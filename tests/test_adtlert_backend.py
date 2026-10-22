"""The optional ADTLERT backend: 2.5D receiver polarity and 3D tetrahedral meshes."""

import numpy as np
import pytest

pg = pytest.importorskip("pygimli")
from pygimli.physics import ert  # noqa: E402

from PyHydroGeophysX.inversion import ert_inversion  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def adtlert_gpu():
    ert_inversion._enable_adtlert_float64()  # before adtlert is first imported
    pytest.importorskip("adtlert")
    if ert_inversion._resolve_ert_engine("adtlert") != "adtlert":
        pytest.skip("the ADTLERT backend needs a CUDA GPU with cuDSS and NVRTC")


def _with_receivers_swapped(data, rows):
    swapped = pg.DataContainerERT(data)
    m, n = np.asarray(data["m"]).copy(), np.asarray(data["n"]).copy()
    m[rows], n[rows] = n[rows], m[rows].copy()
    swapped["m"], swapped["n"] = m, n
    swapped["k"] = ert.createGeometricFactors(swapped)
    return swapped


def test_reversed_receiver_polarity_is_the_same_measurement():
    from PyHydroGeophysX.inversion.ert_mesh import build_inversion_mesh

    x = np.linspace(0.0, 23.0, 24)
    data = ert.createData(elecs=x, schemeName="dd")
    data["k"] = ert.createGeometricFactors(data)
    data["rhoa"] = np.full(data.size(), 100.0)
    mesh = build_inversion_mesh(data, mesh_quality=33, para_depth=8)
    rows = np.arange(0, data.size(), 2)
    swapped = _with_receivers_swapped(data, rows)
    # pyGIMLi's dipole-dipole ordering has negative geometric factors; the swapped rows positive.
    assert np.all(np.asarray(data["k"]) < 0) and np.all(np.asarray(swapped["k"])[rows] > 0)

    forwards = [ert_inversion._build_adtlert_forward(d, mesh)[0] for d in (data, swapped)]
    active = forwards[0].cell_count
    model = np.log(100.0) + np.linspace(-0.5, 0.5, active)
    (response, jacobian), (swapped_response, swapped_jacobian) = (
        forward.forward_and_jacobian(model, log_transform=True) for forward in forwards
    )
    np.testing.assert_allclose(swapped_response, response, rtol=1e-9)
    np.testing.assert_allclose(swapped_jacobian, jacobian, rtol=1e-7,
                               atol=1e-10 * np.abs(jacobian).max())


def _tetrahedral_box(xs, ys, zs, inner):
    """Six tetrahedra per box cell; cells with ``inner(centre)`` get marker 2, the rest 1."""

    mesh = pg.Mesh(3)
    index = np.arange(len(xs) * len(ys) * len(zs)).reshape(len(xs), len(ys), len(zs))
    for x in xs:
        for y in ys:
            for z in zs:
                mesh.createNode(pg.Pos(x, y, z))
    corners = ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            for k in range(len(zs) - 1):
                v = [int(index[i + a, j + b, k + c]) for a, b, c in corners]
                centre = np.mean([[xs[i + a], ys[j + b], zs[k + c]] for a, b, c in corners], axis=0)
                marker = 2 if inner(centre) else 1
                for a, b, c in ((1, 2, 6), (2, 3, 6), (3, 7, 6), (7, 4, 6), (4, 5, 6), (5, 1, 6)):
                    mesh.createTetrahedron(*(mesh.node(v[q]) for q in (0, a, b, c)), marker=marker)
    mesh.createNeighborInfos()
    for boundary in mesh.boundaries():
        if boundary.outside():
            top = abs(boundary.center()[2] - zs[-1]) < 1e-9
            boundary.setMarker(pg.core.MARKER_BOUND_HOMOGEN_NEUMANN if top
                               else pg.core.MARKER_BOUND_MIXED)
    return mesh


def test_adtlert_inverts_a_3d_survey_on_a_tetrahedral_mesh(tmp_path):
    xs = np.concatenate(([-30.0, -15.0], np.arange(-5.0, 25.1, 2.5), [35.0, 50.0]))
    zs = np.concatenate(([-30.0, -18.0], np.arange(-10.0, 0.1, 2.5)))
    mesh = _tetrahedral_box(xs, xs, zs, lambda c: -5 <= c[0] <= 25 and -5 <= c[1] <= 25 and c[2] >= -10)
    mesh_file = tmp_path / "mesh3d.bms"
    mesh.save(str(mesh_file))

    grid = np.arange(0.0, 20.1, 5.0)
    data = pg.DataContainerERT()
    for y in grid:
        for x in grid:
            data.createSensor(pg.Pos(x, y, 0.0))
    n = len(grid)
    lines = [[j * n + i for i in range(n)] for j in range(n)] + [[j * n + i for j in range(n)] for i in range(n)]
    for line in lines:
        for start in range(n - 3):
            for level in range(1, n - 2 - start):
                a, b = line[start], line[start + 1]
                data.createFourPointData(data.size(), a, b, line[start + 1 + level], line[start + 2 + level])
    centres = np.asarray([[c.center()[i] for i in range(3)] for c in mesh.cells()])
    block = (np.abs(centres[:, 0] - 10) < 4) & (np.abs(centres[:, 1] - 10) < 4) & (centres[:, 2] > -6) & (centres[:, 2] < -1.5)
    rho = np.where(block, 20.0, 100.0)
    simulated = ert.simulate(mesh, scheme=data, res=rho, noiseLevel=0.0, verbose=False)
    simulated["err"] = np.full(simulated.size(), 0.03)
    data_file = tmp_path / "survey3d.dat"
    simulated.save(str(data_file), "a b m n rhoa err k")

    result = ert_inversion.run_ert_manager_inversion(
        data_file, tmp_path / "out", engine="adtlert", mesh_file=str(mesh_file),
        error_source="file", lam=20.0, max_iterations=6, max_total_iterations=6,
    )
    assert result["engine"] == "adtlert"
    assert result["metrics"]["backend"] == "adtlert"
    chi2 = result["convergence"]
    assert np.all(np.isfinite(chi2)) and chi2[-1] < 0.2 * chi2[0]
