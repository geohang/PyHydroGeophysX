"""Inverting ERT on a mesh the user built elsewhere.

Generating a mesh from the electrode line is fine for a 2D profile and hopeless
for a 3D domain with topography, boreholes or known structure, which is meshed
externally. An imported mesh fails in ways a generated one cannot, so most of
these tests are about refusing a bad one with a message that says what is wrong.
"""

import numpy as np
import pytest

pytest.importorskip("pygimli")

import pygimli as pg  # noqa: E402
import pygimli.meshtools as mt  # noqa: E402
from pygimli.physics import ert  # noqa: E402

from PyHydroGeophysX.inversion.ert_inversion import (  # noqa: E402
    load_inversion_mesh,
    run_ert_manager_inversion,
)


@pytest.fixture(scope="module")
def survey(tmp_path_factory):
    """A 24-electrode dipole-dipole line over a buried conductive block."""
    scheme = ert.createData(elecs=np.linspace(0, 47, 24), schemeName="dd")
    world = mt.createWorld(start=[-20, 0], end=[67, -30], worldMarker=True)
    block = mt.createRectangle(start=[18, -4], end=[30, -12], marker=2)
    sim = mt.createMesh(mt.mergePLC([world, block]), quality=33, area=1.0)
    data = ert.simulate(sim, res=[[1, 200.0], [2, 25.0]], scheme=scheme,
                        noiseLevel=2, noiseAbs=1e-5, seed=7, verbose=False)
    data.remove(data["rhoa"] <= 0)
    path = tmp_path_factory.mktemp("ert") / "line.dat"
    data.save(str(path))
    return path, data, scheme


@pytest.fixture(scope="module")
def good_mesh(survey, tmp_path_factory):
    _, _, scheme = survey
    mesh = mt.createMesh(
        mt.createParaMeshPLC(scheme.sensorPositions(), paraDepth=20.0,
                             paraDX=0.4, boundary=1.0), quality=33)
    path = tmp_path_factory.mktemp("mesh") / "good.bms"
    mesh.save(str(path))
    return path, mesh


def test_a_valid_mesh_loads(survey, good_mesh):
    _, data, _ = survey
    path, mesh = good_mesh
    loaded = load_inversion_mesh(path, data=data)
    assert loaded.cellCount() == mesh.cellCount()
    assert sum(1 for c in loaded.cells() if c.marker() > 1) > 0


def test_a_mesh_with_no_parameter_domain_is_refused(survey, tmp_path):
    """Every cell tagged background means nothing would be inverted, which
    PyGIMLi reports far downstream as something else entirely."""
    _, data, _ = survey
    mesh = pg.createGrid(x=np.linspace(-20, 67, 30), y=np.linspace(-30, 0, 12))
    for cell in mesh.cells():
        cell.setMarker(1)
    path = tmp_path / "background_only.bms"
    mesh.save(str(path))
    with pytest.raises(ValueError, match="marker"):
        load_inversion_mesh(path, data=data)


def test_a_mesh_that_misses_the_electrodes_is_refused(survey, tmp_path):
    """The usual import mistake is a wrong origin or wrong units."""
    _, data, _ = survey
    mesh = pg.createGrid(x=np.linspace(500.0, 560.0, 20),
                         y=np.linspace(-30, 0, 10))
    for cell in mesh.cells():
        cell.setMarker(2)
    path = tmp_path / "wrong_origin.bms"
    mesh.save(str(path))
    with pytest.raises(ValueError, match="electrode"):
        load_inversion_mesh(path, data=data)


def test_a_missing_file_is_reported_as_one(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_inversion_mesh(tmp_path / "nope.bms")


def test_a_mesh_can_be_checked_before_any_data_is_loaded(good_mesh):
    """Importing a mesh first is allowed; the electrode check waits for data."""
    path, mesh = good_mesh
    assert load_inversion_mesh(path, data=None).cellCount() == mesh.cellCount()


@pytest.mark.parametrize("engine", ["pygimli", "pyhydro"])
def test_the_imported_mesh_is_the_one_inverted(survey, good_mesh, tmp_path,
                                               engine):
    path, _, _ = survey
    mesh_path, mesh = good_mesh
    generated = run_ert_manager_inversion(
        path, tmp_path / f"{engine}_gen", lam=30.0, max_iterations=3,
        auto_lambda=False, engine=engine)
    imported = run_ert_manager_inversion(
        path, tmp_path / f"{engine}_imp", lam=30.0, max_iterations=3,
        auto_lambda=False, engine=engine, mesh_file=str(mesh_path))

    expected = sum(1 for c in mesh.cells() if c.marker() > 1)
    assert imported["mgr"].paraDomain.cellCount() == expected
    assert imported["mgr"].paraDomain.cellCount() != \
        generated["mgr"].paraDomain.cellCount()
    assert np.isfinite(imported["chi2"])


def test_a_bad_mesh_stops_the_run_before_it_starts(survey, tmp_path):
    _, data, _ = survey
    path, _, _ = survey
    mesh = pg.createGrid(x=np.linspace(500.0, 560.0, 20),
                         y=np.linspace(-30, 0, 10))
    for cell in mesh.cells():
        cell.setMarker(2)
    bad = tmp_path / "bad.bms"
    mesh.save(str(bad))
    with pytest.raises(ValueError, match="electrode"):
        run_ert_manager_inversion(path, tmp_path / "run", lam=30.0,
                                  max_iterations=2, auto_lambda=False,
                                  mesh_file=str(bad))
