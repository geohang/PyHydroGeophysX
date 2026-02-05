import numpy as np
import scipy.sparse as sp
import pytest

pytest.importorskip("pygimli")

from PyHydroGeophysX.inversion.cross_constraints import (
    PetrophysicalCoupling,
    StructuralConstraint,
)


class _Cell:
    def __init__(self, idx: int):
        self._idx = idx

    def id(self):
        return self._idx


class _Boundary:
    def __init__(self, left: int, right: int):
        self._left = _Cell(left)
        self._right = _Cell(right)

    def leftCell(self):
        return self._left

    def rightCell(self):
        return self._right


class _Mesh:
    def __init__(self, n_cells: int):
        self._n_cells = n_cells
        self._boundaries = [_Boundary(i, i + 1) for i in range(n_cells - 1)]

    def cellCount(self):
        return self._n_cells

    def cellCenters(self):
        x = np.linspace(0.0, 10.0, self._n_cells)
        z = np.linspace(0.0, 5.0, self._n_cells)
        return np.column_stack((x, z))

    def boundaries(self):
        return self._boundaries


def test_structural_boundary_weights_and_application():
    mesh = _Mesh(3)
    velocity = np.array([1000.0, 3000.0, 3200.0])

    boundary_weights = StructuralConstraint.from_velocity_model(
        velocity_model=velocity,
        mesh=mesh,
        gradient_threshold=0.3,
    )

    Wm = sp.csr_matrix([[1.0, -1.0, 0.0], [0.0, 1.0, -1.0]])
    Wm_weighted = StructuralConstraint.apply_structural_weights_to_Wm(Wm, boundary_weights)

    assert Wm_weighted.shape == Wm.shape
    assert Wm_weighted[0, 0] <= Wm[0, 0]


def test_cross_gradient_operator_shape():
    mesh = _Mesh(5)
    model_a = np.linspace(10.0, 20.0, 5)
    model_b = np.linspace(1000.0, 2000.0, 5)

    cross_op = StructuralConstraint.build_cross_gradient_operator(mesh, model_a, model_b)

    assert cross_op.shape == (5, 5)
    assert sp.issparse(cross_op)


def test_build_linearized_cross_gradient_blocks_shapes():
    mesh = _Mesh(6)
    Wm = sp.csr_matrix(
        [
            [1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, -1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, -1.0],
        ]
    )

    rcm = StructuralConstraint.build_neighborhood_matrix(
        mesh=mesh,
        Wm=Wm,
        source="smoothness",
        threshold=0.0,
        binarize=True,
    )
    X = StructuralConstraint.build_local_design_matrix(mesh)
    model_a = np.linspace(1.0, 2.0, 6)
    model_b = np.linspace(2.0, 3.0, 6)

    B1, B2 = StructuralConstraint.build_linearized_cross_gradient_blocks(
        rcm,
        X,
        model_a,
        model_b,
        mode="direct",
    )

    assert B1.shape == (6, 6)
    assert B2.shape == (6, 6)
    assert np.all(np.isfinite(B1))
    assert np.all(np.isfinite(B2))


def test_petrophysical_coupling_outputs_and_comparison():
    wc = np.array([0.1, 0.15, 0.2, 0.25])
    porosity = np.array([0.3, 0.32, 0.34, 0.36])

    mapped = PetrophysicalCoupling.water_content_to_all_geophysics(
        wc,
        porosity,
        velocity_model="hertz_mindlin",
        bulk_modulus=36.0,
        shear_modulus=45.0,
        mineral_density=2650.0,
    )

    assert set(mapped.keys()) == {"resistivity", "velocity", "conductivity"}
    assert mapped["resistivity"].shape == wc.shape
    assert mapped["velocity"].shape == wc.shape
    assert mapped["conductivity"].shape == wc.shape

    class _Result:
        def __init__(self, final_model=None, recovered_conductivity=None):
            self.final_model = final_model
            self.recovered_conductivity = recovered_conductivity

    ert_result = _Result(final_model=mapped["resistivity"])
    srt_result = _Result(final_model=mapped["velocity"])
    em_result = _Result(recovered_conductivity=mapped["conductivity"])

    stats = PetrophysicalCoupling.compare_inversions_to_hydro(
        hydro_wc=wc,
        ert_result=ert_result,
        srt_result=srt_result,
        em_result=em_result,
        petro_params={"porosity": porosity, "rhos": 100.0, "n": 2.0},
    )

    assert "ert" in stats
    assert "srt" in stats
    assert "em" in stats
    assert "summary" in stats
