"""
Cross-Method Structural Constraints
===================================

This example shows how to:
1. Build structural weights from an SRT velocity model.
2. Apply those weights to an ERT smoothness matrix.
3. Build a cross-gradient operator between two models.
"""

# sphinx_gallery_thumbnail_path = 'auto_examples/images/Ex_cross_constraints_fig_01.png'

import numpy as np
from scipy.sparse import diags

from PyHydroGeophysX.inversion.cross_constraints import StructuralConstraint


class _DummyMesh:
    """Minimal mesh-like object for demonstration."""

    def __init__(self, n_cells: int):
        self._n_cells = n_cells

    def cellCount(self):
        return self._n_cells

    def cellCenters(self):
        x = np.linspace(0.0, 100.0, self._n_cells)
        z = np.linspace(0.0, 30.0, self._n_cells)
        return np.column_stack((x, z))


def run_example():
    n_cells = 60
    mesh = _DummyMesh(n_cells)

    velocity = np.linspace(1200.0, 3200.0, n_cells)
    velocity[25:35] += 1400.0

    conductivity = np.linspace(0.01, 0.2, n_cells)
    conductivity[20:30] *= 2.5

    boundary_weights = StructuralConstraint.from_velocity_model(
        velocity_model=velocity,
        mesh=mesh,
        gradient_threshold=0.3,
    )

    # Dummy smoothness matrix with nearest-neighbor differences.
    w = np.ones(n_cells)
    Wm = diags([-w[:-1], w[:-1]], offsets=[0, 1], shape=(n_cells - 1, n_cells)).tocsr()
    Wm_weighted = StructuralConstraint.apply_structural_weights_to_Wm(Wm, boundary_weights)

    cross_op = StructuralConstraint.build_cross_gradient_operator(
        mesh=mesh,
        model_a=np.log10(1.0 / conductivity),
        model_b=velocity,
    )

    print("Cross-constraint example complete")
    print(f"  smoothness matrix shape: {Wm.shape}")
    print(f"  weighted smoothness shape: {Wm_weighted.shape}")
    print(f"  cross-gradient operator shape: {cross_op.shape}")

    return Wm_weighted, cross_op


if __name__ == "__main__":
    run_example()
