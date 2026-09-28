"""Gravity and magnetics in one namespace.

Nothing is implemented here: preprocessing, gridding, profiles and the grid
writer are in :mod:`PyHydroGeophysX.data_processing.gravmag`, the analytic
bodies in :mod:`PyHydroGeophysX.forward.gravmag`, and the SimPEG inversion in
:mod:`PyHydroGeophysX.inversion.gravmag`. The examples, the generated
walkthroughs and older scripts import them from here.
"""

from PyHydroGeophysX.data_processing.gravmag import (
    build_gravmag_config,
    extract_profile,
    grid_data,
    qc_products,
    regional_residual,
    save_grid,
    spatially_balanced_indices,
)
from PyHydroGeophysX.forward.gravmag import (
    forward_bodies,
    gravity_prism,
    gravity_sphere,
    magnetic_dipole,
)
from PyHydroGeophysX.inversion.gravmag import (
    InversionBackendUnavailable,
    backend_status,
    invert_gravmag,
)

__all__ = [
    "regional_residual",
    "spatially_balanced_indices",
    "qc_products",
    "grid_data",
    "extract_profile",
    "gravity_sphere",
    "gravity_prism",
    "magnetic_dipole",
    "forward_bodies",
    "save_grid",
    "build_gravmag_config",
    "InversionBackendUnavailable",
    "backend_status",
    "invert_gravmag",
]
