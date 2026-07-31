"""Compatibility exports for the Qt gravity/magnetics workbench module."""

from PyHydroGeophysX._internal.deprecations import warn_legacy_path
from PyHydroGeophysX.workflows.gravmag import (
    InversionBackendUnavailable,
    backend_status,
    build_gravmag_config,
    extract_profile,
    forward_bodies,
    gravity_prism,
    gravity_sphere,
    grid_data,
    invert_gravmag,
    magnetic_dipole,
    qc_products,
    regional_residual,
    save_grid,
    spatially_balanced_indices,
)

warn_legacy_path("qt_apps.gravmag_pipeline", "workflows.gravmag")

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
