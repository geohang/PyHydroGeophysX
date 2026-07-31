"""Compatibility re-export for the shared PyVista compatibility helpers."""

from PyHydroGeophysX.visualization.pyvista_compat import (
    ensure_vtk_matplotlib_shim,
    try_import_pyvista,
)

__all__ = ["ensure_vtk_matplotlib_shim", "try_import_pyvista"]
