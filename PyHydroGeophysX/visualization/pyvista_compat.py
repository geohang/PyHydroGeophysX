"""Shared PyVista / PyVistaQt setup used by the 3D viewers.

Used by both the Mesh 3D module and the inversion model viewer, so they share
one VTK shim and one offscreen guard.
"""

from __future__ import annotations

import os
import sys
import types
from typing import Any, Optional, Tuple


def ensure_vtk_matplotlib_shim() -> None:
    """Stub the optional ``vtkmodules.vtkRenderingMatplotlib`` module when absent.

    Some VTK builds (conda-forge ``vtk-base``) omit it, yet pyvista imports it
    unconditionally for a side effect not needed for mesh display.
    """
    if "vtkmodules.vtkRenderingMatplotlib" in sys.modules:
        return
    try:
        import vtkmodules.vtkRenderingMatplotlib  # noqa: F401 - real module present
    except Exception:  # noqa: BLE001 - missing optional submodule; provide a stub
        sys.modules["vtkmodules.vtkRenderingMatplotlib"] = types.ModuleType(
            "vtkmodules.vtkRenderingMatplotlib")


def pygimli_to_pyvista(mesh: Any) -> Any:
    """A PyGIMLi mesh as a PyVista grid, its cell markers as ``Marker``.

    Through PyGIMLi's own VTK writer, which is much faster on a large volume
    than building the grid cell by cell in Python. Holds no Qt object, so it
    can run off the UI thread.
    """
    import shutil
    import tempfile

    import numpy as np

    ensure_vtk_matplotlib_shim()
    import pyvista as pv

    folder = tempfile.mkdtemp(prefix="phgx_mesh_")
    try:
        path = os.path.join(folder, "mesh.vtk")
        mesh.exportVTK(path)
        grid = pv.read(path)
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    grid.cell_data["Marker"] = np.asarray(mesh.cellMarkers(), dtype=int)
    return grid


def try_import_pyvista() -> Tuple[bool, Optional[Any], Optional[Any], str]:
    """Return ``(ok, pyvista_module, QtInteractor, error)``.

    Disabled under the offscreen Qt platform (headless / --self-test), where
    constructing a live ``QtInteractor`` aborts the process at the VTK level.
    """
    try:
        if os.environ.get("QT_QPA_PLATFORM", "").lower() == "offscreen":
            return False, None, None, "offscreen platform (interactive 3D viewer disabled)"
        os.environ.setdefault("QT_API", "pyside6")
        ensure_vtk_matplotlib_shim()
        import pyvista as pv
        from pyvistaqt import QtInteractor
        return True, pv, QtInteractor, ""
    except Exception as exc:  # noqa: BLE001
        return False, None, None, str(exc)
