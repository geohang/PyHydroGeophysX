"""Shared PyVista / PyVistaQt setup used by the 3D viewers (Qt-light).

Factored out of the Mesh 3D module so the inversion model viewer can reuse the
same VTK shim and the same offscreen guard.
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
