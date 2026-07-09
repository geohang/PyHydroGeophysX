"""Module pages for the workbench and a defensive factory to build them.

``build_module`` imports each page lazily so that a failure in one module (for
example a missing optional dependency) degrades to a clean placeholder page
rather than taking down the whole application.
"""

from __future__ import annotations

import importlib
import traceback
from typing import Any, Callable, List, Tuple

from PyHydroGeophysX.qt_apps.modules.base import BaseModule, HomePage, PlaceholderModule

LogFn = Callable[..., None]

# module key -> (submodule, class name, human title)
MODULE_SPECS = {
    "seismic": ("seismic_processing", "SeismicProcessingModule", "Seismic Processing"),
    "ert": ("ert_processing", "ERTProcessingModule", "ERT Processing"),
    "mesh3d": ("mesh3d_processing", "Mesh3DModule", "3D Mesh Builder"),
    "em": ("em_processing", "EMProcessingModule", "EM Processing"),
    "gravmag": ("gravmag_processing", "GravMagProcessingModule", "Gravity / Magnetics"),
    "hydro_geophysics": ("hydro_geophysics", "HydroGeophysicsModule", "Hydro → Geophysics"),
    "geo_hydrology": ("geo_hydrology", "GeoHydrologyModule", "ERT → Water Content"),
    "seismic3d": ("seismic3d", "Seismic3DModule", "Seismic → Structure"),
}

#: Order used to populate the central stack. ``home`` is always first.
MODULE_ORDER: List[str] = ["home", "seismic", "ert", "mesh3d", "em", "gravmag", "hydro_geophysics", "geo_hydrology", "seismic3d"]

#: Install commands for optional packages a module import may be missing.
_INSTALL_HINTS = {
    "obspy": 'pip install "pyhydrogeophysx[seismic-raw]"',
    "segyio": "pip install segyio",
    "pyvista": "pip install pyvista pyvistaqt",
    "pyvistaqt": "pip install pyvista pyvistaqt",
    "pygimli": "conda install -c gimli pygimli",
    "simpeg": 'pip install "pyhydrogeophysx[geophysics]"',
    "resipy": "pip install resipy",
    "qtawesome": 'pip install "pyhydrogeophysx[desktop]"',
    "pyqtgraph": 'pip install "pyhydrogeophysx[desktop]"',
}


def _missing_dependency_hint(exc: BaseException) -> str:
    """Return an HTML hint naming the missing package and how to install it."""
    name = getattr(exc, "name", None)
    if not name:
        return ""
    root = str(name).split(".")[0]
    hint = _INSTALL_HINTS.get(root, f"pip install {root}")
    return (
        f"<br>Missing package: <b>{root}</b>. Install it with "
        f"<code>{hint}</code> and restart the workbench."
    )


def build_module(key: str, state: Any, log: LogFn) -> BaseModule:
    """Instantiate the page for ``key``; never raises."""
    if key == "home":
        return HomePage(state, log)
    spec = MODULE_SPECS.get(key)
    if spec is None:
        return PlaceholderModule(state, log, key, f"Unknown module '{key}'.", key=key)
    submodule, class_name, title = spec
    try:
        mod = importlib.import_module(f"PyHydroGeophysX.qt_apps.modules.{submodule}")
        cls = getattr(mod, class_name)
        return cls(state, log)
    except Exception as exc:  # noqa: BLE001 - we want to catch everything here
        log(f"Failed to load module '{key}': {exc}", "error")
        log(traceback.format_exc(), "debug")
        dependency_hint = _missing_dependency_hint(exc) if isinstance(exc, ImportError) else ""
        return PlaceholderModule(
            state,
            log,
            title,
            f"This module could not be loaded:<br><br><code>{exc}</code>"
            f"{dependency_hint}<br><br>"
            "The rest of the workbench is unaffected.",
            key=key,
        )


__all__ = ["BaseModule", "HomePage", "PlaceholderModule", "build_module", "MODULE_ORDER", "MODULE_SPECS"]
