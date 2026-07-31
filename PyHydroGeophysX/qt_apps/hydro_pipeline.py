"""Compatibility shim for the promoted hydrology-to-geophysics workflow."""

from PyHydroGeophysX._internal.deprecations import warn_legacy_path as _warn_legacy_path
from PyHydroGeophysX.Hydro_modular.hydro_to_geophysics import *  # noqa: F401,F403

_warn_legacy_path("qt_apps.hydro_pipeline", "Hydro_modular.hydro_to_geophysics")
