"""Compatibility shim for raster conversion utilities."""

from PyHydroGeophysX._internal.deprecations import warn_legacy_path as _warn_legacy_path
from PyHydroGeophysX.visualization.raster import *  # noqa: F401,F403

_warn_legacy_path("qt_apps.raster", "visualization.raster")
