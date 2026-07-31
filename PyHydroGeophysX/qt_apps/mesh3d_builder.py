"""Compatibility shim for canonical 3-D mesh generation."""

from PyHydroGeophysX._internal.deprecations import warn_legacy_path as _warn_legacy_path
from PyHydroGeophysX.core.mesh_3d import *  # noqa: F401,F403

_warn_legacy_path("qt_apps.mesh3d_builder", "core.mesh_3d")
