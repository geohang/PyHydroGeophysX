"""Compatibility shim for canonical 3-D ERT forward modeling."""

from PyHydroGeophysX._internal.deprecations import warn_legacy_path as _warn_legacy_path
from PyHydroGeophysX.forward.ert3d import *  # noqa: F401,F403

_warn_legacy_path("qt_apps.ert3d_pipeline", "forward.ert3d")
