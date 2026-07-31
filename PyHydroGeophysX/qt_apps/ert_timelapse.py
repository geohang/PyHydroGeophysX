"""Compatibility shim for canonical time-lapse ERT workflows."""

from PyHydroGeophysX._internal.deprecations import warn_legacy_path as _warn_legacy_path
from PyHydroGeophysX.inversion.time_lapse import (  # noqa: F401
    BackendUnavailable,
    build_timelapse_config,
    default_times,
    run_timelapse_ert,
)

_warn_legacy_path("qt_apps.ert_timelapse", "inversion.time_lapse")

__all__ = [
    "BackendUnavailable",
    "build_timelapse_config",
    "default_times",
    "run_timelapse_ert",
]
