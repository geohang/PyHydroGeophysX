"""Compatibility shim for canonical time-lapse ERT workflows."""

from PyHydroGeophysX._internal.deprecations import warn_legacy_path as _warn_legacy_path
from PyHydroGeophysX.inversion.time_lapse import (  # noqa: F401
    DEFAULT_TL,
    INVERSION_TYPES,
    BackendUnavailable,
    build_timelapse_config,
    default_times,
    run_timelapse_ert,
)
# The modules this read and wrote through in 0.3.0, from where they live now.
from PyHydroGeophysX.data_processing import ert_io as ert_load  # noqa: F401
from PyHydroGeophysX.qt_apps import io_utils  # noqa: F401
from PyHydroGeophysX.visualization import ert_style as ert_plot_style  # noqa: F401

_warn_legacy_path("qt_apps.ert_timelapse", "inversion.time_lapse")

__all__ = [
    "BackendUnavailable",
    "DEFAULT_TL",
    "INVERSION_TYPES",
    "build_timelapse_config",
    "default_times",
    "run_timelapse_ert",
]
