"""Compatibility shim for canonical ERT input loading."""

from PyHydroGeophysX._internal.deprecations import warn_legacy_path as _warn_legacy_path
from PyHydroGeophysX.data_processing.ert_io import *  # noqa: F401,F403

_warn_legacy_path("qt_apps.ert_load", "data_processing.ert_io")
