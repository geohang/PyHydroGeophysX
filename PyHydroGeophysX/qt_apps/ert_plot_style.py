"""Compatibility shim for ERT plot styling."""

from PyHydroGeophysX._internal.deprecations import warn_legacy_path as _warn_legacy_path
from PyHydroGeophysX.visualization.ert_style import *  # noqa: F401,F403

_warn_legacy_path("qt_apps.ert_plot_style", "visualization.ert_style")
