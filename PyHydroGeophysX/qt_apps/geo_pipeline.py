"""Compatibility shim for the promoted ERT-to-water-content workflow."""

from PyHydroGeophysX._internal.deprecations import warn_legacy_path as _warn_legacy_path
from PyHydroGeophysX.Geophy_modular.ERT_to_WC import *  # noqa: F401,F403

_warn_legacy_path("qt_apps.geo_pipeline", "Geophy_modular.ERT_to_WC")
