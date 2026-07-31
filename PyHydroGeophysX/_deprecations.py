"""Compatibility shim for :mod:`PyHydroGeophysX._internal.deprecations`."""

from PyHydroGeophysX._internal.deprecations import warn_legacy_path

warn_legacy_path(
    "PyHydroGeophysX._deprecations",
    "PyHydroGeophysX._internal.deprecations",
)

__all__ = ["warn_legacy_path"]
