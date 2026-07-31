"""Compatibility shim for :mod:`PyHydroGeophysX._internal.utils`."""

from PyHydroGeophysX._internal.deprecations import warn_legacy_path
from PyHydroGeophysX._internal.utils import noop, utc_now

warn_legacy_path("PyHydroGeophysX._utils", "PyHydroGeophysX._internal.utils")

__all__ = ["noop", "utc_now"]
