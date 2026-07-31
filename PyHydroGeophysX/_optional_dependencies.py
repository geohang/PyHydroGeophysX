"""Compatibility shim for :mod:`PyHydroGeophysX._internal.optional_dependencies`."""

from PyHydroGeophysX._internal.deprecations import warn_legacy_path
from PyHydroGeophysX._internal.optional_dependencies import (
    BackendUnavailable,
    INSTALL_HINTS,
    installation_hint,
    missing_dependency_name,
    optional_import_error,
)

warn_legacy_path(
    "PyHydroGeophysX._optional_dependencies",
    "PyHydroGeophysX._internal.optional_dependencies",
)

__all__ = [
    "BackendUnavailable",
    "INSTALL_HINTS",
    "installation_hint",
    "missing_dependency_name",
    "optional_import_error",
]
