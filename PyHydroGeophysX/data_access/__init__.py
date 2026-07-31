"""
Data access abstraction for PyHydroGeophysX.

Provides accessor classes that unify local filesystem and HTTP-based
data loading for the Hydro-to-Geophysics workflow.
"""

from PyHydroGeophysX.data_access.accessors import (
    BaseHydroAccessor,
    LocalHydroAccessor,
    HttpHydroAccessor,
    load_manifest,
    get_manifest_entry,
)

__all__ = [
    "BaseHydroAccessor",
    "LocalHydroAccessor",
    "HttpHydroAccessor",
    "load_manifest",
    "get_manifest_entry",
]
