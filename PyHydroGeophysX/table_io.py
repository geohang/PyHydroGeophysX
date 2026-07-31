"""Compatibility shim for :mod:`PyHydroGeophysX.data_processing.table_io`."""

from PyHydroGeophysX._internal.deprecations import warn_legacy_path
from PyHydroGeophysX.data_processing.table_io import (
    PathLike,
    ensure_dir,
    load_2d_array,
    load_xyz_table,
    read_json,
    write_csv,
    write_json,
)

warn_legacy_path(
    "PyHydroGeophysX.table_io",
    "PyHydroGeophysX.data_processing.table_io",
)

__all__ = [
    "PathLike",
    "ensure_dir",
    "load_2d_array",
    "load_xyz_table",
    "read_json",
    "write_csv",
    "write_json",
]
