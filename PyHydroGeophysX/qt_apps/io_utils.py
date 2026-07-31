"""Lightweight, Qt-free I/O helpers for the desktop workbench.

The generic numeric-table loaders live in
:mod:`PyHydroGeophysX.data_processing.table_io` and are re-exported here so
existing desktop imports keep
working. Only the JSON helpers used by the Streamlit <-> Qt bridge remain
local: ``write_json`` is atomic (temp file + ``os.replace``) so the bridge
poller never sees a half-written document.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

from PyHydroGeophysX.data_processing.table_io import (  # noqa: F401
    ensure_dir,
    load_2d_array,
    load_xyz_table,
    read_json,
    write_csv,
    write_json,
)

PathLike = Union[str, Path]

__all__ = [
    "PathLike",
    "ensure_dir",
    "load_2d_array",
    "load_xyz_table",
    "read_json",
    "write_csv",
    "write_json",
]
