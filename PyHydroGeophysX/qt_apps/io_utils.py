"""Lightweight, Qt-free I/O helpers for the desktop workbench.

These functions only depend on numpy (and optionally pandas) so they can be
imported and unit-tested without a running QApplication. Every loader raises a
``ValueError`` with a human-readable message on failure; callers are expected to
catch it and surface the message in the log panel / a message box rather than
letting the app crash.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np

PathLike = Union[str, Path]

_ARRAY_SUFFIXES = {".npy", ".npz", ".csv", ".txt", ".dat"}


def ensure_dir(path: PathLike) -> Path:
    """Create ``path`` (and parents) if needed and return it as a ``Path``."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _load_text_matrix(path: Path) -> np.ndarray:
    """Load a numeric matrix from a CSV/TXT/DAT file, tolerating a header row."""
    # First try the fast path with a comma delimiter, then whitespace.
    for delimiter in (",", None):
        try:
            arr = np.loadtxt(path, delimiter=delimiter)
            return np.atleast_2d(arr)
        except Exception:
            continue
    # Fall back to pandas, which handles headers and mixed delimiters well.
    try:
        import pandas as pd  # local import; pandas is a desktop dependency
    except Exception as exc:  # pragma: no cover - pandas always present on desktop
        raise ValueError(
            f"Could not parse '{path.name}' as a numeric matrix and pandas is "
            f"unavailable: {exc}"
        )
    for header in ("infer", None):
        try:
            frame = pd.read_csv(path, header=header)
            numeric = frame.select_dtypes(include=[np.number])
            if numeric.size == 0:
                continue
            return np.atleast_2d(numeric.to_numpy(dtype=float))
        except Exception:
            continue
    raise ValueError(f"Could not parse '{path.name}' as a numeric matrix.")


def load_2d_array(path: PathLike) -> np.ndarray:
    """Load a 2D numeric array from ``.npy`` / ``.npz`` / ``.csv`` / ``.txt``.

    For ``.npz`` archives the first stored array is returned. Arrays with more
    than two dimensions are returned unchanged so the caller can choose a slice.
    """
    p = Path(path)
    if not p.exists():
        raise ValueError(f"File not found: {p}")
    suffix = p.suffix.lower()
    if suffix == ".npy":
        try:
            return np.asarray(np.load(p, allow_pickle=False))
        except Exception as exc:
            raise ValueError(f"Failed to read .npy file '{p.name}': {exc}")
    if suffix == ".npz":
        try:
            with np.load(p, allow_pickle=False) as data:
                keys = list(data.files)
                if not keys:
                    raise ValueError(f"'{p.name}' is an empty .npz archive.")
                return np.asarray(data[keys[0]])
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"Failed to read .npz file '{p.name}': {exc}")
    if suffix in (".csv", ".txt", ".dat"):
        return _load_text_matrix(p)
    raise ValueError(
        f"Unsupported file type '{suffix}'. Use one of: "
        f"{', '.join(sorted(_ARRAY_SUFFIXES))}."
    )


def load_xyz_table(path: PathLike, min_cols: int = 2) -> np.ndarray:
    """Load an ``(N, >=min_cols)`` numeric table (e.g. electrode x,z or x,y,z)."""
    arr = load_2d_array(path)
    arr = np.atleast_2d(np.asarray(arr, dtype=float))
    if arr.ndim != 2 or arr.shape[1] < min_cols:
        raise ValueError(
            f"Expected a table with at least {min_cols} columns, got shape "
            f"{arr.shape} from '{Path(path).name}'."
        )
    return arr


def write_json(path: PathLike, obj: Any) -> Path:
    """Write ``obj`` to ``path`` as pretty JSON using ``default=str``.

    The write is atomic (temp file in the same directory + ``os.replace``) so a
    concurrent reader, e.g. the Streamlit bridge poller, never sees a
    half-written document.
    """
    import os
    import tempfile

    p = Path(path)
    ensure_dir(p.parent)
    fd, tmp_name = tempfile.mkstemp(prefix=p.name + ".", suffix=".tmp", dir=str(p.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(obj, handle, indent=2, default=str)
        os.replace(tmp_name, p)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return p


def read_json(path: PathLike) -> Optional[Dict[str, Any]]:
    """Read JSON from ``path``; return ``None`` if missing or malformed."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        with p.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def write_csv(
    path: PathLike,
    rows: Sequence[Sequence[Any]],
    header: Optional[Iterable[str]] = None,
) -> Path:
    """Write ``rows`` to a CSV file. Pure stdlib so it works without pandas."""
    import csv

    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if header is not None:
            writer.writerow(list(header))
        for row in rows:
            writer.writerow(list(row))
    return p
