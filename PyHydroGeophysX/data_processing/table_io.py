"""Lightweight numeric table I/O shared by core and desktop workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Union

import numpy as np

PathLike = Union[str, Path]
_ARRAY_SUFFIXES = {".npy", ".npz", ".csv", ".txt", ".dat"}

__all__ = [
    "PathLike",
    "ensure_dir",
    "load_2d_array",
    "load_xyz_table",
    "write_csv",
    "write_json",
    "read_json",
]


def ensure_dir(path: PathLike) -> Path:
    """Create *path* and its parents if needed, then return it."""
    result = Path(path)
    result.mkdir(parents=True, exist_ok=True)
    return result


def _load_text_matrix(path: Path) -> np.ndarray:
    """Load a numeric text matrix, tolerating a header row."""
    for delimiter in (",", None):
        try:
            return np.atleast_2d(np.loadtxt(path, delimiter=delimiter))
        except Exception:
            continue
    try:
        import pandas as pd
    except Exception as exc:  # pragma: no cover - pandas is a base dependency
        raise ValueError(
            f"Could not parse '{path.name}' and pandas is unavailable: {exc}"
        ) from exc
    for header in ("infer", None):
        try:
            frame = pd.read_csv(path, header=header)
            numeric = frame.select_dtypes(include=[np.number])
            if numeric.size:
                return np.atleast_2d(numeric.to_numpy(dtype=float))
        except Exception:
            continue
    raise ValueError(f"Could not parse '{path.name}' as a numeric matrix.")


def load_2d_array(path: PathLike) -> np.ndarray:
    """Load an array from NPY, NPZ, CSV, TXT, or DAT input."""
    source = Path(path)
    if not source.exists():
        raise ValueError(f"File not found: {source}")
    suffix = source.suffix.lower()
    if suffix == ".npy":
        try:
            return np.asarray(np.load(source, allow_pickle=False))
        except Exception as exc:
            raise ValueError(f"Failed to read .npy file '{source.name}': {exc}") from exc
    if suffix == ".npz":
        try:
            with np.load(source, allow_pickle=False) as data:
                if not data.files:
                    raise ValueError(f"'{source.name}' is an empty .npz archive.")
                return np.asarray(data[data.files[0]])
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"Failed to read .npz file '{source.name}': {exc}") from exc
    if suffix in {".csv", ".txt", ".dat"}:
        return _load_text_matrix(source)
    raise ValueError(
        f"Unsupported file type '{suffix}'. Use one of: "
        f"{', '.join(sorted(_ARRAY_SUFFIXES))}."
    )


def load_xyz_table(path: PathLike, min_cols: int = 2) -> np.ndarray:
    """Load a two-dimensional table with at least *min_cols* columns."""
    array = np.atleast_2d(np.asarray(load_2d_array(path), dtype=float))
    if array.ndim != 2 or array.shape[1] < min_cols:
        raise ValueError(
            f"Expected a table with at least {min_cols} columns, got shape "
            f"{array.shape} from '{Path(path).name}'."
        )
    return array


def write_csv(
    path: PathLike,
    rows: Sequence[Sequence[Any]],
    header: Optional[Iterable[str]] = None,
) -> Path:
    """Write rows to a CSV file."""
    import csv

    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if header is not None:
            writer.writerow(list(header))
        for row in rows:
            writer.writerow(list(row))
    return target


def write_json(path: PathLike, obj: Any) -> Path:
    """Atomically write a JSON document."""
    import json
    import os
    import tempfile

    target = Path(path)
    ensure_dir(target.parent)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=target.name + ".", suffix=".tmp", dir=str(target.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(obj, handle, indent=2, default=str)
        os.replace(temporary_name, target)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise
    return target


def read_json(path: PathLike) -> Optional[dict[str, Any]]:
    """Read JSON, returning ``None`` for a missing or malformed document."""
    import json

    source = Path(path)
    if not source.exists():
        return None
    try:
        with source.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except Exception:
        return None
    return value if isinstance(value, dict) else None
