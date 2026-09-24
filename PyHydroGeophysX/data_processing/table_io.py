"""Lightweight numeric table I/O shared by core and desktop workflows."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Tuple, Union

import numpy as np

PathLike = Union[str, Path]
_ARRAY_SUFFIXES = {".npy", ".npz", ".csv", ".txt", ".dat"}

__all__ = [
    "PathLike",
    "ensure_dir",
    "load_2d_array",
    "load_xyz_table",
    "npy_shape",
    "read_electrode_table",
    "table_header",
    "save_npy_atomic",
    "write_csv",
    "write_json",
    "read_json",
]


def ensure_dir(path: PathLike) -> Path:
    """Create *path* and its parents if needed, then return it."""
    result = Path(path)
    result.mkdir(parents=True, exist_ok=True)
    return result


def npy_shape(path: PathLike) -> Tuple[int, ...]:
    """Read an ``.npy`` file's shape from its header, without opening the data.

    ``np.load(..., mmap_mode="r")`` is the usual way to ask an array how big it
    is, but a mapping keeps the file open for as long as the array lives, and
    Windows then refuses to let anything overwrite it. Reading the header costs
    one short read, closes immediately, and works even while another process is
    rewriting the file.
    """
    with open(path, "rb") as handle:
        version = np.lib.format.read_magic(handle)
        if version == (1, 0):
            shape, _, _ = np.lib.format.read_array_header_1_0(handle)
        else:
            shape, _, _ = np.lib.format.read_array_header_2_0(handle)
    return tuple(int(value) for value in shape)


def save_npy_atomic(path: PathLike, array: Any) -> Path:
    """Write an ``.npy`` via a sibling temp file, then swap it into place.

    A direct ``np.save`` over an existing result truncates it first, so a write
    that fails part way leaves a corrupt file that still looks like a result.
    Staging keeps the previous file intact on failure and never publishes a
    half-written array.

    This does not defeat a lock: replacing a file another process holds mapped
    still raises, by design. It makes that failure clean rather than destructive.
    """
    import tempfile

    target = Path(path)
    ensure_dir(target.parent)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=target.name + ".", suffix=".partial", dir=str(target.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.save(handle, array)
        os.replace(temporary_name, target)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise
    return target


def _load_text_matrix(path: Path) -> np.ndarray:
    """Read numeric tables with an optional header, without dropping columns."""
    import csv
    import io

    lines = [line for line in path.read_text(encoding="utf-8-sig").splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    if not lines:
        raise ValueError(f"'{path.name}' contains no numeric rows.")
    first = lines[0].split("#", 1)[0]
    delimiter = next((sep for sep in (",", ";", "\t") if sep in first), None)
    fields = next(csv.reader([first], delimiter=delimiter)) if delimiter else first.split()
    def is_number(value):
        try:
            float(value)
            return True
        except ValueError:
            return False
    # A header must consist entirely of labels. A damaged first measurement
    # (one bad cell beside numeric coordinates) must never be silently skipped.
    if fields and not any(is_number(value) for value in fields):
        lines = lines[1:]
    if not lines:
        raise ValueError(f"'{path.name}' contains a header but no numeric rows.")
    try:
        return np.loadtxt(io.StringIO("\n".join(lines)), delimiter=delimiter,
                          quotechar='"', ndmin=2)
    except ValueError as exc:
        raise ValueError(f"Could not parse numeric table '{path.name}': {exc}") from exc


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


def table_header(path: PathLike) -> Optional[list]:
    """The header row of a text table, or None when it has none.

    Found the way :func:`load_2d_array` finds it - the first non-comment line,
    the same delimiter rule, a header only when every field is a label - so each
    name lines up with the column the table is returned with.
    """
    import csv

    source = Path(path)
    if source.suffix.lower() in {".npy", ".npz"}:
        return None
    try:
        lines = source.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeDecodeError):
        return None
    first = next((line for line in lines
                  if line.strip() and not line.lstrip().startswith("#")), "")
    first = first.split("#", 1)[0]
    delimiter = next((sep for sep in (",", ";", "\t") if sep in first), None)
    fields = next(csv.reader([first], delimiter=delimiter)) if delimiter else first.split()

    def is_number(value: str) -> bool:
        try:
            float(value)
            return True
        except ValueError:
            return False

    if not fields or any(is_number(value) for value in fields):
        return None
    return [value.strip() for value in fields]


#: Header words that name an electrode's elevation.
_ELEVATION_NAMES = ("z", "elevation", "elev", "height", "altitude")


def read_electrode_table(path: PathLike) -> Tuple[np.ndarray, np.ndarray, str]:
    """Electrode coordinates from a table, its columns chosen by what they are.

    Returns ``(coords, elevation, how)``. ``coords`` holds the coordinate
    columns in the file's own order - ``(x, elevation)`` for a 2-D line,
    ``(x, y, z)`` otherwise - which is what reading the table by position gave
    for a plain two- or three-column file. ``elevation`` is each electrode's
    height, and ``how`` says which columns were read.

    Vendors lay the table out several ways - ``X,Y,Z,Global ID``, a leading or
    trailing electrode number, plain ``x z`` - and reading columns by position
    took an ID column as the elevation, or the electrode number as the position,
    without a word. A header decides when there is one. Without one, an
    electrode-number column (1..n) is set aside, and a layout that still cannot
    be told apart is refused rather than guessed. The elevation is the column a
    header names as such; otherwise, of y and z, the one that varies: PyGIMLi
    writes a 2-D line with its elevation in y, and ``ert_io.electrode_elevation``
    reads a loaded survey by the same rule.
    """
    source = Path(path)
    if source.suffix.lower() in _ARRAY_SUFFIXES:
        table = load_xyz_table(source, min_cols=2)
    else:
        # Electrode files come as .xyz, .elec and the like; they are text tables.
        table = np.atleast_2d(_load_text_matrix(source))
        if table.shape[1] < 2:
            raise ValueError(f"'{source.name}' needs at least two columns: x and the elevation.")
    rows, cols = table.shape
    name = source.name
    header = table_header(source)
    if header is not None and len(header) == cols:
        keys = []
        for label in header:
            words = label.lower().replace("(", " ").replace("[", " ").replace("_", " ").split()
            keys.append(words[0] if words else "")
        x_col = keys.index("x") if "x" in keys else None
        y_col = keys.index("y") if "y" in keys else None
        z_col = next((keys.index(key) for key in _ELEVATION_NAMES if key in keys), None)
        if x_col is None or (y_col is None and z_col is None):
            raise ValueError(
                f"{name}: none of its columns ({', '.join(header)}) reads as x and "
                "the elevation. Name them x and z (or elevation) in the header row.")
        # A named elevation is the elevation; a table of x and y alone is a 2-D
        # line written the PyGIMLi way, with y the vertical.
        height = z_col if z_col is not None else y_col
        picked = [x_col, y_col, z_col] if (y_col is not None and z_col is not None) else [x_col, height]
        return (table[:, picked].copy(), table[:, height].copy(),
                f"x from '{header[x_col]}', elevation from '{header[height]}'")

    def numbering(col: int) -> bool:
        return rows >= 3 and np.array_equal(table[:, col], np.arange(1, rows + 1))

    first, last = numbering(0), numbering(cols - 1)
    if cols == 3 and first and np.ptp(table[:, 1]) == 0:
        # An "x" after the number that never changes would put every electrode
        # in one place: the first column was x, at 1 m spacing.
        first = False
    picked = None
    if cols == 2:
        picked = [0, 1]
    elif cols in (3, 4) and not (first and last):
        rest = [c for c in range(cols)
                if not ((c == 0 and first) or (c == cols - 1 and last))]
        if len(rest) in (2, 3):              # x and elevation, or x y z
            picked = rest
    if picked is None:
        raise ValueError(
            f"{name}: {cols} columns and no header row, so which of them are x "
            "and the elevation cannot be told. Add a header naming them "
            "(x, y, z or elevation).")
    coords = table[:, picked].copy()
    if coords.shape[1] == 2:
        height = 1
    else:
        height = 1 if np.std(coords[:, 1]) >= np.std(coords[:, 2]) else 2
    columns = [c + 1 for c in picked]
    how = ("x, elevation" if picked == [0, 1] else
           f"x from column {columns[0]}, elevation from column {columns[height]}")
    return coords, coords[:, height].copy(), how


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
