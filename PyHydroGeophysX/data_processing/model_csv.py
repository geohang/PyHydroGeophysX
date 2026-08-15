"""Flat CSV views of mesh-based inversion models.

Every 2-D inversion in this package stores its result as a PyGIMLi mesh plus a
value vector ordered by cell.  That pair is exact, but reading it back needs
PyGIMLi installed and the knowledge that the ordering is by ``paraDomain`` cell.
A collaborator who only wants to redraw the section in matplotlib, R, or a GIS
should not have to acquire either.

The tables written here carry the coordinate on every row, so a section can be
redrawn with nothing but the file:

    import pandas as pd, matplotlib.pyplot as plt
    d = pd.read_csv("model_cells.csv")
    plt.tricontourf(d.x, d.z, d.resistivity_ohm_m)

``mesh_nodes.csv`` and ``mesh_cell_nodes.csv`` are written alongside for anyone
who wants the true cell polygons rather than an interpolation through the
centroids.  They are optional to read and cost one pass over the mesh to write.

Coordinates follow the convention used elsewhere in this package: a 2-D PyGIMLi
mesh stores the section plane in its first two components, so those are written
as ``x`` and ``z`` (elevation, positive up) and no ``y`` column appears.  A 3-D
mesh writes all three.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from PyHydroGeophysX.data_processing.table_io import ensure_dir, write_csv

PathLike = Union[str, Path]

CELLS_FILENAME = "model_cells.csv"
NODES_FILENAME = "mesh_nodes.csv"
CELL_NODES_FILENAME = "mesh_cell_nodes.csv"

__all__ = [
    "CELLS_FILENAME",
    "CELL_NODES_FILENAME",
    "NODES_FILENAME",
    "column_name",
    "export_model_csv",
    "model_cell_table",
    "write_grid_model_csv",
    "write_layered_model_csv",
    "write_mesh_geometry_csv",
    "write_model_csv",
]

#: Ten significant digits keeps float32 results exact and float64 results
#: faithful, without writing the repr noise that makes 0.1 + 0.2 unreadable.
_PRECISION = ".10g"


def _num(value: Any) -> str:
    """Format one number for a table; a non-finite value becomes an empty field."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    return format(number, _PRECISION) if np.isfinite(number) else ""


def column_name(value_name: str, units: str = "") -> str:
    """Build a self-describing column header, e.g. ``resistivity_ohm_m``.

    The unit belongs in the header rather than in a sidecar, because a table
    that travels on its own is the point of writing it.
    """
    name = str(value_name or "value").strip().replace(" ", "_")
    unit = str(units or "").strip()
    if not unit:
        return name
    spelled = unit.replace("Ω", "ohm").replace("μ", "u").replace("/", "_per_")
    # Every remaining separator becomes an underscore rather than vanishing:
    # dropping the dot in "ohm.m" would spell the unit "ohmm".
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in spelled)
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    cleaned = cleaned.strip("_")
    return f"{name}_{cleaned}" if cleaned and cleaned not in name else name


def _cell_centres(mesh: Any) -> Tuple[np.ndarray, int]:
    """Return ``(centres, dimension)`` with centres always shaped ``(n_cells, 3)``."""
    centres = np.atleast_2d(np.asarray(mesh.cellCenters(), dtype=float))
    if centres.ndim != 2 or centres.shape[1] < 2:
        raise ValueError(
            f"Mesh cell centres have unusable shape {centres.shape}; expected (n_cells, 2 or 3)."
        )
    if centres.shape[1] == 2:
        centres = np.column_stack((centres, np.zeros(len(centres))))
    return centres[:, :3], int(getattr(mesh, "dim", lambda: 2)())


def _as_cell_columns(values: Any, n_cells: int, what: str) -> np.ndarray:
    """Coerce a model/coverage array to ``(n_cells, n_columns)``.

    A time-lapse result is stored ``(n_cells, n_steps)`` by some callers and
    ``(n_steps, n_cells)`` by others, so the orientation is decided by whichever
    axis matches the mesh rather than by the caller remembering which is which.
    """
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    elif array.ndim != 2:
        raise ValueError(f"{what} must be 1-D or 2-D, got shape {array.shape}.")
    if array.shape[0] == n_cells:
        return array
    if array.shape[1] == n_cells:
        return array.T
    raise ValueError(
        f"{what} has shape {array.shape}, which does not match the mesh's "
        f"{n_cells} cells on either axis."
    )


def _step_headers(base: str, n_columns: int, step_labels: Optional[Sequence[Any]]) -> List[str]:
    if n_columns == 1 and not step_labels:
        return [base]
    if step_labels is not None and len(step_labels) >= n_columns:
        seen: Dict[str, int] = {}
        headers = []
        for label in list(step_labels)[:n_columns]:
            cleaned = "".join(
                ch if (ch.isalnum() or ch == "_") else "_" for ch in str(label).strip()
            ).strip("_") or "step"
            # Two survey dates that clean to the same token would otherwise
            # produce duplicate headers and a silently unreadable table.
            seen[cleaned] = seen.get(cleaned, 0) + 1
            if seen[cleaned] > 1:
                cleaned = f"{cleaned}_{seen[cleaned]}"
            headers.append(f"{base}_{cleaned}")
        return headers
    width = max(2, len(str(n_columns - 1)))
    return [f"{base}_t{index:0{width}d}" for index in range(n_columns)]


def model_cell_table(
    mesh: Any,
    values: Any,
    *,
    value_name: str = "value",
    units: str = "",
    coverage: Any = None,
    step_labels: Optional[Sequence[Any]] = None,
    extra_columns: Optional[Dict[str, Any]] = None,
) -> Tuple[List[str], List[List[str]]]:
    """Build the header and rows of the per-cell table without writing anything.

    Exposed separately so callers can add a column or inspect the table in a
    test without going through the filesystem.
    """
    centres, dimension = _cell_centres(mesh)
    n_cells = len(centres)
    if int(mesh.cellCount()) != n_cells:
        raise ValueError(
            f"Mesh reports {int(mesh.cellCount())} cells but returned {n_cells} centres."
        )

    model = _as_cell_columns(values, n_cells, "The model")
    value_base = column_name(value_name, units)
    value_headers = _step_headers(value_base, model.shape[1], step_labels)

    coverage_headers: List[str] = []
    coverage_columns: Optional[np.ndarray] = None
    if coverage is not None:
        coverage_columns = _as_cell_columns(coverage, n_cells, "The coverage")
        coverage_headers = _step_headers(
            "coverage", coverage_columns.shape[1],
            step_labels if coverage_columns.shape[1] == model.shape[1] else None,
        )

    markers = np.asarray(mesh.cellMarkers(), dtype=int)
    try:
        sizes = np.asarray(mesh.cellSizes(), dtype=float)
    except Exception:  # noqa: BLE001 - cell size is a convenience, not the result
        sizes = np.full(n_cells, np.nan)
    size_header = "cell_volume_m3" if dimension == 3 else "cell_area_m2"

    extras: Dict[str, np.ndarray] = {}
    for name, column in (extra_columns or {}).items():
        flat = np.asarray(column, dtype=float).ravel()
        if flat.size != n_cells:
            raise ValueError(
                f"Extra column {name!r} has {flat.size} values for {n_cells} cells."
            )
        extras[str(name)] = flat

    coordinate_headers = ["x", "y", "z"] if dimension == 3 else ["x", "z"]
    header = (
        ["cell"] + coordinate_headers + ["marker", size_header]
        + value_headers + coverage_headers + list(extras)
    )

    rows: List[List[str]] = []
    for index in range(n_cells):
        if dimension == 3:
            coordinates = [_num(centres[index, 0]), _num(centres[index, 1]),
                           _num(centres[index, 2])]
        else:
            # A 2-D PyGIMLi mesh carries the section plane in components 0 and 1.
            coordinates = [_num(centres[index, 0]), _num(centres[index, 1])]
        row = [str(index)] + coordinates + [str(int(markers[index])), _num(sizes[index])]
        row += [_num(item) for item in model[index]]
        if coverage_columns is not None:
            row += [_num(item) for item in coverage_columns[index]]
        row += [_num(column[index]) for column in extras.values()]
        rows.append(row)
    return header, rows


def write_model_csv(
    path: PathLike,
    mesh: Any,
    values: Any,
    *,
    value_name: str = "value",
    units: str = "",
    coverage: Any = None,
    step_labels: Optional[Sequence[Any]] = None,
    extra_columns: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write one row per mesh cell: coordinate, marker, size, and value(s)."""
    header, rows = model_cell_table(
        mesh, values,
        value_name=value_name, units=units, coverage=coverage,
        step_labels=step_labels, extra_columns=extra_columns,
    )
    return write_csv(path, rows, header=header)


def write_mesh_geometry_csv(out_dir: PathLike, mesh: Any) -> List[Path]:
    """Write the node coordinates and each cell's node ids.

    Together with ``model_cells.csv`` this is enough to draw the true cell
    polygons, which a centroid interpolation cannot reproduce near the
    topography or along a sharp region boundary.
    """
    out = ensure_dir(out_dir)
    positions = np.atleast_2d(np.asarray(mesh.positions(), dtype=float))
    if positions.shape[1] == 2:
        positions = np.column_stack((positions, np.zeros(len(positions))))
    dimension = int(getattr(mesh, "dim", lambda: 2)())
    if dimension == 3:
        node_header = ["node", "x", "y", "z"]
        node_rows = [
            [str(index), _num(row[0]), _num(row[1]), _num(row[2])]
            for index, row in enumerate(positions)
        ]
    else:
        node_header = ["node", "x", "z"]
        node_rows = [
            [str(index), _num(row[0]), _num(row[1])]
            for index, row in enumerate(positions)
        ]
    written = [write_csv(out / NODES_FILENAME, node_rows, header=node_header)]

    cell_ids = [[int(value) for value in mesh.cell(index).ids()]
                for index in range(int(mesh.cellCount()))]
    widest = max((len(ids) for ids in cell_ids), default=0)
    if widest:
        markers = np.asarray(mesh.cellMarkers(), dtype=int)
        cell_header = ["cell", "marker"] + [f"node_{index}" for index in range(widest)]
        # A mixed triangle/quad mesh pads the short rows rather than dropping
        # them, so the cell index stays aligned with model_cells.csv.
        cell_rows = [
            [str(index), str(int(markers[index]))]
            + [str(value) for value in ids] + [""] * (widest - len(ids))
            for index, ids in enumerate(cell_ids)
        ]
        written.append(write_csv(out / CELL_NODES_FILENAME, cell_rows, header=cell_header))
    return written


def write_grid_model_csv(
    path: PathLike,
    edges: Sequence[Any],
    values: Any,
    *,
    value_name: str = "value",
    units: str = "",
    order: str = "F",
) -> Path:
    """Write a regular 3-D grid model as one row per cell.

    The gravity, magnetics, and joint potential-field inversions solve on a
    rectilinear grid rather than a PyGIMLi mesh, so they cannot use
    :func:`write_model_csv`.  ``edges`` is the three edge vectors in x, y, z, and
    ``values`` is shaped ``(nx, ny, nz)`` or flat in ``order`` (SimPEG's tensor
    meshes are Fortran-ordered, which is why that is the default).

    Cell extents are written alongside the centre, so a reader can draw the true
    voxels instead of guessing the spacing back from the centres.
    """
    parsed = [np.asarray(edge, dtype=float).ravel() for edge in edges]
    if len(parsed) != 3:
        raise ValueError(f"A 3-D grid needs three edge vectors, got {len(parsed)}.")
    shape = tuple(int(edge.size - 1) for edge in parsed)
    if min(shape) < 1:
        raise ValueError(f"Edge vectors define an empty grid: shape {shape}.")

    model = np.asarray(values, dtype=float)
    if model.ndim == 1:
        if model.size != int(np.prod(shape)):
            raise ValueError(
                f"Flat model has {model.size} values for a {shape} grid "
                f"({int(np.prod(shape))} cells)."
            )
        model = model.reshape(shape, order=order)
    elif model.shape != shape:
        raise ValueError(f"Model shape {model.shape} does not match grid shape {shape}.")

    centres = [0.5 * (edge[:-1] + edge[1:]) for edge in parsed]
    header = [
        "cell", "i", "j", "k", "x", "y", "z",
        "x_min", "x_max", "y_min", "y_max", "z_min", "z_max",
        column_name(value_name, units),
    ]
    rows: List[List[str]] = []
    index = 0
    # C order over (i, j, k) so the file reads as a stack of x-rows, whatever
    # order the solver used internally.
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                rows.append([
                    str(index), str(i), str(j), str(k),
                    _num(centres[0][i]), _num(centres[1][j]), _num(centres[2][k]),
                    _num(parsed[0][i]), _num(parsed[0][i + 1]),
                    _num(parsed[1][j]), _num(parsed[1][j + 1]),
                    _num(parsed[2][k]), _num(parsed[2][k + 1]),
                    _num(model[i, j, k]),
                ])
                index += 1
    return write_csv(path, rows, header=header)


def write_layered_model_csv(
    path: PathLike,
    thicknesses: Any,
    values: Any,
    *,
    value_name: str = "value",
    units: str = "",
    positions: Optional[Sequence[Any]] = None,
) -> Path:
    """Write a 1-D layered model as one row per layer per sounding.

    The bottom layer of a layered inversion is a half-space, so its lower depth
    is left empty rather than invented.
    """
    thickness = np.asarray(thicknesses, dtype=float).ravel()
    model = np.asarray(values, dtype=float)
    if model.ndim == 1:
        model = model.reshape(1, -1)
    elif model.ndim != 2:
        raise ValueError(f"A layered model must be 1-D or 2-D, got shape {model.shape}.")
    n_soundings, n_layers = model.shape
    if thickness.size < n_layers - 1:
        raise ValueError(
            f"{thickness.size} thicknesses cannot describe {n_layers} layers; "
            f"expected at least {n_layers - 1} (the last layer is a half-space)."
        )
    top = np.r_[0.0, np.cumsum(thickness)][:n_layers]
    bottom = np.r_[top[1:], np.inf][:n_layers]

    labels = (
        [str(item) for item in list(positions)[:n_soundings]]
        if positions is not None and len(positions) >= n_soundings
        else [str(index) for index in range(n_soundings)]
    )
    header = ["sounding", "layer", "depth_top_m", "depth_bottom_m",
              column_name(value_name, units)]
    rows = [
        [labels[sounding], str(layer), _num(top[layer]), _num(bottom[layer]),
         _num(model[sounding, layer])]
        for sounding in range(n_soundings)
        for layer in range(n_layers)
    ]
    return write_csv(path, rows, header=header)


def export_model_csv(
    out_dir: PathLike,
    mesh: Any,
    values: Any,
    *,
    value_name: str = "value",
    units: str = "",
    coverage: Any = None,
    step_labels: Optional[Sequence[Any]] = None,
    extra_columns: Optional[Dict[str, Any]] = None,
    geometry: bool = True,
) -> List[str]:
    """Write the per-cell table and, unless disabled, the mesh geometry.

    Returns the paths written, in the order a reader should meet them.
    """
    out = ensure_dir(out_dir)
    written = [str(write_model_csv(
        out / CELLS_FILENAME, mesh, values,
        value_name=value_name, units=units, coverage=coverage,
        step_labels=step_labels, extra_columns=extra_columns,
    ))]
    if geometry:
        written += [str(path) for path in write_mesh_geometry_csv(out, mesh)]
    return written
