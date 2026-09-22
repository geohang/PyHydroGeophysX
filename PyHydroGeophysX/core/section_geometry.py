"""Geometry of a 2-D geophysical section: ground surface, cell depths, clip masks.

pyGIMLi draws an inverted model on the whole parameter mesh, so the outer edges of
the picture are regularization rather than data: the mesh reaches past the last
electrode and below anything the survey resolves. "Clipping" a section - the
familiar trapezoid of a traditional resistivity image - means trimming it back to
the part the measurements actually constrain.

A per-cell coverage cut does that already, but it cuts cell by cell, so the result
is a ragged edge with isolated islands hanging off it. :func:`coverage_envelope_mask`
turns the same cut into a smooth depth envelope, which is what gives the section a
clean cut rather than a fringe of surviving cells.

The helpers work on any object exposing pyGIMLi's mesh interface (``cellCenters``,
``positions``, ``boundaries``); pyGIMLi itself is never imported here, so the module
stays importable without a geophysics backend.

In a 2-D pyGIMLi mesh the profile coordinate is x and the elevation is y, so every
array below is ``(x, elevation)`` and depth is measured downward from the
interpolated ground surface.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

__all__ = [
    "cell_centers",
    "cell_bottom_depths",
    "surface_line",
    "surface_elevation_at",
    "cell_depths",
    "coverage_mask",
    "coverage_envelope",
    "coverage_envelope_mask",
    "coverage_envelope_polygon",
    "envelope_polygon",
]

#: Boundary marker pyGIMLi puts on the free surface of a 2-D mesh.
_SURFACE_MARKER = -1


def _as_xz(values: Any) -> np.ndarray:
    """Return an ``(n, 2)`` array of ``(x, elevation)`` from anything array-like."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError("expected an (n, >=2) array of positions")
    return np.ascontiguousarray(arr[:, :2])


def cell_centers(mesh: Any) -> np.ndarray:
    """``(n_cells, 2)`` array of cell-centre ``(x, elevation)``."""
    return _as_xz(np.asarray(mesh.cellCenters(), dtype=float))


def _node_positions(mesh: Any) -> Optional[np.ndarray]:
    try:
        return _as_xz(np.asarray(mesh.positions(), dtype=float))
    except Exception:  # noqa: BLE001 - fall back to walking the nodes
        pass
    try:
        return np.asarray([[n.x(), n.y()] for n in mesh.nodes()], dtype=float)
    except Exception:  # noqa: BLE001 - not a mesh we can read
        return None


def _surface_boundary_nodes(mesh: Any) -> Optional[np.ndarray]:
    """Nodes of the boundaries pyGIMLi marks as the free surface, if any."""
    try:
        boundaries = list(mesh.boundaries())
    except Exception:  # noqa: BLE001 - mesh without boundary access
        return None
    points = []
    for boundary in boundaries:
        try:
            if int(boundary.marker()) != _SURFACE_MARKER:
                continue
            for node in boundary.nodes():
                points.append((node.x(), node.y()))
        except Exception:  # noqa: BLE001 - skip anything unreadable
            continue
    if len(points) < 2:
        return None
    return np.asarray(points, dtype=float)


def _upper_envelope(points: np.ndarray,
                    n_columns: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Highest point per x, as a polyline sorted along the profile.

    With ``n_columns`` the points are binned first, which is what makes this work
    on a triangular mesh: interior nodes sit at x values no surface node shares, so
    grouping on the exact coordinate would let an interior node define the surface.
    Within a column the topmost node is on the surface by construction.
    """
    pts = np.asarray(points, dtype=float)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] < 2:
        raise ValueError("need at least two finite points for a surface line")
    x, z = pts[:, 0], pts[:, 1]
    if n_columns is None:
        keys = x
    else:
        lo, hi = float(x.min()), float(x.max())
        width = (hi - lo) or 1.0
        keys = np.clip(((x - lo) / width * n_columns).astype(int), 0, n_columns - 1)
    order = np.lexsort((-z, keys))          # per key, the highest point first
    keys_sorted = keys[order]
    first = np.ones(keys_sorted.size, dtype=bool)
    first[1:] = keys_sorted[1:] != keys_sorted[:-1]
    top = order[first]
    sx, sz = x[top], z[top]
    order_x = np.argsort(sx)
    return sx[order_x], sz[order_x]


def surface_line(mesh: Any = None, sensors: Any = None,
                 n_columns: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """``(x, elevation)`` polyline of the ground surface above the section.

    Electrode positions are the best answer when the caller has them - they are
    where the topography was actually measured. Otherwise the surface is read off
    the mesh: pyGIMLi's free-surface boundary marker when the mesh carries it, and
    the upper envelope of the nodes when it does not (a parameter domain extracted
    from an inversion has usually lost its markers).
    """
    if sensors is not None:
        try:
            pts = _as_xz(sensors)
            if pts.shape[0] >= 2:
                return _upper_envelope(pts)
        except Exception:  # noqa: BLE001 - fall through to the mesh
            pass
    if mesh is None:
        raise ValueError("surface_line needs a mesh or at least two sensor positions")
    marked = _surface_boundary_nodes(mesh)
    if marked is not None:
        return _upper_envelope(marked)
    nodes = _node_positions(mesh)
    if nodes is None or nodes.shape[0] < 2:
        raise ValueError("could not read node positions from the mesh")
    columns = n_columns or int(np.clip(np.sqrt(nodes.shape[0]), 16, 200))
    return _upper_envelope(nodes, n_columns=int(columns))


def surface_elevation_at(x: Any, surface: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Ground elevation at each ``x``, linearly interpolated along ``surface``."""
    sx, sz = surface
    return np.interp(np.asarray(x, dtype=float), np.asarray(sx, dtype=float),
                     np.asarray(sz, dtype=float))


def cell_depths(mesh: Any, sensors: Any = None,
                surface: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> np.ndarray:
    """Depth of every cell centre below the ground surface (positive downward)."""
    centers = cell_centers(mesh)
    line = surface if surface is not None else surface_line(mesh, sensors)
    return surface_elevation_at(centers[:, 0], line) - centers[:, 1]


def cell_bottom_depths(mesh: Any, sensors: Any = None,
                       surface: Optional[Tuple[np.ndarray, np.ndarray]] = None
                       ) -> np.ndarray:
    """Depth of the lowest node of every cell, below the ground surface.

    The bottom of a cell, not its centre, is how deep a resolved cell actually
    reaches. On the coarse triangles an inversion mesh carries at depth the two
    differ by tens of metres, and an envelope built from centres cuts visibly
    above the cells it is meant to keep.
    """
    line = surface if surface is not None else surface_line(mesh, sensors)
    try:
        cells = list(mesh.cells())
        bottoms = []
        for cell in cells:
            nodes = [(n.x(), n.y()) for n in cell.nodes()]
            xs = np.asarray([p[0] for p in nodes], dtype=float)
            zs = np.asarray([p[1] for p in nodes], dtype=float)
            bottoms.append(float(np.max(surface_elevation_at(xs, line) - zs)))
        if len(bottoms) == int(mesh.cellCount()):
            return np.asarray(bottoms, dtype=float)
    except Exception:  # noqa: BLE001 - not a mesh we can walk; centres will do
        pass
    return cell_depths(mesh, sensors=sensors, surface=line)


def coverage_mask(coverage: Any, threshold: float) -> np.ndarray:
    """Per-cell keep/drop mask from a coverage array and a cut.

    A boolean coverage is already the answer (travel-time ray coverage is a yes or
    no); a float array is log10 cumulative sensitivity and is compared with the cut.
    """
    arr = np.asarray(coverage)
    if arr.dtype == bool:
        return arr
    values = arr.astype(float)
    with np.errstate(invalid="ignore"):
        return np.isfinite(values) & (values >= float(threshold))


def coverage_envelope(
    mesh: Any,
    coverage: Any,
    threshold: float = -2.0,
    *,
    sensors: Any = None,
    n_columns: Optional[int] = None,
    smooth_columns: int = 5,
    quantile: float = 1.0,
) -> Optional[Dict[str, Any]]:
    """Reduce a coverage cut to one clipping depth per lateral column.

    :func:`coverage_mask` answers "is this cell resolved?" one cell at a time, which
    leaves a ragged edge and isolated deep islands. The envelope answers "how deep
    does the survey see here?" instead - a single smooth depth profile along the
    line, which is what a traditionally clipped resistivity image is bounded by.

    Args:
        mesh: pyGIMLi mesh the values live on.
        coverage: per-cell coverage (log10 sensitivity) or a boolean mask.
        threshold: coverage cut, in the units of ``coverage``.
        sensors: optional electrode positions defining the topography.
        n_columns: lateral columns the envelope is built from. Defaults to a count
            scaled to the number of surviving cells.
        smooth_columns: width of the median + moving-average smoothing applied to
            the depth profile. ``<= 1`` disables smoothing.
        quantile: quantile of the surviving depths taken as each column's clipping
            depth. 1.0 follows the deepest surviving cell; lower values cut back to
            a more conservative envelope.

    Returns:
        ``{"x", "depth", "surface", "x_range", "mask"}`` - the column centres, the
        clipping depth at each, the surface polyline, the lateral extent and the
        per-cell coverage mask it was built from. ``None`` when the cut keeps
        everything or nothing, or the geometry cannot be read.
    """
    raw = np.asarray(coverage_mask(coverage, threshold), dtype=bool).ravel()
    if not raw.any() or raw.all():
        return None
    try:
        centers = cell_centers(mesh)
        line = surface_line(mesh, sensors)
        depth = cell_bottom_depths(mesh, surface=line)
    except Exception:  # noqa: BLE001 - no geometry to build an envelope from
        return None
    if centers.shape[0] != raw.size or depth.size != raw.size:
        return None

    x = centers[:, 0]
    x_lo, x_hi = float(x[raw].min()), float(x[raw].max())
    if not np.isfinite([x_lo, x_hi]).all() or x_hi <= x_lo:
        return None

    columns = int(np.clip(n_columns or np.sqrt(int(raw.sum())) * 2.0, 12, 120))
    edges = np.linspace(x_lo, x_hi, columns + 1)
    index = np.clip(np.digitize(x, edges) - 1, 0, columns - 1)
    q = float(np.clip(quantile, 0.0, 1.0))

    limit = np.full(columns, np.nan)
    for column in range(columns):
        selected = depth[raw & (index == column)]
        if selected.size:
            limit[column] = np.quantile(selected, q)
    if not np.isfinite(limit).any():
        return None

    return {
        "x": 0.5 * (edges[:-1] + edges[1:]),
        "depth": _fill_and_smooth(limit, smooth_columns),
        "surface": line,
        "x_range": (x_lo, x_hi),
        "mask": raw,
    }


def coverage_envelope_mask(mesh: Any, coverage: Any, threshold: float = -2.0,
                           **kwargs: Any) -> np.ndarray:
    """Keep-mask clipped to the coverage envelope rather than cell by cell.

    Everything above the envelope is kept as a solid shape, everything below is
    blanked. Use this where the values themselves have to be masked - exported
    arrays, statistics over the resolved part. For a figure, prefer
    :func:`coverage_envelope_polygon` with a matplotlib clip path: a mask can only
    ever cut on cell boundaries, so on the coarse triangles at depth it still
    leaves a saw-tooth edge.
    """
    envelope = coverage_envelope(mesh, coverage, threshold, **kwargs)
    raw = np.asarray(coverage_mask(coverage, threshold), dtype=bool).ravel()
    if envelope is None:
        return raw
    centers = cell_centers(mesh)
    x = centers[:, 0]
    depth = surface_elevation_at(x, envelope["surface"]) - centers[:, 1]
    x_lo, x_hi = envelope["x_range"]
    limit = np.interp(x, envelope["x"], envelope["depth"])
    return (depth <= limit) & (x >= x_lo) & (x <= x_hi)


def coverage_envelope_polygon(mesh: Any, coverage: Any, threshold: float = -2.0,
                              **kwargs: Any) -> Optional[np.ndarray]:
    """Closed ``(x, elevation)`` polygon bounding the resolved part of the section."""
    return envelope_polygon(coverage_envelope(mesh, coverage, threshold, **kwargs))


def envelope_polygon(envelope: Optional[Dict[str, Any]]) -> Optional[np.ndarray]:
    """Closed ``(x, elevation)`` polygon around an envelope from :func:`coverage_envelope`.

    Ground surface on top, coverage envelope underneath. Handed to matplotlib as a
    clip path it cuts the drawn image exactly on that line, independently of where
    the mesh happens to put its cell boundaries - the traditional clipped section,
    with a clean edge instead of a fringe of surviving triangles.
    """
    if envelope is None:
        return None
    line = envelope["surface"]
    x_lo, x_hi = envelope["x_range"]
    # Sample the top on the envelope columns and on every surface vertex between
    # them, so a topographic break is not rounded off by the column spacing.
    surface_x = np.asarray(line[0], dtype=float)
    inner = surface_x[(surface_x > x_lo) & (surface_x < x_hi)]
    top_x = np.unique(np.concatenate(([x_lo], envelope["x"], inner, [x_hi])))
    top_z = surface_elevation_at(top_x, line)
    bottom_x = np.unique(np.concatenate(([x_lo], envelope["x"], [x_hi])))
    bottom_z = surface_elevation_at(bottom_x, line) - np.interp(
        bottom_x, envelope["x"], envelope["depth"])
    polygon = np.vstack([
        np.column_stack([top_x, top_z]),
        np.column_stack([bottom_x[::-1], bottom_z[::-1]]),
    ])
    return np.vstack([polygon, polygon[:1]])


def _fill_and_smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Interpolate over empty columns, then median- and mean-smooth the profile."""
    out = np.asarray(values, dtype=float).copy()
    known = np.isfinite(out)
    idx = np.arange(out.size)
    out = np.interp(idx, idx[known], out[known])
    width = int(window)
    if width <= 1:
        return out
    # Median first: it removes the single deep column that one stray well-covered
    # cell creates, which a moving average would only smear into its neighbours.
    half = max(1, width // 2)
    padded = np.pad(out, half, mode="edge")
    median = np.asarray([np.median(padded[i:i + 2 * half + 1]) for i in range(out.size)])
    kernel = np.ones(2 * half + 1) / float(2 * half + 1)
    return np.convolve(np.pad(median, half, mode="edge"), kernel, mode="valid")
