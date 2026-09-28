"""Hydro profiles: sampling the model grid along a line, and meshing the result.

The steps between a hydrological model and a 2D geophysical forward run, shared
by the studio's hydro page (:mod:`PyHydroGeophysX.Hydro_modular.hydro_to_geophysics`,
which re-exports them) and the Streamlit hydro page, which carried copies.
Sampling needs only NumPy and SciPy; :func:`build_profile_mesh` imports the mesh
tools, and so PyGIMLi, when it is called.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import numpy as np


def fill_profile_nans(values: Any) -> np.ndarray:
    """Fill NaNs along the profile direction for each layer.

    A layer with no finite value anywhere on the profile (one that is inactive
    under the whole line) is filled from the nearest layers that have values:
    linearly between the one above and the one below, or copied from the only
    one there is. The Streamlit hydro page did this and the studio raised, so
    the same profile could be modelled on one and not the other.

    Raises
    ------
    RuntimeError
        When no layer has a finite value.

    Examples
    --------
    >>> fill_profile_nans([[1.0, np.nan, 3.0], [np.nan] * 3, [5.0, 5.0, np.nan]]).tolist()
    [[1.0, 2.0, 3.0], [3.0, 3.5, 4.0], [5.0, 5.0, 5.0]]
    """
    arr = np.asarray(values, dtype=float).copy()
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {arr.shape}.")

    x = np.arange(arr.shape[1], dtype=float)
    valid_row_idx = []
    for i in range(arr.shape[0]):
        row = arr[i, :]
        valid = np.isfinite(row)
        if not np.any(valid):
            continue            # filled from its neighbours below
        if np.count_nonzero(valid) == 1:
            row[~valid] = row[valid][0]
        else:
            row[~valid] = np.interp(x[~valid], x[valid], row[valid])
        valid_row_idx.append(i)
        arr[i, :] = row

    if not valid_row_idx:
        raise RuntimeError("Profile interpolation failed: all layers are NaN along this profile.")

    valid_rows = np.asarray(valid_row_idx, dtype=int)
    for i in range(arr.shape[0]):
        if np.any(np.isfinite(arr[i, :])):
            continue
        lower = valid_rows[valid_rows < i]
        upper = valid_rows[valid_rows > i]
        if lower.size and upper.size:
            lo, hi = int(lower[-1]), int(upper[0])
            weight = (i - lo) / float(hi - lo)
            arr[i, :] = (1.0 - weight) * arr[lo, :] + weight * arr[hi, :]
        elif lower.size:
            arr[i, :] = arr[int(lower[-1]), :]
        else:
            arr[i, :] = arr[int(upper[0]), :]
    return arr


def get_mesh_xy(mesh: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Return mesh cell-center x/y arrays."""
    centers = np.asarray(mesh.cellCenters(), dtype=float)
    if centers.ndim == 2 and centers.shape[1] >= 2:
        return centers[:, 0], centers[:, 1]
    x = np.array([float(c[0]) for c in mesh.cellCenters()], dtype=float)
    y = np.array([float(c[1]) for c in mesh.cellCenters()], dtype=float)
    return x, y


def assign_three_layer_markers(
    mesh: Any,
    line1: np.ndarray,
    line2: np.ndarray,
    top_marker: int = 0,
    mid_marker: int = 3,
    bot_marker: int = 2,
) -> np.ndarray:
    """Assign top/middle/bottom markers from two interface lines."""
    x_cell, y_cell = get_mesh_xy(mesh)
    y_line1 = np.interp(x_cell, line1[:, 0], line1[:, 1])
    y_line2 = np.interp(x_cell, line2[:, 0], line2[:, 1])
    markers = np.full(mesh.cellCount(), bot_marker, dtype=int)
    markers[y_cell >= y_line2] = mid_marker
    markers[y_cell >= y_line1] = top_marker
    mesh.setCellMarkers(markers)
    return markers


def interpolate_layer_samples(points: Any, values: Any, query: Any) -> np.ndarray:
    """Values at ``query`` from samples at layer centres: linear, nearest outside them.

    Linear interpolation needs a triangulation of the samples, and there is none
    when they lie on one line: a single layer between flat or evenly sloping
    boundaries, or a single station. SciPy then raised ``QhullError`` ("initial
    simplex is flat") before the nearest-neighbour fallback could run, so such
    a model could not be forward modelled at all. Collinear samples are
    interpolated along their line instead, holding the end values beyond it.

    Parameters
    ----------
    points : ndarray
        ``(n, 2)`` sample positions.
    values : ndarray
        ``(n,)`` sample values.
    query : ndarray
        ``(m, 2)`` positions to evaluate.

    Examples
    --------
    >>> pts = np.array([[0.0, -5.0], [5.0, -5.0], [10.0, -5.0]])   # one flat layer
    >>> out = interpolate_layer_samples(pts, [0.1, 0.2, 0.4], [[2.5, -1.0], [20.0, -9.0]])
    >>> np.round(out, 6).tolist()
    [0.15, 0.4]
    """
    from scipy.interpolate import griddata
    from scipy.spatial import QhullError

    points = np.asarray(points, dtype=float)
    values = np.asarray(values, dtype=float).ravel()
    query = np.asarray(query, dtype=float)
    try:
        out = np.asarray(griddata(points, values, query, method="linear"), dtype=float)
    except QhullError:
        centre = points.mean(axis=0)
        # The direction the samples spread along; any direction when they coincide.
        direction = np.linalg.svd(points - centre, full_matrices=False)[2][0]
        along = (points - centre) @ direction
        order = np.argsort(along, kind="stable")
        return np.interp((query - centre) @ direction, along[order], values[order])
    nan_mask = ~np.isfinite(out)
    if nan_mask.any():
        out[nan_mask] = griddata(points, values, query[nan_mask], method="nearest")
    return out


def interpolate_profile_to_mesh(
    profile_values: Any,
    layer_boundaries: Any,
    x_profile: Any,
    mesh: Any,
) -> np.ndarray:
    """Interpolate a (layers x distance) profile matrix onto mesh cells."""
    values = np.asarray(profile_values, dtype=float)
    bounds = np.asarray(layer_boundaries, dtype=float)
    n_layers, n_profile = values.shape
    if bounds.shape != (n_layers + 1, n_profile):
        raise ValueError(
            f"layer_boundaries shape must be {(n_layers + 1, n_profile)}, got {bounds.shape}."
        )
    layer_centers = 0.5 * (bounds[:-1, :] + bounds[1:, :])
    x2d = np.repeat(np.asarray(x_profile, dtype=float)[np.newaxis, :], n_layers, axis=0)
    points = np.column_stack((x2d.ravel(), layer_centers.ravel()))
    x_cell, y_cell = get_mesh_xy(mesh)
    return interpolate_layer_samples(points, values.ravel(), np.column_stack((x_cell, y_cell)))


def sample_profile(
    water_content_3d: Any,
    porosity_3d: Any,
    top: Any,
    bot: Any,
    point1: Sequence[float],
    point2: Sequence[float],
    num_points: int = 220,
    *,
    clip_to_grid: bool = False,
    fallback_to_active: bool = False,
) -> Dict[str, Any]:
    """Sample the hydro arrays along the straight line between two grid points.

    Used by :func:`extract_profile` and by the Streamlit hydro page, which
    carried its own copy with two extras, kept here as options.

    Parameters
    ----------
    water_content_3d, porosity_3d : ndarray
        ``(n_layers, ny, nx)``.
    top : ndarray
        ``(ny, nx)`` top surface.
    bot : ndarray
        ``(n_layers, ny, nx)`` layer bottoms.
    point1, point2 : sequence of float
        End points as ``(column, row)`` grid indices; rounded to the nearest
        cell.
    num_points : int
        Samples along the profile.
    clip_to_grid : bool
        Clip the end points into the grid, and move one end by a cell when the
        two coincide, so a click outside the grid still gives a profile.
    fallback_to_active : bool
        When the line crosses no cell with both water content and porosity,
        try lines through the active domain instead: the horizontal and
        vertical lines through its median cell, then the two diagonals of its
        bounding box. The first that sees data is used.

    Returns
    -------
    dict
        ``interpolator``, ``L_profile``, ``structure`` (layer boundaries),
        ``water_content_profile`` (clipped to [0, 0.8]), ``porosity_profile``
        (clipped to [0.01, 0.6]) and ``points``, the ``(column, row)`` end
        points actually sampled.
    """
    from PyHydroGeophysX.core.interpolation import ProfileInterpolator

    if clip_to_grid:
        n_rows, n_cols = np.shape(top)
        p1_col = int(np.clip(round(float(point1[0])), 0, n_cols - 1))
        p1_row = int(np.clip(round(float(point1[1])), 0, n_rows - 1))
        p2_col = int(np.clip(round(float(point2[0])), 0, n_cols - 1))
        p2_row = int(np.clip(round(float(point2[1])), 0, n_rows - 1))
        # A zero-length profile samples nothing.
        if p1_col == p2_col and p1_row == p2_row:
            if p2_col < n_cols - 1:
                p2_col += 1
            elif p2_row < n_rows - 1:
                p2_row += 1
            elif p1_col > 0:
                p1_col -= 1
            else:
                p1_row = max(0, p1_row - 1)
    else:
        p1_col, p1_row = int(round(point1[0])), int(round(point1[1]))
        p2_col, p2_row = int(round(point2[0])), int(round(point2[1]))

    def _sample(c1: int, r1: int, c2: int, r2: int):
        interp = ProfileInterpolator(
            point1=[c1, r1],
            point2=[c2, r2],
            surface_data=top,
            origin_x=0.0,
            origin_y=0.0,
            pixel_width=1.0,
            pixel_height=-1.0,
            num_points=int(num_points),
        )
        structure = interp.interpolate_layer_data([top] + [bot[i] for i in range(bot.shape[0])])
        return (interp, structure, interp.interpolate_3d_data(water_content_3d),
                interp.interpolate_3d_data(porosity_3d))

    interpolator, structure, water_content_profile, porosity_profile = _sample(
        p1_col, p1_row, p2_col, p2_row)

    if fallback_to_active and (not np.isfinite(water_content_profile).any()
                               or not np.isfinite(porosity_profile).any()):
        active = np.any(np.isfinite(water_content_3d) & np.isfinite(porosity_3d), axis=0)
        if np.any(active):
            rows, cols = np.where(active)
            min_col, max_col = int(np.min(cols)), int(np.max(cols))
            min_row, max_row = int(np.min(rows)), int(np.max(rows))
            med_col, med_row = int(np.median(cols)), int(np.median(rows))
            for line in ((min_col, med_row, max_col, med_row),
                         (med_col, min_row, med_col, max_row),
                         (min_col, min_row, max_col, max_row),
                         (min_col, max_row, max_col, min_row)):
                tried = _sample(*line)
                if np.isfinite(tried[2]).any() and np.isfinite(tried[3]).any():
                    interpolator, structure, water_content_profile, porosity_profile = tried
                    p1_col, p1_row, p2_col, p2_row = line
                    break

    return {
        "interpolator": interpolator,
        "L_profile": np.asarray(interpolator.L_profile, dtype=float),
        "structure": fill_profile_nans(structure),
        "water_content_profile": np.clip(fill_profile_nans(water_content_profile), 0.0, 0.8),
        "porosity_profile": np.clip(fill_profile_nans(porosity_profile), 0.01, 0.6),
        "points": [(int(p1_col), int(p1_row)), (int(p2_col), int(p2_row))],
    }


def build_profile_mesh(
    L_profile: Any,
    structure: Any,
    water_content_profile: Any,
    porosity_profile: Any,
    *,
    quality: float = 32,
    area: float = 1.0,
) -> Dict[str, Any]:
    """The 2D mesh the ERT and SRT forward runs share, with the profile on it.

    Three regions between the surface and two of the profile's layer
    boundaries (markers 0, 3 and 2 from the top), extending 10 m below the
    deeper one; water content and porosity are interpolated onto the cells.
    The studio's forward run and the Streamlit hydro page both built this.

    Parameters
    ----------
    L_profile : ndarray
        Distance along the profile.
    structure : ndarray
        ``(n_layers + 1, n_profile)`` layer boundaries, surface first.
    water_content_profile, porosity_profile : ndarray
        ``(n_layers, n_profile)``.
    quality, area : float
        Passed to :class:`~PyHydroGeophysX.core.mesh_utils.MeshCreator`.

    Returns
    -------
    dict
        ``mesh``, ``mesh_markers``, ``water_content`` and ``porosity`` (per
        cell), ``layer_idx`` (the structure rows used as surface, middle and
        bottom) and ``layer_markers``.
    """
    from PyHydroGeophysX.core.interpolation import create_surface_lines
    from PyHydroGeophysX.core.mesh_utils import MeshCreator

    n_bounds = structure.shape[0]
    mid_idx = max(1, min(4, n_bounds // 3))
    bot_idx = max(mid_idx + 1, min(12, n_bounds - 2))
    surface, line1, line2 = create_surface_lines(
        L_profile=L_profile, structure=structure, top_idx=0, mid_idx=mid_idx, bot_idx=bot_idx
    )
    mesh, _ = MeshCreator(quality=quality, area=area).create_from_layers(
        surface=surface, layers=[line1, line2], bottom_depth=float(np.min(line2[:, 1]) - 10.0)
    )
    mesh_markers = assign_three_layer_markers(mesh, line1, line2, 0, 3, 2)
    return {
        "mesh": mesh,
        "mesh_markers": mesh_markers,
        "water_content": interpolate_profile_to_mesh(water_content_profile, structure,
                                                     L_profile, mesh),
        "porosity": interpolate_profile_to_mesh(porosity_profile, structure, L_profile, mesh),
        "layer_idx": [0, mid_idx, bot_idx],
        "layer_markers": [0, 3, 2],
    }


__all__ = [
    "fill_profile_nans",
    "get_mesh_xy",
    "assign_three_layer_markers",
    "interpolate_layer_samples",
    "interpolate_profile_to_mesh",
    "sample_profile",
    "build_profile_mesh",
]
