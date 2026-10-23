"""Qt-free 3D ERT mesh builder shared by the desktop studio.

Turns a plain ``config`` dict (the same keys the Streamlit 3D mesh builder uses)
into electrode positions and a PyGIMLi 3D mesh:

* surface arrays with topography - a grid, or electrodes read from a file
  (:data:`FILE_ARRAY`) - -> a PyGIMLi prism mesh (``Mesh3DCreator``);
* box and borehole layouts -> a Gmsh-free PyGIMLi structured grid.

Zones - boxes of known or assumed resistivity, a clay layer, a tank, a plume -
can shape the mesh: every engine can build it so that the zones' faces are
cell faces, and each zone can be its own region, which an inversion on the mesh
does not smooth across (see :func:`normalize_box_zones`).

Nothing here imports Qt, so it can run inside a worker thread and be unit-tested
without a display. ``generate_mesh`` is the single high-level entry point.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Topography
# ---------------------------------------------------------------------------
def topography_function(config: Dict[str, Any]) -> Callable[[float, float], float]:
    """Build a ``z = f(x, y)`` topography callable from the config."""
    topo_type = config.get("topography_type", "Flat")
    if topo_type == "Flat":
        z_flat = float(config.get("z_flat", 0.0))
        return lambda x, y: float(z_flat)

    if topo_type == "Linear tilt":
        z_base = float(config.get("z_base", 0.0))
        tilt_x = float(config.get("tilt_x", 0.0))
        tilt_y = float(config.get("tilt_y", 0.0))
        return lambda x, y: z_base + tilt_x * float(x) + tilt_y * float(y)

    if topo_type == "Gaussian hill":
        z_base = float(config.get("hill_base", 0.0))
        amp = float(config.get("hill_amp", 5.0))
        sigma = max(float(config.get("hill_sigma", 10.0)), 1.0e-9)
        cx = float(config.get("hill_cx", 0.0))
        cy = float(config.get("hill_cy", 0.0))
        return lambda x, y: z_base + amp * np.exp(
            -((float(x) - cx) ** 2 + (float(y) - cy) ** 2) / (2.0 * sigma**2)
        )

    if topo_type.startswith("From file"):
        return _topography_from_points(config.get("topography_points"))

    if topo_type == SENSOR_GROUND:
        return _topography_from_points(config.get("electrode_positions"))

    expr = str(config.get("topography_expr", "0.0"))
    allowed = {
        "np": np, "sin": np.sin, "cos": np.cos, "exp": np.exp,
        "sqrt": np.sqrt, "abs": abs, "pi": np.pi,
    }

    def _custom_topography(x, y):
        try:
            return float(eval(expr, {"__builtins__": {}}, {**allowed, "x": x, "y": y}))  # noqa: S307
        except Exception:
            return 0.0

    return _custom_topography


def _topography_from_points(points: Any) -> Callable[[float, float], float]:
    """Interpolate ``z = f(x, y)`` from loaded ``(x, y, z)`` points.

    Linear interpolation inside the data hull, nearest-neighbour outside it (so
    sensors near the survey edge still get a sensible elevation).
    """
    pts = np.asarray(points, dtype=float) if points is not None else None
    if pts is None or pts.ndim != 2 or pts.shape[1] < 3 or len(pts) == 0:
        return lambda x, y: 0.0
    xy = pts[:, :2]
    z = pts[:, 2]
    if len(pts) < 3:
        z_const = float(np.nanmean(z))
        return lambda x, y: z_const
    try:
        from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

        lin = LinearNDInterpolator(xy, z)
        near = NearestNDInterpolator(xy, z)

        def _topo(x, y):
            val = lin(float(x), float(y))
            return float(val) if np.isfinite(val) else float(near(float(x), float(y)))

        return _topo
    except Exception:  # noqa: BLE001 - scipy missing: nearest-neighbour fallback
        def _topo_nn(x, y):
            d2 = (xy[:, 0] - float(x)) ** 2 + (xy[:, 1] - float(y)) ** 2
            return float(z[int(np.argmin(d2))])

        return _topo_nn


# ---------------------------------------------------------------------------
# Electrodes
# ---------------------------------------------------------------------------
#: The array read from a table of sensor positions (``electrode_positions``).
FILE_ARRAY = "From file (x, y, z)"

#: The ground of a surface array read from a file: the surface through its
#: sensors' own elevations.
SENSOR_GROUND = "Sensor elevations"


def read_sensor_table(path: Any):
    """Sensor positions from a table, as ``(positions, how)``.

    ``positions`` is ``(n, 3)``: x, y and the elevation. The columns are chosen
    the way the ERT page chooses them (``table_io.read_electrode_table``): by
    the header when there is one - ``Cable,Elec,X,Y,Z`` reads X, Y and Z - and
    otherwise with a sensor-number column set aside. A table of x and the
    elevation alone is a line along x, at y = 0.
    """
    from PyHydroGeophysX.data_processing.table_io import read_electrode_table

    coords, elevation, how = read_electrode_table(path)
    if coords.shape[1] == 2:
        positions = np.column_stack([coords[:, 0], np.zeros(len(coords)), elevation])
    else:
        positions = np.asarray(coords[:, :3], dtype=float)
    positions = np.asarray(positions, dtype=float)
    if len(positions) < 2 or not np.isfinite(positions).all():
        raise ValueError("a sensor table needs at least two rows of finite x, y, z.")
    return positions, how


def _file_positions(config: Dict[str, Any]) -> np.ndarray:
    positions = config.get("electrode_positions")
    positions = np.atleast_2d(np.asarray(positions if positions is not None else [],
                                         dtype=float))
    if positions.ndim != 2 or positions.shape[1] < 3 or len(positions) < 2:
        raise ValueError("Load a sensor file (x, y, z) for this array first.")
    return positions[:, :3]


def buried_electrodes(positions: Any) -> bool:
    """Whether some electrodes sit below others: two at one place in plan, to
    the millimetre, at different depths - a borehole. On a surface array no
    electrode is stacked over another."""
    positions = np.atleast_2d(np.asarray(positions, dtype=float))
    if len(positions) < 2:
        return False
    _, column = np.unique(np.round(positions[:, :2], 3), axis=0, return_inverse=True)
    column = np.asarray(column).reshape(-1)
    for index in np.unique(column):
        depths = np.round(positions[column == index, 2], 3)
        if np.unique(depths).size > 1:
            return True
    return False


def surface_array(config: Dict[str, Any]) -> bool:
    """Whether the sensors all lie on the ground: a surface grid, or a file
    with no sensor below another. A surface array is meshed under the
    ground the topography gives; any other around its boreholes."""
    array_type = config.get("array_type", "Surface grid")
    if array_type == "Surface grid":
        return True
    if array_type == FILE_ARRAY:
        try:
            return not buried_electrodes(_file_positions(config))
        except ValueError:
            return True
    return False


def sensor_spacing(electrodes: Any, config: Optional[Dict[str, Any]] = None) -> float:
    """The spacing of a surface array: the grid's larger step, or for any other
    layout the median distance from a sensor to its nearest neighbour."""
    config = config or {}
    if config.get("array_type") == "Surface grid":
        return max(float(config.get("dx", 5.0)), float(config.get("dy", 5.0)), 1.0)
    from scipy.spatial import cKDTree

    xyz = np.unique(np.column_stack([np.asarray(electrodes[axis], dtype=float)
                                     for axis in "xyz"]), axis=0)
    if len(xyz) < 2:
        return 1.0
    nearest = cKDTree(xyz).query(xyz, k=2)[0][:, 1]
    nearest = nearest[np.isfinite(nearest) & (nearest > 1e-9)]
    return max(float(np.median(nearest)), 1.0e-3) if nearest.size else 1.0


def build_electrodes(config: Dict[str, Any], *, create_directory: bool = True):
    """Create a ``Mesh3DCreator`` and an electrode DataFrame from the config.

    Returns ``(creator, electrodes_df)`` where the DataFrame has columns
    ``x, y, z, n`` (electrode number). ``create_directory`` False leaves the
    output directory alone, for a caller that only wants the electrodes.

    A file array (:data:`FILE_ARRAY`) keeps the positions the file gives. On
    a surface array under topography, though, the electrodes are placed on
    the ground the topography sets, as a surface grid's are, unless that
    ground is their own elevations (:data:`SENSOR_GROUND`).
    """
    import pandas as pd

    from PyHydroGeophysX.core.mesh_3d import Mesh3DCreator

    creator = Mesh3DCreator(
        mesh_directory=str(config.get("output_dir", ".")),
        elec_refinement=float(config["electrode_refinement"]),
        node_refinement=float(config["boundary_refinement"]),
        attractor_distance=float(config["attractor_distance"]),
        create_directory=create_directory,
    )
    array_type = config["array_type"]
    mesh_type = config["mesh_type"]

    if array_type == "Surface grid":
        electrodes = creator.create_surface_electrode_array(
            nx=int(config["nx"]), ny=int(config["ny"]),
            dx=float(config["dx"]), dy=float(config["dy"]),
            x_offset=float(config["x_offset"]), y_offset=float(config["y_offset"]),
            z=0.0,
        )
        if mesh_type == "Surface with topography":
            topo_func = topography_function(config)
            electrodes["z"] = [
                topo_func(x_val, y_val) for x_val, y_val in zip(electrodes["x"], electrodes["y"])
            ]
    elif array_type == "Single borehole":
        z_values = np.linspace(float(config["z_start"]), float(config["z_end"]), int(config["n_bh_elec"]))
        electrodes = creator.create_borehole_electrode_array(
            float(config["bh_x"]), float(config["bh_y"]), z_values,
        )
    elif array_type == "Crosshole":
        z_values = np.linspace(float(config["z_start"]), float(config["z_end"]), int(config["n_bh_elec"]))
        electrodes = creator.create_crosshole_electrode_array(list(config["boreholes"]), z_values)
    elif array_type == FILE_ARRAY:
        positions = _file_positions(config)
        electrodes = pd.DataFrame({"x": positions[:, 0], "y": positions[:, 1],
                                   "z": positions[:, 2],
                                   "n": np.arange(1, len(positions) + 1, dtype=int)})
        if (mesh_type == "Surface with topography" and surface_array(config)
                and config.get("topography_type", "Flat") != SENSOR_GROUND):
            topo_func = topography_function(config)
            electrodes["z"] = [
                topo_func(x_val, y_val) for x_val, y_val in zip(electrodes["x"], electrodes["y"])
            ]
    else:  # Surface-to-borehole
        surface = creator.create_surface_electrode_array(
            nx=int(config["n_surface_elec"]), ny=1,
            dx=float(config["surface_dx"]), dy=1.0,
            x_offset=float(config["surface_x0"]), y_offset=float(config["surface_y"]),
            z=float(config["surface_z"]),
        )
        z_values = np.linspace(float(config["z_start"]), float(config["z_end"]), int(config["n_bh_elec"]))
        borehole = creator.create_borehole_electrode_array(
            float(config["bh_x"]), float(config["bh_y"]), z_values,
            electrode_start_number=len(surface) + 1,
        )
        electrodes = pd.concat([surface, borehole], ignore_index=True)
        electrodes["n"] = np.arange(1, len(electrodes) + 1, dtype=int)

    return creator, electrodes


# ---------------------------------------------------------------------------
# Zones: boxes of known or assumed resistivity
# ---------------------------------------------------------------------------
#: Zone ``k`` (0-based) is region ``ZONE_MARKER_START + k`` when zones are
#: regions of their own: marker 1 is the background and 2 the inverted region,
#: and the zones added to an E4D fine zone are numbered on from there.
ZONE_MARKER_START = 3


def normalize_box_zones(zones: Any) -> List[Dict[str, Any]]:
    """Validate 3-D zones and bring each to its canonical form.

    A zone is a box in the mesh's own coordinates, z being the elevation::

        {"name": "Clay", "x": [x_min, x_max], "y": [y_min, y_max],
         "z": [z_bottom, z_top], "resistivity": 20.0}

    Each range may be given in either order. Raises ``ValueError`` naming the
    zone at fault: a range without extent or a resistivity that is not a
    positive number would otherwise surface as an empty zone or a NaN deep in a
    forward run.
    """
    if isinstance(zones, (str, bytes, dict)):
        raise ValueError("Zones are a list, one entry per zone.")
    out: List[Dict[str, Any]] = []
    for index, zone in enumerate(zones or []):
        if not isinstance(zone, dict):
            raise ValueError(f"Zone {index + 1}: a zone has a name, x, y and z ranges and "
                             "a resistivity.")
        zone = dict(zone)
        name = str(zone.get("name") or f"Zone {index + 1}")
        ranges = []
        for axis in ("x", "y", "z"):
            try:
                low, high = sorted(float(value) for value in zone[axis])
            except (KeyError, TypeError, ValueError):
                raise ValueError(f"{name}: a zone needs x, y and z ranges, each given "
                                 "as [from, to].") from None
            if not (np.isfinite(low) and np.isfinite(high)) or high - low <= 0:
                raise ValueError(f"{name}: the {axis} range must have an extent, not "
                                 f"{low:g} to {high:g}.")
            ranges.append([low, high])
        try:
            resistivity = float(zone.get("resistivity"))
        except (TypeError, ValueError):
            raise ValueError(f"{name}: the resistivity must be a number.") from None
        if not np.isfinite(resistivity) or resistivity <= 0:
            raise ValueError(f"{name}: the resistivity must be positive, not "
                             f"{resistivity:g}.")
        out.append({"name": name, "x": ranges[0], "y": ranges[1], "z": ranges[2],
                    "resistivity": resistivity})
    return out


def box_zone_owner(points: Any, zones: Any) -> np.ndarray:
    """The zone each point lies in, -1 for none; where zones overlap, the later
    one takes the point, as in the 2-D zones of the ERT page."""
    points = np.atleast_2d(np.asarray(points, dtype=float))[:, :3]
    owner = np.full(len(points), -1, dtype=int)
    for index, zone in enumerate(normalize_box_zones(zones)):
        inside = np.ones(len(points), dtype=bool)
        for axis, name in enumerate(("x", "y", "z")):
            low, high = zone[name]
            inside &= (points[:, axis] >= low) & (points[:, axis] <= high)
        owner[inside] = index
    return owner


def apply_zone_markers(mesh: Any, zones: Any, *, separate: bool,
                       by_centres: bool = True,
                       reset_inverted: bool = False) -> List[Dict[str, Any]]:
    """Give the inverted cells each zone covers the zone's region, or none.

    With ``separate``, zone ``k`` becomes region ``ZONE_MARKER_START + k``: an
    inversion on the mesh treats it as a region of its own, and PyGIMLi (like
    E4D, zone by zone) puts no smoothness constraint between regions. Without
    it, the zone's cells stay in the inverted region, marker 2, and are smoothed
    across as usual. Only inverted cells (marker above 1) are zoned; the
    background is left alone. ``by_centres`` False keeps the zone markers the
    mesher already gave the cells (E4D does, from its zone seeds), and
    ``reset_inverted`` first returns every inverted cell to marker 2, which
    drops whatever numbering the mesher used inside the inverted region.

    Returns one entry per zone: its ``marker`` and how many ``cells`` it took.
    """
    zones = normalize_box_zones(zones)
    markers = np.asarray(mesh.cellMarkers(), dtype=int)
    if reset_inverted and zones:
        markers[markers > 1] = 2
    owner = np.full(len(markers), -1, dtype=int)
    if by_centres:
        inverted = markers > 1
        centres = np.asarray(mesh.cellCenters(), dtype=float)
        owner[inverted] = box_zone_owner(centres[inverted], zones)
    else:
        for index in range(len(zones)):
            owner[markers == ZONE_MARKER_START + index] = index
    report = []
    for index, zone in enumerate(zones):
        cells = owner == index
        marker = ZONE_MARKER_START + index if separate else 2
        markers[cells] = marker
        report.append({"name": zone["name"], "marker": int(marker),
                       "cells": int(cells.sum()), "resistivity": zone["resistivity"]})
    if zones:
        import pygimli as pg

        mesh.setCellMarkers(pg.IVector([int(value) for value in markers]))
    return report


def clear_inner_face_markers(mesh: Any) -> int:
    """Clear the marker of every face between two cells; returns how many.

    Triangle and TetGen keep an edge or a face only when it is marked, so zone
    faces, E4D's internal boundaries and the prism mesh's plan rectangle reach
    the mesh marked. PyGIMLi reads a marked face inside a region as a known
    interface and puts no smoothness constraint across it - a cut nobody asked
    for, since a zone is decoupled as a region of its own (``decouple_zones``),
    not by its faces. The outer faces keep their markers, which carry the
    boundary conditions.
    """
    cleared = 0
    for boundary in mesh.boundaries():
        if boundary.marker() != 0 and boundary.leftCell() is not None \
                and boundary.rightCell() is not None:
            boundary.setMarker(0)
            cleared += 1
    return cleared


def mark_far_field(mesh: Any) -> Dict[str, int]:
    """Mark a 3-D mesh's outer faces for an ERT forward model: the ground (an
    outer face whose outward normal points up) no flow, every other one the
    mixed condition that stands in for the ground going on beyond the mesh.

    PyGIMLi reads these conditions from the face markers (-1 no flow, -2
    mixed); a face left 0 or carrying a mesher's own number (Gmsh's surface
    group, E4D's 1 and 2) is no flow, so the mesh would be modelled as an
    insulating box. ADTLERT decides from the faces' normals the same way. A
    face marked for a fixed potential (-3, -4) keeps it. Returns how many
    faces are ``ground`` and ``far``.
    """
    import pygimli as pg

    counts = {"ground": 0, "far": 0}
    if int(mesh.dim()) != 3:
        return counts
    mesh.createNeighbourInfos()
    fixed = (pg.core.MARKER_BOUND_HOMOGEN_DIRICHLET, pg.core.MARKER_BOUND_DIRICHLET)
    for boundary in mesh.boundaries():
        left, right = boundary.leftCell(), boundary.rightCell()
        if (left is None) == (right is None) or int(boundary.marker()) in fixed:
            continue
        inside = (left if left is not None else right).center()
        normal, centre = boundary.norm(), boundary.center()
        outward = (normal.x() * (centre.x() - inside.x()) + normal.y() * (centre.y() - inside.y())
                   + normal.z() * (centre.z() - inside.z()))
        up = normal.z() if outward > 0 else -normal.z()
        ground = up > 1.0e-8
        boundary.setMarker(pg.core.MARKER_BOUND_HOMOGEN_NEUMANN if ground
                           else pg.core.MARKER_BOUND_MIXED)
        counts["ground" if ground else "far"] += 1
    return counts


# ---------------------------------------------------------------------------
# Structured (Gmsh-free) mesh for box / borehole layouts
# ---------------------------------------------------------------------------
#: How far a field mesh's outer region reaches beyond its domain, sideways and
#: down, in spreads (the domain's larger horizontal side), as on the ERT page's
#: own mesh: it carries the far boundary away from the array.
OUTER_SPREADS = 4.0


def outer_spreads(config: Dict[str, Any]) -> float:
    """How many spreads (the domain's larger horizontal side) the outer region
    reaches beyond the domain: ``config['outer_spreads']``, by default
    :data:`OUTER_SPREADS`. A Box mesh has none: a tank's walls are where its
    ground ends."""
    if config.get("mesh_type") == "Box mesh":
        return 0.0
    return max(float(config.get("outer_spreads", OUTER_SPREADS)), 0.0)


def graded_offsets(step: float, reach: float, growth: float = 1.5) -> np.ndarray:
    """Offsets from 0 out to ``reach``, the first ``growth`` times ``step`` and
    each next one ``growth`` times the last, all stretched a little so the last
    lands on ``reach``: the outer region's cells, fine beside the domain and
    coarse far from it. An empty reach gives ``[0]``."""
    if reach <= 0:
        return np.zeros(1)
    steps, size = [], float(step)
    while sum(steps) < reach - 1e-9:
        size *= growth
        steps.append(size)
    return np.r_[0.0, np.cumsum(steps) * (reach / sum(steps))]


def _axis_with_points(lower: float, upper: float, spacing: float, required_points: Any,
                      faces: Any = ()) -> np.ndarray:
    """Float axis spanning ``[lower, upper]`` that also includes electrode coords.

    ``faces`` - zone faces the grid must have a node on - are included too, and
    a regular node closer to one than a third of the spacing gives way to it, so
    a face does not leave a sliver of thin cells beside it.
    """
    lower, upper = float(lower), float(upper)
    if upper < lower:
        lower, upper = upper, lower
    if np.isclose(lower, upper):
        pad = max(abs(lower) * 0.05, float(spacing), 1.0)
        lower -= pad
        upper += pad
    spacing = max(float(spacing), 1.0e-6)
    intervals = max(1, int(np.ceil((upper - lower) / spacing)))
    base = np.linspace(lower, upper, intervals + 1, dtype=float)
    points = np.asarray(required_points, dtype=float).ravel()
    points = points[np.isfinite(points)]
    points = points[(points >= lower - 1.0e-9) & (points <= upper + 1.0e-9)]
    faces = np.asarray(list(faces), dtype=float).ravel()
    faces = faces[np.isfinite(faces) & (faces > lower + 1.0e-9) & (faces < upper - 1.0e-9)]
    if faces.size:
        keep = np.min(np.abs(base[:, None] - faces[None, :]), axis=1) > spacing / 3.0
        keep[[0, -1]] = True
        base = base[keep]
    axis = np.unique(np.round(np.concatenate([base, points, faces, [lower, upper]]),
                              8)).astype(float)
    axis.sort()
    if axis.size < 2:
        axis = np.asarray([lower, upper], dtype=float)
    return axis


def _structured_bounds(electrodes_df: Any, config: Dict[str, Any]) -> Dict[str, float]:
    """Estimate a structured 3D mesh domain for box and borehole-style surveys."""
    xs = np.asarray(electrodes_df["x"], dtype=float)
    ys = np.asarray(electrodes_df["y"], dtype=float)
    zs = np.asarray(electrodes_df["z"], dtype=float)
    mesh_type = config.get("mesh_type", "Surface with topography")

    if mesh_type == "Box mesh":
        x_min = min(0.0, float(np.nanmin(xs)))
        x_max = max(float(config.get("box_length", 50.0)), float(np.nanmax(xs)))
        y_min = min(0.0, float(np.nanmin(ys)))
        y_max = max(float(config.get("box_width", 30.0)), float(np.nanmax(ys)))
        z_top = max(0.0, float(np.nanmax(zs)))
        z_bottom = min(-float(config.get("box_height", 25.0)), float(np.nanmin(zs)))
    elif not surface_array(config):
        lateral_padding = float(config.get("borehole_lateral_padding", 10.0))
        top_padding = float(config.get("borehole_top_padding", 2.0))
        bottom_padding = float(config.get("borehole_bottom_padding", 5.0))
        x_min = float(np.nanmin(xs)) - lateral_padding
        x_max = float(np.nanmax(xs)) + lateral_padding
        y_min = float(np.nanmin(ys)) - lateral_padding
        y_max = float(np.nanmax(ys)) + lateral_padding
        z_top = max(0.0, float(np.nanmax(zs)) + top_padding)
        z_bottom = float(np.nanmin(zs)) - bottom_padding
    else:
        spacing = sensor_spacing(electrodes_df, config)
        extension = max(float(config.get("boundary_extension", 1.4)) - 1.0, 0.1)
        x_pad = max(spacing, (float(np.nanmax(xs)) - float(np.nanmin(xs))) * extension * 0.5)
        y_pad = max(spacing, (float(np.nanmax(ys)) - float(np.nanmin(ys))) * extension * 0.5)
        x_min = float(np.nanmin(xs)) - x_pad
        x_max = float(np.nanmax(xs)) + x_pad
        y_min = float(np.nanmin(ys)) - y_pad
        y_max = float(np.nanmax(ys)) + y_pad
        z_top = float(np.nanmax(zs))
        z_bottom = float(np.nanmin(zs)) - float(config.get("para_depth", 20.0))

    min_span = max(float(config.get("borehole_horizontal_cell", config.get("boundary_refinement", 2.0))), 1.0)
    if (x_max - x_min) < min_span:
        center = 0.5 * (x_min + x_max)
        x_min, x_max = center - 0.5 * min_span, center + 0.5 * min_span
    if (y_max - y_min) < min_span:
        center = 0.5 * (y_min + y_max)
        y_min, y_max = center - 0.5 * min_span, center + 0.5 * min_span
    if not z_bottom < z_top:
        z_bottom = z_top - float(config.get("para_depth", 20.0))

    return {
        "x_min": float(x_min), "x_max": float(x_max),
        "y_min": float(y_min), "y_max": float(y_max),
        "z_bottom": float(z_bottom), "z_top": float(z_top),
    }


def create_structured_mesh(electrodes_df: Any, config: Dict[str, Any],
                           zones: Any = None) -> Any:
    """Create a Gmsh-free PyGIMLi structured 3D mesh for box/borehole layouts.

    With ``zones`` the grid lines include every zone face, so the zones'
    faces are cell faces. A field mesh gets an outer region (marker 1) of
    cells doubling outward, reaching beyond the domain sideways and down
    (:func:`outer_spreads`).
    """
    import pygimli as pg

    bounds = _structured_bounds(electrodes_df, config)
    if surface_array(config):
        xy_spacing = float(config.get("boundary_refinement", 2.0))
        z_spacing = float(config.get("dz_fine", 0.5))
    else:
        xy_spacing = float(config.get("borehole_horizontal_cell", 2.0))
        z_spacing = float(config.get("borehole_vertical_cell", 1.0))

    zones = normalize_box_zones(zones)
    faces = {axis: [value for zone in zones for value in zone[axis]] for axis in "xyz"}
    x_axis = _axis_with_points(bounds["x_min"], bounds["x_max"], xy_spacing, electrodes_df["x"],
                               faces["x"])
    y_axis = _axis_with_points(bounds["y_min"], bounds["y_max"], xy_spacing, electrodes_df["y"],
                               faces["y"])
    z_axis = _axis_with_points(bounds["z_bottom"], bounds["z_top"], z_spacing, electrodes_df["z"],
                               faces["z"])

    reach = outer_spreads(config) * max(bounds["x_max"] - bounds["x_min"],
                                        bounds["y_max"] - bounds["y_min"])
    if reach > 0:
        # Doubling, not growing by half: a grid's padding planes cross the
        # whole grid, so every extra one costs a full slab of cells.
        pad = graded_offsets(xy_spacing, reach, growth=2.0)[1:]
        x_axis = np.r_[bounds["x_min"] - pad[::-1], x_axis, bounds["x_max"] + pad]
        y_axis = np.r_[bounds["y_min"] - pad[::-1], y_axis, bounds["y_max"] + pad]
        z_axis = np.r_[bounds["z_bottom"]
                       - graded_offsets(z_spacing, reach, growth=2.0)[1:][::-1], z_axis]

    mesh = pg.createGrid(x=x_axis.astype(float), y=y_axis.astype(float), z=z_axis.astype(float), marker=2)
    para_depth = float(config.get("para_depth", abs(bounds["z_top"] - bounds["z_bottom"])))
    for cell in mesh.cells():
        centre = cell.center()
        depth = bounds["z_top"] - float(centre.z())
        inside = (bounds["x_min"] < centre.x() < bounds["x_max"]
                  and bounds["y_min"] < centre.y() < bounds["y_max"]
                  and centre.z() > bounds["z_bottom"])
        cell.setMarker(2 if inside and depth <= para_depth else 1)
    return mesh


# ---------------------------------------------------------------------------
# Summary / export / high-level entry
# ---------------------------------------------------------------------------
def mesh_summary(mesh: Any) -> Dict[str, Any]:
    """Robust mesh summary metrics (cells/nodes/boundaries/dim)."""
    summary: Dict[str, Any] = {}
    for label, method_name in (
        ("Cells", "cellCount"), ("Nodes", "nodeCount"),
        ("Boundaries", "boundaryCount"), ("Dimension", "dim"),
    ):
        try:
            summary[label] = getattr(mesh, method_name)()
        except Exception:  # noqa: BLE001
            continue
    return summary


def save_outputs(
    mesh: Any, electrodes_df: Any, output_dir: Path, mesh_name: str, formats: List[str]
) -> Dict[str, str]:
    """Save requested outputs (BMS / VTK / electrode CSV). Returns {key: path}."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: Dict[str, str] = {}
    if any("bms" in f.lower() for f in formats):
        path = output_dir / f"{mesh_name}.bms"
        from PyHydroGeophysX.core.mesh_serialization import save_mesh_artifact

        path, sidecar = save_mesh_artifact(mesh, path)
        outputs["bms"] = str(path)
        outputs["mesh_structure"] = str(sidecar)
    if any("vtk" in f.lower() for f in formats):
        path = output_dir / f"{mesh_name}.vtk"
        from PyHydroGeophysX.core.mesh_serialization import via_ascii_path

        # exportVTK opens its own narrow path, so it fails on the same folders
        # that defeat mesh.save; see via_ascii_path for what that means.
        via_ascii_path(mesh.exportVTK, path, mode="write")
        outputs["vtk"] = str(path)
    if any("csv" in f.lower() or "sensor" in f.lower() or "electrode" in f.lower() for f in formats):
        path = output_dir / f"{mesh_name}_sensors.csv"
        electrodes_df.to_csv(path, index=False)
        outputs["sensors_csv"] = str(path)
    return outputs


def find_gmsh_binary() -> Optional[str]:
    """Locate a usable Gmsh executable.

    The pip ``gmsh`` package ships only the Python API (no console binary), so we
    fall back to the ``gmsh.exe`` bundled with resipy when it is installed.
    """
    import shutil

    found = shutil.which("gmsh")
    if found:
        return found
    try:
        import os

        import resipy

        candidate = os.path.join(os.path.dirname(resipy.__file__), "exe", "gmsh.exe")
        if os.path.exists(candidate):
            return candidate
    except Exception:  # noqa: BLE001
        pass
    return None


def _gmsh_box_mesh(creator: Any, electrodes: Any, config: Dict[str, Any],
                   zones: Any = None) -> Any:
    """Refined tetrahedral mesh via Gmsh: a box domain with the sensors embedded.

    High mesh quality with local refinement at the sensors (single region). The
    top is flat, so it suits box / borehole / flat-terrain surveys; strong
    topography is better served by the prism engine. With ``zones`` the zone
    boxes are cut into the domain, so their faces are cell faces.
    """
    binary = find_gmsh_binary()
    if not binary:
        raise RuntimeError(
            "Gmsh executable not found (the pip 'gmsh' package provides only the "
            "Python API). Install Gmsh or resipy, which bundles gmsh.exe.")
    creator.gmsh_path = binary
    xs = np.asarray(electrodes["x"], dtype=float)
    ys = np.asarray(electrodes["y"], dtype=float)
    zs = np.asarray(electrodes["z"], dtype=float)

    if config.get("mesh_type") == "Box mesh":
        length = float(config.get("box_length", 50.0))
        width = float(config.get("box_width", 30.0))
        height = float(config.get("box_height", 25.0))
        origin = (min(0.0, float(xs.min())), min(0.0, float(ys.min())), float(zs.max()) - height)
    else:
        if surface_array(config):
            ext = max(float(config.get("boundary_extension", 1.4)) - 1.0, 0.1)
            lateral = max(float(config.get("boundary_refinement", 2.0)),
                          (float(xs.max()) - float(xs.min())) * ext * 0.5,
                          (float(ys.max()) - float(ys.min())) * ext * 0.5)
        else:
            lateral = float(config.get("borehole_lateral_padding", 10.0))
        depth = float(config.get("para_depth", 20.0)) + float(config.get("borehole_bottom_padding", 5.0))
        origin = (float(xs.min()) - lateral, float(ys.min()) - lateral, float(zs.min()) - depth)
        length = (float(xs.max()) - float(xs.min())) + 2.0 * lateral
        width = (float(ys.max()) - float(ys.min())) + 2.0 * lateral
        height = float(zs.max()) - origin[2]

    reach = outer_spreads(config) * max(length, width)
    mesh = creator.create_box_mesh(length, width, height, electrodes, output_name="gmsh_mesh",
                                   origin=origin, zones=zones, outer=reach)
    if reach > 0:
        # The outer region is background: only the box is inverted.
        upper = (origin[0] + length, origin[1] + width)
        for cell in mesh.cells():
            centre = cell.center()
            if not (origin[0] < centre.x() < upper[0] and origin[1] < centre.y() < upper[1]
                    and centre.z() > origin[2]):
                cell.setMarker(1)
    return mesh


#: The engine that builds meshes from an E4D mesh configuration (see ``core.e4d_mesh``).
E4D_ENGINE = "E4D (Triangle + TetGen)"

#: What the E4D engine reads from the config, and its defaults: those of the
#: Van Nuys crosshole configuration E4D users build from.
E4D_DEFAULTS: Dict[str, Any] = {
    "e4d_fine_padding": 1.0, "e4d_fine_depth_padding": 1.0, "e4d_fine_volume": 1.0,
    "e4d_outer_distance": 100.0, "e4d_bottom_depth": 150.0, "e4d_quality": 1.28,
    "e4d_refine_offset": 0.01, "e4d_conductivity": 0.1, "e4d_mesher": "auto",
    "e4d_tetgen": "", "e4d_config_path": "",
}


def e4d_configuration(config: Dict[str, Any], electrodes: Any = None, zones: Any = None):
    """The E4D mesh configuration the E4D engine will build from.

    A loaded ``.cfg`` (``e4d_config_path``) is used as it is; otherwise one is
    laid out around the sensors as E4D users lay one out, on the surface the
    config's topography describes, with ``zones`` as E4D zones of their own
    inside the fine zone. Returns ``(configuration, sensors)``, the sensors
    being the control points to show when they came from a file.
    """
    import pandas as pd

    from PyHydroGeophysX.core import e4d_mesh as e4d

    options = {**E4D_DEFAULTS, **{k: v for k, v in config.items() if k in E4D_DEFAULTS}}
    source = str(options["e4d_config_path"] or "")
    if source:
        cfg = e4d.read_e4d_config(source)
        shown = cfg.flags != e4d.OUTER
        sensors = pd.DataFrame({"x": cfg.points[shown, 0], "y": cfg.points[shown, 1],
                                "z": cfg.points[shown, 2],
                                "n": np.flatnonzero(shown) + 1})
        return cfg, sensors
    if electrodes is None:
        _, electrodes = build_electrodes(config)
    if zones is None and config.get("conform_to_zones"):
        zones = config.get("zones")
    flat = config.get("topography_type", "Flat") == "Flat"
    cfg, _ = e4d.e4d_config_from_electrodes(
        electrodes,
        surface=float(config.get("z_flat", 0.0)) if flat else topography_function(config),
        fine_padding=float(options["e4d_fine_padding"]),
        fine_depth_padding=float(options["e4d_fine_depth_padding"]),
        outer_distance=float(options["e4d_outer_distance"]),
        bottom_depth=float(options["e4d_bottom_depth"]),
        fine_volume=float(options["e4d_fine_volume"]), quality=float(options["e4d_quality"]),
        refine_offset=float(options["e4d_refine_offset"]),
        conductivity=float(options["e4d_conductivity"]),
        topography_points=0 if flat else 8, zones=normalize_box_zones(zones))
    return cfg, electrodes


def zone_region(config: Dict[str, Any], electrodes: Any = None) -> Dict[str, float]:
    """Where zones act in the mesh ``config`` builds.

    Zones only take inverted cells (see :func:`apply_zone_markers`), and the
    E4D engine meshes them only inside its fine zone, so this is where a new
    zone belongs and what a view clips the zone boxes to: the E4D engine's fine
    zone, or for the other engines the domain they build around the sensors,
    down to the investigation depth (the prism and Gmsh meshes reach a little
    further). Returns ``x_min``, ``x_max``, ``y_min``, ``y_max``, ``z_bottom``
    and ``z_top``, the highest ground over it. Nothing is written.
    """
    if electrodes is None:
        _, electrodes = build_electrodes(config, create_directory=False)
    xs, ys, zs = (np.asarray(electrodes[axis], dtype=float) for axis in ("x", "y", "z"))
    if config.get("mesh_engine") != E4D_ENGINE:
        bounds = _structured_bounds(electrodes, config)
        depth = float(config.get("para_depth", bounds["z_top"] - bounds["z_bottom"]))
        bounds["z_bottom"] = max(bounds["z_bottom"], bounds["z_top"] - depth)
        return bounds
    options = {**E4D_DEFAULTS, **{k: v for k, v in config.items() if k in E4D_DEFAULTS}}
    pad = float(options["e4d_fine_padding"])
    region = {"x_min": float(xs.min()) - pad, "x_max": float(xs.max()) + pad,
              "y_min": float(ys.min()) - pad, "y_max": float(ys.max()) + pad,
              "z_bottom": float(zs.min()) - float(options["e4d_fine_depth_padding"])}
    if config.get("topography_type", "Flat") == "Flat":
        region["z_top"] = float(config.get("z_flat", 0.0))
    else:
        ground = topography_function(config)
        region["z_top"] = max(float(ground(x, y))
                              for x in np.linspace(region["x_min"], region["x_max"], 5)
                              for y in np.linspace(region["y_min"], region["y_max"], 5))
    return region


def _zone_log(say: Callable[[str], None], report: List[Dict[str, Any]], *,
              conform: bool, separate: bool) -> None:
    """What the zones did to the mesh, for the run log."""
    if not report:
        return
    say("Zones: " + "; ".join(f"{entry['name']}: {entry['cells']} cells"
                              + (f" (region {entry['marker']})" if separate else "")
                              for entry in report))
    say("  the mesh follows the zone faces" if conform else
        "  zones take the cells whose centre lies inside them (the mesh does not "
        "follow their faces)")
    missed = [entry["name"] for entry in report if conform and not entry.get("follows", True)]
    if missed:
        say(f"  - except {', '.join(missed)}: on this slope the prism layers cannot bend to "
            "meet their top and bottom without folding cells or squeezing a layer to a "
            "third of its thickness, so they take the cells whose centre lies inside them "
            "(the Gmsh and E4D engines cut zone faces into the mesh on any ground)")
    empty = [entry["name"] for entry in report if not entry["cells"]]
    if empty:
        say(f"  (zone(s) {', '.join(empty)} took no cell - they lie outside the inverted "
            "region, or zones later in the list cover them - and have no effect)")


def generate_mesh(config: Dict[str, Any], log: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    """Build sensors and a 3D mesh from ``config``.

    Honours ``config['mesh_engine']`` (Auto / Gmsh (tetrahedral) / PyGIMLi prism /
    Structured grid / E4D) with a graceful fallback to the structured grid if Gmsh
    fails, and ``config['single_region']`` to collapse the markers to one region.

    ``config['zones']`` are boxes of known or assumed resistivity (see
    :func:`normalize_box_zones`). With ``conform_to_zones`` the mesh is built so
    that their faces are cell faces - E4D meshes them as zones of its own, the
    structured grid puts grid lines on them, the prism mesh puts them into its
    plan triangulation and its layers, Gmsh cuts them into the domain - and with
    ``decouple_zones`` each zone is a region of its own, which an inversion on
    the mesh does not smooth across. Inside a region the smoothness crosses
    every face: faces between cells carry no marker
    (:func:`clear_inner_face_markers`).

    Returns ``{"mesh", "electrodes", "generator", "zones"}``, ``zones`` saying
    which region each zone became, how many cells it took and whether the mesh
    follows its faces (``follows``), and for the E4D
    engine also ``e4d_files`` and ``e4d_zones``. Safe to call from a worker.
    """
    say = log or (lambda *_: None)
    engine = config.get("mesh_engine", "Auto")
    zones = normalize_box_zones(config.get("zones"))
    conform = bool(config.get("conform_to_zones")) and bool(zones)
    separate = bool(config.get("decouple_zones"))
    if engine == E4D_ENGINE:
        from PyHydroGeophysX.core import e4d_mesh as e4d

        loaded = bool(config.get("e4d_config_path"))
        # A loaded E4D configuration carries its own electrodes; the sensor
        # array is built only when the layout is made from it.
        if not loaded:
            say("Building sensor array…")
        cfg, electrodes = e4d_configuration(config, zones=zones if conform else None)
        for note in getattr(cfg, "notes", []):
            say("  " + note)
        say("Building the mesh from the E4D configuration: surface triangulation, then TetGen…")
        built = e4d.build_e4d_mesh(
            cfg, Path(config.get("output_dir") or "."),
            name=str(config.get("e4d_basename") or "e4d_mesh"),
            mesher=str(config.get("e4d_mesher", "auto")),
            tetgen=str(config.get("e4d_tetgen", "") or ""), log=say)
        mesh, label = built["mesh"], f"E4D configuration · {built['mesher']}"
        if config.get("single_region"):
            for cell in mesh.cells():
                cell.setMarker(2)
            label += " · single region"
        report: List[Dict[str, Any]] = []
        if zones and loaded:
            say("  (the zones are not used: the loaded E4D configuration defines its own)")
        elif zones:
            # By cell centre even when E4D meshed the zones: a zone kept off
            # the fine zone's walls in the configuration still reaches them in
            # the mesh, and E4D's numbering shifts when the zones fill it - so
            # the inverted region goes back to marker 2 first.
            report = apply_zone_markers(mesh, zones, separate=separate, reset_inverted=True)
            for entry in report:
                entry["follows"] = conform
            _zone_log(say, report, conform=conform, separate=separate)
        clear_inner_face_markers(mesh)
        mark_far_field(mesh)
        say(f"Mesh ready ({label}).")
        return {"mesh": mesh, "electrodes": electrodes, "generator": label,
                "e4d_files": dict(built["files"]), "e4d_zones": list(built["zones"]),
                "zones": report, "zones_conform": conform and not loaded}

    say("Building sensor array…")
    creator, electrodes = build_electrodes(config)
    mesh_type = config.get("mesh_type")
    surface_topo = mesh_type == "Surface with topography" and surface_array(config)
    shaping = zones if conform else None

    def _prism():
        return creator.create_3d_mesh_with_topography(
            electrode_positions=electrodes,
            topography_func=topography_function(config),
            para_depth=float(config["para_depth"]),
            dz_fine=float(config["dz_fine"]),
            dz_coarse=float(config["dz_coarse"]),
            boundary_extension=float(config["boundary_extension"]),
            use_prism_mesh=True, zones=shaping, outer_spreads=outer_spreads(config),
        ), "PyGIMLi topography prism"

    followed = conform
    try:
        if engine == "Gmsh (tetrahedral)":
            say("Generating Gmsh tetrahedral mesh…")
            mesh, label = _gmsh_box_mesh(creator, electrodes, config, shaping), "Gmsh tetrahedral"
        elif engine == "Structured grid":
            say("Creating PyGIMLi structured grid…")
            mesh, label = create_structured_mesh(electrodes, config, shaping), "PyGIMLi structured grid"
        elif engine == "PyGIMLi prism" and surface_topo:
            say("Creating PyGIMLi topography prism mesh…")
            mesh, label = _prism()
        elif engine == "Auto" and surface_topo:
            say("Creating PyGIMLi topography prism mesh…")
            mesh, label = _prism()
        else:
            say("Creating PyGIMLi structured grid…")
            mesh, label = create_structured_mesh(electrodes, config, shaping), "PyGIMLi structured grid"
    except Exception as exc:  # noqa: BLE001
        if engine == "Gmsh (tetrahedral)":
            say(f"Gmsh failed ({exc}); falling back to structured grid.")
            mesh, label = (create_structured_mesh(electrodes, config, shaping),
                           "PyGIMLi structured grid (Gmsh fallback)")
        else:
            raise

    if config.get("single_region"):
        for cell in mesh.cells():
            cell.setMarker(2)
        label += " · single region"

    report = apply_zone_markers(mesh, zones, separate=separate) if zones else []
    missed = set(getattr(creator, "unfollowed_zones", ()))     # the prism mesh's
    for index, entry in enumerate(report):
        entry["follows"] = followed and index not in missed
    _zone_log(say, report, conform=followed, separate=separate)
    clear_inner_face_markers(mesh)
    mark_far_field(mesh)
    say(f"Mesh ready ({label}).")
    return {"mesh": mesh, "electrodes": electrodes, "generator": label, "zones": report,
            "zones_conform": followed}
