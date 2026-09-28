"""E4D-style tetrahedral meshes: the configuration file, the PLC, and the mesh.

E4D, PNNL's parallel ERT/IP code, does not take a mesh; it takes a *mesh
configuration file* (``.cfg``) describing the geometry, and builds the mesh
from it. The geometry is a list of **control points**, each with a boundary
flag:

* ``1`` - a surface point: a surface electrode, or a point of known elevation;
* ``2`` - a surface boundary point: the corners of the computational domain,
  far from the survey, which E4D joins into vertical planes down to the mesh
  bottom ``m_bot``;
* ``0`` - an internal point: a buried (borehole) electrode, a vertex of an
  internal boundary, or a refinement point.

Internal boundaries are planar polygons through control points that divide the
domain into **zones**, each identified by a seed point and given a maximum
element volume and a starting conductivity. A borehole survey is typically a
small fine zone around the boreholes, at a volume of a cubic metre or so, inside
a coarse outer zone reaching a hundred metres or more in every direction, so the
boundary conditions sit where the potentials have died away.

E4D then builds the mesh in two steps, and this module does the same:

1. The surface and surface boundary points are triangulated in plan (with
   Triangle), the traces of the internal boundaries on the surface included as
   segments, and the new surface nodes are given elevations interpolated from
   the points of known elevation.
2. That surface, the vertical outer walls, the bottom, the internal boundaries
   (carrying every surface node Triangle put on their top edges) and the
   internal points form a piecewise linear complex (PLC), written as a TetGen
   ``.poly`` and tetrahedralized as E4D does it,
   ``tetgen -pnq<quality>a<max volume>aAA``: region attributes become the zone
   numbers of the elements, and each zone's volume constraint applies.

Coordinates are translated first, by the mean of every control point that is
not on the outer boundary, which E4D records in the ``.trn`` file; the mesh this
module returns is back in the survey's own coordinates.

TetGen is run as the program E4D calls when one is found; otherwise through
its Python package (``pip install tetgen``), with the same switches, the
polygon facets triangulated first because the package takes triangles. With
neither, the same PLC is meshed with Gmsh (the build ResIPy bundles is found
automatically), the zones recovered from the seed points and each zone's volume
limit checked, since Gmsh aims at a size rather than enforcing one. The ``.cfg``
and ``.poly`` are written in every case, so E4D itself can be run on the
geometry.

Cell markers of the returned mesh are the zone numbers. Zone 1 of the layouts
built here is the outer zone, which is also what PyGIMLi treats as the
background region (marker 1), so the mesh can be inverted as it is.
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = [
    "E4DMeshConfig",
    "E4DPLC",
    "E4DZone",
    "build_e4d_mesh",
    "build_e4d_plc",
    "e4d_config_from_electrodes",
    "find_tetgen",
    "read_e4d_config",
    "read_e4d_mesh",
    "write_e4d_config",
    "write_tetgen_poly",
]

LogFn = Callable[[str], None]
SURFACE, OUTER, INTERIOR = 1, 2, 0
#: The boundary number E4D's own examples give an internal boundary.
INTERNAL_BOUNDARY = 11
#: A volume constraint at or above this is no constraint at all.
UNCONSTRAINED = 1.0e11


def _noop(_message: str) -> None:
    pass


# ---------------------------------------------------------------------------
# The configuration file
# ---------------------------------------------------------------------------
@dataclass
class E4DZone:
    """One zone: its number, a point inside it, and what E4D gives it."""

    index: int
    seed: Tuple[float, float, float]
    max_volume: float = 1.0e12
    conductivity: float = 0.01
    phase: Optional[float] = None       # spectral IP only


@dataclass
class E4DMeshConfig:
    """The contents of an E4D mesh configuration file.

    ``boundaries`` are the internal boundary planes, each ``(b_num, indices)``
    with 1-based control point indices as the file numbers them.
    """

    quality: float = 1.4                 # m_qual: maximum radius-edge ratio
    max_volume: float = 1.0e12           # max_evol_def: default maximum volume
    bottom: float = -100.0               # m_bot: elevation of the mesh bottom
    build: int = 1                       # tet_build_flag: 1 runs tetgen
    tetgen: str = "tetgen"               # tet_loc
    triangle: str = "triangle"           # tri_loc
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    flags: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    boundaries: List[Tuple[int, List[int]]] = field(default_factory=list)
    holes: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    zones: List[E4DZone] = field(default_factory=list)
    vis_flag: int = 0
    vis_loc: str = "bx"
    trailing: List[str] = field(default_factory=list)   # anything after, kept as is
    notes: List[str] = field(default_factory=list)      # what a layout adjusted; not written

    def __post_init__(self) -> None:
        self.points = np.asarray(self.points, dtype=float).reshape(-1, 3)
        self.flags = np.asarray(self.flags, dtype=int).reshape(-1)
        self.holes = np.asarray(self.holes, dtype=float).reshape(-1, 3)

    def translation(self) -> np.ndarray:
        """E4D's coordinate translation (its ``.trn``): the mean of every control
        point not on the outer boundary."""
        inner = self.points[self.flags != OUTER]
        return inner.mean(axis=0) if len(inner) else np.zeros(3)

    def outer_indices(self) -> List[int]:
        """0-based indices of the surface boundary points, in the order given."""
        return [int(i) for i in np.flatnonzero(self.flags == OUTER)]

    def validate(self) -> List[str]:
        """What is wrong with the configuration, as sentences; empty if nothing."""
        problems: List[str] = []
        n = len(self.points)
        if len(self.flags) != n:
            problems.append(f"{n} control points but {len(self.flags)} boundary flags.")
        if len(self.outer_indices()) < 3:
            problems.append("The outer boundary needs at least three surface boundary "
                            "points (flag 2).")
        for number, (b_num, indices) in enumerate(self.boundaries, start=1):
            if len(indices) < 3:
                problems.append(f"Internal boundary {number} has {len(indices)} points; "
                                "a plane needs three.")
            bad = [i for i in indices if not 1 <= int(i) <= n]
            if bad:
                problems.append(f"Internal boundary {number} refers to control points "
                                f"{bad}, which do not exist (1 to {n}).")
        if not self.zones:
            problems.append("At least one zone is needed.")
        if n and self.bottom >= float(self.points[:, 2].min()):
            problems.append(f"The mesh bottom ({self.bottom:g}) must lie below every "
                            f"control point (the lowest is {self.points[:, 2].min():g}).")
        outer = self.points[self.outer_indices()]
        if len(outer) >= 3:
            from matplotlib.path import Path as _Path

            ring = _Path(outer[:, :2])
            surface = self.points[self.flags == SURFACE]
            outside = ~ring.contains_points(surface[:, :2]) if len(surface) else []
            if np.any(outside):
                problems.append(f"{int(np.sum(outside))} surface point(s) lie on or "
                                "outside the outer boundary; E4D does not allow that.")
        return problems


def _records(path: Union[str, Path]) -> List[List[str]]:
    """The file's lines as token lists: comments dropped, blank lines skipped."""
    records = []
    for raw in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        text = raw.split("#", 1)[0].strip()
        if text:
            records.append(text.split())
    return records


class _Reader:
    """Fortran list-directed reading: a read takes values from as many lines as
    it needs and discards what is left on the last one."""

    def __init__(self, records: List[List[str]]) -> None:
        self._records = records
        self._line = 0

    def take(self, count: int, what: str, *, extra: bool = False) -> List[str]:
        """``count`` values; with ``extra``, also whatever else their last line held."""
        values: List[str] = []
        while len(values) < count:
            if self._line >= len(self._records):
                raise ValueError(f"The configuration ends before {what}.")
            values.extend(self._records[self._line])
            self._line += 1
        return values if extra else values[:count]

    def rest(self) -> List[str]:
        return [" ".join(tokens) for tokens in self._records[self._line:]]


def _unquote(text: str) -> str:
    return text.strip().strip("'\"")


def read_e4d_config(path: Union[str, Path]) -> E4DMeshConfig:
    """Read an E4D mesh configuration file.

    >>> import tempfile, textwrap
    >>> text = textwrap.dedent('''\\
    ...     1.4 1e12
    ...     -50
    ...     1
    ...     'tetgen'
    ...     'triangle'
    ...     5 # points
    ...     1 0 0 0 1
    ...     2 -40 -40 0 2
    ...     3 -40 40 0 2
    ...     4 40 40 0 2
    ...     5 40 -40 0 2
    ...     0 # internal boundaries
    ...     0 # holes
    ...     1 # zones
    ...     1 0 0 -10 1e12 0.01
    ...     0
    ...     'bx'
    ...     ''')
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     cfg_path = Path(folder) / "demo.cfg"
    ...     _ = cfg_path.write_text(text)
    ...     cfg = read_e4d_config(cfg_path)
    >>> len(cfg.points), cfg.flags.tolist(), cfg.bottom, cfg.zones[0].conductivity
    (5, [1, 2, 2, 2, 2], -50.0, 0.01)
    """
    reader = _Reader(_records(path))
    quality, max_volume = (float(v) for v in reader.take(2, "the mesh quality"))
    bottom = float(reader.take(1, "the mesh bottom")[0])
    build = int(float(reader.take(1, "the tetgen build flag")[0]))
    tetgen = _unquote(reader.take(1, "the tetgen path")[0])
    triangle = _unquote(reader.take(1, "the triangle path")[0])

    n_points = int(float(reader.take(1, "the number of control points")[0]))
    points, flags = np.zeros((n_points, 3)), np.zeros(n_points, dtype=int)
    for row in range(n_points):
        number, x, y, z, flag = reader.take(5, f"control point {row + 1}")
        points[row] = (float(x), float(y), float(z))
        flags[row] = int(float(flag))

    boundaries: List[Tuple[int, List[int]]] = []
    for number in range(int(float(reader.take(1, "the number of internal boundaries")[0]))):
        count, b_num = (int(float(v)) for v in reader.take(2, f"internal boundary {number + 1}"))
        indices = [int(float(v)) for v in reader.take(count, f"the points of boundary {number + 1}")]
        boundaries.append((b_num, indices))

    holes = []
    for number in range(int(float(reader.take(1, "the number of holes")[0]))):
        values = reader.take(4, f"hole {number + 1}")
        holes.append([float(v) for v in values[1:4]])

    zones: List[E4DZone] = []
    n_zones = int(float(reader.take(1, "the number of zones")[0]))
    for number in range(n_zones):
        # Six values: the zone, a point in it, its volume and conductivity; a
        # seventh on the same line is the phase of a spectral IP run.
        values = reader.take(6, f"zone {number + 1}", extra=True)
        phase = float(values[6]) if len(values) > 6 else None
        zones.append(E4DZone(index=int(float(values[0])),
                             seed=(float(values[1]), float(values[2]), float(values[3])),
                             max_volume=float(values[4]), conductivity=float(values[5]),
                             phase=phase))

    vis_flag, vis_loc, trailing = 0, "bx", []
    try:
        vis_flag = int(float(reader.take(1, "the visualization flag")[0]))
        vis_loc = _unquote(reader.take(1, "the visualization program")[0])
        trailing = reader.rest()
    except ValueError:
        pass
    return E4DMeshConfig(quality=quality, max_volume=max_volume, bottom=bottom, build=build,
                         tetgen=tetgen, triangle=triangle, points=points, flags=flags,
                         boundaries=boundaries, holes=np.asarray(holes).reshape(-1, 3),
                         zones=zones, vis_flag=vis_flag, vis_loc=vis_loc, trailing=trailing)


def _number(value: float) -> str:
    """A number as E4D and TetGen both read it, without an exponent."""
    return np.format_float_positional(float(value), trim="-")


def write_e4d_config(config: E4DMeshConfig, path: Union[str, Path]) -> Path:
    """Write ``config`` as an E4D mesh configuration file.

    Paths are single-quoted, as the E4D guide asks: Fortran reads an unquoted
    path only up to its first slash.
    """
    lines = [f"{_number(config.quality)} {_number(config.max_volume)}"
             "        # mesh quality (max radius-edge ratio), default max element volume",
             f"{_number(config.bottom)}        # elevation of the mesh bottom",
             f"{int(config.build)}        # 1 = run tetgen",
             f"'{config.tetgen}'", f"'{config.triangle}'", "",
             f"{len(config.points)}        # number of control points: index x y z flag "
             "(1 surface, 2 outer boundary, 0 internal)"]
    for number, (point, flag) in enumerate(zip(config.points, config.flags), start=1):
        lines.append(f"{number}\t{_number(point[0])}\t{_number(point[1])}\t"
                     f"{_number(point[2])}\t{int(flag)}")
    lines += ["", f"{len(config.boundaries)}        # number of internal boundaries"]
    for b_num, indices in config.boundaries:
        lines.append(f"{len(indices)}\t{int(b_num)}")
        lines.append("\t".join(str(int(i)) for i in indices))
    lines += ["", f"{len(config.holes)}        # number of holes"]
    for number, hole in enumerate(config.holes, start=1):
        lines.append(f"{number} {_number(hole[0])} {_number(hole[1])} {_number(hole[2])}")
    lines += ["", f"{len(config.zones)}        # number of zones: index x y z max volume "
                  "conductivity"]
    for zone in config.zones:
        line = (f"{int(zone.index)} {_number(zone.seed[0])} {_number(zone.seed[1])} "
                f"{_number(zone.seed[2])} {_number(zone.max_volume)} "
                f"{_number(zone.conductivity)}")
        if zone.phase is not None:
            line += f" {_number(zone.phase)}"
        lines.append(line)
    lines += ["", f"{int(config.vis_flag)}        # 1 = build the exodus visualization file",
              f"'{config.vis_loc}'"]
    lines += list(config.trailing)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


# ---------------------------------------------------------------------------
# A configuration built the way E4D users build one
# ---------------------------------------------------------------------------
def _surface_function(surface: Union[float, Callable[[float, float], float], None]):
    if callable(surface):
        return lambda x, y: float(surface(float(x), float(y)))
    level = 0.0 if surface is None else float(surface)
    return lambda x, y: level


def e4d_config_from_electrodes(
        electrodes: Any, *, surface: Union[float, Callable[[float, float], float], None] = 0.0,
        fine_padding: float = 1.0, fine_depth_padding: float = 1.0,
        outer_distance: float = 100.0, bottom_depth: float = 150.0,
        fine_volume: float = 1.0, outer_volume: float = 1.0e12, quality: float = 1.28,
        refine_offset: float = 0.01, conductivity: float = 0.1,
        topography_points: int = 0,
        zones: Optional[Sequence[Dict[str, Any]]] = None) -> Tuple[E4DMeshConfig, List[int]]:
    """A mesh configuration for ``electrodes`` laid out as E4D's own examples are.

    Control points, in order: the electrodes (flag 1 at the surface, 0 below
    it); for every buried electrode a refinement point ``refine_offset`` away
    in x and y; every borehole's top on the surface, with a refinement point
    ``5 * refine_offset`` under it; the corners of a fine zone reaching
    ``fine_padding`` beyond the electrodes and ``fine_depth_padding`` below the
    deepest one; and the corners of the outer boundary ``outer_distance``
    beyond the fine zone. The fine zone's four walls and floor are internal
    boundaries. Zone 1 is the outer zone, at ``outer_volume``; zone 2 the fine
    zone, at ``fine_volume``; both start at ``conductivity`` (S/m).

    ``zones`` - boxes as ``core.mesh_3d.normalize_box_zones`` gives them - become
    zones 3, 4, ... inside the fine zone, as E4D models known structure: their
    corners are control points, their faces internal boundaries, and each
    starts at the conductivity of its resistivity, at the fine zone's element
    volume. A zone reaching up to the ground has the ground as its top. A zone
    reaching the fine zone's walls on every side is a layer across it; any
    other is a block, kept a little inside the walls and floor, which may stand
    apart from the others, share a whole face, or lie inside a layer or a block
    listed before it (see :func:`_lay_out_zones`). Anything else, and a zone
    outside the fine zone, is an error.

    ``surface`` is the ground elevation, a number or ``z = f(x, y)``; with
    ``topography_points`` > 0, a grid of that many points a side across the
    fine zone, and one across the whole domain, carry its shape into the mesh.
    A ``refine_offset`` of 0 adds no refinement points.

    Returns ``(config, electrode_indices)``, the second the 0-based control
    point of each electrode.
    """
    xyz = np.asarray(
        electrodes[["x", "y", "z"]].to_numpy() if hasattr(electrodes, "columns") else electrodes,
        dtype=float).reshape(-1, 3)
    if not len(xyz):
        raise ValueError("No electrodes to build a mesh around.")
    ground = _surface_function(surface)
    points: List[Tuple[float, float, float]] = []
    flags: List[int] = []

    def add(x: float, y: float, z: float, flag: int) -> int:
        points.append((float(x), float(y), float(z)))
        flags.append(int(flag))
        return len(points) - 1

    tolerance = 1.0e-3
    electrode_rows, buried = [], []
    for x, y, z in xyz:
        top = ground(x, y)
        if z >= top - tolerance:
            electrode_rows.append(add(x, y, top, SURFACE))
        else:
            electrode_rows.append(add(x, y, z, INTERIOR))
            buried.append((x, y, z))
    offset = float(refine_offset)
    if offset > 0:
        for x, y, z in buried:
            add(x - offset, y - offset, z, INTERIOR)
    # Borehole tops: a surface point where each column of buried electrodes
    # meets the ground, unless an electrode already stands there.
    surface_xy = {(round(p[0], 3), round(p[1], 3)) for p, f in zip(points, flags) if f == SURFACE}
    tops = sorted({(round(x, 3), round(y, 3)) for x, y, _ in buried} - surface_xy)
    for x, y in tops:
        add(x, y, ground(x, y), SURFACE)
    if offset > 0:
        for x, y in tops:
            add(x, y, ground(x, y) - 5.0 * offset, INTERIOR)

    x0, y0 = xyz[:, 0].min() - fine_padding, xyz[:, 1].min() - fine_padding
    x1, y1 = xyz[:, 0].max() + fine_padding, xyz[:, 1].max() + fine_padding
    fine_bottom = float(xyz[:, 2].min()) - float(fine_depth_padding)
    corners = [(x0, y0), (x0, y1), (x1, y1), (x1, y0)]
    top_corner = [add(x, y, ground(x, y), SURFACE) for x, y in corners]
    bottom_corner = [add(x, y, fine_bottom, INTERIOR) for x, y in corners]
    layout = _lay_out_zones(list(zones or []), points, flags, add, ground,
                            (x0, x1, y0, y1, fine_bottom), float(fine_volume),
                            top_corner, bottom_corner)

    ox0, oy0 = x0 - outer_distance, y0 - outer_distance
    ox1, oy1 = x1 + outer_distance, y1 + outer_distance
    if int(topography_points) > 1:
        # Known elevations for the triangulation to interpolate between, kept
        # off the fine zone's walls, strictly inside the outer boundary, and
        # clear of the surface points already there.
        count = int(topography_points)
        for lo_x, hi_x, lo_y, hi_y in ((x0, x1, y0, y1), (ox0, ox1, oy0, oy1)):
            for gx in np.linspace(lo_x, hi_x, count + 2)[1:-1]:
                for gy in np.linspace(lo_y, hi_y, count + 2)[1:-1]:
                    inside_fine = x0 <= gx <= x1 and y0 <= gy <= y1
                    if (lo_x, hi_x) != (x0, x1) and inside_fine:
                        continue
                    if layout["taken"](gx, gy):
                        continue
                    add(gx, gy, ground(gx, gy), SURFACE)
    for x, y in ((ox0, oy0), (ox0, oy1), (ox1, oy1), (ox1, oy0)):
        add(x, y, ground(x, y), OUTER)

    lowest_surface = min(p[2] for p, f in zip(points, flags) if f in (SURFACE, OUTER))
    bottom = lowest_surface - float(bottom_depth)
    if bottom >= fine_bottom:
        raise ValueError(f"The mesh bottom ({bottom:g}) would not lie below the fine "
                         f"zone ({fine_bottom:g}); increase the bottom depth.")
    # Zone 1 is the outer zone, seeded beyond a corner of the fine zone and below
    # it; zone 2 the fine zone, unless the zones fill it; the zones follow, in
    # their order.
    zones = [E4DZone(1, (x0 - 0.5 * outer_distance, y0 - 0.5 * outer_distance,
                         0.5 * (fine_bottom + bottom)), float(outer_volume), float(conductivity))]
    for seed in layout["fine_seeds"]:
        zones.append(E4DZone(2, seed, float(fine_volume), float(conductivity)))
    first = 3 if layout["fine_seeds"] else 2
    for offset, entry in enumerate(layout["entries"]):
        zones.append(E4DZone(first + offset, entry["seed"], float(fine_volume),
                             1.0 / entry["resistivity"]))
    config = E4DMeshConfig(quality=float(quality), max_volume=float(outer_volume),
                           bottom=bottom, points=np.asarray(points), flags=np.asarray(flags),
                           boundaries=layout["boundaries"], zones=zones)
    config.notes = list(layout["notes"])
    return config, electrode_rows


def _lay_out_zones(zones: List[Dict[str, Any]], points: List[Tuple[float, float, float]],
                   flags: List[int], add: Callable[[float, float, float, int], int],
                   ground: Callable[[float, float], float], fine: Tuple[float, ...],
                   volume: float, top_corner: List[int],
                   bottom_corner: List[int]) -> Dict[str, Any]:
    """The fine zone's internal boundaries, with box ``zones`` inside it.

    Two kinds of zone are laid out, the way E4D models known structure:

    * a **layer** - a zone whose plan reaches the fine zone's walls on every
      side - spans the fine zone: its top and bottom are horizontal faces
      across it, and the fine zone's walls are cut into strips where they
      meet, so a layer shares the walls and the layers above and below it;
    * a **block** - any other zone - is a box of its own inside the fine zone,
      stopping short of its walls and floor by about half an element (a zone
      face may not touch them); blocks may stand apart, share a whole face, or
      sit inside a layer or another block listed before them.

    A zone reaching up to the ground has the ground as its top. Anything else
    - zones that overlap, cross a layer's face or touch along part of a face -
    is refused, naming the zones, since TetGen cannot mesh faces that cross.
    Control points are shared wherever zones meet, surface corners next to a
    surface point already there (an electrode) move onto it, and faces are
    written once.

    Returns ``boundaries`` (1-based, as the file numbers them), ``entries`` (per
    zone, in order: its ``seed``, ``resistivity`` and ``kind``), ``fine_seeds``
    (one per piece of the fine zone the zones leave free, none when they fill
    it), ``notes`` on what was adjusted, and
    ``taken(x, y)``, whether a surface point stands at or next to ``(x, y)``.
    """
    x0, x1, y0, y1, fine_bottom = fine
    plan = [(x0, y0), (x0, y1), (x1, y1), (x1, y0)]
    width, narrow = max(x1 - x0, y1 - y0), min(x1 - x0, y1 - y0)
    tiny = 1.0e-6 * width
    element = (6.0 * math.sqrt(2.0) * volume) ** (1.0 / 3.0)   # a regular tet's edge
    margin = min(max(0.01 * width, 0.5 * element), 0.1 * narrow)
    snap = 0.25 * margin
    notes: List[str] = []

    def lowest_ground(bx0: float, bx1: float, by0: float, by1: float) -> float:
        return min(ground(x, y) for x in np.linspace(bx0, bx1, 5) for y in np.linspace(by0, by1, 5))

    def taken(x: float, y: float) -> bool:
        return any(f in (SURFACE, OUTER) and math.hypot(p[0] - x, p[1] - y) <= snap
                   for p, f in zip(points, flags))

    def point(x: float, y: float, z: float, flag: int) -> int:
        for index, (p, f) in enumerate(zip(points, flags)):
            if flag == SURFACE and f in (SURFACE, OUTER) and math.hypot(p[0] - x, p[1] - y) <= tiny:
                return index + 1
            if flag == INTERIOR and f == INTERIOR and math.dist(p, (x, y, z)) <= tiny:
                return index + 1
        return add(x, y, z, flag) + 1

    ground_low = lowest_ground(x0, x1, y0, y1)
    shapes: List[Dict[str, Any]] = []
    for index, zone in enumerate(zones):
        name = zone["name"]
        layer = (zone["x"][0] <= x0 + margin and zone["x"][1] >= x1 - margin
                 and zone["y"][0] <= y0 + margin and zone["y"][1] >= y1 - margin)
        if layer:
            box = [x0, x1, y0, y1]
            roof = ground_low
        else:
            box = [max(zone["x"][0], x0 + margin), min(zone["x"][1], x1 - margin),
                   max(zone["y"][0], y0 + margin), min(zone["y"][1], y1 - margin)]
            if box[1] - box[0] <= margin or box[3] - box[2] <= margin:
                raise ValueError(
                    f"Zone {name} lies outside the E4D fine zone (x {x0:g} to {x1:g}, "
                    f"y {y0:g} to {y1:g}), so E4D cannot mesh it as a zone. Move it "
                    "under the electrodes or widen the fine zone padding.")
            roof = lowest_ground(*box)
        at_ground = zone["z"][1] >= roof - margin
        top = roof if at_ground else float(zone["z"][1])
        bottom = max(float(zone["z"][0]), fine_bottom)
        if layer and bottom <= fine_bottom + margin:
            bottom = fine_bottom                         # down to the fine zone's floor
        elif not layer:
            bottom = max(bottom, fine_bottom + margin)
        if top - bottom <= margin:
            raise ValueError(
                f"Zone {name} lies below the E4D fine zone, which ends at "
                f"{fine_bottom:g} m, or is thinner than {margin:.2g} m (half an element "
                "there), so E4D cannot mesh it as a zone. Raise it, thicken it, increase "
                "the depth below the deepest electrode, or give the fine zone a smaller "
                "element volume.")
        if not layer and ([zone["x"][0], zone["x"][1], zone["y"][0], zone["y"][1]] != box
                          or zone["z"][0] < bottom - tiny):
            notes.append(f"{name} stops {margin:.2g} m short of the fine zone's walls or "
                         "floor in the E4D configuration, as a zone face may not touch "
                         "them; its region in the mesh still reaches them.")
        shapes.append({"index": index, "name": name, "layer": layer, "box": box,
                       "bottom": bottom, "top": top, "at_ground": at_ground,
                       "asked": (float(zone["z"][0]), math.inf if at_ground else top),
                       "resistivity": float(zone["resistivity"])})

    # Layers: one level wherever a layer ends inside the fine zone, shared by the
    # layers that meet there.
    layers = sorted((s for s in shapes if s["layer"]), key=lambda s: s["bottom"])
    raw = sorted({s["bottom"] for s in layers if s["bottom"] > fine_bottom}
                 | {s["top"] for s in layers if not s["at_ground"]})
    levels: List[float] = []
    for value in raw:
        if levels and value - levels[-1] <= margin:
            continue
        levels.append(value)

    def level_of(value: float) -> float:
        return min(levels, key=lambda level: abs(level - value)) if levels else value

    for s in layers:
        if s["bottom"] > fine_bottom:
            s["bottom"] = level_of(s["bottom"])
        if not s["at_ground"]:
            s["top"] = level_of(s["top"])
    for below, above in zip(layers, layers[1:]):
        if above["bottom"] < (math.inf if below["at_ground"] else below["top"]) - tiny:
            raise ValueError(f"Layers {below['name']} and {above['name']} overlap; E4D "
                             "meshes layers that lie one on another.")

    # Blocks: kept off the layers' faces, and checked against one another.
    blocks = [s for s in shapes if not s["layer"]]
    for s in blocks:
        for level in levels:
            low, high = s["asked"]
            if low < level - margin and high > level + margin:
                raise ValueError(f"Zone {s['name']} crosses the face of a layer at "
                                 f"{level:g} m; E4D cannot mesh that. Keep it within one "
                                 "layer, or split it into two zones at the layer face.")
            if abs(s["bottom"] - level) <= margin:
                s["bottom"] = level + margin
                notes.append(f"{s['name']} starts {margin:.2g} m above the layer face "
                             f"at {level:g} m in the E4D configuration.")
            if not s["at_ground"] and abs(s["top"] - level) <= margin:
                s["top"] = level - margin
                notes.append(f"{s['name']} ends {margin:.2g} m below the layer face "
                             f"at {level:g} m in the E4D configuration.")
        if (math.inf if s["at_ground"] else s["top"]) - s["bottom"] <= margin:
            raise ValueError(
                f"Zone {s['name']} does not fit between the layer faces around it: E4D "
                f"keeps a zone {margin:.2g} m (half an element) away from a layer face, "
                "which leaves it no room. Keep its top and bottom that far from the "
                "faces, or give the fine zone a smaller element volume.")

    def extent(s: Dict[str, Any]) -> List[float]:
        return s["box"] + [s["bottom"], math.inf if s["at_ground"] else s["top"]]

    inner: Dict[int, List[Dict[str, Any]]] = {s["index"]: [] for s in shapes}
    for i, a in enumerate(blocks):
        for b in blocks[i + 1:]:
            ea, eb = extent(a), extent(b)
            spans = [min(ea[2 * k + 1], eb[2 * k + 1]) - max(ea[2 * k], eb[2 * k])
                     for k in range(3)]
            if min(spans) < -tiny:
                continue                                          # apart
            touching = [k for k in range(3) if abs(spans[k]) <= tiny]
            if not touching:
                for outer_s, inner_s in ((a, b), (b, a)):
                    eo, ei = extent(outer_s), extent(inner_s)
                    inside = all(ei[2 * k] > eo[2 * k] + tiny for k in range(3)) and all(
                        ei[2 * k + 1] < eo[2 * k + 1] - tiny or (
                            k == 2 and math.isinf(ei[5]) and math.isinf(eo[5]))
                        for k in range(3))
                    if inside:
                        if inner_s["index"] < outer_s["index"]:
                            raise ValueError(
                                f"Zone {inner_s['name']} lies inside zone "
                                f"{outer_s['name']}; list it after {outer_s['name']}, "
                                "so that it keeps its cells.")
                        inner[outer_s["index"]].append(inner_s)
                        break
                else:
                    raise ValueError(f"Zones {a['name']} and {b['name']} overlap; E4D "
                                     "meshes zones that stand apart, share a whole face, "
                                     "or lie one inside another.")
                continue
            k = touching[0]
            same = all(abs(ea[2 * j] - eb[2 * j]) <= tiny and (
                abs(ea[2 * j + 1] - eb[2 * j + 1]) <= tiny
                or (math.isinf(ea[2 * j + 1]) and math.isinf(eb[2 * j + 1])))
                for j in range(3) if j != k)
            if len(touching) > 1 or not same:
                raise ValueError(
                    f"Zones {a['name']} and {b['name']} touch along part of a face. E4D "
                    "can mesh zones that share a whole face, or that stand apart; make "
                    "the shared face the same size in both, or leave a gap.")
    for s in layers:
        inner[s["index"]] = [b for b in blocks
                             if s["bottom"] - tiny <= b["bottom"] and (
                                 s["at_ground"] or b["top"] <= s["top"] + tiny)]

    # Faces, written once each; control points shared where zones meet.
    boundaries: List[Tuple[int, List[int]]] = []
    seen: set = set()

    def face(indices: List[int]) -> None:
        if len(set(indices)) >= 3 and frozenset(indices) not in seen:
            seen.add(frozenset(indices))
            boundaries.append((INTERNAL_BOUNDARY, list(indices)))

    ring_at = {fine_bottom: [i + 1 for i in bottom_corner]}
    for level in levels:
        ring_at[level] = [point(x, y, level, INTERIOR) for x, y in plan]
    stack = [ring_at[fine_bottom]] + [ring_at[level] for level in levels] + [
        [i + 1 for i in top_corner]]
    for lower, upper in zip(stack, stack[1:]):                  # the walls, in strips
        for k in range(4):
            face([upper[k], upper[(k + 1) % 4], lower[(k + 1) % 4], lower[k]])
    face(list(ring_at[fine_bottom]))                             # the floor
    for level in levels:
        face(list(ring_at[level]))                               # the layer faces

    def settle(x: float, y: float) -> Tuple[float, float]:
        """A surface corner beside a surface point already there moves onto it."""
        for p, f in zip(points, flags):
            if f in (SURFACE, OUTER) and math.hypot(p[0] - x, p[1] - y) <= snap:
                return float(p[0]), float(p[1])
        return x, y

    for s in blocks:
        bx0, bx1, by0, by1 = s["box"]
        corners_xy = [(bx0, by0), (bx0, by1), (bx1, by1), (bx1, by0)]
        if s["at_ground"]:
            corners_xy = [settle(x, y) for x, y in corners_xy]
            tops = [point(x, y, ground(x, y), SURFACE) for x, y in corners_xy]
        else:
            tops = [point(x, y, s["top"], INTERIOR) for x, y in corners_xy]
        lows = [point(x, y, s["bottom"], INTERIOR) for x, y in corners_xy]
        for k in range(4):
            face([tops[k], tops[(k + 1) % 4], lows[(k + 1) % 4], lows[k]])
        face(list(lows))
        if not s["at_ground"]:
            face(list(tops))

    # Seeds: a point well inside each zone, and outside the zones inside it.
    fractions = (0.5137, 0.3571, 0.6429, 0.2143, 0.7857, 0.0714, 0.9286)

    def inside(p: Tuple[float, float, float], s: Dict[str, Any]) -> bool:
        bx0, bx1, by0, by1 = s["box"]
        roof = ground(p[0], p[1]) if s["at_ground"] else s["top"]
        return bx0 <= p[0] <= bx1 and by0 <= p[1] <= by1 and s["bottom"] <= p[2] <= roof

    def seed_in(box: List[float], bottom: float, at_ground: bool, top: float,
                avoid: List[Dict[str, Any]]) -> Optional[Tuple[float, float, float]]:
        for fz in fractions:
            for fy in fractions:
                for fx in fractions:
                    x = box[0] + fx * (box[1] - box[0])
                    y = box[2] + fy * (box[3] - box[2])
                    roof = ground(x, y) if at_ground else top
                    candidate = (x, y, bottom + fz * (roof - bottom))
                    if not any(inside(candidate, other) for other in avoid):
                        return candidate
        return None

    entries = []
    for s in shapes:
        seed = seed_in(s["box"], s["bottom"], s["at_ground"], s["top"], inner[s["index"]])
        if seed is None:
            raise ValueError(f"Zone {s['name']} is entirely taken by the zones inside it.")
        entries.append({"seed": seed, "resistivity": s["resistivity"],
                        "kind": "layer" if s["layer"] else "block"})
    # The fine zone: a seed in every piece of it the zones leave free. A buried
    # layer cuts it in two, and TetGen numbers an unseeded piece as a zone of
    # its own.
    centre = (0.5 * (x0 + x1) + 0.013 * (x1 - x0), 0.5 * (y0 + y1) + 0.017 * (y1 - y0))
    fine_seeds: List[Tuple[float, float, float]] = []
    if not shapes:
        fine_seeds.append((centre[0], centre[1], 0.5 * (ground(*centre) + fine_bottom)))
    else:
        fractions = (0.004, 0.996) + fractions               # into the shell by the walls
        taken_spans = {(s["bottom"], None if s["at_ground"] else s["top"]) for s in layers}
        for low, high in zip([fine_bottom] + levels, levels + [None]):
            if (low, high) in taken_spans:
                continue
            seed = seed_in([x0, x1, y0, y1], low, high is None,
                           0.0 if high is None else high, blocks)
            if seed is not None:
                fine_seeds.append(seed)
        if not fine_seeds:
            notes.append("The zones fill the fine zone, so it has no cells of its own; "
                         "the zones are numbered from 2 in the E4D configuration.")
    return {"boundaries": boundaries, "entries": entries, "fine_seeds": fine_seeds,
            "notes": notes, "taken": taken}


# ---------------------------------------------------------------------------
# The piecewise linear complex E4D hands to TetGen
# ---------------------------------------------------------------------------
@dataclass
class E4DPLC:
    """The PLC of an E4D mesh, in translated coordinates.

    ``facets`` are ``(node indices, marker)`` polygons, 0-based: the surface
    triangles (marker 1), the outer walls and bottom (2), and the internal
    boundaries (their ``b_num``). ``regions`` are ``(seed, zone, max volume)``.
    """

    nodes: np.ndarray
    node_markers: np.ndarray
    facets: List[Tuple[List[int], int]]
    regions: List[Tuple[np.ndarray, int, float]]
    holes: np.ndarray
    translation: np.ndarray
    counts: Dict[str, int]
    control_nodes: np.ndarray            # PLC node of every control point

    def internal_markers(self) -> List[int]:
        return sorted({marker for _, marker in self.facets if marker not in (SURFACE, OUTER)})


def _interpolator(xy: np.ndarray, z: np.ndarray):
    """Linear interpolation of elevation in plan, nearest outside the hull."""
    if len(xy) >= 3 and np.ptp(xy[:, 0]) > 0 and np.ptp(xy[:, 1]) > 0:
        from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

        linear, nearest = LinearNDInterpolator(xy, z), NearestNDInterpolator(xy, z)

        def elevation(points: np.ndarray) -> np.ndarray:
            out = linear(points)
            missing = ~np.isfinite(out)
            out[missing] = nearest(points[missing])
            return out

        return elevation
    level = float(np.mean(z)) if len(z) else 0.0
    return lambda points: np.full(len(points), level)


def build_e4d_plc(config: E4DMeshConfig, *, surface_quality: float = 30.0,
                  surface_area: float = 0.0) -> E4DPLC:
    """The PLC E4D builds from ``config``, ready for TetGen or Gmsh.

    ``surface_quality`` is the minimum angle (degrees) of the surface
    triangulation and ``surface_area`` its largest triangle, 0 for none.
    """
    import pygimli as pg
    import pygimli.meshtools as mt

    problems = config.validate()
    if problems:
        raise ValueError("The E4D mesh configuration is not valid: " + " ".join(problems))
    shift = config.translation()
    points = config.points - shift
    flags = config.flags
    surface_ids = [i for i in range(len(points)) if flags[i] in (SURFACE, OUTER)]
    outer = config.outer_indices()

    # 1. The surface in plan: outer boundary and internal-boundary traces as
    #    segments. Triangle may split a segment; the new nodes are found below.
    plan = pg.Mesh(2)
    plan_node = {i: plan.createNode([points[i, 0], points[i, 1], 0.0]) for i in surface_ids}
    edges: Dict[Tuple[int, int], int] = {}
    for a, b in zip(outer, outer[1:] + outer[:1]):
        edges[(min(a, b), max(a, b))] = OUTER
    traces: List[Tuple[int, int, int]] = []           # (a, b, trace marker)
    for number, (_, indices) in enumerate(config.boundaries, start=1):
        rows = [int(i) - 1 for i in indices]
        for a, b in zip(rows, rows[1:] + rows[:1]):
            if flags[a] in (SURFACE, OUTER) and flags[b] in (SURFACE, OUTER):
                key = (min(a, b), max(a, b))
                if key not in edges:
                    edges[key] = 3 + number                # E4D numbers them 4, 5, ...
                    traces.append((a, b, 3 + number))
    for (a, b), marker in edges.items():
        plan.createEdge(plan_node[a], plan_node[b], marker=marker)
    surface = mt.createMesh(plan, quality=float(surface_quality), area=float(surface_area))

    plan_xy = np.asarray(surface.positions(), dtype=float)[:, :2]
    known = np.asarray(surface_ids)
    elevation = _interpolator(points[known, :2], points[known, 2])(plan_xy)
    # The input vertices keep their own elevation, wherever Triangle put them.
    from scipy.spatial import cKDTree

    tree = cKDTree(plan_xy)
    scale = max(float(np.ptp(points[:, :2], axis=0).max()), 1.0)
    control_nodes = np.full(len(points), -1, dtype=int)
    for i in surface_ids:
        distance, node = tree.query(points[i, :2])
        if distance > 1.0e-9 * scale:
            raise RuntimeError(f"Surface point {i + 1} was lost in the surface triangulation.")
        control_nodes[i] = int(node)
        elevation[node] = points[i, 2]
    # Node markers as E4D writes them: 2 on the outer boundary, the trace's
    # number on an internal boundary's surface trace, 1 elsewhere; a control
    # point keeps its own flag.
    surface_markers = np.full(len(plan_xy), SURFACE, dtype=int)
    on_segment: Dict[int, set] = {}
    for boundary in surface.boundaries():
        marker = int(boundary.marker())
        if marker == OUTER or marker > 3:
            for node in boundary.nodes():
                on_segment.setdefault(marker, set()).add(int(node.id()))
                if surface_markers[node.id()] != OUTER:
                    surface_markers[node.id()] = marker
    for i in surface_ids:
        surface_markers[control_nodes[i]] = int(flags[i])

    # 2. The perimeter, walked in the order the outer points are listed.
    neighbours: Dict[int, List[int]] = {}
    for boundary in surface.boundaries():
        if int(boundary.marker()) == OUTER:
            a, b = (int(n.id()) for n in boundary.nodes())
            neighbours.setdefault(a, []).append(b)
            neighbours.setdefault(b, []).append(a)
    start, towards = int(control_nodes[outer[0]]), int(control_nodes[outer[1]])
    ring = [start]
    previous, current = None, start
    while True:
        options = [n for n in neighbours[current] if n != previous]
        if previous is None:
            # Leave the first corner along the side that leads to the second.
            options.sort(key=lambda n: np.linalg.norm(plan_xy[n] - plan_xy[towards]))
        following = options[0]
        if following == start:
            break
        ring.append(following)
        previous, current = current, following
        if len(ring) > len(plan_xy):
            raise RuntimeError("The outer boundary of the surface does not close.")

    n_surface = len(plan_xy)
    nodes = [np.column_stack([plan_xy, elevation])]
    markers = [surface_markers]
    bottom_nodes = n_surface + np.arange(len(ring))
    nodes.append(np.column_stack([plan_xy[ring], np.full(len(ring), config.bottom - shift[2])]))
    markers.append(np.full(len(ring), OUTER, dtype=int))
    interior = [i for i in range(len(points)) if flags[i] not in (SURFACE, OUTER)]
    control_nodes[interior] = n_surface + len(ring) + np.arange(len(interior))
    nodes.append(points[interior])
    markers.append(np.asarray([flags[i] for i in interior], dtype=int))

    # 3. Facets: surface, walls, bottom, then the internal boundaries with the
    #    surface nodes Triangle placed on their top edges.
    facets: List[Tuple[List[int], int]] = [
        ([int(n.id()) for n in cell.nodes()], SURFACE) for cell in surface.cells()]
    for k in range(len(ring)):
        a, b = ring[k], ring[(k + 1) % len(ring)]
        facets.append(([a, b, int(bottom_nodes[(k + 1) % len(ring)]), int(bottom_nodes[k])], OUTER))
    facets.append(([int(n) for n in bottom_nodes], OUTER))
    trace_marker = {(min(a, b), max(a, b)): m for a, b, m in traces}
    for number, (b_num, indices) in enumerate(config.boundaries, start=1):
        rows = [int(i) - 1 for i in indices]
        polygon: List[int] = []
        for a, b in zip(rows, rows[1:] + rows[:1]):
            polygon.append(int(control_nodes[a]))
            key = (min(a, b), max(a, b))
            if key in trace_marker or edges.get(key) == OUTER:
                start_xy, end_xy = plan_xy[control_nodes[a]], plan_xy[control_nodes[b]]
                span = end_xy - start_xy
                length2 = float(span @ span)
                between = []
                for node in on_segment.get(trace_marker.get(key, OUTER), ()):
                    t = float((plan_xy[node] - start_xy) @ span) / length2
                    off = np.linalg.norm(start_xy + t * span - plan_xy[node])
                    if 1e-9 < t < 1 - 1e-9 and off < 1e-7 * scale:
                        between.append((t, node))
                polygon.extend(node for _, node in sorted(between))
        facets.append((polygon, int(b_num)))

    regions = [(np.asarray(zone.seed, dtype=float) - shift, int(zone.index),
                float(zone.max_volume)) for zone in config.zones]
    return E4DPLC(nodes=np.vstack(nodes), node_markers=np.concatenate(markers), facets=facets,
                  regions=regions, holes=config.holes - shift, translation=shift,
                  counts={"surface": n_surface, "bottom": len(ring), "interior": len(interior),
                          "surface_facets": surface.cellCount(),
                          "internal_facets": len(config.boundaries)},
                  control_nodes=control_nodes)


def write_tetgen_poly(plc: E4DPLC, path: Union[str, Path]) -> Path:
    """Write ``plc`` as a TetGen ``.poly``, laid out and numbered as E4D writes it."""
    counts = plc.counts
    lines = [f"{len(plc.nodes)}  3  1  1  # nodes, dimension, attributes, boundary markers",
             f"# The first {counts['surface']} points are the surface, triangulated in plan",
             f"# then {counts['bottom']} points of the lower boundary and "
             f"{counts['interior']} internal points",
             f"# coordinates translated by {' '.join(_number(v) for v in plc.translation)}"]
    for number, (point, marker) in enumerate(zip(plc.nodes, plc.node_markers), start=1):
        lines.append(f"{number} {point[0]:.12e} {point[1]:.12e} {point[2]:.12e} 1 {int(marker)}")
    lines.append(f"{len(plc.facets)}  1  # facets, boundary markers "
                 "(1 surface, 2 walls and bottom, others internal)")
    for polygon, marker in plc.facets:
        lines.append(f"1 0 {int(marker)}")
        lines.append(f"{len(polygon)} " + " ".join(str(i + 1) for i in polygon))
    lines.append(f"{len(plc.holes)}  # holes")
    for number, hole in enumerate(plc.holes, start=1):
        lines.append(f"{number} {hole[0]:.12e} {hole[1]:.12e} {hole[2]:.12e}")
    lines.append(f"{len(plc.regions)}  # regions: seed, zone, maximum volume")
    for number, (seed, zone, volume) in enumerate(plc.regions, start=1):
        lines.append(f"{number} {seed[0]:.12e} {seed[1]:.12e} {seed[2]:.12e} "
                     f"{int(zone)} {_number(volume)}")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="ascii")
    return target


# ---------------------------------------------------------------------------
# Meshing: TetGen as E4D runs it, or Gmsh in its place
# ---------------------------------------------------------------------------
def find_tetgen(hint: str = "") -> Optional[str]:
    """The TetGen program: ``hint`` if it names one, ``$PYHYDRO_TETGEN``, PATH,
    or the running environment's own ``bin`` (a conda-forge ``tetgen``)."""
    import sys

    for candidate in (hint, os.environ.get("PYHYDRO_TETGEN", ""), "tetgen"):
        candidate = str(candidate or "").strip().strip("'\"")
        if not candidate:
            continue
        if Path(candidate).is_file():
            return str(Path(candidate))
        found = shutil.which(candidate)
        if found:
            return found
    for folder in (Path(sys.prefix) / "Library" / "bin", Path(sys.prefix) / "bin"):
        for name in ("tetgen.exe", "tetgen"):
            if (folder / name).is_file():
                return str(folder / name)
    return None


def _tetgen_switches(quality: float, max_volume: float) -> str:
    """E4D's own call: PLC, neighbours, quality, a default volume, region volumes
    and region attributes - ``-pnq1.28a1000000000000.0aAA`` for Van Nuys."""
    return f"-pnq{_number(quality)}a{_number(max_volume)}aAA"


def _tetgen_module():
    """The TetGen Python package (``pip install tetgen``), or None."""
    try:
        import tetgen
    except Exception:  # noqa: BLE001 - optional
        return None
    return tetgen if hasattr(tetgen, "TetGen") else None


def _triangulate_polygon(points: np.ndarray) -> List[Tuple[int, int, int]]:
    """Triangles covering a planar polygon, using its own vertices only.

    Ear clipping in the polygon's plane. A vertex in line with its neighbours
    is never an ear, and an ear with another vertex on or inside it is not
    cut, so vertices lying along an edge - the surface nodes Triangle put on
    an internal boundary's top edge - stay on triangle edges, and the facet
    still meets its neighbours node for node, as TetGen needs.
    """
    n = len(points)
    if n == 3:
        return [(0, 1, 2)]
    normal = np.zeros(3)
    for i in range(n):                         # Newell's normal
        a, b = points[i], points[(i + 1) % n]
        normal += ((a[1] - b[1]) * (a[2] + b[2]), (a[2] - b[2]) * (a[0] + b[0]),
                   (a[0] - b[0]) * (a[1] + b[1]))
    normal /= np.linalg.norm(normal)
    axis = np.eye(3)[int(np.argmin(np.abs(normal)))]
    u = np.cross(normal, axis)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    xy = np.column_stack([(points - points[0]) @ u, (points - points[0]) @ v])
    scale = max(float(np.ptp(xy, axis=0).max()), 1e-300)
    eps = 1e-12 * scale * scale

    def cross(a, b, c):
        return (xy[b, 0] - xy[a, 0]) * (xy[c, 1] - xy[a, 1]) - \
               (xy[b, 1] - xy[a, 1]) * (xy[c, 0] - xy[a, 0])

    ring = list(range(n))
    if sum(cross(0, ring[k], ring[k + 1]) for k in range(1, n - 1)) < 0:
        ring.reverse()                         # counter-clockwise
    triangles: List[Tuple[int, int, int]] = []
    while len(ring) > 3:
        for k in range(len(ring)):
            a, b, c = ring[k - 1], ring[k], ring[(k + 1) % len(ring)]
            if cross(a, b, c) <= eps:
                continue                       # reflex, or in line
            blocked = False
            for j in ring:
                if j in (a, b, c):
                    continue
                if (cross(a, b, j) >= -eps and cross(b, c, j) >= -eps
                        and cross(c, a, j) >= -eps):
                    blocked = True
                    break
            if not blocked:
                triangles.append((a, b, c))
                ring.pop(k)
                break
        else:
            raise ValueError("A facet polygon could not be triangulated; is it simple and planar?")
    if cross(*ring) > eps:
        triangles.append(tuple(ring))
    return triangles


def _mesh_with_tetgen_module(plc: E4DPLC, config: E4DMeshConfig, log: LogFn,
                             tighten: Optional[Dict[int, float]] = None):
    """TetGen through its Python package, with E4D's switches and regions.

    The package takes triangles, so each facet is triangulated first in its
    own plane; TetGen merges coplanar triangles of one marker back into a facet.
    ``tighten`` scales zones' volume limits for this call only.
    """
    import pygimli as pg

    tetgen = _tetgen_module()
    triangles, markers = [], []
    for polygon, marker in plc.facets:
        for a, b, c in _triangulate_polygon(plc.nodes[polygon]):
            triangles.append((polygon[a], polygon[b], polygon[c]))
            markers.append(int(marker))
    generator = tetgen.TetGen(np.ascontiguousarray(plc.nodes, dtype=np.float64),
                              np.asarray(triangles, dtype=np.int32),
                              np.asarray(markers, dtype=np.int32))
    for seed, zone, volume in plc.regions:
        factor = (tighten or {}).get(int(zone), 1.0)
        generator.add_region(int(zone), [float(v) for v in seed], float(volume) * factor)
    for hole in plc.holes:
        generator.add_hole([float(v) for v in hole])
    switches = (f"pzq{_number(config.quality)}a{_number(config.max_volume)}aAAfQ")
    log(f"  TetGen (Python package {getattr(tetgen, '__version__', '')}): -{switches}")
    nodes, elements, attributes, _ = generator.tetrahedralize(switches=switches)
    zone = np.rint(np.asarray(attributes, dtype=float).reshape(len(elements), -1)[:, 0]
                   ).astype(int) if np.size(attributes) else np.zeros(len(elements), dtype=int)
    mesh = pg.Mesh(3)
    for point in np.asarray(nodes, dtype=float):
        mesh.createNode(pg.Pos(*point))
    for ids, marker in zip(np.asarray(elements, dtype=int), zone):
        mesh.createCell([int(i) for i in ids[:4]], marker=int(marker))
    faces = np.asarray(generator.trifaces, dtype=int)
    face_markers = np.asarray(generator.triface_markers, dtype=int)
    for ids, marker in zip(faces, face_markers):
        if marker:
            mesh.createBoundary([int(i) for i in ids[:3]], marker=int(marker))
    mesh.createNeighborInfos()
    mesh.fixBoundaryDirections()
    return mesh


def _run(command: Sequence[str], cwd: Path, log: LogFn, what: str) -> None:
    log(f"  {what}: {' '.join(Path(command[0]).name if i == 0 else c for i, c in enumerate(command))}")
    completed = subprocess.run(list(command), cwd=str(cwd), capture_output=True, text=True,
                               timeout=3600, check=False)
    if completed.returncode != 0:
        tail = (completed.stderr or completed.stdout or "").strip().splitlines()[-6:]
        raise RuntimeError(f"{what} failed (exit {completed.returncode}): " + " ".join(tail))


def _read_table(path: Path) -> Tuple[List[str], np.ndarray]:
    """A TetGen table: its header and its ``count`` rows, whatever follows them."""
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        header, skip = None, 0
        for skip, raw in enumerate(handle, start=1):
            text = raw.split("#", 1)[0].split()
            if text:
                header = text
                break
    if header is None:
        raise ValueError(f"{path.name} is empty.")
    rows = np.loadtxt(path, comments="#", skiprows=skip, max_rows=int(header[0]), ndmin=2)
    return header, rows


def _unmark_inner_faces(mesh) -> None:
    """Leave the faces between cells unmarked, as a PyGIMLi inversion needs.

    E4D smooths across its internal boundaries within a zone and decouples by
    zone; PyGIMLi would cut the smoothness at every marked face. The zones stay
    cell markers, and the outer faces keep theirs.
    """
    from PyHydroGeophysX.core._mesh_3d_builder import clear_inner_face_markers

    clear_inner_face_markers(mesh)


def _mesh_name(base: Path) -> str:
    """The name E4D gave a mesh, from its TetGen base name: TetGen appends the
    iteration, so ``site.v2.1`` is mesh ``site.v2``, whose translation and
    conductivities are ``site.v2.trn`` and ``site.v2.sig``."""
    head, _, tail = base.name.rpartition(".")
    return head if head and tail.isdigit() else base.name


def read_e4d_mesh(base: Union[str, Path], *, translation: Optional[Sequence[float]] = None,
                  conductivity: Union[bool, str, Path] = True):
    """Read an E4D (TetGen) mesh into a PyGIMLi mesh, in survey coordinates.

    ``base`` is the mesh's base name (``VanNuys.1``) or any of its files. The
    translation is read from the ``.trn`` beside it unless given. Cell markers
    are the zone numbers; the outer faces' markers come from the ``.face``
    file, and the faces between cells are left unmarked, so a PyGIMLi
    inversion smooths within a zone as E4D does. With
    ``conductivity``, the ``.sig`` beside it - or the file named - is attached
    as cell data ``conductivity`` (S/m) and ``resistivity`` (ohm-m).

    Written for E4D's files, which number from 1 and whose ``.node`` ends with
    a line E4D appends; PyGIMLi's ``readTetgen`` reads neither.
    """
    import pygimli as pg

    base = Path(base)
    if base.suffix.lower() in (".node", ".ele", ".face", ".neigh", ".edge"):
        base = base.with_suffix("")
    node_path, ele_path = Path(f"{base}.node"), Path(f"{base}.ele")
    if not node_path.is_file() or not ele_path.is_file():
        raise FileNotFoundError(f"{base}.node and {base}.ele are both needed.")
    _, nodes = _read_table(node_path)
    ele_header, elements = _read_table(ele_path)
    first = int(nodes[0, 0])                 # TetGen numbers from 0 or 1

    if translation is None:
        trn = base.parent / f"{_mesh_name(base)}.trn"
        translation = (np.loadtxt(trn).reshape(-1)[:3] if trn.is_file() else np.zeros(3))
    shift = np.asarray(translation, dtype=float).reshape(3)

    mesh = pg.Mesh(3)
    has_marker = nodes.shape[1] > 4
    for row in nodes:
        mesh.createNode(pg.Pos(*(row[1:4] + shift)), int(row[-1]) if has_marker else 0)
    attributes = int(ele_header[2]) if len(ele_header) > 2 else 0
    corner = elements[:, 1:5].astype(int) - first
    zone = elements[:, 5].astype(int) if attributes else np.zeros(len(elements), dtype=int)
    for ids, marker in zip(corner, zone):
        mesh.createCell([int(i) for i in ids], marker=int(marker))
    face_path = Path(f"{base}.face")
    if face_path.is_file():
        face_header, faces = _read_table(face_path)
        marked = int(face_header[1]) if len(face_header) > 1 else 0
        for row in faces:
            mesh.createBoundary([int(i) - first for i in row[1:4]],
                                marker=int(row[4]) if marked else 0)
    mesh.createNeighborInfos()
    mesh.fixBoundaryDirections()
    _unmark_inner_faces(mesh)

    if conductivity:
        sig = Path(conductivity) if isinstance(conductivity, (str, Path)) else \
            base.parent / f"{_mesh_name(base)}.sig"
        if sig.is_file():
            header, values = _read_table(sig)
            if int(header[0]) == mesh.cellCount():
                sigma = values[:, 0]
                mesh["conductivity"] = sigma
                mesh["resistivity"] = np.where(sigma > 0, 1.0 / np.maximum(sigma, 1e-300), np.nan)
    return mesh


def _edge_length(volume: float) -> float:
    """The edge of a regular tetrahedron of ``volume``: V = a^3 / (6 sqrt 2)."""
    return float((6.0 * math.sqrt(2.0) * float(volume)) ** (1.0 / 3.0))


def _write_geo(plc: E4DPLC, path: Path, fine_scale: float = 0.9) -> None:
    """The PLC as a Gmsh geometry: one volume, the internal boundaries and the
    internal points embedded in it, sizes at the points.

    Gmsh aims at a size rather than enforcing a limit, so the finest zone is
    asked for ``fine_scale`` times the edge of its largest allowed element.
    """
    from scipy.spatial import cKDTree

    nodes = plc.nodes
    distance, _ = cKDTree(nodes).query(nodes, k=2)
    nearest = distance[:, 1]
    finite = [volume for _, _, volume in plc.regions if volume < UNCONSTRAINED]
    fine = (float(fine_scale) * _edge_length(min(finite)) if finite
            else float(np.ptp(nodes, axis=0).max()) / 20.0)
    # Sizes: along the surface and the walls, the spacing the triangulation
    # already has there; at internal points and the internal boundaries, the
    # finest zone's edge, or less where points crowd closer than that.
    incident: Dict[int, List[float]] = {}
    lines: Dict[Tuple[int, int], int] = {}
    loops: List[List[int]] = []
    for polygon, _ in plc.facets:
        loop = []
        for a, b in zip(polygon, polygon[1:] + polygon[:1]):
            key = (min(a, b), max(a, b))
            if key not in lines:
                lines[key] = len(lines) + 1
                length = float(np.linalg.norm(nodes[a] - nodes[b]))
                incident.setdefault(a, []).append(length)
                incident.setdefault(b, []).append(length)
            loop.append(lines[key] if key == (a, b) else -lines[key])
        loops.append(loop)
    boundary_like = {i for polygon, marker in plc.facets if marker in (SURFACE, OUTER)
                     for i in polygon}
    size = np.empty(len(nodes))
    for i in range(len(nodes)):
        spacing = float(np.mean(incident[i])) if i in incident else fine
        if i in boundary_like:
            size[i] = min(spacing, 0.9 * nearest[i]) if nearest[i] < spacing else spacing
        else:
            size[i] = min(fine, 0.9 * nearest[i])
    out = ["// E4D-style PLC; coordinates translated as in the .trn file",
           "Mesh.CharacteristicLengthFromPoints = 1;",
           "Mesh.CharacteristicLengthExtendFromBoundary = 1;",
           "Mesh.CharacteristicLengthFromCurvature = 0;",
           "Mesh.Algorithm3D = 1;", "Mesh.Optimize = 1;", "Mesh.MshFileVersion = 2.2;"]
    for i, (point, lc) in enumerate(zip(nodes, size), start=1):
        out.append(f"Point({i}) = {{{point[0]:.12g}, {point[1]:.12g}, {point[2]:.12g}, "
                   f"{lc:.6g}}};")
    for (a, b), number in lines.items():
        out.append(f"Line({number}) = {{{a + 1}, {b + 1}}};")
    for number, loop in enumerate(loops, start=1):
        out.append(f"Curve Loop({number}) = {{{', '.join(str(v) for v in loop)}}};")
        out.append(f"Plane Surface({number}) = {{{number}}};")
    outside = [n for n, (_, m) in enumerate(plc.facets, start=1) if m in (SURFACE, OUTER)]
    inside = [n for n, (_, m) in enumerate(plc.facets, start=1) if m not in (SURFACE, OUTER)]
    out.append(f"Surface Loop(1) = {{{', '.join(map(str, outside))}}};")
    out.append("Volume(1) = {1};")
    if inside:
        out.append(f"Surface{{{', '.join(map(str, inside))}}} In Volume{{1}};")
    on_facets = {i for polygon, _ in plc.facets for i in polygon}
    loose = [i + 1 for i in range(len(nodes)) if i not in on_facets]
    if loose:
        out.append(f"Point{{{', '.join(map(str, loose))}}} In Volume{{1}};")
    by_marker: Dict[int, List[int]] = {}
    for number, (_, marker) in enumerate(plc.facets, start=1):
        by_marker.setdefault(int(marker), []).append(number)
    for marker, numbers in sorted(by_marker.items()):
        out.append(f"Physical Surface({abs(marker)}) = {{{', '.join(map(str, numbers))}}};")
    out.append("Physical Volume(1) = {1};")
    # Sizes set at points thin out between them, so inside the internal
    # boundaries - where the constrained zones are - a box field holds the
    # finest size throughout. Gmsh takes the smallest of its size sources, and
    # outside the box the field asks for nothing.
    internal = [i for polygon, marker in plc.facets if marker not in (SURFACE, OUTER)
                for i in polygon]
    if internal and finite:
        low, high = nodes[internal].min(axis=0), nodes[internal].max(axis=0)
        out += ["Field[1] = Box;", f"Field[1].VIn = {fine:.6g};", "Field[1].VOut = 1e22;",
                f"Field[1].XMin = {low[0]:.12g};", f"Field[1].XMax = {high[0]:.12g};",
                f"Field[1].YMin = {low[1]:.12g};", f"Field[1].YMax = {high[1]:.12g};",
                f"Field[1].ZMin = {low[2]:.12g};", f"Field[1].ZMax = {high[2]:.12g};",
                "Background Field = 1;"]
    path.write_text("\n".join(out) + "\n", encoding="ascii")


def _assign_zones(mesh, plc: E4DPLC, log: LogFn) -> None:
    """Zone numbers for a mesh that knows only its internal boundaries.

    The cells fall apart into connected pieces once the faces lying on an
    internal boundary are cut; each piece takes the zone whose seed it holds.
    """
    import pygimli as pg
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    cells = np.asarray([cell.ids() for cell in mesh.cells()], dtype=np.int64)
    n_nodes = np.int64(mesh.nodeCount())
    faces = np.sort(cells[:, [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]].reshape(-1, 3), axis=1)
    keys = (faces[:, 0] * n_nodes + faces[:, 1]) * n_nodes + faces[:, 2]
    owner = np.repeat(np.arange(len(cells)), 4)
    internal = set(plc.internal_markers())
    cut = [sorted(int(n.id()) for n in boundary.nodes())
           for boundary in mesh.boundaries() if abs(int(boundary.marker())) in
           {abs(m) for m in internal}]
    cut_keys = {(a * int(n_nodes) + b) * int(n_nodes) + c for a, b, c in cut}
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    pairs = np.flatnonzero(sorted_keys[1:] == sorted_keys[:-1])
    left, right = owner[order[pairs]], owner[order[pairs + 1]]
    keep = np.asarray([int(k) not in cut_keys for k in sorted_keys[pairs]], dtype=bool)
    graph = coo_matrix((np.ones(int(keep.sum())), (left[keep], right[keep])),
                       shape=(len(cells), len(cells)))
    count, piece = connected_components(graph, directed=False)
    zone_of_piece: Dict[int, int] = {}
    for seed, zone, _ in plc.regions:
        cell = mesh.findCell(pg.Pos(*seed))
        if cell is None:
            log(f"  zone {zone}: its seed point lies outside the mesh")
            continue
        zone_of_piece.setdefault(int(piece[cell.id()]), int(zone))
    unclaimed = sorted(set(range(count)) - set(zone_of_piece))
    if unclaimed:
        log(f"  {len(unclaimed)} part(s) of the mesh hold no zone seed and are given zone 0")
    for cell in mesh.cells():
        cell.setMarker(zone_of_piece.get(int(piece[cell.id()]), 0))


def _find_gmsh(hint: str = "") -> Optional[str]:
    """Gmsh: ``hint``, PATH, or the build ResIPy bundles - located without
    importing ResIPy, whose import prints a banner and checks its own solvers."""
    if hint and Path(hint).is_file():
        return hint
    found = shutil.which("gmsh")
    if found:
        return found
    import importlib.util

    try:
        spec = importlib.util.find_spec("resipy")
    except (ImportError, ValueError):       # a broken or blocked install
        spec = None
    for folder in (spec.submodule_search_locations or []) if spec else []:
        for name in ("gmsh.exe", "gmsh"):
            candidate = Path(folder) / "exe" / name
            if candidate.is_file():
                return str(candidate)
    return None


def zone_report(mesh, config: E4DMeshConfig) -> List[Dict[str, Any]]:
    """Per zone: cells, the largest element volume and the constraint it had."""
    markers = np.asarray(mesh.cellMarkers(), dtype=int)
    volumes = np.asarray([cell.size() for cell in mesh.cells()], dtype=float)
    report = []
    for zone in {zone.index: zone for zone in config.zones}.values():
        sel = markers == zone.index
        report.append({"zone": int(zone.index), "cells": int(sel.sum()),
                       "largest_volume": float(volumes[sel].max()) if sel.any() else 0.0,
                       "max_volume": float(zone.max_volume),
                       "conductivity": float(zone.conductivity)})
    return report


def build_e4d_mesh(config: E4DMeshConfig, workdir: Union[str, Path], *, name: str = "e4d_mesh",
                   mesher: str = "auto", tetgen: str = "", gmsh: str = "",
                   log: LogFn = _noop) -> Dict[str, Any]:
    """Build the mesh ``config`` describes, the way E4D does.

    Writes ``<name>.cfg``, ``<name>.poly`` and ``<name>.trn`` into ``workdir``
    first - what E4D itself would be run on - then tetrahedralizes with TetGen
    when one is found (``mesher`` "auto" or "tetgen"), which also leaves E4D's
    ``<name>.1.*`` mesh files there, or else with Gmsh ("auto" or "gmsh").

    Returns ``{"mesh", "mesher", "files", "plc", "zones", "translation"}``;
    the mesh is in survey coordinates, cell markers the zone numbers.
    """
    mesher = str(mesher or "auto").lower()
    folder = Path(workdir)
    folder.mkdir(parents=True, exist_ok=True)
    plc = build_e4d_plc(config)
    files = {"cfg": str(write_e4d_config(config, folder / f"{name}.cfg")),
             "poly": str(write_tetgen_poly(plc, folder / f"{name}.poly"))}
    trn = folder / f"{name}.trn"
    trn.write_text("  ".join(f"{v:.12E}" for v in plc.translation) + "\n", encoding="ascii")
    files["trn"] = str(trn)
    log(f"  E4D geometry: {len(config.points)} control points, {len(config.boundaries)} "
        f"internal boundaries, {len(config.zones)} zones; PLC of {len(plc.nodes)} nodes "
        f"and {len(plc.facets)} facets")

    # TetGen as E4D runs it - the program on the .poly - then the same library
    # through its Python package, then Gmsh.
    wants_tetgen = mesher in ("auto", "tetgen")
    tetgen_path = find_tetgen(tetgen or config.tetgen) if wants_tetgen else None
    tetgen_package = _tetgen_module() if wants_tetgen and not tetgen_path else None
    if mesher == "tetgen" and tetgen_path is None and tetgen_package is None:
        raise RuntimeError("TetGen was not found. Install it with `pip install tetgen`, or put "
                           "the tetgen program on PATH (or set PYHYDRO_TETGEN). The .cfg and "
                           f".poly are in {folder}.")
    if tetgen_package is not None:
        import pygimli as pg

        # TetGen 1.6, which the package wraps, can leave a few elements over a
        # zone's volume limit where the older TetGen E4D ships does not. The
        # limit is checked, and tightened for this call by what it missed by;
        # the .cfg and .poly keep the limits as given.
        tighten: Dict[int, float] = {}
        best = None
        for attempt in range(3):
            mesh = _mesh_with_tetgen_module(plc, config, log, tighten)
            over = {entry["zone"]: entry["max_volume"] / entry["largest_volume"]
                    for entry in zone_report(mesh, config)
                    if entry["max_volume"] < UNCONSTRAINED
                    and entry["largest_volume"] > entry["max_volume"]}
            worst = min(over.values()) if over else np.inf
            if best is None or worst > best[0]:
                best = (worst, mesh)
            if not over or attempt == 2:
                break
            for zone, fit in over.items():
                tighten[zone] = tighten.get(zone, 1.0) * 0.95 * fit
            log("  a few elements over their zone's volume limit; meshing again with "
                + ", ".join(f"zone {z} at {f:.2f} of its limit" for z, f in tighten.items()))
        worst, mesh = best
        if worst < 1.0:
            log("  Note: TetGen left elements over a zone's volume limit (see below).")
        mesh.translate(pg.Pos(*plc.translation))
        label = "TetGen (Python package) with E4D's switches"
    elif tetgen_path:
        _run([tetgen_path, _tetgen_switches(config.quality, config.max_volume), f"{name}.poly"],
             folder, log, "TetGen")
        mesh = read_e4d_mesh(folder / f"{name}.1", translation=plc.translation,
                             conductivity=False)
        for suffix in ("node", "ele", "face", "neigh", "edge"):
            produced = folder / f"{name}.1.{suffix}"
            if produced.is_file():
                files[suffix] = str(produced)
        label = "TetGen (as E4D runs it)"
    else:
        gmsh_path = _find_gmsh(gmsh) if mesher in ("auto", "gmsh") else None
        if gmsh_path is None:
            raise RuntimeError("Neither TetGen nor Gmsh was found to build the mesh. The "
                               f"E4D .cfg and .poly are in {folder}; run E4D or TetGen on "
                               "them, or install TetGen (`pip install tetgen`) or ResIPy, "
                               "which bundles Gmsh.")
        import pygimli as pg

        geo, msh = folder / f"{name}.geo", folder / f"{name}.msh"
        # TetGen enforces a zone's volume limit; Gmsh only aims at a size. So
        # the limit is checked, the size tightened by what it missed by, and the
        # attempt that came closest kept - a finer size does not always help.
        scale, best = 0.9, None
        for attempt in range(3):
            _write_geo(plc, geo, fine_scale=scale)
            _run([gmsh_path, geo.name, "-3", "-format", "msh22", "-o", msh.name, "-v", "2"],
                 folder, log, "Gmsh")
            mesh = pg.meshtools.readGmsh(str(msh), verbose=False)
            _assign_zones(mesh, plc, log)
            fits = [entry["max_volume"] / entry["largest_volume"]
                    for entry in zone_report(mesh, config)
                    if entry["max_volume"] < UNCONSTRAINED and entry["largest_volume"] > 0]
            worst = min(fits) if fits else np.inf
            if best is None or worst > best[0]:
                best = (worst, mesh)
            if worst >= 1.0 or attempt == 2:
                break
            scale *= 0.97 * worst ** (1.0 / 3.0)
            log(f"  largest element over its zone's limit; meshing again at "
                f"{scale:.2f} of the edge that limit implies")
        worst, mesh = best
        if worst < 1.0:
            log("  Note: Gmsh could not keep every zone under its volume limit (see "
                "below); TetGen enforces it, and gives the mesh E4D itself would build.")
        mesh.translate(pg.Pos(*plc.translation))
        # The .msh on disk is the last attempt, not necessarily the one kept.
        msh.unlink(missing_ok=True)
        files["geo"] = str(geo)
        label = "Gmsh on the E4D geometry (TetGen not found)"
    zones = zone_report(mesh, config)
    for entry in zones:
        limit = ("unconstrained" if entry["max_volume"] >= UNCONSTRAINED
                 else f"limit {entry['max_volume']:g}")
        log(f"  zone {entry['zone']}: {entry['cells']} cells, largest "
            f"{entry['largest_volume']:.4g} m^3 ({limit})")
    _unmark_inner_faces(mesh)
    return {"mesh": mesh, "mesher": label, "files": files, "plc": plc, "zones": zones,
            "translation": plc.translation}
