"""The mesh an ERT inversion runs on: built, imported, cut at zones, previewed.

One module for every place that needs the inversion mesh - the single-survey
and the time-lapse inversions, and the ERT page's mesh preview - so what the
page shows before a run is the mesh the run then inverts on, cell for cell:

* :func:`build_inversion_mesh` builds PyGIMLi's parameter mesh of a survey,
  along the outlines of a-priori zones (``ert_zones``) when asked, or imports
  one built elsewhere;
* :func:`load_inversion_mesh` reads such a mesh - PyGIMLi, Gmsh, VTK, E4D -
  and checks that it can hold the survey;
* :func:`mark_zone_interfaces` stops the smoothness at the zone outlines;
* :func:`mesh_preview` splits the mesh the way the inversion does, for the
  page to draw.

These were written in ``ert_inversion``, which still exports them.
"""
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pygimli as pg
from pygimli.physics import ert

from PyHydroGeophysX._internal.utils import noop as _noop_log


#: Mesh formats a user can hand the inversion. ``.bms`` is PyGIMLi's own,
#: ``.msh`` is Gmsh (the usual route for a complex 3D domain), ``.node`` and
#: ``.ele`` an E4D (TetGen) mesh, ``.cfg`` an E4D mesh configuration, which is
#: built the way E4D builds it; the rest are what PyGIMLi's loader recognises.
MESH_SUFFIXES = (".bms", ".msh", ".vtk", ".vtu", ".poly", ".node", ".ele", ".cfg")


def load_inversion_mesh(mesh_path: str | Path, data=None,
                        log: Callable[[str], None] = _noop_log):
    """Load a user-supplied inversion mesh and check it can hold this survey.

    Building a mesh from the electrode line is fine for a 2D profile and
    hopeless for a 3D domain with topography, boreholes or known structure, so
    those are meshed externally (usually in Gmsh or E4D) and brought in here.
    An E4D mesh is read with its ``.trn`` translation undone and its zones as
    cell markers, smoothed across its internal boundaries within a zone as E4D
    does; an E4D ``.cfg`` is meshed first (see ``core.e4d_mesh``).

    An imported mesh fails in ways a generated one cannot: electrodes outside
    the domain, or every cell marked background so nothing is inverted. Both
    surface deep inside the forward solver as errors that name nothing useful,
    so they are checked here where the message can say what is wrong.
    """
    path = Path(mesh_path)
    if not path.is_file():
        raise FileNotFoundError(f"No mesh file at {path}.")
    suffix = path.suffix.lower()
    if suffix == ".msh":
        from pygimli.meshtools import readGmsh
        mesh = readGmsh(str(path), verbose=False)
    elif suffix in (".node", ".ele"):
        from PyHydroGeophysX.core.e4d_mesh import read_e4d_mesh
        mesh = read_e4d_mesh(path)
    elif suffix == ".cfg":
        import tempfile

        from PyHydroGeophysX.core.e4d_mesh import build_e4d_mesh, read_e4d_config
        log(f"  building the mesh {path.name} describes, as E4D would")
        built = build_e4d_mesh(read_e4d_config(path), tempfile.mkdtemp(prefix="e4d_"),
                               name=path.stem, log=log)
        mesh = built["mesh"]
    else:
        mesh = pg.load(str(path))
    if mesh is None or int(mesh.cellCount()) == 0:
        raise ValueError(f"{path.name} loaded no cells; is it a mesh file?")

    markers = np.asarray([c.marker() for c in mesh.cells()], dtype=int)
    invertible = int((markers > 1).sum())
    if invertible == 0:
        counts = {int(m): int((markers == m).sum()) for m in np.unique(markers)}
        raise ValueError(
            f"{path.name} has no cells marked as the parameter domain "
            f"(marker > 1); markers present: {counts}. PyGIMLi inverts marker 2 "
            "and above and treats 0 and 1 as background, so nothing here would "
            "be inverted. Re-tag the region to invert with marker 2.")

    if data is not None:
        sensors = np.atleast_2d(np.asarray(data.sensorPositions(), dtype=float))
        if sensors.size:
            outside = _sensors_outside(mesh, sensors)
            if outside:
                raise ValueError(
                    f"{path.name} does not contain {len(outside)} of "
                    f"{len(sensors)} electrodes (first at "
                    f"{np.round(sensors[outside[0]], 2).tolist()}). A mesh that "
                    "does not cover the array cannot be used for this survey; "
                    "check the coordinate origin and units.")
    log(f"  mesh: {path.name}, {mesh.cellCount()} cells "
        f"({invertible} inverted), {mesh.nodeCount()} nodes, {mesh.dim()}D")
    return mesh


#: Triangle stops terminating a little above a 34-degree smallest angle (36
#: already never finishes on a plain profile), so a higher mesh quality is
#: refused up front instead of hanging the run.
MAX_MESH_QUALITY = 34.0

#: Marker a zone outline carries while it is meshed, so Triangle keeps it as
#: cell edges. It is cleared once the mesh is built: pyGIMLi drops the
#: smoothness constraint across every marked edge inside the parameter domain,
#: and whether to do that is a separate choice (:func:`mark_zone_interfaces`).
_ZONE_OUTLINE_MARKER = 11

#: Marker of an edge the smoothness constraint does not cross: one between cells
#: of two different zones, or of a zone and the ground around it.
ZONE_INTERFACE_MARKER = 12


def build_inversion_mesh(data, *, mesh_quality: float = 34.0, para_depth: float = 0.0,
                         para_max_cell_size: float = 0.0, para_boundary: float = 2.0,
                         surface_nodes: int = 1, outer_width: float = 0.0,
                         outer_max_cell_size: float = 0.0,
                         conform_zones: Optional[Sequence[Dict[str, Any]]] = None,
                         mesh_file: str = "", log: Callable[[str], None] = _noop_log,
                         report: Optional[Dict[str, Any]] = None):
    """The mesh an ERT inversion of ``data`` runs on: imported, or built.

    One function for every place that needs it - the single-survey and the
    time-lapse inversions, and the ERT page's mesh preview - so what the page
    shows before a run is the mesh the run then inverts on, cell for cell.

    A generated mesh is PyGIMLi's parameter mesh: an inverted region under the
    electrodes and a coarse outer region around it that carries the boundary
    condition far away. Every sizing option maps onto one of PyGIMLi's, and a
    value left at its default is not passed, so the mesh is the one PyGIMLi
    builds by itself. They are the controls an E4D mesh configuration sets
    too: the inverted region is its fine zone, the outer region its outer
    boundary.

    Parameters
    ----------
    data : pygimli.DataContainerERT
        The survey; its sensor positions define the generated mesh.
    mesh_quality : float
        Smallest angle, in degrees, a triangle of a generated mesh may have:
        higher gives better-shaped and more cells. At most
        :data:`MAX_MESH_QUALITY`.
    para_depth, para_max_cell_size : float
        Depth of the inverted region and its largest cell area (m²); 0 lets
        PyGIMLi choose from the array.
    para_boundary : float
        How far the inverted region reaches past the first and the last
        electrode, in electrode spacings (``paraBoundary``).
    surface_nodes : int
        Nodes placed on the surface between two neighbouring electrodes
        (``addNodes``); more makes the cells near the surface smaller.
    outer_width : float
        How far the outer region reaches beyond the inverted region, sideways
        and down, in lengths of the electrode spread (``boundary``); 0 keeps
        PyGIMLi's four.
    outer_max_cell_size : float
        Largest cell area in the outer region (m², ``boundaryMaxCellSize``); 0
        leaves it unlimited.
    conform_zones : list of dict, optional
        A-priori zones (see ``ert_zones``) whose outlines a generated mesh
        follows: they become cell edges, so no cell straddles a zone's edge.
    mesh_file : str
        A mesh built elsewhere (.bms, .msh, .vtk, .vtu, .poly, E4D). Its own
        domain applies, and the options above are ignored.
    report : dict, optional
        Filled with what was built: ``source`` ("generated" or "imported") and
        ``zone_outline_edges``, the edges the zone outlines added (0 when the
        mesh does not follow them).

    Returns
    -------
    pygimli.Mesh
        The full forward mesh: parameter domain (marker 2 and above) and the
        outer region that carries the boundary far from the array (marker 1).
    """
    from .ert_zones import normalize_zones

    report = report if report is not None else {}
    report.update(source="imported" if str(mesh_file or "") else "generated",
                  zone_outline_edges=0)
    zones = normalize_zones(conform_zones) if conform_zones else []
    if str(mesh_file or ""):
        # A mesh built elsewhere describes its own domain, so the sizing knobs
        # below have nothing to act on. Saying so beats silently ignoring them.
        mesh = load_inversion_mesh(mesh_file, data=data, log=log)
        ignored = [name for name, changed in (
            ("depth", float(para_depth) > 0),
            ("cell size", float(para_max_cell_size) > 0),
            ("side margin", float(para_boundary) != 2.0),
            ("surface nodes", int(surface_nodes) > 1),
            ("outer region", float(outer_width) > 0 or float(outer_max_cell_size) > 0),
            ("zone outlines", bool(zones))) if changed]
        if ignored:
            log(f"  ({', '.join(ignored)} ignored: the mesh is imported and "
                "describes its own domain)")
        return mesh
    quality = float(mesh_quality)
    if quality > MAX_MESH_QUALITY:
        raise ValueError(
            f"A mesh quality of {quality:g} degrees is more than Triangle can "
            f"reach; it would never finish. Use {MAX_MESH_QUALITY:g} or less.")
    if float(para_boundary) <= 0:
        raise ValueError("The side margin (para_boundary) must be positive: the "
                         "inverted region has to reach past the outer electrodes.")
    # PyGIMLi sizes the parameter domain from the array length when paraDepth is
    # left at 0, which for a long line reaches far below anything the data can
    # resolve. Capping it removes unknowns the inversion cannot constrain anyway.
    mesh_kwargs: Dict[str, Any] = {"quality": quality}
    details: List[str] = []
    if float(para_depth) > 0:
        mesh_kwargs["paraDepth"] = float(para_depth)
        details.append(f"parameter domain capped at {float(para_depth):g} m depth")
    if float(para_max_cell_size) > 0:
        mesh_kwargs["paraMaxCellSize"] = float(para_max_cell_size)
        details.append(f"inverted cells at most {float(para_max_cell_size):g} m²")
    if float(para_boundary) != 2.0:
        mesh_kwargs["paraBoundary"] = float(para_boundary)
        details.append(f"{float(para_boundary):g} electrode spacings past the ends")
    if int(surface_nodes) > 1:
        mesh_kwargs["addNodes"] = int(surface_nodes)
        details.append(f"{int(surface_nodes)} surface nodes between electrodes")
    if float(outer_width) > 0:
        mesh_kwargs["boundary"] = float(outer_width)
        details.append(f"outer region {float(outer_width):g} spreads wide")
    if float(outer_max_cell_size) > 0:
        mesh_kwargs["boundaryMaxCellSize"] = float(outer_max_cell_size)
        details.append(f"outer cells at most {float(outer_max_cell_size):g} m²")
    mesh = None
    if zones:
        mesh, edges = _zone_conforming_mesh(data, mesh_kwargs, zones)
        report["zone_outline_edges"] = int(edges)
        if mesh is None:
            log("  (no zone outline reaches the parameter domain, so the mesh "
                "is built without them)")
        else:
            details.append(f"follows the outlines of {len(zones)} zone(s)")
    if mesh is None:
        mesh = ert.ERTManager(data).createMesh(data=data, **mesh_kwargs)
    para_cells = sum(1 for cell in mesh.cells() if cell.marker() > 1)
    log(f"  mesh: {mesh.cellCount()} cells, {para_cells} of them inverted"
        + (f" ({'; '.join(details)})" if details else ""))
    return mesh


def _electrode_spacing(data) -> float:
    """The typical distance between neighbouring electrodes, 1 m without any."""
    positions = np.asarray(data.sensorPositions(), dtype=float)
    if len(positions) < 2:
        return 1.0
    positions = positions[np.argsort(positions[:, 0], kind="stable")]
    gaps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    gaps = gaps[gaps > 1e-9]
    return float(np.median(gaps)) if gaps.size else 1.0


def _zone_conforming_mesh(data, mesh_kwargs: Dict[str, Any], zones):
    """PyGIMLi's parameter mesh of ``data``, with the zone outlines as cell edges.

    The same steps as ``ERTManager.createMesh`` - the parameter-mesh geometry,
    then Triangle with its smoothing - with the outlines added to the geometry
    in between. Returns ``(mesh, edges)``; the mesh is None when no outline
    reaches the parameter domain.
    """
    import pygimli.meshtools as mt

    plc_kwargs = dict(mesh_kwargs)
    quality = plc_kwargs.pop("quality")
    plc = mt.createParaMeshPLC(data.sensors(), **plc_kwargs)
    plc, edges, para_marker = _embed_zone_outlines(
        plc, zones, spacing=_electrode_spacing(data))
    if not edges:
        return None, 0
    mesh = mt.createMesh(plc, quality=quality, smooth=[2, 10])
    for boundary in mesh.boundaries():
        if boundary.marker() == _ZONE_OUTLINE_MARKER:
            boundary.setMarker(0)
    # Every face the outlines cut off was given a seed, so no cell should be
    # left unassigned; one that is belongs to the inverted region it lies in.
    for cell in mesh.cells():
        if cell.marker() == 0:
            cell.setMarker(para_marker)
    return mesh, edges


def _embed_zone_outlines(plc, zones, *, spacing: float):
    """Add the zone outlines to the parameter-mesh geometry ``plc`` as edges.

    Only what lies inside the parameter domain is added: an outline drawn
    across the ground surface, or past the bottom or the sides of the inverted
    region, ends there, on the domain's boundary. Where it meets the surface it
    ends on the nearest surface node, and on a long boundary edge at a node
    inserted there, unless an end of the edge is within half an electrode
    ``spacing``: a new node beside an existing one would force cells as small
    as the gap between them. A vertex within a fifth of the spacing of a node
    or of the boundary is moved onto it, so a vertex placed a little under the
    surface does not leave a sliver of tiny cells between the two, and a
    stretch of outline that runs along the boundary is left to the boundary.
    Every such move is well under what the inversion resolves. Outlines that
    cross each other are noded by Triangle.

    Each face the outlines cut off gets a seed with the inverted region's
    marker and cell size, so every cell stays in the one region the inversion
    inverts, and the regularization sees one domain, as it does without zones.

    Returns ``(plc, edges, marker)``: the new geometry, how many outline edges
    it gained (0 when no outline reaches the parameter domain, and ``plc`` is
    then returned unchanged), and the inverted region's marker.
    """
    import pygimli.meshtools as mt

    nodes = np.array([[n.pos()[0], n.pos()[1]] for n in plc.nodes()], dtype=float)
    ends = np.array([[b.node(0).id(), b.node(1).id()] for b in plc.boundaries()],
                    dtype=int)
    regions = list(plc.regionMarkers())
    para = [r for r in regions if int(r.marker()) > 1 and not r.isHole()]
    para_marker = int(para[0].marker()) if para else 2
    para_area = float(para[0].area()) if para else 0.0
    # The geometry meshed without refinement: a few dozen cells, enough to
    # tell the inverted region from the outer region and from the air above.
    locator = mt.createMesh(plc, quality=0)

    def in_para(point) -> bool:
        cell = locator.findCell(pg.Pos(float(point[0]), float(point[1])))
        return cell is not None and int(cell.marker()) > 1

    start, direction = nodes[ends[:, 0]], nodes[ends[:, 1]] - nodes[ends[:, 0]]
    length2 = np.maximum((direction ** 2).sum(axis=1), 1e-300)
    lengths = np.sqrt(length2)
    tiny = 1e-9 * max(1.0, float(np.abs(nodes).max()))
    snap = 0.2 * float(spacing)
    # How close to an end of each boundary edge a point is taken to be at that
    # end: the whole of a surface edge, half its length either way, and half an
    # electrode spacing of the long edges that close the domain.
    end_snap = np.maximum(snap, 0.5 * np.minimum(lengths, float(spacing)))

    def nearest_edge(point):
        """Distance to the closest boundary edge, that edge, and where along it."""
        along = np.clip(((point - start) * direction).sum(axis=1) / length2, 0.0, 1.0)
        gaps = np.linalg.norm(start + along[:, None] * direction - point, axis=1)
        k = int(np.argmin(gaps))
        return float(gaps[k]), k, float(along[k])

    # A point of the new geometry is an existing node ("node", i), a point on a
    # boundary edge that splits it ("edge", k, s), or a free vertex ("free", x, z).
    def on_edge(k: int, s: float):
        if s * lengths[k] <= end_snap[k] and s <= 0.5:
            return ("node", int(ends[k, 0]))
        if (1.0 - s) * lengths[k] <= end_snap[k]:
            return ("node", int(ends[k, 1]))
        return ("edge", k, s)

    def snapped(vertex):
        gaps = np.linalg.norm(nodes - vertex, axis=1)
        if gaps.min() <= snap:
            return ("node", int(np.argmin(gaps)))
        gap, k, s = nearest_edge(vertex)
        if gap <= snap:
            return on_edge(k, s)
        return ("free", float(vertex[0]), float(vertex[1]))

    def coords(point) -> np.ndarray:
        if point[0] == "node":
            return nodes[point[1]]
        if point[0] == "edge":
            return start[point[1]] + point[2] * direction[point[1]]
        return np.array(point[1:], dtype=float)

    pieces = []
    for zone in zones:
        ring = [snapped(np.asarray(vertex, dtype=float)) for vertex in zone["polygon"]]
        for first, second in zip(ring, ring[1:] + ring[:1]):
            p, q = coords(first), coords(second)
            span = q - p
            if np.linalg.norm(span) <= tiny:
                continue
            # Where this side of the polygon crosses the geometry's edges.
            denom = span[0] * direction[:, 1] - span[1] * direction[:, 0]
            offset = start - p
            valid = np.abs(denom) > 1e-12 * lengths * np.linalg.norm(span)
            with np.errstate(divide="ignore", invalid="ignore"):
                t = (offset[:, 0] * direction[:, 1] - offset[:, 1] * direction[:, 0]) / denom
                u = (offset[:, 0] * span[1] - offset[:, 1] * span[0]) / denom
            hits = np.flatnonzero(valid & (t > 1e-9) & (t < 1 - 1e-9)
                                  & (u >= -1e-9) & (u <= 1 + 1e-9))
            cuts = sorted([(0.0, first), (1.0, second)]
                          + [(float(t[k]), on_edge(int(k), float(np.clip(u[k], 0.0, 1.0))))
                             for k in hits], key=lambda cut: cut[0])
            for (_, a), (_, b) in zip(cuts, cuts[1:]):
                pa, pb = coords(a), coords(b)
                if np.linalg.norm(pb - pa) <= tiny:
                    continue
                middle = 0.5 * (pa + pb)
                if not in_para(middle):
                    continue  # in the outer region or above the ground
                if a[0] != "free" and b[0] != "free" and nearest_edge(middle)[0] <= snap:
                    continue  # runs along the boundary, which is already an edge
                pieces.append((a, b))
    if not pieces:
        return plc, 0, para_marker

    geometry = pg.Mesh(dim=2, isGeometry=True)
    for (x, z), node in zip(nodes, plc.nodes()):
        geometry.createNode(pg.Pos(float(x), float(z)), int(node.marker()))
    splits: Dict[int, List[Tuple[float, int]]] = {}
    free: Dict[Tuple[int, int], int] = {}

    def node_of(point) -> int:
        if point[0] == "node":
            return int(point[1])
        if point[0] == "edge":
            _, k, s = point
            for known, index in splits.get(k, []):
                if abs(known - s) * lengths[k] <= end_snap[k]:
                    return index  # outlines meeting the boundary together share a node
            index = int(geometry.createNode(pg.Pos(*coords(point))).id())
            splits.setdefault(k, []).append((s, index))
            return index
        key = (int(round(point[1] / tiny)), int(round(point[2] / tiny)))
        if key not in free:
            free[key] = int(geometry.createNode(pg.Pos(point[1], point[2])).id())
        return free[key]

    outline: List[Tuple[int, int]] = []
    for a, b in pieces:
        i, j = node_of(a), node_of(b)
        if i != j and (i, j) not in outline and (j, i) not in outline:
            outline.append((i, j))
    for k, boundary in enumerate(plc.boundaries()):
        chain = ([int(ends[k, 0])] + [index for _, index in sorted(splits.get(k, []))]
                 + [int(ends[k, 1])])
        for i, j in zip(chain, chain[1:]):
            geometry.createEdge(geometry.node(i), geometry.node(j), int(boundary.marker()))
    for i, j in outline:
        geometry.createEdge(geometry.node(i), geometry.node(j), _ZONE_OUTLINE_MARKER)
    for region in regions:
        position = pg.Pos(float(region.x()), float(region.y()))
        if region.isHole():
            geometry.addHoleMarker(position)
        else:
            geometry.addRegionMarker(position, int(region.marker()), float(region.area()))
    # A seed a hair to either side of every outline edge: each face the
    # outlines cut off touches at least one of them.
    for i, j in outline:
        a = np.array([geometry.node(i).pos()[0], geometry.node(i).pos()[1]])
        b = np.array([geometry.node(j).pos()[0], geometry.node(j).pos()[1]])
        tangent = b - a
        normal = np.array([-tangent[1], tangent[0]]) / np.linalg.norm(tangent)
        for side in (1.0, -1.0):
            seed = 0.5 * (a + b) + side * 1e-4 * np.linalg.norm(tangent) * normal
            if in_para(seed):
                geometry.addRegionMarker(pg.Pos(*seed), para_marker, para_area)
    return geometry, len(outline), para_marker


def mark_zone_interfaces(mesh, zones, *, marker: int = ZONE_INTERFACE_MARKER):
    """A copy of ``mesh`` on which the smoothness stops at the zone outlines.

    Every edge between two cells of the parameter domain that belong to
    different zones - or to a zone and the ground around it - is given
    ``marker``. PyGIMLi's region manager drops the smoothness constraint
    across a marked edge, so the in-house engines and PyGIMLi's own manager
    let the model jump there; the ADTLERT engine reads the same marks
    (:func:`_interface_regions`). A cell belongs to the zone its centre lies
    in, as it does everywhere else, so on a mesh that follows the outlines
    the cut runs along them exactly, and on any other mesh along the cell
    edges closest to them.

    Returns ``(mesh, edges)``: the marked copy and how many edges were cut; the
    mesh itself, and 0, for no zones or a 3-D mesh, on which zones do not apply.
    """
    from .ert_zones import normalize_zones, zone_prior

    zones = normalize_zones(zones)
    if not zones or int(mesh.dim()) != 2:
        return mesh, 0
    marked = pg.Mesh(mesh)
    markers = np.asarray(marked.cellMarkers(), dtype=int)
    inverted = markers > 1
    owner = np.full(int(marked.cellCount()), -2, dtype=int)  # -2: not inverted
    centres = np.asarray(marked.cellCenters(), dtype=float)[:, :2]
    owner[inverted] = zone_prior(centres[inverted], zones).owner
    edges = 0
    for boundary in marked.boundaries():
        left, right = boundary.leftCell(), boundary.rightCell()
        if left is None or right is None:
            continue
        a, b = owner[left.id()], owner[right.id()]
        if a == -2 or b == -2 or a == b:
            continue
        boundary.setMarker(int(marker))
        edges += 1
    return marked, edges


def _interface_regions(mesh, cell_ids, *, marker: int = ZONE_INTERFACE_MARKER):
    """The parts the marked zone edges cut the parameter domain into, per cell.

    ``cell_ids`` are the parameter cells of ``mesh`` in model order. Cells that
    connect without crossing a marked edge share a label, which is what ADTLERT's
    structure-guided smoothness takes: it drops the constraint between cells of
    different labels, as pyGIMLi does across the marked edge. None when no edge
    is marked.
    """
    position = {int(cell_id): index for index, cell_id in enumerate(cell_ids)}
    parent = list(range(len(position)))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    cut = False
    for boundary in mesh.boundaries():
        left, right = boundary.leftCell(), boundary.rightCell()
        if left is None or right is None:
            continue
        a, b = position.get(int(left.id())), position.get(int(right.id()))
        if a is None or b is None:
            continue
        if int(boundary.marker()) == int(marker):
            cut = True
            continue
        parent[root(a)] = root(b)
    if not cut:
        return None
    _, labels = np.unique([root(index) for index in range(len(parent))],
                          return_inverse=True)
    return labels.astype(np.int32)


def mesh_preview(mesh, data=None) -> Dict[str, Any]:
    """The inversion mesh as the ERT page's mesh tab draws it.

    The mesh is split the way the inversion splits it - through the forward
    operator's region manager - into the parameter domain, the cells that are
    inverted, in the order of the model vector, and the outer region, which is
    never inverted and only carries the boundary condition far from the array:
    pyGIMLi's background region, the "infinite" part of the mesh.

    Parameters
    ----------
    mesh : pygimli.Mesh
        From :func:`build_inversion_mesh`.
    data : pygimli.DataContainerERT, optional
        The survey, for the electrode positions.

    Returns
    -------
    dict
        ``dim`` and the cell counts always. For a 2-D mesh also the cell
        polygons of both parts as ``(cells, corners, 2)`` arrays,
        ``para_centers`` (what :func:`~.ert_zones.zone_prior` takes), both
        extents as ``(xmin, xmax, zmin, zmax)``, and the electrode positions.
    """
    from .ert_zones import cell_centers_2d

    dim = int(mesh.dim())
    sensors = np.zeros((0, 2))
    if data is not None and int(data.sensorCount()):
        sensors = np.asarray(data.sensorPositions(), dtype=float)[:, :2]
    preview: Dict[str, Any] = {
        "dim": dim, "cells": int(mesh.cellCount()), "nodes": int(mesh.nodeCount()),
        "sensors": sensors,
    }
    if dim != 2:
        # Nothing is drawn for a volume, and setting one up for the forward
        # operator refines it, which can take minutes: count by marker instead,
        # by the rule load_inversion_mesh states.
        markers = np.asarray(mesh.cellMarkers(), dtype=int)
        preview.update(para_cells=int((markers > 1).sum()),
                       outer_cells=int((markers <= 1).sum()))
        return preview
    fop = ert.ERTModelling()
    if data is not None:
        fop.setData(data)
    fop.setMesh(mesh)
    para = fop.paraDomain
    manager = fop.regionManager()
    background = [int(m) for m in manager.regionIdxs() if manager.region(m).isBackground()]
    outer = np.isin(np.asarray(mesh.cellMarkers(), dtype=int), background)
    preview.update(para_cells=int(para.cellCount()), outer_cells=int(outer.sum()))

    def polygons(source, keep=None):
        """Corner coordinates of every cell, one array per corner count."""
        positions = np.asarray(source.positions(), dtype=float)[:, :2]
        cells = [cell.ids() for cell, wanted in zip(
            source.cells(), keep if keep is not None else np.ones(source.cellCount(), bool))
            if wanted]
        if cells and len({len(ids) for ids in cells}) == 1:
            return positions[np.asarray(cells, dtype=int)]
        return [positions[np.asarray(ids, dtype=int)] for ids in cells]

    def extent(source):
        xy = np.asarray(source.positions(), dtype=float)[:, :2]
        return (float(xy[:, 0].min()), float(xy[:, 0].max()),
                float(xy[:, 1].min()), float(xy[:, 1].max()))

    preview.update(
        para_polygons=polygons(para), outer_polygons=polygons(mesh, outer),
        para_centers=cell_centers_2d(para),
        para_extent=extent(para), full_extent=extent(mesh))
    return preview


def _sensors_outside(mesh, sensors: np.ndarray) -> List[int]:
    """Indices of electrodes no cell of ``mesh`` contains."""
    missing: List[int] = []
    for index, position in enumerate(sensors):
        coords = list(position[:3]) + [0.0] * (3 - len(position[:3]))
        try:
            cell = mesh.findCell(pg.Pos(*coords[:3]))
        except Exception:  # noqa: BLE001 - fall back to the bounding box
            cell = None
            lower, upper = mesh.boundingBox().min(), mesh.boundingBox().max()
            inside = all(lower[k] - 1e-6 <= coords[k] <= upper[k] + 1e-6
                         for k in range(mesh.dim()))
            if inside:
                continue
        if cell is None:
            missing.append(index)
    return missing
