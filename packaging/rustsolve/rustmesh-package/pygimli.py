"""Conversion to and from pyGIMLi.

:func:`create_mesh` adapts ``pygimli.meshtools.createMesh(plc,
quality, area)`` for a 2D PLC, with rustmesh in place of Triangle. pyGIMLi is
imported lazily, so ``rustmesh`` itself does not need it.
"""

from __future__ import annotations

from typing import Any, Dict
import math
import warnings

import numpy as np

from . import Mesh2D, Mesh3D, Mesh2DQualityError, triangulate


def plc_arrays(plc) -> Dict[str, np.ndarray]:
    """The arrays :func:`rustmesh.triangulate` takes, read from a pyGIMLi PLC."""
    nodes = plc.nodes()
    vertices = np.array([[n.pos()[0], n.pos()[1]] for n in nodes], dtype=float)
    vertex_markers = np.array([n.marker() for n in nodes], dtype=np.int32)
    bounds = plc.boundaries()
    segments = np.array([[b.node(0).id(), b.node(1).id()] for b in bounds],
                        dtype=np.int64).reshape(-1, 2)
    segment_markers = np.array([b.marker() for b in bounds], dtype=np.int32)
    regions, holes = [], []
    for r in plc.regionMarkers():
        if r.isHole():
            holes.append([r.x(), r.y()])
        else:
            regions.append([r.x(), r.y(), r.marker(), max(float(r.area()), 0.0)])
    holes.extend([h[0], h[1]] for h in plc.holeMarker())
    return dict(vertices=vertices, segments=segments, vertex_markers=vertex_markers,
                segment_markers=segment_markers,
                regions=np.array(regions, dtype=float).reshape(-1, 4),
                holes=np.array(holes, dtype=float).reshape(-1, 2))


def to_pygimli(mesh: Mesh2D | Mesh3D):
    """A ``pygimli.Mesh`` with the nodes, cells and markers of ``mesh``.

    Segment edges keep their markers; every other edge gets marker 0.
    """
    import pygimli as pg

    if isinstance(mesh, Mesh3D):
        out = pg.Mesh(3)
        for xyz, marker in zip(mesh.nodes, mesh.node_markers):
            out.createNode(pg.Pos(*map(float, xyz)), int(marker))
        for tet, marker in zip(mesh.tetrahedra, mesh.cell_markers):
            out.createTetrahedron(*(out.node(int(v)) for v in tet), int(marker))
        out.createNeighborInfos()
        boundaries = {tuple(sorted(n.id() for n in b.nodes())): b for b in out.boundaries()}
        for face, marker in zip(mesh.faces, mesh.face_markers):
            boundary = boundaries.get(tuple(sorted(map(int, face))))
            if boundary is None:
                raise RuntimeError("pyGIMLi lost a recovered 3D mesh face")
            boundary.setMarker(int(marker))
        return out

    out = pg.Mesh(2)
    # Keep the returned node handles: looking each one up through Boost.Python
    # again for every edge/cell costs several extra bridge calls per element.
    nodes = [out.createNode(pg.Pos(float(x), float(y)), int(marker))
             for (x, y), marker in zip(mesh.nodes, mesh.node_markers)]
    for (a, b), marker in zip(mesh.edges, mesh.edge_markers):
        out.createEdge(nodes[a], nodes[b], int(marker))
    for (a, b, c), marker in zip(mesh.triangles, mesh.cell_markers):
        out.createTriangle(nodes[a], nodes[b], nodes[c], int(marker))
    out.createNeighborInfos()
    # An edge keeps the triangulation's vertex order, so on the outline its
    # cell may come out on the right; pyGIMLi then does not count the edge as
    # outside() (appendTriangleBoundary, boundary conditions). Triangle's
    # edges have their cell on the left.
    out.fixBoundaryDirections()
    return out


#: pyGIMLi's node marker for sensors (``pg.core.MARKER_NODE_SENSOR``).
SENSOR_MARKER = -99


def map_quality(quality: float, policy: str = "exact") -> Dict[str, Any]:
    """Resolve a 2D angle request; cap30 is an opt-in compatibility policy.

    This is a target mapping, not a mesh-quality or termination guarantee.
    The original and effective targets remain distinct in the adapter report.
    """
    quality = float(quality)
    if not math.isfinite(quality) or not 0.0 <= quality <= 40.0:
        raise ValueError("quality must be finite and between 0 and 40 degrees")
    if policy not in ("exact", "cap30"):
        raise ValueError("angle_policy must be 'exact' or 'cap30'")
    effective = min(quality, 30.0) if policy == "cap30" else quality
    return dict(policy=policy, version=1, requested_min_angle=quality,
                effective_min_angle=effective, mapped=effective != quality)


def _annotate_quality(mesh: Mesh2D, mapping: Dict[str, Any]) -> None:
    requested = mapping["requested_min_angle"]
    count = mesh.report["angle_violations"]
    if mapping["mapped"]:
        p = mesh.nodes[mesh.triangles]
        smallest = np.full(len(p), 180.0)
        for i in range(3):
            a = p[:, (i + 1) % 3] - p[:, i]
            b = p[:, (i + 2) % 3] - p[:, i]
            cross = np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])
            angle = np.degrees(np.arctan2(cross, np.sum(a * b, axis=1)))
            smallest = np.minimum(smallest, angle)
        count = int(np.count_nonzero(smallest < requested - 1e-6))
    mesh.report["angle_mapping"] = dict(mapping,
        requested_angle_violations=count, requested_angle_met=count == 0,
        requested_complete=bool(mesh.report["complete"] and count == 0))


def create_mesh(plc, quality: float = 28.0, area: float = 0.0, *,
                electrode_size: float = 0.0, size_growth: float = 0.5,
                max_vertices: int = 5_000_000, strict: bool = True,
                angle_policy: str = "exact", return_report: bool = False,
                preserveBoundary: bool = False, **unused: Any):
    """Mesh a 2D pyGIMLi PLC like ``pygimli.meshtools.createMesh`` does.

    ``quality`` is the smallest angle in degrees, ``area`` the global largest
    cell area (0: none); regional areas come from the PLC's region markers.
    ``electrode_size`` (0: off) refines around the PLC's sensor nodes (marker
    -99): the target cell edge is ``electrode_size`` there and grows by
    ``size_growth`` metres per metre away from the nearest one.
    ``angle_policy="exact"`` preserves the requested angle. Opt-in ``"cap30"``
    maps requests above 30 to 30 degrees, with a warning; smaller requests
    (including zero) stay unchanged. This does not repair constraint-angle gaps.
    With ``strict``, unmet effective goals raise ``rustmesh.Mesh2DQualityError`` with the
    recovered ``mesh`` and its ``report`` available for inspection. Invalid
    input and backend failures do not carry a partial mesh. Smoothing
    (``smooth=``) is not done; other keywords are ignored.
    ``preserveBoundary=True`` adds no node on the domain's boundary (segments
    with the outside or a hole on one side). That is the behaviour of
    Triangle's ``-Y`` switch, which pyGIMLi's ``createMesh(...,
    preserveBoundary=...)`` uses. Cells along the boundary may then miss the
    angle or area goal, which ``strict`` treats like any unmet goal.
    ``return_report=True`` returns ``(pygimli_mesh, report)``. Native quality and
    completion fields describe the effective target; ``angle_mapping`` also
    records measured violations and completion for the original request. The
    same metadata is added to strict exceptions' original report object.
    """
    if plc.dim() != 2:
        raise ValueError("rustmesh meshes 2D geometries only")
    mapping = map_quality(quality, angle_policy)
    if mapping["mapped"]:
        warnings.warn(f"rustmesh cap30 maps the requested {mapping['requested_min_angle']:g} degree angle "
                      "to 30 degrees; original-request compliance is recorded separately",
                      RuntimeWarning, stacklevel=2)
    arrays = plc_arrays(plc)
    sensors = arrays["vertices"][arrays["vertex_markers"] == SENSOR_MARKER]
    try:
        result = triangulate(**arrays, min_angle=mapping["effective_min_angle"], max_area=area,
                             refine_points=sensors, point_size=electrode_size,
                             size_growth=size_growth, max_vertices=max_vertices, strict=strict,
                             keep_boundary=bool(preserveBoundary))
    except Mesh2DQualityError as error:
        _annotate_quality(error.mesh, mapping)
        raise
    _annotate_quality(result, mapping)
    if not result.report["complete"]:
        warnings.warn("rustmesh returned an incomplete 2D mesh; use return_report=True "
                      "to inspect unmet effective goals", RuntimeWarning, stacklevel=2)
    converted = to_pygimli(result)
    return (converted, result.report) if return_report else converted
