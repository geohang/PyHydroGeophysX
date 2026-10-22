"""Quality 2D and 3D mesh generation with a Rust core.

:func:`triangulate` meshes a planar straight-line graph - vertices, segments
that must become mesh edges, region seeds and hole seeds. :func:`tetrahedralize`
is the experimental 3D counterpart. :mod:`rustmesh.pygimli` converts pyGIMLi
geometries and meshes, and can replace Triangle as pyGIMLi's 2D mesher.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from ._rustmesh import __version__
from ._rustmesh import triangulate as _triangulate
from ._rustmesh import Mesh2DQualityError as _NativeMesh2DQualityError

__all__ = ["Mesh2D", "Mesh2DQualityError", "triangulate", "triangulate_strict", "__version__"]


@dataclass
class Mesh2D:
    """A triangle mesh with its markers and how it maps back to the input.

    ``triangles`` are counter-clockwise node triples. ``cell_regions`` holds
    the index of the region seed that reached each cell (-1 for none), kept
    apart from ``cell_markers`` because several regions may share a marker.
    ``edges`` lists every piece of an input segment plus the domain outline
    where no segment runs; ``edge_segments`` gives the input segment (-1 for
    none) and ``edge_markers`` its marker. ``input_vertex_nodes`` maps input
    vertices to nodes (-1 when outside the domain). ``report`` says whether
    every requested area and angle limit was met (``quality_met``), whether
    refinement stopped without exhausting its budget (``refinement_complete``),
    and their combined status (``complete``). ``issues`` explains failures.
    """

    nodes: np.ndarray
    node_markers: np.ndarray
    input_vertex_nodes: np.ndarray
    triangles: np.ndarray
    cell_markers: np.ndarray
    cell_regions: np.ndarray
    edges: np.ndarray
    edge_markers: np.ndarray
    edge_segments: np.ndarray
    report: Dict[str, Any] = field(default_factory=dict)

    @property
    def cell_areas(self) -> np.ndarray:
        p = self.nodes[self.triangles]
        return 0.5 * ((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1])
                      - (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 1] - p[:, 0, 1]))


class Mesh2DQualityError(RuntimeError):
    """Requested 2D goals were unmet; ``mesh`` and ``report`` retain the result.

    Invalid input, cancellation, backend failure and an empty or nonpositive-area
    result never use this exception. The report covers the requested limits
    only; it is not an overlap audit of the mesh.
    """

    def __init__(self, mesh: Mesh2D):
        self.mesh = mesh
        self.report = mesh.report
        super().__init__("requested 2D mesh goals were not met: " +
                         "; ".join(mesh.report["issues"]))


def _array(values: Any, dtype, cols: Optional[int] = None) -> np.ndarray:
    if values is None:
        shape = (0, cols) if cols else (0,)
        return np.zeros(shape, dtype=dtype)
    a = np.ascontiguousarray(values, dtype=dtype)
    return a.reshape(-1, cols) if cols else a.reshape(-1)


def _strict_integer_input(values: Any, dtype, name: str) -> None:
    """Do not silently truncate malformed integer inputs in the strict API."""
    if values is None:
        return
    raw = np.asarray(values)
    if not raw.size:
        return
    limits = np.iinfo(dtype)
    if (not np.issubdtype(raw.dtype, np.integer)
            or np.any(raw < limits.min) or np.any(raw > limits.max)):
        raise ValueError(f"{name} must contain integers within {np.dtype(dtype)} range")


def triangulate(vertices, segments=None, *, vertex_markers=None, segment_markers=None,
                regions=None, holes=None, min_angle: float = 28.0, max_area: float = 0.0,
                refine_points=None, point_size: float = 0.0, size_growth: float = 0.5,
                max_vertices: int = 5_000_000, strict: bool = False,
                keep_segments=None, keep_boundary: bool = False) -> Mesh2D:
    """Mesh a planar straight-line graph with quality triangles.

    Parameters
    ----------
    vertices : (n, 2) array
        Input vertices; all become mesh nodes (unless outside the domain).
    segments : (k, 2) int array, optional
        Vertex index pairs that must appear as mesh edges. A closed loop of
        segments bounds the domain; anything outside it is dropped.
    vertex_markers, segment_markers : int arrays, optional
        Carried to ``node_markers`` and ``edge_markers``.
    regions : (r, 3) or (r, 4) array, optional
        ``x, y, marker[, max_area]`` per region seed. Every cell reachable
        from the seed without crossing a segment gets the marker; a later seed
        wins. ``max_area`` 0 means no limit.
    holes : (h, 2) array, optional
        Seeds of areas to leave empty.
    min_angle : float
        Smallest cell angle in degrees (0 turns angle refinement off). Up to
        about 30 is a practical target range, not a guarantee. Inspect the
        report: constrained geometries can retain violations even below 30.
    max_area : float
        Largest cell area everywhere (0: none); a smaller regional limit wins.
    refine_points : (p, 2) array, optional
        Points to refine around, usually the electrodes.
    point_size : float
        Target cell edge length at the refine points (0: off). The target
        grows by ``size_growth`` metres per metre of distance from the nearest
        refine point; a cell is split while its area exceeds that of an
        equilateral triangle with the target edge at its centroid.
    size_growth : float
        See ``point_size``.
    max_vertices : int
        Vertex budget; when it runs out the mesh is returned with
        ``report["complete"]`` false and the reason in ``report["issues"]``.
    strict : bool
        With True, unmet refinement/quality goals raise ``Mesh2DQualityError``
        carrying the available ``mesh`` and its original ``report``. Invalid
        input and backend errors remain ordinary errors without a partial mesh.
        With False the mesh is returned anyway; inspect its report.
        ``triangulate_strict`` is shorthand for ``strict=True``.
    keep_segments : int array, optional
        Indices of segments refinement must not split (no node is added on
        them, except where they cross other segments).
    keep_boundary : bool
        Also keep whole every segment piece with the domain on one side only
        (outside or a hole on the other). This is the behaviour of Triangle's
        ``-Y`` switch, which pyGIMLi's ``preserveBoundary`` uses. Cells next to
        kept segments may then miss ``min_angle`` or ``max_area``; the report
        counts them.
    """
    if strict:
        _strict_integer_input(segments, np.int64, "segments")
        _strict_integer_input(vertex_markers, np.int32, "vertex_markers")
        _strict_integer_input(segment_markers, np.int32, "segment_markers")
        if (isinstance(max_vertices, (bool, np.bool_))
                or not isinstance(max_vertices, (int, np.integer)) or max_vertices < 1):
            raise ValueError("max_vertices must be a positive integer")
    v = _array(vertices, np.float64, 2)
    s = _array(segments, np.int64, 2)
    vm = _array(vertex_markers, np.int32)
    sm = _array(segment_markers, np.int32)
    r = _array(regions, np.float64)
    if r.size:
        r = r.reshape(len(regions), -1)
        if r.shape[1] == 3:
            r = np.c_[r, np.zeros(len(r))]
        if r.shape[1] != 4:
            raise ValueError("regions must have 3 or 4 columns: x, y, marker[, max_area]")
        r = np.ascontiguousarray(r)
    else:
        r = np.zeros((0, 4))
    h = _array(holes, np.float64, 2)
    p = _array(refine_points, np.float64, 2)
    k = _array(keep_segments, np.int64)
    try:
        out = _triangulate(v, s, vm, sm, r, h, float(min_angle), float(max_area),
                           int(max_vertices), p, float(point_size), float(size_growth), strict,
                           k, bool(keep_boundary))
    except _NativeMesh2DQualityError as error:
        raise Mesh2DQualityError(Mesh2D(**error.mesh)) from error
    return Mesh2D(**out)


def triangulate_strict(vertices, segments=None, **options) -> Mesh2D:
    """Call :func:`triangulate` with ``strict=True``.

    Accepts the same mesh options (except ``strict``). An incomplete result raises
    :class:`Mesh2DQualityError`, which carries its ``mesh`` and ``report``.
    """
    return triangulate(vertices, segments, strict=True, **options)

from .mesh3d import Mesh3D, MeshQualityError, tetrahedralize
__all__ += ["Mesh3D", "MeshQualityError", "tetrahedralize"]
