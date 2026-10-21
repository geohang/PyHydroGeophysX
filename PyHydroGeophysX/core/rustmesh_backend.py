"""Opt-in 2D meshing with rustmesh in place of Triangle.

pyGIMLi meshes every 2D geometry in ``pygimli.meshtools.createMesh``, whose 2D
branch hands the PLC to Triangle (``pg.core.TriangleWrapper``). Everything
that builds a 2D mesh goes through it: ``ERTManager.createMesh``,
``TravelTimeManager.createMesh``, ``createParaMesh``,
``appendTriangleBoundary`` and PyHydroGeophysX's own calls. :func:`enable`
replaces that branch with rustmesh (MIT OR Apache-2.0), so no call site needs
an edit, and makes any remaining use of ``TriangleWrapper`` (or of libgimli's
C++ ``addTriangleBoundary``, which uses it) raise instead of meshing with
Triangle (or, in a libgimli built without Triangle, returning an empty mesh).

Turn it on with the environment variable ``PYHYDROGEOPHYSX_MESHER=rustmesh``
(read when PyHydroGeophysX is imported; pyGIMLi is patched once it loads) or
in code with :func:`enable` / :func:`rustmesh_meshing`. The default is
unchanged: Triangle.

How pyGIMLi's arguments are honoured:

- ``quality`` (Triangle's smallest angle) is rustmesh's angle target as it is
  (:func:`effective_angle`). pyGIMLi's callers ask for 31-34 degrees; rustmesh
  (since its 2026-10-21 refinement above 30 degrees) meets 34 on the ERT and
  SRT parameter meshes with about as many cells as Triangle. On a geometry
  where refinement does not settle, rustmesh stops at its vertex budget and
  this backend raises rather than lowering the angle quietly.
- ``area`` and the PLC's regional areas are cell-area limits, as in Triangle.
  When the PLC already has cells, their centres and markers seed the regions
  and its region markers are ignored, as ``TriangleWrapper`` does.
- ``smooth`` is libgimli's own ``mesh.smooth``, applied exactly as pyGIMLi
  applies it after Triangle.
- ``preserveBoundary`` (Triangle's ``-Y``) keeps every segment with the outside
  or a hole on one side unsplit; ``-YY`` keeps every segment unsplit.
- Node and edge markers follow Triangle's rules: an unmarked node on a marked
  segment takes the segment's marker (the first such segment in input order),
  and unmarked boundary segments and their unmarked nodes get marker 1.
- Explicit ``switches`` are read for the Triangle options pyGIMLi uses
  (``p z e A Y Q V q<angle> a<area>``); any other option raises. Without ``p``
  (a point list) the mesh is the points' Delaunay triangulation (SciPy);
  refining one (``q`` or ``a`` without ``p``) raises.

rustmesh reports cells that miss the angle or area goal instead of hiding
them; :func:`last_report` returns that report for the most recent mesh,
measured again on the final mesh when it was smoothed. A mesh whose refinement
ran out of its vertex budget, or that came out empty, raises.
"""

from __future__ import annotations

import importlib.abc
import logging
import math
import os
import re
import sys
import warnings
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

ENV_VAR = "PYHYDROGEOPHYSX_MESHER"

_log = logging.getLogger(__name__)
_state: Dict[str, Any] = {"enabled": False, "create_mesh": None, "add_triangle_boundary": None,
                          "triangle_methods": None, "report": None, "calls": 0}


def effective_angle(quality: float) -> float:
    """rustmesh's smallest-angle target for pyGIMLi's ``quality``: the same
    angle, 0 to 40 degrees (rustmesh's range; Triangle warns above 34)."""
    quality = float(quality)
    if not math.isfinite(quality) or not 0.0 <= quality <= 40.0:
        raise ValueError(f"quality must be an angle from 0 to 40 degrees, not {quality}")
    return quality


def is_enabled() -> bool:
    """Whether pyGIMLi's 2D meshing currently goes to rustmesh."""
    return _state["enabled"]


def last_report() -> Optional[Dict[str, Any]]:
    """rustmesh's report on the most recent 2D mesh made while enabled.

    It has rustmesh's own fields (``min_angle``, ``angle_violations``,
    ``area_violations``, ``issues``, ...) plus ``requested_quality``,
    ``min_angle_target``, ``requested_angle_violations`` (cells below the
    requested ``quality``), ``cells`` and ``call`` (a running count). After
    ``smooth``, the angle and area fields describe the smoothed mesh and
    ``before_smoothing`` holds rustmesh's own values.
    """
    return None if _state["report"] is None else dict(_state["report"])


def _triangle_off(what: str) -> RuntimeError:
    return RuntimeError(
        "Triangle is switched off: PyHydroGeophysX meshes 2D geometries with rustmesh "
        f"({ENV_VAR}=rustmesh). Mesh through pygimli.meshtools.createMesh, not {what}.")


def _triangle_wrapper_disabled(*args: Any, **kwargs: Any) -> None:
    raise _triangle_off("pg.core.TriangleWrapper")


def _add_triangle_boundary_disabled(*args: Any, **kwargs: Any) -> None:
    """Stands in for libgimli's ``addTriangleBoundary``, which calls Triangle
    from C++ (pyGIMLi's Python ``appendTriangleBoundary`` does not use it)."""
    raise _triangle_off("pg.core.addTriangleBoundary (use pygimli.meshtools.appendTriangleBoundary)")


def _parse_switches(switches: str) -> Dict[str, Any]:
    """The Triangle options pyGIMLi uses, read from a switch string.

    Numbers after ``q`` and ``a`` are read as Triangle reads them (digits and
    dots only), so ``-zeY`` is z, e, Y.
    """
    text = str(switches).lstrip("-")
    opts = dict(plc=False, regions=False, quality=None, area=0.0, regional_area=False, keep=None)
    keep = 0
    i = 0
    while i < len(text):
        c = text[i]
        i += 1
        if c in "qa":
            number = re.match(r"[0-9.]*", text[i:]).group(0)
            i += len(number)
            if c == "q":
                opts["quality"] = float(number) if number else 20.0
            elif number:
                opts["area"] = float(number)
            else:
                opts["regional_area"] = True
        elif c == "p":
            opts["plc"] = True
        elif c == "A":
            opts["regions"] = True
        elif c == "Y":
            keep += 1
        elif c in "zeQV":
            pass
        else:
            raise NotImplementedError(f"Triangle switch {c!r} (in {switches!r}) has no rustmesh equivalent")
    if not opts["plc"] and (opts["quality"] is not None or opts["area"] > 0 or opts["regional_area"]):
        raise NotImplementedError(f"switches {switches!r} refine a point set without 'p'; "
                                  "the rustmesh backend meshes point sets without refinement only")
    opts["keep"] = {0: None, 1: "boundary"}.get(keep, "all")
    return opts


def _delaunay(poly):
    """Triangle without ``p``: the Delaunay triangulation of the nodes, with
    the convex hull's edges and unmarked nodes marked 1 as Triangle marks them."""
    import numpy as np
    import pygimli as pg
    from scipy.spatial import Delaunay

    points = np.array([[n.pos()[0], n.pos()[1]] for n in poly.nodes()], dtype=float)
    markers = [n.marker() for n in poly.nodes()]
    tri = Delaunay(points)
    hull = {tuple(sorted(e)) for e in tri.convex_hull.tolist()}
    for a, b in hull:
        for k in (a, b):
            if markers[k] == 0:
                markers[k] = 1
    mesh = pg.Mesh(2)
    nodes = [mesh.createNode(pg.Pos(float(x), float(y)), int(m)) for (x, y), m in zip(points, markers)]
    for a, b in sorted(hull):
        mesh.createEdge(nodes[a], nodes[b], 1)
    for a, b, c in tri.simplices:
        p = points[[a, b, c]]
        if (p[1, 0] - p[0, 0]) * (p[2, 1] - p[0, 1]) - (p[2, 0] - p[0, 0]) * (p[1, 1] - p[0, 1]) < 0:
            b, c = c, b
        mesh.createTriangle(nodes[a], nodes[b], nodes[c], 0)
    return mesh


def _triangle_markers(result) -> None:
    """Node and edge markers as Triangle gives them (``-p``), in place."""
    import numpy as np

    tri = result.triangles
    sides = np.sort(np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), axis=1)
    sides, uses = np.unique(sides, axis=0, return_counts=True)
    outline = set(map(tuple, sides[uses == 1].tolist()))
    edges = np.sort(np.asarray(result.edges), axis=1)
    on_outline = np.fromiter((tuple(e) in outline for e in edges.tolist()), bool, len(edges))
    segment = np.asarray(result.edge_segments)
    own = np.asarray(result.edge_markers).copy()
    # Unmarked boundary segments (and pieces of outline) become 1.
    marks = np.where((own == 0) & on_outline, 1, own)
    nodes = np.asarray(result.node_markers).copy()
    unmarked = nodes == 0
    # The first segment in input order with a nonzero marker wins a node.
    order = np.lexsort((np.arange(len(segment)), np.where(segment < 0, np.iinfo(np.int64).max, segment)))
    taken = np.zeros(len(nodes), bool)
    for k in order:
        for node in edges[k]:
            if unmarked[node] and not taken[node] and own[k] != 0:
                nodes[node], taken[node] = own[k], True
    # Then the rest of the nodes on an unmarked boundary segment get its 1.
    for k in order:
        for node in edges[k]:
            if unmarked[node] and not taken[node] and marks[k] != 0:
                nodes[node], taken[node] = marks[k], True
    result.node_markers[:] = nodes
    result.edge_markers[:] = marks


def _quality(nodes, triangles, limits):
    """Smallest angle, every cell's smallest angle, and the count of cells
    above their area limit."""
    import numpy as np

    p = np.asarray(nodes)[np.asarray(triangles)]
    smallest = np.full(len(p), 180.0)
    for i in range(3):
        a = p[:, (i + 1) % 3] - p[:, i]
        b = p[:, (i + 2) % 3] - p[:, i]
        cross = np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])
        smallest = np.minimum(smallest, np.degrees(np.arctan2(cross, np.sum(a * b, axis=1))))
    a, b = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
    area = 0.5 * np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])
    return (float(smallest.min()) if len(p) else 0.0,
            smallest, int(np.count_nonzero(area > limits * (1 + 1e-9))))


def _triangulate(poly, opts: Dict[str, Any]):
    """A pyGIMLi mesh of the 2D PLC ``poly`` by rustmesh, its node order and
    cell order those of rustmesh's result, and the result's cell limits."""
    import numpy as np
    import rustmesh
    from rustmesh.pygimli import plc_arrays, to_pygimli

    arrays = plc_arrays(poly)
    if poly.cellCount() > 0:
        arrays["regions"] = np.array([[c.center()[0], c.center()[1], c.marker(), 0.0]
                                      for c in poly.cells()], dtype=float).reshape(-1, 4)
    if not opts["regional_area"]:
        arrays["regions"][:, 3] = 0.0
    quality = opts["quality"]
    angle = 0.0 if quality is None else effective_angle(quality)
    keep_segments = np.arange(len(arrays["segments"])) if opts["keep"] == "all" else None
    result = rustmesh.triangulate(**arrays, min_angle=angle, max_area=float(opts["area"]),
                                  keep_segments=keep_segments,
                                  keep_boundary=opts["keep"] == "boundary", strict=False)
    report = dict(result.report)
    if not report["refinement_complete"] or len(result.triangles) == 0:
        raise RuntimeError("rustmesh could not mesh the 2D geometry: " + "; ".join(report["issues"]))
    if not opts["regions"]:
        result.cell_markers[:] = 0
    _triangle_markers(result)
    regional = arrays["regions"][:, 3]
    own = np.array([regional[r] if r >= 0 and regional[r] > 0 else np.inf for r in result.cell_regions])
    limits = np.minimum(own, opts["area"] if opts["area"] > 0 else np.inf)
    _state["calls"] += 1
    report.update(requested_quality=quality, min_angle_target=angle,
                  cells=int(len(result.triangles)), call=_state["calls"])
    _state["report"] = report
    mesh = to_pygimli(result)
    # Outline edges with their cell on the left, as Triangle's are: pyGIMLi's
    # outside() needs it (appendTriangleBoundary copies only those edges), and
    # it must hold before smoothing.
    mesh.fixBoundaryDirections()
    return mesh, result, limits


def _record_final(mesh, result, limits, smoothed) -> None:
    """Angle and area figures of the mesh as returned."""
    import numpy as np

    report = _state["report"]
    nodes = np.array(mesh.positions())[:, :2] if smoothed else result.nodes
    smallest_angle, smallest, too_large = _quality(nodes, result.triangles, limits)
    if smoothed:
        report["before_smoothing"] = {k: report[k] for k in
                                      ("min_angle", "angle_violations", "area_violations", "quality_met")}
        target = report["min_angle_target"]
        report["min_angle"] = smallest_angle
        report["angle_violations"] = int(np.count_nonzero(smallest < target - 1e-6)) if target > 0 else 0
        report["area_violations"] = too_large
        report["quality_met"] = report["angle_violations"] == 0 and too_large == 0
        report["smoothed"] = True
    requested = report["requested_quality"]
    report["requested_angle_violations"] = (int(np.count_nonzero(smallest < requested - 1e-6))
                                            if requested else 0)
    if not report["quality_met"] or report["requested_angle_violations"]:
        _log.info("rustmesh 2D mesh: %d cells, smallest angle %.2f, %d below the %.4g degree target, "
                  "%d below the requested %s, %d above their area limit", report["cells"],
                  report["min_angle"], report["angle_violations"], report["min_angle_target"],
                  report["requested_angle_violations"], requested, report["area_violations"])


def _create_mesh(poly, quality=32, area=0.0, smooth=None, switches=None, verbose=False, **kwargs):
    """``pygimli.meshtools.createMesh`` with its 2D branch done by rustmesh."""
    import pygimli as pg

    if not (isinstance(poly, pg.Mesh) and poly.dim() == 2):
        # Lists, point arrays and 3D: pyGIMLi's own code, which comes back
        # here for the 2D mesh of a list or a point set.
        return _state["create_mesh"](poly, quality, area, smooth, switches, verbose, **kwargs)
    pg.checkAndFixLocaleDecimal_point()
    if poly.nodeCount() == 0:
        pg.critical("No nodes in poly to create a valid mesh")
    if switches is None:
        # pyGIMLi's own switches: pzeA, a<area>, q<quality>, a (regional areas), Y
        opts = dict(plc=True, regions=True, quality=float(quality),
                    area=float(area) if area and area > 0 else 0.0, regional_area=True,
                    keep="boundary" if "preserveBoundary" in kwargs else None)
    else:
        opts = _parse_switches(switches)
    if not opts["plc"]:
        mesh = _delaunay(poly)
        mesh.createNeighborInfos()
        mesh.fixBoundaryDirections()
        return mesh
    mesh, result, limits = _triangulate(poly, opts)
    if smooth is not None:
        if smooth is True:
            smooth = [1, 4]
        mesh.smooth(nodeMoving=kwargs.pop("node_move", True), edgeSwapping=False,
                    smoothFunction=smooth[0], smoothIteration=smooth[1])
    _record_final(mesh, result, limits, smoothed=smooth is not None)
    mesh.createNeighborInfos()
    return mesh


def _rebind(old: Any, new: Any) -> None:
    """Point every loaded module's name for ``old`` at ``new`` (except this
    module's own names for its stand-ins)."""
    for module in list(sys.modules.values()):
        if module is sys.modules[__name__]:
            continue
        try:
            names = [name for name, value in list(vars(module).items()) if value is old]
        except TypeError:
            continue
        for name in names:
            setattr(module, name, new)


def _drop_pending_hook() -> None:
    sys.meta_path[:] = [f for f in sys.meta_path if not isinstance(f, _EnableAfterPygimli)]


def enable() -> None:
    """Mesh pyGIMLi's 2D geometries with rustmesh from now on."""
    _drop_pending_hook()
    if is_enabled():
        return
    try:
        import rustmesh  # noqa: F401
        import rustmesh.pygimli  # noqa: F401
    except ImportError as exc:
        raise ImportError("the rustmesh backend needs the rustmesh package "
                          "(https://github.com/geohang/rustmesh)") from exc
    import pygimli as pg
    import pygimli.meshtools.mesh as pg_mesh

    if is_enabled() or pg_mesh.createMesh is _create_mesh:
        return
    _state["create_mesh"] = pg_mesh.createMesh
    _rebind(pg_mesh.createMesh, _create_mesh)
    # The class itself, so instances made earlier and saved references stop too.
    wrapper = pg.core.TriangleWrapper
    _state["triangle_methods"] = (wrapper, wrapper.__init__, wrapper.generate)
    wrapper.__init__ = _triangle_wrapper_disabled
    wrapper.generate = _triangle_wrapper_disabled
    if hasattr(pg.core, "addTriangleBoundary"):
        _state["add_triangle_boundary"] = pg.core.addTriangleBoundary
        _rebind(pg.core.addTriangleBoundary, _add_triangle_boundary_disabled)
    _state["enabled"] = True


def disable() -> None:
    """Give pyGIMLi's 2D meshing back to Triangle (and cancel a pending
    :func:`install_from_environment`)."""
    _drop_pending_hook()
    if not is_enabled():
        return
    _rebind(_create_mesh, _state["create_mesh"])
    wrapper, init, generate = _state["triangle_methods"]
    wrapper.__init__, wrapper.generate = init, generate
    if _state["add_triangle_boundary"] is not None:
        _rebind(_add_triangle_boundary_disabled, _state["add_triangle_boundary"])
    _state["enabled"] = False


@contextmanager
def rustmesh_meshing() -> Iterator[None]:
    """Mesh with rustmesh inside the ``with`` block, as before after it."""
    was = is_enabled()
    enable()
    try:
        yield
    finally:
        if not was:
            disable()


class _EnableAfterPygimli(importlib.abc.MetaPathFinder):
    """Calls :func:`enable` once ``import pygimli`` has finished."""

    def find_spec(self, name, path, target=None):
        if name != "pygimli":
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(name, path, target)
            if spec is not None:
                break
        else:
            return None
        loader_exec = spec.loader.exec_module

        def exec_module(module):
            loader_exec(module)
            if self in sys.meta_path:
                enable()

        spec.loader.exec_module = exec_module
        return spec


def install_from_environment() -> bool:
    """Enable the backend when ``PYHYDROGEOPHYSX_MESHER=rustmesh``.

    pyGIMLi is not imported here: when it is not loaded yet, the backend is
    enabled as soon as it is.
    """
    value = os.environ.get(ENV_VAR, "").strip().lower()
    if value in ("", "triangle"):
        return False
    if value != "rustmesh":
        warnings.warn(f"{ENV_VAR}={value!r} is not a known mesher (use 'rustmesh'); "
                      "pyGIMLi keeps meshing with Triangle", RuntimeWarning, stacklevel=2)
        return False
    if "pygimli" in sys.modules:
        enable()
    elif not any(isinstance(f, _EnableAfterPygimli) for f in sys.meta_path):
        sys.meta_path.insert(0, _EnableAfterPygimli())
    return True
