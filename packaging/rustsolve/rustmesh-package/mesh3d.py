"""Experimental conforming tetrahedral meshes; see docs/mesh3d.md."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import numpy as np
from ._rustmesh import tetrahedralize as _tetrahedralize


def _matrix(values, cols, dtype, name):
    if values is None:
        return np.empty((0, cols), dtype=dtype)
    raw = np.asarray(values)
    if raw.size == 0:
        return np.empty((0, cols), dtype=dtype)
    if raw.ndim != 2 or raw.shape[1] != cols:
        raise ValueError(f"{name} must have shape (n, {cols})")
    if np.issubdtype(np.dtype(dtype), np.integer):
        if not np.issubdtype(raw.dtype, np.integer):
            raise ValueError(f"{name} must contain integer indices")
        info = np.iinfo(dtype)
        if np.any(raw < info.min) or np.any(raw > info.max):
            raise ValueError(f"{name} exceeds {dtype} range")
    return np.ascontiguousarray(raw, dtype=dtype)


def _markers(values, name):
    if values is None:
        return np.empty(0, dtype=np.int32)
    raw = np.asarray(values)
    if raw.size == 0:
        return np.empty(0, dtype=np.int32)
    if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.integer):
        raise ValueError(f"{name} must be a one-dimensional integer array")
    if np.any(raw < np.iinfo(np.int32).min) or np.any(raw > np.iinfo(np.int32).max):
        raise ValueError(f"{name} exceeds int32 range")
    return np.ascontiguousarray(raw, dtype=np.int32)


@dataclass
class Mesh3D:
    nodes: np.ndarray
    node_markers: np.ndarray
    input_vertex_nodes: np.ndarray
    tetrahedra: np.ndarray
    cell_markers: np.ndarray
    cell_regions: np.ndarray
    neighbors: np.ndarray
    faces: np.ndarray
    face_markers: np.ndarray
    face_facets: np.ndarray
    face_cells: np.ndarray
    report: dict[str, Any] = field(default_factory=dict)

    @property
    def cell_volumes(self):
        """Signed volumes; valid returned meshes have strictly positive volumes."""
        p = self.nodes[self.tetrahedra]
        return np.einsum("ij,ij->i", np.cross(p[:, 1]-p[:, 0], p[:, 2]-p[:, 0]),
                         p[:, 3]-p[:, 0]) / 6.0

    @property
    def cell_min_dihedrals(self):
        """Minimum internal dihedral per cell, in degrees (0 for degeneracy).

        Computed from the current arrays, so it also works after loading or
        editing a mesh. Edge scaling avoids overflow for large coordinates.
        """
        p = self.nodes[self.tetrahedra]
        edges = p - p[:, :1]
        scale = np.max(np.abs(edges), axis=(1, 2))
        p = np.divide(edges, scale[:, None, None], out=np.zeros_like(edges),
                      where=scale[:, None, None] > 0)
        faces = ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))
        normals = np.stack([np.cross(p[:, b]-p[:, a], p[:, c]-p[:, a])
                            for a, b, c in faces], axis=1)
        lengths = np.linalg.norm(normals, axis=2)
        normals = np.divide(normals, lengths[:, :, None], out=np.zeros_like(normals),
                            where=lengths[:, :, None] > 0)
        angles = [np.degrees(np.arctan2(
            np.linalg.norm(np.cross(normals[:, i], normals[:, j]), axis=1),
            -np.einsum("ij,ij->i", normals[:, i], normals[:, j])))
            for i in range(4) for j in range(i+1, 4)]
        result = np.min(angles, axis=0)
        result[np.any(lengths == 0, axis=1)] = 0.
        return result

    def write_vtu(self, path, *, include_quality=True):
        """Export tetrahedra, markers and optional per-cell quality to ParaView."""
        data = {"material": (self.cell_markers, "Int32"),
                "region": (self.cell_regions, "Int32")}
        if include_quality:
            data.update(volume=(self.cell_volumes, "Float64"),
                        min_dihedral=(self.cell_min_dihedrals, "Float64"))
        self._write_vtu(path, self.tetrahedra, 10, data)

    def write_boundary_vtu(self, path):
        """Export exterior/interface triangles with marker and input-facet IDs.

        Both incident cell IDs refer to the volume mesh; cell_1 is -1 on the
        exterior. Internal interfaces appear once, even between equal markers.
        """
        self._write_vtu(path, self.faces, 5, {
            "marker": (self.face_markers, "Int32"),
            "input_facet": (self.face_facets, "Int64"),
            "cell_0": (self.face_cells[:, 0], "Int64"),
            "cell_1": (self.face_cells[:, 1], "Int64"),
        })

    def _write_vtu(self, path, connectivity, vtk_type, cell_data):
        from xml.etree.ElementTree import Element, SubElement, ElementTree
        root = Element("VTKFile", type="UnstructuredGrid", version="0.1", byte_order="LittleEndian")
        piece = SubElement(SubElement(root, "UnstructuredGrid"), "Piece",
                           NumberOfPoints=str(len(self.nodes)), NumberOfCells=str(len(connectivity)))
        def array(parent, name, values, kind="Int64", components=None):
            attrs = dict(type=kind, Name=name, format="ascii")
            if components:
                attrs["NumberOfComponents"] = str(components)
            element = SubElement(parent, "DataArray", **attrs)
            element.text = " ".join(map(str, np.asarray(values).ravel()))
        array(SubElement(piece, "Points"), "Points", self.nodes, "Float64", 3)
        cells = SubElement(piece, "Cells")
        array(cells, "connectivity", connectivity)
        array(cells, "offsets", np.arange(1, len(connectivity)+1)*connectivity.shape[1])
        array(cells, "types", np.full(len(connectivity), vtk_type), "UInt8")
        data = SubElement(piece, "CellData")
        for name, (values, kind) in cell_data.items():
            array(data, name, values, kind)
        array(SubElement(piece, "PointData"), "marker", self.node_markers, "Int32")
        ElementTree(root).write(Path(path), encoding="utf-8", xml_declaration=True)


class MeshQualityError(RuntimeError):
    """Requested limits were unmet; ``mesh`` retains the recovered result.

    A geometry-recovery failure raises a different error. The attached mesh
    may need more refinement; ``mesh.report["issues"]`` lists what is unmet.
    """

    def __init__(self, mesh):
        self.mesh = mesh
        super().__init__("rustmesh could not meet the 3D mesh limits: " +
                         "; ".join(mesh.report["issues"]))


def tetrahedralize(vertices, facets=None, *, vertex_markers=None, facet_markers=None,
                   regions=None, holes=None, max_volume=0.0, max_vertices=10_000,
                   max_passes=100, min_dihedral=0.0, strict=True):
    """Mesh a closed triangular PLC, or the convex hull if facets are omitted.

    Regions are (x, y, z, marker[, max_volume]); holes are interior seed points.
    Facets may split, retaining their input index in ``face_facets``. Facet
    winding is arbitrary. Internal interfaces must meet shells conformingly.
    All coordinates use a common unit; volumes use that unit cubed.

    ``min_dihedral`` (degrees, 0: off) is a best-effort quality target: cells
    below it are improved by flips (edge removal, 2-3 flips) and, where flips
    are not enough, by vertices at interior circumcentres; the best recovered
    state is returned, then generated interior vertices are smoothed with
    input vertices and volume limits protected. A bounded local stage then
    subdivides edges and cells and alternates flips with smoothing; generated
    boundary points may slide on the original PLC faces and edges. Up to 16
    local trial rounds supplement the ``max_passes`` global rebuilds, and all
    of them share ``max_vertices``. With 0 the conforming Delaunay cells come
    back as they are, slivers included.

    ``strict=True`` raises ``MeshQualityError`` for unmet volume or angle
    limits; its ``mesh`` attribute keeps the recovered mesh. With
    ``strict=False`` the mesh is returned and ``report["issues"]`` lists what
    is unmet. A facet that cannot be recovered always raises.

    Neighbor slot k is opposite tetrahedron vertex k; -1 means exterior.
    Termination, quality and run time are not guaranteed for general input.
    """
    v = _matrix(vertices, 3, np.float64, "vertices")
    f = _matrix(facets, 3, np.int64, "facets")
    if regions is None or np.asarray(regions).size == 0:
        r = np.empty((0, 5), dtype=float)
    else:
        r = np.asarray(regions, dtype=float)
        if r.ndim != 2 or r.shape[1] not in (4, 5):
            raise ValueError("regions must have 4 or 5 columns: x, y, z, marker[, max_volume]")
        if r.shape[1] == 4:
            r = np.column_stack((r, np.zeros(len(r))))
        r = np.ascontiguousarray(r)
    for name, value in (("max_vertices", max_vertices), ("max_passes", max_passes)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    out = _tetrahedralize(v, f, _markers(vertex_markers, "vertex_markers"),
                         _markers(facet_markers, "facet_markers"), r,
                         _matrix(holes, 3, np.float64, "holes"), float(max_volume),
                         int(max_vertices), int(max_passes), float(min_dihedral))
    mesh = Mesh3D(**out)
    if strict and not mesh.report["complete"]:
        raise MeshQualityError(mesh)
    return mesh
