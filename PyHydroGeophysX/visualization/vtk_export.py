"""VTK export utilities for ParaView visualization."""

import os
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

# ---------------------------------------------------------------------------
# 2-D mesh export (unstructured triangular / quad mesh)
# ---------------------------------------------------------------------------

def export_mesh_to_vtk(
    mesh: Any,
    filename: str,
    cell_data: Optional[Dict[str, np.ndarray]] = None,
) -> str:
    """Export a PyGIMLi mesh to VTK unstructured grid format.

    This uses PyGIMLi's built-in ``mesh.exportVTK`` when available
    and then optionally injects extra cell data fields.

    Parameters
    ----------
    mesh : pygimli.Mesh
        The mesh to export.
    filename : str
        Output ``.vtk`` path.
    cell_data : dict of str -> array, optional
        Additional named cell-data arrays to write.

    Returns
    -------
    str
        Path to the written file.
    """
    if not filename.endswith(".vtk"):
        filename += ".vtk"

    # Use pygimli's native VTK export
    if hasattr(mesh, "exportVTK"):
        mesh.exportVTK(filename)

    # If extra cell data requested, append it
    if cell_data:
        _append_cell_data_to_vtk(filename, cell_data, mesh.cellCount())

    return filename


def _append_cell_data_to_vtk(
    filename: str,
    cell_data: Dict[str, np.ndarray],
    n_cells: int,
):
    """Append CELL_DATA fields to an existing ASCII VTK file."""
    with open(filename, "r", encoding="utf-8") as f:
        content = f.read()

    has_cell_data = "CELL_DATA" in content

    lines = []
    if not has_cell_data:
        lines.append(f"\nCELL_DATA {n_cells}")

    for name, values in cell_data.items():
        arr = np.asarray(values, dtype=float).ravel()
        if arr.size != n_cells:
            raise ValueError(
                f"Cell data '{name}' has {arr.size} values but mesh has {n_cells} cells."
            )
        lines.append(f"SCALARS {name} float 1")
        lines.append("LOOKUP_TABLE default")
        for v in arr:
            lines.append(f"{float(v)}")

    with open(filename, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Structured grid export (for MODFLOW / ParFlow 3-D grids)
# ---------------------------------------------------------------------------

def export_structured_vtk(
    values: np.ndarray,
    filename: str,
    *,
    dx: float = 1.0,
    dy: float = 1.0,
    dz: float = 1.0,
    origin: tuple = (0.0, 0.0, 0.0),
    scalar_name: str = "model",
) -> str:
    """Export a 3-D numpy array to VTK structured-points format.

    Useful for MODFLOW/ParFlow grids that map directly to a regular grid.

    Parameters
    ----------
    values : 3-D array
        Shape ``(nz, ny, nx)`` — layer, row, column ordering.
    filename : str
        Output ``.vtk`` path.
    dx, dy, dz : float
        Cell spacing in each direction.
    origin : tuple of float
        ``(x0, y0, z0)`` origin of the grid.
    scalar_name : str
        Name for the scalar dataset.

    Returns
    -------
    str
        Path to the written file.
    """
    if not filename.endswith(".vtk"):
        filename += ".vtk"

    arr = np.asarray(values, dtype=float)
    if arr.ndim != 3:
        raise ValueError(f"values must be 3-D, got {arr.ndim}-D.")

    nz, ny, nx = arr.shape
    n_points = (nx + 1) * (ny + 1) * (nz + 1)
    n_cells = nx * ny * nz

    with open(filename, "w", encoding="utf-8") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("PyHydroGeophysX structured grid export\n")
        f.write("ASCII\n")
        f.write("DATASET STRUCTURED_POINTS\n")
        f.write(f"DIMENSIONS {nx + 1} {ny + 1} {nz + 1}\n")
        f.write(f"ORIGIN {origin[0]} {origin[1]} {origin[2]}\n")
        f.write(f"SPACING {dx} {dy} {dz}\n")
        f.write(f"\nCELL_DATA {n_cells}\n")
        f.write(f"SCALARS {scalar_name} float 1\n")
        f.write("LOOKUP_TABLE default\n")
        # VTK structured points: x varies fastest, then y, then z
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    f.write(f"{float(arr[k, j, i])}\n")

    return filename


# ---------------------------------------------------------------------------
# Multi-scalar structured export
# ---------------------------------------------------------------------------

def export_structured_vtk_multi(
    scalars: Dict[str, np.ndarray],
    filename: str,
    *,
    dx: float = 1.0,
    dy: float = 1.0,
    dz: float = 1.0,
    origin: tuple = (0.0, 0.0, 0.0),
) -> str:
    """Export multiple 3-D arrays as named scalars in a single VTK file.

    All arrays must have the same shape ``(nz, ny, nx)``.

    Parameters
    ----------
    scalars : dict of str -> 3-D array
        Named scalar datasets (e.g. ``{"resistivity": res, "water_content": wc}``).
    filename : str
        Output ``.vtk`` path.

    Returns
    -------
    str
        Path to the written file.
    """
    if not scalars:
        raise ValueError("scalars dict must not be empty.")

    if not filename.endswith(".vtk"):
        filename += ".vtk"

    shapes = set()
    for name, arr in scalars.items():
        a = np.asarray(arr, dtype=float)
        if a.ndim != 3:
            raise ValueError(f"'{name}' must be 3-D, got {a.ndim}-D.")
        shapes.add(a.shape)
    if len(shapes) != 1:
        raise ValueError("All scalar arrays must have the same shape.")

    nz, ny, nx = shapes.pop()
    n_cells = nx * ny * nz

    with open(filename, "w", encoding="utf-8") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("PyHydroGeophysX multi-scalar export\n")
        f.write("ASCII\n")
        f.write("DATASET STRUCTURED_POINTS\n")
        f.write(f"DIMENSIONS {nx + 1} {ny + 1} {nz + 1}\n")
        f.write(f"ORIGIN {origin[0]} {origin[1]} {origin[2]}\n")
        f.write(f"SPACING {dx} {dy} {dz}\n")
        f.write(f"\nCELL_DATA {n_cells}\n")

        for name, arr_raw in scalars.items():
            arr = np.asarray(arr_raw, dtype=float)
            f.write(f"SCALARS {name} float 1\n")
            f.write("LOOKUP_TABLE default\n")
            for k in range(nz):
                for j in range(ny):
                    for i in range(nx):
                        f.write(f"{float(arr[k, j, i])}\n")

    return filename


# ---------------------------------------------------------------------------
# Time-lapse VTK series (for ParaView animation)
# ---------------------------------------------------------------------------

def export_timelapse_vtk(
    mesh: Any,
    models: Sequence[np.ndarray],
    output_dir: str,
    prefix: str = "timelapse",
    *,
    scalar_name: str = "model",
    extra_data: Optional[Dict[str, Sequence[np.ndarray]]] = None,
) -> List[str]:
    """Export time-lapse models as a numbered VTK series for ParaView.

    Creates files ``prefix_0000.vtk``, ``prefix_0001.vtk``, ... and a
    ``.pvd`` collection file that ParaView can load as a time series.

    Parameters
    ----------
    mesh : pygimli.Mesh
        Mesh shared by all timesteps.
    models : sequence of array-like
        Model values per timestep.
    output_dir : str
        Directory for output files.
    prefix : str
        Filename prefix.
    scalar_name : str
        Name for the primary scalar field.
    extra_data : dict of str -> sequence of arrays, optional
        Additional scalar fields per timestep.

    Returns
    -------
    list of str
        Paths to all generated VTK files.
    """
    os.makedirs(output_dir, exist_ok=True)

    vtk_files = []
    for i, model in enumerate(models):
        fname = os.path.join(output_dir, f"{prefix}_{i:04d}.vtk")
        cell_data = {scalar_name: np.asarray(model, dtype=float).ravel()}
        if extra_data:
            for key, arrs in extra_data.items():
                cell_data[key] = np.asarray(arrs[i], dtype=float).ravel()
        export_mesh_to_vtk(mesh, fname, cell_data=cell_data)
        vtk_files.append(fname)

    # Write PVD collection file
    pvd_path = os.path.join(output_dir, f"{prefix}.pvd")
    _write_pvd(pvd_path, vtk_files)

    return vtk_files


def _write_pvd(pvd_path: str, vtk_files: List[str]):
    """Write a ParaView Collection (.pvd) file referencing the VTK series."""
    with open(pvd_path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="Collection" version="0.1">\n')
        f.write("  <Collection>\n")
        for i, vtk_file in enumerate(vtk_files):
            rel = os.path.basename(vtk_file)
            f.write(f'    <DataSet timestep="{i}" file="{rel}"/>\n')
        f.write("  </Collection>\n")
        f.write("</VTKFile>\n")


# ---------------------------------------------------------------------------
# Structured time-lapse VTK (MODFLOW / ParFlow grids)
# ---------------------------------------------------------------------------

def export_timelapse_structured_vtk(
    models: Sequence[np.ndarray],
    output_dir: str,
    prefix: str = "timelapse",
    *,
    dx: float = 1.0,
    dy: float = 1.0,
    dz: float = 1.0,
    origin: tuple = (0.0, 0.0, 0.0),
    scalar_name: str = "model",
) -> List[str]:
    """Export time-lapse 3-D arrays as a numbered VTK series.

    Parameters
    ----------
    models : sequence of 3-D arrays
        Each entry has shape ``(nz, ny, nx)``.
    output_dir : str
        Output directory.
    prefix : str
        File prefix.
    dx, dy, dz : float
        Grid spacing.
    origin : tuple
        Grid origin.
    scalar_name : str
        Scalar field name.

    Returns
    -------
    list of str
        Paths to generated VTK files.
    """
    os.makedirs(output_dir, exist_ok=True)

    vtk_files = []
    for i, model in enumerate(models):
        fname = os.path.join(output_dir, f"{prefix}_{i:04d}.vtk")
        export_structured_vtk(model, fname, dx=dx, dy=dy, dz=dz,
                              origin=origin, scalar_name=scalar_name)
        vtk_files.append(fname)

    pvd_path = os.path.join(output_dir, f"{prefix}.pvd")
    _write_pvd(pvd_path, vtk_files)

    return vtk_files


# ---------------------------------------------------------------------------
# Point-cloud export (lightweight, no mesh topology)
# ---------------------------------------------------------------------------

def export_points_to_vtk(
    points: np.ndarray,
    scalars: Dict[str, np.ndarray],
    filename: str,
) -> str:
    """Export point-cloud data (e.g. cell centers) to VTK PolyData.

    Parameters
    ----------
    points : array of shape (N, 3)
        XYZ coordinates.
    scalars : dict of str -> array
        Named scalar values at each point.
    filename : str
        Output ``.vtk`` path.

    Returns
    -------
    str
        Path to the written file.
    """
    if not filename.endswith(".vtk"):
        filename += ".vtk"

    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 2:
        raise ValueError("points must be (N, 2) or (N, 3).")
    if pts.shape[1] == 2:
        pts = np.column_stack((pts, np.zeros(pts.shape[0])))

    n = pts.shape[0]

    with open(filename, "w", encoding="utf-8") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("PyHydroGeophysX point export\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")
        f.write(f"POINTS {n} float\n")
        for p in pts:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")
        f.write(f"VERTICES {n} {n * 2}\n")
        for i in range(n):
            f.write(f"1 {i}\n")
        f.write(f"\nPOINT_DATA {n}\n")

        for name, values in scalars.items():
            arr = np.asarray(values, dtype=float).ravel()
            if arr.size != n:
                raise ValueError(
                    f"Scalar '{name}' has {arr.size} values but {n} points."
                )
            f.write(f"SCALARS {name} float 1\n")
            f.write("LOOKUP_TABLE default\n")
            for v in arr:
                f.write(f"{float(v)}\n")

    return filename
