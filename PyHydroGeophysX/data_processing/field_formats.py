"""Field data IO helpers for seismic and EM workflows."""

import csv
from typing import Any, Dict, Optional

import numpy as np


def read_seg2_seismic(file: str) -> Dict[str, np.ndarray]:
    """
    Read SEG-2 seismic file into arrays.

    If ObsPy is available, uses ObsPy SEG-2 reader.
    Otherwise falls back to text loading for already exported arrays.
    """
    try:
        from obspy import read

        stream = read(file, format="SEG2")
        traces = [tr.data.astype(float) for tr in stream]
        sampling_rate = np.array([float(tr.stats.sampling_rate) for tr in stream], dtype=float)

        return {
            "traces": np.array(traces, dtype=object),
            "sampling_rate": sampling_rate,
            "n_traces": np.array([len(traces)], dtype=int),
        }
    except Exception:
        arr = np.loadtxt(file)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return {
            "traces": arr,
            "sampling_rate": np.array([np.nan], dtype=float),
            "n_traces": np.array([arr.shape[0]], dtype=int),
        }


def read_tem_fast(file: str) -> Dict[str, np.ndarray]:
    """Read TEM-FAST style text data into SimPEG-ready arrays."""
    arr = np.loadtxt(file, comments="#")
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    if arr.shape[1] < 2:
        raise ValueError("TEM-FAST file must contain at least time and data columns.")

    result = {
        "time": arr[:, 0].astype(float),
        "data": arr[:, 1].astype(float),
    }

    if arr.shape[1] >= 3:
        result["uncertainty"] = arr[:, 2].astype(float)

    return result


def export_to_vtk(
    result: Any,
    mesh: Any,
    filename: str,
) -> str:
    """
    Export result/model values to a lightweight VTK PolyData file.

    The export writes cell-center points with a single scalar field.
    """
    if mesh is None or not hasattr(mesh, "cellCenters"):
        raise ValueError("mesh must provide cellCenters() for VTK export.")

    values = None
    for key in ["final_model", "recovered_conductivity", "recovered_model"]:
        if hasattr(result, key) and getattr(result, key) is not None:
            values = np.asarray(getattr(result, key), dtype=float).ravel()
            break

    if values is None:
        raise ValueError("result must contain final_model or recovered_conductivity for export.")

    centers_raw = np.asarray(mesh.cellCenters())
    if centers_raw.ndim == 2 and centers_raw.shape[1] >= 2:
        points = centers_raw[:, :3] if centers_raw.shape[1] >= 3 else np.column_stack((centers_raw[:, 0], centers_raw[:, 1], np.zeros(centers_raw.shape[0])))
    else:
        points = []
        for center in mesh.cellCenters():
            if hasattr(center, "x") and hasattr(center, "y") and hasattr(center, "z"):
                points.append([float(center.x()), float(center.y()), float(center.z())])
            elif hasattr(center, "x") and hasattr(center, "y"):
                points.append([float(center.x()), float(center.y()), 0.0])
            else:
                points.append([float(center[0]), float(center[1]), 0.0])
        points = np.asarray(points, dtype=float)

    n_points = points.shape[0]
    if values.size != n_points:
        x_src = np.linspace(0.0, 1.0, values.size)
        x_tgt = np.linspace(0.0, 1.0, n_points)
        values = np.interp(x_tgt, x_src, values)

    with open(filename, "w", encoding="utf-8") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("PyHydroGeophysX export\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")
        f.write(f"POINTS {n_points} float\n")
        for p in points:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")

        f.write(f"VERTICES {n_points} {n_points * 2}\n")
        for i in range(n_points):
            f.write(f"1 {i}\n")

        f.write(f"POINT_DATA {n_points}\n")
        f.write("SCALARS model float 1\n")
        f.write("LOOKUP_TABLE default\n")
        for v in values:
            f.write(f"{float(v)}\n")

    return filename


def export_results_to_csv(
    result: Any,
    filename: str,
) -> str:
    """Export inversion result arrays to a tabular CSV file."""
    fields = [
        "final_model",
        "recovered_model",
        "recovered_conductivity",
        "predicted_data",
        "coverage",
    ]

    arrays = {}
    max_len = 0

    for field in fields:
        if hasattr(result, field) and getattr(result, field) is not None:
            arr = np.asarray(getattr(result, field)).ravel()
            arrays[field] = arr
            max_len = max(max_len, arr.size)

    if not arrays:
        raise ValueError("No exportable arrays found in result object.")

    for key, arr in list(arrays.items()):
        if arr.size < max_len:
            pad = np.full(max_len - arr.size, np.nan)
            arrays[key] = np.hstack((arr.astype(float), pad))
        else:
            arrays[key] = arr.astype(float)

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = list(arrays.keys())
        writer.writerow(header)
        for i in range(max_len):
            writer.writerow([arrays[h][i] for h in header])

    return filename
