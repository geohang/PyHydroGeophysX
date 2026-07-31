"""Gravity/magnetics workflow facade and result export."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from PyHydroGeophysX._internal.utils import noop as _noop, utc_now as _utc_now
from PyHydroGeophysX.data_processing import table_io
from PyHydroGeophysX.data_processing.gravmag import (
    extract_profile,
    grid_data,
    qc_products,
    regional_residual,
    spatially_balanced_indices,
)
from PyHydroGeophysX.forward.gravmag import (
    forward_bodies,
    gravity_prism,
    gravity_sphere,
    magnetic_dipole,
)
from PyHydroGeophysX.inversion.gravmag import (
    InversionBackendUnavailable,
    backend_status,
    invert_gravmag,
)

LogFn = Callable[[str], None]

def save_grid(grid: Dict[str, np.ndarray], out_dir: Path, name: str = "anomaly",
              log: LogFn = _noop) -> List[str]:
    """Save a grid to npy + CSV + VTK (best-effort). Return written paths."""
    out = table_io.ensure_dir(out_dir)
    paths: List[str] = []
    xx, yy, zz = grid["xx"], grid["yy"], grid["zz"]
    np.save(out / f"{name}_grid.npy", np.asarray(zz, float)); paths.append(str(out / f"{name}_grid.npy"))
    rows = list(zip(xx.ravel().tolist(), yy.ravel().tolist(), np.asarray(zz, float).ravel().tolist()))
    table_io.write_csv(out / f"{name}_grid.csv", rows, header=["x", "y", name])
    paths.append(str(out / f"{name}_grid.csv"))
    try:
        import pyvista as pv
        sg = pv.StructuredGrid(np.asarray(xx, float), np.asarray(yy, float),
                               np.zeros_like(np.asarray(xx, float)))
        sg[name] = np.asarray(zz, float).ravel(order="F")
        sg.save(str(out / f"{name}_grid.vtk"))
        paths.append(str(out / f"{name}_grid.vtk"))
    except Exception as exc:  # noqa: BLE001 - VTK is best-effort
        log(f"VTK export skipped: {exc}")
    return paths


def build_gravmag_config(kind: str, settings: Dict[str, Any], bodies: List[Dict[str, Any]],
                         field: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "created_time": _utc_now(),
        "kind": kind,
        "settings": dict(settings),
        "bodies": [dict(b) for b in bodies],
        "field": dict(field) if field else {},
    }

__all__ = [
    "regional_residual",
    "spatially_balanced_indices",
    "qc_products",
    "grid_data",
    "extract_profile",
    "gravity_sphere",
    "gravity_prism",
    "magnetic_dipole",
    "forward_bodies",
    "save_grid",
    "build_gravmag_config",
    "InversionBackendUnavailable",
    "backend_status",
    "invert_gravmag",
]
