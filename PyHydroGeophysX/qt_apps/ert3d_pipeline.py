"""Synthetic 3D ERT forward modeling on a built mesh (Qt-free, worker-safe).

Given a 3D mesh + electrode layout (as produced by
:func:`mesh3d_builder.generate_mesh`) and a resistivity model, this generates
synthetic 3D ERT apparent-resistivity data. It mirrors
``examples/Ex_3D_ERT_forward.py``: build an ERT data container with 3D geometric
factors (:func:`core.mesh_3d.create_3d_ert_data_container`), run pygimli's
``ERTModelling`` response via :class:`forward.ert_forward.ERTForwardModeling`, add
noise, and save a pygimli ``.dat`` file plus a resistivity VTK.

Nothing here imports PySide6, so it is safe to call from a worker thread. pygimli
and the geophysics helpers are imported lazily inside :func:`run_ert3d_forward`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np

from PyHydroGeophysX.qt_apps import io_utils


class BackendUnavailable(RuntimeError):
    """Raised when pygimli / the 3D ERT helpers cannot be imported."""


def run_ert3d_forward(
    mesh: Any,
    electrodes: Any,
    scheme: str = "dd",
    background_res: float = 100.0,
    marker_res: Optional[Dict[int, float]] = None,
    noise: float = 0.03,
    seed: int = 42,
    output_dir: str = ".",
    log: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Run a 3D ERT forward simulation and save synthetic data.

    Parameters
    ----------
    mesh : pygimli.Mesh
        Forward mesh (e.g. from ``mesh3d_builder.generate_mesh``).
    electrodes : pandas.DataFrame
        Electrode positions with ``x, y, z`` columns.
    scheme : str
        Measurement scheme (``dd``, ``wa``, ``slm``, ``wb`` ...).
    background_res : float
        Resistivity (ohm-m) applied to every cell by default.
    marker_res : dict, optional
        Per-region resistivity override, keyed by mesh cell marker.
    noise : float
        Relative Gaussian noise added to the synthetic apparent resistivity.
    seed : int
        Random seed for the noise.
    output_dir : str
        Directory under which an ``ert3d_forward`` folder is written.
    log : callable, optional
        Progress callback taking one string.
    """
    say = log or (lambda *_: None)
    try:
        from pygimli.physics import ert

        from PyHydroGeophysX.core.mesh_3d import create_3d_ert_data_container
        from PyHydroGeophysX.forward.ert_forward import ERTForwardModeling
    except Exception as exc:  # noqa: BLE001
        raise BackendUnavailable(str(exc))

    if mesh is None or electrodes is None:
        raise ValueError("A generated 3D mesh and electrode layout are required.")

    say(f"Building 3D ERT data container (scheme={scheme})…")
    data = create_3d_ert_data_container(electrodes, scheme=scheme, dimension=3)

    n_cells = int(mesh.cellCount())
    res = np.full(n_cells, float(background_res), dtype=float)
    markers = np.asarray(mesh.cellMarkers(), dtype=int)
    applied_markers: Dict[int, float] = {}
    for m, r in (marker_res or {}).items():
        mask = markers == int(m)
        if mask.any():
            res[mask] = float(r)
            applied_markers[int(m)] = float(r)
    res = np.clip(res, 1e-3, 1e6)

    say(f"Running 3D ERT forward ({data.size()} measurements, {n_cells} cells)…")
    fwd = ERTForwardModeling(mesh, data)
    response = np.asarray(fwd.forward(res, log_transform=False), dtype=float)

    rng = np.random.default_rng(int(seed))
    noisy = response * (1.0 + float(noise) * rng.standard_normal(response.size))
    data["rhoa"] = noisy
    try:
        mgr = ert.ERTManager(data)
        data["err"] = mgr.estimateError(data, relativeError=float(noise), absoluteUError=1e-4)
    except Exception:  # noqa: BLE001 - error estimate is best effort
        data["err"] = np.abs(float(noise) * noisy)

    out = io_utils.ensure_dir(Path(output_dir) / "ert3d_forward")
    dat_path = out / "synthetic_3d_ert.dat"
    data.save(str(dat_path))
    say(f"Saved synthetic data to {dat_path}")

    vtk_path: Optional[Path] = out / "forward_3d_resistivity.vtk"
    try:
        mesh["resistivity"] = res
        mesh.exportVTK(str(vtk_path))
    except Exception:  # noqa: BLE001 - VTK export is optional
        vtk_path = None

    finite = response[np.isfinite(response)]
    return {
        "status": "ok",
        "scheme": scheme,
        "n_measurements": int(data.size()),
        "n_cells": n_cells,
        "background_res": float(background_res),
        "marker_res": applied_markers,
        "noise": float(noise),
        "rhoa_min": float(np.nanmin(finite)) if finite.size else None,
        "rhoa_max": float(np.nanmax(finite)) if finite.size else None,
        "data_file": str(dat_path),
        "vtk": str(vtk_path) if vtk_path else "",
        "output_dir": str(out),
    }
