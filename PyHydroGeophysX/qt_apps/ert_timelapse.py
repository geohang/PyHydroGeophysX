"""Time-lapse ERT inversion pipeline for the desktop workbench (Qt-free).

A thin wrapper around ``PyHydroGeophysX.inversion.time_lapse.TimeLapseERTInversion``
(temporal-regularized full time-lapse inversion). It builds a mesh from the first
dataset, runs the inversion over a sequence of ERT data files, renders the
resistivity-evolution panel, and exports the models (npy), mesh (bms), coverage
(npy) and an all-times VTK. pygimli is imported lazily; if it is missing the run
raises ``BackendUnavailable``.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from PyHydroGeophysX.qt_apps import io_utils

LogFn = Callable[[str], None]

INVERSION_TYPES = ("L2", "L1", "L1L2")

DEFAULT_TL = {
    "lambda_val": 50.0, "alpha": 10.0, "inversion_type": "L2",
    "max_iterations": 15, "relativeError": 0.05, "absoluteUError": 0.0,
    "method": "cgls", "mesh_quality": 34.0, "rho_min": 1.0, "rho_max": 1.0e4,
    "windowed": False, "window_size": 3,
}


class BackendUnavailable(RuntimeError):
    """Raised when pygimli / the time-lapse inversion cannot be imported."""


def _noop(_msg: str) -> None:
    return None


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def default_times(n: int) -> List[int]:
    """Sequential measurement times 1..n when the user has none."""
    return list(range(1, int(n) + 1))


def build_timelapse_config(data_files: Sequence[str], measurement_times: Sequence[float],
                           params: Dict[str, Any]) -> Dict[str, Any]:
    """JSON-serializable configuration (no backend needed)."""
    p = {**DEFAULT_TL, **(params or {})}
    return {
        "created_time": _utc_now(),
        "direction": "ert_time_lapse_inversion",
        "n_files": len(data_files),
        "data_files": [str(f) for f in data_files],
        "measurement_times": [float(t) for t in measurement_times],
        "inversion": {k: p[k] for k in (
            "lambda_val", "alpha", "inversion_type", "max_iterations",
            "relativeError", "method", "mesh_quality", "rho_min", "rho_max",
            "windowed", "window_size")},
    }


def run_timelapse_ert(
    data_files: Sequence[str],
    measurement_times: Sequence[float],
    params: Dict[str, Any],
    out_dir: str,
    log: LogFn = _noop,
) -> Dict[str, Any]:
    """Run a full temporal-regularized time-lapse ERT inversion.

    Raises ``BackendUnavailable`` if pygimli / the inversion cannot be imported,
    and propagates other exceptions so the caller can fall back to config export.
    """
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        plt.ioff()
        import pygimli as pg
        from pygimli.physics import ert as pg_ert
        from PyHydroGeophysX.inversion.time_lapse import TimeLapseERTInversion
        from PyHydroGeophysX.inversion.windowed import WindowedTimeLapseERTInversion
    except Exception as exc:  # noqa: BLE001
        raise BackendUnavailable(str(exc))

    files = [str(f) for f in data_files]
    if len(files) < 2:
        raise ValueError("Time-lapse inversion needs at least two ERT data files.")
    times = list(measurement_times) if measurement_times is not None and len(measurement_times) == len(files) \
        else default_times(len(files))
    p = {**DEFAULT_TL, **(params or {})}

    log(f"Building mesh from {Path(files[0]).name} (quality {p['mesh_quality']})")
    data0 = pg_ert.load(files[0])
    mesh = pg_ert.ERTManager(data0).createMesh(data=data0, quality=float(p["mesh_quality"]))

    inv_kwargs = dict(
        lambda_val=float(p["lambda_val"]), alpha=float(p["alpha"]),
        method=str(p["method"]), max_iterations=int(p["max_iterations"]),
        relativeError=float(p["relativeError"]), absoluteUError=float(p.get("absoluteUError", 0.0)),
        inversion_type=str(p["inversion_type"]),
        model_constraints=(float(p["rho_min"]), float(p["rho_max"])),
    )
    import os
    use_windowed = bool(p.get("windowed", False)) and 2 <= int(p["window_size"]) <= len(files)
    if use_windowed and len({os.path.dirname(f) for f in files}) != 1:
        log("Windowed mode needs all files in one folder; falling back to full inversion.")
        use_windowed = False

    if use_windowed:
        window_size = int(p["window_size"])
        data_dir = os.path.dirname(files[0])
        ert_files = [os.path.basename(f) for f in files]
        log(f"Running windowed {p['inversion_type']} time-lapse inversion: {len(files)} steps, "
            f"window={window_size}, lambda={p['lambda_val']}, alpha={p['alpha']}")
        inversion = WindowedTimeLapseERTInversion(
            data_dir=data_dir, ert_files=ert_files, measurement_times=times,
            window_size=window_size, mesh=mesh, **inv_kwargs)
        result = inversion.run(window_parallel=False)
        mode = "windowed"
    else:
        log(f"Running full {p['inversion_type']} time-lapse inversion: {len(files)} steps, "
            f"lambda={p['lambda_val']}, alpha={p['alpha']}")
        inversion = TimeLapseERTInversion(data_files=files, measurement_times=times,
                                          mesh=mesh, **inv_kwargs)
        result = inversion.run()
        mode = "full"

    final_models = np.asarray(result.final_models, dtype=float)
    if final_models.ndim != 2:
        final_models = final_models.reshape(mesh.cellCount(), -1)
    n_time = final_models.shape[1]
    coverage = None
    try:
        coverage = np.asarray(result.all_coverage, dtype=float)
    except Exception:  # noqa: BLE001 - coverage is optional
        coverage = None
    res_mesh = getattr(result, "mesh", mesh)

    out = io_utils.ensure_dir(Path(out_dir) / "qt_ert_timelapse")
    figure_paths: List[str] = []
    data_paths: List[str] = []

    # Resistivity-evolution panel.
    finite = final_models[np.isfinite(final_models)]
    cmin = float(np.percentile(finite, 5)) if finite.size else 1.0
    cmax = float(np.percentile(finite, 95)) if finite.size else 1000.0
    ncol = min(4, n_time)
    nrow = int(np.ceil(n_time / ncol))
    fig = plt.figure(figsize=(3.6 * ncol, 3.0 * nrow))
    for i in range(n_time):
        ax = fig.add_subplot(nrow, ncol, i + 1)
        show_kw = dict(ax=ax, cMap="Spectral_r", cMin=cmin, cMax=cmax,
                       label="Resistivity (Ω·m)", logScale=False)
        if coverage is not None and coverage.shape[0] > i:
            show_kw["coverage"] = coverage[i] > -1.0
        try:
            pg.show(res_mesh, final_models[:, i], **show_kw)
        except Exception:  # noqa: BLE001 - retry without coverage
            show_kw.pop("coverage", None)
            ax.clear()
            pg.show(res_mesh, final_models[:, i], **show_kw)
        ax.set_title(f"t = {times[i]:g}")
    fig.tight_layout()
    panel = out / "timelapse_resistivity.png"
    fig.savefig(panel, dpi=160, bbox_inches="tight"); plt.close(fig)
    figure_paths.append(str(panel))

    # Exports.
    np.save(out / "final_models.npy", final_models); data_paths.append(str(out / "final_models.npy"))
    if coverage is not None:
        np.save(out / "all_coverage.npy", coverage); data_paths.append(str(out / "all_coverage.npy"))
    io_utils.write_csv(out / "measurement_times.csv", [(i, float(t)) for i, t in enumerate(times)],
                       header=["index", "time"])
    data_paths.append(str(out / "measurement_times.csv"))
    try:
        res_mesh.save(str(out / "timelapse_mesh.bms")); data_paths.append(str(out / "timelapse_mesh.bms"))
    except Exception as exc:  # noqa: BLE001
        log(f"Mesh export skipped: {exc}")
    try:
        for i in range(n_time):
            res_mesh[f"resistivity_t{i}"] = final_models[:, i]
        vtk = out / "timelapse_resistivity.vtk"
        res_mesh.exportVTK(str(vtk)); data_paths.append(str(vtk))
    except Exception as exc:  # noqa: BLE001
        log(f"VTK export skipped: {exc}")

    config = build_timelapse_config(files, times, p)
    io_utils.write_json(out / "timelapse_config.json", config)

    return {
        "status": "ok",
        "direction": "ert_time_lapse_inversion",
        "mode": mode,
        "n_times": int(n_time),
        "mesh_cells": int(res_mesh.cellCount()),
        "inversion_type": str(p["inversion_type"]),
        "resistivity_range": [cmin, cmax],
        "figure_paths": figure_paths,
        "data_paths": data_paths,
        "config_path": str(out / "timelapse_config.json"),
        "output_dir": str(out),
    }
