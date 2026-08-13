"""Time-lapse ERT inversion pipeline for the desktop workbench (Qt-free).

A thin wrapper around ``PyHydroGeophysX.inversion.time_lapse.TimeLapseERTInversion``
(temporal-regularized full time-lapse inversion). It builds a mesh from the first
dataset, runs the inversion over a sequence of ERT data files, renders the
resistivity-evolution panel, and exports the models (npy), mesh (bms), coverage
(npy) and an all-times VTK. pygimli is imported lazily; if it is missing the run
raises ``BackendUnavailable``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from PyHydroGeophysX.data_processing import table_io as io_utils
from PyHydroGeophysX._internal.optional_dependencies import BackendUnavailable
from PyHydroGeophysX._internal.utils import noop as _noop, utc_now as _utc_now
from PyHydroGeophysX.data_processing import ert_io as ert_load
from PyHydroGeophysX.visualization import ert_style as ert_plot_style

LogFn = Callable[[str], None]

INVERSION_TYPES = ("L2", "L1", "L1L2")

DEFAULT_TL = {
    "lambda_val": 50.0, "alpha": 10.0, "inversion_type": "L2",
    "max_iterations": 15, "relativeError": 0.05, "absoluteUError": 0.0,
    "method": "cgls", "mesh_quality": 34.0, "rho_min": 1.0, "rho_max": 1.0e4,
    "windowed": False, "window_size": 3, "save_memory": False, "instrument": None,
    "engine": "pyhydro",
    "para_depth": 0.0,
    "max_error": None,
    # Lambda relaxation. A trial here is a full joint inversion over every time
    # step, so the default budget is smaller than the single-inversion search.
    "auto_lambda": False, "target_chi2": 1.0, "chi2_tolerance": 0.2,
    "max_lambda_trials": 4, "lambda_warm_start": True,
}

#: Above this many model unknowns (para cells x time steps) the dense
#: Gauss-Newton matrices get large, so sparse/low-memory mode is auto-enabled
#: unless the caller set ``save_memory`` explicitly.
_AUTO_SPARSE_UNKNOWNS = 15000


def _relax_timelapse_lambda(inversion, first, *, target_chi2: float,
                            chi2_tolerance: float, max_trials: int,
                            warm_start: bool, log: Callable[[str], None]):
    """Relax lambda between converged time-lapse runs until chi2 hits the target.

    Deliberately simpler than the single-inversion search: it only steps
    downward, geometrically, from the requested lambda. A time-lapse trial costs
    a full joint inversion over every time step, so bracketing and bisecting is
    not worth the runtime; and the sweep is warm-started from the previous
    solution, which is where most of the saving comes from.

    Returns ``(result, info)`` and never returns a worse fit than it was given.
    """
    from .ert_inversion import LAMBDA_BOUNDS

    target, tol = float(target_chi2), abs(float(chi2_tolerance))
    start_lambda = float(inversion.parameters["lambda_val"])
    best_result, best_chi2, best_lambda = first, float(first.meta["chi2"]), start_lambda
    trials = [{"lambda": start_lambda, "chi2": best_chi2,
               "iterations": int(first.meta.get("iterations", 0))}]
    info: Dict[str, Any] = {
        "enabled": True, "warm_start": bool(warm_start),
        "lambda_requested": start_lambda, "lambda_used": start_lambda,
        "trials": trials, "status": "already_on_target", "note": "",
    }
    if best_chi2 != best_chi2 or abs(best_chi2 - target) <= tol:
        return best_result, info

    log(f"  chi2 {best_chi2:.2f} outside {target:g} +/- {tol:g}; "
        f"relaxing lambda (max {int(max_trials)} trials)")
    lam = start_lambda
    previous = best_result
    for _ in range(max(0, int(max_trials))):
        # chi2 falls as lambda falls, so only ever step down; stepping up would
        # be moving away from a target the current lambda already overshoots.
        lam = max(lam / 4.0, float(min(LAMBDA_BOUNDS)))
        if lam >= trials[-1]["lambda"]:
            info["status"] = "best_effort"
            break
        inversion.parameters["lambda_val"] = float(lam)
        seed = np.asarray(previous.final_models, dtype=float) if warm_start else None
        trial = inversion.run(initial_model=seed)
        chi2 = float(trial.meta["chi2"])
        trials.append({"lambda": float(lam), "chi2": chi2,
                       "iterations": int(trial.meta.get("iterations", 0))})
        log(f"  lam {lam:g}" + (" warm" if warm_start else " cold")
            + f" -> chi2 {chi2:.3f} ({trial.meta.get('iterations', 0)} it)")
        if chi2 == chi2 and abs(chi2 - target) < abs(best_chi2 - target):
            best_result, best_chi2, best_lambda = trial, chi2, float(lam)
        previous = trial
        if chi2 == chi2 and abs(chi2 - target) <= tol:
            info["status"] = "converged"
            break
    else:
        info["status"] = "best_effort"

    inversion.parameters["lambda_val"] = start_lambda  # leave the object as found
    info["lambda_used"] = best_lambda
    if best_lambda != start_lambda:
        info["note"] = (f"Auto-λ: {start_lambda:g} → {best_lambda:g}, "
                        f"χ² {trials[0]['chi2']:.2f} → {best_chi2:.2f} in "
                        f"{len(trials) - 1} trial(s).")
    else:
        info["status"] = "no_improvement"
        info["note"] = f"Auto-λ: no λ beat {start_lambda:g} (χ² {best_chi2:.2f})."
    log("  " + info["note"])
    return best_result, info


def default_times(n: int) -> List[int]:
    """Sequential measurement times 1..n when the user has none."""
    return list(range(1, int(n) + 1))


def _step_titles(labels: Sequence[str], times: Sequence[float], n_time: int) -> List[str]:
    """Clear per-step titles so the panel always says what the number means:
    a parsed date stays as the date; a plain 1..n sequence becomes "Time step N";
    any other numeric time becomes "t = <value>"."""
    labels = list(labels or [])
    is_dated = any("-" in str(lbl) for lbl in labels)
    is_sequence = labels == [str(i + 1) for i in range(n_time)]
    titles: List[str] = []
    for i in range(n_time):
        lbl = labels[i] if i < len(labels) else str(i + 1)
        if is_dated:
            titles.append(str(lbl))
        elif is_sequence:
            titles.append(f"Time step {i + 1}")
        else:
            t = times[i] if i < len(times) else (i + 1)
            titles.append(f"t = {t:g}" if isinstance(t, (int, float)) else f"t = {lbl}")
    return titles


def _safe_label(text: str) -> str:
    """Filename-safe version of a time label."""
    out = "".join(c if c.isalnum() or c in "-_." else "_" for c in str(text))
    return out.strip("_") or "step"


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
        "instrument": p.get("instrument"),
        "inversion": {k: p[k] for k in (
            "lambda_val", "alpha", "inversion_type", "max_iterations",
            "relativeError", "method", "mesh_quality", "rho_min", "rho_max",
            "windowed", "window_size", "save_memory", "engine", "para_depth",
            "max_error")},
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

    import os

    source_files = [str(f) for f in data_files]
    if len(source_files) < 2:
        raise ValueError("Time-lapse inversion needs at least two ERT data files.")
    provided = dict(params or {})
    p = {**DEFAULT_TL, **provided}
    from .ert_inversion import _resolve_ert_engine

    requested_engine = str(p.get("engine", "pyhydro")).lower()
    engine = _resolve_ert_engine(requested_engine, log=log)
    if engine == "adtlert":
        # Defaults from the paper branch's real-data time-lapse example.
        # Explicit caller choices still win.
        if "windowed" not in provided:
            p["windowed"] = True
        if "window_size" not in provided:
            p["window_size"] = 3
        if "max_iterations" not in provided:
            p["max_iterations"] = 5
        if "rho_min" not in provided:
            p["rho_min"] = 1.0
        if "rho_max" not in provided:
            p["rho_max"] = 1.0e5
        if "para_depth" not in provided:
            p["para_depth"] = 90.0
        p["method"] = "cgls"  # maps to GPU CGLS for the CUDA-only backend
    use_windowed = (
        bool(p.get("windowed", False))
        and 2 <= int(p["window_size"]) <= len(source_files)
    )
    if engine == "adtlert" and not use_windowed:
        raise ValueError(
            "engine='adtlert' currently requires windowed=True for time-lapse ERT"
        )
    if engine not in ("pyhydro", "adtlert"):
        raise ValueError(
            "Time-lapse ERT engine must be 'pyhydro' or windowed 'adtlert'"
        )

    # Derive measurement times + display labels. Filenames that embed dates (e.g.
    # the E4D monthly series 2021-10-08_1400.ohm) give elapsed-day times and date
    # labels; otherwise fall back to a sequential 1..n.
    if measurement_times is not None and len(measurement_times) == len(source_files):
        times = [float(t) for t in measurement_times]
        labels = [f"{t:g}" for t in times]
    else:
        times, labels = ert_load.measurement_times_for(source_files)

    # Load every file through the robust device-aware loader and re-write it as a
    # clean pygimli file. Raw ``ert.load`` cannot parse index-prefixed / header-less
    # formats (E4D etc.) and would drop all data + topography; normalizing first is
    # what makes the time-lapse inversion actually work on those files.
    instrument = p.get("instrument")
    log(f"Preparing {len(source_files)} ERT files"
        + (f" (instrument: {instrument})" if instrument else " (auto-detect)") + " …")
    clean_dir, basenames, containers = ert_load.normalize_for_timelapse(
        source_files, instrument, out_dir, log=log,
        max_error=p.get("max_error"))
    files = [os.path.join(clean_dir, b) for b in basenames]

    log(f"Building mesh from {Path(source_files[0]).name} (quality {p['mesh_quality']})")
    data0 = containers[0]
    mesh_kwargs: Dict[str, Any] = {"quality": float(p["mesh_quality"])}
    if float(p.get("para_depth", 0.0)) > 0.0:
        mesh_kwargs["paraDepth"] = float(p["para_depth"])
    mesh = pg_ert.ERTManager(data0).createMesh(data=data0, **mesh_kwargs)

    # Pick dense vs. sparse (low-memory) solve. Honor an explicit choice; otherwise
    # auto-enable sparse once the dense Gauss-Newton matrices would get large.
    save_memory = bool(p.get("save_memory", False))
    n_unknowns = int(mesh.cellCount()) * len(files)
    if (engine == "pyhydro" and "save_memory" not in (params or {})
            and not save_memory and n_unknowns > _AUTO_SPARSE_UNKNOWNS):
        save_memory = True
        log(f"Auto-enabling low-memory (sparse) mode: ~{n_unknowns} model unknowns "
            f"({mesh.cellCount()} cells x {len(files)} steps).")

    inv_kwargs = dict(
        lambda_val=float(p["lambda_val"]), alpha=float(p["alpha"]),
        method=str(p["method"]), max_iterations=int(p["max_iterations"]),
        relativeError=float(p["relativeError"]), absoluteUError=float(p.get("absoluteUError", 0.0)),
        inversion_type=str(p["inversion_type"]),
        model_constraints=(float(p["rho_min"]), float(p["rho_max"])),
        save_memory=save_memory,
    )
    lambda_report: Dict[str, Any] = {"enabled": False}

    if use_windowed:
        window_size = int(p["window_size"])
        log(f"Running {engine} windowed {p['inversion_type']} time-lapse "
            f"inversion: {len(files)} steps, "
            f"window={window_size}, lambda={p['lambda_val']}, alpha={p['alpha']}")
        inversion = WindowedTimeLapseERTInversion(
            data_dir=clean_dir, ert_files=basenames, measurement_times=times,
            window_size=window_size, mesh=mesh, engine=engine, log=log,
            **inv_kwargs)
        result = inversion.run(window_parallel=False)
        mode = "windowed"
    else:
        log(f"Running full {p['inversion_type']} time-lapse inversion: {len(files)} steps, "
            f"lambda={p['lambda_val']}, alpha={p['alpha']}")
        inversion = TimeLapseERTInversion(data_files=files, measurement_times=times,
                                          mesh=mesh, **inv_kwargs)
        result = inversion.run()
        mode = "full"
        if bool(p.get("auto_lambda", False)):
            result, lambda_info = _relax_timelapse_lambda(
                inversion, result,
                target_chi2=float(p.get("target_chi2", 1.0)),
                chi2_tolerance=float(p.get("chi2_tolerance", 0.2)),
                max_trials=int(p.get("max_lambda_trials", 4)),
                warm_start=bool(p.get("lambda_warm_start", True)),
                log=log,
            )
            lambda_report = lambda_info

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

    # ADTLERT windowed results expose the actual iteration trace separately from
    # ``all_chi2`` (which contains one final value per window). Prefer that trace
    # so a one-window paper-style run does not look like a one-iteration run.
    chi2_history: List[float] = []
    try:
        raw_history = getattr(result, "iteration_chi2", None)
        if raw_history is None or len(raw_history) == 0:
            raw_history = getattr(result, "all_chi2", None) or []
        for entry in raw_history:
            arr = np.asarray(entry, dtype=float).ravel()
            if arr.size:
                chi2_history.append(float(arr[0]))
    except Exception:  # noqa: BLE001 - convergence history is optional
        chi2_history = []
    final_chi2 = chi2_history[-1] if chi2_history else float("nan")
    n_data_total = int(sum(int(c.size()) for c in containers))

    out = io_utils.ensure_dir(Path(out_dir) / "qt_ert_timelapse")
    figure_paths: List[str] = []
    data_paths: List[str] = []

    # Per-step titles: a parsed date is shown as-is (already unambiguous); a plain
    # sequence reads "Time step N"; any other numeric time reads "t = <value>" so
    # the panel always says what the number means.
    panel_titles = _step_titles(labels, times, n_time)

    # Resistivity-evolution panel. Use the same per-model, logarithmic ERT
    # rendering convention as the interactive Resistivity model view.
    finite = final_models[np.isfinite(final_models) & (final_models > 0.0)]
    if finite.size:
        # One scale across every date, as in the paper figures. Robust limits
        # keep a handful of poorly covered extreme cells from washing out the
        # time-lapse signal everywhere else.
        rho_min = float(np.nanpercentile(finite, 2.0))
        rho_max = float(np.nanpercentile(finite, 98.0))
    else:
        rho_min, rho_max = 1.0, 1000.0
    if rho_max <= rho_min:
        rho_max = rho_min * 1.01
    ncol = min(4, n_time)
    nrow = int(np.ceil(n_time / ncol))
    fig = plt.figure(figsize=(3.6 * ncol, 3.0 * nrow))
    for i in range(n_time):
        ax = fig.add_subplot(nrow, ncol, i + 1)
        show_kw = ert_plot_style.ert_model_plot_kwargs()
        show_kw.update(
            ax=ax,
            label=ert_plot_style.ERT_RESISTIVITY_LABEL,
            cMin=rho_min,
            cMax=rho_max,
        )
        if coverage is not None and coverage.shape[0] > i:
            show_kw["coverage"] = coverage[i]
        try:
            pg.show(res_mesh, final_models[:, i], **show_kw)
        except Exception:  # noqa: BLE001 - retry without coverage
            show_kw.pop("coverage", None)
            ax.clear()
            pg.show(res_mesh, final_models[:, i], **show_kw)
        ax.set_title(panel_titles[i])
    span = f": {labels[0]} → {labels[-1]}" if n_time and any("-" in str(l) for l in labels) else ""
    fig.suptitle(f"Time-lapse resistivity ({mode}, {n_time} time steps){span}", y=1.0)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    panel = out / "timelapse_resistivity.png"
    fig.savefig(panel, dpi=160, bbox_inches="tight"); plt.close(fig)
    figure_paths.append(str(panel))

    # Exports go through a sibling temp file that is swapped into place, so a run
    # that cannot replace the target leaves the previous file intact rather than
    # truncated, and never publishes a half-written array.
    models_path = out / "final_models.npy"
    io_utils.save_npy_atomic(models_path, final_models); data_paths.append(str(models_path))
    coverage_path = None
    if coverage is not None:
        coverage_path = out / "all_coverage.npy"
        io_utils.save_npy_atomic(coverage_path, coverage); data_paths.append(str(coverage_path))
    io_utils.write_csv(
        out / "measurement_times.csv",
        [(i, float(times[i]), labels[i] if i < len(labels) else "",
          Path(source_files[i]).name if i < len(source_files) else "") for i in range(n_time)],
        header=["index", "time", "label", "source_file"])
    data_paths.append(str(out / "measurement_times.csv"))
    mesh_path = out / "timelapse_mesh.bms"
    try:
        res_mesh.save(str(mesh_path)); data_paths.append(str(mesh_path))
    except Exception as exc:  # noqa: BLE001
        mesh_path = None
        log(f"Mesh export skipped: {exc}")
    # Per-step VTKs (one resistivity field each) — a clean ParaView time series.
    vtk_step_paths: List[str] = []
    try:
        steps_dir = io_utils.ensure_dir(out / "vtk_steps")
        for i in range(n_time):
            step_mesh = pg.Mesh(res_mesh)  # copy before the combined fields are added
            step_mesh["resistivity"] = final_models[:, i]
            if coverage is not None and coverage.shape[0] > i:
                step_mesh["coverage"] = np.asarray(coverage[i], dtype=float)
            lbl = _safe_label(labels[i]) if i < len(labels) else f"{i:03d}"
            sp = steps_dir / f"resistivity_t{i:03d}_{lbl}.vtk"
            step_mesh.exportVTK(str(sp))
            vtk_step_paths.append(str(sp)); data_paths.append(str(sp))
    except Exception as exc:  # noqa: BLE001
        log(f"Per-step VTK export skipped: {exc}")
    # Combined VTK: every time step as a separate field on one mesh.
    vtk_combined = ""
    try:
        for i in range(n_time):
            res_mesh[f"resistivity_t{i}"] = final_models[:, i]
        vtk = out / "timelapse_resistivity.vtk"
        res_mesh.exportVTK(str(vtk)); data_paths.append(str(vtk))
        vtk_combined = str(vtk)
    except Exception as exc:  # noqa: BLE001
        log(f"VTK export skipped: {exc}")

    config = build_timelapse_config(source_files, times, p)
    io_utils.write_json(out / "timelapse_config.json", config)

    return {
        "status": "ok",
        "direction": "ert_time_lapse_inversion",
        "mode": mode,
        "engine": engine,
        "engine_requested": requested_engine,
        "backend_version": str(result.meta.get("backend_version", "")),
        "linearized_solver": str(result.meta.get("linearized_solver", "")),
        "sensitivity_profile": str(
            result.meta.get("sensitivity_profile", "")
        ),
        "normal_sensitivity": result.meta.get("normal_sensitivity"),
        "include_robin_boundary_derivative": result.meta.get(
            "include_robin_boundary_derivative"
        ),
        "n_times": int(n_time),
        "mesh_cells": int(res_mesh.cellCount()),
        "inversion_type": str(p["inversion_type"]),
        "instrument": instrument,
        "save_memory": bool(save_memory),
        "chi2": final_chi2,
        "chi2_history": chi2_history,
        "auto_lambda": lambda_report,
        "lambda_used": float(lambda_report.get("lambda_used", p["lambda_val"])),
        "n_data": n_data_total,
        "measurement_times": [float(t) for t in times],
        "time_labels": list(labels),
        "resistivity_range": [rho_min, rho_max],
        "figure_paths": figure_paths,
        "data_paths": data_paths,
        "vtk_combined": vtk_combined,
        "vtk_step_paths": vtk_step_paths,
        "normalized_dir": clean_dir,
        "config_path": str(out / "timelapse_config.json"),
        "output_dir": str(out),
        "model_bundle": {
            "mesh": str(mesh_path) if mesh_path is not None else "",
            "models": str(models_path),
            "coverage": str(coverage_path) if coverage_path is not None else "",
        },
        # In-memory results for interactive per-step display (a pyGIMLi mesh +
        # arrays; not JSON-serializable, so the UI strips these before publishing).
        "mesh": res_mesh,
        "final_models": final_models,
        "coverage": coverage,
        "step_titles": panel_titles,
    }
