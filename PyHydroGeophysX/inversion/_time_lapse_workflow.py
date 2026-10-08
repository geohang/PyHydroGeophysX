"""Time-lapse ERT inversion pipeline for the desktop studio (Qt-free).

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

#: The overview figure draws at most this many time steps. pyGIMLi lays the
#: whole figure out again after every section it draws, so the figure's cost
#: grows with the square of its panel count: a 420-step series spent most of an
#: hour, after the inversion had finished, on one image 315 inches tall that
#: nobody could read. A longer series shows this many steps, evenly spaced from
#: the first to the last; every step is still saved in ``final_models.npy`` and
#: the per-step VTK files, and the studio's step viewer shows any of them.
OVERVIEW_MAX_PANELS = 12

DEFAULT_TL = {
    "lambda_val": 50.0, "alpha": 10.0, "inversion_type": "L2",
    "max_iterations": 15, "relativeError": 0.05, "absoluteUError": 0.0,
    # This dict is passed straight through to TimeLapseERTInversion, so it
    # overrides that class's own default. It has to move with it: the matrix the
    # solver receives is the Gauss-Newton normal matrix, and 'cgls' is a
    # least-squares method, which works on its square. The adtlert branch below
    # forces 'cgls' back, because there the string selects that backend's own
    # GPU CGLS rather than anything in solvers/linear_solvers.py.
    "method": "spd_cholesky", "mesh_quality": 34.0, "rho_min": 1.0, "rho_max": 1.0e4,
    "windowed": False, "window_size": 3, "save_memory": False, "instrument": None,
    "engine": "pyhydro",
    "para_depth": 0.0,
    "para_max_cell_size": 0.0,
    # The rest of the generated mesh's sizing, as build_inversion_mesh takes it;
    # at these values PyGIMLi builds its own default mesh.
    "para_boundary": 2.0, "surface_nodes": 1, "outer_width": 0.0,
    "outer_max_cell_size": 0.0,
    # A mesh built elsewhere to invert on instead of the generated one, and the
    # a-priori resistivity zones drawn on it (see ert_zones): whether the mesh
    # follows their outlines, and whether the smoothness stops at them.
    "mesh_file": "",
    "zones": None,
    "conform_to_zones": False,
    "decouple_zones": False,
    "max_error": None,
    # A reciprocal error model {"m", "b", "floor", ...} to set every reading's
    # error from (ert_io.reciprocal_model_errors); None keeps each file's own.
    "error_model": None,
    # Lambda relaxation. A trial here is a full joint inversion over every time
    # step, so the default budget is smaller than the single-inversion search.
    "auto_lambda": False, "target_chi2": 1.0, "chi2_tolerance": 0.2,
    # The run ends once an iteration lowers chi2 by less than this fraction:
    # the in-house engine (full or windowed) and E4D take it; None leaves each
    # its own default (1 % in-house, 0.5 % for E4D). ADTLERT stops on the size
    # of its model step, and R2 and R3t decide for themselves.
    "plateau_tolerance": None,
    "max_lambda_trials": 4, "lambda_warm_start": True,
    # Distribute the temporal constraint by the interval between surveys, so it
    # penalizes the rate of change rather than the raw difference. Normalized by
    # the median interval: an evenly sampled series is unaffected and alpha keeps
    # its meaning. "uniform" reproduces a run from before this existed.
    "temporal_weighting": "interval",
    "temporal_weight_limit": 10.0,
    # Correct every step to one reference temperature before the sections are
    # compared. None or {"enabled": False} leaves the models as inverted; see
    # PyHydroGeophysX.petrophysics.temperature.DEFAULT_TEMPERATURE_SPEC.
    "temperature_correction": None,
    # How the exported panels are trimmed to the resolved part of the section:
    # "coverage" is pyGIMLi's own per-cell alpha fade, "envelope" is the
    # traditional clean cut along a smooth clipping depth, "none" draws the whole
    # parameter mesh.
    "figure_clip": "coverage",
    "figure_clip_threshold": -2.0,
    # How many time steps the overview figure draws at most; a longer series
    # shows that many, evenly spaced. 0 draws every step.
    "figure_max_panels": OVERVIEW_MAX_PANELS,
}

#: Above this many model unknowns (para cells x time steps) the dense
#: Gauss-Newton matrices get large, so sparse/low-memory mode is auto-enabled
#: unless the caller set ``save_memory`` explicitly.
_AUTO_SPARSE_UNKNOWNS = 15000

#: Roughly how many progress lines the per-step VTK export writes, however
#: long the series: often enough to keep a progress bar moving, few enough not
#: to bury the run's log under 420 lines that say the same thing.
_VTK_PROGRESS_LINES = 20


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


def _coerce_datetime(value: Any):
    """A datetime from a datetime, a date or an ISO-ish string; None otherwise."""
    import datetime as dt

    if isinstance(value, dt.datetime):
        return value
    if isinstance(value, dt.date):
        return dt.datetime(value.year, value.month, value.day)
    if isinstance(value, str) and value.strip():
        try:
            return dt.datetime.fromisoformat(value.strip())
        except ValueError:
            from PyHydroGeophysX.data_processing.survey_timing import parse_timestamp

            found = parse_timestamp(value)
            return found[0] if found else None
    return None


def _resolve_timing(source_files: Sequence[str],
                    measurement_times: Optional[Sequence[float]],
                    time_labels: Optional[Sequence[str]],
                    timestamps: Optional[Sequence[Any]]):
    """Acquisition times for the run, from whichever source has them.

    Order of preference: timestamps handed in by the caller, then timestamps
    recovered from the caller's own labels (the Studio stages the files under
    generated names but records the parsed dates as labels, so this is what keeps
    a bundled run dated), then the filenames themselves.

    A caller that also supplies numeric ``measurement_times`` keeps them: they may
    be in a unit of its own choosing and they are what the temporal regularization
    sees. Only the reporting - the labels, the intervals, the span - comes from the
    timestamps, and the dates the temperature correction runs on; the filenames
    are read for them whether or not numeric times came with the files.
    """
    from dataclasses import replace

    from PyHydroGeophysX.data_processing.survey_timing import (
        SurveyTiming, survey_timing,
    )

    files = [str(f) for f in source_files]
    n = len(files)
    stamps = None
    if timestamps is not None and len(timestamps) == n:
        candidate = [_coerce_datetime(value) for value in timestamps]
        if all(value is not None for value in candidate):
            stamps, source = candidate, "acquisition times supplied with the run"
    if stamps is None and time_labels is not None and len(time_labels) == n:
        candidate = [_coerce_datetime(str(label)) for label in time_labels]
        if all(value is not None for value in candidate) and len(set(candidate)) == n:
            stamps, source = candidate, "the acquisition dates recorded with the files"

    has_times = measurement_times is not None and len(measurement_times) == n
    if stamps is not None:
        origin = min(stamps)
        derived = [(s - origin).total_seconds() / 86400.0 for s in stamps]
        times = [float(t) for t in measurement_times] if has_times else derived
        labels = ([str(lbl) for lbl in time_labels]
                  if time_labels is not None and len(time_labels) == n
                  else [s.strftime("%Y-%m-%d %H:%M") for s in stamps])
        return SurveyTiming(files=files, timestamps=stamps, times=times,
                            labels=labels, source=source, unit="d")

    named = survey_timing(files)
    if not has_times:
        return named
    # Numeric times place each survey for the temporal regularization; they say
    # nothing about when it was measured, and the file names often do. Numeric
    # times used to skip the names, so the same two dated files had a seasonal
    # temperature correction without times and none with them.
    supplied = [float(t) for t in measurement_times]
    if time_labels is not None and len(time_labels) == n:
        labels = [str(lbl) for lbl in time_labels]
    elif named.dated:
        labels = list(named.labels)
    else:
        labels = [f"{t:g}" for t in supplied]
    if named.dated:
        return replace(named, times=supplied, labels=labels)
    return SurveyTiming(files=files, timestamps=[None] * n, times=supplied,
                        labels=labels, source="supplied times", unit="")


def _sensor_positions(data) -> Optional[np.ndarray]:
    """Electrode ``(x, elevation)`` positions, or None when they cannot be read."""
    try:
        positions = np.asarray(data.sensors(), dtype=float)
        return positions[:, :2] if positions.ndim == 2 and positions.shape[1] >= 2 else None
    except Exception:  # noqa: BLE001 - topography will come from the mesh instead
        return None


def _apply_temperature_correction(spec: Any, models: np.ndarray, mesh: Any, data,
                                  times: Sequence[float],
                                  timestamps: Sequence[Any],
                                  log: LogFn) -> Dict[str, Any]:
    """Correct the inverted series to one reference temperature, if asked to.

    Returns a report dict carrying the corrected models under ``"models"`` when it
    succeeded. A correction that was asked for but could not be built is reported
    as ``applied: False`` with the reason, and logged as a warning: a section
    silently left uncorrected looks exactly like a corrected one, so the run has to
    say which it is.
    """
    if not spec or not bool(dict(spec).get("enabled", True)):
        return {"applied": False, "requested": False}
    try:
        from PyHydroGeophysX.core import section_geometry
        from PyHydroGeophysX.petrophysics import temperature as temperature_model

        depths = section_geometry.cell_depths(mesh, sensors=_sensor_positions(data))
        dates = [t for t in timestamps] if timestamps and all(
            t is not None for t in timestamps) else None
        corrected, report = temperature_model.correct_time_lapse_models(
            models, dict(spec), depths, days=list(times), dates=dates)
    except Exception as exc:  # noqa: BLE001 - never lose the inversion over this
        log(f"WARNING: temperature correction was requested but could not be "
            f"applied ({exc}). The sections below are the raw inverted "
            f"resistivity, uncorrected for temperature.")
        return {"applied": False, "requested": True, "error": str(exc)}
    log(f"Temperature correction: {report['note']}")
    report["models"] = corrected
    return report


def _envelope_polygon(mesh, coverage, threshold: float,
                      sensors: Optional[np.ndarray], log: LogFn):
    """Clipping polygon for the panels, or None if one cannot be built."""
    try:
        from PyHydroGeophysX.core.section_geometry import coverage_envelope_polygon

        polygon = coverage_envelope_polygon(
            mesh, coverage, threshold, sensors=sensors)
        if polygon is None:
            log(f"Section clipping skipped: the {threshold:g} coverage cut keeps "
                f"either all of the section or none of it.")
        return polygon
    except Exception as exc:  # noqa: BLE001 - a clip is cosmetic, never fatal
        log(f"Section clipping skipped: {exc}")
        return None


def _clip_axes(ax, polygon, log: LogFn) -> bool:
    """Clip one drawn panel to ``polygon``; never fatal if it cannot."""
    try:
        from PyHydroGeophysX.visualization.section_clip import clip_axes_to_polygon

        return clip_axes_to_polygon(ax, polygon, outline=True, tighten=True)
    except Exception as exc:  # noqa: BLE001 - a clip is cosmetic
        log(f"Section clipping skipped: {exc}")
        return False


def _step_titles(labels: Sequence[str], times: Sequence[float], n_time: int,
                 time_unit: str = "") -> List[str]:
    """Clear per-step titles so the panel always says what the number means:
    a parsed date stays as the date; a plain 1..n sequence becomes "Time step N";
    any other numeric time becomes "t = <value>", carrying ``time_unit`` when the
    caller knows it — a bare number on a section leaves the reader guessing
    whether it counts hours, days or surveys."""
    labels = list(labels or [])
    unit = f" {time_unit.strip()}" if str(time_unit).strip() else ""
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
            titles.append(f"t = {t:g}{unit}" if isinstance(t, (int, float))
                          else f"t = {lbl}{unit}")
    return titles


def _safe_label(text: str) -> str:
    """Filename-safe version of a time label."""
    out = "".join(c if c.isalnum() or c in "-_." else "_" for c in str(text))
    return out.strip("_") or "step"


def overview_steps(n_time: int, max_panels: int = OVERVIEW_MAX_PANELS) -> List[int]:
    """The time steps the overview figure draws, as 0-based indices in order.

    Every step of a series of up to ``max_panels``; otherwise ``max_panels``
    steps spread evenly over it, the first and the last always among them.
    ``max_panels`` of 0 or less draws every step.

    >>> overview_steps(5)
    [0, 1, 2, 3, 4]
    >>> overview_steps(420, 4)
    [0, 140, 279, 419]
    """
    n_time, limit = int(n_time), int(max_panels or 0)
    if limit <= 0 or n_time <= limit:
        return list(range(max(0, n_time)))
    # At least two, so a capped series always shows where it starts and ends.
    picks = np.round(np.linspace(0, n_time - 1, max(2, limit))).astype(int)
    return sorted({int(i) for i in picks})


def _progress(log: LogFn, current: int, total: int, label: str) -> None:
    """Log one ``[progress current/total] label`` line.

    The studio reads these off the workflow process's output into its progress
    bar and status bar (``qt_apps.workers.ProcessWorkflowWorker``); anywhere
    else they are ordinary, readable log lines.
    """
    log(f"[progress {int(current)}/{int(total)}] {label}")


class _JointIterationProgress:
    """Report each iteration of the full (all steps at once) inversion.

    The windowed inversion reports every window it finishes; the full one
    solves every step together and said nothing a progress bar could read, so
    the studio showed a bar that only knew the run was busy. This turns the
    inversion's own iteration events into ``[progress i/n]`` lines, ``n`` being
    the most iterations the run can take; one that converges early stops short.
    """

    def __init__(self, n_steps: int, log: LogFn) -> None:
        self.n_steps = int(n_steps)
        self.log = log

    def __call__(self, event: Dict[str, Any]) -> None:
        if event.get("event") != "timelapse_iteration_done":
            return
        iteration = int(event.get("iteration", 0))
        maximum = max(1, int(event.get("max_iterations", 1)))
        irls, irls_total = int(event.get("irls_iteration", 1)), int(event.get("irls_iterations", 1))
        current = (irls - 1) * maximum + iteration
        total = max(current, irls_total * maximum)
        reweight = f", reweighting pass {irls}/{irls_total}" if irls_total > 1 else ""
        self.log(f"[progress {current}/{total}] Inverting all {self.n_steps} steps together"
                 f"{reweight}, iteration {iteration}/{maximum}: "
                 f"chi2 {float(event.get('chi2', float('nan'))):.3f}")


def _log_ticks(lo: float, hi: float) -> List[float]:
    """Round values to label a logarithmic scale from ``lo`` to ``hi``.

    Decades when the range spans three or more; otherwise 1-2-5 steps, or
    finer ones for a range under a decade, so a bar always carries a few
    readable labels rather than the ends' arbitrary values.

    >>> _log_ticks(385.0, 2685.0)
    [500.0, 1000.0, 2000.0]
    >>> _log_ticks(1.0, 10000.0)
    [1.0, 10.0, 100.0, 1000.0, 10000.0]
    """
    if not (hi > lo > 0.0):
        return [lo, hi] if lo > 0.0 else []
    first, last = int(np.floor(np.log10(lo))), int(np.ceil(np.log10(hi)))
    for subs in ((1.0,), (1.0, 2.0, 5.0), (1.0, 1.5, 2.0, 3.0, 5.0, 7.0)):
        ticks = [float(f"{m * 10.0 ** k:.6g}") for k in range(first, last + 1)
                 for m in subs if lo <= m * 10.0 ** k <= hi]
        if len(ticks) >= 3:
            return ticks
    return [float(f"{lo:.2g}"), float(f"{hi:.2g}")]


def _draw_overview(path: Path, mesh: Any, models: np.ndarray,
                   coverage: Optional[np.ndarray], *, steps: Sequence[int],
                   titles: Sequence[str], rho_range: Sequence[float],
                   clip_mode: str, clip_polygon: Any, title: str,
                   log: LogFn) -> None:
    """Draw the resistivity sections of ``steps`` in rows of four and save them.

    Every section uses the same per-model, logarithmic rendering convention as
    the interactive Resistivity model view, on one colour scale ``rho_range``.
    """
    import matplotlib.pyplot as plt
    from matplotlib import ticker
    import pygimli as pg

    n_panels = len(steps)
    ncol = min(4, n_panels)
    nrow = int(np.ceil(n_panels / ncol))
    _progress(log, 0, n_panels, "Saving results: drawing the overview figure")
    # The shared colour bar below gets its own 0.85 inch, added to the
    # figure rather than taken from the panels.
    height = 3.0 * nrow + 0.85
    fig = plt.figure(figsize=(3.6 * ncol, height))
    mappable = None
    try:
        for slot, i in enumerate(steps):
            ax = fig.add_subplot(nrow, ncol, slot + 1)
            show_kw = ert_plot_style.ert_model_plot_kwargs()
            # One colour bar for the whole figure, drawn below: every panel is
            # on the same scale, and a bar under each 3.6-inch panel pressed
            # its five labels into one another ("385626101716522685").
            show_kw.update(
                ax=ax,
                cMin=float(rho_range[0]),
                cMax=float(rho_range[1]),
                colorBar=False,
            )
            step_coverage = (coverage[i] if coverage is not None and coverage.shape[0] > i
                             else None)
            if step_coverage is not None and clip_mode == "coverage":
                show_kw["coverage"] = step_coverage
            try:
                pg.show(mesh, models[:, i], **show_kw)
            except Exception:  # noqa: BLE001 - retry without coverage
                show_kw.pop("coverage", None)
                ax.clear()
                pg.show(mesh, models[:, i], **show_kw)
            if mappable is None and ax.collections:
                mappable = ax.collections[0]   # the model's cells: cmap and LogNorm
            if clip_polygon is not None:
                _clip_axes(ax, clip_polygon, log)
            ax.set_title(titles[i])
            _progress(log, slot + 1, n_panels,
                      f"Saving results: overview figure, panel {slot + 1}/{n_panels}")
        fig.suptitle(title, y=1.0)
        fig.tight_layout(rect=(0.0, 0.85 / height, 1.0, 0.97))
        if mappable is not None:
            cax = fig.add_axes((0.3, 0.32 / height, 0.4, 0.13 / height))
            bar = fig.colorbar(mappable, cax=cax, orientation="horizontal")
            bar.set_label(ert_plot_style.ERT_RESISTIVITY_LABEL)
            bar.set_ticks(_log_ticks(float(rho_range[0]), float(rho_range[1])))
            bar.ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
            bar.ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        fig.savefig(path, dpi=160, bbox_inches="tight")
    finally:
        plt.close(fig)


class _VTKTemplate:
    """pyGIMLi's VTK file of one time step, with its field values left open.

    Every per-step file holds the same mesh - points, cells, markers - and only
    its resistivity and coverage differ, yet ``exportVTK`` formats the whole
    geometry again for each one: on a 60 000-cell mesh three quarters of a
    file's 0.3 s, repeated for every step. The template is cut from the first
    file pyGIMLi writes and fills in later steps' values the way pyGIMLi
    formats them. It is used only after it reproduced that first file byte for
    byte, and a step it cannot render the same way (a NaN, say, whose spelling
    is the C++ library's own) is left to pyGIMLi.
    """

    def __init__(self, segments: List[Any]) -> None:
        #: Runs of the file's lines as bytes, and ``(field, line ending)`` where
        #: a field's values go.
        self.segments = segments
        self.fields = {item[0] for item in segments if isinstance(item, tuple)}

    @staticmethod
    def values_line(values: Any) -> Optional[bytes]:
        """One field's values as pyGIMLi writes them; None if they are not all finite."""
        array = np.asarray(values, dtype=float).ravel()
        if not np.all(np.isfinite(array)):
            return None
        return " ".join(["%.14g" % value for value in array.tolist()]).encode("ascii") + b" "

    @classmethod
    def from_reference(cls, reference: bytes,
                       fields: Dict[str, Any]) -> Optional["_VTKTemplate"]:
        """The template of ``reference``, a file pyGIMLi wrote holding ``fields``;
        None when it does not reproduce that file exactly."""
        lines = reference.split(b"\n")
        holes: Dict[int, Any] = {}
        for name in fields:
            header = f"SCALARS {name} double 1".encode("ascii")
            found = [k for k, line in enumerate(lines) if line.rstrip(b"\r") == header]
            if len(found) != 1 or found[0] + 2 >= len(lines):
                return None
            k = found[0] + 2
            holes[k] = (name, b"\r" if lines[k].endswith(b"\r") else b"")
        segments: List[Any] = []
        run: List[bytes] = []
        for k, line in enumerate(lines):
            if k in holes:
                if run:
                    segments.append(b"\n".join(run))
                    run = []
                segments.append(holes[k])
            else:
                run.append(line)
        if run:
            segments.append(b"\n".join(run))
        template = cls(segments)
        return template if template.render(fields) == reference else None

    def render(self, fields: Dict[str, Any]) -> Optional[bytes]:
        """The file for ``fields``; None when it has to come from pyGIMLi instead."""
        if set(fields) != self.fields:
            return None
        pieces: List[bytes] = []
        for item in self.segments:
            if isinstance(item, tuple):
                line = self.values_line(fields[item[0]])
                if line is None:
                    return None
                pieces.append(line + item[1])
            else:
                pieces.append(item)
        return b"\n".join(pieces)


def _export_vtks(out: Path, mesh: Any, models: np.ndarray,
                 coverage: Optional[np.ndarray], labels: Sequence[str],
                 log: LogFn):
    """Write one VTK per time step and one holding every step.

    The per-step files carry one resistivity field each (and that step's
    coverage) - a clean ParaView time series; the combined file carries every
    step as a field of its own on one mesh. Returns ``(step_paths,
    combined_path)``; a file that cannot be written is logged and left out,
    never fatal.
    """
    import pygimli as pg

    n_time = int(models.shape[1])
    every = max(1, -(-n_time // _VTK_PROGRESS_LINES))   # ceil
    step_paths: List[str] = []
    try:
        steps_dir = io_utils.ensure_dir(out / "vtk_steps")
        _progress(log, 0, n_time, f"Saving results: per-step VTK 0/{n_time}")
        template = None
        for i in range(n_time):
            fields = {"resistivity": models[:, i]}
            if coverage is not None and coverage.shape[0] > i:
                fields["coverage"] = np.asarray(coverage[i], dtype=float)
            lbl = _safe_label(labels[i]) if i < len(labels) else f"{i:03d}"
            sp = steps_dir / f"resistivity_t{i:03d}_{lbl}.vtk"
            text = template.render(fields) if template is not None else None
            if text is not None:
                sp.write_bytes(text)
            else:
                step_mesh = pg.Mesh(mesh)
                for name, values in fields.items():
                    step_mesh[name] = values
                step_mesh.exportVTK(str(sp))
                if i == 0:
                    template = _VTKTemplate.from_reference(sp.read_bytes(), fields)
            step_paths.append(str(sp))
            if (i + 1) % every == 0 or i + 1 == n_time:
                _progress(log, i + 1, n_time,
                          f"Saving results: per-step VTK {i + 1}/{n_time}")
    except Exception as exc:  # noqa: BLE001
        log(f"Per-step VTK export skipped: {exc}")
    # Every time step as a separate field, on a copy: the mesh handed back to
    # the caller stays the bare parameter mesh rather than carrying one field
    # per step for the rest of its life.
    combined = ""
    try:
        _progress(log, 0, 1, f"Saving results: combined VTK ({n_time} steps in one file)")
        combined_mesh = pg.Mesh(mesh)
        for i in range(n_time):
            combined_mesh[f"resistivity_t{i}"] = models[:, i]
        vtk = out / "timelapse_resistivity.vtk"
        combined_mesh.exportVTK(str(vtk))
        combined = str(vtk)
    except Exception as exc:  # noqa: BLE001
        log(f"VTK export skipped: {exc}")
    return step_paths, combined


#: The range of relative data errors each time-lapse engine accepts from the
#: ``err`` column; a value outside it is moved to the nearer limit.
#: TimeLapseERTInversion clips to 1-50 % (time_lapse.py), the windowed ADTLERT
#: backend raises anything below 1 % (windowed.py), and E4D, R2 and R3t take
#: the column as it is (e4d._survey_data).
_ENGINE_ERROR_RANGE = {"pyhydro": (0.01, 0.50), "adtlert": (0.01, None)}


def _data_error_report(engine: str, containers, error_model: Optional[Dict[str, Any]],
                       log: LogFn = _noop) -> Dict[str, Any]:
    """What the run hands the engine as data errors, and what the engine
    makes of them.

    Every time-lapse engine reads each survey's ``err`` column - set from the
    reciprocal error model when there is one - but not all of them take every
    value: the limits in ``_ENGINE_ERROR_RANGE`` move the errors outside
    them, which is said here rather than left to look as if the model's
    errors were used as they are.
    """
    errors = [np.asarray(c["err"], dtype=float) for c in containers if c.haveData("err")]
    if len(errors) != len(containers) or not errors:
        return {"source": "estimate (the surveys carry no error column)"}
    values = np.concatenate(errors)
    # ADTLERT inverts the union of every survey's readings, and a reading a
    # survey lacks is filled in at 100 % error (align_timelapse_abmn); those
    # placeholders are counted apart, not as errors the data were given.
    filled = int((values == 1.0).sum()) if engine == "adtlert" else 0
    if filled and filled < values.size:
        values = values[values != 1.0]
    low, high = _ENGINE_ERROR_RANGE.get(engine, (None, None))
    raised = int((values < low).sum()) if low is not None else 0
    lowered = int((values > high).sum()) if high is not None else 0
    report = {
        "source": ("reciprocal error model" if error_model else "each survey's err column"),
        "median": float(np.median(values)), "min": float(values.min()),
        "max": float(values.max()), "readings": int(values.size),
        "engine_range": [low, high], "raised_to_engine_minimum": raised,
        "lowered_to_engine_maximum": lowered, "filled_at_100_percent": filled,
    }
    if error_model:
        report["model"] = {key: error_model[key] for key in (
            "m", "b", "floor", "r2", "r2_raw", "pairs", "surveys", "fitted_over")
            if key in error_model}
    if error_model or raised or lowered:
        log(f"Data errors: {report['source']}, median {100.0 * report['median']:.3g} % "
            f"(from {100.0 * report['min']:.3g} to {100.0 * report['max']:.3g} %) over "
            f"{report['readings']} readings"
            + (f"; besides them, {filled} reading(s) a survey lacks were filled in at "
               "100 % error to align the surveys" if filled else "") + ".")
        if raised or lowered:
            limits = " and ".join(text for text in (
                f"below {100.0 * low:g} %" if low is not None else "",
                f"above {100.0 * high:g} %" if high is not None else "") if text)
            log(f"  Note: the {engine} engine takes no error {limits}: {raised} "
                f"reading(s) were raised and {lowered} lowered to that limit, so "
                "those readings are fitted to the limit, not to their own error.")
        elif error_model:
            log(f"  The {engine} engine uses these errors as they are.")
    return report


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
            "para_max_cell_size", "para_boundary", "surface_nodes", "outer_width",
            "outer_max_cell_size", "mesh_file", "zones", "conform_to_zones",
            "decouple_zones",
            "max_error", "error_model", "temporal_weighting", "temporal_weight_limit",
            "plateau_tolerance")},
        # Post-processing that changes what the sections show has to travel with
        # the configuration, or a re-run reproduces different pictures.
        "temperature_correction": p.get("temperature_correction"),
        "figure_clip": p.get("figure_clip"),
        "figure_clip_threshold": p.get("figure_clip_threshold"),
        "figure_max_panels": p.get("figure_max_panels"),
    }


def run_timelapse_ert(
    data_files: Sequence[str],
    measurement_times: Sequence[float],
    params: Dict[str, Any],
    out_dir: str,
    log: LogFn = _noop,
    time_labels: Optional[Sequence[str]] = None,
    time_unit: str = "",
    timestamps: Optional[Sequence[Any]] = None,
    electrode_file: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a full temporal-regularized time-lapse ERT inversion.

    ``electrode_file`` places the electrodes of every survey, rows matched to
    each file's electrodes in order, as it does for a single survey; a survey
    whose electrode count differs from the file's is refused. Without it each
    file's own electrode table is used.

    ``time_labels`` is what the per-step panel titles read — acquisition dates,
    typically. Pass them whenever ``measurement_times`` is given: the times alone
    are bare numbers, and a caller that staged its files under generated names
    (the Studio writes them as ``step_0000.*``) is the only place the original
    dates still exist. ``time_unit`` labels the numbers when there is nothing
    better to show, e.g. ``"d"``.

    ``timestamps`` are the absolute acquisition times (datetimes or ISO strings).
    They are what lets the run report the real duration between surveys instead of
    an elapsed-day number, and they are required by the seasonal mode of the
    temperature correction, which needs to know where in the year each survey sits.
    When they are not given they are read back off the source filenames.

    Raises ``BackendUnavailable`` if pygimli / the inversion cannot be imported,
    and propagates other exceptions so the caller can fall back to config export.
    """
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        plt.ioff()
        import pygimli  # noqa: F401 - a missing backend is reported here, up front
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
    if engine in ("e4d", "r2", "r3t"):
        # E4D runs the series itself (its ERT4 mode), one survey after another;
        # R2 and R3t invert each survey against the baseline.
        use_windowed = False
    if engine not in ("pyhydro", "adtlert", "e4d", "r2", "r3t"):
        raise ValueError(
            "Time-lapse ERT engine must be 'pyhydro', windowed 'adtlert', 'e4d', 'r2' "
            "or 'r3t'"
        )

    # Acquisition times. Absolute timestamps are what make the sequence readable —
    # they turn "step 3" into a date and an interval — so they are carried through
    # when the caller has them and read back off the filenames when it does not.
    timing = _resolve_timing(source_files, measurement_times, time_labels, timestamps)
    times, labels = list(timing.times), list(timing.labels)
    if timing.dated:
        time_unit = time_unit or "d"
    log("Survey timing: " + timing.summary())
    if not timing.dated:
        log("  No timestamps were found, so every step is one unit apart and the "
            "panels are headed by an index. Name the files with their acquisition "
            "time (e.g. site_2024-06-12_1430.dat) or enter the times in the "
            "interface to get real dates and intervals.")

    # Load every file through the robust device-aware loader and re-write it as a
    # clean pygimli file. Raw ``ert.load`` cannot parse index-prefixed / header-less
    # formats (E4D etc.) and would drop all data + topography; normalizing first is
    # what makes the time-lapse inversion actually work on those files.
    instrument = p.get("instrument")
    log(f"Preparing {len(source_files)} ERT files"
        + (f" (instrument: {instrument})" if instrument else " (auto-detect)")
        + (f", electrode positions from {Path(electrode_file).name}" if electrode_file else "")
        + " …")
    clean_dir, basenames, containers = ert_load.normalize_for_timelapse(
        source_files, instrument, out_dir, log=log,
        max_error=p.get("max_error"), engine=engine,
        electrode_file=str(electrode_file) if electrode_file else None,
        error_model=p.get("error_model") or None)
    files = [os.path.join(clean_dir, b) for b in basenames]
    data_error = _data_error_report(engine, containers, p.get("error_model") or None,
                                    log=log)

    # The same mesh builder as the single-survey run and the ERT page's mesh
    # preview, so the preview of the first survey is the mesh inverted here.
    from .ert_inversion import _zone_notes
    from .ert_mesh import build_inversion_mesh, mark_zone_interfaces
    from .ert_zones import normalize_zones, zone_prior

    data0 = containers[0]
    p["mesh_file"] = str(p.get("mesh_file") or "")
    zones = normalize_zones(p.get("zones"))
    p["zones"] = zones or None
    if p["mesh_file"]:
        log(f"Loading the inversion mesh {Path(p['mesh_file']).name}")
    else:
        log(f"Building mesh from {Path(source_files[0]).name} (quality {p['mesh_quality']})")
    mesh_report: Dict[str, Any] = {}
    mesh = build_inversion_mesh(
        data0, mesh_quality=float(p["mesh_quality"]),
        para_depth=float(p.get("para_depth", 0.0) or 0.0),
        para_max_cell_size=float(p.get("para_max_cell_size", 0.0) or 0.0),
        para_boundary=float(p.get("para_boundary", 2.0) or 2.0),
        surface_nodes=int(p.get("surface_nodes", 1) or 1),
        outer_width=float(p.get("outer_width", 0.0) or 0.0),
        outer_max_cell_size=float(p.get("outer_max_cell_size", 0.0) or 0.0),
        conform_zones=zones if p.get("conform_to_zones") else None,
        mesh_file=p["mesh_file"], log=log, report=mesh_report)
    # The smoothness stops at the zone outlines for every engine: the in-house
    # one and PyGIMLi read the marked edges, ADTLERT turns them into units.
    decoupled_edges = 0
    if p.get("decouple_zones") and zones:
        mesh, decoupled_edges = mark_zone_interfaces(mesh, zones)
        log(f"  smoothness dropped across {decoupled_edges} cell edges along the "
            "zone outlines" if decoupled_edges else
            "  (no cell edge lies on a zone outline, so the smoothness is unchanged)")

    # A-priori zones. The PyHydro engine starts every survey from them and holds
    # the fixed ones; the ADTLERT backend builds its own start model and takes
    # none, so a zone list it would silently drop is reported instead.
    zone_report: List[Dict[str, Any]] = []
    zones_not_applied: List[str] = []
    if zones and engine == "adtlert":
        log(f"Note: the ADTLERT time-lapse backend does not take a-priori zone "
            f"values, so the {len(zones)} zone(s) defined set neither its start "
            "nor its reference model"
            + (" (the smoothness still stops at their outlines)" if decoupled_edges
               else "")
            + ". Use the PyHydro engine to invert with them.")
        zones_not_applied = [zone["name"] for zone in zones]
        zones = []
    elif zones:
        fop = pg_ert.ERTModelling()
        fop.setData(data0)
        fop.setMesh(mesh)
        prior = zone_prior(fop.paraDomain, zones,
                           bounds=(float(p["rho_min"]), float(p["rho_max"])))
        for line in _zone_notes(engine, prior, engine in ("r2", "r3t")):
            log(line)
        zone_report = [dict(entry) for entry in prior.report]

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
        temporal_weighting=str(p.get("temporal_weighting", "interval")),
        temporal_weight_limit=p.get("temporal_weight_limit", 10.0),
    )
    if zones:
        inv_kwargs["zones"] = zones
    # The in-house solver's plateau, TimeLapseERTInversion's convergence
    # tolerance. Not handed on to ADTLERT, which reads that name as the size of
    # a model step below which it stops - a different measure altogether.
    if engine == "pyhydro" and p.get("plateau_tolerance"):
        inv_kwargs["convergence_tolerance"] = float(p["plateau_tolerance"])
    lambda_report: Dict[str, Any] = {"enabled": False}
    # The GPU backend builds its own temporal operator and does not take these,
    # so say that rather than let the setting look as if it applied.
    if engine == "adtlert" and str(p.get("temporal_weighting", "interval")) == "interval":
        log("Note: the ADTLERT backend applies its own uniform temporal "
            "constraint; the interval weighting is used by the PyHydro engine "
            "only.")

    if engine == "e4d":
        from .e4d import invert_e4d_time_lapse

        log(f"Running E4D time-lapse inversion (ERT4): {len(files)} steps, "
            f"lambda (E4D beta) = {p['lambda_val']}")
        ignored = [name for name, used in (
            ("alpha", float(p.get("alpha", 0.0)) > 0),
            ("auto-lambda", bool(p.get("auto_lambda", False))),
            ("iteration limit", True)) if used]
        log("  Note: E4D runs each survey until its misfit stops falling or meets the "
            f"target, from the solution before it; {', '.join(ignored)} "
            f"{'do' if len(ignored) > 1 else 'does'} not apply to it.")
        settings = dict(p.get("e4d") or {})
        settings.setdefault("workdir", str(Path(out_dir) / "e4d_timelapse"))
        result = invert_e4d_time_lapse(
            containers, mesh, lam=float(p["lambda_val"]),
            plateau_tolerance=float(p.get("plateau_tolerance", 0.005) or 0.005),
            target_chi2=float(p.get("target_chi2", 1.0)),
            model_constraints=(float(p["rho_min"]), float(p["rho_max"])),
            zones=zones or None, options=settings,
            outer_width=float(p.get("outer_width", 0.0) or 0.0),
            relative_error=float(p["relativeError"]), log=log)
        mode = "e4d"
    elif engine in ("r2", "r3t"):
        from .r2 import ENGINES as _R2_PROGRAMS, invert_r2_time_lapse

        program = _R2_PROGRAMS[engine]
        log(f"Running {program} time-lapse inversion (difference inversion against the "
            f"first survey): {len(files)} steps")
        ignored = [name for name, used in (
            ("lambda", True), ("alpha", float(p.get("alpha", 0.0)) > 0),
            ("auto-lambda", bool(p.get("auto_lambda", False)))) if used]
        log(f"  Note: {program} chooses its own smoothing weight at every iteration, and "
            f"every later survey starts from the baseline model; {', '.join(ignored)} "
            f"{'do' if len(ignored) > 1 else 'does'} not apply to it.")
        settings = dict(p.get("r2") or {})
        settings.setdefault("workdir", str(Path(out_dir) / f"{engine}_timelapse"))
        result = invert_r2_time_lapse(
            containers, mesh, program=engine, max_iterations=int(p["max_iterations"]),
            target_chi2=float(p.get("target_chi2", 1.0)),
            model_constraints=(float(p["rho_min"]), float(p["rho_max"])),
            zones=zones or None, options=settings,
            outer_width=float(p.get("outer_width", 0.0) or 0.0),
            relative_error=float(p["relativeError"]), log=log)
        mode = engine
    elif use_windowed:
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
        inversion = TimeLapseERTInversion(
            data_files=files, measurement_times=times, mesh=mesh,
            progress_callback=_JointIterationProgress(len(files), log), **inv_kwargs)
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

    # What the solver did with the temporal constraint. Two runs with the same
    # alpha are not the same inversion if one weighted by the interval, so the
    # run says which it was.
    temporal_report = dict(result.meta.get("temporal_weighting") or {})
    if temporal_report.get("note"):
        log(temporal_report["note"])

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

    # Temperature correction. Resistivity falls about 2 % per degC, which over a
    # season is the same size as the change moisture produces, so an uncorrected
    # series shows the ground "drying" as it cools. Applied here, after the
    # inversion, so the corrected models are what every panel, change plot and
    # export below is built from; the raw models are kept beside them.
    raw_models = final_models
    temperature_report = _apply_temperature_correction(
        p.get("temperature_correction"), final_models, res_mesh, data0,
        times=times, timestamps=timing.timestamps, log=log)
    if temperature_report.get("applied"):
        final_models = np.asarray(temperature_report.pop("models"), dtype=float)
        raw_path = out / "final_models_uncorrected.npy"
        io_utils.save_npy_atomic(raw_path, raw_models)
        data_paths.append(str(raw_path))
        temperature_report["uncorrected_models"] = str(raw_path)
    else:
        temperature_report.pop("models", None)

    # Per-step titles: a parsed date is shown as-is (already unambiguous); a plain
    # sequence reads "Time step N"; any other numeric time reads "t = <value>" so
    # the panel always says what the number means.
    panel_titles = _step_titles(labels, times, n_time, time_unit)

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
    # How the panels are trimmed. "envelope" is the traditional clean cut: the
    # drawing is clipped to a smooth clipping depth instead of being masked cell by
    # cell, so the edge follows how deep the survey sees rather than where the mesh
    # put a triangle.
    clip_mode = str(p.get("figure_clip", "coverage") or "none").strip().lower()
    clip_cut = float(p.get("figure_clip_threshold", -2.0))
    sensors = _sensor_positions(data0)
    # One envelope for the whole sequence, from the mean coverage. Per-step
    # envelopes would frame each panel slightly differently, and a series whose
    # panels are not on the same axes cannot be compared by eye - which is the
    # only reason to draw them side by side.
    clip_polygon = None
    if coverage is not None and clip_mode == "envelope":
        clip_polygon = _envelope_polygon(
            res_mesh, np.nanmean(coverage, axis=0), clip_cut, sensors, log)
    # A long series is drawn as a sample of its steps (OVERVIEW_MAX_PANELS),
    # and the title says so, so the figure is never mistaken for the whole run.
    max_panels = int(p.get("figure_max_panels", OVERVIEW_MAX_PANELS) or 0)
    shown_steps = overview_steps(n_time, max_panels)
    span = f": {labels[0]} → {labels[-1]}" if n_time and any("-" in str(l) for l in labels) else ""
    corrected = (f", corrected to "
                 f"{temperature_report.get('reference_temperature_C', 25.0):g} °C"
                 if temperature_report.get("applied") else "")
    counted = (f"{n_time} time steps" if len(shown_steps) == n_time
               else f"{len(shown_steps)} of {n_time} time steps, evenly spaced")
    panel = out / "timelapse_resistivity.png"
    _draw_overview(panel, res_mesh, final_models, coverage, steps=shown_steps,
                   titles=panel_titles, rho_range=(rho_min, rho_max),
                   clip_mode=clip_mode, clip_polygon=clip_polygon,
                   title=f"Time-lapse resistivity ({mode}, {counted}){corrected}{span}",
                   log=log)
    figure_paths.append(str(panel))
    if len(shown_steps) < n_time:
        log(f"The overview figure shows {len(shown_steps)} of the {n_time} time steps, "
            f"evenly spaced from the first to the last. Every step is saved in "
            f"final_models.npy and the per-step VTK files.")

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
        # Name the unit in the header rather than leaving a column of bare
        # numbers that only the code knows how to read.
        header=["index", f"time[{time_unit}]" if time_unit else "time",
                "label", "source_file"])
    data_paths.append(str(out / "measurement_times.csv"))
    # The acquisition log: absolute timestamp, elapsed time and the gap to the
    # previous survey, in a unit a reader recognises. This is the table that says
    # whether "step 3 to step 4" was an hour or a month.
    timing_rows = timing.rows()
    if timing_rows:
        io_utils.write_csv(out / "survey_times.csv", timing_rows,
                           header=timing.csv_header())
        data_paths.append(str(out / "survey_times.csv"))
    mesh_path = out / "timelapse_mesh.bms"
    try:
        res_mesh.save(str(mesh_path)); data_paths.append(str(mesh_path))
    except Exception as exc:  # noqa: BLE001
        mesh_path = None
        log(f"Mesh export skipped: {exc}")
    # Per-step VTKs (a clean ParaView time series) and the combined one.
    vtk_step_paths, vtk_combined = _export_vtks(
        out, res_mesh, final_models, coverage, labels, log)
    data_paths.extend(vtk_step_paths)
    if vtk_combined:
        data_paths.append(vtk_combined)

    config = build_timelapse_config(source_files, times, p)
    config["electrode_file"] = str(electrode_file) if electrode_file else None
    # Which steps the overview figure drew, 0-based; the setting that chose
    # them is in the configuration's figure_max_panels.
    config["overview_steps"] = list(shown_steps)
    io_utils.write_json(out / "timelapse_config.json", config)
    # The workflow layer reads every file once more to record its checksum,
    # which for a long series on a large mesh is gigabytes; say so meanwhile.
    _progress(log, 1, 1, "Saving results: recording the files written")

    return {
        "status": "ok",
        "direction": "ert_time_lapse_inversion",
        "mode": mode,
        "engine": engine,
        "engine_requested": requested_engine,
        # E4D's own run folder: its log, every step's model, the 3-D models.
        "e4d": dict(result.meta.get("e4d") or {}),
        # R2's or R3t's run folders (baseline, then the difference inversion),
        # and the smoothing weight each survey settled on.
        "r2": dict(result.meta.get("r2") or {}),
        "r3t": dict(result.meta.get("r3t") or {}),
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
        "mesh_file": p["mesh_file"],
        # The a-priori zones as applied (cells covered, values after clipping);
        # empty when none were defined or the engine could not take them.
        "zones": zone_report,
        "zones_fixed_held": bool(any(z["fixed"] and z["cells"] for z in zone_report)),
        "zones_not_applied": zones_not_applied,
        "mesh_zone_outline_edges": int(mesh_report.get("zone_outline_edges", 0)),
        "zone_edges_decoupled": int(decoupled_edges),
        "inversion_type": str(p["inversion_type"]),
        "instrument": instrument,
        # Where the electrode positions came from, when not each file's header.
        "electrode_file": Path(electrode_file).name if electrode_file else "",
        "save_memory": bool(save_memory),
        # The data errors the engine was handed, and what it did with them.
        "data_error": data_error,
        "chi2": final_chi2,
        "chi2_history": chi2_history,
        "auto_lambda": lambda_report,
        "lambda_used": float(lambda_report.get("lambda_used", p["lambda_val"])),
        "n_data": n_data_total,
        "measurement_times": [float(t) for t in times],
        "time_labels": list(labels),
        "time_unit": str(time_unit),
        # Absolute acquisition times and the gaps between them, so the report can
        # state the monitoring interval instead of a bare step count.
        "survey_timing": timing.to_dict(),
        "temporal_weighting": temporal_report,
        "temperature_correction": temperature_report,
        "figure_clip": clip_mode,
        # The steps the overview figure drew, 0-based: all of them unless the
        # series was longer than figure_max_panels.
        "overview_steps": list(shown_steps),
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
