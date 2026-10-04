"""How well a seismic, TDEM or MT model explains its data, judged as the ERT model is.

An ERT run checks its inversion before anything is built on it
(``InversionEvaluationAgent``): a quality score out of 100 from the data fit,
the model and the convergence, with recommendations, and a retry when the fit
is off. The other methods went straight from their inversion to the report.
This module gives them the same step:

- **Seismic refraction.** The fit of the picks (chi-squared against the target
  of one, the misfit in milliseconds, picks far outside their error), the
  share of the model the rays cover, cells pinned at a velocity bound, and the
  convergence of the inversion. The workflow retries with lambda changed when
  the fit is off, as ERT's evaluation does.
- **TDEM.** For a survey, the chi-squared of every sounding and the share
  fitted within twice their errors, soundings that failed or resolved nothing
  above their depth of investigation, and resistivities outside the physical
  range. For one sounding, its chi-squared and the decay against the model's.
- **MT.** The RMS of every site against Occam's target of one, the static
  shift each needed, and resistivities outside the physical range.

The data-fit rule is the ERT evaluation's own (:func:`chi2_fit_score`), so a
score means the same for every method. The weights and thresholds are
heuristics, and the report calls the score that. Each evaluation draws its fit
as a figure.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ._figstyle import detached_figure

#: Chi-squared the inversions aim for, and the range accepted around it.
CHI2_TARGET = 1.0
CHI2_ACCEPTABLE = (0.8, 1.5)
#: Above this a datum, sounding or site is said to be fitted poorly.
POOR_CHI2 = 2.0
#: The overall score a result must reach, as in the ERT evaluation.
QUALITY_THRESHOLD = 70.0
#: Resistivities outside this range (ohm-m) are not physical for the near surface.
RESISTIVITY_RANGE = (0.1, 1e5)
#: A cell within this fraction of a velocity bound is said to sit on it.
BOUND_MARGIN = 0.02
#: A static shift beyond this factor is worth a reader's attention.
STATIC_SHIFT_FACTOR = 2.5

DPI = 150


# ---------------------------------------------------------------------------
# Shared rules
# ---------------------------------------------------------------------------
def chi2_fit_score(chi2: Optional[float], target: float = CHI2_TARGET,
                   acceptable: Tuple[float, float] = CHI2_ACCEPTABLE
                   ) -> Tuple[Optional[float], str]:
    """Score a final chi-squared out of 100, and say which side of the target it is.

    Inside the accepted range the score falls 20 points per unit away from
    the target; below it (errors overstated, or noise fitted) it runs from 40
    to 60; above it (data not explained within their errors) it falls 10
    points per unit from 60. ``InversionEvaluationAgent`` scores ERT this way.

    Examples
    --------
    >>> chi2_fit_score(1.0)
    (100.0, 'good')
    >>> chi2_fit_score(0.4)
    (50.0, 'overfit')
    >>> chi2_fit_score(3.5)
    (40.0, 'underfit')
    >>> chi2_fit_score(None)
    (None, 'unknown')
    """
    if chi2 is None or not np.isfinite(chi2) or chi2 < 0:
        return None, "unknown"
    low, high = acceptable
    if low <= chi2 <= high:
        score, status = 100.0 - abs(chi2 - target) * 20.0, "good"
    elif chi2 < low:
        score, status = 40.0 + chi2 / low * 20.0, "overfit"
    else:
        score, status = max(0.0, 60.0 - (chi2 - high) * 10.0), "underfit"
    return float(np.clip(round(score, 6), 0.0, 100.0)), status


def convergence_score(history: Optional[Sequence[float]], stop_at: Optional[float] = None
                      ) -> Tuple[Optional[float], Dict[str, Any]]:
    """Score how settled the chi-squared was over the last iterations.

    The mean relative improvement of the last two steps: below 0.1 % scores
    100, below 1 % 90, below 5 % 70, and anything faster 50, as the ERT
    evaluation scores it. Fewer than three values give no score. An inversion
    that stops once chi-squared reaches ``stop_at`` - pyGIMLi's stops at one -
    and got there converged, however fast it was still falling.

    >>> convergence_score([100.0, 10.0, 1.21, 1.2, 1.199])[0]
    90.0
    >>> convergence_score([1026.5, 33.5, 1.24, 0.68], stop_at=1.0)[0]
    100.0
    >>> convergence_score([5.0, 1.0])[0] is None
    True
    """
    values = [float(v) for v in (history or []) if v is not None and np.isfinite(v)]
    if len(values) < 3:
        return None, {"iterations": len(values)}
    if stop_at is not None and values[-1] <= stop_at:
        return 100.0, {"iterations": len(values), "initial_chi2": values[0],
                       "final_chi2": values[-1], "status": "target_reached"}
    steps = [(values[i] - values[i + 1]) / values[i] for i in (-3, -2) if values[i] > 0]
    improvement = float(np.mean(steps)) if steps else 0.0
    score = (100.0 if improvement < 0.001 else 90.0 if improvement < 0.01
             else 70.0 if improvement < 0.05 else 50.0)
    return score, {"iterations": len(values), "initial_chi2": values[0],
                   "final_chi2": values[-1], "last_improvement": improvement,
                   "status": "converged" if improvement < 0.01 else "still_improving"}


def _combine(scores: Mapping[str, Optional[float]], weights: Mapping[str, float]) -> float:
    """Weighted mean of the components that could be scored."""
    known = [(scores[k], weights[k]) for k in weights if scores.get(k) is not None]
    if not known:
        return 50.0
    return float(sum(s * w for s, w in known) / sum(w for _, w in known))


def _result(method: str, scores: Dict[str, Optional[float]], weights: Mapping[str, float],
            metrics: Dict[str, Any], recommendations: List[str], clauses: List[str],
            threshold: float, figure: Optional[str]) -> Dict[str, Any]:
    score = _combine(scores, weights)
    fit = scores.get("data_fit")
    ok = score >= threshold and (fit is None or fit >= 60.0)
    verdict = (f"Quality {score:.0f}/100, "
               + ("which meets" if ok else "below") + f" the {threshold:.0f} threshold")
    return {
        "status": "success" if ok else "needs_review",
        "method": method,
        "quality_score": score,
        "component_scores": {k: v for k, v in scores.items() if v is not None},
        "not_scored": [k for k in weights if scores.get(k) is None],
        "weights": dict(weights),
        "threshold": float(threshold),
        "metrics": metrics,
        "recommendations": recommendations or ["The result meets the quality criteria; no "
                                               "change to the inversion is needed."],
        "summary": verdict + (": " + "; ".join(clauses) if clauses else "") + ".",
        "figure": figure,
    }


def _velocity_bounds(limits: Any) -> Optional[Tuple[float, float]]:
    """Velocity bounds in m/s, whether given as velocity or (as pyGIMLi keeps them) slowness.

    >>> _velocity_bounds([1 / 8000, 1 / 86])
    (86.0, 8000.0)
    >>> _velocity_bounds(None) is None
    True
    """
    try:
        low, high = sorted(float(v) for v in limits)
    except (TypeError, ValueError):
        return None
    if not low > 0:
        return None
    if high < 1.0:
        low, high = sorted((1.0 / low, 1.0 / high))
    return round(low, 6), round(high, 6)


def _plausible(values: Any) -> Tuple[Optional[float], Dict[str, Any]]:
    """Score resistivities by the share inside :data:`RESISTIVITY_RANGE`."""
    values = np.asarray(values, dtype=float).ravel()
    values = values[np.isfinite(values) & (values > 0)]
    if not values.size:
        return None, {}
    low, high = RESISTIVITY_RANGE
    outside = float(np.mean((values < low) | (values > high)))
    return (float(np.clip(100.0 - outside * 200.0, 0.0, 100.0)),
            {"fraction_outside": outside, "range_ohm_m": [float(values.min()), float(values.max())]})


def _fit_words(status: str, chi2: float) -> str:
    return {"good": f"fits its data within their errors (chi-squared {chi2:.2f})",
            "overfit": f"fits its data more closely than their errors (chi-squared {chi2:.2f})",
            "underfit": f"does not fit its data within their errors (chi-squared {chi2:.2f})"
            }.get(status, "has no reported misfit")


def _save(figure: Any, path: Optional[Path]) -> Optional[str]:
    if path is None:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=DPI, bbox_inches="tight")
    return str(path)


# ---------------------------------------------------------------------------
# Seismic refraction
# ---------------------------------------------------------------------------
def evaluate_seismic(results: Mapping[str, Any], *, figure_path: Optional[Path] = None,
                     unit: Optional[str] = None,
                     threshold: float = QUALITY_THRESHOLD) -> Dict[str, Any]:
    """Score a refraction tomography from what ``SeismicAgent.execute`` returned.

    Parameters
    ----------
    results : dict
        The seismic results: ``chi2``, ``rrms``, ``coverage``,
        ``velocity_model`` and ``inversion_params`` - and, from this version
        of the agent on, ``traveltimes``, ``predicted_times``,
        ``relative_errors`` and ``chi2_history``. What is missing is not scored.
    figure_path : path, optional
        Where to draw the picks against the model's times.
    unit : str, optional
        Length unit of the figure's position axis.
    threshold : float
        The score a result must reach.

    Returns
    -------
    dict
        ``status`` (``success`` or ``needs_review``), ``quality_score``,
        ``component_scores``, ``metrics``, ``recommendations``, ``summary``
        and ``figure``.
    """
    metrics: Dict[str, Any] = {}
    scores: Dict[str, Optional[float]] = {}
    advice: List[str] = []
    clauses: List[str] = []

    chi2 = results.get("chi2")
    scores["data_fit"], status = chi2_fit_score(None if chi2 is None else float(chi2))
    metrics["data_fit"] = {"chi2": chi2, "rrms_percent": results.get("rrms"), "status": status}
    table = results.get("traveltimes") or {}
    observed = np.asarray(table.get("time_s", []), dtype=float)
    predicted = np.asarray(results.get("predicted_times", []), dtype=float)
    relative = np.asarray(results.get("relative_errors", []), dtype=float)
    paired = observed.size and observed.size == predicted.size
    if paired:
        residual = observed - predicted
        metrics["data_fit"]["rms_ms"] = float(np.sqrt(np.mean(residual ** 2)) * 1e3)
        if relative.size == observed.size:
            normalized = residual / np.maximum(relative * np.abs(observed), 1e-12)
            far = int(np.count_nonzero(np.abs(normalized) > 3.0))
            metrics["data_fit"]["picks_beyond_3_errors"] = far
            if far > 0.05 * observed.size:
                advice.append(f"{far} of {observed.size} picks lie more than three errors from "
                              "the model's times; check them against the traces, and the "
                              "positions of their shots.")
    if status != "unknown":
        clauses.append("the model " + _fit_words(status, float(chi2))
                       + (f", RMS {metrics['data_fit']['rms_ms']:.2f} ms"
                          if "rms_ms" in metrics["data_fit"] else ""))
    if status == "underfit":
        advice.append(f"The picks are not explained within their errors (chi-squared "
                      f"{float(chi2):.2f}): look for outlying picks and misplaced shots first; "
                      "a smaller lambda lets the model follow the picks more closely.")
    elif status == "overfit":
        advice.append(f"Chi-squared {float(chi2):.2f} is below the target of one: the errors "
                      "are probably overstated (pyGIMLi assumes 3 % when the file gives none), "
                      "or the model follows noise; a larger lambda smooths it.")

    coverage = np.asarray(results.get("coverage", []), dtype=float)
    covered = coverage > 0 if coverage.size else np.zeros(0, dtype=bool)
    if coverage.size:
        share = float(np.mean(covered))
        scores["ray_coverage"] = float(np.clip(share / 0.5 * 100.0, 0.0, 100.0))
        metrics["ray_coverage"] = {"fraction_covered": share}
        clauses.append(f"rays cover {share:.0%} of the model")
        if share < 0.4:
            advice.append(f"Rays cover only {share:.0%} of the model; below the deepest rays "
                          "the velocities are the starting gradient, not data. Report the "
                          "model only where it is covered.")

    velocity = np.asarray(results.get("velocity_model", []), dtype=float)
    limits = (results.get("inversion_params") or {}).get("limits")
    if velocity.size:
        inside = velocity[covered] if covered.size == velocity.size and covered.any() else velocity
        pinned, bounds = 0.0, _velocity_bounds(limits)
        if bounds is not None:
            low, high = bounds
            pinned = float(np.mean((inside <= low * (1 + BOUND_MARGIN))
                                   | (inside >= high * (1 - BOUND_MARGIN))))
        unphysical = float(np.mean((inside < 100.0) | (inside > 7000.0)))
        scores["physical_plausibility"] = float(np.clip(
            100.0 - 300.0 * pinned - 300.0 * unphysical, 0.0, 100.0))
        metrics["physical_plausibility"] = {
            "fraction_at_bound": pinned, "fraction_outside_100_7000": unphysical,
            "velocity_range": [float(inside.min()), float(inside.max())],
            "limits": list(bounds) if bounds is not None else None}
        if pinned > 0.02:
            advice.append(f"{pinned:.0%} of the covered cells sit at a velocity bound "
                          f"({bounds[0]:.0f} or {bounds[1]:.0f} m/s): the bound, "
                          "not the data, set them. Widen the limits or check the picks there.")
            clauses.append(f"{pinned:.0%} of covered cells at a velocity bound")

    # pyGIMLi stops once chi-squared reaches one, so a history ending there converged.
    scores["convergence"], metrics["convergence"] = convergence_score(
        results.get("chi2_history"), stop_at=CHI2_TARGET)
    if metrics["convergence"].get("status") == "still_improving":
        advice.append("The misfit was still falling when the inversion stopped; allow more "
                      "iterations.")

    figure = None
    if figure_path is not None and paired and table.get("geophone_x") is not None:
        figure = _seismic_figure(table, predicted, relative, Path(figure_path), unit)
    return _result("seismic", scores, {"data_fit": 0.45, "ray_coverage": 0.2,
                                       "physical_plausibility": 0.2, "convergence": 0.15},
                   metrics, advice, clauses, threshold, figure)


def _seismic_figure(table: Mapping[str, Any], predicted: np.ndarray, relative: np.ndarray,
                    path: Path, unit: Optional[str]) -> Optional[str]:
    from matplotlib import colormaps

    from PyHydroGeophysX.visualization.axis_units import set_length_axis

    shots = np.asarray(table["shot"])
    x = np.asarray(table["geophone_x"], dtype=float)
    observed = np.asarray(table["time_s"], dtype=float)
    order = np.unique(shots)
    colours = colormaps["viridis"](np.linspace(0.05, 0.9, max(1, order.size)))
    figure = detached_figure((11.0, 4.2))
    grid = figure.add_gridspec(1, 2, width_ratios=[1.7, 1])
    ax = figure.add_subplot(grid[0, 0])
    for colour, shot in zip(colours, order):
        on = shots == shot
        rank = np.argsort(x[on])
        ax.plot(x[on][rank], observed[on][rank] * 1e3, "o", ms=2.5, color=colour)
        ax.plot(x[on][rank], predicted[on][rank] * 1e3, "-", lw=1.0, color=colour)
    set_length_axis(ax, "x", "Geophone position", unit=unit)
    ax.set_ylabel("Travel time (ms)")
    ax.set_ylim(bottom=0.0)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.25)
    ax.set_title("Picks (dots) and the model's travel times (lines)", fontsize=10)
    hist = figure.add_subplot(grid[0, 1])
    residual = (observed - predicted) * 1e3
    hist.hist(residual, bins=30, color="#5e81ac", alpha=0.85)
    if relative.size == observed.size:
        typical = float(np.median(relative * np.abs(observed)) * 1e3)
        for edge in (-typical, typical):
            hist.axvline(edge, color="#bf616a", ls="--", lw=1)
    hist.set_xlabel("Pick minus model (ms)")
    hist.set_ylabel("Picks")
    hist.set_title("Residuals; dashed: the typical pick error", fontsize=10)
    figure.tight_layout()
    return _save(figure, path)


# ---------------------------------------------------------------------------
# TDEM
# ---------------------------------------------------------------------------
def evaluate_tdem(results: Mapping[str, Any], *, figure_path: Optional[Path] = None,
                  unit: Optional[str] = None,
                  threshold: float = QUALITY_THRESHOLD) -> Dict[str, Any]:
    """Score a TDEM survey's or sounding's inversion from what ``TDEMAgent`` returned.

    Parameters and returns as :func:`evaluate_seismic`. A survey is scored on
    its soundings' chi-squared, how many soundings resolved a model, and the
    physical range of the resistivities; one sounding on its chi-squared and
    its resistivities.
    """
    metrics: Dict[str, Any] = {}
    scores: Dict[str, Optional[float]] = {}
    advice: List[str] = []
    clauses: List[str] = []
    figure = None

    if results.get("survey"):
        chi2s = np.asarray(results.get("chi2_list", []), dtype=float)
        chi2s = chi2s[np.isfinite(chi2s)]
        median = results.get("chi2_sounding_median")
        median = float(median) if median is not None else (float(np.median(chi2s))
                                                            if chi2s.size else None)
        base, status = chi2_fit_score(median)
        within = float(np.mean(chi2s <= POOR_CHI2)) if chi2s.size else None
        scores["data_fit"] = (None if base is None else
                              base if within is None else 0.6 * base + 40.0 * within)
        metrics["data_fit"] = {"chi2_sounding_median": median, "chi2_global": results.get("chi2"),
                               "fraction_within_2": within, "status": status,
                               "n_soundings": int(results.get("n_soundings") or chi2s.size)}
        if median is not None:
            clauses.append("the median sounding " + _fit_words(status, median)
                           + (f", {within:.0%} of soundings within chi-squared {POOR_CHI2:g}"
                              if within is not None else ""))
        if within is not None and within < 0.8:
            poor = int(np.count_nonzero(chi2s > POOR_CHI2))
            advice.append(f"{poor} of {chi2s.size} soundings are fitted at chi-squared above "
                          f"{POOR_CHI2:g}; look at their decays for noisy late gates or coupling, "
                          "and re-invert with those gates dropped.")
        n = int(results.get("n_soundings") or 0)
        lost = int(results.get("failed_soundings") or 0) + int(results.get("unresolved_soundings") or 0)
        if n:
            share = 1.0 - lost / n
            scores["resolution"] = float(np.clip(share * 100.0, 0.0, 100.0))
            metrics["resolution"] = {"fraction_resolved": share, "failed": results.get("failed_soundings"),
                                     "unresolved": results.get("unresolved_soundings")}
            doi = np.asarray(results.get("doi", []), dtype=float)
            if doi.size and np.isfinite(doi).any():
                metrics["resolution"]["doi_median_m"] = float(np.nanmedian(doi))
                clauses.append(f"median depth of investigation {float(np.nanmedian(doi)):.0f} m")
            if lost:
                advice.append(f"{lost} of {n} soundings resolved no model above their depth of "
                              "investigation and are blank in the section.")
        robust = results.get("robust") or {}
        if robust.get("enabled") and robust.get("downweighted") is not None:
            metrics["data_fit"]["gates_downweighted"] = robust.get("downweighted")
        if figure_path is not None and chi2s.size:
            figure = _tdem_survey_figure(results, Path(figure_path), unit)
    else:
        chi2 = results.get("chi2")
        scores["data_fit"], status = chi2_fit_score(None if chi2 is None else float(chi2))
        metrics["data_fit"] = {"chi2": chi2, "status": status}
        if status != "unknown":
            clauses.append("the model " + _fit_words(status, float(chi2)))
        times = np.asarray(results.get("times", []), dtype=float)
        if figure_path is not None and times.size and np.size(results.get("predicted_data")) == times.size:
            figure = _tdem_sounding_figure(results, Path(figure_path))

    status = metrics["data_fit"].get("status")
    if status == "underfit":
        advice.append("The decays are not explained within their errors: check the noise "
                      "floor and the late gates, and whether the error model is too tight.")
    elif status == "overfit":
        advice.append("Chi-squared is below the target of one: the errors are probably "
                      "overstated, so the model may carry less structure than the data allow.")
    # No retry here, unlike the seismic evaluation: the line inversion has its
    # own smoothness search (auto_lambda), and the ground-TEM preset turns it
    # off on purpose - with four or five gates a sounding, chasing chi-squared
    # one fits noise. Say which way it went, and how to change it.
    searched = (results.get("inversion_settings") or {}).get("auto_lambda")
    if results.get("survey") and searched is not None:
        metrics["regularization"] = {"auto_lambda": bool(searched)}
        if not searched and status in ("underfit", "overfit"):
            advice.append("The smoothness was held fixed: the ground-TEM preset does not "
                          "search it, because with a handful of gates a sounding, chasing "
                          "chi-squared one fits noise. To let the line inversion search it "
                          "anyway, set auto_lambda to true in tdem_params.")
    scores["physical_plausibility"], metrics["physical_plausibility"] = _plausible(
        results.get("recovered_resistivity", []))
    if (metrics["physical_plausibility"] or {}).get("fraction_outside", 0) > 0.02:
        advice.append("Part of the model lies outside 0.1-100,000 ohm-m; those values are "
                      "set by the regularization, not by the data.")
    return _result("tdem", scores, {"data_fit": 0.55, "resolution": 0.25,
                                    "physical_plausibility": 0.2},
                   metrics, advice, clauses, threshold, figure)


def _tdem_survey_figure(results: Mapping[str, Any], path: Path, unit: Optional[str]) -> Optional[str]:
    from PyHydroGeophysX.visualization.axis_units import set_length_axis

    chi2 = np.asarray(results.get("chi2_list", []), dtype=float)
    positions = np.asarray(results.get("positions", []), dtype=float)
    if positions.size != chi2.size:
        positions = np.arange(chi2.size, dtype=float)
        along = False
    else:
        along = True
    lines = np.asarray(results.get("line_numbers", np.zeros(chi2.size)))
    doi = np.asarray(results.get("doi", []), dtype=float)
    figure = detached_figure((11.0, 5.2))
    top = figure.add_subplot(211)
    top.axhspan(*CHI2_ACCEPTABLE, color="#a3be8c", alpha=0.25, lw=0)
    top.axhline(POOR_CHI2, color="#bf616a", ls="--", lw=1)
    top.scatter(positions, chi2, c=lines if lines.size == chi2.size else None, cmap="tab20",
                s=10)
    top.set_yscale("log")
    top.set_ylabel("Chi-squared")
    top.grid(True, which="both", alpha=0.2)
    top.set_title("Fit of each sounding; green: accepted range, dashed: poorly fitted",
                  fontsize=10)
    bottom = figure.add_subplot(212, sharex=top)
    if doi.size == chi2.size:
        bottom.plot(positions, doi, ".", ms=3, color="#4c566a")
        bottom.invert_yaxis()
        set_length_axis(bottom, "y", "Depth of investigation", unit=unit)
    if along:
        set_length_axis(bottom, "x", "Position along line", unit=unit)
    else:
        bottom.set_xlabel("Sounding")
    bottom.grid(True, alpha=0.2)
    figure.tight_layout()
    return _save(figure, path)


def _tdem_sounding_figure(results: Mapping[str, Any], path: Path) -> Optional[str]:
    times = np.asarray(results["times"], dtype=float) * 1e3
    observed = np.asarray(results.get("observed_data", []), dtype=float)
    predicted = np.asarray(results["predicted_data"], dtype=float)
    errors = np.asarray(results.get("uncertainties", []), dtype=float)
    figure = detached_figure((10.0, 4.2))
    left, right = figure.subplots(1, 2)
    if observed.size == times.size:
        if errors.size == times.size:
            left.errorbar(times, np.abs(observed), yerr=errors, fmt="o", ms=3, color="#5e81ac",
                          label="Observed", capsize=2, lw=0.8)
        else:
            left.plot(times, np.abs(observed), "o", ms=3, color="#5e81ac", label="Observed")
    left.plot(times, np.abs(predicted), "-", color="#bf616a", lw=1.5, label="Model")
    left.set_xscale("log")
    left.set_yscale("log")
    left.set_xlabel("Time after turn-off (ms)")
    left.set_ylabel("|Response|")
    left.grid(True, which="both", alpha=0.2)
    left.legend(fontsize=8)
    left.set_title("Decay and the model's response", fontsize=10)
    if observed.size == times.size and errors.size == times.size:
        right.semilogx(times, (observed - predicted) / np.maximum(errors, 1e-30), "o", ms=3,
                       color="#4c566a")
        for edge in (-1, 1):
            right.axhline(edge, color="#bf616a", ls="--", lw=1)
        right.set_ylabel("Residual / error")
    right.set_xlabel("Time after turn-off (ms)")
    right.grid(True, which="both", alpha=0.2)
    right.set_title("Normalized residuals", fontsize=10)
    figure.tight_layout()
    return _save(figure, path)


# ---------------------------------------------------------------------------
# MT
# ---------------------------------------------------------------------------
def evaluate_mt(results: Mapping[str, Any], *, figure_path: Optional[Path] = None,
                threshold: float = QUALITY_THRESHOLD) -> Dict[str, Any]:
    """Score MT sites' 1D models from what the workflow's MT step returned.

    Occam's inversion aims at an RMS of one, so a site's RMS squared is
    scored as a chi-squared. The static shift each site needed is scored
    against :data:`STATIC_SHIFT_FACTOR`.
    """
    sites = [dict(site) for site in results.get("sites") or []]
    metrics: Dict[str, Any] = {}
    scores: Dict[str, Optional[float]] = {}
    advice: List[str] = []
    clauses: List[str] = []
    rms = np.asarray([float(site.get("rms", np.nan)) for site in sites], dtype=float)
    finite = rms[np.isfinite(rms)]
    if finite.size:
        median = float(np.median(finite))
        base, status = chi2_fit_score(median ** 2)
        within = float(np.mean(finite <= POOR_CHI2))
        scores["data_fit"] = 0.6 * base + 40.0 * within
        metrics["data_fit"] = {"rms_median": median, "rms_by_site": {
            site.get("station", f"site {k + 1}"): float(site.get("rms", np.nan))
            for k, site in enumerate(sites)}, "fraction_within_2": within, "status": status}
        clauses.append(f"median site RMS {median:.2f} against Occam's target of 1"
                       + (f", {within:.0%} of sites within RMS {POOR_CHI2:g}" if len(sites) > 1 else ""))
        poor = [site.get("station", "?") for site, value in zip(sites, rms) if value > POOR_CHI2]
        if poor:
            advice.append(f"{', '.join(map(str, poor))} {'is' if len(poor) == 1 else 'are'} "
                          f"fitted at RMS above {POOR_CHI2:g}: check the transfer functions for "
                          "noisy bands and remove them before inverting again.")
    else:
        metrics["data_fit"] = {"status": "unknown"}
        scores["data_fit"] = None
    shifts = []
    for site in sites:
        factors = [float(v) for v in (site.get("static_shift") or {}).values()
                   if v is not None and np.isfinite(v) and v > 0]
        if factors:
            shifts.append((site.get("station", "?"), max(factors, key=lambda f: abs(np.log10(f)))))
    if shifts:
        large = [(name, f) for name, f in shifts
                 if abs(np.log10(f)) > np.log10(STATIC_SHIFT_FACTOR)]
        scores["static_shift"] = float(100.0 * (1.0 - len(large) / len(shifts)))
        metrics["static_shift"] = {"factors": {name: f for name, f in shifts}}
        if large:
            advice.append("Static shifts larger than a factor of "
                          f"{STATIC_SHIFT_FACTOR:g} at " + ", ".join(
                              f"{name} (x{f:.2f})" for name, f in large)
                          + ": the shallow resistivity there is uncertain by that factor; "
                            "tie it to TEM or ERT over the same site.")
    values = np.concatenate([np.asarray(site.get("resistivity_ohm_m", []), dtype=float).ravel()
                             for site in sites]) if sites else np.zeros(0)
    scores["physical_plausibility"], metrics["physical_plausibility"] = _plausible(values)
    profile = results.get("profile") or {}
    if profile.get("rms") is not None:
        metrics["profile"] = {"rms": float(profile["rms"])}
        clauses.append(f"the 2D section fits to RMS {float(profile['rms']):.2f}")
    figure = None
    if figure_path is not None and finite.size:
        figure = _mt_figure(sites, Path(figure_path))
    return _result("mt", scores, {"data_fit": 0.6, "static_shift": 0.2,
                                  "physical_plausibility": 0.2},
                   metrics, advice, clauses, threshold, figure)


def _mt_figure(sites: Sequence[Mapping[str, Any]], path: Path) -> Optional[str]:
    names = [str(site.get("station", f"site {k + 1}")) for k, site in enumerate(sites)]
    rms = [float(site.get("rms", np.nan)) for site in sites]
    figure = detached_figure((max(5.0, 0.6 * len(sites) + 3.0), 3.8))
    ax = figure.add_subplot(111)
    colours = ["#a3be8c" if value <= CHI2_ACCEPTABLE[1] else "#ebcb8b" if value <= POOR_CHI2
               else "#bf616a" for value in rms]
    ax.bar(range(len(sites)), rms, color=colours)
    ax.axhline(1.0, color="#4c566a", lw=1)
    ax.axhline(POOR_CHI2, color="#bf616a", ls="--", lw=1)
    ax.set_xticks(range(len(sites)), names, rotation=45 if len(sites) > 6 else 0, ha="right"
                  if len(sites) > 6 else "center")
    ax.set_ylabel("RMS")
    ax.set_title("Fit of each site; solid: Occam's target, dashed: poorly fitted", fontsize=10)
    ax.grid(True, axis="y", alpha=0.25)
    figure.tight_layout()
    return _save(figure, path)


# ---------------------------------------------------------------------------
# Gravity and magnetics
# ---------------------------------------------------------------------------
def evaluate_gravmag(results: Mapping[str, Any], *, figure_path: Optional[Path] = None,
                     unit: Optional[str] = None,
                     threshold: float = QUALITY_THRESHOLD) -> Dict[str, Any]:
    """Score a 3D density-contrast or susceptibility model from ``GravMagAgent``'s results.

    The data fit is the residual anomaly's chi-squared at the beta the
    inversion chose; the stations misfit by more than three errors are
    counted; a model that reaches the solver's bound (1 g/cc, 0.5 SI) is
    marked down, since the bound and not the data then set it; and the beta
    search is scored on whether it landed on the target. Potential fields
    carry no depth resolution of their own - the depth of the model comes from
    the sensitivity weighting - so that is always said.
    """
    inversion = dict(results.get("inversion") or {})
    metrics: Dict[str, Any] = {}
    scores: Dict[str, Optional[float]] = {}
    advice: List[str] = []
    clauses: List[str] = []
    chi2 = inversion.get("chi2")
    scores["data_fit"], status = chi2_fit_score(None if chi2 is None else float(chi2))
    metrics["data_fit"] = {"chi2": chi2, "status": status, "n_data": inversion.get("n_data")}
    unit_name = results.get("unit") or ""
    fit = inversion.get("fit") or {}
    observed = np.asarray(fit.get("observed", []), dtype=float)
    predicted = np.asarray(fit.get("predicted", []), dtype=float)
    std = np.asarray(fit.get("std", []), dtype=float)
    paired = observed.size and observed.size == predicted.size
    if paired:
        residual = observed - predicted
        metrics["data_fit"]["rms"] = float(np.sqrt(np.mean(residual ** 2)))
        if std.size == observed.size:
            far = int(np.count_nonzero(np.abs(residual) > 3.0 * np.maximum(std, 1e-30)))
            metrics["data_fit"]["stations_beyond_3_errors"] = far
            if far > 0.05 * observed.size:
                advice.append(f"{far} of {observed.size} stations lie more than three errors "
                              "from the model's anomaly; look for spikes, a wrong tie or a "
                              "station position error there.")
    if status != "unknown":
        clauses.append("the model " + _fit_words(status, float(chi2))
                       + (f", RMS {metrics['data_fit']['rms']:.3g} {unit_name}"
                          if "rms" in metrics["data_fit"] else ""))
    if status == "underfit":
        advice.append("The residual anomaly is not explained within its errors: check the "
                      "regional trend removed and the noise floor before trusting the model.")
    elif status == "overfit":
        advice.append("Chi-squared is below the target: the errors (3 % plus the noise floor) "
                      "are probably overstated, so the model shows less than the data allow.")
    if inversion:
        pinned = bool(inversion.get("at_bound"))
        scores["physical_plausibility"] = 40.0 if pinned else 100.0
        metrics["physical_plausibility"] = {"model_range": inversion.get("model_range"),
                                            "at_bound": pinned, "bound": inversion.get("bound")}
        if pinned:
            advice.append(f"The model reaches the solver's bound (±{inversion.get('bound')}); "
                          "there the bound, not the data, set the value. A source closer to "
                          "the surface or a larger contrast than the bound allows is likely.")
            clauses.append("the model reaches the solver's bound")
    beta = inversion.get("beta") or {}
    if beta.get("status"):
        scores["convergence"] = {"converged": 100.0, "fixed": 70.0}.get(str(beta["status"]), 60.0)
        metrics["convergence"] = {"beta": beta.get("beta"), "status": beta.get("status"),
                                  "reason": beta.get("reason"),
                                  "chi2_at_each_beta": inversion.get("convergence")}
        if beta["status"] not in ("converged", "fixed"):
            advice.append("The beta search did not land on the target chi-squared "
                          f"({beta.get('reason') or beta['status']}); the model is the "
                          "closest trial.")
    advice.append("Gravity and magnetic data alone do not fix the depth of a source; the "
                  "depth in this model comes from the sensitivity weighting. Tie it to a "
                  "borehole, a seismic line or the geology before reading depths from it.")
    figure = None
    if figure_path is not None and paired and np.size(fit.get("x")) == observed.size:
        figure = _gravmag_figure(fit, results, Path(figure_path), unit)
    return _result("gravmag", scores, {"data_fit": 0.6, "physical_plausibility": 0.25,
                                       "convergence": 0.15},
                   metrics, advice, clauses, threshold, figure)


def _gravmag_figure(fit: Mapping[str, Any], results: Mapping[str, Any], path: Path,
                    unit: Optional[str]) -> Optional[str]:
    from matplotlib.colors import Normalize, TwoSlopeNorm

    from PyHydroGeophysX.visualization.axis_units import set_length_axis

    x, y = np.asarray(fit["x"], dtype=float), np.asarray(fit["y"], dtype=float)
    observed = np.asarray(fit["observed"], dtype=float)
    predicted = np.asarray(fit["predicted"], dtype=float)
    std = np.asarray(fit.get("std", np.ones_like(observed)), dtype=float)
    field = results.get("unit") or ""
    low, high = np.percentile(np.concatenate([observed, predicted]), [2, 98])
    figure = detached_figure((13.0, 4.4))
    axes = figure.subplots(1, 3)
    panels = ((observed, f"Residual anomaly ({field})", "Data", "RdBu_r", Normalize(low, high)),
              (predicted, f"Model anomaly ({field})", "Model", "RdBu_r", Normalize(low, high)),
              ((observed - predicted) / np.maximum(std, 1e-30), "Misfit / error",
               "Data minus model, in errors", "PuOr_r", TwoSlopeNorm(0.0, -3.0, 3.0)))
    for ax, (values, label, title, cmap, norm) in zip(axes, panels):
        shown = ax.scatter(x, y, c=values, cmap=cmap, norm=norm, s=14, edgecolors="none")
        figure.colorbar(shown, ax=ax, label=label, shrink=0.85)
        ax.set_aspect("equal", adjustable="datalim")
        ax.ticklabel_format(useOffset=False, style="plain")
        set_length_axis(ax, "x", "Easting", unit=unit)
        set_length_axis(ax, "y", "Northing", unit=unit, labelled=ax is axes[0])
        ax.set_title(title, fontsize=10)
    figure.tight_layout()
    return _save(figure, path)


def describe(evaluation: Mapping[str, Any]) -> str:
    """The step summary: the verdict, and whether a retry replaced the model.

    >>> describe({'summary': 'Quality 82/100, which meets the 70 threshold.', 'attempts': 1})
    'Quality 82/100, which meets the 70 threshold.'
    """
    text = str(evaluation.get("summary") or "")
    attempts = int(evaluation.get("attempts") or 1)
    if attempts > 1:
        adopted = evaluation.get("adopted_attempt")
        text += (f" Tried {attempts} settings; the run keeps attempt {adopted}"
                 + (f" (lambda {evaluation['adjusted_params']['lam']:g})"
                    if (evaluation.get("adjusted_params") or {}).get("lam") is not None else "")
                 + "." if adopted else f" Tried {attempts} settings.")
    return text
