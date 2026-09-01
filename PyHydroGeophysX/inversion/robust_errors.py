"""Bounded Huber IRLS for EM data only; regularisation is never reweighted.

These are effective fitting uncertainties, not revised measurement errors.
Every imported gate remains present. Recomputing each factor from the original
uncertainty permits a gate to regain weight and prevents cumulative inflation.
"""

from __future__ import annotations

from typing import Any, Callable, Dict

import numpy as np


def robust_error_options(inv: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "threshold": inv.get("robust_threshold", 3.0),
        "passes": inv.get("robust_passes", 3),
        "max_error_factor": inv.get("robust_max_error_factor", 10.0),
        "min_unchanged_fraction": inv.get("robust_min_unchanged_fraction", 0.0),
        "target_chi2": inv.get("robust_target_chi2", 0.0),
        "target_tolerance": inv.get("robust_target_tolerance", 0.25),
    }


def huber_error_factor(residual, threshold: float, max_error_factor: float):
    """sigma_eff/sigma_base, with inverse-variance weight min(1, k/|r|).

    The maximum error factor bounds the loss of influence. Beyond that cap this
    is a bounded Huber-style reweighting, not an exact unbounded Huber loss.
    """
    threshold, maximum = float(threshold), float(max_error_factor)
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("robust_threshold must be finite and positive.")
    if not np.isfinite(maximum) or maximum < 1:
        raise ValueError("robust_max_error_factor must be finite and >= 1.")
    residual = np.asarray(residual, dtype=float)
    if not np.all(np.isfinite(residual)):
        raise ValueError("robust reweighting received non-finite residuals.")
    return np.sqrt(np.clip(np.abs(residual) / threshold, 1.0, maximum ** 2))


def select_error_factors(residual, threshold=3.0, max_error_factor=10.0, *,
                         min_unchanged_fraction=0.0, target_chi2=0.0):
    """Limit changes to the worst residuals; optionally calibrate their errors.

    This target mode is NOT a Huber likelihood or independently estimated noise.
    For the current model it searches a bounded error multiplier, then the caller
    must refit. Protected gates have factor exactly 1, including when the target
    is unreachable. Ranks may change between passes; ties have stable ordering.
    The quota is across the input vector (one LCI run, or one independent fit).
    """
    r = np.asarray(residual, dtype=float).ravel()
    f = huber_error_factor(r, threshold, max_error_factor)
    fraction, target = float(min_unchanged_fraction), float(target_chi2)
    if not np.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("robust_min_unchanged_fraction must be in [0, 1].")
    if not np.isfinite(target) or target < 0:
        raise ValueError("robust_target_chi2 must be finite and >= 0 (0 disables it).")
    if not r.size:
        raise ValueError("error selection needs at least one gate.")
    n_fixed = int(np.ceil(fraction * r.size))
    protected = np.argsort(np.abs(r), kind="stable")[:n_fixed]
    f[protected] = 1.0
    eligible = f > 1.0
    lower_factors = np.where(eligible, float(max_error_factor), 1.0)
    lower_chi2 = float(np.mean((r / lower_factors) ** 2))
    report = {"minimum_unchanged": n_fixed, "eligible_for_inflation": int(eligible.sum()),
              "fixed_model_min_chi2": lower_chi2, "target_chi2": target,
              "status": "huber_only", "inflation_strength": 1.0}
    if target == 0:
        return f, report
    if float(np.mean(r*r)) <= target:
        report.update(status="already_below_target", inflation_strength=0.0)
        return np.ones_like(r), report
    if lower_chi2 >= target:
        report.update(status="limited_by_unchanged_fraction_threshold_or_cap",
                      inflation_strength=None)
        return lower_factors, report

    # Strength=1 reproduces Huber errors on the eligible subset. Search strength
    # above OR below 1 instead of inflating the protected 70% to make chi2=1.
    shape = np.where(eligible, np.maximum(np.abs(r) / float(threshold) - 1., 0.), 0.)
    def factors_at(strength):
        return np.sqrt(np.minimum(1. + strength * shape, float(max_error_factor)**2))
    low, high = 0., 1.
    for _ in range(128):
        if float(np.mean((r / factors_at(high))**2)) <= target:
            break
        high *= 2.
    for _ in range(60):
        mid = (low + high) / 2.
        if float(np.mean((r / factors_at(mid))**2)) > target:
            low = mid
        else:
            high = mid
    report.update(status="target_matched_at_current_model", inflation_strength=high)
    return factors_at(high), report


def reweight_errors(
    observed, uncertainty, solve: Callable, predict: Callable, *,
    threshold: float = 3.0, passes: int = 3, max_error_factor: float = 10.0,
    min_unchanged_fraction: float = 0.0, target_chi2: float = 0.0,
    target_tolerance: float = 0.2,
    solver_ready: Callable = lambda outcome: True,
    stage_statistics: Callable = lambda outcome, original_residual: {},
    history: Callable = lambda outcome: [], log: Callable = lambda message: None,
):
    """Solve, update uncertainties, and warm-start without ever masking a gate.

    ``solve(sigma, previous)`` is responsible for freezing regularisation after the
    initial solve. The returned report always scores the final model against ALL
    original observations, both with original and with actually used errors.
    ``solver_ready`` can veto further error inflation after an incomplete or
    failed inner solve. Budget exhaustion is not evidence of contaminated data.
    """
    observed = np.asarray(observed, dtype=float).ravel().copy()
    base = np.asarray(uncertainty, dtype=float).ravel().copy()
    if observed.size != base.size or not observed.size:
        raise ValueError("robust fitting needs nonempty, matching data and errors.")
    if not np.all(np.isfinite(observed)) or not np.all(np.isfinite(base) & (base > 0)):
        raise ValueError("robust fitting requires finite data and positive errors.")
    if (not isinstance(passes, (int, float, np.integer, np.floating))
            or isinstance(passes, bool) or not np.isfinite(passes)
            or int(passes) != passes or int(passes) < 1):
        raise ValueError("robust_passes must be a positive integer.")
    def choose(residual):
        return select_error_factors(residual, threshold, max_error_factor,
                                    min_unchanged_fraction=min_unchanged_fraction,
                                    target_chi2=target_chi2)
    factors, _ = choose(np.zeros(base.size))
    if not np.isfinite(target_tolerance) or target_tolerance < 0:
        raise ValueError("target_tolerance must be finite and nonnegative.")
    effective = base.copy()
    stages = []

    def score(outcome):
        predicted = np.asarray(predict(outcome), dtype=float).ravel()
        if predicted.size != observed.size or not np.all(np.isfinite(predicted)):
            raise ValueError("robust forward response must be finite and preserve every gate.")
        residual = (predicted - observed) / base
        return predicted, residual

    def stage(index, outcome, residual):
        return {
            "pass": index, "kept": int(base.size), "dropped": 0,
            "downweighted": int(np.count_nonzero(factors > 1.0)),
            "unchanged": int(np.count_nonzero(factors == 1.0)),
            "chi2_original": float(np.mean(residual ** 2)),
            "chi2_effective": float(np.mean((residual / factors) ** 2)),
            "convergence": [float(v) for v in history(outcome)],
            "solver_ready": bool(solver_ready(outcome)),
            **stage_statistics(outcome, residual),
        }

    outcome = solve(effective.copy(), None)
    predicted, residual = score(outcome)
    initial = stage(0, outcome, residual)
    stopped = "maximum reweighting passes reached"
    for index in range(1, int(passes) + 1):
        if not solver_ready(outcome):
            stopped = "inner solver incomplete; error inflation paused"
            log(f"  Robust weighting stopped: {stopped}.")
            break
        proposed, selection = choose(residual)
        if np.allclose(proposed, factors, rtol=0.01, atol=0.0):
            stopped = "effective errors stable (1% tolerance)"
            break
        factors = proposed
        effective = base * factors
        outcome = solve(effective.copy(), outcome)
        predicted, residual = score(outcome)
        entry = stage(index, outcome, residual)
        entry["error_selection"] = selection
        stages.append(entry)
        log(f"  Robust pass {index}: {entry['downweighted']}/{base.size} gates "
            f"downweighted, none removed; chi2 original={entry['chi2_original']:.3f}, "
            f"effective={entry['chi2_effective']:.3f}")
        if not solver_ready(outcome):
            stopped = "inner solver incomplete; error inflation paused"
            log(f"  Robust weighting stopped: {stopped}.")
            break
        if float(target_chi2) > 0 and abs(entry["chi2_effective"] - float(target_chi2)) <= target_tolerance:
            stopped = "effective chi2 target band reached (error-calibrated, not raw fit)"
            break

    _, final_limits = choose(residual)
    final_chi2 = float(np.mean((residual / factors) ** 2))
    info = {
        "enabled": True,
        "method": "bounded_target_error_scaling" if float(target_chi2) > 0 else "bounded_huber_irls",
        "threshold": float(threshold),
        "min_unchanged_fraction": float(min_unchanged_fraction),
        "fraction_scope": "input_vector",
        "unchanged": int(np.count_nonzero(factors == 1.0)),
        "unchanged_fraction": float(np.mean(factors == 1.0)),
        "target_chi2": float(target_chi2), "target_tolerance": float(target_tolerance),
        "target_reached": (bool(solver_ready(outcome)) and abs(final_chi2 - float(target_chi2)) <= target_tolerance
                           if float(target_chi2) > 0 else None),
        "final_model_error_limits": final_limits,
        "max_error_factor": float(max_error_factor), "passes_requested": int(passes),
        "initial": initial, "passes": stages, "stopped_because": stopped,
        "solver_ready": bool(solver_ready(outcome)),
        "n_start": int(base.size), "kept": int(base.size), "dropped": 0,
        "downweighted": int(np.count_nonzero(factors > 1.0)),
        "effective_data_count": float(np.sum(1.0 / factors ** 2)),
        "chi2_original": float(np.mean(residual ** 2)),
        "chi2_effective": final_chi2,
        "observed": observed.tolist(), "predicted": predicted.tolist(),
        "uncertainty_original": base.tolist(), "uncertainty_effective": effective.tolist(),
        "error_factor": factors.tolist(), "weights": (1.0 / factors ** 2).tolist(),
        "residual_original": residual.tolist(),
    }
    return outcome, effective, info
