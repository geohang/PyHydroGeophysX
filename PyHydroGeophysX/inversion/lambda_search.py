"""Chi-squared targeted regularization search, shared by every method.

ERT, SRT, time-lapse, potential fields, and the EM1D line inversion all face the
same question: the fixed regularization weight the user asked for did not land
the misfit near its target, so which weight would? The search below answers it
without knowing anything about the physics, so it lives here rather than in any
one method's module. Keeping it free of pygimli and SimPEG imports is the point:
the SimPEG-only paths must not pull in pygimli to adjust a scalar.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from PyHydroGeophysX._internal.utils import noop as _noop_log

# Lambda is a log-scale dial, so the search below moves in log space and the
# bounds are deliberately wide: under ~1e-3 the inversion is effectively
# unregularized, over ~1e5 the model is flat whatever the data says. The Qt
# spin box uses the same bounds so the UI cannot ask for a lambda the search
# would refuse to visit.
LAMBDA_BOUNDS: Tuple[float, float] = (1e-3, 1e5)

# Each bracketing step multiplies or divides lambda by this factor. Four is
# about one binary order of magnitude per inversion, which brackets a typical
# chi2 miss in two or three runs without overshooting far past the target.
_BRACKET_FACTOR = 4.0


def search_lambda_for_chi2(
    evaluate: Callable[[float], float],
    *,
    start_lambda: float,
    start_chi2: float,
    target_chi2: float = 1.0,
    tolerance: float = 0.2,
    max_trials: int = 6,
    bounds: Tuple[float, float] = LAMBDA_BOUNDS,
    log: Callable[[str], None] = _noop_log,
) -> Dict[str, Any]:
    """Search for the lambda whose chi-squared lands closest to ``target_chi2``.

    ``evaluate(lam)`` runs one inversion and returns its chi-squared; it is called
    at most ``max_trials`` times, on top of the run that produced ``start_chi2``.

    The search assumes chi2 grows with lambda, because more smoothing fits the
    data less well. It brackets the target by multiplying or dividing lambda by
    ``_BRACKET_FACTOR``, then bisects in log-lambda. Monotonicity only steers the
    next guess: the reported lambda is whichever trial came closest to the
    target, so a noisy or non-monotonic response degrades to "best of the trials"
    instead of failing.

    Returns a dict with ``lam``, ``chi2``, ``trials`` (every visited pair),
    ``status``, and ``reason``.
    """
    lam_min, lam_max = (float(min(bounds)), float(max(bounds)))
    target = float(target_chi2)
    tol = abs(float(tolerance))
    trials: List[Dict[str, float]] = [
        {"lambda": float(start_lambda), "chi2": float(start_chi2)}
    ]
    best_lam, best_chi2 = float(start_lambda), float(start_chi2)

    def _on_target(value: float) -> bool:
        return abs(value - target) <= tol

    # lo brackets the target from below in lambda (fits at least as well as the
    # target); hi brackets it from above (fits worse than the target).
    lo = float(start_lambda) if start_chi2 <= target else None
    hi = None if start_chi2 <= target else float(start_lambda)

    budget = max(0, int(max_trials))
    lam = float(start_lambda)
    reason = ""

    def _record(value_lam: float, value_chi2: float) -> None:
        nonlocal best_lam, best_chi2
        trials.append({"lambda": float(value_lam), "chi2": float(value_chi2)})
        if abs(value_chi2 - target) < abs(best_chi2 - target):
            best_lam, best_chi2 = float(value_lam), float(value_chi2)

    # Phase 1: expand geometrically until the target is bracketed.
    while budget > 0 and (lo is None or hi is None):
        nxt = lam * _BRACKET_FACTOR if hi is None else lam / _BRACKET_FACTOR
        nxt = min(max(nxt, lam_min), lam_max)
        if abs(nxt - lam) <= 1e-12 * max(1.0, lam):
            edge = "upper" if hi is None else "lower"
            reason = f"lambda reached its {edge} bound ({nxt:g})"
            break
        lam = nxt
        chi2 = evaluate(lam)
        budget -= 1
        if chi2 != chi2:  # NaN: the inversion did not report a usable chi2
            reason = f"inversion at lambda = {lam:g} returned no chi2"
            break
        _record(lam, chi2)
        if _on_target(chi2):
            return {"lam": best_lam, "chi2": best_chi2, "trials": trials,
                    "status": "converged", "reason": ""}
        if chi2 <= target:
            lo = lam
        else:
            hi = lam

    # Phase 2: bisect the bracket in log-lambda.
    while budget > 0 and lo is not None and hi is not None and lo < hi:
        lam = float(np.sqrt(lo * hi))
        chi2 = evaluate(lam)
        budget -= 1
        if chi2 != chi2:
            reason = f"inversion at lambda = {lam:g} returned no chi2"
            break
        _record(lam, chi2)
        if _on_target(chi2):
            return {"lam": best_lam, "chi2": best_chi2, "trials": trials,
                    "status": "converged", "reason": ""}
        if chi2 <= target:
            lo = lam
        else:
            hi = lam

    if not reason:
        reason = f"used all {int(max_trials)} trial(s) without landing inside the band"
    log(f"  lambda search stopped: {reason}")
    return {"lam": best_lam, "chi2": best_chi2, "trials": trials,
            "status": "best_effort", "reason": reason}


__all__ = ["LAMBDA_BOUNDS", "search_lambda_for_chi2"]
