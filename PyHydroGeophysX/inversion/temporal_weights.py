"""Weights for the temporal constraint of a time-lapse inversion.

The temporal regularization penalizes the difference between adjacent surveys,
:math:`\\alpha \\| W_t m \\|^2` with :math:`W_t` a first difference in time. With
one weight per adjacent pair that penalty is on the raw difference, which is only
the right question when the surveys are evenly spaced. A campaign that samples
hourly for a week and then monthly asks the solver to hold a one-hour change and
a one-month change equally still, so the sparse end of the series is smoothed as
hard as the dense end and real change between the widely spaced surveys is pushed
into the data misfit.

Weighting each pair by :math:`1/\\Delta t` turns the penalty into one on the *rate*
of change, :math:`|\\Delta m / \\Delta t|^2`: a long gap is allowed a proportionally
larger change, a short gap only a small one.

The weights are normalized by the median interval, which is what keeps ``alpha``
meaning the same thing it did before. On an evenly sampled series every weight is
then exactly 1 and the inversion is unchanged; only irregular sampling is
affected, which is the case the weighting exists for.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

import numpy as np

__all__ = ["MODES", "temporal_weights", "describe_temporal_weights"]

MODES = ("interval", "uniform")

#: Default cap on how far a weight may depart from the median interval, either
#: way. Without it a pair of surveys minutes apart inside a monthly series would
#: be weighted hundreds of times more heavily than the rest and effectively
#: frozen together.
DEFAULT_LIMIT = 10.0


def temporal_weights(measurement_times: Sequence[float], *,
                     mode: str = "interval",
                     limit: Optional[float] = DEFAULT_LIMIT,
                     decay_rate: float = 0.0) -> Tuple[np.ndarray, dict]:
    """One weight per adjacent survey pair, with a report of what was applied.

    Args:
        measurement_times: measurement time of every survey, in any one unit.
        mode: ``'interval'`` weights each pair by the median interval over its own
            (a penalty on the rate of change); ``'uniform'`` weights every pair
            equally, which is what the inversion did before this existed.
        limit: cap on the weight ratio, either way. ``None`` or a value ``<= 1``
            disables the cap.
        decay_rate: the existing exponential down-weighting of long gaps,
            ``exp(-decay_rate * dt)``. Applied on top, and inert at its default 0.

    Returns:
        ``(weights, report)`` with one weight per pair (``n_times - 1`` of them)
        and a JSON-safe dict describing the weighting, for the log and the run
        record. Falls back to uniform weights - and says so in the report - when
        the times cannot support the weighting: a zero or negative interval means
        two surveys carry the same time, or the sequence is out of order.
    """
    times = np.asarray(measurement_times, dtype=float).ravel()
    n_pairs = max(times.size - 1, 0)
    report: dict = {"mode": "uniform", "requested": str(mode), "applied": False,
                    "limit": None, "note": ""}
    if n_pairs == 0:
        return np.ones(0, dtype=float), report

    intervals = np.diff(times)
    base = np.exp(-float(decay_rate) * intervals)
    if str(mode).strip().lower() != "interval":
        report["note"] = "Temporal constraint weighted equally over every pair."
        return base, report

    if not np.isfinite(intervals).all() or np.any(intervals <= 0.0):
        report["note"] = (
            "Temporal constraint weighted equally: the measurement times do not "
            "increase (a zero or negative interval), so an interval weighting "
            "cannot be formed. Check the order of the survey files.")
        return base, report

    reference = float(np.median(intervals))
    scale = reference / intervals
    capped = 0
    if limit is not None and float(limit) > 1.0:
        cap = float(limit)
        capped = int(np.sum((scale > cap) | (scale < 1.0 / cap)))
        scale = np.clip(scale, 1.0 / cap, cap)
        report["limit"] = cap

    report.update({
        "mode": "interval",
        "applied": True,
        "reference_interval": reference,
        "interval_range": [float(intervals.min()), float(intervals.max())],
        "weight_range": [float(scale.min()), float(scale.max())],
        "capped_pairs": capped,
    })
    spread = float(scale.max() / scale.min()) if scale.min() > 0 else float("inf")
    if spread <= 1.0 + 1e-9:
        report["note"] = (
            "Temporal constraint weighted by the interval between surveys; the "
            "sampling is even, so every pair carries the same weight.")
    else:
        report["note"] = (
            f"Temporal constraint weighted by the interval between surveys "
            f"(median {reference:g}): weights {scale.min():.2f} to {scale.max():.2f}, "
            f"so a long gap is allowed proportionally more change than a short one."
            + (f" {capped} pair(s) hit the {report['limit']:g}x cap." if capped else ""))
    return base * scale, report


def describe_temporal_weights(report: Any) -> str:
    """The one line a run should log about its temporal weighting."""
    return str((report or {}).get("note") or "")
