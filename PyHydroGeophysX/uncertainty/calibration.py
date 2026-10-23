"""Checking a reported uncertainty against a known answer.

A water content reported as mean +/- standard deviation is a claim about how
far the truth may lie from the mean. Where the truth is known - a synthetic
model, cores, logs - :func:`interval_coverage` tests that claim: how often the
band holds the truth, and whether the misses are spread as a calibrated
Gaussian error would spread them.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


def _block(truth: np.ndarray, mean: np.ndarray, std: np.ndarray,
           p10: Optional[np.ndarray], p90: Optional[np.ndarray],
           weights: Optional[np.ndarray]) -> Dict[str, float]:
    """The coverage figures for one set of cells (all of them, or one group)."""
    n = int(truth.size)
    if n == 0:
        return {"n": 0}
    w = np.ones(n) if weights is None else weights / weights.sum() * n
    error = mean - truth
    spread = std > 0
    z = np.full(n, np.nan)
    z[spread] = (truth[spread] - mean[spread]) / std[spread]
    # A cell with no spread holds the truth only when its mean is the truth.
    distance = np.where(spread, np.abs(z), np.where(error == 0, 0.0, np.inf))
    finite = z[np.isfinite(z)]
    median = float(np.median(finite)) if finite.size else float("nan")
    out = {
        "n": n,
        "within_1sigma": float(np.average(distance <= 1.0, weights=w)),
        "within_2sigma": float(np.average(distance <= 2.0, weights=w)),
        "z_mean": float(np.mean(finite)) if finite.size else float("nan"),
        "z_std": float(np.std(finite)) if finite.size else float("nan"),
        "z_median": median,
        "z_robust_std": (float(1.4826 * np.median(np.abs(finite - median)))
                         if finite.size else float("nan")),
        "rmse": float(np.sqrt(np.average(error ** 2, weights=w))),
        "bias": float(np.average(error, weights=w)),
        "mean_std": float(np.average(std, weights=w)),
        "rms_std": float(np.sqrt(np.average(std ** 2, weights=w))),
        "zero_std": int((~spread).sum()),
    }
    out["rmse_over_rms_std"] = (out["rmse"] / out["rms_std"] if out["rms_std"] > 0
                                else float("inf"))
    if p10 is not None and p90 is not None:
        out["within_p10_p90"] = float(np.average((truth >= p10) & (truth <= p90), weights=w))
    return out


def interval_coverage(truth: Any, mean: Any, std: Any, groups: Any = None, *,
                      p10: Any = None, p90: Any = None,
                      weights: Any = None) -> Dict[str, Any]:
    """How often an estimate's mean +/- k standard deviations holds the truth.

    A calibrated Gaussian error bar holds the truth in 68.3 % of cells at one
    standard deviation and 95.4 % at two, z = (truth - mean) / std has mean 0
    and standard deviation 1, and p10-p90 holds it in 80 %. A band that passes
    the counts with z off-centre and narrow is wide, not calibrated: read the z
    figures and the per-group numbers (depth bins, units, coverage terciles),
    not only the totals. When the Monte Carlo draws one parameter set per unit,
    the cells of a unit are correlated, so the fractions are not binomial
    proportions and move with the seed.

    Parameters
    ----------
    truth, mean, std : array_like
        True value, estimated mean and standard deviation per cell, one size.
    groups : array_like, optional
        One label per cell; every label gets the same figures under ``groups``.
    p10, p90 : array_like, optional
        Percentiles of the estimate; add ``within_p10_p90``.
    weights : array_like, optional
        Cell weights, such as cell areas, so small shallow cells do not dominate.

    Returns
    -------
    dict
        ``overall`` and, with ``groups``, ``groups`` (label -> the same block),
        each with ``n``, ``within_1sigma``, ``within_2sigma``,
        ``within_p10_p90`` (when given), ``z_mean``, ``z_std``, ``z_median``,
        ``z_robust_std`` (1.4826 x MAD), ``rmse``, ``bias`` (mean - truth),
        ``mean_std``, ``rms_std``, ``rmse_over_rms_std`` and ``zero_std`` (cells
        with a standard deviation of zero: covered only where the mean is the
        truth, and left out of z); plus ``excluded``, the cells with a
        non-finite input or a negative standard deviation.

    Raises
    ------
    ValueError
        When the inputs differ in size.

    Examples
    --------
    >>> r = interval_coverage([0.10, 0.20, 0.30, 0.40], [0.10, 0.25, 0.20, 0.40], [0.05] * 4)
    >>> round(r["overall"]["within_1sigma"], 2), r["overall"]["within_2sigma"]
    (0.75, 1.0)
    >>> round(r["overall"]["bias"], 4)
    -0.0125
    >>> by_depth = interval_coverage([0.1, 0.2], [0.1, 0.4], [0.05, 0.05], groups=["0-1 m", "1-3 m"])
    >>> by_depth["groups"]["1-3 m"]["within_2sigma"]
    0.0
    """
    truth, mean, std = (np.asarray(a, dtype=float).ravel() for a in (truth, mean, std))
    extra = [None if a is None else np.asarray(a, dtype=float).ravel()
             for a in (p10, p90, weights)]
    label = None if groups is None else np.asarray(groups).ravel()
    sizes = {a.size for a in (truth, mean, std, *[e for e in extra if e is not None])}
    if label is not None:
        sizes.add(label.size)
    if len(sizes) > 1:
        raise ValueError(f"truth, mean, std and the optional arrays differ in size: "
                         f"{sorted(sizes)}.")
    keep = np.isfinite(truth) & np.isfinite(mean) & np.isfinite(std) & (std >= 0)
    for array in extra:
        if array is not None:
            keep &= np.isfinite(array)

    def pick(array: Optional[np.ndarray], mask: np.ndarray) -> Optional[np.ndarray]:
        return None if array is None else array[mask]

    result: Dict[str, Any] = {
        "overall": _block(truth[keep], mean[keep], std[keep], pick(extra[0], keep),
                          pick(extra[1], keep), pick(extra[2], keep)),
        "excluded": int((~keep).sum()),
    }
    if label is not None:
        result["groups"] = {}
        for key in dict.fromkeys(label[keep].tolist()):
            mask = keep & (label == key)
            result["groups"][key] = _block(truth[mask], mean[mask], std[mask],
                                           pick(extra[0], mask), pick(extra[1], mask),
                                           pick(extra[2], mask))
    return result


__all__ = ["interval_coverage"]
