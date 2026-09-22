"""Whether a derived product is precise enough to be interpreted.

A number with an error bar wider than the signal is not a weak result, it is an
absence of one — but printed as "water content 0.028 to 0.162" it reads like a
finding. This module turns the uncertainty the Monte Carlo already computed into
a statement about what the numbers can and cannot support.

The run that prompted it converted resistivity to water content across five
surveys and reported values spanning 0.028-0.162 with a mean standard deviation
of 0.081. That error bar covers 60% of the whole range: the driest and wettest
cells are not distinguishable, so no wetting or drying pattern in that image is
established. Two causes compound, and both are worth naming separately because
the user can act on the second:

- The petrophysical parameters were not supplied, so Archie exponents and
  saturated resistivity were generated at a "low information" level with 100%
  standard-deviation scaling. That spread propagates straight into the answer.
- Every mesh cell carried a distinct marker, so the layered model collapsed to
  one layer and a single parameter set was applied to soil, weathered rock and
  bedrock alike.
"""

from typing import Any, Dict, List, Optional

import numpy as np

#: Above this ratio of typical error bar to the spread being interpreted, the
#: image cannot separate its own extremes and no pattern in it is established.
_UNUSABLE = 0.5

#: Between the two, differences large compared with the error bar can still be
#: read qualitatively, but no value should be quoted as a measurement.
_QUALITATIVE = 0.25


def water_content_reliability(results: Dict[str, Any],
                              config: Optional[Dict[str, Any]] = None
                              ) -> Optional[Dict[str, Any]]:
    """How far the water-content estimate can be trusted, from its own spread.

    Parameters
    ----------
    results : dict
        Workflow results carrying ``water_content_mean`` / ``water_content_std``,
        or a ``time_lapse_water_content`` list of per-step petrophysics results.
    config : dict, optional
        Workflow configuration; ``petrophysical_params`` is read to tell a
        calibrated conversion from one run on generated defaults.

    Returns
    -------
    dict or None
        ``{"ratio", "usable", "calibrated", "sentence"}`` where ``ratio`` is the
        mean standard deviation divided by the range of the mean values, and
        ``sentence`` states in words what the estimate supports. None when the
        run produced no water content or no uncertainty to judge it by.

    Raises
    ------
    None

    Examples
    --------
    >>> tight = {"water_content_mean": [0.10, 0.30], "water_content_std": [0.01, 0.01]}
    >>> water_content_reliability(tight, {"petrophysical_params": {"n": 2}})["usable"]
    True
    >>> loose = {"water_content_mean": [0.03, 0.16], "water_content_std": [0.08, 0.08]}
    >>> water_content_reliability(loose, {})["usable"]
    False
    >>> water_content_reliability({}, {}) is None
    True
    """
    means, stds = _gather(results)
    if means is None or stds is None or means.size == 0 or stds.size == 0:
        return None
    spread = float(np.nanmax(means) - np.nanmin(means))
    typical = float(np.nanmean(stds))
    if not np.isfinite(spread) or not np.isfinite(typical) or spread <= 0:
        return None
    ratio = typical / spread
    calibrated = bool((config or {}).get("petrophysical_params"))

    if ratio >= _UNUSABLE:
        verdict = (
            f"The mean uncertainty (+/-{typical:.3f}) covers {ratio:.0%} of the range "
            f"of the estimate itself ({float(np.nanmin(means)):.3f} to "
            f"{float(np.nanmax(means)):.3f}), so the wettest and driest parts of this "
            f"image are not distinguishable and no moisture pattern in it is "
            f"established.")
    elif ratio >= _QUALITATIVE:
        verdict = (
            f"The mean uncertainty (+/-{typical:.3f}) is {ratio:.0%} of the range of "
            f"the estimate, so only differences much larger than that error bar should "
            f"be read, and no single value should be quoted as a measurement.")
    else:
        verdict = (
            f"The mean uncertainty (+/-{typical:.3f}) is {ratio:.0%} of the range of "
            f"the estimate, so contrasts across the section are resolved well enough "
            f"to compare.")

    if not calibrated:
        verdict += (" No petrophysical parameters were supplied for this site, so the "
                    "conversion ran on generated defaults; calibrating them against "
                    "samples or logs is what would narrow this.")
    return {"ratio": ratio, "usable": ratio < _UNUSABLE,
            "calibrated": calibrated, "sentence": verdict}


def _gather(results: Dict[str, Any]):
    """Mean and standard-deviation arrays from either result shape."""
    if not isinstance(results, dict):
        return None, None
    per_step: List[Dict[str, Any]] = results.get("time_lapse_water_content") or []
    if per_step:
        means = [np.asarray(step.get("water_content_mean"), dtype=float).ravel()
                 for step in per_step if step.get("water_content_mean") is not None]
        stds = [np.asarray(step.get("water_content_std"), dtype=float).ravel()
                for step in per_step if step.get("water_content_std") is not None]
        if means and stds:
            return np.concatenate(means), np.concatenate(stds)
    mean, std = results.get("water_content_mean"), results.get("water_content_std")
    if mean is None or std is None:
        return None, None
    return (np.asarray(mean, dtype=float).ravel(),
            np.asarray(std, dtype=float).ravel())


def result_caveats(config: Dict[str, Any], results: Dict[str, Any]) -> List[str]:
    """Warnings about products that exist but cannot carry the weight put on them.

    Distinct from a missing product: this is the case where a number was
    produced, will be read, and should not be read at face value.

    Parameters
    ----------
    config : dict
        Workflow configuration.
    results : dict
        Finished workflow results.

    Returns
    -------
    list of str
        One sentence per caveat; empty when nothing needs qualifying.

    Raises
    ------
    None

    Examples
    --------
    >>> loose = {"water_content_mean": [0.03, 0.16], "water_content_std": [0.08, 0.08]}
    >>> len(result_caveats({}, loose))
    1
    """
    caveats: List[str] = []
    reliability = water_content_reliability(results, config)
    if reliability and not reliability["usable"]:
        caveats.append("Water content is too uncertain to interpret quantitatively. "
                       + reliability["sentence"])
    return caveats
