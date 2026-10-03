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

from typing import Any, Dict, List, Optional, Sequence

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
    relationship = _relationship(results)
    calibrated = (relationship == "user" if relationship
                  else bool((config or {}).get("petrophysical_params")))
    if np.isfinite(typical) and typical == 0.0:
        # Every draw gave the same value: one realization, or nothing varied.
        # That is no uncertainty estimate, not a perfectly certain one.
        return {"ratio": float("inf"), "usable": False, "calibrated": calibrated,
                "sentence": "No uncertainty was computed for this water content - every "
                            "Monte Carlo draw gave the same value - so its reliability "
                            "cannot be judged."}
    if not np.isfinite(spread) or not np.isfinite(typical) or spread <= 0:
        return None
    ratio = typical / spread

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
        verdict += (" It rests on petrophysical parameters that were not calibrated for "
                    "this site (stated with their ranges in the warnings); calibrating "
                    "them against samples or logs is what would narrow this.")
    return {"ratio": ratio, "usable": ratio < _UNUSABLE,
            "calibrated": calibrated, "sentence": verdict}


def _relationship(results: Dict[str, Any]) -> str:
    """``user``, ``partial`` or ``default`` as the conversion recorded it, or ""."""
    if not isinstance(results, dict):
        return ""
    if results.get("petrophysical_relationship"):
        return str(results["petrophysical_relationship"])
    for step in results.get("time_lapse_water_content") or []:
        if isinstance(step, dict) and step.get("petrophysical_relationship"):
            return str(step["petrophysical_relationship"])
    return ""


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


# -- what the conversion assumed ----------------------------------------------
#
# A water content is only as good as the petrophysical relationship it went
# through, and the people reading it know that. When nobody gave that
# relationship the conversion still runs - on generic defaults with wide
# spreads - and the number it prints looks the same as a calibrated one. So the
# run says, every time, which parameters it drew and over what ranges, and
# whether they were the user's or defaults: a warning when any were defaults,
# a statement in the report either way.

#: Fewest Monte Carlo draws that estimate a spread at all. Asking for one draw
#: printed a standard deviation of exactly zero, which reads as certainty.
MIN_REALIZATIONS = 50

#: The parameters of the relationship, in the order they are stated, with a
#: name a reader recognises, a unit and how many decimals to print.
_PARAMETERS = (
    ("rho_sat", "saturated resistivity", " Ω·m", 0),
    ("m", "cementation exponent m", "", 1),
    ("rho_fluid", "pore-fluid resistivity", " Ω·m", 0),
    ("n", "saturation exponent n", "", 1),
    ("porosity", "porosity", "", 2),
    ("sigma_sur", "surface conductivity", " S/m", 3),
)
#: Parameters whose absence makes a relationship incomplete. Surface
#: conductivity is a small correction a supplied relationship rarely states.
_CORE = {"rho_sat", "m", "rho_fluid", "n", "porosity"}


def realizations(config: Dict[str, Any], note=None) -> int:
    """The number of Monte Carlo draws to use: as asked, but never too few.

    >>> realizations({'n_realizations': 1}, note=print)
    n_realizations was raised from 1 to 50: fewer draws cannot estimate an uncertainty, and a water content needs one.
    50
    >>> realizations({})
    100
    """
    asked = (config or {}).get("n_realizations", 100)
    try:
        count = int(asked)
    except (TypeError, ValueError):
        count = 100
    if count < MIN_REALIZATIONS:
        if note is not None:
            note(f"n_realizations was raised from {asked} to {MIN_REALIZATIONS}: fewer "
                 "draws cannot estimate an uncertainty, and a water content needs one.")
        count = MIN_REALIZATIONS
    return count


def _layer_parameters(draws: Dict[str, Any]) -> List[str]:
    """The parameters a layer's draws actually used, by its route through the model."""
    use_rho_sat = np.asarray(draws.get("use_rho_sat", [0.0]), dtype=float)
    if use_rho_sat.size and float(np.nanmean(use_rho_sat)) >= 0.5:
        return ["rho_sat", "n", "porosity", "sigma_sur"]
    return ["m", "rho_fluid", "n", "porosity", "sigma_sur"]


def prior_ranges(params_used: Dict[Any, Dict[str, Any]]) -> Dict[int, Dict[str, tuple]]:
    """The 5th and 95th percentile of every parameter actually drawn, per layer.

    From the draws themselves, after the sampler's bounds - not from the
    normal distributions requested, which a wide prior's clipping changes.

    >>> draws = {0: {'n': np.linspace(1.0, 4.0, 101), 'porosity': np.full(101, 0.3),
    ...              'm': np.full(101, 1.5), 'rho_fluid': np.full(101, 20.0),
    ...              'sigma_sur': np.zeros(101), 'use_rho_sat': np.zeros(101)}}
    >>> {k: tuple(round(v, 2) for v in r) for k, r in prior_ranges(draws)[0].items()}
    {'m': (1.5, 1.5), 'rho_fluid': (20.0, 20.0), 'n': (1.15, 3.85), 'porosity': (0.3, 0.3), 'sigma_sur': (0.0, 0.0)}
    """
    out: Dict[int, Dict[str, tuple]] = {}
    for marker, draws in (params_used or {}).items():
        layer: Dict[str, tuple] = {}
        for name in _layer_parameters(draws):
            values = np.asarray(draws.get(name, []), dtype=float)
            values = values[np.isfinite(values)]
            if values.size:
                low, high = np.percentile(values, [5, 95])
                layer[name] = (float(low), float(high))
        out[int(marker)] = layer
    return out


def _number(value: float, digits: int) -> str:
    text = f"{value:.{digits}f}"
    return text.rstrip("0").rstrip(".") if "." in text and digits >= 2 else text


def _span(name: str, bounds: tuple) -> str:
    _key, label, unit, digits = next(p for p in _PARAMETERS if p[0] == name)
    low, high = bounds
    if abs(high - low) <= 1e-9 * max(1.0, abs(high)):
        return f"{label} {_number(low, digits)}{unit} (fixed)"
    return f"{label} {_number(low, digits)}–{_number(high, digits)}{unit}"


def describe_prior(params_used: Dict[Any, Dict[str, Any]],
                   given: Optional[Dict[Any, Sequence[str]]] = None,
                   n_realizations: Optional[int] = None) -> Dict[str, Any]:
    """Whose petrophysical relationship the conversion used, and over what ranges.

    Parameters
    ----------
    params_used : dict
        The Monte Carlo draws per layer and parameter
        (:func:`~PyHydroGeophysX.petrophysics.monte_carlo.run_petrophysics_monte_carlo`).
    given : dict, optional
        Per layer, the parameters the user supplied. Anything else drawn is a
        default.
    n_realizations : int, optional
        The number of draws, for the statement.

    Returns
    -------
    dict
        ``relationship`` - ``"user"`` (every core parameter supplied),
        ``"partial"`` or ``"default"``; ``ranges`` - :func:`prior_ranges`;
        ``ranges_text`` - the ranges in words; ``statement`` - what a reader
        must be told, which is a warning unless ``relationship`` is ``"user"``.

    Examples
    --------
    >>> draws = {0: {'n': np.linspace(1.0, 4.0, 101), 'porosity': np.linspace(0.01, 0.9, 101),
    ...              'm': np.linspace(1.0, 3.5, 101), 'rho_fluid': np.full(101, 20.0),
    ...              'sigma_sur': np.linspace(0, 0.015, 101), 'use_rho_sat': np.zeros(101)}}
    >>> prior = describe_prior(draws, {}, 101)
    >>> prior['relationship']
    'default'
    >>> print(prior['statement'])  # doctest: +NORMALIZE_WHITESPACE
    Water content was produced, but it is not reliable: no petrophysical relationship was
    given, so the conversion used generic default parameters - cementation exponent m
    1.1–3.4, pore-fluid resistivity 20 Ω·m (fixed), saturation exponent n 1.1–3.9,
    porosity 0.05–0.86, surface conductivity 0.001–0.014 S/m (5–95% of 101 Monte Carlo
    draws). These ranges are assumptions, not values measured for this site, and the
    ± uncertainty reported reflects them. Give the site's relationship - Archie m, n and
    porosity, or a saturated resistivity, from cores, logs or published values for this
    formation - to make the estimate reliable.
    >>> describe_prior(draws, {0: ['m', 'n', 'porosity', 'rho_fluid']}, 101)['relationship']
    'user'
    """
    ranges = prior_ranges(params_used)
    given = {int(k): set(v or ()) for k, v in (given or {}).items()}
    layers = sorted(ranges)
    defaulted = {marker: [name for name in ranges[marker]
                          if name in _CORE and name not in given.get(marker, set())]
                 for marker in layers}
    supplied = {marker: [name for name in ranges[marker] if name in given.get(marker, set())]
                for marker in layers}
    if not any(supplied.values()):
        relationship = "default"
    elif any(defaulted.values()):
        relationship = "partial"
    else:
        relationship = "user"

    def layer_text(marker: int, mark_defaults: bool) -> str:
        parts = []
        for name, *_rest in _PARAMETERS:
            if name in ranges[marker]:
                text = _span(name, ranges[marker][name])
                if mark_defaults and name not in given.get(marker, set()):
                    text = (text[:-1] + ", default)" if text.endswith("(fixed)")
                            else text + " (default)")
                parts.append(text)
        return ", ".join(parts)

    mark = relationship == "partial"
    texts = [layer_text(marker, mark) for marker in layers]
    if len(set(texts)) <= 1:
        ranges_text = texts[0] if texts else ""
    else:
        ranges_text = "; ".join(f"unit {index + 1}: {text}" for index, text in enumerate(texts))
    draws = (f" (5–95% of {n_realizations} Monte Carlo draws)" if n_realizations
             else " (5–95% of the Monte Carlo draws)")
    ask = ("Give the site's relationship - Archie m, n and porosity, or a saturated "
           "resistivity, from cores, logs or published values for this formation - to "
           "make the estimate reliable.")
    if relationship == "default":
        statement = (
            "Water content was produced, but it is not reliable: no petrophysical "
            "relationship was given, so the conversion used generic default parameters - "
            f"{ranges_text}{draws}. These ranges are assumptions, not values measured "
            "for this site, and the ± uncertainty reported reflects them. " + ask)
    elif relationship == "partial":
        missing = sorted({name for names in defaulted.values() for name in names},
                         key=lambda n: [p[0] for p in _PARAMETERS].index(n))
        labels = [next(p[1] for p in _PARAMETERS if p[0] == n) for n in missing]
        listed = (labels[0] if len(labels) == 1
                  else ", ".join(labels[:-1]) + " and " + labels[-1])
        statement = (
            "Water content rests partly on default parameters, so it is only as "
            f"reliable as they are: the {listed} "
            f"{'was' if len(labels) == 1 else 'were'} not given and took generic "
            f"defaults. Parameters used - {ranges_text}{draws}. " + ask)
    else:
        statement = (f"The petrophysical relationship given was used: {ranges_text}{draws}.")
    return {"relationship": relationship, "ranges": ranges, "ranges_text": ranges_text,
            "statement": statement}
