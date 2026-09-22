"""What the user asked for, and whether the run actually produced it.

A workflow that quietly drops a requested product is worse than one that fails:
the report still looks complete, so nobody checks. This module is the guard
against that, and it exists because of a real run. The request

    "help me processs ERT data and estimate the water conent"

produced a full time-lapse ERT report with no water content anywhere and no
statement that it had been skipped. Two things had to go wrong together:

1. Intent was decided by substring matching against a keyword list, and the
   typo "conent" matched none of them - not ``water content``, not ``moisture``,
   nothing. The user asked in plain English and the workflow read "no".
2. The branch that ran had no petrophysics step at all, while the single-survey
   branch did, so even correct intent would have produced nothing.

Reading intent is a job for the model, not for a phrase list: a list only ever
recognises the wordings someone thought of in advance, and "I want to know how
wet the soil is" is not on it. :func:`llm_deliverables` is therefore the primary
path - the request parser asks the model one small structured question and the
answer becomes ``convert_to_water_content`` on the config.

Matching remains underneath it, for runs with no API key and for calls that
fail. :func:`wants_water_content` resolves in that order: the explicit flag, then
supplied parameters, then *fuzzy* phrase matching that survives a typo and
covers Chinese as well as English. Where it is genuinely ambiguous it answers
yes, because computing a product nobody wanted costs some time while skipping
one somebody asked for costs them the result.

:func:`unmet_requests` addresses the second failure, as a safety net that does
not care which branch ran or how intent was decided: it compares what was asked
for against what came back, and names anything missing so the caller can warn
instead of claiming success.
"""

import json
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

#: Phrases that mean "convert resistivity to water content". Matched fuzzily,
#: so near-misses and ordinary misspellings still count.
WATER_CONTENT_PHRASES = (
    "water content", "watercontent", "moisture", "saturation", "petrophysics",
    "petrophysical", "archie", "porosity", "rho_sat", "soil water",
    # The user writes requests in Chinese as often as English. These are matched
    # as plain substrings; the fuzzy path below is word-based and does not apply
    # to a script without spaces between words.
    "含水量", "含水率", "水分", "饱和度", "岩石物理",
)

#: Phrases that mean "stop at the resistivity model".
RESISTIVITY_ONLY_PHRASES = (
    "ert inversion only", "resistivity only", "just inversion", "only invert",
    "inversion only", "resistivity imaging", "只做反演", "只要电阻率",
)

#: How close a phrase in the request has to be to count as a match. 0.85 accepts
#: "water conent" (0.96 against "water content") and ordinary plural or spacing
#: differences, while still rejecting unrelated words.
_SIMILARITY = 0.85


def _mentions(text: str, phrases) -> bool:
    """Whether ``text`` contains any of ``phrases``, tolerating a typo.

    Exact substring first, then a sliding comparison over word windows of the
    same length as each phrase, so a misspelling inside an otherwise matching
    phrase still registers.
    """
    lowered = (text or "").lower()
    if any(phrase in lowered for phrase in phrases):
        return True
    words = lowered.replace("-", " ").split()
    for phrase in phrases:
        span = len(phrase.split())
        for start in range(len(words) - span + 1):
            window = " ".join(words[start:start + span])
            if SequenceMatcher(None, window, phrase).ratio() >= _SIMILARITY:
                return True
    return False


def wants_water_content(config: Dict[str, Any]) -> bool:
    """Whether this run should convert resistivity to water content.

    Parameters
    ----------
    config : dict
        Workflow configuration. ``convert_to_water_content`` is honoured first
        when present (the request parser or the user set it deliberately), then
        non-empty ``petrophysical_params``, then the wording of
        ``user_request``.

    Returns
    -------
    bool
        True when the product was asked for, explicitly or in words.

    Raises
    ------
    None

    Examples
    --------
    >>> wants_water_content({"user_request": "estimate the water conent"})
    True
    >>> wants_water_content({"user_request": "just inversion, no conversion"})
    False
    >>> wants_water_content({"user_request": "帮我算含水量"})
    True
    >>> wants_water_content({"convert_to_water_content": False,
    ...                      "user_request": "water content please"})
    False
    """
    explicit = config.get("convert_to_water_content")
    if explicit is not None:
        return bool(explicit)
    params = config.get("petrophysical_params") or {}
    if params:
        return True
    request = str(config.get("user_request", ""))
    # Asymmetric on purpose. Recognising a request is fuzzy, because missing one
    # costs the user their result; recognising a refusal is exact, because a
    # false positive there skips the deliverable - which is the whole failure
    # this module exists to prevent. ("resistivity to" scores 0.87 against
    # "resistivity only", which is close enough to have silently opted a user
    # out of the product they asked for in the same sentence.)
    if any(phrase in request.lower() for phrase in RESISTIVITY_ONLY_PHRASES):
        return False
    return _mentions(request, WATER_CONTENT_PHRASES)


#: Phrases that mean "bring meteorology into this". Matched like the water
#: content ones: fuzzily for a request, exactly for a refusal.
CLIMATE_PHRASES = (
    "climate", "precipitation", "rainfall", "weather", "meteorolog",
    "evapotranspiration", "temperature", "snowmelt", "daymet",
    "气象", "降水", "降雨", "蒸散", "气温", "融雪",
)


def wants_climate(config: Dict[str, Any]) -> bool:
    """Whether this run should retrieve and correlate meteorological data.

    Parameters
    ----------
    config : dict
        Workflow configuration. ``use_climate`` and a supplied
        ``climate_config`` are honoured first, then the wording of
        ``user_request``.

    Returns
    -------
    bool
        True when meteorological integration was asked for.

    Raises
    ------
    None

    Examples
    --------
    >>> wants_climate({"use_climate": True})
    True
    >>> wants_climate({"user_request": "compare the changes against rainfall"})
    True
    >>> wants_climate({"user_request": "just invert the surveys"})
    False
    """
    if config.get("use_climate") is not None:
        if bool(config.get("use_climate")):
            return True
    if config.get("climate_config"):
        return True
    return _mentions(str(config.get("user_request", "")), CLIMATE_PHRASES)


def climate_blocker(config: Dict[str, Any]) -> Optional[str]:
    """Why meteorological data cannot be retrieved, in the user's terms.

    Daymet is sampled at a point, so a position is required. A survey in a local
    coordinate frame has none - the electrode "y" values are elevations, not
    latitudes - and that is a fixable situation the user can only act on if they
    are told which piece is missing.

    Parameters
    ----------
    config : dict
        Workflow configuration.

    Returns
    -------
    str or None
        A sentence naming the obstacle, or None when nothing blocks retrieval.

    Raises
    ------
    None

    Examples
    --------
    >>> climate_blocker({"climate_config": {"coords": [-106.2, 41.4]}}) is None
    True
    >>> climate_blocker({"crs": "local"})
    'no site coordinates: the survey is in a local coordinate frame (crs="local"), which carries no longitude or latitude, and the request did not name the site'
    >>> climate_blocker({"crs": "local", "site_location": "Nowhere Creek"})
    'no site coordinates: the site was given as "Nowhere Creek" but that name could not be resolved to a position'
    """
    from ._geocode import coords_from_config

    if coords_from_config(config) is not None:
        return None
    crs = str(config.get("crs", "") or "").strip().lower()
    place = str(config.get("site_location", "") or "").strip()
    if place:
        # A name was read from the request but did not resolve, which is a
        # different problem from never having said where the site is.
        detail = (f'the site was given as "{place}" but that name could not be '
                  "resolved to a position")
    elif crs in ("local", "none", ""):
        detail = (f'the survey is in a local coordinate frame (crs="{crs or "unset"}"), '
                  "which carries no longitude or latitude, and the request did not "
                  "name the site")
    else:
        detail = (f'the survey coordinates are in "{crs}" but no site longitude and '
                  "latitude were supplied or resolved")
    return f"no site coordinates: {detail}"


def unmet_requests(config: Dict[str, Any], results: Dict[str, Any]) -> List[str]:
    """Products the request named that the results do not contain.

    The safety net, deliberately independent of which workflow branch ran: it
    reads the request and the finished results, not the plan. A caller turns
    each entry into a warning so the run is not reported as a clean success.

    Parameters
    ----------
    config : dict
        Workflow configuration, including ``user_request``.
    results : dict
        The finished workflow results.

    Returns
    -------
    list of str
        One plain-English sentence per missing product; empty when the run
        delivered everything it was asked for.

    Raises
    ------
    None

    Examples
    --------
    >>> unmet_requests({"user_request": "estimate water content"}, {"status": "success"})
    ['Water content was requested but this run produced none.']
    >>> unmet_requests({"user_request": "invert the ERT data"}, {"status": "success"})
    []
    """
    missing: List[str] = []
    if not isinstance(results, dict):
        return missing
    if wants_water_content(config):
        produced = (results.get("water_content_mean") is not None
                    or results.get("water_content") is not None
                    or bool(results.get("petrophysics_results"))
                    or bool(results.get("time_lapse_water_content")))
        if not produced:
            missing.append("Water content was requested but this run produced none.")
    if wants_climate(config) and not results.get("climate_data"):
        blocker = climate_blocker(config)
        reason = f" ({blocker})" if blocker else ""
        missing.append("Meteorological data were requested but none were "
                       f"retrieved{reason}.")
    return missing


#: What the model is asked about the request. Deliberately small and closed: one
#: question, a fixed answer shape, and an explicit "not stated" option so the
#: model is never pushed into guessing. Phrased to survive misspellings and any
#: language, which is the whole reason this beats a keyword list.
DELIVERABLES_PROMPT = """You decide which products a geophysics user is asking for.

Read the request and answer with JSON only - no prose, no code fence:
{"water_content": true, "resistivity_only": false, "site_location": null,
 "aspects": {"climate": false, "fusion": false, "tdem": false,
             "seismic": false, "hydro_model": false}}

Fields:
- water_content: true if they want water content, moisture, saturation or a
  petrophysical conversion estimated from the resistivity. false if they clearly
  do not want it. null if the request does not say either way.
- resistivity_only: true if they explicitly want to stop at the resistivity
  model and not convert it. Otherwise false.
- site_location: the place the survey is from, as a NAME a map service could
  look up - "Medicine Bow, Wyoming", "Reynolds Creek, Idaho". null if the
  request does not say where. Do NOT give latitude or longitude: a name that is
  looked up can be checked, a remembered coordinate cannot.
- aspects: which topics this request involves, so that only the matching
  configuration is extracted. Each is true or false:
    climate     - meteorological data: precipitation, temperature, PET, snow
    fusion      - combining two or more geophysical methods in one interpretation
    tdem        - time-domain or transient electromagnetic soundings
    seismic     - seismic refraction, travel times, velocity models
    hydro_model - outputs of a hydrological model such as MODFLOW or ParFlow
  Mark one false only when the request clearly does not involve it. If in doubt,
  mark it true: an extra question costs a moment, a missed one loses settings the
  user asked for.

The request may be in any language and may contain misspellings; judge what the
user means, not how they spelled it.

Request:
{request}"""


def llm_deliverables(user_request, query) -> Dict[str, Any]:
    """Ask the model which products ``user_request`` is asking for.

    This is the primary path. A model reads "estimate the water conent", or the
    same thing in Chinese, or "I want to know how wet the soil is" - none of
    which a phrase list recognises without someone having thought of them first.
    Every workflow here already parses the request with an LLM, so this is one
    more small question rather than new infrastructure.

    Parameters
    ----------
    user_request : str
        The user's own words.
    query : callable
        Takes a prompt string and returns the model's reply, typically an
        agent's ``query_llm``.

    Returns
    -------
    dict
        Keys present only when the model answered them definitely. ``{}`` when
        there is no model, the call fails, or the reply cannot be parsed - the
        caller then falls back to :func:`wants_water_content`'s matching.

    Raises
    ------
    None

    Examples
    --------
    >>> llm_deliverables("estimate water content", lambda p: '{"water_content": true}')
    {'convert_to_water_content': True}
    >>> llm_deliverables("invert this", lambda p: '{"water_content": null}')
    {}
    >>> llm_deliverables("anything", lambda p: 'sorry, I cannot help')
    {}
    """
    if not user_request or query is None:
        return {}
    try:
        reply = query(DELIVERABLES_PROMPT.replace("{request}", str(user_request)))
    except Exception:  # noqa: BLE001 - no key, rate limit, network: fall back
        return {}
    answer = _parse_json_object(reply)
    if not isinstance(answer, dict):
        return {}
    return _config_from(answer)


def _config_from(answer: Dict[str, Any]) -> Dict[str, Any]:
    """Workflow-configuration keys implied by the model's answer."""
    out: Dict[str, Any] = {}
    if answer.get("resistivity_only") is True:
        out["convert_to_water_content"] = False
    elif isinstance(answer.get("water_content"), bool):
        out["convert_to_water_content"] = answer["water_content"]
    place = answer.get("site_location")
    if isinstance(place, str) and place.strip():
        # A name, to be resolved by a gazetteer. Anything that parses as a pair
        # of numbers is the model having supplied coordinates after all, which
        # is exactly what must not be trusted.
        cleaned = place.strip()
        parts = cleaned.replace(",", " ").split()
        looks_numeric = len(parts) == 2 and all(
            part.replace("-", "").replace(".", "").isdigit() for part in parts)
        if not looks_numeric:
            out["site_location"] = cleaned
    return out


def _parse_json_object(reply) -> Any:
    """The first JSON object in ``reply``, or None.

    Models wrap JSON in prose or a code fence often enough that requiring a bare
    object would throw away good answers.
    """
    if not isinstance(reply, str):
        return None
    text = reply.strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except ValueError:
        return None


#: The extraction stages the request parser can run, and what each is for. A
#: stage costs an LLM call with a prompt of several hundred tokens, so running
#: all of them on every request spent most of the parsing time answering
#: questions the request had not raised.
ROUTABLE_ASPECTS = {
    "climate": "meteorological data: precipitation, temperature, PET, snow",
    "fusion": "combining two or more geophysical methods in one interpretation",
    "tdem": "time-domain or transient electromagnetic soundings",
    "seismic": "seismic refraction, travel times, velocity models",
    "hydro_model": "outputs of a hydrological model such as MODFLOW or ParFlow",
}


def _aspects_from(answer: Dict[str, Any]) -> Dict[str, bool]:
    """Which extraction stages the model says this request involves.

    Values are True, False, or None for "the model did not say". The third is
    not the same as True: a stage that also has a keyword gate should fall back
    to that gate when the model is silent, and no signal should be able to
    quietly outvote another. :func:`stage_enabled` does the composing.
    """
    aspects = answer.get("aspects")
    if not isinstance(aspects, dict):
        return {name: None for name in ROUTABLE_ASPECTS}
    return {name: (aspects[name] if isinstance(aspects.get(name), bool) else None)
            for name in ROUTABLE_ASPECTS}


def read_request(user_request, query) -> Tuple[Dict[str, Any], Dict[str, bool]]:
    """One model call: what the request asks for, and which stages to run.

    The parser used to make a call per topic whether or not the request
    mentioned the topic - for a plain ERT request, two of its calls returned
    only ``fusion_pattern: null``, ``use_climate: false`` and
    ``use_seismic: false``, at about seven seconds each. Asking once which
    topics are in play, then extracting only those, is the same decision made
    for a fraction of the cost, and it does not get slower as stages are added.

    Parameters
    ----------
    user_request : str
        The user's own words.
    query : callable
        Takes a prompt and returns the model's reply.

    Returns
    -------
    tuple
        ``(config_updates, aspects)`` - keys to merge into the workflow
        configuration, and a stage-name -> bool map. On any failure the config
        is empty and every stage is enabled, which is the old behaviour.

    Raises
    ------
    None

    Examples
    --------
    >>> reply = '{"water_content": true, "aspects": {"climate": false, "tdem": false}}'
    >>> config, aspects = read_request("estimate water content", lambda p: reply)
    >>> config, aspects["climate"], aspects["tdem"], aspects["fusion"]
    ({'convert_to_water_content': True}, False, False, None)
    >>> read_request("anything", lambda p: 'not json')[1]["climate"] is None
    True
    """
    if not user_request or query is None:
        return {}, {name: None for name in ROUTABLE_ASPECTS}
    try:
        reply = query(DELIVERABLES_PROMPT.replace("{request}", str(user_request)))
    except Exception:  # noqa: BLE001 - no key, rate limit, network
        return {}, {name: None for name in ROUTABLE_ASPECTS}
    answer = _parse_json_object(reply)
    if not isinstance(answer, dict):
        return {}, {name: None for name in ROUTABLE_ASPECTS}
    return _config_from(answer), _aspects_from(answer)


def stage_enabled(aspects: Dict[str, Any], name: str,
                  keyword_hit: Optional[bool] = None) -> bool:
    """Whether an extraction stage should run, from the router and its keywords.

    Composed so neither signal can quietly outvote the other. A keyword gate
    that matches always runs its stage - "modflow" in the request means the
    request is about MODFLOW, whatever the model thought - and the router can
    only skip stages that have no keyword gate of their own. When the model said
    nothing, the keyword gate decides alone, which is the behaviour that existed
    before the router.

    Parameters
    ----------
    aspects : dict
        Stage name -> True, False, or None, as :func:`read_request` returns.
    name : str
        The stage to decide.
    keyword_hit : bool, optional
        Whether this stage's own keyword gate matched. None when it has none.

    Returns
    -------
    bool
        True when the stage should run.

    Raises
    ------
    None

    Examples
    --------
    >>> stage_enabled({"climate": False}, "climate")          # router skips it
    False
    >>> stage_enabled({"tdem": False}, "tdem", keyword_hit=True)   # keywords win
    True
    >>> stage_enabled({"tdem": None}, "tdem", keyword_hit=False)   # model silent
    False
    >>> stage_enabled({}, "fusion")                            # nothing known
    True
    """
    stated = aspects.get(name)
    if keyword_hit is None:
        return stated is not False
    if stated is None:
        return bool(keyword_hit)
    return bool(stated) or bool(keyword_hit)
