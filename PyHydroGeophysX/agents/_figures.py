"""Which figures a report should carry, decided from the request.

The figure set used to be whatever the plotting code happened to reach. That is
how a run asking for water content produced four resistivity figures and none
of the product it was asked for: the water-content figure was written inside an
exception handler and so was drawn only when the others failed. Nothing
compared the figures produced against the figures the request implied, so
nothing noticed.

This module is that comparison. The request is read by the model - the same
model already parsing it - and turned into a set of *topics*. Topics are
matched against the figures whose data the run actually produced, and the
result says three things rather than one:

- ``draw``: figures to render, ordered so the requested topics come first.
- ``omitted``: figures deliberately left out, each with the reason, so a
  narrow request ("just the resistivity section") gets a short report instead
  of everything the code can plot.
- ``missing``: a topic the user asked about for which no data exists. This is
  the one that matters: it becomes a warning the user sees, rather than a
  silent absence in the figure list.

The asymmetry is deliberate and matches :mod:`PyHydroGeophysX.agents._intent`:
a topic is included on weak evidence and excluded only on explicit evidence.
Drawing a figure nobody asked for costs a scroll; omitting one they did ask for
is the failure this exists to prevent. So when the model cannot be reached, or
answers unusably, every available figure is drawn.
"""

import json
from typing import Any, Callable, Dict, List, Optional, Sequence

#: Figure key -> the topic it belongs to and how to describe it. The keys are
#: the ones the report agent stores in its ``vis_files`` mapping.
FIGURE_CATALOG: Dict[str, Dict[str, str]] = {
    "baseline_resistivity": {
        "topic": "resistivity",
        "caption": "Recovered resistivity model for the baseline survey.",
    },
    "timelapse_all_resistivity": {
        "topic": "resistivity",
        "caption": "Recovered resistivity model for each survey, on a shared "
                   "colour scale.",
    },
    "timelapse_changes_percent": {
        "topic": "change",
        "caption": "Resistivity change relative to the baseline, as a percentage.",
    },
    "timelapse_changes_absolute": {
        "topic": "change",
        "caption": "Resistivity change relative to the baseline, in ohm-m.",
    },
    "timelapse_water_content": {
        "topic": "water_content",
        "caption": "Volumetric water content derived from each resistivity "
                   "model, on a shared colour scale.",
    },
    "water_content": {
        "topic": "water_content",
        "caption": "Volumetric water content derived from the resistivity model.",
    },
    "water_content_uncertainty": {
        "topic": "uncertainty",
        "caption": "Standard deviation of the water-content estimate from the "
                   "Monte Carlo ensemble.",
    },
    "climate_correlation": {
        "topic": "climate",
        "caption": "Meteorological series over the monitoring period against "
                   "the recovered resistivity.",
    },
    "workflow_summary": {
        "topic": "workflow",
        "caption": "Summary of the workflow that produced this report.",
    },
}

#: Topic -> what it means, for the prompt and for the warning text.
FIGURE_TOPICS: Dict[str, str] = {
    "resistivity": "the recovered resistivity model or section",
    "change": "how resistivity changed between surveys",
    "water_content": "water content, moisture or saturation",
    "uncertainty": "the uncertainty or spread of an estimate",
    "climate": "precipitation, temperature or other meteorological series",
    "workflow": "a diagram of the processing workflow itself",
}

FIGURES_PROMPT = """You decide which figures a geophysics report should show.

Read the request and answer with JSON only - no prose, no code fence:
{"topics": ["resistivity", "water_content"], "only_these": false,
 "style": {"size": null, "colormap": null}}

topics: the subjects the user wants to SEE plotted. Choose from:
- resistivity: the recovered resistivity model or section
- change: how resistivity changed between surveys
- water_content: water content, moisture or saturation
- uncertainty: the uncertainty or spread of an estimate
- climate: precipitation, temperature or other meteorological series
- workflow: a diagram of the processing workflow itself

Include a topic whenever the request asks for that quantity at all, whether or
not it says "plot" or "figure" - someone who asks you to estimate water content
wants to see it. If the request does not narrow things down, list the topics it
involves and leave the rest out.

only_these: true ONLY if the user explicitly limited what they want shown, for
example "just the resistivity section" or "no uncertainty plots". false in every
other case, including when you are unsure. Setting it true suppresses figures,
so set it true only on explicit instruction.

style: how the figures should look, when the request says. Both fields are null
unless the request actually asks.
- size: "compact", "normal" or "large". Use it for "make the figures bigger",
  "smaller plots", "these are huge".
- colormap: a matplotlib colormap name the request asks for by name, such as
  "viridis" or "cividis". Do not invent one and do not choose on the user's
  behalf; an unrecognised name is ignored.

The request may be in any language and may contain misspellings; judge what the
user means, not how they spelled it.

Request:
{request}"""


def llm_figure_topics(user_request: str, query: Callable[[str], str]
                      ) -> Optional[Dict[str, Any]]:
    """Ask the model which subjects the request wants plotted.

    Parameters
    ----------
    user_request : str
        The request in the user's own words, in any language.
    query : callable
        Takes a prompt and returns the model's reply. Anything it raises is
        treated as "no answer": a missing key or a failed call must not decide
        the figure set.

    Returns
    -------
    dict or None
        ``{"topics": set, "only_these": bool}``, or None when the model could
        not be reached or did not answer usably.

    Raises
    ------
    None

    Examples
    --------
    >>> reply = '{"topics": ["water_content"], "only_these": false}'
    >>> answer = llm_figure_topics('estimate water content', lambda p: reply)
    >>> sorted(answer['topics']), answer['only_these'], answer['style']
    (['water_content'], False, {})
    >>> llm_figure_topics('x', lambda p: 'no idea') is None
    True
    """
    if not str(user_request or "").strip():
        return None
    try:
        reply = query(FIGURES_PROMPT.replace("{request}", str(user_request)))
    except Exception:  # noqa: BLE001 - no key, no network, rate limit: all "no answer"
        return None
    answer = _parse_json_object(reply)
    if not isinstance(answer, dict):
        return None
    raw = answer.get("topics")
    if not isinstance(raw, (list, tuple)):
        return None
    topics = {str(t).strip().lower() for t in raw if str(t).strip().lower()
              in FIGURE_TOPICS}
    if not topics:
        return None
    return {"topics": topics, "only_these": bool(answer.get("only_these")),
            "style": _style_request(answer.get("style"))}


def _style_request(style: Any) -> Dict[str, Any]:
    """Style preferences the request expressed, in the shape the style reads.

    A colormap is not assigned to a quantity here: "use cividis" means the
    magnitude plots, since a diverging map is what a change plot needs and
    replacing it with a sequential one would hide the sign of the change.
    :func:`PyHydroGeophysX.agents._figstyle.style_from_config` drops anything
    it does not recognise, so a name the model invented changes nothing.

    Examples
    --------
    >>> _style_request({'size': 'large', 'colormap': 'cividis'})
    {'size': 'large', 'resistivity_cmap': 'cividis', 'water_content_cmap': 'cividis'}
    >>> _style_request({'size': None, 'colormap': None})
    {}
    >>> _style_request('not a dict')
    {}
    """
    if not isinstance(style, dict):
        return {}
    out: Dict[str, Any] = {}
    size = style.get("size")
    if isinstance(size, str) and size.strip():
        out["size"] = size.strip().lower()
    colormap = style.get("colormap")
    if isinstance(colormap, str) and colormap.strip():
        out["resistivity_cmap"] = colormap.strip()
        out["water_content_cmap"] = colormap.strip()
    return out


def _parse_json_object(reply: Any) -> Any:
    """The first JSON object in ``reply``, or None."""
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


def plan_figures(available: Sequence[str], user_request: str = "",
                 query: Optional[Callable[[str], str]] = None,
                 topics: Optional[Dict[str, Any]] = None) -> Dict[str, List[str]]:
    """Which figures to show, which to leave out, and which are missing.

    Parameters
    ----------
    available : sequence of str
        Figure keys the run actually produced, as stored in ``vis_files``.
    user_request : str, optional
        The request in the user's own words. Without it nothing can be judged,
        so every available figure is drawn.
    query : callable, optional
        The model, as taken by :func:`llm_figure_topics`. Omit to skip the
        model entirely - useful when the caller has already asked.
    topics : dict, optional
        An answer already obtained from :func:`llm_figure_topics`, to avoid a
        second call.

    Returns
    -------
    dict
        ``{"draw": [...], "omitted": [...], "missing": [...]}``. ``draw`` is
        ordered with the requested topics first. ``missing`` holds topic names,
        not figure keys: the user asked to see something the run has no data
        for, which is a fact about the run rather than about plotting.

    Raises
    ------
    None

    Examples
    --------
    >>> plan = plan_figures(['baseline_resistivity', 'timelapse_water_content'])
    >>> plan['draw']
    ['baseline_resistivity', 'timelapse_water_content']
    >>> asked = {'topics': {'water_content'}, 'only_these': True}
    >>> plan = plan_figures(['baseline_resistivity', 'timelapse_water_content'],
    ...                     topics=asked)
    >>> plan['draw'], plan['omitted']
    (['timelapse_water_content'], ['baseline_resistivity'])
    >>> plan_figures([], topics={'topics': {'water_content'}, 'only_these': False})
    {'draw': [], 'omitted': [], 'missing': ['water_content']}
    """
    keys = [k for k in available if k]
    if topics is None and query is not None:
        topics = llm_figure_topics(user_request, query)
    if not topics or not topics.get("topics"):
        # Nothing was established about intent, so nothing is suppressed.
        return {"draw": list(keys), "omitted": [], "missing": []}

    wanted = set(topics["topics"])
    only = bool(topics.get("only_these"))

    def topic_of(key: str) -> str:
        return FIGURE_CATALOG.get(key, {}).get("topic", "")

    requested = [k for k in keys if topic_of(k) in wanted]
    others = [k for k in keys if topic_of(k) not in wanted]
    draw = requested + ([] if only else others)
    omitted = others if only else []
    present = {topic_of(k) for k in keys}
    missing = sorted(t for t in wanted if t not in present)
    return {"draw": draw, "omitted": omitted, "missing": missing}


def missing_figure_warnings(missing: Sequence[str]) -> List[str]:
    """One sentence per topic the user wanted to see and the run cannot show.

    Parameters
    ----------
    missing : sequence of str
        Topic names, as returned by :func:`plan_figures`.

    Returns
    -------
    list of str
        Warnings for the run's warning list, empty when nothing is missing.

    Raises
    ------
    None

    Examples
    --------
    >>> missing_figure_warnings(['water_content'])
    ['The request asked to see water content, moisture or saturation, but the run produced no data to plot for it.']
    >>> missing_figure_warnings([])
    []
    """
    return [f"The request asked to see {FIGURE_TOPICS.get(topic, topic)}, but "
            f"the run produced no data to plot for it."
            for topic in missing]
