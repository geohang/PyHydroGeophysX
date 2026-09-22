"""What the time-lapse inversion actually does, in one place.

The reports described the run as a *difference inversion*, and it is not one.
``time_lapse_method`` is read in
:meth:`PyHydroGeophysX.agents.ert_inversion_agent.ERTInversionAgent.run_time_lapse_inversion`
only to be written to the log; it never reaches
:class:`PyHydroGeophysX.inversion.time_lapse.TimeLapseERTInversion`, which
implements exactly one scheme and applies it regardless.

That scheme inverts every survey at once. Its objective carries a data-misfit
term per survey, a spatial smoothness term ``lambda * ||W_m m||^2``, and a
temporal term ``alpha * ||W_t m||^2`` in which ``W_t`` takes first differences
between the model blocks of adjacent time steps (see
``_sparse_temporal_difference_matrix``), optionally decayed by the interval
between surveys. The per-iteration objective the solver records,
``[chi2_ert, phi_m, phi_t]``, is those three terms.

A difference inversion is a different thing: it inverts the *change* in the
data relative to a baseline survey against a fixed baseline model, one time
step at a time, and has no term coupling the steps to one another. Reporting
one as the other misstates what produced the numbers - it implies each time
step was solved independently against the baseline, when in fact every survey
constrained every other through ``W_t``, which is precisely why the recovered
changes are as smooth in time as they are.

So the label is taken from here rather than from the configuration, and a
configuration that asks for a scheme this package does not implement is
reported as an unhonoured request rather than quietly relabelled.
"""

from typing import Any, Dict, Optional, Tuple

#: The only time-lapse scheme this package implements.
IMPLEMENTED_SCHEME = "temporal_constraint"

#: How to name it in a report heading or a one-line summary.
SCHEME_LABEL = "Simultaneous inversion with temporal constraints (4D)"

#: How to name it where only a few words fit.
SCHEME_SHORT_LABEL = "Temporally constrained (4D)"

SCHEME_DESCRIPTION = (
    "All surveys were inverted simultaneously in a single 4D problem rather "
    "than one at a time. The objective function combines a data-misfit term "
    "for each survey, a spatial smoothness constraint (lambda) applied within "
    "each time step, and a temporal constraint (alpha) applied to the "
    "first differences between the models of successive time steps. The "
    "temporal term penalises change between consecutive surveys, so a change "
    "is recovered only where the data require it; this suppresses the "
    "inversion artefacts that appear when independently inverted models are "
    "differenced, at the cost of damping genuine abrupt change. Recovered "
    "changes are therefore not independent between time steps, and the "
    "magnitude of a change depends on the value of alpha as well as on the "
    "data."
)

#: Schemes a configuration may name that this package does not implement. The
#: text of each says what it would have done, so a report can state plainly
#: what was asked for and what ran instead.
ALTERNATIVE_SCHEMES = {
    "difference": "inverts the change in the data relative to a baseline "
                  "survey, one time step at a time, with no constraint "
                  "coupling the time steps",
    "ratio": "inverts the ratio of each survey to the baseline, one time step "
             "at a time",
    "joint": "a separate joint-inversion formulation across data types",
}

#: Configuration values that mean "the scheme that is implemented".
_ALIASES = {
    IMPLEMENTED_SCHEME, "temporal", "temporally_constrained", "4d", "fourd",
    "simultaneous", "temporal_regularization", "temporal_regularisation",
}


def resolve_scheme(requested: Optional[str]) -> Tuple[str, Optional[str]]:
    """The scheme that ran, and a note when it is not the one asked for.

    Parameters
    ----------
    requested : str or None
        ``time_lapse_method`` from the workflow configuration. None or blank
        means nothing was asked for, which is not a mismatch.

    Returns
    -------
    tuple
        ``(scheme_id, note)``. ``scheme_id`` is always
        :data:`IMPLEMENTED_SCHEME` - it is what the solver does. ``note`` is a
        sentence naming the discrepancy, or None when there is none.

    Raises
    ------
    None

    Examples
    --------
    >>> resolve_scheme('temporal_constraint')
    ('temporal_constraint', None)
    >>> scheme, note = resolve_scheme('difference')
    >>> note.startswith('The configuration requested a difference inversion')
    True
    >>> resolve_scheme(None)
    ('temporal_constraint', None)
    """
    name = (requested or "").strip().lower().replace("-", "_").replace(" ", "_")
    if not name or name in _ALIASES:
        return IMPLEMENTED_SCHEME, None
    behaviour = ALTERNATIVE_SCHEMES.get(name)
    if behaviour is None:
        note = (f"The configuration requested a time-lapse scheme named "
                f"'{requested}', which this package does not implement. "
                f"{SCHEME_LABEL} was used instead.")
    else:
        note = (f"The configuration requested a {name} inversion, which "
                f"{behaviour}. This package does not implement it; "
                f"{_scheme_lower()} was used instead, so the time steps are "
                f"coupled to one another and the recovered changes are not "
                f"independent.")
    return IMPLEMENTED_SCHEME, note


def _scheme_lower() -> str:
    """The scheme label lower-cased for mid-sentence use."""
    return SCHEME_LABEL[0].lower() + SCHEME_LABEL[1:]


def scheme_note(config: Dict[str, Any]) -> Optional[str]:
    """The mismatch note for a workflow configuration, or None.

    Parameters
    ----------
    config : dict
        Workflow configuration; ``time_lapse_method`` is read.

    Returns
    -------
    str or None
        A sentence for the report's warnings, or None when the configuration
        asked for what actually runs.

    Raises
    ------
    None

    Examples
    --------
    >>> scheme_note({'time_lapse_method': 'temporal_constraint'}) is None
    True
    >>> 'difference inversion' in scheme_note({'time_lapse_method': 'difference'})
    True
    """
    if not isinstance(config, dict):
        return None
    return resolve_scheme(config.get("time_lapse_method"))[1]
