"""What the loop does about a step that failed.

Before this, a failed step was recorded and the run moved on. That is right for
a step nothing can be done about, and wrong for the ones that fail because a
setting was unsuitable rather than because the work is impossible - an inversion
that diverged at the configured regularisation, a conversion given a threshold
no cell meets, a loader pointed at the wrong column layout. A person looking at
the error would change one number and run it again, and the loop could not.

The inversion already had a hand-written version of this: evaluate the result,
adjust lambda, invert again. That special case is what this generalises.

Three things are deliberately separated:

- **Diagnosis** is the model's, because reading an error message and saying what
  it means is what a model is for. It never runs anything.
- **The decision** is bounded here: the only recovery this performs on its own
  is re-running the same tool with different *configuration*. Changing numbers
  is reversible, inspectable, and recorded in the plan; it cannot reach outside
  the run.
- **Anything beyond configuration** - a generated reader for a coordinate file
  nobody anticipated - is a proposal that goes to the user through
  :meth:`RunContext.ask` with the source in front of them, and runs only on an
  explicit yes. See :mod:`~PyHydroGeophysX.agents.runtime.adapters`. Code a
  model wrote, executed unattended against somebody's data, is not a thing this
  package does on its own initiative: a run with nobody to ask declines.

A recovery is itself a step in the transcript, with the diagnosis as its reason,
so a reader of the report sees what failed, what the run thought about it, what
it changed, and whether that worked.
"""

import json
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from .context import RunContext

#: How many times one tool may be recovered within a run. One retry catches the
#: unsuitable-setting case; a tool that fails twice with different settings is
#: failing for a reason settings will not fix, and going round again only spends
#: the user's time and tokens.
MAX_RECOVERIES = 1

#: Configuration keys a diagnosis is allowed to change. Everything here is a
#: number or a switch that selects between behaviours already implemented and
#: tested. Absent on purpose: every path key, because a recovery that can point
#: the run at a different file is a recovery that can read anything on the disk,
#: and the user chose those files.
ADJUSTABLE = (
    "inversion_params", "temporal_regularization", "time_lapse_method",
    "quality_threshold", "max_attempts", "auto_adjust",
    "velocity_threshold", "seismic_inversion_params", "align_origin",
    "petrophysical_params", "n_realizations", "coverage_threshold",
    "instrument", "crs", "first_break_params", "tdem_params",
)

RECOVERY_PROMPT = """A step of a hydrogeophysical workflow failed. Say what to do.

The run so far:
{transcript}

The step that failed: {tool}
What it is for: {purpose}
The error: {error}

The settings it ran with (only these may be changed):
{settings}

Coordinate files in this run that a reader could be written for, if one of
them is what could not be read: {adaptable}

Answer with JSON and nothing else:
{{"cause": "one sentence on why this failed",
  "action": "retry" | "adapt" | "give_up",
  "changes": {{"<setting>": <new value>}},
  "file_key": "<one of the coordinate files above, for adapt only>",
  "why": "one sentence on why this should work"}}

Rules:
- "retry" only when a different setting could plausibly fix it. Re-running with
  the same settings is never worth doing.
- "adapt" only when a coordinate file above holds the right numbers in a layout
  the reader does not handle. Name it in "file_key". A human will read the
  reader you would write and decide whether to run it, so do not propose this
  for a file whose contents are simply wrong or missing.
- "give_up" when the failure is about missing data, an unreadable instrument
  format, an unimplemented method, or anything neither of the above can change.
  This is a normal answer, not a defeat; say the cause plainly so the report can
  state it.
- Only the settings listed above may appear in "changes". Keep the same types.
- Change as little as possible: one or two values, not a new configuration."""


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """The first JSON object in ``text``, or None.

    Models wrap JSON in prose and in code fences however they are asked not to.

    Examples
    --------
    >>> _extract_json('```json\\n{"action": "retry"}\\n```')
    {'action': 'retry'}
    >>> _extract_json('no json here') is None
    True
    """
    if not text:
        return None
    match = re.search(r"\{.*\}", str(text), re.S)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _merge(current: Any, change: Any) -> Any:
    """Apply ``change`` to ``current``, merging one level of dict.

    A diagnosis that wants a different lambda says ``{"inversion_params":
    {"lambda": 5}}`` and means "that one key", not "replace every inversion
    setting with this".

    Examples
    --------
    >>> _merge({'lambda': 15, 'max_iterations': 10}, {'lambda': 5})
    {'lambda': 5, 'max_iterations': 10}
    >>> _merge(15.0, 5.0)
    5.0
    """
    if isinstance(current, dict) and isinstance(change, dict):
        merged = dict(current)
        merged.update(change)
        return merged
    return change


def settings_for(ctx: RunContext) -> Dict[str, Any]:
    """The adjustable part of the configuration, as the diagnosis sees it."""
    return {key: ctx.config[key] for key in ADJUSTABLE if key in ctx.config}


def diagnose(ctx: RunContext, tool: Any, error: str,
             ask: Optional[Callable[[str], str]]) -> Dict[str, Any]:
    """Ask what went wrong and what to change, and keep only what is allowed.

    Parameters
    ----------
    ctx : RunContext
        The run, for the transcript and the current settings.
    tool : Tool or None
        The tool that failed.
    error : str
        Its error message.
    ask : callable or None
        Takes a prompt, returns the model's reply. Without one there is no
        diagnosis: a deterministic run does not guess at causes.

    Returns
    -------
    dict
        ``{"cause", "action", "changes", "why"}``. ``action`` is ``"retry"``
        only when a diagnosis asked for it *and* named at least one allowed
        setting that would actually differ; otherwise ``"give_up"``, so a model
        that answers "retry" with nothing to change cannot loop the run.

    Raises
    ------
    None
        An unreachable or unparseable model gives up, which is what the loop
        did before any of this existed.

    Examples
    --------
    >>> ctx = RunContext('goal', {'inversion_params': {'lambda': 15}})
    >>> reply = '{"cause": "lambda too high", "action": "retry",' \\
    ...         ' "changes": {"inversion_params": {"lambda": 5}},' \\
    ...         ' "why": "less smoothing"}'
    >>> plan = diagnose(ctx, None, 'diverged', lambda prompt: reply)
    >>> plan['action'], plan['changes']
    ('retry', {'inversion_params': {'lambda': 5}})
    >>> diagnose(ctx, None, 'diverged', None)['action']
    'give_up'
    """
    blank = {"cause": "", "action": "give_up", "changes": {}, "file_key": "",
             "why": ""}
    if not callable(ask):
        return blank
    settings = settings_for(ctx)
    from .adapters import ADAPTABLE_FILES

    present = [key for key in ADAPTABLE_FILES if ctx.config.get(key)]
    prompt = RECOVERY_PROMPT.format(
        transcript=ctx.transcript(limit=12),
        tool=getattr(tool, "name", "") or "the step",
        purpose=getattr(tool, "description", "") or "(not described)",
        error=str(error)[:800],
        settings=json.dumps(settings, indent=2, default=str)[:2000],
        adaptable=", ".join(present) or "(none)")
    try:
        parsed = _extract_json(ask(prompt))
    except Exception:  # noqa: BLE001 - an unreachable model is not a crash
        parsed = None
    if not parsed:
        return blank
    # Only the allowed keys, and only values that would actually differ. A
    # "retry" that changes nothing is a re-run of the thing that just failed.
    changes = {key: value for key, value in (parsed.get("changes") or {}).items()
               if key in ADJUSTABLE
               and _merge(ctx.config.get(key), value) != ctx.config.get(key)}
    wanted = str(parsed.get("action"))
    file_key = str(parsed.get("file_key") or "")
    if wanted == "adapt" and file_key in ADAPTABLE_FILES and ctx.config.get(file_key):
        action = "adapt"
    elif wanted == "retry" and changes:
        action = "retry"
    else:
        action = "give_up"
        file_key = ""
    return {"cause": str(parsed.get("cause") or "")[:300],
            "action": action,
            "changes": changes,
            "file_key": file_key,
            "why": str(parsed.get("why") or "")[:300]}


def apply_changes(ctx: RunContext, changes: Dict[str, Any]) -> List[str]:
    """Put the diagnosis's settings into the configuration, and say which.

    Returns
    -------
    list of str
        ``"key: old -> new"`` per change, for the transcript. The run's
        configuration is what the report prints, so a number that changed
        mid-run has to be visible as having changed.

    Examples
    --------
    >>> ctx = RunContext('goal', {'inversion_params': {'lambda': 15}})
    >>> apply_changes(ctx, {'inversion_params': {'lambda': 5}})
    ["inversion_params: {'lambda': 15} -> {'lambda': 5}"]
    """
    applied = []
    for key, value in (changes or {}).items():
        if key not in ADJUSTABLE:
            continue
        before = ctx.config.get(key)
        after = _merge(before, value)
        ctx.config[key] = after
        applied.append(f"{key}: {before} -> {after}")
    return applied


def recover(ctx: RunContext, tool: Any, error: str,
            ask: Optional[Callable[[str], str]],
            run: Callable[[str], str],
            attempts: Optional[Dict[str, int]] = None) -> Tuple[str, str]:
    """Diagnose a failed step and, if a setting can be blamed, run it again.

    Parameters
    ----------
    ctx : RunContext
        The run. Its configuration is what a recovery changes.
    tool : Tool
        The tool that failed.
    error : str
        Its error.
    ask : callable or None
        The model, for the diagnosis.
    run : callable
        Takes the tool's name and runs it again, returning the new status. The
        controller passes its own ``invoke`` so the retry is an ordinary step,
        recorded like any other.
    attempts : dict, optional
        Tool name to recoveries already spent, carried across a run.

    Returns
    -------
    tuple
        ``(status, note)`` - the status after any retry, and a sentence for the
        warnings. ``status`` is the original ``"failed"`` when nothing was tried.

    Raises
    ------
    None

    Examples
    --------
    >>> from .tools import Tool
    >>> ctx = RunContext('goal', {'inversion_params': {'lambda': 15}})
    >>> tool = Tool('invert', 'Invert.', lambda c: ('', {}),
    ...             label='Run ERT inversion')
    >>> reply = '{"cause": "too smooth", "action": "retry",' \\
    ...         ' "changes": {"inversion_params": {"lambda": 5}}, "why": "ok"}'
    >>> recover(ctx, tool, 'diverged', lambda p: reply, lambda name: 'ok')
    ('ok', "Run ERT inversion failed (diverged). Diagnosis: too smooth. Retried with inversion_params: {'lambda': 15} -> {'lambda': 5}, which worked.")
    """
    spent = attempts if attempts is not None else {}
    name = getattr(tool, "name", "") or ""
    if spent.get(name, 0) >= MAX_RECOVERIES:
        return "failed", ""
    plan = diagnose(ctx, tool, error, ask)
    label = getattr(tool, "label", "") or name or "the step"
    if plan["action"] == "adapt":
        # A coordinate file whose layout nobody wrote a reader for. The model
        # writes one, the user reads it and decides, and only then does anything
        # run. Declining is the default and a perfectly good outcome.
        from .adapters import adapt_file

        spent[name] = spent.get(name, 0) + 1
        key = plan["file_key"]
        rewritten = adapt_file(ctx, key, error, ask)
        if not rewritten:
            return "failed", (f"{label} failed ({error}). Diagnosis: "
                              f"{plan['cause']} A reader for {key} was not run, "
                              f"so the step was left undone.")
        original = ctx.config.get(key)
        ctx.config[key] = rewritten
        status = run(name)
        outcome = "which worked" if status == "ok" else "which did not help either"
        return status, (f"{label} failed ({error}). Diagnosis: {plan['cause']} "
                        f"{key} was re-read by approved generated code and the "
                        f"step retried against {rewritten} in place of "
                        f"{original}, {outcome}.")
    if plan["action"] != "retry":
        if plan["cause"]:
            return "failed", (f"{label} failed ({error}). Diagnosis: "
                              f"{plan['cause']} Nothing in the settings would "
                              f"change that, so it was left undone.")
        return "failed", ""
    spent[name] = spent.get(name, 0) + 1
    applied = apply_changes(ctx, plan["changes"])
    ctx.note(f"{label} was retried with changed settings after it failed: "
             f"{'; '.join(applied)}. Reason given: {plan['why'] or plan['cause']}")
    status = run(name)
    outcome = "which worked" if status == "ok" else "which did not help either"
    return status, (f"{label} failed ({error}). Diagnosis: {plan['cause']}. "
                    f"Retried with {'; '.join(applied)}, {outcome}.")
