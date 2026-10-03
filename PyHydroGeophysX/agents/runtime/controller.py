"""The loop: read the transcript, choose one action, observe, choose again.

This is the whole difference from the pipeline it replaces. A pipeline decides
everything before it starts; a loop decides one step at a time, with the
results of the previous steps in front of it. That is what lets a run react -
to an inversion that did not converge, to a mesh with no layering, to a file
that turned out to hold something other than what its name suggested.

Two properties matter more than the loop itself:

**It always terminates, and always produces something.** A model that keeps
choosing badly is bounded by ``max_steps``; a model that cannot be reached at
all falls through to :func:`next_by_policy`, which walks the tools in
dependency order. A run without an API key therefore behaves exactly like the
old pipeline - it simply stops adapting. This matters because the previous
design's failure mode was an exception mid-sequence that lost the whole run.

**A refusal is data.** When the controller names a tool that cannot run, or one
that does not exist, that becomes a recorded step with the reason, and the loop
continues. The model sees its own mistake in the transcript on the next turn,
which is how it corrects. Raising instead would end the run over a typo.
"""

import json
from typing import Any, Callable, Dict, List, Optional

from ..._internal.utils import parse_json_object
from . import steering
from .context import PROJECTED, RunContext
from .tools import TOOLS, invoke, menu, runnable_tools

#: The artifact that has to exist before the controller may finish. AQUAH's
#: runs end with a report; an assistant whose last product is something else -
#: a reviewed report, say - passes its own key as ``finish``.
DEFAULT_FINISH = "report_files"

#: A run that has taken this many steps is looping rather than progressing.
#: Generous: a time-lapse run with a retry and a report is about eight.
DEFAULT_MAX_STEPS = 24

CONTROLLER_PROMPT = """You are running a geophysics processing workflow. Choose \
the single next action.

{transcript}

Actions you can take now:
{menu}

Reply with JSON only - no prose, no code fence. Give your reasoning first:
{{"why": "<one or two sentences: what the run has shown so far, and so what to do next>", "tool": "<name from the list>"}}
or, when the goal has been met and a report has been written:
{{"why": "<one short sentence>", "done": true}}

Rules:
- Choose only from the list above. A tool not listed cannot run yet, usually
  because something it needs has not been produced.
- Produce everything the goal asks for before finishing. If the user asked for
  water content, a resistivity model alone does not meet the goal.
- Write the report last, once the products it should describe exist.
- If a step failed, do not repeat it unchanged. Either try the alternative that
  addresses the failure, or move on and let the report state what is missing.
- Prefer finishing over exploring. This is a processing run, not an investigation.
"""


def _parse_choice(reply: Any) -> Optional[Dict[str, Any]]:
    """The controller's decision, or None when the reply is unusable.

    ``settings`` is kept when the reply carries a dict of them: it is how a
    decision follows a note from the user (:data:`STEERING_PROMPT`).
    """
    answer = parse_json_object(reply)
    if not isinstance(answer, dict):
        return None
    extra = ({"settings": answer["settings"]}
             if isinstance(answer.get("settings"), dict) and answer["settings"] else {})
    if answer.get("done"):
        return {"done": True, "why": str(answer.get("why", "")), **extra}
    tool = answer.get("tool")
    if not isinstance(tool, str) or not tool.strip():
        return None
    return {"tool": tool.strip(), "why": str(answer.get("why", "")), **extra}


#: Added to the decision prompt when the user has just said something. The
#: settings it may change are the recovery's (:data:`~.recovery.ADJUSTABLE`):
#: numbers and switches between behaviours already implemented, never a path.
STEERING_PROMPT = """
The user has just told you, while the run is going:
{notes}
Follow it where you can. If it asks for a different setting, add the changed
keys to your JSON as "settings", using these current values as the template
(change only what the user asked for):
{settings}
If it asks for something none of the actions or settings can do, say so in
"why" and carry on with the best next action.
"""


def next_by_policy(ctx: RunContext,
                   tools: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """The next tool when there is no model to ask.

    Takes the first tool whose requirements are met and which has not already
    run. Registration order is dependency order, so this reproduces the fixed
    pipeline - which is the right fallback: without a model the run should
    still do the obvious thing, it just cannot reconsider.

    Parameters
    ----------
    ctx : RunContext
        The run as it stands.

    Returns
    -------
    str or None
        A tool name, or None when nothing can run.

    Raises
    ------
    None

    Examples
    --------
    >>> from .tools import Tool
    >>> ctx = RunContext('x')
    >>> only = {'first': Tool('first', 'One.', lambda c: ('', {'a': 1}),
    ...                       produces=('a',)),
    ...         'second': Tool('second', 'Two.', lambda c: ('', {}),
    ...                        requires=('a',))}
    >>> next_by_policy(ctx, only)
    'first'
    >>> ctx.put('a', 1)
    >>> _ = ctx.begin('first'); ctx.finish()
    >>> next_by_policy(ctx, only)
    'second'
    """
    for tool in runnable_tools(ctx, tools):
        return tool.name
    return None


def route_ahead(ctx: RunContext, tools: Optional[Dict[str, Any]] = None,
                finish: str = DEFAULT_FINISH, limit: int = 12) -> List[Dict[str, str]]:
    """The steps still between the run as it stands and its final product.

    Worked out on a copy of the run: :func:`next_by_policy` takes a step, the
    step is assumed to make what it declares, and so on until ``finish``
    exists or nothing more can run. It is the dependency order applied to
    what the run actually has, not a promise - the controller may choose
    differently, a step may fail - which is why the desktop recomputes it after
    every step and calls it the route *as things stand*. The products of a
    projected step are :data:`~.context.PROJECTED`, which the ``when`` gates
    that look inside an artifact recognise.

    Parameters
    ----------
    ctx : RunContext
        The run as it stands; not modified.
    tools : dict, optional
        The registry to project over; the global one by default.
    finish : str
        The artifact that ends the run.
    limit : int
        At most this many steps are projected.

    Returns
    -------
    list of dict
        ``{"tool", "label", "module"}`` per step, in order. Empty once the
        final product exists, or when nothing can run.

    Raises
    ------
    None

    Examples
    --------
    >>> from .tools import Tool
    >>> chain = {'load': Tool('load', 'Load.', lambda c: ('', {}), produces=('data',),
    ...                       label='Load data'),
    ...          'report': Tool('report', 'Report.', lambda c: ('', {}),
    ...                         requires=('data',), produces=('report_files',),
    ...                         label='Write the report')}
    >>> [step['label'] for step in route_ahead(RunContext('x'), chain)]
    ['Load data', 'Write the report']
    >>> ctx = RunContext('x')
    >>> ctx.put('data', [1])
    >>> _ = ctx.begin('load'); ctx.finish()
    >>> [step['tool'] for step in route_ahead(ctx, chain)]
    ['report']
    """
    goal = finish or DEFAULT_FINISH
    shadow = RunContext(ctx.goal, ctx.config, ctx.output_dir)
    shadow.artifacts = dict(ctx.artifacts)
    shadow.steps = list(ctx.steps)
    projected: List[Dict[str, str]] = []
    while len(projected) < limit and not shadow.has(goal):
        planned = {step["tool"] for step in projected}
        # A repeatable tool stays on offer after it has run; on paper it runs once.
        options = [tool for tool in runnable_tools(shadow, tools)
                   if tool.name not in planned]
        if not options:
            break
        tool = options[0]
        shadow.begin(tool.name, "projected")
        shadow.finish()
        for key in tool.produces:
            if not shadow.has(key):
                shadow.artifacts[key] = PROJECTED
        projected.append({"tool": tool.name, "label": tool.label or tool.name,
                          "module": tool.module or ""})
    return projected


#: What an ``on_step`` hook may answer. ``proceed`` is the default for any
#: other value, including None, so a hook that forgets to return cannot stall a
#: run.
PROCEED, SKIP, STOP = "proceed", "skip", "stop"

#: How the loop ended, as ``RunContext.ended`` records it: the model finished
#: once a report existed, nothing more could run, the user stopped the run, or
#: it used every step it was allowed.
FINISHED, EXHAUSTED, STOPPED, STEP_LIMIT = "finished", "exhausted", "stopped", "step_limit"


def run_controller(ctx: RunContext, ask: Optional[Callable[[str], str]] = None,
                   max_steps: int = DEFAULT_MAX_STEPS,
                   progress: Optional[Callable[[str, float, str], None]] = None,
                   tools: Optional[Dict[str, Any]] = None,
                   on_step: Optional[Callable[..., Optional[str]]] = None,
                   on_result: Optional[Callable[..., None]] = None,
                   recovery: bool = True,
                   prompt: Optional[str] = None,
                   finish: str = DEFAULT_FINISH
                   ) -> RunContext:
    """Drive the run to completion, one chosen action at a time.

    Parameters
    ----------
    ctx : RunContext
        The run. Mutated in place and returned.
    ask : callable, optional
        Takes a prompt, returns the model's reply. Omit to run the
        deterministic policy - which is what happens without an API key.
    max_steps : int
        Hard bound on iterations, so a model that keeps choosing badly still
        terminates with whatever it has produced.
    progress : callable, optional
        ``(step, fraction, detail)``, called before each action so the desktop
        app can show the run advancing.
    on_step : callable, optional
        ``(ctx, tool, reason) -> 'proceed' | 'skip' | 'stop'``, called before
        each action with the tool about to run. This is the single difference
        between the app's two modes: "auto to report" supplies a hook that
        notes the tool's module and returns immediately, while "step by step"
        supplies one that shows the user what is about to happen and waits.
        Both drive the same loop over the same tools, so a step behaves
        identically whichever way it was reached - the previous design had two
        separate code paths that could and did drift.
    recovery : bool
        Whether a failed step may be diagnosed and retried once with different
        settings. On by default, because the alternative - recording the failure
        and moving on - loses runs that one changed number would have saved.
        Only configuration is ever changed, never code and never a file path;
        see :mod:`~PyHydroGeophysX.agents.runtime.recovery`.
    on_result : callable, optional
        ``(ctx, step)``, called after each action with the step it finished.
        ``on_step`` fires *before* a step and so can only say what is about to
        happen; this is where what a step actually concluded becomes available
        to a caller while the run is still going.
    prompt : str, optional
        The decision prompt, with ``{transcript}`` and ``{menu}`` fields.
        Defaults to :data:`CONTROLLER_PROMPT`; an assistant for another domain
        passes its own framing.
    finish : str
        The artifact that must exist before "done" is accepted
        (:data:`DEFAULT_FINISH`, the report, unless the assistant says otherwise).

    Returns
    -------
    RunContext
        The same context, carrying the steps taken, the artifacts produced, any
        warnings raised and, in ``ended``, how the loop ended. A model's "done"
        ends it only once a report exists; before that the next available step
        is taken instead.

    Raises
    ------
    None
        Tool failures are recorded, not raised: the point of the loop is to
        keep deciding after one.

    Examples
    --------
    >>> from .tools import Tool
    >>> only = {'load': Tool('load', 'Load.',
    ...                      lambda c: ('Loaded 3 files.', {'data': [1, 2, 3]}),
    ...                      produces=('data',)),
    ...         'report': Tool('report', 'Report.', lambda c: ('Wrote it.', {}),
    ...                        requires=('data',))}
    >>> ctx = run_controller(RunContext('process my data'), tools=only)
    >>> [s.tool for s in ctx.steps]
    ['load', 'report']
    >>> ctx.steps[0].summary
    'Loaded 3 files.'
    >>> ctx.ended
    'exhausted'
    """
    # Recoveries spent per tool, so one bad setting cannot be retried forever.
    spent: Dict[str, int] = {}
    steer = steering.current.get()
    for index in range(max_steps):
        # Between steps is where the user's pause and notes take effect: the
        # step that was running has finished, and nothing new has started.
        if steer is not None and not steer.checkpoint(
                ctx.steps[-1].description or ctx.steps[-1].tool if ctx.steps else ""):
            ctx.ended = STOPPED
            break
        notes = _take_notes(ctx, steer)
        options = runnable_tools(ctx, tools)
        if not options:
            ctx.ended = EXHAUSTED
            break
        choice = _decide(ctx, ask, tools, prompt, finish, notes)
        if notes:
            _answer_notes(ctx, steer, notes, choice, heard=ask is not None)
        if choice is not None and choice.get("done") and not _may_finish(ctx, finish):
            # "done" is a valid answer only once a report exists: the prompt
            # says so and forced_choice counts on it. Taken at its word here, a
            # model answering "done" with steps on offer ended the run on the
            # spot - nothing run, no report, reported as a success. The step
            # taken instead records why, which is how the model reads the
            # refusal on its next turn.
            name = next_by_policy(ctx, tools)
            choice = None if name is None else {
                "tool": name,
                "why": "the model answered 'done' before a report was written, "
                       "so the next available step was taken"}
        if choice is None or choice.get("done"):
            ctx.ended = FINISHED if choice is not None else EXHAUSTED
            break
        name = choice["tool"]
        tool = (TOOLS if tools is None else tools).get(name)
        reason = choice.get("why", "")
        if tool is not None and name not in {t.name for t in options}:
            # Only what the menu offered may run. `invoke` checks requirements
            # and nothing else, so a reply naming an unlisted tool ran it: a
            # single-survey inversion replaced a time-lapse result, a climate
            # retrieval ran with climate switched off. Recorded as blocked - it
            # never ran - so the model reads why on its next turn, and the tool
            # stays on offer for when it does apply.
            why = tool.unavailable_because(ctx)
            ctx.begin(name, reason, agent=tool.agent,
                      description=tool.label or name, purpose=tool.description)
            ctx.finish(status="blocked",
                       error=f"Not offered, so not run: {why}. Choose one of: "
                             f"{', '.join(t.name for t in options)}.")
            continue

        verdict = PROCEED
        if on_step is not None:
            try:
                verdict = on_step(ctx, tool, reason) or PROCEED
            except Exception:  # noqa: BLE001 - a UI hook must not stop a run
                verdict = PROCEED
        if verdict == STOP:
            ctx.begin(name, reason, agent=getattr(tool, "agent", ""),
                      description=getattr(tool, "label", "") or name)
            ctx.finish(status="skipped", error="Stopped here at the user's request.")
            ctx.ended = STOPPED
            break
        if verdict == SKIP:
            # Recorded, not silently dropped: a step the user chose to skip is
            # a fact about the run, and the report has to be able to say the
            # product it would have made is missing.
            ctx.begin(name, reason, agent=getattr(tool, "agent", ""),
                      description=getattr(tool, "label", "") or name)
            ctx.finish(status="skipped", error="Skipped at the user's request.")
            continue

        if progress is not None:
            fraction = min(0.95, (index + 1) / float(max_steps))
            try:
                progress(tool.label if tool and tool.label else name, fraction, reason)
            except Exception:  # noqa: BLE001 - a UI callback must not stop a run
                pass
        status = invoke(ctx, name, reason, tools)
        if status == "failed" and recovery is not False:
            # A step that failed because a setting was unsuitable is worth one
            # more go with a different setting; a person reading the error would
            # do the same. The retry is an ordinary recorded step, so the plan
            # shows both attempts and what changed between them.
            from .recovery import recover

            status, note = recover(
                ctx, tool, ctx.steps[-1].error, ask,
                lambda again, why=reason: invoke(
                    ctx, again, f"retried after it failed; {why}", tools),
                spent)
            if note:
                ctx.note(note)
        if on_result is not None and ctx.steps:
            try:
                on_result(ctx, ctx.steps[-1])
            except Exception:  # noqa: BLE001 - a UI hook must not stop a run
                pass
    else:
        ctx.ended = STEP_LIMIT
    return ctx


def forced_choice(ctx: RunContext,
                  tools: Optional[Dict[str, Any]] = None,
                  finish: str = DEFAULT_FINISH) -> Optional[Dict[str, Any]]:
    """The decision to take without asking, when the menu leaves only one.

    The model may answer with a tool from the menu, or with "done" once a
    report has been written (:data:`CONTROLLER_PROMPT`). With one tool on the
    menu and no report yet there is exactly one valid answer, and asking for it
    cost a full model request: three of the six decisions in an ERT-to-water-
    content run - load, invert, write the report - were of this kind. Taking it
    directly is the same decision, and the step says so, so the plan and the
    transcript show that nothing was chosen.

    Parameters
    ----------
    ctx : RunContext
        The run as it stands.
    tools : dict, optional
        The registry to choose from; the global one by default.
    finish : str
        The artifact whose existence makes "done" a valid answer.

    Returns
    -------
    dict or None
        ``{"tool": name, "why": "only one action was available: name"}`` when
        the choice is forced; None when the model has something to decide,
        including whether to finish.

    Raises
    ------
    None

    Examples
    --------
    >>> from .tools import Tool
    >>> only = {'load': Tool('load', 'Load.', lambda c: ('', {'data': 1}),
    ...                      produces=('data',))}
    >>> forced_choice(RunContext('x'), only)
    {'tool': 'load', 'why': 'only one action was available: load'}
    >>> ctx = RunContext('x')
    >>> ctx.put('report_files', {'report_markdown': 'r.md'})
    >>> forced_choice(ctx, only) is None        # "done" is a valid answer now
    True
    """
    options = runnable_tools(ctx, tools)
    if len(options) != 1 or _may_finish(ctx, finish):
        return None
    name = options[0].name
    return {"tool": name, "why": f"only one action was available: {name}"}


def _take_notes(ctx: RunContext, steer: Optional["steering.Steering"]) -> List[str]:
    """The user's new notes, added to the run's record of what it was told."""
    notes = steer.take_notes() if steer is not None else []
    ctx.guidance.extend(notes)
    return notes


def _answer_notes(ctx: RunContext, steer: Any, notes: List[str],
                  choice: Optional[Dict[str, Any]], heard: bool) -> None:
    """Act on the settings a decision changed for the user, and say so.

    The answer goes back to whoever sent the notes: what the controller made of
    them (its reason) and which settings changed, or that this run has no
    model to read them.
    """
    from .recovery import apply_changes

    changes: List[str] = []
    if heard and choice is not None and choice.get("settings"):
        changes = apply_changes(ctx, choice["settings"])
        for change in changes:
            ctx.note(f"Changed at the user's request: {change}.")
    if not heard:
        ctx.note("The user sent notes during the run, but it had no model to read "
                 "them: " + "; ".join(notes))
    if steer is not None:
        steer.tell({"phase": "steer", "notes": list(notes), "changes": changes,
                    "heard": bool(heard),
                    "why": str((choice or {}).get("why", "")) if heard else "",
                    "tool": str((choice or {}).get("tool", "") or "")})


def _may_finish(ctx: RunContext, finish: str = DEFAULT_FINISH) -> bool:
    """Whether "done" is a valid answer: once the run's final product exists."""
    return ctx.has(finish or DEFAULT_FINISH)


def _decide(ctx: RunContext, ask: Optional[Callable[[str], str]],
            tools: Optional[Dict[str, Any]] = None,
            template: Optional[str] = None,
            finish: str = DEFAULT_FINISH,
            notes: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
    """One decision: the model's if it can make one, the policy's otherwise.

    ``notes`` are what the user has just said; with them the model is asked
    even when the menu leaves one choice, since a note can change a setting
    or end the run as well as pick a step.
    """
    if ask is not None:
        # Nothing to decide, nothing to ask. Without a model the policy below
        # takes the same step, as it always has.
        forced = None if notes else forced_choice(ctx, tools, finish)
        if forced is not None:
            return forced
        prompt = (template or CONTROLLER_PROMPT).format(transcript=ctx.transcript(),
                                                        menu=menu(ctx, tools))
        if notes:
            from .recovery import settings_for

            prompt += STEERING_PROMPT.format(
                notes="\n".join(f"- {note}" for note in notes),
                settings=json.dumps(settings_for(ctx), default=str, indent=1) or "{}")
        try:
            choice = _parse_choice(ask(prompt))
        except Exception:  # noqa: BLE001 - an unreachable model is not a failed run
            choice = None
        if choice is not None:
            return choice
        # The model could not be reached or did not answer usably. Recording it
        # matters: a run that silently fell back to the fixed order looks
        # identical to one the model drove, and the two are not the same run.
        ctx.note("The controller could not get a usable decision from the model "
                 "and fell back to running the available steps in order.")
    name = next_by_policy(ctx, tools)
    return None if name is None else {"tool": name, "why": "next available step"}
