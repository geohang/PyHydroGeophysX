"""The loop: read the transcript, choose one action, observe, choose again.

A fixed pipeline decides everything before it starts; this loop decides one
step at a time, with the results of the previous steps in front of it. That is
what lets a run react -
to an inversion that did not converge, to a mesh with no layering, to a file
that turned out to hold something other than what its name suggested.

Two properties matter more than the loop itself:

**It always terminates, and always produces something.** A model that keeps
choosing badly is bounded by ``max_steps``; a model that cannot be reached at
all falls through to :func:`next_by_policy`, which walks the tools in
dependency order. A run without an API key therefore behaves like the fixed
pipeline - it simply stops adapting. An exception mid-sequence does not lose
the whole run.

**A refusal is data.** When the controller names a tool that cannot run, or one
that does not exist, that becomes a recorded step with the reason, and the loop
continues. The model sees its own mistake in the transcript on the next turn,
which is how it corrects. Raising instead would end the run over a typo.
"""

import json
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..._internal.utils import parse_json_object
from . import steering, toolbox
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


#: Steps a flexible run may take: the standard route is about eight, and the
#: rest is room for looks, a re-run and the steps after it.
FLEXIBLE_MAX_STEPS = 40

FLEXIBLE_PROMPT = """You are running a geophysics processing workflow for the user's \
goal. Think the whole task through - what the goal needs, what the run has shown so \
far, what is left, what could go better - then choose the single next action.

{transcript}

The standard route from here, as things stand:
{route}

Standard steps you can take now (they run without asking):
{menu}

Off the standard route (each is put to the user first, with your reason as the
recommendation - except a look and leaving steps out, which are yours to decide):
{toolbox}

Settings that may be changed, with their current values:
{settings}

Reply with JSON only - no prose, no code fence. Your reasoning comes first:
{{"why": "<what the evidence so far shows, and why this is the best next action>",
  "confidence": <0 to 1: how likely this is the action an expert would take here>,
  "plan": ["<the step names you intend to run after this one, in order>"],
  "check_after": ["<steps in your plan whose results you want to see before going on>"],
  "tool": "<a step name from either list>"}}

You are asked again only when you want to be. After this step, the steps in
"plan" run in order without asking you, until a step in "check_after" has run,
a step fails or raises a warning, the user says something, a planned step is not
on offer, or the plan runs out. Put a step in "check_after" when its result could
change what to do next (an inversion's fit, an evaluation's verdict); leave out
the ones whose outcome cannot (loading data, writing the report). A plan that ends
with the report and is not checked after it ends the run there.
Variants of the same reply:
- with changed settings for that step, or to run a finished step again with them:
  add "settings": {{"<key>": <new value>}}
- to look at a product's numbers before deciding: "inspect": "<a name under Available now>"
  in place of "tool"
- to read a written procedure (when the list above has them): "read_skill": "<its name>"
  in place of "tool"
- to use a studio page (listed above when the studio is open): "studio": {{"module":
  "<page key>", "action": "describe" or an action name, "args": {{...}}}} in place of "tool"
- to leave out route steps that do not serve the goal (data the request never named,
  a method it did not ask for): "skip": ["<step name>", ..] in place of "tool". This
  is your call, not put to the user; the steps are recorded as left out with your
  reason, and the steps after them that need their products drop off the route too.
  A step the report or a product the goal asks for still needs is not left out.
  With a confidence below {low_confidence} it is put to the user instead.
- for something no step does: "script": "<what a short Python script should do>" in
  place of "tool"
- when a choice is the user's to make: "ask": "<question>", "options": ["..", ".."],
  "recommend": "<one option>" in place of "tool"
- once the goal is met and the report is written: {{"why": "...", "done": true}}

Rules:
- Follow the standard route unless the evidence gives a concrete reason that another
  action will answer the user's goal better or more completely: a poor fit or an
  implausible model, a product the goal implies that the route leaves out, a check
  that would confirm or overturn a conclusion. Put that reason in "why"; it is what
  the user reads when asked.
- Look before you change: inspect a product when its numbers decide what to do, and
  read a written procedure (listed above, when there are any) before a step it covers.
- Give "confidence" honestly: low when the evidence is thin or two actions are close.
- A step run again changes what the steps after it read; they run again after it.
  If the report was already written, write it again so it describes the final results.
- If the user declined a proposal, accept it and carry on; never propose it again.
- Produce everything the goal asks for, and write the report last.
- If a step failed, do not repeat it unchanged.
- If a studio page action failed, read its error and the page's state: run it again
  with other arguments, use another action or page, propose a script for what no page
  does, or go back to the standard route. A failed page action leaves the run's own
  results as they were.
"""


#: Below this confidence, a decision the model may otherwise take alone -
#: leaving route steps out - is put to the user instead. A model's stated
#: confidence is a rough signal, so it only ever adds a question; it never
#: lets a decision skip one.
LOW_CONFIDENCE = 0.5


def _confidence(value: Any) -> Optional[float]:
    """A stated confidence as a number in [0, 1], or None.

    Examples
    --------
    >>> _confidence(0.83), _confidence("70%"), _confidence("high"), _confidence(3)
    (0.83, 0.7, None, None)
    """
    text = str(value).strip() if value is not None and not isinstance(value, bool) else ""
    scale = 100.0 if text.endswith("%") else 1.0
    try:
        number = float(text.rstrip("%")) / scale
    except ValueError:
        return None
    return round(number, 3) if 0.0 <= number <= 1.0 else None


def _parse_choice(reply: Any) -> Optional[Dict[str, Any]]:
    """The controller's decision, or None when the reply is unusable.

    ``settings`` is kept when the reply carries a dict of them: it is how a
    decision follows a note from the user (:data:`STEERING_PROMPT`), and in a
    flexible run how it proposes a change. ``plan``, ``confidence``, ``inspect``,
    ``read_skill``, ``script`` and ``ask`` are the flexible run's
    (:data:`FLEXIBLE_PROMPT`); a confidence is kept only as a number in [0, 1].

    Examples
    --------
    >>> _parse_choice('{"why": "fit is poor", "plan": ["redo"], "tool": "invert_ert",'
    ...               ' "settings": {"inversion_params": {"lambda": 20}}}')['settings']
    {'inversion_params': {'lambda': 20}}
    >>> _parse_choice('{"why": "check", "inspect": "inversion_results"}')['inspect']
    'inversion_results'
    >>> _parse_choice('{"why": "qc first", "confidence": 0.4, "read_skill": "ert-data-quality"}')
    {'read_skill': 'ert-data-quality', 'why': 'qc first', 'confidence': 0.4}
    >>> _parse_choice('{"why": "no TDEM data", "skip": "load_tdem_data"}')['skip']
    ['load_tdem_data']
    >>> _parse_choice('not json') is None
    True
    """
    answer = parse_json_object(reply)
    if not isinstance(answer, dict):
        return None
    extra: Dict[str, Any] = (
        {"settings": answer["settings"]}
        if isinstance(answer.get("settings"), dict) and answer["settings"] else {})
    if isinstance(answer.get("plan"), list):
        extra["plan"] = [str(p)[:160] for p in answer["plan"][:12]]
    if isinstance(answer.get("check_after"), list):
        extra["check_after"] = [str(p)[:160] for p in answer["check_after"][:12]]
    confidence = _confidence(answer.get("confidence"))
    if confidence is not None:
        extra["confidence"] = confidence
    why = str(answer.get("why", ""))
    if answer.get("done"):
        return {"done": True, "why": why, **extra}
    page = answer.get("studio")
    if isinstance(page, dict) and str(page.get("module") or "").strip():
        return {"studio": {"module": str(page["module"]).strip(),
                           "action": str(page.get("action") or "describe").strip(),
                           "args": page.get("args") if isinstance(page.get("args"), dict) else {}},
                "why": why, **extra}
    skip = answer.get("skip")
    skip = [skip] if isinstance(skip, str) else skip
    if isinstance(skip, list) and any(str(s).strip() for s in skip):
        return {"skip": [str(s).strip() for s in skip if str(s).strip()][:12],
                "why": why, **extra}
    for kind in ("inspect", "read_skill", "script", "ask"):
        value = answer.get(kind)
        if isinstance(value, str) and value.strip():
            choice = {kind: value.strip(), "why": why, **extra}
            if kind == "ask":
                choice["options"] = [str(o) for o in answer.get("options") or []
                                     if str(o).strip()]
                choice["recommend"] = str(answer.get("recommend") or "")
            return choice
    tool = answer.get("tool")
    if not isinstance(tool, str) or not tool.strip():
        return None
    return {"tool": tool.strip(), "why": why, **extra}


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
    pipeline - the right fallback: without a model the run should
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
    shadow.reopened = dict(ctx.reopened)
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
#: The loop's own code failed again after the run had resumed past it
#: (:func:`.entry.drive`).
BROKE = "broke"


def run_controller(ctx: RunContext, ask: Optional[Callable[[str], str]] = None,
                   max_steps: int = DEFAULT_MAX_STEPS,
                   progress: Optional[Callable[[str, float, str], None]] = None,
                   tools: Optional[Dict[str, Any]] = None,
                   on_step: Optional[Callable[..., Optional[str]]] = None,
                   on_result: Optional[Callable[..., None]] = None,
                   recovery: bool = True,
                   prompt: Optional[str] = None,
                   finish: str = DEFAULT_FINISH,
                   flexible: bool = False,
                   approves_steps: bool = False
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
        identically whichever way it was reached.
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
    flexible : bool
        Let the model step off the standard route (:mod:`.toolbox`): look at a
        product's numbers, add a step the route has not scheduled, run a
        finished step again with other settings, change a setting, leave route
        steps out, write a temporary script, or ask the user. Everything but a look is put to the
        user first with the model's reason (:func:`.toolbox.propose`), and the
        route is the default. The model is asked with :data:`FLEXIBLE_PROMPT`,
        and says in its answer which steps it means to run next and which
        results it wants to see; those steps then run without asking it again
        (:func:`_follow_plan`) until it wants to look, or something it could
        not have planned for happens. Ignored without ``ask``, and with a
        ``prompt`` of the assistant's own, which does not offer it.
    approves_steps : bool
        ``on_step`` asks the user before every step (step by step), so an
        off-route step is approved there, with its reason, rather than in a
        question of its own.

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
    registry = TOOLS if tools is None else tools
    # Latitude needs someone to exercise it: without a model the route is all.
    # And the default prompt, which is the one that offers it.
    flexible = bool(flexible and ask is not None and prompt is None)
    # The model's own plan, followed without asking it again until it wants to
    # be asked (FLEXIBLE_PROMPT): the steps it means to run next, and the ones
    # whose results it wants to see first. Asking before every step would make
    # a flexible run much slower than the fixed route for the same steps.
    planned: List[str] = []
    checks: List[str] = []
    warned = len(ctx.warnings)
    # The model's stated confidence in its last decision, and the step it went
    # to: kept on that step, so runs can be checked for whether it means anything.
    stamp: Optional[tuple] = None
    for index in range(max_steps):
        _stamp_confidence(ctx, stamp)
        stamp = None
        # Between steps is where the user's pause and notes take effect: the
        # step that was running has finished, and nothing new has started.
        if steer is not None and not steer.checkpoint(
                ctx.steps[-1].description or ctx.steps[-1].tool if ctx.steps else ""):
            ctx.ended = STOPPED
            break
        notes = _take_notes(ctx, steer)
        options = runnable_tools(ctx, tools)
        follow = (_follow_plan(ctx, planned, checks, options, warned, notes, finish)
                  if flexible else None)
        if follow == "finish":
            ctx.ended = FINISHED
            break
        if not options and follow is None and not (flexible and _room_to_deviate(ctx)):
            ctx.ended = EXHAUSTED
            break
        if isinstance(follow, dict):
            choice = follow
        else:
            choice = _decide(ctx, ask, tools, prompt, finish, notes, flexible=flexible)
            if choice is not None and choice.get("confidence") is not None:
                stamp = (len(ctx.steps), choice["confidence"])
            if flexible and choice is not None:
                # A look, a script or a question is read before going on.
                planned = ([p for p in choice.get("plan") or [] if p in registry]
                           if choice.get("tool") else [])
                checks = list(choice.get("check_after") or [])
        warned = len(ctx.warnings)
        if notes:
            _answer_notes(ctx, steer, notes, choice, heard=ask is not None)
        if choice is not None and choice.get("done") and not _may_finish(ctx, finish):
            # "done" is a valid answer only once a report exists: the prompt
            # says so and forced_choice counts on it. Taken at its word, a
            # model answering "done" with steps on offer would end the run on
            # the spot - nothing run, no report, reported as a success. The
            # step taken instead records why, which is how the model reads the
            # refusal on its next turn.
            name = next_by_policy(ctx, tools)
            stamp = None
            choice = None if name is None else {
                "tool": name,
                "why": "the model answered 'done' before a report was written, "
                       "so the next available step was taken"}
        if choice is None or choice.get("done"):
            ctx.ended = FINISHED if choice is not None else EXHAUSTED
            break
        reason = choice.get("why", "")
        if flexible and not choice.get("tool"):
            # A look, a studio page, a script or a question: no catalog tool runs.
            if _act_off_route(ctx, choice, ask, on_result, on_step, approves_steps,
                              _route_next(ctx, tools, finish), tools=tools,
                              finish=finish) == STOP:
                ctx.ended = STOPPED
                break
            continue
        name = choice["tool"]
        tool = registry.get(name)
        # Settings a user's note asked for were applied with the notes; any
        # others in the decision are the model's own proposal.
        proposed = None if notes else choice.get("settings")
        deviation = (_deviation(ctx, tool, options, registry, proposed, finish)
                     if flexible and tool is not None else None)
        if deviation is not None and deviation.get("refuse"):
            ctx.begin(name, reason, agent=getattr(tool, "agent", ""),
                      description=getattr(tool, "label", "") or name)
            ctx.finish(status="blocked", error=deviation["refuse"])
            continue
        if deviation is None and tool is not None and name not in {t.name for t in options}:
            # Only what the menu offered may run. `invoke` checks requirements
            # and nothing else, so a reply naming an unlisted tool would run
            # it: a single-survey inversion could replace a time-lapse result,
            # a climate retrieval could run with climate switched off. Recorded
            # as blocked - it never ran - so the model reads why on its next
            # turn, and the tool stays on offer for when it does apply.
            why = tool.unavailable_because(ctx)
            ctx.begin(name, reason, agent=tool.agent,
                      description=tool.label or name, purpose=tool.description)
            ctx.finish(status="blocked",
                       error=f"Not offered, so not run: {why}. Choose one of: "
                             f"{', '.join(t.name for t in options)}.")
            continue
        if deviation is not None:
            reason = f"Off the standard route - {deviation['title']}. {reason}".strip()
            if not approves_steps:
                # Asked here; with a hook that approves every step, that
                # approval below is the user's answer instead.
                accepted = toolbox.propose(
                    ctx, deviation["key"], deviation["title"], choice.get("why", ""),
                    _route_next(ctx, tools, finish), name=_assistant(ctx))
                if not accepted:
                    if deviation["kind"] != "settings":
                        continue
                    # Declined new settings: the step runs as configured.
                    deviation, reason = None, choice.get("why", "")

        verdict = PROCEED
        if on_step is not None:
            try:
                verdict = on_step(ctx, tool, reason) or PROCEED
            except Exception:  # noqa: BLE001 - a UI hook must not stop a run
                verdict = PROCEED
        if deviation is not None and approves_steps:
            ctx.proposals.append({"key": deviation["key"], "title": deviation["title"],
                                  "why": choice.get("why", ""),
                                  "accepted": verdict == PROCEED})
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
        if deviation is not None and deviation.get("changes"):
            from .recovery import apply_changes

            for change in apply_changes(ctx, deviation["changes"]):
                ctx.note(f"Changed off the standard route, with the user's agreement: "
                         f"{change}.")

        if progress is not None:
            fraction = min(0.95, (index + 1) / float(max_steps))
            try:
                progress(tool.label if tool and tool.label else name, fraction, reason)
            except Exception:  # noqa: BLE001 - a UI callback must not stop a run
                pass
        # A warning the step itself raises interrupts the plan; the notes this
        # loop wrote about the change of course just made do not.
        warned = len(ctx.warnings)
        status = invoke(ctx, name, reason, tools)
        if status == "ok" and deviation is not None and tool is not None:
            # What this step made has changed under the steps after it: they
            # run again, and so does a report already written.
            stale = [n for n in toolbox.downstream(registry, tool.produces) if n != name]
            stale += [n for n, t in registry.items()
                      if finish in t.produces and ctx.attempted(n) and n != name]
            ctx.reopen(dict.fromkeys(stale))
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
    _stamp_confidence(ctx, stamp)
    return ctx


def _assistant(ctx: RunContext) -> str:
    """The assistant's name for what the user reads."""
    return str(ctx.settings.get("assistant_name") or "The assistant")


def _route_next(ctx: RunContext, tools: Optional[Dict[str, Any]], finish: str) -> str:
    """The standard route's next step, in words, for a proposal to set beside."""
    try:
        ahead = route_ahead(ctx, tools, finish, limit=1)
    except Exception:  # noqa: BLE001 - a projection must not stop a run
        ahead = []
    return ahead[0]["label"] if ahead else ""


def _follow_plan(ctx: RunContext, planned: List[str], checks: List[str],
                 options: List[Any], warned: int, notes: List[str],
                 finish: str) -> Any:
    """The next step of the model's plan, taken without asking it; or None to ask.

    The model is asked again when it said it wanted to see the last step's
    result (``checks``), when something it could not have planned for
    happened - the step failed, a warning was raised, the user said something
    - and when its plan has run out or names a step that is not on offer.
    "finish" when the plan ended with the step that wrote the final product
    and the model did not ask to look at it: the run is done.

    Examples
    --------
    >>> from .tools import Tool
    >>> ctx = RunContext('x')
    >>> _ = ctx.begin('load'); ctx.finish()
    >>> step = Tool('invert', '', lambda c: ('', {}))
    >>> _follow_plan(ctx, ['invert'], [], [step], 0, [], 'report_files')['tool']
    'invert'
    >>> _follow_plan(ctx, ['invert'], ['load'], [step], 0, [], 'report_files') is None
    True
    """
    if notes or not ctx.steps:
        return None
    last = ctx.steps[-1]
    if last.status != "ok" or last.tool in checks or len(ctx.warnings) > warned:
        return None
    if not planned:
        return "finish" if finish in (last.produced or []) and _may_finish(ctx, finish) else None
    if planned[0] not in {tool.name for tool in options}:
        return None
    return {"tool": planned.pop(0), "why": "next in the plan", "planned": True}


def _room_to_deviate(ctx: RunContext) -> bool:
    """Whether a flexible run with nothing left on the route may still act."""
    return (toolbox.proposals_left(ctx) > 0 or toolbox.looks_left(ctx) > 0
            or bool(toolbox.studio_pages_allowed(ctx)))


def _deviation(ctx: RunContext, tool: Any, options: List[Any], registry: Dict[str, Any],
               settings: Optional[Dict[str, Any]], finish: str) -> Optional[Dict[str, Any]]:
    """What running ``tool`` (with ``settings``) would change about the route.

    None when it is the route: an offered step with no new settings. Otherwise
    ``{"kind", "key", "title", "changes"}`` - ``settings`` for an offered step
    with changed settings, ``extra`` for a step the route has not scheduled,
    ``redo`` for a finished step run again - or ``{"refuse": why}`` when it
    cannot be proposed at all.
    """
    from .recovery import ADJUSTABLE, _merge

    label = tool.label or tool.name
    changes = {key: value for key, value in (settings or {}).items()
               if key in ADJUSTABLE and _merge(ctx.config.get(key), value) != ctx.config.get(key)}
    described = "; ".join(
        f"{key} {'the default' if ctx.config.get(key) is None else repr(ctx.config.get(key))}"
        f" -> {_merge(ctx.config.get(key), value)!r}" for key, value in changes.items())
    offered = tool.name in {t.name for t in options}
    if offered and not changes:
        return None
    if offered:
        kind, key, title = "settings", f"settings:{tool.name}", f"{label}, with {described}"
    elif tool.name in {t.name for t in toolbox.extra_tools(ctx, registry)}:
        kind, key = "extra", f"extra:{tool.name}"
        title = f"{label}, which the standard route had not scheduled" + (
            f", with {described}" if changes else "")
    elif tool.name in {t.name for t in toolbox.redo_candidates(ctx, registry, finish)}:
        if not changes:
            return {"refuse": f"{label} has already run; running it again unchanged would "
                              "only repeat it. Give changed \"settings\" to run it again."}
        kind, key, title = "redo", f"redo:{tool.name}", f"{label} again, with {described}"
    else:
        return {"refuse": f"{label} is not on offer: {tool.unavailable_because(ctx)}."}
    if key in toolbox.declined(ctx):
        return {"refuse": f"The user declined this ({title}); keep to the standard route."}
    if not toolbox.proposals_left(ctx):
        return {"refuse": "No more off-route proposals in this run; keep to the "
                          "standard route."}
    return {"kind": kind, "key": key, "title": title, "changes": changes}


def _projected_products(ctx: RunContext, tools: Optional[Dict[str, Any]], finish: str,
                        skipped: Sequence[str] = ()) -> set:
    """What the run has, plus what its route ahead would make with ``skipped`` left out."""
    registry = TOOLS if tools is None else tools
    shadow = RunContext(ctx.goal, ctx.config, ctx.output_dir)
    shadow.artifacts = dict(ctx.artifacts)
    shadow.steps = list(ctx.steps)
    shadow.reopened = dict(ctx.reopened)
    for name in skipped:
        shadow.begin(name, "projected")
        shadow.finish(status="skipped")
    made = {key for key in shadow.artifacts if shadow.has(key)}
    for step in route_ahead(shadow, tools, finish, limit=40):
        made.update(getattr(registry.get(step["tool"]), "produces", ()))
    return made


def _needed_products(ctx: RunContext, finish: str) -> Dict[str, str]:
    """The products a skip must not cost the run: the report, and what the request asks for."""
    from .._intent import wants_climate, wants_spatial_map, wants_water_content

    needed = {finish: "the report"}
    for key, label, wanted in (("water_content", "water content", wants_water_content),
                               ("climate_data", "the climate series", wants_climate),
                               ("plan_maps", "the plan-view maps", wants_spatial_map)):
        try:
            if wanted(ctx.config):
                needed[key] = label
        except Exception:  # noqa: BLE001 - an unreadable request protects nothing extra
            pass
    return needed


def _skip_steps(ctx: RunContext, names: List[str], why: str,
                tools: Optional[Dict[str, Any]], finish: str) -> None:
    """Leave route steps out on the model's judgement, recording each as skipped.

    Not put to the user: a step for data the request never named, or a method it
    did not ask for, is the model's call, and asking about it only slowed the
    run. What guards the user instead is that nothing they asked for can be lost
    this way: a step is left out only when the report and every product the
    request asks for can still be made without it (:func:`_projected_products`).
    A skipped step counts as tried, so it is not offered again, and the steps
    after it that need its products drop off the route with it. The Live tab and
    the report show what was left out and why.

    Examples
    --------
    >>> from .tools import Tool
    >>> reg = {'tem': Tool('tem', '', lambda c: ('', {}), produces=('t',), label='Load TEM'),
    ...        'ert': Tool('ert', '', lambda c: ('', {}), produces=('model',)),
    ...        'rep': Tool('rep', '', lambda c: ('', {}), requires=('model',),
    ...                    produces=('report_files',))}
    >>> ctx = RunContext('x')
    >>> _skip_steps(ctx, ['tem', 'ert', 'rep'], 'no TEM data', reg, 'report_files')
    >>> [(s.tool, s.status) for s in ctx.steps]
    [('tem', 'skipped'), ('skip_steps', 'blocked')]
    >>> print(ctx.steps[-1].error)
    Not left out: ert (the run needs it for the report); rep (it writes the report or is not on the route ahead).
    """
    registry = TOOLS if tools is None else tools
    try:
        ahead = [step["tool"] for step in route_ahead(ctx, tools, finish, limit=40)]
        before = _projected_products(ctx, tools, finish)
    except Exception:  # noqa: BLE001 - a projection must not stop a run
        ahead, before = [], set()
    allowed = set(toolbox.skippable(ctx, registry, ahead, finish))
    needed = {key: label for key, label in _needed_products(ctx, finish).items()
              if key in before}
    kept: List[str] = []
    refused: List[str] = []
    for name in dict.fromkeys(names):
        if name not in allowed:
            refused.append(f"{name} (it writes the report or is not on the route ahead)")
            continue
        try:
            lost = [label for key, label in needed.items()
                    if key not in _projected_products(ctx, tools, finish, kept + [name])]
        except Exception:  # noqa: BLE001 - when in doubt, keep the step
            lost = ["the run"]
        if lost:
            refused.append(f"{name} (the run needs it for {', '.join(lost)})")
        else:
            kept.append(name)
    for name in kept:
        tool = registry[name]
        ctx.begin(name, why, agent=getattr(tool, "agent", ""),
                  description=tool.label or name)
        ctx.finish(status="skipped", error=f"Left out by {_assistant(ctx)}: {why}")
    if refused:
        ctx.begin("skip_steps", why, description="Leave out route steps")
        ctx.finish(status="blocked", error="Not left out: " + "; ".join(refused) + ".")


def _act_off_route(ctx: RunContext, choice: Dict[str, Any], ask: Any,
                   on_result: Optional[Callable[..., None]],
                   on_step: Optional[Callable[..., Optional[str]]] = None,
                   approves_steps: bool = False, route_next: str = "",
                   tools: Optional[Dict[str, Any]] = None,
                   finish: str = DEFAULT_FINISH) -> str:
    """Carry out a look, a studio page action, skipped steps, a script or a question.

    Returns the step hook's verdict when a studio action was put to it -
    :data:`STOP` ends the run - and :data:`PROCEED` otherwise.

    Examples
    --------
    >>> from .tools import Tool
    >>> reg = {name: Tool(name, '', lambda c: ('', {}), produces=(name,))
    ...        for name in ('a', 'b', 'c', 'report_files')}
    >>> ctx = RunContext('x')
    >>> _skip_agreed(ctx, ['a', 'b'], 'maybe not needed', 'Load data')
    False
    >>> for choice in ({'skip': ['a'], 'confidence': 0.9}, {'skip': ['b']},
    ...                {'skip': ['a', 'b', 'c'], 'confidence': 0.9}):
    ...     verdict = _act_off_route(ctx, choice, None, None, tools=reg)
    >>> [(s.tool, s.status) for s in ctx.steps]
    [('skip_steps', 'blocked'), ('skip_steps', 'blocked'), ('c', 'skipped')]
    >>> print(ctx.steps[0].error)
    The user kept these steps on the route: a.
    """
    why = str(choice.get("why", ""))
    before = len(ctx.steps)
    verdict = PROCEED
    if choice.get("inspect"):
        if not toolbox.looks_left(ctx):
            ctx.begin("inspect", why, description="Inspect")
            ctx.finish(status="blocked", error="No more looks in this run; act on what "
                                               "the run has shown.")
        else:
            toolbox.inspect_artifact(ctx, str(choice["inspect"]))
    elif choice.get("read_skill"):
        if not toolbox.skill_reads_left(ctx):
            ctx.begin("skill:", why, description="Read a procedure")
            ctx.finish(status="blocked", error="No more procedure reads in this run; act "
                                               "on what you have read.")
        else:
            toolbox.read_skill(ctx, str(choice["read_skill"]), why)
    elif choice.get("studio"):
        verdict = _use_studio(ctx, choice["studio"], why, on_step, approves_steps,
                              route_next)
    elif choice.get("skip"):
        kept = {name for key in toolbox.declined(ctx) if key.startswith("skip:")
                for name in key[len("skip:"):].split(",")}
        names = [name for name in choice["skip"] if name not in kept]
        if not names:
            ctx.begin("skip_steps", why, agent=_assistant(ctx), description="Leave steps out")
            ctx.finish(status="blocked", error="The user kept these steps on the route: "
                       + ", ".join(choice["skip"]) + ".")
        else:
            # The model's own call unless it is unsure; steps the user kept stay.
            unsure = choice.get("confidence") is not None and choice["confidence"] < LOW_CONFIDENCE
            if not unsure or _skip_agreed(ctx, names, why, route_next):
                _skip_steps(ctx, names, why, tools, finish)
    elif not toolbox.proposals_left(ctx):
        ctx.begin("ask_user" if choice.get("ask") else "temporary_script", why,
                  description="Off-route proposal")
        ctx.finish(status="blocked", error="No more off-route proposals in this run; "
                                           "keep to the standard route.")
    elif choice.get("script"):
        from .recovery import script_task

        ctx.note(script_task(ctx, str(choice["script"]), ask, why))
    elif choice.get("ask"):
        chosen = toolbox.ask_choice(ctx, str(choice["ask"]), choice.get("options") or [],
                                    str(choice.get("recommend") or ""), why)
        ctx.begin("ask_user", why, agent=_assistant(ctx), description="Ask the user",
                  purpose=str(choice["ask"]))
        if chosen:
            ctx.finish(summary=f"Asked: {choice['ask']} The user chose: {chosen}.")
        else:
            ctx.finish(status="blocked", error="A question needs two to four options.")
    # Every step this recorded - a skip can record several.
    for step in ctx.steps[before:] if on_result is not None else []:
        try:
            on_result(ctx, step)
        except Exception:  # noqa: BLE001 - a UI hook must not stop a run
            pass
    return verdict


def _skip_agreed(ctx: RunContext, names: List[str], why: str, route_next: str) -> bool:
    """Put route steps the model is unsure of leaving out to the user; True if agreed.

    Staying on the route is the default, so with nobody to answer, or no
    questions left, the steps stay.

    Examples
    --------
    >>> ctx = RunContext('x')
    >>> _skip_agreed(ctx, ['load_tdem_data'], 'maybe no TEM', 'Load TEM')
    False
    >>> ctx.proposals[0]['title'], ctx.proposals[0]['accepted']
    ('leave out load_tdem_data', False)
    """
    key = "skip:" + ",".join(sorted(names))
    if key in toolbox.declined(ctx) or not toolbox.proposals_left(ctx):
        ctx.begin("skip_steps", why, agent=_assistant(ctx), description="Leave steps out")
        ctx.finish(status="blocked", error=(
            "The user kept these steps on the route." if key in toolbox.declined(ctx) else
            "Not sure enough to leave these steps out alone, and no questions are left: "
            "they stay on the route."))
        return False
    return toolbox.propose(ctx, key, f"leave out {', '.join(names)}", why, route_next,
                           name=_assistant(ctx))


def _stamp_confidence(ctx: RunContext, stamp: Optional[tuple]) -> None:
    """Record a decision's stated confidence on the step it led to."""
    if stamp is not None and stamp[0] < len(ctx.steps):
        step = ctx.steps[stamp[0]]
        if step.confidence is None:
            step.confidence = stamp[1]


def _use_studio(ctx: RunContext, request: Dict[str, Any], why: str,
                on_step: Optional[Callable[..., Optional[str]]], approves_steps: bool,
                route_next: str) -> str:
    """Look at a studio page, or run one of its actions (:mod:`.studio`).

    A look is free. The first action on a page is put to the user with the
    model's reason (unless every step is approved anyway); once they agree,
    the run may use that page. The action is announced like a step, so the
    Live tab shows it while it runs, and recorded as one.
    """
    from . import studio
    from .tools import Tool

    titles = {page["key"]: page["title"] for page in studio.catalog()}
    module = str(request.get("module") or "")
    action = str(request.get("action") or "describe")
    if module not in titles:
        ctx.begin(f"look:{module}", why, description="Studio page")
        ctx.finish(status="blocked", error=f"No studio page '{module}'"
                   + (f"; the pages are {', '.join(titles)}." if titles else
                      "; this run has no studio."))
        return PROCEED
    title = titles[module]
    if action in ("describe", "describe_current_module", "get_status"):
        exhausted = toolbox.looks_left(ctx) <= 0
        ctx.begin(f"look:{module}", why, agent="Studio",
                  description=f"Look at the {title} page",
                  purpose="See a studio page's actions and state; changes nothing.")
        if exhausted:
            ctx.finish(status="blocked", error="No more looks in this run.")
            return PROCEED
        answer = studio.call("describe", module=module)
        if answer.get("status") != "ok":
            ctx.finish(status="failed", error=str(answer.get("error") or "No description."))
        else:
            busy = f"\nbusy: {answer['busy']}" if answer.get("busy") else ""
            detail = str((request.get("args") or {}).get("action") or "")
            ctx.finish(summary=studio.describe_compact(answer.get("describe"), detail) + busy)
        return PROCEED
    allowed = module in toolbox.studio_pages_allowed(ctx)
    key = f"studio:{module}"
    label = f"{title} page · {action}"
    if not allowed and not approves_steps:
        refused = key in toolbox.declined(ctx)
        if refused or not toolbox.proposals_left(ctx):
            ctx.begin(f"studio:{module}.{action}", why, description=label)
            ctx.finish(status="blocked", error=(
                f"The user declined the use of the {title} page." if refused
                else "No more off-route proposals in this run; keep to the standard route."))
            return PROCEED
        if not toolbox.propose(ctx, key, f"use the {title} page, starting with {action}",
                               why, route_next, name=_assistant(ctx),
                               detail=f"Arguments: {request.get('args') or {}}"):
            return PROCEED
    step = Tool(f"studio:{module}.{action}", f"The {title} page's action {action}.",
                lambda c: ("", {}), label=label, module=module)
    verdict = PROCEED
    if on_step is not None:
        try:
            verdict = on_step(ctx, step, why) or PROCEED
        except Exception:  # noqa: BLE001 - a UI hook must not stop a run
            verdict = PROCEED
    if approves_steps and not allowed:
        ctx.proposals.append({"key": key, "title": f"use the {title} page", "why": why,
                              "accepted": verdict == PROCEED})
    if verdict in (STOP, SKIP):
        ctx.begin(step.name, why, agent="Studio", description=label)
        ctx.finish(status="skipped", error="Stopped here at the user's request."
                   if verdict == STOP else "Skipped at the user's request.")
        return verdict
    ctx.begin(step.name, why, agent="Studio", description=label,
              purpose=f"Run the {title} page's action {action}.")
    answer = studio.call("apply", module=module, action=action, args=request.get("args") or {})
    failed = str(answer.get("status") or "").lower() in ("failed", "error", "declined")
    ctx.finish(summary=studio.result_summary(answer), status="failed" if failed else "ok",
               error=str(answer.get("error") or answer.get("hint") or "") if failed else "")
    return PROCEED


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
            notes: Optional[List[str]] = None,
            flexible: bool = False) -> Optional[Dict[str, Any]]:
    """One decision: the model's if it can make one, the policy's otherwise.

    ``notes`` are what the user has just said; with them the model is asked
    even when the menu leaves one choice, since a note can change a setting
    or end the run as well as pick a step. A ``flexible`` run is always asked:
    one step on the route is not the only thing it may do.
    """
    if ask is not None:
        # Nothing to decide, nothing to ask. Without a model the policy below
        # takes the same step, as it always has.
        forced = None if (notes or flexible) else forced_choice(ctx, tools, finish)
        if forced is not None:
            return forced
        if flexible and template is None:
            from .recovery import settings_for

            ahead = route_ahead(ctx, tools, finish)
            prompt = FLEXIBLE_PROMPT.format(
                transcript=ctx.transcript(), menu=menu(ctx, tools),
                route="\n".join(f"- {step['label']} ({step['tool']})" for step in ahead)
                      or "- (nothing left: the route is complete)",
                toolbox=toolbox.toolbox_text(ctx, TOOLS if tools is None else tools,
                                             route=[step["tool"] for step in ahead],
                                             finish=finish),
                settings=json.dumps(settings_for(ctx), default=str)[:2500] or "{}",
                low_confidence=LOW_CONFIDENCE)
        else:
            prompt = (template or CONTROLLER_PROMPT).format(transcript=ctx.transcript(),
                                                            menu=menu(ctx, tools))
        if notes:
            from .recovery import settings_for

            prompt += STEERING_PROMPT.format(
                notes="\n".join(f"- {note}" for note in notes),
                settings=json.dumps(settings_for(ctx), default=str, indent=1) or "{}")
        try:
            reply = ask(prompt)
            choice = _parse_choice(reply)
            problem = "" if choice is not None else "its reply was not a usable decision"
        except Exception as exc:  # noqa: BLE001 - an unreachable model is not a failed run
            message = str(exc).strip()
            choice = None
            problem = message.splitlines()[0][:300] if message else type(exc).__name__
        if choice is not None:
            return choice
        # The model could not be reached or did not answer usably. Recording it
        # matters: a run that silently falls back to the fixed order looks
        # identical to one the model drove, and the two are not the same run.
        # The reason goes with it - a usage limit is not a bad reply.
        ctx.note(f"The controller could not get a usable decision from the model "
                 f"({problem}) and fell back to running the available steps in order.")
        name = next_by_policy(ctx, tools)
        return None if name is None else {
            "tool": name, "why": f"next available step; the model could not decide ({problem})"}
    name = next_by_policy(ctx, tools)
    return None if name is None else {"tool": name, "why": "next available step"}
