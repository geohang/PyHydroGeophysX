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

from .context import RunContext
from .tools import TOOLS, invoke, menu, runnable_tools

#: A run that has taken this many steps is looping rather than progressing.
#: Generous: a time-lapse run with a retry and a report is about eight.
DEFAULT_MAX_STEPS = 24

CONTROLLER_PROMPT = """You are running a geophysics processing workflow. Choose \
the single next action.

{transcript}

Actions you can take now:
{menu}

Reply with JSON only - no prose, no code fence:
{{"tool": "<name from the list>", "why": "<one short sentence>"}}
or, when the goal has been met and a report has been written:
{{"done": true, "why": "<one short sentence>"}}

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
    """The controller's decision, or None when the reply is unusable."""
    if not isinstance(reply, str):
        return None
    text = reply.strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        answer = json.loads(text[start:end + 1])
    except ValueError:
        return None
    if not isinstance(answer, dict):
        return None
    if answer.get("done"):
        return {"done": True, "why": str(answer.get("why", ""))}
    tool = answer.get("tool")
    if not isinstance(tool, str) or not tool.strip():
        return None
    return {"tool": tool.strip(), "why": str(answer.get("why", ""))}


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


#: What an ``on_step`` hook may answer. ``proceed`` is the default for any
#: other value, including None, so a hook that forgets to return cannot stall a
#: run.
PROCEED, SKIP, STOP = "proceed", "skip", "stop"


def run_controller(ctx: RunContext, ask: Optional[Callable[[str], str]] = None,
                   max_steps: int = DEFAULT_MAX_STEPS,
                   progress: Optional[Callable[[str, float, str], None]] = None,
                   tools: Optional[Dict[str, Any]] = None,
                   on_step: Optional[Callable[..., Optional[str]]] = None,
                   on_result: Optional[Callable[..., None]] = None,
                   recovery: bool = True
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

    Returns
    -------
    RunContext
        The same context, carrying the steps taken, the artifacts produced and
        any warnings raised.

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
    """
    # Recoveries spent per tool, so one bad setting cannot be retried forever.
    spent: Dict[str, int] = {}
    for index in range(max_steps):
        options = runnable_tools(ctx, tools)
        if not options:
            break
        choice = _decide(ctx, ask, tools)
        if choice is None or choice.get("done"):
            break
        name = choice["tool"]
        tool = (TOOLS if tools is None else tools).get(name)
        reason = choice.get("why", "")

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
    return ctx


def _decide(ctx: RunContext, ask: Optional[Callable[[str], str]],
            tools: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """One decision: the model's if it can make one, the policy's otherwise."""
    if ask is not None:
        prompt = CONTROLLER_PROMPT.format(transcript=ctx.transcript(),
                                         menu=menu(ctx, tools))
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
