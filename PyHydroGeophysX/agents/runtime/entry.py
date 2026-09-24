"""The workflow entry point, as a controller run.

``BaseAgent.run_unified_agent_workflow`` keeps its signature and its four-value
return, because the desktop app, the Streamlit app and the one-click runner all
call it. What changed is everything behind that signature: instead of
classifying the request into one of eight workflow types and running that
type's fixed sequence, it builds a run context and lets the controller choose
each step from the tools whose inputs exist.

The old implementation is still present as
``BaseAgent.run_legacy_agent_workflow``. It is not dead weight and not
indecision: it covers workflow types whose tools cannot be exercised on this
machine - there is no ParFlow output, no SEG-Y file and no GPU here - and
keeping it reachable means a run that behaves differently can be compared
against the code it replaced instead of argued about. Set
``PHGX_LEGACY_WORKFLOW=1`` to use it.
"""

import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from . import catalog  # noqa: F401 - importing registers every tool
from .catalog import summarise_run
from .context import RunContext
from .controller import PROCEED, SKIP, STOP, run_controller
from .modes import announcement, auto, completion

#: Set to 1 to run the pre-controller implementation instead.
LEGACY_ENV = "PHGX_LEGACY_WORKFLOW"


def _make_ask(api_key: Optional[str], model: Optional[str], provider: str
              ) -> Optional[Callable[[str], str]]:
    """A one-shot question to the model, or None when there is no key.

    Returning None is what makes the controller fall back to its deterministic
    policy, so an unconfigured run still executes rather than failing.
    """
    if not api_key:
        return None

    from ..base_agent import BaseAgent

    class _Controller(BaseAgent):
        """Exists only to reach ``query_llm``; it runs no workflow itself."""

        def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
            return {}

    agent = _Controller("workflow_controller", api_key=api_key, model=model,
                        llm_provider=provider)

    def ask(prompt: str) -> str:
        # Low temperature: this is a dispatch decision, not a piece of writing,
        # and a creative answer here means a tool name that does not exist.
        return agent.query_llm(prompt, temperature=0.0, max_tokens=300)

    return ask


def announced(on_step: Callable[..., Optional[str]],
              announce: Callable[[Dict[str, Any]], None]) -> Callable[..., Optional[str]]:
    """``on_step``, followed by auto mode's "now running" event once it approves.

    Step-by-step answered the approval and then said nothing until the step
    had finished, so the desktop showed "Continuing · proceed" for the whole of
    an inversion and the choice of the next step after it. Auto mode announces
    each step as it starts; this sends the same announcement once the user has
    approved one, and nothing for a step that was skipped or that ended the run.

    Parameters
    ----------
    on_step : callable
        The pause hook, as :func:`run_controller` takes it.
    announce : callable
        Receives the step's :func:`~.modes.announcement`.

    Returns
    -------
    callable
        A hook returning exactly what ``on_step`` returned. An ``announce``
        that raises is ignored, as auto mode ignores it: a display must not
        stop a run.

    Raises
    ------
    None
        What ``on_step`` raises propagates, for the controller to handle as it
        always has.

    Examples
    --------
    >>> from .tools import Tool
    >>> seen = []
    >>> hook = announced(lambda ctx, tool, why: 'proceed', seen.append)
    >>> tool = Tool('invert_ert', 'Invert.', lambda c: ('', {}), label='Run ERT inversion')
    >>> hook(RunContext('goal'), tool, 'the data are loaded')
    'proceed'
    >>> seen[0]['label']
    'Run ERT inversion'
    >>> announced(lambda ctx, tool, why: 'skip', seen.append)(RunContext('g'), tool, '')
    'skip'
    >>> len(seen)
    1
    """
    def hook(ctx: RunContext, tool: Any, reason: str) -> Optional[str]:
        verdict = on_step(ctx, tool, reason)
        # The controller reads a missing answer as "proceed", and so does this.
        if (verdict or PROCEED) not in (STOP, SKIP):
            try:
                announce(announcement(ctx, tool, reason))
            except Exception:  # noqa: BLE001 - a display must not stop a run
                pass
        return verdict

    return hook


def run_workflow(workflow_config: Dict[str, Any], api_key: Optional[str],
                 llm_model: Optional[str], llm_provider: str, output_dir: Any,
                 progress_callback: Optional[Callable[[str, float, str], None]] = None,
                 on_step: Optional[Callable[..., Optional[str]]] = None,
                 ask_user: Optional[Callable[[Dict[str, Any]], str]] = None
                 ) -> Tuple[Dict[str, Any], list, str, Dict[str, str]]:
    """Run one workflow and return what its callers expect.

    Parameters
    ----------
    workflow_config : dict
        The parsed configuration, as ``ContextInputAgent`` produces it.
    api_key, llm_model, llm_provider : str
        Model access. Without a key the run still executes, deterministically.
    output_dir : str or Path
        Where results are written.
    progress_callback : callable, optional
        ``(step, fraction, detail, module)`` for the desktop progress bar; the
        fourth argument is optional and older callbacks still work.
    on_step : callable, optional
        The pause policy. Omit for "auto to report"; pass
        :func:`~PyHydroGeophysX.agents.runtime.modes.step_by_step` to walk the
        run one approved step at a time.
    ask_user : callable, optional
        Answers a question a tool could not decide for itself, given
        ``{"question", "options", "default"}`` and returning an option id.
        Without it a run takes the stated default and records that nobody was
        asked, so a headless run never hangs waiting for an answer.

    Returns
    -------
    tuple
        ``(results, execution_plan, interpretation, report_files)`` - the same
        four values the pipeline returned. ``execution_plan`` is now derived
        from the steps that ran, so it can no longer describe a step that did
        not happen.

    Raises
    ------
    ValueError
        Only when the run produced nothing at all - no artifacts and no
        successful step. Individual tool failures are recorded and reported
        rather than raised, so a run that got most of the way still delivers
        what it has.

    Examples
    --------
    >>> results, plan, text, files = run_workflow(
    ...     {'user_request': 'nothing to do'}, None, None, 'openai', '.')
    >>> plan
    []
    >>> 'Workflow completed' in text
    True
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    ctx = RunContext(
        goal=str(workflow_config.get("user_request") or ""),
        config=workflow_config,
        output_dir=str(output),
        settings={"api_key": api_key, "model": llm_model,
                  "llm_provider": llm_provider,
                  "progress_callback": progress_callback,
                  "ask_user": ask_user},
    )

    def announce(event):
        """Tell the caller where the run is working, before it starts working.

        The progress callback is how the desktop learns which studio module to
        bring forward, so this fires before the step rather than after it - by
        the time a step reports its result the user has missed it happening.
        Callbacks that predate the fourth argument still work: it is optional.
        """
        if progress_callback is None:
            return
        # How far along the run is. The honest answer is that nothing knows:
        # the loop decides each step from what the previous one produced, so
        # the total is not available until the run is over. Two attempts at a
        # denominator both failed - DEFAULT_MAX_STEPS crawled to 31% across a
        # complete six-step run and then jumped to 100, and counting currently
        # runnable tools made the bar go *backwards* (95%, 95%, 73%) because
        # only one tool can run before anything is loaded.
        #
        # So this is a monotone curve that approaches but never reaches the
        # end, which is what an indeterminate run actually looks like. It
        # claims no precision it does not have.
        fraction = min(0.95, 0.1 + 0.85 * (1.0 - 0.7 ** event["step"]))
        try:
            progress_callback(event["label"], fraction, event["reason"],
                              event["module"])
        except TypeError:
            progress_callback(event["label"], fraction, event["reason"])

    def report_result(_ctx, step):
        """Say what the step concluded, as soon as it has concluded it.

        The announcement above can only say what is about to be attempted. The
        studio's module panels hold their own state and the workflow runs
        headless in another process, so without this the user watching a panel
        the run had navigated to saw an empty tool and no evidence anything had
        happened.
        """
        if progress_callback is None:
            return
        event = completion(_ctx, step)
        if not event["summary"]:
            return
        fraction = min(0.95, 0.1 + 0.85 * (1.0 - 0.7 ** event["step"]))
        try:
            progress_callback(event["label"], fraction, event["summary"],
                              event["module"])
        except TypeError:
            progress_callback(event["label"], fraction, event["summary"])

    # One loop, two policies: without a pause hook this runs to the end, with
    # one it stops before each step and asks. Both announce the same events, so
    # the app follows along either way.
    run_controller(ctx, ask=_make_ask(api_key, llm_model, llm_provider),
                   on_step=announced(on_step, announce) if on_step else auto(announce),
                   on_result=report_result)

    failed = [s for s in ctx.steps if s.status == "failed"]
    # A step that failed is reported, not hidden: the request may have named the
    # product it would have made, and an absence nobody states reads as a
    # product nobody wanted.
    for step in failed:
        ctx.note(f"{step.description or step.tool} did not complete: {step.error}")
    # A result that rests on a choice somebody made is not the same as one the
    # data forced, and the report has to be able to say which it is.
    for asked in ctx.questions:
        ctx.note(f"A choice was put to the user: {asked['question']} "
                 f"Answered '{asked['answer']}'.")

    # Raise only when nothing at all worked. The previous condition was
    # `not ctx.has(a, b, c, d)`, and `has` requires *every* named artifact - so
    # it was true whenever any one was missing, which is nearly always. One
    # failed step therefore aborted runs that had produced everything else,
    # which is the opposite of what the loop exists to do: a seismic file with
    # a malformed geometry header killed a run that had gone on to invert the
    # ERT surveys successfully.
    if ctx.steps and not any(step.status == "ok" for step in ctx.steps):
        raise ValueError(f"The workflow produced no result. {failed[0].tool} failed: "
                         f"{failed[0].error}" if failed else
                         "The workflow produced no result.")

    results = ctx.results()
    # The callers read the inversion's own keys off the top level, as they did
    # when the pipeline returned that dictionary directly.
    inversion = ctx.get("inversion_results")
    if isinstance(inversion, dict):
        results = {**inversion, **results}
    results["status"] = "success" if ctx.steps and not failed else results.get(
        "status", "success")
    return results, ctx.plan(), summarise_run(ctx), ctx.get("report_files") or {}


def use_legacy() -> bool:
    """Whether this process should run the pre-controller implementation."""
    return str(os.environ.get(LEGACY_ENV, "")).strip().lower() in {"1", "true", "yes"}
