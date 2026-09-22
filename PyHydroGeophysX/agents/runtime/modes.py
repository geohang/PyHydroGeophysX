"""Auto and step-by-step, as two hooks on one loop.

The studio offers the same work two ways: "Auto to report" runs to the end, and
the chat walks through it a step at a time. Those were separate code paths -
different dispatch, different ordering, different error handling - so a step
could behave one way when a user approved it and another when the workflow ran
it unattended, and a fix to one did not reach the other.

They are now the same :func:`~PyHydroGeophysX.agents.runtime.controller.run_controller`
over the same registry, differing only in the ``on_step`` hook they supply:

- :func:`auto` never pauses. It announces the step and where in the studio it is
  happening, so the app can bring that module to the front while the work runs.
- :func:`step_by_step` asks first and honours the answer, including "skip this
  one" and "stop here".

Both announce the same thing in the same order, which is what makes the two
modes feel like one feature rather than two implementations.

Nothing here imports Qt. A hook reports a module *key*; what the desktop does
with it - navigate, highlight, ignore - is the desktop's decision, and the
Streamlit app and the tests can pass their own.
"""

from typing import Any, Callable, Dict, List, Optional

from .context import RunContext
from .controller import PROCEED, SKIP, STOP

#: What an announcement carries. Kept small and JSON-safe on purpose: the
#: desktop runs the workflow in a separate process and this crosses it as one
#: line of JSON.
def announcement(ctx: RunContext, tool: Any, reason: str) -> Dict[str, Any]:
    """What the app is told about the step that is about to run.

    Parameters
    ----------
    ctx : RunContext
        The run as it stands.
    tool : Tool or None
        The tool about to run. None when the controller named something that
        is not registered, which is itself worth announcing.
    reason : str
        Why the controller chose it.

    Returns
    -------
    dict
        ``{"tool", "label", "module", "reason", "step"}`` - all JSON-safe.
        ``module`` is a studio module key or ``""``.

    Raises
    ------
    None

    Examples
    --------
    >>> from .tools import Tool
    >>> ctx = RunContext('goal')
    >>> tool = Tool('invert_ert', 'Invert.', lambda c: ('', {}),
    ...             label='Run ERT inversion', module='ert')
    >>> announcement(ctx, tool, 'the data are loaded') == {
    ...     'tool': 'invert_ert', 'label': 'Run ERT inversion', 'module': 'ert',
    ...     'reason': 'the data are loaded', 'step': 1}
    True
    >>> announcement(ctx, None, '')['label']
    'Unknown step'
    """
    return {"tool": getattr(tool, "name", "") or "",
            "label": (getattr(tool, "label", "") or getattr(tool, "name", "")
                      or "Unknown step"),
            "module": getattr(tool, "module", "") or "",
            "reason": str(reason or ""),
            "step": len(ctx.steps) + 1}


def completion(ctx: RunContext, step: Any) -> Dict[str, Any]:
    """What the app is told about a step that has just finished.

    The counterpart to :func:`announcement`. That one fires before a step and so
    can only say what is about to happen; this carries what the step actually
    concluded, which is the only evidence of work a user can see while a run is
    still going - the studio's own panels stay empty, because the workflow runs
    headless in its own process.

    Returns
    -------
    dict
        ``{"tool", "label", "module", "summary", "status", "produced", "step"}``
        - all JSON-safe, since this crosses a process boundary as one line.

    Examples
    --------
    >>> ctx = RunContext('goal')
    >>> ctx.begin('load_ert_surveys', 'it is first',
    ...           description='Load ERT data')
    ... # doctest: +ELLIPSIS
    Step(tool='load_ert_surveys', ...)
    >>> ctx.put('ert_data', [1, 2])
    >>> ctx.finish('Loaded 5 surveys, 812 measurements.')
    >>> event = completion(ctx, ctx.steps[-1])
    >>> event['summary'], event['status'], event['produced']
    ('Loaded 5 surveys, 812 measurements.', 'ok', ['ert_data'])
    """
    from .tools import TOOLS

    tool = TOOLS.get(getattr(step, "tool", ""))
    return {"tool": getattr(step, "tool", "") or "",
            "label": (getattr(step, "description", "")
                      or getattr(tool, "label", "") or getattr(step, "tool", "")
                      or "Unknown step"),
            "module": getattr(tool, "module", "") or "",
            "summary": str(getattr(step, "summary", "")
                           or getattr(step, "error", "")),
            "status": str(getattr(step, "status", "")),
            "produced": list(getattr(step, "produced", []) or []),
            "step": len(ctx.steps)}


def auto(announce: Optional[Callable[[Dict[str, Any]], None]] = None
         ) -> Callable[..., str]:
    """An ``on_step`` hook that runs straight through, narrating as it goes.

    Parameters
    ----------
    announce : callable, optional
        Receives the :func:`announcement` for each step before it runs. The
        desktop uses it to bring the relevant module to the front, so the user
        watches the work happen rather than a progress bar.

    Returns
    -------
    callable
        Suitable as ``run_controller(..., on_step=...)``. Always proceeds; an
        announcement that raises is ignored, because a display problem must not
        stop a computation.

    Raises
    ------
    None

    Examples
    --------
    >>> from .tools import Tool
    >>> seen = []
    >>> hook = auto(seen.append)
    >>> tool = Tool('invert_ert', 'Invert.', lambda c: ('', {}), module='ert')
    >>> hook(RunContext('goal'), tool, 'because')
    'proceed'
    >>> seen[0]['module']
    'ert'
    >>> auto(lambda event: 1 / 0)(RunContext('g'), tool, '')
    'proceed'
    """
    def hook(ctx: RunContext, tool: Any, reason: str) -> str:
        if announce is not None:
            try:
                announce(announcement(ctx, tool, reason))
            except Exception:  # noqa: BLE001 - a display must not stop a run
                pass
        return PROCEED

    return hook


def step_by_step(approve: Callable[[Dict[str, Any]], Any],
                 announce: Optional[Callable[[Dict[str, Any]], None]] = None
                 ) -> Callable[..., str]:
    """An ``on_step`` hook that asks before each step and honours the answer.

    Parameters
    ----------
    approve : callable
        Receives the :func:`announcement` and returns what to do: ``True`` or
        ``'proceed'`` to run it, ``'skip'`` to leave it out, ``'stop'`` or
        ``False`` to end the run here. Blocking is expected - in the desktop
        this is where the user reads the step and presses a button.
    announce : callable, optional
        Called before ``approve``, with the same event, so the app can navigate
        to the module first and let the user see the panel they are deciding
        about.

    Returns
    -------
    callable
        Suitable as ``run_controller(..., on_step=...)``.

    Raises
    ------
    None
        An ``approve`` that raises is treated as "stop": a run whose approval
        gate is broken must not carry on unapproved.

    Examples
    --------
    >>> from .tools import Tool
    >>> tool = Tool('invert_ert', 'Invert.', lambda c: ('', {}), module='ert')
    >>> ctx = RunContext('goal')
    >>> step_by_step(lambda event: True)(ctx, tool, '')
    'proceed'
    >>> step_by_step(lambda event: 'skip')(ctx, tool, '')
    'skip'
    >>> step_by_step(lambda event: False)(ctx, tool, '')
    'stop'
    >>> step_by_step(lambda event: 1 / 0)(ctx, tool, '')
    'stop'
    """
    def hook(ctx: RunContext, tool: Any, reason: str) -> str:
        event = announcement(ctx, tool, reason)
        if announce is not None:
            try:
                announce(event)
            except Exception:  # noqa: BLE001 - a display must not stop a run
                pass
        try:
            answer = approve(event)
        except Exception:  # noqa: BLE001 - an unanswerable gate is not consent
            return STOP
        if answer is True or answer == PROCEED:
            return PROCEED
        if answer is False or answer == STOP:
            return STOP
        if answer == SKIP:
            return SKIP
        # Anything else is not an approval. Defaulting to "proceed" here would
        # run unapproved work on a typo.
        return STOP

    return hook


def collect(events: List[Dict[str, Any]]) -> Callable[[Dict[str, Any]], None]:
    """An ``announce`` that appends to a list, for tests and for replay.

    Examples
    --------
    >>> seen = []
    >>> collect(seen)({'tool': 'x'})
    >>> seen
    [{'tool': 'x'}]
    """
    def announce(event: Dict[str, Any]) -> None:
        events.append(event)

    return announce
