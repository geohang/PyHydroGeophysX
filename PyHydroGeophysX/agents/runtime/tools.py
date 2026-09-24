"""Every capability the workflow has, described well enough to be chosen.

A tool is the unit the controller selects. It is deliberately not the same
thing as an agent class: an agent is an implementation, a tool is an offer -
what it needs, what it produces, and one sentence of what it is for. The
controller never sees ``ERTInversionAgent``; it sees "invert_time_lapse:
recover a resistivity model for every survey at once, needs ert_data".

Three fields carry the weight:

``requires``
    Artifacts that must exist. A tool whose requirements are unmet is never
    offered, which is what makes an impossible sequence unrepresentable rather
    than merely discouraged. The old pipeline enforced ordering by writing the
    calls in order; here the data dependencies say it.

``produces``
    What will exist afterwards, so the controller can reason about reaching a
    goal it has not been given a recipe for.

``description``
    Written for a reader who does not know this codebase, because that is what
    the model is. This is the same reason a tool description matters more than
    a function docstring: nothing else tells the caller when to reach for it.

Handlers take the :class:`~PyHydroGeophysX.agents.runtime.context.RunContext`
and return ``(summary, outputs)``. Raising is allowed and is not fatal - the
controller records the failure as an observation and carries on deciding.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .context import RunContext

#: Every registered tool, by name.
TOOLS: Dict[str, "Tool"] = {}


@dataclass
class Tool:
    """One capability the controller can choose."""

    name: str
    description: str
    handler: Callable[[RunContext], Tuple[str, Dict[str, Any]]]
    requires: Sequence[str] = field(default_factory=tuple)
    produces: Sequence[str] = field(default_factory=tuple)
    #: Which agent class implements it, for the derived execution plan.
    agent: str = ""
    #: A short imperative label for the plan, e.g. "Run time-lapse inversion".
    label: str = ""
    #: The desktop studio module that shows this kind of work, as keyed in
    #: ``PyHydroGeophysX.qt_apps.modules.MODULE_SPECS``. The runtime never
    #: imports Qt; it reports the key and the desktop decides what to do with
    #: it, which is what lets a run walk the user through the studio while it
    #: works instead of leaving them on a progress bar.
    module: str = ""
    #: Extra condition beyond ``requires`` - for tools gated by configuration
    #: rather than by data, such as the climate step.
    when: Optional[Callable[[RunContext], bool]] = None
    #: Tools that make no sense to run twice. The controller will not re-offer
    #: one that has already been *attempted* unless it is marked repeatable.
    repeatable: bool = False

    def available(self, ctx: RunContext) -> bool:
        """Whether this tool can run against the context as it stands."""
        if not ctx.has(*self.requires):
            return False
        # Attempted, not merely succeeded. Excluding only successes leaves a
        # failing tool offered forever: the deterministic policy takes the
        # first offer, so a single failure became twenty-four identical
        # failures and the step that could have salvaged the run was never
        # reached. A tool that should be retried says so with `repeatable`,
        # and then owns the decision to vary what it does.
        if not self.repeatable and ctx.attempted(self.name):
            return False
        if self.when is not None:
            try:
                return bool(self.when(ctx))
            except Exception:  # noqa: BLE001 - a broken gate must not hide the tool
                return True
        return True

    def missing(self, ctx: RunContext) -> List[str]:
        """Requirements that are not satisfied, for a blocked-step message."""
        return [key for key in self.requires if not ctx.has(key)]

    def unavailable_because(self, ctx: RunContext) -> str:
        """Why :meth:`available` is False, in words the controller can act on."""
        absent = self.missing(ctx)
        if absent:
            return f"it needs {', '.join(absent)}, which the run has not produced"
        if not self.repeatable and ctx.attempted(self.name):
            return "it has already been tried in this run and is not repeated"
        return "it does not apply to this run as configured"

    def offer(self) -> str:
        """The tool as one line of the controller's menu."""
        needs = ", ".join(self.requires) or "nothing"
        gives = ", ".join(self.produces) or "a side effect"
        return f"- {self.name}: {self.description} (needs: {needs}; gives: {gives})"


def register(tool: Tool) -> Tool:
    """Add ``tool`` to the registry, replacing any tool of the same name.

    Parameters
    ----------
    tool : Tool
        The tool to register.

    Returns
    -------
    Tool
        The same tool, so this can wrap a definition.

    Raises
    ------
    ValueError
        If the tool has no name or no handler - a nameless tool cannot be
        chosen and a handlerless one cannot be run, and both fail far from
        here.

    Examples
    --------
    >>> t = register(Tool('noop', 'Do nothing.', lambda ctx: ('done', {})))
    >>> TOOLS['noop'] is t
    True
    >>> del TOOLS['noop']
    """
    if not tool.name or tool.handler is None:
        raise ValueError("a tool needs a name and a handler")
    TOOLS[tool.name] = tool
    return tool


def runnable_tools(ctx: RunContext,
                   tools: Optional[Dict[str, "Tool"]] = None) -> List[Tool]:
    """Every tool whose requirements the context currently satisfies.

    Parameters
    ----------
    ctx : RunContext
        The run as it stands.

    Returns
    -------
    list of Tool
        In registration order, which is roughly the order a run needs them -
        so a controller that simply takes the first is already a sane pipeline.

    Examples
    --------
    >>> ctx = RunContext('x')
    >>> only = {'needs_data': Tool('needs_data', 'Needs data.',
    ...                            lambda c: ('', {}), requires=('data',))}
    >>> [t.name for t in runnable_tools(ctx, only)]
    []
    >>> ctx.put('data', [1])
    >>> [t.name for t in runnable_tools(ctx, only)]
    ['needs_data']
    """
    return [tool for tool in (TOOLS if tools is None else tools).values()
            if tool.available(ctx)]


def menu(ctx: RunContext, tools: Optional[Dict[str, "Tool"]] = None) -> str:
    """The runnable tools as the controller sees them, one per line."""
    offers = [tool.offer() for tool in runnable_tools(ctx, tools)]
    return "\n".join(offers) if offers else "- (nothing can run against the current state)"


def invoke(ctx: RunContext, name: str, reason: str = "",
           tools: Optional[Dict[str, "Tool"]] = None) -> str:
    """Run one tool, recording the step whatever happens.

    Only ``requires`` is checked here, not the ``when`` gate or the no-repeat
    rule: a recovery retries a tool that has just failed, which the no-repeat
    rule would refuse. Refusing a tool the menu did not offer is the
    controller's job, before it gets this far.

    Parameters
    ----------
    ctx : RunContext
        The run; the tool reads and writes it.
    name : str
        Registered tool name.
    reason : str
        Why this step was chosen, for the transcript and the plan.

    Returns
    -------
    str
        The step's status: ``'ok'``, ``'failed'``, ``'blocked'`` or
        ``'skipped'`` (the last for an unknown tool name).

    Raises
    ------
    None
        A tool that raises is recorded as failed. The loop has to keep its
        ability to decide what to do about a failure, which it loses if the
        exception propagates out of the run.

    Examples
    --------
    >>> ctx = RunContext('x')
    >>> only = {'boom': Tool('boom', 'Raise.', lambda c: 1 / 0)}
    >>> invoke(ctx, 'boom', tools=only)
    'failed'
    >>> ctx.steps[-1].error.startswith('ZeroDivisionError')
    True
    >>> invoke(ctx, 'no_such_tool', tools=only)
    'skipped'
    """
    tool = (TOOLS if tools is None else tools).get(name)
    if tool is None:
        ctx.begin(name, reason)
        ctx.finish(status="skipped",
                   error=f"No tool named '{name}'. Choose one that is offered.")
        return "skipped"
    ctx.begin(name, reason, agent=tool.agent,
              description=tool.label or tool.name, purpose=tool.description)
    absent = tool.missing(ctx)
    if absent:
        ctx.finish(status="blocked",
                   error=f"Needs {', '.join(absent)}, which the run has not produced.")
        return "blocked"
    try:
        summary, outputs = tool.handler(ctx)
    except Exception as exc:  # noqa: BLE001 - a failure is an observation
        ctx.finish(status="failed", error=f"{type(exc).__name__}: {exc}")
        return "failed"
    from .context import merge_outputs
    merge_outputs(ctx, outputs or {})
    ctx.finish(summary=summary or "Completed.")
    return "ok"
