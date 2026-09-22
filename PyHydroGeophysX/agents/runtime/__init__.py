"""A controller loop for the geophysics agents, in place of a fixed pipeline.

The workflow used to be a classifier followed by one of eight hard-coded
sequences: ``_detect_workflow_type`` picked a branch of a 1770-line ``if/elif``
and that branch called its agents in a fixed order. Three things followed from
that shape, and all three were reported as bugs rather than recognised as one
design:

- The execution plan was written by hand in nine separate places, so it
  described the run someone expected rather than the run that happened. A plan
  listed ``ClimateDataAgent`` on a run whose configuration had climate disabled.
- ``BaseAgent.context`` is per-agent and never shared, so no agent could see
  what another had concluded. The report agent received the evaluation's
  numbers but not its judgement, and could not say why a result needed review.
- Nothing could react. A stage that found a problem had no way to change what
  happened next, beyond one special case for re-running an inversion.

This package replaces the dispatcher with the shape Claude Code uses: one
transcript, one tool per capability, and a controller that reads the transcript,
chooses the next action, observes its result, and chooses again.

- :mod:`~PyHydroGeophysX.agents.runtime.context` is the transcript: the goal,
  every step taken, what each step found, and the artifacts they produced.
- :mod:`~PyHydroGeophysX.agents.runtime.tools` is the registry: each capability
  described well enough for a model to choose it, with the preconditions that
  say when it can run at all.
- :mod:`~PyHydroGeophysX.agents.runtime.controller` is the loop.

The loop degrades to a deterministic policy when there is no model to ask, so a
run without an API key still executes - it simply stops adapting.
"""

from .context import RunContext, Step
from .controller import PROCEED, SKIP, STOP, run_controller
from .modes import announcement, auto, completion, step_by_step
from .tools import TOOLS, Tool, register, runnable_tools

__all__ = ["RunContext", "Step", "Tool", "TOOLS", "register", "runnable_tools",
           "run_controller", "auto", "step_by_step", "announcement",
           "completion", "PROCEED", "SKIP", "STOP"]
