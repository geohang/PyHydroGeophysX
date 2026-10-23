"""A controller loop for the geophysics agents.

Rather than a fixed pipeline, a run is a loop with one shared transcript and
one tool per capability. The controller reads the transcript, chooses the next
action, observes its result, and chooses again. This has three consequences:

- The execution plan is derived from the steps that ran, so it describes the
  run that happened rather than the run someone expected.
- Every tool sees what the earlier steps concluded, not only their numbers: the
  report can say why a result needs review.
- A stage that finds a problem can change what happens next, for example by
  re-running an inversion with different settings.

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
