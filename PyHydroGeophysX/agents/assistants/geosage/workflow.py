"""GeoSAGE's run: what "Auto to report" does when GeoSAGE is the assistant.

Follows :data:`PyHydroGeophysX.agents.assistants.WORKFLOW_CONTRACT`. The loop,
the step events the studio shows, step-by-step approval and questions to the
user all come from :func:`PyHydroGeophysX.agents.runtime.entry.drive`; what is
GeoSAGE's own is the configuration (:func:`configure`), the tools
(:mod:`.tools`) and the decision prompt (:data:`CONTROLLER_PROMPT`).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional

#: How the controller is told what kind of run this is. ``{transcript}`` and
#: ``{menu}`` are filled in by the controller.
CONTROLLER_PROMPT = """You are running GeoSAGE, a geological-interpretation workflow \
from joint gravity and magnetic inversion. Choose the single next action.

{transcript}

Actions you can take now:
{menu}

Reply with JSON only - no prose, no code fence. Give your reasoning first:
{{"why": "<one or two sentences: what the run has shown so far, and so what to do next>", "tool": "<name from the list>"}}
or, when the reviewed report exists:
{{"why": "<one short sentence>", "done": true}}

Rules:
- Choose only from the list above. A tool not listed cannot run yet, usually
  because something it needs has not been produced.
- Compile the geological priors before building the quasi-geological model when
  the user supplied geological information; without priors the grouping falls
  back to GMM/BIC, and the report must say so.
- The report is final only once it has been reviewed.
- If a step failed, do not repeat it unchanged; move on and let the report state
  what is missing.
"""


def configure(payload: Dict[str, Any]) -> Dict[str, Any]:
    """The run's configuration from the request: GeoSAGE's ContextAgent.

    Placeholder for the port: today it only carries the request and the input
    files through. The ContextAgent turns the request into a complete
    executable configuration - data fields, inversion settings, grouping
    settings - on top of the defaults.
    """
    config = {"user_request": str(payload.get("request") or "").strip()}
    config.update(payload.get("inputs") or {})
    return config


def run(payload: Dict[str, Any], progress: Callable[..., None], *,
        approve: Optional[Callable[..., Any]] = None,
        on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
        events: Optional[list] = None, **_hooks: Any) -> Dict[str, Any]:
    """Run GeoSAGE on ``payload`` and return the studio's result."""
    from PyHydroGeophysX.agents.runtime.context import RunContext
    from PyHydroGeophysX.agents.runtime.entry import drive, make_ask
    from PyHydroGeophysX.agents.runtime.modes import step_by_step

    from .tools import TOOLS

    request = str(payload.get("request") or "").strip()
    if not request:
        raise ValueError("Describe the exploration objective you want GeoSAGE to work on.")
    output = Path(payload["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    # The agents call Claude 'claude'; the chat adapters call it 'anthropic'.
    provider = {"anthropic": "claude"}.get(payload.get("provider") or "openai",
                                           payload.get("provider") or "openai")
    progress("Understanding request", 0.02, "Preparing the GeoSAGE configuration")
    config = configure(payload)
    ctx = RunContext(goal=request, config=config, output_dir=str(output),
                     settings={"api_key": payload.get("api_key"),
                               "model": payload.get("model"),
                               "llm_provider": provider,
                               "progress_callback": progress,
                               "ask_user": approve})
    on_step = step_by_step(approve) if payload.get("step_mode") and approve else None
    drive(ctx, tools=TOOLS,
          ask=make_ask(payload.get("api_key"), payload.get("model"), provider),
          progress_callback=progress, on_step=on_step, on_event=on_event,
          prompt=CONTROLLER_PROMPT, finish="report_files", recovery=False)

    failed = [step for step in ctx.steps if step.status == "failed"]
    if ctx.steps and not any(step.status == "ok" for step in ctx.steps):
        raise RuntimeError(f"GeoSAGE produced nothing: {failed[0].error}" if failed
                           else "GeoSAGE produced nothing.")
    warnings = [f"{step.description or step.tool} did not complete: {step.error}"
                for step in failed] + list(ctx.warnings)
    reported = ctx.has("report_files")
    status = "incomplete" if not reported else ("needs_review" if warnings else "success")
    draft = ctx.get("draft_report")
    interpretation = (draft.get("summary") if isinstance(draft, dict) else None) or (
        "GeoSAGE finished; the reviewed report is in the run folder." if reported else
        "GeoSAGE stopped before a reviewed report was written.")
    return {"status": status, "interpretation": interpretation, "warnings": warnings,
            "report_files": ctx.get("report_files") or {}, "output_dir": str(output),
            "execution_plan": ctx.plan(), "workflow_config": config}
