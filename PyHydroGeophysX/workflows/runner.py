"""Public workflow execution entry point."""

from __future__ import annotations

from typing import Any, Mapping

from .models import RunContext, WorkflowRunResult, WorkflowSpec
from .registry import get_workflow


def run_workflow(
    spec: WorkflowSpec | Mapping[str, Any],
    context: RunContext | None = None,
) -> WorkflowRunResult:
    """Validate and execute one registered workflow."""
    if not isinstance(spec, WorkflowSpec):
        spec = WorkflowSpec.from_dict(spec)
    descriptor = get_workflow(spec.workflow_id)
    spec.validate(stochastic=descriptor.stochastic)
    runtime = context or RunContext()
    runtime.prepare()
    runtime.progress(f"Starting {spec.workflow_id}")
    result = descriptor.load_handler()(spec, runtime)
    if not isinstance(result, WorkflowRunResult):
        raise TypeError(
            f"Workflow {spec.workflow_id!r} returned {type(result).__name__}; "
            "handlers must return WorkflowRunResult."
        )
    result.to_dict()
    runtime.progress(f"Finished {spec.workflow_id}: {result.status}")
    return result


__all__ = ["run_workflow"]

