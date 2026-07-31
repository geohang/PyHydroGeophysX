"""Versioned JSON recipe IO."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .models import WorkflowSpec
from .registry import get_workflow


def save_recipe(spec: WorkflowSpec, path: str | Path) -> Path:
    """Validate and save a workflow recipe as UTF-8 JSON."""
    descriptor = get_workflow(spec.workflow_id)
    spec.validate(stochastic=descriptor.stochastic)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(spec.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return destination


def load_recipe(path: str | Path) -> WorkflowSpec:
    source = Path(path)
    payload: Mapping[str, Any] = json.loads(source.read_text(encoding="utf-8"))
    spec = WorkflowSpec.from_dict(payload)
    descriptor = get_workflow(spec.workflow_id)
    spec.validate(stochastic=descriptor.stochastic)
    return spec


__all__ = ["load_recipe", "save_recipe"]

