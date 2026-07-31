"""Deterministic workflow code export with optional LLM explanation."""

from __future__ import annotations

import json
from pathlib import Path
import traceback
from typing import Any, Dict, Mapping, Optional

from .base_agent import BaseAgent
from PyHydroGeophysX.workflows import (
    WorkflowSpec,
    export_workflow_bundle,
    load_recipe,
)


class CodeGenerationAgent(BaseAgent):
    """Export reproducible workflow code without asking an LLM to write code.

    LLM access is limited to prose explanations and parameter suggestions. The
    executable file always comes from the versioned workflow generator.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        llm_provider: str = "openai",
    ) -> None:
        super().__init__("code_generator", api_key, model, llm_provider)
        self.system_message = (
            "Explain PyHydroGeophysX workflow recipes and suggest parameter "
            "changes. Never invent or execute Python code."
        )

    @staticmethod
    def _spec_from_input(input_data: Mapping[str, Any]) -> WorkflowSpec:
        recipe_path = input_data.get("recipe_path")
        if recipe_path:
            return load_recipe(str(recipe_path))
        raw = input_data.get("workflow_spec")
        if isinstance(raw, WorkflowSpec):
            return raw
        if isinstance(raw, Mapping):
            return WorkflowSpec.from_dict(raw)
        workflow_id = str(input_data.get("workflow_id") or "").strip()
        if workflow_id:
            seed = input_data.get("seed")
            return WorkflowSpec(
                workflow_id=workflow_id,
                inputs=dict(input_data.get("inputs") or {}),
                parameters=dict(input_data.get("parameters") or {}),
                seed=None if seed is None else int(seed),
                dependencies=list(input_data.get("dependencies") or []),
                metadata=dict(input_data.get("metadata") or {}),
            )
        raise ValueError(
            "A recipe_path, workflow_spec, or workflow_id is required. "
            "LLM-authored executable code is no longer supported."
        )

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a spec and export its recipe plus deterministic Python."""
        self._log_execution("Exporting deterministic workflow code")
        try:
            spec = self._spec_from_input(input_data)
            output_dir = Path(input_data.get("output_dir") or "results/generated")
            stem = str(input_data.get("stem") or spec.workflow_id.replace(".", "_"))
            recipe_path, code_path = export_workflow_bundle(
                spec, output_dir, stem=stem
            )
            code = code_path.read_text(encoding="utf-8")
            interpretation = self.explain_recipe(
                spec,
                user_request=str(input_data.get("user_request") or ""),
            )
            self.results = {
                "status": "success",
                "code": code,
                "code_file": str(code_path),
                "recipe_file": str(recipe_path),
                "output": (
                    "Recipe and deterministic runner exported. "
                    "The agent did not execute generated code."
                ),
                "error": None,
                "interpretation": interpretation,
                "output_dir": str(output_dir),
                "workflow_id": spec.workflow_id,
            }
        except Exception as exc:  # noqa: BLE001
            self.results = {
                "status": "failed",
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "interpretation": (
                    "Materialize every workflow input and provide a valid recipe "
                    "before exporting code."
                ),
            }
        return self.results

    def explain_recipe(
        self,
        spec: WorkflowSpec,
        *,
        user_request: str = "",
    ) -> str:
        """Optionally ask the LLM for prose; never use its response as code."""
        if not self.api_key:
            return (
                f"Exported workflow {spec.workflow_id!r} with "
                f"{len(spec.inputs)} input field(s) and "
                f"{len(spec.parameters)} parameter field(s)."
            )
        prompt = (
            "Explain this workflow recipe in concise user-facing prose. Mention "
            "its inputs, outputs, seed, and portability considerations. Do not "
            "write code.\n\n"
            f"User request: {user_request or 'not provided'}\n"
            f"Recipe:\n{json.dumps(spec.to_dict(), indent=2, sort_keys=True)}"
        )
        try:
            return self.query_llm(
                prompt, self.system_message, temperature=0.2, max_tokens=400
            )
        except Exception:  # noqa: BLE001
            return f"Exported deterministic workflow {spec.workflow_id!r}."

    def suggest_parameters(
        self,
        spec: WorkflowSpec,
        user_request: str,
    ) -> str:
        """Return advisory suggestions without mutating the recipe."""
        if not self.api_key:
            return "Parameter suggestions require an LLM API key."
        prompt = (
            "Suggest parameter changes for this recipe. Be explicit that they "
            "are advisory and return prose only; do not write code.\n\n"
            f"Request: {user_request}\n"
            f"Recipe: {json.dumps(spec.to_dict(), sort_keys=True)}"
        )
        return self.query_llm(
            prompt, self.system_message, temperature=0.3, max_tokens=400
        )

    def check_request_scope(
        self,
        user_request: str,
        workflow_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Report whether a serializable registered workflow was supplied."""
        has_recipe = bool(
            workflow_config.get("recipe_path")
            or workflow_config.get("workflow_spec")
            or workflow_config.get("workflow_id")
        )
        return {
            "in_scope": has_recipe,
            "out_of_scope_parts": [] if has_recipe else [user_request or "custom request"],
            "recommendation": (
                "Export the registered workflow recipe."
                if has_recipe
                else "Map the request to a registered workflow and materialize its inputs."
            ),
        }

    def _log_execution(self, message: str, level: str = "INFO") -> None:
        print(f"[{self.name}] [{level}] {message}")


__all__ = ["CodeGenerationAgent"]
