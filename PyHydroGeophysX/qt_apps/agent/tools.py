"""Provider-neutral tool specs for the AQUAH workbench command layer.

Each spec is ``{"name", "description", "parameters"(JSON schema)}``. The provider
adapters in :mod:`providers` wrap these into OpenAI ``function`` tools or
Anthropic ``input_schema`` tools. The set is deliberately small and generic: the
model learns each module's actions at run time via ``describe_current_module``
and acts through ``apply_action``, so a new module needs no new tool here.
"""

from __future__ import annotations

from typing import Any, Dict, List


def tool_specs() -> List[Dict[str, Any]]:
    """Return the neutral tool spec list for the workbench command layer."""
    return [
        {
            "name": "list_modules",
            "description": "List the workbench modules that can be opened, with their keys and titles.",
            "parameters": {"type": "object", "properties": {}},
        },
        {
            "name": "navigate",
            "description": "Open a workbench module by key (e.g. 'hydro_geophysics'). Returns a description of the opened module.",
            "parameters": {
                "type": "object",
                "properties": {
                    "module": {
                        "type": "string",
                        "description": "Module key from list_modules, e.g. 'hydro_geophysics'.",
                    }
                },
                "required": ["module"],
            },
        },
        {
            "name": "describe_current_module",
            "description": (
                "Describe the currently open module: its supported actions and current parameter "
                "values. Call this before apply_action so you use real action names and argument keys."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
        {
            "name": "apply_action",
            "description": (
                "Run one action on the current module (for example set parameters, pick a profile, "
                "choose methods, or run processing). Use describe_current_module first to learn valid "
                "action names and argument keys."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "Action name reported by describe_current_module, e.g. 'select_methods' or 'run'.",
                    },
                    "args": {
                        "type": "object",
                        "description": "Arguments for the action as a JSON object. May be empty.",
                        "additionalProperties": True,
                    },
                },
                "required": ["action"],
            },
        },
        {
            "name": "get_workbench_state",
            "description": "Read the overall workbench state: project context, the selected module, and which modules already have results.",
            "parameters": {"type": "object", "properties": {}},
        },
    ]
