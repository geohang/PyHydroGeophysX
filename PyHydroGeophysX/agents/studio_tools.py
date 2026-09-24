"""Provider-neutral tool specs for the AQUAH studio command layer.

Each spec is ``{"name", "description", "parameters"(JSON schema)}``. The provider
adapters in :mod:`providers` wrap these into OpenAI ``function`` tools or
Anthropic ``input_schema`` tools. The set is deliberately small and generic: the
model learns each module's actions at run time from the ``navigate`` result (or
``describe_current_module``, which returns the same description) and acts
through ``apply_action``, so a new module needs no new tool here.

``capture_view`` is the one conditional entry: it returns a picture, so it is
only offered when the selected model can read images.
"""

from __future__ import annotations

from typing import Any, Dict, List


def tool_specs(vision: bool = False) -> List[Dict[str, Any]]:
    """Return the neutral tool spec list for the studio command layer.

    Pass ``vision=True`` when the selected model accepts image input, which adds
    the ``capture_view`` screenshot tool.
    """
    specs = [
        {
            "name": "list_modules",
            "description": "List the studio modules that can be opened, with their keys and titles.",
            "parameters": {"type": "object", "properties": {}},
        },
        {
            "name": "navigate",
            "description": (
                "Open a studio module by key (e.g. 'hydro_geophysics'). Returns the opened "
                "module's description under 'current_module' - its actions with their argument "
                "keys, its current state and its views - so no describe_current_module call "
                "is needed after it."
            ),
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
                "values. navigate already returns this for the module it opens; call this to "
                "refresh the state later, or when no navigate result is at hand."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
        {
            "name": "apply_action",
            "description": (
                "Run one action on the current module (for example set parameters, pick a profile, "
                "choose methods, or run processing). Use the action names and argument keys from "
                "the module's description, as navigate or describe_current_module returned it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "Action name from the module's description, e.g. 'select_methods' or 'run'.",
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
            "name": "get_studio_state",
            "description": "Read the overall studio state: project context, the selected module, and which modules already have results.",
            "parameters": {"type": "object", "properties": {}},
        },
    ]
    if vision:
        specs.append({
            "name": "capture_view",
            "description": (
                "Take a picture of a panel in the current module and look at it. Use this to "
                "check a result you cannot judge from numbers alone: whether an inversion "
                "section has artefacts, whether first-break picks follow the real arrival, "
                "whether a mesh or 3D model looks right. The module's description (from navigate "
                "or describe_current_module) lists the view names under 'views'; omit 'view' "
                "to capture the whole page. "
                "Some views also return a 'context' field of exact values for what the plot "
                "shows (for a seismic gather, the pick time of every trace). Those numbers are "
                "authoritative for identity; read quality from the image, not indices."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "view": {
                        "type": "string",
                        "description": "View name from the 'views' list of the module's description, e.g. 'gather' or 'quality'.",
                    }
                },
            },
        })
    return specs
