"""Qt-free LLM provider layer shared by the desktop and web chat agents.

:mod:`PyHydroGeophysX.llm.providers` holds the neutral message state machine
and the OpenAI / Anthropic / OpenAI-compatible adapters. The module used to
live at ``PyHydroGeophysX.qt_apps.agent.providers`` and is still re-exported
from there for compatibility; new code should import from here.
"""

from PyHydroGeophysX.llm import providers  # noqa: F401

__all__ = ["providers"]
