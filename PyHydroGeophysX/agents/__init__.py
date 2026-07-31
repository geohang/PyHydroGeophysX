"""
Multi-Agent System for Automated Geophysical Workflows.

The package exposes lightweight entry points eagerly and loads heavier agent
classes on demand. This keeps dry-run previews and docs examples usable even
when optional scientific dependencies for a specific agent are not installed.
"""

from importlib import import_module
from typing import Any

from .agent_coordinator import AgentCoordinator
from .base_agent import AgentResult, BaseAgent
from .context_input_agent import ContextInputAgent

_LAZY_IMPORTS = {
    "ERTLoaderAgent": ".ert_loader_agent",
    "ERTInversionAgent": ".ert_inversion_agent",
    "InversionEvaluationAgent": ".inversion_evaluation_agent",
    "WaterContentAgent": ".water_content_agent",
    "ReportAgent": ".report_agent",
    "SeismicAgent": ".seismic_agent",
    "ClimateDataAgent": ".climate_data_agent",
    "DataFusionAgent": ".data_fusion_agent",
    "StructureConstraintAgent": ".structure_constraint_agent",
    "PetrophysicsAgent": ".petrophysics_agent",
    "CodeGenerationAgent": ".code_generation_agent",
    "TDEMAgent": ".tdem_agent",
    "ModelOutputAgent": ".model_output_agent",
    "GeophysicalInversionAgent": ".geophysical_inversion_agent",
}

__all__ = [
    "AgentCoordinator",
    "AgentResult",
    "BaseAgent",
    "ContextInputAgent",
    *_LAZY_IMPORTS.keys(),
]


def __getattr__(name: str) -> Any:
    """Lazily import optional agent classes.

    Parameters
    ----------
    name : str
        Public class name requested from ``PyHydroGeophysX.agents``.

    Returns
    -------
    Any
        Imported class object.

    Raises
    ------
    AttributeError
        If ``name`` is not a public agent export.

    Examples
    --------
    >>> "AgentCoordinator" in __all__
    True
    """
    module_name = _LAZY_IMPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
