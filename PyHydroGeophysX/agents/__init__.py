"""
Multi-Agent System for Automated Geophysical Workflows.

The package exposes lightweight entry points eagerly and loads heavier agent
classes on demand. This keeps dry-run previews and docs examples usable even
when optional scientific dependencies for a specific agent are not installed.
"""

import warnings
from importlib import import_module
from typing import Any

from PyHydroGeophysX._internal.deprecations import DEPRECATED_IN, register_removed_module

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
    "TDEMAgent": ".tdem_agent",
    "ModelOutputAgent": ".model_output_agent",
}

__all__ = [
    "AgentCoordinator",
    "AgentResult",
    "BaseAgent",
    "ContextInputAgent",
    *_LAZY_IMPORTS.keys(),
]

#: Agents removed in 0.5.0, and what does their job now. Asking for one of
#: these names warns with its replacement, so a script that only imports it
#: keeps running; creating one raises the same message.
_REMOVED_AGENTS = {
    "WorkflowOrchestratorAgent": (
        "the controller behind BaseAgent.run_unified_agent_workflow() now chooses "
        "each step of a run, and AgentCoordinator runs the fixed ERT pipeline"),
    "CodeGenerationAgent": (
        "export a workflow's code with PyHydroGeophysX.workflows.export_workflow_bundle()"),
    "GeophysicalInversionAgent": (
        "use ERTInversionAgent, SeismicAgent or TDEMAgent, or run SRTInversion, "
        "TimeLapseSRTInversion, FDEMInversion or JointERTSRTInversion directly"),
}


#: The modules that held them in 0.3.0, for code that imported the module path.
_REMOVED_MODULES = {
    "workflow_orchestrator_agent": "WorkflowOrchestratorAgent",
    "code_generation_agent": "CodeGenerationAgent",
    "geophysical_inversion_agent": "GeophysicalInversionAgent",
}


def _stand_in(name: str) -> type:
    """A class named ``name`` that refuses to run, saying what replaced it."""
    message = f"{name} was removed in PyHydroGeophysX {DEPRECATED_IN}: {_REMOVED_AGENTS[name]}."

    def refuse(self, *args, **kwargs):
        raise RuntimeError(message)

    return type(name, (), {"__init__": refuse, "__doc__": message, "__module__": __name__})


def _removed_agent(name: str, stacklevel: int = 2) -> type:
    """Warn that ``name`` was removed, and return a stand-in that refuses to run.

    ``stacklevel`` counts from the caller, so the warning points at the line
    that asked for the name.
    """
    stand_in = _stand_in(name)
    warnings.warn(stand_in.__doc__, DeprecationWarning, stacklevel=stacklevel + 1)
    return stand_in


# Importing the old module path warns once, with the same replacement; its
# class is the same refusing stand-in.
for _module, _name in _REMOVED_MODULES.items():
    register_removed_module(
        f"{__name__}.{_module}",
        f"{__name__}.{_module} was removed in PyHydroGeophysX {DEPRECATED_IN}: "
        f"{_REMOVED_AGENTS[_name]}.",
        lambda module, name=_name: setattr(module, name, _stand_in(name)))
del _module, _name

#: The PyDaymet script that ClimateDataAgent.fetch_climate_data_with_conda()
#: ran in a conda environment of its own in 0.3.0; the module path warns the
#: same way, and its main() refuses with the same message.
_CLIMATE_SCRIPT = f"{__name__}.fetch_climate_data"
_CLIMATE_SCRIPT_REMOVED = (
    f"{_CLIMATE_SCRIPT} was removed in PyHydroGeophysX {DEPRECATED_IN}: climate data "
    "now come from the Open-Meteo historical-weather API (ERA5), with no conda "
    "environment; pass the coordinates and dates to ClimateDataAgent.execute(), "
    "with an output_dir to save the series as CSV.")


def _climate_script(module) -> None:
    """The 0.3.0 script's names: its variable list, and a ``main`` that refuses."""
    def main(argv=None):
        raise RuntimeError(_CLIMATE_SCRIPT_REMOVED)

    module.DEFAULT_VARIABLES = ["prcp", "tmin", "tmax", "srad", "vp", "dayl"]
    module.main = main


register_removed_module(_CLIMATE_SCRIPT, _CLIMATE_SCRIPT_REMOVED, _climate_script)


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
    if name in _REMOVED_AGENTS:
        return _removed_agent(name)
    module_name = _LAZY_IMPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
