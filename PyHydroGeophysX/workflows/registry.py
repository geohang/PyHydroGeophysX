"""Workflow and desktop-module registries."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Callable, Dict, Iterable, Tuple

from .models import RunContext, WorkflowRunResult, WorkflowSpec

WorkflowHandler = Callable[[WorkflowSpec, RunContext], WorkflowRunResult]


@dataclass(frozen=True)
class WorkflowDescriptor:
    workflow_id: str
    handler_path: str
    description: str
    stochastic: bool = False
    module_key: str = ""

    def load_handler(self) -> WorkflowHandler:
        module_name, separator, attribute = self.handler_path.partition(":")
        if not separator:
            raise ValueError(
                f"Handler path for {self.workflow_id!r} must use 'module:attribute'."
            )
        module = importlib.import_module(module_name)
        handler = getattr(module, attribute)
        if not callable(handler):
            raise TypeError(f"Workflow handler {self.handler_path!r} is not callable.")
        return handler


@dataclass(frozen=True)
class ModuleDescriptor:
    navigation_key: str
    result_key: str
    workflow_ids: Tuple[str, ...]
    aliases: Tuple[str, ...] = ()


_WORKFLOWS: Dict[str, WorkflowDescriptor] = {}


def register_workflow(descriptor: WorkflowDescriptor, *, replace: bool = False) -> None:
    existing = _WORKFLOWS.get(descriptor.workflow_id)
    if existing is not None and not replace:
        raise KeyError(f"Workflow {descriptor.workflow_id!r} is already registered.")
    _WORKFLOWS[descriptor.workflow_id] = descriptor


def list_workflows() -> Tuple[WorkflowDescriptor, ...]:
    return tuple(_WORKFLOWS[key] for key in sorted(_WORKFLOWS))


def get_workflow(workflow_id: str) -> WorkflowDescriptor:
    try:
        return _WORKFLOWS[str(workflow_id)]
    except KeyError as exc:
        choices = ", ".join(sorted(_WORKFLOWS))
        raise KeyError(f"Unknown workflow {workflow_id!r}. Available: {choices}") from exc


MODULE_DESCRIPTORS: Dict[str, ModuleDescriptor] = {
    "seismic": ModuleDescriptor(
        "seismic", "seismic_processing", ("seismic.srt_inversion",),
        aliases=("seismic_processing",),
    ),
    "ert": ModuleDescriptor(
        "ert", "ert_processing", ("ert.single_inversion", "ert.timelapse_inversion"),
        aliases=("ert_processing",),
    ),
    "mesh3d": ModuleDescriptor(
        "mesh3d", "mesh3d", ("mesh3d.build", "ert3d.forward"),
    ),
    "em": ModuleDescriptor(
        "em", "em_processing", ("em.inversion",), aliases=("em_processing",),
    ),
    "gravmag": ModuleDescriptor(
        "gravmag", "gravmag_processing",
        ("gravmag.process", "gravmag.forward_bodies", "gravmag.invert"),
        aliases=("gravmag_processing",),
    ),
    "joint_inversion": ModuleDescriptor(
        "joint_inversion", "joint_inversion", ("joint_inversion.run",),
    ),
    "hydro_geophysics": ModuleDescriptor(
        "hydro_geophysics", "hydro_geophysics", ("hydro_geophysics.forward",),
    ),
    "geo_hydrology": ModuleDescriptor(
        "geo_hydrology", "geo_hydrology", ("geo_hydrology.ert_to_wc",),
    ),
    "seismic3d": ModuleDescriptor(
        "seismic3d", "seismic3d", ("seismic3d.build",),
    ),
}


def _register_builtins(descriptors: Iterable[WorkflowDescriptor]) -> None:
    for descriptor in descriptors:
        register_workflow(descriptor)


_register_builtins([
    WorkflowDescriptor(
        "gravmag.process",
        "PyHydroGeophysX.workflows.builtin:run_gravmag_process",
        "QC, regional/residual separation, gridding, and optional profile extraction.",
        module_key="gravmag",
    ),
    WorkflowDescriptor(
        "gravmag.forward_bodies",
        "PyHydroGeophysX.workflows.builtin:run_gravmag_forward_bodies",
        "Analytic gravity or magnetic response for a set of bodies.",
        module_key="gravmag",
    ),
    WorkflowDescriptor(
        "seismic.srt_inversion",
        "PyHydroGeophysX.workflows.builtin:run_srt_inversion",
        "PyGIMLi travel-time inversion from a persisted travel-time artifact.",
        module_key="seismic",
    ),
    WorkflowDescriptor(
        "gravmag.invert",
        "PyHydroGeophysX.workflows.domain:run_gravmag_inversion",
        "SimPEG gravity or magnetic inversion.",
        stochastic=True,
        module_key="gravmag",
    ),
    WorkflowDescriptor(
        "ert.single_inversion",
        "PyHydroGeophysX.workflows.domain:run_ert_single",
        "Single-time ERT inversion from a persisted dataset.",
        module_key="ert",
    ),
    WorkflowDescriptor(
        "ert.timelapse_inversion",
        "PyHydroGeophysX.workflows.domain:run_ert_timelapse",
        "Temporal-regularized ERT inversion from persisted datasets.",
        module_key="ert",
    ),
    WorkflowDescriptor(
        "em.inversion",
        "PyHydroGeophysX.workflows.domain:run_em_inversion",
        "FDEM or TDEM inversion from a persisted sounding.",
        module_key="em",
    ),
    WorkflowDescriptor(
        "mesh3d.build",
        "PyHydroGeophysX.workflows.domain:run_mesh3d",
        "Generate a 3-D ERT mesh from serializable configuration.",
        module_key="mesh3d",
    ),
    WorkflowDescriptor(
        "ert3d.forward",
        "PyHydroGeophysX.workflows.domain:run_ert3d_forward",
        "Generate synthetic 3-D ERT observations on a persisted mesh.",
        stochastic=True,
        module_key="mesh3d",
    ),
    WorkflowDescriptor(
        "joint_inversion.run",
        "PyHydroGeophysX.workflows.domain:run_joint",
        "Run a registered two-method joint inversion.",
        module_key="joint_inversion",
    ),
    WorkflowDescriptor(
        "hydro_geophysics.forward",
        "PyHydroGeophysX.workflows.domain:run_hydro_geophysics",
        "Hydrology-to-geophysics cross-modal forward workflow.",
        stochastic=True,
        module_key="hydro_geophysics",
    ),
    WorkflowDescriptor(
        "geo_hydrology.ert_to_wc",
        "PyHydroGeophysX.workflows.domain:run_geo_hydrology",
        "Seeded ERT-to-water-content uncertainty workflow.",
        stochastic=True,
        module_key="geo_hydrology",
    ),
    WorkflowDescriptor(
        "seismic3d.build",
        "PyHydroGeophysX.workflows.domain:run_seismic3d",
        "Integrate 2-D seismic sections into a 3-D structure model.",
        module_key="seismic3d",
    ),
])


__all__ = [
    "MODULE_DESCRIPTORS",
    "ModuleDescriptor",
    "WorkflowDescriptor",
    "get_workflow",
    "list_workflows",
    "register_workflow",
]
