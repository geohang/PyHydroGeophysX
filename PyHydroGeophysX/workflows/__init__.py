"""Qt-free workflow, recipe, CLI, and code-generation API."""

from .codegen import generate_python
from .bundle import export_workflow_bundle, teaching_paths
from .models import (
    ArtifactRef,
    RunContext,
    WorkflowRunResult,
    WorkflowSpec,
    WorkflowValidationError,
    iter_artifact_refs,
)
from .recipe import load_recipe, save_recipe
from .registry import (
    MODULE_DESCRIPTORS,
    ModuleDescriptor,
    WorkflowDescriptor,
    get_workflow,
    list_workflows,
    module_descriptor_for,
    navigation_key_for,
    register_workflow,
    result_key_for,
)
from .runner import run_workflow
from .walkthrough import generate_notebook, generate_walkthrough

__all__ = [
    "ArtifactRef",
    "MODULE_DESCRIPTORS",
    "ModuleDescriptor",
    "RunContext",
    "WorkflowDescriptor",
    "WorkflowRunResult",
    "WorkflowSpec",
    "WorkflowValidationError",
    "generate_notebook",
    "generate_python",
    "generate_walkthrough",
    "export_workflow_bundle",
    "get_workflow",
    "list_workflows",
    "iter_artifact_refs",
    "load_recipe",
    "module_descriptor_for",
    "navigation_key_for",
    "register_workflow",
    "result_key_for",
    "run_workflow",
    "save_recipe",
    "teaching_paths",
]
