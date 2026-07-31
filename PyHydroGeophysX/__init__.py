"""Public, import-light API for :mod:`PyHydroGeophysX`.

The package root intentionally contains no eager scientific imports.  Public
objects are resolved on first access so importing a lightweight submodule does
not also initialize plotting libraries or optional geophysics engines.
"""

from __future__ import annotations

import importlib
import os as _os
from typing import Dict, Tuple

__version__ = "0.4.0"

# Keep qtpy-based optional packages on the same Qt binding as the desktop app.
# ``setdefault`` continues to respect an explicit user override.
_os.environ.setdefault("QT_API", "pyside6")

_Export = Tuple[str, str]


def _exports(module: str, *names: str) -> Dict[str, _Export]:
    return {name: (module, name) for name in names}


_EXPORTS: Dict[str, _Export] = {}
_EXPORTS.update(_exports(
    "PyHydroGeophysX.workflows",
    "ArtifactRef",
    "RunContext",
    "WorkflowRunResult",
    "WorkflowSpec",
    "generate_python",
    "get_workflow",
    "list_workflows",
    "load_recipe",
    "run_workflow",
    "save_recipe",
))
_EXPORTS.update(_exports(
    "PyHydroGeophysX.core.interpolation",
    "ProfileInterpolator",
    "interpolate_to_profile",
    "setup_profile_coordinates",
    "interpolate_structure_to_profile",
    "prepare_2D_profile_data",
    "interpolate_to_mesh",
))
_EXPORTS.update(_exports(
    "PyHydroGeophysX.core.mesh_utils",
    "MeshCreator",
    "create_mesh_from_layers",
))
_EXPORTS.update(_exports(
    "PyHydroGeophysX.model_output.water_content",
    "MODFLOWWaterContent",
    "MODFLOWPorosity",
))
_EXPORTS["HydroModelOutput"] = (
    "PyHydroGeophysX.model_output.base",
    "HydroModelOutput",
)
_EXPORTS.update(_exports(
    "PyHydroGeophysX.model_output.parflow_output",
    "ParflowSaturation",
    "ParflowPorosity",
))
_EXPORTS.update(_exports(
    "PyHydroGeophysX.forward.ert_forward",
    "ERTForwardModeling",
    "ertforward",
    "ertforward2",
    "ertforandjac",
    "ertforandjac2",
))
_EXPORTS.update(_exports(
    "PyHydroGeophysX.inversion.base",
    "InversionResult",
    "TimeLapseInversionResult",
    "InversionBase",
))
_EXPORTS["ERTInversion"] = (
    "PyHydroGeophysX.inversion.ert_inversion",
    "ERTInversion",
)
_EXPORTS["TimeLapseERTInversion"] = (
    "PyHydroGeophysX.inversion.time_lapse",
    "TimeLapseERTInversion",
)
_EXPORTS["WindowedTimeLapseERTInversion"] = (
    "PyHydroGeophysX.inversion.windowed",
    "WindowedTimeLapseERTInversion",
)
_EXPORTS.update(_exports(
    "PyHydroGeophysX.solvers.linear_solvers",
    "generalized_solver",
    "LinearSolver",
    "CGLSSolver",
    "LSQRSolver",
    "RRLSQRSolver",
    "RRLSSolver",
    "direct_solver",
    "TikhonvRegularization",
    "IterativeRefinement",
    "get_optimal_solver",
))
_EXPORTS.update(_exports(
    "PyHydroGeophysX.petrophysics.resistivity_models",
    "water_content_to_resistivity",
    "resistivity_to_water_content",
    "resistivity_to_saturation",
))
_EXPORTS.update(_exports(
    "PyHydroGeophysX.petrophysics.velocity_models",
    "BaseVelocityModel",
    "VRHModel",
    "BrieModel",
    "DEMModel",
    "HertzMindlinModel",
    "VRH_model",
    "satK",
    "velDEM",
    "vel_porous",
))
_EXPORTS.update(_exports(
    "PyHydroGeophysX.data_processing",
    "load_ert_resipy",
    "qc_and_visualize",
    "export_for_inversion",
    "LocalRef",
    "FirstBreakPick",
    "read_seg2_seismic",
    "read_segy",
    "apply_agc",
    "normalize_traces",
    "pick_first_breaks",
    "export_first_breaks",
    "first_breaks_to_traveltime",
    "read_tem_fast",
    "export_to_vtk",
    "export_results_to_csv",
))
_EXPORTS.update(_exports(
    "PyHydroGeophysX.agents",
    "AgentCoordinator",
    "ERTLoaderAgent",
    "ERTInversionAgent",
    "WaterContentAgent",
    "ReportAgent",
    "SeismicAgent",
    "GeophysicalInversionAgent",
))
_EXPORTS.update(_exports(
    "PyHydroGeophysX.forward.fdem_forward",
    "FDEMForwardModeling",
    "FDEMSurveyConfig",
))
_EXPORTS["SRTInversion"] = (
    "PyHydroGeophysX.inversion.srt_inversion",
    "SRTInversion",
)
_EXPORTS["TimeLapseSRTInversion"] = (
    "PyHydroGeophysX.inversion.srt_time_lapse",
    "TimeLapseSRTInversion",
)
_EXPORTS.update(_exports(
    "PyHydroGeophysX.inversion.fdem_inversion",
    "FDEMInversion",
    "FDEMInversionResult",
))
_EXPORTS["GeophysicalInversion"] = (
    "PyHydroGeophysX.inversion.multi_method",
    "GeophysicalInversion",
)
_EXPORTS.update(_exports(
    "PyHydroGeophysX.inversion.cross_constraints",
    "StructuralConstraint",
    "PetrophysicalCoupling",
))
_EXPORTS.update(_exports(
    "PyHydroGeophysX.inversion.joint_api",
    "JointInversionRequest",
    "JointInversionResult",
    "JointPairCapability",
    "get_joint_capabilities",
))
_EXPORTS.update(_exports(
    "PyHydroGeophysX.inversion.joint_ert_srt",
    "JointERTSRTInversion",
    "JointERTSRTResult",
))
_EXPORTS.update(_exports(
    "PyHydroGeophysX.inversion.joint_fdem_tdem",
    "JointFDEMTDEMInversion",
    "JointFDEMTDEMResult",
))
_EXPORTS.update(_exports(
    "PyHydroGeophysX.inversion.joint_gravity_magnetics",
    "JointGravityMagneticsInversion",
    "GravityMagneticsJointResult",
))
_EXPORTS.update(_exports(
    "PyHydroGeophysX.visualization",
    "plot_model_section",
    "plot_timelapse_snapshots",
    "plot_difference_map",
    "plot_convergence",
    "plot_pseudosection_matrix",
    "plot_electrode_layout",
    "plot_topography",
    "plot_monitoring_timeseries",
    "plot_coverage",
    "create_timelapse_gif",
    "create_timelapse_mp4",
    "create_difference_gif",
    "export_mesh_to_vtk",
    "export_structured_vtk",
    "export_structured_vtk_multi",
    "export_timelapse_vtk",
    "export_timelapse_structured_vtk",
    "export_points_to_vtk",
))
_EXPORTS.update(_exports(
    "PyHydroGeophysX.data_processing.em1d",
    "load_sounding",
    "load_line_geometry",
))
_EXPORTS.update(_exports(
    "PyHydroGeophysX.inversion.em1d",
    "tdem_invert",
    "fdem_invert",
))
_EXPORTS.update(_exports(
    "PyHydroGeophysX.workflows.em1d",
    "invert_line",
    "estimate_data_scale",
    "calibrate_to_reference",
))
_EXPORTS.update(_exports(
    "PyHydroGeophysX.data_processing.gravmag",
    "regional_residual",
    "qc_products",
    "grid_data",
    "extract_profile",
))
_EXPORTS.update(_exports(
    "PyHydroGeophysX.forward.gravmag",
    "gravity_sphere",
    "gravity_prism",
    "magnetic_dipole",
    "forward_bodies",
))
_EXPORTS.update(_exports(
    "PyHydroGeophysX.inversion.gravmag",
    "invert_gravmag",
))
_EXPORTS["BackendUnavailable"] = (
    "PyHydroGeophysX._internal.optional_dependencies",
    "BackendUnavailable",
)

# These names historically degraded to ``None`` when their optional backend
# could not be imported.  Keep that 0.4 behavior while making access lazy.
_OPTIONAL_EXPORTS = {
    "MeshCreator",
    "create_mesh_from_layers",
    "ParflowSaturation",
    "ParflowPorosity",
    "ERTForwardModeling",
    "ertforward",
    "ertforward2",
    "ertforandjac",
    "ertforandjac2",
    "InversionResult",
    "TimeLapseInversionResult",
    "InversionBase",
    "ERTInversion",
    "TimeLapseERTInversion",
    "WindowedTimeLapseERTInversion",
    "AgentCoordinator",
    "ERTLoaderAgent",
    "ERTInversionAgent",
    "WaterContentAgent",
    "ReportAgent",
    "SeismicAgent",
    "GeophysicalInversionAgent",
    "FDEMForwardModeling",
    "FDEMSurveyConfig",
    "SRTInversion",
    "TimeLapseSRTInversion",
    "FDEMInversion",
    "FDEMInversionResult",
    "GeophysicalInversion",
    "StructuralConstraint",
    "PetrophysicalCoupling",
    "JointERTSRTInversion",
    "JointERTSRTResult",
    "JointFDEMTDEMInversion",
    "JointFDEMTDEMResult",
    "JointGravityMagneticsInversion",
    "GravityMagneticsJointResult",
}

_FEATURE_FLAGS = {"PYGIMLI_AVAILABLE", "SIMPEG_AVAILABLE"}


def _backend_available(name: str) -> bool:
    modules = ("pygimli",) if name == "PYGIMLI_AVAILABLE" else ("simpeg", "SimPEG")
    for module_name in modules:
        try:
            importlib.import_module(module_name)
            return True
        except ImportError:
            continue
    return False


def run_joint_inversion(*args, **kwargs):
    """Lazily load and run the backend selected by a joint request."""
    from PyHydroGeophysX.inversion.joint import run_joint_inversion as _run

    return _run(*args, **kwargs)


def __getattr__(name: str):
    if name in _FEATURE_FLAGS:
        value = _backend_available(name)
        globals()[name] = value
        return value
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    try:
        value = getattr(importlib.import_module(module_name), attribute)
    except ImportError:
        if name not in _OPTIONAL_EXPORTS:
            raise
        value = None
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))


__all__ = [
    "PYGIMLI_AVAILABLE",
    "SIMPEG_AVAILABLE",
    *_EXPORTS,
    "run_joint_inversion",
]
