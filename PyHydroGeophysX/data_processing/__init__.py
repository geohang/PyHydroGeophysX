"""Lazy data-processing exports.

Importing a lightweight reader must not initialize ERT plotting code or probe
optional geophysics engines.  Public objects are therefore loaded only when
they are requested.
"""

from __future__ import annotations

import importlib
from typing import Dict, Tuple

from PyHydroGeophysX._internal.optional_dependencies import optional_import_error

_EXPORTS: Dict[str, Tuple[str, str]] = {
    "calculate_reciprocal_errors": (
        "PyHydroGeophysX.data_processing.ert_data_agent",
        "calculate_reciprocal_errors",
    ),
    "export_ert_dataset": (
        "PyHydroGeophysX.data_processing.ert_data_agent",
        "export_ert_dataset",
    ),
    "export_for_inversion": (
        "PyHydroGeophysX.data_processing.ert_data_agent",
        "export_for_inversion",
    ),
    "LocalRef": ("PyHydroGeophysX.data_processing.ert_data_agent", "LocalRef"),
    "load_ert_resipy": (
        "PyHydroGeophysX.data_processing.ert_data_agent",
        "load_ert_resipy",
    ),
    "qc_and_visualize": (
        "PyHydroGeophysX.data_processing.ert_data_agent",
        "qc_and_visualize",
    ),
    "read_seg2_seismic": (
        "PyHydroGeophysX.data_processing.field_formats",
        "read_seg2_seismic",
    ),
    "read_tem_fast": (
        "PyHydroGeophysX.data_processing.field_formats",
        "read_tem_fast",
    ),
    "export_to_vtk": (
        "PyHydroGeophysX.data_processing.field_formats",
        "export_to_vtk",
    ),
    "export_results_to_csv": (
        "PyHydroGeophysX.data_processing.field_formats",
        "export_results_to_csv",
    ),
    "FirstBreakPick": (
        "PyHydroGeophysX.data_processing.seismic",
        "FirstBreakPick",
    ),
    "SegyMetadata": ("PyHydroGeophysX.data_processing.seismic", "SegyMetadata"),
    "SeismicDataset": (
        "PyHydroGeophysX.data_processing.seismic",
        "SeismicDataset",
    ),
    "SeismicShotGather": (
        "PyHydroGeophysX.data_processing.seismic",
        "SeismicShotGather",
    ),
    "SeismicTraceHeader": (
        "PyHydroGeophysX.data_processing.seismic",
        "SeismicTraceHeader",
    ),
    "TravelTimeModel": (
        "PyHydroGeophysX.data_processing.seismic",
        "TravelTimeModel",
    ),
    "TravelTimeModelSegment": (
        "PyHydroGeophysX.data_processing.seismic",
        "TravelTimeModelSegment",
    ),
    "apply_agc": ("PyHydroGeophysX.data_processing.seismic", "apply_agc"),
    "bandpass_filter": (
        "PyHydroGeophysX.data_processing.seismic",
        "bandpass_filter",
    ),
    "export_first_breaks": (
        "PyHydroGeophysX.data_processing.seismic",
        "export_first_breaks",
    ),
    "export_traveltime_container": (
        "PyHydroGeophysX.data_processing.seismic",
        "export_traveltime_container",
    ),
    "fit_velocity_traveltime_model": (
        "PyHydroGeophysX.data_processing.seismic",
        "fit_velocity_traveltime_model",
    ),
    "first_breaks_to_traveltime": (
        "PyHydroGeophysX.data_processing.seismic",
        "first_breaks_to_traveltime",
    ),
    "normalize_traces": (
        "PyHydroGeophysX.data_processing.seismic",
        "normalize_traces",
    ),
    "pick_first_breaks": (
        "PyHydroGeophysX.data_processing.seismic",
        "pick_first_breaks",
    ),
    "predict_velocity_traveltimes": (
        "PyHydroGeophysX.data_processing.seismic",
        "predict_velocity_traveltimes",
    ),
    "read_geometrics_dat": (
        "PyHydroGeophysX.data_processing.seismic",
        "read_geometrics_dat",
    ),
    "read_segy": ("PyHydroGeophysX.data_processing.seismic", "read_segy"),
    "tukey_taper": ("PyHydroGeophysX.data_processing.seismic", "tukey_taper"),
    "load_joint_observations": (
        "PyHydroGeophysX.data_processing.joint_io",
        "load_joint_observations",
    ),
    "save_joint_observations": (
        "PyHydroGeophysX.data_processing.joint_io",
        "save_joint_observations",
    ),
    "save_edited_ert_container": (
        "PyHydroGeophysX.data_processing.ert_io",
        "save_edited_ert_container",
    ),
    "ensure_dir": ("PyHydroGeophysX.data_processing.table_io", "ensure_dir"),
    "load_2d_array": (
        "PyHydroGeophysX.data_processing.table_io",
        "load_2d_array",
    ),
    "load_xyz_table": (
        "PyHydroGeophysX.data_processing.table_io",
        "load_xyz_table",
    ),
    "write_csv": ("PyHydroGeophysX.data_processing.table_io", "write_csv"),
    "write_json": ("PyHydroGeophysX.data_processing.table_io", "write_json"),
    "read_json": ("PyHydroGeophysX.data_processing.table_io", "read_json"),
    "TEMCOMPANY_MOMENTS": (
        "PyHydroGeophysX.data_processing.em1d",
        "TEMCOMPANY_MOMENTS",
    ),
    "is_temcompany_source": (
        "PyHydroGeophysX.data_processing.em1d",
        "is_temcompany_source",
    ),
    "is_ttem_source": (
        "PyHydroGeophysX.data_processing.ttem",
        "is_ttem_source",
    ),
    "load_temcompany_sounding": (
        "PyHydroGeophysX.data_processing.em1d",
        "load_temcompany_sounding",
    ),
    "load_ttem_sounding": (
        "PyHydroGeophysX.data_processing.ttem",
        "load_ttem_sounding",
    ),
    "load_sounding": ("PyHydroGeophysX.data_processing.em1d", "load_sounding"),
    "load_line_geometry": (
        "PyHydroGeophysX.data_processing.em1d",
        "load_line_geometry",
    ),
    "regional_residual": (
        "PyHydroGeophysX.data_processing.gravmag",
        "regional_residual",
    ),
    "spatially_balanced_indices": (
        "PyHydroGeophysX.data_processing.gravmag",
        "spatially_balanced_indices",
    ),
    "qc_products": ("PyHydroGeophysX.data_processing.gravmag", "qc_products"),
    "grid_data": ("PyHydroGeophysX.data_processing.gravmag", "grid_data"),
    "extract_profile": (
        "PyHydroGeophysX.data_processing.gravmag",
        "extract_profile",
    ),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    try:
        value = getattr(importlib.import_module(module_name), attribute)
    except ImportError as exc:
        raise optional_import_error(name, exc) from exc
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
