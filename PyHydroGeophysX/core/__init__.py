"""
Core utilities for geophysical modeling and inversion.

Names are loaded from their submodules on first use, as the package root does:
importing one core module - the 3-D mesh builder a workflow process runs, say -
no longer loads SciPy's signal and image tools, pandas and the kriging stack
for the others, which took a second before any work began. The availability
flags and the ``None`` stand-ins for an optional module that is not installed
are as they were.
"""

from __future__ import annotations

import importlib
from typing import Dict, Tuple

_Export = Tuple[str, str]


def _exports(module: str, *names: str) -> Dict[str, _Export]:
    return {name: (module, name) for name in names}


_EXPORTS: Dict[str, _Export] = {}
# Mesh utilities (requires pygimli)
_EXPORTS.update(_exports("mesh_utils", "MeshCreator", "create_mesh_from_layers",
                         "extract_velocity_interface", "add_velocity_interface"))
# Interpolation utilities
_EXPORTS.update(_exports("interpolation", "ProfileInterpolator", "interpolate_to_profile",
                         "setup_profile_coordinates", "interpolate_structure_to_profile",
                         "prepare_2D_profile_data", "interpolate_to_mesh",
                         "create_surface_lines"))
# Plan-view gridding (numpy/scipy only, no optional dependency)
_EXPORTS.update(_exports("plan_interpolation", "plan_grid", "write_plan_grid",
                         "ordinary_kriging", "inverse_distance", "empirical_variogram",
                         "fit_variogram", "auto_variogram", "variogram_function"))
# 3D kriging utilities (optional, requires gstools and pyvista)
_EXPORTS.update(_exports("kriging_3d", "create_3d_structured_grid",
                         "estimate_directional_variograms", "optimize_variogram_model",
                         "krige_seismic_velocity_3d", "krige_from_2d_profiles"))
# 3D mesh utilities
_EXPORTS.update(_exports("mesh_3d", "Mesh3DCreator", "create_3d_ert_mesh_from_modflow",
                         "interpolate_modflow_to_3d_mesh", "create_3d_ert_data_container",
                         "export_electrodes_to_csv"))

#: Modules whose names stand in as None when the module cannot be imported;
#: plan-view gridding needs nothing optional, so its failure is an error.
_OPTIONAL_MODULES = {"mesh_utils", "interpolation", "kriging_3d", "mesh_3d"}

_FEATURE_FLAGS = {
    "MESH_UTILS_AVAILABLE": "mesh_utils",
    "KRIGING_3D_AVAILABLE": "kriging_3d",
    "MESH_3D_AVAILABLE": "mesh_3d",
}

_SUBMODULES = {"_mesh_3d_builder", "e4d_mesh", "interpolation", "kriging_3d", "mesh_3d",
               "mesh_serialization", "mesh_utils", "plan_interpolation", "plt_utils",
               "section_geometry"}


def __getattr__(name: str):
    if name in _FEATURE_FLAGS:
        try:
            importlib.import_module(f"{__name__}.{_FEATURE_FLAGS[name]}")
            value = True
        except ImportError:
            value = False
        globals()[name] = value
        return value
    if name in _SUBMODULES:
        # ``core.mesh_3d`` without importing it first, as eager loading allowed.
        return importlib.import_module(f"{__name__}.{name}")
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attribute = target
    try:
        value = getattr(importlib.import_module(f"{__name__}.{module}"), attribute)
    except ImportError:
        if module not in _OPTIONAL_MODULES:
            raise
        # Expose unavailable optional entries as None, not callable stubs.
        value = None
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))


__all__ = [
    # Availability flags
    'MESH_UTILS_AVAILABLE',
    'KRIGING_3D_AVAILABLE',
    'MESH_3D_AVAILABLE',

    # Mesh utilities
    'MeshCreator',
    'create_mesh_from_layers',
    'extract_velocity_interface',
    'add_velocity_interface',

    # 3D Mesh utilities
    'Mesh3DCreator',
    'create_3d_ert_mesh_from_modflow',
    'interpolate_modflow_to_3d_mesh',
    'create_3d_ert_data_container',
    'export_electrodes_to_csv',

    # Interpolation utilities
    'ProfileInterpolator',
    'interpolate_to_profile',
    'setup_profile_coordinates',
    'interpolate_structure_to_profile',
    'prepare_2D_profile_data',
    'interpolate_to_mesh',
    'create_surface_lines',

    # Plan-view (map) gridding
    'plan_grid',
    'write_plan_grid',
    'ordinary_kriging',
    'inverse_distance',
    'empirical_variogram',
    'fit_variogram',
    'auto_variogram',
    'variogram_function',

    # 3D kriging utilities
    'create_3d_structured_grid',
    'estimate_directional_variograms',
    'optimize_variogram_model',
    'krige_seismic_velocity_3d',
    'krige_from_2d_profiles',
]
