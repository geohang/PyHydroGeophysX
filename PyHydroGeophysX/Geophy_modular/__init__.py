"""Cross-modal geophysics-to-hydrology and structure integration APIs."""

from __future__ import annotations

import importlib
from typing import Dict, Tuple

from PyHydroGeophysX._internal.optional_dependencies import optional_import_error

_EXPORTS: Dict[str, Tuple[str, str]] = {
    "extract_velocity_structure": (
        "PyHydroGeophysX.Geophy_modular.seismic_processor",
        "extract_velocity_structure",
    ),
    "process_seismic_tomography": (
        "PyHydroGeophysX.Geophy_modular.seismic_processor",
        "process_seismic_tomography",
    ),
    "seismic_velocity_classifier": (
        "PyHydroGeophysX.Geophy_modular.seismic_processor",
        "seismic_velocity_classifier",
    ),
    "ERTtoWC": ("PyHydroGeophysX.Geophy_modular.ERT_to_WC", "ERTtoWC"),
    "plot_time_series": (
        "PyHydroGeophysX.Geophy_modular.ERT_to_WC",
        "plot_time_series",
    ),
    "run_ert_to_wc": (
        "PyHydroGeophysX.Geophy_modular.ERT_to_WC",
        "run_ert_to_wc",
    ),
}

__all__ = list(_EXPORTS)

#: Removed from ``structure_integration`` when it became the 3D-model pipeline.
#: Still exported, they raised a bare AttributeError on first use - and the ERT
#: inversion agent, catching it, ran a requested structure-constrained inversion
#: on an unconstrained mesh. The replacement is named instead.
_REMOVED = {
    name: "PyHydroGeophysX.core.mesh_utils.add_velocity_interface"
    for name in ("create_ert_mesh_with_structure", "integrate_velocity_interface",
                 "create_joint_inversion_mesh")
}


def __getattr__(name: str):
    target = _EXPORTS.get(name)
    if target is None:
        if name in _REMOVED:
            raise AttributeError(f"{name} was removed; build the interface mesh "
                                 f"with {_REMOVED[name]}")
        raise AttributeError(name)
    module_name, attribute = target
    try:
        value = getattr(importlib.import_module(module_name), attribute)
    except ImportError as exc:
        raise optional_import_error(name, exc) from exc
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
