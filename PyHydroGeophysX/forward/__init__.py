"""Lazy forward-modeling exports with independently optional backends."""

from __future__ import annotations

import importlib
from typing import Dict, Tuple

from PyHydroGeophysX._internal.optional_dependencies import optional_import_error

_EXPORTS: Dict[str, Tuple[str, str]] = {
    "ERTForwardModeling": (
        "PyHydroGeophysX.forward.ert_forward",
        "ERTForwardModeling",
    ),
    "ertforward": ("PyHydroGeophysX.forward.ert_forward", "ertforward"),
    "ertforward2": ("PyHydroGeophysX.forward.ert_forward", "ertforward2"),
    "ertforandjac": ("PyHydroGeophysX.forward.ert_forward", "ertforandjac"),
    "ertforandjac2": ("PyHydroGeophysX.forward.ert_forward", "ertforandjac2"),
    "SeismicForwardModeling": (
        "PyHydroGeophysX.forward.srt_forward",
        "SeismicForwardModeling",
    ),
    "TDEMForwardModeling": (
        "PyHydroGeophysX.forward.tdem_forward",
        "TDEMForwardModeling",
    ),
    "TDEMSurveyConfig": (
        "PyHydroGeophysX.forward.tdem_forward",
        "TDEMSurveyConfig",
    ),
    "create_tdem_survey": (
        "PyHydroGeophysX.forward.tdem_forward",
        "create_tdem_survey",
    ),
    "hydro_to_tdem": ("PyHydroGeophysX.forward.tdem_forward", "hydro_to_tdem"),
    "simulate_tdem_sounding_from_hydro": (
        "PyHydroGeophysX.forward.tdem_forward",
        "simulate_tdem_sounding_from_hydro",
    ),
    "FDEMForwardModeling": (
        "PyHydroGeophysX.forward.fdem_forward",
        "FDEMForwardModeling",
    ),
    "FDEMSurveyConfig": (
        "PyHydroGeophysX.forward.fdem_forward",
        "FDEMSurveyConfig",
    ),
    "gravity_sphere": (
        "PyHydroGeophysX.forward.gravmag",
        "gravity_sphere",
    ),
    "gravity_prism": ("PyHydroGeophysX.forward.gravmag", "gravity_prism"),
    "magnetic_dipole": (
        "PyHydroGeophysX.forward.gravmag",
        "magnetic_dipole",
    ),
    "forward_bodies": ("PyHydroGeophysX.forward.gravmag", "forward_bodies"),
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
