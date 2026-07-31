"""Lazy inversion-framework exports."""

from __future__ import annotations

import importlib
from typing import Dict, Tuple

from PyHydroGeophysX._internal.optional_dependencies import optional_import_error

_EXPORTS: Dict[str, Tuple[str, str]] = {
    "InversionBase": ("PyHydroGeophysX.inversion.base", "InversionBase"),
    "InversionResult": ("PyHydroGeophysX.inversion.base", "InversionResult"),
    "TimeLapseInversionResult": (
        "PyHydroGeophysX.inversion.base",
        "TimeLapseInversionResult",
    ),
    "ERTInversion": ("PyHydroGeophysX.inversion.ert_inversion", "ERTInversion"),
    "TDEMInversion": ("PyHydroGeophysX.inversion.tdem_inversion", "TDEMInversion"),
    "TDEMInversionResult": (
        "PyHydroGeophysX.inversion.tdem_inversion",
        "TDEMInversionResult",
    ),
    "run_tdem_inversion": (
        "PyHydroGeophysX.inversion.tdem_inversion",
        "run_tdem_inversion",
    ),
    "FDEMInversion": ("PyHydroGeophysX.inversion.fdem_inversion", "FDEMInversion"),
    "FDEMInversionResult": (
        "PyHydroGeophysX.inversion.fdem_inversion",
        "FDEMInversionResult",
    ),
    "TimeLapseERTInversion": (
        "PyHydroGeophysX.inversion.time_lapse",
        "TimeLapseERTInversion",
    ),
    "WindowedTimeLapseERTInversion": (
        "PyHydroGeophysX.inversion.windowed",
        "WindowedTimeLapseERTInversion",
    ),
    "SRTInversion": ("PyHydroGeophysX.inversion.srt_inversion", "SRTInversion"),
    "TimeLapseSRTInversion": (
        "PyHydroGeophysX.inversion.srt_time_lapse",
        "TimeLapseSRTInversion",
    ),
    "GeophysicalInversion": (
        "PyHydroGeophysX.inversion.multi_method",
        "GeophysicalInversion",
    ),
    "StructuralConstraint": (
        "PyHydroGeophysX.inversion.cross_constraints",
        "StructuralConstraint",
    ),
    "PetrophysicalCoupling": (
        "PyHydroGeophysX.inversion.cross_constraints",
        "PetrophysicalCoupling",
    ),
    "JointERTSRTInversion": (
        "PyHydroGeophysX.inversion.joint_ert_srt",
        "JointERTSRTInversion",
    ),
    "JointERTSRTResult": (
        "PyHydroGeophysX.inversion.joint_ert_srt",
        "JointERTSRTResult",
    ),
    "JointFDEMTDEMInversion": (
        "PyHydroGeophysX.inversion.joint_fdem_tdem",
        "JointFDEMTDEMInversion",
    ),
    "JointFDEMTDEMResult": (
        "PyHydroGeophysX.inversion.joint_fdem_tdem",
        "JointFDEMTDEMResult",
    ),
    "JointGravityMagneticsInversion": (
        "PyHydroGeophysX.inversion.joint_gravity_magnetics",
        "JointGravityMagneticsInversion",
    ),
    "GravityMagneticsJointResult": (
        "PyHydroGeophysX.inversion.joint_gravity_magnetics",
        "GravityMagneticsJointResult",
    ),
    "JointInversionRequest": (
        "PyHydroGeophysX.inversion.joint_api",
        "JointInversionRequest",
    ),
    "JointInversionResult": (
        "PyHydroGeophysX.inversion.joint_api",
        "JointInversionResult",
    ),
    "JointPairCapability": (
        "PyHydroGeophysX.inversion.joint_api",
        "JointPairCapability",
    ),
    "get_joint_capabilities": (
        "PyHydroGeophysX.inversion.joint_api",
        "get_joint_capabilities",
    ),
    "run_joint_inversion": (
        "PyHydroGeophysX.inversion.joint",
        "run_joint_inversion",
    ),
    "fdem_invert": ("PyHydroGeophysX.inversion.em1d", "fdem_invert"),
    "tdem_invert": ("PyHydroGeophysX.inversion.em1d", "tdem_invert"),
    "tdem_joint_invert": (
        "PyHydroGeophysX.inversion.em1d",
        "tdem_joint_invert",
    ),
    "InversionBackendUnavailable": (
        "PyHydroGeophysX.inversion.gravmag",
        "InversionBackendUnavailable",
    ),
    "invert_gravmag": (
        "PyHydroGeophysX.inversion.gravmag",
        "invert_gravmag",
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
