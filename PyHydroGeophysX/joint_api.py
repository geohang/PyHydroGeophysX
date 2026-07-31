"""Compatibility shim for :mod:`PyHydroGeophysX.inversion.joint_api`."""

from PyHydroGeophysX._internal.deprecations import warn_legacy_path
from PyHydroGeophysX.inversion.joint_api import (
    METHODS,
    JointInversionRequest,
    JointInversionResult,
    JointMethodAdapter,
    JointPairCapability,
    get_joint_capabilities,
    get_joint_capability,
    normalize_joint_pair,
    pair_joint_soundings,
    split_joint_soundings,
    validate_profile_interface,
)

warn_legacy_path(
    "PyHydroGeophysX.joint_api",
    "PyHydroGeophysX.inversion.joint_api",
)

__all__ = [
    "JointInversionRequest",
    "JointInversionResult",
    "JointMethodAdapter",
    "JointPairCapability",
    "METHODS",
    "get_joint_capabilities",
    "get_joint_capability",
    "normalize_joint_pair",
    "pair_joint_soundings",
    "split_joint_soundings",
    "validate_profile_interface",
]
