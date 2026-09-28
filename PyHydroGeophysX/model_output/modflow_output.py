"""Deprecated import path for canonical MODFLOW water-content readers."""

from PyHydroGeophysX._internal.deprecations import warn_legacy_path as _warn_legacy_path
from .base import HydroModelOutput  # imported here in 0.3.0, so still reachable here
from .water_content import MODFLOWPorosity, MODFLOWWaterContent, binaryread

_warn_legacy_path("model_output.modflow_output", "model_output.water_content")

__all__ = ["HydroModelOutput", "MODFLOWPorosity", "MODFLOWWaterContent", "binaryread"]
