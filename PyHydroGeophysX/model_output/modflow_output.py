"""Deprecated import path for canonical MODFLOW water-content readers."""

from PyHydroGeophysX._internal.deprecations import warn_legacy_path as _warn_legacy_path
from .water_content import MODFLOWPorosity, MODFLOWWaterContent, binaryread

_warn_legacy_path("model_output.modflow_output", "model_output.water_content")

__all__ = ["MODFLOWPorosity", "MODFLOWWaterContent", "binaryread"]
