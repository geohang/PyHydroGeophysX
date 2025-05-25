"""
Model Output Processing Subpackage.

This subpackage provides classes and functions for reading, processing, and
interpreting outputs from various hydrological models. It aims to offer a
consistent interface for accessing common hydrological variables like water content,
porosity, and saturation, regardless of the underlying model format.

Currently supported models (or functionalities) include:
- MODFLOW: For water content and porosity.
- Parflow: For saturation and porosity (if available).

The base class `HydroModelOutput` defines a common structure, and specific
model implementations inherit from it.
"""

from .base import HydroModelOutput
from .modflow_output import (
    MODFLOWWaterContent,
    MODFLOWPorosity,
    binaryread
)

# Import if implemented
try:
    from .parflow_output import (
        ParflowSaturation,
        ParflowPorosity
    )
    PARFLOW_AVAILABLE = True
except ImportError:
    PARFLOW_AVAILABLE = False

__all__ = [
    'HydroModelOutput',
    'MODFLOWWaterContent',
    'MODFLOWPorosity',
    'binaryread' # Utility function from modflow_output
]

if PARFLOW_AVAILABLE:
    __all__ += ['ParflowSaturation', 'ParflowPorosity']
# Note: water_content.py is expected to be removed, its contents merged into modflow_output.py or deprecated.