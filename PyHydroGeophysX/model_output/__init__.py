"""
Module for processing model outputs from various hydrological models.
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
    'binaryread'
]

if PARFLOW_AVAILABLE:
    __all__ += ['ParflowSaturation', 'ParflowPorosity']