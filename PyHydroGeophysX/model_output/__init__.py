"""
Module for processing model outputs from various hydrological models.
"""

from .base import HydroModelOutput
from .ats_output import ATSOutput, ATSSaturation, ATSPorosity, ATSWaterContent
from .pflotran_output import (
    PFLOTRANOutput, PFLOTRANSaturation, PFLOTRANPorosity, PFLOTRANWaterContent,
)
from .water_content import (
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
    'ATSOutput', 'ATSSaturation', 'ATSPorosity', 'ATSWaterContent',
    'PFLOTRANOutput', 'PFLOTRANSaturation', 'PFLOTRANPorosity', 'PFLOTRANWaterContent',
    'HydroModelOutput',
    'MODFLOWWaterContent',
    'MODFLOWPorosity',
    'binaryread'
]

if PARFLOW_AVAILABLE:
    __all__ += ['ParflowSaturation', 'ParflowPorosity']
