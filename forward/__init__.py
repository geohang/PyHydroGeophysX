"""
Forward modeling utilities for geophysical simulations.
"""

# Import ERT forward modeling utilities
from PyHydroGeophysX.forward.ert_forward import (
    ERTForwardModeling,
    ertforward,
    ertforward2,
    ertforandjac,
    ertforandjac2
)

# Import SRT forward modeling utilities
from PyHydroGeophysX.forward.srt_forward import (
    SeismicForwardModeling
)

# Define the public API for this module
__all__ = [
    # ERT forward modeling
    'ERTForwardModeling',
    'ertforward',
    'ertforward2',
    'ertforandjac',
    'ertforandjac2',
    
    # SRT forward modeling
    'SeismicForwardModeling'
]