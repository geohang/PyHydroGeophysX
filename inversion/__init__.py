"""
Inversion framework for geophysical applications.
"""

# Import inversion base classes
from PyHydroGeophysX.inversion.base import (
    InversionBase,
    InversionResult,
    TimeLapseInversionResult
)

# Import ERT inversion classes
from PyHydroGeophysX.inversion.ert_inversion import (
    ERTInversion
)

# Import time-lapse inversion classes
from PyHydroGeophysX.inversion.time_lapse import (
    TimeLapseERTInversion
)

# Import windowed inversion classes
from PyHydroGeophysX.inversion.windowed import (
    WindowedTimeLapseERTInversion
)

__all__ = [
    # Base classes
    'InversionBase',
    'InversionResult',
    'TimeLapseInversionResult',
    
    # ERT inversion
    'ERTInversion',
    
    # Time-lapse inversion
    'TimeLapseERTInversion',
    
    # Windowed inversion
    'WindowedTimeLapseERTInversion'
]