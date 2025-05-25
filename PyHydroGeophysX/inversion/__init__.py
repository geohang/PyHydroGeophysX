"""
Geophysical Inversion Framework Subpackage.

This subpackage provides a comprehensive framework for performing geophysical
inversions, with a focus on Electrical Resistivity Tomography (ERT). It includes
base classes for defining inversion processes and results, specific implementations
for standard ERT inversion, and advanced methods for time-lapse and windowed
time-lapse ERT inversions.

Key Components:
- `InversionBase`: An abstract base class defining the core interface for inversion algorithms.
- `InversionResult`, `TimeLapseInversionResult`: Classes for storing and visualizing
  inversion outcomes.
- `ERTInversion`: For standard 2D/3D ERT inversion.
- `TimeLapseERTInversion`: For time-lapse ERT, incorporating temporal regularization
  to link inversions across different timesteps.
- `WindowedTimeLapseERTInversion`: Extends time-lapse ERT by processing data in
  moving windows, suitable for large time-series datasets and parallel processing.

The framework is designed to be extensible for other geophysical methods in the future.
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