"""
Geophysical Forward Modeling Subpackage.

This subpackage provides classes and functions for performing forward geophysical
simulations, primarily focusing on Electrical Resistivity Tomography (ERT) and
Seismic Refraction Tomography (SRT). These tools allow users to generate synthetic
geophysical data based on given subsurface models.

Key functionalities include:
- ERT forward modeling: Calculating apparent resistivity data and Jacobians
  for various electrode configurations and subsurface resistivity distributions.
  Includes options for log-transformed parameters and data.
- SRT forward modeling: Calculating seismic travel times based on velocity models.

The `ERTForwardModeling` and `SeismicForwardModeling` classes offer structured
approaches to these simulations, while standalone functions provide more direct
access to core calculations, some of which might be helpers for inversion routines
or specialized use cases.
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