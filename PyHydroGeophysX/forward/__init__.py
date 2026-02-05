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

# Import TDEM forward modeling utilities
from PyHydroGeophysX.forward.tdem_forward import (
    TDEMForwardModeling,
    TDEMSurveyConfig,
    create_tdem_survey,
    hydro_to_tdem
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
    'SeismicForwardModeling',
    
    # TDEM forward modeling
    'TDEMForwardModeling',
    'TDEMSurveyConfig',
    'create_tdem_survey',
    'hydro_to_tdem'
]

# FDEM forward modeling
from PyHydroGeophysX.forward.fdem_forward import FDEMForwardModeling, FDEMSurveyConfig

__all__ += [
    'FDEMForwardModeling',
    'FDEMSurveyConfig',
]
