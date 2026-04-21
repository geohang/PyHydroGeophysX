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

# Import TDEM inversion classes (simpeg is optional)
try:
    from PyHydroGeophysX.inversion.tdem_inversion import (
        TDEMInversion,
        TDEMInversionResult,
        run_tdem_inversion
    )
except ImportError:
    TDEMInversion = None
    TDEMInversionResult = None
    run_tdem_inversion = None

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
    
    # TDEM inversion
    'TDEMInversion',
    'TDEMInversionResult',
    'run_tdem_inversion',
    
    # Time-lapse inversion
    'TimeLapseERTInversion',
    
    # Windowed inversion
    'WindowedTimeLapseERTInversion'
]

# SRT inversion
from PyHydroGeophysX.inversion.srt_inversion import SRTInversion

# SRT time-lapse inversion
from PyHydroGeophysX.inversion.srt_time_lapse import TimeLapseSRTInversion

# FDEM inversion
from PyHydroGeophysX.inversion.fdem_inversion import FDEMInversion, FDEMInversionResult

# Multi-method interface
from PyHydroGeophysX.inversion.multi_method import GeophysicalInversion

# Cross-method constraints
from PyHydroGeophysX.inversion.cross_constraints import StructuralConstraint, PetrophysicalCoupling

__all__ += [
    'SRTInversion',
    'TimeLapseSRTInversion',
    'FDEMInversion',
    'FDEMInversionResult',
    'GeophysicalInversion',
    'StructuralConstraint',
    'PetrophysicalCoupling',
]

# Joint ERT-SRT inversion
from PyHydroGeophysX.inversion.joint_ert_srt import JointERTSRTInversion, JointERTSRTResult

__all__ += [
    'JointERTSRTInversion',
    'JointERTSRTResult',
]
