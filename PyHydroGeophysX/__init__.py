"""
WatershedGeo - A comprehensive package for geophysical modeling and inversion in watershed monitoring.

This package integrates MODFLOW hydrological model outputs with geophysical forward modeling
and inversion, specializing in electrical resistivity tomography (ERT) and seismic velocity modeling.
"""

__version__ = '0.1.0'

# Track available features
PYGIMLI_AVAILABLE = False
SIMPEG_AVAILABLE = False

# Core utilities - interpolation (no pygimli dependency)
try:
    from PyHydroGeophysX.core.interpolation import (
        ProfileInterpolator, 
        interpolate_to_profile,
        setup_profile_coordinates,
        interpolate_structure_to_profile,
        prepare_2D_profile_data,
        interpolate_to_mesh
    )
except ImportError:
    ProfileInterpolator = None
    interpolate_to_profile = None
    setup_profile_coordinates = None
    interpolate_structure_to_profile = None
    prepare_2D_profile_data = None
    interpolate_to_mesh = None

# Core utilities - mesh (requires pygimli)
try:
    from PyHydroGeophysX.core.mesh_utils import (
        MeshCreator,
        create_mesh_from_layers
    )
    PYGIMLI_AVAILABLE = True
except ImportError:
    MeshCreator = None
    create_mesh_from_layers = None

# Import from model_output module
try:
    from PyHydroGeophysX.model_output.modflow_output import MODFLOWWaterContent, MODFLOWPorosity
    from PyHydroGeophysX.model_output.base import HydroModelOutput
except ImportError:
    MODFLOWWaterContent = None
    MODFLOWPorosity = None
    HydroModelOutput = None

# Try to import ParFlow classes if available
try:
    from PyHydroGeophysX.model_output.parflow_output import ParflowSaturation, ParflowPorosity
except ImportError:
    pass

# Forward modeling (requires pygimli)
try:
    from PyHydroGeophysX.forward.ert_forward import (
        ERTForwardModeling,
        ertforward,
        ertforward2,
        ertforandjac,
        ertforandjac2
    )
except ImportError:
    ERTForwardModeling = None
    ertforward = None
    ertforward2 = None
    ertforandjac = None
    ertforandjac2 = None

# Inversion utilities (requires pygimli)
try:
    from PyHydroGeophysX.inversion.base import (
        InversionResult,
        TimeLapseInversionResult,
        InversionBase
    )

    from PyHydroGeophysX.inversion.ert_inversion import (
        ERTInversion
    )

    from PyHydroGeophysX.inversion.time_lapse import (
        TimeLapseERTInversion
    )

    from PyHydroGeophysX.inversion.windowed import (
        WindowedTimeLapseERTInversion
    )
except ImportError:
    InversionResult = None
    TimeLapseInversionResult = None
    InversionBase = None
    ERTInversion = None
    TimeLapseERTInversion = None
    WindowedTimeLapseERTInversion = None

# Linear solvers
try:
    from PyHydroGeophysX.solvers.linear_solvers import (
        generalized_solver,
        LinearSolver,
        CGLSSolver,
        LSQRSolver,
        RRLSQRSolver,
        RRLSSolver,
        direct_solver,
        TikhonvRegularization,
        IterativeRefinement,
        get_optimal_solver
    )
except ImportError:
    generalized_solver = None
    LinearSolver = None
    CGLSSolver = None
    LSQRSolver = None
    RRLSQRSolver = None
    RRLSSolver = None
    direct_solver = None
    TikhonvRegularization = None
    IterativeRefinement = None
    get_optimal_solver = None

# Petrophysics - Resistivity models
try:
    from PyHydroGeophysX.petrophysics.resistivity_models import (
        water_content_to_resistivity,
        resistivity_to_water_content,
        resistivity_to_saturation,
    )
except ImportError:
    water_content_to_resistivity = None
    resistivity_to_water_content = None
    resistivity_to_saturation = None

# Petrophysics - Velocity models
try:
    from PyHydroGeophysX.petrophysics.velocity_models import (
        BaseVelocityModel,
        VRHModel,
        BrieModel,
        DEMModel,
        HertzMindlinModel,
        VRH_model,
        satK,
        velDEM,
        vel_porous
    )
except ImportError:
    BaseVelocityModel = None
    VRHModel = None
    BrieModel = None
    DEMModel = None
    HertzMindlinModel = None
    VRH_model = None
    satK = None
    velDEM = None
    vel_porous = None

# Define what gets imported with 'from watershed_geophysics import *'
__all__ = [
    # Feature flags
    'PYGIMLI_AVAILABLE',
    'SIMPEG_AVAILABLE',
    
    # Core - Interpolation
    'ProfileInterpolator', 
    'interpolate_to_profile',
    'setup_profile_coordinates',
    'interpolate_structure_to_profile',
    'prepare_2D_profile_data',
    'interpolate_to_mesh',
    
    # Core - Mesh
    'MeshCreator',
    'create_mesh_from_layers',
    
    # MODFLOW
    'MODFLOWWaterContent',
    
    # Forward modeling
    'ERTForwardModeling',
    'ertforward',
    'ertforward2',
    'ertforandjac',
    'ertforandjac2',
    
    # Inversion base
    'InversionResult',
    'TimeLapseInversionResult',
    'InversionBase',
    
    # ERT inversion
    'ERTInversion',
    
    # Time-lapse inversion
    'TimeLapseERTInversion',
    
    # Windowed inversion
    'WindowedTimeLapseERTInversion',
    
    # Linear solvers
    'generalized_solver',
    'LinearSolver',
    'CGLSSolver',
    'LSQRSolver',
    'RRLSQRSolver',
    'RRLSSolver',
    'direct_solver',
    'TikhonvRegularization',
    'IterativeRefinement',
    'get_optimal_solver',
    
    # Resistivity models
    'water_content_to_resistivity',
    'resistivity_to_water_content',
    'resistivity_to_saturation',
    
    # Velocity models
    'BaseVelocityModel',
    'VRHModel',
    'BrieModel',
    'DEMModel',
    'HertzMindlinModel',
    'VRH_model',
    'satK',
    'velDEM',
    'vel_porous'
]

# Data processing utilities
try:
    from .data_processing import (
        load_ert_resipy,
        qc_and_visualize,
        export_for_inversion,
        LocalRef,
    )
    __all__ += [
        "load_ert_resipy",
        "qc_and_visualize",
        "export_for_inversion",
        "LocalRef",
    ]
except ImportError:
    pass

# Multi-agent system (optional - requires openai package)
try:
    from .agents import (
        AgentCoordinator,
        ERTLoaderAgent,
        ERTInversionAgent,
        WaterContentAgent,
        ReportAgent,
        SeismicAgent
    )
    __all__ += [
        "AgentCoordinator",
        "ERTLoaderAgent",
        "ERTInversionAgent",
        "WaterContentAgent",
        "ReportAgent",
        "SeismicAgent"
    ]
except ImportError:
    # Agents module requires openai package
    pass

# Additional forward modeling (FDEM)
try:
    from PyHydroGeophysX.forward.fdem_forward import (
        FDEMForwardModeling,
        FDEMSurveyConfig,
    )
    __all__ += [
        "FDEMForwardModeling",
        "FDEMSurveyConfig",
    ]
except ImportError:
    FDEMForwardModeling = None
    FDEMSurveyConfig = None

# Additional inversion modules
try:
    from PyHydroGeophysX.inversion.srt_inversion import SRTInversion
    from PyHydroGeophysX.inversion.srt_time_lapse import TimeLapseSRTInversion
    from PyHydroGeophysX.inversion.fdem_inversion import (
        FDEMInversion,
        FDEMInversionResult,
    )
    from PyHydroGeophysX.inversion.multi_method import GeophysicalInversion
    from PyHydroGeophysX.inversion.cross_constraints import (
        StructuralConstraint,
        PetrophysicalCoupling,
    )
    __all__ += [
        "SRTInversion",
        "TimeLapseSRTInversion",
        "FDEMInversion",
        "FDEMInversionResult",
        "GeophysicalInversion",
        "StructuralConstraint",
        "PetrophysicalCoupling",
    ]
except ImportError:
    SRTInversion = None
    TimeLapseSRTInversion = None
    FDEMInversion = None
    FDEMInversionResult = None
    GeophysicalInversion = None
    StructuralConstraint = None
    PetrophysicalCoupling = None

# Additional multi-method agent
try:
    from .agents.geophysical_inversion_agent import GeophysicalInversionAgent
    __all__ += [
        "GeophysicalInversionAgent",
    ]
except ImportError:
    GeophysicalInversionAgent = None

# Additional joint inversion module
try:
    from PyHydroGeophysX.inversion.joint_ert_srt import (
        JointERTSRTInversion,
        JointERTSRTResult,
    )
    __all__ += [
        "JointERTSRTInversion",
        "JointERTSRTResult",
    ]
except ImportError:
    JointERTSRTInversion = None
    JointERTSRTResult = None
