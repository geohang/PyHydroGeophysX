"""
Core utilities for geophysical modeling and inversion.
"""

# Track availability
MESH_UTILS_AVAILABLE = False
KRIGING_3D_AVAILABLE = False
MESH_3D_AVAILABLE = False

# Import mesh utilities (requires pygimli)
try:
    from PyHydroGeophysX.core.mesh_utils import (
        MeshCreator,
        create_mesh_from_layers,
        extract_velocity_interface,
        add_velocity_interface
    )
    MESH_UTILS_AVAILABLE = True
except ImportError:
    MeshCreator = None
    create_mesh_from_layers = None
    extract_velocity_interface = None
    add_velocity_interface = None

# Import interpolation utilities
try:
    from PyHydroGeophysX.core.interpolation import (
        ProfileInterpolator,
        interpolate_to_profile,
        setup_profile_coordinates,
        interpolate_structure_to_profile,
        prepare_2D_profile_data,
        interpolate_to_mesh,
        create_surface_lines
    )
except ImportError:
    ProfileInterpolator = None
    interpolate_to_profile = None
    setup_profile_coordinates = None
    interpolate_structure_to_profile = None
    prepare_2D_profile_data = None
    interpolate_to_mesh = None
    create_surface_lines = None

# Import 3D kriging utilities (optional, requires gstools and pyvista)
try:
    from PyHydroGeophysX.core.kriging_3d import (
        create_3d_structured_grid,
        estimate_directional_variograms,
        optimize_variogram_model,
        krige_seismic_velocity_3d,
        krige_from_2d_profiles
    )
    KRIGING_3D_AVAILABLE = True
except ImportError:
    # Define placeholder functions if dependencies not available
    create_3d_structured_grid = None
    estimate_directional_variograms = None
    optimize_variogram_model = None
    krige_seismic_velocity_3d = None
    krige_from_2d_profiles = None

# Import 3D mesh utilities
try:
    from PyHydroGeophysX.core.mesh_3d import (
        Mesh3DCreator,
        create_3d_ert_mesh_from_modflow,
        interpolate_modflow_to_3d_mesh,
        create_3d_ert_data_container,
        export_electrodes_to_csv
    )
    MESH_3D_AVAILABLE = True
except ImportError:
    Mesh3DCreator = None
    create_3d_ert_mesh_from_modflow = None
    interpolate_modflow_to_3d_mesh = None
    create_3d_ert_data_container = None
    export_electrodes_to_csv = None

__all__ = [
    # Availability flags
    'MESH_UTILS_AVAILABLE',
    'KRIGING_3D_AVAILABLE',
    'MESH_3D_AVAILABLE',
    
    # Mesh utilities
    'MeshCreator',
    'create_mesh_from_layers',
    'extract_velocity_interface',
    'add_velocity_interface',
    
    # 3D Mesh utilities
    'Mesh3DCreator',
    'create_3d_ert_mesh_from_modflow',
    'interpolate_modflow_to_3d_mesh',
    'create_3d_ert_data_container',
    'export_electrodes_to_csv',
    
    # Interpolation utilities
    'ProfileInterpolator',
    'interpolate_to_profile',
    'setup_profile_coordinates',
    'interpolate_structure_to_profile',
    'prepare_2D_profile_data',
    'interpolate_to_mesh',
    'create_surface_lines',
    
    # 3D kriging utilities
    'create_3d_structured_grid',
    'estimate_directional_variograms',
    'optimize_variogram_model',
    'krige_seismic_velocity_3d',
    'krige_from_2d_profiles',
]
