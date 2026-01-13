"""
Core utilities for geophysical modeling and inversion.
"""

# Import mesh utilities
from PyHydroGeophysX.core.mesh_utils import (
    MeshCreator,
    create_mesh_from_layers,
    extract_velocity_interface,
    add_velocity_interface
)

# Import interpolation utilities
from PyHydroGeophysX.core.interpolation import (
    ProfileInterpolator,
    interpolate_to_profile,
    setup_profile_coordinates,
    interpolate_structure_to_profile,
    prepare_2D_profile_data,
    interpolate_to_mesh,
    create_surface_lines
)

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
    KRIGING_3D_AVAILABLE = False

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
    MESH_3D_AVAILABLE = False

__all__ = [
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
    'MESH_3D_AVAILABLE',
    
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
    'KRIGING_3D_AVAILABLE'
]