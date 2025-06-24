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

__all__ = [
    # Mesh utilities
    'MeshCreator',
    'create_mesh_from_layers',
    'extract_velocity_interface',
    'add_velocity_interface',
    
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