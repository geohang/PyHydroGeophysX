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
    interpolate_to_mesh
)

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
    'interpolate_to_mesh'
]