"""
Seismic data processing module for structure identification.
"""
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pygimli as pg
from pygimli.physics import traveltime as tt

from PyHydroGeophysX.core.mesh_utils import _velocity_interface


# ---------------------------------------------------------------------------
# process seismic tomography
# ---------------------------------------------------------------------------
def process_seismic_tomography(
    ttData: Any,
    mesh: Any = None,
    **kwargs: Any,
) -> Any:
    """
    Process seismic tomography data and perform inversion.
    
    Args:
        ttData: Travel time data container
        mesh: Mesh for inversion (optional, created if None)
        **kwargs: Additional parameters including:
            - lam: Regularization parameter (default: 50)
            - zWeight: Vertical regularization weight (default: 0.2)
            - vTop: Top velocity constraint (default: 500)
            - vBottom: Bottom velocity constraint (default: 5000)
            - quality: Mesh quality if creating new mesh (default: 31)
            - paraDepth: Maximum depth for parametric domain (default: 30)
            - verbose: Verbosity level (default: 1)
            
    Returns:
        TravelTimeManager object with inversion results
    """
    # Set default parameters
    params = {
        'lam': 50,
        'zWeight': 0.2,
        'vTop': 500,
        'vBottom': 5000,
        'quality': 31,
        'paraDepth': 30.0,
        'verbose': 1,
        'limits': [400., 10000.]
    }
    
    # Update with user-provided parameters
    params.update(kwargs)
    
    # Create travel time manager
    TT = pg.physics.traveltime.TravelTimeManager()
    
    # Set or create mesh
    if mesh is not None:
        TT.setMesh(mesh)
    else:
        # Create mesh from data if not provided
        # For a more sophisticated mesh creation, we could use createParaMesh
        pass
    
    # Run inversion
    TT.invert(ttData, 
              lam=params['lam'],
              zWeight=params['zWeight'], 
              vTop=params['vTop'], 
              vBottom=params['vBottom'],
              verbose=params['verbose'], 
              limits=params['limits'])
    
    return TT


# ---------------------------------------------------------------------------
# seismic velocity classifier
# ---------------------------------------------------------------------------
def seismic_velocity_classifier(
    velocity_data: Any,
    mesh: Any,
    threshold: Any = 1200,
) -> Any:
    """
    Classify mesh cells based on velocity threshold.
    
    Args:
        velocity_data: Velocity values for each cell
        mesh: PyGIMLi mesh
        threshold: Velocity threshold for classification (default: 1200)
        
    Returns:
        Array of cell markers (1: below threshold, 2: above threshold)
    """
    # Initialize classification array
    thresholded = np.ones_like(velocity_data, dtype=int)
    
    # Get cell centers
    cell_centers = mesh.cellCenters()
    x_coords = cell_centers[:,0]  # X-coordinates of cell centers
    z_coords = cell_centers[:,1]  # Z-coordinates (depth) of cell centers
    
    # Get unique x-coordinates (horizontal distances)
    unique_x = np.unique(x_coords)
    
    # For each vertical column (each unique x-coordinate)
    for x in unique_x:
        # Get indices of cells in this column, shallowest first. z is an
        # elevation (negative downward), so an ascending sort starts at the
        # deepest cell: once that one passed the threshold, every cell above it
        # became 2 as well, an 800 m/s near-surface cell included.
        column_indices = np.where(x_coords == x)[0]
        column_indices = column_indices[np.argsort(-z_coords[column_indices], kind="stable")]
        
        # Check if any cell in this column exceeds the threshold
        threshold_crossed = False
        
        # Process cells from top to bottom
        for idx in column_indices:
            if velocity_data[idx] >= threshold or threshold_crossed:
                thresholded[idx] = 2
                threshold_crossed = True
    
    return thresholded


# ---------------------------------------------------------------------------
# extract velocity structure
# ---------------------------------------------------------------------------
def extract_velocity_structure(
    mesh: Any,
    velocity_data: Any,
    threshold: Any = 1200,
    interval: Any = 4.0,
) -> Any:
    """
    Extract structure interface from velocity model at the specified threshold.
    
    Args:
        mesh: PyGIMLi mesh
        velocity_data: Velocity values for each cell
        threshold: Velocity threshold defining interface (default: 1200)
        interval: Horizontal sampling interval (default: 4.0)
        
    Returns:
        x_coords: Horizontal coordinates of interface points
        z_coords: Vertical coordinates of interface points
        interface_data: Dictionary with interface information
    """
    # The same extraction as core.mesh_utils.extract_velocity_interface, over
    # the mesh's whole x-range, plus the points the curve was fitted to.
    x_dense, z_dense, points = _velocity_interface(mesh, velocity_data, threshold, interval)

    # Prepare interface data dictionary
    interface_data = {
        'threshold': threshold,
        'raw_x': points['raw_x'],
        'raw_z': points['raw_z'],
        'smooth_x': x_dense,
        'smooth_z': z_dense,
        'min_x': points['min_x'],
        'max_x': points['max_x']
    }
    
    return x_dense, z_dense, interface_data


# ---------------------------------------------------------------------------
# save velocity structure
# ---------------------------------------------------------------------------
def save_velocity_structure(
    filename: Any,
    x_coords: Any,
    z_coords: Any,
    interface_data: Any = None,
) -> None:
    """
    Save velocity structure data to file.
    
    Args:
        filename: Output filename
        x_coords: X coordinates of interface
        z_coords: Z coordinates of interface
        interface_data: Additional data to save (optional)
    """
    # Create dictionary with data
    save_data = {
        'x_coords': x_coords,
        'z_coords': z_coords
    }
    
    # Add additional data if provided
    if interface_data is not None:
        save_data.update(interface_data)
    
    # Save as numpy file
    np.savez(filename, **save_data)
    
    # Also save as CSV for easier viewing
    csv_filename = filename.replace('.npz', '.csv')
    with open(csv_filename, 'w', encoding='utf-8') as f:
        f.write('x,z\n')
        for x, z in zip(x_coords, z_coords):
            f.write(f"{x},{z}\n")
