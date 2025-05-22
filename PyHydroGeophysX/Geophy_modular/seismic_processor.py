"""
Seismic data processing module for structure identification.
"""
import numpy as np
import pygimli as pg
from pygimli.physics import traveltime as tt
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from typing import Tuple, List, Optional, Union, Dict, Any


def process_seismic_tomography(ttData, mesh=None, **kwargs):
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
        'limits': [100., 6000.]
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


def seismic_velocity_classifier(velocity_data, mesh, threshold=1200):
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
        # Get indices of cells in this column, sorted by depth (z-coordinate)
        column_indices = np.where(x_coords == x)[0]
        column_indices = column_indices[np.argsort(z_coords[column_indices])]
        
        # Check if any cell in this column exceeds the threshold
        threshold_crossed = False
        
        # Process cells from top to bottom
        for idx in column_indices:
            if velocity_data[idx] >= threshold or threshold_crossed:
                thresholded[idx] = 2
                threshold_crossed = True
    
    return thresholded


def extract_velocity_structure(mesh, velocity_data, threshold=1200, interval=4.0):
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
    # Get cell centers
    cell_centers = mesh.cellCenters()
    x_coords = cell_centers[:,0]
    z_coords = cell_centers[:,1]
    
    # Get x-range for complete boundary
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    
    # Create bins across the entire x-range
    x_bins = np.arange(x_min, x_max + interval, interval)
    
    # Arrays to store interface points
    interface_x = []
    interface_z = []
    
    # For each bin, find the velocity interface
    for i in range(len(x_bins)-1):
        # Get all cells in this x-range
        bin_indices = np.where((x_coords >= x_bins[i]) & (x_coords < x_bins[i+1]))[0]
        
        if len(bin_indices) > 0:
            # Get velocity values and depths for this bin
            bin_velocities = velocity_data[bin_indices]
            bin_depths = z_coords[bin_indices]
            
            # Sort by depth
            sort_indices = np.argsort(bin_depths)
            bin_velocities = bin_velocities[sort_indices]
            bin_depths = bin_depths[sort_indices]
            
            # Find where velocity crosses the threshold
            for j in range(1, len(bin_velocities)):
                if (bin_velocities[j-1] < threshold and bin_velocities[j] >= threshold) or \
                   (bin_velocities[j-1] >= threshold and bin_velocities[j] < threshold):
                    # Linear interpolation for exact interface depth
                    v1 = bin_velocities[j-1]
                    v2 = bin_velocities[j]
                    z1 = bin_depths[j-1]
                    z2 = bin_depths[j]
                    
                    # Calculate the interpolated z-value where velocity = threshold
                    ratio = (threshold - v1) / (v2 - v1)
                    interface_depth = z1 + ratio * (z2 - z1)
                    
                    interface_x.append((x_bins[i] + x_bins[i+1]) / 2)
                    interface_z.append(interface_depth)
                    break
    
    # Ensure we have interface points for the entire range
    # If first point is missing, extrapolate from the first available points
    if len(interface_x) > 0 and interface_x[0] > x_min + interval:
        interface_x.insert(0, x_min)
        # Use the slope of the first two points to extrapolate
        if len(interface_x) > 2:
            slope = (interface_z[1] - interface_z[0]) / (interface_x[1] - interface_x[0])
            interface_z.insert(0, interface_z[0] - slope * (interface_x[1] - x_min))
        else:
            interface_z.insert(0, interface_z[0])
    
    # If last point is missing, extrapolate from the last available points
    if len(interface_x) > 0 and interface_x[-1] < x_max - interval:
        interface_x.append(x_max)
        # Use the slope of the last two points to extrapolate
        if len(interface_x) > 2:
            slope = (interface_z[-1] - interface_z[-2]) / (interface_x[-1] - interface_x[-2])
            interface_z.append(interface_z[-1] + slope * (x_max - interface_x[-1]))
        else:
            interface_z.append(interface_z[-1])
    
    # Create a dense interpolation grid for smoothing
    x_dense = np.linspace(x_min, x_max, 500)  # 500 points for smooth curve
    
    # Apply cubic interpolation for smoother interface
    if len(interface_x) > 3:
        try:
            interp_func = interp1d(interface_x, interface_z, kind='cubic', 
                                  bounds_error=False, fill_value="extrapolate")
            z_dense = interp_func(x_dense)
            
            # Apply additional smoothing
            z_dense = savgol_filter(z_dense, window_length=31, polyorder=3)
        except:
            # Fall back to linear interpolation if cubic fails
            interp_func = interp1d(interface_x, interface_z, kind='linear',
                                  bounds_error=False, fill_value="extrapolate")
            z_dense = interp_func(x_dense)
    else:
        # Not enough points for cubic interpolation
        interp_func = interp1d(interface_x, interface_z, kind='linear',
                              bounds_error=False, fill_value="extrapolate")
        z_dense = interp_func(x_dense)
    
    # Prepare interface data dictionary
    interface_data = {
        'threshold': threshold,
        'raw_x': interface_x,
        'raw_z': interface_z,
        'smooth_x': x_dense,
        'smooth_z': z_dense,
        'min_x': x_min,
        'max_x': x_max
    }
    
    return x_dense, z_dense, interface_data


def save_velocity_structure(filename, x_coords, z_coords, interface_data=None):
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
    with open(csv_filename, 'w') as f:
        f.write('x,z\n')
        for x, z in zip(x_coords, z_coords):
            f.write(f"{x},{z}\n")