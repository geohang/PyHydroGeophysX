"""
Mesh utilities for geophysical modeling and inversion.
"""
import numpy as np
import pygimli as pg
import pygimli.meshtools as mt
from typing import Tuple, List, Optional, Union
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

def create_mesh_from_layers(surface: np.ndarray,
                          line1: np.ndarray,
                          line2: np.ndarray,
                          bottom_depth: float = 30.0,
                          quality: float = 28,
                          area: float = 40) -> Tuple[pg.Mesh, np.ndarray, np.ndarray]:
    """
    Create mesh from layer boundaries and get cell centers and markers.
    
    Args:
        surface: Surface coordinates [[x,z],...] 
        line1: First layer boundary coordinates 
        line2: Second layer boundary coordinates 
        bottom_depth: Depth below surface minimum for mesh bottom
        quality: Mesh quality parameter
        area: Maximum cell area
        
    Returns:
        mesh: PyGIMLI mesh
        mesh_centers: Array of cell center coordinates
        markers: Array of cell markers
    """
    # Calculate bottom elevation from normalized surface
    min_surface_elev = np.nanmin(surface[:,1])
    bottom_elev = bottom_depth #min_surface_elev - bottom_depth
    
    # Create reversed lines for polygon creation
    line1r = line1.copy()
    line1r[:,0] = np.flip(line1[:,0])
    line1r[:,1] = np.flip(line1[:,1])
    
    line2r = line2.copy()
    line2r[:,0] = np.flip(line2[:,0])
    line2r[:,1] = np.flip(line2[:,1])
    
    # Create surface layer
    layer1 = mt.createPolygon(surface,
                             isClosed=False, 
                             marker=2, 
                             boundaryMarker=-1,
                             interpolate='linear', 
                             area=0.1)
    
    # Create middle layer
    Gline1 = mt.createPolygon(np.vstack((line1, line2r)),
                             isClosed=True, 
                             marker=3, 
                             boundaryMarker=1,
                             interpolate='linear', 
                             area=1)
    
    # Create bottom boundary
    Gline2 = mt.createPolygon([[surface[0,0], surface[0,1]],
                              [line2[0,0], bottom_elev],
                              [line2[-1,0], bottom_elev],
                              [surface[-1,0], surface[-1,1]]],
                             isClosed=False, 
                             marker=2, 
                             boundaryMarker=1,
                             interpolate='linear', 
                             area=2)
    
    # Create bottom layer
    layer2 = mt.createPolygon(np.vstack((line2r,
                                        [[line2[0,0], line2[0,1]],
                                         [line2[0,0], bottom_elev],
                                         [line2[-1,0], bottom_elev],
                                         [line2[-1,0], line2[-1,1]]])),
                             isClosed=True, 
                             marker=2, 
                             area=2, 
                             boundaryMarker=1)
    
    # Combine all geometries
    geom = layer1 + layer2 + Gline1 + Gline2
    
    # Create mesh
    mesh = mt.createMesh(geom, quality=quality, area=area)
    
    # Get cell centers and markers
    mesh_centers = np.array(mesh.cellCenters())
    markers = np.array(mesh.cellMarkers())
    
    return mesh, mesh_centers, markers,geom







def extract_velocity_interface(mesh, velocity_data, threshold=1200, interval=4.0):
    """
    Extract the interface where velocity equals the threshold value.
    
    Args:
        mesh: The PyGIMLi mesh
        velocity_data: The velocity values
        threshold: The velocity value defining the interface (default: 1200)
        interval: Spacing between x-coordinate points (default: 4.0)
        
    Returns:
        x_dense, z_dense: Arrays with x and z coordinates of the smooth interface
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
    
    return x_dense, z_dense


def add_velocity_interface(ertData, smooth_x, smooth_z, paraBoundary=2, boundary=1):
    """
    Add a velocity interface line to the geometry and create a mesh with different markers:
    - Outside survey area: marker = 1
    - Inside survey area, above velocity line: marker = 2
    - Inside survey area, below velocity line: marker = 3
    
    Args:
        ertData: ERT data with sensor positions
        smooth_x, smooth_z: Arrays with x and z coordinates of the velocity interface
        paraBoundary: Parameter boundary size (default: 2)
        boundary: Boundary marker (default: 1)
        
    Returns:
        markers: Array with cell markers
        meshafter: The created mesh with updated markers
    """
    # Create the initial parameter mesh
    geo = mt.createParaMeshPLC(ertData, quality=32, paraMaxCellSize=30,
                               paraBoundary=paraBoundary, paraDepth=30.0,
                               boundaryMaxCellSize=500)
    
    # Stack x and z coordinates for the interface
    interface_points = np.vstack((smooth_x, smooth_z)).T
    
    # Extend the interface line beyond the data range by paraBoundary
    input_points = np.vstack((
        np.array([[interface_points[0][0] - paraBoundary, interface_points[0][1]]]),
        interface_points,
        np.array([[interface_points[-1][0] + paraBoundary, interface_points[-1][1]]])
    ))
    
    # Create a polygon line for the interface
    interface_line = mt.createPolygon(input_points.tolist(), isClosed=False,
                                     interpolate='linear', marker=99)
    
    # Add the interface to the geometry
    geo_with_interface = geo + interface_line
    
    # Create a mesh from the combined geometry
    meshafter = mt.createMesh(geo_with_interface, quality=28)
    
    # Initialize all markers to 1 (outside region)
    markers = np.ones(meshafter.cellCount())
    
    # Identify the survey area
    survey_left = ertData.sensors()[0][0] - paraBoundary
    survey_right = ertData.sensors()[-1][0] + paraBoundary
    
    # Process each cell
    for i in range(meshafter.cellCount()):
        cell_x = meshafter.cell(i).center().x()
        cell_y = meshafter.cell(i).center().y()
        
        # Only modify markers within the survey area
        if cell_x >= survey_left and cell_x <= survey_right:
            # Interpolate the interface height at this x position
            interface_y = np.interp(cell_x, input_points[:, 0], input_points[:, 1])
            
            # Set marker based on position relative to interface
            if abs(cell_y) < abs(interface_y):
                markers[i] = 2  # Below interface
            else:
                markers[i] = 3  # Above interface
    
    # Keep original markers for outside cells
    markers[meshafter.cellMarkers()==1] = 1
    
    # Set the updated markers
    meshafter.setCellMarkers(markers)
    
    return markers, meshafter





class MeshCreator:
    """Class for creating and managing meshes for geophysical inversion."""
    
    def __init__(self, quality: float = 28, area: float = 40):
        """
        Initialize MeshCreator with quality and area parameters.
        
        Args:
            quality: Mesh quality parameter (higher is better)
            area: Maximum cell area
        """
        self.quality = quality
        self.area = area
    
    def create_from_layers(self, surface: np.ndarray, 
                          layers: List[np.ndarray],
                          bottom_depth: float = 30.0,
                          markers: List[int] = None) -> pg.Mesh:
        """
        Create a mesh from surface and layer boundaries.
        
        Args:
            surface: Surface coordinates [[x,z],...]
            layers: List of layer boundary coordinates
            bottom_depth: Depth below surface minimum for mesh bottom
            markers: List of markers for each layer (default: [2, 3, 2, ...])
            
        Returns:
            PyGIMLI mesh
        """
        if len(layers) < 1:
            raise ValueError("At least one layer boundary is required")
            
        # Create default markers if not provided
        if markers is None:
            markers = [2] * (len(layers) + 1)
            if len(layers) > 0:
                markers[1] = 3  # Middle layer
        
        # Normalize elevation by maximum elevation
        max_ele = np.nanmax(surface[:,1])
        surface_norm = surface.copy()
        surface_norm[:,1] = surface_norm[:,1]  #- max_ele
        
        layers_norm = []
        for layer in layers:
            layer_norm = layer.copy()
            layer_norm[:,1] = layer_norm[:,1] # - max_ele
            layers_norm.append(layer_norm)
        
        # Create mesh using specific implementation
        if len(layers) == 2:
            mesh, centers, markers_array,geom = create_mesh_from_layers(
                surface_norm, layers_norm[0], layers_norm[1], 
                bottom_depth, self.quality, self.area
            )
            return mesh,geom
        else:
            # Implement custom mesh creation for different number of layers
            raise NotImplementedError("Currently only 2-layer mesh creation is implemented")
    
    def create_from_ert_data(self, data, max_depth: float = 30.0, quality: float = 34):
        """
        Create a mesh suitable for ERT inversion from ERT data.
        
        Args:
            data: PyGIMLI ERT data object
            max_depth: Maximum depth of the mesh
            quality: Mesh quality parameter
            
        Returns:
            PyGIMLI mesh for ERT inversion
        """
        from pygimli.physics import ert
        ert_manager = ert.ERTManager(data)
        return ert_manager.createMesh(data=data, quality=quality)
