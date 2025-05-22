"""
Interpolation utilities for geophysical data processing.
"""
import numpy as np
from scipy.interpolate import griddata
from typing import Tuple, List, Optional, Union


def interpolate_to_profile(data: np.ndarray, 
                         X_grid: np.ndarray, 
                         Y_grid: np.ndarray,
                         X_pro: np.ndarray, 
                         Y_pro: np.ndarray,
                         method: str = 'linear') -> np.ndarray:
    """
    Interpolate 2D data onto a profile line
    
    Args:
        data: 2D array of values to interpolate
        X_grid: X coordinates of original grid (meshgrid)
        Y_grid: Y coordinates of original grid (meshgrid)
        X_pro: X coordinates of profile points
        Y_pro: Y coordinates of profile points
        method: Interpolation method ('linear' or 'nearest')
        
    Returns:
        Interpolated values along profile
    """
    
    X_new = X_grid.ravel()
    Y_new = Y_grid.ravel()
    
    return griddata((X_new, Y_new), np.array(data).ravel(),
                   (np.array(X_pro).ravel(), np.array(Y_pro).ravel()),
                   method=method)


def setup_profile_coordinates(point1: List[int], 
                            point2: List[int],
                            surface_data: np.ndarray,
                            origin_x: float = 0.0,
                            origin_y: float = 0.0,
                            pixel_width: float = 1.0,
                            pixel_height: float = -1.0,
                            num_points: int = 200) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Set up profile coordinates based on surface elevation data between two points
    
    Args:
        point1: Starting point indices [col, row]
        point2: Ending point indices [col, row]
        surface_data: 2D array of surface elevation data
        origin_x: X coordinate of origin
        origin_y: Y coordinate of origin
        pixel_width: Width of each pixel
        pixel_height: Height of each pixel (negative for top-down)
        num_points: Number of points along profile
        
    Returns:
        X_pro: X coordinates along profile
        Y_pro: Y coordinates along profile
        L_profile: Distances along profile
        XX: X coordinate grid
        YY: Y coordinate grid
    """
    # Create coordinate grids
    x = origin_x + pixel_width * np.arange(surface_data.shape[1])
    y = origin_y + pixel_height * np.arange(surface_data.shape[0])
    XX, YY = np.meshgrid(x, y)
    
    # Handle no-data values
    surface_data = surface_data.copy()
    surface_data[surface_data == 0] = np.nan
    
    # Calculate start and end positions
    P1_pos = np.array([x[point1[0]], y[point1[1]]])
    P2_pos = np.array([x[point2[0]], y[point2[1]]])
    
    # Calculate total distance
    dis = np.sqrt(np.sum((P1_pos - P2_pos)**2))
    
    # Generate profile coordinates
    X_pro = (x[point1[0]] - x[point2[0]])/dis * np.linspace(0, dis, num_points)[:-1] + x[point2[0]]
    Y_pro = (y[point1[1]] - y[point2[1]])/dis * np.linspace(0, dis, num_points)[:-1] + y[point2[1]]
    
    # Calculate profile distances
    L_profile = np.sqrt((X_pro - X_pro[0])**2 + (Y_pro - Y_pro[0])**2)
    
    return X_pro, Y_pro, L_profile, XX, YY


def interpolate_structure_to_profile(structure_data: List[np.ndarray],
                                   X_grid: np.ndarray,
                                   Y_grid: np.ndarray,
                                   X_pro: np.ndarray,
                                   Y_pro: np.ndarray) -> np.ndarray:
    """
    Interpolate multiple structure layers onto profile
    
    Args:
        structure_data: List of 2D arrays for each layer
        X_grid: X coordinates of original grid
        Y_grid: Y coordinates of original grid
        X_pro: X coordinates of profile points
        Y_pro: Y coordinates of profile points
        
    Returns:
        Array of interpolated values with shape (n_layers, n_points)
    """
    structure = []
    for layer in structure_data:
        interpolated = interpolate_to_profile(layer, X_grid, Y_grid,
                                           X_pro, Y_pro)
        structure.append(interpolated)
    return np.array(structure)


def prepare_2D_profile_data(data: np.ndarray, 
                          XX: np.ndarray, 
                          YY: np.ndarray,
                          X_pro: np.ndarray,
                          Y_pro: np.ndarray) -> np.ndarray:
    """
    Interpolate multiple 2D gridded data layers onto a profile line.
    
    Args:
        data: 3D array of gridded data (n_layers, ny, nx)
        XX, YY: Coordinate grids from meshgrid
        X_pro, Y_pro: Profile line coordinates
        
    Returns:
        Interpolated values along profile (n_layers, n_profile_points)
    """
    n_layers = data.shape[0]
    profile_values = []
    
    X_new = XX.ravel()
    Y_new = YY.ravel()
    
    for i in range(n_layers):
        layer_values = griddata((X_new, Y_new), 
                              data[i].ravel(), 
                              (X_pro.ravel(), Y_pro.ravel()), 
                              method='linear')
        profile_values.append(layer_values)
    
    return np.array(profile_values)


def interpolate_to_mesh(property_values: np.ndarray,
                       profile_distance: np.ndarray,
                       depth_values: np.ndarray,
                       mesh_x: np.ndarray,
                       mesh_y: np.ndarray,
                       mesh_markers: np.ndarray,
                       ID,
                       layer_markers: list = [3, 0, 2]) -> np.ndarray:
    """
    Interpolate property values from profile to mesh with layer-specific handling.
    
    Args:
        property_values: Property values array (n_points)
        profile_distance: Distance along profile (n_points)
        depth_values: Depth values array (n_layers, n_points)
        mesh_x: X coordinates of mesh cells
        mesh_y: Y coordinates of mesh cells
        mesh_markers: Markers indicating different layers in mesh
        layer_markers: List of marker values for each layer
    
    Returns:
        Interpolated values for mesh cells
    """
    # Initialize output array
    result = np.zeros_like(mesh_markers, dtype=float)

    # print(profile_distance.shape)
    # print(depth_values.shape)
    # print(property_values.shape)
    L_profile_new = np.repeat(profile_distance.reshape(1,-1),property_values.shape[0],axis=0)

    Depth = depth_values[:14]

    maxele = 0 # set 0 here

    for marker in layer_markers:
        # For each layer marker, interpolate property values to mesh grid
        # Note: ID is used to identify which layer to interpolate for
        # Interpolate property values to mesh grid for each layer
        grid_z1 = griddata((L_profile_new[ID==marker].ravel(),Depth[ID==marker].ravel()- maxele), property_values[ID==marker].ravel(), (mesh_x[mesh_markers==marker], mesh_y[mesh_markers==marker]), method='linear')
        temp_ID = np.isnan(grid_z1)
        grid_z2 = griddata((L_profile_new[ID==marker].ravel(),Depth[ID==marker].ravel()- maxele), property_values[ID==marker].ravel(), (mesh_x[mesh_markers==marker], mesh_y[mesh_markers==marker]), method='nearest')
        grid_z1[temp_ID] = grid_z2[temp_ID]
        result[mesh_markers==marker] = grid_z1.copy()

    # # Interpolate property values to mesh grid for each layer
    # grid_z1 = griddata((L_profile_new[ID==0].ravel(),Depth[ID==0].ravel()- maxele), property_values[ID==0].ravel(), (mesh_x[mesh_markers==0], mesh_y[mesh_markers==0]), method='linear')
    # temp_ID = np.isnan(grid_z1)
    # grid_z2 = griddata((L_profile_new[ID==0].ravel(),Depth[ID==0].ravel()- maxele), property_values[ID==0].ravel(), (mesh_x[mesh_markers==0], mesh_y[mesh_markers==0]), method='nearest')
    # grid_z1[temp_ID] = grid_z2[temp_ID]
    # result[mesh_markers==0] = grid_z1.copy()



    #result =  griddata((L_profile_new.ravel(),depth_values[:14].ravel()), property_values.ravel(), (mesh_x, mesh_y), method='nearest')


    
    return result


class ProfileInterpolator:
    """Class for handling interpolation of data to/from profiles."""
    
    def __init__(self, point1: List[int], point2: List[int], 
                surface_data: np.ndarray,
                origin_x: float = 0.0, origin_y: float = 0.0,
                pixel_width: float = 1.0, pixel_height: float = -1.0,
                num_points: int = 200):
        """
        Initialize profile interpolator with reference points and surface data.
        
        Args:
            point1: Starting point indices [col, row]
            point2: Ending point indices [col, row]
            surface_data: 2D array of surface elevation data
            origin_x, origin_y: Coordinates of origin
            pixel_width, pixel_height: Pixel dimensions
            num_points: Number of points along profile
        """
        self.point1 = point1
        self.point2 = point2
        self.surface_data = surface_data
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.pixel_width = pixel_width
        self.pixel_height = pixel_height
        self.num_points = num_points
        
        # Set up profile coordinates
        self.X_pro, self.Y_pro, self.L_profile, self.XX, self.YY = setup_profile_coordinates(
            point1, point2, surface_data, origin_x, origin_y, 
            pixel_width, pixel_height, num_points
        )
        
        # Get surface profile
        self.surface_profile = interpolate_to_profile(
            surface_data, self.XX, self.YY, self.X_pro, self.Y_pro
        )
    
    def interpolate_layer_data(self, layer_data: List[np.ndarray]) -> np.ndarray:
        """
        Interpolate multiple layer data to profile.
        
        Args:
            layer_data: List of 2D arrays for each layer
            
        Returns:
            Array of interpolated values (n_layers, n_profile_points)
        """
        return interpolate_structure_to_profile(
            layer_data, self.XX, self.YY, self.X_pro, self.Y_pro
        )
    
    def interpolate_3d_data(self, data: np.ndarray) -> np.ndarray:
        """
        Interpolate 3D data (n_layers, ny, nx) to profile.
        
        Args:
            data: 3D array of values
            
        Returns:
            Array of interpolated values (n_layers, n_profile_points)
        """
        return prepare_2D_profile_data(
            data, self.XX, self.YY, self.X_pro, self.Y_pro
        )
    
    def interpolate_to_mesh(self, property_values: np.ndarray,
                          depth_values: np.ndarray,
                          mesh_x: np.ndarray,
                          mesh_y: np.ndarray,
                          mesh_markers: np.ndarray,
                          ID: np.ndarray,
                          layer_markers: list = [3, 0, 2]) -> np.ndarray:
        """
        Interpolate property values from profile to mesh with layer-specific handling.
        
        Args:
            property_values: Property values array (n_points or n_layers, n_points)
            depth_values: Depth values array (n_layers, n_points)
            mesh_x, mesh_y: Coordinates of mesh cells
            mesh_markers: Markers indicating different layers in mesh
            layer_markers: List of marker values for each layer
        
        Returns:
            Interpolated values for mesh cells
        """
        return interpolate_to_mesh(
            property_values, self.L_profile, depth_values,
            mesh_x, mesh_y, mesh_markers, ID,layer_markers
        )


def create_surface_lines(L_profile: np.ndarray,
                        structure: np.ndarray,
                        top_idx: int = 0,
                        mid_idx: int = 4,
                        bot_idx: int = 12) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create surface and boundary lines from structure data
    
    Args:
        L_profile: Distance along profile
        structure: Interpolated structure data
        top_idx: Index for top surface
        mid_idx: Index for middle boundary
        bot_idx: Index for bottom boundary
        
    Returns:
        surface: Surface coordinates
        line1: First boundary coordinates
        line2: Second boundary coordinates
    """
    # Extract and reshape structure layers
    S1 = structure[top_idx,:].reshape(-1,1)
    S2 = structure[mid_idx,:].reshape(-1,1)
    S3 = structure[bot_idx,:].reshape(-1,1)
    
    # Create coordinate arrays
    surface = np.hstack((L_profile.reshape(-1,1), S1))
    line1 = np.hstack((L_profile.reshape(-1,1), S2))
    line2 = np.hstack((L_profile.reshape(-1,1), S3))
    
    # Normalize by maximum elevation
    #maxele = np.nanmax(surface[:,1])
    surface[:,1] = surface[:,1] #- maxele
    line1[:,1] = line1[:,1] #- maxele
    line2[:,1] = line2[:,1] #- maxele
    
    return surface, line1, line2