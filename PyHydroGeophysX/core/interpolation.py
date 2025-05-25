"""
Geophysical Data Interpolation Utilities.

This module provides functions and a class for various interpolation tasks 
commonly encountered in hydrogeophysical data processing. These tasks include:
- Interpolating 2D gridded data onto a profile line.
- Setting up profile coordinates from surface elevation data.
- Interpolating multiple geological structure layers onto a profile.
- Preparing and interpolating 3D data (e.g., from multiple geophysical surveys) onto a profile.
- Interpolating property values from a profile representation to a 2D mesh, 
  considering different geological layers.
- Extracting specific surface and boundary lines from interpolated structural data.

The `ProfileInterpolator` class encapsulates profile definition and provides methods
for streamlined interpolation of different data types onto that profile and subsequently
onto a 2D mesh.
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
    Interpolate 2D gridded data onto a specified profile line.
    
    This function takes values from a 2D grid and interpolates them at
    the coordinates defining a profile line.
    
    Args:
        data (np.ndarray): 2D array of values to interpolate.
        X_grid (np.ndarray): 2D array of X coordinates of the original grid (e.g., from `np.meshgrid`).
        Y_grid (np.ndarray): 2D array of Y coordinates of the original grid (e.g., from `np.meshgrid`).
        X_pro (np.ndarray): 1D array of X coordinates of the profile points.
        Y_pro (np.ndarray): 1D array of Y coordinates of the profile points.
        method (str, optional): Interpolation method to use. Defaults to 'linear'.
                                 Acceptable values are 'linear', 'nearest', 'cubic' (from scipy.interpolate.griddata).
        
    Returns:
        np.ndarray: 1D array of interpolated values along the profile line.
    """
    # Flatten the grid coordinates and data for griddata function
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
    """
    Set up profile coordinates based on surface elevation data between two specified points.
    
    This function defines a profile line between two points (given by their indices on a grid
    of `surface_data`). It calculates the X and Y coordinates in world units along this profile,
    the cumulative distance along the profile, and also returns the coordinate grids (XX, YY)
    for the entire `surface_data` domain.
    
    Args:
        point1 (List[int]): Starting point grid indices [column_index, row_index] for the profile.
        point2 (List[int]): Ending point grid indices [column_index, row_index] for the profile.
        surface_data (np.ndarray): 2D array of surface elevation data. This grid defines the
                                   spatial context (shape) for interpreting `point1` and `point2`.
        origin_x (float, optional): X coordinate of the world origin corresponding to the
                                   grid's (0,0) index (typically top-left or bottom-left). Defaults to 0.0.
        origin_y (float, optional): Y coordinate of the world origin corresponding to the
                                   grid's (0,0) index. Defaults to 0.0.
        pixel_width (float, optional): Width of each pixel/cell in `surface_data` in world units. 
                                      Defaults to 1.0.
        pixel_height (float, optional): Height of each pixel/cell in `surface_data` in world units. 
                                     Typically negative if row indices increase downwards (image/matrix convention). 
                                     Defaults to -1.0.
        num_points (int, optional): Number of equidistant points to generate along the profile line. 
                                   Defaults to 200.
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            - X_pro (np.ndarray): 1D array of X coordinates (world units) along the profile. Empty if num_points <= 0.
            - Y_pro (np.ndarray): 1D array of Y coordinates (world units) along the profile. Empty if num_points <= 0.
            - L_profile (np.ndarray): 1D array of cumulative distances from the start of the profile 
                                      to each point on the profile. Empty if num_points <= 0.
            - XX (np.ndarray): 2D array of X coordinates (world units) for the entire grid.
            - YY (np.ndarray): 2D array of Y coordinates (world units) for the entire grid.
    """
    # Create 1D arrays representing the world coordinates of grid cell centers or edges
    x_coords_grid_axis = origin_x + pixel_width * np.arange(surface_data.shape[1])
    y_coords_grid_axis = origin_y + pixel_height * np.arange(surface_data.shape[0])
    # Create 2D meshgrid representing X and Y world coordinates for each point in surface_data
    XX, YY = np.meshgrid(x_coords_grid_axis, y_coords_grid_axis)
    
    # Make a copy of surface_data to avoid modifying the input array.
    # Replace 0 with NaN, assuming 0 is a no-data value. This should be documented or parameterized if no-data value varies.
    surface_data_processed = surface_data.copy()
    surface_data_processed[surface_data_processed == 0] = np.nan
    
    # Get world coordinates for the start (P1) and end (P2) points of the profile
    P1_world_coords = np.array([x_coords_grid_axis[point1[0]], y_coords_grid_axis[point1[1]]])
    P2_world_coords = np.array([x_coords_grid_axis[point2[0]], y_coords_grid_axis[point2[1]]])
    
    # Generate `num_points` equally spaced points along the profile line, from P1 to P2.
    if num_points <= 0: # Handle cases with no or non-positive number of points
        X_pro = np.array([])
        Y_pro = np.array([])
    elif num_points == 1: # Single point profile: use the starting point P1.
        X_pro = np.array([P1_world_coords[0]])
        Y_pro = np.array([P1_world_coords[1]])
    else: # For num_points > 1
        X_pro = np.linspace(P1_world_coords[0], P2_world_coords[0], num_points)
        Y_pro = np.linspace(P1_world_coords[1], P2_world_coords[1], num_points)
    
    # Calculate cumulative distances along the profile from its starting point (X_pro[0], Y_pro[0])
    if num_points > 0: # L_profile can only be calculated if there are points
        L_profile = np.sqrt((X_pro - X_pro[0])**2 + (Y_pro - Y_pro[0])**2)
    else:
        L_profile = np.array([]) # Empty if no points on profile
            
    return X_pro, Y_pro, L_profile, XX, YY


def interpolate_structure_to_profile(structure_data: List[np.ndarray],
                                   X_grid: np.ndarray,
                                   Y_grid: np.ndarray,
                                   X_pro: np.ndarray,
                                   Y_pro: np.ndarray) -> np.ndarray:
    """
    """
    Interpolate multiple geological structure layers onto a defined profile line.
    
    Each layer is represented by a 2D array of values (e.g., elevation or depth data
    on the original grid). This function iterates through each layer, interpolates its 
    values onto the profile coordinates using `interpolate_to_profile` (defaulting to 
    linear interpolation), and stacks the results.
    
    Args:
        structure_data (List[np.ndarray]): A list where each element is a 2D numpy array 
                                          representing a geological layer's data on the original grid.
                                          Each array in the list should have the same spatial shape.
        X_grid (np.ndarray): 2D array of X coordinates of the original grid (e.g., from `np.meshgrid`).
                             Must match the spatial dimensions of the data arrays in `structure_data`.
        Y_grid (np.ndarray): 2D array of Y coordinates of the original grid (e.g., from `np.meshgrid`).
                             Must match the spatial dimensions of the data arrays in `structure_data`.
        X_pro (np.ndarray): 1D array of X coordinates of the profile points.
        Y_pro (np.ndarray): 1D array of Y coordinates of the profile points.
        
    Returns:
        np.ndarray: An array of interpolated values with shape (n_layers, n_profile_points),
                    where n_layers is the number of arrays in `structure_data`, and
                    n_profile_points is the number of points along the profile.
    """
    interpolated_layers_list = []
    for layer_data_on_grid in structure_data:
        # Interpolate current layer to the profile line using linear interpolation by default
        # (as per interpolate_to_profile's default method).
        interpolated_layer_on_profile = interpolate_to_profile(
            layer_data_on_grid, X_grid, Y_grid, X_pro, Y_pro, method='linear'
        )
        interpolated_layers_list.append(interpolated_layer_on_profile)
    return np.array(interpolated_layers_list)


def prepare_2D_profile_data(data: np.ndarray, 
                          XX: np.ndarray, 
                          YY: np.ndarray,
                          X_pro: np.ndarray,
                          Y_pro: np.ndarray) -> np.ndarray:
    """
    """
    Interpolate multiple 2D gridded data layers (stacked in a 3D array) onto a profile line.
    
    This function is similar to `interpolate_structure_to_profile` but is designed for
    a single 3D numpy array where the first dimension represents different layers or datasets
    (e.g., multiple geophysical survey results at different depths or times, or different properties).
    
    Args:
        data (np.ndarray): 3D array of gridded data with shape (n_layers, ny, nx),
                           where n_layers is the number of data layers/properties, 
                           ny is the number of rows (Y dimension of the grid), and 
                           nx is the number of columns (X dimension of the grid).
        XX (np.ndarray): 2D array of X coordinates of the grid (e.g., from `np.meshgrid(x_coords, y_coords)`). 
                         Shape must be (ny, nx), matching `data.shape[1:]`.
        YY (np.ndarray): 2D array of Y coordinates of the grid (e.g., from `np.meshgrid(x_coords, y_coords)`).
                         Shape must be (ny, nx), matching `data.shape[1:]`.
        X_pro (np.ndarray): 1D array of X coordinates of the profile points where data is to be interpolated.
        Y_pro (np.ndarray): 1D array of Y coordinates of the profile points where data is to be interpolated.
        
    Returns:
        np.ndarray: Interpolated values along the profile, with shape (n_layers, n_profile_points),
                    where n_profile_points is the number of points in `X_pro`/`Y_pro`.
    """
    # Input validation
    if data.ndim != 3:
        raise ValueError(f"Input data must be a 3D array (n_layers, ny, nx), got {data.ndim} dimensions.")
    if XX.shape != YY.shape:
        raise ValueError(f"XX and YY meshgrid shapes must match. Got XX: {XX.shape}, YY: {YY.shape}")
    if data.shape[1:] != XX.shape: # Check if data's spatial dimensions (ny, nx) match meshgrid shape
        raise ValueError(f"Spatial dimensions of data ({data.shape[1:]}) must match XX, YY grid ({XX.shape}).")

    n_layers = data.shape[0]
    interpolated_profile_values_list = []
    
    # Flatten the grid coordinates for use in scipy.interpolate.griddata.
    # These are the "known" points from which to interpolate.
    grid_X_flat = XX.ravel()
    grid_Y_flat = YY.ravel()
    
    # Ensure profile coordinates are 1D arrays (ravel just in case they are not).
    # These are the "query" points where we want to interpolate values.
    profile_X_flat = X_pro.ravel()
    profile_Y_flat = Y_pro.ravel()

    for i in range(n_layers):
        current_layer_data_flat = data[i].ravel() # Flatten the data for the current layer
        # Interpolate the i-th layer to the profile coordinates using linear interpolation.
        layer_profile_values = griddata(
            (grid_X_flat, grid_Y_flat),      # Points (X,Y) where data values are known
            current_layer_data_flat,         # Known data values for the current layer
            (profile_X_flat, profile_Y_flat),# Points (X,Y) along the profile where interpolated values are desired
            method='linear'                  # Interpolation method (can be parameterized if needed)
        )
        interpolated_profile_values_list.append(layer_profile_values)
    
    return np.array(interpolated_profile_values_list)


def interpolate_to_mesh(property_values: np.ndarray,
                       profile_distance: np.ndarray,
                       depth_values: np.ndarray,
                       mesh_x: np.ndarray,
                       mesh_y: np.ndarray,
                       mesh_markers: np.ndarray,
                       ID,
                       layer_markers: list = [3, 0, 2]) -> np.ndarray:
    """
    """
    Interpolate property values from a profile representation to a 2D mesh, 
    with layer-specific handling using markers.

    The core interpolation strategy is as follows:
    1. Iterate through each unique `marker_val` specified in `layer_markers`.
    2. For the current `marker_val`, identify the relevant data points from `property_values` 
       and their corresponding `depth_values` using the `ID` array. The `ID` array acts as
       a map, indicating which layer (`marker_val`) each point in `property_values` and 
       `depth_values` belongs to.
    3. The `profile_distance` array (repeated to match dimensions of `property_values`) provides the X-coordinates 
       (distance along profile) for these points. The selected `depth_values` (potentially adjusted 
       by `maxele`) provide the Y-coordinates (depth).
    4. These (X, Y, property_value) triplets, specific to the current `marker_val`, are then 
       interpolated onto the cells of the `mesh_x`, `mesh_y` grid that are designated by this 
       same `marker_val` in `mesh_markers`.
    5. Linear interpolation (`method='linear'`) is attempted first. If it results in NaNs (e.g., for points
       outside the convex hull of the input data for that layer), nearest neighbor 
       interpolation (`method='nearest'`) is used to fill these NaNs for the affected points.
    
    Args:
        property_values (np.ndarray): Array of property values to be interpolated. 
                                     Expected shape (n_profile_layers, n_profile_points), where
                                     `n_profile_layers` corresponds to distinct sets of properties
                                     (e.g., resistivity at different investigation depths along the profile)
                                     and `n_profile_points` is the number of points along the profile.
        profile_distance (np.ndarray): 1D array of distances along the profile (n_profile_points). 
                                      This serves as the X-coordinate for the profile data points.
        depth_values (np.ndarray): Array of depth values corresponding to `property_values`.
                                  Expected shape (n_profile_layers, n_profile_points). This serves
                                  as the Y-coordinate (depth) for the profile data points.
                                  Note: The line `Depth = depth_values[:14]` means that currently
                                  only the first 14 layers/rows of `depth_values` are actively used.
        mesh_x (np.ndarray): 1D or 2D array of X coordinates of the mesh cell centers to interpolate onto.
                             Must be broadcastable with `mesh_y` to the shape of `mesh_markers`.
        mesh_y (np.ndarray): 1D or 2D array of Y coordinates of the mesh cell centers to interpolate onto.
                             Must be broadcastable with `mesh_x` to the shape of `mesh_markers`.
        mesh_markers (np.ndarray): An array with the same shape as the target mesh grid (implied by `mesh_x`, `mesh_y`),
                                  where each cell contains an integer marker indicating the geological
                                  layer or region it belongs to. This controls which parts of the mesh
                                  receive interpolated values from which layer-specific data.
        ID (np.ndarray): An array with the same shape as `property_values` and `depth_values`
                         (i.e., (n_profile_layers, n_profile_points)). It contains markers that link
                         each data point in `property_values` and `depth_values` to a specific layer marker
                         (those found in `layer_markers`). This is crucial for selecting the correct
                         subset of profile data for interpolating onto the corresponding mesh region.
        layer_markers (list, optional): A list of integer marker values. The interpolation will be performed
                                        iteratively for each marker in this list. For each marker,
                                        the corresponding data from `property_values` (filtered by `ID == marker`)
                                        will be interpolated onto the mesh cells marked with this same `marker`
                                        in `mesh_markers`. Defaults to `[3, 0, 2]`.
    
    Returns:
        np.ndarray: An array of the same shape as `mesh_markers` containing the interpolated 
                    values for the mesh cells. Cells in `mesh_markers` not corresponding to any
                    marker in `layer_markers` will retain their initial zero value.
    """
    # Initialize output array with zeros, matching the shape of mesh_markers.
    # Values will be float as geophysical properties are typically continuous.
    result = np.zeros_like(mesh_markers, dtype=float)

    # `L_profile_new` repeats `profile_distance` for each "layer" defined in `property_values`.
    # This creates a 2D array of shape (property_values.shape[0], profile_distance.shape[0]),
    # making it align with `property_values`, `depth_values`, and `ID` for easy masking.
    L_profile_new = np.repeat(profile_distance.reshape(1, -1), property_values.shape[0], axis=0)

    # `Depth` currently uses only the first 14 layers/rows from the input `depth_values`.
    # This is a fixed assumption that might originate from a specific dataset structure or model convention
    # (e.g. a model with a maximum of 14 distinct subsurface layers relevant for this interpolation).
    # If the number of relevant layers in `depth_values` can vary or exceed 14,
    # this slicing should be made dynamic (e.g., using `depth_values.shape[0]`, a parameter,
    # or inferred from the unique values in `ID` that are also in `layer_markers`).
    # TODO: Re-evaluate the fixed slice `[:14]`. Consider parameterizing or using `depth_values.shape[0]`.
    Depth = depth_values[:14] # Slices to use only the first 14 rows of depth_values.

    # `maxele` is intended for vertical datum adjustment of the `Depth` values.
    # Setting `maxele = 0` means no adjustment is currently applied; `Depth` values are used as is (after slicing).
    # If, for example, `depth_values` represented absolute elevations and `maxele` was set to
    # a reference elevation (e.g., `np.nanmax(surface_profile_elevation)`), then `Depth - maxele` 
    # would convert these elevations to depths relative to that reference.
    # Its current value of 0 implies that depth_values are already relative to the desired datum, or absolute and used as such.
    # TODO: Evaluate if `maxele` should be a parameter, calculated (e.g., from surface data), or removed if always 0.
    maxele = 0  # Purpose: Vertical datum adjustment. Set to 0 implies no adjustment to input depth_values.

    # Iterate through each layer marker specified in the `layer_markers` list (e.g., [3, 0, 2])
    for marker_val in layer_markers:
        # `current_layer_data_mask` is a boolean array. It's true where `ID` matches the current `marker_val`.
        # This mask is used to select the relevant portions of `L_profile_new`, `Depth` (the sliced one),
        # and `property_values` that correspond to the current geological layer/marker.
        # Note: This assumes `ID` is also structured such that its first dimension aligns with `property_values.shape[0]`
        # and that relevant IDs are within the first 14 layers if `Depth` is sliced.
        current_layer_data_mask = (ID == marker_val) # Shape: (property_values.shape[0], profile_distance.shape[0])
        
        # Define the known (X, Y) points for interpolation from the profile data for the current layer.
        # `L_profile_new[current_layer_data_mask]` extracts distances along profile.
        # `(Depth[current_layer_data_mask] - maxele)` extracts adjusted depths.
        # `.ravel()` flattens them into 1D arrays suitable for `griddata`.
        source_points_xy = (L_profile_new[current_layer_data_mask].ravel(), 
                            (Depth[current_layer_data_mask] - maxele).ravel())
        
        # Define the property values at these known source points.
        source_values = property_values[current_layer_data_mask].ravel()
        
        # Define the target (X,Y) points on the mesh where we want to interpolate values.
        # This is restricted to only those mesh cells that are marked with the current `marker_val`.
        target_mesh_cells_mask = (mesh_markers == marker_val) # Boolean mask for cells in the current layer.
        target_points_xy = (mesh_x[target_mesh_cells_mask], mesh_y[target_mesh_cells_mask])

        # Check if there are any source points for the current marker. If not, skip interpolation for this marker.
        if source_points_xy[0].size == 0 or source_values.size == 0:
            # print(f"Warning: No source data points found for marker {marker_val} in ID array. Skipping interpolation for this marker.")
            continue
        
        # Check if there are any target mesh cells for the current marker. If not, skip.
        if target_points_xy[0].size == 0:
            # print(f"Warning: No target mesh cells found for marker {marker_val} in mesh_markers. Skipping interpolation for this marker.")
            continue

        # Perform linear interpolation for the current layer.
        interpolated_linear = griddata(source_points_xy, source_values, target_points_xy, method='linear')
        
        # Identify NaNs where linear interpolation failed (e.g., target points outside the convex hull of source points).
        nan_mask_linear = np.isnan(interpolated_linear)
        
        # If there are any NaNs after linear interpolation, fill them using nearest neighbor interpolation.
        # This ensures that all cells in the target layer region receive an estimated value.
        if np.any(nan_mask_linear):
            # Interpolate using 'nearest' only for the points that were NaN after linear interpolation.
            # This avoids re-calculating for already successfully interpolated points.
            target_points_for_nearest = (target_points_xy[0][nan_mask_linear], target_points_xy[1][nan_mask_linear])
            
            # Ensure there are still points to interpolate for nearest neighbors (i.e., nan_mask_linear was not all False for target_points_xy)
            if target_points_for_nearest[0].size > 0:
                 interpolated_nearest = griddata(source_points_xy, source_values, target_points_for_nearest, method='nearest')
                 interpolated_linear[nan_mask_linear] = interpolated_nearest # Fill NaNs
            
        # Assign the (potentially filled) interpolated values to the corresponding cells in the `result` array.
        result[target_mesh_cells_mask] = interpolated_linear
    
    return result


class ProfileInterpolator:
    """
    A class to streamline interpolation tasks related to a specific profile.

    Upon initialization, this class defines a profile using `setup_profile_coordinates`
    based on two points on a surface data grid. This profile includes X, Y coordinates (world units),
    distances along the profile (`L_profile`), and the original grid coordinates (`XX`, `YY`)
    of the input `surface_data`'s domain. It also pre-calculates and stores the
    surface elevation along this defined profile (`surface_profile`).

    The class then provides methods to:
    - Interpolate multiple 2D layer data (e.g., geological layers from grids) onto this profile.
    - Interpolate 3D gridded data (e.g., geophysical data volumes) onto this profile.
    - Interpolate property values from the profile representation (using profile distance and associated depths)
      to a 2D mesh, with layer-specific handling via markers.
    """
    
    def __init__(self, point1: List[int], point2: List[int], 
                surface_data: np.ndarray,
                origin_x: float = 0.0, origin_y: float = 0.0,
                pixel_width: float = 1.0, pixel_height: float = -1.0,
                num_points: int = 200):
        """
        Initialize the ProfileInterpolator with parameters to define a profile.
        
        This constructor sets up the fundamental profile geometry (X-Y coordinates in world units,
        distance along profile `L_profile`, and the meshgrid `XX`, `YY` of the original data space)
        by calling `setup_profile_coordinates`. It also computes and stores the
        elevation of the `surface_data` interpolated onto this defined profile as `self.surface_profile`.
        
        Args:
            point1 (List[int]): Starting point grid indices [column_index, row_index] for the profile,
                                referring to indices within the `surface_data` grid.
            point2 (List[int]): Ending point grid indices [column_index, row_index] for the profile,
                                referring to indices within the `surface_data` grid.
            surface_data (np.ndarray): 2D array of surface elevation data. This grid defines the
                                     spatial context (shape, cell size through pixel_width/height)
                                     for interpreting `point1`, `point2`, and for subsequent interpolations.
            origin_x (float, optional): X coordinate of the world origin corresponding to the
                                       grid's (0,0) index (e.g., top-left or bottom-left corner). Defaults to 0.0.
            origin_y (float, optional): Y coordinate of the world origin corresponding to the
                                       grid's (0,0) index. Defaults to 0.0.
            pixel_width (float, optional): Width of each pixel/cell in `surface_data` in world units. 
                                          Defaults to 1.0.
            pixel_height (float, optional): Height of each pixel/cell in `surface_data` in world units. 
                                         Typically negative if row indices of `surface_data` increase downwards
                                         (standard image/matrix convention). Defaults to -1.0.
            num_points (int, optional): Number of equidistant points to generate along the profile line. 
                                       Defaults to 200.
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
            layer_data (List[np.ndarray]): A list of 2D numpy arrays, where each array represents
                                          a layer's data (e.g., elevation, thickness) on the 
                                          original grid (which corresponds to `self.XX`, `self.YY`).
            
        Returns:
            np.ndarray: Array of interpolated values for each layer along the defined profile.
                        The shape will be (n_layers, n_profile_points), where n_layers is
                        the number of arrays in `layer_data` and n_profile_points is
                        `self.num_points` (the number of points on the profile).
                        Returns an empty array if the profile has no points.
        """
        if self.X_pro.size == 0: # Or self.num_points <= 0
            return np.array([])
        return interpolate_structure_to_profile(
            layer_data, self.XX, self.YY, self.X_pro, self.Y_pro
        )
    
    def interpolate_3d_data(self, data: np.ndarray) -> np.ndarray:
        """
        Interpolate 3D gridded data (e.g., a geophysical data volume) onto this interpolator's defined profile.
        
        The input `data` is expected to be a 3D array where the first dimension
        represents different layers, slices, or depths of the volume. The other two
        dimensions (ny, nx) must match the grid defined by `self.XX` and `self.YY`
        (which was derived from the `surface_data` provided at initialization).
        
        Args:
            data (np.ndarray): 3D array of values with shape (n_layers, ny, nx).
            
        Returns:
            np.ndarray: Array of interpolated values along the profile for each layer/slice, 
                        with shape (n_layers, n_profile_points), where n_profile_points
                        is `self.num_points`. Returns an empty array if the profile has no points.
        """
        if self.X_pro.size == 0: # Or self.num_points <= 0
            return np.array([])
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
            property_values (np.ndarray): Array of property values to be interpolated from the profile
                                         representation to the 2D mesh. Expected shape is typically
                                         (n_profile_layers, n_profile_points), analogous to the
                                         `property_values` in the standalone `interpolate_to_mesh` function.
                                         `n_profile_points` should match `self.num_points`.
            depth_values (np.ndarray): Array of depth values corresponding to `property_values` along the profile.
                                      Expected shape (n_profile_layers, n_profile_points).
                                      Important: The called `interpolate_to_mesh` function currently hardcodes
                                      the use of `depth_values[:14]`, implying an assumption that only the
                                      first 14 layers/rows of this array are relevant.
            mesh_x (np.ndarray): 1D or 2D array of X coordinates of the target mesh cell centers.
            mesh_y (np.ndarray): 1D or 2D array of Y coordinates of the target mesh cell centers.
            mesh_markers (np.ndarray): An array defining the layer marker for each cell in the target mesh.
                                      Used to guide layer-specific interpolation.
            ID (np.ndarray): An array, typically with the same shape as `property_values` and `depth_values`
                             (i.e., (n_profile_layers, n_profile_points)). It links each data point in
                             `property_values` and `depth_values` to a specific layer marker (from `layer_markers`).
            layer_markers (list, optional): A list of integer marker values that specify which layers
                                           to process during interpolation. Defaults to `[3, 0, 2]`.
        
        Returns:
            np.ndarray: An array of the same shape as `mesh_markers` containing the interpolated values 
                        for the mesh cells. Returns an empty array or array of zeros if the profile has no points.

        Note on behavior inherited from `interpolate_to_mesh`:
            - The `depth_values` input is internally sliced to `depth_values[:14]` by the
              standalone `interpolate_to_mesh` function. This implies a fixed assumption about the
              number of effective depth layers used from the input.
            - A variable `maxele` (intended for vertical datum adjustment of depths) is hardcoded to 0
              within `interpolate_to_mesh`, meaning no such adjustment is currently performed on `depth_values`.
              Depths are used as provided (after slicing).
        """
        if self.L_profile.size == 0: # Or self.num_points <= 0
             # Return zeros matching mesh_markers if profile is empty, as per interpolate_to_mesh behavior
            return np.zeros_like(mesh_markers, dtype=float)
            
        # `self.L_profile` (distance along this interpolator's defined profile) is passed as `profile_distance`.
        return interpolate_to_mesh(
            property_values, self.L_profile, depth_values,
            mesh_x, mesh_y, mesh_markers, ID, layer_markers
        )


def create_surface_lines(L_profile: np.ndarray,
                        structure: np.ndarray,
                        top_idx: int = 0,
                        mid_idx: int = 4,
                        bot_idx: int = 12) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    """
    Create coordinate pairs for plotting surface and boundary lines from structure data 
    that has already been interpolated onto a profile.
    
    This function extracts specific layers (identified by `top_idx`, `mid_idx`, `bot_idx`)
    from the `structure` data (which should be values along a profile) and pairs them
    with the corresponding distances along the profile (`L_profile`). The result is
    a set of (distance, value) coordinate pairs suitable for plotting lines.
    The values are returned as is, without any normalization.
    
    Args:
        L_profile (np.ndarray): 1D array of distances along the profile. This will form the
                                X-coordinates of the output lines. Shape (n_profile_points,).
        structure (np.ndarray): 2D array of interpolated structure data, where rows
                                correspond to different layers/surfaces and columns correspond to
                                points along the profile. Shape (n_layers, n_profile_points).
        top_idx (int, optional): Index of the row in `structure` that represents the top surface. 
                               Defaults to 0.
        mid_idx (int, optional): Index of the row in `structure` that represents a middle boundary line. 
                               Defaults to 4.
        bot_idx (int, optional): Index of the row in `structure` that represents a bottom boundary line. 
                               Defaults to 12.
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - surface (np.ndarray): Coordinate pairs [distance, value] for the top surface line.
                                    Shape (n_profile_points, 2). Values are taken directly from `structure`.
            - line1 (np.ndarray): Coordinate pairs [distance, value] for the middle boundary line.
                                  Shape (n_profile_points, 2). Values are taken directly from `structure`.
            - line2 (np.ndarray): Coordinate pairs [distance, value] for the bottom boundary line.
                                  Shape (n_profile_points, 2). Values are taken directly from `structure`.
    """
    # Extract specified layers (rows) from the structure data.
    # .reshape(-1, 1) converts the 1D array of layer values into a 2D column vector.
    S1_values = structure[top_idx, :].reshape(-1, 1) # Values for the top surface
    S2_values = structure[mid_idx, :].reshape(-1, 1) # Values for the middle boundary
    S3_values = structure[bot_idx, :].reshape(-1, 1) # Values for the bottom boundary
    
    # Reshape L_profile to be a column vector to enable horizontal stacking with layer values.
    L_profile_col_vec = L_profile.reshape(-1, 1)
    
    # Create coordinate arrays by horizontally stacking distance (L_profile) and layer values.
    # Each resulting array will have shape (n_profile_points, 2).
    surface = np.hstack((L_profile_col_vec, S1_values))
    line1 = np.hstack((L_profile_col_vec, S2_values))
    line2 = np.hstack((L_profile_col_vec, S3_values))
    
    # The original code had commented-out lines suggesting normalization by maxele (maximum elevation).
    # Those lines have been removed to clarify that this function returns absolute values from the structure.
    # If normalization is needed, it should be implemented explicitly outside or in a dedicated function.
    
    return surface, line1, line2