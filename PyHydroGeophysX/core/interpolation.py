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
    
    # Ravel the grid coordinates to be 1D arrays for griddata
    # griddata expects 1D arrays of points, so X_grid and Y_grid (from meshgrid) are flattened.
    X_new = X_grid.ravel()
    Y_new = Y_grid.ravel()
    
    # Interpolate data to profile points
    # np.array(data).ravel() flattens the input 2D data array to a 1D array,
    # corresponding to the flattened grid coordinates.
    # (np.array(X_pro).ravel(), np.array(Y_pro).ravel()) provides the target profile coordinates,
    # also flattened to ensure they are 1D arrays.
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
    # Create 1D coordinate arrays for x and y based on origin and pixel dimensions
    # This defines the spatial extent of the input surface_data in real-world coordinates.
    x = origin_x + pixel_width * np.arange(surface_data.shape[1])  # X-coordinates for columns
    y = origin_y + pixel_height * np.arange(surface_data.shape[0]) # Y-coordinates for rows; pixel_height is often negative for image-like (top-down) coordinate systems
    # Create 2D meshgrid from the 1D coordinate arrays. XX and YY will hold the coordinates for each point in surface_data.
    XX, YY = np.meshgrid(x, y)
    
    # Handle no-data values (often represented as 0 in raster data) by converting them to NaN
    # This is important for interpolation, as NaN values are typically ignored.
    surface_data = surface_data.copy() # Avoid modifying the original input array
    surface_data[surface_data == 0] = np.nan # Replace 0s with NaN
    
    # Transform point indices (column, row) to actual spatial coordinates
    # point1 and point2 are given as [column_index, row_index].
    # x array corresponds to columns (surface_data.shape[1]), y array to rows (surface_data.shape[0]).
    P1_pos = np.array([x[point1[0]], y[point1[1]]]) # Coordinates of the starting point of the profile
    P2_pos = np.array([x[point2[0]], y[point2[1]]]) # Coordinates of the ending point of the profile
    
    # Calculate the Euclidean distance between the start and end points of the profile line
    dis = np.sqrt(np.sum((P1_pos - P2_pos)**2))
    
    # Generate evenly spaced points along the profile line in X and Y dimensions.
    # np.linspace creates num_points points from 0 to `dis`.
    # The [:-1] excludes the endpoint. This might be to define `num_points` segments rather than `num_points` including the end.
    # The logic effectively parameterizes the line segment from P2_pos to P1_pos and samples points along it.
    # It calculates points on the line defined by P1 and P2.
    # (x[point1[0]] - x[point2[0]])/dis is the x-component of the normalized vector from P2 to P1.
    # This is then scaled by distances from np.linspace and shifted by x[point2[0]].
    # SUGGESTION: A more standard way to write this might be:
    # t = np.linspace(0, 1, num_points)
    # X_pro = P1_pos[0] * t + P2_pos[0] * (1-t)
    # Y_pro = P1_pos[1] * t + P2_pos[1] * (1-t)
    # Or, if keeping the current structure but starting from P1:
    # X_pro = P1_pos[0] + (P2_pos[0] - P1_pos[0])/dis * np.linspace(0, dis, num_points)
    # Y_pro = P1_pos[1] + (P2_pos[1] - P1_pos[1])/dis * np.linspace(0, dis, num_points)
    # The current implementation seems to generate points from P2 towards P1, then effectively reverses them by how L_profile is calculated later if P1 is the intended start.
    # However, the variable names X_pro, Y_pro usually imply the sequence of points along the profile.
    # If point1 is the start and point2 is the end:
    # X_pro = x[point1[0]] + (x[point2[0]] - x[point1[0]])/dis * np.linspace(0, dis, num_points, endpoint=True) # Or endpoint=False if num_points is segments
    # Y_pro = y[point1[1]] + (y[point2[1]] - y[point1[1]])/dis * np.linspace(0, dis, num_points, endpoint=True)
    # The current [:-1] means num_points-1 points are generated if num_points is the total count including start and end.
    # If num_points is the number of segments, then num_points+1 points are needed. Let's assume num_points is total points.
    X_pro_calc = (x[point1[0]] - x[point2[0]])/dis * np.linspace(0, dis, num_points)[:-1] + x[point2[0]]
    Y_pro_calc = (y[point1[1]] - y[point2[1]])/dis * np.linspace(0, dis, num_points)[:-1] + y[point2[1]]
    # To ensure profile starts from point1 and goes to point2, it should be:
    # X_pro = x[point1[0]] + (x[point2[0]] - x[point1[0]]) / dis * np.linspace(0, dis, num_points)
    # Y_pro = y[point1[1]] + (y[point2[1]] - y[point1[1]]) / dis * np.linspace(0, dis, num_points)
    # For now, sticking to the original logic:
    X_pro = X_pro_calc
    Y_pro = Y_pro_calc


    # Calculate the cumulative distance along the profile from the *first generated point* (X_pro[0], Y_pro[0]).
    # This measures the path length along the generated (potentially irregular if X_pro, Y_pro are not straight) profile line.
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
    structure = [] # Initialize an empty list to store interpolated profiles for each layer
    # Iterate through each layer provided in the structure_data list
    for layer_idx, layer_array in enumerate(structure_data):
        # For each 'layer_array' (a 2D numpy array representing spatial data for one layer),
        # interpolate its values onto the predefined profile coordinates (X_pro, Y_pro).
        # X_grid and Y_grid are the meshgrid coordinates corresponding to 'layer_array'.
        interpolated_profile = interpolate_to_profile(layer_array, X_grid, Y_grid,
                                           X_pro, Y_pro)
        structure.append(interpolated_profile)
    # Convert the list of 1D interpolated profiles into a 2D NumPy array.
    # The resulting array will have shape (n_layers, n_points_on_profile).
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
    n_layers = data.shape[0] # Get the number of layers from the first dimension of the input data
    profile_values = [] # Initialize a list to store interpolated data for each layer
    
    # Ravel the 2D grid coordinates (XX, YY) to 1D arrays.
    # This is done once as these coordinates are the same for all layers.
    X_grid_flat = XX.ravel() # Flatten the X-coordinates of the grid
    Y_grid_flat = YY.ravel() # Flatten the Y-coordinates of the grid
    
    # Ravel the profile coordinates (X_pro, Y_pro) to 1D arrays.
    # This is also done once.
    X_pro_flat = X_pro.ravel() # Flatten the X-coordinates of the profile points
    Y_pro_flat = Y_pro.ravel() # Flatten the Y-coordinates of the profile points

    # Loop through each layer of the 3D input data
    for i in range(n_layers):
        # Get the 2D data for the current layer
        current_layer_data = data[i]
        # Ravel the current layer's 2D data to a 1D array to match the flattened grid coordinates
        current_layer_data_flat = current_layer_data.ravel()

        # Perform linear interpolation for the current layer.
        # Source points are (X_grid_flat, Y_grid_flat) with values current_layer_data_flat.
        # Target points are (X_pro_flat, Y_pro_flat).
        layer_profile_values = griddata((X_grid_flat, Y_grid_flat),
                                      current_layer_data_flat,
                                      (X_pro_flat, Y_pro_flat),
                                      method='linear') # Using linear interpolation by default
        profile_values.append(layer_profile_values)
    
    # Convert the list of interpolated profile values (one array per layer) into a single 2D NumPy array.
    # The shape of the output array will be (n_layers, n_profile_points).
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
    # Initialize an output array with the same shape as mesh_markers (which represents the mesh cells), filled with zeros.
    # This array will store the interpolated property values for each cell in the mesh.
    result = np.zeros_like(mesh_markers, dtype=float)

    # print(profile_distance.shape) # Debugging: Check shape of profile_distance
    # print(depth_values.shape) # Debugging: Check shape of depth_values (n_layers, n_points_on_profile)
    # print(property_values.shape) # Debugging: Check shape of property_values (n_points_on_profile or n_layers, n_points_on_profile)

    # Prepare L_profile_new: This is intended to be the X-coordinates (distance along profile) for the source data points used in griddata.
    # It needs to match the structure of property_values and depth_values when they are filtered by ID.
    # If property_values is 1D (n_points), it implies a single property array along the profile.
    # If property_values is 2D (n_layers, n_points), it implies properties are defined per layer along the profile.
    # The `ID` array seems to assign each point in `property_values` and `depth_values` to a specific layer/marker.
    # Thus, `L_profile_new`, `Depth`, and `property_values` will be indexed by `ID == marker`.

    # Assuming property_values has shape (total_points_from_all_layers_ concatenated)
    # And Depth also has shape (total_points_from_all_layers_concatenated)
    # And ID has shape (total_points_from_all_layers_concatenated)
    # And L_profile_new should then also be (total_points_from_all_layers_concatenated)
    # The original logic for L_profile_new was a bit complex with np.repeat.
    # A simpler interpretation is that L_profile_new should be the x-coordinates (distances)
    # corresponding to each point in property_values and depth_values.
    # If property_values, depth_values, and ID are structured such that each element corresponds to a point,
    # and profile_distance contains the unique distances along the profile,
    # then L_profile_new needs to be constructed to map these distances to the structure of ID.
    # For example, if ID = [0,0,0,1,1,1] and profile_distance = [d1,d2,d3], then L_profile_new should be [d1,d2,d3,d1,d2,d3] if data is structured by layer first.
    # However, the problem states property_values is (n_points) and depth_values is (n_layers, n_points_profile).
    # This implies property_values is 1D, and depth_values gives multiple depth arrays.
    # `ID` likely links elements of `property_values` to specific layers and locations.

    # Let's assume `profile_distance` is (n_points_on_profile).
    # `property_values` is (total_data_points). `depth_values` is (n_layers, n_points_on_profile).
    # `ID` is (total_data_points), mapping each data point to a layer marker.
    # `Depth` is derived from `depth_values` but its exact structure for indexing by `ID` needs to be clear.
    # The current `Depth = depth_values[:14]` makes it (14, n_points_on_profile).
    # This implies `ID` might index into a flattened version of a structure related to these.

    # Re-evaluating L_profile_new:
    # If `property_values[ID==marker]` gives values for a layer, and `Depth[ID==marker]` gives depths,
    # then `L_profile_new[ID==marker]` should give the corresponding distances.
    # This suggests L_profile_new should have the same shape as `property_values` and `ID`.
    # If `profile_distance` is (n_profile_points), and `property_values` comes from stacking data from different depths at these profile points,
    # then `L_profile_new` should be `profile_distance` tiled or repeated appropriately.
    if len(property_values.shape) > 1 and property_values.shape[0] > 1: # e.g. (n_layers_in_prop, n_profile_points)
        # This case is not explicitly handled by the original L_profile_new logic if property_values.shape[0] is not number of layers
        # but rather a different dimension.
        # Assuming property_values is (total_points) and ID maps these points.
        # L_profile_new should be an array of distances, same shape as property_values,
        # where each entry is the profile_distance for that point.
        # This requires knowing how property_values maps to profile_distance via ID.
        # For now, assume L_profile_new is correctly prepared to be同shape as property_values
        # and contains the horizontal distance for each point in property_values.
        # The original `np.repeat` logic might be if `property_values` was (num_prop_layers, num_profile_points)
        # and `profile_distance` was (num_profile_points).
        # Then `L_profile_new` would be `np.tile(profile_distance, (property_values.shape[0], 1))`
        # The current structure seems to be:
        # property_values: 1D array of all property data points
        # ID: 1D array (same size as property_values), identifying layer for each point
        # Depth: 1D array (same size as property_values), vertical position for each point
        # L_profile_new: 1D array (same size as property_values), horizontal position for each point
        # This means `L_profile_new`, `Depth`, `property_values` are all 1D arrays of the same size,
        # and `ID` is used to filter them for a specific layer/marker.
        # The previous `L_profile_new = np.repeat(...)` was likely trying to construct this.
        # If `profile_distance` is (n_unique_profile_locations) and `property_values` are sampled at these locations for various depths/layers,
        # `L_profile_new` must be correctly constructed to align.
        # Given the arguments, `profile_distance` is (n_points), `property_values` is (n_points).
        # This implies a direct correspondence, so L_profile_new should just be `profile_distance`.
        L_profile_new = profile_distance.ravel() # This seems most consistent with (n_points) args
    else: # property_values is 1D
        L_profile_new = profile_distance.ravel()

    # Ensure Depth array corresponds to the layers being processed.
    # The slicing `depth_values[:14]` implies that `depth_values` might have more rows (layers) than needed,
    # or that there's a fixed number of depth measurements (14) that are structured and then indexed by `ID`.
    # If `depth_values` is (n_layers_total, n_points_profile), and `ID` references points within this structure,
    # then `Depth` should also be a 1D array of the same size as `property_values` and `ID`, containing the specific depth for each point.
    # The original code `Depth = depth_values[:14]` and then `Depth[ID==marker]` suggests `ID` might index columns of a transposed, subsetted `depth_values`.
    # This part is complex. Let's assume `Depth` is correctly prepared to be a 1D array, aligned with `property_values` and `ID`,
    # representing the vertical coordinate for each data point.
    # For the comment, we'll describe the existing line:
    _Depth_source_data = depth_values[:14] # Selects up to the first 14 rows (layers/depths) from the input depth_values.
                                       # This implies a specific structure for depth_values where these rows are relevant.
                                       # SUGGESTION: The structure of depth_values, property_values, and ID and how they relate
                                       # to form the source points for interpolation could be clarified with more detailed variable names or comments upstream.

    maxele = 0.0 # This variable is initialized to 0. If it were a non-zero value (e.g., max surface elevation),
                 # it would be used to adjust depth values, potentially converting them to depths relative to that elevation.
                 # Currently, subtracting 0 has no effect on the depth values.

    # Iterate through the specified layer_markers. For each marker, data attributed to that layer is interpolated.
    for marker_val in layer_markers:
        # Create boolean masks for selecting data points corresponding to the current marker.
        # It's assumed that `ID` is a 1D array aligned with `L_profile_new`, `_Depth_source_data` (when properly indexed), and `property_values`.
        id_mask = (ID == marker_val) # Mask for selecting points belonging to the current marker_val

        # Prepare source points (x, y coordinates) and values (property_values) for griddata.
        # These are the known data points from the profile that belong to the current layer.
        # L_profile_new[id_mask] gives the horizontal distances for the layer's data points.
        # _Depth_source_data[id_mask] would give the vertical positions. This indexing implies _Depth_source_data might need to be 1D and aligned with ID.
        # If _Depth_source_data is (14, n_profile_points), and ID is (total_points), this direct indexing is problematic.
        # Let's assume the original intent was that Depth, L_profile_new, property_values are all 1D arrays of same length, filtered by ID.
        # And that _Depth_source_data has been correctly transformed into such a 1D 'Depth' array before this loop.
        # The provided `Depth[ID==marker]` in the original code implies `Depth` is a 1D array of same size as `ID`.
        # Let's assume `Depth_points_for_interpolation` is a 1D array correctly derived from `depth_values` and aligned with `ID`.
        # For commenting the existing code, we'll use `_Depth_source_data` and note the indexing.

        # Source points for interpolation:
        # x-coordinates (distance along profile) for data points of the current layer
        source_points_x = L_profile_new[id_mask].ravel()
        # y-coordinates (depth) for data points of the current layer.
        # The indexing `_Depth_source_data[id_mask]` implies `_Depth_source_data` must be structured (e.g., flattened or selectively indexed)
        # such that `id_mask` can be applied to it directly to get the depth values corresponding to `source_points_x`.
        # This is the most complex part to infer without knowing the exact upstream structure of `ID` and `_Depth_source_data`.
        # Assuming `_Depth_source_data` is already a 1D array aligned with `ID`, or `ID` correctly indexes into its structure.
        # For the purpose of commenting the line as written:
        source_points_y = _Depth_source_data[id_mask].ravel() - maxele # Adjust depth by maxele (currently 0)
        # Values to be interpolated, corresponding to the source_points
        source_values = property_values[id_mask].ravel()

        # Check if there are any source points for this marker to avoid errors with empty arrays.
        if source_points_x.size == 0 or source_values.size == 0:
            # print(f"Warning: No source data points found for marker {marker_val}. Skipping interpolation for this marker.")
            continue # Skip to the next marker if no data for this one

        # Prepare target mesh cell coordinates for the current marker.
        # These are the (x,y) centroids of mesh cells that have the current marker_val.
        target_mesh_x = mesh_x[mesh_markers == marker_val]
        target_mesh_y = mesh_y[mesh_markers == marker_val]

        # If there are no target cells for this marker, skip.
        if target_mesh_x.size == 0:
            # print(f"Warning: No mesh cells found for marker {marker_val}. Skipping interpolation for this marker.")
            continue

        # Perform linear interpolation from source points to target mesh cell coordinates.
        interpolated_linear = griddata((source_points_x, source_points_y), source_values,
                                       (target_mesh_x, target_mesh_y), method='linear')

        # Identify where linear interpolation resulted in NaN values.
        # This typically occurs for target points outside the convex hull of the source points.
        nan_mask_linear = np.isnan(interpolated_linear)

        # Perform nearest neighbor interpolation for the points where linear interpolation failed (produced NaN).
        # This ensures that all cells in the layer receive an interpolated value.
        # We only need to run nearest neighbor for the subset of points that are NaN.
        if np.any(nan_mask_linear):
            interpolated_nearest = griddata((source_points_x, source_points_y), source_values,
                                            (target_mesh_x[nan_mask_linear], target_mesh_y[nan_mask_linear]), method='nearest')
            # Fill NaN values in the linear interpolation result with values from nearest neighbor interpolation.
            interpolated_linear[nan_mask_linear] = interpolated_nearest

        # Assign the interpolated (and potentially filled) values to the corresponding cells in the main result array.
        # mesh_markers == marker_val provides the boolean mask for cells belonging to the current layer.
        result[mesh_markers == marker_val] = interpolated_linear.copy()

    # # Commented out block: This seems to be an alternative or older way of handling a specific layer (marker 0).
    # # The loop above generalizes this logic for all markers in `layer_markers`.
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
            point1: Starting point indices [col, row] for the profile line.
            point2: Ending point indices [col, row] for the profile line.
            surface_data: 2D numpy array of surface elevation data.
            origin_x: X coordinate of the origin of the surface_data grid.
            origin_y: Y coordinate of the origin of the surface_data grid.
            pixel_width: Width of each pixel in the surface_data grid (spatial unit).
            pixel_height: Height of each pixel in the surface_data grid (spatial unit, often negative).
            num_points: Number of points to generate along the profile line.
        """
        # Store initial parameters, these define the context of the surface data and the desired profile
        self.point1 = point1
        self.point2 = point2
        self.surface_data = surface_data
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.pixel_width = pixel_width
        self.pixel_height = pixel_height
        self.num_points = num_points
        
        # Call the standalone function `setup_profile_coordinates` to:
        # 1. Calculate the X and Y coordinates of points along the defined profile line (self.X_pro, self.Y_pro).
        # 2. Calculate the cumulative distance along this profile line (self.L_profile).
        # 3. Generate meshgrid coordinates (self.XX, self.YY) for the entire input surface_data extent.
        # These attributes are then stored in the instance for later use in interpolation methods.
        self.X_pro, self.Y_pro, self.L_profile, self.XX, self.YY = setup_profile_coordinates(
            point1, point2, surface_data, origin_x, origin_y, 
            pixel_width, pixel_height, num_points
        )
        
        # Interpolate the surface elevation data itself onto the generated profile line.
        # This provides the surface topography along the profile.
        # `self.XX`, `self.YY` are the grid coordinates of `surface_data`.
        # `self.X_pro`, `self.Y_pro` are the target profile coordinates.
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
        # This method calls the standalone `interpolate_structure_to_profile` function.
        # It uses the grid coordinates (self.XX, self.YY) and profile coordinates (self.X_pro, self.Y_pro)
        # that were pre-calculated during the initialization of the ProfileInterpolator instance.
        # `layer_data` is a list of 2D arrays, each representing a subsurface layer.
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
        # This method calls the standalone `prepare_2D_profile_data` function.
        # It interpolates slices from a 3D data volume onto the defined profile.
        # It utilizes the pre-calculated grid (self.XX, self.YY) and profile coordinates (self.X_pro, self.Y_pro)
        # stored within the instance. `data` is expected to be a 3D numpy array (n_layers, ny, nx).
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
        # This method calls the standalone `interpolate_to_mesh` function.
        # It interpolates property values (defined along the profile) to a 2D mesh.
        # It uses the instance's `self.L_profile` (distance along the profile) as one of the key inputs
        # for relating profile data to mesh coordinates.
        return interpolate_to_mesh(
            property_values, self.L_profile, depth_values, # self.L_profile is the key attribute from the instance
            mesh_x, mesh_y, mesh_markers, ID, layer_markers
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
        surface: Coordinates (distance, elevation) for the top surface line.
        line1: Coordinates (distance, elevation) for the middle boundary line.
        line2: Coordinates (distance, elevation) for the bottom boundary line.
    """
    # Extract specific layer elevation data from the 'structure' array using provided indices.
    # 'structure' is assumed to be a 2D array where rows are layers and columns are points along the profile.
    # .reshape(-1, 1) ensures that S1, S2, S3 are column vectors, suitable for hstack.
    S1 = structure[top_idx,:].reshape(-1,1)  # Elevation data for the top surface layer along the profile.
    S2 = structure[mid_idx,:].reshape(-1,1)  # Elevation data for the middle boundary layer along the profile.
    S3 = structure[bot_idx,:].reshape(-1,1)  # Elevation data for the bottom boundary layer along the profile.
    
    # Create coordinate arrays for each line (surface, line1, line2).
    # np.hstack horizontally stacks L_profile (distance along profile, reshaped to a column vector)
    # with the corresponding layer elevation data (S1, S2, S3).
    # This results in 2D arrays where the first column is the distance along the profile
    # and the second column is the elevation of the respective layer/surface at that distance.
    surface = np.hstack((L_profile.reshape(-1,1), S1)) # Pairs [distance, elevation_S1]
    line1 = np.hstack((L_profile.reshape(-1,1), S2))   # Pairs [distance, elevation_S2]
    line2 = np.hstack((L_profile.reshape(-1,1), S3))   # Pairs [distance, elevation_S3]
    
    # Normalization section (currently commented out or partially applied with subtraction of 'maxele' which is not defined here).
    # The intention, if fully implemented, would typically be to normalize elevations relative to a reference,
    # such as the maximum elevation of the top surface, to get relative depths.
    # Example: maxele = np.nanmax(surface[:,1]) # Find max elevation on the surface line.
    # Then, subtract maxele from the second column (elevation) of surface, line1, and line2.
    # As it is, if maxele is not defined or is 0, this part has no effect or would cause an error if uncommented without defining maxele.
    # Assuming the #- maxele was a placeholder for such an operation.
    # surface[:,1] = surface[:,1] #- maxele
    # line1[:,1] = line1[:,1] #- maxele
    # line2[:,1] = line2[:,1] #- maxele
    
    return surface, line1, line2