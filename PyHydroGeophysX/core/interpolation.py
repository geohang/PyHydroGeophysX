"""
Interpolation utilities for geophysical data processing.

This module provides functions and a class for interpolating geophysical data,
primarily for creating 2D profiles from 3D datasets and for interpolating
profile data onto a 2D mesh.
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

    This function takes 2D gridded data and interpolates it to a series of
    points defining a profile line. It uses `scipy.interpolate.griddata`
    for the interpolation.

    Args:
        data: 2D numpy array of values to interpolate.
        X_grid: 2D numpy array of X coordinates of the original grid (e.g., from np.meshgrid).
        Y_grid: 2D numpy array of Y coordinates of the original grid (e.g., from np.meshgrid).
        X_pro: 1D numpy array of X coordinates of the profile points.
        Y_pro: 1D numpy array of Y coordinates of the profile points.
        method: Interpolation method to use. Options are 'linear', 'nearest', 'cubic'.
                Defaults to 'linear'.

    Returns:
        1D numpy array of interpolated values along the profile.
        Returns an array of NaNs if interpolation fails for some points.
    """
    # Ravel the grid coordinates and data to 1D arrays for griddata
    X_new = X_grid.ravel()
    Y_new = Y_grid.ravel()
    data_ravel = np.array(data).ravel() # Ensure data is a numpy array

    # Ravel profile coordinates
    X_pro_ravel = np.array(X_pro).ravel()
    Y_pro_ravel = np.array(Y_pro).ravel()

    # Perform interpolation
    # Potential issue: If X_pro, Y_pro are outside the convex hull of X_grid, Y_grid,
    # 'linear' and 'cubic' methods will result in NaNs for those points.
    # 'nearest' will extrapolate, which might be desired or not depending on the use case.
    interpolated_values = griddata((X_new, Y_new), data_ravel,
                                   (X_pro_ravel, Y_pro_ravel),
                                   method=method)
    return interpolated_values


def setup_profile_coordinates(point1: List[int],
                            point2: List[int],
                            surface_data: np.ndarray,
                            origin_x: float = 0.0,
                            origin_y: float = 0.0,
                            pixel_width: float = 1.0,
                            pixel_height: float = -1.0,
                            num_points: int = 200) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Set up profile coordinates based on surface elevation data between two points.

    This function defines a profile line between two points (specified by their
    indices in the `surface_data` grid) and generates coordinates along this
    profile. It also returns the meshgrid coordinates for the entire surface.

    Args:
        point1: List or tuple of [column_index, row_index] for the starting point of the profile.
        point2: List or tuple of [column_index, row_index] for the ending point of the profile.
        surface_data: 2D numpy array of surface elevation data. Used to determine grid dimensions.
        origin_x: X coordinate of the origin (bottom-left corner) of the grid. Defaults to 0.0.
        origin_y: Y coordinate of the origin (bottom-left corner) of the grid. Defaults to 0.0.
        pixel_width: Width of each pixel/cell in the grid. Defaults to 1.0.
        pixel_height: Height of each pixel/cell in the grid.
                      Defaults to -1.0 (assuming image-like coordinates where Y increases downwards).
                      If Y increases upwards, this should be positive.
        num_points: Number of points to generate along the profile line. Defaults to 200.

    Returns:
        Tuple containing:
            - X_pro (np.ndarray): 1D array of X coordinates along the profile.
            - Y_pro (np.ndarray): 1D array of Y coordinates along the profile.
            - L_profile (np.ndarray): 1D array of distances along the profile from the start point.
            - XX (np.ndarray): 2D array of X coordinates for the entire grid (meshgrid).
            - YY (np.ndarray): 2D array of Y coordinates for the entire grid (meshgrid).
    """
    # Create 1D coordinate arrays for the grid axes
    # Potential improvement: Ensure surface_data.shape[1] and surface_data.shape[0] are at least 1
    # to avoid issues with np.arange if a dimension is 0.
    x = origin_x + pixel_width * np.arange(surface_data.shape[1])
    y = origin_y + pixel_height * np.arange(surface_data.shape[0])
    XX, YY = np.meshgrid(x, y) # Create 2D coordinate grids

    # Handle no-data values in surface_data if necessary (currently not used by this function
    # but good practice if surface_data itself were being interpolated here).
    # surface_data_copy = surface_data.copy()
    # surface_data_copy[surface_data_copy == 0] = np.nan # Assuming 0 represents no-data

    # Calculate the actual Cartesian coordinates of the start and end points of the profile
    # Potential issue: point1 and point2 indices should be validated to be within the bounds of x and y arrays.
    # For example, point1[0] < len(x) and point1[1] < len(y).
    try:
        P1_pos = np.array([x[point1[0]], y[point1[1]]])
        P2_pos = np.array([x[point2[0]], y[point2[1]]])
    except IndexError:
        # Potential bug: If point indices are out of bounds, this will raise an IndexError.
        # Consider adding error handling or pre-validation.
        raise ValueError("point1 or point2 indices are out of bounds for the given surface_data dimensions.")

    # Calculate the total Euclidean distance between P1 and P2
    dis = np.sqrt(np.sum((P1_pos - P2_pos)**2))
    if dis == 0:
        # Handle the case where point1 and point2 are the same
        # Potential issue: if dis is zero, division by zero will occur.
        # If num_points is 1, linspace(0,0,1) is [0.]. If num_points > 1, linspace(0,0,N) is [0., 0., ...].
        # The `[:-1]` slice might lead to an empty array if num_points is 1.
        if num_points == 1:
            X_pro = np.array([x[point1[0]]])
            Y_pro = np.array([y[point1[1]]])
        else: # Create num_points identical points
            X_pro = np.full(num_points, x[point1[0]])
            Y_pro = np.full(num_points, y[point1[1]])
    else:
        # Generate evenly spaced points along the profile line
        # The `[:-1]` at the end of linspace was likely a bug or specific requirement.
        # Typically, you'd want all `num_points`. If `num_points` truly means segments,
        # then `num_points+1` points are needed. Assuming `num_points` means points.
        # Corrected to generate `num_points` along the line.
        # Original code: `np.linspace(0, dis, num_points)[:-1]`
        # This would generate `num_points-1` points if num_points > 0.
        # If the intention was to exclude the end point for some reason, this should be documented.
        # Assuming we want `num_points` including start and end:
        line_space = np.linspace(0, dis, num_points)
        X_pro = (x[point1[0]] - x[point2[0]]) / dis * line_space + x[point2[0]]
        Y_pro = (y[point1[1]] - y[point2[1]]) / dis * line_space + y[point2[1]]
        # The original formulation for X_pro and Y_pro was a bit unusual.
        # A more standard way to parameterize a line segment from P2 to P1:
        # P(t) = P2 + t * (P1 - P2) for t in [0, 1]
        # X_pro = P2_pos[0] + (P1_pos[0] - P2_pos[0]) * np.linspace(0, 1, num_points)
        # Y_pro = P2_pos[1] + (P1_pos[1] - P2_pos[1]) * np.linspace(0, 1, num_points)
        # The original code seems to go from P1 towards P2, but the factors are from P1.
        # Let's re-verify the original logic:
        # X_pro = x_P1 - (x_P1 - x_P2)/dis * linspace_val
        # Y_pro = y_P1 - (y_P1 - y_P2)/dis * linspace_val
        # This means at linspace_val = 0, X_pro = x_P1. At linspace_val = dis, X_pro = x_P2.
        # So the profile goes from P1 to P2.
        # The original calculation:
        # X_pro = (x[point1[0]] - x[point2[0]])/dis * np.linspace(0, dis, num_points)[:-1] + x[point2[0]]
        # If num_points = 1, linspace(0,dis,1)[:-1] is empty. This is a bug.
        # If num_points = 0, linspace(0,dis,0) is empty. Also problematic.
        # Let's assume num_points >= 1.
        # If num_points = 1, it should return just point1.
        if num_points == 0:
             X_pro = np.array([])
             Y_pro = np.array([])
        elif num_points == 1:
             X_pro = np.array([x[point1[0]]])
             Y_pro = np.array([y[point1[1]]])
        else:
            # The original code `... * np.linspace(0, dis, num_points)[:-1] + x[point2[0]]`
            # seems to intend to generate points from point1 towards point2, but the last point is excluded.
            # And the reference point added is x[point2[0]]. This logic is confusing.
            # Let P_start = (x[point1[0]], y[point1[1]]) and P_end = (x[point2[0]], y[point2[1]])
            # Vector V = P_end - P_start
            # Points X_p = x_P_start + t * V_x, Y_p = y_P_start + t * V_y, where t from 0 to 1 over num_points steps
            t = np.linspace(0, 1, num_points)
            X_pro = x[point1[0]] + t * (x[point2[0]] - x[point1[0]])
            Y_pro = y[point1[1]] + t * (y[point2[1]] - y[point1[1]])


    # Calculate cumulative distances along the profile from the starting point
    if num_points > 0:
        L_profile = np.sqrt((X_pro - X_pro[0])**2 + (Y_pro - Y_pro[0])**2)
    else:
        L_profile = np.array([])

    return X_pro, Y_pro, L_profile, XX, YY


def interpolate_structure_to_profile(structure_data: List[np.ndarray],
                                   X_grid: np.ndarray,
                                   Y_grid: np.ndarray,
                                   X_pro: np.ndarray,
                                   Y_pro: np.ndarray,
                                   method: str = 'linear') -> np.ndarray:
    """
    Interpolate multiple structure layers (2D grids) onto a defined profile.

    This function iterates through a list of 2D arrays, each representing a
    geological layer or structure, and interpolates each onto the given profile
    coordinates.

    Args:
        structure_data: List of 2D numpy arrays. Each array is a grid of values
                        representing a layer.
        X_grid: 2D numpy array of X coordinates of the original grid (meshgrid).
        Y_grid: 2D numpy array of Y coordinates of the original grid (meshgrid).
        X_pro: 1D numpy array of X coordinates of the profile points.
        Y_pro: 1D numpy array of Y coordinates of the profile points.
        method: Interpolation method to use (e.g., 'linear', 'nearest').
                Passed to `interpolate_to_profile`. Defaults to 'linear'.

    Returns:
        2D numpy array of interpolated values. Shape is (n_layers, n_profile_points).
    """
    structure_on_profile = []
    for layer_data in structure_data:
        # For each layer, interpolate its data onto the profile
        interpolated_layer = interpolate_to_profile(layer_data, X_grid, Y_grid,
                                                    X_pro, Y_pro, method=method)
        structure_on_profile.append(interpolated_layer)
    return np.array(structure_on_profile)


def prepare_2D_profile_data(data: np.ndarray,
                          XX: np.ndarray,
                          YY: np.ndarray,
                          X_pro: np.ndarray,
                          Y_pro: np.ndarray,
                          method: str = 'linear') -> np.ndarray:
    """
    Interpolate multiple 2D gridded data layers (from a 3D array) onto a profile line.

    This function is similar to `interpolate_structure_to_profile` but takes a single
    3D numpy array where the first dimension represents different layers.

    Args:
        data: 3D numpy array of gridded data with shape (n_layers, ny, nx).
        XX: 2D numpy array of X coordinates of the original grid (meshgrid from input grid dimensions).
        YY: 2D numpy array of Y coordinates of the original grid (meshgrid from input grid dimensions).
        X_pro: 1D numpy array of X coordinates of the profile points.
        Y_pro: 1D numpy array of Y coordinates of the profile points.
        method: Interpolation method to use for `griddata`. Defaults to 'linear'.

    Returns:
        2D numpy array of interpolated values along the profile.
        Shape: (n_layers, n_profile_points).
    """
    if data.ndim != 3:
        raise ValueError(f"Input 'data' must be a 3D array (n_layers, ny, nx), got {data.ndim} dimensions.")
    if XX.ndim != 2 or YY.ndim != 2:
        raise ValueError(f"Input 'XX' and 'YY' must be 2D arrays, got {XX.ndim} and {YY.ndim} dimensions.")
    if X_pro.ndim != 1 or Y_pro.ndim != 1:
        raise ValueError(f"Input 'X_pro' and 'Y_pro' must be 1D arrays, got {X_pro.ndim} and {Y_pro.ndim} dimensions.")
    if X_pro.shape[0] != Y_pro.shape[0]:
        raise ValueError(f"X_pro and Y_pro must have the same number of points, got {X_pro.shape[0]} and {Y_pro.shape[0]}.")


    n_layers = data.shape[0]
    profile_values = []

    # Ravel the grid coordinates once
    X_grid_ravel = XX.ravel()
    Y_grid_ravel = YY.ravel()

    # Ravel profile coordinates once
    X_pro_ravel = X_pro.ravel() # Should already be 1D, but ravel handles it.
    Y_pro_ravel = Y_pro.ravel()

    for i in range(n_layers):
        # Extract the current layer's data and ravel it
        layer_data_ravel = data[i].ravel()
        # Perform interpolation for the current layer
        # Potential issue: Behavior for points outside convex hull (NaNs for 'linear'/'cubic').
        layer_interpolated_values = griddata((X_grid_ravel, Y_grid_ravel),
                                             layer_data_ravel,
                                             (X_pro_ravel, Y_pro_ravel),
                                             method=method)
        profile_values.append(layer_interpolated_values)

    return np.array(profile_values)


def interpolate_to_mesh(property_values: np.ndarray,
                       profile_distance: np.ndarray,
                       depth_values: np.ndarray,
                       mesh_x: np.ndarray,
                       mesh_y: np.ndarray,
                       mesh_markers: np.ndarray,
                       ID: np.ndarray, # Changed from any to np.ndarray for clarity
                       layer_markers: Optional[List[int]] = None) -> np.ndarray: # Default None
    """
    Interpolate property values from a 2D profile (defined by distance and depth)
    onto a 2D unstructured mesh. This function handles layer-specific interpolation
    based on markers.

    The input data `property_values`, `profile_distance`, and `depth_values`
    are expected to be structured in a way that they can be filtered by `ID`
    which corresponds to `layer_markers` in the mesh.

    Args:
        property_values: 1D or 2D numpy array of property values.
                         If 2D, assumed shape (n_layers_in_profile_data, n_points_along_profile).
                         If 1D, it's treated as a single layer or must be indexable by `ID`.
        profile_distance: 1D numpy array of distances along the profile.
                          It's repeated to match dimensions if `property_values` is 2D.
        depth_values: 2D numpy array of depth values, typically (n_layers_in_profile_data, n_points_along_profile).
                      The slice `depth_values[:14]` suggests a fixed number of depth layers expected.
                      This should be clarified or made more flexible.
        mesh_x: 1D numpy array of X coordinates of the mesh cell centers.
        mesh_y: 1D numpy array of Y coordinates of the mesh cell centers (typically depths).
        mesh_markers: 1D numpy array of integer markers for each mesh cell, indicating its layer/region.
        ID: 1D or 2D numpy array used for filtering `property_values`, `profile_distance`, and `depth_values`
            to match specific `layer_markers`. Its shape should be compatible with `property_values`.
            This `ID` array seems to map points in `property_values` to specific layers.
        layer_markers: List of integer marker values present in `mesh_markers` for which
                       interpolation should be performed. If None, it defaults to [3, 0, 2].
                       It's recommended to derive this from `np.unique(mesh_markers)` if appropriate.

    Returns:
        1D numpy array of interpolated values for each mesh cell, preserving the order of `mesh_x`/`mesh_y`.
    """
    if layer_markers is None:
        layer_markers = [3, 0, 2] # Default layer markers

    # Initialize output array for mesh cell properties
    result = np.zeros_like(mesh_markers, dtype=float) # mesh_markers defines the shape of the mesh

    # Prepare L_profile_new:
    # This repeats profile_distance for each "layer" in property_values.
    # This assumes property_values might be (n_prop_layers, n_profile_points).
    # If property_values is 1D, property_values.shape[0] would be the number of points,
    # which makes np.repeat act differently. This logic seems fragile.
    # Let's assume property_values is (n_data_layers, n_profile_points)
    # and profile_distance is (n_profile_points).
    if property_values.ndim == 1:
        # If property_values is 1D, it implies a single set of values along the profile.
        # ID must then be 1D and of the same length.
        if ID.ndim != 1 or ID.shape[0] != property_values.shape[0]:
            raise ValueError("If property_values is 1D, ID must be 1D and match its length.")
        L_profile_new = profile_distance # No repeat needed, will be filtered by ID directly
        Depth_for_interp = depth_values # Assumes depth_values is (n_data_layers, n_profile_points)
                                        # or compatible with ID indexing.
        prop_values_for_interp = property_values
    elif property_values.ndim == 2:
        if ID.ndim != 2 or ID.shape != property_values.shape:
             raise ValueError("If property_values is 2D, ID must be 2D and match its shape.")
        # This repeats each profile_distance value for property_values.shape[0] times.
        # Example: if profile_distance is (100,) and property_values is (3, 100),
        # L_profile_new becomes (3, 100) where each row is profile_distance.
        L_profile_new = np.repeat(profile_distance.reshape(1, -1), property_values.shape[0], axis=0)
        Depth_for_interp = depth_values # Assumed (n_data_layers, n_profile_points)
        prop_values_for_interp = property_values
    else:
        raise ValueError(f"property_values must be 1D or 2D, got {property_values.ndim} dimensions.")


    # Potential issue: Hardcoded slice `depth_values[:14]`.
    # This implies `depth_values` always has at least 14 layers, or that only these are relevant.
    # This should be a parameter or derived. For now, let's call it `Depth_subset`.
    # Depth = depth_values[:14] # Original line. This is risky.
    # Assuming Depth_for_interp is correctly shaped now.
    # If depth_values itself is supposed to be (num_actual_depth_layers, n_profile_points),
    # then the slicing by ID should handle which depths are used.
    # The `[:14]` might be an artifact or a specific domain constraint not generalized.

    maxele = 0  # This variable is named 'maxele' (max elevation?) but set to 0.
                # If this is for normalization, it should be calculated or passed.
                # Comment says "set 0 here", implying it's intentional for this context.

    for marker_val in layer_markers:
        # Create a boolean mask for the current layer in the mesh
        mesh_layer_mask = (mesh_markers == marker_val)
        if not np.any(mesh_layer_mask):
            # print(f"Warning: No mesh cells found for marker {marker_val}. Skipping.")
            continue

        # Create a boolean mask for the current layer in the profile data using ID
        # ID should have the same shape as property_values and depth_values for this to work element-wise.
        profile_data_mask = (ID == marker_val)
        if not np.any(profile_data_mask):
            # print(f"Warning: No profile data found for ID {marker_val} corresponding to mesh marker. Skipping interpolation for this marker.")
            continue

        # Select data points for interpolation:
        # Source points: (L_profile_new[profile_data_mask].ravel(), Depth_for_interp[profile_data_mask].ravel() - maxele)
        # Source values: prop_values_for_interp[profile_data_mask].ravel()
        # Target points: (mesh_x[mesh_layer_mask], mesh_y[mesh_layer_mask])

        source_x = L_profile_new[profile_data_mask].ravel()
        source_y = Depth_for_interp[profile_data_mask].ravel() - maxele # Apply maxele offset
        source_vals = prop_values_for_interp[profile_data_mask].ravel()

        if source_x.size == 0 or source_vals.size == 0 :
            # print(f"Warning: No valid source data points for marker {marker_val} after filtering. Skipping.")
            # result[mesh_layer_mask] will remain 0 or NaN if initialized that way.
            # Or fill with a specific no-data value.
            result[mesh_layer_mask] = np.nan # Or some other fill_value
            continue
        
        target_x = mesh_x[mesh_layer_mask]
        target_y = mesh_y[mesh_layer_mask]

        # Perform linear interpolation
        # Potential issue: If source points are collinear or too few, griddata can be slow or fail.
        interpolated_linear = griddata(
            (source_x, source_y), source_vals,
            (target_x, target_y), method='linear'
        )

        # Identify NaNs from linear interpolation (points outside convex hull)
        nan_mask = np.isnan(interpolated_linear)
        if np.any(nan_mask):
            # Perform nearest neighbor interpolation for points where linear failed
            # Potential improvement: Check if source_x for nearest is non-empty.
            interpolated_nearest = griddata(
                (source_x, source_y), source_vals,
                (target_x[nan_mask], target_y[nan_mask]), method='nearest'
            )
            interpolated_linear[nan_mask] = interpolated_nearest
        
        # Assign interpolated values to the corresponding part of the result array
        result[mesh_layer_mask] = interpolated_linear

    # Commented out sections from original code:
    # The original code had a loop and then a specific block for marker 0.
    # The loop should handle all markers in layer_markers. If marker 0 needs
    # special handling, it should be explicit. The current loop is generic.
    # Example of the commented block:
    # # grid_z1 = griddata((L_profile_new[ID==0]..., Depth[ID==0]...), property_values[ID==0]..., (mesh_x[mesh_markers==0]...), method='linear')
    # # ...
    # # result[mesh_markers==0] = grid_z1.copy()

    # The final commented line seems like an attempt at global interpolation without layers:
    # #result =  griddata((L_profile_new.ravel(),depth_values[:14].ravel()), property_values.ravel(), (mesh_x, mesh_y), method='nearest')
    # This would ignore the layer_markers logic.

    return result


class ProfileInterpolator:
    """
    A class to simplify the process of interpolating various types of geophysical data
    onto a defined 2D profile.

    The class handles the setup of profile coordinates and provides methods
    to interpolate surface data, layered structures, and 3D gridded data onto
    this profile. It can also interpolate from the profile data to a 2D mesh.
    """

    def __init__(self, point1: List[int], point2: List[int],
                surface_data: np.ndarray,
                origin_x: float = 0.0, origin_y: float = 0.0,
                pixel_width: float = 1.0, pixel_height: float = -1.0, # Typically negative if Y is row index
                num_points: int = 200):
        """
        Initialize the ProfileInterpolator.

        This involves setting up the profile coordinates based on two points
        on a surface grid and calculating the surface elevation along this profile.

        Args:
            point1: List or tuple [col_idx, row_idx] for the starting grid cell of the profile.
            point2: List or tuple [col_idx, row_idx] for the ending grid cell of the profile.
            surface_data: 2D numpy array representing surface elevation or another key surface.
                          Used for defining grid dimensions and for optional surface profile interpolation.
            origin_x: X-coordinate of the grid's origin (e.g., bottom-left corner).
            origin_y: Y-coordinate of the grid's origin.
            pixel_width: The width of a single cell in the grid in Cartesian units.
            pixel_height: The height of a single cell in the grid in Cartesian units.
                          Often negative if row index increases downwards but Y-coordinate increases upwards.
            num_points: Number of points to discretize the profile into.
        """
        self.point1 = point1
        self.point2 = point2
        # Store parameters
        self.surface_data_grid = surface_data # Store the original grid data
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.pixel_width = pixel_width
        self.pixel_height = pixel_height
        self.num_points = num_points

        # Set up profile coordinates (X, Y positions and distance along profile)
        # Also get the meshgrid (XX, YY) for the entire input surface_data domain
        self.X_pro, self.Y_pro, self.L_profile, self.XX, self.YY = setup_profile_coordinates(
            point1, point2, self.surface_data_grid,
            origin_x, origin_y,
            pixel_width, pixel_height, num_points
        )

        # Interpolate the provided surface_data onto the profile line
        # This gives the elevation (or other gridded value) along the profile path.
        # Potential issue: if X_pro, Y_pro are empty (e.g. num_points=0), this might fail.
        # setup_profile_coordinates should handle num_points=0 returning empty arrays.
        # interpolate_to_profile should ideally handle empty X_pro, Y_pro gracefully.
        if self.X_pro.size > 0 and self.Y_pro.size > 0 :
            self.surface_profile = interpolate_to_profile(
                self.surface_data_grid, self.XX, self.YY, self.X_pro, self.Y_pro, method='linear' # Default linear
            )
        else:
            self.surface_profile = np.array([]) # Empty profile if no points

    def interpolate_layer_data(self, layer_data: List[np.ndarray], method: str = 'linear') -> np.ndarray:
        """
        Interpolate multiple layers of 2D gridded data onto the defined profile.

        Each layer is a 2D numpy array with the same dimensions as the initial
        `surface_data` grid.

        Args:
            layer_data: A list of 2D numpy arrays. Each array represents a layer's
                        data on the original grid.
            method: Interpolation method ('linear', 'nearest', 'cubic') to be used by
                    `interpolate_structure_to_profile`. Defaults to 'linear'.

        Returns:
            A 2D numpy array containing the interpolated values for each layer along
            the profile. Shape: (n_layers, n_profile_points).
        """
        if not self.X_pro.size > 0: # Or check self.num_points > 0
            return np.array([]).reshape(len(layer_data), 0) # Return empty array with correct first dim

        return interpolate_structure_to_profile(
            layer_data, self.XX, self.YY, self.X_pro, self.Y_pro, method=method
        )

    def interpolate_3d_data(self, data_3d: np.ndarray, method: str = 'linear') -> np.ndarray:
        """
        Interpolate 3D gridded data onto the defined profile.

        The 3D data is expected to be an array of shape (n_layers, ny, nx),
        where (ny, nx) are the dimensions of the original `surface_data` grid.

        Args:
            data_3d: A 3D numpy array of data to interpolate.
                     Shape: (n_layers, ny, nx).
            method: Interpolation method ('linear', 'nearest', 'cubic') to be used by
                    `prepare_2D_profile_data`. Defaults to 'linear'.


        Returns:
            A 2D numpy array containing the interpolated values for each layer of
            the 3D data along the profile. Shape: (n_layers, n_profile_points).
        """
        if not self.X_pro.size > 0: # Or check self.num_points > 0
             # Return empty array with correct first dimension if data_3d is not empty
            return np.array([]).reshape(data_3d.shape[0] if data_3d.ndim == 3 and data_3d.shape[0] > 0 else 0, 0)


        return prepare_2D_profile_data(
            data_3d, self.XX, self.YY, self.X_pro, self.Y_pro, method=method
        )

    def interpolate_to_mesh(self,
                          property_values: np.ndarray,
                          depth_values: np.ndarray, # Depths corresponding to property_values along profile
                          mesh_x: np.ndarray,       # Mesh cell X coordinates
                          mesh_y: np.ndarray,       # Mesh cell Y coordinates (depths)
                          mesh_markers: np.ndarray, # Mesh cell layer markers
                          ID: np.ndarray,           # Profile data layer/ID markers
                          layer_markers: Optional[List[int]] = None) -> np.ndarray:
        """
        Interpolate property values from the profile representation (distance vs. depth)
        to a 2D unstructured mesh, using layer-specific interpolation.

        Args:
            property_values: 1D or 2D numpy array of property values along the profile.
                             These values are associated with `self.L_profile` (distance)
                             and `depth_values`.
            depth_values: 2D numpy array (n_profile_layers, n_profile_points) of depth coordinates
                          for the `property_values` along the profile.
            mesh_x: 1D numpy array of X-coordinates for mesh cell centers.
                    For profile-to-mesh, this is often equivalent to distance along profile.
            mesh_y: 1D numpy array of Y-coordinates (depths) for mesh cell centers.
            mesh_markers: 1D numpy array of integer markers for each mesh cell,
                          indicating its geological layer or region.
            ID: 1D or 2D numpy array that maps points in `property_values` and `depth_values`
                to specific layers, aligning them with `layer_markers` in the mesh.
                Shape should be compatible with `property_values`.
            layer_markers: Optional list of integer marker values in `mesh_markers`
                           to perform interpolation for. Defaults to [3, 0, 2] if None.

        Returns:
            1D numpy array of interpolated property values for each cell in the mesh.
        """
        if not self.L_profile.size > 0:
            # If profile has no points, cannot interpolate to mesh based on it.
            # Return array of NaNs or zeros matching mesh size.
            return np.full_like(mesh_x, np.nan, dtype=float)

        return interpolate_to_mesh(
            property_values, self.L_profile, depth_values,
            mesh_x, mesh_y, mesh_markers, ID, layer_markers=layer_markers
        )


def create_surface_lines(L_profile: np.ndarray,
                        structure_on_profile: np.ndarray,
                        top_idx: int = 0,
                        mid_idx: int = 4,
                        bot_idx: int = 12) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract specific layer boundaries from interpolated structure data along a profile
    to create 2D lines (e.g., for plotting or mesh generation).

    The function takes structure data (already interpolated onto a profile) and
    extracts rows specified by indices to represent different geological surfaces
    or boundaries. These are returned as (N, 2) arrays where N is the number
    of points along the profile, column 0 is distance along profile, and column 1 is elevation/depth.

    Args:
        L_profile: 1D numpy array of distances along the profile.
        structure_on_profile: 2D numpy array (n_layers, n_profile_points) of interpolated
                              values (e.g., elevations or depths) for each layer
                              along the profile.
        top_idx: Index of the row in `structure_on_profile` that represents the top surface.
        mid_idx: Index of the row for a middle boundary/surface.
        bot_idx: Index of the row for a bottom boundary/surface.

    Returns:
        Tuple containing three (N, 2) numpy arrays:
            - surface_line: Coordinates [L_profile, S1_elevations] for the top surface.
            - boundary_line1: Coordinates [L_profile, S2_elevations] for the middle boundary.
            - boundary_line2: Coordinates [L_profile, S3_elevations] for the bottom boundary.
    """
    # Potential issue: Assumes structure_on_profile has enough layers to satisfy indices.
    # Add checks for indices bounds.
    num_layers, num_points = structure_on_profile.shape
    if not (0 <= top_idx < num_layers and 0 <= mid_idx < num_layers and 0 <= bot_idx < num_layers):
        raise IndexError(f"One or more indices (top_idx={top_idx}, mid_idx={mid_idx}, bot_idx={bot_idx}) "
                         f"are out of bounds for structure_on_profile with {num_layers} layers.")

    if L_profile.shape[0] != num_points:
        raise ValueError(f"L_profile length ({L_profile.shape[0]}) must match "
                         f"number of points in structure_on_profile ({num_points}).")

    # Extract specified layers (rows) from the structure data
    S1_elevations = structure_on_profile[top_idx, :]
    S2_elevations = structure_on_profile[mid_idx, :]
    S3_elevations = structure_on_profile[bot_idx, :]

    # Reshape L_profile to (N,1) for hstack
    L_profile_col = L_profile.reshape(-1, 1)

    # Create coordinate arrays for each line: [distance_along_profile, elevation/depth]
    surface_line = np.hstack((L_profile_col, S1_elevations.reshape(-1, 1)))
    boundary_line1 = np.hstack((L_profile_col, S2_elevations.reshape(-1, 1)))
    boundary_line2 = np.hstack((L_profile_col, S3_elevations.reshape(-1, 1)))

    # Normalization section (commented out in original):
    # The original code includes a commented-out normalization step.
    # If normalization is intended, `maxele` should be clearly defined (e.g., np.nanmax(S1_elevations)).
    # Example:
    # max_elevation = np.nanmax(surface_line[:, 1]) # Or from a global reference
    # surface_line[:, 1] -= max_elevation
    # boundary_line1[:, 1] -= max_elevation
    # boundary_line2[:, 1] -= max_elevation
    # This would shift all elevations relative to this max_elevation.
    # The current code does not apply any normalization as `maxele` was set to 0 and then lines commented.

    return surface_line, boundary_line1, boundary_line2