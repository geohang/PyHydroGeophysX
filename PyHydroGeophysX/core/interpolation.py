"""
Interpolation utilities for geophysical data processing.
"""
from typing import Any, List, Optional, Tuple, Union

import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import map_coordinates


# ---------------------------------------------------------------------------
# fast regular-grid sampling helpers
# ---------------------------------------------------------------------------
def _detect_regular_grid(X_grid: np.ndarray, Y_grid: np.ndarray):
    """Return ``(x0, dx, y0, dy)`` if (X_grid, Y_grid) is a uniform meshgrid.

    Returns ``None`` for any non-uniform / irregular grid so callers can fall
    back to general scattered interpolation.
    """
    try:
        X_grid = np.asarray(X_grid, dtype=float)
        Y_grid = np.asarray(Y_grid, dtype=float)
        if X_grid.ndim != 2 or Y_grid.ndim != 2 or X_grid.shape != Y_grid.shape:
            return None
        x = X_grid[0, :]
        y = Y_grid[:, 0]
        if x.size < 2 or y.size < 2:
            return None
        if not (np.allclose(X_grid, x[np.newaxis, :]) and np.allclose(Y_grid, y[:, np.newaxis])):
            return None
        dx = np.diff(x)
        dy = np.diff(y)
        if not (np.allclose(dx, dx[0]) and np.allclose(dy, dy[0])):
            return None
        if dx[0] == 0 or dy[0] == 0:
            return None
        return float(x[0]), float(dx[0]), float(y[0]), float(dy[0])
    except Exception:
        return None


def _sample_grid_along_profile(data: np.ndarray,
                               X_grid: np.ndarray,
                               Y_grid: np.ndarray,
                               X_pro: np.ndarray,
                               Y_pro: np.ndarray,
                               method: str = 'linear') -> np.ndarray:
    """Sample ``data`` along a profile line.

    When (X_grid, Y_grid) form a uniform regular grid -- the usual case for data
    produced by :func:`setup_profile_coordinates` -- this uses fast
    bilinear/nearest :func:`scipy.ndimage.map_coordinates` sampling, which is
    avoids constructing a Delaunay triangulation. Bilinear interpolation need
    not equal triangulation-based linear interpolation. For irregular grids it
    falls back to ``griddata`` so the function remains general.
    """
    x_pro = np.asarray(X_pro, dtype=float).ravel()
    y_pro = np.asarray(Y_pro, dtype=float).ravel()
    grid = _detect_regular_grid(X_grid, Y_grid)
    if grid is None:
        return griddata((np.asarray(X_grid).ravel(), np.asarray(Y_grid).ravel()),
                        np.asarray(data, dtype=float).ravel(),
                        (x_pro, y_pro), method=method)
    x0, dx, y0, dy = grid
    cols = (x_pro - x0) / dx
    rows = (y_pro - y0) / dy
    data_arr = np.asarray(data, dtype=float)
    if method == 'nearest':
        return map_coordinates(data_arr, [rows, cols], order=0, mode='nearest')
    # 'linear' (and anything else): bilinear, NaN outside the grid to mirror the
    # convex-hull behavior of griddata on a regular grid.
    return map_coordinates(data_arr, [rows, cols], order=1, mode='constant', cval=np.nan)


# ---------------------------------------------------------------------------
# interpolate to profile
# ---------------------------------------------------------------------------
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
    
    return _sample_grid_along_profile(data, X_grid, Y_grid, X_pro, Y_pro, method=method)


# ---------------------------------------------------------------------------
# setup profile coordinates
# ---------------------------------------------------------------------------
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

    The profile starts at ``point2`` (distance zero) and runs toward ``point1``,
    stopping one step short of it: ``num_points - 1`` samples, spaced
    ``distance / (num_points - 1)`` apart. The examples, and the data bundled
    with them (``TL_measurements``, ``Seismic``), were built on this layout, and
    ``create_mesh_from_layers`` places each layer's region marker from the first
    node of the profile, so a profile run the other way puts the regolith and
    bedrock markers in each other's layers.

    Args:
        point1: End point indices [col, row]; the profile stops one step before it
        point2: Start point indices [col, row]; distance zero
        surface_data: 2D array of surface elevation data
        origin_x: X coordinate of origin
        origin_y: Y coordinate of origin
        pixel_width: Width of each pixel
        pixel_height: Height of each pixel (negative for top-down)
        num_points: Size of the spacing along the profile (an integer >= 2); the
            profile holds ``num_points - 1`` of its points

    Returns:
        X_pro: X coordinates along profile
        Y_pro: Y coordinates along profile
        L_profile: Distances along profile
        XX: X coordinate grid
        YY: Y coordinate grid

    Raises:
        ValueError: ``num_points`` is not an integer >= 2, or the endpoints are
            the same location.
    """
    if isinstance(num_points, (bool, np.bool_)) or int(num_points) != num_points or num_points < 2:
        raise ValueError("num_points must be an integer >= 2.")
    num_points = int(num_points)
    # Create coordinate grids
    x = origin_x + pixel_width * np.arange(surface_data.shape[1])
    y = origin_y + pixel_height * np.arange(surface_data.shape[0])
    XX, YY = np.meshgrid(x, y)
    
    # Calculate start and end positions
    P1_pos = np.array([x[point1[0]], y[point1[1]]])
    P2_pos = np.array([x[point2[0]], y[point2[1]]])
    
    # Calculate total distance
    dis = np.sqrt(np.sum((P1_pos - P2_pos)**2))
    
    if not np.isfinite(dis) or dis <= 0:
        raise ValueError("Profile endpoints must be distinct finite locations.")
    # Generate profile coordinates: from point2 toward point1, num_points - 1
    # samples (see the docstring for why this layout is kept).
    X_pro = (x[point1[0]] - x[point2[0]])/dis * np.linspace(0, dis, num_points)[:-1] + x[point2[0]]
    Y_pro = (y[point1[1]] - y[point2[1]])/dis * np.linspace(0, dis, num_points)[:-1] + y[point2[1]]
    
    # Calculate profile distances
    L_profile = np.sqrt((X_pro - X_pro[0])**2 + (Y_pro - Y_pro[0])**2)
    
    return X_pro, Y_pro, L_profile, XX, YY


# ---------------------------------------------------------------------------
# interpolate structure to profile
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# prepare 2 D profile data
# ---------------------------------------------------------------------------
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

    for i in range(n_layers):
        layer_values = _sample_grid_along_profile(data[i], XX, YY, X_pro, Y_pro, method='linear')
        profile_values.append(layer_values)

    return np.array(profile_values)


# ---------------------------------------------------------------------------
# interpolate to mesh
# ---------------------------------------------------------------------------
def interpolate_to_mesh(
    property_values: np.ndarray,
    profile_distance: np.ndarray,
    depth_values: np.ndarray,
    mesh_x: np.ndarray,
    mesh_y: np.ndarray,
    mesh_markers: np.ndarray,
    ID: Any,
    layer_markers: list = [3, 0, 2],
) -> np.ndarray:
    """
    Interpolate property values from profile to mesh with layer-specific handling.
    
    Args:
        property_values: Array shaped (layers, profile points), any number of
            layers.
        profile_distance: Distance along profile (n_points)
        depth_values: Vertical coordinates (at least as many rows as
            property_values, profile points), in the same datum/units as
            mesh_y. Row i locates property row i, so with the usual layer
            surfaces (top plus one bottom per layer) each layer is placed at
            its top; rows beyond property_values' are not used.
        mesh_x: X coordinates of mesh cells
        mesh_y: Y coordinates of mesh cells
        mesh_markers: One layer marker per mesh cell. Cells outside
            layer_markers retain zero in the output.
        ID: Source layer labels shaped like property_values.
        layer_markers: List of marker values for each layer

    Returns:
        Interpolated values for mesh cells

    Raises:
        ValueError: depth_values has fewer rows, or a different number of
            profile points, than property_values.
    """
    # Initialize output array
    result = np.zeros_like(mesh_markers, dtype=float)

    L_profile_new = np.repeat(profile_distance.reshape(1,-1),property_values.shape[0],axis=0)

    # One depth row per property row. This took the first 14 rows whatever
    # the layer count, so only a 14-layer model could be interpolated.
    Depth = np.asarray(depth_values)[:property_values.shape[0]]
    if Depth.shape != np.shape(property_values):
        raise ValueError(
            f"depth_values {np.shape(depth_values)} needs at least one row per row of "
            f"property_values {np.shape(property_values)}, over the same profile points.")

    maxele = 0  # No elevation shift; source and mesh coordinates must agree.

    for marker in layer_markers:
        # Restrict interpolation to matching source/mesh layers; use nearest
        # values where the linear interpolation lies outside the convex hull.
        grid_z1 = griddata((L_profile_new[ID==marker].ravel(),Depth[ID==marker].ravel()- maxele), property_values[ID==marker].ravel(), (mesh_x[mesh_markers==marker], mesh_y[mesh_markers==marker]), method='linear')
        temp_ID = np.isnan(grid_z1)
        grid_z2 = griddata((L_profile_new[ID==marker].ravel(),Depth[ID==marker].ravel()- maxele), property_values[ID==marker].ravel(), (mesh_x[mesh_markers==marker], mesh_y[mesh_markers==marker]), method='nearest')
        grid_z1[temp_ID] = grid_z2[temp_ID]
        result[mesh_markers==marker] = grid_z1.copy()






    
    return result


# ---------------------------------------------------------------------------
# Profile Interpolator
# ---------------------------------------------------------------------------
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
            property_values: Property values array (n_layers, n_points), any
                number of layers
            depth_values: Vertical coordinates, at least one row per
                property row, in the same datum/units as mesh_y; see the
                module-level interpolate_to_mesh.
            mesh_x, mesh_y: Coordinates of mesh cells
            mesh_markers: One layer marker per mesh cell. Cells outside
            layer_markers retain zero in the output.
        ID: Source layer labels shaped like property_values.
            layer_markers: List of marker values for each layer
        
        Returns:
            Interpolated values for mesh cells
        """
        return interpolate_to_mesh(
            property_values, self.L_profile, depth_values,
            mesh_x, mesh_y, mesh_markers, ID,layer_markers
        )


# ---------------------------------------------------------------------------
# create surface lines
# ---------------------------------------------------------------------------
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
    
    # Preserve the input elevation datum; no vertical normalization is applied.
    surface[:,1] = surface[:,1]
    line1[:,1] = line1[:,1]
    line2[:,1] = line2[:,1]
    
    return surface, line1, line2
