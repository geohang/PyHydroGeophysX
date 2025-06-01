"""
Mesh utilities for geophysical modeling and inversion using PyGIMLi.

This module provides functions and a class for creating and manipulating
2D meshes, particularly for incorporating geological layer information or
velocity interfaces into the mesh structure.
"""
import numpy as np
import pygimli as pg
import pygimli.meshtools as mt
from typing import Tuple, List, Optional, Union
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

def create_mesh_from_layers(surface: np.ndarray,
                          line1: np.ndarray, # Coordinates of the first subsurface layer boundary
                          line2: np.ndarray, # Coordinates of the second subsurface layer boundary
                          bottom_depth: float = 30.0, # Absolute depth for the bottom of the mesh
                          quality: float = 28, # Mesh quality (higher for finer mesh, affects triangle quality)
                          area: float = 40) -> Tuple[pg.Mesh, np.ndarray, np.ndarray, pg.meshtools.geom.Polygon]: # Added geom to return
    """
    Create a 2D PyGIMLi mesh from three defined layer boundaries (surface, line1, line2).

    This function constructs a mesh suitable for geophysical modeling where distinct
    geological layers need to be represented with different markers. It creates
    polygons for each layer and then meshes the combined geometry.

    The markers are typically assigned as:
    - Marker 2: Top layer (surface to line1) and bottom layer (below line2 to mesh bottom)
    - Marker 3: Middle layer (between line1 and line2)

    Args:
        surface: Numpy array of (x, z) coordinates defining the ground surface.
                 Shape: (N_surface_points, 2).
        line1: Numpy array of (x, z) coordinates for the first subsurface boundary.
               Shape: (N_line1_points, 2).
        line2: Numpy array of (x, z) coordinates for the second subsurface boundary.
               Shape: (N_line2_points, 2).
        bottom_depth: The absolute z-coordinate for the bottom of the mesh.
                      Note: The original comment "Depth below surface minimum" seems
                      inconsistent with its usage `bottom_elev = bottom_depth`.
                      Assuming `bottom_depth` is an absolute elevation for the bottom.
                      If relative, `min_surface_elev - bottom_depth` would be used.
        quality: PyGIMLi mesh quality parameter. Controls the minimum angle of triangles.
                 Typical values are 20-35. Higher values mean better quality but more nodes.
        area: Maximum desired area for mesh cells. Smaller values lead to a finer mesh.

    Returns:
        Tuple containing:
            - mesh (pg.Mesh): The generated PyGIMLi mesh object.
            - mesh_centers (np.ndarray): Array of (x,y) coordinates for each cell center.
            - markers (np.ndarray): Array of integer markers for each cell in the mesh.
            - geom (pg.meshtools.geom.Polygon): The geometry object used to create the mesh.
    """
    # Determine bottom elevation for the mesh.
    # Original logic: bottom_elev = bottom_depth. This implies bottom_depth is an absolute elevation.
    # If bottom_depth was intended to be relative to min_surface_elev:
    # min_surface_elev = np.nanmin(surface[:,1])
    # bottom_elev = min_surface_elev - bottom_depth
    bottom_elev = bottom_depth # Assuming absolute depth for mesh bottom

    # Create reversed versions of line1 and line2 for constructing closed polygons.
    # This is necessary because createPolygon often requires points in a specific order (e.g., clockwise)
    # to define the inside/outside of a polygon, especially for `isClosed=True`.
    line1r = line1.copy()
    line1r[:, 0] = np.flip(line1[:, 0]) # Reverse x-coordinates
    line1r[:, 1] = np.flip(line1[:, 1]) # Reverse z-coordinates

    line2r = line2.copy()
    line2r[:, 0] = np.flip(line2[:, 0])
    line2r[:, 1] = np.flip(line2[:, 1])

    # Define the geometry components as per the original implementation.
    # 1. `layer1`: An open polyline representing the surface.
    #    Marker 2, area=0.1 (suggests refinement near surface).
    layer1 = mt.createPolygon(surface, isClosed=False, marker=2, boundaryMarker=-1, interpolate='linear', area=0.1)

    # 2. `Gline1`: A closed polygon for the middle layer (between line1 and line2r).
    #    Marker 3. This defines the middle region.
    Gline1 = mt.createPolygon(np.vstack((line1, line2r)), isClosed=True, marker=3, boundaryMarker=1, interpolate='linear', area=1)

    # 3. `Gline2`: An open polyline. Defines part of the outer boundary or an internal one.
    #    Points: [surface_start, point_under_line2_start, point_under_line2_end, surface_end].
    #    Marker 2, area=2.
    #    This polyline seems to define the "floor" and parts of the "walls" of the model domain.
    gline2_points = [
        [surface[0,0], surface[0,1]],      # Surface start point
        [line2[0,0], bottom_elev],         # Point below line2 start at bottom_elev
        [line2[-1,0], bottom_elev],        # Point below line2 end at bottom_elev
        [surface[-1,0], surface[-1,1]],    # Surface end point
    ]
    Gline2 = mt.createPolygon(gline2_points, isClosed=False, marker=2, boundaryMarker=1, interpolate='linear', area=2)

    # 4. `layer2`: A closed polygon for the bottom layer (below line2).
    #    Marker 2, area=2.
    #    The points define the region: along line2r (reversed line2), then connecting
    #    the ends of line2r down to bottom_elev, across at bottom_elev, and back up.
    layer2_points = np.vstack((
        line2r,                             # Path along reversed line2 (e.g., right to left)
        [[line2[0,0], line2[0,1]]],         # Connects end of line2r back to start of line2 (closes along line2)
                                            # This point is line2_left_end. line2r ends at line2_left_end.
                                            # This makes the path trace line2.
        [[line2[0,0], bottom_elev]],        # From line2_left_end down to bottom_elev
        [[line2[-1,0], bottom_elev]],       # Across at bottom_elev to a point under line2_right_end
        [[line2[-1,0], line2[-1,1]]]        # Up from bottom_elev to line2_right_end
                                            # Polygon closes by connecting this to start of line2r (line2_right_end).
    ))
    # Potential issue: If line2 is not perfectly aligned (e.g. x-coordinates) with surface or line1,
    # the polygons might have gaps or overlaps that could complicate meshing or marker assignment.
    # The construction of layer2_points assumes line2r provides a path that, when combined with the
    # vertical and bottom segments, forms a well-defined closed region.
    layer2 = mt.createPolygon(layer2_points, isClosed=True, marker=2, area=2, boundaryMarker=1)

    # Combine all geometry components into a single PLC (Piecewise Linear Complex).
    # The order of addition can sometimes matter for how PyGIMLi resolves overlapping features
    # or assigns properties, though typically it handles merging appropriately.
    geom = layer1 + layer2 + Gline1 + Gline2
    # Note: The top layer (surface down to line1) is not defined as an explicit *closed* polygon
    # in this sum. It's expected to be formed implicitly by the mesher using `layer1` (surface polyline)
    # as its top and the upper edge of `Gline1` (which is line1) as its bottom.
    # The marker for this implicit top region will likely be determined by the marker of `layer1` (surface polyline), which is 2.

    # Create the mesh from the composite geometry PLC.
    mesh = mt.createMesh(geom, quality=quality, area=area)

    # Extract cell centers and markers from the generated mesh.
    mesh_centers = np.array(mesh.cellCenters()) # Nx2 array of (x, z) for 2D meshes
    markers = np.array(mesh.cellMarkers())      # 1D array of integer markers for each cell

    return mesh, mesh_centers, markers, geom # Return geom for potential inspection or debugging







def extract_velocity_interface(mesh: pg.Mesh,
                               velocity_data: np.ndarray,
                               threshold: float = 1200.0,
                               interval: float = 4.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract an interface (e.g., bedrock) from velocity data on a mesh.

    The function identifies points where the cell velocity values cross a given
    threshold. It bins cells by their x-coordinates, finds the depth of the
    threshold crossing within each bin via linear interpolation, and then
    smooths the resulting interface line using cubic interpolation and a
    Savitzky-Golay filter.

    Args:
        mesh: PyGIMLi mesh object containing the cells.
        velocity_data: 1D numpy array of velocity values, one per cell in `mesh`.
                       Order must correspond to mesh.cellIDs().
        threshold: The velocity value that defines the interface (e.g., velocity of bedrock).
        interval: The width of x-coordinate bins used to sample the interface.
                  Smaller intervals give more detail but may be noisier.

    Returns:
        Tuple (x_dense, z_dense):
            - x_dense (np.ndarray): 1D array of x-coordinates for the smoothed interface.
            - z_dense (np.ndarray): 1D array of z-coordinates (depths) for the smoothed interface.
              Coordinate system (e.g. depth positive down, or elevation positive up) should
              be consistent with the input mesh's coordinate system.
    """
    # Get cell center coordinates (x, z)
    # PyGIMLi's 2D meshes use x and y coordinates. For profile views, y is often depth.
    if mesh.dim() != 2:
        raise ValueError(f"Mesh must be 2D. Got dimension: {mesh.dim()}")
    
    cell_centers_tuples = mesh.cellCenters() # Returns list of RVector3 like objects for pg.Mesh
                                         # For 2D mesh, z component is often 0 or unused.
                                         # We assume x is horizontal, y is vertical (depth/elevation).
    
    # Convert to numpy array for easier slicing. Assuming we need x and y (depth).
    # If mesh.cell(0).center() returns (x,y,z), then use [c.x(), c.y()] or [c.x(), c.z()]
    # depending on mesh orientation. Standard for 2D pg.Mesh is x,y.
    try:
        # Attempt to get x,y assuming centers are directly convertible or attributes
        x_coords = np.array([c.x() for c in cell_centers_tuples])
        z_coords = np.array([c.y() for c in cell_centers_tuples]) # y-coordinate treated as depth/elevation
    except AttributeError:
        # Fallback if .x() / .y() not present (e.g. if cell_centers was already an array from a specific source)
        if isinstance(cell_centers_tuples, np.ndarray) and cell_centers_tuples.shape[1] >= 2:
            x_coords = cell_centers_tuples[:, 0]
            z_coords = cell_centers_tuples[:, 1]
        else:
            raise TypeError("Could not extract x, z coordinates from mesh.cellCenters().")


    if mesh.cellCount() != len(velocity_data):
        raise ValueError("Length of velocity_data must match the number of cells in the mesh.")

    # Determine the horizontal range of the mesh for binning
    x_min_mesh, x_max_mesh = np.min(x_coords), np.max(x_coords)

    # Create bins across the horizontal extent of the mesh
    x_bins = np.arange(x_min_mesh, x_max_mesh + interval, interval)

    interface_x_raw = []
    interface_z_raw = []

    # Iterate through each bin to find where velocity crosses the threshold
    for i in range(len(x_bins) - 1):
        bin_cell_indices = np.where((x_coords >= x_bins[i]) & (x_coords < x_bins[i+1]))[0]

        if len(bin_cell_indices) > 0:
            bin_velocities = velocity_data[bin_cell_indices]
            bin_depths = z_coords[bin_cell_indices]

            # Sort cells in the bin by depth to allow systematic search for threshold crossing
            # This assumes z_coords represent depth where increasing value means deeper.
            # If z_coords is elevation (increasing upwards), sorting might need to be reversed
            # or the logic for "above/below" adapted.
            sort_indices = np.argsort(bin_depths) 
            bin_velocities_sorted = bin_velocities[sort_indices]
            bin_depths_sorted = bin_depths[sort_indices]

            found_in_bin = False
            for j in range(1, len(bin_velocities_sorted)):
                v_prev, z_prev = bin_velocities_sorted[j-1], bin_depths_sorted[j-1]
                v_curr, z_curr = bin_velocities_sorted[j], bin_depths_sorted[j]

                if (v_prev < threshold <= v_curr) or (v_prev >= threshold > v_curr):
                    if v_curr == v_prev: 
                        interpolated_depth = z_prev if abs(threshold - v_prev) < abs(threshold - v_curr) else z_curr
                    else:
                        ratio = (threshold - v_prev) / (v_curr - v_prev)
                        interpolated_depth = z_prev + ratio * (z_curr - z_prev)
                    
                    interface_x_raw.append((x_bins[i] + x_bins[i+1]) / 2.0) 
                    interface_z_raw.append(interpolated_depth)
                    found_in_bin = True
                    break 
            
    interface_x_raw = np.array(interface_x_raw)
    interface_z_raw = np.array(interface_z_raw)

    if len(interface_x_raw) < 2:
        # print("Warning: Less than 2 raw interface points found. Cannot interpolate reliably.")
        return np.array([]), np.array([]) # Return empty if not enough points

    # Extrapolation logic (from original code)
    # Check if first found x is significantly after mesh start
    # A small tolerance (e.g. interval/100) added to avoid floating point issues in comparison
    if interface_x_raw[0] > x_min_mesh + interval / 2.0 + 1e-9 : 
        extrap_x_start = x_min_mesh
        if len(interface_x_raw) >= 2: 
            # Avoid division by zero if x-coordinates are identical
            dx_start = interface_x_raw[1] - interface_x_raw[0]
            slope_start = (interface_z_raw[1] - interface_z_raw[0]) / dx_start if dx_start != 0 else 0
            extrap_z_start = interface_z_raw[0] - slope_start * (interface_x_raw[0] - x_min_mesh)
        else: 
            extrap_z_start = interface_z_raw[0]
        interface_x_raw = np.insert(interface_x_raw, 0, extrap_x_start)
        interface_z_raw = np.insert(interface_z_raw, 0, extrap_z_start)

    # Check if last found x is significantly before mesh end
    if len(interface_x_raw) > 0 and interface_x_raw[-1] < x_max_mesh - interval / 2.0 - 1e-9:
        extrap_x_end = x_max_mesh
        if len(interface_x_raw) >= 2: 
            dx_end = interface_x_raw[-1] - interface_x_raw[-2]
            slope_end = (interface_z_raw[-1] - interface_z_raw[-2]) / dx_end if dx_end != 0 else 0
            extrap_z_end = interface_z_raw[-1] + slope_end * (x_max_mesh - interface_x_raw[-1])
        else: 
            extrap_z_end = interface_z_raw[-1]
        interface_x_raw = np.append(interface_x_raw, extrap_x_end)
        interface_z_raw = np.append(interface_z_raw, extrap_z_end)

    if len(interface_x_raw) < 2: 
        # print("Warning: Still less than 2 interface points after extrapolation.")
        return np.array([]), np.array([])

    x_dense = np.linspace(x_min_mesh, x_max_mesh, 500)
    z_smooth_dense = np.array([]) # Initialize to ensure it's defined

    try:
        sort_idx = np.argsort(interface_x_raw)
        interface_x_sorted = interface_x_raw[sort_idx]
        interface_z_sorted = interface_z_raw[sort_idx]

        unique_x, unique_idx = np.unique(interface_x_sorted, return_index=True)
        interface_x_unique = interface_x_sorted[unique_idx]
        interface_z_unique = interface_z_sorted[unique_idx]
        
        if len(interface_x_unique) < 2 :
             raise ValueError("Not enough unique x-points for interpolation after processing duplicates.")

        interp_kind = 'linear'
        if len(interface_x_unique) >= 4: # Cubic interpolation needs at least 4 unique points
            interp_kind = 'cubic'
        
        interp_func = interp1d(interface_x_unique, interface_z_unique, kind=interp_kind,
                               bounds_error=False, fill_value="extrapolate")
        z_interp_dense = interp_func(x_dense)

        # Savitzky-Golay filter conditions
        # Window length must be odd, > polyorder, and <= number of data points.
        window_len = 31
        poly_order = 3
        if interp_kind == 'cubic' and len(z_interp_dense) >= window_len : 
            z_smooth_dense = savgol_filter(z_interp_dense, window_length=window_len, polyorder=poly_order)
        else:
            z_smooth_dense = z_interp_dense
            
    except ValueError as e:
        # print(f"Interpolation or smoothing failed: {e}. Attempting fallback or returning empty.")
        # Fallback to linear if cubic failed and enough points for linear
        if interp_kind == 'cubic' and len(interface_x_unique) >= 2:
            try:
                interp_func_linear = interp1d(interface_x_unique, interface_z_unique, kind='linear',
                                              bounds_error=False, fill_value="extrapolate")
                z_smooth_dense = interp_func_linear(x_dense)
            except: # Final fallback if linear also fails
                 pass # z_smooth_dense remains empty
        # If linear was primary, or if already tried linear as fallback, z_smooth_dense might still be empty or contain previous result
        elif 'z_interp_dense' in locals() and z_interp_dense.size > 0 : # Check if z_interp_dense was calculated
             z_smooth_dense = z_interp_dense

    if z_smooth_dense.size == 0:
        x_dense = np.array([]) # Ensure x_dense is also empty for consistency

    return x_dense, z_smooth_dense
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


def add_velocity_interface(ertData: pg.DataContainer,
                               smooth_x: np.ndarray, # X-coordinates of the interface
                               smooth_z: np.ndarray, # Z-coordinates of the interface
                               paraBoundary_ext: float = 2.0, # How much to extend interface beyond sensor range
                               default_marker_outside: int = 1,
                               marker_above_interface: int = 3, # Swapped from original comment based on typical visualization
                               marker_below_interface: int = 2  # Swapped from original comment
                               ) -> Tuple[np.ndarray, pg.Mesh]:
    """
    Incorporate a smoothed velocity interface into a mesh generated for ERT data.

    This function first creates a standard ERT parameter mesh using PyGIMLi's
    `createParaMeshPLC`. It then adds the provided `smooth_x, smooth_z` interface
    as a polyline to this geometry. Finally, it re-meshes and assigns markers
    to cells based on their position relative to this interface within the main
    survey area.

    The definition of "survey area" and "above/below" depends on coordinate system conventions.
    Typically, for PyGIMLi 2D meshes, Y increases downwards (depth).
    - `marker_above_interface` (e.g., 3): Cells physically shallower than the interface.
    - `marker_below_interface` (e.g., 2): Cells physically deeper than or on the interface.

    Args:
        ertData: PyGIMLi data container (e.g., `pg.DataContainer`). Must have sensor positions.
        smooth_x: 1D numpy array of x-coordinates for the smoothed interface line.
        smooth_z: 1D numpy array of z-coordinates for the smoothed interface line.
                  The y-coordinate from PyGIMLi mesh cells (depth) will be compared against these.
        paraBoundary_ext: Extension distance for the interface line beyond the
                          x-range of ERT sensors. Also used by `createParaMeshPLC` for defining
                          the outer, less refined mesh region. Default is 2.0 units.
        default_marker_outside: Marker for cells considered outside the primary survey/parametric region. Default is 1.
        marker_above_interface: Marker for cells inside survey region, physically above (shallower than) the interface. Default is 3.
        marker_below_interface: Marker for cells inside survey region, on or below (deeper than) the interface. Default is 2.

    Returns:
        Tuple (markers, meshafter):
            - markers (np.ndarray): 1D array of final cell markers for `meshafter`.
            - meshafter (pg.Mesh): The new PyGIMLi mesh with the interface incorporated
                                   and cell markers assigned.
    Raises:
        ValueError: If `smooth_x` or `smooth_z` are not 1D arrays of the same length,
                    or if they have fewer than 2 points.
    """
    if not (isinstance(smooth_x, np.ndarray) and isinstance(smooth_z, np.ndarray) and
            smooth_x.ndim == 1 and smooth_z.ndim == 1 and len(smooth_x) == len(smooth_z)):
        raise ValueError("smooth_x and smooth_z must be 1D numpy arrays of the same length.")
    if len(smooth_x) < 2:
        raise ValueError("Interface (smooth_x, smooth_z) must have at least 2 points to define a line.")

    # Create an initial mesh geometry based on ERT sensor positions.
    # Parameters like quality, paraMaxCellSize, paraDepth, boundaryMaxCellSize are set to typical values.
    # These could be exposed as arguments to this function for more flexibility.
    geo_base = mt.createParaMeshPLC(ertData, quality=32, paraMaxCellSize=30,
                                    paraBoundary=paraBoundary_ext, paraDepth=30.0,
                                    boundaryMaxCellSize=500)

    # Prepare the interface points. Stack to get (N, 2) array of [x, z] points.
    # The z-coordinates here are from `smooth_z` and represent the interface's depth/elevation.
    interface_coords = np.vstack((smooth_x, smooth_z)).T

    # Extend the interface line using interpolation to ensure it spans wider than the sensor coverage,
    # including the `paraBoundary_ext` region, so it properly cuts through the entire mesh.
    sensor_x_positions = ertData.sensorPositions()[:, 0] # Assuming 2D or 3D positions from DataContainer
    min_sensor_x, max_sensor_x = np.min(sensor_x_positions), np.max(sensor_x_positions)

    # Sort the input interface line by x-coordinate for reliable interpolation
    sort_indices_interface = np.argsort(interface_coords[:, 0])
    interface_coords_sorted = interface_coords[sort_indices_interface, :]
    
    # Create an interpolator for the interface line, allowing extrapolation if needed.
    f_interp_interface = interp1d(interface_coords_sorted[:, 0], interface_coords_sorted[:, 1],
                                  kind='linear', fill_value="extrapolate")
    
    # Define the x-range for the polyline to be added to the geometry.
    # It should span from (min_sensor_x - paraBoundary_ext) to (max_sensor_x + paraBoundary_ext).
    # Add a small safety margin to ensure it fully cuts the PLC boundary.
    safety_margin = paraBoundary_ext * 0.05 # e.g., 5% of paraBoundary_ext
    extended_line_x_start = min_sensor_x - paraBoundary_ext - safety_margin
    extended_line_x_end = max_sensor_x + paraBoundary_ext + safety_margin
    
    # Create points for the extended polyline.
    # Use original interface points that fall within this new extended range, plus the new endpoints.
    relevant_points_mask = (interface_coords_sorted[:, 0] >= extended_line_x_start) & \
                           (interface_coords_sorted[:, 0] <= extended_line_x_end)
    x_for_poly = interface_coords_sorted[relevant_points_mask, 0]
    z_for_poly = interface_coords_sorted[relevant_points_mask, 1]

    # Prepend the new start point if it's not already covered.
    if not x_for_poly.size or not np.isclose(x_for_poly[0], extended_line_x_start):
        x_for_poly = np.insert(x_for_poly, 0, extended_line_x_start)
        z_for_poly = np.insert(z_for_poly, 0, f_interp_interface(extended_line_x_start))
        
    # Append the new end point if it's not already covered.
    if not x_for_poly.size or not np.isclose(x_for_poly[-1], extended_line_x_end): # Check size again in case it was empty
        x_for_poly = np.append(x_for_poly, extended_line_x_end)
        z_for_poly = np.append(z_for_poly, f_interp_interface(extended_line_x_end))
        
    # Remove duplicate x-points that might have arisen from prepending/appending if original line was short
    unique_x_poly, unique_indices_poly = np.unique(x_for_poly, return_index=True)
    extended_interface_polyline_coords = np.vstack((unique_x_poly, z_for_poly[unique_indices_poly])).T
    
    # Create a PyGIMLi polyline from these extended interface coordinates.
    # Marker 99 is arbitrary, just to identify this line in the PLC if needed during debugging.
    interface_as_polyline = mt.createPolygon(extended_interface_polyline_coords.tolist(),
                                             isClosed=False, interpolate='linear', marker=99)

    # Add the interface polyline to the base ERT geometry. This acts as an internal boundary.
    geo_combined = geo_base + interface_as_polyline

    # Create a new mesh from the combined geometry. Quality 28 is an example value.
    meshafter = mt.createMesh(geo_combined, quality=28) # Area constraint could be added.

    # Initialize cell markers for the new mesh.
    final_markers = np.full(meshafter.cellCount(), default_marker_outside, dtype=int)
    
    # Define the horizontal limits of the "survey area" where detailed marking based on interface occurs.
    # This area corresponds to the region extended by `paraBoundary_ext` around sensors.
    survey_region_x_min = min_sensor_x - paraBoundary_ext
    survey_region_x_max = max_sensor_x + paraBoundary_ext

    # Iterate over each cell in the new mesh to assign markers.
    for i, cell in enumerate(meshafter.cells()): # Iterate through cells
        cell_center_x = cell.center().x()
        cell_center_y_depth = cell.center().y() # PyGIMLi's y-coordinate, typically depth (increases downwards)

        # Check if the cell center is within the defined horizontal survey region.
        if survey_region_x_min <= cell_center_x <= survey_region_x_max:
            # Interpolate the z-coordinate (depth) of the interface at the cell's x-center.
            # Use `extended_interface_polyline_coords` for this interpolation.
            interface_z_at_cell_x = np.interp(cell_center_x,
                                              extended_interface_polyline_coords[:, 0], # X values
                                              extended_interface_polyline_coords[:, 1])  # Z values (depths)
            
            # Compare cell center depth with interface depth.
            # Assuming PyGIMLi's y-coordinate for 2D mesh increases downwards (depth convention):
            # - Cell is "above" (shallower) if its y_depth is LESS than interface_z_at_cell_x.
            # - Cell is "below" (deeper) if its y_depth is GREATER than interface_z_at_cell_x.
            # The original code's `abs(cell_y) < abs(interface_y)` for "below" was potentially confusing
            # if z-values could be positive/negative (elevations).
            # With depth convention (Y positive downwards):
            if cell_center_y_depth < interface_z_at_cell_x: # Cell is shallower than the interface
                final_markers[i] = marker_above_interface
            else: # Cell is deeper than or on the interface
                final_markers[i] = marker_below_interface
        # else: cell is outside survey_region_x_min/max, keeps `default_marker_outside` assigned initially.

    # Apply the calculated markers to the mesh.
    meshafter.setCellMarkers(final_markers)

    return final_markers, meshafter
    
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
    
    """
    A utility class for creating PyGIMLi meshes, particularly for layered models
    or standard ERT surveys.
    """
    def __init__(self, quality: float = 30.0, area: Optional[float] = None):
        """
        Initialize MeshCreator.

        Args:
            quality: Default mesh quality parameter for `mt.createMesh`.
                     Higher values (e.g., 30-35) generally result in better-shaped
                     triangles but can increase node count. Default is 30.0.
            area: Default maximum cell area constraint for `mt.createMesh`.
                  If None, no specific area constraint is applied by default, allowing
                  quality to primarily dictate cell sizes. Can be overridden in methods.
                  A smaller value leads to a finer mesh in the specified region.
        """
        self.quality = quality
        self.area = area 

    def create_from_layers(self,
                           surface: np.ndarray,
                           layers: List[np.ndarray], 
                           bottom_depth: float, 
                           layer_markers: Optional[List[int]] = None, # Note: Not used by current underlying call
                           quality: Optional[float] = None,
                           area: Optional[float] = None 
                           ) -> Tuple[pg.Mesh, pg.meshtools.geom.Polygon]:
        """
        Create a 2D PyGIMLi mesh from a surface polyline and a list of subsurface layer boundary polylines.

        This method constructs a mesh where regions between the provided surface and
        layer boundaries are assigned specific markers. It's designed for models
        with a defined number of horizontal-like layers.

        Currently, this implementation specifically calls the global `create_mesh_from_layers`
        function, which is hardcoded for exactly two subsurface layers (i.e., three regions total:
        surface-to-layer1, layer1-to-layer2, layer2-to-bottom).
        The `layer_markers` argument here is NOT used by the current underlying global function call.

        Args:
            surface: Numpy array of (x, z) coordinates for the ground surface. (N_surface, 2).
                     Z is typically elevation or depth, ensure consistency with `layers` and `bottom_depth`.
            layers: List of numpy arrays. Each array contains (x, z) coordinates
                    for a subsurface layer boundary. E.g., `[layer1_coords, layer2_coords]`.
            bottom_depth: Absolute z-coordinate for the bottom of the entire mesh.
                          Convention (e.g., elevation positive up, depth positive down) must be consistent.
            layer_markers: Optional list of integer markers. Intended for future use if this
                           method implements generic multi-layer meshing. Currently not passed through.
            quality: Mesh quality parameter. Overrides class default `self.quality` if provided.
            area: Maximum cell area for the mesh. Overrides class default `self.area` if provided.
                  If None, uses class default. If class default is also None, PyGIMLi's default (often 0, no constraint) is used.

        Returns:
            Tuple (pg.Mesh, pg.meshtools.geom.Polygon):
                - mesh: The generated PyGIMLi mesh.
                - geom: The PLC geometry used to create the mesh.

        Raises:
            NotImplementedError: If the number of layers in the `layers` list is not 2.
            ValueError: If `layers` list is empty.
        """
        if not layers: 
            raise ValueError("The 'layers' list must contain at least one layer boundary.")

        current_quality = quality if quality is not None else self.quality
        # `mt.createMesh` expects area constraint as a float. If None, pass 0.0 (no constraint).
        current_area_val = area if area is not None else (self.area if self.area is not None else 0.0)

        # Normalization of coordinates (e.g., subtracting max_ele) was present in the original
        # class method but is omitted here. The global `create_mesh_from_layers` function
        # expects absolute coordinates. If normalization is required, it should be performed
        # by the user before calling this method, ensuring `surface`, `layers`, and `bottom_depth`
        # are all in a consistent coordinate system.

        if len(layers) == 2:
            # This calls the global function `create_mesh_from_layers`.
            # That function has its own hardcoded marker logic (marker 2 for top/bottom, 3 for middle).
            # The `layer_markers` argument of this class method is not used by the global function.
            mesh, _, _, geom = create_mesh_from_layers( # mesh, mesh_centers, cell_markers, geom_plc
                surface, layers[0], layers[1], 
                bottom_depth=bottom_depth,    
                quality=current_quality,
                area=current_area_val 
            )
            # Future enhancement: If `layer_markers` is provided, and a generic multi-layer scheme is implemented,
            # those markers should be used to set cell markers in the mesh.
            return mesh, geom
        else:
            # This part would require a more generalized PLC construction:
            # 1. Define an outer boundary polygon for the entire domain.
            # 2. Add each layer from `layers` as an internal polyline interface.
            # 3. Create a PLC from these components.
            # 4. Mesh the PLC.
            # 5. Assign cell markers based on which region (defined by interfaces) each cell center falls into.
            raise NotImplementedError(
                "Mesh creation for a number of layers other than 2 (i.e., surface + two subsurface boundaries) "
                "is not yet implemented in this version of `create_from_layers`. "
                "The current implementation is specific to a 3-region model via the global function."
            )

    def create_from_ert_data(self,
                             data: pg.DataContainer, 
                             max_depth: Optional[float] = None, 
                             quality: Optional[float] = None, 
                             paraMaxCellSize: Optional[float] = None, 
                             paraBoundary: Optional[float] = None, 
                             boundaryMaxCellSize: Optional[float] = None 
                             ) -> pg.Mesh:
        """
        Create a mesh suitable for ERT (Electrical Resistivity Tomography) inversion.

        This method utilizes PyGIMLi's `ERTManager` to generate a mesh tailored to
        the sensor configuration found in the `data` object. It allows customization
        of various meshing parameters relevant to ERT.

        Args:
            data: PyGIMLi DataContainer object. Must contain sensor positions.
            max_depth: Maximum depth of the parametric (inversion) domain. (Corresponds to `paraDepth` in ERTManager).
                       If None, `ERTManager` might use a default or auto-calculate.
            quality: Mesh quality parameter. Overrides class default `self.quality` if provided.
            paraMaxCellSize: Maximum cell size in the parametric domain (central region with data sensitivity).
            paraBoundary: Width of the less refined boundary region surrounding the parametric domain.
            boundaryMaxCellSize: Maximum cell size in this outer boundary region.

        Returns:
            pg.Mesh: The generated PyGIMLi mesh, suitable for ERT forward modeling and inversion.
        
        Raises:
            ImportError: If `pygimli.physics.ert` cannot be imported.
        """
        try:
            from pygimli.physics import ert # Local import for ERT-specific functionality
        except ImportError as e:
            raise ImportError("Module `pygimli.physics.ert` is required for `create_from_ert_data` "
                              "but could not be imported. Please ensure PyGIMLi is fully installed.") from e

        ert_manager = ert.ERTManager(data=data, verbose=False) # verbose=False to suppress console output

        # Prepare dictionary of arguments for ERTManager's createMesh method.
        # Only include arguments if they are explicitly provided or set in class defaults.
        mesh_args = {}
        
        current_quality = quality if quality is not None else self.quality
        if current_quality is not None: # Ensure quality is not None before adding
            mesh_args['quality'] = current_quality
        
        if max_depth is not None: 
            mesh_args['paraDepth'] = max_depth # ERTManager uses 'paraDepth'
        
        if paraMaxCellSize is not None:
            mesh_args['paraMaxCellSize'] = paraMaxCellSize
            
        if paraBoundary is not None:
            mesh_args['paraBoundary'] = paraBoundary

        if boundaryMaxCellSize is not None:
            mesh_args['boundaryMaxCellSize'] = boundaryMaxCellSize
        
        # Generate the mesh using ERTManager.
        created_mesh = ert_manager.createMesh(data=data, **mesh_args)
        
        return created_mesh
