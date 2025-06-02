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
    # Calculate the elevation for the bottom of the mesh.
    # Currently, `bottom_depth` is treated as an absolute elevation value for the mesh bottom.
    # The commented-out line `min_surface_elev = np.nanmin(surface[:,1])` and the subsequent comment
    # `bottom_elev = min_surface_elev - bottom_depth` suggest that `bottom_depth` might have
    # previously been interpreted as a relative depth below the minimum surface elevation.
    # min_surface_elev = np.nanmin(surface[:,1]) # Example: find minimum elevation of the surface points
    bottom_elev = bottom_depth # Current: bottom_elev is an absolute Z value.
                               # Alternative: bottom_elev = min_surface_elev - bottom_depth (if bottom_depth is relative depth)

    # Create reversed versions of line1 and line2 (boundary lines for layers).
    # These are needed to define closed polygons for mesh layers by connecting
    # the end of one line segment (e.g., line1) to the start of another (e.g., line2r, which is line2 reversed).
    # This allows forming closed loops for defining regions in PyGIMLi.
    line1r = line1.copy() # Create a copy to avoid modifying the original array
    line1r[:,0] = np.flip(line1[:,0]) # Flip the order of x-coordinates
    line1r[:,1] = np.flip(line1[:,1]) # Flip the order of z-coordinates (elevations)

    line2r = line2.copy() # Create a copy
    line2r[:,0] = np.flip(line2[:,0]) # Flip x-coordinates
    line2r[:,1] = np.flip(line2[:,1]) # Flip z-coordinates

    # Define the first geometric component: the region above 'line1' (top layer).
    # This PLC segment ('layer1') is defined by the 'surface' points.
    # `isClosed=False` indicates this segment itself doesn't form a closed polygon but is a boundary.
    # The region it helps define (with marker 2) is bounded by 'surface' from above and 'line1' from below.
    # `marker=2`: Cells in the region whose top is `surface` and bottom is `line1` will get this marker.
    # `boundaryMarker=-1`: Boundary marker for the `surface` segment itself.
    # `area=0.1`: Meshing constraint (e.g., desired max cell area or refinement factor) near this segment.
    layer1 = mt.createPolygon(surface,       # Points defining the top boundary (surface line)
                             isClosed=False,  # This path itself is not a closed loop.
                             marker=2,        # Target region marker for the area defined by this (as upper bound) and line1 (as lower bound).
                             boundaryMarker=-1, # Marker for the surface boundary segment.
                             interpolate='linear', # Linear interpolation between the points of the surface line.
                             area=0.1)        # Meshing constraint related to this segment.
                             # SUGGESTION: The area parameter in createPolygon is a refinement attribute for elements near this line.

    # Define the second geometric component: the region between 'line1' and 'line2' (middle layer).
    # This polygon ('Gline1') is formed by concatenating 'line1' and the reversed 'line2' (line2r).
    # This creates a closed loop: from start of line1 to end of line1, then from end of line2 (start of line2r)
    # to start of line2 (end of line2r), effectively closing the polygon.
    # `marker=3`: Cells within this closed polygon (between line1 and line2) get this marker.
    # `boundaryMarker=1`: Marker for the boundary segments of this polygon (i.e., line1 and line2 themselves).
    Gline1 = mt.createPolygon(np.vstack((line1, line2r)), # Vertical stack of line1 points and reversed line2 points.
                             isClosed=True,   # This forms a closed polygon loop.
                             marker=3,        # Region marker for cells inside this polygon (between line1 and line2).
                             boundaryMarker=1,  # Marker for the boundary segments of this polygon.
                             interpolate='linear',
                             area=1)          # Meshing constraint for this region.

    # Define the third geometric component: an outer boundary definition ('Gline2').
    # This PLC segment defines parts of the overall model boundary, particularly for the region with marker=2 (top layer).
    # It connects:
    #   1. Start of the surface `[surface[0,0], surface[0,1]]` (top-left of model).
    #   2. A point vertically below the start of line2, at `bottom_elev` `[line2[0,0], bottom_elev]` (bottom-left of this bounding box).
    #   3. A point horizontally across at `bottom_elev`, aligned with the end of line2 `[line2[-1,0], bottom_elev]` (bottom-right of this box).
    #   4. End of the surface `[surface[-1,0], surface[-1,1]]` (top-right of model).
    # `isClosed=False` indicates it's a set of boundary segments.
    # `marker=2`: This associates these boundary segments with the region that has marker 2.
    # It effectively forms the left, bottom (of the part above line2's general elevation), and right boundaries of the top layer's extent.
    # SUGGESTION: "Gline2" is a bit generic. Could be `OuterBoundaryForLayer1`.
    Gline2 = mt.createPolygon([[surface[0,0], surface[0,1]],     # Top-left point of the model (start of surface).
                              [line2[0,0], bottom_elev],       # Bottom-left point for the bounding box of the region above line2.
                              [line2[-1,0], bottom_elev],      # Bottom-right point for this bounding box.
                              [surface[-1,0], surface[-1,1]]], # Top-right point of the model (end of surface).
                             isClosed=False,  # This is a sequence of lines, not a filled polygon.
                             marker=2,        # Associates these lines with region marker 2 (the top layer).
                             boundaryMarker=1, # Marker for these specific boundary segments.
                             interpolate='linear',
                             area=2)          # Meshing constraint.

    # Define the fourth geometric component: the region below 'line2' (bottom layer).
    # This polygon ('layer2') is formed by `line2r` (reversed line2) as its top boundary,
    # and then closed by segments going down to `bottom_elev`, across, and back up to `line2r`'s start.
    # `marker=2`: Cells in this bottom region also get marker 2. This means the top and bottom layers in this
    # specific 3-layer setup (surface-line1, line1-line2, line2-bottom) will share the same marker.
    # SUGGESTION: If distinct properties are needed for the geological top and bottom layers, this marker should be different.
    # The vertices for this polygon are:
    #   - Points of line2r (from end of original line2, tracing back to start of original line2).
    #   - Then, a point vertically below the start of original line2, at `bottom_elev`.
    #   - Then, a point vertically below the end of original line2, at `bottom_elev`.
    # `isClosed=True` tells `createPolygon` to connect the last point `[line2[-1,0], bottom_elev]`
    # back to the first point of `line2r` (which is `line2[-1,:]`, the end of original line2).
    layer2_boundary_nodes = np.vstack((
        line2r,                                 # Path along reversed line2 (from end of line2 to start of line2).
        np.array([[line2[0,0], bottom_elev]]),  # Point below the start of original line2, at `bottom_elev`.
        np.array([[line2[-1,0], bottom_elev]]) # Point below the end of original line2, at `bottom_elev`.
    ))
    layer2 = mt.createPolygon(layer2_boundary_nodes,
                             isClosed=True,  # Forms a closed polygon for the bottom layer.
                             marker=2,       # Region marker for cells below line2.
                             area=2,         # Meshing constraint for this region.
                             boundaryMarker=1) # Marker for the boundary segments of this polygon.

    # Combine all defined geometric parts (PLC segments and polygons) into a single geometry object.
    # The `+` operator merges these PLC objects. PyGIMLi's meshing tools use this combined geometry.
    # The order of addition generally doesn't matter for `mt.createMesh` as it resolves regions based on markers and geometry.
    geom = layer1 + layer2 + Gline1 + Gline2
    
    # Create the final computational mesh from the combined geometry object.
    # `quality` is a constraint on mesh element shape (e.g., minimum angle of triangles).
    # `area` is a global constraint on maximum cell area (can be locally refined by `area` in `createPolygon`).
    mesh = mt.createMesh(geom, quality=quality, area=area)
    
    # Retrieve the coordinates of the center of each cell in the generated mesh.
    # `mesh.cellCenters()` returns a list of pg.Pos objects; convert to NumPy array for easier use.
    mesh_centers = np.array([c.center() for c in mesh.cells()]) # Resulting shape (N_cells, 2) for 2D.
    # Retrieve the marker assigned to each cell. These markers correspond to those set in `createPolygon` calls.
    markers = np.array(mesh.cellMarkers()) # Array of integers, one per cell.
    
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
    # Get cell center coordinates from the mesh.
    # mesh.cellCenters() returns a list of pg.Pos objects (or similar for 2D).
    # It's better to convert to numpy array directly if all are same dimension.
    # For 2D meshes, cell centers are often returned as (x, y, 0.0) if using RVector3.
    # Or pg.Pos which has .x() .y() .z() methods.
    # Assuming .x() and .y() give horizontal and vertical respectively.
    _cell_centers_pg = mesh.cellCenters() # Get list of pg.Pos objects
    x_coords = np.array([c.x() for c in _cell_centers_pg]) # Extract x-coordinates
    z_coords = np.array([c.y() for c in _cell_centers_pg]) # Extract y-coordinates (used as z/depth)

    # Determine the minimum and maximum x-coordinates from cell centers to define the horizontal extent of the analysis.
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    
    # Create a series of x-coordinate bins (intervals) across the determined horizontal range.
    # These bins are used to segment the mesh vertically and find an interface point within each segment.
    x_bins = np.arange(x_min, x_max + interval, interval) # `interval` defines the width of each bin.
    
    # Initialize lists to store the (x, z) coordinates of the identified interface points.
    interface_x_list = [] # Stores the x-coordinate (center of the bin) for each found interface point.
    interface_z_list = [] # Stores the z-coordinate (interpolated depth) of the interface for that bin.
    
    # Iterate through each x-bin to find the depth where the velocity crosses the specified threshold.
    for i in range(len(x_bins)-1): # Loop over pairs of bin edges [x_bins[i], x_bins[i+1])
        # Find indices of mesh cells whose centers fall horizontally within the current x-bin.
        # Note: using a strict < for the upper bound of the bin.
        bin_indices = np.where((x_coords >= x_bins[i]) & (x_coords < x_bins[i+1]))[0]
        
        if len(bin_indices) > 0: # Proceed only if there are cells in the current bin
            # Get velocity values and z-coordinates (depths) for the cells within this bin.
            # Assumes `velocity_data` is a NumPy array ordered consistently with mesh cells, or indexable by `bin_indices`.
            bin_velocities = velocity_data[bin_indices]
            bin_depths = z_coords[bin_indices]
            
            # Sort these cells by depth (z_coords). This is crucial for correctly finding the first
            # crossing of the velocity threshold along a vertical profile within the bin.
            # The direction of sorting (ascending/descending) should match the physical setup (e.g., depth increasing downwards).
            # np.argsort default is ascending, which is fine if z_coords are depths (increasing downwards)
            # or elevations (increasing upwards) as long as consistent.
            sort_indices = np.argsort(bin_depths)
            bin_velocities_sorted = bin_velocities[sort_indices]
            bin_depths_sorted = bin_depths[sort_indices]
            
            # Iterate through the depth-sorted cells in the bin to find where velocity crosses the threshold.
            # Start from j=1 to compare cell j with cell j-1.
            for j in range(1, len(bin_velocities_sorted)):
                v_prev = bin_velocities_sorted[j-1] # Velocity of the cell "above" (or previous in sorted list)
                v_curr = bin_velocities_sorted[j]   # Velocity of the cell "below" (or current in sorted list)
                z_prev = bin_depths_sorted[j-1]     # Depth/elevation of the cell "above"
                z_curr = bin_depths_sorted[j]       # Depth/elevation of the cell "below"
                                                    # Original code had a typo `z2 = bin_depths[j-1]`, now corrected to `z_curr = bin_depths_sorted[j]`

                # Check if the threshold is crossed between the current cell (j) and the previous cell (j-1).
                # This handles crossings in both directions (e.g., velocity increasing or decreasing across threshold).
                is_crossing = (v_prev < threshold <= v_curr) or \
                              (v_prev >= threshold > v_curr) # Note: includes equality for one side to catch exact threshold hits.

                if is_crossing:
                    # Perform linear interpolation to find a more precise z-coordinate for the threshold crossing.
                    # Avoid division by zero if v_curr happens to be equal to v_prev (though crossing implies they should differ if threshold is between them).
                    if v_curr == v_prev:
                        # If velocities are same and they cross threshold (e.g. both are equal to threshold),
                        # take midpoint or one of the z values. Midpoint is safer.
                        interface_depth = (z_prev + z_curr) / 2.0
                    else:
                        # Calculate the interpolation weight/ratio.
                        ratio = (threshold - v_prev) / (v_curr - v_prev)
                        # Calculate the interpolated depth/elevation.
                        interface_depth = z_prev + ratio * (z_curr - z_prev)
                    
                    # Store the midpoint of the current x-bin as the x-coordinate for this interface point.
                    interface_x_list.append((x_bins[i] + x_bins[i+1]) / 2.0)
                    # Store the interpolated depth/elevation as the z-coordinate.
                    interface_z_list.append(interface_depth)
                    break # Exit the inner loop (over j); assumes we need the first crossing encountered in sorted order.

    # Convert lists of found interface points to numpy arrays for easier manipulation.
    interface_x = np.array(interface_x_list)
    interface_z = np.array(interface_z_list)

    # Extrapolation for missing start/end points:
    # Ensure the identified interface spans the entire model range from x_min to x_max.

    # Handle extrapolation at the beginning (x_min).
    if len(interface_x) == 0: # No interface points found at all.
        # If no points, cannot extrapolate. `x_dense` and `z_dense` will be based on empty/NaN data.
        pass # `interface_x` and `interface_z` remain empty.
    elif interface_x[0] > x_min + interval / 2.0: # If first found point is significantly far from x_min.
                                               # Using interval/2.0 as a heuristic for "significantly far".
        # If there's only one original point, extrapolate horizontally (constant z).
        if len(interface_x) == 1:
            extrapolated_z_start = interface_z[0]
        else: # More than one point, extrapolate using the slope defined by the first two original points.
            slope_start = (interface_z[1] - interface_z[0]) / (interface_x[1] - interface_x[0])
            extrapolated_z_start = interface_z[0] - slope_start * (interface_x[0] - x_min)
        # Insert the extrapolated point at the beginning of the interface arrays.
        interface_x = np.insert(interface_x, 0, x_min)
        interface_z = np.insert(interface_z, 0, extrapolated_z_start)

    # Handle extrapolation at the end (x_max).
    if len(interface_x) == 0: # Still no points (e.g., if previous block was skipped).
        pass
    elif interface_x[-1] < x_max - interval / 2.0: # If last found point is significantly far from x_max.
        # If there's only one point (could be after start extrapolation), extrapolate horizontally.
        if len(interface_x) == 1:
            extrapolated_z_end = interface_z[-1]
        else: # More than one point, extrapolate using the slope of the last two points.
            slope_end = (interface_z[-1] - interface_z[-2]) / (interface_x[-1] - interface_x[-2])
            extrapolated_z_end = interface_z[-1] + slope_end * (x_max - interface_x[-1])
        # Append the extrapolated point at the end of the interface arrays.
        interface_x = np.append(interface_x, x_max)
        interface_z = np.append(interface_z, extrapolated_z_end)
        # SUGGESTION: The original extrapolation logic using list.insert was more convoluted.
        # Switching to numpy array operations earlier (np.array(interface_x_list)) simplifies this.

    # Create a dense grid of x-coordinates for generating a smooth, interpolated interface line.
    x_dense = np.linspace(x_min, x_max, num=500)  # `num=500` points provide a visually smooth curve.

    # Interpolate the (potentially extrapolated) interface z-coordinates onto the dense x-grid.
    if len(interface_x) > 3: # Cubic interpolation generally requires at least 4 points for good behavior.
        try:
            # Attempt cubic spline interpolation for a smoother curve.
            # `bounds_error=False` and `fill_value="extrapolate"` allow interp1d to estimate values beyond the range of `interface_x`.
            interp_func_cubic = interp1d(interface_x, interface_z, kind='cubic',
                                         bounds_error=False, fill_value="extrapolate")
            z_dense_cubic = interp_func_cubic(x_dense)
            
            # Apply Savitzky-Golay filter for additional smoothing of the interpolated curve.
            # `window_length` must be odd and <= number of data points in `z_dense_cubic`.
            # `polyorder` must be < `window_length`.
            # SUGGESTION: Make window_length and polyorder parameters, or adapt them to data size more robustly.
            # For example, window_length could be min(31, len(z_dense_cubic) - (1 if len(z_dense_cubic) % 2 == 0 else 0) ).
            if len(z_dense_cubic) >= 31: # Ensure enough points for the default filter window (31).
                 z_dense_filtered = savgol_filter(z_dense_cubic, window_length=31, polyorder=3)
                 z_dense = z_dense_filtered
            else: # Not enough points for the specified Savitzky-Golay filter, use cubic result directly.
                z_dense = z_dense_cubic
        except Exception as e:
            # print(f"Cubic interpolation or Sav-Gol filter failed: {e}. Falling back to linear.") # Optional debug/info message
            # Fall back to linear interpolation if cubic interpolation or Sav-Gol filter fails.
            if len(interface_x) >= 2: # Linear interpolation needs at least 2 points.
                interp_func_linear = interp1d(interface_x, interface_z, kind='linear',
                                              bounds_error=False, fill_value="extrapolate")
                z_dense = interp_func_linear(x_dense)
            elif len(interface_x) == 1: # Only one point, create a flat line.
                 z_dense = np.full_like(x_dense, interface_z[0])
            else: # No points, fill with NaNs.
                z_dense = np.full_like(x_dense, np.nan)
    elif len(interface_x) >= 2 : # If 2 or 3 points, use linear interpolation.
        interp_func_linear = interp1d(interface_x, interface_z, kind='linear',
                                      bounds_error=False, fill_value="extrapolate")
        z_dense = interp_func_linear(x_dense)
    elif len(interface_x) == 1: # If only one interface point was found.
        z_dense = np.full_like(x_dense, interface_z[0]) # Create a flat interface at that depth/elevation.
    else: # No interface points were found at all (interface_x is empty).
        z_dense = np.full_like(x_dense, np.nan) # Return NaNs or a suitable default (e.g., a predefined average depth).
    
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
    # Create an initial "parameter mesh" (PLC - Planar Straight Line Complex) using PyGIMLi's utility for ERT data.
    # This function generates a geometry suitable for geophysical surveys, extending beyond sensor coverage.
    # `ertData`: PyGIMLi data container, used to get sensor positions.
    # `quality`: Controls the minimum angle of mesh triangles (e.g., 32 degrees for `createParaMeshPLC`).
    # `paraMaxCellSize`: Maximum cell size in the "parameter domain" (central survey area where parameters are typically estimated).
    # `paraBoundary`: Lateral extension (padding) of the parameter domain beyond the first/last sensor.
    # `paraDepth`: Depth of the parameter domain.
    # `boundaryMaxCellSize`: Maximum cell size at the far boundaries of the overall mesh.
    geo = mt.createParaMeshPLC(ertData, quality=32, paraMaxCellSize=30,
                               paraBoundary=paraBoundary, paraDepth=30.0,
                               boundaryMaxCellSize=500)
    
    # Combine the smoothed x and z coordinates of the velocity interface into a 2D numpy array of points (N_points, 2).
    # `smooth_x` and `smooth_z` are assumed to be 1D arrays of the same length representing the interface.
    interface_points = np.vstack((smooth_x, smooth_z)).T # .T transposes from (2, N_points) to (N_points, 2)

    # Extend the extracted velocity interface line horizontally at both ends.
    # This ensures the interface line spans the entire width of the geometry defined by `geo`,
    # including the `paraBoundary` extensions. The z-value (vertical coordinate) at the extended points
    # is kept the same as the first/last point of the original `interface_points`.
    # This creates `input_points` which will be added as a line to the geometry.
    # Note: `interface_points[0,0]` is x of first point, `interface_points[0,1]` is z of first point.
    # `interface_points[-1,0]` is x of last point, `interface_points[-1,1]` is z of last point.
    input_points = np.vstack((
        np.array([[interface_points[0,0] - paraBoundary, interface_points[0,1]]]), # Extend to the left: x is first_x - paraBoundary, z is first_z
        interface_points,                                                            # Original interface points
        np.array([[interface_points[-1,0] + paraBoundary, interface_points[-1,1]]])  # Extend to the right: x is last_x + paraBoundary, z is last_z
    ))
    
    # Create a PyGIMLi polygon object (which can also represent a polyline if not closed) for the extended velocity interface.
    # `isClosed=False` indicates it's a line segment, not a closed area to be filled.
    # `marker=99` is a temporary marker assigned to this line segment itself. This marker can be used by
    # the meshing algorithm to ensure the line is preserved (triangle edges align with it),
    # but it's not a region marker for cell assignment.
    interface_line_plc_segment = mt.createPolygon(input_points.tolist(), isClosed=False, # .tolist() converts numpy array to list of [x,z] pairs
                                                  interpolate='linear', marker=99)      # marker for the line itself

    # Add this interface line segment to the initial geometry (`geo`).
    # The `+` operator for PyGIMLi PLC objects merges them. The meshing algorithm will now "see" this line
    # and incorporate it into the mesh structure.
    geo_with_interface = geo + interface_line_plc_segment

    # Create a new mesh using the augmented geometry that now includes the velocity interface line.
    # The meshing algorithm will attempt to align triangle edges with this line.
    # `quality=28` (or another value) controls the quality (e.g., minimum angles) of the final mesh elements.
    meshafter = mt.createMesh(geo_with_interface, quality=28)
    
    # Initialize an array to store new cell markers for `meshafter`.
    # Default all cell markers to 1. This marker is intended for cells "Outside survey area".
    new_cell_markers = np.ones(meshafter.cellCount(), dtype=int)

    # Define the horizontal extent of the primary "survey area" for detailed marker assignment (above/below interface).
    # This uses the x-positions of the first and last ERT sensors from `ertData` and adjusts by `paraBoundary`.
    # This should ideally match the horizontal extent of the parametric domain set up by `createParaMeshPLC`.
    # `ertData.sensorPositions()` returns an array of sensor coordinates (pg.Pos objects or similar).
    # We assume `ertData.sensors()` gives a list/array of [x,y,z] or [x,y] sensor coordinates.
    sensor_x_coords = np.array([s[0] for s in ertData.sensors()]) # Get all x-coordinates of sensors
    survey_left_edge = np.min(sensor_x_coords) - paraBoundary   # Left edge of the survey area for detailed marking
    survey_right_edge = np.max(sensor_x_coords) + paraBoundary # Right edge

    # Iterate through each cell of the newly created mesh (`meshafter`) to assign specific markers.
    for i, cell_obj in enumerate(meshafter.cells()): # Iterate over cell objects
        cell_center_pos = cell_obj.center() # Get the center coordinates (pg.Pos object) of the cell
        cell_x = cell_center_pos.x()
        cell_y = cell_center_pos.y() # In PyGIMLi 2D meshes, 'y' is typically the vertical coordinate.
                                     # Convention: y is often elevation (positive upwards) or depth (positive downwards).
        
        # Check if the cell center is horizontally within the defined "survey area".
        if cell_x >= survey_left_edge and cell_x <= survey_right_edge:
            # If inside the survey area, determine if the cell is above or below the velocity interface.
            # Interpolate the y-coordinate (vertical position) of the interface at the cell's x-coordinate.
            # `input_points` contains the (x,y) coordinates of the (extended) velocity interface line.
            interface_y_at_cell_x = np.interp(cell_x, input_points[:, 0], input_points[:, 1])
            
            # Compare cell's y-coordinate with the interface's y-coordinate at that x.
            # Markers are assigned as:
            # - Marker 2: Inside survey area, above velocity line.
            # - Marker 3: Inside survey area, below velocity line.
            # This implies a coordinate system where "above" means a greater y-value if y is elevation,
            # or a smaller y-value if y is depth starting from 0 at the surface.
            # Assuming y is elevation (increases upwards, common in PyGIMLi meshes unless specified):
            if cell_y > interface_y_at_cell_x: # Cell center's y-value is greater than interface's y-value
                new_cell_markers[i] = 2  # Cell is physically above the interface line.
            else: # Cell center is below or on the interface line.
                new_cell_markers[i] = 3  # Cell is physically below the interface line.
        # Cells outside this horizontal range [survey_left_edge, survey_right_edge] will retain marker 1 by default.

    # Preserve original boundary markers (e.g., for padding cells defined by createParaMeshPLC).
    # `meshafter.cellMarkers()` at this point would reflect markers assigned by `mt.createMesh`
    # based on the regions in `geo_with_interface`. These are generally based on the `marker`
    # attribute of the `mt.createPolygon` calls used to build the PLC.
    # The `boundary` argument (default 1) is used to identify cells that should retain this specific marker,
    # potentially overriding the 1, 2, 3 logic above if those cells also fall into `boundary` regions.
    # This step ensures that cells designated by `createParaMeshPLC` as part of its specific boundary regions
    # (often padding or far-field cells, typically marked as 1) are correctly maintained.
    current_mesh_plc_markers = meshafter.cellMarkers() # Markers from PLC regions after meshing.
    for i in range(meshafter.cellCount()):
        if current_mesh_plc_markers[i] == boundary: # If the cell's original PLC region marker was `boundary` (e.g., 1)
            new_cell_markers[i] = boundary # Ensure this marker is preserved. This primarily affects cells outside survey_left/right
                                           # or cells within if their original PLC region marker was `boundary`.
                                           # SUGGESTION: The interaction could be complex. A clearer approach might be to
                                           # first assign based on PLC markers, then refine only a specific target PLC region.
                                           # However, this approach is common: set a base, refine, then restore specific original markers.

    # Apply the newly defined markers to the mesh object.
    meshafter.setCellMarkers(new_cell_markers)
    
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
        self.quality = quality # Store mesh quality parameter (e.g., min angle for triangles)
        self.area = area       # Store max cell area constraint for mesh generation
    
    def create_from_layers(self, surface: np.ndarray, 
                          layers: List[np.ndarray],
                          bottom_depth: float = 30.0,
                          markers: List[int] = None) -> Tuple[pg.Mesh, pg.meshtools.MeshPoly PLC]: # Corrected type hint for PLC
        """
        Create a mesh from surface topography and a list of subsurface layer boundary lines.
        
        Args:
            surface: Numpy array of shape (N, 2) defining surface coordinates [[x,z],...].
                     It's assumed z is elevation.
            layers: List of numpy arrays, where each array is shape (M, 2) defining a layer boundary [[x,z],...].
                    These should be ordered from shallowest to deepest.
            bottom_depth: Absolute elevation for the bottom of the entire mesh.
                          SUGGESTION: Consider if this should be relative depth below lowest layer or min surface.
                          Current use in `create_mesh_from_layers` treats it as absolute elevation.
            markers: Optional list of integer markers for each region created by the layers.
                     A "region" is the space between two consecutive boundaries (surface is the 0th boundary).
                     Number of regions = len(layers) + 1.
                     Example: surface, layer_boundary_1, layer_boundary_2 -> 3 regions are formed.
                              Region 0: Between surface and layers[0].
                              Region 1: Between layers[0] and layers[1].
                              Region 2: Below layers[1] down to bottom_depth.
                     So, `markers` list should have `len(layers) + 1` elements.
                     If None, a default marking scheme is applied: [2, 3, 2, 2, ...].
            
        Returns:
            Tuple containing the generated PyGIMLi mesh and the PLC (Planar Straight Line Complex) geometry object.
        """
        if len(layers) < 1:
            # If there are no subsurface layers, the behavior might be to mesh only down to bottom_depth based on surface.
            # However, the current structure, especially the default markers, implies at least one subsurface layer.
            # The downstream `create_mesh_from_layers` is hardcoded for 2 layers.
            raise ValueError("At least one subsurface layer boundary is required. This method is primarily for layered structures.")
            
        # --- Marker Setup ---
        # Create default region markers if not provided by the user.
        # Number of regions = number of layer boundaries + 1.
        num_regions = len(layers) + 1
        if markers is None:
            # Default marking: Most regions get marker 2.
            # The second region from the top (between first and second subsurface layer, or below first if only one) gets marker 3.
            # This is a common pattern where marker 3 might denote a specific layer of interest (e.g., an aquifer).
            effective_markers = [2] * num_regions
            if num_regions > 1: # If there is at least one subsurface layer (i.e., at least two regions)
                                # effective_markers[0] is for region: surface to layers[0]
                                # effective_markers[1] is for region: layers[0] to layers[1] (if exists) or layers[0] to bottom
                effective_markers[1] = 3
        else: # User provided markers
            if len(markers) != num_regions:
                raise ValueError(f"Length of markers list ({len(markers)}) does not match the number of regions ({num_regions}).")
            effective_markers = markers

        # --- Elevation Normalization (Currently Inactive) ---
        # The code includes commented-out lines for normalizing surface and layer elevations.
        # If active, `max_ele` (max surface elevation) would be subtracted from all z-coordinates,
        # making surface elevations <= 0 and effectively converting elevations to depths relative to the highest point.
        # This can be useful if subsequent calculations assume a common reference, but here it's off.
        # max_ele = np.nanmax(surface[:,1]) # Example: Find maximum surface elevation for normalization.
        surface_processed = surface.copy()
        # surface_processed[:,1] = surface_processed[:,1] - max_ele # Normalization is currently inactive.
        
        layers_processed = [] # List to hold processed layer boundary coordinates.
        for layer_boundary_points in layers:
            layer_norm_temp = layer_boundary_points.copy()
            # layer_norm_temp[:,1] = layer_norm_temp[:,1] - max_ele # Normalization is currently inactive.
            layers_processed.append(layer_norm_temp)
        
        # --- Mesh Creation Strategy ---
        # The current implementation has a specific path for a "2-layer" case, which means
        # two subsurface boundaries (i.e., `len(layers) == 2`). This results in three distinct regions:
        # 1. Region 0: Between surface and `layers_processed[0]`.
        # 2. Region 1: Between `layers_processed[0]` and `layers_processed[1]`.
        # 3. Region 2: Below `layers_processed[1]` down to `bottom_depth`.
        
        if len(layers_processed) == 2:
            # This specific case uses the standalone `create_mesh_from_layers` function.
            # Note: `create_mesh_from_layers` has its own hardcoded markers:
            #   - Region above first boundary (layers_processed[0]): marker 2
            #   - Region between first and second boundary: marker 3
            #   - Region below second boundary (layers_processed[1]): marker 2
            # These will override the `effective_markers` computed above unless `create_mesh_from_layers` is modified
            # or the markers are re-applied to the mesh after its creation.
            # SUGGESTION: For consistency, `create_mesh_from_layers` should accept a markers list,
            # or this function should re-apply `effective_markers` to the resulting mesh if user markers are prioritized.
            # The current implementation returns the mesh with markers as defined within `create_mesh_from_layers`.
            mesh, _, returned_markers_array, geom = create_mesh_from_layers( # Ignoring centers, using returned_markers for now
                surface_processed, layers_processed[0], layers_processed[1],
                bottom_depth=bottom_depth, quality=self.quality, area=self.area
            )
            # Example of how user-defined markers could be (conditionally) applied if desired:
            # if markers is not None: # If user actually supplied markers
            #     # This would require a mapping if create_mesh_from_layers internal regions don't align with `effective_markers` intent
            #     # For now, this is a placeholder for a more complex marker reconciliation logic if needed.
            #     # mesh.setCellMarkers(effective_markers_mapped_to_geometry)
            #     pass # Pass for now, as the policy on marker priority isn't fully clear.
            return mesh, geom # Return the mesh and the PLC geometry object.
        else:
            # Cases other than exactly two subsurface layers are not handled by the direct call above.
            # SUGGESTION: Generalize mesh creation for N layers. This would involve:
            # 1. Creating PLC polygons for each region defined by surface, layers, and bottom_depth.
            #    - Top region: bounded by surface, layers_processed[0], and model sides. Marker: effective_markers[0].
            #    - Intermediate regions (j): bounded by layers_processed[j-1], layers_processed[j], and model sides. Marker: effective_markers[j].
            #    - Bottom region: bounded by layers_processed[-1], bottom_depth line, and model sides. Marker: effective_markers[-1].
            # 2. Combining these polygons into a single PLC.
            # 3. Calling mt.createMesh() on the combined PLC.
            raise NotImplementedError("Currently, this method in MeshCreator only has a specific implementation for "
                                      "exactly 2 subsurface layer boundaries (resulting in three regions). "
                                      "A generalized solution for other numbers of layers is needed.")
    
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
