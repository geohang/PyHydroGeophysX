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
        pg.physics.traveltime.TravelTimeManager: The PyGIMLi TravelTimeManager object,
                                                 which contains the inversion results (e.g., velocity model,
                                                 achieved chi-squared, etc.).
    """
    # --- Set Default Inversion Parameters ---
    # These parameters control the behavior of the PyGIMLi travel time inversion.
    # Users can override these by passing them as keyword arguments (`**kwargs`).
    default_params = {
        'lam': 50,          # Regularization parameter (lambda): Controls the trade-off between data fit and model smoothness. Higher values mean smoother models.
        'zWeight': 0.2,     # Vertical regularization weight: Weight for vertical smoothness relative to horizontal. <1 means stronger horizontal smoothness.
        'vTop': 500,        # Velocity constraint for the top layer/region [m/s]. Used as a starting or reference velocity.
        'vBottom': 5000,    # Velocity constraint for the bottom layer/region [m/s].
        'quality': 31,      # Mesh quality parameter for `createParaMesh`, if mesh is auto-generated. (Not used if mesh is provided).
                            # See PyGIMLi documentation for `quality` parameter meaning.
        'paraDepth': 30.0,  # Maximum depth of the parametric domain for inversion if mesh is auto-generated [m].
        'verbose': 1,       # Verbosity level for PyGIMLi's inversion output (0=silent, 1=normal, >1=more verbose).
        'limits': [100., 6000.] # Min and max velocity limits [m/s] for the inverted model. Helps constrain results to physically plausible values.
        # SUGGESTION: 'maxIter' (maximum iterations) and 'robustData' (use robust L1-like data weighting) are also common and useful parameters.
    }
    
    # Update default parameters with any user-provided keyword arguments.
    # If a key exists in `kwargs`, its value will override the default.
    current_params = default_params.copy() # Start with defaults
    current_params.update(kwargs)      # Update with user inputs
    
    # --- Initialize PyGIMLi TravelTimeManager ---
    # The TravelTimeManager class is the main interface for SRT inversion in PyGIMLi.
    # It handles mesh setup, forward operator, inversion algorithm, etc.
    TT_manager = pg.physics.traveltime.TravelTimeManager()
    
    # --- Set or Create Mesh ---
    # If a mesh is provided by the user, set it in the manager.
    if mesh is not None:
        TT_manager.setMesh(mesh)
    else:
        # If no mesh is provided, PyGIMLi's ERTManager (often used as a base or utility within TTManager context for mesh)
        # or TravelTimeManager itself might create a default mesh based on the data geometry.
        # The original code has `pass` here, implying that if `mesh` is None,
        # `TT.invert` is expected to handle mesh creation internally, possibly using `paraDepth` and sensor locations.
        # This is typical PyGIMLi behavior: `TT.invert(data)` can create a mesh if one isn't set.
        # The `quality` and `paraDepth` parameters from `params` would be used by `createParaMesh` if called by `invert`.
        if current_params['verbose'] > 0:
            print("Mesh not provided by user. PyGIMLi's TravelTimeManager will attempt to create one based on data and parameters.")
        # SUGGESTION: Explicitly call `TT_manager.createMesh(ttData, quality=params['quality'], paraMaxCellSize=..., paraDepth=params['paraDepth'])`
        # here if more control over default mesh creation is desired before calling `invert`.
        # This would also allow returning the created mesh more easily if needed.
        pass
    
    # --- Run the Inversion ---
    # The `invert` method of TravelTimeManager performs the actual tomographic inversion.
    # It takes the travel time data (`ttData`) and various inversion control parameters.
    # The result (e.g., velocity model) is stored within the `TT_manager` object.
    TT_manager.invert(ttData,
                      lam=current_params['lam'],
                      zWeight=current_params['zWeight'],
                      vTop=current_params['vTop'],
                      vBottom=current_params['vBottom'],
                      verbose=current_params['verbose'],
                      limits=current_params['limits']
                      # Other parameters like maxIter, robustData, etc., can be passed if supported by TT.invert
                     )
    # After `invert` runs, `TT_manager.model` will contain the inverted slowness model,
    # and `TT_manager.coverage()` can provide resolution information.
    
    return TT_manager # Return the manager object, which now contains the inversion results.


def seismic_velocity_classifier(velocity_data, mesh, threshold=1200):
    """
    Classify mesh cells based on velocity threshold.
    
    Args:
        velocity_data: Velocity values for each cell
        mesh: PyGIMLi mesh
        threshold: Velocity threshold for classification (default: 1200)
        
    Returns:
        Array of cell markers (1: below threshold, 2: above threshold).
        Shape: (number_of_cells,).
    """
    # Initialize an array to store the classification markers for each cell.
    # Default marker is 1 (below threshold).
    classified_markers = np.ones_like(velocity_data, dtype=int)
    
    # Get the coordinates of the center of each cell in the mesh.
    # `mesh.cellCenters()` returns a list/array of pg.Pos objects or similar.
    # Convert to NumPy array for easier manipulation.
    cell_center_coords = np.array(mesh.cellCenters()) # Shape: (n_cells, 2) or (n_cells, 3)
    x_cell_coords = cell_center_coords[:,0]  # X-coordinates of cell centers.
    z_cell_coords = cell_center_coords[:,1]  # Z-coordinates (depths) of cell centers.
                                          # Assumes y-axis (index 1) is depth, typically negative downwards in PyGIMLi.
    
    # Get unique x-coordinates. This identifies unique vertical columns of cells in structured meshes,
    # or pseudo-columns if the mesh is unstructured but data is binned this way.
    unique_x_positions = np.unique(x_cell_coords)
    
    # --- Iterate Through Each Vertical Column (or pseudo-column) ---
    # The logic classifies cells based on whether any cell *above* them in the same column
    # (or at the same x-location and shallower depth) has crossed the velocity threshold.
    # This implies a layer-cake like classification from top down at each x-location.
    for x_pos in unique_x_positions:
        # Find indices of all cells that fall into the current vertical column (same x_pos).
        cells_in_column_indices = np.where(x_cell_coords == x_pos)[0]
        
        # Sort these cell indices by their depth (z_cell_coords).
        # This ensures processing from shallowest to deepest within the column.
        # `np.argsort` gives indices that would sort `z_cell_coords[cells_in_column_indices]`.
        sorted_order_in_column = np.argsort(z_cell_coords[cells_in_column_indices])
        # Re-index `cells_in_column_indices` to get original cell indices in depth-sorted order.
        depth_sorted_cell_indices_in_column = cells_in_column_indices[sorted_order_in_column]
        
        # Flag to track if the velocity threshold has been crossed by any cell above in this column.
        threshold_has_been_crossed_in_column = False

        # Process cells in this column from top (shallowest) to bottom (deepest).
        for cell_idx in depth_sorted_cell_indices_in_column:
            # If the current cell's velocity is above the threshold, OR
            # if a cell above it in this column already crossed the threshold,
            # then mark this cell (and all subsequent deeper cells in this column) as 'above threshold' (marker 2).
            if velocity_data[cell_idx] >= threshold or threshold_has_been_crossed_in_column:
                classified_markers[cell_idx] = 2 # Mark as 'above threshold'.
                threshold_has_been_crossed_in_column = True # Set flag for remaining deeper cells in this column.
            # Otherwise, it remains marker 1 (below threshold), as initialized.
    
    return classified_markers


def extract_velocity_structure(mesh, velocity_data, threshold=1200, interval=4.0):
    """
    Extract structure interface from velocity model at the specified threshold.
    
    Args:
        mesh: PyGIMLi mesh
        velocity_data: Velocity values for each cell
        threshold: Velocity threshold defining interface (default: 1200)
        interval: Horizontal sampling interval (default: 4.0)
        
    Returns:
        x_coords_smooth (np.ndarray): Smoothed horizontal coordinates of the interface.
        z_coords_smooth (np.ndarray): Smoothed vertical coordinates (depths) of the interface.
        interface_details (Dict): Dictionary containing raw and smoothed interface points, threshold, and extent.
    """
    # --- Get Cell Center Coordinates ---
    # This is similar to `seismic_velocity_classifier`.
    cell_center_coords = np.array(mesh.cellCenters())
    x_cell_coords = cell_center_coords[:,0]
    z_cell_coords = cell_center_coords[:,1] # Assumed to be depth, typically negative downwards.
    
    # Determine the horizontal extent (min and max x-coordinates) of the model.
    x_model_min, x_model_max = np.min(x_cell_coords), np.max(x_cell_coords)
    
    # --- Binning and Interface Picking ---
    # Create horizontal bins (intervals) across the model's x-range.
    # The interface depth will be determined within each bin.
    x_bins_edges = np.arange(x_model_min, x_model_max + interval, interval)
    
    # Lists to store the (x,z) coordinates of the raw, picked interface points.
    raw_interface_x_coords = []
    raw_interface_z_coords = []
    
    # Iterate through each bin along the x-axis.
    for i_bin in range(len(x_bins_edges)-1):
        # Define the current bin's horizontal extent.
        bin_x_start, bin_x_end = x_bins_edges[i_bin], x_bins_edges[i_bin+1]

        # Find indices of cells whose centers fall within the current x-bin.
        cells_in_bin_mask = (x_cell_coords >= bin_x_start) & (x_cell_coords < bin_x_end)
        cells_in_bin_indices = np.where(cells_in_bin_mask)[0]
        
        if len(cells_in_bin_indices) > 0: # If there are cells in this bin
            # Get velocities and depths (z-coordinates) for cells in this bin.
            velocities_in_bin = velocity_data[cells_in_bin_indices]
            depths_in_bin = z_cell_coords[cells_in_bin_indices]
            
            # Sort these cells by depth to facilitate finding the threshold crossing.
            # `np.argsort` gives indices that would sort `depths_in_bin`.
            depth_sorted_order = np.argsort(depths_in_bin)
            sorted_velocities_in_bin = velocities_in_bin[depth_sorted_order]
            sorted_depths_in_bin = depths_in_bin[depth_sorted_order]
            
            # --- Find Where Velocity Crosses the Threshold ---
            # Iterate through depth-sorted cells in the bin to find the first crossing.
            for j_cell_idx in range(1, len(sorted_velocities_in_bin)):
                v_cell_above = sorted_velocities_in_bin[j_cell_idx-1]
                v_cell_below = sorted_velocities_in_bin[j_cell_idx]
                z_cell_above = sorted_depths_in_bin[j_cell_idx-1]
                z_cell_below = sorted_depths_in_bin[j_cell_idx]

                # Check if the threshold is crossed between cell_above and cell_below.
                # This handles both increasing and decreasing velocity with depth at the threshold.
                if (v_cell_above < threshold <= v_cell_below) or \
                   (v_cell_above >= threshold > v_cell_below):
                    # --- Linear Interpolation for Exact Interface Depth ---
                    # Interpolate z where velocity = threshold, between z_cell_above and z_cell_below.
                    # Avoid division by zero if velocities are identical (should not happen if threshold is between them).
                    if np.isclose(v_cell_below, v_cell_above): # Velocities are too close to interpolate reliably.
                        # Take the average depth or depth of the cell closer to threshold. Here, average.
                        interpolated_interface_depth = (z_cell_above + z_cell_below) / 2.0
                    else:
                        # Ratio for linear interpolation: (target - v1) / (v2 - v1)
                        interpolation_ratio = (threshold - v_cell_above) / (v_cell_below - v_cell_above)
                        interpolated_interface_depth = z_cell_above + interpolation_ratio * (z_cell_below - z_cell_above)
                    
                    # Store the midpoint of the bin's x-range and the interpolated interface depth.
                    raw_interface_x_coords.append((bin_x_start + bin_x_end) / 2.0)
                    raw_interface_z_coords.append(interpolated_interface_depth)
                    break # Found interface in this bin, move to next bin.
    
    # --- Extrapolate Interface to Model Boundaries if Needed ---
    # Ensure the picked interface spans the entire horizontal model range (x_model_min to x_model_max).
    # This handles cases where no threshold crossing was found at the very start or end of the model.
    
    # Extrapolate to the beginning (x_model_min) if first picked point is beyond it.
    if raw_interface_x_coords and raw_interface_x_coords[0] > x_model_min + interval: # Check if first point is too far from min_x
        raw_interface_x_coords.insert(0, x_model_min) # Add x_model_min as the first x-coordinate.
        if len(raw_interface_x_coords) > 2: # Need at least two existing points (now at index 1 and 2) to extrapolate slope.
            # Slope between the (new) second and third points (original first and second).
            slope = (raw_interface_z_coords[1] - raw_interface_z_coords[0]) / (raw_interface_x_coords[2] - raw_interface_x_coords[1]) # Typo, should be raw_interface_x_coords[1] vs raw_interface_x_coords[2]
            # Corrected slope calculation for extrapolation: based on the first two *original* points (now indices 1 and 2)
            # slope = (raw_interface_z_coords[1] - raw_interface_z_coords[0]) / (raw_interface_x_coords[1] - raw_interface_x_coords[0]) # This uses the newly inserted x_min.
            # The original code had: slope = (interface_z[1] - interface_z[0]) / (interface_x[1] - interface_x[0])
            # which, after insertion, means slope between (x_model_min, z_unknown_yet) and (original_x0, original_z0).
            # This is not well-defined. A better way is to use the first two *actual* data points for slope.
            # Let's assume the code's intent was to use slope of the first segment of existing data.
            # If list was [x0, x1, ...], after insert: [x_min, x0, x1, ...]. Slope (z1-z0)/(x1-x0).
            # Extrapolated z = z0 - slope * (x0 - x_min).
            # The original code's `interface_z.insert(0, interface_z[0] - slope * (interface_x[1] - x_min))`
            # used `interface_z[0]` which was the z of the first original point.
            # This seems okay if `interface_z[0]` is the original first z, and `interface_x[1]` is original first x.
            original_first_z = raw_interface_z_coords[0] # This is actually the z of the point that *was* first.
            original_first_x = raw_interface_x_coords[1] # This is the x of the point that *was* first (now at index 1).
            if len(raw_interface_x_coords) > 2: # i.e. original list had at least 2 points
                slope = (raw_interface_z_coords[1] - original_first_z) / (raw_interface_x_coords[2] - original_first_x)
                extrap_z = original_first_z - slope * (original_first_x - x_model_min)
            else: # Original list had only one point
                extrap_z = original_first_z # Horizontal extrapolation
            raw_interface_z_coords.insert(0, extrap_z)
        elif raw_interface_z_coords: # Only one point originally, extrapolate horizontally
             raw_interface_z_coords.insert(0, raw_interface_z_coords[0])
        # Else: if raw_interface_x was empty, it remains empty.
    
    # Extrapolate to the end (x_model_max) if last picked point is before it.
    if raw_interface_x_coords and raw_interface_x_coords[-1] < x_model_max - interval:
        raw_interface_x_coords.append(x_model_max)
        if len(raw_interface_x_coords) > 2: # Need at least two points before this new one to define slope.
            # Slope between the (now) second-to-last and third-to-last points (original last two).
            slope = (raw_interface_z_coords[-2] - raw_interface_z_coords[-3]) / \
                    (raw_interface_x_coords[-2] - raw_interface_x_coords[-3])
            extrap_z = raw_interface_z_coords[-2] + slope * (x_model_max - raw_interface_x_coords[-2])
            raw_interface_z_coords.append(extrap_z)
        elif raw_interface_z_coords: # Only one point existed before adding x_model_max
            raw_interface_z_coords.append(raw_interface_z_coords[-2]) # Horizontal extrapolation (use z of previous point)
    
    # --- Interpolate and Smooth the Interface ---
    # Create a dense set of x-coordinates for a smooth interpolated interface line.
    x_coords_smooth = np.linspace(x_model_min, x_model_max, 500) # 500 points for smoothness.

    z_coords_smooth: np.ndarray
    if len(raw_interface_x_coords) > 3: # Cubic interpolation requires at least 4 points.
        try:
            # Perform cubic spline interpolation.
            # `bounds_error=False` allows extrapolation if x_dense is outside raw_interface_x range.
            # `fill_value="extrapolate"` tells interp1d how to extrapolate.
            cubic_interp_func = interp1d(raw_interface_x_coords, raw_interface_z_coords, kind='cubic',
                                         bounds_error=False, fill_value="extrapolate")
            z_coords_smooth = cubic_interp_func(x_coords_smooth)
            
            # Apply Savitzky-Golay filter for additional smoothing.
            # `window_length` must be odd and <= number of data points. `polyorder` < `window_length`.
            # SUGGESTION: Check if len(z_coords_smooth) is sufficient for savgol_filter window_length.
            # Make window_length adaptive or ensure enough points.
            if len(z_coords_smooth) >= 31: # Default window_length is 31.
                 z_coords_smooth = savgol_filter(z_coords_smooth, window_length=31, polyorder=3)
            elif len(z_coords_smooth) > 0 : # If fewer points than 31, try a smaller odd window
                 savgol_win_len = min(len(z_coords_smooth) - (1 if len(z_coords_smooth) % 2 == 0 else 0 ), 5) # Min window 5 or less
                 if savgol_win_len > 3: # Polyorder must be less than window
                    z_coords_smooth = savgol_filter(z_coords_smooth, window_length=savgol_win_len, polyorder=min(3,savgol_win_len-1))


        except Exception as e_smooth: # Catch errors during interpolation/smoothing (e.g., too few unique points for cubic).
            print(f"Warning: Cubic interpolation/smoothing failed ('{e_smooth}'). Falling back to linear interpolation.")
            # Fall back to linear interpolation if cubic or Savitzky-Golay fails.
            linear_interp_func = interp1d(raw_interface_x_coords, raw_interface_z_coords, kind='linear',
                                          bounds_error=False, fill_value="extrapolate")
            z_coords_smooth = linear_interp_func(x_coords_smooth)
    elif len(raw_interface_x_coords) > 0 : # If 1-3 points, use linear interpolation. (interp1d needs at least 1 point for fill_value, 2 for actual interpolation line)
        linear_interp_func = interp1d(raw_interface_x_coords, raw_interface_z_coords, kind='linear',
                                      bounds_error=False, fill_value="extrapolate")
        z_coords_smooth = linear_interp_func(x_coords_smooth)
    else: # No raw interface points found at all.
        print("Warning: No raw interface points found. Returning NaNs for smooth interface.")
        z_coords_smooth = np.full_like(x_coords_smooth, np.nan) # Fill with NaNs.
    
    # --- Prepare Output Dictionary ---
    # Contains raw picked points, smoothed points, threshold, and extent.
    interface_details = {
        'threshold': threshold,
        'raw_x': np.array(raw_interface_x_coords), # Ensure numpy array
        'raw_z': np.array(raw_interface_z_coords), # Ensure numpy array
        'smooth_x': x_coords_smooth,
        'smooth_z': z_coords_smooth,
        'min_x': x_model_min,
        'max_x': x_model_max
    }
    
    return x_coords_smooth, z_coords_smooth, interface_details


def save_velocity_structure(filename, x_coords, z_coords, interface_data=None):
    """
    Save velocity structure data to file.
    
    Args:
        filename: Output filename
        x_coords: X coordinates of interface
        z_coords: Z coordinates of interface
        interface_data: Additional data to save (optional)
    """
    # Create a dictionary containing the primary data to be saved.
    # This always includes the (smoothed) x and z coordinates of the interface.
    data_to_save = {
        'x_coords': x_coords, # Smoothed x-coordinates of the interface.
        'z_coords': z_coords  # Smoothed z-coordinates (depths) of the interface.
    }
    
    # If additional `interface_data` (like raw points, threshold) is provided, add it to the dictionary.
    # This `interface_data` typically comes from `extract_velocity_structure`.
    if interface_data is not None:
        data_to_save.update(interface_data) # Merge additional data into save_data.
    
    # --- Save Data ---
    # Save the data in NumPy's compressed .npz format.
    # This format can store multiple arrays in a single file.
    # `**save_data` unpacks the dictionary into keyword arguments for `np.savez`.
    np.savez(filename, **data_to_save)
    if os.path.exists(filename): print(f"Velocity structure data saved to {filename}")
    else: print(f"Failed to save velocity structure to {filename}")

    # Also save the primary smoothed interface (x,z coordinates) as a CSV file for easier human inspection or use in other software.
    # Replace the .npz extension (if present) with .csv.
    csv_filename = filename
    if filename.lower().endswith('.npz'):
        csv_filename = filename[:-4] + '.csv'
    else: # If no .npz extension, just append .csv (or user can provide full csv name)
        csv_filename = filename + '.csv'

    try:
        with open(csv_filename, 'w') as csv_file:
            csv_file.write('x,z\n') # Write header row for CSV.
            # Iterate through x and z coordinates and write each pair as a row.
            for x_val, z_val in zip(x_coords, z_coords):
                csv_file.write(f"{x_val},{z_val}\n")
        if os.path.exists(csv_filename): print(f"Smoothed interface also saved to CSV: {csv_filename}")
    except Exception as e_csv:
        print(f"Warning: Could not save interface to CSV file '{csv_filename}': {e_csv}")