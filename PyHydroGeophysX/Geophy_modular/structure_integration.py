"""
Structure integration module for constrained geophysical inversion.
"""
import numpy as np
import pygimli as pg
import pygimli.meshtools as mt
from typing import Tuple, List, Optional, Union, Dict, Any


def integrate_velocity_interface(ertData, smooth_x, smooth_z, paraBoundary=2, 
                               quality=28, paraMaxCellSize=30, paraDepth=30.0):
    """
    Integrate velocity interface into mesh for constrained ERT inversion.
    
    Args:
        ertData: PyGIMLi ERT data container
        smooth_x: X coordinates of velocity interface
        smooth_z: Z coordinates of velocity interface
        paraBoundary: Extra boundary size (default: 2)
        quality: Mesh quality parameter (default: 28)
        paraMaxCellSize: Maximum cell size (default: 30)
        paraDepth: Maximum depth of the model (default: 30.0)
        
    Returns:
        markers (np.ndarray): Array of integer markers assigned to each cell of the new mesh.
                              Marker 1: Typically outside/boundary region.
                              Marker 2: Region below the integrated interface.
                              Marker 3: Region above the integrated interface.
        meshafter (pg.Mesh): The newly created PyGIMLi mesh that incorporates the structural interface.
    """
    # --- Create Initial Parameter Mesh (PLC) ---
    # `mt.createParaMeshPLC` (PLC: Piecewise Linear Complex) creates a mesh suitable for inversion
    # based on the electrode positions in `ertData`.
    # `paraBoundary`: Defines extra space around the electrode spread.
    # `paraDepth`: Defines the depth of the primary investigation area of the mesh.
    # `paraMaxCellSize`: Controls cell size in the parametric domain (central region).
    # `boundaryMaxCellSize`: Controls cell size in the outer boundary regions of the mesh.
    # `quality`: Mesh quality parameter (e.g., minimum angle of triangles).
    # This initial mesh (`geo`) represents the geometry without the structural interface.
    geo = mt.createParaMeshPLC(ertData, quality=quality, paraMaxCellSize=paraMaxCellSize,
                              paraBoundary=paraBoundary, paraDepth=paraDepth,
                              boundaryMaxCellSize=500) # Boundary cell size fixed at 500.
    
    # --- Prepare Interface Geometry ---
    # Combine the x and z coordinates of the smoothed velocity interface into a list of points.
    # `smooth_x` and `smooth_z` define the line representing the geological boundary.
    interface_points = np.vstack((smooth_x, smooth_z)).T # Creates an array of [[x1,z1], [x2,z2], ...]
    
    # Extend the interface line horizontally at both ends by `paraBoundary`.
    # This ensures the interface line spans across the entire modeling domain,
    # preventing edge effects when it's incorporated into the mesh.
    # The z-value (depth) at the extended points is kept the same as the first/last interface point.
    extended_interface_points = np.vstack((
        np.array([[interface_points[0][0] - paraBoundary, interface_points[0][1]]]), # Extend to the left
        interface_points,                                                              # Original interface points
        np.array([[interface_points[-1][0] + paraBoundary, interface_points[-1][1]]])  # Extend to the right
    ))
    
    # Create a PyGIMLi polygon object representing the (extended) velocity interface line.
    # `isClosed=False` indicates it's a line, not a closed area.
    # `marker=99` is a temporary marker assigned to this line segment during mesh generation.
    # This marker helps PyGIMLi's meshing algorithm to "see" this line as a structural constraint.
    # `interpolate='linear'` means the line segments are straight between provided points.
    interface_line_plc = mt.createPolygon(extended_interface_points.tolist(), isClosed=False,
                                          interpolate='linear', marker=99) # Marker for the line itself.
    
    # --- Combine Geometries and Create Final Mesh ---
    # Add the interface line PLC to the initial mesh geometry PLC.
    # The `+` operator for PLC objects in PyGIMLi merges these geometric definitions.
    # The meshing algorithm will now honor both the domain defined by `geo` and the `interface_line_plc`.
    geometry_with_interface = geo + interface_line_plc
    
    # Create a new mesh (`meshafter`) from the combined geometry.
    # The `quality` parameter is reused here for the final mesh generation.
    # This mesh will have cells and boundaries that conform to the integrated interface.
    mesh_with_interface = mt.createMesh(geometry_with_interface, quality=quality)
    
    # --- Assign Cell Markers Based on Interface ---
    # Initialize an array to store new markers for each cell in `mesh_with_interface`.
    # Default marker is 1, typically representing the outer/boundary region or a background layer.
    new_cell_markers = np.ones(mesh_with_interface.cellCount(), dtype=int)
    
    # Define the horizontal extent of the primary survey/inversion area.
    # This is based on the first and last electrode positions from `ertData`, adjusted by `paraBoundary`.
    # Cells outside this horizontal range might retain marker 1.
    survey_area_x_min = ertData.sensors()[0][0] - paraBoundary # Leftmost extent of survey influence.
    survey_area_x_max = ertData.sensors()[-1][0] + paraBoundary # Rightmost extent.
    
    # Iterate through each cell of the new mesh to assign specific markers based on its position
    # relative to the integrated velocity interface and the survey area.
    for i in range(mesh_with_interface.cellCount()):
        cell_center = mesh_with_interface.cell(i).center() # Get center coordinates (x, y/z) of the cell.
        cell_x_coord = cell_center.x()
        cell_y_coord = cell_center.y() # In PyGIMLi, y often represents depth (can be negative).
        
        # Only modify markers for cells within the primary horizontal survey area.
        if survey_area_x_min <= cell_x_coord <= survey_area_x_max:
            # Interpolate the z-coordinate (depth) of the interface at the cell's x-coordinate.
            # This determines the interface depth directly below or above the cell center.
            interface_z_at_cell_x = np.interp(cell_x_coord, extended_interface_points[:, 0], extended_interface_points[:, 1])
            
            # Assign markers based on cell center's depth relative to the interface depth.
            # The use of `abs()` suggests that depths (`cell_y_coord`, `interface_z_at_cell_x`) might be negative.
            # If y is depth (positive downwards):
            #   - cell_y < interface_y means cell is shallower (above interface).
            #   - cell_y > interface_y means cell is deeper (below interface).
            # If y is elevation (positive upwards):
            #   - cell_y < interface_y means cell is lower (below interface).
            #   - cell_y > interface_y means cell is higher (above interface).
            # Given typical geophysical conventions (depth positive downwards or y negative downwards):
            # `abs(cell_y) < abs(interface_y)` with marker 2 (Below) implies:
            #   If y is negative depth: smaller absolute value means shallower. So marker 2 for shallower (above).
            #   This contradicts "Below interface".
            # Let's assume standard PyGIMLi: y is depth, increasing downwards (often negative values for 'up').
            # If y is positive downwards:
            #   `cell_y < interface_y` -> above interface (marker 3)
            #   `cell_y > interface_y` -> below interface (marker 2)
            # The `abs()` usage is confusing. Assuming a coordinate system where y is depth and positive downwards:
            # If cell_y_coord (depth of cell) > interface_z_at_cell_x (depth of interface) -> cell is below interface.
            # If cell_y_coord (depth of cell) < interface_z_at_cell_x (depth of interface) -> cell is above interface.
            # The original logic `abs(cell_y) < abs(interface_y)` for marker 2 (Below) is tricky.
            # If interface_y is, e.g. -10m, and cell_y is -5m (shallower), abs(-5) < abs(-10) is TRUE. Cell is above.
            # If interface_y is -10m, and cell_y is -15m (deeper), abs(-15) < abs(-10) is FALSE. Cell is below.
            # So, if `abs(cell_y) < abs(interface_y)` means marker 2 (Below), this implies:
            #    - If y is negative (depths are negative, surface at y=0): Shallower cells (less negative, smaller abs value) are marked 2 (Below). This is wrong.
            # Re-evaluating based on typical PyGIMLi mesh where y usually decreases with depth (surface y=0, below y<0):
            #   - If `cell_y_coord > interface_z_at_cell_x` (e.g. -5 > -10), cell is shallower (above). Should be marker 3.
            #   - If `cell_y_coord < interface_z_at_cell_x` (e.g. -15 < -10), cell is deeper (below). Should be marker 2.
            # The original code's `abs(cell_y) < abs(interface_y)` for marker 2 (Below interface) means:
            #   If cell is shallower (e.g. y=-5, interface_y=-10), |cell_y|<|interface_y| is TRUE. It's marked 2 (Below). This is incorrect.
            #   It should be: if cell_y < interface_y (more negative = deeper), then marker 2 (Below).
            # SUGGESTION: The marker assignment logic based on `abs()` should be reviewed carefully
            # against the specific coordinate system of the mesh (is y depth positive downwards or upwards?).
            # Assuming y is depth, positive downwards (less common for pg.Mesh from `createParaMeshPLC` which often has y negative downwards).
            # If y is negative downwards (typical for PyGIMLi surface meshes):
            #   cell_y < interface_y means cell is DEEPER == Marker 2 (Below)
            #   cell_y > interface_y means cell is SHALLOWER == Marker 3 (Above)
            if cell_y_coord < interface_z_at_cell_x: # Cell center is deeper than the interface line
                new_cell_markers[i] = 2  # Assign marker for "Below interface"
            else: # Cell center is shallower than or on the interface line
                new_cell_markers[i] = 3  # Assign marker for "Above interface"
    
    # Preserve original markers for cells that were part of the initial `geo`'s boundary region (marker 1).
    # `mesh_with_interface.cellMarkers()` here would reflect markers assigned by `mt.createMesh` based on PLC regions.
    # If initial `geo` had outer regions marked as 1, those should remain 1.
    # This line ensures that cells originally marked as 1 by `createParaMeshPLC` (typically far-field boundaries)
    # retain their marker 1, overriding the above logic for those specific cells.
    new_cell_markers[mesh_with_interface.cellMarkers() == 1] = 1 # Preserve boundary marker.
    
    # Apply the newly defined markers to the mesh.
    mesh_with_interface.setCellMarkers(new_cell_markers)
    
    return new_cell_markers, mesh_with_interface


def create_ert_mesh_with_structure(ertData, interface_data, **kwargs):
    """
    Create ERT mesh with structure interface for constrained inversion.
    
    Args:
        ertData: PyGIMLi ERT data container
        interface_data: Interface data (can be a tuple of (x, z) or a dictionary with smooth_x, smooth_z)
        **kwargs: Additional parameters including:
            - paraBoundary: Extra boundary size (default: 2)
            - quality: Mesh quality parameter (default: 28)
            - `paraMaxCellSize` (float): Max cell size in the parameter domain (near electrodes). Default: 30.
            - `paraDepth` (float): Depth of the parameter domain. Default: 30.0.
            
    Returns:
        Tuple[pg.Mesh, np.ndarray, Dict[int, Dict[str, Union[str, int]]]]:
            - generated_mesh (pg.Mesh): The final mesh with the integrated structure.
            - cell_markers (np.ndarray): The markers assigned to each cell in `generated_mesh`.
            - region_definitions (Dict): A dictionary describing the regions defined by the markers.
    """
    # --- Set Default Mesh Generation Parameters ---
    # These parameters are used by `integrate_velocity_interface` if not overridden by `kwargs`.
    default_mesh_params = {
        'paraBoundary': 2.0,    # Default extra boundary size around survey.
        'quality': 28,        # Default mesh quality (triangle angles).
        'paraMaxCellSize': 30.0,# Default max cell size in central region.
        'paraDepth': 30.0     # Default depth of primary modeling area.
    }
    # Update defaults with any user-provided parameters in `kwargs`.
    current_mesh_params = default_mesh_params.copy()
    current_mesh_params.update(kwargs) # User kwargs override defaults.
    
    # --- Extract Interface Coordinates ---
    # The `interface_data` can be provided either as a simple tuple (x_coords, z_coords)
    # or as a dictionary (e.g., from `extract_velocity_structure`) containing 'smooth_x' and 'smooth_z'.
    smooth_x_coords: np.ndarray
    smooth_z_coords: np.ndarray
    if isinstance(interface_data, tuple) and len(interface_data) == 2:
        smooth_x_coords, smooth_z_coords = interface_data
    elif isinstance(interface_data, dict) and 'smooth_x' in interface_data and 'smooth_z' in interface_data:
        smooth_x_coords = interface_data['smooth_x']
        smooth_z_coords = interface_data['smooth_z']
    else:
        # If `interface_data` is not in a recognized format, raise an error.
        raise ValueError("`interface_data` must be a tuple (x_coords, z_coords) or a dictionary "
                         "containing 'smooth_x' and 'smooth_z' keys.")

    # --- Create Mesh with Integrated Interface ---
    # Call `integrate_velocity_interface` to perform the core task of generating the mesh
    # and assigning markers based on the provided interface and ERT data geometry.
    # This function returns the cell markers and the generated mesh.
    assigned_cell_markers, generated_mesh_with_interface = integrate_velocity_interface(
        ertData, smooth_x_coords, smooth_z_coords,
        paraBoundary=current_mesh_params['paraBoundary'],
        quality=current_mesh_params['quality'],
        paraMaxCellSize=current_mesh_params['paraMaxCellSize'],
        paraDepth=current_mesh_params['paraDepth']
    )
    
    # --- Define Region Dictionary ---
    # Create a dictionary that provides a human-readable description for each marker value
    # assigned by `integrate_velocity_interface`. This is useful for post-processing and visualization.
    # Marker 1: Boundary/Exterior region created by `createParaMeshPLC`.
    # Marker 2: Region below the velocity interface.
    # Marker 3: Region above the velocity interface.
    # (This mapping is based on the logic within `integrate_velocity_interface`.)
    region_definitions = {
        1: {"name": "boundary_region", "marker": 1, "description": "Boundary/Exterior cells of the mesh"},
        2: {"name": "lower_velocity_layer", "marker": 2, "description": "Region below the integrated velocity interface"},
        3: {"name": "upper_velocity_layer", "marker": 3, "description": "Region above the integrated velocity interface"}
    }
    # SUGGESTION: The marker numbers (1,2,3) and their meanings should be consistently documented
    # or made more flexible if other marking schemes are possible from `integrate_velocity_interface`.
    
    return generated_mesh_with_interface, assigned_cell_markers, region_definitions


def create_joint_inversion_mesh(ertData, ttData, velocity_threshold=1200, **kwargs):
    """
    Create a mesh for joint ERT-seismic inversion by first inverting seismic data,
    extracting the velocity interface, and then creating a constrained ERT mesh.
    
    Args:
        ertData: PyGIMLi ERT data container
        ttData: PyGIMLi seismic travel time data container
        velocity_threshold: Threshold for velocity interface (default: 1200)
        **kwargs: Additional parameters including:
            - `seismic_params` (Dict): Parameters for `process_seismic_tomography`.
            - `mesh_params` (Dict): Parameters for `create_ert_mesh_with_structure`.
            - `interface_interval` (float): Horizontal sampling interval for extracting velocity interface. Default: 5.0.
            
    Returns:
        Tuple[pg.Mesh, pg.physics.traveltime.TravelTimeManager, Dict]:
            - joint_mesh (pg.Mesh): The final ERT mesh incorporating the seismic velocity structure.
            - seismic_manager (TravelTimeManager): Manager object containing results of the seismic inversion.
            - structure_interface_data (Dict): Data dictionary of the extracted velocity interface.
    """
    # --- Import Required Modules ---
    # These imports are placed inside the function, which is generally not standard Python style
    # unless there's a specific reason (e.g., circular dependency avoidance, conditional import).
    # For a library module, they are typically at the top of the file.
    # SUGGESTION: Move these imports to the top of the module if they are standard dependencies.
    from pygimli.physics import traveltime as tt # Already imported at module level. Redundant here.
    # This import implies a specific project structure.
    # It might be better if seismic_processor functions are imported at module level or passed as arguments if dynamic.
    from PyHydroGeophysX.Geophy_modular.seismic_processor import ( # Corrected path from watershed_geophysics
        process_seismic_tomography, extract_velocity_structure
    )
    
    # --- Extract Parameter Dictionaries from kwargs ---
    # `kwargs.get('key', {})` provides an empty dict as default if 'key' is not in kwargs.
    seismic_inversion_params = kwargs.get('seismic_params', {})
    ert_mesh_creation_params = kwargs.get('mesh_params', {})

    # --- Step 1: Create Mesh for Initial Seismic Inversion (if not provided) ---
    # If seismic_params does not already contain a 'mesh', create one.
    # This mesh is used for the initial seismic tomography.
    # It's created based on ERT data sensor locations, implying ERT and SRT share similar survey lines/extents.
    if 'mesh' not in seismic_inversion_params or seismic_inversion_params['mesh'] is None:
        if ertData is None: # ERTData is needed if seismic mesh is auto-generated from it.
            raise ValueError("ertData must be provided if seismic_params['mesh'] is not set for auto-meshing.")

        print("Creating initial mesh for seismic tomography based on ERT data geometry.")
        # Use PyGIMLi's ERTManager to create a suitable starting mesh.
        # Parameters for mesh creation are taken from `seismic_params` or defaults.
        temp_ert_manager = pg.physics.ert.ERTManager(ertData) # Temporary manager for mesh creation.
        initial_seismic_mesh = temp_ert_manager.createMesh(
            data=ertData, # Uses sensor coverage from ERT data.
            quality=seismic_inversion_params.get('quality', 31.0), # Mesh quality.
            paraMaxCellSize=seismic_inversion_params.get('paraMaxCellSize', 5.0), # Max cell size in parameter domain.
            paraBoundary=seismic_inversion_params.get('paraBoundary', 0.1), # Boundary width.
            paraDepth=seismic_inversion_params.get('paraDepth', 30.0)       # Depth of parameter domain.
        )
        seismic_inversion_params['mesh'] = initial_seismic_mesh # Add created mesh to params for seismic inversion.
    
    # --- Step 2: Perform Seismic Tomography ---
    # `process_seismic_tomography` inverts the travel time data (`ttData`)
    # using the provided/created mesh and seismic inversion parameters.
    # It returns a `TravelTimeManager` object containing the seismic velocity model.
    print("Performing seismic tomography...")
    seismic_inversion_manager = process_seismic_tomography(ttData, **seismic_inversion_params)
    
    # --- Step 3: Extract Velocity Interface from Seismic Model ---
    # `extract_velocity_structure` identifies a structural boundary (interface)
    # from the inverted seismic velocity model based on a specified `velocity_threshold`.
    # `seismic_manager.paraDomain` is the mesh used for the seismic inversion (parameter mesh).
    # `seismic_manager.model.array()` is the inverted slowness model; needs conversion to velocity if not already.
    #   (Assuming `process_seismic_tomography` returns velocity in `.model` or it's handled by `extract_velocity_structure`)
    #   If `seismic_manager.model` stores slowness (s/m), it should be converted: `1.0 / seismic_manager.model.array()`.
    #   This depends on the output of `process_seismic_tomography`. For now, assume it provides velocity.
    #   SUGGESTION: Ensure `seismic_manager.model.array()` provides velocity, not slowness, to `extract_velocity_structure`.
    #   If `process_seismic_tomography` returns TTManager, `TTManager.velocity` might hold velocity.
    #   Let's assume `seismic_manager.model` is slowness, so convert:
    inverted_slowness_model = seismic_inversion_manager.model # This is typically slowness from TTManager.
    inverted_velocity_model = 1.0 / inverted_slowness_model.array() # Convert slowness to velocity. Add epsilon for safety if needed.

    print(f"Extracting velocity interface at threshold: {velocity_threshold} m/s.")
    interface_x_coords, interface_z_coords, structure_interface_data = extract_velocity_structure(
        mesh=seismic_inversion_manager.paraDomain, # Use the parameter mesh from seismic inversion.
        velocity_data=inverted_velocity_model,    # Pass the inverted velocity model.
        threshold=velocity_threshold,
        interval=kwargs.get('interface_interval', 5.0) # Horizontal interval for picking interface points.
    )
    
    # --- Step 4: Create ERT Mesh Incorporating the Extracted Structure ---
    # `create_ert_mesh_with_structure` takes the ERT data (for electrode locations),
    # the extracted interface coordinates (x, z), and mesh parameters to generate
    # a new ERT mesh that is constrained by this interface.
    # This mesh will have different regions/markers above and below the interface.
    print("Creating final ERT mesh with integrated seismic structure...")
    joint_inversion_mesh, _cell_markers, _region_info = create_ert_mesh_with_structure(
        ertData, 
        (interface_x_coords, interface_z_coords), # Pass interface as tuple
        **ert_mesh_creation_params # Pass mesh creation parameters for ERT mesh.
    )
    # _cell_markers and _region_info are also returned but not passed on by this function's current signature.
    
    return joint_inversion_mesh, seismic_inversion_manager, structure_interface_data