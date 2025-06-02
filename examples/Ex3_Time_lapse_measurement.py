# Script-level comment:
# This script, Ex3_Time_lapse_measurement.py, focuses on generating synthetic
# time-lapse Electrical Resistivity Tomography (ERT) data. It simulates a common
# scenario in hydrogeophysics where changes in subsurface water content over time
# are monitored using ERT.
#
# Workflow steps covered:
# 1. Setup: Defines paths, loads initial domain information (idomain, topography, porosity),
#    and sets up a 2D profile and geophysical mesh, similar to Ex2_workflow.py.
#    This establishes the spatial framework for the time-lapse simulation.
# 2. Time-series data processing:
#    - Loads a time-series of 3D water content data (presumably from a hydrological model like MODFLOW).
#    - For each timestep:
#        - Interpolates the 3D water content to the 2D profile.
#        - Maps the profile water content data onto the 2D geophysical mesh.
#        - Converts the mesh-based water content (and static porosity) to a resistivity model
#          using petrophysical relationships (e.g., Archie's law or Waxman-Smits).
#        - Saves the generated water content and resistivity models for each timestep.
# 3. Synthetic ERT data generation (commented out but shown as example):
#    - Optionally, for each timestep's resistivity model, ERT forward modeling can be performed
#      to generate a synthetic ERT dataset (apparent resistivities).
#    - Includes examples for both non-parallel and parallel (using joblib) processing
#      of ERT forward modeling across multiple timesteps. This is crucial for efficiency
#      when dealing with many time steps.
# 4. Visualization:
#    - Demonstrates loading and plotting one of the saved synthetic ERT datasets.
#    - Shows how to load all synthetic apparent resistivity data and visualize them as a
#      time-lapse pseudosection (apparent resistivity vs. time vs. measurement number),
#      often plotted alongside precipitation data for context.
#    - Visualizes the generated water content and resistivity models on the mesh for selected timesteps.
#    - Creates an animation (GIF) of the water content changes on the mesh over the entire time series.
#
# Assumptions:
# - Similar to Ex2, this script assumes example data files are in a specific directory structure.
#   Hardcoded absolute paths are used for `output_dir` and `data_dir`.
#   - `examples/data/id.txt`, `top.txt`, `Porosity.npy`, `bot.npy` (as in Ex2).
#   - `examples/data/Watercontent.npy`: A 4D NumPy array representing time-series water content
#     (e.g., [time, nlay, nrow, ncol]).
#   - `examples/data/precip.npy`: Precipitation data for plotting alongside time-lapse results.
# - The script demonstrates saving intermediate models (water content, resistivity per timestep)
#   and synthetic ERT data, which can be useful for detailed analysis or as input for time-lapse inversion.
#
# Expected output:
# - Console output: Progress messages for data processing and (if uncommented) parallel ERT simulation.
# - Saved files:
#   - `mesh.bms`: The 2D geophysical mesh.
#   - `synwcmodel/synwcmodel[i].npy`: Water content on mesh for each timestep `i`.
#   - `synresmodel/synresmodel[i].npy`: Resistivity model on mesh for each timestep `i`.
#   - (If ERT simulation part is run) `appres/synthetic_data[i].dat`: Synthetic ERT data for timestep `i`.
#   - `apparent_resistivity.tiff`: Plot of time-lapse apparent resistivity vs. precipitation.
#   - `water_content_model.tiff`: Plot of water content models at selected timesteps.
#   - `resistivity_model.tiff`: Plot of resistivity models at selected timesteps.
#   - `WCanimation.gif`: Animation of water content changes over time.
# - Matplotlib plots: Various plots displayed during script execution, including example synthetic ERT data,
#   time-series plots, and model snapshots.

"""
Ex3. Creating Synthetic Time-Lapse ERT Measurements
==============================================

This example demonstrates how to create synthetic time-lapse electrical 
resistivity tomography (ERT) measurements for watershed monitoring applications.

The example covers:
1. Loading time-series water content data from MODFLOW simulations
2. Converting water content to resistivity for each timestep
3. Forward modeling to generate synthetic ERT data for multiple time periods
4. Parallel processing for efficient computation across multiple timesteps
5. Visualization of apparent resistivity changes over time
6. Creating animations showing temporal water content variations

This workflow is essential for testing time-lapse inversion algorithms
and understanding the sensitivity of ERT measurements to hydrological changes.
"""

# %%
# --- Standard library and third-party imports ---
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pygimli as pg # PyGIMLi for geophysical modeling, mesh handling, and ERT functionalities
from pygimli.physics import ert # Specifically for ERT related tasks
from mpl_toolkits.axes_grid1 import make_axes_locatable # For advanced colorbar placement

# --- Setup package path for development ---
# This ensures the script can find the PyHydroGeophysX package when run directly,
# especially if the package is not formally installed in the Python environment.
try:
    # This works when running as a standard Python script.
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # This fallback is for environments like Jupyter notebooks where __file__ is not defined.
    current_dir = os.getcwd()

# Add the parent directory (expected to be the project root containing PyHydroGeophysX) to sys.path.
parent_dir = os.path.dirname(current_dir) # Navigate one level up from "examples"
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# --- Import PyHydroGeophysX specific modules ---
# `MODFLOWWaterContent`: For loading water content data (though not directly used for initial loading in this script, kept for context).
from PyHydroGeophysX.model_output.modflow_output import MODFLOWWaterContent
# Core utilities for creating profiles, interpolating data, and mesh generation.
from PyHydroGeophysX.core.interpolation import ProfileInterpolator, create_surface_lines
from PyHydroGeophysX.core.mesh_utils import MeshCreator
# Petrophysical model for converting water content to resistivity.
from PyHydroGeophysX.petrophysics.resistivity_models import water_content_to_resistivity
# ERT forward modeling class (though PyGIMLi's native tools are also used directly).
from PyHydroGeophysX.forward.ert_forward import ERTForwardModeling

# %%
# --- Define Output Directory ---
# Specifies where all results (plots, data files, animations) from this example will be saved.
# IMPORTANT: This uses a hardcoded absolute path, which will likely need to be changed by the user
# to match their own system's directory structure.
# SUGGESTION: Use a relative path for better portability, e.g.:
# output_dir = os.path.join(current_dir, "results", "Ex3_TL_measurements_results")
output_dir = "C:/Users/HChen8/Documents/GitHub/PyHydroGeophysX/examples/results/TL_measurements" # User-specific path.
os.makedirs(output_dir, exist_ok=True) # Create the directory if it doesn't exist, no error if it does.

# %%
# --- Step 1: Setup Profile and Mesh (similar to Ex2_workflow.py) ---
# This section establishes the 2D spatial domain (profile and mesh) onto which
# time-varying hydrological data will be mapped.
print("Step 1: Set up the ERT profiles like in the workflow example.")

# Define path to example data files.
# IMPORTANT: This is also a hardcoded absolute path.
# SUGGESTION: Use data_dir = os.path.join(current_dir, "data")
data_dir = "C:/Users/HChen8/Documents/GitHub/PyHydroGeophysX/examples/data/"
modflow_dir = os.path.join(data_dir, "modflow") # Subdirectory for MODFLOW related example files.

# Load domain information from files:
# - `idomain`: Defines active/inactive cells in the original 3D hydrological model.
# - `top`: Represents the elevation of the top surface of the 3D model.
# - `porosity`: Array of porosity values for the 3D domain.
# Users should replace these with paths to their actual data files.
idomain = np.loadtxt(os.path.join(data_dir, "id.txt"))      # Expected: 2D array (nrows, ncols).
top = np.loadtxt(os.path.join(data_dir, "top.txt"))          # Expected: 2D array (nrows, ncols).
porosity = np.load(os.path.join(data_dir, "Porosity.npy"))  # Expected: 3D array (nlay, nrows, ncol).

# Define profile endpoints using grid indices [column_index, row_index] from the hydrological model.
point1 = [115, 70]  # Start point of the 2D profile.
point2 = [95, 180]  # End point of the 2D profile.

# Initialize ProfileInterpolator for handling geometric transformations and interpolation to the profile.
# Parameters define the geometry and origin of the source 3D hydrological model grid.
interpolator = ProfileInterpolator(
    point1=point1,
    point2=point2,
    surface_data=top, # `top` array defines the surface for profile extraction.
    origin_x=569156.2983333333, # Real-world X-coordinate of the grid origin.
    origin_y=4842444.17,    # Real-world Y-coordinate of the grid origin.
    pixel_width=1.0,        # Cell width in the hydrological model grid.
    pixel_height=-1.0       # Cell height (negative if Y-axis origin is top-left).
                            # num_points for profile discretization defaults to 200 if not specified.
)

# Interpolate the 3D porosity field onto the 2D profile line.
# Result `porosity_profile` will have shape (nlay_hydro, num_profile_points).
porosity_profile = interpolator.interpolate_3d_data(porosity)


# Load 3D data for subsurface layer boundaries (e.g., bottom elevation of each layer).
# `bot.npy` is assumed to store these boundaries.
bot = np.load(os.path.join(data_dir, "bot.npy")) # Expected: (num_boundaries, nlay_hydro, ncol_hydro)

# Interpolate these 3D layer boundaries onto the 2D profile.
# `structure` will store elevations of all boundaries along the profile: (num_boundaries_total, num_profile_points).
structure = interpolator.interpolate_layer_data([top] + bot.tolist()) # Prepend `top` as the uppermost boundary.

# Create specific surface and layer lines for defining the geophysical mesh geometry.
# Indices `top_idx`, `mid_idx`, `bot_idx` select which boundaries from `structure`
# will define the main geological units in the mesh (e.g., surface, top of layer A, top of layer B).
top_idx=int(0)     # Index for ground surface in `structure`.
mid_idx=int(4)     # Index for top of a middle layer (e.g., fractured bedrock).
bot_idx=int(12)    # Index for top of a bottom layer (e.g., fresh bedrock).
surface, line1, line2 = create_surface_lines(
    L_profile=interpolator.L_profile, # Distances along the profile (X-coordinates for mesh).
    structure=structure,              # All interpolated layer boundary elevations.
    top_idx=top_idx, # Use the defined indices
    mid_idx=mid_idx,
    bot_idx=bot_idx
)

# Create the 2D geophysical mesh using MeshCreator.
# The mesh will conform to `surface`, `line1`, and `line2`.
mesh_creator = MeshCreator(quality=32) # `quality` controls mesh element shape.
mesh, geom = mesh_creator.create_from_layers( # `geom` is the PyGIMLi PLC object.
    surface=surface,
    layers=[line1, line2], # Define two internal boundaries.
    bottom_depth= np.min(line2[:,1])-10 # Mesh extends 10 units below the lowest point of `line2`.
)

# Save the generated mesh to a file.
mesh.save(os.path.join(output_dir, "mesh.bms")) # Saved in PyGIMLi binary mesh format.


# --- Prepare for mapping hydrological properties to mesh cells ---
# Create an `ID1` array to assign material/layer IDs to the `porosity_profile` data.
# These IDs help map hydrological model layers to geophysical mesh regions/markers during interpolation.
# The markers (0, 3, 2) should correspond to how regions are defined/marked in `mesh`.
# `mid_idx` and `bot_idx` (relative to the full stack of hydro layers in `porosity_profile`)
# are used here to define which parts of the `porosity_profile` belong to which material ID.
ID1 = porosity_profile.copy() # Initialize with same shape as porosity_profile.
ID1[:mid_idx] = 0             # Material ID 0 (e.g., regolith) for hydro layers above mid_idx.
ID1[mid_idx:bot_idx] = 3      # Material ID 3 (e.g., fractured bedrock) for hydro layers between mid_idx and bot_idx.
ID1[bot_idx:] = 2             # Material ID 2 (e.g., fresh bedrock) for hydro layers below bot_idx.

# Get cell centers and markers from the generated geophysical mesh.
# These are needed for the `interpolate_to_mesh` function.
mesh_centers = np.array(mesh.cellCenters()) # Coordinates of mesh cell centers.
mesh_markers = np.array(mesh.cellMarkers()) # Markers assigned to mesh cells by MeshCreator.

# Interpolate the profiled porosity data (which now has material IDs in `ID1`) onto the geophysical mesh cells.
# `layer_markers` specifies which material IDs from `ID1` should be processed and mapped.
porosity_mesh = interpolator.interpolate_to_mesh(
    property_values=porosity_profile, # Porosity data along the profile.
    depth_values=structure,           # Depths/elevations of hydro layers along profile.
    mesh_x=mesh_centers[:, 0],        # X-coordinates of geophysical mesh cell centers.
    mesh_y=mesh_centers[:, 1],        # Y-coordinates of geophysical mesh cell centers.
    mesh_markers=mesh_markers,        # Markers of geophysical mesh cells (target regions).
    ID=ID1,                           # Source material/layer IDs assigned to profile data points.
    layer_markers = [0,3,2]           # List of unique material IDs in `ID1` to be interpolated.
)

# Load the full time-series of 3D water content data.
# `Water_Content.npy` is assumed to be a 4D array: [time_index, nlay_hydro, nrow_hydro, ncol_hydro].
Water_Content = np.load(os.path.join(data_dir, "Watercontent.npy"))

# %%
# --- Create Subdirectories for Saving Time-Lapse Models ---
# These directories will store the water content and resistivity models generated for each timestep.
# `exist_ok=True` means `makedirs` won't raise an error if directories already exist.
# SUGGESTION: Use `output_dir` as the base for these paths for consistency.
# E.g., os.makedirs(os.path.join(output_dir, "synwcmodel"), exist_ok=True)
os.makedirs("results/TL_measurements/synwcmodel", exist_ok=True) # Path is relative to where script is run from, not `output_dir`.
os.makedirs("results/TL_measurements/synresmodel", exist_ok=True)


# --- Process Each Timestep: Interpolate WC, Convert to Resistivity, Save Models ---
# This loop iterates through each timestep in the `Water_Content` array.
for i in range(len(Water_Content)): # `len(Water_Content)` gives the number of timesteps.
    water_content_timestep = Water_Content[i] # Extract 3D water content for the current timestep `i`.

    # Interpolate this 3D water content to the 2D profile.
    water_content_profile = interpolator.interpolate_3d_data(water_content_timestep)

    # Interpolate the profiled water content onto the 2D geophysical mesh.
    # `ID1` (material IDs for profile layers) and `mesh_markers` guide this interpolation.
    wc_mesh = interpolator.interpolate_to_mesh(
        property_values=water_content_profile,
        depth_values=structure,           # Using the same overall layer structure definition.
        mesh_x=mesh_centers[:, 0],
        mesh_y=mesh_centers[:, 1],
        mesh_markers=mesh_markers,
        ID=ID1,                           # Material IDs for profile data.
        layer_markers=[0, 3, 2]           # Unique material IDs to process.
    )

    # Convert the mesh-based water content to a resistivity model using petrophysical parameters.
    # `marker_labels` should align with the markers in `mesh_markers` and the order of `rho_sat`, `n_val`, `sigma_s`.
    marker_labels = [0, 3, 2]       # Markers for different geological units in the mesh.
    rho_sat = [100, 500, 2400]      # Saturated resistivity for each unit [Ohm-m]. Example values.
    n_val = [2.2, 1.8, 2.5]         # Saturation exponent (n) for each unit. Example values.
    sigma_s = [0.002, 0, 0]         # Surface conductivity for each unit [S/m]. Example values.
                                    # Note: Original script comment mentioned `sigma_s` was [1/500, 0, 0].

    res_models = np.zeros_like(wc_mesh) # Initialize resistivity model array for the current timestep.

    # Apply petrophysical conversion for each layer/marker.
    # Layer 1 (marker_labels[0], e.g., regolith)
    mask_layer0 = (mesh_markers == marker_labels[0])
    if np.any(mask_layer0): # Process only if cells with this marker exist.
        res_models[mask_layer0] = water_content_to_resistivity(
            wc_mesh[mask_layer0],
            float(rho_sat[0]),
            float(n_val[0]),
            porosity_mesh[mask_layer0], # Use the static porosity model for this layer.
            sigma_s[0]
        )

    # Layer 2 (marker_labels[1], e.g., fractured bedrock)
    mask_layer1 = (mesh_markers == marker_labels[1])
    if np.any(mask_layer1):
        res_models[mask_layer1] = water_content_to_resistivity(
            wc_mesh[mask_layer1],
            float(rho_sat[1]),
            float(n_val[1]),
            porosity_mesh[mask_layer1],
            sigma_s[1]
        )

    # Layer 3 (marker_labels[2], e.g., fresh bedrock)
    mask_layer2 = (mesh_markers == marker_labels[2])
    if np.any(mask_layer2):
        res_models[mask_layer2] = water_content_to_resistivity(
            wc_mesh[mask_layer2],
            float(rho_sat[2]),
            float(n_val[2]),
            porosity_mesh[mask_layer2],
            sigma_s[2]
        )

    # Save the generated water content model and resistivity model for the current timestep `i`.
    # Files are saved as .npy arrays in the subdirectories created earlier.
    # SUGGESTION: Use `output_dir` as base for consistency.
    # E.g., np.save(os.path.join(output_dir, "synwcmodel", f"synwcmodel{i}.npy"), wc_mesh)
    np.save(os.path.join("results/TL_measurements/synwcmodel", f"synwcmodel{i}" ), wc_mesh) # Using f-string for filename
    np.save(os.path.join("results/TL_measurements/synresmodel", f"synresmodel{i}" ), res_models) # Using f-string


# %%
# --- Non-parallel ERT Forward Modeling (Commented Out) ---
# This section shows how to perform ERT forward modeling for each timestep sequentially.
# It's commented out in the original script, likely because a parallel version is preferred for efficiency.
# os.makedirs("results/TL_measurements/appres", exist_ok=True) # Ensure output dir for apparent resistivity data.

# for i in range(2): # Example: only process first 2 timesteps for this demo.
#     # Load the previously saved resistivity model for timestep `i`.
#     # res_model = np.load(os.path.join(output_dir, "synresmodel/synresmodel" + str(i) + ".npy")) # Added .npy extension
#     xpos = np.linspace(15,15+72 - 1,72) # Define electrode x-positions.
#     ypos = np.interp(xpos,interpolator.L_profile,interpolator.surface_profile) # Drape electrodes on surface.
#     pos = np.hstack((xpos.reshape(-1,1),ypos.reshape(-1,1))) # Combine into (N,2) array.

#     schemeert = ert.createData(elecs=pos,schemeName='wa') # Create ERT measurement scheme.

#     # Prepare mesh for forward modeling (add boundary, set markers).
#     # `mesh` is the geophysical mesh created earlier.
#     mesh.setCellMarkers(np.ones(mesh.cellCount())*2) # Mark all core cells as region 2.
#     grid = pg.meshtools.appendTriangleBoundary(mesh, marker=1, # Boundary cells get marker 1.
#                                             xbound=100, ybound=100) # Define extent of boundary.

#     # The ERTForwardModeling class from PyHydroGeophysX is instantiated but not directly used for `fob.response`.
#     # fwd_operator = ERTForwardModeling(mesh=grid, data=schemeert) # This line is effectively unused if fob below is used.

#     # Use PyGIMLi's ERTModelling for the forward calculation.
#     synth_data = schemeert.copy() # Create a data container for results.
#     fob = ert.ERTModelling()      # Initialize PyGIMLi's forward operator.
#     fob.setData(schemeert)        # Assign measurement scheme.
#     fob.setMesh(grid)             # Assign the mesh with boundary.
#     # dr = fob.response(res_model)  # Calculate "true" apparent resistivities using the resistivity model.

#     # Add synthetic noise to the data.
#     # dr *= 1. + pg.randn(dr.size()) * 0.05 # 5% relative Gaussian noise.
#     # ert_manager = ert.ERTManager(synth_data) # For error estimation.
#     # synth_data['rhoa'] = dr
#     # synth_data['err'] = ert_manager.estimateError(synth_data, absoluteUError=0.0, relativeError=0.05) # Assign errors.

#     # Save the synthetic ERT data for this timestep.
#     # synth_data.save(os.path.join(output_dir, "appres/synthetic_data"+str(i)+".dat"))
#     pass # Pass to indicate this block is intentionally commented out / conceptual.

# %%
# --- Parallel ERT Forward Modeling (Commented Out) ---
# This section outlines how to parallelize the ERT forward modeling across multiple timesteps
# using `joblib.Parallel` and `joblib.delayed`. This can significantly speed up computations
# for large numbers of timesteps. The actual execution is commented out.

# import os # Already imported.
# import numpy as np # Already imported.
# import pygimli as pg # Already imported.
# from pygimli.physics import ert # Already imported.
# from joblib import Parallel, delayed # For parallel processing.

# def process_timestep(i, output_dir_parallel, mesh_filepath, interpolator_L_profile_data, interpolator_surface_profile_data): # Renamed args for clarity
#     """
#     Processes a single timestep: loads resistivity model, sets up ERT survey,
#     runs forward model, adds noise, and saves synthetic ERT data.
#     This function is designed to be called in parallel for each timestep.
#     """
#     try:
#         # Load the resistivity model for the current timestep `i`.
#         res_model_path = os.path.join(output_dir_parallel, "synresmodel", f"synresmodel{i}.npy") # Used f-string
#         res_model = np.load(res_model_path)
        
#         # Create electrode positions (draped on surface).
#         xpos = np.linspace(15, 15 + 72 - 1, 72)
#         ypos = np.interp(xpos, interpolator_L_profile_data, interpolator_surface_profile_data)
#         pos = np.hstack((xpos.reshape(-1,1), ypos.reshape(-1,1)))
        
#         # Create ERT measurement scheme.
#         schemeert = ert.createData(elecs=pos, schemeName='wa')
        
#         # Load the base mesh (saved earlier).
#         mesh_loaded = pg.load(mesh_filepath) # Renamed
#         # Prepare mesh for forward modeling (set core/boundary markers).
#         mesh_loaded.setCellMarkers(np.ones(mesh_loaded.cellCount())*2) # Core cells marker 2.
#         grid_fwd = pg.meshtools.appendTriangleBoundary(mesh_loaded, marker=1, xbound=100, ybound=100) # Boundary marker 1. Renamed.
        
#         # Set up PyGIMLi's forward operator.
#         fwd_operator_parallel = ert.ERTModelling() # Renamed
#         fwd_operator_parallel.setData(schemeert)
#         fwd_operator_parallel.setMesh(grid_fwd)
        
#         # Perform forward modeling to get true apparent resistivities.
#         synth_data_container = schemeert.copy() # Renamed
#         true_rhoa_parallel = fwd_operator_parallel.response(res_model) # Renamed
        
#         # Add 5% relative Gaussian noise.
#         noisy_rhoa_parallel = true_rhoa_parallel * (1. + pg.randn(true_rhoa_parallel.size()) * 0.05) # Renamed
        
#         # Store noisy data and estimate errors for the data container.
#         ert_manager_parallel = ert.ERTManager(synth_data_container) # Renamed
#         synth_data_container['rhoa'] = noisy_rhoa_parallel
#         synth_data_container['err'] = ert_manager_parallel.estimateError(synth_data_container, absoluteUError=0.0, relativeError=0.05)
        
#         # Save the synthetic ERT data for this timestep.
#         # Ensure the "appres" subdirectory exists within `output_dir_parallel`.
#         appres_dir_parallel = os.path.join(output_dir_parallel, "appres") # Renamed
#         os.makedirs(appres_dir_parallel, exist_ok=True)
#         synth_data_container.save(os.path.join(appres_dir_parallel, f"synthetic_data{i}.dat")) # Used f-string
        
#         return i, True, None  # Return timestep index, success status, and no error.
#     except Exception as e:
#         return i, False, str(e)  # Return timestep index, failure status, and error message.

# %%
# --- Execute Parallel Processing (Commented Out) ---
# This block would set up and run the parallel ERT forward modeling.
# It's commented out, so it won't run as part of this script execution.

# # Create the specific output directory for apparent resistivity results if it doesn't exist.
# # appres_output_dir = os.path.join(output_dir, "appres") # Renamed
# # os.makedirs(appres_output_dir, exist_ok=True)


# # # Extract necessary data from the `interpolator` object to pass to parallel workers.
# # # This is because the full `interpolator` object might not be easily serializable by joblib.
# # interpolator_L_profile_data = interpolator.L_profile.copy() # Pass a copy of the L_profile array.
# # interpolator_surface_profile_data = interpolator.surface_profile.copy() # Pass a copy of surface_profile.
# # mesh_filepath_for_parallel = os.path.join(output_dir, "mesh.bms") # Path to the saved mesh.

# # # Use joblib to run `process_timestep` in parallel for each timestep.
# # # `n_jobs=2`: Use 2 CPU cores. Adjust as needed (e.g., -1 to use all available cores).
# # # `verbose=10`: Print progress messages.
# # # `Water_Content.shape[0]` gives the total number of timesteps.
# # results_parallel = Parallel(n_jobs=2, verbose=10)( # Renamed
# #     delayed(process_timestep)(
# #         i,
# #         output_dir, # Pass the main output directory.
# #         mesh_filepath_for_parallel,
# #         interpolator_L_profile_data,
# #         interpolator_surface_profile_data
# #     ) for i in range(Water_Content.shape[0]) # Iterate over all timesteps.
# # )

# # # Check and report results from parallel processing.
# # success_count = sum(1 for _, success, _ in results_parallel if success)
# # print(f"Successfully processed {success_count} out of {len(results_parallel)} timesteps in parallel.")

# # # Print any errors that occurred during parallel execution.
# # for i, success, error_msg in results_parallel: # Renamed loop var
# #     if not success:
# #         print(f"Error in parallel processing of timestep {i}: {error_msg}")
# pass # Indicates this conceptual block is complete.


# %%
# --- Load and Show Example Synthetic ERT Data ---
# This loads and displays one of the synthetic ERT datasets, assuming the forward modeling
# (either sequential or parallel version) was run and saved the .dat files.
# If the forward modeling parts are commented out, this cell might error if files don't exist.
# It attempts to load data for timestep 1.
example_timestep_to_load = 1
synthetic_data_example_path = os.path.join(output_dir, "appres", f"synthetic_data{example_timestep_to_load}.dat") # Used f-string

# Check if the example data file exists before trying to load it.
if os.path.exists(synthetic_data_example_path):
    syn_data_example = pg.load(synthetic_data_example_path) # Renamed
    ert.show(syn_data_example, label="Apparent Resistivity ($\Omega$m)") # Added label
    plt.title(f"Synthetic ERT Data (Timestep {example_timestep_to_load})")
    plt.show()
else:
    print(f"Synthetic ERT data file not found (expected at {synthetic_data_example_path}). "
          "Skipping plot. Please run the ERT forward modeling section first.")

# %%
# --- Load All Synthetic ERT Data for Time-Lapse Plotting ---
# This loop attempts to load all generated synthetic ERT data files (one per timestep)
# and stores the apparent resistivity values in a list.
all_syn_data_rhoa = [] # Renamed
# Iterate through all timesteps based on the shape of the loaded `Water_Content` array.
for i in range(Water_Content.shape[0]):
    try:
        # Construct the path to the .dat file for the current timestep `i`.
        data_file_path = os.path.join(output_dir, "appres", f"synthetic_data{i}.dat") # Used f-string
        if os.path.exists(data_file_path):
            syn_data_timestep = pg.load(data_file_path) # Renamed
            all_syn_data_rhoa.append(np.array(syn_data_timestep['rhoa'])) # Append apparent resistivity array.
        else:
            print(f"Warning: Synthetic ERT data for timestep {i} not found at {data_file_path}. Skipping.")
            # Optionally, append NaNs or a placeholder if maintaining full list length is crucial.
            # all_syn_data_rhoa.append(np.nan) # Example placeholder
    except Exception as e:
        print(f"Error loading synthetic ERT data for timestep {i}: {e}")
        # all_syn_data_rhoa.append(np.nan) # Example placeholder

# %%
# --- Plot Time-Lapse Apparent Resistivity Data with Precipitation ---
# This section visualizes the changes in apparent resistivity over time,
# alongside precipitation data for hydro-climatic context.
import pandas as pd # For handling time series data (dates).
import matplotlib.pylab as pylab # For advanced plot styling.

# Define custom Matplotlib parameters for plot appearance.
params = {'legend.fontsize': 13,
          #'figure.figsize': (15, 5), # Figure size is set locally below.
         'axes.labelsize': 13,
         'axes.titlesize':13,
         'xtick.labelsize':13,
         'ytick.labelsize':13}
pylab.rcParams.update(params) # Apply the custom parameters.
plt.rcParams["font.family"] = "Arial" # Set global font family.


# Create a date range corresponding to the time-lapse period (assumed to be 365 days).
# This is example data; real applications would use actual dates.
# rng = pd.date_range(start="09/01/2011", end="08/30/2012", freq="D") # This is unused if x-axis is just days.
# Load precipitation data (assumed to be a NumPy array of daily values).
precip_data = np.load(os.path.join(data_dir, "precip.npy")) # Renamed
# Convert the list of apparent resistivity arrays into a 2D NumPy array.
# Each row corresponds to a measurement configuration, each column to a timestep.
# This might fail if not all data files were loaded (e.g., if `all_syn_data_rhoa` contains NaNs or is ragged).
# Filter out potential None/NaN entries if some files were missing
valid_syn_data_arrays = [arr for arr in all_syn_data_rhoa if isinstance(arr, np.ndarray)]
if valid_syn_data_arrays:
    syn_data_rhoa_array = np.array(valid_syn_data_arrays) # Renamed
    print(f"Shape of stacked apparent resistivity data: {syn_data_rhoa_array.shape}") # (n_timesteps, n_measurements)
else:
    syn_data_rhoa_array = np.array([]) # Empty array if no data loaded
    print("Warning: No synthetic ERT data loaded for time-lapse plot.")


# Create the figure for time-lapse apparent resistivity and precipitation.
plt.figure(figsize=(12, 6))

# Subplot 1 (Top): Precipitation data.
plt.subplot(211) # 2 rows, 1 column, 1st plot.
# Assumes `precip_data` has length corresponding to number of days (e.g., 365).
plt.bar(np.arange(len(precip_data)), precip_data, color='k') # Bar chart for precipitation.
plt.xlim([0, len(precip_data)-1]) # Set x-axis limits.
plt.ylabel('Precipitation (mm)')
plt.xlabel('Time (days)') # X-axis shows days from start.

# Subplot 2 (Bottom): Time-lapse apparent resistivity pseudosection.
plt.subplot(212) # 2 rows, 1 column, 2nd plot.
if syn_data_rhoa_array.ndim == 2 and syn_data_rhoa_array.shape[0] > 0: # Check if array is valid 2D
    # `imshow` displays the data: rows are measurement configurations, columns are time.
    # `.T` transposes so time is on x-axis, measurements on y-axis.
    # `pg.utils.cMap('rhoa')` provides a suitable colormap for resistivity.
    # `vmin`, `vmax` set the color scale limits.
    plt.imshow(syn_data_rhoa_array.T, aspect='auto', cmap=pg.utils.cMap('rhoa'), vmin=200, vmax=2000)
    plt.colorbar(label='Apparent Resistivity ($\Omega$m)') # Add colorbar.
else:
    plt.text(0.5, 0.5, "Apparent resistivity data not available or invalid.", horizontalalignment='center', verticalalignment='center')
plt.ylabel('Measurement #') # Y-axis represents different ERT measurement configurations.
plt.xlabel('Time (days)')   # X-axis is time.

plt.tight_layout() # Adjust layout.
plt.savefig(os.path.join(output_dir, "apparent_resistivity_timelapse.tiff"), dpi=300) # Save figure. Renamed.
plt.show() # Display plot.

# %%
# --- Additional Plot: Apparent Resistivity Pseudosection (Standalone) ---
# This cell seems to be for generating a larger, standalone version of the apparent resistivity
# time-lapse plot, possibly for better detail in the colorbar or presentation.
plt.figure(figsize=(12, 6)) # New figure.
# plt.subplot(211) # This was likely a copy-paste error, should be just one plot or adjust subplotting.
if syn_data_rhoa_array.ndim == 2 and syn_data_rhoa_array.shape[0] > 0:
    plt.imshow(syn_data_rhoa_array.T, aspect='auto', cmap=pg.utils.cMap('rhoa'), vmin=200, vmax=2000)
    plt.colorbar(label='Apparent Resistivity (Ω·m)')
    plt.ylabel('Measurement #')
    plt.xlabel('Time (days)')
    plt.title("Time-Lapse Apparent Resistivity") # Added title
else:
    plt.text(0.5, 0.5, "Apparent resistivity data not available or invalid.", horizontalalignment='center', verticalalignment='center')
    plt.title("Time-Lapse Apparent Resistivity (Data Missing)")
plt.show()

# %%
# This cell seems to be empty or incomplete in the original script.

# %%
# --- Visualize Water Content Models at Selected Timesteps ---
# This section plots the water content distribution on the geophysical mesh
# for four selected timesteps (Day 30, 150, 210, 320) to show temporal changes.
fig_wc_timesteps, axes_wc_timesteps = plt.subplots(1, 4, figsize=(16, 14)) # Renamed for clarity

# Import a specific colormap for water content visualization.
from palettable.lightbartlein.diverging import BlueDarkRed18_18_r # Example colormap.
fixed_cmap_wc = BlueDarkRed18_18_r.mpl_colormap # Renamed for clarity.

# Timestep 1: Day 30
ax_wc_ts1 = axes_wc_timesteps[0] # Renamed for clarity
# Load the saved water content model for day 30 (timestep index 30).
# Assuming file synwcmodel30.npy exists in the "synwcmodel" subdirectory.
# SUGGESTION: Use output_dir as base path. E.g. os.path.join(output_dir, "synwcmodel", "synwcmodel30.npy")
wc_day30 = np.load(os.path.join("results/TL_measurements/synwcmodel", "synwcmodel30.npy")) # Renamed
# Display the water content on the mesh using PyGIMLi's show function.
pg.show(mesh, wc_day30, ax=ax_wc_ts1, cMap=fixed_cmap_wc, logScale=False,
        cMin=0.0, cMax=0.32, label='Water Content (-)',
        xlabel='Distance (m)', ylabel='Elevation (m)')
ax_wc_ts1.set_title("Day 30")


# Timestep 2: Day 150
ax_wc_ts2 = axes_wc_timesteps[1] # Renamed
wc_day150 = np.load(os.path.join("results/TL_measurements/synwcmodel", "synwcmodel150.npy")) # Renamed
pg.show(mesh, wc_day150, ax=ax_wc_ts2, cMap=fixed_cmap_wc, logScale=False,
        cMin=0.0, cMax=0.32, label='Water Content (-)',
        xlabel='Distance (m)', ylabel='Elevation (m)')
ax_wc_ts2.set_title("Day 150")


# Timestep 3: Day 210
ax_wc_ts3 = axes_wc_timesteps[2] # Renamed
wc_day210 = np.load(os.path.join("results/TL_measurements/synwcmodel", "synwcmodel210.npy")) # Renamed
pg.show(mesh, wc_day210, ax=ax_wc_ts3, cMap=fixed_cmap_wc, logScale=False,
        cMin=0.0, cMax=0.32, label='Water Content (-)',
        xlabel='Distance (m)', ylabel='Elevation (m)')
ax_wc_ts3.set_title("Day 210")


# Timestep 4: Day 320 (original script had 320, file name might be synwcmodel320.npy)
ax_wc_ts4 = axes_wc_timesteps[3] # Renamed
# Original script loaded "synwcmodel320.npy" but title was "Day 330". Assuming data for day 320.
wc_day320 = np.load(os.path.join("results/TL_measurements/synwcmodel", "synwcmodel320.npy")) # Renamed
pg.show(mesh, wc_day320, ax=ax_wc_ts4, cMap=fixed_cmap_wc, logScale=False,
        cMin=0.0, cMax=0.32, label='Water Content (-)',
        xlabel='Distance (m)', ylabel='Elevation (m)')
ax_wc_ts4.set_title("Day 320") # Corrected title to match data if it's day 320.

fig_wc_timesteps.tight_layout() # Adjust layout.
plt.savefig(os.path.join(output_dir, "water_content_snapshots.tiff"), dpi=300) # Renamed file for clarity
plt.show() # Display plot.

# %%
# --- Visualize Resistivity Models at Selected Timesteps ---
# This section plots the resistivity distribution on the geophysical mesh
# for four selected timesteps (Day 30, 150, 210, 330) to show temporal changes.
fig_res_timesteps, axes_res_timesteps = plt.subplots(1, 4, figsize=(16, 14)) # Renamed

# Import a different colormap for resistivity visualization.
from palettable.lightbartlein.diverging import BlueDarkRed18_18 # Example colormap (non-reversed version).
fixed_cmap_res = BlueDarkRed18_18.mpl_colormap # Renamed

# Timestep 1: Day 30
ax_res_ts1 = axes_res_timesteps[0] # Renamed
# Load the saved resistivity model for day 30.
# SUGGESTION: Use `output_dir` as base path.
res_day30 = np.load(os.path.join("results/TL_measurements/synresmodel", "synresmodel30.npy")) # Renamed
pg.show(mesh, res_day30, ax=ax_res_ts1, cMap=fixed_cmap_res, logScale=False, showColorBar=True, # logScale often True for resistivity
        xlabel="Distance (m)", ylabel="Elevation (m)",
        label='Resistivity (Ω·m)', cMin=100, cMax=3000) # Define color scale limits.
ax_res_ts1.set_title("Day 30")


# Timestep 2: Day 150
ax_res_ts2 = axes_res_timesteps[1] # Renamed
res_day150 = np.load(os.path.join("results/TL_measurements/synresmodel", "synresmodel150.npy")) # Renamed
pg.show(mesh, res_day150, ax=ax_res_ts2, cMap=fixed_cmap_res, logScale=False, showColorBar=True,
        xlabel="Distance (m)", ylabel="Elevation (m)",
        label='Resistivity (Ω·m)', cMin=100, cMax=3000)
ax_res_ts2.set_title("Day 150")


# Timestep 3: Day 210
ax_res_ts3 = axes_res_timesteps[2] # Renamed
res_day210 = np.load(os.path.join("results/TL_measurements/synresmodel", "synresmodel210.npy")) # Renamed
pg.show(mesh, res_day210, ax=ax_res_ts3, cMap=fixed_cmap_res,
        logScale=False, showColorBar=True,
        xlabel="Distance (m)", ylabel="Elevation (m)",
        label='Resistivity (Ω·m)', cMin=100, cMax=3000)
ax_res_ts3.set_title("Day 210")


# Timestep 4: Day 330
ax_res_ts4 = axes_res_timesteps[3] # Renamed
res_day330 = np.load(os.path.join("results/TL_measurements/synresmodel", "synresmodel330.npy")) # Renamed
pg.show(mesh, res_day330, ax=ax_res_ts4, cMap=fixed_cmap_res, logScale=False, showColorBar=True,
        xlabel="Distance (m)", ylabel="Elevation (m)",
        label='Resistivity (Ω·m)', cMin=100, cMax=3000)
ax_res_ts4.set_title("Day 330")


fig_res_timesteps.tight_layout() # Adjust layout.
plt.savefig(os.path.join(output_dir, "resistivity_model_snapshots.tiff"), dpi=300) # Renamed file for clarity
plt.show() # Display plot.

# %%
# --- Create Animation (GIF) of Water Content Changes Over Time ---
# This section generates a GIF animation showing how water content on the mesh
# evolves through all 365 simulated days.
# import numpy as np # Already imported
# import matplotlib.pyplot as plt # Already imported
# import os # Already imported
from PIL import Image # For image manipulation (creating GIF).
import io # For handling byte streams (saving plots to buffer).

# Import the colormap used for water content plots.
from palettable.lightbartlein.diverging import BlueDarkRed18_18_r
fixed_cmap_animation = BlueDarkRed18_18_r.mpl_colormap # Renamed

# List to store individual frames (images) of the animation.
frames = []

# Set the DPI (dots per inch) for consistent figure size in the animation.
dpi = 100

# Loop through all 365 timesteps to create each frame of the animation.
num_timesteps_animation = 365 # Assuming 365 days of data.
for i in range(num_timesteps_animation):
    # Print progress update every 10 frames.
    if i % 10 == 0:
        print(f"Processing animation frame {i} of {num_timesteps_animation}")
        
    # Set up a new Matplotlib figure for each frame.
    # Figure size is adjusted to minimize empty space around the plot.
    fig_anim_frame = plt.figure(figsize=[8, 2.2]) # Renamed
    
    # Adjust subplot parameters to use more of the figure space.
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    
    ax_anim_frame = fig_anim_frame.add_subplot(1, 1, 1) # Renamed
    
    # Load the water content data for the current timestep `i`.
    # Assumes files are named synwcmodel0.npy, synwcmodel1.npy, etc.
    # SUGGESTION: Use `output_dir` as base path.
    wc_anim_frame = np.load(os.path.join("results/TL_measurements/synwcmodel", f"synwcmodel{i}.npy")) # Renamed, used f-string
    
    # Plot the water content data on the mesh for the current frame.
    # `pad=0.3` adds padding around the colorbar.
    # Labels and ticks are removed to save space and focus on the visual change.
    pg.show(mesh, wc_anim_frame, pad=0.3, orientation="vertical",
            cMap=fixed_cmap_animation, cMin=0.00, cMax=0.32, # Fixed color scale for consistency.
            xlabel="", ylabel="",  # Remove axis labels.
            label='Water content', ax=ax_anim_frame) # Pass the axes object.
    
    # Style adjustments to make the animation cleaner.
    ax_anim_frame.spines['top'].set_visible(False)
    ax_anim_frame.spines['right'].set_visible(False)
    ax_anim_frame.spines['bottom'].set_visible(False)
    ax_anim_frame.spines['left'].set_visible(False)
    ax_anim_frame.get_xaxis().set_ticks([]) # Remove x-axis ticks.
    ax_anim_frame.get_yaxis().set_ticks([]) # Remove y-axis ticks.
    
    # Add a day counter text to the plot for temporal reference.
    # `transform=ax.transAxes` positions text relative to axes dimensions (0,0 is bottom-left, 1,1 is top-right).
    ax_anim_frame.text(0.1, 0.1, f'Day: {i}', transform=ax_anim_frame.transAxes,
                       fontsize=12, fontweight='bold', color='black',
                       bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=3)) # Bounding box for text.
    
    # Add compact axis labels (Distance, Elevation) if desired, despite removing ticks.
    ax_anim_frame.text(0.5, 0.02, 'Distance (m)', transform=ax_anim_frame.transAxes,
                       ha='center', fontsize=8)
    ax_anim_frame.text(0.02, 0.5, 'Elevation (m)', transform=ax_anim_frame.transAxes,
                       va='center', rotation=90, fontsize=8)
    
    # Save the current plot to an in-memory buffer instead of a file on disk.
    buf = io.BytesIO() # Create a byte stream buffer.
    plt.savefig(buf, format='png', dpi=dpi, bbox_inches='tight') # Save plot to buffer as PNG.
    plt.close(fig_anim_frame)  # Close the figure to free memory.
    
    # Convert buffer content to a PIL (Pillow) Image object and append to the `frames` list.
    buf.seek(0) # Rewind buffer to the beginning.
    img = Image.open(buf)
    frames.append(img.copy())  # Append a copy of the image.
    buf.close() # Close the buffer.

print("All animation frames processed!")

# Save the collected frames as an animated GIF.
gif_path = os.path.join(output_dir, "WCanimation.gif")
# Define frame durations: first frame longer (500ms), subsequent frames shorter (100ms).
durations_gif = [500] + [100] * (len(frames) - 1) # Renamed

# Save the GIF using PIL.
frames[0].save(
    gif_path,
    format='GIF',
    append_images=frames[1:], # Append subsequent frames.
    save_all=True,            # Save all frames.
    duration=durations_gif,   # Assign durations to frames.
    loop=0,                   # 0 means loop the GIF indefinitely.
    optimize=True             # Optimize GIF palette and size.
)

print(f"Animation GIF saved successfully to {gif_path}")

# %%
# This cell is empty in the original script.
# End of Example 3.



