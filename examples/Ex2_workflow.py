# Script-level comment:
# This script, Ex2_workflow.py, demonstrates a comprehensive end-to-end workflow
# using PyHydroGeophysX. It showcases the integration of hydrological model data
# with geophysical forward modeling and inversion.
#
# Workflow steps covered:
# 1. Loading hydrological model data:
#    - Domain information (idomain, surface topography from `top.txt`).
#    - Static properties (porosity from `Porosity.npy`).
#    - Dynamic properties (water content for a specific timestep from `Watercontent.npy`).
# 2. Profile setup: Defining a 2D profile line on the hydrological model grid.
# 3. Interpolation: Interpolating 3D hydrological data (water content, porosity, layer boundaries)
#    onto the 2D profile using `ProfileInterpolator`.
# 4. Mesh generation: Creating a 2D geophysical mesh that conforms to the interpolated
#    geological layer structure using `MeshCreator`.
# 5. Data mapping: Interpolating the 2D profile data (water content, porosity) onto
#    the cells of the 2D geophysical mesh.
# 6. Petrophysical conversion:
#    - Converting water content and porosity to electrical resistivity using `water_content_to_resistivity`.
#    - Converting water content and porosity to P-wave seismic velocity (Vp) using
#      `HertzMindlinModel` and `DEMModel`.
# 7. Geophysical forward modeling:
#    - Simulating synthetic ERT data using the resistivity model and `ERTModelling` from PyGIMLi.
#    - Simulating synthetic seismic travel time data using the Vp model and `TravelTimeManager` from PyGIMLi.
# 8. Geophysical inversion:
#    - Performing ERT inversion on the synthetic ERT data using the custom `ERTInversion`
#      class from PyHydroGeophysX.
#    - Performing ERT inversion using PyGIMLi's default `ERTManager` for comparison.
# 9. "One-step" approach: Demonstrating higher-level wrapper functions (`hydro_to_ert`, `hydro_to_srt`)
#    for direct conversion from hydrological model outputs to synthetic geophysical data.
#
# Assumptions:
# - Example data files are located in a specific directory structure (e.g., `examples/data/`).
#   Paths are currently hardcoded and may need adjustment by the user.
#   - `examples/data/id.txt`: MODFLOW idomain array.
#   - `examples/data/top.txt`: Surface topography data.
#   - `examples/data/Porosity.npy`: 3D porosity array.
#   - `examples/data/Watercontent.npy`: 4D water content array (time, layers, rows, cols); timestep 50 is used.
#   - `examples/data/bot.npy`: 3D array defining boundaries of subsurface geological layers.
#
# Expected output:
# - Console output: Prints shapes of data, progress messages, min/max of calculated properties.
# - Matplotlib plots: Several figures showing:
#   - Surface elevation map with profile line, and the extracted 2D elevation profile.
#   - Porosity and saturation distributions on the 2D geophysical mesh.
#   - Final resistivity and P-wave velocity models on the mesh.
#   - Synthetic ERT pseudosection and SRT travel time picks.
#   - Comparison of true and inverted resistivity models.
#   - Results from the "one-step" ERT and SRT simulations.
# - Saved files:
#   - `workflow_mesh.bms`: The generated 2D geophysical mesh.
#   - `Fig_Resistivity_Velocity_Models.tiff`: Plot of resistivity and velocity models.
#   - `Fig2_Synthetic_Geophysical_Models_and_Data.png`: Plot of synthetic models and data.
#   - `Fig3_Inversion_Comparison.png`: Plot comparing inversion results.
#   - Data files from one-step approaches (e.g., `synthetic_ert_data_onestep.dat`).
#   - All outputs are saved into a user-specific directory defined by `output_dir`
#     (e.g., "C:/Users/HChen8/Documents/GitHub/PyHydroGeophysX/examples/results/workflow_example").
#     SUGGESTION: Change this to a relative path for better portability.

"""
Ex 2. Complete Workflow: From Hydrological Models to Geophysical Inversion
====================================================================

This example demonstrates the complete workflow for integrating hydrological 
model outputs with geophysical forward modeling and inversion using PyHydroGeophysX.

The workflow includes:
1. Loading MODFLOW water content data
2. Setting up 2D profile interpolation from 3D model data
3. Creating meshes with geological layer structure
4. Converting water content to resistivity using petrophysical relationships
5. Converting water content to seismic velocity using rock physics models
6. Forward modeling to create synthetic ERT and seismic data
7. Performing ERT inversion to recover resistivity models

This example serves as a comprehensive tutorial showing the integration
of hydrological and geophysical modeling for watershed monitoring applications.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pygimli as pg # PyGIMLi for geophysical modeling and mesh handling
from pygimli.physics import ert # For ERT specific functionalities
from pygimli.physics import TravelTimeManager # For seismic (not explicitly used here but often in SRT context)
import pygimli.physics.traveltime as tt # For seismic travel time utilities
from mpl_toolkits.axes_grid1 import make_axes_locatable # For colorbar adjustments in plots

# --- Setup for running the script ---
# This section determines the script's current directory and adjusts `sys.path`
# to ensure that the `PyHydroGeophysX` package can be imported. This is particularly
# useful when running examples directly from the `examples` folder, allowing the interpreter
# to find the package in the parent directory.
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # For Jupyter notebooks or interactive environments where __file__ is not defined,
    # os.getcwd() provides the current working directory.
    current_dir = os.getcwd() # Fallback for interactive environments

# Add the parent directory of `PyHydroGeophysX` (i.e., the project root) to `sys.path`.
# This line assumes the `examples` directory is located one level below the project root.
# If `PyHydroGeophysX` is properly installed (e.g., via pip install -e . or a distribution),
# this explicit path manipulation might not be necessary as the package would be in the Python path.
parent_dir = os.path.dirname(os.path.dirname(current_dir)) # Navigates up two levels from script's dir to project root
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# --- Import PyHydroGeophysX specific modules ---
# Core utilities for interpolation and mesh creation
from PyHydroGeophysX.core.interpolation import ProfileInterpolator, create_surface_lines
from PyHydroGeophysX.core.mesh_utils import MeshCreator
# Petrophysical models for converting between hydro and geophy properties
from PyHydroGeophysX.petrophysics.resistivity_models import water_content_to_resistivity
from PyHydroGeophysX.petrophysics.velocity_models import HertzMindlinModel, DEMModel
# Forward modeling classes
from PyHydroGeophysX.forward.ert_forward import ERTForwardModeling
# Inversion classes
from PyHydroGeophysX.inversion.ert_inversion import ERTInversion
# Hydrological to Geophysical direct conversion modules (though not all used directly here, shows availability)
from PyHydroGeophysX.Hydro_modular import hydro_to_ert # Example of one such module


# %%
# --- Define Output Directory ---
# Specifies where results (plots, data files) from this example will be saved.
# Uses an absolute path, which is not ideal for portability across different machines.
# SUGGESTION: Use a path relative to the script's location or the user's home directory
# for better portability. For example:
# output_dir = os.path.join(current_dir, "results", "workflow_example")
output_dir = "C:/Users/HChen8/Documents/GitHub/PyHydroGeophysX/examples/results/workflow_example" # User-specific absolute path.
# Create the output directory if it doesn't already exist.
# `exist_ok=True` prevents an error if the directory already exists.
os.makedirs(output_dir, exist_ok=True)

# %% [markdown]
# ## Step by Step approach
# <!-- Marks the beginning of the detailed, sequential workflow steps. -->

# %% [markdown]
# ### Loading domain information...
# <!-- This section focuses on loading basic domain geometry and static hydrological model data like porosity. -->
# <!-- These data define the physical framework of the model. -->

# %%
# --- Define Data Directory and Load Initial Hydrological Model Data ---
# Path to the directory containing example data files.
# This is also a user-specific absolute path.
# SUGGESTION: For example scripts, it's better to use relative paths from `current_dir`
# e.g., data_dir = os.path.join(current_dir, "data")
data_dir = "C:/Users/HChen8/Documents/GitHub/PyHydroGeophysX/examples/data/"
modflow_dir = os.path.join(data_dir, "modflow") # Specific subdirectory, may or may not be used if data_dir has all files.

# Load domain information:
# - `idomain`: Defines active (value > 0) and inactive (value = 0) cells in the MODFLOW model grid.
#   Crucial for masking areas outside the valid model domain. Expected shape: (nrows, ncols).
# - `top`: Represents the elevation of the top surface of the model grid.
#   Expected shape: (nrows, ncols).
# - `porosity`: Array of porosity values for the domain.
#   Expected shape: (nlay, nrows, ncols) for a 3D model.
# These are loaded from text or NumPy binary files. User should replace with their actual file paths and formats.
idomain = np.loadtxt(os.path.join(data_dir, "id.txt"))
top = np.loadtxt(os.path.join(data_dir, "top.txt"))
porosity = np.load(os.path.join(data_dir, "Porosity.npy"))

# %% [markdown]
# ### Loading MODFLOW water content data..
# <!-- This section demonstrates loading time-specific water content data. -->

# %%
# --- Load MODFLOW Water Content Data for a Specific Timestep ---
# This example loads pre-processed water content data from a .npy file.
# This approach is often used for convenience or if the original binary MODFLOW outputs
# are very large or require complex reading steps not covered by a simple script.
# In a typical direct workflow, one might use the `MODFLOWWaterContent` class from
# `PyHydroGeophysX.model_output.modflow_output` to load directly from MODFLOW's binary "WaterContent" file (often from UZF package).

# "Watercontent.npy" is assumed to be a 4D array (e.g., time_index, nlay_uzf, nrows, ncols)
# or a list/dict of 3D arrays. Here, it's loaded, and then a specific timestep (index 50) is selected.
# The resulting `water_content` variable should be a 3D array (nlay_uzf, nrows, ncols) for that single time instance.
Water_Content_All_Timesteps = np.load(os.path.join(data_dir, "Watercontent.npy")) # Load the entire dataset.
selected_timestep_index = 50 # Define which timestep to extract.
water_content = Water_Content_All_Timesteps[selected_timestep_index]
# Print the shape of the selected water content data for verification.
print(f"Selected Water Content data shape (expected: nlay, nrow, ncol): {water_content.shape}")

# %% [markdown]
# ### Set up profile for 2D section
# <!-- This section defines a 2D profile line (a cross-section) on the 3D hydrological model grid. -->
# <!-- Data from the 3D model will be interpolated onto this 2D profile, which is a common step -->
# <!-- when preparing for 2D geophysical modeling or visualization. -->

# %%
# Step 3: Set up profile for 2D section
print("Step 3: Setting up profile for 2D section...")

# Define profile endpoints using grid indices [column_index, row_index] from the original hydrological model grid.
# These indices specify the start and end points of the 2D profile line on the surface (or a specific layer).
# Point1: [column_index_start, row_index_start]
# Point2: [column_index_end, row_index_end]
point1_indices = [115, 70]  # Example start point indices.
point2_indices = [95, 180]  # Example end point indices.

# Initialize the ProfileInterpolator.
# This object handles geometric calculations for the profile and subsequent interpolation of data onto it.
# Parameters:
# - `point1`, `point2`: Start and end point indices of the profile.
# - `surface_data`: A 2D array of surface elevations (e.g., the `top` array loaded earlier).
# - `origin_x`, `origin_y`: Real-world coordinates (e.g., UTM) of the hydrological model grid's origin
#   (typically the bottom-left or top-left corner of the grid).
# - `pixel_width`, `pixel_height`: Cell dimensions (spatial resolution) of the hydrological model grid.
#   `pixel_height` is often negative if the y-axis origin of the grid is at the top (common in raster data).
# - `num_points`: Number of discrete points to define along the 2D profile line for interpolation.
profile_interpolator_instance = ProfileInterpolator( # Renamed variable for clarity
    point1=point1_indices,
    point2=point2_indices,
    surface_data=top,      # Using the 'top' array (surface elevations) for profile setup.
    origin_x=569156.0,     # Example UTM Easting of the model grid's origin.
    origin_y=4842444.0,    # Example UTM Northing of the model grid's origin.
    pixel_width=1.0,       # Width of each cell in the hydro model grid (e.g., in meters).
    pixel_height=-1.0,     # Height of each cell (e.g., in meters; negative if y-axis is downwards from origin).
    num_points=400         # Discretize the profile into 400 points for interpolation.
)

# %% [markdown]
# ### Interpolating data to profile
# <!-- Hydrological data (e.g., water content, porosity, layer boundaries) are interpolated from the 3D grid -->
# <!-- onto the previously defined 2D profile line. This transforms 3D model data into a 2D cross-section format. -->

# %%
# Step 4: Interpolate 3D hydrological data to the 2D profile.
print("Step 4: Interpolating hydrological data to the 2D profile...")

# Interpolate 3D water content data (for the selected MODFLOW timestep) to the defined 2D profile.
# `water_content` is assumed to be a 3D array (nlay_hydro, nrow_hydro, ncol_hydro).
# `profile_interpolator_instance.interpolate_3d_data` projects these values onto the profile line.
# The result `water_content_on_profile` will be a 2D array (nlay_hydro, num_profile_points).
water_content_on_profile = profile_interpolator_instance.interpolate_3d_data(water_content) # Renamed variable

# Interpolate 3D porosity data to the 2D profile.
# `porosity` is assumed to be a 3D array (nlay_hydro, nrow_hydro, ncol_hydro).
# The result `porosity_on_profile` will be a 2D array (nlay_hydro, num_profile_points).
porosity_on_profile = profile_interpolator_instance.interpolate_3d_data(porosity) # Renamed variable

# %% [markdown]
# ### Creating mesh
# <!-- This section focuses on creating a 2D geophysical mesh. The mesh geometry is based on -->
# <!-- geological layer information (e.g., surface topography, subsurface layer boundaries) -->
# <!-- that has also been interpolated onto the 2D profile. This structured mesh is suitable -->
# <!-- for geophysical forward modeling and inversion. -->

# %%
# --- Define Geological Structure along the Profile for Mesh Creation ---
# Load boundary data for subsurface geological layers (e.g., bottom elevations of different units).
# `bot.npy` is assumed to contain an array where each "row" (first dimension) represents a different layer boundary.
# Expected shape might be (num_layer_boundaries, nrow_hydro, ncol_hydro).
layer_boundary_data_3d = np.load(os.path.join(data_dir, "bot.npy")) # Renamed for clarity

# Interpolate these 3D layer boundaries onto the 2D profile.
# `profile_interpolator_instance.interpolate_layer_data` takes a list of 3D arrays (one for each boundary).
# The `top` array (surface elevation) is prepended to the list of boundaries from `bot.npy`.
# `structure_on_profile` will be a 2D array (num_total_boundaries, num_profile_points) holding elevations of boundaries along the profile.
structure_on_profile = profile_interpolator_instance.interpolate_layer_data([top] + layer_boundary_data_3d.tolist()) # Renamed

# --- Create Surface and Layer Boundary Lines for Mesh Generation ---
# The `create_surface_lines` function extracts specific layer boundaries from `structure_on_profile`
# to define the main geological units for the geophysical mesh (e.g., regolith, fractured bedrock, fresh bedrock).
# `top_idx`, `mid_idx`, `bot_idx` are indices into the layers of `structure_on_profile` (0 is surface).
# These indices select which interpolated boundaries will define the mesh layers.
# Example: Layer 0 is the surface, layer 4 is the top of fractured bedrock, layer 12 is top of fresh bedrock.
layer_interface_idx_top = int(0)     # Index in `structure_on_profile` corresponding to the ground surface.
layer_interface_idx_middle = int(4)  # Index for the boundary between, e.g., regolith and fractured bedrock.
layer_interface_idx_bottom = int(12) # Index for the boundary between, e.g., fractured and fresh bedrock.

# `surface_line`, `boundary_line1`, `boundary_line2` will be 2D arrays of shape [n_points_on_profile, 2],
# where columns are (x_distance_along_profile, z_elevation) for these selected boundaries.
surface_line, boundary_line1, boundary_line2 = create_surface_lines(
    L_profile=profile_interpolator_instance.L_profile, # Distances along the profile (serves as x-coordinates for the 2D mesh).
    structure=structure_on_profile,                    # All interpolated layer boundary elevations along the profile.
    top_idx=layer_interface_idx_top,                   # Index for the surface line.
    mid_idx=layer_interface_idx_middle,                # Index for the first internal layer boundary.
    bot_idx=layer_interface_idx_bottom                 # Index for the second internal layer boundary.
)

# --- Create 2D Geophysical Mesh using MeshCreator ---
# Instantiate `MeshCreator` with a desired mesh quality parameter (e.g., minimum angle for triangles).
mesh_creator_instance = MeshCreator(quality=32) # Renamed variable
# Create the mesh using the defined surface and internal layer boundary lines.
# The resulting mesh will conform to these structures, allowing different physical properties to be assigned per region.
# `bottom_depth` defines how far below the last specified interface (`boundary_line2`) the mesh should extend.
# `_geom_plc` is the PyGIMLi Planar Straight Line Complex (PLC) object representing the geometry, also returned.
geophysical_mesh, _geom_plc = mesh_creator_instance.create_from_layers( # Renamed `mesh` to `geophysical_mesh`
    surface=surface_line,
    layers=[boundary_line1, boundary_line2], # List of internal layer boundaries to be incorporated into the mesh.
    bottom_depth= np.min(boundary_line2[:,1]) - 10 # Mesh bottom is 10 units below the minimum elevation of `boundary_line2`.
)

# --- Save the Generated Geophysical Mesh ---
# The mesh is saved in PyGIMLi's binary mesh format (.bms), which can be reloaded later.
# Output path is constructed using `output_dir` defined at the beginning of the script.
mesh_save_path = os.path.join(output_dir, "workflow_mesh.bms") # Changed filename for clarity
geophysical_mesh.save(mesh_save_path)
print(f"Generated geophysical mesh saved to: {mesh_save_path}")

# %%
# --- Visualize Mesh Setup: 3D Surface View and 2D Profile View ---
# This cell generates a plot to help understand the relationship between the original 3D model's
# surface, the selected 2D profile line, and the extracted 2D elevation profile that forms
# the top boundary of the 2D geophysical mesh.
# import matplotlib.pyplot as plt # Already imported earlier.

plt.figure(figsize=(15, 5)) # Create a new figure for the two subplots.
# Create a copy of the `top` data and mask out inactive cells (where idomain is 0)
# by setting them to NaN for clearer visualization of the model's active surface area.
top_masked_for_plot = top.copy()
top_masked_for_plot[idomain==0] = np.nan

# Subplot 1 (Left): Surface Elevation Map with the 2D Profile Line
plt.subplot(121) # 1 row, 2 columns, 1st subplot.
plt.imshow(top_masked_for_plot, origin='lower') # Display the 2D `top` array as an image. `origin='lower'` for standard cartesian.
plt.colorbar(label='Top Elevation (m)') # Add a colorbar indicating elevation values.
# Mark the start and end points of the profile on the map using their grid indices.
plt.plot(point1_indices[0], point1_indices[1], 'ro', label='Profile Start') # Start point in red.
plt.plot(point2_indices[0], point2_indices[1], 'bo', label='Profile End')   # End point in blue.
# Draw a dashed red line connecting the start and end points to visualize the profile path.
plt.plot([point1_indices[0], point2_indices[0]], [point1_indices[1], point2_indices[1]], 'r--')
plt.legend() # Show legend for profile points.
plt.title('Surface Elevation Map with 2D Profile Line')
plt.xlabel('Model Column Index') # X-axis represents column indices of the 3D model grid.
plt.ylabel('Model Row Index')   # Y-axis represents row indices.


# Subplot 2 (Right): Extracted 2D Elevation Profile
# This plot shows the actual (x, z) coordinates of the surface along the defined profile line.
# `surface_line` was generated by `create_surface_lines` and contains [distance_along_profile, elevation].
plt.subplot(122)
plt.plot(surface_line[:,0], surface_line[:,1], color='blue') # Plot elevation vs. distance along profile.
plt.title('Extracted 2D Elevation Profile for Geophysical Mesh')
plt.xlabel('Distance Along Profile (m)') # X-axis is distance from the start of the profile.
plt.ylabel('Elevation (m)')             # Y-axis is elevation.
plt.grid(True) # Add a grid for better readability.

plt.tight_layout() # Adjust subplot spacing to prevent overlaps.
plt.show() # Display the figure. Output: Matplotlib plot window.

# %% [markdown]
# ### Interpolating data to mesh
# <!-- This section maps the 2D profile data (water content, porosity) onto the cells of the 2D geophysical mesh. -->

# %%
# Step 6: Interpolate data from the 2D profile to the 2D geophysical mesh cells.
print("Step 6: Interpolating profile data to the geophysical mesh cells...")

# Create an ID array (`ID_profile_layers`) to assign layer markers to the interpolated profile data.
# This is based on the layer boundary indices (`layer_interface_idx_...`) defined earlier.
# These markers should correspond to the markers used in the geophysical mesh (`geophysical_mesh.cellMarkers()`).
# The `porosity_on_profile` array (shape n_hydro_layers x n_profile_points) is used as a template for shape and structure.
# marker_labels defines the actual integer markers to assign to these layers.
# Example: Layer from surface to `mid_idx` boundary gets marker_labels[0].
# The mesh created by `MeshCreator` assigns markers (e.g., 2, 3, then others for subsequent layers).
# Ensure `marker_labels` here matches how `MeshCreator` assigned them or how they are intended for petrophysics.
# The `MeshCreator` in `mesh_utils.py` uses markers 2, 3, 2 for a 3-layer system (surface, layer_A, layer_B).
# The `ID_profile_layers` should reflect the *source* layers from the hydro model along the profile.
# The `interpolate_to_mesh` function then uses `mesh_markers` (from the geophysical mesh) and `layer_markers`
# (which should be the unique values in `ID_profile_layers`) to map correctly.

# Define marker labels for the layers consistent with how `MeshCreator` would have marked them,
# or how they are defined for petrophysical properties.
# Assuming mesh created by `create_from_layers` has markers:
# Region above line1 (boundary_line1) -> typically marker 2 by default by `createPolygon` in `create_mesh_from_layers` if it's the first "layer" polygon.
# Region between line1 and line2 -> typically marker 3.
# Region below line2 -> typically marker 2 again (if default markers are used and it's the second main layer polygon).
# This needs to be very consistent. The example `layer_markers = [0,3,2]` seems to imply custom marker meanings.
# Let's assume the markers in `geophysical_mesh` are [RegionAboveLine1, RegionBetweenLine1And2, RegionBelowLine2].
# And `ID_profile_layers` needs to map hydro model layers to these geophysical mesh region markers along the profile.

# This part is tricky: `ID_profile_layers` should represent the "source layer ID" for each point along the profile *for each hydro layer*.
# The `interpolate_to_mesh` then takes these *profile points with assigned source layer IDs* and interpolates them
# to the geophysical mesh cells, guided by `mesh_markers` of the target geophysical mesh.
# The `layer_markers` argument in `interpolate_to_mesh` should be the list of unique IDs present in `ID_profile_layers`.

# Create `ID_profile_layers` based on hydro model layering structure along the profile.
# `porosity_on_profile` has shape (num_hydro_model_layers, num_profile_points).
# We need to assign a "material ID" to each of these points.
# For simplicity, let's assume the hydro model layers map directly to the geophysical regions defined by structure.
# Example: hydro layers 0 to (mid_idx-1) map to geophysical region marker_labels[0] (e.g., regolith).
# hydro layers mid_idx to (bot_idx-1) map to geophysical region marker_labels[1] (e.g., fractured bedrock).
# hydro layers bot_idx onwards map to geophysical region marker_labels[2] (e.g., fresh bedrock).
ID_profile_layers = np.zeros_like(porosity_on_profile, dtype=int) # Same shape as profile data
defined_marker_labels = [0, 3, 2] # These are the markers that will be used in `interpolate_to_mesh`
                                  # and should match markers in `geophysical_mesh`.
                                  # Assuming these are: Regolith, Fractured, Fresh.

# Assign these based on the hydro model layer indices used to define structure:
# This assumes `porosity_on_profile` has at least `bot_idx` layers.
for k_layer in range(porosity_on_profile.shape[0]): # Iterate over each hydro model layer in the profile data
    if k_layer < layer_interface_idx_middle: # Layers above the first main interface (e.g., regolith)
        ID_profile_layers[k_layer, :] = defined_marker_labels[0]
    elif k_layer < layer_interface_idx_bottom: # Layers between first and second interface (e.g., fractured)
        ID_profile_layers[k_layer, :] = defined_marker_labels[1]
    else: # Layers below the second interface (e.g., fresh)
        ID_profile_layers[k_layer, :] = defined_marker_labels[2]

# Get cell centers and actual markers from the generated geophysical mesh.
mesh_cell_centers_np = np.array(geophysical_mesh.cellCenters()) # Renamed
mesh_cell_markers_np = np.array(geophysical_mesh.cellMarkers()) # Renamed

# Interpolate porosity from profile to the geophysical mesh cells.
porosity_on_mesh = profile_interpolator.interpolate_to_mesh( # Renamed
    property_values=porosity_on_profile,   # Porosity data along the profile (n_hydro_layers, n_profile_points)
    depth_values=structure_on_profile,     # Depths of hydro layers along profile
    mesh_x=mesh_cell_centers_np[:, 0],     # X-coords of geophysical mesh cell centers
    mesh_y=mesh_cell_centers_np[:, 1],     # Y-coords (depths) of geophysical mesh cell centers
    mesh_markers=mesh_cell_markers_np,     # Markers of geophysical mesh cells
    ID=ID_profile_layers,                  # Material/Layer IDs assigned to profile data points
    layer_markers=defined_marker_labels    # The unique marker labels being mapped to mesh regions
)

# Interpolate water content from profile to the geophysical mesh cells.
wc_on_mesh = profile_interpolator.interpolate_to_mesh( # Renamed
    property_values=water_content_on_profile,
    depth_values=structure_on_profile,
    mesh_x=mesh_cell_centers_np[:, 0],
    mesh_y=mesh_cell_centers_np[:, 1],
    mesh_markers=mesh_cell_markers_np,
    ID=ID_profile_layers,
    layer_markers=defined_marker_labels
)

# %%
# ID1 from original script was likely a typo or shorthand for ID_profile_layers.
# print(ID_profile_layers) # For debugging the generated layer IDs on the profile.

# %%
# --- Import necessary libraries for this plotting section ---
# import matplotlib.pyplot as plt # Already imported
# import matplotlib as mpl # Already imported
# import numpy as np # Already imported
# import pygimli as pg # Already imported
from palettable.cartocolors.diverging import Earth_7 # Specific colormap palette

# --- Font Settings for Publication Quality Plots (Optional) ---
# These settings configure matplotlib to use Arial font and specific font sizes
# for various plot elements, suitable for publications.
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['axes.titlesize'] = 14
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['legend.fontsize'] = 12
mpl.rcParams['figure.dpi'] = 150 # Set DPI for saved figures.

# --- Preprocessing Data for Combined Plot ---
# `top_masked_viz` was created in the previous cell for visualizing the surface elevation map.
# Recalculate saturation on the geophysical mesh for plotting.
# `wc_on_mesh` and `porosity_on_mesh` are the interpolated values on the `geophysical_mesh`.
porosity_on_mesh_safe_viz = np.maximum(porosity_on_mesh, 1e-4) # Avoid division by zero. Renamed.
saturation_on_mesh_viz = np.clip(wc_on_mesh / porosity_on_mesh_safe_viz, 0.0, 1.0) # Renamed.
# Select a colormap for porosity plot.
custom_porosity_cmap = Earth_7.mpl_colormap # Renamed.

# --- Create 2x2 Figure for Combined Visualization ---
# This figure will display:
# 1. The original 3D model's top surface with the 2D profile line.
# 2. The extracted 2D elevation profile.
# 3. Porosity distribution on the 2D geophysical mesh.
# 4. Saturation distribution on the 2D geophysical mesh.
fig_combined_display, axs_combined_display = plt.subplots(2, 2, figsize=(14, 10)) # Renamed fig and axs

# --- Top Left Subplot: Surface Elevation Map and Profile Line ---
ax_tl_viz = axs_combined_display[0, 0] # Top-left axes. Renamed.
im_surface = ax_tl_viz.imshow(top_masked_for_plot, origin='lower', cmap='terrain') # Renamed im0
# ax_tl_viz.invert_yaxis() # Original had this; map views usually don't invert Y. Removed for typical map view.

# Plot profile line and points (using `point1_indices`, `point2_indices` from earlier).
ax_tl_viz.plot(point1_indices[0], point1_indices[1], 'ro', label='Profile Start')
ax_tl_viz.plot(point2_indices[0], point2_indices[1], 'bo', label='Profile End')
ax_tl_viz.plot([point1_indices[0], point2_indices[0]], [point1_indices[1], point2_indices[1]], 'r--')

# Styling for top-left subplot
ax_tl_viz.set_xticks([]) # Remove x-axis ticks for map view.
ax_tl_viz.set_yticks([]) # Remove y-axis ticks.
for spine_obj in ax_tl_viz.spines.values(): # Make plot border invisible. Renamed.
    spine_obj.set_visible(False)
ax_tl_viz.set_title('Surface Elevation Map & Profile') # Set title.
cbar_surface = fig_combined_display.colorbar(im_surface, ax=ax_tl_viz, orientation='vertical', shrink=0.8) # Add colorbar. Renamed.
cbar_surface.set_label('Elevation (m)')
ax_tl_viz.legend(loc='upper right')

# --- Top Right Subplot: Extracted 2D Elevation Profile ---
ax_tr_viz = axs_combined_display[0, 1] # Top-right axes. Renamed.
# `surface_line` contains (distance_along_profile, elevation) for the surface.
ax_tr_viz.plot(surface_line[:, 0], surface_line[:, 1], color='darkgreen')
ax_tr_viz.set_xlabel('Distance Along Profile (m)')
ax_tr_viz.set_ylabel('Elevation (m)')
ax_tr_viz.set_title('2D Elevation Profile')
ax_tr_viz.grid(True)

# --- Bottom Left Subplot: Porosity on Geophysical Mesh ---
ax_bl_viz = axs_combined_display[1, 0] # Bottom-left axes. Renamed.
# Use PyGIMLi's `show` utility to display cell data (`porosity_on_mesh`) on `geophysical_mesh`.
pg.show(geophysical_mesh, data=porosity_on_mesh, ax=ax_bl_viz, orientation="vertical",
        cMap=custom_porosity_cmap, cMin=0.05, cMax=0.45, # Define porosity range for color scale.
        xlabel="Distance (m)", ylabel="Elevation (m)",
        label='Porosity (-)', showColorBar=True)
ax_bl_viz.set_title('Porosity on Geophysical Mesh')


# --- Bottom Right Subplot: Saturation on Geophysical Mesh ---
ax_br_viz = axs_combined_display[1, 1] # Bottom-right axes. Renamed.
# Display `saturation_on_mesh_viz`.
pg.show(geophysical_mesh, data=saturation_on_mesh_viz, ax=ax_br_viz, orientation="vertical",
        cMap='Blues', cMin=0.0, cMax=1.0, # Saturation range [0,1].
        xlabel="Distance (m)", ylabel="Elevation (m)",
        label='Saturation (-)', showColorBar=True)
ax_br_viz.set_title('Saturation on Geophysical Mesh')

# Adjust overall layout to prevent overlapping titles/labels.
plt.tight_layout(pad=3.0) # Increased padding.
plt.show() # Display the combined figure.


# %%
# --- Print Min/Max of Calculated Water Content and Saturation on Mesh ---
# Useful for checking the range of values after interpolation and calculation.
# `wc_on_mesh` and `saturation_on_mesh_viz` are the values on the geophysical mesh cells.
# Filter for finite values to avoid errors with NaN if masking occurred.
finite_wc_on_mesh = wc_on_mesh[np.isfinite(wc_on_mesh)]
finite_sat_on_mesh = saturation_on_mesh_viz[np.isfinite(saturation_on_mesh_viz)]

print(f"Water Content on mesh min/max: "
      f"{np.min(finite_wc_on_mesh) if finite_wc_on_mesh.size > 0 else 'N/A':.4f} / "
      f"{np.max(finite_wc_on_mesh) if finite_wc_on_mesh.size > 0 else 'N/A':.4f}")
print(f"Saturation on mesh min/max: "
      f"{np.min(finite_sat_on_mesh) if finite_sat_on_mesh.size > 0 else 'N/A':.4f} / "
      f"{np.max(finite_sat_on_mesh) if finite_sat_on_mesh.size > 0 else 'N/A':.4f}")

# %% [markdown]
# ### Calculating saturation
# <!-- This section seems to recalculate saturation. It was already calculated above for plotting (as `saturation_on_mesh_viz`). -->
# <!-- This might be redundant or for ensuring the variable `saturation` is explicitly defined for subsequent steps. -->
# <!-- For clarity, the variable `saturation_on_mesh` (or `saturation_on_mesh_viz`) from above will be used/confirmed. -->

# %%

# Ensure porosity is not zero to avoid division by zero.
# `porosity_on_mesh` is used here (should be same as `porosity_on_mesh_safe_viz` if floor is same).
# This step re-calculates `saturation` if it wasn't already the primary variable holding the latest calculation.
# porosity_safe_calc = np.maximum(porosity_on_mesh, 0.01) # Original variable name was porosity_mesh
# saturation_calc = np.clip(wc_on_mesh / porosity_safe_calc, 0.0, 1.0) # Renamed to avoid confusion.
# Using `saturation_on_mesh_viz` that was calculated before plot, as it's the most up-to-date.
saturation = saturation_on_mesh_viz # Assign to `saturation` for use in subsequent cells.
print("Saturation for petrophysics obtained from previous calculation (used for plotting).")


# %% [markdown]
# ### Converting to resistivity
# <!-- This section converts the water content (via saturation) and porosity on the geophysical mesh -->
# <!-- to electrical resistivity using a petrophysical model (e.g., Waxman-Smits). -->
# <!-- This creates the resistivity model for ERT forward simulation. -->

# %%

# Step 8: Convert water content on mesh to resistivity model using petrophysical relationships.
print("Step 8: Converting water content to resistivity model...")

# Define petrophysical parameters for different geological layers.
# These parameters are for the Waxman-Smits model as used in `water_content_to_resistivity`.
# `petro_marker_labels` should correspond to the markers in `geophysical_mesh` and the order of params.
# `defined_marker_labels` was [0, 3, 2] from the interpolation step.
petro_marker_labels = defined_marker_labels # e.g., [Regolith_marker, Fractured_marker, Fresh_marker]
# Saturated resistivity (rho_sat) for each layer [Ohm-m].
rho_sat_layer_params = [100.0, 500.0, 2400.0] # Example values matching markers [0, 3, 2]
# Saturation exponent (n) for each layer.
n_exponent_layer_params = [2.2, 1.8, 2.5] # Example values
# Surface conductivity (sigma_s) for each layer [S/m].
sigma_s_layer_params = [1.0/500.0, 0.0, 0.0] # Example values

# Initialize an array to store the resistivity model for all mesh cells.
# `wc_on_mesh` is used for shape, implying resistivity model has same cell mapping.
resistivity_model_on_mesh = np.zeros_like(wc_on_mesh) # Renamed `res_models`

# Iterate through each defined layer/marker to apply its specific petrophysical parameters.
for i, current_marker in enumerate(petro_marker_labels):
    # Create a boolean mask for cells belonging to the current layer marker.
    # `mesh_cell_markers_np` holds the marker for each cell in `geophysical_mesh`.
    current_layer_mask = (mesh_cell_markers_np == current_marker)

    if not np.any(current_layer_mask): # If no cells found for this marker, skip.
        print(f"Warning: No cells found for marker {current_marker} in resistivity conversion.")
        continue

    # Calculate resistivity for the cells in the current layer.
    # `water_content_to_resistivity` function applies the Waxman-Smits model.
    # It takes water content, rhos, n, porosity, and sigma_sur.
    # Using `wc_on_mesh` and `porosity_on_mesh` which are per-cell values.
    resistivity_for_layer = water_content_to_resistivity(
        wc_on_mesh[current_layer_mask],                  # Water content for cells in this layer.
        float(rho_sat_layer_params[i]),                  # rhos for this layer.
        float(n_exponent_layer_params[i]),               # n for this layer.
        porosity_on_mesh[current_layer_mask],            # Porosity for cells in this layer.
        float(sigma_s_layer_params[i])                   # sigma_sur for this layer.
    )
    resistivity_model_on_mesh[current_layer_mask] = resistivity_for_layer # Assign to the overall model.

# %%
# --- Print Resistivity Range for Each Layer (Debugging/Verification) ---
# This helps check if the calculated resistivities are plausible for each defined layer.
print("Resistivity ranges per layer:")
for i, marker_val in enumerate(petro_marker_labels): # Use petro_marker_labels for consistency
    layer_mask_check = (mesh_cell_markers_np == marker_val) # Use mesh_cell_markers_np
    if np.any(layer_mask_check):
        # Filter out NaNs or Infs before min/max to avoid runtime warnings if bad values exist
        res_layer_values = resistivity_model_on_mesh[layer_mask_check]
        res_layer_values_finite = res_layer_values[np.isfinite(res_layer_values)]
        if res_layer_values_finite.size > 0:
            print(f"  Layer (Marker {marker_val}): min={np.min(res_layer_values_finite):.2f} Ohm-m, max={np.max(res_layer_values_finite):.2f} Ohm-m")
        else:
            print(f"  Layer (Marker {marker_val}): No finite resistivity values calculated.")
    else:
        # This case was already handled by a warning in the loop above.
        pass


# %% [markdown]
# ### Converting to P wave velocity
# <!-- This section converts hydrological properties (porosity, saturation) on the geophysical mesh -->
# <!-- to P-wave seismic velocity (Vp) using rock physics models (Hertz-Mindlin, DEM). -->
# <!-- This creates the velocity model for SRT forward simulation. -->

# %%
# Step 9: Convert hydrological properties to P-wave velocity (Vp) model.
print("Step 9: Converting to P-wave velocity model...")

# Initialize petrophysical velocity models (Hertz-Mindlin and DEM).
# These models have default parameters that can be overridden if needed.
# Critical porosity and coordination number are key for Hertz-Mindlin.
hertz_mindlin_petro_model = HertzMindlinModel(critical_porosity=0.4, coordination_number=6.0) # Renamed
dem_petro_model = DEMModel() # Renamed

# Initialize an array to store the Vp model for all mesh cells [m/s].
velocity_model_on_mesh = np.zeros_like(wc_on_mesh, dtype=float) # Renamed. Ensure float type.

# --- Define Layer-Specific Rock Physics Parameters for Velocity Calculation ---
# These parameters are examples and should be adjusted for the specific geological materials.
# `petro_marker_labels` ([0, 3, 2]) from resistivity section are reused here for layer identification.
# It's crucial that this marker list and the parameter dictionaries below are consistently ordered.

# Parameters for the top layer (e.g., regolith, marker defined by petro_marker_labels[0]), to be used with Hertz-Mindlin model.
top_layer_mask_vel = (mesh_cell_markers_np == petro_marker_labels[0]) # Renamed mask
if np.any(top_layer_mask_vel):
    top_bulk_modulus_matrix_vel = 30.0  # Bulk modulus of mineral matrix [GPa]. Renamed.
    top_shear_modulus_matrix_vel = 20.0  # Shear modulus of mineral matrix [GPa]. Renamed.
    top_mineral_density_vel = 2650.0     # Density of mineral matrix [kg/m³]. Renamed.
    top_effective_depth_vel = 1.0      # Effective depth for pressure calculation in Hertz-Mindlin [m]. Renamed.

    # Calculate Vp using Hertz-Mindlin model for the top layer.
    # It returns high and low bounds for Vp; average is taken.
    Vp_high_top, Vp_low_top = hertz_mindlin_petro_model.calculate_velocity(
        porosity=porosity_on_mesh[top_layer_mask_vel],
        saturation=saturation[top_layer_mask_vel], # Use the globally defined saturation array
        bulk_modulus=top_bulk_modulus_matrix_vel,
        shear_modulus=top_shear_modulus_matrix_vel,
        mineral_density=top_mineral_density_vel,
        depth=top_effective_depth_vel
    )
    velocity_model_on_mesh[top_layer_mask_vel] = (Vp_high_top + Vp_low_top) / 2.0 # Average Vp [m/s]

# Parameters for the middle layer (e.g., fractured bedrock, marker petro_marker_labels[1]), to be used with DEM model.
mid_layer_mask_vel = (mesh_cell_markers_np == petro_marker_labels[1]) # Renamed mask
if np.any(mid_layer_mask_vel):
    mid_bulk_modulus_matrix_vel = 50.0  # GPa
    mid_shear_modulus_matrix_vel = 35.0 # GPa
    mid_mineral_density_vel = 2670.0  # kg/m³
    mid_pore_aspect_ratio_vel = 0.05   # Pore aspect ratio for DEM model

    # Calculate Vp using DEM model for the middle layer.
    # DEM model returns K_eff, G_eff, Vp. We only need Vp.
    _Keff_mid_vel, _Geff_mid_vel, Vp_mid_vel = dem_petro_model.calculate_velocity( # Renamed outputs
        porosity=porosity_on_mesh[mid_layer_mask_vel],
        saturation=saturation[mid_layer_mask_vel],
        bulk_modulus=mid_bulk_modulus_matrix_vel,
        shear_modulus=mid_shear_modulus_matrix_vel,
        mineral_density=mid_mineral_density_vel,
        aspect_ratio=mid_pore_aspect_ratio_vel
    )
    velocity_model_on_mesh[mid_layer_mask_vel] = Vp_mid_vel # Vp [m/s]

# Parameters for the bottom layer (e.g., fresh bedrock, marker petro_marker_labels[2]), to be used with DEM model.
bot_layer_mask_vel = (mesh_cell_markers_np == petro_marker_labels[2]) # Renamed mask
if np.any(bot_layer_mask_vel):
    bot_bulk_modulus_matrix_vel = 55.0  # GPa
    bot_shear_modulus_matrix_vel = 50.0  # GPa
    bot_mineral_density_vel = 2680.0  # kg/m³
    bot_pore_aspect_ratio_vel = 0.03   # Pore aspect ratio for DEM model

    # Calculate Vp using DEM model for the bottom layer.
    _Keff_bot_vel, _Geff_bot_vel, Vp_bot_vel = dem_petro_model.calculate_velocity( # Renamed outputs
        porosity=porosity_on_mesh[bot_layer_mask_vel],
        saturation=saturation[bot_layer_mask_vel],
        bulk_modulus=bot_bulk_modulus_matrix_vel,
        shear_modulus=bot_shear_modulus_matrix_vel,
        mineral_density=bot_mineral_density_vel,
        aspect_ratio=bot_pore_aspect_ratio_vel
    )
    velocity_model_on_mesh[bot_layer_mask_vel] = Vp_bot_vel # Vp [m/s]

# %%
# --- Print Velocity Range for Each Layer (Debugging/Verification) ---
print("P-wave velocity ranges per layer (after petrophysical conversion):")
for i, marker_val in enumerate(petro_marker_labels):
    layer_mask_check_vel = (mesh_cell_markers_np == marker_val) # Renamed
    if np.any(layer_mask_check_vel) and np.any(np.isfinite(velocity_model_on_mesh[layer_mask_check_vel])): # Check for finite values
        vel_layer_values = velocity_model_on_mesh[layer_mask_check_vel][np.isfinite(velocity_model_on_mesh[layer_mask_check_vel])]
        if vel_layer_values.size > 0:
             print(f"  Layer (Marker {marker_val}): min={np.min(vel_layer_values):.2f} m/s, max={np.max(vel_layer_values):.2f} m/s")
        else:
             print(f"  Layer (Marker {marker_val}): Contains no finite velocity values.")
    elif not np.any(layer_mask_check_vel):
         print(f"  Layer (Marker {marker_val}): No cells with this marker in the mesh.") # More informative
    else: # All values are NaN or Inf
         print(f"  Layer (Marker {marker_val}): All velocity values are non-finite (NaN or Inf).")


# %%
# --- Visualize Final Resistivity and Velocity Models on the Geophysical Mesh ---
from palettable.cartocolors.diverging import Earth_7 # A specific colormap palette.
custom_cmap_viz = Earth_7.mpl_colormap # Renamed for clarity

# Create a figure with 1 row and 2 columns for side-by-side plots.
fig_models_viz, axs_models_viz = plt.subplots(1, 2, figsize=(14, 6)) # Renamed fig and axs

# --- Left Subplot: Final Resistivity Model ---
# `resistivity_model_on_mesh` contains the calculated resistivities for `geophysical_mesh`.
# `pg.show` handles plotting cell data on the mesh.
pg.show(geophysical_mesh, data=resistivity_model_on_mesh, ax=axs_models_viz[0], orientation="vertical",
        cMap=custom_cmap_viz, logScale=True, showColorBar=True, # Resistivity often plotted on log scale.
        xlabel="Distance (m)", ylabel="Elevation (m)",
        label='Resistivity (Ω·m)', cMin=100, cMax=3000) # Fixed color range for consistency.
axs_models_viz[0].set_title("Final Resistivity Model on Mesh")


# --- Right Subplot: Final P-wave Velocity Model ---
# `velocity_model_on_mesh` contains the calculated Vp for `geophysical_mesh`.
pg.show(geophysical_mesh, data=velocity_model_on_mesh, ax=axs_models_viz[1], orientation="vertical",
        cMap=custom_cmap_viz, cMin=500, cMax=5000, showColorBar=True, # Fixed color range.
        xlabel="Distance (m)", ylabel="Elevation (m)",
        label='P-wave Velocity (m/s)')
axs_models_viz[1].set_title("Final P-wave Velocity Model on Mesh")

# --- Print Overall Velocity Range for Debugging ---
# Ensure to use finite values for min/max if NaNs/Infs might be present.
finite_vel_viz = velocity_model_on_mesh[np.isfinite(velocity_model_on_mesh)]
if finite_vel_viz.size > 0:
    print(f"Overall plotting velocity model range: {np.min(finite_vel_viz):.2f} - {np.max(finite_vel_viz):.2f} m/s")
else:
    print("Overall plotting velocity model contains no finite values.")

# Adjust layout and save the figure.
plt.tight_layout(pad=3.0) # Increased padding.
# Save to the main output_dir (defined at the start of the script).
plt.savefig(os.path.join(output_dir, "Fig_Resistivity_Velocity_Models.tiff"), dpi=300) # Renamed figure file
plt.show()

# %% [markdown]
# ### ERT forward modeling simulation
# <!-- This section generates synthetic ERT data using the calculated resistivity model. -->

# %%

# --- Define ERT Survey Parameters ---
# Electrode positions (x-coordinates). Assumes a linear array.
# Creates 72 electrodes starting at x=15m with 1m spacing.
electrode_x_coords_ert = np.linspace(15, 15 + 72 - 1, 72) # Renamed
# Interpolate y-coordinates (elevations) for electrodes from the surface profile.
# `profile_interpolator` is from the earlier profile setup.
electrode_y_coords_ert = np.interp(electrode_x_coords_ert, profile_interpolator.L_profile, profile_interpolator.surface_profile) # Renamed
# Combine into a 2D array of [x, y] positions.
electrode_positions_ert = np.hstack((electrode_x_coords_ert.reshape(-1,1), electrode_y_coords_ert.reshape(-1,1))) # Renamed

# Create an ERT measurement scheme (e.g., Wenner alpha 'wa').
ert_measurement_scheme = ert.createData(elecs=electrode_positions_ert, schemeName='wa') # Renamed

# --- Prepare Mesh and Model for ERT Forward Modeling ---
# Step 10: Forward modeling to create synthetic ERT data.
print("Step 10: Performing ERT forward modeling to generate synthetic data...")

# The mesh used for inversion (`geophysical_mesh`) might need boundary elements for forward modeling.
# `simulation_grid_ert` will be this mesh with an appended triangular boundary.
# Cell markers are set: original mesh cells get marker 2, boundary cells get marker 1.
# This matches the setup in `hydro_to_ert.py` where `final_res_model_for_fwd` was created.
# We need to ensure `resistivity_model_on_mesh` is correctly mapped to this `simulation_grid_ert`.
geophysical_mesh.setCellMarkers(np.ones(geophysical_mesh.cellCount(), dtype=int) * 2) # Mark all original cells as 2
simulation_grid_ert = pg.meshtools.appendTriangleBoundary(geophysical_mesh, marker=1, # Boundary cells are marker 1
                                                          xbound=100, ybound=100) # Extent of boundary
                                                                                # SUGGESTION: xbound/ybound should be relative to mesh size.

# Create the full resistivity model for `simulation_grid_ert`.
# Values for original cells (marker 2) come from `resistivity_model_on_mesh`.
# Values for boundary cells (marker 1) get a default background resistivity.
ert_model_for_fwd_sim = np.ones(simulation_grid_ert.cellCount()) # Renamed
ert_model_for_fwd_sim[simulation_grid_ert.cellMarkers() == 2] = resistivity_model_on_mesh # Assign core model
ert_model_for_fwd_sim[simulation_grid_ert.cellMarkers() == 1] = 100.0 # Default background for boundary (Ohm-m)
                                                                # SUGGESTION: Make this background a parameter.

# The ERTForwardModeling class from this package is not used here; instead, PyGIMLi's ert.ERTModelling is used directly.
# This is fine, just a note on consistency if a wrapper class exists.
# fwd_operator_custom_instance = ERTForwardModeling(mesh=simulation_grid_ert, data=ert_measurement_scheme) # This instance is unused in the original script.


# --- Simulate ERT Data ---
synthetic_ert_data_container = ert_measurement_scheme.copy() # Create a data container for synthetic data. Renamed.
# Initialize PyGIMLi's forward operator.
fwd_op_gimli_ert_sim = ert.ERTModelling() # Renamed
fwd_op_gimli_ert_sim.setData(ert_measurement_scheme)
fwd_op_gimli_ert_sim.setMesh(simulation_grid_ert) # Use the grid with boundaries.
# Calculate "true" apparent resistivities using the prepared model.
true_apparent_resistivities_vec = fwd_op_gimli_ert_sim.response(ert_model_for_fwd_sim) # pg.Vector. Renamed.

# Add synthetic noise to the true data.
# Noise model: multiplicative Gaussian noise (5% relative error in this example, as per `noise_level_ert`).
# `pg.randn` creates normally distributed random numbers (mean 0, std 1).
noise_level_ert = 0.05 # Example noise level, make it a parameter if varied.
noisy_apparent_resistivities_vec = true_apparent_resistivities_vec * \
                                 (1.0 + pg.randn(true_apparent_resistivities_vec.size()) * noise_level_ert) # Renamed.

# Store noisy data and estimate errors for the data container.
synthetic_ert_data_container['rhoa'] = noisy_apparent_resistivities_vec
# Use ERTManager to estimate errors based on an error model (here, 0% absolute, 5% relative).
# These errors are typically used for weighting in the inversion.
ert_manager_for_error_est = ert.ERTManager(synthetic_ert_data_container) # Temporary manager. Renamed.
synthetic_ert_data_container['err'] = ert_manager_for_error_est.estimateError(
    synthetic_ert_data_container, absoluteUError=0.0, relativeError=0.05
)
# Display the synthetic ERT data as a pseudosection.
ert.showData(synthetic_ert_data_container, logscale=True, label=r"Synthetic Apparent Resistivity ($\Omega$m)") # Added label
plt.title("Synthetic ERT Data with Noise")
plt.show() # Display plot.


# %%
# --- Seismic Data Simulation ---
################## Seismic data #####################
# Step 11: Create seismic survey design and simulate synthetic travel time data.
print("Step 11: Creating seismic survey design and simulating data...")

# Define sensor (geophone) layout for the seismic survey.
num_geophones_srt = 72 # Number of geophones. Renamed.
shot_spacing_srt = 5.0  # Distance between shot points [m]. Renamed.
# X-coordinates for geophones, similar to ERT electrodes.
geophone_x_coords_srt = np.linspace(15, 15 + num_geophones_srt - 1, num_geophones_srt) # Renamed.

# Create a seismic refraction scheme using `pg.physics.traveltime.createRAData`.
# This defines shot-receiver pairs based on sensor locations and shot spacing.
srt_measurement_scheme = pg.physics.traveltime.createRAData(geophone_x_coords_srt, shotDistance=shot_spacing_srt) # Renamed.

# --- Drape Sensors onto Surface Topography ---
# `pos_srt_on_surface` will store the (x,y) coordinates of sensors on the actual surface. Renamed.
# `surface_line` (from `create_surface_lines`) defines the surface topography.
pos_srt_on_surface = np.zeros((num_geophones_srt, 2)) # Initialize (num_sensors, 2) array
for i in range(num_geophones_srt):
    # For each geophone's nominal x-position, find the corresponding surface elevation.
    # This uses linear interpolation based on the `surface_line` data.
    # `surface_line[:,0]` are x-distances, `surface_line[:,1]` are elevations.
    # np.interp is suitable here if surface_line x-coords are sorted.
    # The original loop was a bit complex using `np.where(minusx== np.amin(minusx))`; np.interp is more direct for this.
    # Assuming surface_line[:,0] (which is L_profile) are sorted distances for interpolation.
    current_sensor_x = geophone_x_coords_srt[i]
    # Check if sensor_x is within the bounds of the profile's x-coordinates to avoid extrapolation errors with interp.
    if surface_line[0,0] <= current_sensor_x <= surface_line[-1,0]:
        y_surface_interp = np.interp(current_sensor_x, surface_line[:,0], surface_line[:,1])
        pos_srt_on_surface[i, 0] = current_sensor_x
        pos_srt_on_surface[i, 1] = y_surface_interp
    else: # Fallback if sensor is outside profile range (e.g., use endpoint elevation and clamp x)
        if current_sensor_x < surface_line[0,0]:
            pos_srt_on_surface[i, 0] = surface_line[0,0]
            pos_srt_on_surface[i, 1] = surface_line[0,1]
        else: # current_sensor_x > surface_line[-1,0]
            pos_srt_on_surface[i, 0] = surface_line[-1,0]
            pos_srt_on_surface[i, 1] = surface_line[-1,1]
        print(f"Warning: Sensor {i} at x={current_sensor_x} is outside profile range. Clamped to endpoint x={pos_srt_on_surface[i,0]}.")


# Set the (potentially draped) sensor positions in the seismic scheme.
srt_measurement_scheme.setSensors(pos_srt_on_surface)


# --- Simulate Seismic Travel Times ---
# Initialize PyGIMLi's TravelTimeManager for the forward simulation.
srt_manager_sim = TravelTimeManager() # Renamed `mgr`
# `simulate` calculates travel times.
# It requires a slowness model (1/velocity). `velocity_model_on_mesh` is Vp [m/s].
# `mesh` here should be the geophysical mesh (`geophysical_mesh`) where `velocity_model_on_mesh` is defined.
# `noiseLevel` and `noiseAbs` add noise to the synthetic travel times.
synthetic_srt_data = srt_manager_sim.simulate(
    slowness=1.0 / velocity_model_on_mesh, # Convert velocity to slowness. Add epsilon if velocities can be 0.
    scheme=srt_measurement_scheme,
    mesh=geophysical_mesh, # Use the same mesh as for ERT property definition.
    noiseLevel=0.05,       # Relative noise (5%).
    noiseAbs=0.00001,      # Absolute noise (10 microseconds).
    seed=1334,             # Random seed for noise reproducibility.
    verbose=True
)


# %%
# --- Function to Plot First Arrival Travel Times ---
# This function is defined locally in the example.
# It helps visualize the travel time data (picks) for different shots.
def drawFirstPicks(ax, data, tt=None, plotva=False, **kwargs):
    """Plot first arrivals as lines.

    Parameters
    ----------
    ax : matplotlib.axes
        axis to draw the lines in
    data : :gimliapi:`GIMLI::DataContainer`
        data containing shots ("s"), geophones ("g") and traveltimes ("t")
    tt : array, optional
        traveltimes to use instead of data("t")
    plotva : bool, optional
        plot apparent velocity instead of traveltimes

    Return
    ------
    ax : matplotlib.axes
        the modified axis
    """
    # Extract coordinates
    # `pg.x(data)` gets unique x-positions of sensors from the data container.
    # This assumes sensors are primarily defined by their x-coordinate for this plot.
    sensor_x_coordinates = pg.x(data) # Unique sorted sensor x-positions.
    # Get geophone (receiver) x-positions for each measurement.
    geophone_x = np.array([sensor_x_coordinates[int(g)] for g in data("g")])
    # Get shot (source) x-positions for each measurement.
    shot_x = np.array([sensor_x_coordinates[int(s)] for s in data("s")])

    # Get traveltimes to plot.
    travel_times_plot: np.ndarray
    if tt is None:
        # If no specific travel times are provided, use the 't' token from the data container.
        travel_times_plot = np.array(data("t"))
    else:
        travel_times_plot = np.asarray(tt) # Ensure it's a NumPy array.

    y_axis_plot_label = "Traveltime (s)" # Default y-axis label
    if plotva: # If apparent velocity is requested.
        # Calculate apparent velocity: V_app = |x_geophone - x_shot| / travel_time.
        distance_offset = np.abs(geophone_x - shot_x)
        # Add small epsilon to travel_times_plot to prevent division by zero.
        travel_times_plot = distance_offset / (travel_times_plot + 1e-12)
        y_axis_plot_label = "Apparent velocity (m s$^{-1}$)"

    # Find unique source (shot) positions to plot each shot gather.
    unique_shot_positions = np.unique(shot_x)

    # Define default plotting style for lines.
    plot_kwargs = {'color': 'black', 'linestyle': '--', 'linewidth': 0.9, 'marker': None}
    plot_kwargs.update(kwargs) # Allow user to override defaults.

    # Plot data for each shot.
    for i_shot, shot_pos_x in enumerate(unique_shot_positions):
        # Select data for the current shot.
        mask_current_shot = (shot_x == shot_pos_x)
        times_current_shot = travel_times_plot[mask_current_shot]
        geophones_current_shot = geophone_x[mask_current_shot]

        # Sort by geophone position for a connected line plot.
        sort_indices_geophones = geophones_current_shot.argsort()

        # Plot the travel time curve (or apparent velocity curve).
        ax.plot(geophones_current_shot[sort_indices_geophones], times_current_shot[sort_indices_geophones], **plot_kwargs)

        # Add a marker for the shot position itself at y=0 (or top of plot if y-axis inverted).
        # This y-position might need adjustment depending on `plotva` and y-axis limits.
        shot_marker_y = 0.0
        if not plotva and ax.get_ylim()[0] > ax.get_ylim()[1]: # If y-axis inverted (travel times)
            shot_marker_y = ax.get_ylim()[1] # Place at visual top
        elif plotva and ax.get_ylim()[0] < ax.get_ylim()[1]: # If y-axis normal (velocities)
             shot_marker_y = ax.get_ylim()[0] # Place at visual bottom

        ax.plot(shot_pos_x, shot_marker_y, marker='s', color='black', markersize=4,
                markeredgecolor='black', markeredgewidth=0.5, linestyle='None')

    # Style the plot axes.
    ax.grid(True, linestyle='-', linewidth=0.2, color='lightgray')
    ax.set_ylabel(y_axis_plot_label)
    ax.set_xlabel("Distance (m)")

    # Invert y-axis for travel time plots (smaller times at top).
    if not plotva:
        ax.invert_yaxis()

    return ax

# --- Example Usage of drawFirstPicks ---
# Create a figure and axes for the travel time plot.
fig_srt_picks, ax_srt_picks = plt.subplots(figsize=(3.5, 2.5), dpi=300)
# Call the function to draw the first arrival picks on the created axes.
drawFirstPicks(ax_srt_picks, synthetic_srt_data) # Pass the synthetic SRT data.
plt.title("Synthetic First Arrival Travel Times") # Add a title.
plt.show() # Display the plot.

# %%
# --- Combined Plot of Resistivity, Velocity, ERT Data, and SRT Data ---
# This cell creates a 2x2 subplot figure to show various results from the workflow.
# import numpy as np # Already imported
# import matplotlib.pyplot as plt # Already imported
# import pygimli as pg # Already imported

# Assume `geophysical_mesh`, `resistivity_model_on_mesh`, `velocity_model_on_mesh`,
# `custom_velocity_resistivity_cmap` (Earth_7), `synthetic_ert_data_container`, `ert_measurement_scheme` (for ERT sensors),
# `synthetic_srt_data`, and `drawFirstPicks` are all defined from previous cells.

# Create 2x2 figure and axes. `hspace` and `wspace` control spacing between subplots.
fig_combined_results, axs_combined_results = plt.subplots(2, 2, figsize=(14, 10), # Renamed
                                                          gridspec_kw={'hspace': 0.1, 'wspace': 0.4})

# Flatten the 2x2 array of axes for easy indexing (ax1, ax2, ax3, ax4).
ax1_res_model, ax2_vel_model, ax3_ert_data, ax4_srt_data = axs_combined_results.flatten() # Renamed for clarity

# --- Top Left: Resistivity Model ---
# Show the true resistivity model used for ERT forward simulation.
pg.show(geophysical_mesh, data=resistivity_model_on_mesh, ax=ax1_res_model, orientation="vertical",
        cMap=custom_cmap_viz, logScale=True, showColorBar=True, # Using custom_cmap_viz from earlier cell
        xlabel="Distance (m)", ylabel="Elevation (m)",
        label='Resistivity (Ω·m)', cMin=100, cMax=3000)
ax1_res_model.set_title("True Resistivity Model")
# Draw ERT electrode positions on this plot.
pg.viewer.mpl.drawSensors(ax1_res_model, ert_measurement_scheme.sensors(), diam=0.8,
                         facecolor='black', edgecolor='black')


# --- Top Right: P-wave Velocity Model ---
# Show the true P-wave velocity model used for SRT forward simulation.
pg.show(geophysical_mesh, data=velocity_model_on_mesh, ax=ax2_vel_model, orientation="vertical",
        cMap=custom_cmap_viz, cMin=500, cMax=5000, showColorBar=True, # Using custom_cmap_viz
        xlabel="Distance (m)", ylabel="Elevation (m)",
        label='P-wave Velocity (m/s)')
ax2_vel_model.set_title("True P-wave Velocity Model")
# Draw sensor positions (same as ERT for this example, assuming co-located or similar line).
pg.viewer.mpl.drawSensors(ax2_vel_model, ert_measurement_scheme.sensors(), diam=0.8,
                         facecolor='black', edgecolor='black')


# Print velocity range for debugging, using finite values.
finite_vel_plot_combined = velocity_model_on_mesh[np.isfinite(velocity_model_on_mesh)] # Renamed
if finite_vel_plot_combined.size > 0:
    print(f"Plotting velocity range (combined plot): {np.min(finite_vel_plot_combined):.2f} - {np.max(finite_vel_plot_combined):.2f} m/s")

# --- Bottom Left: Synthetic ERT Data Pseudosection ---
# `ert.showData` displays apparent resistivity data in pseudosection format.
ert.showData(synthetic_ert_data_container, ax=ax3_ert_data, logscale=True, cMin=500, cMax=2000, cmap='jet', # Renamed ax
             label=r"Apparent Resistivity ($\Omega$m)")
ax3_ert_data.set_xlabel("Distance (m)")
ax3_ert_data.set_ylabel("Apparent Depth (m)") # Pseudosection y-axis is often apparent depth
ax3_ert_data.set_title("Synthetic ERT Data (Pseudosection)")
# Remove top and right spines for cleaner look, if desired.
# ax3_ert_data.spines['top'].set_visible(False)
# ax3_ert_data.spines['right'].set_visible(False)

# --- Bottom Right: Synthetic SRT First-Arrival Picks ---
# Use the `drawFirstPicks` function defined earlier.
drawFirstPicks(ax=ax4_srt_data, data=synthetic_srt_data) # Renamed ax
ax4_srt_data.set_title("Synthetic SRT Data (First Arrivals)")
# ax4_srt_data.set_xlabel("Distance (m)") # Already set by drawFirstPicks
# ax4_srt_data.set_ylabel("First arrival time (s)") # Already set by drawFirstPicks
# ax4_srt_data.spines['top'].set_visible(False)
# ax4_srt_data.spines['right'].set_visible(False)

plt.tight_layout() # Adjust layout to prevent overlaps.
plt.savefig(os.path.join(output_dir, "Fig2_Synthetic_Geophysical_Models_and_Data.png"), dpi=300) # Save figure
plt.show() # Display the plot.


# %%
# ### Run ERT inversion on synthetic data
# <!-- This section demonstrates inverting the synthetic ERT data generated previously. -->
# <!-- It shows usage of the custom ERTInversion class and compares with PyGIMLi's default ERTManager inversion. -->

# --- Inversion using the custom ERTInversion class from PyHydroGeophysX ---
print("Running ERT inversion using PyHydroGeophysX ERTInversion class...")
# Initialize the ERTInversion class.
# `data_file`: Path to the synthetic ERT data (saved previously or use `synthetic_ert_data_container` object directly if API allows).
# The example assumes data was saved. If not, `ERTInversion` might need to accept a DataContainer.
# For this example, let's assume synthetic_ert_data_container needs to be saved first if ERTInversion strictly needs a file.
synthetic_data_filepath = os.path.join(output_dir, "synthetic_data_for_inversion.dat") # Defined path for saving
synthetic_ert_data_container.save(synthetic_data_filepath) # Saved the data

custom_inverter = ERTInversion( # Renamed
    data_file=synthetic_data_filepath, # Use the saved synthetic data file.
    lambda_val=10.0,          # Regularization strength.
    method="cgls",            # Solver method.
    use_gpu=True,             # Attempt to use GPU (will fallback to CPU if not available).
    max_iterations=10,        # Maximum number of inversion iterations.
    lambda_rate= 1.0          # Lambda cooling factor (1.0 means constant lambda unless other strategy).
                               # A value < 1 (e.g. 0.8) would decrease lambda each iteration.
)
# Run the inversion. This will use the mesh created during `custom_inverter.setup()` if not provided in __init__.
# If `geophysical_mesh` or `simulation_grid_ert` should be used, it needs to be passed to ERTInversion constructor.
# For this example, let's pass the `simulation_grid_ert` (mesh with boundaries) to ensure consistency.
custom_inverter.set_mesh(simulation_grid_ert) # Explicitly set the mesh for inversion
custom_inversion_result = custom_inverter.run() # `initial_model` is None, so it will be auto-generated.

# %%
# --- Inversion using PyGIMLi's default ERTManager ---
# This provides a comparison to a standard PyGIMLi inversion workflow.
print("\nRunning ERT inversion using PyGIMLi ERTManager...")
ert_mgr_pygimli = ert.ERTManager(synthetic_data_filepath) # Load data into ERTManager. Renamed.
# `mgr.invert()` performs inversion. `lam` is regularization. `quality` for auto-mesh.
# `paraModel` after inversion will store the resulting resistivity model.
# `mgr.paraDomain` will be the mesh used/created for this inversion.
# To use the same mesh as custom_inverter for a fair comparison:
# ert_mgr_pygimli.setMesh(simulation_grid_ert) # This would require ensuring mesh is suitable for ERTManager's expectations.
# For default ERTManager behavior, it will create its own mesh.
inverted_resistivity_pygimli_model = ert_mgr_pygimli.invert(lam=10, verbose=True, quality=34) # Renamed `inv` to `inverted_resistivity_pygimli_model`

# %%
# --- Plot Comparison of True Model, Custom Inversion, and PyGIMLi Inversion ---
fig_comparison_inv, axes_comparison_inv = plt.subplots(1, 3, figsize=(18, 7)) # Increased figsize for 3 plots. Renamed.

# Plot 1: True Resistivity Model
# `ert_model_for_fwd_sim` was the true model on `simulation_grid_ert`.
ax_true_comp = axes_comparison_inv[0] # Renamed
pg.show(simulation_grid_ert, ert_model_for_fwd_sim, ax=ax_true_comp, cMap='jet', logScale=False,
        cMin=100, cMax=3000, label='Resistivity ($\Omega$m)') # Consistent units
ax_true_comp.set_title("True Resistivity Model")
pg.viewer.mpl.drawSensors(ax_true_comp, ert_measurement_scheme.sensors(), diam=0.8, facecolor='black', edgecolor='black')


# Plot 2: Inverted Model from PyHydroGeophysX ERTInversion
ax_custom_inv_comp = axes_comparison_inv[1] # Renamed
# `custom_inversion_result.mesh` is the mesh used by this inversion.
# `custom_inversion_result.final_model` is the inverted resistivity.
# `custom_inversion_result.coverage` can be used to mask poorly resolved areas.
pg.show(custom_inversion_result.mesh, custom_inversion_result.final_model, ax=ax_custom_inv_comp,
        cMap='jet', logScale=False, cMin=100, cMax=3000, label='Resistivity ($\Omega$m)',
        coverage=(custom_inversion_result.coverage > -1 if custom_inversion_result.coverage is not None else None)) # Example coverage threshold
ax_custom_inv_comp.set_title("Inverted Model (PyHydroGeophysX)")
pg.viewer.mpl.drawSensors(ax_custom_inv_comp, ert_measurement_scheme.sensors(), diam=0.8, facecolor='black', edgecolor='black')


# Plot 3: Inverted Model from PyGIMLi ERTManager
ax_pygimli_inv_comp = axes_comparison_inv[2] # Renamed
# `ert_mgr_pygimli.paraDomain` is the mesh, `ert_mgr_pygimli.paraModel()` is the model.
pg.show(ert_mgr_pygimli.paraDomain, ert_mgr_pygimli.paraModel(), ax=ax_pygimli_inv_comp,
        cMap='jet', logScale=False, cMin=100, cMax=3000, label='Resistivity ($\Omega$m)',
        coverage=(ert_mgr_pygimli.coverage() > -1 if ert_mgr_pygimli.coverage() is not None else None)) # Example coverage
ax_pygimli_inv_comp.set_title("Inverted Model (PyGIMLi Default)")
pg.viewer.mpl.drawSensors(ax_pygimli_inv_comp, ert_measurement_scheme.sensors(), diam=0.8, facecolor='black', edgecolor='black')

# Adjust layout for the comparison plot.
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "Fig3_Inversion_Comparison.png"), dpi=300) # Save comparison
plt.show()


# Informative print statement about the comparison.
print("# The inversion results are almost same from this code and Pygimli default inversion.")
print("# the difference is that the chi2 value for stop inversion is not the same, we chose 1.5 while Pygimli is 1.0")

# %% [markdown]
# ## One step approach
# <!-- This section demonstrates using higher-level wrapper functions for direct hydro-to-geophysical conversion. -->

# %% [markdown]
# ### ERT one step from HM to GM
# <!-- Shows direct conversion from hydrological model outputs to synthetic ERT data using `hydro_to_ert`. -->
# <!-- "HM to GM" might mean "Hydrological Model to Geophysical Model" or refer to specific model types. -->

# %%
# --- Setup Directories and Load Data (similar to start of script) ---
# This redefines some variables, ensure paths are correct.
output_dir_onestep_ert = os.path.join(output_dir, "onestep_ert_example") # Renamed for clarity
os.makedirs(output_dir_onestep_ert, exist_ok=True)

# Re-load data (or use previously loaded data if still in scope and appropriate)
# data_dir = "C:/Users/HChen8/Documents/GitHub/PyHydroGeophysX/examples/data/" # Already defined
# idomain = np.loadtxt(os.path.join(data_dir, "id.txt")) # Already defined
# top = np.loadtxt(os.path.join(data_dir, "top.txt")) # Already defined
# porosity = np.load(os.path.join(data_dir, "Porosity.npy")) # Already defined
# water_content_step50 = np.load(os.path.join(data_dir, "Watercontent.npy"))[50]  # Time step 50. Renamed.
# Using previously loaded `water_content` for consistency.

# --- Re-Setup Profile Interpolator and Mesh (as done in steps 3-5) ---
# This is needed because `hydro_to_ert` requires these as inputs.
# `point1_indices`, `point2_indices` are from earlier definition.
profile_interpolator_onestep = ProfileInterpolator( # Renamed
    point1=point1_indices, point2=point2_indices, surface_data=top,
    origin_x=569156.0, origin_y=4842444.0, pixel_width=1.0, pixel_height=-1.0, num_points=400
)
# Re-create mesh structure inputs. `layer_boundary_data_3d` from earlier.
structure_on_profile_onestep = profile_interpolator_onestep.interpolate_layer_data([top] + layer_boundary_data_3d.tolist()) # Renamed
# `layer_interface_idx_...` are from earlier.
surface_line_os, boundary_line1_os, boundary_line2_os = create_surface_lines( # Renamed outputs
    L_profile=profile_interpolator_onestep.L_profile, structure=structure_on_profile_onestep,
    top_idx=layer_interface_idx_top, mid_idx=layer_interface_idx_middle, bot_idx=layer_interface_idx_bottom
)
mesh_creator_os = MeshCreator(quality=32) # Renamed
geophysical_mesh_os, _geom_plc_os = mesh_creator_os.create_from_layers( # Renamed mesh and geom
    surface=surface_line_os, layers=[boundary_line1_os, boundary_line2_os],
    bottom_depth=np.min(boundary_line2_os[:,1])-10
)

# --- Define Layer Markers and Petrophysical Parameters for `hydro_to_ert` ---
# These should be consistent with the layers defined in the mesh and expected by the function.
# `defined_marker_labels` were [0,3,2] earlier. Assuming these map to top, middle, bottom based on `rho_parameters`.
layer_markers_for_hydro_to_ert = defined_marker_labels
# Resistivity parameters for each layer, corresponding to `layer_markers_for_hydro_to_ert`.
resistivity_params_for_hydro_to_ert = {
    'rho_sat': [100.0, 500.0, 2400.0],      # rho_sat for marker 0, then 3, then 2
    'n': [2.2, 1.8, 2.5],                   # n for marker 0, then 3, then 2
    'sigma_s': [1.0/500.0, 0.0, 0.0]        # sigma_s for marker 0, then 3, then 2
}
# Get cell markers from the created mesh.
mesh_cell_markers_for_hydro_to_ert = np.array(geophysical_mesh_os.cellMarkers()) # Renamed


# --- Call `hydro_to_ert` for Direct Conversion and Forward Modeling ---
# This function encapsulates steps from hydrological data interpolation to synthetic ERT data generation.
synthetic_ert_data_onestep, ert_resistivity_model_onestep = hydro_to_ert( # Renamed outputs
    water_content=water_content, # Using the single timestep water content loaded earlier.
    porosity=porosity,           # Full 3D porosity field.
    mesh=geophysical_mesh_os,    # The generated geophysical mesh.
    mesh_markers=mesh_cell_markers_for_hydro_to_ert, # Markers for this mesh.
    profile_interpolator=profile_interpolator_onestep, # Profile interpolator instance.
    layer_idx=[layer_interface_idx_top, layer_interface_idx_middle, layer_interface_idx_bottom], # Indices for layer boundaries in hydro data. (Original was layer_idx, using more descriptive name from context)
    structure=structure_on_profile_onestep, # Interpolated hydro layer depths on profile.
    marker_labels=layer_markers_for_hydro_to_ert, # Markers to assign properties to.
    rho_parameters=resistivity_params_for_hydro_to_ert, # Petrophysical parameters.
    electrode_spacing=1.0,    # ERT survey params
    electrode_start=15.0,     # Start position of electrode array.
    num_electrodes=72,
    scheme_name='wa',         # Wenner alpha array.
    noise_level=0.05,         # 5% relative noise.
    abs_error=0.0,            # Absolute error component for error model.
    rel_error=0.05,           # Relative error component for error model.
    save_path=os.path.join(output_dir_onestep_ert, "synthetic_ert_data_onestep.dat"), # Save output.
    verbose=True,
    seed=42                   # Random seed for noise.
)

# Display the generated synthetic ERT data.
ert.showData(synthetic_ert_data_onestep, logscale=True, label=r"Apparent Resistivity ($\Omega$m) - One Step")
plt.title("Synthetic ERT Data (One-Step Workflow)")
plt.show()

# %% [markdown]
# ### SRT one step from HM to GM
# <!-- Shows direct conversion from hydrological model outputs to synthetic SRT data using `hydro_to_srt`. -->

# %%
# --- Re-import modules if running this section independently ---
# import os #(already imported)
# import numpy as np #(already imported)
# import matplotlib.pyplot as plt #(already imported)
# import pygimli as pg #(already imported)
# from PyHydroGeophysX.core.interpolation import ProfileInterpolator, create_surface_lines #(already imported)
# from PyHydroGeophysX.core.mesh_utils import MeshCreator #(already imported)
from PyHydroGeophysX.Hydro_modular.hydro_to_srt import hydro_to_srt # Specific import for this section.

# --- Setup Directories and Load Data (largely repeated from ERT one-step, ensure consistency or reuse) ---
output_dir_onestep_srt = os.path.join(output_dir, "onestep_srt_example") # Renamed
os.makedirs(output_dir_onestep_srt, exist_ok=True)

# Data loading (assuming these are still the same as defined earlier in the script)
# data_dir = "C:/Users/HChen8/Documents/GitHub/PyHydroGeophysX/examples/data/"
# idomain = np.loadtxt(os.path.join(data_dir, "id.txt"))
# top = np.loadtxt(os.path.join(data_dir, "top.txt"))
# porosity = np.load(os.path.join(data_dir, "Porosity.npy"))
# water_content = np.load(os.path.join(data_dir, "Watercontent.npy"))[50] # Using `water_content` from earlier

# Profile Interpolator (assuming same profile as for ERT one-step, using `profile_interpolator_onestep`)
# Mesh Structure (assuming same mesh structure and layers as for ERT one-step, using `geophysical_mesh_os`)
# Layer boundary indices (`layer_interface_idx_top`, etc.) and `structure_on_profile_onestep` are reused.
# Layer markers `layer_markers_for_hydro_to_ert` are reused.

# --- Define Rock Physics Parameters for Velocity Calculation (for `hydro_to_srt`) ---
# This dictionary defines parameters for Hertz-Mindlin and DEM models for different layers.
# The keys ('top', 'mid', 'bot') should correspond to a mapping strategy within `hydro_to_srt`
# which then links them to `marker_labels`.
velocity_petro_params = { # Renamed
    'top': { # Parameters for the top layer (e.g., regolith, likely marker_labels[0])
        'bulk_modulus': 30.0,         # Matrix bulk modulus [GPa]
        'shear_modulus': 20.0,        # Matrix shear modulus [GPa]
        'mineral_density': 2650.0,    # Matrix mineral density [kg/m³]
        'depth': 1.0                  # Effective depth for Hertz-Mindlin pressure [m]
    },
    'mid': { # Parameters for the middle layer (e.g., fractured bedrock, marker_labels[1])
        'bulk_modulus': 50.0,
        'shear_modulus': 35.0,
        'mineral_density': 2670.0,
        'aspect_ratio': 0.05          # Pore aspect ratio for DEM model
    },
    'bot': { # Parameters for the bottom layer (e.g., fresh bedrock, marker_labels[2])
        'bulk_modulus': 55.0,
        'shear_modulus': 50.0,
        'mineral_density': 2680.0,
        'aspect_ratio': 0.03
    }
}
# Cell markers from the mesh that `hydro_to_srt` will use for assigning these parameters.
# mesh_cell_markers_for_hydro_to_srt = np.array(geophysical_mesh_os.cellMarkers()) # Renamed

# --- Call `hydro_to_srt` for Direct Conversion and SRT Forward Modeling ---
# This function encapsulates interpolation, petrophysical conversion to velocity, and SRT forward modeling.
synthetic_srt_data_onestep, srt_velocity_model_onestep = hydro_to_srt( # Renamed outputs
    water_content=water_content,       # Using the single timestep water content.
    porosity=porosity,                 # Full 3D porosity field.
    mesh=geophysical_mesh_os,          # The common geophysical mesh.
    profile_interpolator=profile_interpolator_onestep, # Profile interpolator.
    layer_idx=[layer_interface_idx_top, layer_interface_idx_middle, layer_interface_idx_bottom], # Hydro layer boundary indices.
    structure=structure_on_profile_onestep, # Interpolated hydro layer depths on profile.
    marker_labels=layer_markers_for_hydro_to_ert, # Markers for assigning petro params.
    vel_parameters=velocity_petro_params, # Velocity petrophysical parameters.
    sensor_spacing=1.0,
    sensor_start=15.0,
    num_sensors=72,
    shot_distance=5.0, # Original was int, float is fine.
    noise_level=0.05,
    noise_abs=0.00001,
    save_path=os.path.join(output_dir_onestep_srt, "synthetic_seismic_data_onestep.dat"), # Save output.
    mesh_markers=mesh_cell_markers_for_hydro_to_ert, # Pass the mesh markers.
    verbose=True,
    seed=1334
)

# --- Visualize the Results from `hydro_to_srt` ---
from PyHydroGeophysX.forward.srt_forward import SeismicForwardModeling # For drawFirstPicks helper

# Create a figure with two subplots (velocity model and travel time data).
fig_srt_onestep, axes_srt_onestep = plt.subplots(2, 1, figsize=(10, 10)) # Renamed

# Plot the generated P-wave velocity model on the mesh.
pg.show(geophysical_mesh_os, data=srt_velocity_model_onestep, ax=axes_srt_onestep[0],
        cMap='viridis', # Changed colormap for variety
        cMin=500, cMax=5000, label='P-wave Velocity (m/s)', showColorBar=True,
        xlabel="Distance (m)", ylabel="Elevation (m)")
axes_srt_onestep[0].set_title("P-wave Velocity Model (One-Step Workflow)")

# Plot the synthetic first-arrival travel times.
SeismicForwardModeling.draw_first_picks(axes_srt_onestep[1], synthetic_srt_data_onestep)
axes_srt_onestep[1].set_title('Synthetic First-Arrival Travel Times (One-Step Workflow)')

plt.tight_layout()
plt.savefig(os.path.join(output_dir_onestep_srt, "Fig_SRT_OneStep_Results.png"), dpi=300) # Save figure. Added .png
plt.show()


# %% [markdown]
# ### Calculating saturation

# %%


# Ensure porosity is not zero to avoid division by zero
porosity_safe = np.maximum(porosity_mesh, 0.01)
saturation = np.clip(wc_mesh / porosity_safe, 0.0, 1.0)


# %% [markdown]
# ### Converting to resistivity

# %%

# Step 8: Convert to resistivity using petrophysical model


marker_labels = [0, 3, 2] # top. mid, bottom layers (example values)
rho_sat = [100, 500, 2400] # Saturated resistivity for each layer (example values)
n = [2.2, 1.8, 2.5] # Cementation exponent for each layer (example values)
sigma_s = [1/500, 0, 0] # Saturated resistivity of the surface conductivity see Chen & Niu, (2022) for each layer (example values)
# Convert water content back to resistivity

res_models = np.zeros_like(wc_mesh)  # Initialize an array for resistivity values

mask = (mesh_markers == marker_labels[0])
top_res = water_content_to_resistivity(
    wc_mesh[mask],                  # Water content values for this layer
    float(rho_sat[0]),              # Use a scalar value instead of an array
    float(n[0]),                    # Use a scalar value instead of an array
    porosity_mesh[mask],            # Porosity values for this layer
    sigma_s[0] # Use a scalar value
)
res_models[mask] = top_res

mask = (mesh_markers == marker_labels[1])
mid_res = water_content_to_resistivity(
    wc_mesh[mask],                  # Water content values for this layer
    float(rho_sat[1]),              # Use a scalar value instead of an array
    float(n[1]),                    # Use a scalar value instead of an array
    porosity_mesh[mask],            # Porosity values for this layer
    sigma_s[1]  # Use a scalar value
)
res_models[mask] = mid_res


mask = (mesh_markers == marker_labels[2])
bot_res = water_content_to_resistivity(
    wc_mesh[mask],                  # Water content values for this layer
    float(rho_sat[2]),              # Use a scalar value instead of an array
    float(n[2]),                    # Use a scalar value instead of an array
    porosity_mesh[mask],            # Porosity values for this layer
    sigma_s[2]
)
res_models[mask] = bot_res


# %%
print(np.min(top_res), np.max(top_res))
print(np.min(mid_res), np.max(mid_res))
print(np.min(bot_res), np.max(bot_res))

# %% [markdown]
# ### Converting to P wave velocity

# %%
# Step 9: Convert to P wave velocity using petrophysical model


# Initialize velocity models
hm_model = HertzMindlinModel(critical_porosity=0.4, coordination_number=6.0)
dem_model = DEMModel()

# Initialize velocity model
velocity_mesh = np.zeros_like(wc_mesh)




top_mask = (mesh_markers == marker_labels[0])
top_bulk_modulus = 30.0  # GPa
top_shear_modulus = 20.0  # GPa
top_mineral_density = 2650  # kg/m³
top_depth = 1.0  # m

# Get Vp values using Hertz-Mindlin model
Vp_high, Vp_low = hm_model.calculate_velocity(
    porosity=porosity_mesh[top_mask],
    saturation=saturation[top_mask],
    bulk_modulus=top_bulk_modulus,
    shear_modulus=top_shear_modulus,
    mineral_density=top_mineral_density,
    depth=top_depth
)

# Use average of high and low bounds
velocity_mesh[top_mask] = (Vp_high + Vp_low) / 2



mid_mask = (mesh_markers == marker_labels[1])

mid_bulk_modulus = 50.0  # GPa
mid_shear_modulus = 35.0 # GPa
mid_mineral_density = 2670  # kg/m³
mid_aspect_ratio = 0.05

# Get Vp values using DEM model
_, _, Vp = dem_model.calculate_velocity(
    porosity=porosity_mesh[mid_mask],
    saturation=saturation[mid_mask],
    bulk_modulus=mid_bulk_modulus,
    shear_modulus=mid_shear_modulus,
    mineral_density=mid_mineral_density,
    aspect_ratio=mid_aspect_ratio
)

velocity_mesh[mid_mask] = Vp

bot_mask = (mesh_markers == marker_labels[2])
bot_bulk_modulus = 55  # GPa
bot_shear_modulus = 50  # GPa
bot_mineral_density = 2680  # kg/m³
bot_aspect_ratio = 0.03

# Get Vp values using DEM model
_, _, Vp = dem_model.calculate_velocity(
    porosity=porosity_mesh[bot_mask],
    saturation=saturation[bot_mask],
    bulk_modulus=bot_bulk_modulus,
    shear_modulus=bot_shear_modulus,
    mineral_density=bot_mineral_density,
    aspect_ratio=bot_aspect_ratio
)

velocity_mesh[bot_mask] = Vp

# %%
print(np.min(velocity_mesh[top_mask]), np.max(velocity_mesh[top_mask]))
print(np.min(velocity_mesh[mid_mask]), np.max(velocity_mesh[mid_mask]))
print(np.min(velocity_mesh[bot_mask]), np.max(velocity_mesh[bot_mask]))

# %%
from palettable.lightbartlein.diverging import BlueDarkRed18_18
fixed_cmap = BlueDarkRed18_18.mpl_colormap



# --- Create figure with 1 row, 2 columns ---
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# --- Left: Resistivity with log scale ---
pg.show(mesh, res_models, ax=axs[0], orientation="vertical",
        cMap=fixed_cmap, logScale=True, showColorBar=True,
        xlabel="Distance (m)", ylabel="Elevation (m)",
        label='Resistivity (Ω·m)', cMin=500, cMax=3000)


# --- Right: P-wave velocity with fixed color scale ---
pg.show(mesh, velocity_mesh, ax=axs[1], orientation="vertical",
        cMap=fixed_cmap, cMin=500, cMax=5000, showColorBar=True,
        xlabel="Distance (m)", ylabel="Elevation (m)",
        label='Velocity (m/s)')

# --- Print value range for debugging ---
print("Velocity range:", np.min(velocity_mesh), np.max(velocity_mesh))

# --- Final layout ---
plt.tight_layout(pad=3)

plt.savefig(os.path.join(output_dir, "res_vel.tiff"), dpi=300)

# %% [markdown]
# ### ERT forward modeling simulation

# %%



xpos = np.linspace(15,15+72 - 1,72)
ypos = np.interp(xpos,interpolator.L_profile,interpolator.surface_profile)
pos = np.hstack((xpos.reshape(-1,1),ypos.reshape(-1,1)))

schemeert = ert.createData(elecs=pos,schemeName='wa')

# Step 10: Forward modeling to create synthetic ERT data

mesh.setCellMarkers(np.ones(mesh.cellCount())*2)
grid = pg.meshtools.appendTriangleBoundary(mesh, marker=1,
                                          xbound=100, ybound=100)

fwd_operator = ERTForwardModeling(mesh=grid, data=schemeert)


synth_data = schemeert.copy()
fob = ert.ERTModelling()
fob.setData(schemeert)
fob.setMesh(grid)
dr = fob.response(res_models)

dr *= 1. + pg.randn(dr.size()) * 0.05
ert_manager = ert.ERTManager(synth_data)
synth_data['rhoa'] = dr
synth_data['err'] = ert_manager.estimateError(synth_data, absoluteUError=0.0, relativeError=0.05)
ert.showData(synth_data,  logscale=True)


# %%
################## Seismic data #####################

print("Step 11: Creating seismic survey design...")

numberGeophones = 72
shotDistance = 5
sensors = np.linspace(15,15 + 72 - 1, numberGeophones)
scheme = pg.physics.traveltime.createRAData(sensors,shotDistance=shotDistance)



for i in range(numberGeophones):
    minusx = np.abs(surface[:,0]-sensors[i])
    index = np.where(minusx== np.amin(minusx))
    new_x = surface[index,0]
    new_y = surface[index,1]
    pos[i, 0] = new_x
    pos[i, 1] = new_y


scheme.setSensors(pos)


mgr = TravelTimeManager()
datasrt = mgr.simulate(slowness=1.0 / velocity_mesh, scheme=scheme, mesh=mesh,
                    noiseLevel=0.05, noiseAbs=0.00001, seed=1334
                    ,verbose=True)


# %%
def drawFirstPicks(ax, data, tt=None, plotva=False, **kwargs):
    """Plot first arrivals as lines.
    
    Parameters
    ----------
    ax : matplotlib.axes
        axis to draw the lines in
    data : :gimliapi:`GIMLI::DataContainer`
        data containing shots ("s"), geophones ("g") and traveltimes ("t")
    tt : array, optional
        traveltimes to use instead of data("t")
    plotva : bool, optional
        plot apparent velocity instead of traveltimes
    
    Return
    ------
    ax : matplotlib.axes
        the modified axis
    """
    # Extract coordinates
    px = pg.x(data)
    gx = np.array([px[int(g)] for g in data("g")])
    sx = np.array([px[int(s)] for s in data("s")])
    
    # Get traveltimes
    if tt is None:
        tt = np.array(data("t"))
    if plotva:
        tt = np.absolute(gx - sx) / tt
    
    # Find unique source positions    
    uns = np.unique(sx)
    
    # Override kwargs with clean, minimalist style
    kwargs['color'] = 'black'
    kwargs['linestyle'] = '--'
    kwargs['linewidth'] = 0.9
    kwargs['marker'] = None  # No markers on the lines
    
    # Plot for each source
    for i, si in enumerate(uns):
        ti = tt[sx == si]
        gi = gx[sx == si]
        ii = gi.argsort()
        
        # Plot line
        ax.plot(gi[ii], ti[ii], **kwargs)
        
        # Add source marker as black square at top
        ax.plot(si, 0.0, 's', color='black', markersize=4, 
                markeredgecolor='black', markeredgewidth=0.5)
    
    # Clean grid style
    ax.grid(True, linestyle='-', linewidth=0.2, color='lightgray')
    
    # Set proper axis labels with units
    if plotva:
        ax.set_ylabel("Apparent velocity (m s$^{-1}$)")
    else:
        ax.set_ylabel("Traveltime (s)")
    
    ax.set_xlabel("Distance (m)")
    

    

    
    # Invert y-axis for traveltimes
    ax.invert_yaxis()

    return ax

# Usage
fig, ax = plt.subplots(figsize=(3.5, 2.5), dpi=300) 
drawFirstPicks(ax, datasrt)

# %%
import numpy as np
import matplotlib.pyplot as plt
import pygimli as pg

# Assume mesh, res_models, velocity_mesh, fixed_cmap, synth_data, datasrt, ert, drawFirstPicks are already defined

# Create 2×2 axes
fig, axs = plt.subplots(2, 2, figsize=(14, 10),
                        gridspec_kw={'hspace': 0.1, 'wspace': 0.4})

# Flatten for easy indexing
ax1, ax2, ax3, ax4 = axs.flatten()

# --- Top left: Resistivity (log scale) ---
pg.show(mesh, res_models, ax=ax1, orientation="vertical",
        cMap=fixed_cmap, logScale=True, showColorBar=True,
        xlabel="Distance (m)", ylabel="Elevation (m)",
        label='Resistivity (Ω·m)', cMin=100, cMax=3000)
# Invert y (so elevation decreases downward)
pg.viewer.mpl.drawSensors(ax1, schemeert.sensors(), diam=0.8,
                         facecolor='black', edgecolor='black')


# --- Top right: P-wave velocity (fixed scale) ---
pg.show(mesh, velocity_mesh, ax=ax2, orientation="vertical",
        cMap=fixed_cmap, cMin=500, cMax=5000, showColorBar=True,
        xlabel="Distance (m)", ylabel="Elevation (m)",
        label='Velocity (m/s)')
pg.viewer.mpl.drawSensors(ax2, schemeert.sensors(), diam=0.8,
                         facecolor='black', edgecolor='black')


# Print range for debugging
print("Velocity range:", np.min(velocity_mesh), np.max(velocity_mesh))

# --- Bottom left: Synthetic ERT data ---
ert.showData(synth_data, logscale=True, ax=ax3, cMin=500, cMax=2000,cmap='jet')

ax3.set_xlabel("Distance (m)")
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

# --- Bottom right: First-break picks ---
drawFirstPicks(ax=ax4, data=datasrt)

ax4.set_xlabel("Distance (m)")
ax4.set_ylabel("First arrival time (s)")
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)

plt.tight_layout()



# %%
# ### Run ERT inversion on synthetic data

# using my code to the inversion

# Create ERT inversion object
inversion = ERTInversion(
    data_file=os.path.join(output_dir, "synthetic_data.dat"),
    lambda_val=10.0,
    method="cgls",
    use_gpu=True,
    max_iterations=10,
    lambda_rate= 1.0
)
inversion_result = inversion.run()

# %%
# ### Using Pygimili default to the inversion
mgr = ert.ERTManager(os.path.join(output_dir, "synthetic_data.dat"))
inv = mgr.invert(lam=10, verbose=True,quality=34)

# %%
fig, axes = plt.subplots(1, 3, figsize=(10, 12))

# True resistivity model
ax1 = axes[0]
cbar1 = pg.show(mesh, res_models, ax=ax1, cMap='jet', logScale=False, 
              cMin=100, cMax=3000, label='Resistivity [Ohm-m]')
ax1.set_title("True Resistivity Model")

# Inverted model
ax2 = axes[1]
cbar2 = pg.show(inversion_result.mesh, inversion_result.final_model, ax=ax2, cMap='jet', logScale=False, 
              cMin=100, cMax=3000, label='Resistivity [Ohm-m]',coverage=inversion_result.coverage>-1)
ax2.set_title("Inverted Resistivity Model (Our Code)")

ax3 = axes[2]
cbar2 = pg.show(mgr.paraDomain, mgr.paraModel(), ax=ax3, cMap='jet', logScale=False, 
              cMin=100, cMax=3000, label='Resistivity [Ohm-m]',coverage=mgr.coverage()>-1)
ax3.set_title("Inverted Resistivity Model (Pygimli)")
# Adjust layout
plt.tight_layout()



# The inversion results are almost same from this code and Pygimli default inversion.
# the difference is that the chi2 value for stop inversion is not the same, we chose 1.5 while Pygimli is 1.0

# %% [markdown]
# ## One step approach

# %% [markdown]
# ### ERT one step from HM to GM

# %%
# Set up directories
output_dir = "C:/Users/HChen8/Documents/GitHub/PyHydroGeophysX/examples/results/hydro_to_ert_example"
os.makedirs(output_dir, exist_ok=True)

# Load your data
data_dir = "C:/Users/HChen8/Documents/GitHub/PyHydroGeophysX/examples/data/"
idomain = np.loadtxt(os.path.join(data_dir, "id.txt"))
top = np.loadtxt(os.path.join(data_dir, "top.txt"))
porosity = np.load(os.path.join(data_dir, "Porosity.npy"))
water_content = np.load(os.path.join(data_dir, "Watercontent.npy"))[50]  # Time step 50

# Set up profile
point1 = [115, 70]  
point2 = [95, 180]  

interpolator = ProfileInterpolator(
    point1=point1,
    point2=point2,
    surface_data=top,
    origin_x=569156.0,
    origin_y=4842444.0,
    pixel_width=1.0,
    pixel_height=-1.0,
    num_points=400
)

# Create mesh structure
bot = np.load(os.path.join(data_dir, "bot.npy"))
layer_idx = [0, 4, 12]  # Example indices for top, middle, and bottom layers
structure = interpolator.interpolate_layer_data([top] + bot.tolist())
surface, line1, line2 = create_surface_lines(
    L_profile=interpolator.L_profile,
    structure=structure,
    top_idx=layer_idx[0],
    mid_idx=layer_idx[1],
    bot_idx=layer_idx[2]
)

# Create mesh
mesh_creator = MeshCreator(quality=32)
mesh, geom = mesh_creator.create_from_layers(
    surface=surface,
    layers=[line1, line2],
    bottom_depth=np.min(line2[:,1])-10
)

# Define layer markers
marker_labels = [0, 3, 2]  # top, middle, bottom layers

# Define resistivity parameters for each layer
rho_parameters = {
    'rho_sat': [100, 500, 2400],      # Saturated resistivity for each layer (Ohm-m)
    'n': [2.2, 1.8, 2.5],             # Cementation exponent for each layer
    'sigma_s': [1/500, 0, 0]          # Surface conductivity for each layer (S/m)
}

mesh_markers = np.array(mesh.cellMarkers())


# Generate ERT response directly
synth_data, res_model = hydro_to_ert(
    water_content=water_content,
    porosity=porosity,
    mesh=mesh,
    mesh_markers = mesh_markers,
    profile_interpolator=interpolator,
    layer_idx=layer_idx,
    structure = structure,
    marker_labels=marker_labels,
    rho_parameters=rho_parameters,
    electrode_spacing=1.0,
    electrode_start=15,
    num_electrodes=72,
    scheme_name='wa',
    noise_level=0.05,
    abs_error=0.0,
    rel_error=0.05,
    save_path=os.path.join(output_dir, "synthetic_ert_data.dat"),
    verbose=True,
    seed=42,
)

ert.showData(synth_data,  logscale=True)

# %% [markdown]
# ### SRT one step from HM to GM

# %%
import os
import numpy as np
import matplotlib.pyplot as plt
import pygimli as pg

# Import PyHydroGeophysX modules
from PyHydroGeophysX.core.interpolation import ProfileInterpolator, create_surface_lines
from PyHydroGeophysX.core.mesh_utils import MeshCreator
from PyHydroGeophysX.Hydro_modular.hydro_to_srt import hydro_to_srt

# 1. Set up output directory
output_dir = "C:/Users/HChen8/Documents/GitHub/PyHydroGeophysX/examples/results/srt_example"
os.makedirs(output_dir, exist_ok=True)

# Load your data
data_dir = "C:/Users/HChen8/Documents/GitHub/PyHydroGeophysX/examples/data/"
idomain = np.loadtxt(os.path.join(data_dir, "id.txt"))
top = np.loadtxt(os.path.join(data_dir, "top.txt"))
porosity = np.load(os.path.join(data_dir, "Porosity.npy"))
water_content = np.load(os.path.join(data_dir, "Watercontent.npy"))[50]  # Time step 50

# Set up profile
point1 = [115, 70]  
point2 = [95, 180]  

interpolator = ProfileInterpolator(
    point1=point1,
    point2=point2,
    surface_data=top,
    origin_x=569156.0,
    origin_y=4842444.0,
    pixel_width=1.0,
    pixel_height=-1.0,
    num_points=400
)

# Create mesh structure
bot = np.load(os.path.join(data_dir, "bot.npy"))
layer_idx = [0, 4, 12]  # Example indices for top, middle, and bottom layers
structure = interpolator.interpolate_layer_data([top] + bot.tolist())
surface, line1, line2 = create_surface_lines(
    L_profile=interpolator.L_profile,
    structure=structure,
    top_idx=layer_idx[0],
    mid_idx=layer_idx[1],
    bot_idx=layer_idx[2]
)

# Create mesh
mesh_creator = MeshCreator(quality=32)
mesh, geom = mesh_creator.create_from_layers(
    surface=surface,
    layers=[line1, line2],
    bottom_depth=np.min(line2[:,1])-10
)

# Define layer markers
marker_labels = [0, 3, 2]  # top, middle, bottom layers

# Rock physics parameters for each layer
vel_parameters = {
    'top': {
        'bulk_modulus': 30.0,         # GPa
        'shear_modulus': 20.0,        # GPa
        'mineral_density': 2650,      # kg/m³
        'depth': 1.0                  # m
    },
    'mid': {
        'bulk_modulus': 50.0,         # GPa
        'shear_modulus': 35.0,        # GPa
        'mineral_density': 2670,      # kg/m³
        'aspect_ratio': 0.05          # Crack aspect ratio
    },
    'bot': {
        'bulk_modulus': 55.0,         # GPa
        'shear_modulus': 50.0,        # GPa
        'mineral_density': 2680,      # kg/m³
        'aspect_ratio': 0.03          # Crack aspect ratio
    }
}
mesh_markers = np.array(mesh.cellMarkers())
# 13. Now we call hydro_to_srt with the pre-processed mesh values
synth_data, velocity_mesh = hydro_to_srt(
    water_content=water_content,           # Use pre-interpolated mesh values
    porosity=porosity,          # Use pre-interpolated mesh values
    mesh=mesh,
    profile_interpolator=interpolator,
    layer_idx=layer_idx,
    structure = structure,
    marker_labels=marker_labels,
    vel_parameters=vel_parameters,
    sensor_spacing=1.0,              
    sensor_start=15.0,               
    num_sensors=72,                  
    shot_distance=5,                 
    noise_level=0.05,                
    noise_abs=0.00001,               
    save_path=os.path.join(output_dir, "synthetic_seismic_data.dat"),
    mesh_markers=mesh_markers,       # Pass the mesh markers directly
    verbose=True,
    seed=1334                        
)

# 14. Visualize the results
from PyHydroGeophysX.forward.srt_forward import SeismicForwardModeling

# Create a figure
fig, axes = plt.subplots(2, 1, figsize=(10, 10))

# Plot velocity model
pg.show(mesh, velocity_mesh, ax=axes[0], cMap='jet', 
        cMin=500, cMax=5000, label='Velocity (m/s)',
        xlabel="Distance (m)", ylabel="Elevation (m)")

# Plot first-arrival travel times
SeismicForwardModeling.draw_first_picks(axes[1], synth_data)
axes[1].set_title('Synthetic First-Arrival Travel Times')

plt.tight_layout()