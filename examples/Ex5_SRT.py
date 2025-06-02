# Script-level comment:
# This script, Ex5_SRT.py, demonstrates a workflow for Seismic Refraction Tomography (SRT),
# starting from hydrological model outputs, converting them to seismic velocity,
# performing SRT forward modeling to generate synthetic travel times, and finally,
# inverting these synthetic travel times to recover a velocity model.
# The example shows this for a "long" profile and then briefly for a "short" profile.
#
# Workflow steps covered:
# 1. Hydrological to Velocity Model:
#    - Loads hydrological model data (domain info, porosity, water content for a specific timestep).
#    - Sets up a 2D profile and interpolates hydrological data onto it.
#    - Creates a 2D geophysical mesh conforming to geological structures.
#    - Maps profile data (porosity, water content) to the mesh cells.
#    - Calculates saturation from water content and porosity.
#    - Converts mesh-based hydrological properties to a P-wave velocity (Vp) model
#      using rock physics models (Hertz-Mindlin, DEM).
# 2. SRT Survey Setup and Forward Modeling:
#    - Defines seismic survey geometry (geophone locations, shot spacing) for a "long" profile.
#    - Drapes sensor positions onto the surface topography.
#    - Generates synthetic first-arrival travel time data using the Vp model and PyGIMLi's TravelTimeManager.
#    - Saves the synthetic travel time data.
# 3. SRT Inversion (Long Profile):
#    - Uses PyGIMLi's TravelTimeManager to perform SRT inversion on the synthetic data.
#    - Creates an inversion mesh.
#    - Sets inversion parameters (regularization, smoothness constraints, velocity limits).
#    - Visualizes the inverted velocity model, highlighting specific velocity contours.
#    - Includes helper functions for hole filling in coverage maps and creating triangle data for plotting.
# 4. SRT Inversion (Short Profile):
#    - Loads previously generated synthetic travel time data (presumably from Ex2_workflow.py, representing a shorter profile).
#    - Performs SRT inversion on this data using a new mesh.
#    - Visualizes the inverted velocity model for the short profile.
#
# Assumptions:
# - Example data files (idomain, top, porosity, water content, bot) are available in the
#   specified `data_dir`. Paths are hardcoded and may need user adjustment.
# - The "short profile" data (`synthetic_seismic_data.dat`) is assumed to exist in the
#   output directory of `Ex2_workflow.py`.
#
# Expected output:
# - Console output: Progress messages (e.g., "Step 1: ...", "Step 9: ...").
# - Saved files:
#   - `synthetic_seismic_data_long.dat`: Synthetic travel time data for the long profile.
#   - Plots are shown interactively but not explicitly saved to files in this script,
#     though they could be with `plt.savefig()`.
# - Matplotlib plots:
#   - Geometry of the long profile with sensor locations.
#   - Synthetic first-arrival travel times for the long profile.
#   - Inverted P-wave velocity model for the long profile, with contours.
#   - Inverted P-wave velocity model for the short profile, with contours.

"""
Ex 5. Seismic Refraction Tomography (SRT) Forward Modeling
====================================================

This example demonstrates seismic refraction tomography forward modeling
for watershed structure characterization using PyHydroGeophysX.

The workflow includes:
1. Converting water content to seismic P-wave velocity using rock physics models
2. Creating seismic survey geometry along topographic profiles
3. Forward modeling to generate synthetic travel time data
4. Seismic tomography inversion to recover velocity structure
5. Visualization of velocity models and first-arrival picks

SRT is valuable for determining subsurface structure and bedrock interface
geometry, which provides important constraints for hydrogeophysical modeling
and interpretation of ERT data.
"""
# --- Standard library and third-party imports ---
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pygimli as pg # PyGIMLi for geophysical modeling, mesh, and SRT.
from pygimli.physics import ert # For ERT (not directly used but often complementary).
from pygimli.physics import TravelTimeManager # Core class for SRT.
import pygimli.physics.traveltime as tt # Travel time utilities.
from mpl_toolkits.axes_grid1 import make_axes_locatable # For colorbar adjustments.
import pygimli.meshtools as mt # For mesh creation and utilities.

# --- Setup package path for development ---
# Ensures the script can find PyHydroGeophysX when run directly.
try:
    # For regular Python scripts.
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # Fallback for interactive environments (e.g., Jupyter).
    current_dir = os.getcwd()

# Add the parent directory (project root) to Python path.
parent_dir = os.path.dirname(current_dir) # Navigate one level up.
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# --- Import PyHydroGeophysX specific modules ---
from PyHydroGeophysX.model_output.modflow_output import MODFLOWWaterContent # For MODFLOW data (not directly used here but good for context).
from PyHydroGeophysX.core.interpolation import ProfileInterpolator, create_surface_lines # For profile and geometry setup.
from PyHydroGeophysX.core.mesh_utils import MeshCreator # For mesh generation.
from PyHydroGeophysX.petrophysics.velocity_models import HertzMindlinModel, DEMModel # For rock physics conversions.

# %%
# --- Define Output Directory ---
# Specifies where results (e.g., saved synthetic data) from this example will be stored.
# IMPORTANT: This uses a hardcoded absolute path. Users should modify this.
# SUGGESTION: Use a relative path for portability:
# output_dir_srt = os.path.join(current_dir, "results", "Ex5_SRT_results")
output_dir = "C:/Users/HChen8/Documents/GitHub/PyHydroGeophysX/examples/results/seismic_example" # User-specific path.
os.makedirs(output_dir, exist_ok=True) # Create directory if it doesn't exist.

# %% [markdown]
# ## Long seismic profile
# <!-- This section details the setup, forward modeling, and inversion for a "long" seismic profile. -->

# %%
# --- Step 1: Initial Setup - Mesh and Hydrological Model Data ---
# This part largely replicates the setup from Ex2_workflow.py to get a geophysical mesh
# and interpolated hydrological properties (porosity, water content) on that mesh.
print("Step 1: Follow the workflow to create the mesh and model...")

# Define path to example data files.
# IMPORTANT: Hardcoded absolute path.
data_dir = "C:/Users/HChen8/Documents/GitHub/PyHydroGeophysX/examples/data/"
modflow_dir = os.path.join(data_dir, "modflow") # Not directly used if .npy files contain all needed data.

# Load domain information (idomain, surface topography, 3D porosity).
idomain = np.loadtxt(os.path.join(data_dir, "id.txt"))
top = np.loadtxt(os.path.join(data_dir, "top.txt"))
porosity = np.load(os.path.join(data_dir, "Porosity.npy"))

# Load time-series water content and select a specific timestep (index 50).
Water_Content = np.load(os.path.join(data_dir, "Watercontent.npy"))
water_content = Water_Content[50] # Using data from timestep 50.
print(f"Shape of selected water content data: {water_content.shape}")


# --- Step 3 (from Ex2 context): Set up profile for 2D section ---
# Define profile endpoints using grid indices from the hydrological model.
point1 = [115, 70]  # [col_idx, row_idx] for start point of the profile.
point2 = [95, 180]  # [col_idx, row_idx] for end point of the profile.

# Initialize ProfileInterpolator.
interpolator = ProfileInterpolator(
    point1=point1,
    point2=point2,
    surface_data=top, # Surface elevation data.
    origin_x=569156.2983333333, # Real-world X-coordinate of grid origin.
    origin_y=4842444.17,    # Real-world Y-coordinate of grid origin.
    pixel_width=1.0,        # Cell width in hydro model.
    pixel_height=-1.0,      # Cell height (negative if Y-axis origin is top).
    num_points = 400        # Number of points to discretize the profile.
)

# Interpolate 3D water content and porosity fields to the 2D profile.
water_content_profile = interpolator.interpolate_3d_data(water_content)
porosity_profile = interpolator.interpolate_3d_data(porosity)

# %%
# --- Step 2 (from Ex2 context, adapted for SRT): Creating Geometry and Mesh ---
print("Step 2: Creating geometry for the seismic refraction survey...")

# Load 3D data for subsurface layer boundaries.
bot = np.load(os.path.join(data_dir, "bot.npy"))
# Interpolate these layer boundaries to the 2D profile.
structure = interpolator.interpolate_layer_data([top] + bot.tolist())

# Define main geological layer interfaces from the interpolated structure.
top_idx=int(0)     # Index for surface.
mid_idx=int(4)     # Index for top of middle layer.
bot_idx=int(12)    # Index for top of bottom layer.
surface, line1, line2 = create_surface_lines(
    L_profile=interpolator.L_profile, # Distances along profile.
    structure=structure,              # Interpolated layer elevations.
    top_idx=top_idx,
    mid_idx=mid_idx,
    bot_idx=bot_idx
)

# Create the 2D geophysical mesh using MeshCreator.
mesh_creator = MeshCreator(quality=32) # Mesh quality parameter.
mesh, geom = mesh_creator.create_from_layers( # `geom` is the PLC object.
    surface=surface,
    layers=[line1, line2], # Internal layer boundaries.
    bottom_depth= np.min(line2[:,1])-10 # Mesh bottom 10 units below lowest point of `line2`.
)

# %%
# --- Visualize the Generated Mesh Geometry (PLC) ---
# `pg.show` can display the Planar Straight Line Complex (PLC) defining the mesh regions.
pg.show(geom) # Output: Matplotlib plot of the mesh geometry.
plt.title("Geophysical Mesh Geometry (PLC)")
plt.xlabel("Distance (m)")
plt.ylabel("Elevation (m)")
plt.show()

# %%
# --- Define Seismic Survey Geometry for the Long Profile ---
numberGeophones = 90 # Number of geophones in the survey line.
shotDistance = 5     # Spacing between shot points (in terms of geophone indices, or meters if sensors are 1m spaced).
                     # The `createRAData` function's `shotDistance` is often in terms of number of geophones.
                     # Here, `sensors` array below seems to define actual x-positions.

# Define nominal x-positions for geophones (e.g., from 1m to 110m).
sensors = np.linspace(1,110, numberGeophones) # Nominal x-coordinates of geophones.

# Create a seismic refraction data scheme (shot-receiver pairs).
# `shotDistance` here likely means shots are placed every `shotDistance` *meters* if `sensors` are actual positions,
# or every `shotDistance` *geophone index* if `sensors` are just indices.
# Given `sensors` are positions, this should define shots at specific x-locations.
# However, PyGIMLi's `createRAData` often takes sensor *indices* if not given a full DataContainer.
# If `sensors` are actual x-coordinates, this might not work as expected directly.
# A common way: scheme = DataContainer(); scheme.setSensors(sensor_positions_array); then define shots.
# Or, if `sensors` are indices 0 to N-1: createRAData(sensors_indices, shotDistance=shot_skip_count)
# The current usage is a bit ambiguous without knowing specific `createRAData` overload.
# Assuming `sensors` are x-positions, and `createRAData` can handle this to define a scheme.
# For clarity: it's better to first define sensor positions, then create scheme.
scheme = pg.physics.traveltime.createRAData(sensors,shotDistance=shotDistance)

# --- Drape Sensor Positions onto Surface Topography ---
# `pos` will store the actual (x, y) coordinates of geophones on the surface.
pos = np.zeros((numberGeophones,2)) # Array to hold (x, elevation) for each geophone.
               
# For each nominal geophone x-position, find the corresponding surface elevation from `surface_line`.
# `surface_line` contains [distance_along_profile, elevation_profile].
for i in range(numberGeophones):
    # Find the closest point on the `surface_line` (which represents the topography along the profile)
    # to the nominal sensor x-position `sensors[i]`.
    # `surface_line[:,0]` contains the x-coordinates (distances along profile) of the surface profile.
    # `surface_line[:,1]` contains the elevations at these x-coordinates.
    minusx = np.abs(surface[:,0]-sensors[i]) # Difference between sensor x and profile x points.
    index = np.where(minusx== np.amin(minusx)) # Find index of closest profile x point.
    # This finds the *nearest existing point* on the profile; interpolation is usually better.
    # SUGGESTION: Use np.interp(sensors[i], surface[:,0], surface[:,1]) for smoother draping,
    # provided `surface[:,0]` is monotonically increasing.
    new_x = surface[index,0] # X-coordinate on profile.
    new_y = surface[index,1] # Y-coordinate (elevation) on profile.
    pos[i, 0] = new_x
    pos[i, 1] = new_y

# Set the calculated (draped) sensor positions in the measurement scheme.
scheme.setSensors(pos)

# %%
# --- Visualize Survey Geometry on Mesh ---
# Plot the mesh geometry (`geom`) and overlay the draped sensor positions.
fig_geom_sensors = plt.figure(figsize=[8,6]) # Renamed for clarity
ax_geom_sensors = fig_geom_sensors.add_subplot(1,1,1) # Renamed

pg.show(geom, ax=ax_geom_sensors) # Show the PLC geometry.
# Draw sensor positions on the plot.
pg.viewer.mpl.drawSensors(ax_geom_sensors, scheme.sensors(), diam=0.5,
                         facecolor='black', edgecolor='black')
ax_geom_sensors.set_title("Mesh Geometry with Draped Seismic Sensors")
ax_geom_sensors.set_xlabel("Distance (m)")
ax_geom_sensors.set_ylabel("Elevation (m)")
plt.show() # Output: Matplotlib plot.

# %%
# --- Step 6 & 7 (from Ex2): Interpolate Data to Mesh and Calculate Saturation ---
print("Step 6: Interpolating data to mesh...")

# Create an ID array (`ID1`) to map hydrological model layers to geophysical mesh regions.
# This uses `porosity_profile` shape and layer indices (`mid_idx`, `bot_idx`) from earlier.
# Markers 0, 3, 2 are assigned to represent different geological units.
ID1 = porosity_profile.copy()
ID1[:mid_idx] = 0 # Regolith material ID.
ID1[mid_idx:bot_idx] = 3 # Fractured bedrock material ID.
ID1[bot_idx:] = 2 # Fresh bedrock material ID.

# Get cell centers and markers from the generated `mesh`.
mesh_centers = np.array(mesh.cellCenters())
mesh_markers = np.array(mesh.cellMarkers()) # Markers assigned by MeshCreator.

# Interpolate porosity from the 2D profile to the 2D geophysical mesh cells.
porosity_mesh = interpolator.interpolate_to_mesh(
    property_values=porosity_profile,
    depth_values=structure, # Using the full interpolated layer structure.
    mesh_x=mesh_centers[:, 0],
    mesh_y=mesh_centers[:, 1],
    mesh_markers=mesh_markers, # Target markers on the geophysical mesh.
    ID=ID1,                    # Source material IDs from hydro model layers on profile.
    layer_markers = [0,3,2]    # List of unique material IDs to process.
)

# Interpolate water content to the geophysical mesh cells.
wc_mesh = interpolator.interpolate_to_mesh(
    property_values=water_content_profile,
    depth_values=structure,
    mesh_x=mesh_centers[:, 0],
    mesh_y=mesh_centers[:, 1],
    mesh_markers=mesh_markers,
    ID=ID1,
    layer_markers = [0,3,2]
)

print("Step 7: Calculating saturation on the mesh...")
# Calculate saturation from water content and porosity on the mesh.
# Ensure porosity is not zero to avoid division errors; use a small floor value.
porosity_safe = np.maximum(porosity_mesh, 0.01) # Avoid porosity being too close to zero.
saturation = np.clip(wc_mesh / porosity_safe, 0.0, 1.0) # Clip saturation to physically realistic range [0,1].

# %%
# --- Step 9 (from Ex2): Convert Hydrological Properties to P-wave Velocity ---
print("Step9: Converting to P wave velocity ..")
# Define material markers corresponding to those in `ID1` and `mesh_markers`.
marker_labels = [0, 3, 2] # Top (regolith), mid (fractured), bottom (fresh bedrock).

# Initialize rock physics models (Hertz-Mindlin for unconsolidated, DEM for fractured/solid rock).
hm_model = HertzMindlinModel(critical_porosity=0.4, coordination_number=6.0)
dem_model = DEMModel()

# Initialize an array to store the P-wave velocity model on the mesh cells.
velocity_mesh = np.zeros_like(wc_mesh) # Same shape as water content and porosity on mesh.


# --- Calculate Vp for the Top Layer (Regolith) using Hertz-Mindlin ---
# Parameters specific to the top layer (marker_labels[0]).
top_mask = (mesh_markers == marker_labels[0]) # Boolean mask for cells in the top layer.
if np.any(top_mask): # Proceed if there are any cells with this marker.
    top_bulk_modulus = 30.0  # Bulk modulus of mineral matrix [GPa].
    top_shear_modulus = 20.0 # Shear modulus of mineral matrix [GPa].
    top_mineral_density = 2650.0 # Density of mineral matrix [kg/m³].
    top_depth = 1.0          # Effective depth for pressure calculation [m].

    # Calculate Vp using Hertz-Mindlin model. It returns high and low velocity bounds.
    Vp_high, Vp_low = hm_model.calculate_velocity(
        porosity=porosity_mesh[top_mask],
        saturation=saturation[top_mask],
        bulk_modulus=top_bulk_modulus,
        shear_modulus=top_shear_modulus,
        mineral_density=top_mineral_density,
        depth=top_depth
    )
    # Use the average of the high and low bounds as the representative Vp.
    velocity_mesh[top_mask] = (Vp_high + Vp_low) / 2.0
else:
    print(f"Warning: No cells found for top layer (marker {marker_labels[0]}) in Vp calculation.")


# --- Calculate Vp for the Middle Layer (Fractured Bedrock) using DEM Model ---
mid_mask = (mesh_markers == marker_labels[1]) # Boolean mask for middle layer cells.
if np.any(mid_mask):
    mid_bulk_modulus = 50.0  # GPa
    mid_shear_modulus = 35.0 # GPa
    mid_mineral_density = 2670.0  # kg/m³
    mid_aspect_ratio = 0.05   # Pore aspect ratio for DEM model (characterizes cracks/pores).

    # DEM model returns effective bulk (K_eff), shear (G_eff), and P-wave velocity (Vp).
    _, _, Vp_mid = dem_model.calculate_velocity( # Use Vp_mid to avoid conflict if Vp was defined before.
        porosity=porosity_mesh[mid_mask],
        saturation=saturation[mid_mask],
        bulk_modulus=mid_bulk_modulus,
        shear_modulus=mid_shear_modulus,
        mineral_density=mid_mineral_density,
        aspect_ratio=mid_aspect_ratio
    )
    velocity_mesh[mid_mask] = Vp_mid
else:
    print(f"Warning: No cells found for middle layer (marker {marker_labels[1]}) in Vp calculation.")

# --- Calculate Vp for the Bottom Layer (Fresh Bedrock) using DEM Model ---
bot_mask = (mesh_markers == marker_labels[2]) # Boolean mask for bottom layer cells.
if np.any(bot_mask):
    bot_bulk_modulus = 55.0  # GPa
    bot_shear_modulus = 50.0 # GPa
    bot_mineral_density = 2680.0  # kg/m³
    bot_aspect_ratio = 0.03   # Pore aspect ratio.

    _, _, Vp_bot = dem_model.calculate_velocity( # Use Vp_bot.
        porosity=porosity_mesh[bot_mask],
        saturation=saturation[bot_mask],
        bulk_modulus=bot_bulk_modulus,
        shear_modulus=bot_shear_modulus,
        mineral_density=bot_mineral_density,
        aspect_ratio=bot_aspect_ratio
    )
    velocity_mesh[bot_mask] = Vp_bot
else:
    print(f"Warning: No cells found for bottom layer (marker {marker_labels[2]}) in Vp calculation.")


# %%
# --- SRT Forward Modeling: Generate Synthetic Travel Time Data ---
# Initialize PyGIMLi's TravelTimeManager for seismic forward modeling.
mgr = TravelTimeManager()
# `mgr.simulate` calculates first-arrival travel times.
# Input:
#   - `slowness`: Slowness model (1 / P-wave velocity). Ensure `velocity_mesh` doesn't contain zeros.
#                 Add a small epsilon to `velocity_mesh` if necessary to prevent division by zero.
#   - `scheme`: The seismic measurement scheme (shot-receiver geometry) defined earlier.
#   - `mesh`: The geophysical mesh on which `velocity_mesh` is defined.
#   - `noiseLevel`, `noiseAbs`: Parameters to add synthetic noise to the travel times.
#   - `seed`: For reproducible noise generation.
#   - `verbose`: Print progress information.
slowness_model = 1.0 / np.maximum(velocity_mesh, 1e-6) # Avoid division by zero, ensure Vp is positive.
datasrt = mgr.simulate(slowness=slowness_model, scheme=scheme, mesh=mesh,
                    noiseLevel=0.05, noiseAbs=0.00001, seed=1334,
                    verbose=True)
# Save the generated synthetic travel time data to a file.
datasrt.save(os.path.join(output_dir, "synthetic_seismic_data_long.dat"))

# %%
# --- Define Helper Function to Plot First Arrival Picks ---
# This function visualizes seismic travel time data (shot gathers).
def drawFirstPicks(ax, data, tt=None, plotva=False, **kwargs):
    """Plot first arrivals as lines.
    
    Parameters
    ----------
    ax : matplotlib.axes
        Axis to draw the lines in.
    data : pygimli.DataContainer
        Data container with "s" (shots), "g" (geophones), and "t" (traveltimes).
    tt : array, optional
        Traveltimes to use instead of data("t").
    plotva : bool, optional
        If True, plot apparent velocity instead of traveltimes.
    
    Return
    ------
    ax : matplotlib.axes
        The modified axis.
    """
    # Extract sensor (x) coordinates from the data container.
    px = pg.x(data) # Unique x-positions of all sensors.
    # Map geophone and shot indices to their x-positions for each measurement.
    gx = np.array([px[int(g)] for g in data("g")]) # Geophone x-positions.
    sx = np.array([px[int(s)] for s in data("s")]) # Shot x-positions.
    
    # Get traveltimes to plot.
    travel_times_plot: np.ndarray
    if tt is None:
        travel_times_plot = np.array(data("t")) # Use 't' from data container.
    else:
        travel_times_plot = np.asarray(tt) # Use provided travel times.

    y_axis_plot_label = "Traveltime (s)" # Default y-axis label.
    if plotva: # If plotting apparent velocity.
        distance_offset = np.abs(gx - sx) # Distance between shot and geophone.
        travel_times_plot = distance_offset / (travel_times_plot + 1e-12) # V_app = dist/time. Add epsilon to avoid div by zero.
        y_axis_plot_label = "Apparent velocity (m s$^{-1}$)"
    
    # Find unique shot positions to plot each shot gather separately.
    unique_shot_positions = np.unique(sx)
    
    # Define default plotting style for the travel time curves.
    plot_kwargs = {'color': 'black', 'linestyle': '--', 'linewidth': 0.9, 'marker': None}
    plot_kwargs.update(kwargs) # Allow user to override default styles.
    
    # Plot data for each shot.
    for i_shot_loop, shot_pos_x_loop in enumerate(unique_shot_positions): # Renamed loop vars
        # Select data (times, geophone positions) for the current shot.
        mask_current_shot = (sx == shot_pos_x_loop)
        times_current_shot = travel_times_plot[mask_current_shot]
        geophones_current_shot = gx[mask_current_shot]
        
        # Sort by geophone position for a connected line plot.
        sort_indices_geophones = geophones_current_shot.argsort()
        
        # Plot the travel time curve (or apparent velocity curve).
        ax.plot(geophones_current_shot[sort_indices_geophones], times_current_shot[sort_indices_geophones], **plot_kwargs)

        # Add a marker for the shot position itself.
        # Position marker at y=0 or adjust based on y-axis limits for visibility.
        shot_marker_y_pos = 0.0 # Renamed
        current_ylim_plot = ax.get_ylim() # Renamed
        if not plotva and current_ylim_plot[0] > current_ylim_plot[1]: # If y-axis inverted (travel times are plotted positive down).
            shot_marker_y_pos = current_ylim_plot[1] # Place at visual top of plot.
        elif plotva and current_ylim_plot[0] < current_ylim_plot[1]: # If y-axis normal (velocities).
             shot_marker_y_pos = current_ylim_plot[0] # Place at visual bottom.

        ax.plot(shot_pos_x_loop, shot_marker_y_pos, marker='s', color='black', markersize=4,
                markeredgecolor='black', markeredgewidth=0.5, linestyle='None') # Shot marker.
    
    # Style the plot axes.
    ax.grid(True, linestyle='-', linewidth=0.2, color='lightgray') # Add a light grid.
    ax.set_ylabel(y_axis_plot_label)
    ax.set_xlabel("Distance (m)")
    
    # Invert y-axis for travel time plots (standard convention: smaller times at top).
    if not plotva:
        ax.invert_yaxis()

    return ax

# --- Example Usage of drawFirstPicks to Visualize Synthetic Data ---
# Create a figure and axes for the plot.
fig_picks_long, ax_picks_long = plt.subplots(figsize=(3.5, 2.5), dpi=300) # Renamed
# Call the function to draw the first arrival picks from `datasrt`.
drawFirstPicks(ax_picks_long, datasrt)
ax_picks_long.set_title("Synthetic Travel Times (Long Profile)")
plt.show() # Display plot. Output: Matplotlib plot of shot gathers.

# %%
# --- SRT Inversion for the Long Profile ---
# Initialize PyGIMLi's TravelTimeManager for inversion.
TT = pg.physics.traveltime.TravelTimeManager() # Instance for inversion.
# Create an inversion mesh based on the synthetic data (`datasrt`).
# `paraMaxCellSize`, `quality`, `paraDepth` control mesh generation.
mesh_inv = TT.createMesh(datasrt, paraMaxCellSize=2, quality=32, paraDepth = 50.0)
# Perform SRT inversion.
# `lam`: Regularization strength.
# `zWeight`: Vertical smoothness weight.
# `vTop`, `vBottom`: Constraints on velocity near top/bottom of model.
# `limits`: Min/max allowed velocities in the inverted model [m/s].
TT.invert(datasrt, mesh = mesh_inv,lam=50,
          zWeight=0.2,vTop=500, vBottom=5500,
          verbose=1, limits=[300., 8000.])
# The inverted velocity model is stored in `TT.model`.

# %%
# --- Post-Inversion: Coverage and Hole Filling (for Visualization) ---
# Calculate standardized coverage (ray density) of the inversion mesh.
cov = TT.standardizedCoverage()
# Get cell center positions of the inversion mesh.
pos = np.array(mesh_inv.cellCenters())


# %%
# --- Define Helper Function to Fill Holes in 2D Scattered Data (Coverage Map) ---
# This function is used to improve the visual appearance of coverage maps by filling
# small regions of zero coverage that are surrounded by regions with coverage.
import numpy as np # Already imported.
import matplotlib.pyplot as plt # Already imported.
from scipy import ndimage # For image processing tasks like hole filling.
from scipy.interpolate import griddata # For interpolating scattered data to a grid.

def fill_holes_2d(pos_coords, coverage_values, grid_resolution=100): # Renamed args
    """
    Fill holes (0 values) surrounded by 1 values in 2D scattered data representing coverage.
    This is primarily a cosmetic step for better visualization of coverage.
    
    Parameters:
    -----------
    pos_coords : ndarray of shape (n_points, 2) or (n_points, 3)
        Position array where first two columns are x,y coordinates of data points.
    coverage_values : ndarray of shape (n_points,)
        Coverage values (typically 0 or 1, or continuous values that will be thresholded) at each point.
    grid_resolution : int
        Resolution of the temporary regular grid used for interpolation and hole filling.
        
    Returns:
    --------
    filled_coverage_on_scattered : ndarray of shape (n_points,)
        Updated coverage values with holes filled, mapped back to original scattered point locations.
    """
    # Extract only the first two columns (x, y) from `pos_coords` for 2D operations.
    pos_2d = pos_coords[:, :2]
    
    # Determine min and max coordinates to define the boundaries of the temporary regular grid.
    min_coords = np.min(pos_2d, axis=0)
    max_coords = np.max(pos_2d, axis=0)
    
    # Create a regular 2D grid (X, Y coordinates).
    x_grid_nodes = np.linspace(min_coords[0], max_coords[0], grid_resolution) # Renamed
    y_grid_nodes = np.linspace(min_coords[1], max_coords[1], grid_resolution) # Renamed
    X_grid_mesh, Y_grid_mesh = np.meshgrid(x_grid_nodes, y_grid_nodes) # Renamed
    
    # Interpolate the scattered coverage data (`coverage_values`) onto the regular grid.
    # `griddata` with 'nearest' method assigns each grid node the value of the nearest scattered point.
    grid_points_for_interp = np.vstack([X_grid_mesh.ravel(), Y_grid_mesh.ravel()]).T # Renamed
    grid_cov_interpolated = griddata(pos_2d, coverage_values, grid_points_for_interp, method='nearest').reshape(X_grid_mesh.shape) # Renamed
    
    # Convert the interpolated grid coverage to a binary format (True for coverage > 0.5, False otherwise).
    # This assumes coverage values are roughly 0 or 1.
    binary_grid_coverage = (grid_cov_interpolated > 0.5) # Renamed
    
    # Fill holes in the binary grid using `scipy.ndimage.binary_fill_holes`.
    # This fills regions of False (no coverage) that are completely surrounded by True (coverage).
    filled_binary_grid = ndimage.binary_fill_holes(binary_grid_coverage) # Renamed
    
    # Convert the filled binary grid back to float type (0.0 and 1.0).
    filled_grid_float = filled_binary_grid.astype(float) # Renamed
    
    # Interpolate the filled grid values back to the original scattered point locations.
    # This assigns each original point the value from the nearest node in the filled regular grid.
    filled_coverage_on_scattered = griddata(grid_points_for_interp, filled_grid_float.ravel(), pos_2d, method='nearest') # Renamed
    
    return filled_coverage_on_scattered

# Example usage comment from original script (illustrative):
# # Assuming you have your data loaded as pos and cov
# # cov = np.array([0, 1, 0, ...])  # Your original coverage values

# Apply the hole filling function to the calculated coverage `cov`.
# `pos` contains cell centers of `mesh_inv`.
filled_cov = fill_holes_2d(pos, cov) # `pos` and `cov` are from the SRT inversion result.


# %%
# --- Define Helper Function to Create Triangle Data for Plotting ---
# This function extracts node coordinates and triangle connectivity from a PyGIMLi mesh,
# which is needed for some Matplotlib plotting functions like `tricontour`.
def createTriangles(mesh_obj): # Renamed arg
    """Generate triangle connectivity and node coordinates for mesh plotting.

    Creates triangle definitions for each 2D triangle cell. If the mesh contains quads,
    they are split into two triangles. For 3D meshes, it extracts boundary faces.
    The result is cached into `mesh_obj._triData` to speed up repeated calls.

    Parameters
    ----------
    mesh_obj : pygimli.Mesh
        A 2D or 3D PyGIMLi mesh object.

    Returns
    -------
    x_nodes : numpy.ndarray
        X-coordinates of all nodes in the mesh.
    y_nodes : numpy.ndarray
        Y-coordinates of all nodes in the mesh.
    triangles_connectivity : numpy.ndarray (shape: N_triangles, 3)
        Node indices forming each triangle.
    z_nodes : numpy.ndarray
        Z-coordinates of all nodes (0 for 2D meshes).
    data_indices_map : list of int
        List mapping triangle indices back to original cell/boundary IDs.
    """
    # Check if triangle data is already cached for this mesh to avoid re-computation.
    if hasattr(mesh_obj, '_triData'):
        if hash(mesh_obj) == mesh_obj._triData[0]: # Compare hash to see if mesh is unchanged.
            return mesh_obj._triData[1:] # Return cached data.

    # Get node coordinates from the mesh.
    x_nodes = pg.x(mesh_obj)
    y_nodes = pg.y(mesh_obj)
    z_nodes = pg.z(mesh_obj) # Typically zero for 2D meshes.

    # Determine entities to triangulate (cells for 2D, specific boundaries for 3D).
    if mesh_obj.dim() == 2:
        ents_to_triangulate = mesh_obj.cells() # Use all cells for a 2D mesh.
    else: # For 3D meshes, extract boundary faces that are actual model boundaries.
        ents_to_triangulate = mesh_obj.boundaries(mesh_obj.boundaryMarkers() != 0)
        if len(ents_to_triangulate) == 0: # Fallback: get all exterior-facing boundaries.
            for b_face in mesh_obj.boundaries(): # Renamed loop var
                if b_face.leftCell() is None or b_face.rightCell() is None:
                    ents_to_triangulate.append(b_face)

    triangles_connectivity = [] # List to store node indices for each triangle.
    data_indices_map = []     # List to map triangles back to cell/boundary IDs.

    # Iterate over selected entities (cells or boundary faces).
    for current_ent in ents_to_triangulate: # Renamed loop var
        # For triangles, directly use the first three nodes.
        triangles_connectivity.append([current_ent.node(0).id(), current_ent.node(1).id(), current_ent.node(2).id()])
        data_indices_map.append(current_ent.id())

        # If the entity is a quadrilateral, split it into two triangles.
        if current_ent.shape().nodeCount() == 4:
            triangles_connectivity.append([current_ent.node(0).id(), current_ent.node(2).id(), current_ent.node(3).id()])
            data_indices_map.append(current_ent.id()) # Both triangles map to the same quad cell ID.

    # Cache the computed data in the mesh object for future use.
    mesh_obj._triData = [hash(mesh_obj), x_nodes, y_nodes, triangles_connectivity, z_nodes, data_indices_map]

    return x_nodes, y_nodes, triangles_connectivity, z_nodes, data_indices_map

# %%
# --- Prepare Data for Contour Plotting on Inversion Mesh (Long Profile) ---
# Extract triangle connectivity and node coordinates from the inversion mesh `mesh_inv`.
x_nodes_long, y_nodes_long, triangles_long, _, _ = createTriangles(mesh_inv) # Renamed vars
# Convert cell-based inverted velocity model (`TT.model`) to node-based data for smoother contours.
# `TT.model` is the result from `TT.invert()` call.
z_node_velocities_long = pg.meshtools.cellDataToNodeData(mesh_inv, TT.model.array()) # Renamed

# %%
# --- Plot Inverted Velocity Model with Contours (Long Profile) ---
# Set Matplotlib styling parameters for the plot.
params_long_srt_plot = {'legend.fontsize': 15, # Renamed
                        'axes.labelsize': 15,
                        'axes.titlesize':16,
                        'xtick.labelsize':15,
                        'ytick.labelsize':15}
pylab.rcParams.update(params_long_srt_plot)
plt.rcParams["font.family"] = "Arial"

# Import and set a colormap.
from palettable.lightbartlein.diverging import BlueDarkRed18_18
fixed_cmap_long_srt = BlueDarkRed18_18.mpl_colormap # Renamed

# Create the figure and axes.
fig_long_srt_inv, ax_long_srt_inv = plt.figure(figsize=[8,9]), None # Renamed, ax created by pg.show
ax_long_srt_inv = fig_long_srt_inv.add_subplot(1,1,1) # Create axes

# Display the inverted velocity model on `mesh_inv`.
# `coverage=filled_cov` uses the hole-filled coverage to mask poorly resolved areas.
pg.show(mesh_inv, TT.model.array(), cMap=fixed_cmap_long_srt, coverage=filled_cov, ax=ax_long_srt_inv,
        label='Velocity (m s$^{-1}$)', xlabel="Distance (m)", ylabel="Elevation (m)",
        pad=0.3, cMin=500, cMax=5000, orientation="vertical")

# Overlay velocity contours on the plot.
# `levels` specifies the velocity values for which to draw contour lines.
ax_long_srt_inv.tricontour(x_nodes_long, y_nodes_long, triangles_long, z_node_velocities_long,
                           levels=[1200], linewidths=1.0, colors='k', linestyles='dashed') # E.g., bedrock interface.
ax_long_srt_inv.tricontour(x_nodes_long, y_nodes_long, triangles_long, z_node_velocities_long,
                           levels=[4300], linewidths=1.0, colors='k', linestyles='-')   # E.g., deeper interface.

# Draw sensor positions on the plot.
pg.viewer.mpl.drawSensors(ax_long_srt_inv, datasrt.sensors(), diam=0.9,
                         facecolor='black', edgecolor='black')
ax_long_srt_inv.set_title("Inverted Velocity Model (Long Profile)")
plt.show() # Output: Matplotlib plot.


# %% [markdown]
# ## Short seismic profiles
# <!-- This section handles a "short" seismic profile, likely loading pre-existing synthetic data -->
# <!-- (possibly generated by Ex2_workflow.py) and performing SRT inversion on it. -->

# %%
# --- Load Synthetic Travel Time Data for a Short Profile ---
# This data is assumed to be generated from a different (shorter) survey geometry
# or represents a different part of the overall study area.
# IMPORTANT: Path is hardcoded. This file is expected to be an output of Ex2_workflow.py.
short_profile_data_path = "C:/Users/HChen8/Documents/GitHub/PyHydroGeophysX/examples/results/workflow_example/synthetic_seismic_data.dat"
ttData_short = tt.load(short_profile_data_path) # Renamed variable

# --- SRT Inversion for the Short Profile ---
TT_short = pg.physics.traveltime.TravelTimeManager() # New TravelTimeManager instance for this inversion.
# Create an inversion mesh specifically for this short profile data.
# `paraMaxCellSize`, `quality`, `paraDepth` control mesh generation.
mesh_inv_short = TT_short.createMesh(ttData_short, paraMaxCellSize=2, quality=32, paraDepth=30.0) # Renamed
# Perform SRT inversion. Parameters are similar to the long profile inversion.
# Note: The original script uses `mesh=mesh_inv` (the long profile mesh) here, which seems like a mistake
# if `mesh_inv_short` was intended for this short profile data.
# For commenting, I will assume the intention was to use `mesh_inv_short`.
# If `mesh=mesh_inv` was indeed intentional, it implies inverting short profile data on a potentially ill-suited (long) mesh.
# Using `mesh_inv_short` as likely intended:
TT_short.invert(ttData_short, mesh=mesh_inv_short, lam=50, # Corrected to use mesh_inv_short
                zWeight=0.2, vTop=500, vBottom=5500,
                verbose=1, limits=[300., 8000.])
# Inverted model is in `TT_short.model`.

# %%
# --- Prepare Data for Contour Plotting on Inversion Mesh (Short Profile) ---
# Extract triangle connectivity and node coordinates from the short profile inversion mesh `mesh_inv_short`.
x_nodes_short, y_nodes_short, triangles_short, _, _ = createTriangles(mesh_inv_short) # Renamed variables
# Convert cell-based inverted velocity model to node-based data for `mesh_inv_short`.
z_node_velocities_short = pg.meshtools.cellDataToNodeData(mesh_inv_short, np.array(TT_short.model)) # Renamed, ensured model is array
# Get cell centers and calculate hole-filled coverage for `mesh_inv_short`.
pos_short = np.array(mesh_inv_short.cellCenters()) # Renamed
coverage_short = TT_short.standardizedCoverage() # Renamed
filled_cov_short = fill_holes_2d(pos_short, coverage_short) # Renamed

# %%
# --- Plot Inverted Velocity Model with Contours (Short Profile) ---
# Set Matplotlib styling parameters.
params_short_srt_plot = {'legend.fontsize': 15, # Renamed
                         'axes.labelsize': 15,
                         'axes.titlesize':16,
                         'xtick.labelsize':15,
                         'ytick.labelsize':15}
pylab.rcParams.update(params_short_srt_plot)
plt.rcParams["font.family"] = "Arial"

# Import and set colormap.
from palettable.lightbartlein.diverging import BlueDarkRed18_18 # Already imported but good for clarity.
fixed_cmap_short_srt = BlueDarkRed18_18.mpl_colormap # Renamed

# Create figure and axes.
fig_short_srt_inv, ax_short_srt_inv = plt.figure(figsize=[8,9]), None # Renamed
ax_short_srt_inv = fig_short_srt_inv.add_subplot(1,1,1) # Create axes

# Display the inverted velocity model for the short profile on `mesh_inv_short`.
pg.show(mesh_inv_short, TT_short.model.array(), cMap=fixed_cmap_short_srt, coverage=filled_cov_short, # Used filled_cov_short
        ax=ax_short_srt_inv, label='Velocity (m s$^{-1}$)',
        xlabel="Distance (m)", ylabel="Elevation (m)", pad=0.3, cMin=500, cMax=5000,
        orientation="vertical")

# Overlay velocity contours.
ax_short_srt_inv.tricontour(x_nodes_short, y_nodes_short, triangles_short, z_node_velocities_short,
                            levels=[1200], linewidths=1.0, colors='k', linestyles='dashed')

# Draw sensor positions from `ttData_short`.
pg.viewer.mpl.drawSensors(ax_short_srt_inv, ttData_short.sensors(), diam=0.8,
                         facecolor='black', edgecolor='black')
ax_short_srt_inv.set_title("Inverted Velocity Model (Short Profile)")
plt.show() # Output: Matplotlib plot.
# End of Example 5.