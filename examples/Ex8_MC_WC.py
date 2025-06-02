# Script-level comment:
# This script, Ex8_MC_WC.py, focuses on quantifying the uncertainty in water content
# estimates derived from time-lapse Electrical Resistivity Tomography (ERT) data.
# It employs a Monte Carlo (MC) simulation approach, where petrophysical parameters
# are sampled from defined probability distributions to translate inverted resistivity
# models into a probabilistic distribution of water content.
#
# Workflow steps covered:
# 1. Data Loading:
#    - Loads time-lapse inverted resistivity models (e.g., from Ex7 or similar).
#    - Loads associated coverage data and cell markers (identifying geological layers).
#    - Loads the geophysical mesh used for the ERT inversion.
# 2. Monte Carlo Setup:
#    - Defines the number of MC realizations (iterations).
#    - Specifies probability distributions (mean and standard deviation) for key
#      petrophysical parameters for different geological layers/units. These include:
#        - Saturated resistivity (`rhos`).
#        - Saturation exponent (`n`).
#        - Surface conductivity (`sigma_sur`).
#        - Porosity.
#    - Initializes arrays to store the results of all MC realizations (water content, saturation)
#      and the parameters sampled in each realization.
# 3. Monte Carlo Simulation Loop:
#    - For each realization:
#        - Samples petrophysical parameters (`rhos`, `n`, `sigma_sur`, `porosity`) for each layer
#          from their defined normal distributions. Values are clipped to ensure physical plausibility
#          (e.g., porosity > 0.05, rhos > 1.0).
#        - Assigns the sampled porosity to mesh cells based on their geological marker.
#        - For each timestep in the input resistivity data:
#            - Converts the resistivity values to saturation using the `resistivity_to_saturation`
#              function and the sampled layer-specific parameters (`rhos`, `n`, `sigma_sur`).
#            - Calculates water content as saturation multiplied by the sampled porosity for that cell/layer.
#        - Stores the resulting 3D (n_cells, n_timesteps) water content and saturation arrays for this realization.
# 4. Statistical Analysis of MC Results:
#    - Calculates statistics (mean, standard deviation, P10, P50/median, P90 percentiles)
#      of the water content across all MC realizations for each cell and timestep.
# 5. Visualization:
#    - Plots the mean water content for all timesteps on the mesh.
#    - Compares estimated water content with "true" water content (loaded from synthetic model outputs)
#      at specific locations by extracting and plotting time-series with uncertainty bands
#      (mean +/- standard deviation).
#    - Helper functions `extract_mc_time_series` and `extract_true_values_at_positions` are defined
#      for this time-series extraction.
#
# Assumptions:
# - Results from a previous structure-constrained time-lapse ERT inversion are available:
#   - `resmodel.npy`: Time-series of inverted resistivity models.
#   - `all_coverage.npy`: Corresponding coverage/sensitivity data.
#   - `index_marker.npy`: Cell markers identifying geological layers on the mesh.
#   - `mesh_res.bms`: The inversion mesh.
# - "True" water content models (`synwcmodel[i].npy`) and the mesh they are defined on (`mesh.bms` from
#   Ex3 results) are available for comparison purposes.
# - Paths to these input files are hardcoded and may need adjustment.
#
# Expected output:
# - Console output: (Potentially tqdm progress bar if uncommented).
# - Matplotlib plots:
#   - A figure showing the grid of 12 mean water content models over time.
#   - A plot showing a single timestep mean water content with specific points marked.
#   - Time-series plots of estimated water content (with uncertainty bands) versus true water content
#     at selected locations in different geological layers.
# - Saved files: This script primarily focuses on analysis and plotting; explicit saving of
#   MC results arrays (like `water_content_all`) is not shown but could be added.

"""
Ex 8. Monte Carlo Uncertainty Quantification for Water Content Estimation
====================================================================

This example demonstrates Monte Carlo uncertainty quantification for 
converting ERT resistivity models to water content estimates.

The analysis includes:
1. Loading inverted resistivity models from time-lapse ERT
2. Defining parameter distributions for different geological layers
3. Monte Carlo sampling of petrophysical parameters
4. Probabilistic water content estimation with uncertainty bounds
5. Statistical analysis and visualization of results
6. Time series extraction with confidence intervals

Uncertainty quantification is essential for reliable hydrological 
interpretation of geophysical data, providing confidence bounds on
water content estimates and identifying regions of high/low certainty.
"""
# --- Standard library and third-party imports ---
import numpy as np
import matplotlib.pyplot as plt
import os
import pygimli as pg # PyGIMLi for mesh handling and visualization.
import sys
from tqdm import tqdm # For displaying progress bars during loops (optional).

# --- Setup package path for development ---
# Ensures the script can find PyHydroGeophysX when run directly.
try:
    # For regular Python scripts.
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # Fallback for interactive environments (e.g., Jupyter).
    current_dir = os.getcwd()

# Add the parent directory (project root) to Python path.
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# --- Import PyHydroGeophysX modules ---
# `resistivity_to_saturation` is a key petrophysical conversion function.
from PyHydroGeophysX.petrophysics.resistivity_models import resistivity_to_saturation

# --- Load Input Data from Previous Inversion Results ---
# These files are assumed to be outputs from a preceding script (e.g., Ex7_structure_TLresinv.py).
# IMPORTANT: Paths are hardcoded and will likely need to be changed by the user.
# SUGGESTION: Use relative paths or configuration files for managing data paths.
base_results_path = "C:/Users/HChen8/Documents/GitHub/PyHydroGeophysX/examples/results/Structure_WC"

# Load the time-series of inverted resistivity models.
# Expected shape: (n_cells, n_timesteps) or (n_realizations_if_any, n_cells, n_timesteps).
# Here, it's (n_cells, n_timesteps) from the windowed inversion.
resistivity_values = np.load(os.path.join(base_results_path, "resmodel.npy"))
# Load coverage data (ray density or sensitivity) associated with the resistivity models.
coverage = np.load(os.path.join(base_results_path, "all_coverage.npy"))
# Load cell markers that identify different geological layers or regions within the mesh.
# This is crucial for applying layer-specific petrophysical parameters.
# `index_marker.npy` is assumed to be a 1D array of markers, one for each cell.
cell_markers = np.load(os.path.join(base_results_path, "index_marker.npy"))
# Load the geophysical mesh used for the ERT inversion.
mesh = pg.load(os.path.join(base_results_path, "mesh_res.bms"))

# --- Monte Carlo Simulation Setup ---
# Define the number of Monte Carlo realizations (iterations) to perform.
# More realizations provide better statistics but increase computation time.
n_realizations = 100 # Number of samples to draw for each parameter.

# Define probability distributions (mean and standard deviation) for petrophysical parameters.
# These distributions represent the uncertainty in these parameters for each geological layer.
# Parameters are for a model like Waxman-Smits or Archie's Law.
# Layer 1 (assumed to correspond to cells with marker 3, e.g., a top layer/regolith).
layer1_dist = {
    'rhos': {'mean': 100.0, 'std': 20.0},       # Saturated bulk resistivity (rho_sat) [Ohm-m].
    'n': {'mean': 2.2, 'std': 0.2},             # Saturation exponent (Archie's n).
    'sigma_sur': {'mean': 1/500, 'std': 1/2000},# Surface conductivity (sigma_s) [S/m].
    'porosity': {'mean': 0.40, 'std': 0.05}     # Porosity (phi) [-].
}
# Layer 2 (assumed to correspond to cells with marker 2, e.g., a bottom layer/bedrock).
layer2_dist = {
    'rhos': {'mean': 500.0, 'std': 100.0},
    'n': {'mean': 1.8, 'std': 0.2},
    'sigma_sur': {'mean': 0.0, 'std': 1/50000}, # Very low mean surface conductivity for bedrock.
    'porosity': {'mean': 0.35, 'std': 0.1}
}
# Note: Ensure these marker associations (3 for layer1, 2 for layer2) are consistent
# with how `cell_markers` were defined and used in previous scripts (e.g., Ex6, Ex7).

# --- Initialize Arrays to Store Monte Carlo Results ---
# `water_content_all`: Stores water content for each realization, cell, and timestep.
# Shape: (n_realizations, n_cells, n_timesteps).
water_content_all = np.zeros((n_realizations, *resistivity_values.shape))
# `Saturation_all`: Stores saturation for each realization, cell, and timestep.
Saturation_all = np.zeros((n_realizations, *resistivity_values.shape))

# `params_used`: Dictionary to store the actual parameter values sampled in each realization.
# This is useful for analyzing parameter sensitivity or correlations later.
params_used = {
    'layer1': {'rhos': np.zeros(n_realizations), 'n': np.zeros(n_realizations),
               'sigma_sur': np.zeros(n_realizations), 'porosity': np.zeros(n_realizations)},
    'layer2': {'rhos': np.zeros(n_realizations), 'n': np.zeros(n_realizations),
               'sigma_sur': np.zeros(n_realizations), 'porosity': np.zeros(n_realizations)}
}

# --- Perform Monte Carlo Simulation Loop ---
# `tqdm` creates a progress bar for the loop if installed.
for mc_idx in tqdm(range(n_realizations), desc="Monte Carlo Realizations"): # Added tqdm for progress
    # --- Sample Petrophysical Parameters for Current Realization ---
    # For each layer, sample `rhos`, `n`, and `sigma_sur` from their defined normal distributions.
    # `max(value, lower_bound)` is used to clip sampled values at physically plausible minimums (e.g., rhos > 1.0).
    layer1_params_sampled = { # Renamed for clarity
        'rhos': max(1.0, np.random.normal(layer1_dist['rhos']['mean'], layer1_dist['rhos']['std'])),
        'n': max(1.0, np.random.normal(layer1_dist['n']['mean'], layer1_dist['n']['std'])),
        'sigma_sur': max(0.0, np.random.normal(layer1_dist['sigma_sur']['mean'], layer1_dist['sigma_sur']['std']))
    }
    layer2_params_sampled = { # Renamed
        'rhos': max(1.0, np.random.normal(layer2_dist['rhos']['mean'], layer2_dist['rhos']['std'])),
        'n': max(1.0, np.random.normal(layer2_dist['n']['mean'], layer2_dist['n']['std'])),
        'sigma_sur': max(0.0, np.random.normal(layer2_dist['sigma_sur']['mean'], layer2_dist['sigma_sur']['std']))
    }
    
    # Sample porosity for each layer and assign to mesh cells based on `cell_markers`.
    # Porosity is also clipped to a physical range (e.g., 0.05 to 0.6).
    porosity_sampled_mc = np.zeros_like(cell_markers, dtype=float) # Renamed
    layer1_porosity_sampled = np.clip(np.random.normal(layer1_dist['porosity']['mean'],
                                                       layer1_dist['porosity']['std']), 0.05, 0.6) # Renamed
    layer2_porosity_sampled = np.clip(np.random.normal(layer2_dist['porosity']['mean'],
                                                       layer2_dist['porosity']['std']), 0.05, 0.6) # Renamed
    
    porosity_sampled_mc[cell_markers == 3] = layer1_porosity_sampled  # Assign to Layer 1 (marker 3) cells.
    porosity_sampled_mc[cell_markers == 2] = layer2_porosity_sampled  # Assign to Layer 2 (marker 2) cells.
    
    # Store the sampled parameters for this realization (`mc_idx`).
    params_used['layer1']['rhos'][mc_idx] = layer1_params_sampled['rhos']
    params_used['layer1']['n'][mc_idx] = layer1_params_sampled['n']
    params_used['layer1']['sigma_sur'][mc_idx] = layer1_params_sampled['sigma_sur']
    params_used['layer1']['porosity'][mc_idx] = layer1_porosity_sampled
    params_used['layer2']['rhos'][mc_idx] = layer2_params_sampled['rhos']
    params_used['layer2']['n'][mc_idx] = layer2_params_sampled['n']
    params_used['layer2']['sigma_sur'][mc_idx] = layer2_params_sampled['sigma_sur']
    params_used['layer2']['porosity'][mc_idx] = layer2_porosity_sampled
    
    # Initialize arrays for water content and saturation for the current realization.
    # These will store results for all cells and all timesteps for this `mc_idx`.
    water_content_realization = np.zeros_like(resistivity_values) # Renamed
    saturation_realization = np.zeros_like(resistivity_values) # Renamed
    
    # --- Process Each Timestep within the Current Realization ---
    for t in range(resistivity_values.shape[1]): # Iterate over number of timesteps.
        # Extract the inverted resistivity model for the current timestep `t`.
        resistivity_t_step = resistivity_values[:, t] # Renamed
        
        # --- Convert Resistivity to Saturation for Each Layer ---
        # Layer 1 (marker 3)
        mask_layer1 = (cell_markers == 3) # Boolean mask for cells belonging to Layer 1.
        if np.any(mask_layer1): # Check if any cells belong to this layer.
            saturation_realization[mask_layer1, t] = resistivity_to_saturation(
                resistivity_t_step[mask_layer1],    # Resistivity values for Layer 1 cells.
                layer1_params_sampled['rhos'],      # Sampled rhos for Layer 1.
                layer1_params_sampled['n'],         # Sampled n for Layer 1.
                layer1_params_sampled['sigma_sur']  # Sampled sigma_sur for Layer 1.
            )
        
        # Layer 2 (marker 2)
        mask_layer2 = (cell_markers == 2) # Boolean mask for cells belonging to Layer 2.
        if np.any(mask_layer2):
            saturation_realization[mask_layer2, t] = resistivity_to_saturation(
                resistivity_t_step[mask_layer2],    # Resistivity values for Layer 2 cells.
                layer2_params_sampled['rhos'],      # Sampled rhos for Layer 2.
                layer2_params_sampled['n'],         # Sampled n for Layer 2.
                layer2_params_sampled['sigma_sur']  # Sampled sigma_sur for Layer 2.
            )
        
        # Convert saturation to water content: water_content = saturation * porosity.
        # `porosity_sampled_mc` is the array of sampled porosities for all cells for this realization.
        water_content_realization[:, t] = saturation_realization[:, t] * porosity_sampled_mc
    
    # Store the water content and saturation results for the current realization `mc_idx`.
    water_content_all[mc_idx] = water_content_realization
    Saturation_all[mc_idx] = saturation_realization # Storing all saturation realizations.

# %%
# --- Calculate Statistical Summary of Monte Carlo Water Content Results ---
# After all realizations are complete, calculate statistics across the realizations dimension (axis=0).
water_content_mean = np.mean(water_content_all, axis=0)  # Mean water content for each cell and timestep.
water_content_std = np.std(water_content_all, axis=0)    # Standard deviation.
water_content_p10 = np.percentile(water_content_all, 10, axis=0)  # 10th percentile.
water_content_p50 = np.percentile(water_content_all, 50, axis=0)  # 50th percentile (Median).
water_content_p90 = np.percentile(water_content_all, 90, axis=0)  # 90th percentile.
# These arrays (e.g., `water_content_mean`) will have shape (n_cells, n_timesteps).

# %%
# --- Visualize Mean Water Content Over Time ---
# This section plots the mean water content model for each of the 12 timesteps.
from palettable.lightbartlein.diverging import BlueDarkRed18_18_r # Import a colormap.
import matplotlib.pyplot as plt # Already imported.
import numpy as np # Already imported.
import matplotlib.pylab as pylab # Already imported.

# Set Matplotlib styling parameters for the plot.
params_mc_plot = {'legend.fontsize': 13, # Renamed params dict
                  'axes.labelsize': 13,
                  'axes.titlesize':13,
                  'xtick.labelsize':13,
                  'ytick.labelsize':13}
pylab.rcParams.update(params_mc_plot)
plt.rcParams["font.family"] = "Arial"

fixed_cmap_wc_mean = BlueDarkRed18_18_r.mpl_colormap # Colormap for mean water content. Renamed.
fig_wc_mean, axes_wc_mean = plt.figure(figsize=[16, 6]), [] # Create figure. Renamed fig and axes.

# Adjust subplot spacing.
plt.subplots_adjust(wspace=0.05, hspace=0.05)

# Loop through each timestep (assuming 12 timesteps from `resistivity_values.shape[1]`).
for i in range(12): # Or use `water_content_mean.shape[1]`
    row, col = i // 4, i % 4 # Calculate subplot row and column.
    ax = fig_wc_mean.add_subplot(3, 4, i + 1) # Add subplot.
    axes_wc_mean.append(ax) # Store axis.
    
    # Determine if Y-axis label should be shown.
    ylabel_wc_text = "Elevation (m)" if col == 0 else None # Renamed.
    
    # Determine if colorbar label should be shown (only for one subplot).
    wc_cbar_label = 'Water Content (-)' if (i == 7) else None # Renamed.
    
    # Control tick label visibility for cleaner grid.
    if col != 0:
        ax.set_yticklabels([]) # Hide Y-tick labels if not first column.
    if row != 2:
        ax.set_xticklabels([]) # Hide X-tick labels if not bottom row.
    else:
        ax.set_xlabel("Distance (m)") # Show X-label for bottom row.
    
    # Plot the mean water content for the current timestep `i`.
    # `mesh` is the loaded geophysical mesh.
    # `coverage[i,:]` provides the coverage for this timestep to mask poorly resolved areas.
    # Assumes `coverage` array is (n_timesteps, n_cells).
    model_to_plot_wc = water_content_mean[:, i] # Renamed
    coverage_to_plot_wc = coverage[i,:] > -1.2 if coverage is not None and i < coverage.shape[0] else None # Renamed

    _, cbar = pg.show(mesh, model_to_plot_wc, pad=0.3, orientation="vertical",
                      cMap=fixed_cmap_wc_mean, cMin=0, cMax=0.32, # Set color scale for water content.
                      ylabel=ylabel_wc_text, label=wc_cbar_label, ax=ax, logScale=False,
                      coverage=coverage_to_plot_wc)
    
    # Remove individual colorbars if label is not set (i.e., for all but i=7).
    if i != 7:
        if cbar: cbar.remove()
    ax.set_title(f"Mean WC - Day {(i+1)*30}") # Example title assuming 30-day intervals.

fig_wc_mean.tight_layout() # Adjust layout.
plt.suptitle("Mean Water Content from Monte Carlo Simulation", fontsize=16, y=1.02) # Overall title.
plt.show() # Display plot. Output: Figure with 12 subplots of mean water content.


# %%
# --- Load True Water Content Models for Comparison ---
# This section loads the "true" water content models that were used to generate
# the synthetic ERT data in the first place (e.g., from Ex3). This allows for
# direct comparison of MC-estimated water content against the ground truth.
WC_true_list = [] # Renamed list

# Loop through timesteps (assuming 30-day intervals, matching `ert_files` naming).
for i in np.arange(30,361,30): # Corresponds to days 30, 60, ..., 360.
    # IMPORTANT: Hardcoded path to true water content models.
    # These files (`synwcmodel[day].npy`) are expected outputs from Ex3.
    true_wc_filepath = f"C:/Users/HChen8/Documents/GitHub/PyHydroGeophysX/examples/results/TL_measurements/synwcmodel/synwcmodel{i}.npy" # Used f-string
    true_values_timestep = np.load(true_wc_filepath) # Load true WC for current timestep. Renamed.
    WC_true_list.append(true_values_timestep)

# Load the mesh on which the true water content models are defined.
# This might be different from `mesh` (the structurally constrained inversion mesh)
# if the true models were generated on a simpler/different mesh.
# IMPORTANT: Hardcoded path.
mesh_true_wc_path = "C:/Users/HChen8/Documents/GitHub/PyHydroGeophysX/examples/results/TL_measurements/mesh.bms" # Renamed
mesh_true = pg.load(mesh_true_wc_path) # Mesh for true WC models.

# Convert the list of true water content arrays into a single 3D NumPy array.
# Shape: (n_timesteps, n_cells_in_true_mesh).
WC_true_array = np.array(WC_true_list) # Renamed
print(f"Shape of true water content array: {WC_true_array.shape}") # (n_timesteps, n_cells_true_mesh)

# %%
# --- Visualize a Single Timestep of Mean Estimated Water Content ---
# This plot focuses on one timestep (index 6, corresponding to Day 210 or the 7th month)
# of the mean water content estimated from MC, and marks specific points of interest.
fig_single_wc, ax_single_wc = plt.subplots(1, 1, figsize=[6, 3]) # Renamed

# `water_content_mean[:, 6]` is the mean WC for the 7th timestep (0-indexed).
# `coverage[6,:]` is the coverage for this timestep.
# `ylabel` and `label` are used from the previous loop's context, which might be unintended.
# For clarity, they should be redefined or passed explicitly if needed.
pg.show(mesh, water_content_mean[:, 6], pad=0.3, orientation="vertical",
        cMap=fixed_cmap_wc_mean, cMin=0, cMax=0.32,
        ylabel="Elevation (m)", label='Water Content (-)', # Explicitly set labels
        ax=ax_single_wc, logScale=False,
        coverage=(coverage[6,:] > -1.2 if coverage is not None and coverage.shape[0] > 6 else None))
ax_single_wc.set_title("Mean Water Content (Day 210) with Sample Points")

# Mark specific (x,y) locations on the plot for time-series extraction.
ax_single_wc.plot([40],[1607],'*',c='k', markersize=8, label="Point A (Regolith)") # Example point A
ax_single_wc.plot([80],[1621],'*',c='k', markersize=8, label="Point B (Regolith)") # Example point B
ax_single_wc.plot([30],[1604],'*',c='k', markersize=8, label="Point C (Fractured)") # Example point C
ax_single_wc.plot([65],[1608],'*',c='k', markersize=8, label="Point D (Fractured)") # Example point D
# plt.legend() # Optional: add legend for points if desired.
plt.show()

# %%
# --- Helper Function to Extract Monte Carlo Time Series at Specific Locations ---
# This function finds the mesh cells closest to given (x,y) positions and extracts
# the time series of a variable (e.g., water content) for all MC realizations at these cells.
def extract_mc_time_series(mesh_obj, mc_values_all, target_positions): # Renamed args
    """
    Extract Monte Carlo time series at specific (x,y) positions from mesh cell data.
    
    Args:
        mesh_obj : pygimli.Mesh
            The PyGIMLi mesh on which `mc_values_all` are defined.
        mc_values_all : numpy.ndarray
            Array of all Monte Carlo realization results, expected shape:
            (n_realizations, n_cells, n_timesteps).
        target_positions : list of tuples
            List of (x,y) coordinate tuples where time series are to be extracted.

    Returns:
        tuple (numpy.ndarray, list)
            - time_series_at_pos: Array of shape (n_positions, n_realizations, n_timesteps)
                                 containing the extracted time series.
            - cell_indices_found: List of cell indices in `mesh_obj` corresponding to `target_positions`.
    """
    n_realizations_mc = mc_values_all.shape[0] # Renamed
    n_timesteps_mc = mc_values_all.shape[2]    # Renamed
    
    # Find indices of mesh cells closest to the specified (x,y) positions.
    cell_indices_found = [] # Renamed
    for x_pos, y_pos in target_positions:
        # Calculate Euclidean distances from all cell centers to the target (x,y) position.
        cell_centers_coords = np.array(mesh_obj.cellCenters()) # Renamed
        distances_to_pos = np.sqrt((cell_centers_coords[:, 0] - x_pos)**2 + (cell_centers_coords[:, 1] - y_pos)**2) # Renamed
        closest_cell_idx = np.argmin(distances_to_pos) # Index of the cell with minimum distance. Renamed.
        cell_indices_found.append(closest_cell_idx)
    
    # Extract the time series data for each realization at each identified cell index.
    time_series_at_pos = np.zeros((len(target_positions), n_realizations_mc, n_timesteps_mc)) # Renamed
    
    for i_pos, cell_idx_val in enumerate(cell_indices_found): # Renamed loop vars
        for i_mc, mc_realization_idx in enumerate(range(n_realizations_mc)): # Renamed loop vars
            time_series_at_pos[i_pos, i_mc, :] = mc_values_all[mc_realization_idx, cell_idx_val, :]

    return time_series_at_pos, cell_indices_found

# --- Helper Function to Extract True Values at Specific Locations ---
def extract_true_values_at_positions(mesh_obj_true, true_values_data, target_positions): # Renamed args
    """
    Extract true (reference) water content values at specific (x,y) positions.
    This is used for comparing estimated values against ground truth.
    
    Args:
        mesh_obj_true : pygimli.Mesh
            The PyGIMLi mesh on which `true_values_data` are defined (can be different from inversion mesh).
        true_values_data : numpy.ndarray
            Array of true water content values. Expected shape: (n_cells_true_mesh, n_timesteps)
            or (n_timesteps, n_cells_true_mesh) - needs consistency with how it's indexed.
            The original script uses WC_true.T, implying (n_cells, n_timesteps) if WC_true is (n_timesteps, n_cells).
        target_positions : list of tuples
            List of (x,y) coordinate tuples where true values are to be extracted.

    Returns:
        tuple (numpy.ndarray, list)
            - true_values_at_pos: Extracted true values at each position. Shape depends on `true_values_data`.
            - cell_indices_true_found: List of cell indices in `mesh_obj_true` for `target_positions`.
    """
    cell_indices_true_found = [] # Renamed
    for x_pos, y_pos in target_positions:
        cell_centers_true = np.array(mesh_obj_true.cellCenters()) # Renamed
        distances_true = np.sqrt((cell_centers_true[:, 0] - x_pos)**2 + (cell_centers_true[:, 1] - y_pos)**2) # Renamed
        closest_cell_idx_true = np.argmin(distances_true) # Renamed
        cell_indices_true_found.append(closest_cell_idx_true)
    
    # Extract true values at the identified cell indices.
    # Handles cases where `true_values_data` might be single snapshot (1D per cell) or time-series (2D per cell).
    if true_values_data.ndim == 1:  # If true_values_data is (n_cells_true_mesh,).
        true_values_at_pos = true_values_data[cell_indices_true_found]
    elif true_values_data.ndim == 2:  # If true_values_data is (n_cells_true_mesh, n_timesteps) or (n_timesteps, n_cells_true_mesh).
                                     # Based on `WC_true.T` usage, it's (n_cells_true_mesh, n_timesteps).
        true_values_at_pos = true_values_data[cell_indices_true_found, :]
    else:
        raise ValueError(f"Unexpected shape for true_values_data: {true_values_data.shape}")
    
    return true_values_at_pos, cell_indices_true_found


# %%
# --- Extract and Plot Time Series for Points in Top Layer (Regolith) ---
# Define (x,y) positions of interest, assumed to be in the regolith layer.
# These coordinates should be relative to the mesh's coordinate system.
positions_regolith = [ # Renamed
    (80, 1621),  # Example Point B from plot.
    (40, 1607),  # Example Point A from plot.
]

# Extract Monte Carlo time series data for these regolith positions.
# `mesh` is the inversion mesh, `water_content_all` holds all MC realizations.
time_series_regolith, _ = extract_mc_time_series(mesh, water_content_all, positions_regolith) # Renamed, ignore cell_indices output

# Extract true water content time series at these positions for comparison.
# `mesh_true` is the mesh for true WC, `WC_true_array.T` transposes true WC to (n_cells, n_timesteps).
true_wc_regolith, _ = extract_true_values_at_positions(mesh_true, WC_true_array.T, positions_regolith) # Renamed

# %%
# --- Plot Time Series for Regolith Points ---
plt.figure(figsize=(12, 3)) # Create a new figure.

# Define time axis for plotting (e.g., days, assuming 30-day intervals for 12 timesteps).
measurement_times_days = np.arange(30, 361, 30) # Renamed

# Plot for the first regolith position.
ax_regolith1 = plt.subplot(1, 2, 1) # Renamed
mean_ts_regolith1 = np.mean(time_series_regolith[0, :, :], axis=0) # Mean of MC results for position 0. Renamed.
std_ts_regolith1 = np.std(time_series_regolith[0, :, :], axis=0)   # Std dev for position 0. Renamed.

ax_regolith1.plot(measurement_times_days, mean_ts_regolith1, 'o-', color='tab:blue', label='Estimated Mean WC') # Plot mean.
# Plot uncertainty band (mean +/- 1 std dev).
ax_regolith1.fill_between(measurement_times_days, mean_ts_regolith1 - std_ts_regolith1,
                          mean_ts_regolith1 + std_ts_regolith1, color='tab:blue', alpha=0.2, label='Estimated +/- 1 std')
ax_regolith1.plot(measurement_times_days, true_wc_regolith[0, :], 'tab:blue', ls='--', label='True WC') # Plot true WC.
ax_regolith1.grid(True)
ax_regolith1.legend(frameon=False)
ax_regolith1.set_xlabel('Time (Days)')
ax_regolith1.set_ylabel('Water Content (-)')
ax_regolith1.set_ylim(0, 0.35) # Set consistent y-axis limits.
ax_regolith1.set_title(f"Regolith Point 1 ({positions_regolith[0][0]},{positions_regolith[0][1]})") # Add title with coords

# Plot for the second regolith position.
ax_regolith2 = plt.subplot(1, 2, 2) # Renamed
mean_ts_regolith2 = np.mean(time_series_regolith[1, :, :], axis=0) # Mean for position 1. Renamed.
std_ts_regolith2 = np.std(time_series_regolith[1, :, :], axis=0)   # Std dev for position 1. Renamed.

ax_regolith2.plot(measurement_times_days, mean_ts_regolith2, 'o-', color='tab:blue', label='Estimated Mean WC')
ax_regolith2.fill_between(measurement_times_days, mean_ts_regolith2 - std_ts_regolith2,
                          mean_ts_regolith2 + std_ts_regolith2, color='tab:blue', alpha=0.2, label='Estimated +/- 1 std')
ax_regolith2.plot(measurement_times_days, true_wc_regolith[1, :], 'tab:blue', ls='--', label='True WC')
ax_regolith2.set_xlabel('Time (Days)')
ax_regolith2.set_ylabel('Water Content (-)')
ax_regolith2.set_ylim(0, 0.35)
ax_regolith2.legend(frameon=False)
ax_regolith2.grid(True)
ax_regolith2.set_title(f"Regolith Point 2 ({positions_regolith[1][0]},{positions_regolith[1][1]})")

plt.tight_layout()
plt.suptitle("Water Content Time Series (Regolith Layer)", fontsize=14, y=1.05) # Add overall title
plt.show() # Output: Matplotlib plot.


# %%
# --- Extract and Plot Time Series for Points in Fractured Bedrock Layer ---
# Define (x,y) positions assumed to be in the fractured bedrock layer.
positions_fractured = [ # Renamed
    (30, 1604),  # Example Point C from plot.
    (65, 1608),  # Example Point D from plot.
]

# Extract Monte Carlo time series data for these fractured bedrock positions.
time_series_fractured, _ = extract_mc_time_series(mesh, water_content_all, positions_fractured) # Renamed

# Extract true water content time series at these positions.
true_wc_fractured, _ = extract_true_values_at_positions(mesh_true, WC_true_array.T, positions_fractured) # Renamed

# %%
# --- Plot Time Series for Fractured Bedrock Points ---
plt.figure(figsize=(12, 3)) # Create a new figure.

# Plot for the first fractured bedrock position.
ax_fractured1 = plt.subplot(1, 2, 1) # Renamed
mean_ts_fractured1 = np.mean(time_series_fractured[0, :, :], axis=0) # Mean for position 0. Renamed.
std_ts_fractured1 = np.std(time_series_fractured[0, :, :], axis=0)   # Std dev for position 0. Renamed.

ax_fractured1.plot(measurement_times_days, mean_ts_fractured1, 'o-', color='tab:brown', label='Estimated Mean WC')
ax_fractured1.fill_between(measurement_times_days, mean_ts_fractured1 - std_ts_fractured1,
                           mean_ts_fractured1 + std_ts_fractured1, color='tab:brown', alpha=0.2, label='Estimated +/- 1 std')
ax_fractured1.plot(measurement_times_days, true_wc_fractured[0, :], 'tab:brown', ls='--', label='True WC')
ax_fractured1.grid(True)
ax_fractured1.legend(frameon=False)
ax_fractured1.set_xlabel('Time (Days)')
ax_fractured1.set_ylabel('Water Content (-)')
ax_fractured1.set_ylim(0, 0.35)
ax_fractured1.set_title(f"Fractured Point 1 ({positions_fractured[0][0]},{positions_fractured[0][1]})")

# Plot for the second fractured bedrock position.
ax_fractured2 = plt.subplot(1, 2, 2) # Renamed
mean_ts_fractured2 = np.mean(time_series_fractured[1, :, :], axis=0) # Mean for position 1. Renamed.
std_ts_fractured2 = np.std(time_series_fractured[1, :, :], axis=0)   # Std dev for position 1. Renamed.

ax_fractured2.plot(measurement_times_days, mean_ts_fractured2, 'o-', color='tab:brown', label='Estimated Mean WC')
ax_fractured2.fill_between(measurement_times_days, mean_ts_fractured2 - std_ts_fractured2,
                           mean_ts_fractured2 + std_ts_fractured2, color='tab:brown', alpha=0.2, label='Estimated +/- 1 std')
ax_fractured2.plot(measurement_times_days, true_wc_fractured[1, :], 'tab:brown', ls='--', label='True WC')
ax_fractured2.set_xlabel('Time (Days)')
ax_fractured2.set_ylabel('Water Content (-)')
ax_fractured2.set_ylim(0, 0.35)
ax_fractured2.legend(frameon=False)
ax_fractured2.grid(True)
ax_fractured2.set_title(f"Fractured Point 2 ({positions_fractured[1][0]},{positions_fractured[1][1]})")

plt.tight_layout()
plt.suptitle("Water Content Time Series (Fractured Bedrock Layer)", fontsize=14, y=1.05) # Add overall title
plt.show() # Output: Matplotlib plot.
# End of Example 8.



