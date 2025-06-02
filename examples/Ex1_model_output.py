=======
# Script-level comment:
# This script, Ex1_model_output.py, serves as a basic demonstration of how to use
# PyHydroGeophysX to load and perform initial processing of outputs from
# common hydrological models, specifically ParFlow and MODFLOW.
#
# Key functionalities showcased:
# 1. Instantiation of data processing classes for ParFlow (ParflowSaturation, ParflowPorosity)
#    and MODFLOW (MODFLOWWaterContent, MODFLOWPorosity).
# 2. Loading specific output variables:
#    - ParFlow: Saturation data for a given timestep, static porosity field, and domain mask.
#    - MODFLOW: Water content data from UZF package for a given timestep, and porosity
#               data derived from model input packages (e.g., LPF, UPW, STO via Flopy).
# 3. Basic data manipulation: Applying a mask to ParFlow data to handle inactive cells (setting to NaN).
# 4. Visualization: Plotting 2D slices of the loaded porosity and saturation/water content
#    data using Matplotlib to provide a quick look at the model outputs.
#
# Assumptions:
# - The script assumes that example data files are present in a specific subdirectory
#   structure relative to the script's location:
#   `examples/`
#   `|-- data/`
#   `    |-- parflow/`
#   `    |   |-- test2/  (contains ParFlow output files like test2.out.satur.00200.pfb, test2.out.porosity.pfb, test2.out.mask.pfb)`
#   `    |-- modflow/`
#   `        |-- id.txt  (MODFLOW idomain array for masking)`
#   `        |-- TLnewtest2sfb2.nam (MODFLOW name file and other associated model files like .lpf, .uzf.binary)`
# - Users trying to run this example with their own data should update the file paths
#   (`model_directory_parflow`, `parflow_run_name`, `modflow_sim_workspace`, `modflow_model_name`, `idomain_array` path)
#   and relevant parameters (e.g., timestep numbers, slice indices) accordingly.
#
# Expected output:
# - Console output: Prints the shapes of the loaded ParFlow and MODFLOW data arrays.
# - Matplotlib plots: Two figures, one showing a slice of ParFlow porosity and saturation,
#   and another showing a slice of MODFLOW porosity and water content. These plots are displayed interactively.

"""
Ex 1. Loading and Processing Hydrological Model Outputs
==================================================

This example demonstrates how to load and process outputs from different 
hydrological models using PyHydroGeophysX. We show examples for both 
ParFlow and MODFLOW models.

The example covers:
- Loading ParFlow saturation and porosity data
- Loading MODFLOW water content and porosity data  
- Basic visualization of the loaded data

This is typically the first step in any workflow where you want to
convert hydrological model outputs to geophysical data.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# --- Setup for running the script ---
# This section adjusts the Python path to ensure the PyHydroGeophysX package
# can be imported correctly, especially when running the example directly
# from within the examples directory or a similar context.

# This section determines the script's current directory and adjusts `sys.path`
# to ensure that the `PyHydroGeophysX` package can be imported. This is particularly
# useful when running examples directly from the `examples` folder, allowing the interpreter
# to find the package in the parent directory.
# For Jupyter notebooks, use the current working directory
try:
    # For regular Python scripts, __file__ is defined.
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # For Jupyter notebooks or interactive environments where __file__ is not defined,
    # os.getcwd() provides the current working directory.
    current_dir = os.getcwd()

# Add the parent directory of `PyHydroGeophysX` (i.e., the project root) to `sys.path`.
# This line assumes the `examples` directory is located one level below the project root.
# If `PyHydroGeophysX` is properly installed (e.g., via pip install -e . or a distribution),
# this explicit path manipulation might not be necessary as the package would be in the Python path.
parent_dir = os.path.dirname(os.path.dirname(current_dir)) # Navigates up two levels from the script's directory.
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# --- Import PyHydroGeophysX modules ---
# These lines import the necessary classes from the PyHydroGeophysX package
# for handling ParFlow and MODFLOW model outputs.
from PyHydroGeophysX.model_output.parflow_output import ParflowSaturation, ParflowPorosity
from PyHydroGeophysX.model_output.modflow_output import MODFLOWWaterContent, MODFLOWPorosity


# %% [markdown]
# ## 1. Parflow example
# <!-- This section demonstrates loading hydrological data from a ParFlow simulation. -->
# <!-- ParFlow output files are typically in .pfb (ParFlow Binary) format. -->
# <!-- Key variables of interest are often saturation and porosity. -->

# %%

# --- Define Path to ParFlow Model Directory and Run Name ---
# The `current_dir` variable holds the path to the directory where this example script is located.
# Example ParFlow data for this script is expected to be in a subdirectory: "data/parflow/test2/".
# Users should modify `model_directory_parflow` to point to their own ParFlow output directory structure.
model_directory_parflow = os.path.join(current_dir, "data", "parflow", "test2")
# The `parflow_run_name` variable corresponds to the base name used for ParFlow output files.
# For example, if saturation files are named "myrun.out.satur.00001.pfb", then `parflow_run_name` is "myrun".
parflow_run_name = "test2" # Example run name.

# --- Load ParFlow Saturation Data ---
# Instantiate the `ParflowSaturation` class, providing the path to the model directory and the run name.
# This processor is specifically designed to handle ParFlow saturation (.satur.pfb) files.
saturation_processor = ParflowSaturation(
    model_directory=model_directory_parflow,
    run_name=parflow_run_name
)
# Load saturation data for a specific ParFlow timestep.
# The `load_timestep` method requires the integer timestep number as it appears in the ParFlow output filenames
# (e.g., for "test2.out.satur.00200.pfb", the timestep is 200).
parflow_timestep_to_load = 200
saturation_data_pf = saturation_processor.load_timestep(parflow_timestep_to_load)

# --- Load ParFlow Porosity Data ---
# Instantiate the `ParflowPorosity` class, similar to the saturation processor.
# This handles ParFlow porosity files (e.g., .porosity.pfb).
porosity_processor_pf = ParflowPorosity(
    model_directory=model_directory_parflow,
    run_name=parflow_run_name
)
# Porosity is often a static field (not time-dependent) in ParFlow models.
# The `load_porosity` method reads the corresponding porosity file.
porosity_data_pf = porosity_processor_pf.load_porosity()

# --- Load ParFlow Mask Data and Apply to Porosity/Saturation ---
# ParFlow models often use a mask file (.mask.pfb) to define active (value=1) and inactive (value=0) cells
# within the model domain. Applying this mask is crucial for correct data interpretation and visualization.
mask_data_pf = porosity_processor_pf.load_mask() # The ParflowPorosity class can also load the mask.
# print(f"ParFlow Mask data shape: {mask_data_pf.shape}") # Optional: uncomment to check the shape of the mask.

# Apply the mask: where the mask is 0 (inactive cells), set the corresponding porosity and saturation values to NaN.
# NaN (Not a Number) values are typically ignored by plotting functions or treated specially in calculations.
# This step assumes that porosity_data_pf, saturation_data_pf, and mask_data_pf have compatible shapes.
porosity_data_pf[mask_data_pf==0] = np.nan    # Set inactive porosity cells to NaN.
saturation_data_pf[mask_data_pf==0] = np.nan  # Set inactive saturation cells to NaN.

# %%
# --- Print Shapes and Visualize ParFlow Data ---
# This block prints the shape of the loaded saturation data and then visualizes
# 2D slices of porosity and saturation for a quick inspection.

# Print the shape of the loaded ParFlow saturation data array.
# For 3D ParFlow output, this is typically (number_of_z_layers, number_of_y_cells, number_of_x_cells).
print(f"ParFlow Saturation data shape: {saturation_data_pf.shape}")

# Define the index of the Z-layer to be plotted. ParFlow data is often (nz, ny, nx).
# Index 19 refers to the 20th layer from the top (0-indexed).
slice_idx_pf = 19 # Users might need to adjust this based on their model's dimensions.
# SUGGESTION: Add a check to ensure `slice_idx_pf` is a valid index for `porosity_data_pf.shape[0]`
# to prevent runtime errors if the data has fewer layers. E.g.:
# if not (0 <= slice_idx_pf < porosity_data_pf.shape[0]):
#     raise ValueError(f"Invalid slice_idx_pf: {slice_idx_pf} for data with {porosity_data_pf.shape[0]} layers.")

# Create a Matplotlib figure with two subplots side-by-side.
plt.figure(figsize=(10, 4)) # Adjust figure size as needed.

# Plot Porosity Slice:
plt.subplot(1, 2, 1) # (rows, columns, panel number)
# `imshow` is used to display the 2D array as an image.
# `origin='lower'` places the (0,0) index of the array at the bottom-left corner of the plot.
# `cmap='viridis'` is a perceptually uniform colormap, good for representing scalar data.
plt.imshow(porosity_data_pf[slice_idx_pf, :, :], cmap='viridis', origin='lower')
plt.colorbar(label='Porosity (-)') # Add a colorbar with a descriptive label.
plt.title(f'ParFlow Porosity (Layer {slice_idx_pf})') # Set a title for this subplot.
# Note on axis orientation: For a map view of a layer (X-Y plane at a certain depth/Z-layer),
# `origin='lower'` means Y increases upwards. If `porosity_data_pf` is (z,y,x), this displays
# the y-x plane for the selected z-layer.

# Plot Saturation Slice:
plt.subplot(1, 2, 2) # Second subplot.
plt.imshow(saturation_data_pf[slice_idx_pf, :, :], cmap='viridis', origin='lower')
plt.colorbar(label='Saturation (-)')
plt.title(f'ParFlow Saturation (Layer {slice_idx_pf}, Timestep {parflow_timestep_to_load})')

plt.tight_layout() # Adjusts subplot params for a tight layout, preventing overlapping titles/labels.
plt.show() # Display the generated plots. Output is an interactive Matplotlib window.

# %% [markdown]
# ## 2. MODFLOW example
# <!-- This section demonstrates loading hydrological data from a MODFLOW simulation. -->
# <!-- MODFLOW outputs can include binary files (e.g., from UZF package for water content) -->
# <!-- and standard package files (e.g., LPF, UPW for porosity-related parameters). -->
# <!-- An `idomain` array is often used to identify active model cells. -->

# %%
# This empty cell was likely for users to add their specific MODFLOW path setup if different from below.

# %%
# --- Define Paths and Load MODFLOW Data ---
# Set up paths to the MODFLOW example data.
# `current_dir` is the directory of this script (Ex1_model_output.py).
# MODFLOW example data is assumed to be in "examples/data/modflow/".
# Users should modify `modflow_sim_workspace` to their MODFLOW model's root directory.
data_dir_base = os.path.join(current_dir, "data") # Base path for example data.
modflow_sim_workspace = os.path.join(data_dir_base, "modflow") # Path to the MODFLOW example dataset.

# Load the `idomain` array.
# The `idomain` array in MODFLOW specifies active (value > 0), inactive (value = 0),
# or fixed-head (value < 0) cells. It's crucial for masking results to valid model areas.
# Here, it's loaded from a simple text file "id.txt", assumed to contain a 2D array (nrows x ncols).
# This file should be prepared by the user, representing their model grid's active status.
idomain_filepath = os.path.join(modflow_sim_workspace, "id.txt")
idomain_array = np.loadtxt(idomain_filepath)

# --- Load MODFLOW Water Content Data ---
# This example uses the `MODFLOWWaterContent` class, which is specifically designed to read
# binary "WaterContent" output files generated by the MODFLOW UZF (Unsaturated Zone Flow) package.
# Instantiate the water content processor.
# `model_directory`: Path to the MODFLOW simulation workspace (where .uzf.binary files are).
# `idomain`: The loaded idomain array, used for mapping UZF data to the grid and masking.
water_content_processor_mf = MODFLOWWaterContent(
    model_directory=modflow_sim_workspace,
    idomain=idomain_array
)

# Load water content for a specific MODFLOW timestep.
# `load_timestep` takes a zero-based timestep index relative to the available UZF output times.
# The number of UZF layers (`nlay_uzf`) might need to be specified if it's not the default (3 layers).
# This example loads the data for the second timestep (index 1).
target_timestep_idx = 1
water_content_data_mf = water_content_processor_mf.load_timestep(target_timestep_idx)
# The shape of `water_content_data_mf` is expected to be (nlay_uzf, nrows, ncols).

# Print the shape of the loaded MODFLOW water content data.
print(f"MODFLOW Water Content data shape: {water_content_data_mf.shape}")


# Define the `model_name` for the MODFLOW simulation.
# This is the base name of the MODFLOW model, as used in the main MODFLOW name file (e.g., "mymodel" for "mymodel.nam").
# It's required by Flopy to correctly locate and parse model files.
modflow_model_name = "TLnewtest2sfb2"  # Replace with your actual MODFLOW model name.

# --- Load MODFLOW Porosity Data ---
# Porosity in MODFLOW models is typically defined within specific input packages:
# - LPF (Layer Property Flow) or UPW (Upstream Weighting Package) often contain Specific Yield (SY),
#   which is effectively drainable porosity for unconfined layers, and Specific Storage (SS).
# - STO (Storage Package, common in MODFLOW 6) directly defines SY and SS.
# The `MODFLOWPorosity` class utilizes Flopy to read these package files and extract porosity.
# 1. Create an instance of the MODFLOWPorosity class:
porosity_loader_mf = MODFLOWPorosity(
    model_directory=modflow_sim_workspace, # Path to the MODFLOW model workspace.
    model_name=modflow_model_name          # Base name of the MODFLOW model.
)
# 2. Load the porosity data:
# The `load_porosity` method attempts to find porosity-related parameters (like SY or SS, prioritizing SY for porosity).
# It returns a NumPy array, typically of shape (nlay_model, nrows, ncols).
porosity_data_mf = porosity_loader_mf.load_porosity()

# %%
# --- Visualize MODFLOW Data ---
# This block plots 2D slices of the loaded MODFLOW porosity and water content for a quick visual check.
# Slices are typically taken from the first model layer or first UZF layer.

# Prepare porosity data for plotting:
# Select the data for the first model layer (index 0).
porosity_slice_mf = porosity_data_mf[0, :, :].copy() # Use .copy() to avoid unintended modifications to original data.
# Apply the idomain mask: set cells to NaN where idomain is 0 (inactive).
# This ensures that visualizations only show data from active parts of the model.
# This assumes `idomain_array` is 2D and its dimensions match `porosity_slice_mf`.
porosity_slice_mf[idomain_array == 0] = np.nan

# Prepare water content data for plotting:
# Select data for the first UZF layer (index 0).
water_content_slice_mf = water_content_data_mf[0, :, :].copy()
# Apply the idomain mask to the water content slice as well for consistency.
water_content_slice_mf[idomain_array == 0] = np.nan


# Create a Matplotlib figure for the MODFLOW data plots.
plt.figure(figsize=(10, 4))

# Plot MODFLOW Porosity Slice:
plt.subplot(1, 2, 1) # First subplot in a 1x2 grid.
plt.imshow(porosity_slice_mf, cmap='viridis', origin='lower') # Display data as an image.
plt.colorbar(label='Porosity (-)') # Add a color scale.
plt.title(f'MODFLOW Porosity (Model Layer 0)') # Add a title.

# Plot MODFLOW Water Content Slice:
plt.subplot(1, 2, 2) # Second subplot.
plt.imshow(water_content_slice_mf, cmap='viridis', origin='lower')
plt.colorbar(label='Water Content (-)')
plt.title(f'MODFLOW Water Content (UZF Layer 0, Timestep {target_timestep_idx})')

plt.tight_layout() # Adjust subplot layout to prevent overlaps.
plt.show() # Display the plots. Expected output: Matplotlib window with two subplots.

# %%
# End of Example 1.
# This script has demonstrated the basic loading and visualization of ParFlow and MODFLOW
# model outputs using PyHydroGeophysX. More complex data processing, integration,
# and conversion to geophysical parameters are covered in subsequent examples.
#
# Users should adapt file paths (e.g., `model_directory_parflow`, `modflow_sim_workspace`),
# model-specific names (`parflow_run_name`, `modflow_model_name`), and
# parameters like timestep numbers or slice indices to match their own datasets and requirements.



