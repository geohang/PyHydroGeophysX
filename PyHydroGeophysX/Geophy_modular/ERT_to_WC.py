"""
Module for converting Electrical Resistivity Tomography (ERT) resistivity models to
volumetric water content, incorporating structural information (geological layers)
and quantifying uncertainty using Monte Carlo simulations.

This module provides the `ERTtoWC` class, which takes ERT resistivity data,
a corresponding mesh, cell markers identifying different layers, and optional
coverage information. It allows users to define petrophysical parameter
distributions (saturated resistivity `rhos`, saturation exponent `n`,
surface conductivity `sigma_sur`, and porosity `φ`) for each layer.
The core functionality involves running Monte Carlo simulations to sample these
parameters and convert resistivity to water content for each realization,
thereby providing a distribution of possible water content values.
Statistics (mean, std, percentiles) can then be calculated from these distributions.
The module also includes utilities for plotting results and extracting time series.
"""

import numpy as np
import pygimli as pg
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Union, Callable


from ..petrophysics.resistivity_models import resistivity_to_saturation



class ERTtoWC:
    """Class for converting ERT resistivity models to water content."""
    
    def __init__(self, 
                 mesh: pg.Mesh,                        # The geophysical mesh
                 resistivity_values: np.ndarray,       # Inverted resistivity models over time
                 cell_markers: np.ndarray,             # Markers linking cells to layers/materials
                 coverage: Optional[np.ndarray] = None # Optional ERT coverage/sensitivity map
                 ):
        """
        Initialize the ERTtoWC converter.

        This class takes time-lapse ERT inversion results (resistivity models) and
        converts them to volumetric water content (θ) using petrophysical relationships.
        It supports uncertainty quantification through Monte Carlo simulations by sampling
        petrophysical parameters for different geological layers defined by cell markers.

        Args:
            mesh (pg.Mesh): The PyGIMLi mesh object that corresponds to the resistivity models
                            and cell markers.
            resistivity_values (np.ndarray): A 2D NumPy array where rows are cells and columns are
                                             timesteps (n_cells x n_timesteps), containing the
                                             inverted electrical resistivity values (in Ω·m).
            cell_markers (np.ndarray): A 1D NumPy array of integers, with length equal to the
                                       number of cells in the mesh. Each integer is a marker
                                       identifying the geological layer or material type for that cell.
                                       This is used to apply layer-specific petrophysical parameters.
            coverage (Optional[np.ndarray], optional): A 1D or 2D NumPy array representing the
                                                       ERT survey coverage or sensitivity for each cell
                                                       (and optionally for each timestep if 2D).
                                                       Used for filtering or weighting results. Defaults to None.
                                                       SUGGESTION: Clarify expected shape and use of coverage if 2D.
                                                       If 1D, assumed constant for all timesteps.
        """
        self.mesh = mesh # Store the PyGIMLi mesh.
        self.resistivity_values = resistivity_values # Store time-lapse resistivity models.
        self.cell_markers = cell_markers # Store cell markers.
        self.coverage = coverage # Store coverage/sensitivity information.

        # Initialize attributes that will be populated by other methods.
        self.layer_distributions: Optional[Dict[int, Dict[str, Dict[str, float]]]] = None # To store parameter distributions.
        self.layer_markers: Optional[List[int]] = None # To store unique layer markers.
        self.water_content_all: Optional[np.ndarray] = None # To store all MC realizations of water content.
        self.saturation_all: Optional[np.ndarray] = None    # To store all MC realizations of saturation.
        self.params_used: Optional[Dict[int, Dict[str, np.ndarray]]] = None # To store sampled parameters.

    
    def setup_layer_distributions(self, 
                                layer_distributions: Dict[int, Dict[str, Dict[str, float]]]
                                ) -> None:
        """
        Set up the statistical distributions for petrophysical parameters for each geological layer.

        Each layer (identified by a marker in `cell_markers`) can have different mean and
        standard deviation for its petrophysical properties (rhos, n, sigma_sur, porosity).
        These distributions are used in the Monte Carlo simulation to sample parameter sets.

        Args:
            layer_distributions (Dict[int, Dict[str, Dict[str, float]]]):
                A dictionary where keys are integer cell markers (identifying layers)
                and values are dictionaries of parameters. Each parameter dictionary
                (e.g., for 'rhos') contains 'mean' and 'std' (standard deviation) keys.
                Example:
                {
                    1: { # Layer with marker 1
                        'rhos': {'mean': 100.0, 'std': 20.0},      # Saturated resistivity (Ω·m)
                        'n': {'mean': 2.0, 'std': 0.1},            # Saturation exponent
                        'sigma_sur': {'mean': 0.0, 'std': 0.001},  # Surface conductivity (S/m)
                        'porosity': {'mean': 0.3, 'std': 0.05}     # Porosity (φ, fraction)
                    },
                    2: { ... } # Parameters for layer with marker 2
                }
        """
        self.layer_distributions = layer_distributions # Store the provided distributions.
        # Get a list of unique layer markers for which distributions are defined.
        self.layer_markers = list(layer_distributions.keys())
    
    def run_monte_carlo(self, n_realizations: int = 100, progress_bar: bool = True
                        ) -> Tuple[np.ndarray, np.ndarray, Dict[int, Dict[str, np.ndarray]]]:
        """
        Run Monte Carlo simulation for uncertainty quantification.
        
        Args:
            n_realizations: Number of Monte Carlo realizations
            progress_bar: Whether to show progress bar
        
        Returns:
            Tuple[np.ndarray, np.ndarray, Dict[int, Dict[str, np.ndarray]]]:
                - water_content_all (np.ndarray): All realizations of water content
                                                  (n_realizations, n_cells, n_timesteps).
                - saturation_all (np.ndarray): All realizations of saturation
                                               (n_realizations, n_cells, n_timesteps).
                - params_used (Dict): Dictionary storing the sampled petrophysical parameters
                                      for each realization and each layer.
        """
        # Ensure that layer parameter distributions have been set up before running.
        if self.layer_distributions is None or self.layer_markers is None: # Check both for robustness
            raise ValueError("Layer distributions not set. Call setup_layer_distributions first.")
        
        # --- Initialize Arrays to Store Monte Carlo Results ---
        # `water_content_all`: Stores water content for each realization, cell, and timestep.
        # `saturation_all`: Stores saturation for each realization, cell, and timestep.
        # Shape: (n_realizations, n_cells, n_timesteps)
        # `self.resistivity_values.shape` is (n_cells, n_timesteps).
        num_cells = self.resistivity_values.shape[0]
        num_timesteps = self.resistivity_values.shape[1] if self.resistivity_values.ndim > 1 else 1
        # Adjust shape if resistivity_values is 1D (single timestep)
        if self.resistivity_values.ndim == 1: # Single timestep case
            resistivity_values_2d = self.resistivity_values.reshape(-1, 1)
            num_timesteps = 1
        else:
            resistivity_values_2d = self.resistivity_values

        water_content_all = np.zeros((n_realizations, num_cells, num_timesteps))
        saturation_all = np.zeros((n_realizations, num_cells, num_timesteps))
        
        # `params_used`: A nested dictionary to store the specific petrophysical parameters
        # sampled for each realization and for each layer.
        # Structure: params_used[marker][parameter_name][realization_index] = value
        params_used_mc = {
            marker: {param: np.zeros(n_realizations)
                     for param in ['rhos', 'n', 'sigma_sur', 'porosity']}
            for marker in self.layer_markers
        }
        
        # Use `tqdm` for a progress bar if requested, otherwise a simple range.
        realization_iterator = tqdm(range(n_realizations), desc="Monte Carlo Simulations") if progress_bar else range(n_realizations)
        
        # --- Monte Carlo Simulation Loop ---
        for mc_idx in realization_iterator:
            # For each realization, sample a new set of petrophysical parameters for each layer.
            # `layer_sampled_params` will store {marker: {param_name: value}} for this realization.
            layer_sampled_params: Dict[int, Dict[str, float]] = {}
            # `porosity_realization` will be an array of porosity values for all cells,
            # sampled based on the cell's layer marker for this realization.
            porosity_realization = np.zeros_like(self.cell_markers, dtype=float)
            
            for marker_id in self.layer_markers: # Iterate through each defined layer marker.
                current_layer_dist = self.layer_distributions[marker_id] # Get distributions for this layer.

                # Sample each petrophysical parameter from its normal distribution (mean, std).
                # Apply constraints (e.g., rhos > 1.0, n > 1.0, sigma_sur >= 0.0, porosity within bounds).
                # SUGGESTION: Consider using truncated normal distributions or other distributions
                # if normal distribution can lead to too many physically implausible samples (e.g. negative n).
                # `max(min_val, ...)` is a simple way to enforce lower bound. Clipping for upper bound.
                sampled_rhos = max(1.0, np.random.normal(current_layer_dist['rhos']['mean'], current_layer_dist['rhos']['std']))
                sampled_n = max(1.0, np.random.normal(current_layer_dist['n']['mean'], current_layer_dist['n']['std']))
                sampled_sigma_sur = max(0.0, np.random.normal(current_layer_dist['sigma_sur']['mean'], current_layer_dist['sigma_sur']['std']))
                
                layer_sampled_params[marker_id] = {
                    'rhos': sampled_rhos,
                    'n': sampled_n,
                    'sigma_sur': sampled_sigma_sur
                }
                
                # Sample porosity and clip to a physically plausible range [0.05, 0.6].
                # SUGGESTION: Porosity bounds could also be part of `layer_distributions`.
                sampled_porosity_value = np.clip(
                    np.random.normal(current_layer_dist['porosity']['mean'], current_layer_dist['porosity']['std']),
                    0.05, 0.6
                )
                # Assign this sampled porosity to all cells belonging to the current layer marker.
                porosity_realization[self.cell_markers == marker_id] = sampled_porosity_value
                
                # Store the sampled parameters for this realization and layer.
                params_used_mc[marker_id]['rhos'][mc_idx] = sampled_rhos
                params_used_mc[marker_id]['n'][mc_idx] = sampled_n
                params_used_mc[marker_id]['sigma_sur'][mc_idx] = sampled_sigma_sur
                params_used_mc[marker_id]['porosity'][mc_idx] = sampled_porosity_value
            
            # --- Process Each Timestep for the Current Realization ---
            for t_idx in range(num_timesteps):
                resistivity_at_timestep_t = resistivity_values_2d[:, t_idx] # Resistivity model for current timestep.
                
                # Calculate saturation for each layer using its sampled parameters.
                for marker_id in self.layer_markers:
                    layer_cell_mask = (self.cell_markers == marker_id) # Mask for cells in the current layer.
                    if np.any(layer_cell_mask): # If any cells belong to this layer.
                        current_params_for_layer = layer_sampled_params[marker_id]
                        # `resistivity_to_saturation` converts resistivity to saturation using Waxman-Smits.
                        # It requires rhos, n, sigma_sur for the layer. Porosity is not directly used by it,
                        # as it solves for S.
                        saturation_values_layer = resistivity_to_saturation(
                            resistivity=resistivity_at_timestep_t[layer_cell_mask],
                            rhos=current_params_for_layer['rhos'],
                            n=current_params_for_layer['n'],
                            sigma_sur=current_params_for_layer['sigma_sur']
                        )
                        saturation_all[mc_idx, layer_cell_mask, t_idx] = saturation_values_layer
                
                # Convert saturation to volumetric water content (θ = S * φ) for this realization and timestep.
                # `porosity_realization` is the array of porosity values for all cells for this mc_idx.
                water_content_all[mc_idx, :, t_idx] = saturation_all[mc_idx, :, t_idx] * porosity_realization
        
        # Store all results in instance attributes.
        self.water_content_all = water_content_all
        self.saturation_all = saturation_all
        self.params_used = params_used_mc # Store the dict of parameters used
        
        return self.water_content_all, self.saturation_all, self.params_used
    
    def get_statistics(self) -> Dict[str, np.ndarray]:
        """Calculate statistics across Monte Carlo realizations."""
        if not hasattr(self, 'water_content_all') or self.water_content_all is None:
            raise ValueError("Monte Carlo results are not available. Please run `run_monte_carlo()` first.")
        
        # Calculate mean, standard deviation, and percentiles (10th, 50th=median, 90th)
        # of the water content across all Monte Carlo realizations (axis=0).
        # The results will have shape (n_cells, n_timesteps).
        return {
            'mean': np.mean(self.water_content_all, axis=0),      # Mean water content
            'std': np.std(self.water_content_all, axis=0),        # Standard deviation of water content
            'p10': np.percentile(self.water_content_all, 10, axis=0), # 10th percentile
            'p50': np.percentile(self.water_content_all, 50, axis=0), # 50th percentile (median)
            'p90': np.percentile(self.water_content_all, 90, axis=0)  # 90th percentile
        }
    
    def extract_time_series(self, positions: List[Tuple[float, float]]) -> Tuple[np.ndarray, List[int]]:
        """Extract time series at specific positions."""
        if not hasattr(self, 'water_content_all') or self.water_content_all is None:
            raise ValueError("Monte Carlo results are not available. Please run `run_monte_carlo()` first.")
        
        # --- Find Mesh Cell Indices Closest to Specified (x, y) Positions ---
        cell_indices: List[int] = [] # To store the index of the mesh cell closest to each position.
        mesh_cell_centers = np.array(self.mesh.cellCenters()) # Get all cell center coordinates.
        
        for x_pos, y_pos in positions: # For each requested (x,y) position
            # Calculate Euclidean distances from this position to all cell centers.
            distances_to_centers = np.sqrt(
                (mesh_cell_centers[:, 0] - x_pos)**2 + (mesh_cell_centers[:, 1] - y_pos)**2
            )
            # Find the index of the cell with the minimum distance.
            closest_cell_idx = np.argmin(distances_to_centers)
            cell_indices.append(closest_cell_idx)
        
        # --- Extract Water Content Time Series for These Cells ---
        # `self.water_content_all` has shape (n_realizations, n_cells, n_timesteps).
        # We need to select data for the found `cell_indices`.
        num_realizations = self.water_content_all.shape[0]
        num_timesteps = self.water_content_all.shape[2] # Assuming 3rd dimension is time
        
        # Initialize an array to store the extracted time series.
        # Shape: (num_positions, n_realizations, n_timesteps)
        extracted_time_series = np.zeros((len(positions), num_realizations, num_timesteps))

        # For each requested position (identified by its closest cell_idx),
        # extract the water content data across all realizations and timesteps.
        for pos_idx, cell_idx_val in enumerate(cell_indices):
            extracted_time_series[pos_idx, :, :] = self.water_content_all[:, cell_idx_val, :]

        return extracted_time_series, cell_indices # Return the data and the cell indices used.
    
    def plot_water_content(self, time_idx: int = 0, ax=None, 
                         cmap: str = 'jet', cmin: float = 0.0, cmax: float = 0.32,
                         coverage_threshold: Optional[float] = None):
        """Plot water content for a specific time step."""
        import matplotlib.pyplot as plt
        
        if not hasattr(self, 'water_content_all') or self.water_content_all is None: # Check if MC results exist
            raise ValueError("No Monte Carlo results available. Please run `run_monte_carlo()` first.")
        
        # Get the mean water content for the specified timestep across all Monte Carlo realizations.
        # `self.get_statistics()` returns a dict; `['mean']` is (n_cells, n_timesteps).
        mean_wc_all_timesteps = self.get_statistics()['mean']
        # Select the data for the specified `time_idx`.
        values_to_plot = mean_wc_all_timesteps[:, time_idx]
        
        # Create a new figure and axes if none are provided by the user.
        if ax is None:
            _fig, ax = plt.subplots(figsize=(10, 6)) # Underscore for fig if not returned/used.
        
        # --- Apply Coverage Mask (Optional) ---
        # If a coverage threshold and coverage data are available, mask cells with low coverage.
        # This prevents plotting water content in areas where ERT resolution is poor.
        values_for_plotting = values_to_plot # Default to unmasked values.
        if coverage_threshold is not None and self.coverage is not None:
            # Determine the coverage mask.
            # self.coverage could be 1D (static coverage) or 2D (time-lapse coverage: n_timesteps, n_cells).
            # The original code seems to handle this by checking ndim.
            # Assuming coverage values are higher for better coverage.
            current_coverage_data: np.ndarray
            if self.coverage.ndim == 2: # Time-lapse coverage
                if time_idx < self.coverage.shape[0]: # Check if time_idx is valid for coverage array
                    current_coverage_data = self.coverage[time_idx, :]
                else:
                    print(f"Warning: time_idx {time_idx} out of bounds for coverage array. Using coverage from t=0.")
                    current_coverage_data = self.coverage[0, :] # Fallback
            elif self.coverage.ndim == 1: # Static coverage
                current_coverage_data = self.coverage
            else:
                raise ValueError(f"Coverage data has unexpected dimension: {self.coverage.ndim}. Expected 1D or 2D.")

            # Create mask: True for cells where coverage is *below* the threshold (these will be masked out).
            coverage_mask = current_coverage_data < coverage_threshold
            # Apply the mask to the water content values.
            values_for_plotting = np.ma.array(values_to_plot, mask=coverage_mask)
        
        # --- Create Plot using PyGIMLi's `show` function ---
        # `pg.show` is a versatile plotting utility for meshes and cell data.
        # It returns the axes object, which can be useful for further customization.
        # SUGGESTION: Use a more perceptually uniform colormap than 'jet' if possible (e.g., 'viridis', 'cividis').
        plot_axes = pg.show(
            self.mesh,                      # The PyGIMLi mesh to plot on.
            data=values_for_plotting,       # The (masked) water content values for each cell.
            cMap=cmap,                      # Colormap for visualizing water content.
            cMin=cmin,                      # Minimum value for the color scale.
            cMax=cmax,                      # Maximum value for the color scale.
            label='Water Content (-)',      # Label for the color bar.
            logScale=False,                 # Use linear color scale (log often used for resistivity).
            ax=ax                           # The matplotlib axes to plot on.
        )
        return plot_axes # Return the axes for potential further manipulation.
    
    def save_results(self, output_dir: str, base_filename: str) -> None:
        """Save Monte Carlo results to files."""
        # Import `os` module locally if not already imported at module level.
        # import os # Already imported at the top of the file.

        # Create the output directory if it doesn't already exist.
        os.makedirs(output_dir, exist_ok=True)
        
        # --- Save Calculated Statistics ---
        # Get the statistics (mean, std, percentiles) of water content.
        # These are typically shaped (n_cells, n_timesteps).
        calculated_stats = self.get_statistics()
        for stat_name, stat_array_values in calculated_stats.items():
            # Construct filename for each statistic (e.g., "my_results_mean.npy").
            file_path = os.path.join(output_dir, f"{base_filename}_{stat_name}.npy")
            # Save the NumPy array to a .npy file.
            np.save(file_path, stat_array_values)
            print(f"Saved {stat_name} to {file_path}")

        # --- Optionally, Save All Monte Carlo Realizations ---
        # This can be large but is useful for more detailed post-analysis.
        # SUGGESTION: Add flags to control saving of all realizations, as these files can be very large.
        # e.g., if self.parameters.get('save_all_realizations', False):
        if self.water_content_all is not None:
            wc_all_path = os.path.join(output_dir, f"{base_filename}_water_content_all_realizations.npy")
            np.save(wc_all_path, self.water_content_all)
            print(f"Saved all water content realizations to {wc_all_path}")

        if self.saturation_all is not None:
            sat_all_path = os.path.join(output_dir, f"{base_filename}_saturation_all_realizations.npy")
            np.save(sat_all_path, self.saturation_all)
            print(f"Saved all saturation realizations to {sat_all_path}")

        if self.params_used is not None:
            # Saving a dictionary of numpy arrays (params_used) might be better with np.savez or pickle,
            # or save each parameter set individually if needed.
            # For simplicity with .npy, one might convert it to a structured array or save components.
            # Current: Not saving params_used directly to a single .npy due to its nested dict structure.
            # SUGGESTION: Implement saving for `params_used`, perhaps to a .npz file or JSON for dict structure.
            # Example for npz: np.savez(os.path.join(output_dir, f"{base_filename}_params_used.npz"), **self.params_used)
            # This would require params_used to be structured in a way np.savez can handle (e.g. dict of arrays).
            # The current structure is Dict[int, Dict[str, np.ndarray]]. This needs careful handling for saving.
            print("Note: Sampled parameters (`params_used`) are not saved by this basic save_results method yet.")


def plot_time_series(time_steps: np.ndarray,      # Array of time values for x-axis
                    time_series_data: np.ndarray, # Extracted time series data
                    true_values: Optional[np.ndarray] = None, # Optional true values for comparison
                    labels: Optional[List[str]] = None,       # Labels for each position/time series
                    colors: Optional[List[str]] = None,       # Colors for each time series plot
                    output_file: Optional[str] = None         # Path to save the figure
                    ) -> plt.Figure : # Type hint for matplotlib Figure
    """
    Plot time series data, typically mean water content with uncertainty bands (std dev).
    Allows for plotting multiple positions/locations on separate subplots.

    Args:
        time_steps (np.ndarray): 1D array of time values (e.g., hours, days) for the x-axis.
        time_series_data (np.ndarray): 3D array containing time series data from Monte Carlo.
                                       Shape: (n_positions, n_realizations, n_timesteps).
                                       `n_positions` is the number of different locations/cells.
        true_values (Optional[np.ndarray], optional): 2D array of true values for comparison.
                                                      Shape: (n_positions, n_timesteps). Defaults to None.
        labels (Optional[List[str]], optional): List of labels for each position's subplot title.
                                                Defaults to generic "Position X".
        colors (Optional[List[str]], optional): List of colors for plotting each position's series.
                                                Defaults to a predefined list of matplotlib colors.
        output_file (Optional[str], optional): If provided, the plot will be saved to this file path.
                                               Defaults to None (plot displayed but not saved).

    Returns:
        plt.Figure: The matplotlib Figure object containing the plot(s).
    """
    import matplotlib.pyplot as plt # Local import for plotting.
    
    num_positions = time_series_data.shape[0] # Number of different positions/cells to plot.
    
    # --- Setup Default Labels and Colors if Not Provided ---
    if labels is None:
        labels = [f"Position {i+1}" for i in range(num_positions)]
    
    if colors is None:
        # Define a default list of distinct colors. Cycle through them if n_positions > len(colors).
        default_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
        colors = [default_colors[i % len(default_colors)] for i in range(num_positions)]
    
    # --- Create Figure and Axes ---
    # Create a figure with `n_positions` subplots arranged horizontally.
    # `figsize` can be adjusted based on the number of positions.
    fig, axes = plt.subplots(nrows=1, ncols=num_positions,
                             figsize=(max(12, 4 * num_positions), 4), # Adjust width based on num_positions
                             sharey=True) # Share y-axis for easier comparison if data ranges are similar.
    # If only one position, `axes` will not be an array but a single Axes object. Ensure it's always iterable.
    axes = np.atleast_1d(axes)
    
    # --- Plot Data for Each Position ---
    for i in range(num_positions):
        ax = axes[i] # Current subplot axes.

        # Calculate statistics for the current position's time series data.
        # `time_series_data[i]` has shape (n_realizations, n_timesteps).
        # Calculate mean and standard deviation across realizations (axis=0).
        mean_time_series = np.mean(time_series_data[i], axis=0) # Mean over realizations for this position.
        std_dev_time_series = np.std(time_series_data[i], axis=0)  # Std dev over realizations.
        
        # Plot the mean time series. 'o-' means markers and a line.
        ax.plot(time_steps, mean_time_series, 'o-', color=colors[i], label='Estimated (Mean)')
        
        # Plot the uncertainty band (mean ± 1 standard deviation).
        # `fill_between` creates the shaded region. `alpha` controls transparency.
        ax.fill_between(time_steps,
                        mean_time_series - std_dev_time_series,
                        mean_time_series + std_dev_time_series,
                        color=colors[i], alpha=0.2, label='Mean ±1 Std Dev')
        
        # Plot true values if provided for comparison.
        if true_values is not None and i < true_values.shape[0]: # Ensure true_values exist for this position
            ax.plot(time_steps, true_values[i], ls='--', color=colors[i], label='True Value')
        
        # --- Styling for Each Subplot ---
        ax.set_xlabel('Time') # X-axis label.
        if i == 0: # Set y-axis label only for the first subplot if y-axes are shared.
            ax.set_ylabel('Water Content (-)')
        ax.grid(True, linestyle=':', alpha=0.7) # Add a subtle grid.
        ax.set_title(labels[i] if i < len(labels) else f"Position {i+1}") # Subplot title.
        
        # Add legend to the first subplot for clarity if multiple lines per plot.
        if i == 0: # Or `ax.legend()` for each if preferred.
            ax.legend(frameon=False)
    
    plt.tight_layout() # Adjust subplot params for a tight layout.
    
    # Save the figure to a file if `output_file` path is provided.
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Time series plot saved to: {output_file}")
    
    return fig # Return the Figure object.