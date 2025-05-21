"""
Module for converting ERT resistivity models to water content with structural information.
"""
import numpy as np
import pygimli as pg
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Union, Callable


from ..petrophysics.resistivity_models import resistivity_to_saturation



class ERTtoWC:
    """Class for converting ERT resistivity models to water content."""
    
    def __init__(self, 
                 mesh: pg.Mesh,
                 resistivity_values: np.ndarray,
                 cell_markers: np.ndarray,
                 coverage: Optional[np.ndarray] = None):
        """
        Initialize converter.
        
        Args:
            mesh: PyGIMLI mesh
            resistivity_values: Resistivity values, shape (n_cells, n_timesteps)
            cell_markers: Cell markers to identify different geological layers
            coverage: Coverage values (optional)
        """
        self.mesh = mesh
        self.resistivity_values = resistivity_values
        self.cell_markers = cell_markers
        self.coverage = coverage
    
    def setup_layer_distributions(self, 
                                layer_distributions: Dict[int, Dict[str, Dict[str, float]]]) -> None:
        """
        Set up parameter distributions for different layers.
        
        Args:
            layer_distributions: Dictionary mapping layer markers to parameter distributions
        """
        self.layer_distributions = layer_distributions
        self.layer_markers = list(layer_distributions.keys())
    
    def run_monte_carlo(self, n_realizations: int = 100, progress_bar: bool = True) -> Tuple:
        """
        Run Monte Carlo simulation for uncertainty quantification.
        
        Args:
            n_realizations: Number of Monte Carlo realizations
            progress_bar: Whether to show progress bar
        
        Returns:
            Tuple of (water_content_all, saturation_all, params_used)
        """
        if not hasattr(self, 'layer_distributions'):
            raise ValueError("Layer distributions not set. Call setup_layer_distributions first.")
        
        # Initialize arrays to store results
        water_content_all = np.zeros((n_realizations, *self.resistivity_values.shape))
        saturation_all = np.zeros((n_realizations, *self.resistivity_values.shape))
        
        # Store parameters used for each realization
        params_used = {marker: {param: np.zeros(n_realizations) for param in 
                              ['rhos', 'n', 'sigma_sur', 'porosity']} 
                     for marker in self.layer_markers}
        
        # Setup iterator
        iterator = tqdm(range(n_realizations), desc="Monte Carlo Simulations") if progress_bar else range(n_realizations)
        
        # Run Monte Carlo simulation
        for mc_idx in iterator:
            # Sample parameters for each layer
            layer_params = {}
            porosity = np.zeros_like(self.cell_markers, dtype=float)
            
            for marker in self.layer_markers:
                # Get distribution for this layer
                layer_dist = self.layer_distributions[marker]
                
                # Sample parameters
                layer_params[marker] = {
                    'rhos': max(1.0, np.random.normal(layer_dist['rhos']['mean'], layer_dist['rhos']['std'])),
                    'n': max(1.0, np.random.normal(layer_dist['n']['mean'], layer_dist['n']['std'])),
                    'sigma_sur': max(0.0, np.random.normal(layer_dist['sigma_sur']['mean'], layer_dist['sigma_sur']['std']))
                }
                
                # Sample porosity
                porosity_value = np.clip(np.random.normal(layer_dist['porosity']['mean'], 
                                                         layer_dist['porosity']['std']), 0.05, 0.6)
                porosity[self.cell_markers == marker] = porosity_value
                
                # Store parameters
                for param, value in {'rhos': layer_params[marker]['rhos'], 
                                   'n': layer_params[marker]['n'],
                                   'sigma_sur': layer_params[marker]['sigma_sur'],
                                   'porosity': porosity_value}.items():
                    params_used[marker][param][mc_idx] = value
            
            # Process each timestep
            for t in range(self.resistivity_values.shape[1]):
                resistivity_t = self.resistivity_values[:, t]
                
                # Process each layer
                for marker in self.layer_markers:
                    mask_layer = self.cell_markers == marker
                    if np.any(mask_layer):
                        params = layer_params[marker]
                        saturation_all[mc_idx, mask_layer, t] = resistivity_to_saturation(
                            resistivity_t[mask_layer],
                            params['rhos'],
                            params['n'],
                            params['sigma_sur']
                        )
                
                # Convert saturation to water content
                water_content_all[mc_idx, :, t] = saturation_all[mc_idx, :, t] * porosity
        
        # Store results
        self.water_content_all = water_content_all
        self.saturation_all = saturation_all
        self.params_used = params_used
        
        return water_content_all, saturation_all, params_used
    
    def get_statistics(self) -> Dict[str, np.ndarray]:
        """Calculate statistics across Monte Carlo realizations."""
        if not hasattr(self, 'water_content_all'):
            raise ValueError("No Monte Carlo results. Run run_monte_carlo first.")
        
        return {
            'mean': np.mean(self.water_content_all, axis=0),
            'std': np.std(self.water_content_all, axis=0),
            'p10': np.percentile(self.water_content_all, 10, axis=0),
            'p50': np.percentile(self.water_content_all, 50, axis=0),
            'p90': np.percentile(self.water_content_all, 90, axis=0)
        }
    
    def extract_time_series(self, positions: List[Tuple[float, float]]) -> Tuple[np.ndarray, List[int]]:
        """Extract time series at specific positions."""
        if not hasattr(self, 'water_content_all'):
            raise ValueError("No Monte Carlo results. Run run_monte_carlo first.")
        
        # Find indices of cells closest to specified positions
        cell_indices = []
        for x_pos, y_pos in positions:
            cell_centers = np.array(self.mesh.cellCenters())
            distances = np.sqrt((cell_centers[:, 0] - x_pos)**2 + (cell_centers[:, 1] - y_pos)**2)
            cell_indices.append(np.argmin(distances))
        
        # Extract time series
        n_realizations = self.water_content_all.shape[0]
        n_timesteps = self.water_content_all.shape[2]
        time_series = np.zeros((len(positions), n_realizations, n_timesteps))
        
        for pos_idx, cell_idx in enumerate(cell_indices):
            time_series[pos_idx] = self.water_content_all[:, cell_idx, :]
        
        return time_series, cell_indices
    
    def plot_water_content(self, time_idx: int = 0, ax=None, 
                         cmap: str = 'jet', cmin: float = 0.0, cmax: float = 0.32,
                         coverage_threshold: Optional[float] = None):
        """Plot water content for a specific time step."""
        import matplotlib.pyplot as plt
        
        if not hasattr(self, 'water_content_all'):
            raise ValueError("No Monte Carlo results. Run run_monte_carlo first.")
        
        # Get values to plot
        values = self.get_statistics()['mean'][:, time_idx]
        
        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        # Apply coverage mask if needed
        if coverage_threshold is not None and self.coverage is not None:
            if self.coverage.ndim == 2:
                coverage_mask = self.coverage[time_idx, :] < coverage_threshold
            else:
                coverage_mask = self.coverage < coverage_threshold
                
            values_masked = np.ma.array(values, mask=coverage_mask)
        else:
            values_masked = values
        
        # Create plot
        return pg.show(
            self.mesh,
            values_masked,
            cMap=cmap,
            cMin=cmin,
            cMax=cmax,
            label='Water Content (-)',
            logScale=False,
            ax=ax
        )
    
    def save_results(self, output_dir: str, base_filename: str) -> None:
        """Save Monte Carlo results to files."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save statistics
        stats = self.get_statistics()
        for stat_name, stat_values in stats.items():
            np.save(os.path.join(output_dir, f"{base_filename}_{stat_name}.npy"), stat_values)


def plot_time_series(time_steps: np.ndarray, time_series_data: np.ndarray,
                    true_values: Optional[np.ndarray] = None,
                    labels: Optional[List[str]] = None,
                    colors: Optional[List[str]] = None,
                    output_file: Optional[str] = None):
    """Plot time series with uncertainty bands."""
    import matplotlib.pyplot as plt
    
    n_positions = time_series_data.shape[0]
    
    # Default labels and colors
    if labels is None:
        labels = [f"Position {i+1}" for i in range(n_positions)]
    
    if colors is None:
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
        colors = [colors[i % len(colors)] for i in range(n_positions)]
    
    # Create figure
    fig, axes = plt.subplots(1, n_positions, figsize=(12, 4))
    axes = np.atleast_1d(axes)
    
    # Plot each position
    for i in range(n_positions):
        ax = axes[i]
        
        # Calculate statistics
        mean_ts = np.mean(time_series_data[i], axis=0)
        std_ts = np.std(time_series_data[i], axis=0)
        
        # Plot mean and uncertainty
        ax.plot(time_steps, mean_ts, 'o-', color=colors[i], label='Estimated')
        ax.fill_between(time_steps, mean_ts-std_ts, mean_ts+std_ts, color=colors[i], alpha=0.2)
        
        # Plot true values if provided
        if true_values is not None:
            ax.plot(time_steps, true_values[i], ls='--', color=colors[i], label='True')
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Water Content (-)')
        ax.grid(True)
        ax.set_title(labels[i] if i < len(labels) else f"Position {i+1}")
        
        if i == 0:
            ax.legend(frameon=False)
    
    plt.tight_layout()
    
    # Save figure if output_file is provided
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    return fig