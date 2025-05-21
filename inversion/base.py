"""
Base classes for geophysical inversion.
"""
import numpy as np
import pygimli as pg
import matplotlib.pyplot as plt
from typing import Optional, Union, List, Dict, Any, Tuple


class InversionResult:
    """Base class to store inversion results."""
    
    def __init__(self):
        """Initialize inversion result container."""
        self.final_model = None  # Final model parameters
        self.predicted_data = None  # Predicted data
        self.coverage = None  # Coverage values
        self.mesh = None  # Mesh used in inversion
        self.iteration_models = []  # List to store model parameters at each iteration
        self.iteration_data_errors = []  # List to store data errors at each iteration
        self.iteration_chi2 = []  # List to store chi-squared values at each iteration
        self.meta = {}  # Dictionary for additional metadata
    
    def save(self, filename: str) -> None:
        """
        Save results to file.
        
        Args:
            filename: Path to save file
        """
        import pickle
        
        # Create a dictionary with all relevant attributes
        data = {
            'final_model': self.final_model,
            'predicted_data': self.predicted_data,
            'coverage': self.coverage,
            'iteration_models': self.iteration_models,
            'iteration_data_errors': self.iteration_data_errors,
            'iteration_chi2': self.iteration_chi2,
            'meta': self.meta
        }
        
        # Save mesh separately if it exists
        if self.mesh is not None:
            mesh_filename = filename + '.bms'
            pg.save(self.mesh, mesh_filename)
            data['mesh_file'] = mesh_filename
        
        # Save data using pickle
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
    
    @classmethod
    def load(cls, filename: str) -> 'InversionResult':
        """
        Load results from file.
        
        Args:
            filename: Path to load file
            
        Returns:
            Loaded InversionResult
        """
        import pickle
        
        # Load data using pickle
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        # Create new InversionResult
        result = cls()
        
        # Assign attributes
        result.final_model = data['final_model']
        result.predicted_data = data['predicted_data']
        result.coverage = data['coverage']
        result.iteration_models = data['iteration_models']
        result.iteration_data_errors = data['iteration_data_errors']
        result.iteration_chi2 = data['iteration_chi2']
        result.meta = data['meta']
        
        # Load mesh if it exists
        if 'mesh_file' in data:
            result.mesh = pg.load(data['mesh_file'])
        
        return result
    
    def plot_model(self, ax=None, cmap='jet', coverage_threshold=None, **kwargs):
        """
        Plot the final model.
        
        Args:
            ax: Matplotlib axis for plotting (creates new if None)
            cmap: Colormap for model values
            coverage_threshold: Threshold for coverage masking (None for no mask)
            **kwargs: Additional arguments for plotting
            
        Returns:
            fig, ax: Figure and axis objects
        """
        if self.final_model is None or self.mesh is None:
            raise ValueError("Cannot plot model: model or mesh is missing")
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.figure
        
        # Get model to plot (considering coverage mask if specified)
        model_to_plot = self.final_model
        
        if coverage_threshold is not None and self.coverage is not None:
            # Create masked array for plotting
            mask = self.coverage < coverage_threshold
            model_to_plot = np.ma.array(self.final_model, mask=mask)
        
        # Plot model on mesh
        cb = pg.show(self.mesh, model_to_plot, ax=ax, cMap=cmap, **kwargs)
        
        return fig, ax
    
    def plot_convergence(self, ax=None, **kwargs):
        """
        Plot convergence curve.
        
        Args:
            ax: Matplotlib axis for plotting (creates new if None)
            **kwargs: Additional arguments for plotting
            
        Returns:
            fig, ax: Figure and axis objects
        """
        if not self.iteration_chi2:
            raise ValueError("No convergence data available")
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        else:
            fig = ax.figure
        
        iterations = range(1, len(self.iteration_chi2) + 1)
        ax.plot(iterations, self.iteration_chi2, 'o-', **kwargs)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Chi²')
        ax.grid(True)
        
        return fig, ax


class TimeLapseInversionResult(InversionResult):
    """Class to store time-lapse inversion results."""
    
    def __init__(self):
        """Initialize time-lapse inversion result container."""
        super().__init__()
        self.final_models = None  # 2D array (cells, timesteps)
        self.timesteps = None  # Array of timestep values
        self.all_coverage = []  # List of coverage values for each timestep
        self.all_chi2 = []  # List of chi-squared values for each timestep
    
    def plot_time_slice(self, timestep_idx: int, ax=None, cmap='jet', 
                       coverage_threshold=None, **kwargs):
        """
        Plot a single time slice from the results.
        
        Args:
            timestep_idx: Index of the timestep to plot
            ax: Matplotlib axis for plotting
            cmap: Colormap for model values
            coverage_threshold: Threshold for coverage masking
            **kwargs: Additional arguments for plotting
            
        Returns:
            fig, ax: Figure and axis objects
        """
        if self.final_models is None or self.mesh is None:
            raise ValueError("Cannot plot model: models or mesh is missing")
        
        if timestep_idx < 0 or timestep_idx >= self.final_models.shape[1]:
            raise ValueError(f"Invalid timestep index: {timestep_idx}")
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.figure
        
        # Get model slice for this timestep
        model_slice = self.final_models[:, timestep_idx]
        
        if coverage_threshold is not None and len(self.all_coverage) > 0:
            # Create masked array for plotting
            coverage = self.all_coverage[timestep_idx] if len(self.all_coverage) > timestep_idx else self.all_coverage[0]
            mask = coverage < coverage_threshold
            model_slice = np.ma.array(model_slice, mask=mask)
        
        # Plot model on mesh
        cb = pg.show(self.mesh, model_slice, ax=ax, cMap=cmap, **kwargs)
        
        # Add timestep information if available
        if self.timesteps is not None:
            ax.set_title(f"Timestep {timestep_idx} (t = {self.timesteps[timestep_idx]})")
        
        return fig, ax

    def create_time_lapse_animation(self, filename, 
                                  cmap='jet', coverage_threshold=None,
                                  dpi=100, fps=2, **kwargs):
        """
        Create an animation of time-lapse results.
        
        Args:
            filename: Output filename (e.g., 'animation.mp4')
            cmap: Colormap for model values
            coverage_threshold: Threshold for coverage masking
            dpi: DPI for the output animation
            fps: Frames per second
            **kwargs: Additional arguments for plotting
        """
        import matplotlib.animation as animation
        
        if self.final_models is None or self.mesh is None:
            raise ValueError("Cannot create animation: models or mesh is missing")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Setup function to update each frame
        def update_frame(frame):
            ax.clear()
            self.plot_time_slice(frame, ax=ax, cmap=cmap, 
                              coverage_threshold=coverage_threshold, **kwargs)
            return [ax]
        
        # Create animation
        ani = animation.FuncAnimation(
            fig, update_frame, frames=self.final_models.shape[1], 
            blit=True
        )
        
        # Save animation
        ani.save(filename, dpi=dpi, writer='ffmpeg', fps=fps)
        plt.close(fig)


class InversionBase:
    """Base class for geophysical inversion."""
    
    def __init__(self, data: pg.DataContainer, mesh: Optional[pg.Mesh] = None, **kwargs):
        """
        Initialize inversion with data and mesh.
        
        Args:
            data: Observed data
            mesh: Mesh for inversion (created if None)
            **kwargs: Additional parameters for inversion
        """
        self.data = data
        self.mesh = mesh
        self.result = InversionResult()
        self.parameters = kwargs
        
        # Default parameters
        self.default_parameters = {
            'lambda': 10.0,  # Regularization parameter
            'max_iterations': 20,  # Maximum number of iterations
            'min_chi2': 1.0,  # Target chi-squared value
            'tolerance': 0.01,  # Convergence tolerance
            'model_constraints': (1e-6, 1e7),  # Min and max model values
        }
        
        # Update defaults with provided kwargs
        for key, value in self.default_parameters.items():
            if key not in self.parameters:
                self.parameters[key] = value
    
    def setup(self):
        """Set up inversion (create operators, matrices, etc.)"""
        # Create mesh if not provided
        if self.mesh is None:
            # This should be implemented in derived classes as it depends on the specific inversion type
            raise NotImplementedError("Mesh creation must be implemented in derived classes")
    
    def run(self):
        """
        Run inversion and return results.
        
        Returns:
            InversionResult with inversion results
        """
        # This should be implemented in derived classes
        raise NotImplementedError("Run method must be implemented in derived classes")
    
    def compute_jacobian(self, model: np.ndarray):
        """
        Compute Jacobian matrix for a given model.
        
        Args:
            model: Model parameters
            
        Returns:
            Jacobian matrix
        """
        # This should be implemented in derived classes
        raise NotImplementedError("Jacobian computation must be implemented in derived classes")
    
    def objective_function(self, model: np.ndarray, data: Optional[np.ndarray] = None):
        """
        Compute objective function value.
        
        Args:
            model: Model parameters
            data: Observed data (uses self.data if None)
            
        Returns:
            Value of the objective function
        """
        # This should be implemented in derived classes
        raise NotImplementedError("Objective function must be implemented in derived classes")
