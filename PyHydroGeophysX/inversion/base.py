"""
Base classes for geophysical inversion frameworks.

This module defines:
- `InversionResult`: A base class to store and manage common results from an inversion process,
  including final model, predicted data, convergence history, and plotting utilities.
- `TimeLapseInversionResult`: A specialized version of `InversionResult` for time-lapse studies,
  handling multiple models over time and providing time-slice plotting and animation.
- `InversionBase`: An abstract base class outlining the common structure and interface
  for various geophysical inversion methods (e.g., ERT, SRT). It handles data, mesh,
  and basic parameter management.
"""
import numpy as np
import pygimli as pg
import matplotlib.pyplot as plt # Used for plotting methods
import os # For path manipulation in save/load
from typing import Optional, Union, List, Dict, Any, Tuple # Dict is used in InversionResult.meta


class InversionResult:
    """
    Base class to store, save, load, and plot results from a geophysical inversion.

    Attributes:
        final_model (Optional[np.ndarray]): The final inverted model parameters (e.g., resistivity, velocity).
                                            Typically a 1D array corresponding to mesh cells.
        predicted_data (Optional[np.ndarray]): The data predicted by the forward model using the `final_model`.
        coverage (Optional[np.ndarray]): Coverage or sensitivity values for the model parameters,
                                         often derived from the Jacobian or resolution matrix.
        mesh (Optional[pg.Mesh]): The PyGIMLi mesh object used in the inversion.
        iteration_models (List[np.ndarray]): A list storing the model parameters at each iteration of the inversion.
        iteration_data_errors (List[np.ndarray]): A list storing the data misfit (e.g., residuals) at each iteration.
        iteration_chi2 (List[float]): A list storing the chi-squared (χ²) value or a similar misfit metric
                                      at each iteration.
        meta (Dict[str, Any]): A dictionary to store any additional metadata about the inversion run
                               (e.g., inversion parameters, timings, comments).
    """
    
    def __init__(self):
        """Initialize an empty InversionResult container."""
        self.final_model: Optional[np.ndarray] = None
        self.predicted_data: Optional[np.ndarray] = None
        self.coverage: Optional[np.ndarray] = None
        self.mesh: Optional[pg.Mesh] = None
        self.iteration_models: List[np.ndarray] = []
        self.iteration_data_errors: List[np.ndarray] = []
        self.iteration_chi2: List[float] = []
        self.meta: Dict[str, Any] = {} # For storing inversion parameters, etc.
    
    def save(self, filename: str) -> None:
        """
        Save the inversion results to a file using Python's pickle format.
        If a mesh is present, it is saved separately as a PyGIMLi binary mesh file (.bms).

        Args:
            filename (str): The base path (including filename without extension) to save the results.
                            The main data will be saved as `filename.pkl` (or just `filename` if user includes .pkl).
                            The mesh will be saved as `filename.bms` or `filename.pkl.bms`.
                            It's recommended to provide `filename` without `.pkl`.

        Raises:
            IOError: If there's an error during file writing.
            pickle.PicklingError: If an object cannot be pickled.
        """
        import pickle # Local import for a standard library module is fine.

        # Ensure filename doesn't inadvertently include .pkl if we append it later.
        base_filename, ext = os.path.splitext(filename)
        if ext.lower() == '.pkl':
            pickle_filename = filename
            mesh_save_filename_base = base_filename
        else: # No extension or different extension, assume filename is base
            pickle_filename = filename + ".pkl" # Standard extension for pickled files
            mesh_save_filename_base = filename

        data_to_save = {
            'final_model': self.final_model,
            'predicted_data': self.predicted_data,
            'coverage': self.coverage,
            'iteration_models': self.iteration_models,
            'iteration_data_errors': self.iteration_data_errors,
            'iteration_chi2': self.iteration_chi2,
            'meta': self.meta,
            'mesh_file': None # Placeholder, will be updated if mesh is saved
        }
        
        # Save mesh separately using PyGIMLi's binary format if it exists
        if self.mesh is not None:
            # Potential Issue: Appending '.bms' to a filename that might already have an extension
            # (e.g. 'results.pkl') could lead to 'results.pkl.bms'.
            # A cleaner way might be to derive mesh filename from the base filename.
            mesh_specific_filename = mesh_save_filename_base + '.bms'
            try:
                pg.save(self.mesh, mesh_specific_filename)
                data_to_save['mesh_file'] = mesh_specific_filename # Store relative path or just name
                print(f"Mesh saved to: {mesh_specific_filename}")
            except Exception as e:
                # Log error but continue to save other data if possible
                print(f"Warning: Could not save mesh to '{mesh_specific_filename}'. Error: {e}")
                data_to_save['mesh_file'] = None # Ensure it's None if saving failed

        # Save the rest of the data using pickle
        try:
            with open(pickle_filename, 'wb') as f:
                pickle.dump(data_to_save, f)
            print(f"Inversion results (excluding mesh) saved to: {pickle_filename}")
        except (IOError, pickle.PicklingError) as e:
            # Clean up mesh file if main data saving fails to avoid partial save?
            # For now, just raise the error.
            raise IOError(f"Failed to save inversion results to '{pickle_filename}': {e}")
    
    @classmethod
    def load(cls, filename: str) -> 'InversionResult':
        """
        Load inversion results from a file previously saved by the `save` method.

        Args:
            filename (str): The base path to the saved results file.
                            If saved as `name.pkl` and `name.bms`, provide `name.pkl` or `name`.

        Returns:
            InversionResult: An instance of `InversionResult` (or a subclass if called on one)
                             populated with the loaded data.
        
        Raises:
            FileNotFoundError: If the main data file or associated mesh file (if referenced) is not found.
            IOError: If there's an error during file reading.
            pickle.UnpicklingError: If the file cannot be unpickled.
        """
        import pickle # Local import
        
        # Determine the pickle filename
        base_filename, ext = os.path.splitext(filename)
        pickle_filename = filename if ext.lower() == '.pkl' else filename + ".pkl"

        if not os.path.exists(pickle_filename):
            # Try original filename if .pkl was added and not found
            if ext.lower() != '.pkl' and os.path.exists(filename):
                 pickle_filename = filename # User might have provided full name without .pkl
            else:
                 raise FileNotFoundError(f"Pickle data file not found: {pickle_filename} (or {filename})")

        # Load the main data dictionary using pickle
        with open(pickle_filename, 'rb') as f:
            loaded_data = pickle.load(f)

        # Create a new instance of the class (allows loading into subclasses like TimeLapseInversionResult)
        result_instance = cls()

        # Assign attributes from the loaded dictionary
        # Use .get() for robustness against missing keys if format changes, though direct access is fine if format is fixed.
        result_instance.final_model = loaded_data.get('final_model')
        result_instance.predicted_data = loaded_data.get('predicted_data')
        result_instance.coverage = loaded_data.get('coverage')
        result_instance.iteration_models = loaded_data.get('iteration_models', [])
        result_instance.iteration_data_errors = loaded_data.get('iteration_data_errors', [])
        result_instance.iteration_chi2 = loaded_data.get('iteration_chi2', [])
        result_instance.meta = loaded_data.get('meta', {})

        # Load the mesh if a mesh file path is stored in the data
        mesh_file_path = loaded_data.get('mesh_file')
        if mesh_file_path:
            # Check if the path is absolute or needs to be relative to the pickle file's directory
            if not os.path.isabs(mesh_file_path):
                pickle_dir = os.path.dirname(os.path.abspath(pickle_filename))
                mesh_file_path = os.path.join(pickle_dir, mesh_file_path)

            if os.path.exists(mesh_file_path):
                try:
                    result_instance.mesh = pg.load(mesh_file_path)
                    print(f"Mesh loaded from: {mesh_file_path}")
                except Exception as e:
                    print(f"Warning: Could not load mesh from '{mesh_file_path}'. Error: {e}")
            else:
                print(f"Warning: Mesh file '{mesh_file_path}' referenced in pickle file but not found.")

        return result_instance
    
    def plot_model(self, ax: Optional[plt.Axes] = None, cmap: str = 'viridis', # Changed default cmap
                   coverage_threshold: Optional[float] = None, **kwargs: Any) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot the final inverted model on its associated mesh.

        Args:
            ax (Optional[plt.Axes], optional): A matplotlib Axes object to plot on.
                                               If None, a new figure and axes are created. Defaults to None.
            cmap (str, optional): The colormap to use for visualizing model values.
                                  Defaults to 'viridis'.
            coverage_threshold (Optional[float], optional):
                If provided, cells with coverage values below this threshold will be masked
                (made transparent or semi-transparent) in the plot. Requires `self.coverage`
                to be populated. Defaults to None (no masking).
            **kwargs (Any): Additional keyword arguments passed directly to `pygimli.show`
                            (e.g., cMin, cMax, orientation, logScale).

        Returns:
            Tuple[plt.Figure, plt.Axes]: The matplotlib Figure and Axes objects of the plot.

        Raises:
            ValueError: If `self.final_model` or `self.mesh` is None.
        """
        if self.final_model is None or self.mesh is None:
            raise ValueError("Cannot plot model: final_model or mesh is not available in InversionResult.")
        
        fig: plt.Figure
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6)) # Default figure size
        else:
            fig = ax.figure # Get figure from existing axes
        
        model_to_display = np.array(self.final_model) # Ensure it's a NumPy array
        
        # Apply coverage masking if a threshold is provided and coverage data exists
        if coverage_threshold is not None and self.coverage is not None:
            if len(self.coverage) != len(model_to_display):
                print("Warning: Coverage array length does not match model length. Cannot apply coverage mask.")
            else:
                # Create a mask where True means "mask out" (i.e., coverage < threshold)
                mask = np.array(self.coverage) < coverage_threshold
                model_to_display = np.ma.array(model_to_display, mask=mask)

        # Use PyGIMLi's show function to plot the model on the mesh
        # `pg.show` can handle masked arrays automatically.
        # It returns the colorbar instance, which can be useful for customization.
        # Common kwargs for pg.show: cMin, cMax, cMap (cmap), orientation, label, logScale
        # Ensure cMap is passed correctly.
        cb = pg.show(self.mesh, data=model_to_display, ax=ax, cMap=cmap, **kwargs)
        # Example: add a colorbar label if not automatically set by pg.show or if desired
        # if 'label' not in kwargs and cb is not None:
        #     cb.set_label("Model Parameter Value")
        
        return fig, ax
    
    def plot_convergence(self, ax: Optional[plt.Axes] = None, **kwargs: Any) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot the convergence curve (chi-squared misfit vs. iteration number).

        Args:
            ax (Optional[plt.Axes], optional): A matplotlib Axes object to plot on.
                                               If None, a new figure and axes are created. Defaults to None.
            **kwargs (Any): Additional keyword arguments passed directly to `ax.plot`
                            (e.g., color, marker, linestyle).

        Returns:
            Tuple[plt.Figure, plt.Axes]: The matplotlib Figure and Axes objects of the plot.

        Raises:
            ValueError: If `self.iteration_chi2` is empty (no convergence data).
        """
        if not self.iteration_chi2: # Check if the list is empty
            raise ValueError("No convergence data (iteration_chi2) available to plot.")
        
        fig: plt.Figure
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5)) # Default figure size
        else:
            fig = ax.figure
        
        iterations = range(1, len(self.iteration_chi2) + 1) # Iterations usually start from 1
        ax.plot(iterations, self.iteration_chi2, marker='o', linestyle='-', **kwargs) # Default style
        ax.set_xlabel('Iteration Number')
        ax.set_ylabel('Chi-Squared (χ²)')
        ax.set_title('Inversion Convergence Curve')
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.set_yscale('log') # Chi-squared is often plotted on a log scale for y-axis
        # Optional: Set x-axis to integer ticks if many iterations
        if len(iterations) > 10:
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        return fig, ax


class TimeLapseInversionResult(InversionResult):
    """
    Specialized class to store and manage results from time-lapse inversions.
    Inherits from `InversionResult` and adds attributes specific to time-lapse data.

    Attributes:
        final_models (Optional[np.ndarray]): A 2D NumPy array where each column represents
                                              the inverted model for a specific timestep
                                              (shape: num_cells x num_timesteps).
        timesteps (Optional[np.ndarray]): An array or list of time values (e.g., hours, days)
                                          corresponding to each model slice in `final_models`.
        all_coverage (List[np.ndarray]): A list where each element is the coverage array
                                         for the corresponding timestep's model.
        all_chi2 (List[Any]): A list where each element is a list of chi-squared values
                                      per iteration for the inversion of that specific timestep or window.
                                      (Note: Original was List[float], if chi2 is per window, might need adjustment)
    """
    
    def __init__(self):
        """Initialize an empty TimeLapseInversionResult container."""
        super().__init__() # Initialize base class attributes
        self.final_models: Optional[np.ndarray] = None  # Shape: (num_cells, num_timesteps)
        self.timesteps: Optional[np.ndarray] = None     # Timestamps corresponding to model slices
        self.all_coverage: List[np.ndarray] = []       # List of coverage arrays, one per timestep
        self.all_chi2: List[Any] = [] # Could be List[List[float]] if each time step has its own convergence
                                      # Or List[float] if it's a global chi2 for joint/sequential inversion.
                                      # Original was List[float], implies one chi2 list for the whole process.
                                      # If from windowed inversion, this might be a list of chi2 lists.
    
    # Potential Improvement: Override save/load to handle time-lapse specific attributes if needed,
    # especially if their structure is complex or requires special handling beyond what pickle does.
    # For current attributes (mostly lists of NumPy arrays), default pickle should work if they are added to data_to_save dict.
    # The current save/load in InversionResult does not handle these time-lapse specific attributes.
    # This needs to be overridden.

    def save(self, filename: str) -> None:
        """
        Save time-lapse inversion results. Overrides base class `save`.
        """
        import pickle
        base_filename, ext = os.path.splitext(filename)
        pickle_filename = filename if ext.lower() == '.pkl' else filename + ".pkl"
        mesh_save_filename_base = base_filename

        data_to_save = {
            # Base attributes
            'final_model': self.final_model, # Might be None if final_models is primary
            'predicted_data': self.predicted_data,
            'coverage': self.coverage, # Might be None if all_coverage is primary
            'iteration_models': self.iteration_models,
            'iteration_data_errors': self.iteration_data_errors,
            'iteration_chi2': self.iteration_chi2,
            'meta': self.meta,
            'mesh_file': None,
            # Time-lapse specific attributes
            'final_models': self.final_models,
            'timesteps': self.timesteps,
            'all_coverage': self.all_coverage,
            'all_chi2_timelapse': self.all_chi2 # Renamed to avoid confusion with base iteration_chi2
        }
        if self.mesh is not None:
            mesh_specific_filename = mesh_save_filename_base + '.bms'
            try:
                pg.save(self.mesh, mesh_specific_filename)
                data_to_save['mesh_file'] = mesh_specific_filename
                print(f"Mesh saved to: {mesh_specific_filename}")
            except Exception as e:
                print(f"Warning: Could not save mesh to '{mesh_specific_filename}'. Error: {e}")

        try:
            with open(pickle_filename, 'wb') as f:
                pickle.dump(data_to_save, f)
            print(f"TimeLapseInversion results saved to: {pickle_filename}")
        except (IOError, pickle.PicklingError) as e:
            raise IOError(f"Failed to save TimeLapseInversion results to '{pickle_filename}': {e}")

    @classmethod
    def load(cls, filename: str) -> 'TimeLapseInversionResult':
        """
        Load time-lapse inversion results. Overrides base class `load`.
        """
        import pickle
        base_filename, ext = os.path.splitext(filename)
        pickle_filename = filename if ext.lower() == '.pkl' else filename + ".pkl"

        if not os.path.exists(pickle_filename):
            if ext.lower() != '.pkl' and os.path.exists(filename):
                 pickle_filename = filename
            else:
                 raise FileNotFoundError(f"Pickle data file not found: {pickle_filename} (or {filename})")

        with open(pickle_filename, 'rb') as f:
            loaded_data = pickle.load(f)

        result_instance = cls() # Creates instance of TimeLapseInversionResult

        # Load base attributes
        result_instance.final_model = loaded_data.get('final_model')
        result_instance.predicted_data = loaded_data.get('predicted_data')
        result_instance.coverage = loaded_data.get('coverage')
        result_instance.iteration_models = loaded_data.get('iteration_models', [])
        result_instance.iteration_data_errors = loaded_data.get('iteration_data_errors', [])
        result_instance.iteration_chi2 = loaded_data.get('iteration_chi2', []) # Base chi2 list
        result_instance.meta = loaded_data.get('meta', {})

        # Load time-lapse specific attributes
        result_instance.final_models = loaded_data.get('final_models')
        result_instance.timesteps = loaded_data.get('timesteps')
        result_instance.all_coverage = loaded_data.get('all_coverage', [])
        result_instance.all_chi2 = loaded_data.get('all_chi2_timelapse', loaded_data.get('all_chi2', [])) # Handle old naming

        mesh_file_path = loaded_data.get('mesh_file')
        if mesh_file_path:
            if not os.path.isabs(mesh_file_path):
                pickle_dir = os.path.dirname(os.path.abspath(pickle_filename))
                mesh_file_path = os.path.join(pickle_dir, mesh_file_path)
            if os.path.exists(mesh_file_path):
                try:
                    result_instance.mesh = pg.load(mesh_file_path)
                    print(f"Mesh loaded from: {mesh_file_path}")
                except Exception as e:
                    print(f"Warning: Could not load mesh from '{mesh_file_path}'. Error: {e}")
            else:
                print(f"Warning: Mesh file '{mesh_file_path}' not found.")
        return result_instance

    def plot_time_slice(self, timestep_idx: int, ax: Optional[plt.Axes] = None, cmap: str = 'viridis',
                       coverage_threshold: Optional[float] = None, **kwargs: Any) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot a single time slice (inverted model at a specific timestep) from the results.

        Args:
            timestep_idx (int): The zero-based index of the timestep to plot.
            ax (Optional[plt.Axes], optional): Matplotlib Axes to plot on. If None, creates new.
            cmap (str, optional): Colormap for model values. Defaults to 'viridis'.
            coverage_threshold (Optional[float], optional): Threshold for coverage masking.
                                                             Defaults to None.
            **kwargs (Any): Additional arguments passed to `pg.show`.

        Returns:
            Tuple[plt.Figure, plt.Axes]: The matplotlib Figure and Axes objects.

        Raises:
            ValueError: If models or mesh are missing, or if `timestep_idx` is out of range.
        """
        if self.final_models is None or self.mesh is None:
            raise ValueError("Cannot plot time slice: final_models or mesh is not available.")
        
        if not (0 <= timestep_idx < self.final_models.shape[1]):
            raise ValueError(f"Invalid timestep_idx: {timestep_idx}. "
                             f"Must be between 0 and {self.final_models.shape[1] - 1}.")
        
        fig: plt.Figure
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.figure
        
        # Get the model slice for the specified timestep
        model_slice_to_plot = self.final_models[:, timestep_idx]

        # Apply coverage masking if specified
        current_coverage_array: Optional[np.ndarray] = None
        if coverage_threshold is not None and self.all_coverage:
            if timestep_idx < len(self.all_coverage) and self.all_coverage[timestep_idx] is not None:
                current_coverage_array = np.array(self.all_coverage[timestep_idx])
                if len(current_coverage_array) != len(model_slice_to_plot):
                    print(f"Warning: Coverage array length for timestep {timestep_idx} does not match model slice length. Skipping coverage mask.")
                else:
                    mask = current_coverage_array < coverage_threshold
                    model_slice_to_plot = np.ma.array(model_slice_to_plot, mask=mask)
            elif self.all_coverage and self.all_coverage[0] is not None and coverage_threshold is not None :
                # Fallback to first coverage if specific one not found (original behavior)
                # This might be misleading if coverages vary significantly.
                print(f"Warning: Coverage for timestep {timestep_idx} not found or incompatible. Attempting to use coverage from first timestep as fallback for masking.")
                current_coverage_array = np.array(self.all_coverage[0])
                if len(current_coverage_array) == len(model_slice_to_plot):
                     mask = current_coverage_array < coverage_threshold
                     model_slice_to_plot = np.ma.array(model_slice_to_plot, mask=mask)
                else:
                     print("Warning: Fallback coverage array length also mismatched. No mask applied for this slice.")
            else:
                 print(f"Warning: No suitable coverage data found for masking timestep {timestep_idx}.")


        # Plot the model slice on the mesh
        # Pass any additional kwargs to pg.show (e.g., cMin, cMax, label for colorbar)
        cb = pg.show(self.mesh, data=model_slice_to_plot, ax=ax, cMap=cmap, **kwargs)
        
        # Add title with timestep information if available
        title = f"Time Slice (Index: {timestep_idx})"
        if self.timesteps is not None and timestep_idx < len(self.timesteps):
            title += f" - Time: {self.timesteps[timestep_idx]}" # Assuming self.timesteps stores actual time values
        ax.set_title(title)
        
        return fig, ax

    def create_time_lapse_animation(self, output_filename: str, # Renamed from filename
                                  cmap: str = 'viridis', coverage_threshold: Optional[float] = None,
                                  dpi: int = 100, fps: int = 2, **kwargs: Any) -> None:
        """
        Create and save an animation (e.g., MP4 video) of the time-lapse inversion results.

        Requires `ffmpeg` or another Matplotlib-supported animation writer to be installed.
        
        Args:
            output_filename (str): The filename for the output animation (e.g., 'timelapse_animation.mp4').
            cmap (str, optional): Colormap for model values. Defaults to 'viridis'.
            coverage_threshold (Optional[float], optional): Threshold for coverage masking. Defaults to None.
            dpi (int, optional): Dots Per Inch for the output animation. Defaults to 100.
            fps (int, optional): Frames Per Second for the animation. Defaults to 2.
            **kwargs (Any): Additional keyword arguments passed to `plot_time_slice` for each frame.

        Raises:
            ValueError: If `final_models` or `mesh` is missing.
            ImportError: If `matplotlib.animation` cannot be imported.
        """
        try:
            import matplotlib.animation as animation
        except ImportError:
            raise ImportError("matplotlib.animation is required to create animations. Please install or check your matplotlib setup.")
        
        if self.final_models is None or self.mesh is None:
            raise ValueError("Cannot create animation: final_models or mesh is not available.")
        
        num_timesteps = self.final_models.shape[1]
        if num_timesteps == 0:
            print("Warning: No time slices available in final_models to animate.")
            return

        fig, ax = plt.subplots(figsize=(10, 6)) # Adjust figsize as needed

        # Animation update function: called for each frame
        def update_animation_frame(frame_index: int) -> List[plt.Artist]: # Return list of artists to draw
            ax.clear() # Clear previous frame's contents
            # Use self.plot_time_slice to draw the model for the current frame_index
            # All plot_time_slice arguments need to be passed.
            # kwargs from create_time_lapse_animation are passed through.
            self.plot_time_slice(timestep_idx=frame_index, ax=ax, cmap=cmap,
                                 coverage_threshold=coverage_threshold, **kwargs)
            # FuncAnimation expects a list of artists that were updated
            # This includes images, collections (like contour lines), patches, lines, texts.
            # Safest to return all artists associated with the axes.
            return list(ax.images) + list(ax.collections) + list(ax.patches) + list(ax.lines) + list(ax.texts)


        # Create the animation object
        # blit=True optimizes drawing by only redrawing what has changed.
        # However, blit=True can be tricky with complex plots or changing axis limits/titles.
        # If issues occur, try blit=False.
        ani = animation.FuncAnimation(
            fig, update_animation_frame, frames=num_timesteps,
            blit=False, # Set to False for safety with pg.show and title changes per frame
            repeat=False # Don't repeat animation once done
        )
        
        # Save the animation
        # Requires a writer like ffmpeg. User needs to have it installed.
        try:
            ani.save(output_filename, dpi=dpi, writer='ffmpeg', fps=fps)
            print(f"Time-lapse animation saved to: {output_filename}")
        except Exception as e:
            # Common errors: ffmpeg not found, or issues with plot contents.
            print(f"Error saving animation '{output_filename}': {e}. Ensure ffmpeg is installed and in PATH.")
        finally:
            plt.close(fig) # Close the figure to free memory, especially if run in a loop.


class InversionBase:
    """
    Abstract base class for geophysical inversion methods.

    This class provides a foundational structure for specific inversion techniques
    (e.g., ERT, SRT). It manages observed data, mesh, common inversion parameters,
    and an `InversionResult` object. Subclasses must implement methods for
    setting up the inversion, running the inversion loop, computing Jacobians,
    and evaluating the objective function.
    """
    
    def __init__(self, data: pg.DataContainer, mesh: Optional[pg.Mesh] = None, **kwargs: Any):
        """
        Initialize the base inversion class.

        Args:
            data (pg.DataContainer): PyGIMLi DataContainer holding the observed geophysical data.
            mesh (Optional[pg.Mesh], optional): A PyGIMLi mesh for the inversion.
                                                If None, it's expected that subclasses will create or
                                                load a mesh in their `setup` method. Defaults to None.
            **kwargs (Any): Additional keyword arguments for configuring the inversion.
                            These are stored in `self.parameters` and can override defaults.
        """
        if not isinstance(data, pg.DataContainer):
            raise TypeError("data must be a PyGIMLi pg.DataContainer object.")
        if mesh is not None and not isinstance(mesh, pg.Mesh):
            raise TypeError("mesh must be a PyGIMLi pg.Mesh object or None.")

        self.data: pg.DataContainer = data
        self.mesh: Optional[pg.Mesh] = mesh
        self.result: InversionResult = InversionResult() # Initialize with a basic result container
                                                        # Subclasses might replace this with a more specific one (e.g. TimeLapseInversionResult)

        # Store user-provided parameters, allowing them to override defaults.
        self.parameters: Dict[str, Any] = kwargs

        # Define default parameters common to many inversion types.
        # Subclasses can extend or override these.
        default_inversion_parameters: Dict[str, Any] = {
            'lambda_reg': 10.0,          # Regularization strength parameter (lambda)
            'max_iterations': 20,        # Maximum number of inversion iterations
            'target_chi_squared': 1.0,   # Target chi-squared (χ²) value for convergence
            'convergence_tolerance': 0.01, # Relative change in chi-squared for convergence
            'model_constraints': (1e-6, 1e7), # Min and max bounds for model parameters (e.g., resistivity)
                                              # Should be appropriate for the physical property being inverted.
            # Other common params could be: error_model (relative, absolute), regularization_type, etc.
        }
        
        # Update instance parameters: user kwargs take precedence over defaults.
        # This loop ensures that if a key is in kwargs, it uses that value, otherwise it uses default.
        # A more pythonic way: self.parameters = {**default_inversion_parameters, **kwargs} (Python 3.5+)
        for key, default_value in default_inversion_parameters.items():
            if key not in self.parameters: # Only set default if not provided by user
                self.parameters[key] = default_value
    
    def setup(self) -> None:
        """
        Abstract method to set up the inversion specifics.

        This should be implemented by derived classes to prepare everything needed
        for the inversion, such as:
        - Creating or validating the mesh if not already done.
        - Initializing forward modeling operators.
        - Preparing data weighting and model regularization matrices.
        - Setting up initial models.
        """
        # Example: Mesh creation if not provided
        if self.mesh is None:
            # This part is highly dependent on the type of inversion (ERT, SRT, etc.)
            # and thus must be implemented in the specific subclass.
            raise NotImplementedError("Mesh creation or loading must be implemented in the setup() method of derived classes if mesh is not provided during initialization.")

        # Other setup tasks like initializing forward operator,
        # regularization matrices, etc., would go here or in subclass's setup.
        self.result.mesh = self.mesh # Store mesh in result object
        self.result.meta['inversion_parameters'] = self.parameters # Store used params

    def run(self) -> InversionResult:
        """
        Abstract method to run the main inversion loop.
        
        Derived classes must implement this to perform the iterative optimization process.
        The method should populate and return an `InversionResult` (or subclass) object.

        Returns:
            InversionResult: An object containing the results of the inversion.
        """
        raise NotImplementedError("The run() method must be implemented in derived classes.")
    
    def compute_jacobian(self, model: np.ndarray) -> np.ndarray:
        """
        Abstract method to compute the Jacobian matrix (sensitivity matrix).

        The Jacobian J_ij = ∂d_i / ∂m_j relates changes in model parameters (m_j)
        to changes in observed data (d_i).
        
        Args:
            model (np.ndarray): The current model parameter vector for which to compute the Jacobian.
            
        Returns:,
            np.ndarray: The computed Jacobian matrix, typically of shape (n_data, n_model_params).
        """
        raise NotImplementedError("Jacobian computation (compute_jacobian) must be implemented in derived classes.")
    
    def objective_function(self, model: np.ndarray, data_to_fit: Optional[np.ndarray] = None) -> float:
        """
        Abstract method to compute the value of the objective function.

        The objective function typically includes a data misfit term and one or more
        regularization terms: Φ(m) = Φ_d(m) + λ * Φ_m(m).
        
        Args:
            model (np.ndarray): The current model parameter vector.
            data_to_fit (Optional[np.ndarray], optional): The observed data to fit.
                                                          If None, `self.data` (from DataContainer)
                                                          is typically used. Defaults to None.
            
        Returns:
            float: The calculated value of the objective function.
        """
        raise NotImplementedError("Objective function calculation must be implemented in derived classes.")
