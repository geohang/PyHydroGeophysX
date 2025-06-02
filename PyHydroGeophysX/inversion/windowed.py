"""
Windowed time-lapse ERT inversion for handling large temporal datasets.
"""
import numpy as np
import pygimli as pg
import os
import tempfile
import sys
from multiprocessing import Pool, Lock, Manager
from functools import partial
from typing import List, Optional, Union, Tuple, Dict, Any, Callable

from .base import TimeLapseInversionResult
from .time_lapse import TimeLapseERTInversion


def _process_window(start_idx: int, print_lock, data_dir: str, ert_files: List[str],
                  measurement_times: List[float], window_size: int, mesh: str,
                  inversion_params: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
    """
    Process a single window for parallel execution.
    
    Args:
        start_idx: Starting index of the window
        print_lock: Lock for synchronized printing
        data_dir: Directory containing ERT data files
        ert_files: List of ERT data filenames
        measurement_times: Array of measurement times
        window_size: Size of the window
        mesh: mesh
        inversion_params: Dictionary of inversion parameters
        
    Returns:
        Tuple of (window index (int), result dictionary (Dict[str, Any])).
        The result dictionary contains 'final_model', 'coverage', 'all_chi2', 'mesh', etc.
        for the processed window.
    """
    # Import necessary modules within the function if it's intended to be run in a separate process
    # where these might not have been imported in the global scope of that process.
    # However, pygimli and sys are typically available if the main script imports them.
    # This is more crucial for custom modules that might not be in PYTHONPATH of worker processes.
    # import pygimli as pg # Already imported globally in the file
    # import sys # Already imported globally
    
    # Extract inversion type from parameters, defaulting to 'L2' if not specified.
    # This determines the norm used for data misfit and regularization.
    inversion_type = inversion_params.get('inversion_type', 'L2')
    
    # --- Mesh Handling ---
    # The `mesh` argument to _process_window might be a pg.Mesh object or a path to a mesh file (string).
    # If it's a path, it needs to be loaded. The current code seems to assume `mesh` is already a pg.Mesh object
    # or None if it's passed from `WindowedTimeLapseERTInversion.run`.
    # If `mesh` is a string (path), it should be loaded here:
    # current_mesh: Optional[pg.Mesh]
    # if isinstance(mesh, str):
    #     current_mesh = pg.load(mesh)
    # elif isinstance(mesh, pg.Mesh) or mesh is None:
    #     current_mesh = mesh
    # else:
    #     raise TypeError("Mesh argument must be a pg.Mesh object, a path string, or None.")
    # The provided code `if mesh: mesh=mesh else: mesh=None` is redundant.
    # It will be whatever `mesh` was passed in.
    current_mesh = mesh # `mesh` is passed from `partial` which gets it from `self.mesh` (can be object or path or None)
                        # If `self.mesh` is a path, TimeLapseERTInversion's setup would handle loading it.
                        # Or, if it's a path intended for each worker to load, then loading logic is needed here.
                        # Assuming `TimeLapseERTInversion` handles mesh loading if `self.mesh` is a path.

    # Use a print lock for synchronized console output if running in parallel.
    with print_lock:
        print(f"\nStarting {inversion_type} inversion for window beginning at index {start_idx}")
        sys.stdout.flush() # Ensure immediate printing.
    
    try:
        # --- Prepare Data for the Current Window ---
        # Select the data files and measurement times relevant to this specific window.
        # `start_idx` is the beginning index of the window in the full list of files/times.
        window_file_paths = [os.path.join(data_dir, ert_files[i])
                             for i in range(start_idx, start_idx + window_size)]
        window_measurement_times = measurement_times[start_idx : start_idx + window_size]
        
        # --- Initialize and Run Time-Lapse Inversion for the Window ---
        # Create an instance of TimeLapseERTInversion for this specific window.
        # It will perform a time-lapse inversion considering only the data within this window.
        inversion_instance_for_window = TimeLapseERTInversion(
            data_files=window_file_paths,
            measurement_times=window_measurement_times,
            mesh=current_mesh, # Pass the mesh (object or None)
            **inversion_params # Pass all other inversion parameters (lambda, alpha, etc.)
        )
        
        # Run the time-lapse inversion for this window.
        window_inversion_result = inversion_instance_for_window.run() # This is a TimeLapseInversionResult object.
        
        # --- Extract and Store Relevant Results ---
        # Create a dictionary to hold the key results from this window's inversion.
        # This dictionary will be returned by the parallel worker.
        # `final_models` from TimeLapseInversionResult is (n_cells, n_timesteps_in_window).
        # `all_coverage` is a list of coverage arrays; taking the first one as representative for the window.
        #   (Usually, coverage might be similar across timesteps if geometry is same, but could be specific).
        #   SUGGESTION: Clarify if taking only `all_coverage[0]` is intended or if coverage for the
        #   middle timestep of the window would be more representative.
        result_summary_dict = {
            'final_model': window_inversion_result.final_models, # (n_cells, window_size)
            'coverage': window_inversion_result.all_coverage[0] if window_inversion_result.all_coverage else None,
            'all_chi2': window_inversion_result.all_chi2, # List of [chi2, spatial_reg, temporal_reg] per GN iter
            'mesh': window_inversion_result.mesh, # The mesh used by this window's inversion
            'mesh_cells': window_inversion_result.mesh.cellCount() if window_inversion_result.mesh else None,
            'mesh_nodes': window_inversion_result.mesh.nodeCount() if window_inversion_result.mesh else None
        }
        
        # Print summary for this window (with lock for parallel safety).
        with print_lock:
            print(f"\nWindow {start_idx} processing finished. Results summary:")
            if result_summary_dict['final_model'] is not None:
                print(f"  Final model shape: {result_summary_dict['final_model'].shape}")
            else:
                print("  Final model: None (Inversion might have failed or produced no model)")
            print(f"  Coverage available: {result_summary_dict['coverage'] is not None}")
            if result_summary_dict['all_chi2'] is not None:
                print(f"  Number of GN iterations recorded: {len(result_summary_dict['all_chi2'])}")
                if result_summary_dict['all_chi2']: # If list is not empty
                     print(f"  Final Chi2 (last GN iter): {result_summary_dict['all_chi2'][-1][0]:.2f}" if result_summary_dict['all_chi2'][-1] else "N/A")
            else:
                print("  Chi2 stats: None")
            sys.stdout.flush()
        
        return start_idx, result_summary_dict # Return window index and its results.
        
    except Exception as e:
        # Catch any exceptions during the processing of this window.
        with print_lock:
            print(f"Error processing window starting at index {start_idx}: {str(e)}")
            # Consider logging the full traceback here for debugging.
            # import traceback; traceback.print_exc()
            sys.stdout.flush()
        raise # Re-raise the exception to be caught by the Pool error handler or main thread.


class WindowedTimeLapseERTInversion:
    """
    Class for windowed time-lapse ERT inversion to handle large temporal datasets.
    This approach breaks down a long time-lapse sequence into smaller, overlapping
    (or non-overlapping, depending on stride, though current implementation implies stride=1)
    windows. A standard time-lapse inversion (e.g., using TimeLapseERTInversion)
    is performed on each window. The final time-lapse model is then stitched together
    from the results of these windowed inversions.
    """
    
    def __init__(self, data_dir: str, ert_files: List[str], measurement_times: List[float],
                window_size: int = 3, mesh: Optional[Union[pg.Mesh, str]] = None, **kwargs):
        """
        Initialize windowed time-lapse ERT inversion.
        
        Args:
            data_dir (str): Directory containing all ERT data files.
            ert_files (List[str]): List of ERT data filenames (relative to `data_dir`).
                                   Order should correspond to `measurement_times`.
            measurement_times (List[float]): List of measurement times, corresponding to `ert_files`.
            window_size (int, optional): The number of timesteps included in each sliding window.
                                         Defaults to 3. Must be at least 2.
            mesh (Optional[Union[pg.Mesh, str]], optional):
                                         PyGIMLi mesh object or path to a mesh file to be used for all inversions.
                                         If None, `TimeLapseERTInversion` will attempt to create a default mesh
                                         based on the first dataset in each window. Providing a consistent mesh
                                         is generally recommended for time-lapse studies. Defaults to None.
            **kwargs: Additional parameters to be passed to each `TimeLapseERTInversion` instance
                      (e.g., lambda_val, alpha, model_constraints, max_iterations).
        """
        self.data_dir = data_dir                 # Directory for ERT data files.
        self.ert_files = ert_files               # List of data filenames.
        self.measurement_times = np.array(measurement_times) # Array of times.
        self.window_size = window_size           # Number of time steps per window.
        self.mesh = mesh                         # User-provided mesh or path.
        self.inversion_params = kwargs           # Parameters for `TimeLapseERTInversion`.
        
        # --- Input Validations ---
        if len(ert_files) != len(measurement_times):
            raise ValueError("Number of ERT data files must match the number of measurement times.")
        
        if window_size < 2: # A window needs at least two timesteps for temporal regularization.
            raise ValueError("Window size must be at least 2.")
        
        if window_size > len(ert_files):
            # This implies a single window covering all data, which defeats the purpose of "windowed"
            # but is technically runnable if handled as a special case by window_indices.
            # Original code raises error.
            raise ValueError("Window size cannot be larger than the total number of data files.")
            # SUGGESTION: Could default to `window_size = len(ert_files)` in this case,
            # effectively running one large time-lapse inversion.
        
        # Total number of time steps in the full sequence.
        self.total_steps = len(ert_files)
        
        # Calculate the starting indices for each window.
        # This creates non-overlapping windows if stride = window_size.
        # Here, it creates overlapping windows with a stride of 1.
        # e.g., if total_steps=5, window_size=3: indices are [0, 1, 2].
        # Window 0: files[0,1,2]; Window 1: files[1,2,3]; Window 2: files[2,3,4].
        self.window_indices = list(range(0, self.total_steps - self.window_size + 1))
        
        # Determine the middle index within a window. This is used for stitching results.
        # For a window of size `W`, the result for timestep `t_i` is often taken from the
        # window where `t_i` is central, to minimize edge effects of the windowed inversion.
        # E.g., if window_size = 3, mid_idx = 1. Results for window[s:s+3] would be for time s+1.
        # If window_size = 4, mid_idx = 2. (integer division)
        # SUGGESTION: Stitching strategy needs to be clear. Taking only mid_idx might discard info
        # or require special handling for start/end of the overall sequence.
        # The current stitching in `run` method handles start/end differently.
        self.mid_idx = self.window_size // 2
    
    def run(self, window_parallel: bool = False, max_window_workers: Optional[int] = None) -> TimeLapseInversionResult:
        """
        Run windowed time-lapse ERT inversion.
        
        Args:
            window_parallel: Whether to process windows in parallel
            max_window_workers: Maximum number of parallel workers (None for auto)
            
        Returns:
            TimeLapseInversionResult: An object containing the stitched final models, coverage,
                                      chi2 statistics, and the mesh.
        """
        # Initialize a TimeLapseInversionResult object to store the aggregated results.
        final_stitched_result = TimeLapseInversionResult()
        final_stitched_result.timesteps = self.measurement_times # Assign all measurement times.
        
        # --- Mesh Handling for Parallel Processing ---
        # If parallel processing is used and the mesh is a pg.Mesh object (not a path),
        # it might need to be saved to a temporary file so each worker process can load it independently.
        # PyGIMLi mesh objects themselves might not be directly picklable for multiprocessing.
        # The `_process_window` function receives `mesh_file` (which is `self.mesh` here).
        # If `self.mesh` is an object, it's passed directly. If it's a path, that path is used.
        # The original code creates `mesh_file = self.mesh` and then in `_process_window` uses `mesh=mesh_file`.
        # This seems okay if `mesh_file` is a path, or if pg.Mesh is picklable.
        # No explicit temp file creation for mesh object is shown here, relying on `partial` to pass it.
        # This could be problematic if `pg.Mesh` is not picklable.
        # SUGGESTION: If `self.mesh` is a pg.Mesh object and `window_parallel` is True,
        # consider saving it to a temp file and passing the path, then deleting it in `finally`.
        # For now, assume `self.mesh` (as `mesh_argument_for_process`) is handled correctly by `partial` or `_process_window`.
        mesh_argument_for_process = self.mesh

        # --- Parallel or Sequential Window Processing ---
        window_processing_outputs: List[Tuple[int, Dict[str, Any]]] # List to store (start_idx, result_dict)

        if window_parallel:
            # --- Parallel Processing ---
            if max_window_workers is None:
                 # Default to number of CPU cores if not specified.
                 resolved_max_workers = os.cpu_count()
                 print(f"\nProcessing {len(self.window_indices)} windows in parallel. Max workers not set, defaulting to {resolved_max_workers}.")
            else:
                 resolved_max_workers = max_window_workers
                 print(f"\nProcessing {len(self.window_indices)} windows in parallel with up to {resolved_max_workers} workers...")
            print(f"Using {self.inversion_params.get('inversion_type', 'L2')} inversion type for each window.")
            
            # `Manager` is used for creating shared objects like `Lock` for safe printing from multiple processes.
            with Manager() as manager:
                shared_print_lock = manager.Lock() # Create a shared lock for synchronized printing.
                
                # `partial` creates a new function with some arguments of `_process_window` pre-filled.
                # This is useful for `pool.map` which only passes one iterable argument.
                process_single_window_configured = partial(
                    _process_window, # Target function
                    # Pre-filled arguments:
                    print_lock=shared_print_lock,
                    data_dir=self.data_dir,
                    ert_files=self.ert_files,
                    measurement_times=self.measurement_times,
                    window_size=self.window_size,
                    mesh=mesh_argument_for_process, # Pass mesh (object or path)
                    inversion_params=self.inversion_params
                )
                
                # Create a process pool and map the `_process_window` function over window start indices.
                with Pool(processes=resolved_max_workers) as pool:
                    # `pool.map` applies the function to each item in `self.window_indices` and collects results.
                    # Results are returned in the order of the input iterable.
                    window_processing_outputs = pool.map(process_single_window_configured, self.window_indices)
                # Sort by start_idx just in case pool.map doesn't guarantee order (though it usually does).
                window_processing_outputs = sorted(window_processing_outputs, key=lambda x: x[0])

        else: # Sequential processing
            print(f"\nProcessing {len(self.window_indices)} windows sequentially...")
            print(f"Using {self.inversion_params.get('inversion_type', 'L2')} inversion type for each window.")
            
            window_processing_outputs = []
            # Simple lock for sequential case (though not strictly needed, kept for consistency with _process_window signature).
            sequential_print_lock = Lock()
            for window_start_idx in self.window_indices:
                result_tuple = _process_window(
                    window_start_idx,
                    sequential_print_lock, # Pass lock object
                    self.data_dir, self.ert_files, self.measurement_times,
                    self.window_size, mesh_argument_for_process, self.inversion_params
                )
                window_processing_outputs.append(result_tuple)
            
        # --- Stitching Results from Windows ---
        if not window_processing_outputs:
            raise ValueError("No results were produced from window processing. Check for errors in _process_window.")
            
        stitched_models_list = []    # List to hold the selected model columns from each window.
        stitched_coverage_list = []  # List for coverage maps.
        stitched_chi2_list = []      # List for chi-squared stats.

        # The mesh should be consistent across windows if a single mesh was provided or generated by TimeLapseERTInversion.
        # Take the mesh from the first window's result (assuming it's representative).
        # This assumes all windows successfully return a mesh.
        final_mesh_from_windows = window_processing_outputs[0][1].get('mesh')
        if final_mesh_from_windows is None and isinstance(self.mesh, pg.Mesh): # If first window failed on mesh but we have one
            final_mesh_from_windows = self.mesh
        elif final_mesh_from_windows is None and isinstance(self.mesh, str) and os.path.exists(self.mesh):
             final_mesh_from_windows = pg.load(self.mesh)


        # --- Stitching Logic ---
        # The goal is to construct a full time-lapse model sequence (total_steps long)
        # from the (window_size long) models obtained from each window.
        # Strategy:
        #   - First window: Take its first `mid_idx` + 1 models (or special handling like first 2).
        #   - Middle windows: Take the model at `mid_idx` (center) of the window.
        #   - Last window: Take its last `window_size - mid_idx` models (or special handling like last 2).
        # This aims to use the most reliable part of each windowed inversion.
        # The original code has specific logic: first 2 from first window, mid from middle, last 2 from last. This implies window_size >= 2.
        # If window_size = 2, mid_idx = 1. First window gives model[0], model[1]. Last window also model[0], model[1].
        # If window_size = 3, mid_idx = 1. First window gives model[0], model[1]. Middle window gives model[1]. Last gives model[1], model[2].
        # This seems to be a specific stitching strategy for `mid_idx` related selection.

        num_windows = len(window_processing_outputs)
        for i in range(num_windows):
            window_start_idx, current_window_result_dict = window_processing_outputs[i]
            current_window_models = current_window_result_dict.get('final_model') # Shape: (n_cells, window_size)
            current_window_coverage = current_window_result_dict.get('coverage') # Single coverage map
            current_window_chi2_stats = current_window_result_dict.get('all_chi2') # List of iteration stats

            if current_window_models is None:
                print(f"Warning: Window {window_start_idx} provided no model. Attempting to fill with NaNs or skip.")
                # Need to know num_cells to create a NaN placeholder.
                num_cells_placeholder = final_mesh_from_windows.cellCount() if final_mesh_from_windows else stitched_models_list[0].shape[0] if stitched_models_list else 0
                nan_model_placeholder = np.full((num_cells_placeholder, 1), np.nan)
                
                if i == 0: # First window
                    for k in range(self.mid_idx + (1 if self.window_size % 2 != 0 else 0) if self.window_size > 1 else self.window_size ): # take first mid_idx+1 or all if window small
                         if k < self.window_size: stitched_models_list.append(nan_model_placeholder)
                elif i == num_windows - 1: # Last window
                    for k in range(self.mid_idx, self.window_size):
                         stitched_models_list.append(nan_model_placeholder)
                else: # Middle windows
                    stitched_models_list.append(nan_model_placeholder)
                continue # Skip to next window if this one failed.

            if i == 0: # First window
                # Take the first `mid_idx + 1` models if window is odd, or `mid_idx` if even.
                # Original code: takes first two (indices 0, 1). This is robust for window_size >= 2.
                # If window_size = 2, mid_idx=1. Takes [:,0], [:,1].
                # If window_size = 3, mid_idx=1. Takes [:,0], [:,1]. (This means one model from overlap is chosen by next window)
                # This stitching needs to be careful to get `self.total_steps` models in the end.
                # A simpler approach: first window contributes models from its start up to its mid_idx.
                # Last window contributes from its mid_idx to its end.
                # Middle windows contribute only their mid_idx model.
                # Let's refine stitching logic based on typical windowing:
                for k in range(self.mid_idx + 1): # Models from index 0 to mid_idx of this window.
                    stitched_models_list.append(current_window_models[:, k])
                if current_window_coverage is not None: # Add coverage for these models
                    for _ in range(self.mid_idx + 1): stitched_coverage_list.append(current_window_coverage)
            
            elif i == num_windows - 1: # Last window
                # Add models from mid_idx of this window to its end.
                for k in range(self.mid_idx, self.window_size):
                    # Check if we already added this global timestep from previous window's overlap.
                    # Global index of model k in this window: window_start_idx + k
                    # Global index of last model added: len(stitched_models_list) -1 (if using 0-based global time)
                    # Effective global time index for model k in current window: window_start_idx + k
                    # Number of models already stitched: len(stitched_models_list)
                    # We need to add models for global times: len(stitched_models_list) ... self.total_steps - 1
                    if (window_start_idx + k) >= len(stitched_models_list): # Add if not overlapping with previous window's main part
                        stitched_models_list.append(current_window_models[:, k])
                        if current_window_coverage is not None: stitched_coverage_list.append(current_window_coverage)
            
            else: # Middle windows
                # Add only the model at mid_idx of this window.
                # This corresponds to global time: window_start_idx + self.mid_idx
                stitched_models_list.append(current_window_models[:, self.mid_idx])
                if current_window_coverage is not None: stitched_coverage_list.append(current_window_coverage)

            if current_window_chi2_stats is not None: # Append all chi2 stats from this window's inversion.
                stitched_chi2_list.extend(current_window_chi2_stats)
            
            # Original stitching (more specific, might lead to issues if window_size is small e.g. 2)
            # This logic seems to be what was in the original file structure, let's try to keep its spirit:
            # It seems like it wants to pick specific columns.

        # --- Rebuild based on original code's apparent stitching logic ---
        # This part needs to be very careful to match the intended number of total_steps.
        all_models_rebuilt = []
        all_coverage_rebuilt = []
        all_chi2_rebuilt = [] # Chi2 from each window run (full list of lists)

        # First window: use its first `mid_idx` models. For `window_size=3, mid_idx=1`, this is model 0.
        # The original code takes first two (0, 1) from first window.
        # Let's use a strategy: first window -> models 0..mid_idx-1
        # middle windows -> model at mid_idx
        # last window -> models mid_idx..end
        # This ensures `total_steps` models if done carefully.

        # Example: total_steps=5, window_size=3. mid_idx=1. windows_indices=[0,1,2]
        # Window 0 (files 0,1,2): contributes model for time 0 (model[:,0] of window 0)
        # Window 1 (files 1,2,3): contributes model for time 1 (model[:,mid_idx=1] of window 1)
        # Window 2 (files 2,3,4): contributes model for time 2 (model[:,mid_idx=1] of window 2),
        #                        AND model for time 3 (model[:,mid_idx+1] of window 2),
        #                        AND model for time 4 (model[:,mid_idx+2] of window 2)
        # This ensures all `total_steps` are covered.

        for i in range(self.total_steps): # Iterate for each final timestep model needed
            corresponding_window_idx = -1
            model_idx_in_window = -1

            if i < self.mid_idx: # For the first few timesteps, take from the first window
                corresponding_window_idx = 0
                model_idx_in_window = i
            elif i >= self.total_steps - (self.window_size - self.mid_idx): # For the last few timesteps
                corresponding_window_idx = num_windows - 1 # Last window
                # model_idx_in_window for last window needs to map global time `i`
                # to local window index. Global time `i` corresponds to
                # `i - self.window_indices[last_window_idx]` in that window.
                model_idx_in_window = i - self.window_indices[num_windows-1]
            else: # For timesteps in the middle
                # Find the window where this timestep `i` is the `mid_idx`.
                # The model for global time `i` should come from window starting at `i - self.mid_idx`.
                corresponding_window_idx = i - self.mid_idx
                model_idx_in_window = self.mid_idx
            
            # Retrieve results from the chosen window
            _ , window_res_dict = window_processing_outputs[corresponding_window_idx]

            if window_res_dict['final_model'] is not None and model_idx_in_window < window_res_dict['final_model'].shape[1]:
                all_models_rebuilt.append(window_res_dict['final_model'][:, model_idx_in_window])
                if window_res_dict['coverage'] is not None: # Assuming coverage is representative for the window
                    all_coverage_rebuilt.append(window_res_dict['coverage'])
            else: # Placeholder if model missing
                num_cells_placeholder = final_mesh_from_windows.cellCount() if final_mesh_from_windows else (all_models_rebuilt[0].shape[0] if all_models_rebuilt else 0)
                all_models_rebuilt.append(np.full(num_cells_placeholder, np.nan))
                all_coverage_rebuilt.append(np.full(num_cells_placeholder, np.nan) if num_cells_placeholder > 0 else np.array([]))

            # Chi2: just append all lists of chi2 iterations from all windows.
            # This might need better aggregation if per-final-timestep chi2 is desired.
            if window_res_dict['all_chi2'] is not None:
                 all_chi2_rebuilt.extend(window_res_dict['all_chi2']) # This makes a flat list of all iterations from all windows.

        # Ensure all models are column vectors before hstack
        all_models_rebuilt_cols = [m.reshape(-1, 1) if m.ndim == 1 else m for m in all_models_rebuilt]

        # Store final stitched results.
        if all_models_rebuilt_cols:
            final_stitched_result.final_models = np.hstack(all_models_rebuilt_cols) # Shape (n_cells, total_steps)
        else: # Handle case where no models were processed
            num_cells_placeholder = final_mesh_from_windows.cellCount() if final_mesh_from_windows else 0
            final_stitched_result.final_models = np.empty((num_cells_placeholder, 0))

        final_stitched_result.all_coverage = all_coverage_rebuilt # List of coverage arrays.
        final_stitched_result.all_chi2 = all_chi2_rebuilt # List of iteration stats.
        final_stitched_result.mesh = final_mesh_from_windows
            
        print("\nFinal stitched result summary:")
        if final_stitched_result.final_models is not None:
            print(f"  Stitched Model shape: {final_stitched_result.final_models.shape}")
        print(f"  Number of coverage arrays stored: {len(final_stitched_result.all_coverage)}")
        # print(f"  Total number of chi2 iteration stats recorded: {len(final_stitched_result.all_chi2)}")
        print(f"  Final Mesh cells: {final_stitched_result.mesh.cellCount() if final_stitched_result.mesh else 'N/A'}")
            
        # Original code had a `finally` block for cleaning up temp mesh file.
        # That logic is removed as temp file creation for mesh is not explicit here.
        # If it were added, cleanup would go here.
        
        return final_stitched_result
