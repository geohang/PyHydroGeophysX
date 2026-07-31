"""
Windowed time-lapse ERT inversion for handling large temporal datasets.
"""
import os
import sys
import tempfile
from multiprocessing import Lock
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pygimli as pg
from joblib import Parallel, delayed

from .base import TimeLapseInversionResult
from .time_lapse import TimeLapseERTInversion


# ---------------------------------------------------------------------------
# process window
# ---------------------------------------------------------------------------
def _process_window(start_idx: int, print_lock, data_dir: str, ert_files: List[str],
                  measurement_times: List[float], window_size: int,
                  mesh: Optional[Union[pg.Mesh, str]],
                  inversion_params: Dict[str, Any],
                  include_mesh: bool = True,
                  result_mesh_path: Optional[str] = None) -> Tuple[int, Dict[str, Any]]:
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
        include_mesh: Include the non-pickleable PyGIMLi mesh in the result
        result_mesh_path: Optional filename for returning the inversion mesh
        
    Returns:
        Tuple of (window index, result dictionary)
    """
    import sys

    import pygimli as pg

    # Extract inversion type
    inversion_type = inversion_params.get('inversion_type', 'L2')
    
    # PyGIMLi meshes cannot be pickled. Parallel callers therefore pass a
    # temporary mesh filename and each worker loads its own local instance.
    if isinstance(mesh, (str, os.PathLike)):
        mesh = pg.load(str(mesh))
    
    if print_lock is not None:
        print_lock.acquire()
    try:
        print(f"\nStarting {inversion_type} inversion for window {start_idx}")
        sys.stdout.flush()
    finally:
        if print_lock is not None:
            print_lock.release()
    
    try:
        # Get data file paths for this window
        window_files = [os.path.join(data_dir, ert_files[i]) for i in range(start_idx, start_idx + window_size)]
        window_times = measurement_times[start_idx:start_idx + window_size]
        
        # Create TimeLapseERTInversion instance
        inversion = TimeLapseERTInversion(
            data_files=window_files,
            measurement_times=window_times,
            mesh=mesh,
            **inversion_params
        )
        
        # Run inversion
        window_result = inversion.run()
        if result_mesh_path is not None and window_result.mesh is not None:
            window_result.mesh.save(result_mesh_path)
        
        # Extract relevant information for the result dictionary
        result_dict = {
            'final_model': window_result.final_models,
            'coverage': window_result.all_coverage[0] if window_result.all_coverage else None,
            'all_chi2': window_result.all_chi2,
            'mesh': window_result.mesh if include_mesh else None,
            'mesh_path': result_mesh_path,
            'mesh_cells': window_result.mesh.cellCount() if window_result.mesh else None,
            'mesh_nodes': window_result.mesh.nodeCount() if window_result.mesh else None
        }
        
        if print_lock is not None:
            print_lock.acquire()
        try:
            print(f"\nWindow {start_idx} results:")
            print(f"Model shape: {window_result.final_models.shape if window_result.final_models is not None else None}")
            print(f"Coverage available: {window_result.all_coverage is not None}")
            print(f"Number of iterations: {len(window_result.all_chi2) if window_result.all_chi2 is not None else 0}")
            sys.stdout.flush()
        finally:
            if print_lock is not None:
                print_lock.release()
        
        return start_idx, result_dict
        
    except Exception as e:
        if print_lock is not None:
            print_lock.acquire()
        try:
            print(f"Error in process {start_idx}: {str(e)}")
            sys.stdout.flush()
        finally:
            if print_lock is not None:
                print_lock.release()
        raise


# ---------------------------------------------------------------------------
# Windowed Time Lapse ERTInversion
# ---------------------------------------------------------------------------
class WindowedTimeLapseERTInversion:
    """
    Class for windowed time-lapse ERT inversion to handle large temporal datasets.
    """
    
    def __init__(self, data_dir: str, ert_files: List[str], measurement_times: List[float],
                window_size: int = 3, mesh: Optional[Union[pg.Mesh, str]] = None, **kwargs):
        """
        Initialize windowed time-lapse ERT inversion.
        
        Args:
            data_dir: Directory containing ERT data files
            ert_files: List of ERT data filenames
            measurement_times: List of measurement times
            window_size: Size of sliding window
            mesh: Mesh for inversion or path to mesh file
            **kwargs: Additional parameters to pass to TimeLapseERTInversion
        """
        self.data_dir = data_dir
        self.ert_files = ert_files
        self.measurement_times = np.array(measurement_times)
        self.window_size = window_size
        self.mesh = mesh
        self.inversion_params = kwargs
        
        # Validate inputs
        if len(ert_files) != len(measurement_times):
            raise ValueError("Number of data files must match number of measurement times")
        
        if window_size < 2:
            raise ValueError("Window size must be at least 2")
        
        if window_size > len(ert_files):
            raise ValueError("Window size cannot be larger than number of data files")
        
        # Total number of time steps
        self.total_steps = len(ert_files)
        
        # Calculate window indices
        self.window_indices = list(range(0, self.total_steps - window_size + 1))
        
        # Middle index for extracting results from windows
        self.mid_idx = window_size // 2
    
    def run(self, window_parallel: bool = False, max_window_workers: Optional[int] = None) -> TimeLapseInversionResult:
        """
        Run windowed time-lapse ERT inversion.
        
        Args:
            window_parallel: Whether to process windows in parallel
            max_window_workers: Maximum number of parallel workers (None for auto)
            
        Returns:
            TimeLapseInversionResult with stitched results
        """
        # Initialize result
        result = TimeLapseInversionResult()
        result.timesteps = self.measurement_times
        
        # Create a temporary mesh file because PyGIMLi meshes are not
        # pickleable across worker processes.
        mesh_file = None
        result_mesh_file = None
        try:
            # Process all windows
            if window_parallel:
                if max_window_workers is not None and max_window_workers < 1:
                    raise ValueError("max_window_workers must be at least 1")
                if self.mesh is None:
                    raise ValueError(
                        "window_parallel=True requires an explicit mesh or mesh filename"
                    )
                if isinstance(self.mesh, pg.Mesh):
                    handle = tempfile.NamedTemporaryFile(suffix=".bms", delete=False)
                    mesh_file = handle.name
                    handle.close()
                    self.mesh.save(mesh_file)
                else:
                    mesh_file = str(self.mesh)

                handle = tempfile.NamedTemporaryFile(suffix=".bms", delete=False)
                result_mesh_file = handle.name
                handle.close()

                print(f"\nProcessing {len(self.window_indices)} windows in parallel with {max_window_workers} workers...")
                print(f"Using {self.inversion_params.get('inversion_type', 'L2')} inversion")

                # A real ERT window can use several GB of RAM. Keep automatic
                # parallelism conservative; callers with more memory can opt
                # into a larger worker count explicitly.
                n_jobs = (
                    min(2, len(self.window_indices))
                    if max_window_workers is None
                    else max_window_workers
                )
                window_results = sorted(
                    Parallel(n_jobs=n_jobs, backend="loky")(
                        delayed(_process_window)(
                            idx,
                            None,
                            self.data_dir,
                            self.ert_files,
                            self.measurement_times,
                            self.window_size,
                            mesh_file,
                            self.inversion_params,
                            False,
                            result_mesh_file if idx == self.window_indices[0] else None,
                        )
                        for idx in self.window_indices
                    ),
                    key=lambda x: x[0],
                )
            else:
                mesh_file = self.mesh
                print(f"\nProcessing {len(self.window_indices)} windows sequentially...")
                print(f"Using {self.inversion_params.get('inversion_type', 'L2')} inversion")
                
                window_results = []
                for idx in self.window_indices:
                    result_tuple = _process_window(
                        idx,
                        Lock(),
                        self.data_dir,
                        self.ert_files,
                        self.measurement_times,
                        self.window_size,
                        mesh_file,
                        self.inversion_params,
                    )
                    window_results.append(result_tuple)
            
            # Process window results
            if not window_results:
                raise ValueError("No results produced from window processing")
            
            result_by_start = dict(window_results)
            temp_mesh = window_results[0][1]['mesh']
            if temp_mesh is None and result_mesh_file is not None:
                temp_mesh = pg.load(result_mesh_file)
            all_models = []
            all_coverage = []

            # Select the most centered available window for every global time
            # step. This handles any valid window size, including one window.
            last_start = self.window_indices[-1]
            for timestep in range(self.total_steps):
                start_idx = min(max(timestep - self.mid_idx, 0), last_start)
                window_result = result_by_start[start_idx]
                if window_result['final_model'] is None:
                    raise ValueError(f"Window {start_idx} produced no model results")
                local_idx = timestep - start_idx
                all_models.append(window_result['final_model'][:, local_idx])
                if window_result['coverage'] is not None:
                    all_coverage.append(window_result['coverage'])

            all_chi2 = []
            for _, window_result in window_results:
                if window_result['all_chi2'] is not None:
                    all_chi2.extend(window_result['all_chi2'])
            
            # Convert models to 2D arrays
            all_models = [m.reshape(-1, 1) if len(m.shape) == 1 else m for m in all_models]
            
            if len(all_models) != self.total_steps:
                print(f"Warning: Number of processed models ({len(all_models)}) does not match input size ({self.total_steps})")
            
            # Store final results
            result.final_models = np.hstack(all_models)
            result.all_coverage = all_coverage
            result.all_chi2 = all_chi2
            result.mesh = temp_mesh
            
            print("\nFinal result summary:")
            print(f"Model shape: {result.final_models.shape if result.final_models is not None else None}")
            print(f"Number of coverage arrays: {len(result.all_coverage)}")
            print(f"Number of chi2 values: {len(result.all_chi2)}")
            print(f"Mesh exists: {result.mesh is not None}")
            
        finally:
            # Clean up temporary mesh file
            if window_parallel and mesh_file and isinstance(self.mesh, pg.Mesh):
                try:
                    os.unlink(mesh_file)
                except Exception:
                    pass
            if result_mesh_file:
                try:
                    os.unlink(result_mesh_file)
                except Exception:
                    pass
        
        return result
