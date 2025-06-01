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
        Tuple of (window index, result dictionary)
    """
    import pygimli as pg
    import sys
    
    # Extract inversion type
    inversion_type = inversion_params.get('inversion_type', 'L2')
    
    # Load mesh for each process
    if mesh:
        mesh = mesh
    else:
        mesh = None
    
    with print_lock:
        print(f"\nStarting {inversion_type} inversion for window {start_idx}")
        sys.stdout.flush()
    
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
        
        # Extract relevant information for the result dictionary
        result_dict = {
            'final_model': window_result.final_models,
            'coverage': window_result.all_coverage[0] if window_result.all_coverage else None,
            'all_chi2': window_result.all_chi2,
            'mesh': window_result.mesh,
            'mesh_cells': window_result.mesh.cellCount() if window_result.mesh else None,
            'mesh_nodes': window_result.mesh.nodeCount() if window_result.mesh else None
        }
        
        with print_lock:
            print(f"\nWindow {start_idx} results:")
            print(f"Model shape: {window_result.final_models.shape if window_result.final_models is not None else None}")
            print(f"Coverage available: {window_result.all_coverage is not None}")
            print(f"Number of iterations: {len(window_result.all_chi2) if window_result.all_chi2 is not None else 0}")
            sys.stdout.flush()
        
        return start_idx, result_dict
        
    except Exception as e:
        with print_lock:
            print(f"Error in process {start_idx}: {str(e)}")
            sys.stdout.flush()
        raise


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
        
        # Create temporary mesh file for parallel processing
        mesh_file = None
        try:
            mesh_file = self.mesh
            
            # Process all windows
            if window_parallel:
                print(f"\nProcessing {len(self.window_indices)} windows in parallel with {max_window_workers} workers...")
                print(f"Using {self.inversion_params.get('inversion_type', 'L2')} inversion")
                
                with Manager() as manager:
                    print_lock = manager.Lock()
                    
                    process_window_partial = partial(
                        _process_window,
                        print_lock=print_lock,
                        data_dir=self.data_dir,
                        ert_files=self.ert_files,
                        measurement_times=self.measurement_times,
                        window_size=self.window_size,
                        mesh_file=mesh_file,
                        inversion_params=self.inversion_params
                    )
                    
                    with Pool(processes=max_window_workers) as pool:
                        window_results = sorted(
                            pool.map(process_window_partial, self.window_indices),
                            key=lambda x: x[0]
                        )
            else:
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
                        self.inversion_params
                    )
                    window_results.append(result_tuple)
            
            # Process window results
            if not window_results:
                raise ValueError("No results produced from window processing")
            
            all_models = []
            all_coverage = []
            all_chi2 = []
            
            # Process first window
            _, first_result = window_results[0]
            if first_result['final_model'] is None:
                raise ValueError("First window produced no model results")
            
            # Store first two timesteps from first window
            all_models.append(first_result['final_model'][:, 0])
            all_models.append(first_result['final_model'][:, 1])
            temp_mesh = first_result['mesh']
            if first_result['all_chi2'] is not None:
                all_chi2.extend(first_result['all_chi2'])
            
            if first_result['coverage'] is not None:
                all_coverage.extend([first_result['coverage']] * 2)
            
            # Process middle windows
            for i, (win_idx, window_result) in enumerate(window_results[1:-1], 1):
                if window_result['final_model'] is None:
                    print(f"Warning: Window {win_idx} produced no model results. Using previous window.")
                    continue
                    
                # Extract middle timestep from each window
                all_models.append(window_result['final_model'][:, self.mid_idx])
                
                if window_result['all_chi2'] is not None:
                    all_chi2.extend(window_result['all_chi2'])
                    
                if window_result['coverage'] is not None:
                    all_coverage.append(window_result['coverage'])
            
            # Process last window
            _, last_result = window_results[-1]
            if last_result['final_model'] is not None:
                all_models.append(last_result['final_model'][:, -2])
                all_models.append(last_result['final_model'][:, -1])
                
                if last_result['all_chi2'] is not None:
                    all_chi2.extend(last_result['all_chi2'])
                    
                if last_result['coverage'] is not None:
                    all_coverage.extend([last_result['coverage']] * 2)
            
            # Convert models to 2D arrays
            all_models = [m.reshape(-1, 1) if len(m.shape) == 1 else m for m in all_models]
            
            if len(all_models) != self.total_steps:
                print(f"Warning: Number of processed models ({len(all_models)}) does not match input size ({self.total_steps})")
            
            # Store final results
            result.final_models = np.hstack(all_models)
            result.all_coverage = all_coverage
            result.all_chi2 = all_chi2
            result.mesh = temp_mesh if isinstance(self.mesh, pg.Mesh) else None
            
            print("\nFinal result summary:")
            print(f"Model shape: {result.final_models.shape if result.final_models is not None else None}")
            print(f"Number of coverage arrays: {len(result.all_coverage)}")
            print(f"Number of chi2 values: {len(result.all_chi2)}")
            print(f"Mesh exists: {result.mesh is not None}")
            
        finally:
            # Clean up temporary mesh file
            if window_parallel and mesh_file and not isinstance(self.mesh, str):
                try:
                    os.unlink(mesh_file)
                except:
                    pass
        
        return result
