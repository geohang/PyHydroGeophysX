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
from pygimli.physics import ert

from .base import TimeLapseInversionResult
from .ert_inversion import _adtlert_solver_name, _build_adtlert_forward
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
    
    def __init__(
        self,
        data_dir: str,
        ert_files: List[str],
        measurement_times: List[float],
        window_size: int = 3,
        mesh: Optional[Union[pg.Mesh, str]] = None,
        engine: str = "pyhydro",
        **kwargs,
    ):
        """
        Initialize windowed time-lapse ERT inversion.
        
        Args:
            data_dir: Directory containing ERT data files
            ert_files: List of ERT data filenames
            measurement_times: List of measurement times
            window_size: Size of sliding window
            mesh: Mesh for inversion or path to mesh file
            engine: ``"pyhydro"`` or the optional GPU ``"adtlert"`` backend
            **kwargs: Additional parameters to pass to TimeLapseERTInversion
        """
        self.data_dir = data_dir
        self.ert_files = ert_files
        self.measurement_times = np.array(measurement_times)
        self.window_size = window_size
        self.mesh = mesh
        self.engine = str(engine).lower()
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

    @staticmethod
    def _survey_arrays(data) -> Tuple[np.ndarray, np.ndarray]:
        sensors = np.asarray(
            [[float(pos[0]), float(pos[1])] for pos in data.sensorPositions()],
            dtype=float,
        )
        abmn = np.column_stack(
            [
                np.asarray(data[key], dtype=np.int32)
                for key in ("a", "b", "m", "n")
            ]
        )
        return sensors, abmn

    def _load_adtlert_series(self):
        paths = [os.path.join(self.data_dir, name) for name in self.ert_files]
        datasets = [ert.load(path) for path in paths]
        reference_sensors, reference_abmn = self._survey_arrays(datasets[0])
        observed_rows = []
        error_rows = []
        relative_error = float(
            self.inversion_params.get("relativeError", 0.05)
        )
        absolute_error = float(
            self.inversion_params.get(
                "absoluteError",
                self.inversion_params.get("absoluteUError", 0.0),
            )
        )

        for index, data in enumerate(datasets):
            sensors, abmn = self._survey_arrays(data)
            if sensors.shape != reference_sensors.shape or not np.allclose(
                sensors, reference_sensors, rtol=0.0, atol=1.0e-8
            ):
                raise ValueError(
                    "ADTLERT windowed inversion requires identical electrode "
                    f"positions at every timestep; timestep {index} differs"
                )
            if not np.array_equal(abmn, reference_abmn):
                raise ValueError(
                    "ADTLERT windowed inversion requires identical ABMN "
                    "ordering "
                    f"at every timestep; timestep {index} differs"
                )

            observed = np.asarray(data["rhoa"], dtype=float)
            if observed.shape != (reference_abmn.shape[0],):
                raise ValueError(
                    f"timestep {index} has {observed.size} apparent "
                    "resistivities; "
                    f"expected {reference_abmn.shape[0]}"
                )
            if not np.all(np.isfinite(observed)) or np.any(observed <= 0.0):
                raise ValueError(
                    f"timestep {index} contains invalid apparent resistivity "
                    "values"
                )

            errors = np.asarray(data["err"], dtype=float)
            invalid_errors = (
                errors.shape != observed.shape
                or not np.all(np.isfinite(errors))
                or np.any(errors <= 0.0)
            )
            if invalid_errors:
                if "r" in data.dataMap():
                    resistance = np.abs(np.asarray(data["r"], dtype=float))
                else:
                    factors = np.abs(np.asarray(data["k"], dtype=float))
                    resistance = observed / np.maximum(factors, 1.0e-12)
                errors = relative_error + absolute_error / np.maximum(
                    resistance, 1.0e-12
                )
            observed_rows.append(observed)
            error_rows.append(np.clip(errors, 0.01, 0.50))

        return datasets, np.vstack(observed_rows), np.vstack(error_rows)

    def _run_adtlert(self) -> TimeLapseInversionResult:
        try:
            from adtlert.inversion import (
                InversionConfig,
                invert_windowed_timelapse_log_resistivity,
            )
        except ImportError as exc:
            from PyHydroGeophysX._internal.optional_dependencies import (
                BackendUnavailable,
            )

            raise BackendUnavailable(
                "The ADTLERT ERT backend is unavailable. Install it with "
                "`pip install \"pyhydrogeophysx[adtlert]\"`."
            ) from exc

        datasets, observed, relative_errors = self._load_adtlert_series()
        if isinstance(self.mesh, (str, os.PathLike)):
            mesh = pg.load(str(self.mesh))
        elif self.mesh is None:
            mesh = ert.ERTManager(datasets[0]).createMesh(
                data=datasets[0],
                quality=float(self.inversion_params.get("mesh_quality", 34.0)),
            )
        else:
            mesh = self.mesh

        forward, result_mesh, active_ids, version = _build_adtlert_forward(
            datasets[0], mesh
        )
        inversion_type = str(
            self.inversion_params.get("inversion_type", "L2")
        ).upper()
        norm_config = {
            "L2": ("weighted_log_l2", "first_order", "temporal_smoothness"),
            "L1": ("weighted_log_l1", "first_order_l1", "first_order_l1"),
            "L1L2": ("weighted_log_huber", "first_order_l1", "first_order_l1"),
        }
        try:
            norm_values = norm_config[inversion_type]
            data_misfit, spatial_regularization, temporal_regularization = (
                norm_values
            )
        except KeyError as exc:
            raise ValueError(
                "ADTLERT windowed inversion_type must be L2, L1 or L1L2"
            ) from exc

        linearized_solver = _adtlert_solver_name(
            self.inversion_params.get("method", "cgls"),
            prefer_gpu=True,
        )
        config = InversionConfig(
            max_iterations=int(
                self.inversion_params.get("max_iterations", 15)
            ),
            data_std=np.log1p(relative_errors),
            data_misfit=data_misfit,
            regularization=float(
                self.inversion_params.get("lambda_val", 50.0)
            ),
            spatial_regularization=spatial_regularization,
            temporal_regularization=float(
                self.inversion_params.get("alpha", 10.0)
            ),
            temporal_regularization_mode="separate",
            temporal_regularization_type=temporal_regularization,
            model_bounds=tuple(
                float(value)
                for value in self.inversion_params.get(
                    "model_constraints", (1.0e-2, 1.0e5)
                )
            ),
            target_chi2=float(
                self.inversion_params.get("target_chi_squared", 1.0)
            ),
            step_tolerance=float(
                self.inversion_params.get("convergence_tolerance", 0.01)
            ),
            linearized_solver=linearized_solver,
        )
        initial = np.vstack(
            [
                np.full(active_ids.size, np.median(row), dtype=float)
                for row in observed
            ]
        ).T
        inverted = invert_windowed_timelapse_log_resistivity(
            forward,
            observed,
            initial,
            window_size=int(self.window_size),
            window_step=int(self.inversion_params.get("window_step", 1)),
            config=config,
        )

        result = TimeLapseInversionResult()
        result.final_models = np.asarray(inverted.final_models, dtype=float)
        result.predicted_data = np.asarray(
            inverted.predicted_data, dtype=float
        )
        result.coverage = np.asarray(inverted.coverage, dtype=float)
        result.timesteps = self.measurement_times.copy()
        result.mesh = result_mesh
        coverage_by_time: List[List[np.ndarray]] = [
            [] for _ in range(self.total_steps)
        ]
        for report, window_coverage in zip(
            inverted.window_reports, inverted.all_coverage
        ):
            coverage_values = np.asarray(window_coverage, dtype=float)
            for time_index in range(
                int(report["start_idx"]), int(report["end_idx"]) + 1
            ):
                coverage_by_time[time_index].append(coverage_values)
        result.all_coverage = [
            (
                np.nanmedian(np.vstack(values), axis=0)
                if values
                else result.coverage.copy()
            )
            for values in coverage_by_time
        ]
        result.all_chi2 = [float(value) for value in inverted.all_chi2]
        result.iteration_chi2 = [
            float(value) for value in inverted.iteration_chi2
        ]
        result.meta.update(
            backend="adtlert",
            backend_version=version,
            chi2=(
                result.iteration_chi2[-1]
                if result.iteration_chi2
                else float("nan")
            ),
            iterations=len(result.iteration_chi2),
            lambda_val=float(config.regularization),
            window_size=int(self.window_size),
            window_step=int(self.inversion_params.get("window_step", 1)),
            linearized_solver=linearized_solver,
            window_reports=list(inverted.window_reports),
        )
        return result
    
    def run(
        self,
        window_parallel: bool = False,
        max_window_workers: Optional[int] = None,
    ) -> TimeLapseInversionResult:
        """
        Run windowed time-lapse ERT inversion.
        
        Args:
            window_parallel: Whether to process windows in parallel
            max_window_workers: Maximum number of parallel workers (None for auto)
            
        Returns:
            TimeLapseInversionResult with stitched results
        """
        if self.engine == "adtlert":
            if window_parallel:
                raise ValueError(
                    "ADTLERT windowed inversion uses one shared GPU/cuDSS context; "
                    "window_parallel=True would duplicate GPU memory"
                )
            return self._run_adtlert()
        if self.engine != "pyhydro":
            raise ValueError(
                "WindowedTimeLapseERTInversion engine must be 'pyhydro' or 'adtlert'"
            )

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
