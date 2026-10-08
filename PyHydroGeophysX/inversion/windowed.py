"""
Windowed time-lapse ERT inversion for handling large temporal datasets.
"""
import dataclasses
import os
import sys
import tempfile
import time
from multiprocessing import Lock
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pygimli as pg
from joblib import Parallel, delayed
from pygimli.physics import ert

from .base import TimeLapseInversionResult
from .ert_inversion import (
    _adtlert_solver_name,
    _adtlert_spatial_term,
    _build_adtlert_forward,
    _resolve_ert_engine,
)
from .time_lapse import TimeLapseERTInversion, _voltage_error
from .temporal_weights import DEFAULT_LIMIT as _TEMPORAL_LIMIT
from .temporal_weights import temporal_weights


class _ADTLERTWindowProgress:
    """Forward ADTLERT's structured window events to workflow text logs.

    The ``[progress current/total]`` prefix is deliberately readable on its
    own and is also recognized by ``ProcessWorkflowWorker`` so the desktop can
    turn the same line into a determinate progress bar.
    """

    def __init__(self, log: Callable[[str], None]) -> None:
        self.log = log
        self.iteration_chi2: List[float] = []
        self.window_index = 0
        self.n_windows = 0

    def __call__(self, event: Dict[str, Any]) -> None:
        name = str(event.get("event", ""))
        if name == "windowed_start":
            self.n_windows = int(event.get("n_windows", 0))
            n_times = int(event.get("n_times", 0))
            window_size = int(event.get("window_size", 0))
            self.log(
                f"[progress 0/{self.n_windows}] ADTLERT preparing "
                f"{self.n_windows} windows ({n_times} steps, window={window_size})"
            )
        elif name == "window_start":
            self.window_index = int(event.get("window_index", 0))
            self.n_windows = int(event.get("n_windows", self.n_windows))
            start = int(event.get("start_idx", 0)) + 1
            end = int(event.get("end_idx", start - 1)) + 1
            self.log(
                f"ADTLERT window {self.window_index}/{self.n_windows} started "
                f"(time steps {start}-{end})"
            )
        elif name == "timelapse_iteration_done":
            value = event.get("chi2")
            if value is None:
                return
            chi2 = float(value)
            self.iteration_chi2.append(chi2)
            iteration = int(event.get("iteration", 0))
            maximum = int(event.get("max_iterations", 0))
            self.log(
                f"ADTLERT window {self.window_index}/{self.n_windows}, "
                f"iteration {iteration}/{maximum}: chi2 {chi2:.3f}"
            )
        elif name == "window_done":
            current = int(event.get("window_index", self.window_index))
            total = int(event.get("n_windows", self.n_windows))
            value = event.get("final_chi2")
            suffix = "" if value is None else f", chi2 {float(value):.3f}"
            self.log(
                f"[progress {current}/{total}] ADTLERT window "
                f"{current}/{total} complete{suffix}"
            )
        elif name == "windowed_prediction_start":
            n_times = int(event.get("n_times", 0))
            self.log(
                f"[progress 0/{n_times}] ADTLERT windows complete; assembling "
                f"predictions for {n_times} time steps"
            )
        elif name == "windowed_prediction_step":
            done, n_times = int(event.get("done", 0)), int(event.get("n_times", 0))
            self.log(f"[progress {done}/{n_times}] ADTLERT predicted data, "
                     f"step {done}/{n_times}")
        elif name == "windowed_done":
            total = int(event.get("n_windows", self.n_windows))
            value = event.get("final_chi2")
            suffix = "" if value is None else f", final chi2 {float(value):.3f}"
            self.log(f"ADTLERT windowed inversion complete: {total}/{total} windows{suffix}")


def _adtlert_windows(forward, observed, initial, *, window_size: int,
                     window_step: int, config, progress):
    """ADTLERT's sliding-window time-lapse inversion, one window at a time.

    The windows, the stitching - a time step's model is the geometric mean of
    the windows holding it - and the progress events are those of adtlert's
    ``invert_windowed_timelapse_log_resistivity``, each window inverted by
    ``invert_timelapse_log_resistivity`` as there. That function also keeps a
    forward-and-Jacobian cache across windows, up to window x (iterations + 2)
    x 4 dense Jacobians (84 by default, 128 at most), until the run ends. It
    reuses the few a window shares with the one before: on the DAS-1 example,
    942 readings and 6024 cells over five surveys, 4 of 54, while the cache
    grew to 2 GB and the run's peak to 9.2 GB. At a reported run's 63 MB per
    survey it fills with 5.3 GB, and that run ran out of memory.
    """
    from adtlert.inversion import (
        TimeLapseERTInversionResult,
        invert_timelapse_log_resistivity,
    )

    observed = np.asarray(observed, dtype=float)
    n_times = int(observed.shape[0])
    step = max(1, int(window_step))
    starts = sorted(set(range(0, n_times - window_size + 1, step)) | {n_times - window_size})
    data_std = np.asarray(config.data_std, dtype=float)
    progress({"event": "windowed_start", "n_windows": len(starts), "n_times": n_times,
              "window_size": int(window_size)})
    contributions: List[List[np.ndarray]] = [[] for _ in range(n_times)]
    coverage_bank, window_chi2, reports = [], [], []
    for index, start in enumerate(starts, start=1):
        end = start + int(window_size)
        progress({"event": "window_start", "window_index": index, "n_windows": len(starts),
                  "start_idx": start, "end_idx": end - 1})
        window_config = config if data_std.ndim == 0 else dataclasses.replace(
            config, data_std=np.broadcast_to(data_std, observed.shape)[start:end].copy())
        began = time.perf_counter()
        window = invert_timelapse_log_resistivity(
            forward, observed[start:end], initial[:, start:end], config=window_config)
        for local, log_model in enumerate(window.final_log_models.T):
            contributions[start + local].append(log_model)
        if window.coverage is not None:
            coverage_bank.append(np.asarray(window.coverage, dtype=float).ravel())
        final_chi2 = float(window.iteration_chi2[-1]) if window.iteration_chi2 else None
        if final_chi2 is not None:
            window_chi2.append(final_chi2)
        reports.append({"start_idx": start, "end_idx": end - 1, "final_chi2_data": final_chi2,
                        "iterations": len(window.iteration_chi2),
                        "elapsed_sec": time.perf_counter() - began})
        progress({"event": "window_done", "window_index": index, "n_windows": len(starts),
                  "final_chi2": final_chi2})
        del window

    final_log = np.column_stack([np.mean(np.column_stack(models), axis=1)
                                 for models in contributions])
    progress({"event": "windowed_prediction_start", "n_times": n_times})
    # One forward run per step: about a minute for a 420-step series, which
    # without a count reads as a hang right after the last window finished.
    every = max(1, n_times // 20)
    predicted_rows = []
    for t in range(n_times):
        predicted_rows.append(np.asarray(
            forward.forward(final_log[:, t], log_transform=True), dtype=float))
        if (t + 1) % every == 0 or t + 1 == n_times:
            progress({"event": "windowed_prediction_step", "done": t + 1, "n_times": n_times})
    predicted_log = np.vstack(predicted_rows)
    progress({"event": "windowed_done", "n_windows": len(starts),
              "final_chi2": window_chi2[-1] if window_chi2 else None})
    return TimeLapseERTInversionResult(
        final_models=np.exp(final_log),
        final_log_models=final_log,
        predicted_data=np.exp(predicted_log),
        predicted_log_data=predicted_log,
        coverage=(np.nanmedian(np.column_stack(coverage_bank), axis=1) if coverage_bank
                  else np.zeros(final_log.shape[0])),
        all_coverage=coverage_bank,
        all_chi2=np.asarray(window_chi2, dtype=float),
        iteration_chi2=window_chi2,
        window_reports=reports,
    )


class _PyHydroWindowProgress:
    """Map one PyHydro window's inner iterations onto global work units."""

    _OUTER_ITERATIONS = {"L1": 5, "L1L2": 8}

    def __init__(
        self,
        *,
        start_idx: int,
        n_windows: int,
        window_size: int,
        inversion_type: str,
        max_iterations: int,
        log: Callable[[str], None],
    ) -> None:
        self.start_idx = int(start_idx)
        self.window_index = self.start_idx + 1
        self.n_windows = int(n_windows)
        self.window_size = int(window_size)
        self.inversion_type = str(inversion_type).upper()
        self.max_iterations = int(max_iterations)
        outer = self._OUTER_ITERATIONS.get(self.inversion_type, 1)
        self.slots_per_window = max(1, outer * self.max_iterations)
        self.total_units = self.n_windows * self.slots_per_window
        self.log = log

    def start(self) -> None:
        completed = self.start_idx * self.slots_per_window
        first_step = self.start_idx + 1
        last_step = self.start_idx + self.window_size
        self.log(
            f"[progress {completed}/{self.total_units}] PyHydro window "
            f"{self.window_index}/{self.n_windows} started "
            f"(time steps {first_step}-{last_step})"
        )

    def __call__(self, event: Dict[str, Any]) -> None:
        if event.get("event") != "timelapse_iteration_done":
            return
        iteration = int(event.get("iteration", 0))
        irls_iteration = int(event.get("irls_iteration", 1))
        local_unit = (irls_iteration - 1) * self.max_iterations + iteration
        current = self.start_idx * self.slots_per_window + local_unit
        current = max(0, min(current, self.total_units))
        chi2 = float(event.get("chi2", float("nan")))
        dphi = float(event.get("dphi", float("nan")))
        irls = (
            f", IRLS {irls_iteration}/{int(event.get('irls_iterations', 1))}"
            if int(event.get("irls_iterations", 1)) > 1 else ""
        )
        self.log(
            f"[progress {current}/{self.total_units}] PyHydro window "
            f"{self.window_index}/{self.n_windows}{irls}, iteration "
            f"{iteration}/{self.max_iterations}: chi2 {chi2:.3f}, dPhi {dphi:.3g}"
        )

    def done(self, *, chi2: Optional[float], iterations: int) -> None:
        completed = self.window_index * self.slots_per_window
        suffix = "" if chi2 is None else f", chi2 {float(chi2):.3f}"
        self.log(
            f"[progress {completed}/{self.total_units}] PyHydro window "
            f"{self.window_index}/{self.n_windows} complete "
            f"({int(iterations)} iterations{suffix})"
        )


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
    
    def emit(message: str) -> None:
        if print_lock is not None:
            print_lock.acquire()
        try:
            print(message, flush=True)
        finally:
            if print_lock is not None:
                print_lock.release()

    n_windows = len(ert_files) - int(window_size) + 1
    window_progress = _PyHydroWindowProgress(
        start_idx=start_idx,
        n_windows=n_windows,
        window_size=window_size,
        inversion_type=inversion_type,
        max_iterations=int(inversion_params.get("max_iterations", 15)),
        log=emit,
    )
    window_progress.start()
    
    try:
        # Get data file paths for this window
        window_files = [os.path.join(data_dir, ert_files[i]) for i in range(start_idx, start_idx + window_size)]
        window_times = measurement_times[start_idx:start_idx + window_size]
        
        # Create TimeLapseERTInversion instance
        window_params = dict(inversion_params)
        # These weights were normalized over the entire campaign. Normalizing
        # inside a two-survey window would turn every interval weight into 1.
        if "temporal_pair_weights" in window_params:
            window_params["temporal_pair_weights"] = np.asarray(
                window_params["temporal_pair_weights"]
            )[start_idx:start_idx + window_size - 1]
        window_params["progress_callback"] = window_progress
        # The callback above is flushed after every completed iteration.  Keep
        # the legacy verbose prints off so buffered stdout does not later dump
        # a duplicate block of the same convergence history.
        window_params["verbose"] = False
        inversion = TimeLapseERTInversion(
            data_files=window_files,
            measurement_times=window_times,
            mesh=mesh,
            **window_params
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
        
        history = list(window_result.all_chi2 or [])
        final_chi2 = None
        if history:
            final_values = np.asarray(history[-1], dtype=float).ravel()
            if final_values.size:
                final_chi2 = float(final_values[0])
        window_progress.done(chi2=final_chi2, iterations=len(history))
        
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
        log: Optional[Callable[[str], None]] = None,
        **kwargs,
    ):
        """
        Initialize windowed time-lapse ERT inversion.
        
        Args:
            data_dir: Directory containing ERT data files
            ert_files: List of ERT data filenames
            measurement_times: List of measurement times
            window_size: Size of sliding window
            mesh: Mesh for inversion or path to mesh file. If omitted, one
                shared mesh is built from all surveys' electrode positions.
            engine: ``"pyhydro"`` or the optional GPU ``"adtlert"`` backend
            **kwargs: Additional parameters to pass to TimeLapseERTInversion
        """
        self.data_dir = data_dir
        self.ert_files = ert_files
        self.measurement_times = np.array(measurement_times)
        self.window_size = window_size
        self.mesh = mesh
        self.requested_engine = str(engine).lower()
        self.engine = _resolve_ert_engine(self.requested_engine)
        self.log = log or (lambda _message: None)
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

    def _shared_mesh(self):
        """Use the same spatial parameters for every window, including auto mesh."""
        if self.mesh is not None:
            return self.mesh
        paths = [os.path.join(self.data_dir, name) for name in self.ert_files]
        # Keep the first survey's measurement scheme, and add the other sensor
        # positions for meshing only. Each forward operator still uses its own
        # survey's electrodes and readings.
        combined = pg.DataContainerERT(ert.load(paths[0]))
        for path in paths[1:]:
            data = ert.load(path)
            for position in data.sensorPositions():
                combined.createSensor(position)
        combined.sortSensorsX()
        return ert.ERTManager(combined).createMesh(data=combined, quality=34)

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
        from PyHydroGeophysX.data_processing.ert_io import align_timelapse_abmn
        datasets = align_timelapse_abmn(datasets)
        reference_sensors, reference_abmn = self._survey_arrays(datasets[0])
        observed_rows = []
        error_rows = []
        relative_error = float(
            self.inversion_params.get("relativeError", 0.05)
        )
        # absoluteError is an ohm floor and absoluteUError a voltage error in
        # volts, the same two meanings TimeLapseERTInversion gives them; the
        # voltage used to be read as ohms here.
        absolute_error = float(self.inversion_params.get("absoluteError", 0.0))
        absolute_u_error = self.inversion_params.get("absoluteUError", 0.0)
        absolute_current = float(self.inversion_params.get("absoluteCurrent", 0.1))

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
                # haveData, not "in dataMap": pyGIMLi lists a zero-filled 'r'
                # for every ERT container, and |r| = 0 made the error 5e11.
                if data.haveData("r"):
                    resistance = np.abs(np.asarray(data["r"], dtype=float))
                else:
                    factors = np.abs(np.asarray(data["k"], dtype=float))
                    resistance = observed / np.maximum(factors, 1.0e-12)
                errors = relative_error + absolute_error / np.maximum(
                    resistance, 1.0e-12
                ) + _voltage_error(data, absolute_u_error, absolute_current)
            observed_rows.append(observed)
            error_rows.append(np.maximum(errors, 0.01))

        return datasets, np.vstack(observed_rows), np.vstack(error_rows)

    def _run_adtlert(self) -> TimeLapseInversionResult:
        try:
            from adtlert.inversion import InversionConfig
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

        # No time-lapse step solves a model twice: the candidate's Jacobian is
        # computed with its fields, and an accepted candidate's is reused. Kept,
        # the forward's field cache only held 1 GB on the DAS-1 example (eight
        # 130 MB entries, and their GPU copies) for no hit in 59 lookups.
        forward, result_mesh, active_ids, version = _build_adtlert_forward(
            datasets[0], mesh, field_cache_entries=0
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
        progress = _ADTLERTWindowProgress(self.log)

        config = InversionConfig(
            max_iterations=int(
                self.inversion_params.get("max_iterations", 15)
            ),
            data_std=np.log1p(relative_errors),
            data_misfit=data_misfit,
            regularization=float(
                self.inversion_params.get("lambda_val", 50.0)
            ),
            spatial_regularization=_adtlert_spatial_term(
                spatial_regularization, forward),
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
            target_chi2=(
                None
                if self.inversion_params.get("target_chi_squared") is None
                else float(self.inversion_params["target_chi_squared"])
            ),
            step_tolerance=float(
                self.inversion_params.get("convergence_tolerance", 1.0e-4)
            ),
            linearized_solver=linearized_solver,
            normal_sensitivity=True,
            include_robin_boundary_derivative=False,
            max_log_step=1.0,
            line_search=True,
            progress_callback=progress,
        )
        initial = np.vstack(
            [
                np.full(active_ids.size, np.median(row), dtype=float)
                for row in observed
            ]
        ).T
        inverted = _adtlert_windows(
            forward,
            observed,
            initial,
            window_size=int(self.window_size),
            window_step=int(self.inversion_params.get("window_step", 1)),
            config=config,
            progress=progress,
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
        result.iteration_chi2 = (
            progress.iteration_chi2
            if progress.iteration_chi2
            else [float(value) for value in inverted.iteration_chi2]
        )
        result.meta["temporal_weighting"] = {
            "mode": "uniform", "applied": False,
            "requested": str(self.inversion_params.get("temporal_weighting", "interval")),
            "note": ("The ADTLERT backend builds its own temporal operator and "
                     "constrains every adjacent pair equally; the interval "
                     "weighting applies to the PyHydro engine only."),
        }
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
            sensitivity_profile="paper",
            normal_sensitivity=True,
            include_robin_boundary_derivative=False,
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
                    "ADTLERT windowed inversion uses one shared GPU context; "
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
        # Normalize once, then slice the same pair weights for every window.
        pair_weights, temporal_report = temporal_weights(
            self.measurement_times,
            mode=str(self.inversion_params.get("temporal_weighting", "interval")),
            limit=self.inversion_params.get("temporal_weight_limit", _TEMPORAL_LIMIT),
            decay_rate=float(self.inversion_params.get("decay_rate", 0.0)),
        )
        if temporal_report.get("applied"):
            temporal_report["note"] += " Normalized over the full series and sliced into each window."
        result.meta["temporal_weighting"] = temporal_report
        window_params = dict(self.inversion_params)
        window_params["temporal_pair_weights"] = pair_weights
        window_params["_temporal_weight_report"] = temporal_report
        shared_mesh = self._shared_mesh()
        
        # Create a temporary mesh file because PyGIMLi meshes are not
        # pickleable across worker processes.
        mesh_file = None
        result_mesh_file = None
        try:
            # Process all windows
            if window_parallel:
                if max_window_workers is not None and max_window_workers < 1:
                    raise ValueError("max_window_workers must be at least 1")
                if isinstance(shared_mesh, pg.Mesh):
                    handle = tempfile.NamedTemporaryFile(suffix=".bms", delete=False)
                    mesh_file = handle.name
                    handle.close()
                    shared_mesh.save(mesh_file)
                else:
                    mesh_file = str(shared_mesh)

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
                            window_params,
                            False,
                            result_mesh_file if idx == self.window_indices[0] else None,
                        )
                        for idx in self.window_indices
                    ),
                    key=lambda x: x[0],
                )
            else:
                mesh_file = shared_mesh
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
                        window_params,
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
            if window_parallel and mesh_file and isinstance(shared_mesh, pg.Mesh):
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
