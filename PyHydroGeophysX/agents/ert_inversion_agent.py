"""
ERT Inversion Agent

Specialized agent for performing ERT inversion with optional structural constraints.
"""

import os
from typing import Any, Dict, Optional

import numpy as np

from ._method import IMPLEMENTED_SCHEME, SCHEME_LABEL
from .base_agent import BaseAgent


# ---------------------------------------------------------------------------
# ERTInversion Agent
# ---------------------------------------------------------------------------
class ERTInversionAgent(BaseAgent):
    """
    Agent specialized in ERT inversion.
    
    Uses PyHydroGeophysX inversion module to perform resistivity inversion
    with optional structural constraints from seismic data.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None,
                 llm_provider: str = "openai"):
        """Initialize ERT Inversion Agent."""
        super().__init__("ert_inversion", api_key, model, llm_provider)
        self.system_message = """You are an expert in electrical resistivity tomography (ERT) 
inversion. Your role is to configure and execute ERT inversions, select appropriate 
regularization parameters, and interpret inversion results. You understand smoothness 
constraints, structural constraints, and convergence criteria."""
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform ERT inversion (standard or time-lapse).
        
        Args:
            input_data: Dictionary containing:
                - ert_data: Loaded ERT data (for standard inversion)
                - inversion_mode: 'standard' or 'time-lapse'
                - time_lapse_data: List of ERT datasets (for time-lapse)
                - time_lapse_method: recorded only; the solver implements one
                  scheme (see PyHydroGeophysX.agents._method)
                - temporal_regularization: Temporal smoothing weight (for time-lapse)
                - inversion_params: Inversion parameters (lambda, max_iter, etc.)
                - use_structure_constraint: Whether to use seismic structure (default: False)
                - seismic_structure: Optional seismic structure data
                - output_dir: Directory for saving results
                
        Returns:
            Dictionary containing inversion results
        """
        inversion_mode = input_data.get('inversion_mode', 'standard')
        
        if inversion_mode == 'time-lapse':
            return self._execute_time_lapse(input_data)
        else:
            return self._execute_standard(input_data)
    
    def _execute_standard(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform standard (single-time) ERT inversion.
        
        Args:
            input_data: Dictionary containing standard inversion parameters
                
        Returns:
            Dictionary containing inversion results
        """
        self._log_execution("Starting standard ERT inversion")
        
        try:
            import pygimli as pg

            from PyHydroGeophysX.data_processing.ert_data_agent import export_for_inversion
            from PyHydroGeophysX.inversion.ert_inversion import ERTInversion

            # Extract parameters
            ert_data = input_data.get('ert_data')
            inversion_params = input_data.get('inversion_params', {})
            use_structure = input_data.get('use_structure_constraint', False)
            seismic_structure = input_data.get('seismic_structure')
            output_dir = input_data.get('output_dir', 'results/ert_inversion')
            
            os.makedirs(output_dir, exist_ok=True)
            
            if ert_data is None:
                raise ValueError("ert_data is required")
            
            # Export ERT data for inversion
            self._log_execution("Exporting data to inversion format")
            use_source_error = input_data.get(
                'use_source_error',
                inversion_params.get('use_source_error', True)
            )
            export_strategy = input_data.get(
                'export_strategy',
                inversion_params.get('export_strategy', 'default')
            )
            self._log_execution(
                "Export mode: "
                + f"{export_strategy}, "
                + ("source (dataset-provided)" if bool(use_source_error) else "reciprocal (auto-estimated)")
            )
            data_file = export_for_inversion(
                ert_data, 
                outdir=output_dir, 
                fmt='pgimli',
                use_source_error=bool(use_source_error),
                export_strategy=str(export_strategy),
                default_relative_error=float(inversion_params.get('default_relative_error', 0.01)),
                default_absolute_error=float(inversion_params.get('default_absolute_error', 0.001)),
                default_rhoa_limits=tuple(inversion_params.get('default_rhoa_limits', (0.1, 10000.0))),
                default_reciprocal_percent=float(inversion_params.get('default_reciprocal_percent', 10.0)),
                default_fit_error_lin=bool(inversion_params.get('default_fit_error_lin', True)),
            )
            self._log_execution(f"ERT data exported to: {data_file}")
            
            # Get LLM recommendations for inversion parameters if API is available
            if self.api_key and not inversion_params:
                self._log_execution("Requesting LLM recommendations for inversion parameters")
                inversion_params = self._get_recommended_params(ert_data)
            
            # Set default parameters if not provided
            lambda_val = inversion_params.get('lambda', 20.0)
            max_iterations = inversion_params.get('max_iterations', 10)
            method = inversion_params.get('method', 'cgls')
            use_gpu = inversion_params.get('use_gpu', False)
            self._log_execution(f"Inversion parameters: lambda={lambda_val}, "
                              f"max_iter={max_iterations}, method={method}")
            
            # Handle structure-constrained inversion if seismic data provided
            mesh = None
            constraint_warning = None
            if use_structure:
                self._log_execution("Creating mesh with seismic structure constraints")
                try:
                    # From the exported container: the interface mesh is built
                    # around pyGIMLi sensor positions.
                    mesh = self._create_structured_mesh(data_file, seismic_structure)
                except Exception as exc:  # noqa: BLE001 - reported on the result
                    # Said on the result, not only in the log: a constrained
                    # inversion that quietly ran unconstrained is what this
                    # used to do, because the mesh builder it imported was gone.
                    constraint_warning = (
                        f"A structure-constrained inversion was requested but the "
                        f"constraint could not be built ({exc}); this model is an "
                        f"unconstrained inversion.")
                    self._log_execution(constraint_warning, level='WARNING')
            
            # Perform inversion
            self._log_execution("Running ERT inversion...")
            inversion = ERTInversion(
                data_file=data_file,
                lambda_val=lambda_val,
                method=method,
                use_gpu=use_gpu,
                max_iterations=max_iterations,
                mesh=mesh,
                lambda_rate=inversion_params.get('lambda_rate', 1.0),
                lambda_min=inversion_params.get('lambda_min', 1.0),
                model_constraints=tuple(inversion_params.get('model_constraints', (1e-6, 1e7))),
                quality=inversion_params.get('quality', 34.0),
                paraDX=inversion_params.get('paraDX', 0.3),
                paraMaxCellSize=inversion_params.get('paraMaxCellSize', 0.0),
                paraDepth=inversion_params.get('paraDepth', 0.0),
                absoluteUError=inversion_params.get('absoluteUError', 0.0),
                relativeError=inversion_params.get('relativeError', 0.05),
                maxLineSearch=inversion_params.get('maxLineSearch'),
                zWeight=inversion_params.get('zWeight'),
                robustData=inversion_params.get('robustData'),
                blockyModel=inversion_params.get('blockyModel'),
                verbose=inversion_params.get('verbose', False),
            )
            
            inversion_result = inversion.run()
            
            self._log_execution("Inversion completed successfully")
            
            # Store results
            self.update_context('inversion_result', inversion_result)
            self.update_context('data_file', data_file)
            
            # Get LLM interpretation of results
            interpretation = None
            if self.api_key:
                self._log_execution("Generating interpretation of results")
                interpretation = self._interpret_results(inversion_result, inversion_params)
            
            self.results = {
                'status': 'success',
                'inversion_result': inversion_result,
                'mesh': inversion_result.mesh,
                'resistivity_model': inversion_result.final_model,
                'coverage': inversion_result.coverage,
                'chi2': float(inversion_result.meta.get('final_chi2')) if inversion_result.meta.get('final_chi2') is not None else (float(np.asarray(inversion_result.iteration_chi2[-1]).item()) if inversion_result.iteration_chi2 else None),
                'iterations': int(inversion_result.meta.get('native_iterations')) if inversion_result.meta.get('native_iterations') is not None else len(inversion_result.iteration_chi2),
                'interpretation': interpretation,
                'output_dir': output_dir
            }
            self.results['processing'] = {
                'input_measurements': len(ert_data.observations),
                'inverted_measurements': int(inversion.data.size()),
                'exported_data_file': str(data_file),
                'export_strategy': str(export_strategy),
                'use_source_error': bool(use_source_error),
                'inversion_params': dict(inversion_params),
            }
            if use_structure:
                self.results['structure_constrained'] = mesh is not None
                if mesh is not None:
                    self.results['inversion_method'] = (
                        "Structure-constrained inversion: the seismic velocity "
                        "interface is built into the mesh as a boundary, so the "
                        "smoothing does not act across it")
                else:
                    self.results['warnings'] = [constraint_warning]
            
            final_chi2 = float(inversion_result.meta.get('final_chi2')) if inversion_result.meta.get('final_chi2') is not None else (float(np.asarray(inversion_result.iteration_chi2[-1]).item()) if inversion_result.iteration_chi2 else 0.0)
            n_iterations = int(inversion_result.meta.get('native_iterations')) if inversion_result.meta.get('native_iterations') is not None else len(inversion_result.iteration_chi2)
            self._log_execution(f"Final chi2: {final_chi2:.3f}, "
                              f"Iterations: {n_iterations}")
            
            return self.results
            
        except Exception as e:
            self._log_execution(f"Error during inversion: {str(e)}", level='ERROR')
            self.results = {
                'status': 'failed',
                'error': str(e)
            }
            raise
    
    def _get_recommended_params(self, ert_data) -> Dict[str, Any]:
        """
        Get LLM recommendations for inversion parameters.
        
        Args:
            ert_data: Loaded ERT data
            
        Returns:
            Recommended parameters dictionary
        """
        try:
            data_info = f"""
            ERT Data Characteristics:
            - Number of electrodes: {len(ert_data.electrodes)}
            - Number of measurements: {len(ert_data.observations)}
            - Array type: Wenner-Schlumberger
            """
            
            prompt = f"""Based on the following ERT survey characteristics, recommend 
appropriate inversion parameters:

{data_info}

Provide recommendations for:
1. Lambda (regularization parameter): typical range 10-50
2. Maximum iterations: typical range 5-20

Return as: lambda=XX, max_iterations=YY"""
            
            response = self.query_llm(prompt, self.system_message, temperature=0.3, max_tokens=150)
            
            # Parse response with robust error handling
            params = {'lambda': 20.0, 'max_iterations': 10}  # defaults
            
            try:
                import re

                # Try to extract lambda value
                lambda_match = re.search(r'lambda[=:\s]+(\d+\.?\d*)', response, re.IGNORECASE)
                if lambda_match:
                    params['lambda'] = float(lambda_match.group(1))
                
                # Try to extract max_iterations value
                iter_match = re.search(r'max[_\s]*iterations[=:\s]+(\d+)', response, re.IGNORECASE)
                if iter_match:
                    params['max_iterations'] = int(iter_match.group(1))
                    
            except (ValueError, AttributeError) as e:
                self._log_execution(f"Could not parse LLM response: {e}, using defaults")
            
            self._log_execution(f"LLM recommended: lambda={params['lambda']}, "
                              f"max_iterations={params['max_iterations']}")
            
            return params
        except Exception as e:
            self._log_execution(f"Could not get LLM recommendations: {e}, using defaults")
            return {'lambda': 20.0, 'max_iterations': 10}
    
    def _create_structured_mesh(self, data, seismic_structure) -> Any:
        """
        Create mesh with seismic structural constraints.

        Args:
            data: The survey as a pyGIMLi ERT container, or the path of one
            seismic_structure: A SeismicAgent result (``interfaces`` keyed by
                velocity threshold) or anything carrying ``interface_coords``
                as ``(x, z)``

        Returns:
            Mesh with the velocity interface built in as a region boundary
            (markers 2 above it, 3 below, 1 outside the survey)

        Raises:
            ValueError: If there is no interface to build in. The caller says
                so on its result; returning None used to leave the inversion
                running unconstrained with nothing but a log line to show it.
        """
        from pygimli.physics import ert

        # create_ert_mesh_with_structure, which this imported, was removed from
        # Geophy_modular.structure_integration; this is the builder the
        # structure-constraint agent uses.
        from PyHydroGeophysX.core.mesh_utils import add_velocity_interface

        structure = seismic_structure or {}
        coords = structure.get('interface_coords')
        if coords is None:
            # SeismicAgent returns {threshold: {'x': ..., 'z': ...}}, and the
            # coordinator hands that result over as it is.
            from .runtime.catalog import interface_coords

            thresholds = (structure.get('velocity_thresholds')
                          or [structure.get('velocity_threshold', 1200.0)])
            coords = interface_coords(structure, thresholds[0])
        if coords is None or len(coords) != 2 or not len(coords[0]):
            raise ValueError("the seismic structure carries no velocity interface")
        container = ert.load(str(data)) if isinstance(data, (str, os.PathLike)) else data
        _, mesh = add_velocity_interface(container,
                                         np.asarray(coords[0], dtype=float),
                                         np.asarray(coords[1], dtype=float))
        self._log_execution(f"Created structured mesh with seismic constraints "
                            f"({mesh.cellCount()} cells)")
        return mesh
    
    def _interpret_results(self, inversion_result, params) -> str:
        """
        Get LLM interpretation of inversion results.
        
        Args:
            inversion_result: Inversion results object
            params: Inversion parameters used
            
        Returns:
            Interpretation string
        """
        try:
            # Extract final chi2 as scalar (it's stored as numpy array)
            final_chi2 = float(np.asarray(inversion_result.iteration_chi2[-1]).item()) if inversion_result.iteration_chi2 else 0.0
            n_iterations = len(inversion_result.iteration_chi2)
            
            results_summary = f"""
            Inversion Results:
            - Final chi2: {final_chi2:.3f}
            - Number of iterations: {n_iterations}
            - Lambda used: {params.get('lambda', 'N/A')}
            - Resistivity range: {np.min(inversion_result.final_model):.1f} to {np.max(inversion_result.final_model):.1f} Ohm-m
            """
            
            prompt = f"""Interpret these ERT inversion results and assess data quality:

{results_summary}

Provide a brief interpretation (2-3 sentences) about:
1. Quality of the inversion (based on chi2)
2. What the resistivity range suggests about subsurface materials"""
            
            interpretation = self.query_llm(prompt, self.system_message, 
                                          temperature=0.5, max_tokens=200)
            return interpretation
        except Exception:
            return "Could not generate interpretation"
    
    def _execute_time_lapse(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform time-lapse ERT inversion.
        
        Args:
            input_data: Dictionary containing:
                - time_lapse_data: List of ERT datasets (or file paths) in temporal order
                - time_lapse_method: recorded only; see PyHydroGeophysX.agents._method
                - temporal_regularization: Temporal smoothing weight (default: 10.0)
                - baseline_index: Index of baseline dataset (default: 0)
                - inversion_params: Standard inversion parameters
                - output_dir: Directory for saving results
                
        Returns:
            Dictionary containing time-lapse inversion results
        """
        self._log_execution("Starting time-lapse ERT inversion")
        
        try:
            import pygimli as pg

            from PyHydroGeophysX.data_processing.ert_data_agent import export_for_inversion
            from PyHydroGeophysX.inversion.time_lapse import TimeLapseERTInversion

            # Extract parameters
            time_lapse_data = input_data.get('time_lapse_data', [])
            tl_method = input_data.get('time_lapse_method', IMPLEMENTED_SCHEME)
            temporal_reg = input_data.get('temporal_regularization', 10.0)
            baseline_idx = input_data.get('baseline_index', 0)
            inversion_params = input_data.get('inversion_params', {})
            output_dir = input_data.get('output_dir', 'results/ert_time_lapse')
            
            os.makedirs(output_dir, exist_ok=True)
            
            if not time_lapse_data or len(time_lapse_data) < 2:
                raise ValueError("Time-lapse inversion requires at least 2 datasets")
            
            self._log_execution(f"Processing {len(time_lapse_data)} time-lapse datasets")
            # Log what will actually run. The requested value is logged beside
            # it, not in place of it, because the solver ignores it.
            self._log_execution(f"Scheme: {SCHEME_LABEL} (configuration requested '{tl_method}'), "
                                f"temporal constraint alpha={temporal_reg}")
            
            # Export all datasets to inversion format
            use_source_error = input_data.get(
                'use_source_error',
                inversion_params.get('use_source_error', True)
            )
            data_files = []
            for i, ert_data in enumerate(time_lapse_data):
                self._log_execution(f"Exporting dataset {i+1}/{len(time_lapse_data)}")
                data_file = export_for_inversion(
                    ert_data,
                    outdir=output_dir,
                    fmt='pgimli',
                    use_source_error=bool(use_source_error)
                )
                # Rename to include time index
                import shutil
                new_file = os.path.join(output_dir, f'tl_data_{i:03d}.dat')
                shutil.move(data_file, new_file)
                data_files.append(new_file)
            
            # Set inversion parameters following Ex_TL_inversion.py pattern
            lambda_val = inversion_params.get('lambda', 50.0)
            alpha = temporal_reg  # Temporal regularization parameter
            decay_rate = inversion_params.get('decay_rate', 0.0)
            method = inversion_params.get('method', 'cgls')
            model_constraints = inversion_params.get('model_constraints', (0.001, 1e4))
            max_iterations = inversion_params.get('max_iterations', 15)
            absoluteUError = inversion_params.get('absoluteUError', 0.0)
            relativeError = inversion_params.get('relativeError', 0.05)
            lambda_rate = inversion_params.get('lambda_rate', 1.0)
            lambda_min = inversion_params.get('lambda_min', 1.0)
            inversion_type = inversion_params.get('inversion_type', 'L2')
            
            self._log_execution(f"Inversion parameters: lambda={lambda_val}, alpha={alpha}, "
                              f"max_iter={max_iterations}, method={method}, type={inversion_type}")
            
            # Measurement times. A sequential index treats a one-hour gap and a
            # one-month gap as the same spacing and gives the report a step number
            # where it wants a date, so the acquisition times are read off the
            # original file names whenever the caller did not supply them.
            source_files = [str(f) for f in (input_data.get('source_files') or [])]
            timing = None
            measurement_times = input_data.get('measurement_times')
            if measurement_times is None:
                from PyHydroGeophysX.data_processing.survey_timing import survey_timing

                if len(source_files) == len(time_lapse_data):
                    timing = survey_timing(source_files)
                if timing is not None and timing.dated:
                    measurement_times = list(timing.times)
                    self._log_execution(f"Survey timing: {timing.summary()}")
                else:
                    timing = None
                    measurement_times = list(range(1, len(time_lapse_data) + 1))
                    self._log_execution(
                        "No acquisition times could be read from the survey files; "
                        "using a sequential index, so the run cannot report the "
                        "real interval between surveys.", level='WARNING')
            
            # Create mesh for inversion
            from pygimli.physics import ert
            data = ert.load(data_files[0])
            ert_manager = ert.ERTManager(data)
            mesh = ert_manager.createMesh(data=data, quality=31, paraMaxCellSize=25,paraDepth=28)
            
            # Perform time-lapse inversion
            self._log_execution("Running time-lapse inversion...")
            tl_inversion = TimeLapseERTInversion(
                data_files=data_files,
                measurement_times=measurement_times,
                mesh=mesh,
                lambda_val=lambda_val,
                alpha=alpha,
                decay_rate=decay_rate,
                method=method,
                model_constraints=model_constraints,
                max_iterations=max_iterations,
                absoluteUError=absoluteUError,
                relativeError=relativeError,
                lambda_rate=lambda_rate,
                lambda_min=lambda_min,
                inversion_type=inversion_type
            )
            
            tl_result = tl_inversion.run()

            self._log_execution("Time-lapse inversion completed successfully")

            # Temperature correction. Resistivity falls about 2 % per degC, so over
            # a monitoring season the temperature signal is the same size as the
            # moisture signal the survey is run to see. Applied here so every
            # downstream consumer - the report, the petrophysics - reads the
            # corrected series; the raw one is kept alongside it.
            final_models = tl_result.final_models
            temperature_report = self._correct_for_temperature(
                inversion_params.get('temperature_correction'), final_models,
                tl_result, data, measurement_times, timing)
            if temperature_report.get('applied'):
                final_models = temperature_report.pop('models')
            else:
                temperature_report.pop('models', None)
            
            # Store results
            self.update_context('time_lapse_result', tl_result)
            self.update_context('data_files', data_files)
            
            # Get LLM interpretation of time-lapse results
            interpretation = None
            if self.api_key:
                self._log_execution("Generating interpretation of time-lapse results")
                interpretation = self._interpret_time_lapse_results(tl_result, tl_method)
            
            self.results = {
                'status': 'success',
                'inversion_mode': 'time-lapse',
                'time_lapse_result': tl_result,
                'final_models': final_models,  # 2D array: cells x timesteps
                'baseline_model': final_models[:, 0] if final_models is not None else None,
                'time_lapse_models': [final_models[:, i] for i in range(final_models.shape[1])] if final_models is not None else [],
                # What the models above are: inverted resistivity, or resistivity
                # reported at a reference temperature. A corrected section looks
                # exactly like an uncorrected one, so the run has to say which.
                'temperature_correction': temperature_report,
                'final_models_uncorrected': (
                    tl_result.final_models if temperature_report.get('applied') else None),
                'survey_timing': timing.to_dict() if timing is not None else None,
                'mesh': tl_result.mesh,
                # Two different things that were both called 'method'. The
                # report printed this one under "Solver Method" and showed the
                # time-lapse scheme there; the linear solver that actually ran
                # is the one in inversion_params.
                'time_lapse_method_requested': tl_method,
                'method': method,
                'temporal_regularization': temporal_reg,
                # Recorded on the result so the report and the audit describe the
                # run that happened, instead of showing N/A for settings the
                # caller passed in.
                'inversion_params': dict(inversion_params or {}),
                'lambda': dict(inversion_params or {}).get('lambda'),
                'max_iterations': dict(inversion_params or {}).get('max_iterations'),
                'n_timesteps': tl_result.final_models.shape[1] if tl_result.final_models is not None else len(time_lapse_data),
                'chi2_values': tl_result.all_chi2 if hasattr(tl_result, 'all_chi2') else None,
                'coverage': tl_result.all_coverage if hasattr(tl_result, 'all_coverage') else [],
                'timesteps': tl_result.timesteps if hasattr(tl_result, 'timesteps') else measurement_times,
                'interpretation': interpretation,
                'output_dir': output_dir
            }
            
            self._log_execution(f"Processed {len(time_lapse_data)} time steps")
            
            return self.results
            
        except Exception as e:
            self._log_execution(f"Error during time-lapse inversion: {str(e)}", level='ERROR')
            self.results = {
                'status': 'failed',
                'error': str(e),
                'inversion_mode': 'time-lapse'
            }
            raise
    
    def _correct_for_temperature(self, spec, models, tl_result, data,
                                 measurement_times, timing) -> dict:
        """Report every time step at one reference temperature, if asked to.

        Returns a report dict, carrying the corrected models under ``"models"``
        when it succeeded. A correction that was requested but could not be built
        is returned as ``applied: False`` with the reason and logged as a warning:
        a section left uncorrected looks exactly like a corrected one, so the run
        has to say which it is rather than quietly produce the wrong one.
        """
        if not spec or not bool(dict(spec).get('enabled', True)):
            return {'applied': False, 'requested': False}
        if models is None:
            return {'applied': False, 'requested': True,
                    'error': 'the inversion produced no models to correct'}
        try:
            import numpy as _np

            from PyHydroGeophysX.core import section_geometry
            from PyHydroGeophysX.petrophysics import temperature as temperature_model

            try:
                sensors = _np.asarray(data.sensors(), dtype=float)[:, :2]
            except Exception:  # noqa: BLE001 - topography comes from the mesh then
                sensors = None
            depths = section_geometry.cell_depths(tl_result.mesh, sensors=sensors)
            dates = list(timing.timestamps) if timing is not None and timing.dated else None
            corrected, report = temperature_model.correct_time_lapse_models(
                models, dict(spec), depths, days=list(measurement_times), dates=dates)
        except Exception as exc:  # noqa: BLE001 - never lose the inversion over this
            self._log_execution(
                f"Temperature correction was requested but could not be applied "
                f"({exc}); the models below are the raw inverted resistivity.",
                level='WARNING')
            return {'applied': False, 'requested': True, 'error': str(exc)}
        self._log_execution(f"Temperature correction: {report['note']}")
        report['models'] = corrected
        return report

    def _interpret_time_lapse_results(self, tl_result, method: str) -> str:
        """
        Get LLM interpretation of time-lapse inversion results.
        
        Args:
            tl_result: Time-lapse inversion results object
            method: Time-lapse method used
            
        Returns:
            Interpretation string
        """
        try:
            # TimeLapseInversionResult carries final_models, one column per
            # time step. The attributes read here before - baseline_model,
            # time_lapse_models, changes - are not on it, so every time-lapse
            # run logged "object has no attribute 'baseline_model'" and
            # produced no interpretation at all.
            models = getattr(tl_result, 'final_models', None)
            if models is None or getattr(models, 'ndim', 0) != 2 or models.shape[1] == 0:
                self._log_execution('No time-lapse models to interpret', level='WARNING')
                return None
            n_timesteps = int(models.shape[1])
            baseline = models[:, 0]

            results_summary = f"""
            Time-Lapse Inversion Results:
            - Number of time steps: {n_timesteps}
            - Method: {method}
            - Baseline resistivity range: {np.nanmin(baseline):.1f} to {np.nanmax(baseline):.1f} Ohm-m
            """

            if n_timesteps > 1:
                changes = models[:, 1:] - baseline[:, None]
                max_change = float(np.nanmax(np.abs(changes)))
                mean_changes = [float(np.nanmean(changes[:, i]))
                                for i in range(changes.shape[1])]
                results_summary += (
                    f"\n            - Maximum absolute resistivity change: {max_change:.1f} Ohm-m"
                    f"\n            - Mean change per step: "
                    + ", ".join(f"{value:+.2f}" for value in mean_changes) + " Ohm-m")
            
            prompt = f"""Interpret these time-lapse ERT inversion results:

{results_summary}

Provide a brief interpretation (2-3 sentences) about:
1. The temporal dynamics observed
2. What the resistivity changes might indicate (e.g., moisture movement, groundwater changes)
3. Quality assessment of the time-lapse inversion"""
            
            interpretation = self.query_llm(prompt, self.system_message,
                                          temperature=0.5, max_tokens=250)
            return interpretation
        except Exception as e:
            self._log_execution(f"Could not generate time-lapse interpretation: {e}")
            return "Could not generate interpretation"
    
    def _log_execution(self, message: str, level: str = 'INFO'):
        """Log execution message."""
        print(f"[{self.name}] [{level}] {message}")
