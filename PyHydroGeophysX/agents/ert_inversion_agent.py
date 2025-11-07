"""
ERT Inversion Agent

Specialized agent for performing ERT inversion with optional structural constraints.
"""

from typing import Dict, Any, Optional
import numpy as np
import os
from .base_agent import BaseAgent


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
                - time_lapse_method: 'difference', 'ratio', or 'joint' (for time-lapse)
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
            from PyHydroGeophysX.inversion.ert_inversion import ERTInversion
            from PyHydroGeophysX.data_processing.ert_data_agent import export_for_inversion
            import pygimli as pg
            
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
            data_file = export_for_inversion(
                ert_data, 
                outdir=output_dir, 
                fmt='pgimli'
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
            if use_structure and seismic_structure:
                self._log_execution("Creating mesh with seismic structure constraints")
                mesh = self._create_structured_mesh(ert_data, seismic_structure)
            
            # Perform inversion
            self._log_execution("Running ERT inversion...")
            inversion = ERTInversion(
                data_file=data_file,
                lambda_val=lambda_val,
                method=method,
                use_gpu=use_gpu,
                max_iterations=max_iterations,
                mesh=mesh
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
                'chi2': float(inversion_result.iteration_chi2[-1]) if inversion_result.iteration_chi2 else None,
                'iterations': len(inversion_result.iteration_chi2),
                'interpretation': interpretation,
                'output_dir': output_dir
            }
            
            final_chi2 = float(inversion_result.iteration_chi2[-1]) if inversion_result.iteration_chi2 else 0.0
            n_iterations = len(inversion_result.iteration_chi2)
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
    
    def _create_structured_mesh(self, ert_data, seismic_structure) -> Any:
        """
        Create mesh with seismic structural constraints.
        
        Args:
            ert_data: ERT data
            seismic_structure: Seismic structure information
            
        Returns:
            Structured mesh
        """
        try:
            from PyHydroGeophysX.Geophy_modular.structure_integration import (
                create_ert_mesh_with_structure
            )
            
            interface_coords = seismic_structure.get('interface_coords')
            if interface_coords:
                mesh, markers, regions = create_ert_mesh_with_structure(
                    ert_data,
                    interface_coords
                )
                self._log_execution("Created structured mesh with seismic constraints")
                return mesh
        except Exception as e:
            self._log_execution(f"Could not create structured mesh: {str(e)}")
        
        return None
    
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
            final_chi2 = float(inversion_result.iteration_chi2[-1]) if inversion_result.iteration_chi2 else 0.0
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
        except:
            return "Could not generate interpretation"
    
    def _execute_time_lapse(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform time-lapse ERT inversion.
        
        Args:
            input_data: Dictionary containing:
                - time_lapse_data: List of ERT datasets (or file paths) in temporal order
                - time_lapse_method: 'difference', 'ratio', or 'joint'
                - temporal_regularization: Temporal smoothing weight (default: 10.0)
                - baseline_index: Index of baseline dataset (default: 0)
                - inversion_params: Standard inversion parameters
                - output_dir: Directory for saving results
                
        Returns:
            Dictionary containing time-lapse inversion results
        """
        self._log_execution("Starting time-lapse ERT inversion")
        
        try:
            from PyHydroGeophysX.inversion.time_lapse import TimeLapseInversion
            from PyHydroGeophysX.data_processing.ert_data_agent import export_for_inversion
            import pygimli as pg
            
            # Extract parameters
            time_lapse_data = input_data.get('time_lapse_data', [])
            tl_method = input_data.get('time_lapse_method', 'difference')
            temporal_reg = input_data.get('temporal_regularization', 10.0)
            baseline_idx = input_data.get('baseline_index', 0)
            inversion_params = input_data.get('inversion_params', {})
            output_dir = input_data.get('output_dir', 'results/ert_time_lapse')
            
            os.makedirs(output_dir, exist_ok=True)
            
            if not time_lapse_data or len(time_lapse_data) < 2:
                raise ValueError("Time-lapse inversion requires at least 2 datasets")
            
            self._log_execution(f"Processing {len(time_lapse_data)} time-lapse datasets")
            self._log_execution(f"Method: {tl_method}, Temporal regularization: {temporal_reg}")
            
            # Export all datasets to inversion format
            data_files = []
            for i, ert_data in enumerate(time_lapse_data):
                self._log_execution(f"Exporting dataset {i+1}/{len(time_lapse_data)}")
                data_file = export_for_inversion(
                    ert_data,
                    outdir=output_dir,
                    fmt='pgimli'
                )
                # Rename to include time index
                import shutil
                new_file = os.path.join(output_dir, f'tl_data_{i:03d}.dat')
                shutil.move(data_file, new_file)
                data_files.append(new_file)
            
            # Set inversion parameters
            lambda_val = inversion_params.get('lambda', 20.0)
            max_iterations = inversion_params.get('max_iterations', 10)
            method = inversion_params.get('method', 'cgls')
            use_gpu = inversion_params.get('use_gpu', False)
            
            self._log_execution(f"Inversion parameters: lambda={lambda_val}, "
                              f"max_iter={max_iterations}, method={method}")
            
            # Perform time-lapse inversion
            self._log_execution("Running time-lapse inversion...")
            tl_inversion = TimeLapseInversion(
                data_files=data_files,
                lambda_val=lambda_val,
                method=method,
                use_gpu=use_gpu,
                max_iterations=max_iterations,
                temporal_regularization=temporal_reg,
                baseline_index=baseline_idx,
                time_lapse_method=tl_method
            )
            
            tl_result = tl_inversion.run()
            
            self._log_execution("Time-lapse inversion completed successfully")
            
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
                'baseline_model': tl_result.baseline_model,
                'time_lapse_models': tl_result.time_lapse_models,
                'changes': tl_result.changes if hasattr(tl_result, 'changes') else None,
                'mesh': tl_result.mesh,
                'method': tl_method,
                'temporal_regularization': temporal_reg,
                'n_timesteps': len(time_lapse_data),
                'chi2_values': tl_result.chi2_values if hasattr(tl_result, 'chi2_values') else None,
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
            n_timesteps = len(tl_result.time_lapse_models) if hasattr(tl_result, 'time_lapse_models') else 0
            
            results_summary = f"""
            Time-Lapse Inversion Results:
            - Number of time steps: {n_timesteps}
            - Method: {method}
            - Baseline resistivity range: {np.min(tl_result.baseline_model):.1f} to {np.max(tl_result.baseline_model):.1f} Ohm-m
            """
            
            if hasattr(tl_result, 'changes') and tl_result.changes:
                max_change = np.max([np.abs(c).max() for c in tl_result.changes])
                results_summary += f"\n            - Maximum resistivity change: {max_change:.1f} Ohm-m"
            
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
