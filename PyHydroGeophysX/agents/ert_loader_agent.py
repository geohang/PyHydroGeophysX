"""
ERT Loader Agent

Specialized agent for loading and quality-checking ERT field data.
"""

from typing import Dict, Any, Optional
import numpy as np
from .base_agent import BaseAgent


class ERTLoaderAgent(BaseAgent):
    """
    Agent specialized in loading ERT data from various instruments.
    
    Uses PyHydroGeophysX data_processing module to load, validate,
    and prepare ERT data for inversion.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize ERT Loader Agent."""
        super().__init__("ert_loader", api_key)
        self.system_message = """You are an expert in electrical resistivity tomography (ERT) 
data processing. Your role is to load and validate ERT field data from various commercial 
instruments, perform quality control, and prepare data for inversion. You understand 
different data formats, coordinate systems, and common data quality issues."""
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load and process ERT data.
        
        Args:
            input_data: Dictionary containing:
                - data_file: Path to ERT data file
                - instrument: Instrument type (E4D, Syscal, ABEM, etc.)
                - project_dir: Project directory
                - crs: Coordinate reference system ('local' or EPSG code)
                - quality_check: Whether to perform quality checks (default: True)
                
        Returns:
            Dictionary containing loaded ERT data and quality metrics
        """
        self._log_execution("Starting ERT data loading")
        
        try:
            from PyHydroGeophysX.data_processing.ert_data_agent import (
                load_ert_resipy, qc_and_visualize, export_for_inversion, LocalRef
            )
            
            # Extract parameters
            data_file = input_data.get('data_file')
            instrument = input_data.get('instrument', 'E4D')
            project_dir = input_data.get('project_dir', '.')
            crs = input_data.get('crs', 'local')
            quality_check = input_data.get('quality_check', True)
            
            if not data_file:
                raise ValueError("data_file is required")
            
            self._log_execution(f"Loading data from {data_file} ({instrument} format)")
            
            # Set up coordinate reference
            local_ref = LocalRef(origin_x=0.0, origin_y=0.0, azimuth_deg=90.0)
            
            # Load ERT data
            ert_data = load_ert_resipy(
                project_dir=project_dir,
                data_file=data_file,
                instrument=instrument,
                crs=crs,
                local_ref=local_ref if crs == 'local' else None
            )
            
            self._log_execution(f"Loaded {len(ert_data.electrodes)} electrodes, "
                              f"{len(ert_data.observations)} measurements")
            
            # Store in context
            self.update_context('ert_data', ert_data)
            self.update_context('data_file', data_file)
            self.update_context('instrument', instrument)
            
            # Perform quality checks if requested
            qc_results = None
            if quality_check:
                self._log_execution("Running quality control checks")
                qc_output_dir = input_data.get('output_dir', 'results/ert_loader')
                qc_artifacts = qc_and_visualize(ert_data, outdir=qc_output_dir)
                qc_results = self._analyze_data_quality(ert_data)
                self.update_context('qc_artifacts', qc_artifacts)
                self.update_context('qc_results', qc_results)
            
            # Use LLM to provide intelligent insights about the data
            if self.api_key:
                data_summary = f"""
                Data Summary:
                - Instrument: {instrument}
                - Number of electrodes: {len(ert_data.electrodes)}
                - Number of measurements: {len(ert_data.observations)}
                - Coordinate system: {crs}
                """
                
                if qc_results:
                    data_summary += f"\n- Quality metrics: {qc_results}"
                
                insights = self._get_llm_insights(data_summary)
                self.update_context('llm_insights', insights)
            else:
                insights = "LLM insights not available (API key not provided)"
            
            self.results = {
                'status': 'success',
                'ert_data': ert_data,
                'num_electrodes': len(ert_data.electrodes),
                'num_measurements': len(ert_data.observations),
                'qc_results': qc_results,
                'insights': insights
            }
            
            self._log_execution("ERT data loading completed successfully")
            return self.results
            
        except Exception as e:
            self._log_execution(f"Error loading ERT data: {str(e)}", level='ERROR')
            self.results = {
                'status': 'failed',
                'error': str(e)
            }
            raise
    
    def _analyze_data_quality(self, ert_data) -> Dict[str, Any]:
        """
        Analyze data quality metrics.
        
        Args:
            ert_data: Loaded ERT data object
            
        Returns:
            Dictionary of quality metrics
        """
        try:
            # Extract apparent resistivity values
            rhoa_values = np.array([obs.resistance for obs in ert_data.observations])
            
            # Calculate basic statistics
            metrics = {
                'mean_rhoa': float(np.mean(rhoa_values)),
                'std_rhoa': float(np.std(rhoa_values)),
                'min_rhoa': float(np.min(rhoa_values)),
                'max_rhoa': float(np.max(rhoa_values)),
                'num_negative': int(np.sum(rhoa_values < 0)),
                'num_outliers': int(np.sum(np.abs(rhoa_values - np.mean(rhoa_values)) > 3 * np.std(rhoa_values)))
            }
            
            return metrics
        except:
            return {'error': 'Could not compute quality metrics'}
    
    def _get_llm_insights(self, data_summary: str) -> str:
        """
        Get intelligent insights from LLM about the data.
        
        Args:
            data_summary: Summary of loaded data
            
        Returns:
            LLM-generated insights
        """
        try:
            prompt = f"""Based on the following ERT data summary, provide brief insights 
about the data quality and any recommendations for the inversion process:

{data_summary}

Provide concise, practical insights (2-3 sentences)."""
            
            insights = self.query_llm(prompt, self.system_message, temperature=0.5, max_tokens=200)
            return insights
        except:
            return "Could not generate LLM insights"
    
    def _log_execution(self, message: str, level: str = 'INFO'):
        """Log execution message."""
        print(f"[{self.name}] [{level}] {message}")
