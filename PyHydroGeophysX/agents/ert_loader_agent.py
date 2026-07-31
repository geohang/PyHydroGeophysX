"""
ERT Loader Agent

Specialized agent for loading and quality-checking ERT field data.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
from .base_agent import AgentResult, BaseAgent


class ERTLoaderAgent(BaseAgent):
    """
    Agent specialized in loading ERT data from various instruments.
    
    Uses PyHydroGeophysX data_processing module to load, validate,
    and prepare ERT data for inversion.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None,
                 llm_provider: str = "openai"):
        """Initialize ERT Loader Agent."""
        super().__init__("ert_loader", api_key, model, llm_provider)
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
            # Extract parameters
            data_file = input_data.get('data_file')
            instrument = input_data.get('instrument', 'E4D')
            project_dir = input_data.get('project_dir', '.')
            electrode_file = input_data.get('electrode_file')  # Extract electrode file
            crs = input_data.get('crs', 'local')
            quality_check = input_data.get('quality_check', True)

            if data_file and not Path(str(data_file)).expanduser().is_absolute():
                project_candidate = Path(str(project_dir)) / str(data_file)
                if project_candidate.exists():
                    data_file = str(project_candidate)

            if electrode_file and not Path(str(electrode_file)).expanduser().is_absolute():
                project_candidate = Path(str(project_dir)) / str(electrode_file)
                if project_candidate.exists():
                    electrode_file = str(project_candidate)

            validation_error = self.validate_input_file(
                data_file,
                supported_extensions=[".ohm", ".bin", ".dat", ".stg", ".txt", ".data"],
                field_name="data_file",
                max_size_mb=input_data.get("max_file_size_mb"),
            )
            if validation_error:
                return validation_error

            if electrode_file:
                electrode_validation = self.validate_input_file(
                    electrode_file,
                    supported_extensions=[".dat", ".txt", ".csv"],
                    field_name="electrode_file",
                    max_size_mb=input_data.get("max_file_size_mb"),
                )
                if electrode_validation:
                    return electrode_validation

            detected_instrument = self._detect_instrument_from_header(data_file)
            if detected_instrument and instrument and detected_instrument != instrument:
                return AgentResult(
                    status="needs_review",
                    summary="The declared ERT instrument does not match the file header.",
                    data={
                        "declared_instrument": instrument,
                        "detected_instrument": detected_instrument,
                        "data_file": data_file,
                    },
                error_fix_hint=(
                    f"Change instrument to '{detected_instrument}' or verify that "
                    f"the file really uses '{instrument}' format. See: "
                    "https://geohang.github.io/PyHydroGeophysX/agents/troubleshooting.html#wrong-ert-instrument"
                ),
            )
            
            from PyHydroGeophysX.data_processing.ert_data_agent import (
                load_ert_resipy, qc_and_visualize, export_for_inversion, LocalRef
            )

            self._log_execution(f"Loading data from {data_file} ({instrument} format)")
            if electrode_file:
                self._log_execution(f"Using electrode file: {electrode_file}")
            
            # Set up coordinate reference
            local_ref = LocalRef(origin_x=0.0, origin_y=0.0, azimuth_deg=90.0)
            
            # Load ERT data
            ert_data = load_ert_resipy(
                project_dir=project_dir,
                data_file=data_file,
                instrument=instrument,
                electrode_file=electrode_file,  # Pass electrode file
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
            self.results = AgentResult(
                status="failed",
                summary="ERT data could not be loaded.",
                data={},
                error=str(e),
                error_fix_hint=(
                    "Check the file path, instrument name, electrode file, and file format. See: "
                    "https://geohang.github.io/PyHydroGeophysX/agents/troubleshooting.html#data-file-not-found"
                ),
            )
            return self.results

    def _detect_instrument_from_header(self, data_file: str) -> Optional[str]:
        """Detect a likely ERT instrument from the first two text lines.

        Parameters
        ----------
        data_file : str
            Path to an ERT data file.

        Returns
        -------
        str or None
            Detected canonical instrument name.

        Raises
        ------
        None

        Examples
        --------
        >>> ERTLoaderAgent()._detect_instrument_from_header("")
        """
        try:
            with open(data_file, "r", encoding="utf-8", errors="ignore") as handle:
                header = " ".join([handle.readline(), handle.readline()]).lower()
        except Exception:
            return None

        if "e4d" in header:
            return "E4D"
        if "syscal" in header:
            return "Syscal"
        if "abem" in header or "terameter" in header:
            return "ABEM-Lund"
        if "das" in header or "das-1" in header:
            return "DAS-1"
        if "bert" in header:
            return "BERT"
        if "sting" in header:
            return "Sting"
        if "ares" in header:
            return "ARES"
        return None
    
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
        except Exception:
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
        except Exception:
            return "Could not generate LLM insights"
    
    def _log_execution(self, message: str, level: str = 'INFO'):
        """Log execution message."""
        print(f"[{self.name}] [{level}] {message}")
