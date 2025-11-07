"""
Seismic Data Processing Agent

Specialized agent for processing seismic refraction data and extracting velocity structures.
"""

from typing import Dict, Any, Optional
import numpy as np
import os
from .base_agent import BaseAgent


class SeismicAgent(BaseAgent):
    """
    Agent specialized in seismic refraction tomography (SRT) processing.
    
    Uses PyHydroGeophysX seismic processing module to invert seismic data
    and extract velocity interfaces for structural constraints.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None,
                 llm_provider: str = "openai"):
        """Initialize Seismic Agent."""
        super().__init__("seismic_processor", api_key, model, llm_provider)
        self.system_message = """You are an expert in seismic refraction tomography (SRT). 
Your role is to process seismic travel time data, perform velocity inversions, and 
extract geological structure interfaces. You understand velocity-depth relationships 
and how to identify layer boundaries."""
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process seismic data and extract velocity structure.
        
        Args:
            input_data: Dictionary containing:
                - seismic_data: Seismic travel time data
                - velocity_threshold: Threshold for interface detection (default: 1200 m/s)
                - inversion_params: Parameters for seismic inversion
                - output_dir: Directory for saving results
                
        Returns:
            Dictionary containing velocity model and interface coordinates
        """
        self._log_execution("Starting seismic data processing")
        
        try:
            from PyHydroGeophysX.Geophy_modular.seismic_processor import (
                process_seismic_tomography,
                extract_velocity_structure,
                save_velocity_structure
            )
            
            # Extract parameters
            seismic_data = input_data.get('seismic_data')
            velocity_threshold = input_data.get('velocity_threshold', 1200)
            inversion_params = input_data.get('inversion_params', {})
            output_dir = input_data.get('output_dir', 'results/seismic')
            
            os.makedirs(output_dir, exist_ok=True)
            
            if seismic_data is None:
                raise ValueError("seismic_data is required")
            
            self._log_execution("Processing seismic tomography")
            
            # Get LLM recommendations for inversion parameters if needed
            if self.api_key and not inversion_params:
                self._log_execution("Requesting LLM recommendations for seismic inversion")
                inversion_params = self._get_recommended_params(seismic_data)
            
            # Set default parameters
            lam = inversion_params.get('lam', 50)
            zWeight = inversion_params.get('zWeight', 0.2)
            vTop = inversion_params.get('vTop', 500)
            vBottom = inversion_params.get('vBottom', 5000)
            
            self._log_execution(f"Inversion parameters: lam={lam}, zWeight={zWeight}")
            
            # Process seismic tomography
            TT_manager = process_seismic_tomography(
                seismic_data,
                lam=lam,
                zWeight=zWeight,
                vTop=vTop,
                vBottom=vBottom,
                verbose=1
            )
            
            self._log_execution("Seismic inversion completed")
            
            # Extract velocity structure
            self._log_execution(f"Extracting velocity interface at {velocity_threshold} m/s")
            interface_x, interface_z, interface_data = extract_velocity_structure(
                TT_manager.paraDomain,
                TT_manager.model.array(),
                threshold=velocity_threshold,
                interval=4.0
            )
            
            # Save velocity structure
            structure_file = os.path.join(output_dir, 'velocity_interface.npz')
            save_velocity_structure(structure_file, interface_x, interface_z, interface_data)
            
            self._log_execution(f"Velocity interface extracted with {len(interface_x)} points")
            
            # Get LLM interpretation
            interpretation = None
            if self.api_key:
                self._log_execution("Generating interpretation of seismic results")
                interpretation = self._interpret_results(TT_manager, interface_data)
            
            self.results = {
                'status': 'success',
                'velocity_model': TT_manager.model.array(),
                'mesh': TT_manager.paraDomain,
                'interface_coords': (interface_x, interface_z),
                'interface_data': interface_data,
                'velocity_threshold': velocity_threshold,
                'interpretation': interpretation,
                'output_dir': output_dir
            }
            
            velocity_range = [np.min(TT_manager.model.array()), 
                            np.max(TT_manager.model.array())]
            self._log_execution(f"Velocity range: {velocity_range[0]:.0f} - {velocity_range[1]:.0f} m/s")
            
            return self.results
            
        except Exception as e:
            self._log_execution(f"Error during seismic processing: {str(e)}", level='ERROR')
            self.results = {
                'status': 'failed',
                'error': str(e)
            }
            raise
    
    def _get_recommended_params(self, seismic_data) -> Dict[str, Any]:
        """
        Get LLM recommendations for seismic inversion parameters.
        
        Args:
            seismic_data: Seismic travel time data
            
        Returns:
            Recommended parameters dictionary
        """
        try:
            data_info = f"""
            Seismic Data Characteristics:
            - Data type: Travel time data
            - Expected geology: Regolith over bedrock
            """
            
            prompt = f"""Based on typical seismic refraction surveys over regolith-bedrock 
systems, recommend inversion parameters:

{data_info}

Provide recommendations for:
1. Lambda (regularization): typical range 20-100
2. zWeight (vertical regularization): typical range 0.1-0.5
3. vTop (top velocity): typical 300-800 m/s for soil/regolith
4. vBottom (bottom velocity): typical 3000-6000 m/s for bedrock

Return as: lam=XX, zWeight=XX, vTop=XX, vBottom=XX"""
            
            response = self.query_llm(prompt, self.system_message, temperature=0.3, max_tokens=200)
            
            # Parse response with robust error handling
            params = {'lam': 50, 'zWeight': 0.2, 'vTop': 500, 'vBottom': 5000}  # defaults
            
            try:
                import re
                for key in ['lam', 'zWeight', 'vTop', 'vBottom']:
                    pattern = rf'{key}[=:\s]+(\d+\.?\d*)'
                    match = re.search(pattern, response, re.IGNORECASE)
                    if match:
                        params[key] = float(match.group(1))
            except (ValueError, AttributeError) as e:
                self._log_execution(f"Could not parse LLM response: {e}, using defaults")
            
            self._log_execution(f"LLM recommended parameters: {params}")
            
            return params
        except Exception as e:
            self._log_execution(f"Could not get LLM recommendations: {e}, using defaults")
            return {'lam': 50, 'zWeight': 0.2, 'vTop': 500, 'vBottom': 5000}
    
    def _interpret_results(self, TT_manager, interface_data) -> str:
        """
        Get LLM interpretation of seismic results.
        
        Args:
            TT_manager: Travel time manager with results
            interface_data: Interface extraction data
            
        Returns:
            Interpretation string
        """
        try:
            velocity_model = TT_manager.model.array()
            results_summary = f"""
            Seismic Inversion Results:
            - Velocity range: {np.min(velocity_model):.0f} to {np.max(velocity_model):.0f} m/s
            - Interface threshold: {interface_data['threshold']} m/s
            - Interface depth range: {np.min(interface_data['smooth_z']):.1f} to {np.max(interface_data['smooth_z']):.1f} m
            """
            
            prompt = f"""Interpret these seismic refraction results:

{results_summary}

Provide a brief interpretation (2-3 sentences) about:
1. What the velocity structure suggests about subsurface geology
2. Confidence in the interface location"""
            
            interpretation = self.query_llm(prompt, self.system_message, 
                                          temperature=0.5, max_tokens=200)
            return interpretation
        except:
            return "Could not generate interpretation"
    
    def _log_execution(self, message: str, level: str = 'INFO'):
        """Log execution message."""
        print(f"[{self.name}] [{level}] {message}")
