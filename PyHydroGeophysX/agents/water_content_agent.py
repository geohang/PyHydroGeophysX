"""
Water Content Conversion Agent

Specialized agent for converting resistivity to water content using petrophysical models.
"""

import os
from typing import Any, Dict, Optional

import numpy as np

from .base_agent import BaseAgent


# ---------------------------------------------------------------------------
# Water Content Agent
# ---------------------------------------------------------------------------
class WaterContentAgent(BaseAgent):
    """
    Agent specialized in converting resistivity to water content.
    
    Uses PyHydroGeophysX petrophysical models and Monte Carlo uncertainty
    quantification to estimate water content from resistivity.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None,
                 llm_provider: str = "openai"):
        """Initialize Water Content Agent."""
        super().__init__("water_content", api_key, model, llm_provider)
        self.system_message = """You are an expert in petrophysical relationships and 
rock physics. Your role is to convert electrical resistivity to water content using 
appropriate models (Archie's law, Waxman-Smits), select suitable parameters for 
different geological layers, and quantify uncertainties."""
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert resistivity to water content.
        
        Args:
            input_data: Dictionary containing:
                - inversion_results: ERT inversion results
                - petrophysical_params: Parameters for each layer (rhos, n, porosity, etc.)
                - uncertainty_analysis: Whether to run Monte Carlo (default: False)
                - n_realizations: Number of MC realizations (default: 100)
                - output_dir: Directory for saving results
                
        Returns:
            Dictionary containing water content estimates and uncertainties
        """
        self._log_execution("Starting water content conversion")
        
        try:
            from PyHydroGeophysX.Geophy_modular.ERT_to_WC import ERTtoWC

            # Extract parameters
            inversion_results = input_data.get('inversion_results')
            petro_params = input_data.get('petrophysical_params', {})
            run_uncertainty = input_data.get('uncertainty_analysis', False)
            n_realizations = input_data.get('n_realizations', 100)
            output_dir = input_data.get('output_dir', 'results/water_content')
            
            os.makedirs(output_dir, exist_ok=True)
            
            if inversion_results is None:
                raise ValueError("inversion_results is required")
            
            # Get mesh and resistivity values
            mesh = inversion_results['mesh']
            resistivity = inversion_results['resistivity_model']
            coverage = inversion_results.get('coverage')
            
            # Prepare resistivity array for time-series format (even if single time)
            if resistivity.ndim == 1:
                resistivity_values = resistivity.reshape(-1, 1)
            else:
                resistivity_values = resistivity
            
            # Get cell markers
            cell_markers = np.array(mesh.cellMarkers())
            
            # Get LLM recommendations for petrophysical parameters if needed
            if self.api_key and not petro_params:
                self._log_execution("Requesting LLM recommendations for petrophysical parameters")
                petro_params = self._get_recommended_petro_params(resistivity, cell_markers)
            
            # Set up default layer distributions (even if petro_params has some values)
            layer_distributions = self._setup_layer_distributions(petro_params, cell_markers)
            
            self._log_execution(f"Converting resistivity to water content for "
                              f"{len(np.unique(cell_markers))} geological layers")
            
            # Initialize converter
            converter = ERTtoWC(
                mesh=mesh,
                resistivity_values=resistivity_values,
                cell_markers=cell_markers,
                coverage=coverage
            )
            
            # Setup layer distributions
            converter.setup_layer_distributions(layer_distributions)
            
            # Run Monte Carlo if requested
            if run_uncertainty:
                self._log_execution(f"Running Monte Carlo analysis with {n_realizations} realizations")
                wc_all, sat_all, params_used = converter.run_monte_carlo(
                    n_realizations=n_realizations,
                    progress_bar=True
                )
                
                # Get statistics
                stats = converter.get_statistics()
                
                # Save results
                converter.save_results(output_dir, 'water_content')
                
                self._log_execution("Monte Carlo analysis completed")
            else:
                self._log_execution("Running single water content estimate")
                # Single realization
                wc_all, sat_all, params_used = converter.run_monte_carlo(
                    n_realizations=1,
                    progress_bar=False
                )
                stats = {
                    'mean': wc_all[0],
                    'std': np.zeros_like(wc_all[0])
                }
            
            # Get LLM interpretation
            interpretation = None
            if self.api_key:
                self._log_execution("Generating interpretation of water content results")
                interpretation = self._interpret_wc_results(stats, layer_distributions)
            
            self.results = {
                'status': 'success',
                'water_content_mean': stats['mean'],
                'water_content_std': stats['std'],
                'saturation_all': sat_all if run_uncertainty else sat_all[0],
                'layer_distributions': layer_distributions,
                'params_used': params_used if run_uncertainty else None,
                'interpretation': interpretation,
                'output_dir': output_dir,
                'mesh': mesh,
                'coverage': coverage
            }
            
            # Log summary statistics
            mean_wc_value = None
            if isinstance(stats['mean'], np.ndarray):
                if coverage is not None:
                    mean_wc_value = np.mean(stats['mean'][coverage > 0])
                else:
                    mean_wc_value = np.mean(stats['mean'])
            
            if mean_wc_value is not None:
                self._log_execution(f"Mean water content: {mean_wc_value:.3f}")
            
            return self.results
            
        except Exception as e:
            self._log_execution(f"Error during water content conversion: {str(e)}", level='ERROR')
            self.results = {
                'status': 'failed',
                'error': str(e)
            }
            raise
    
    def _setup_layer_distributions(self, petro_params: Dict, cell_markers: np.ndarray) -> Dict:
        """
        Setup layer parameter distributions for Monte Carlo.
        
        Args:
            petro_params: Petrophysical parameters
            cell_markers: Cell markers identifying layers
            
        Returns:
            Layer distributions dictionary
        """
        unique_markers = np.unique(cell_markers)
        
        # Default parameters for different layer types
        default_params = {
            'regolith': {
                'rhos': {'mean': 100.0, 'std': 20.0},
                'n': {'mean': 2.2, 'std': 0.2},
                'sigma_sur': {'mean': 0.002, 'std': 0.0005},
                'porosity': {'mean': 0.40, 'std': 0.05}
            },
            'fractured_bedrock': {
                'rhos': {'mean': 500.0, 'std': 100.0},
                'n': {'mean': 1.8, 'std': 0.2},
                'sigma_sur': {'mean': 0.0, 'std': 0.0},
                'porosity': {'mean': 0.30, 'std': 0.05}
            },
            'bedrock': {
                'rhos': {'mean': 2000.0, 'std': 500.0},
                'n': {'mean': 2.0, 'std': 0.2},
                'sigma_sur': {'mean': 0.0, 'std': 0.0},
                'porosity': {'mean': 0.10, 'std': 0.03}
            }
        }
        
        # Build layer distributions
        layer_distributions = {}
        
        for i, marker in enumerate(unique_markers):
            # Use provided parameters or defaults
            if marker in petro_params:
                layer_distributions[marker] = petro_params[marker]
            else:
                # Assign default based on layer index
                if i == 0:  # Top layer
                    layer_distributions[marker] = default_params['regolith']
                elif i == len(unique_markers) - 1:  # Bottom layer
                    layer_distributions[marker] = default_params['bedrock']
                else:  # Middle layer(s)
                    layer_distributions[marker] = default_params['fractured_bedrock']
        
        return layer_distributions
    
    def _get_recommended_petro_params(self, resistivity: np.ndarray, 
                                     cell_markers: np.ndarray) -> Dict:
        """
        Get LLM recommendations for petrophysical parameters.
        
        Args:
            resistivity: Resistivity values
            cell_markers: Cell markers
            
        Returns:
            Recommended parameters dictionary
        """
        try:
            unique_markers = np.unique(cell_markers)
            
            # Calculate resistivity statistics per layer
            layer_stats = []
            for marker in unique_markers:
                mask = cell_markers == marker
                res_layer = resistivity[mask]
                layer_stats.append(f"Layer {marker}: mean={np.mean(res_layer):.1f}, "
                                 f"range=[{np.min(res_layer):.1f}, {np.max(res_layer):.1f}] Ohm-m")
            
            info = f"""
            Resistivity Statistics by Layer:
            {chr(10).join(layer_stats)}
            
            Number of layers: {len(unique_markers)}
            """
            
            prompt = f"""Based on these resistivity values, suggest appropriate petrophysical 
parameters for water content conversion:

{info}

For each layer, consider:
- Low resistivity (<200 Ohm-m): likely saturated regolith/soil
- Medium resistivity (200-1000 Ohm-m): partially saturated or fractured rock
- High resistivity (>1000 Ohm-m): dry or competent bedrock

Provide brief recommendations."""
            
            response = self.query_llm(prompt, self.system_message, temperature=0.5, max_tokens=300)
            self.update_context('petro_recommendations', response)
            
        except:
            self._log_execution("Could not get LLM recommendations for petrophysical parameters")
        
        return {}  # Return empty, will use defaults
    
    def _interpret_wc_results(self, stats: Dict, layer_distributions: Dict) -> str:
        """
        Get LLM interpretation of water content results.
        
        Args:
            stats: Water content statistics
            layer_distributions: Layer parameter distributions
            
        Returns:
            Interpretation string
        """
        try:
            mean_wc = stats['mean']
            wc_summary = f"""
            Water Content Results:
            - Mean: {np.mean(mean_wc):.3f}
            - Range: [{np.min(mean_wc):.3f}, {np.max(mean_wc):.3f}]
            - Number of layers: {len(layer_distributions)}
            """
            
            prompt = f"""Interpret these water content results:

{wc_summary}

Provide a brief interpretation (2-3 sentences) about:
1. What these water content values suggest about subsurface hydrology
2. Any notable patterns or concerns"""
            
            interpretation = self.query_llm(prompt, self.system_message, 
                                          temperature=0.5, max_tokens=200)
            return interpretation
        except:
            return "Could not generate interpretation"
    
    def _log_execution(self, message: str, level: str = 'INFO'):
        """Log execution message."""
        print(f"[{self.name}] [{level}] {message}")
