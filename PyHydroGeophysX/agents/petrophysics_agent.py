"""
Petrophysics Agent

Converts resistivity models to hydrological properties (water content, saturation, porosity)
using structure-constrained petrophysical models with Monte Carlo uncertainty quantification.
Implements the workflow from Ex_MC_Hydro.py.
"""

import os
from typing import Any, Dict, Optional

import numpy as np
from tqdm import tqdm

from .base_agent import BaseAgent


# ---------------------------------------------------------------------------
# Petrophysics Agent
# ---------------------------------------------------------------------------
class PetrophysicsAgent(BaseAgent):
    """
    Agent for converting resistivity to hydrological properties with uncertainty.
    
    This agent uses Archie's law and modified petrophysical models to convert
    resistivity to water content, incorporating:
    - Layer-specific parameters from structural constraints
    - Monte Carlo uncertainty quantification
    - Surface conductivity effects in clay-rich materials
    """
    
    # Default parameter distributions for common geological layers
    DEFAULT_LAYER_PARAMS = {
        'regolith': {
            'm': {'mean': 1.3, 'std': 0.1},
            'n': {'mean': 2.1, 'std': 0.1},
            'sigma_sur': {'mean': 1/200, 'std': 1/200},
            'porosity': {'mean': 0.42, 'std': 0.05},
            'rho_fluid': 20.0
        },
        'bedrock': {
            'm': {'mean': 1.9, 'std': 0.2},
            'n': {'mean': 1.7, 'std': 0.2},
            'sigma_sur': {'mean': 0.0, 'std': 0.0},
            'porosity': {'mean': 0.25, 'std': 0.15},
            'rho_fluid': 20.0
        }
    }
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None,
                 llm_provider: str = "openai"):
        """Initialize Petrophysics Agent."""
        super().__init__("petrophysics", api_key, model, llm_provider)
        self.system_message = """You are an expert in petrophysical modeling and hydrogeophysics.
You understand how to convert electrical resistivity to water content using Archie's law
and modified petrophysical relationships. You can recommend appropriate parameters for
different geological materials and quantify uncertainties."""
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert resistivity to water content with uncertainty quantification.
        
        Args:
            input_data: Dictionary containing:
                - resistivity_model: Resistivity values (can be 1D or 2D for time-lapse)
                - mesh: PyGIMLI mesh with cell markers
                - cell_markers: Array identifying geological layers
                - layer_params: Dictionary of parameters for each layer (optional)
                - n_realizations: Number of Monte Carlo samples (default: 100)
                - output_dir: Directory for saving results
                
        Returns:
            Dictionary containing water content statistics and uncertainty
        """
        self._log_execution("Starting petrophysical conversion with uncertainty analysis")
        
        try:
            from PyHydroGeophysX.petrophysics.resistivity_models import resistivity_to_saturation

            # Extract parameters
            resistivity_model = input_data.get('resistivity_model')
            mesh = input_data.get('mesh')
            cell_markers = input_data.get('cell_markers')
            layer_params = input_data.get('layer_params', None)
            # Normalize legacy key name
            petrophysical_params = input_data.get('petrophysical_params', None) or input_data.get('petrophysical_parameters', None)
            geological_context = input_data.get('geological_context', 'generic')
            n_realizations = input_data.get('n_realizations', 100)
            output_dir = input_data.get('output_dir', 'results/petrophysics')
            
            os.makedirs(output_dir, exist_ok=True)
            
            if resistivity_model is None:
                raise ValueError("resistivity_model is required")
            if cell_markers is None:
                raise ValueError("cell_markers is required")
            
            # Ensure resistivity_model is 2D (cells × timesteps)
            resistivity_array = np.array(resistivity_model)
            if resistivity_array.ndim == 1:
                resistivity_array = resistivity_array.reshape(-1, 1)
            
            n_cells, n_timesteps = resistivity_array.shape
            
            self._log_execution(f"Processing {n_cells} cells, {n_timesteps} time step(s)")
            self._log_execution(f"Monte Carlo realizations: {n_realizations}")
            
            # Check if cell_markers need simplification (too many unique markers)
            unique_layers = np.unique(cell_markers)
            n_unique = len(unique_layers)
            
            # If too many unique markers (each cell has unique marker), treat as single layer
            if n_unique > 100:
                self._log_execution(f"Too many unique markers ({n_unique}), treating as single layer")
                cell_markers = np.zeros(n_cells, dtype=int)
                unique_layers = np.array([0])
                self._log_execution("Created single-layer model")
            
            self._log_execution(f"Found {len(unique_layers)} geological layers: {unique_layers}")
            
            # Determine information level for uncertainty scaling
            # If layer_params are provided (from natural language), that's high information
            if layer_params is not None and len(layer_params) > 0:
                info_level = 'high'
                self._log_execution(f"Layer-specific parameters provided: {list(layer_params.keys())}")
            else:
                info_level = self._assess_information_level(geological_context, petrophysical_params)
            
            self._log_execution(f"Information level: {info_level}")
            
            # Get layer parameters
            if layer_params is None:
                self._log_execution("Generating layer parameters based on available information")
                layer_params = self._get_layer_params_with_uncertainty(
                    cell_markers, 
                    info_level,
                    petrophysical_params,
                    geological_context
                )
            else:
                # Convert named layer parameters to numeric IDs if needed
                layer_params = self._convert_named_to_numeric_params(
                    layer_params, 
                    unique_layers,
                    info_level
                )
            
            # Get LLM recommendations if available
            if self.api_key and petrophysical_params is None and 'generic' not in geological_context.lower():
                self._log_execution("Requesting LLM recommendations for petrophysical parameters")
                llm_params = self._get_recommended_params(resistivity_array, cell_markers, geological_context)
                if llm_params:
                    layer_params = llm_params
            
            # Monte Carlo simulation
            self._log_execution("Starting Monte Carlo simulation...")
            mc_results = self._run_monte_carlo(
                resistivity_array,
                cell_markers,
                layer_params,
                n_realizations
            )
            
            # Calculate statistics
            water_content_all = mc_results['water_content_all']
            saturation_all = mc_results['saturation_all']
            
            water_content_mean = np.mean(water_content_all, axis=0)
            water_content_std = np.std(water_content_all, axis=0)
            water_content_p10 = np.percentile(water_content_all, 10, axis=0)
            water_content_p50 = np.percentile(water_content_all, 50, axis=0)
            water_content_p90 = np.percentile(water_content_all, 90, axis=0)
            
            saturation_mean = np.mean(saturation_all, axis=0)
            saturation_std = np.std(saturation_all, axis=0)
            
            self._log_execution("Monte Carlo simulation completed")
            
            # Save results
            np.save(os.path.join(output_dir, 'water_content_mean.npy'), water_content_mean)
            np.save(os.path.join(output_dir, 'water_content_std.npy'), water_content_std)
            np.save(os.path.join(output_dir, 'saturation_mean.npy'), saturation_mean)
            np.save(os.path.join(output_dir, 'saturation_std.npy'), saturation_std)
            
            self._log_execution("Results saved to disk")
            
            # Get LLM interpretation
            interpretation = None
            if self.api_key:
                self._log_execution("Generating interpretation of petrophysical results")
                interpretation = self._interpret_results(
                    water_content_mean,
                    water_content_std,
                    layer_params,
                    cell_markers
                )
            
            # Calculate statistics
            wc_mean_overall = np.mean(water_content_mean)
            wc_std_overall = np.mean(water_content_std)
            
            self.results = {
                'status': 'success',
                'water_content_mean': water_content_mean,
                'water_content_std': water_content_std,
                'water_content_p10': water_content_p10,
                'water_content_p50': water_content_p50,
                'water_content_p90': water_content_p90,
                'saturation_mean': saturation_mean,
                'saturation_std': saturation_std,
                'cell_markers': cell_markers,  # Include the markers used for layer-specific analysis
                'layer_params': layer_params,
                'layer_params_used': layer_params,
                'petrophysical_params': petrophysical_params or {},
                'params_used': mc_results['params_used'],
                'statistics': {
                    'mean_water_content': wc_mean_overall,
                    'mean_uncertainty': wc_std_overall,
                    'wc_range': [np.min(water_content_mean), np.max(water_content_mean)],
                    'n_realizations': n_realizations,
                    'n_layers': len(unique_layers)
                },
                'interpretation': interpretation,
                'output_dir': output_dir
            }
            
            self._log_execution(f"Water content range: {self.results['statistics']['wc_range'][0]:.4f} - "
                              f"{self.results['statistics']['wc_range'][1]:.4f}")
            self._log_execution(f"Mean uncertainty: {wc_std_overall:.4f}")
            
            return self.results
            
        except Exception as e:
            self._log_execution(f"Error during petrophysical conversion: {str(e)}", level='ERROR')
            self.results = {
                'status': 'failed',
                'error': str(e)
            }
            raise
    
    def _get_default_layer_params(self, cell_markers: np.ndarray) -> Dict:
        """
        Get default parameters based on cell markers.
        
        Args:
            cell_markers: Array of layer markers
            
        Returns:
            Dictionary of layer parameters
        """
        unique_layers = np.unique(cell_markers)
        layer_params = {}
        
        for i, layer_id in enumerate(unique_layers):
            # Use regolith params for top layer, bedrock for others
            template = 'regolith' if i == 0 else 'bedrock'
            layer_params[int(layer_id)] = self.DEFAULT_LAYER_PARAMS[template].copy()
        
        return layer_params
    
    def _assess_information_level(self, geological_context: str, 
                                   petrophysical_params: Optional[Dict]) -> str:
        """
        Assess the level of information available for parameter estimation.
        
        Args:
            geological_context: Description of geological context
            petrophysical_params: Explicit petrophysical measurements
            
        Returns:
            'high', 'medium', or 'low' information level
        """
        # Log what we received for debugging
        has_petro_params = petrophysical_params is not None and len(petrophysical_params) > 0
        self._log_execution(f"Petrophysical params provided: {has_petro_params}")
        if has_petro_params:
            self._log_execution(f"Parameters: {list(petrophysical_params.keys())}")
        
        if has_petro_params:
            # Explicit field measurements provided - highest confidence
            return 'high'
        elif geological_context and 'generic' not in geological_context.lower():
            # Geological description provided - medium confidence
            if len(geological_context) > 50:  # Detailed description
                return 'medium'
            else:
                return 'low'
        else:
            # Minimal information - lowest confidence, highest uncertainty
            return 'low'
    
    def _convert_named_to_numeric_params(self, layer_params: Dict, 
                                        unique_layers: np.ndarray,
                                        info_level: str) -> Dict:
        """
        Convert named layer parameters (e.g., 'regolith', 'fractured_bedrock') 
        to numeric layer IDs with proper uncertainty structure.
        
        Args:
            layer_params: Dictionary with named layer keys (e.g., {'regolith': {...}, 'fractured_bedrock': {...}})
            unique_layers: Array of numeric layer IDs from cell markers
            info_level: Information level for uncertainty scaling
            
        Returns:
            Dictionary with numeric layer IDs as keys
        """
        # Check if already numeric
        if all(isinstance(k, (int, np.integer)) for k in layer_params.keys()):
            return layer_params
        
        # Mapping for named layers to expected markers
        # From add_velocity_interface: marker 2 = above interface (regolith), marker 3 = below interface (bedrock)
        layer_name_mapping = {
            'regolith': 2,
            'fractured_bedrock': 3,
            'bedrock': 3,
            'background': 1
        }
        
        numeric_params = {}
        uncertainty_scale = {'high': 0.05, 'medium': 0.25, 'low': 0.75}.get(info_level, 0.25)
        
        for layer_name, params in layer_params.items():
            # Get numeric ID for this layer
            layer_id = layer_name_mapping.get(layer_name.lower())
            if layer_id is None or layer_id not in unique_layers:
                # Try to map to available layers
                if len(unique_layers) == 1:
                    layer_id = unique_layers[0]
                elif len(unique_layers) >= 2:
                    # Assume first named layer maps to first numeric layer, etc.
                    idx = list(layer_params.keys()).index(layer_name)
                    layer_id = unique_layers[min(idx, len(unique_layers)-1)]
                else:
                    continue
            
            # Convert range parameters to mean/std format expected by Monte Carlo
            converted = {}
            
            # Handle rho_sat_range
            if 'rho_sat_range' in params:
                rho_min, rho_max = params['rho_sat_range']
                rho_mean = (rho_min + rho_max) / 2
                rho_std = (rho_max - rho_min) / 4  # ~95% within range
                converted['rho_sat'] = {'mean': rho_mean, 'std': rho_std}
                converted['use_rho_sat'] = True
            
            # Handle n_range (cementation exponent)
            if 'n_range' in params:
                n_min, n_max = params['n_range']
                n_mean = (n_min + n_max) / 2
                n_std = (n_max - n_min) / 4
                converted['n'] = {'mean': n_mean, 'std': n_std}
            
            # Handle porosity_range
            if 'porosity_range' in params:
                phi_min, phi_max = params['porosity_range']
                phi_mean = (phi_min + phi_max) / 2
                phi_std = (phi_max - phi_min) / 4
                converted['porosity'] = {'mean': phi_mean, 'std': phi_std}
            
            # Add default sigma_sur (surface conductivity)
            if 'sigma_sur' not in converted:
                converted['sigma_sur'] = {'mean': 0.0, 'std': 0.001}
            
            numeric_params[layer_id] = converted
            
            self._log_execution(f"Mapped '{layer_name}' to layer ID {layer_id}")
            self._log_execution(f"  ρ_sat: {converted.get('rho_sat', {}).get('mean', 'N/A'):.1f} ± {converted.get('rho_sat', {}).get('std', 'N/A'):.1f} Ωm")
            self._log_execution(f"  n: {converted.get('n', {}).get('mean', 'N/A'):.2f} ± {converted.get('n', {}).get('std', 'N/A'):.2f}")
            self._log_execution(f"  φ: {converted.get('porosity', {}).get('mean', 'N/A'):.2f} ± {converted.get('porosity', {}).get('std', 'N/A'):.2f}")
        
        return numeric_params
    
    
    def _guess_params_from_geology(self, geological_context: str, layer_index: int = 0) -> Dict[str, float]:
        """Heuristic petrophysical guesses from geological text. Returns mean values; uncertainty is applied later."""
        ctx = (geological_context or '').lower()
        base_regolith = {'m': 1.5, 'n': 2.0, 'porosity': 0.40, 'sigma_sur': 1/200, 'rho_fluid': 20.0}
        base_bedrock = {'m': 1.8, 'n': 2.0, 'porosity': 0.22, 'sigma_sur': 1/400, 'rho_fluid': 20.0}
        guess = base_regolith if layer_index == 0 else base_bedrock

        if any(k in ctx for k in ['sand', 'sandstone']):
            guess = {'m': 1.3, 'n': 1.8, 'porosity': 0.38, 'sigma_sur': 1/300, 'rho_fluid': 20.0}
        elif any(k in ctx for k in ['clay', 'shale', 'mud']):
            guess = {'m': 1.8, 'n': 2.2, 'porosity': 0.45, 'sigma_sur': 1/100, 'rho_fluid': 20.0}
        elif any(k in ctx for k in ['carbonate', 'limestone', 'dolomite']):
            guess = {'m': 1.7, 'n': 2.0, 'porosity': 0.25, 'sigma_sur': 1/500, 'rho_fluid': 20.0}
        elif any(k in ctx for k in ['fractured', 'bedrock', 'granite', 'basalt']):
            guess = {'m': 2.0, 'n': 2.1, 'porosity': 0.18, 'sigma_sur': 1/600, 'rho_fluid': 20.0}
        elif any(k in ctx for k in ['soil', 'regolith', 'weathered']):
            guess = {'m': 1.5, 'n': 2.0, 'porosity': 0.42, 'sigma_sur': 1/220, 'rho_fluid': 20.0}

        return guess

    def _get_layer_params_with_uncertainty(self, cell_markers: np.ndarray,
                                           info_level: str,
                                           petrophysical_params: Optional[Dict],
                                           geological_context: str) -> Dict:
        """
        Generate layer parameters with uncertainty scaled by information level.
        
        Uncertainty levels:
        - Low info (scenario 1): High uncertainty (std = 50-100% of mean)
        - Medium info (scenario 2): Moderate uncertainty (std = 20-30% of mean)
        - High info (scenario 3): Low uncertainty (std = 5-10% of mean)
        
        Args:
            cell_markers: Array of layer markers
            info_level: 'high', 'medium', or 'low'
            petrophysical_params: Explicit measurements if available
            geological_context: Geological description
            
        Returns:
            Dictionary of layer parameters with appropriate uncertainty
        """
        unique_layers = np.unique(cell_markers)
        layer_params = {}
        
        # Uncertainty multipliers based on information level
        uncertainty_scales = {
            'high': 0.05,   # explicit measurements: tight bounds
            'medium': 0.50,  # geology-described: generous uncertainty
            'low': 1.00     # minimal information: very high uncertainty
        }
        
        scale = uncertainty_scales.get(info_level, 0.75)
        
        # Base parameters vary by information level
        if info_level == 'high' and petrophysical_params:
            # Use explicit measurements
            for i, layer_id in enumerate(unique_layers):
                template = 'regolith' if i == 0 else 'bedrock'
                base = self.DEFAULT_LAYER_PARAMS[template].copy()
                
                # Override with explicit values if provided
                porosity_val = petrophysical_params.get('porosity', base['porosity']['mean'])
                n_val = petrophysical_params.get('n', base['n']['mean'])
                m_val = petrophysical_params.get('m', base['m']['mean'])
                rho_sat_val = petrophysical_params.get('rho_sat', None)
                
                # When rho_sat is provided, use it directly instead of calculating via m
                # This is more accurate: S = (rho_sat / rho)^(1/n)
                if rho_sat_val:
                    # Use much smaller uncertainty for rho_sat since it's a direct measurement
                    # and has larger absolute value (uncertainty propagates non-linearly)
                    rho_sat_uncertainty_scale = scale * 0.1
                    layer_params[int(layer_id)] = {
                        'n': {'mean': n_val, 'std': n_val * scale},
                        'sigma_sur': base['sigma_sur'].copy(),
                        'porosity': {'mean': porosity_val, 'std': porosity_val * scale},
                        'rho_sat': {'mean': rho_sat_val, 'std': rho_sat_val * rho_sat_uncertainty_scale},  # Use rho_sat directly
                        'use_rho_sat': True  # Flag to use rho_sat instead of m
                    }
                else:
                    layer_params[int(layer_id)] = {
                        'm': {'mean': m_val, 'std': m_val * scale},
                        'n': {'mean': n_val, 'std': n_val * scale},
                        'sigma_sur': base['sigma_sur'].copy(),
                        'porosity': {'mean': porosity_val, 'std': porosity_val * scale},
                        'rho_fluid': base['rho_fluid'],
                        'use_rho_sat': False
                    }
                
        elif info_level == 'medium':
            # Geology-informed guess with generous uncertainty
            for idx, layer_id in enumerate(unique_layers):
                base = self._guess_params_from_geology(geological_context, layer_index=idx)
                m_mean = base['m']; n_mean = base['n']; phi_mean = base['porosity']; sigma_sur = base['sigma_sur']
                layer_params[int(layer_id)] = {
                    'm': {'mean': m_mean, 'std': max(abs(m_mean) * scale, 0.3)},
                    'n': {'mean': n_mean, 'std': max(abs(n_mean) * scale, 0.3)},
                    'sigma_sur': {'mean': sigma_sur, 'std': max(abs(sigma_sur), 1/250)},
                    'porosity': {'mean': phi_mean, 'std': max(abs(phi_mean) * scale, 0.08)},
                    'rho_fluid': base.get('rho_fluid', 20.0)
                }
                
        else:  # 'low' information level
            # Minimal information: defaults with very high uncertainty
            for idx, layer_id in enumerate(unique_layers):
                base = self._guess_params_from_geology(geological_context, layer_index=idx)
                m_mean = base['m']; n_mean = base['n']; phi_mean = base['porosity']; sigma_sur = base['sigma_sur']
                layer_params[int(layer_id)] = {
                    'm': {'mean': m_mean, 'std': max(abs(m_mean) * scale, 0.7)},
                    'n': {'mean': n_mean, 'std': max(abs(n_mean) * scale, 0.7)},
                    'sigma_sur': {'mean': sigma_sur, 'std': max(abs(sigma_sur), 1/150)},
                    'porosity': {'mean': phi_mean, 'std': max(abs(phi_mean) * scale, 0.15)},
                    'rho_fluid': base.get('rho_fluid', 20.0)
                }

        self._log_execution(f"Generated parameters with {info_level} information level (std scale: {scale:.0%})")
        
        return layer_params
    
    def _run_monte_carlo(self, resistivity_array: np.ndarray, 
                        cell_markers: np.ndarray,
                        layer_params: Dict,
                        n_realizations: int) -> Dict:
        """
        Run Monte Carlo uncertainty quantification.
        
        Args:
            resistivity_array: Resistivity values (cells × timesteps)
            cell_markers: Layer markers
            layer_params: Parameters for each layer
            n_realizations: Number of MC samples
            
        Returns:
            Dictionary with MC results
        """
        from PyHydroGeophysX.petrophysics.resistivity_models import resistivity_to_saturation
        
        n_cells, n_timesteps = resistivity_array.shape
        unique_layers = np.unique(cell_markers)
        
        # Initialize storage
        water_content_all = np.zeros((n_realizations, n_cells, n_timesteps))
        saturation_all = np.zeros((n_realizations, n_cells, n_timesteps))
        
        # Store parameters used for each realization
        params_used = {int(layer_id): {
            'm': np.zeros(n_realizations),
            'rho_fluid': np.zeros(n_realizations),
            'rho_sat': np.zeros(n_realizations),
            'n': np.zeros(n_realizations),
            'sigma_sur': np.zeros(n_realizations),
            'porosity': np.zeros(n_realizations),
        } for layer_id in unique_layers}
        
        # Monte Carlo loop
        for mc_idx in tqdm(range(n_realizations), desc="MC Realizations"):
            # Sample parameters for each layer
            sampled_params = {}
            for layer_id in unique_layers:
                lp = layer_params[int(layer_id)]
                use_rho_sat = lp.get('use_rho_sat', False)
                
                if use_rho_sat:
                    # Use rho_sat directly - no need for m or rho_fluid
                    sampled = {
                        'rho_sat': max(0.0, np.random.normal(lp['rho_sat']['mean'], lp['rho_sat']['std'])),
                        'n': max(0.0, np.random.normal(lp['n']['mean'], lp['n']['std'])),
                        'sigma_sur': max(0.0, np.random.normal(lp['sigma_sur']['mean'], 
                                                               lp['sigma_sur']['std'])),
                        'porosity': max(0.0, np.random.normal(lp['porosity']['mean'], 
                                                             lp['porosity']['std'])),
                        'use_rho_sat': True
                    }
                else:
                    # Traditional approach using m and rho_fluid
                    sampled = {
                        'm': max(0.0, np.random.normal(lp['m']['mean'], lp['m']['std'])),
                        'n': max(0.0, np.random.normal(lp['n']['mean'], lp['n']['std'])),
                        'sigma_sur': max(0.0, np.random.normal(lp['sigma_sur']['mean'], 
                                                               lp['sigma_sur']['std'])),
                        'porosity': max(0.0, np.random.normal(lp['porosity']['mean'], 
                                                             lp['porosity']['std'])),
                        'rho_fluid': lp['rho_fluid'],
                        'use_rho_sat': False
                    }
                
                sampled_params[layer_id] = sampled
                
                # Store used parameters
                if use_rho_sat:
                    params_used[int(layer_id)]['rho_sat'][mc_idx] = sampled['rho_sat']
                else:
                    params_used[int(layer_id)]['m'][mc_idx] = sampled['m']
                    params_used[int(layer_id)]['rho_fluid'][mc_idx] = sampled['rho_fluid']
                params_used[int(layer_id)]['n'][mc_idx] = sampled['n']
                params_used[int(layer_id)]['sigma_sur'][mc_idx] = sampled['sigma_sur']
                params_used[int(layer_id)]['porosity'][mc_idx] = sampled['porosity']
            
            # Create porosity array
            porosity = np.zeros(n_cells)
            for layer_id in unique_layers:
                mask = cell_markers == layer_id
                porosity[mask] = sampled_params[layer_id]['porosity']
            
            # Process each timestep
            saturation = np.zeros((n_cells, n_timesteps))
            for t in range(n_timesteps):
                resistivity_t = resistivity_array[:, t]
                
                # Process each layer
                for layer_id in unique_layers:
                    mask = cell_markers == layer_id
                    if np.any(mask):
                        params = sampled_params[layer_id]
                        
                        if params.get('use_rho_sat', False):
                            # Direct calculation using rho_sat: S = (rho_sat / rho)^(1/n)
                            rho_sat = params['rho_sat']
                            n = params['n']
                            # Clip saturation to [0, 1]
                            saturation[mask, t] = np.clip((rho_sat / resistivity_t[mask])**(1.0/n), 0.0, 1.0)
                        else:
                            # Traditional approach using resistivity_to_saturation
                            saturation[mask, t] = resistivity_to_saturation(
                                resistivity=resistivity_t[mask],
                                porosity=params['porosity'],
                                m=params['m'],
                                rho_fluid=params['rho_fluid'],
                                n=params['n'],
                                sigma_sur=params['sigma_sur']
                            )
            
            # Convert to water content
            water_content = saturation * porosity[:, np.newaxis]
            
            # Store results
            water_content_all[mc_idx] = water_content
            saturation_all[mc_idx] = saturation
        
        return {
            'water_content_all': water_content_all,
            'saturation_all': saturation_all,
            'params_used': params_used
        }
    
    def _get_recommended_params(self, resistivity_array: np.ndarray,
                               cell_markers: np.ndarray,
                               geological_context: str = '') -> Dict:
        """
        Get LLM recommendations for petrophysical parameters.
        
        Args:
            resistivity_array: Resistivity values
            cell_markers: Layer markers
            geological_context: Geological description for context
            
        Returns:
            Recommended parameters dictionary
        """
        try:
            unique_layers = np.unique(cell_markers)
            res_stats = {
                'mean': np.mean(resistivity_array),
                'std': np.std(resistivity_array),
                'range': [np.min(resistivity_array), np.max(resistivity_array)]
            }
            
            context_info = f"\nGeological Context: {geological_context}" if geological_context else ""
            
            prompt = f"""Recommend petrophysical parameters for converting resistivity to water content:

Resistivity Statistics:
- Mean: {res_stats['mean']:.1f} Ωm
- Range: {res_stats['range'][0]:.1f} - {res_stats['range'][1]:.1f} Ωm
- Number of layers: {len(unique_layers)}{context_info}

Provide parameters for each layer:
1. Cementation exponent (m): 1.3-2.0
2. Saturation exponent (n): 1.5-2.5
3. Porosity: 0.2-0.5
4. Fluid resistivity: ~20 Ωm

For top layer (regolith): Higher porosity, lower m
For bottom layers (bedrock): Lower porosity, higher m"""
            
            response = self.query_llm(prompt, self.system_message,
                                     temperature=0.3, max_tokens=300)
            
            # Use defaults if parsing fails
            return self._get_default_layer_params(cell_markers)
            
        except Exception:
            return self._get_default_layer_params(cell_markers)
    
    def _interpret_results(self, water_content_mean: np.ndarray,
                          water_content_std: np.ndarray,
                          layer_params: Dict,
                          cell_markers: np.ndarray) -> str:
        """
        Get LLM interpretation of petrophysical results.
        
        Args:
            water_content_mean: Mean water content
            water_content_std: Water content uncertainty
            layer_params: Parameters used
            cell_markers: Layer markers
            
        Returns:
            Interpretation string
        """
        try:
            unique_layers = np.unique(cell_markers)
            
            results_summary = f"""
            Petrophysical Conversion Results:
            - Water content range: {np.min(water_content_mean):.4f} - {np.max(water_content_mean):.4f}
            - Mean uncertainty: {np.mean(water_content_std):.4f}
            - Number of layers: {len(unique_layers)}
            """
            
            prompt = f"""Interpret these petrophysical conversion results:

{results_summary}

Provide a brief interpretation (2-3 sentences) about:
1. What the water content values suggest about subsurface hydrology
2. The reliability of the estimates based on uncertainty"""
            
            interpretation = self.query_llm(prompt, self.system_message,
                                           temperature=0.5, max_tokens=200)
            return interpretation
        except Exception:
            return "Petrophysical conversion completed with uncertainty quantification"
    
    def _log_execution(self, message: str, level: str = 'INFO'):
        """Log execution message."""
        print(f"[{self.name}] [{level}] {message}")
