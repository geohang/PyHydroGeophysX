"""
Petrophysics Agent

Converts resistivity models to hydrological properties (water content, saturation, porosity)
using structure-constrained petrophysical models with Monte Carlo uncertainty quantification.
Implements the workflow from Ex_MC_Hydro.py.
"""

import copy
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .base_agent import BaseAgent


# ---------------------------------------------------------------------------
# Petrophysics Agent
# ---------------------------------------------------------------------------
#: A layer has to hold enough cells for its own parameters to mean anything.
#: Below this a "layer" is a handful of cells whose statistics are noise.
MIN_CELLS_PER_LAYER = 10

#: The parameters of the relationship a user can supply.
_RELATIONSHIP_KEYS = ("rho_sat", "m", "rho_fluid", "n", "porosity")

#: Spread given to a parameter the user did NOT supply when they supplied others:
#: half its value, as for a geology-informed guess - not the 5% a measured value gets.
_UNGIVEN_SCALE = 0.5


def _distribution_of(given: Any, default_mean: float, relative: float,
                     floor: float) -> Dict[str, float]:
    """A parameter's distribution: from the value or ``[low, high]`` the user gave,
    or, when they gave none, the default mean with a generous spread.

    >>> _distribution_of(2.0, 1.7, 0.05, 0.3)
    {'mean': 2.0, 'std': 0.1}
    >>> _distribution_of([0.3, 0.4], 0.42, 0.05, 0.08)
    {'mean': 0.35, 'std': 0.025}
    >>> _distribution_of(None, 2.1, 0.05, 0.3)
    {'mean': 2.1, 'std': 1.05}
    """
    bounds = _as_range(given) if isinstance(given, (list, tuple, str)) else None
    if bounds is not None:
        low, high = bounds
        return {'mean': round((low + high) / 2, 12), 'std': round(abs(high - low) / 4, 12)}
    if given is not None:
        mean = float(given)
        return {'mean': mean, 'std': round(abs(mean) * relative, 12)}
    return {'mean': float(default_mean),
            'std': round(max(abs(float(default_mean)) * _UNGIVEN_SCALE, floor), 12)}


#: The per-layer ranges a request can give, and the parameter each one sets.
LAYER_RANGE_KEYS = (("rho_sat_range", "rho_sat"), ("n_range", "n"),
                    ("porosity_range", "porosity"))


def _as_range(value: Any) -> Optional[Tuple[float, float]]:
    """``(low, high)`` of a two-number range, or None when ``value`` is not one.

    A parsed request can carry ``None``, ``[None, None]`` or a string where a
    range belongs; unpacking those used to stop the whole conversion.

    >>> _as_range([250, 50]), _as_range([None, None]), _as_range("50-250")
    ((50.0, 250.0), None, None)
    """
    if isinstance(value, (str, bytes)):
        return None
    try:
        low, high = (float(bound) for bound in value)
    except (TypeError, ValueError):
        return None
    if not (np.isfinite(low) and np.isfinite(high)):
        return None
    return (low, high) if low <= high else (high, low)


def resolve_layers(cell_markers, n_cells: int,
                   min_cells: int = MIN_CELLS_PER_LAYER):
    """Group cells into geological layers, or into one unit when there is none.

    Cell markers mean different things depending on where the mesh came from. A
    mesh built with regions carries a handful of markers that genuinely separate
    soil from weathered rock from bedrock. An inversion parameter mesh often
    carries one marker per cell, which identifies cells and says nothing about
    geology. Reading the second as layering gives every cell its own
    petrophysical parameter set estimated from one sample.

    The test applied here is whether the markers *group* the mesh: at least two
    values, each covering enough cells to estimate parameters from. When they do
    not, the section is one unit - which is the honest description of a model
    with no structural information, not a failure to find any.

    Parameters
    ----------
    cell_markers : array-like
        One marker per cell, as the mesh reports them.
    n_cells : int
        Number of model cells.
    min_cells : int, optional
        Fewest cells a layer may hold and still be treated as one.

    Returns
    -------
    tuple
        ``(markers, unique_layers, description)`` - the markers to use (all
        zeros for a single unit), their distinct values, and a sentence for the
        log and the report saying which case this is and why.

    Raises
    ------
    None

    Examples
    --------
    >>> import numpy as np
    >>> markers, unique, why = resolve_layers(np.arange(832), 832)
    >>> len(unique), 'single unit' in why
    (1, True)
    >>> markers, unique, why = resolve_layers(np.r_[np.zeros(400), np.ones(432)], 832)
    >>> len(unique), 'layers' in why
    (2, True)
    """
    markers = np.asarray(cell_markers).ravel()
    if markers.size != n_cells:
        markers = np.zeros(n_cells, dtype=int)
        return markers, np.array([0]), (
            "Cell markers did not match the model, so the section is treated as a "
            "single unit with one petrophysical parameter set.")

    unique, counts = np.unique(markers, return_counts=True)
    if unique.size < 2:
        return (markers.astype(int), unique,
                "The mesh reports one region, so the section is a single unit with "
                "one petrophysical parameter set.")
    if counts.min() >= min_cells:
        sizes = ", ".join(str(int(c)) for c in counts)
        return (markers.astype(int), unique,
                f"Using {unique.size} geological layers from the mesh markers "
                f"(cells per layer: {sizes}).")

    # Markers exist but do not group the mesh - typically one per cell, which
    # identifies cells rather than geology.
    detail = (f"{unique.size} markers for {n_cells} cells"
              if unique.size > n_cells // 2 else
              f"the smallest of {unique.size} marker groups holds {int(counts.min())} "
              f"cell(s)")
    return (np.zeros(n_cells, dtype=int), np.array([0]),
            f"The cell markers do not describe geological structure ({detail}), so the "
            f"section is treated as a single unit with one petrophysical parameter set. "
            f"Supplying a layered model is what would let parameters vary with depth.")


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
        # Per layer, the relationship parameters the user supplied (execute).
        self._given_by_layer: Dict[int, set] = {}
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
            from ._uncertainty import describe_prior, realizations

            realization_notes: List[str] = []
            n_realizations = realizations(input_data, note=realization_notes.append)
            seed = int(input_data.get('seed', 7))
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
            
            # Decide whether the markers describe geology or merely number the
            # cells. A fixed "more than 100 is too many" threshold called the
            # latter a failure; the honest reading is that a model with no
            # structural information is one unit.
            cell_markers, unique_layers, layering_note = resolve_layers(cell_markers, n_cells)
            self._log_execution(layering_note)
            
            # Determine information level for uncertainty scaling
            # If layer_params are provided (from natural language), that's high information
            if layer_params is not None and len(layer_params) > 0:
                info_level = 'high'
                self._log_execution(f"Layer-specific parameters provided: {list(layer_params.keys())}")
            else:
                info_level = self._assess_information_level(geological_context, petrophysical_params)
            
            self._log_execution(f"Information level: {info_level}")
            
            # Get layer parameters. ``given`` records, per layer, which of
            # them the user supplied - everything else drawn is a default, and
            # the result says so with the ranges it drew (describe_prior).
            layer_param_notes: List[str] = []
            self._given_by_layer = {}
            global_given = {key for key in _RELATIONSHIP_KEYS
                            if petrophysical_params and petrophysical_params.get(key) is not None}
            requested_layers = layer_params
            if layer_params is None:
                self._log_execution("Generating layer parameters based on available information")
                layer_params = self._get_layer_params_with_uncertainty(
                    cell_markers,
                    info_level,
                    petrophysical_params,
                    geological_context
                )
            else:
                # Convert named layer parameters to numeric IDs if needed. A
                # layer the request did not describe, or a parameter it left
                # out, takes what the conversion would have used without them;
                # a partial set used to stop the conversion outright.
                layer_params, layer_param_notes = self._convert_named_to_numeric_params(
                    layer_params,
                    unique_layers,
                    info_level,
                    lambda: self._get_layer_params_with_uncertainty(
                        cell_markers,
                        self._assess_information_level(geological_context, petrophysical_params),
                        petrophysical_params,
                        geological_context),
                )
                for note in layer_param_notes:
                    self._log_execution(note, level='WARNING')
            
            # Monte Carlo simulation
            self._log_execution("Starting Monte Carlo simulation...")
            mc_results = self._run_monte_carlo(
                resistivity_array,
                cell_markers,
                layer_params,
                n_realizations,
                seed=seed,
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
            if self.llm_enabled:
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

            given = {}
            for layer_id in (int(layer) for layer in unique_layers):
                keys = set(global_given) | set(self._given_by_layer.get(layer_id, ()))
                if (requested_layers and not self._given_by_layer
                        and isinstance(requested_layers.get(layer_id), dict)):
                    # Numeric layer parameters handed over whole, by a script.
                    keys |= {k for k in requested_layers[layer_id] if k in _RELATIONSHIP_KEYS}
                given[layer_id] = sorted(keys)
            prior = describe_prior(mc_results['params_used'], given, n_realizations)
            if prior['relationship'] != 'user':
                self._log_execution(prior['statement'], level='WARNING')
            
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
                # Recorded so the report can say how the section was divided, and
                # why, rather than leaving it to be inferred from a log line.
                'layering': layering_note,
                'n_layers': int(len(unique_layers)),
                'layer_params': layer_params,
                'layer_params_used': layer_params,
                # Where the per-layer parameters were completed or overridden.
                'layer_param_notes': layer_param_notes,
                'petrophysical_params': petrophysical_params or {},
                'params_used': mc_results['params_used'],
                # Whose relationship this was and the ranges actually drawn: a
                # water content means little without them.
                'param_given': given,
                'petrophysical_relationship': prior['relationship'],
                'prior_ranges': prior['ranges'],
                'prior_ranges_text': prior['ranges_text'],
                'prior_statement': prior['statement'],
                'realization_notes': realization_notes,
                'n_realizations': n_realizations,
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
                                        info_level: str,
                                        defaults: Callable[[], Dict]) -> Tuple[Dict, List[str]]:
        """
        Convert named layer parameters (e.g., 'regolith', 'fractured_bedrock')
        to numeric layer IDs with proper uncertainty structure.

        A request may describe only some layers, or only some parameters of a
        layer, and a parsed one may leave a range empty. Every layer of the
        model still needs a complete set, so whatever the request does not give
        comes from ``defaults()`` - the parameters the conversion uses when no
        per-layer values are given - and each substitution is noted.

        Args:
            layer_params: Dictionary with named layer keys (e.g., {'regolith': {...}, 'fractured_bedrock': {...}})
            unique_layers: Array of numeric layer IDs from cell markers
            info_level: Information level for uncertainty scaling
            defaults: Returns complete parameters per numeric layer ID; called
                only when something has to be filled in.

        Returns:
            Dictionary with numeric layer IDs as keys, and notes on what was
            filled in or overridden
        """
        notes: List[str] = []
        layers = [int(layer) for layer in unique_layers]
        cached: Dict[str, Dict] = {}

        def default_for(layer_id: int) -> Dict:
            if 'params' not in cached:
                cached['params'] = defaults()
            return cached['params'].get(layer_id, {})

        # Check if already numeric
        if all(isinstance(k, (int, np.integer)) for k in layer_params.keys()):
            numeric_params = dict(layer_params)
        else:
            numeric_params = self._layers_from_ranges(layer_params, layers, default_for, notes)

        # A layer no parameters were given for keeps the default set; it used
        # to raise a KeyError that ended the conversion.
        for layer_id in layers:
            if layer_id not in numeric_params:
                numeric_params[layer_id] = copy.deepcopy(default_for(layer_id))
                notes.append(f"No layer parameters were given for layer {layer_id} of this "
                             "model, so it used the default parameters.")
        return numeric_params, notes

    def _layers_from_ranges(self, layer_params: Dict, layers: List[int],
                            default_for: Callable[[int], Dict],
                            notes: List[str]) -> Dict:
        """Numeric-ID parameters from ranges given per named layer."""
        # Mapping for named layers to expected markers
        # From add_velocity_interface: marker 2 = above interface (regolith), marker 3 = below interface (bedrock)
        layer_name_mapping = {
            'regolith': 2,
            'fractured_bedrock': 3,
            'bedrock': 3,
            'background': 1
        }

        numeric_params: Dict = {}
        placed: Dict[int, str] = {}
        for position, (layer_name, params) in enumerate(layer_params.items()):
            # Get numeric ID for this layer
            layer_id = layer_name_mapping.get(str(layer_name).lower())
            if layer_id is None or layer_id not in layers:
                # Try to map to available layers
                if len(layers) == 1:
                    layer_id = layers[0]
                elif len(layers) >= 2:
                    # Assume first named layer maps to first numeric layer, etc.
                    layer_id = layers[min(position, len(layers) - 1)]
                else:
                    continue
            if layer_id in placed:
                notes.append(f"Layer parameters for '{placed[layer_id]}' and '{layer_name}' "
                             f"both fall on layer {layer_id} of this model; "
                             f"'{layer_name}' was applied there.")
            placed[layer_id] = layer_name
            params = params if isinstance(params, dict) else {}

            # Convert range parameters to mean/std format expected by Monte Carlo
            converted: Dict[str, Any] = {}
            for range_key, key in LAYER_RANGE_KEYS:
                if range_key not in params:
                    continue
                bounds = _as_range(params[range_key])
                if bounds is None:
                    notes.append(f"The {range_key} given for '{layer_name}' "
                                 f"({params[range_key]!r}) is not a pair of numbers, "
                                 "so it was not used.")
                    continue
                low, high = bounds
                # ~95% within range
                converted[key] = {'mean': (low + high) / 2, 'std': (high - low) / 4}
                if key == 'rho_sat':
                    converted['use_rho_sat'] = True
                self._given_by_layer.setdefault(layer_id, set()).add(key)

            # Add default sigma_sur (surface conductivity)
            if 'sigma_sur' not in converted:
                converted['sigma_sur'] = {'mean': 0.0, 'std': 0.001}

            # What the request left out takes the default for this layer. With
            # no saturated resistivity, the layer follows whichever route its
            # default takes: a rho_sat of its own, or Archie from m and the fluid.
            # The defaults are generated only when something is missing: making
            # them logs a low-information assessment a complete request is not.
            filled = []
            missing = [key for key in ('rho_sat', 'n', 'porosity') if key not in converted]
            default = default_for(layer_id) if missing else {}
            if 'rho_sat' not in converted:
                for key in ('rho_sat', 'use_rho_sat', 'm', 'rho_fluid'):
                    if key in default:
                        converted[key] = copy.deepcopy(default[key])
                filled.append('saturated resistivity')
            for key, label in (('n', 'saturation exponent n'), ('porosity', 'porosity')):
                if key not in converted and key in default:
                    converted[key] = copy.deepcopy(default[key])
                    filled.append(label)
            if filled:
                items = (filled[0] if len(filled) == 1
                         else ", ".join(filled[:-1]) + " or " + filled[-1])
                notes.append(f"'{layer_name}' gave no usable {items}, so layer {layer_id} "
                             f"used the default for {'that' if len(filled) == 1 else 'those'}.")

            numeric_params[layer_id] = converted

            self._log_execution(f"Mapped '{layer_name}' to layer ID {layer_id}")
            for key, label, unit, digits in (('rho_sat', 'ρ_sat', ' Ωm', 1), ('n', 'n', '', 2),
                                             ('porosity', 'φ', '', 2)):
                value = converted.get(key)
                if isinstance(value, dict):
                    self._log_execution(f"  {label}: {value['mean']:.{digits}f} ± "
                                        f"{value['std']:.{digits}f}{unit}")

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
                
                # A value given is used with a tight spread (or the range given);
                # one not given keeps the default with a generous spread. It
                # used to take the default at the given values' 5%, which
                # made a guess look like a measurement.
                def value(key, default_mean, floor, relative=scale):
                    return _distribution_of(petrophysical_params.get(key), default_mean,
                                            relative, floor)

                porosity_val = value('porosity', base['porosity']['mean'], 0.08)
                n_val = value('n', base['n']['mean'], 0.3)
                m_val = value('m', base['m']['mean'], 0.3)
                rho_sat_given = petrophysical_params.get('rho_sat')

                # When rho_sat is provided, use it directly instead of calculating via m
                # With zero surface conductivity, S = (rho_sat / rho)^(1/n).
                if rho_sat_given is not None and rho_sat_given != 0:
                    # Heuristic prior: use one tenth of the generic relative
                    # uncertainty for a supplied rho_sat; this is not a measured error.
                    rho_sat_val = _distribution_of(rho_sat_given, 100.0, scale * 0.1, 0.0)
                    layer_params[int(layer_id)] = {
                        'n': n_val,
                        'sigma_sur': base['sigma_sur'].copy(),
                        'porosity': porosity_val,
                        'rho_sat': rho_sat_val,  # Use rho_sat directly
                        'use_rho_sat': True  # Flag to use rho_sat instead of m
                    }
                else:
                    fluid = petrophysical_params.get('rho_fluid')
                    layer_params[int(layer_id)] = {
                        'm': m_val,
                        'n': n_val,
                        'sigma_sur': base['sigma_sur'].copy(),
                        'porosity': porosity_val,
                        'rho_fluid': (_distribution_of(fluid, 20.0, scale, 0.0)
                                      if fluid is not None else base['rho_fluid']),
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
    
    def _run_monte_carlo(
        self,
        resistivity_array: np.ndarray,
        cell_markers: np.ndarray,
        layer_params: Dict,
        n_realizations: int,
        *,
        seed: int = 7,
    ) -> Dict:
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
        from PyHydroGeophysX.petrophysics.monte_carlo import (
            run_petrophysics_monte_carlo,
        )

        layers = []
        for marker in np.unique(cell_markers):
            parameters = dict(layer_params[int(marker)])
            parameters["marker"] = int(marker)
            layers.append(parameters)
        return run_petrophysics_monte_carlo(
            resistivity_array,
            cell_markers,
            layers,
            products=("water_content",),
            n_realizations=n_realizations,
            seed=seed,
            progress=self._log_execution,
            return_realizations=True,
        )
    
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
