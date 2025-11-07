"""
Context Input Agent for Natural Language Workflow Configuration

Translates user's natural language requests into structured workflow configurations.
Supports multiple LLM providers (OpenAI GPT, Google Gemini, Anthropic Claude).
"""

from typing import Dict, Any, Optional, List
import json
from .base_agent import BaseAgent


class ContextInputAgent(BaseAgent):
    """
    Agent that interprets natural language requests and generates workflow configurations.
    
    This agent uses LLM to understand user intent and create appropriate configuration
    dictionaries for the AgentCoordinator, including parameters for:
    - Data loading (file paths, instruments, CRS)
    - Inversion settings (regularization, iterations, time-lapse mode)
    - Petrophysical parameters
    - Climate data integration
    - Seismic constraints
    - Uncertainty quantification
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4", 
                 llm_provider: str = "openai"):
        """
        Initialize the context input agent.
        
        Args:
            api_key: LLM API key
            model: Model name (e.g., 'gpt-4', 'gemini-pro', 'claude-3-opus')
            llm_provider: Provider ('openai', 'gemini', 'claude')
        """
        super().__init__("ContextInputAgent", api_key, model, llm_provider)
    
    def parse_request(self, user_request: str, available_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Parse natural language request into workflow configuration.
        
        Args:
            user_request: Natural language description of desired workflow
            available_data: Optional dict with available data files, instruments, etc.
            
        Returns:
            Dict containing workflow_config ready for AgentCoordinator
        """
        # Build context for LLM
        context = self._build_context(available_data)
        
        # Create prompt for LLM
        prompt = self._create_parsing_prompt(user_request, context)
        
        # Get LLM response
        response = self.query_llm(prompt)
        
        # Parse JSON from response
        workflow_config = self._extract_config_from_response(response)
        
        # Validate and set defaults
        workflow_config = self._validate_and_complete_config(workflow_config)
        
        return workflow_config
    
    def _build_context(self, available_data: Optional[Dict[str, Any]]) -> str:
        """Build context string about available data and options."""
        context_parts = []
        
        if available_data:
            if 'data_files' in available_data:
                context_parts.append(f"Available data files: {', '.join(available_data['data_files'])}")
            if 'instruments' in available_data:
                context_parts.append(f"Supported instruments: {', '.join(available_data['instruments'])}")
            if 'site_info' in available_data:
                context_parts.append(f"Site information: {json.dumps(available_data['site_info'], indent=2)}")
        
        return "\n".join(context_parts) if context_parts else "No additional context provided."
    
    def _create_parsing_prompt(self, user_request: str, context: str) -> str:
        """Create prompt for LLM to parse user request."""
        prompt = f"""You are an expert geophysicist helping configure an ERT (Electrical Resistivity Tomography) workflow.

User Request:
{user_request}

Available Context:
{context}

Your task is to generate a workflow configuration in JSON format. The configuration should include:

1. **Data source configuration**:
   - data_file: Path to ERT data file
   - project_dir: Project directory path
   - instrument: One of ['E4D', 'Syscal', 'ABEM-Lund', 'Protocol DC', 'BERT', 'Sting', 'ARES', 'Custom']
   - crs: Coordinate system ('local' or 'EPSG:XXXX')

2. **Inversion type** (determine from request):
   - inversion_mode: 'standard' or 'time-lapse'
   - If time-lapse:
     - time_lapse_files: List of data files in temporal order
     - baseline_file: Optional baseline/background file
     - time_lapse_method: 'difference' or 'ratio' or 'joint'
     - temporal_regularization: Optional temporal smoothing weight

3. **Inversion parameters**:
   - lambda: Regularization parameter (default: 20.0, range: 1-100)
   - max_iterations: Maximum iterations (default: 10, range: 5-30)
   - method: Solver method ('cgls' is default)
   - use_gpu: Boolean for GPU acceleration

4. **Petrophysical parameters** (if water content conversion requested):
   - Auto-suggest or use defaults if not specified
   - Include layer-specific parameters if mentioned

5. **Climate data integration** (if mentioned):
   - use_climate: Boolean
   - climate_config with coords, dates, variables, pet_method

6. **Seismic constraints** (if mentioned):
   - use_seismic: Boolean
   - velocity_threshold: Velocity threshold in m/s

7. **Uncertainty quantification**:
   - run_uncertainty: Boolean (default: True if requested)
   - n_realizations: Number of Monte Carlo runs (default: 100)

Important:
- Detect if user wants TIME-LAPSE inversion (keywords: time-lapse, temporal, monitoring, time series, repeated, 4D)
- For time-lapse, set inversion_mode to 'time-lapse' and include time_lapse_files list
- Use reasonable defaults where information is missing
- Return ONLY valid JSON, no explanatory text

Example time-lapse config:
{{
  "inversion_mode": "time-lapse",
  "time_lapse_files": ["data1.ohm", "data2.ohm", "data3.ohm"],
  "baseline_file": "data1.ohm",
  "time_lapse_method": "difference",
  "temporal_regularization": 10.0,
  "data_file": "data1.ohm",
  "project_dir": "data/ERT/E4D",
  "instrument": "E4D",
  "inversion_params": {{"lambda": 20.0, "max_iterations": 10}}
}}

Generate the JSON configuration now:"""
        
        return prompt
    
    def _extract_config_from_response(self, response: str) -> Dict[str, Any]:
        """Extract JSON configuration from LLM response."""
        # Try to find JSON in response
        try:
            # First, try direct parsing
            config = json.loads(response)
            return config
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks
            import re
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                try:
                    config = json.loads(json_match.group(1))
                    return config
                except json.JSONDecodeError:
                    pass
            
            # Try to find anything between { and }
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                try:
                    config = json.loads(json_match.group(0))
                    return config
                except json.JSONDecodeError:
                    pass
        
        # If all fails, return minimal config and log warning
        print(f"⚠️  Warning: Could not parse JSON from LLM response. Using minimal config.")
        return {}
    
    def _validate_and_complete_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate configuration and add missing defaults."""
        # Set defaults
        defaults = {
            'instrument': 'E4D',
            'crs': 'local',
            'inversion_mode': 'standard',  # or 'time-lapse'
            'inversion_params': {
                'lambda': 20.0,
                'max_iterations': 10,
                'method': 'cgls',
                'use_gpu': False
            },
            'use_climate': False,
            'use_seismic': False,
            'run_uncertainty': True,
            'n_realizations': 100,
            'petrophysical_params': {}
        }
        
        # Merge with defaults
        for key, value in defaults.items():
            if key not in config:
                config[key] = value
            elif isinstance(value, dict) and isinstance(config[key], dict):
                # Merge nested dicts
                for subkey, subvalue in value.items():
                    if subkey not in config[key]:
                        config[key][subkey] = subvalue
        
        # Validate time-lapse specific fields
        if config.get('inversion_mode') == 'time-lapse':
            if 'time_lapse_files' not in config or not config['time_lapse_files']:
                print("⚠️  Warning: time-lapse mode requested but no time_lapse_files provided")
                config['inversion_mode'] = 'standard'
            else:
                # Set defaults for time-lapse
                if 'time_lapse_method' not in config:
                    config['time_lapse_method'] = 'difference'
                if 'temporal_regularization' not in config:
                    config['temporal_regularization'] = 10.0
        
        return config
    
    def explain_config(self, config: Dict[str, Any]) -> str:
        """
        Generate human-readable explanation of workflow configuration.
        
        Args:
            config: Workflow configuration dictionary
            
        Returns:
            Formatted explanation string
        """
        prompt = f"""Explain the following ERT workflow configuration in clear, user-friendly language:

Configuration:
{json.dumps(config, indent=2)}

Provide a concise explanation covering:
1. What type of analysis will be performed (standard or time-lapse)
2. Key inversion parameters and their meaning
3. Any special features enabled (climate data, seismic constraints, uncertainty analysis)
4. Expected outputs

Keep it brief (3-5 sentences) and avoid technical jargon where possible."""
        
        explanation = self.query_llm(prompt)
        return explanation
    
    def suggest_improvements(self, config: Dict[str, Any], 
                           site_conditions: Optional[str] = None) -> str:
        """
        Suggest improvements to configuration based on best practices.
        
        Args:
            config: Current workflow configuration
            site_conditions: Optional description of site conditions
            
        Returns:
            Suggestions for improving the configuration
        """
        context = f"\nSite conditions: {site_conditions}" if site_conditions else ""
        
        prompt = f"""As an expert in geophysical inversion, review this ERT workflow configuration and suggest improvements:

Configuration:
{json.dumps(config, indent=2)}
{context}

Provide specific, actionable suggestions for:
1. Inversion parameters (lambda, iterations)
2. Petrophysical parameters (if applicable)
3. Quality control steps
4. Potential issues or missing considerations

Format as a numbered list of 3-5 key suggestions."""
        
        suggestions = self.query_llm(prompt)
        return suggestions
