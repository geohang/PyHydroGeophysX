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
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the context input agent (parse natural language request).
        
        Args:
            input_data: Dictionary containing:
                - user_request: Natural language workflow description
                - available_data: Optional dict with available files/instruments
                
        Returns:
            Dictionary containing:
                - status: 'success' or 'failed'
                - workflow_config: Generated configuration
                - explanation: Human-readable explanation
        """
        try:
            user_request = input_data.get('user_request', '')
            available_data = input_data.get('available_data')
            
            if not user_request:
                raise ValueError("user_request is required")
            
            # Parse the request
            workflow_config = self.parse_request(user_request, available_data)
            
            # Generate explanation
            explanation = self.explain_config(workflow_config)
            
            return {
                'status': 'success',
                'workflow_config': workflow_config,
                'explanation': explanation
            }
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    def parse_request(self, user_request: str, available_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Parse natural language request into workflow configuration.
        
        Uses TWO focused prompts for better reliability:
        1. Inversion configuration prompt (ERT-specific parameters)
        2. Climate configuration prompt (meteorological parameters)
        
        Args:
            user_request: Natural language description of desired workflow
            available_data: Optional dict with available data files, instruments, etc.
            
        Returns:
            Dict containing workflow_config ready for AgentCoordinator
        """
        user_request_lower = user_request.lower()
        print("Parsing request with multi-stage extraction:")
        print("  Stage 1: Extracting ERT inversion configuration...")
        
        # Build context for LLM
        context = self._build_context(available_data)
        
        # STAGE 1: Extract inversion configuration
        inversion_prompt = self._create_inversion_prompt(user_request, context)
        inversion_response = self.query_llm(inversion_prompt)
        inversion_config = self._extract_config_from_response(inversion_response)
        
        # STAGE 2: Extract data fusion configuration  
        print("  Stage 2: Extracting data fusion configuration...")
        fusion_prompt = self._create_data_fusion_prompt(user_request)
        fusion_response = self.query_llm(fusion_prompt)
        fusion_config = self._extract_config_from_response(fusion_response)
        
        # STAGE 3: Extract climate configuration
        print("  Stage 3: Extracting climate/site configuration...")
        climate_prompt = self._create_climate_prompt(user_request)
        climate_response = self.query_llm(climate_prompt)
        climate_config = self._extract_config_from_response(climate_response)

        # STAGE 4: Extract hydrological model output configuration (MODFLOW / ParFlow)
        hydro_config = {}
        hydro_keywords = ['modflow', 'parflow', 'par flow', 'watercontent', 'saturation', 'porosity', 'hydrological model']
        if any(kw in user_request_lower for kw in hydro_keywords):
            print("  Stage 4: Extracting hydrological model output configuration...")
            hydro_prompt = self._create_hydro_model_prompt(user_request)
            hydro_response = self.query_llm(hydro_prompt)
            hydro_config = self._extract_config_from_response(hydro_response)
        
        # STAGE 5: Extract TDEM configuration (if electromagnetic mentioned)
        tdem_config = {}
        tdem_keywords = ['tdem', 'tem ', 'time-domain electromagnetic', 'electromagnetic sounding',
                        'loop source', 'transient electromagnetic', 'simpeg']
        if any(kw in user_request.lower() for kw in tdem_keywords):
            print("  Stage 5: Extracting TDEM configuration...")
            tdem_prompt = self._create_tdem_prompt(user_request)
            tdem_response = self.query_llm(tdem_prompt)
            tdem_config = self._extract_config_from_response(tdem_response)
        
        # STAGE 6: Extract seismic configuration (if seismic mentioned without ERT fusion)
        seismic_config = {}
        seismic_keywords = ['seismic refraction', 'srt inversion', 'travel time', 'velocity model',
                           'velocity tomography', 'p-wave', 'first arrival', 'seismic tomography']
        # Only extract seismic config if seismic is mentioned but not as part of ERT fusion
        is_seismic_only = (any(kw in user_request_lower for kw in seismic_keywords) and
                          'ert' not in user_request_lower and 
                          'resistivity' not in user_request_lower and
                          'fusion' not in user_request_lower)
        if is_seismic_only:
            print("  Stage 6: Extracting seismic configuration...")
            seismic_prompt = self._create_seismic_prompt(user_request)
            seismic_response = self.query_llm(seismic_prompt)
            seismic_config = self._extract_config_from_response(seismic_response)
        
        # Merge configurations
        workflow_config = {**inversion_config, **fusion_config, **climate_config, **hydro_config, **tdem_config, **seismic_config}

        # Detect ERT data processing intent (QC/export without inversion)
        processing_keywords = ['data processing', 'quality control', 'qc', 'preprocess', 'export', 'resipy']
        inversion_keywords = ['invert', 'inversion', 'tomography', 'time-lapse', 'timelapse']
        if any(kw in user_request_lower for kw in processing_keywords) and not any(kw in user_request_lower for kw in inversion_keywords):
            workflow_config['ert_data_processing'] = True
        
        # Fallback: Extract petrophysical parameters using regex if LLM missed them
        if not workflow_config.get('petrophysical_params') or len(workflow_config.get('petrophysical_params', {})) == 0:
            params = self._extract_params_regex(user_request)
            if params:
                workflow_config['petrophysical_params'] = params
        
        # Fallback: Extract electrode file using regex if LLM missed it
        if not workflow_config.get('electrode_file'):
            electrode_file = self._extract_electrode_file_regex(user_request)
            if electrode_file:
                workflow_config['electrode_file'] = electrode_file
                print(f"  ⚠️  LLM did not extract electrode file. Using regex fallback: {electrode_file}")

        # Fallback: Extract hydrological model config using regex if LLM missed it
        if any(kw in user_request_lower for kw in ['modflow', 'parflow', 'par flow']):
            hydro_regex = self._extract_hydro_model_regex(user_request)
            for key, value in hydro_regex.items():
                if value is not None and key not in workflow_config:
                    workflow_config[key] = value
        
        # Validate and set defaults
        workflow_config = self._validate_and_complete_config(workflow_config)
        
        # Fallback: Extract ERT data files using regex if LLM missed them
        # Also detect time-lapse mode if multiple files are found
        extracted_files = self._extract_files_regex(user_request)
        if extracted_files and len(extracted_files) > 1:
            # Multiple files suggest time-lapse workflow
            if workflow_config.get('inversion_mode') != 'time-lapse':
                print(f"  ⚠️  Detected {len(extracted_files)} data files in request - setting inversion_mode to 'time-lapse'")
                workflow_config['inversion_mode'] = 'time-lapse'
            
            if not workflow_config.get('time_lapse_files') or len(workflow_config.get('time_lapse_files', [])) == 0:
                workflow_config['time_lapse_files'] = extracted_files
                print(f"  ⚠️  LLM did not extract file list. Using regex fallback: found {len(extracted_files)} files")
        
        # Normalize time-lapse file paths if in time-lapse mode
        if workflow_config.get('inversion_mode') == 'time-lapse':
            workflow_config = self._normalize_timelapse_paths(workflow_config)
        
        # Process and normalize climate configuration if climate data is requested
        if workflow_config.get('use_climate') or workflow_config.get('climate_config'):
            workflow_config = self._normalize_climate_config(workflow_config)
        
        # Process and normalize data fusion configuration if multi-method analysis detected
        if workflow_config.get('fusion_pattern'):
            workflow_config = self._normalize_fusion_config(workflow_config)
        
        print("✓ Multi-stage extraction complete")
        return workflow_config
    
    def _extract_params_regex(self, text: str) -> Dict[str, float]:
        """
        Fallback method to extract petrophysical parameters using regex.
        Looks for patterns like: rho_sat = 541, porosity = 0.37, n = 1.24, m = 1.5
        """
        import re
        params = {}
        
        # Pattern to match: parameter_name = number (with optional decimal)
        # Common parameter names: rho_sat, porosity, phi, n, m
        patterns = {
            'rho_sat': r'rho_sat\s*=\s*([\d.]+)',
            'porosity': r'porosity\s*=\s*([\d.]+)',
            'phi': r'phi\s*=\s*([\d.]+)',
            'n': r'\bn\s*=\s*([\d.]+)',  # \b for word boundary
            'm': r'\bm\s*=\s*([\d.]+)',
        }
        
        for param_name, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    value = float(match.group(1))
                    # Map phi to porosity
                    if param_name == 'phi':
                        params['porosity'] = value
                    else:
                        params[param_name] = value
                except ValueError:
                    pass
        
        return params
    
    def _extract_electrode_file_regex(self, text: str) -> str:
        """
        Fallback method to extract electrode file path using regex.
        Looks for patterns like:
        - "electrode file in path/to/electrodes.dat"
        - "electrode file: electrodes.dat"
        - "electrodes at path/electrodes.dat"
        """
        import re
        
        # Pattern 1: "electrode file in/at/: <path>"
        patterns = [
            r'electrode[s]?\s+file\s+(?:in|at|:)\s+([^\s,]+\.dat)',
            r'electrode[s]?\s+(?:in|at|:)\s+([^\s,]+\.dat)',
            r'using\s+electrode[s]?\s+file\s+([^\s,]+\.dat)',
            r'with\s+electrode[s]?\s+file\s+([^\s,]+\.dat)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None
    
    def _extract_files_regex(self, text: str) -> list:
        """
        Fallback method to extract ERT data file names using regex.
        Looks for common ERT file patterns like:
        - *.ohm (E4D format)
        - *.dat (various formats)
        - *.Data (DAS format)
        - Files with dates in names (e.g., 2021-10-08_1400.ohm)
        """
        import re
        files = []
        
        # Pattern to match common ERT file extensions
        # Look for lines with bullet points or dashes followed by filenames
        patterns = [
            r'[-*•]\s+([^\s]+\.ohm)',  # Bullet + .ohm files
            r'[-*•]\s+([^\s]+\.dat)',  # Bullet + .dat files
            r'[-*•]\s+([^\s]+\.Data)',  # Bullet + .Data files
            r'(\d{4}-\d{2}-\d{2}[^\s]*\.ohm)',  # Date-based .ohm files (anywhere)
            r'(\d{4}-\d{2}-\d{2}[^\s]*\.dat)',  # Date-based .dat files (anywhere)
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.MULTILINE | re.IGNORECASE)
            for match in matches:
                if match not in files:  # Avoid duplicates
                    files.append(match)
        
        return files

    def _extract_hydro_model_regex(self, text: str) -> Dict[str, Any]:
        """
        Extract MODFLOW/ParFlow configuration using regex as a fallback.

        Looks for:
        - modflow_dir / parflow_dir / model_directory
        - run_name / model_name
        - idomain file
        - timestep / nlay / plot_layer
        """
        import re

        def _find_path(patterns: list) -> Optional[str]:
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    value = match.group(1).strip().strip('"').strip("'")
                    return value
            return None

        modflow_dir = _find_path([
            r'modflow\s+(?:dir|directory|folder|path)\s*[:=]\s*([^\s,]+)',
            r'modflow\s+workspace\s*[:=]\s*([^\s,]+)',
        ])
        parflow_dir = _find_path([
            r'parflow\s+(?:dir|directory|folder|path)\s*[:=]\s*([^\s,]+)',
            r'par\s*flow\s+(?:dir|directory|folder|path)\s*[:=]\s*([^\s,]+)',
        ])
        model_directory = _find_path([
            r'model\s+(?:dir|directory|folder|path)\s*[:=]\s*([^\s,]+)'
        ])

        run_name = _find_path([r'run[_\s]*name\s*[:=]\s*([^\s,]+)'])
        model_name = _find_path([r'model[_\s]*name\s*[:=]\s*([^\s,]+)'])
        idomain_file = _find_path([
            r'idomain\s*(?:file|path)?\s*[:=]\s*([^\s,]+)',
            r'\b(id\.txt)\b'
        ])

        def _find_int(pattern: str) -> Optional[int]:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    return int(match.group(1))
                except ValueError:
                    return None
            return None

        timestep = _find_int(r'timestep\s*[:=]\s*(\d+)')
        nlay = _find_int(r'nlay\s*[:=]\s*(\d+)')
        plot_layer = _find_int(r'plot\s*layer\s*[:=]\s*(\d+)')

        hydro_model = None
        if "modflow" in text.lower() and "parflow" in text.lower():
            hydro_model = "both"
        elif "modflow" in text.lower():
            hydro_model = "modflow"
        elif "parflow" in text.lower() or "par flow" in text.lower():
            hydro_model = "parflow"

        return {
            "hydro_model": hydro_model,
            "modflow_dir": modflow_dir,
            "parflow_dir": parflow_dir,
            "model_directory": model_directory,
            "run_name": run_name,
            "model_name": model_name,
            "idomain_file": idomain_file,
            "timestep": timestep,
            "nlay": nlay,
            "plot_layer": plot_layer,
        }
    
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
    
    def _create_inversion_prompt(self, user_request: str, context: str) -> str:
        """Create focused prompt for ERT inversion configuration only."""
        prompt = f"""You are an expert geophysicist configuring an ERT (Electrical Resistivity Tomography) inversion.

User Request:
{user_request}

Available Context:
{context}

Extract ONLY ERT inversion configuration in JSON format:

1. **Data source**:
   - data_file: Path to ERT data file (REQUIRED - extract from text)
   - project_dir: Project directory path (extract folder path, e.g., "data/ERT/E4D")
   - instrument: One of ['DAS-1', 'Syscal', 'ABEM-Lund', 'Protocol DC', 'BERT', 'Sting', 'ARES', 'E4D', 'Custom']
     * Match EXACT instrument name from request
     * "E4D" → use "E4D", "DAS" → use "DAS-1", "Syscal" → use "Syscal"
   - crs: Coordinate system ('local' or 'EPSG:XXXX')

2. **Inversion type**:
   - inversion_mode: 'standard' OR 'time-lapse'
   - If time-lapse (keywords: time-lapse, temporal, monitoring, 4D, repeated surveys):
     * time_lapse_files: List of ALL data files in temporal order (extract all filenames)
     * time_lapse_method: 'difference', 'ratio', or 'joint'
     * temporal_regularization: Temporal smoothing weight (extract if mentioned)

3. **Inversion parameters**:
   - lambda: Regularization parameter (extract if mentioned, default: 15-20)
   - max_iterations: Maximum iterations (extract if mentioned, default: 10)
   - method: Solver method (extract if mentioned, default: 'cgls')
   - use_gpu: Boolean for GPU acceleration

4. **Petrophysical parameters** (ONLY if water content conversion mentioned):
   - Extract explicit values: rho_sat, porosity, n, m
   - Example: "rho_sat = 541" → {{"rho_sat": 541}}
   - Return empty dict {{}} if not mentioned

IMPORTANT:
- Focus ONLY on inversion/ERT parameters
- DO NOT extract climate, meteorological, or site information
- Return ONLY valid JSON, no explanatory text
- For time-lapse: extract ALL filenames mentioned in numbered lists or bullet points

Example output:
{{
  "inversion_mode": "time-lapse",
  "time_lapse_files": ["2021-10-08_1400.ohm", "2021-11-08_1230.ohm", "2021-12-08_1230.ohm"],
  "time_lapse_method": "difference",
  "temporal_regularization": 15.0,
  "data_file": "2021-10-08_1400.ohm",
  "project_dir": "data/ERT/E4D",
  "instrument": "E4D",
  "inversion_params": {{"lambda": 15.0, "max_iterations": 10, "method": "cgls"}}
}}

Generate JSON now:"""
        return prompt
    
    def _create_data_fusion_prompt(self, user_request: str) -> str:
        """Create focused prompt for data fusion configuration."""
        prompt = f"""You are extracting multi-method geophysical data fusion configuration from a request.

User Request:
{user_request}

Extract ONLY data fusion parameters in JSON format:

1. **Data fusion type** (if multi-method analysis mentioned):
   - fusion_pattern: 'full_integration', 'structure_constraint', 'cross_validation', or null
   - methods: List of methods ['seismic', 'ert', 'petrophysics']
   
2. **Seismic data** (if mentioned):
   - seismic_file: Path to seismic data file
   - velocity_threshold: Velocity threshold for interface extraction (m/s)
   - seismic_params:
     * lam: Regularization (default: 50)
     * zWeight: Vertical smoothing weight (default: 0.2)
     * vTop: Top layer velocity constraint (default: 500)
     * vBottom: Bottom layer velocity constraint (default: 5000)
     
3. **ERT data** (if mentioned):
   - ert_file: Path to ERT data file
   - ert_params:
     * lambda: Regularization parameter (extract if mentioned)
     * max_iterations: Maximum iterations (default: 20)
     * limits: [min_rho, max_rho] resistivity limits
     
4. **Mesh parameters** (if mentioned):
   - mesh_quality: Mesh quality parameter (default: 31)
   - mesh_params:
     * paraDepth: Parameter domain depth
     * paraDX: Parameter spacing
     * paraMaxCellSize: Maximum cell size
     
5. **Petrophysical parameters** (if mentioned):
   - n_realizations: Number of Monte Carlo realizations (default: 100)
   - layer_params: Layer-specific parameters dict
     * Extract for each layer (e.g., "regolith", "fractured_bedrock"):
       - rho_sat_range: [min, max] (Ωm)
       - n_range: [min, max] (cementation exponent)
       - porosity_range: [min, max] (-)
       
6. **Output settings**:
   - output_dir: Output directory path
   - coverage_threshold: Coverage threshold for plotting (default: -1.0)

IMPORTANT:
- Focus ONLY on data fusion parameters
- Return null for fusion_pattern if NOT multi-method
- Return ONLY valid JSON, no explanatory text
- Extract layer names exactly as mentioned (e.g., "regolith", "fractured_bedrock")

Example output (full data fusion):
{{
  "fusion_pattern": "full_integration",
  "methods": ["seismic", "ert", "petrophysics"],
  "seismic_file": "data/Seismic/srtfieldline2.dat",
  "ert_file": "data/ERT/Bert/fielddataline2.dat",
  "velocity_threshold": 1000,
  "seismic_params": {{"lam": 50, "zWeight": 0.2, "vTop": 500, "vBottom": 5000}},
  "ert_params": {{"lambda": 20, "max_iterations": 20, "limits": [1.0, 10000.0]}},
  "mesh_quality": 31,
  "n_realizations": 100,
  "layer_params": {{
    "regolith": {{
      "rho_sat_range": [50, 250],
      "n_range": [1.3, 2.2],
      "porosity_range": [0.25, 0.35]
    }},
    "fractured_bedrock": {{
      "rho_sat_range": [165, 350],
      "n_range": [2.0, 2.2],
      "porosity_range": [0.2, 0.3]
    }}
  }},
  "output_dir": "results/data_fusion_field",
  "coverage_threshold": -1.0
}}

Example output (no fusion):
{{
  "fusion_pattern": null
}}

Generate JSON now:"""
        return prompt
    
    def _create_climate_prompt(self, user_request: str) -> str:
        """Create focused prompt for climate data configuration only."""
        prompt = f"""You are extracting climate/meteorological data requirements from a geophysical monitoring request.

User Request:
{user_request}

Extract ONLY climate data configuration in JSON format:

1. **Climate data required?**
   - use_climate: true if request mentions: climate, weather, precipitation, temperature, ET, evapotranspiration, meteorological
   - use_climate: false otherwise

2. **If use_climate = true**, extract climate_config:
   - coords: [longitude, latitude] in decimal degrees
     * Extract from text: "38.92584°N, -106.97998°W" → [-106.97998, 38.92584]
     * Longitude first (negative for W), latitude second (positive for N)
   - dates: [start_date, end_date] in "YYYY-MM-DD" format
     * Extract explicit date range if mentioned
     * Format: "September 2021 to March 2022" → ["2021-09-01", "2022-03-31"]
   - variables: List of climate variables mentioned
     * Common: ["prcp", "tmin", "tmax", "srad", "vp", "dayl"]
     * Extract if explicitly requested
   - pet_method: PET calculation method if mentioned
     * "Penman-Monteith" → "penman_monteith"
     * "Hargreaves" → "hargreaves"
   - time_scale: "daily" or "monthly" (extract if mentioned)
   - antecedent_days: List of antecedent periods if mentioned
     * "7-day and 14-day cumulative" → [7, 14]

3. **Site information** (if provided):
   - name: Site name
   - location: Location description
   - coordinates: Formatted coordinate string (e.g., "38.92584°N, -106.97998°W")
   - elevation: Elevation with units (e.g., "3,150 meters")

IMPORTANT:
- Focus ONLY on climate/meteorological/site parameters
- DO NOT extract inversion or ERT parameters
- If no climate data mentioned, return {{"use_climate": false}}
- Return ONLY valid JSON, no explanatory text

Example output (climate requested):
{{
  "use_climate": true,
  "climate_config": {{
    "coords": [-106.97998, 38.92584],
    "dates": ["2021-09-01", "2022-03-31"],
    "variables": ["prcp", "tmin", "tmax", "srad", "dayl"],
    "pet_method": "penman_monteith",
    "time_scale": "daily",
    "antecedent_days": [7, 14]
  }},
  "site_info": {{
    "name": "Mt. Snodgrass Monitoring Site",
    "location": "Near Crested Butte, Colorado, USA",
    "coordinates": "38.92584°N, -106.97998°W",
    "elevation": "3,150 meters"
  }}
}}

Example output (no climate):
{{
  "use_climate": false
}}

Generate JSON now:"""
        return prompt

    def _create_hydro_model_prompt(self, user_request: str) -> str:
        """Create focused prompt for hydrological model output configuration (MODFLOW / ParFlow)."""
        prompt = f"""You are extracting hydrological model output loading configuration.

User Request:
{user_request}

Extract ONLY hydrological model output parameters in JSON format:

1. **Model type**:
   - hydro_model: "modflow", "parflow", "both", or null

2. **Directories / paths**:
   - modflow_dir: Path to MODFLOW model directory
   - parflow_dir: Path to ParFlow model directory
   - model_directory: If a shared model directory is given, put it here
   - idomain_file: Path to MODFLOW idomain/id.txt file (if mentioned)

3. **Model identifiers**:
   - model_name: MODFLOW model name (for porosity via flopy)
   - run_name: ParFlow run name (prefix for .out.satur.*.pfb files)

4. **Load options**:
   - timestep: MODFLOW timestep index (integer)
   - parflow_timestep: ParFlow timestep index (integer)
   - nlay: Number of MODFLOW layers (integer)
   - plot_layer: Layer index to visualize (integer)
   - load_water_content: true/false
   - load_porosity: true/false
   - load_saturation: true/false

IMPORTANT:
- Focus ONLY on model output loading (not inversion/ERT/seismic)
- Return ONLY valid JSON, no explanatory text
- If a field is not mentioned, omit it or set to null

Example output (MODFLOW):
{{
  "hydro_model": "modflow",
  "modflow_dir": "examples/data/modflow",
  "idomain_file": "examples/data/modflow/id.txt",
  "model_name": "TLnewtest2sfb2",
  "timestep": 1,
  "nlay": 3,
  "plot_layer": 0,
  "load_water_content": true,
  "load_porosity": true
}}

Example output (ParFlow):
{{
  "hydro_model": "parflow",
  "parflow_dir": "examples/data/parflow/test2",
  "run_name": "test2",
  "parflow_timestep": 0,
  "plot_layer": 19,
  "load_saturation": true,
  "load_porosity": true
}}

Generate JSON now:"""
        return prompt
    
    def _create_tdem_prompt(self, user_request: str) -> str:
        """Create focused prompt for TDEM configuration extraction."""
        prompt = f"""You are extracting Time-Domain Electromagnetic (TDEM) configuration from a request.

User Request:
{user_request}

Extract ONLY TDEM parameters in JSON format:

1. **TDEM Mode**:
   - tdem_mode: 'inversion', 'forward', or 'hydro_to_tdem'
   
2. **Data source** (for inversion mode):
   - tdem_file: Path to TDEM data file (.txt format with columns: TIME BZ UNCERTAINTY)
   
3. **Survey parameters**:
   - source_radius: Loop radius in meters (extract if mentioned, default: 10)
   - times: Time channels array (if specified, e.g., "10µs to 10ms")
   
4. **Inversion parameters** (for inversion mode):
   - n_layers: Number of layers (extract if mentioned, default: 20)
   - min_thickness: Minimum layer thickness in meters (default: 0.5)
   - max_thickness: Maximum layer thickness in meters (default: 10)
   - starting_conductivity: Starting conductivity in S/m (default: 0.001)
   - use_irls: Use sparse regularization (default: true)
   - max_iterations: Maximum iterations (default: 50)

5. **Forward modeling parameters** (for forward mode):
   - thicknesses: Layer thicknesses array (meters)
   - conductivity: Layer conductivities array (S/m)
   - noise_level: Noise level as fraction (default: 0.05 = 5%)

6. **Hydro-to-TDEM parameters** (for hydro_to_tdem mode):
   - water_content: Path to water content data
   - porosity: Path to porosity data
   - layer_thicknesses: Layer thicknesses
   - Petrophysical parameters:
     * sigma_w: Pore water conductivity (S/m, default: 0.05)
     * m: Cementation exponent (default: 1.5)
     * n: Saturation exponent (default: 2.0)

IMPORTANT:
- Only extract TDEM-related parameters
- Return {{"tdem_mode": null}} if no TDEM processing mentioned
- Return ONLY valid JSON, no explanatory text

Example output (inversion):
{{
  "tdem_mode": "inversion",
  "tdem_file": "tdem_synthetic_data.txt",
  "source_radius": 10.0,
  "n_layers": 20,
  "use_irls": true,
  "max_iterations": 50
}}

Example output (no TDEM):
{{
  "tdem_mode": null
}}

Generate JSON now:"""
        return prompt
    
    def _create_seismic_prompt(self, user_request: str) -> str:
        """Create focused prompt for seismic refraction configuration extraction."""
        prompt = f"""You are extracting seismic refraction tomography (SRT) configuration from a request.

User Request:
{user_request}

Extract ONLY seismic parameters in JSON format:

1. **Data source**:
   - seismic_file: Path to seismic travel time data file (.dat format)
   
2. **Inversion parameters**:
   - lam: Lambda/regularization parameter (extract if mentioned, default: 50)
   - z_weight: Vertical smoothing weight (extract if mentioned, default: 0.2)
   - v_top: Top velocity constraint in m/s (extract if mentioned, default: 500)
   - v_bottom: Bottom velocity constraint in m/s (extract if mentioned, default: 5000)
   - para_depth: Parametric depth in meters (extract if mentioned, default: 30)
   - velocity_limits: [min, max] velocity limits in m/s (default: [300, 8000])
   
3. **Interface extraction**:
   - extract_interfaces: Whether to extract velocity interfaces (default: true)
   - velocity_threshold: Main threshold for interface extraction in m/s (extract if mentioned, default: 1200)
   - velocity_thresholds: List of thresholds for multiple interfaces (e.g., [1200, 3000, 5000])
   
4. **Geological interpretation hints**:
   - < 1200 m/s: Regolith/weathered soil
   - 1200-3000 m/s: Fractured rock
   - > 3000 m/s: Competent bedrock

IMPORTANT:
- Only extract seismic-related parameters
- Look for velocity values mentioned in the request
- Return {{"seismic_only": false}} if no seismic processing mentioned
- Set "seismic_only": true for standalone seismic workflows
- Return ONLY valid JSON, no explanatory text

Example output (seismic inversion):
{{
  "seismic_only": true,
  "seismic_file": "synthetic_seismic_data.dat",
  "lam": 50,
  "z_weight": 0.2,
  "v_top": 500,
  "v_bottom": 5000,
  "para_depth": 30,
  "extract_interfaces": true,
  "velocity_thresholds": [1200, 5000]
}}

Example output (no seismic):
{{
  "seismic_only": false
}}

Generate JSON now:"""
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
        # First, flatten nested structures if present
        # Handle 'data_source' nested structure
        if 'data_source' in config and isinstance(config['data_source'], dict):
            data_source = config.pop('data_source')
            # Extract nested fields to top level if not already present
            for key in ['instrument', 'data_file', 'electrode_file', 'project_dir', 'crs']:
                if key in data_source and key not in config:
                    config[key] = data_source[key]
        
        # Handle 'inversion_type' nested structure
        if 'inversion_type' in config and isinstance(config['inversion_type'], dict):
            inversion_type = config.pop('inversion_type')
            if 'inversion_mode' in inversion_type and 'inversion_mode' not in config:
                config['inversion_mode'] = inversion_type['inversion_mode']
        
        # Handle 'uncertainty_quantification' nested structure
        if 'uncertainty_quantification' in config and isinstance(config['uncertainty_quantification'], dict):
            uncertainty = config.pop('uncertainty_quantification')
            if 'run_uncertainty' in uncertainty and 'run_uncertainty' not in config:
                config['run_uncertainty'] = uncertainty['run_uncertainty']
            if 'n_realizations' in uncertainty and 'n_realizations' not in config:
                config['n_realizations'] = uncertainty['n_realizations']
        
        # Set defaults
        defaults = {
            'instrument': 'Syscal',  # Changed from E4D to more common instrument
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
    
    def _normalize_timelapse_paths(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize file paths for time-lapse inversion workflows.
        
        When time-lapse mode is detected, this method ensures all file paths
        are properly formatted with the correct base directory and that filenames
        are extracted from any full paths provided.
        
        Args:
            config: Workflow configuration with time-lapse files
            
        Returns:
            Updated configuration with normalized paths
        """
        from pathlib import Path
        
        # Extract or set base data directory
        if 'project_dir' in config:
            base_data_dir = Path(config['project_dir'])
        else:
            # Try to infer from data_file or time_lapse_files
            if 'data_file' in config:
                base_data_dir = Path(config['data_file']).parent
            elif 'time_lapse_files' in config and config['time_lapse_files']:
                base_data_dir = Path(config['time_lapse_files'][0]).parent
            else:
                # Default to common E4D location
                base_data_dir = Path('data/ERT/E4D')
        
        # Clean up project_dir (remove leading ./)
        if str(base_data_dir).startswith('./'):
            base_data_dir = Path(str(base_data_dir)[2:])
        
        # Ensure project_dir is set
        config['project_dir'] = str(base_data_dir)
        
        # Normalize baseline_file if present
        if 'baseline_file' in config:
            baseline_name = Path(config['baseline_file']).name
            config['baseline_file'] = str(base_data_dir / baseline_name)
            # Set data_file to match baseline
            config['data_file'] = config['baseline_file']
        elif 'data_file' in config:
            # Use data_file as baseline
            data_file_name = Path(config['data_file']).name
            config['data_file'] = str(base_data_dir / data_file_name)
            config['baseline_file'] = config['data_file']
        
        # Normalize time_lapse_files list
        if 'time_lapse_files' in config and config['time_lapse_files']:
            normalized_files = []
            for file_path in config['time_lapse_files']:
                # Extract just the filename
                filename = Path(file_path).name
                # Construct full path with base directory
                full_path = str(base_data_dir / filename)
                normalized_files.append(full_path)
            
            config['time_lapse_files'] = normalized_files
            
            # If baseline_file not set, use first time-lapse file
            if 'baseline_file' not in config:
                config['baseline_file'] = config['time_lapse_files'][0]
                config['data_file'] = config['baseline_file']
        
        # Normalize electrode_file if present
        if 'electrode_file' in config:
            electrode_name = Path(config['electrode_file']).name
            config['electrode_file'] = str(base_data_dir / electrode_name)
        
        return config
    
    def _normalize_climate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize and validate climate data configuration.
        
        Ensures climate_config has all required fields with proper formatting.
        If dates are not specified, infers them from ERT survey dates.
        
        Args:
            config: Workflow configuration with climate settings
            
        Returns:
            Updated configuration with normalized climate_config
        """
        from pathlib import Path
        from datetime import datetime, timedelta
        
        # Ensure use_climate flag is set
        if 'climate_config' in config and not config.get('use_climate'):
            config['use_climate'] = True
        
        # Initialize climate_config if not present
        if 'climate_config' not in config:
            config['climate_config'] = {}
        
        climate_cfg = config['climate_config']
        
        # Set default climate variables if not specified
        if 'variables' not in climate_cfg:
            climate_cfg['variables'] = ["prcp", "tmin", "tmax", "srad", "vp", "dayl"]
        
        # Set default PET method
        if 'pet_method' not in climate_cfg:
            climate_cfg['pet_method'] = "penman_monteith"
        
        # Set default time scale
        if 'time_scale' not in climate_cfg:
            climate_cfg['time_scale'] = "daily"
        
        # Set default region
        if 'region' not in climate_cfg:
            climate_cfg['region'] = "na"
        
        # Set default antecedent days
        if 'antecedent_days' not in climate_cfg:
            climate_cfg['antecedent_days'] = [1, 3, 7, 14, 30]
        
        # Set default CRS
        if 'crs' not in climate_cfg:
            climate_cfg['crs'] = 4326  # WGS84
        
        # Infer dates from ERT survey if not provided
        if 'dates' not in climate_cfg or not climate_cfg['dates']:
            dates = self._infer_climate_dates(config)
            if dates:
                climate_cfg['dates'] = dates
        
        # Set default output path if not specified
        if 'output' not in climate_cfg:
            # Use site name if available, otherwise default
            site_name = config.get('site_info', {}).get('name', 'site').lower().replace(' ', '_')
            climate_cfg['output'] = f"data/climate/{site_name}_climate.csv"
        
        # Ensure output directory exists in path
        output_path = Path(climate_cfg['output'])
        climate_cfg['output'] = str(output_path)
        
        return config
    
    def _infer_climate_dates(self, config: Dict[str, Any]) -> Optional[List[str]]:
        """
        Infer climate date range from ERT survey dates.
        
        Args:
            config: Workflow configuration
            
        Returns:
            [start_date, end_date] in "YYYY-MM-DD" format, or None if cannot infer
        """
        from datetime import datetime, timedelta
        import re
        
        dates = []
        
        # Try to extract dates from time-lapse files
        if config.get('inversion_mode') == 'time-lapse' and config.get('time_lapse_files'):
            for filepath in config['time_lapse_files']:
                # Look for date patterns: YYYY-MM-DD, YYYYMMDD, YYYY_MM_DD
                date_match = re.search(r'(\d{4})[_-]?(\d{2})[_-]?(\d{2})', filepath)
                if date_match:
                    try:
                        date_str = f"{date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)}"
                        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
                        dates.append(date_obj)
                    except ValueError:
                        continue
        
        # If we found dates, extend range by buffer period
        if dates:
            min_date = min(dates)
            max_date = max(dates)
            
            # Add 1 month buffer before and after
            start_date = min_date - timedelta(days=30)
            end_date = max_date + timedelta(days=30)
            
            return [start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")]
        
        return None
    
    def _normalize_fusion_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize and validate data fusion configuration.
        
        Ensures all fusion-related parameters are properly formatted and
        have sensible defaults.
        
        Args:
            config: Workflow configuration with fusion settings
            
        Returns:
            Updated configuration with normalized fusion parameters
        """
        from pathlib import Path
        
        # Set default fusion parameters
        fusion_defaults = {
            'coverage_threshold': -1.0,
            'mesh_quality': 31,
            'n_realizations': 100
        }
        
        for key, value in fusion_defaults.items():
            if key not in config:
                config[key] = value
        
        # Normalize file paths
        if 'seismic_file' in config:
            seismic_path = Path(config['seismic_file'])
            if not seismic_path.is_absolute():
                config['seismic_file'] = str(seismic_path)
        
        if 'ert_file' in config:
            ert_path = Path(config['ert_file'])
            if not ert_path.is_absolute():
                config['ert_file'] = str(ert_path)
        
        # Set default seismic parameters
        if 'seismic_params' not in config:
            config['seismic_params'] = {}
        
        seismic_defaults = {
            'lam': 50,
            'zWeight': 0.2,
            'vTop': 500,
            'vBottom': 5000
        }
        
        for key, value in seismic_defaults.items():
            if key not in config['seismic_params']:
                config['seismic_params'][key] = value
        
        # Set default ERT parameters for fusion
        if 'ert_params' not in config:
            config['ert_params'] = {}
        
        ert_defaults = {
            'lambda': 20,
            'max_iterations': 20,
            'limits': [1.0, 10000.0]
        }
        
        for key, value in ert_defaults.items():
            if key not in config['ert_params']:
                config['ert_params'][key] = value
        
        # Set default mesh parameters
        if 'mesh_params' not in config:
            config['mesh_params'] = {}
        
        mesh_defaults = {
            'paraDepth': 22.0,
            'paraDX': 0.5,
            'paraMaxCellSize': 5
        }
        
        for key, value in mesh_defaults.items():
            if key not in config['mesh_params']:
                config['mesh_params'][key] = value
        
        # Set default output directory
        if 'output_dir' not in config:
            config['output_dir'] = 'results/data_fusion'
        
        # Ensure methods list exists
        if 'methods' not in config:
            # Infer from available data
            methods = []
            if config.get('seismic_file'):
                methods.append('seismic')
            if config.get('ert_file') or config.get('data_file'):
                methods.append('ert')
            if config.get('layer_params') or config.get('n_realizations'):
                methods.append('petrophysics')
            config['methods'] = methods
        
        return config
    
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
