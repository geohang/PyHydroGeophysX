"""
Base Agent Class for Multi-Agent System

Provides the foundation for all specialized agents in the workflow.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import os
import json
from pathlib import Path
import numpy as np


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the multi-agent system.
    
    Each agent is specialized for a specific task and can communicate
    with other agents through the coordinator.
    """
    
    def __init__(self, name: str, api_key: Optional[str] = None, model: Optional[str] = None, 
                 llm_provider: str = "openai"):
        """
        Initialize the base agent.
        
        Args:
            name: Name identifier for this agent
            api_key: LLM API key (uses provider-specific env var if not provided)
            model: LLM model to use (default: gpt-4 for OpenAI, gemini-pro for Gemini, 
                   claude-3-opus-20240229 for Claude)
            llm_provider: LLM provider to use ('openai', 'gemini', or 'claude')
        """
        self.name = name
        self.llm_provider = llm_provider.lower()
        
        # Set API key and default model based on provider
        if self.llm_provider == "openai":
            self.api_key = api_key or os.getenv('OPENAI_API_KEY')
            self.model = model or os.getenv('OPENAI_MODEL', 'gpt-4')
        elif self.llm_provider == "gemini":
            self.api_key = api_key or os.getenv('GEMINI_API_KEY')
            self.model = model or os.getenv('GEMINI_MODEL', 'gemini-pro')
        elif self.llm_provider == "claude":
            self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
            self.model = model or os.getenv('CLAUDE_MODEL', 'claude-3-opus-20240229')
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}. "
                           f"Supported providers: 'openai', 'gemini', 'claude'")
        
        self.context = {}
        self.results = {}
        
    @abstractmethod
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the agent's primary task.
        
        Args:
            input_data: Input data dictionary
            
        Returns:
            Dictionary containing execution results
        """
        pass
    
    def query_llm(self, prompt: str, system_message: str = None, 
                  temperature: float = 0.7, max_tokens: int = 1000) -> str:
        """
        Query the LLM API for assistance. Supports multiple LLM providers:
        OpenAI (GPT), Google (Gemini), and Anthropic (Claude).
        
        Args:
            prompt: User prompt for the LLM
            system_message: System message defining agent behavior
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens in response
            
        Returns:
            LLM response as string
        """
        if not self.api_key:
            raise ValueError(
                f"{self.llm_provider.upper()} API key not found. Set the appropriate "
                f"environment variable or pass api_key during initialization."
            )
        
        try:
            if self.llm_provider == "openai":
                return self._query_openai(prompt, system_message, temperature, max_tokens)
            elif self.llm_provider == "gemini":
                return self._query_gemini(prompt, system_message, temperature, max_tokens)
            elif self.llm_provider == "claude":
                return self._query_claude(prompt, system_message, temperature, max_tokens)
            else:
                # This should never happen due to __init__ validation, but handle it anyway
                raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")
        except ImportError as e:
            raise ImportError(
                f"Required package for {self.llm_provider} not installed. "
                f"Install with: pip install {self._get_package_name()}"
            )
        except Exception as e:
            raise RuntimeError(f"Error querying {self.llm_provider} LLM: {str(e)}")
    
    def _query_openai(self, prompt: str, system_message: str, 
                      temperature: float, max_tokens: int) -> str:
        """Query OpenAI GPT API."""
        import openai
        client = openai.OpenAI(api_key=self.api_key)
        
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content
    
    def _query_gemini(self, prompt: str, system_message: str,
                      temperature: float, max_tokens: int) -> str:
        """Query Google Gemini API."""
        import google.generativeai as genai
        genai.configure(api_key=self.api_key)
        
        model = genai.GenerativeModel(self.model)
        
        # Combine system message with prompt for Gemini
        full_prompt = prompt
        if system_message:
            full_prompt = f"{system_message}\n\n{prompt}"
        
        response = model.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens
            )
        )
        
        return response.text
    
    def _query_claude(self, prompt: str, system_message: str,
                      temperature: float, max_tokens: int) -> str:
        """Query Anthropic Claude API."""
        import anthropic
        client = anthropic.Anthropic(api_key=self.api_key)
        
        message = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_message if system_message else "",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        return message.content[0].text
    
    def _get_package_name(self) -> str:
        """Get the package name for the current LLM provider."""
        packages = {
            "openai": "openai",
            "gemini": "google-generativeai",
            "claude": "anthropic"
        }
        return packages.get(self.llm_provider, "unknown")
    
    def update_context(self, key: str, value: Any):
        """Update agent's context with new information."""
        self.context[key] = value
    
    def get_context(self, key: str, default: Any = None) -> Any:
        """Get value from agent's context."""
        return self.context.get(key, default)
    
    def save_results(self, output_dir: str):
        """
        Save agent results to file.
        
        Args:
            output_dir: Directory to save results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save context and results as JSON
        output_file = os.path.join(output_dir, f"{self.name}_results.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'agent': self.name,
                'context': {k: str(v) for k, v in self.context.items()},
                'results': {k: str(v) for k, v in self.results.items()}
            }, f, indent=2)
        
        return output_file
    
    @staticmethod
    def run_unified_agent_workflow(workflow_config, api_key, llm_model, llm_provider, output_dir, progress_callback=None):
        """
        Unified agent workflow: infers task type from config and runs the appropriate pipeline.
        Supported: data fusion, time-lapse, direct ERT conversion.
        Returns: results dict, execution plan, interpretation, report files
        
        Args:
            workflow_config: Configuration dictionary from ContextInputAgent
            api_key: LLM API key
            llm_model: LLM model name
            llm_provider: LLM provider ('openai', 'gemini', 'claude')
            output_dir: Output directory path
            progress_callback: Optional callback function(step: str, progress: float, details: str)
        """
        def update_progress(step: str, progress: float, details: str = ""):
            """Update progress if callback is available."""
            if progress_callback:
                progress_callback(step, progress, details)
            print(f"[Progress {progress*100:.0f}%] {step}: {details}")
        
        # 1. Infer workflow type from configuration keys
        # More specific detection: check for unique indicators of each workflow
        config_keys = set(workflow_config.keys())
        print(f'\nDetecting workflow type from config keys: {config_keys}')
        
        # Normalize key names: ContextInputAgent may use 'data_file' or 'ert_file'
        if 'data_file' in workflow_config and 'ert_file' not in workflow_config:
            workflow_config['ert_file'] = workflow_config['data_file']
            print(f"  → Normalized 'data_file' to 'ert_file'")

        # Detect workflow type with priority order
        user_request_lower = workflow_config.get('user_request', '').lower()
        
        # TDEM: check for TDEM-specific keys
        if (workflow_config.get('tdem_file') or 
            workflow_config.get('tdem_data') or
            'tdem' in user_request_lower or
            'tem ' in user_request_lower or
            'electromagnetic' in user_request_lower):
            workflow_type = 'tdem'
        
        # Seismic: check for seismic-specific keys (standalone seismic refraction)
        elif (workflow_config.get('seismic_file') and not workflow_config.get('ert_file') or
              workflow_config.get('seismic_only', False) or
              ('seismic' in user_request_lower and 'ert' not in user_request_lower and 
               'resistivity' not in user_request_lower and 'fusion' not in user_request_lower) or
              'srt inversion' in user_request_lower or
              'seismic refraction' in user_request_lower or
              'travel time' in user_request_lower):
            workflow_type = 'seismic'
            
        # Time-lapse: check for time-lapse specific keys
        elif ('timelapse_files' in config_keys or 
            'timelapse_params' in config_keys or 
            'climate_config' in config_keys):
            workflow_type = 'time_lapse'
            
        # Data fusion: check for fusion-specific indicators WITH actual values
        # Note: ContextInputAgent may add 'fusion_pattern' or 'methods' keys even for ERT-only requests
        # We need to check if they have meaningful (non-None, non-empty) values
        elif (workflow_config.get('velocity_threshold') or  # Has velocity threshold
              (workflow_config.get('ert_file') and workflow_config.get('seismic_file')) or  # Has both files
              (workflow_config.get('fusion_pattern') and 
               workflow_config.get('fusion_pattern') not in [None, 'None', '']) or  # Has valid fusion pattern
              (workflow_config.get('methods') and 
               len(workflow_config.get('methods', [])) > 1 and  # Has multiple methods
               'seismic' in workflow_config.get('methods', []))):  # Including seismic
            workflow_type = 'data_fusion'
            
        # Direct ERT: single ERT file without real fusion indicators
        elif workflow_config.get('ert_file'):
            workflow_type = 'direct_ert'
            
        else:
            # Could not determine workflow type - check if this is an out-of-scope request
            workflow_type = 'custom'
            print("  → No standard workflow detected, will attempt custom code generation")

        print(f'\n===== WORKFLOW TYPE: {workflow_type.upper()} =====')
        execution_plan = None
        interpretation = None
        results = {}
        report_files = {}

        if workflow_type == 'data_fusion':
            # Use DataFusionAgent for complete workflow execution
            from .data_fusion_agent import DataFusionAgent
            fusion_agent = DataFusionAgent(
                api_key=api_key,
                model=llm_model,
                llm_provider=llm_provider
            )

            fusion_input = {
                'fusion_pattern': workflow_config.get('fusion_pattern', 'full_integration'),
                'methods': workflow_config.get('methods', ['seismic', 'ert', 'petrophysics']),
                'workflow_config': workflow_config,
                'data': {
                    'seismic': workflow_config.get('seismic_file'),
                    'ert': workflow_config.get('ert_file')
                },
                'output_dir': workflow_config.get('output_dir', str(output_dir))
            }

            update_progress("Planning data fusion workflow", 0.15, "Analyzing multi-method integration")
            print('Getting data fusion execution plan...')
            plan_result = fusion_agent.execute(fusion_input)
            execution_plan = plan_result.get('execution_plan')
            interpretation = plan_result.get('interpretation')

            print('\n' + '='*70)
            print('DATA FUSION EXECUTION PLAN')
            print('='*70)
            print(f"Pattern: {plan_result.get('fusion_pattern')}")
            print(f"\nInterpretation: {interpretation}")
            print(f"\nExecution Steps ({len(execution_plan)} total):")
            for i, step in enumerate(execution_plan, 1):
                print(f"\n  Step {i}: {step['step']}")
                print(f"    Agent: {step['agent']}")
                print(f"    Description: {step['description']}")
                print(f"    Outputs: {', '.join(step['outputs'])}")

            update_progress("Executing data fusion workflow", 0.30, "Processing seismic and ERT data")
            print('\nExecuting complete data fusion workflow...')
            results = fusion_agent.execute_full_workflow(fusion_input)
            update_progress("Data fusion complete", 0.75, "Generating report")
            
            # Update interpretation with detailed results
            if results.get('status') == 'success':
                import numpy as np
                
                # Build layer parameters summary
                layer_params = workflow_config.get('layer_params', {})
                params_summary = ""
                if layer_params:
                    for layer_name, params in layer_params.items():
                        params_summary += f"\n  - {layer_name}: "
                        if 'rho_sat_range' in params:
                            params_summary += f"ρ_sat={params['rho_sat_range']}, "
                        if 'porosity_range' in params:
                            params_summary += f"φ={params['porosity_range']}, "
                        if 'n_range' in params:
                            params_summary += f"n={params['n_range']}"
                
                # Get water content stats if available
                wc_stats = ""
                if 'water_content_mean' in results:
                    wc_mean = results['water_content_mean']
                    wc_std = results.get('water_content_std', np.zeros_like(wc_mean))
                    wc_stats = f"\n- Mean water content: {np.nanmean(wc_mean):.3f} ± {np.nanmean(wc_std):.3f}"
                    wc_stats += f"\n- Water content range: {np.nanmin(wc_mean):.3f} - {np.nanmax(wc_mean):.3f}"
                
                interpretation = f"""Data Fusion workflow completed successfully.

**Multi-Method Integration:**
- Seismic file: {workflow_config.get('seismic_file', 'N/A')}
- ERT file: {workflow_config.get('ert_file', 'N/A')}
- Velocity threshold: {workflow_config.get('velocity_threshold', 1200)} m/s
- Fusion pattern: {workflow_config.get('fusion_pattern', 'full_integration')}

**Petrophysical Parameters:**{params_summary if params_summary else ' Default Archie parameters applied'}

**Water Content Results:**{wc_stats if wc_stats else ' Not computed'}

**Key Benefits:**
- Seismic-derived structure constrains ERT inversion
- Layer-specific petrophysics improves accuracy
- Monte Carlo provides uncertainty quantification
"""

            # Generate comprehensive report with layer-specific statistics
            if results.get('status') == 'success':
                from .report_agent import ReportAgent
                report_agent = ReportAgent(api_key=api_key, model=llm_model, llm_provider=llm_provider)
                workflow_config['user_request'] = workflow_config.get('user_request', 'Data fusion workflow')

                # Add layer-specific statistics if available
                if 'cell_markers' in results and 'water_content_mean' in results:
                    import numpy as np
                    cell_markers = results['cell_markers']
                    water_content_mean = results['water_content_mean'].ravel()
                    unique_markers = np.unique(cell_markers)

                    layer_stats = {}
                    for layer_id in unique_markers:
                        mask = cell_markers == layer_id
                        layer_wc = water_content_mean[mask]

                        # Map layer ID to name (from petrophysics: marker 2=regolith, 3=bedrock)
                        layer_name = f"Layer {layer_id}"
                        for name in workflow_config.get('layer_params', {}).keys():
                            if layer_id == 2 and 'regolith' in name.lower():
                                layer_name = "Regolith"
                            elif layer_id == 3 and ('bedrock' in name.lower() or 'fractured' in name.lower()):
                                layer_name = "Fractured Bedrock"

                        layer_stats[str(layer_id)] = {
                            'name': layer_name,
                            'mean_wc': float(np.nanmean(layer_wc)),
                            'std_wc': float(np.nanstd(layer_wc)),
                            'min_wc': float(np.nanmin(layer_wc)),
                            'max_wc': float(np.nanmax(layer_wc)),
                            'n_cells': int(np.sum(mask))
                        }

                    results['layer_statistics'] = layer_stats

                report_input = {
                    'structure_results': results,
                    'petro_results': results if 'water_content_mean' in results else None,
                    'workflow_config': workflow_config,
                    'output_dir': str(output_dir)
                }
                report_results = report_agent.generate_data_fusion_report(report_input)
                if report_results.get('status') == 'success':
                    # Collect all report artifacts (md/html/pdf + visualizations)
                    report_files = {}
                    if report_results.get('report_file'):
                        report_files['report_markdown'] = report_results['report_file']
                    if report_results.get('html_file'):
                        report_files['report_html'] = report_results['html_file']
                    if report_results.get('pdf_file'):
                        report_files['report_pdf'] = report_results['pdf_file']
                    for vis_name, vis_path in (report_results.get('visualization_files') or {}).items():
                        report_files[f'visualization_{vis_name}'] = vis_path

        elif workflow_type == 'time_lapse':
            # Use ERT agents for time-lapse workflow
            from .ert_loader_agent import ERTLoaderAgent
            from .ert_inversion_agent import ERTInversionAgent
            from .inversion_evaluation_agent import InversionEvaluationAgent
            ert_loader = ERTLoaderAgent(api_key=api_key, model=llm_model, llm_provider=llm_provider)
            ert_inversion = ERTInversionAgent(api_key=api_key, model=llm_model, llm_provider=llm_provider)
            eval_agent = InversionEvaluationAgent(api_key=api_key, model=llm_model, llm_provider=llm_provider)

            update_progress("Starting time-lapse workflow", 0.15, "Loading multiple ERT datasets")
            print('Running time-lapse ERT workflow...')
            
            # Set execution plan for time-lapse workflow
            execution_plan = [
                {'step': 'Load Time-Lapse ERT Data', 'agent': 'ERTLoaderAgent', 
                 'description': 'Load multiple ERT datasets for time-lapse monitoring', 
                 'outputs': ['ert_data_list']},
                {'step': 'Fetch Climate Data', 'agent': 'ClimateDataAgent', 
                 'description': 'Fetch meteorological data (precipitation, temperature, PET) via conda environment', 
                 'outputs': ['climate_data']},
                {'step': 'Time-Lapse Inversion', 'agent': 'ERTInversionAgent', 
                 'description': 'Run time-lapse inversion with temporal regularization', 
                 'outputs': ['resistivity_changes', 'temporal_models']},
                {'step': 'Evaluate Inversion Quality', 'agent': 'InversionEvaluationAgent', 
                 'description': 'Assess inversion quality and optimize parameters if needed', 
                 'outputs': ['quality_metrics', 'optimized_results']},
                {'step': 'Generate Time-Lapse Report', 'agent': 'ReportAgent', 
                 'description': 'Create comprehensive report with climate correlation analysis', 
                 'outputs': ['html_report', 'visualizations', 'climate_resistivity_correlation']}
            ]
            
            # Initial interpretation - will be updated with actual results later
            interpretation = None  # Will be set after inversion completes

            # Load time-lapse data files (check both naming conventions)
            time_lapse_files = workflow_config.get('time_lapse_files') or workflow_config.get('timelapse_files', [])
            if not time_lapse_files:
                raise ValueError('No time-lapse files specified in configuration')
            
            print(f'  → Found {len(time_lapse_files)} time-lapse files')

            # Get electrode file for topography (if provided)
            electrode_file = workflow_config.get('electrode_file')
            if electrode_file:
                electrode_file_path = Path(electrode_file)
                project_dir = workflow_config.get('project_dir', '.')
                
                # Normalize electrode file path
                if not electrode_file_path.exists():
                    if project_dir and project_dir != '.':
                        combined_path = Path(project_dir) / electrode_file_path.name
                        if combined_path.exists():
                            electrode_file_path = combined_path
                        else:
                            if len(combined_path.parts) > 0 and combined_path.parts[0] == 'examples':
                                combined_path = Path(*combined_path.parts[1:])
                                if combined_path.exists():
                                    electrode_file_path = combined_path
                    if not electrode_file_path.exists() and len(electrode_file_path.parts) > 0 and electrode_file_path.parts[0] == 'examples':
                        alt_path = Path(*electrode_file_path.parts[1:])
                        if alt_path.exists():
                            electrode_file_path = alt_path
                
                electrode_file = str(electrode_file_path)
                print(f'  → Using electrode file: {electrode_file_path.name}')

            time_lapse_data = []
            for i, data_file in enumerate(time_lapse_files):
                # Normalize each time-lapse file path
                data_file_path = Path(data_file)
                project_dir = workflow_config.get('project_dir', '.')
                
                if not data_file_path.exists():
                    # Try combining with project_dir
                    if project_dir and project_dir != '.':
                        combined_path = Path(project_dir) / data_file_path.name
                        if combined_path.exists():
                            data_file_path = combined_path
                            print(f'  → Resolved: {project_dir} + {data_file_path.name}')
                        else:
                            # Handle duplicate 'examples/' prefix
                            if len(combined_path.parts) > 0 and combined_path.parts[0] == 'examples':
                                combined_path = Path(*combined_path.parts[1:])
                                if combined_path.exists():
                                    data_file_path = combined_path
                    
                    # Try removing 'examples/' prefix if present
                    if not data_file_path.exists() and len(data_file_path.parts) > 0 and data_file_path.parts[0] == 'examples':
                        alt_path = Path(*data_file_path.parts[1:])
                        if alt_path.exists():
                            data_file_path = alt_path
                            print(f'  → Removed examples/ prefix')
                
                data_file = str(data_file_path)
                print(f'Loading dataset {i+1}/{len(time_lapse_files)}: {Path(data_file).name}')
                
                result = ert_loader.execute({
                    'data_file': data_file,
                    'instrument': workflow_config.get('instrument', 'E4D'),
                    'project_dir': workflow_config.get('project_dir', '.'),
                    'electrode_file': electrode_file,  # Pass electrode file for topography
                    'crs': workflow_config.get('crs', 'local')
                })
                if result['status'] != 'success':
                    print(f'Failed to load {data_file}: {result.get("error")}')
                    continue
                time_lapse_data.append(result['ert_data'])

            if len(time_lapse_data) < 2:
                raise ValueError(f'Need at least 2 datasets for time-lapse, got {len(time_lapse_data)}')
            
            update_progress("Data loaded", 0.30, f"Loaded {len(time_lapse_data)} time-lapse datasets")

            # Fetch climate data if requested
            climate_results = None
            if workflow_config.get('use_climate', False) or workflow_config.get('climate_config'):
                print('\nFetching climate data for correlation analysis...')
                from .climate_data_agent import ClimateDataAgent
                import json
                
                climate_agent = ClimateDataAgent(api_key=api_key, model=llm_model, llm_provider=llm_provider)
                
                climate_config = workflow_config.get('climate_config', {})
                if climate_config:
                    # Save climate config to JSON file for conda environment
                    climate_config_file = output_dir / 'climate_config.json'
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Prepare climate config for saving
                    config_to_save = {
                        'coords': climate_config.get('coords'),
                        'dates': climate_config.get('dates'),
                        'variables': climate_config.get('variables', ['prcp', 'tmin', 'tmax', 'srad', 'dayl']),
                        'pet_method': climate_config.get('pet_method', 'penman_monteith'),
                        'time_scale': climate_config.get('time_scale', 'daily'),
                        'region': climate_config.get('region', 'na'),
                        'crs': climate_config.get('crs', 4326),
                        'output': str(output_dir / 'climate_data.csv')
                    }
                    
                    with open(climate_config_file, 'w', encoding='utf-8') as f:
                        json.dump(config_to_save, f, indent=2)
                    
                    print(f"  → Climate config saved: {climate_config_file.name}")
                    
                    # Fetch climate data using separate conda environment
                    fetch_result = climate_agent.fetch_climate_data_with_conda(
                        config_file=str(climate_config_file),
                        conda_path=None,  # Auto-detect
                        env_name="climate_fetch"
                    )
                    
                    if fetch_result.get('success'):
                        csv_path = Path(fetch_result['csv_path'])
                        print(f"  ✓ Climate data fetched: {csv_path.name}")
                        
                        # Load the fetched climate data from CSV
                        import re
                        from datetime import datetime, timedelta
                        
                        # Extract ERT dates from filenames
                        ert_dates = []
                        for fname in time_lapse_files:
                            match = re.search(r'(\d{4}-\d{2}-\d{2})', str(fname))
                            if match:
                                ert_dates.append(match.group(1))
                        
                        if ert_dates:
                            # Load climate data with extended range for visualization
                            first_date = datetime.strptime(ert_dates[0], '%Y-%m-%d')
                            last_date = datetime.strptime(ert_dates[-1], '%Y-%m-%d')
                            start_date = (first_date - timedelta(days=30)).strftime('%Y-%m-%d')
                            end_date = (last_date + timedelta(days=30)).strftime('%Y-%m-%d')
                            
                            climate_input = {
                                'csv_file': str(csv_path),
                                'ert_timestamps': ert_dates,
                                'start_date': start_date,
                                'end_date': end_date
                            }
                            
                            climate_results = climate_agent.execute(climate_input)
                            
                            if climate_results.get('data_source') == 'pre_fetched_csv':
                                workflow_config['climate_data'] = climate_results
                                print(f"  ✓ Climate data loaded from CSV")
                            else:
                                print(f"  ⚠️  Could not load climate CSV: {climate_results.get('error', 'Unknown error')}")
                        else:
                            print("  ⚠️  Could not extract dates from time-lapse filenames")
                    else:
                        print(f"  ⚠️  Climate data fetch failed: {fetch_result.get('message', 'Unknown error')}")

            # Run time-lapse inversion
            update_progress("Running time-lapse inversion", 0.45, "This may take several minutes...")
            inversion_input = {
                'time_lapse_data': time_lapse_data,
                'inversion_mode': 'time-lapse',
                'time_lapse_method': workflow_config.get('time_lapse_method', 'difference'),
                'temporal_regularization': workflow_config.get('temporal_regularization', 10.0),
                'baseline_index': 0,
                'inversion_params': workflow_config.get('inversion_params', {
                    'lambda': 15.0,
                    'max_iterations': 10,
                    'method': 'cgls'
                }),
                'output_dir': str(output_dir / 'inversion')
            }

            print('\nRunning time-lapse inversion...')
            results = ert_inversion.execute(inversion_input)
            update_progress("Inversion complete", 0.65, f"Processed {results.get('n_timesteps', 'N/A')} time steps")
            
            print(f"  → Inversion status: {results.get('status')}")
            if results.get('status') == 'success':
                print(f"  → Number of timesteps: {results.get('n_timesteps', 'N/A')}")
                print(f"  → Chi² values: {results.get('chi2_values', 'N/A')}")

            # Evaluate and optimize inversion quality if successful
            evaluation_results = None
            if results.get('status') == 'success':
                print('Evaluating inversion quality and optimizing parameters...')
                eval_input = {
                    'inversion_results': results,
                    'ert_data': time_lapse_data[0],  # Use baseline data for evaluation
                    'time_lapse_data': time_lapse_data,
                    'inversion_mode': 'time-lapse',
                    'inversion_params': workflow_config.get('inversion_params', {
                        'lambda': 15.0,
                        'max_iterations': 10,
                        'method': 'cgls'
                    }),
                    'auto_adjust': True,  # Automatically adjust and re-run if needed
                    'max_attempts': 2,    # Maximum 3 attempts to improve
                    'project_dir': workflow_config.get('project_dir', 'data/ERT/E4D'),
                    'instrument': workflow_config.get('instrument', 'E4D')
                }

                evaluation_results = eval_agent.execute(eval_input)

                # Update results if optimization improved them
                if evaluation_results.get('status') == 'success' and evaluation_results.get('attempts', 1) > 1:
                    print('✓ Inversion was optimized! Using improved results.')
                    results = evaluation_results['final_results']
            
            # Build detailed interpretation after inversion
            if results.get('status') == 'success':
                n_timesteps = results.get('n_timesteps', len(time_lapse_data))
                chi2_values = results.get('chi2_values', [])
                chi2_summary = f"{min(chi2_values):.3f} - {max(chi2_values):.3f}" if chi2_values else "N/A"
                
                interpretation = f"""Time-lapse ERT monitoring workflow completed successfully.

**Survey Summary:**
- Number of time steps: {n_timesteps}
- Data files processed: {len(time_lapse_files)}

**Inversion Results:**
- Chi-squared range: {chi2_summary}
- Temporal regularization: {workflow_config.get('temporal_regularization', 10.0)}
- Inversion method: {workflow_config.get('time_lapse_method', 'difference')}

**Climate Integration:**
- Climate data: {'Available' if workflow_config.get('climate_data') else 'Not requested'}

**Key Findings:**
Time-lapse resistivity changes capture subsurface moisture dynamics over the monitoring period.
Decreasing resistivity indicates increased soil moisture (wetting events).
Increasing resistivity indicates soil drying (evapotranspiration or drainage).
"""

            # Generate comprehensive report with climate integration if available
            if results.get('status') == 'success':
                from .report_agent import ReportAgent
                report_agent = ReportAgent(api_key=api_key, model=llm_model, llm_provider=llm_provider)

                # Prepare comprehensive report input
                climate_config = workflow_config.get('climate_config', {})
                dates = climate_config.get('dates', ['N/A', 'N/A']) if climate_config else ['N/A', 'N/A']

                # Ensure dates is a list and extract start/end as strings
                if not isinstance(dates, list):
                    dates = ['N/A', 'N/A']
                start_date = str(dates[0]) if len(dates) > 0 else 'N/A'
                end_date = str(dates[-1]) if len(dates) > 1 else 'N/A'

                # Get site coordinates
                site_info_config = workflow_config.get('site_info', {})
                coordinates_str = str(site_info_config.get('coordinates', 'N/A'))
                if coordinates_str == 'N/A':
                    # Try to get from climate_config
                    coords_list = climate_config.get('coords', [])
                    if coords_list and len(coords_list) == 2:
                        coordinates_str = f"{coords_list[1]:.5f}°N, {coords_list[0]:.5f}°W"

                # Prepare comparison DataFrame for climate-resistivity correlation if climate data available
                comparison_df = None
                if workflow_config.get('climate_data'):
                    import pandas as pd
                    import numpy as np

                    climate_results = workflow_config['climate_data']
                    if climate_results.get('ert_alignment') and 'ert_aligned_data' in climate_results['ert_alignment']:
                        aligned_df = climate_results['ert_alignment']['ert_aligned_data']

                        # Calculate mean resistivity changes for each time step
                        final_models = results.get('final_models')
                        if final_models is not None:
                            baseline = final_models[:, 0]
                            resistivity_changes = []
                            for i in range(1, final_models.shape[1]):
                                change = final_models[:, i] - baseline
                                mean_change = np.mean(change)
                                resistivity_changes.append(mean_change)

                            # Create comparison dataframe
                            prcp_vals = aligned_df.get('prcp', [0] * (len(aligned_df) - 1))[1:]
                            tmin_vals = aligned_df.get('tmin', [0] * (len(aligned_df) - 1))[1:]
                            tmax_vals = aligned_df.get('tmax', [0] * (len(aligned_df) - 1))[1:]
                            pet_vals = aligned_df.get('pet', [0] * (len(aligned_df) - 1))[1:]

                            # Convert dates to strings for report compatibility
                            date_strings = [str(dt.date()) if hasattr(dt, 'date') else str(dt) for dt in aligned_df.index[1:]]

                            comparison_df = pd.DataFrame({
                                'Date': date_strings,
                                'Mean_Resistivity_Change_Ohm_m': resistivity_changes[:len(aligned_df)-1],
                                'Precipitation_mm': prcp_vals,
                                'Temp_Min_C': tmin_vals,
                                'Temp_Max_C': tmax_vals,
                                'Temp_Mean_C': (np.array(tmin_vals) + np.array(tmax_vals)) / 2,
                                'PET_mm': pet_vals
                            })

                site_info = {
                    'name': str(site_info_config.get('name', 'Time-Lapse ERT Monitoring Site')),
                    'location': str(site_info_config.get('location', coordinates_str)),
                    'coordinates': str(coordinates_str),
                    'elevation': str(site_info_config.get('elevation', 'N/A')),
                    'study_period': f"{start_date} to {end_date}",
                    'description': str('Time-lapse ERT monitoring with climate integration for subsurface moisture dynamics.')
                }

                report_input = {
                    'inversion_results': results,
                    'climate_data': workflow_config.get('climate_data'),
                    'site_info': site_info,
                    'comparison_data': comparison_df,
                    'evaluation_results': evaluation_results,
                    'workflow_config': workflow_config,
                    'time_lapse_method': workflow_config.get('time_lapse_method', 'difference'),
                    'output_dir': str(output_dir)
                }
                
                update_progress("Generating report", 0.85, "Creating visualizations and analysis")
                print('\n' + '='*70)
                print('GENERATING TIME-LAPSE REPORT')
                print('='*70)
                print(f"  → Output directory: {output_dir}")
                print(f"  → Time-lapse method: {workflow_config.get('time_lapse_method', 'difference')}")
                print(f"  → Climate data available: {workflow_config.get('climate_data') is not None}")
                
                report_results = report_agent.generate_timelapse_report(report_input)
                update_progress("Report complete", 0.95, "Saving files")
                
                if report_results.get('status') == 'success':
                    report_files = {}
                    if report_results.get('report_file'):
                        report_files['report_markdown'] = report_results['report_file']
                    if report_results.get('html_file'):
                        report_files['report_html'] = report_results['html_file']
                    if report_results.get('pdf_file'):
                        report_files['report_pdf'] = report_results['pdf_file']
                    for vis_name, vis_path in (report_results.get('visualization_files') or {}).items():
                        report_files[f'visualization_{vis_name}'] = vis_path
                    print("\nReport generation completed successfully!")
                    print(f"  Generated {len(report_results.get('visualization_files', {}))} visualization files")
                    print(f"  Report file: {report_results.get('report_file')}")
                    print(f"  HTML file: {report_results.get('html_file')}")
                else:
                    print(f"\nReport generation failed: {report_results.get('error', 'Unknown error')}")
                    print("  Check logs for details")

        elif workflow_type == 'direct_ert':
            # Use ERT agents for direct ERT workflow
            from .ert_loader_agent import ERTLoaderAgent
            from .ert_inversion_agent import ERTInversionAgent
            from .petrophysics_agent import PetrophysicsAgent
            from .inversion_evaluation_agent import InversionEvaluationAgent
            ert_loader = ERTLoaderAgent(api_key=api_key, model=llm_model, llm_provider=llm_provider)
            ert_inversion = ERTInversionAgent(api_key=api_key, model=llm_model, llm_provider=llm_provider)
            petrophysics_agent = PetrophysicsAgent(api_key=api_key, model=llm_model, llm_provider=llm_provider)
            eval_agent = InversionEvaluationAgent(api_key=api_key, model=llm_model, llm_provider=llm_provider)

            # Detect if user wants water content conversion
            # Check explicit flag or presence of petrophysical parameters
            user_request = workflow_config.get('user_request', '').lower()
            petro_params = workflow_config.get('petrophysical_params', {})
            has_petro_params = petro_params and len(petro_params) > 0
            
            # Keywords indicating water content conversion is desired
            wc_keywords = ['water content', 'moisture', 'saturation', 'petrophysic', 
                          'archie', 'porosity', 'rho_sat', 'hydro']
            wants_water_content = (
                any(kw in user_request for kw in wc_keywords) or
                has_petro_params or
                workflow_config.get('convert_to_water_content', None) is True
            )
            
            # Keywords indicating ERT-only is desired
            ert_only_keywords = ['ert inversion only', 'resistivity only', 'just inversion', 
                                'only invert', 'inversion result', 'resistivity imaging']
            explicitly_ert_only = (
                any(kw in user_request for kw in ert_only_keywords) or
                workflow_config.get('convert_to_water_content') is False
            )
            
            # Determine workflow mode
            skip_petrophysics = explicitly_ert_only or (not wants_water_content and not has_petro_params)
            
            if skip_petrophysics:
                print('Running ERT inversion workflow (no water content conversion)...')
                update_progress("Running ERT inversion", 0.20, "ERT-only mode detected")
            else:
                print('Running direct ERT to water content workflow...')
                update_progress("Running ERT to water content workflow", 0.20, "Full conversion mode")

            # Load ERT data
            data_file = workflow_config.get('ert_file')
            if not data_file:
                raise ValueError('No ERT file specified in configuration')
            
            # Normalize path: handle various scenarios
            data_file_path = Path(data_file)
            project_dir = workflow_config.get('project_dir', '.')
            
            # Try to find the file in different locations
            if not data_file_path.exists():
                # Scenario 1: Try combining project_dir + data_file
                if project_dir and project_dir != '.':
                    combined_path = Path(project_dir) / data_file_path.name
                    if combined_path.exists():
                        data_file_path = combined_path
                        print(f'  → Combined path: {project_dir} + {data_file_path.name} = {data_file_path}')
                    else:
                        # Scenario 2: Maybe project_dir has 'examples/' prefix but we're in examples/
                        if combined_path.parts[0] == 'examples':
                            combined_path = Path(*combined_path.parts[1:])
                            if combined_path.exists():
                                data_file_path = combined_path
                                print(f'  → Fixed duplicate: {data_file_path}')
                
                # Scenario 3: Try removing 'examples/' prefix from original path
                if not data_file_path.exists() and data_file_path.parts[0] == 'examples':
                    alt_path = Path(*data_file_path.parts[1:])
                    if alt_path.exists():
                        data_file_path = alt_path
                        print(f'  → Removed examples/ prefix: {data_file_path}')
            
            data_file = str(data_file_path)
            
            # Update project_dir to match the data file location
            if not project_dir or project_dir == '.':
                workflow_config['project_dir'] = str(data_file_path.parent)
            else:
                # Make sure project_dir matches where the file actually is
                workflow_config['project_dir'] = str(data_file_path.parent)

            update_progress("Loading ERT data", 0.25, f"File: {data_file_path.name}")
            
            # Handle electrode file if provided
            electrode_file = workflow_config.get('electrode_file')
            if electrode_file:
                electrode_file_path = Path(electrode_file)
                # Normalize electrode file path similar to data_file
                if not electrode_file_path.exists():
                    # Try combining with project_dir
                    if project_dir and project_dir != '.':
                        combined_path = Path(project_dir) / electrode_file_path.name
                        if combined_path.exists():
                            electrode_file_path = combined_path
                        elif combined_path.parts[0] == 'examples':
                            combined_path = Path(*combined_path.parts[1:])
                            if combined_path.exists():
                                electrode_file_path = combined_path
                    # Try removing 'examples/' prefix
                    if not electrode_file_path.exists() and electrode_file_path.parts[0] == 'examples':
                        alt_path = Path(*electrode_file_path.parts[1:])
                        if alt_path.exists():
                            electrode_file_path = alt_path
                electrode_file = str(electrode_file_path)
                print(f'  → Electrode file: {electrode_file_path.name}')
            
            print(f'Loading ERT data: {Path(data_file).name}')
            load_result = ert_loader.execute({
                'data_file': data_file,
                'instrument': workflow_config.get('instrument', 'DAS-1'),
                'project_dir': workflow_config.get('project_dir', '.'),
                'electrode_file': electrode_file,
                'crs': workflow_config.get('crs', 'local')
            })

            if load_result['status'] != 'success':
                raise ValueError(f'Failed to load ERT data: {load_result.get("error")}')

            ert_data = load_result['ert_data']
            update_progress("Data loaded successfully", 0.35, f"{len(ert_data.electrodes)} electrodes, {len(ert_data.observations)} measurements")

            # Run inversion
            update_progress("Running ERT inversion", 0.40, "This may take a few minutes...")
            inversion_input = {
                'ert_data': ert_data,
                'instrument': workflow_config.get('instrument', 'DAS-1'),
                'project_dir': workflow_config.get('project_dir', '.'),
                'output_dir': str(output_dir / 'inversion'),
                'inversion_params': workflow_config.get('inversion_params', {
                    'lambda': 20.0,
                    'max_iterations': 12,
                    'method': 'cgls'
                })
            }

            inversion_results = ert_inversion.execute(inversion_input)

            if inversion_results.get('status') != 'success':
                raise ValueError(f'Inversion failed: {inversion_results.get("error")}')
            
            chi2_value = inversion_results.get('chi2', 'N/A')
            update_progress("Inversion complete", 0.55, f"Chi² = {chi2_value}")

            # Evaluate inversion quality
            evaluation_results = None
            if inversion_results.get('status') == 'success':
                update_progress("Evaluating inversion quality", 0.60, "Checking convergence and data fit")
                eval_input = {
                    'inversion_results': inversion_results,
                    'ert_data': ert_data,
                    'quality_threshold': 0.7
                }
                evaluation_results = eval_agent.execute(eval_input)

            # If skipping petrophysics (ERT-only mode)
            if skip_petrophysics:
                update_progress("Preparing ERT results", 0.75, "Skipping water content conversion")
                results = {
                    'status': 'success',
                    'ert_data': ert_data,
                    'inversion_results': inversion_results,
                    'evaluation_results': evaluation_results,
                    'skip_petrophysics': True
                }
                
                # Set execution plan for ERT-only workflow
                execution_plan = [
                    {'step': 'Load ERT Data', 'agent': 'ERTLoaderAgent', 
                     'description': 'Load ERT data file', 'outputs': ['ert_data']},
                    {'step': 'Run Inversion', 'agent': 'ERTInversionAgent', 
                     'description': 'Invert for resistivity model', 'outputs': ['resistivity_model', 'mesh']},
                    {'step': 'Evaluate Quality', 'agent': 'InversionEvaluationAgent', 
                     'description': 'Assess inversion quality', 'outputs': ['quality_score']},
                    {'step': 'Generate Report', 'agent': 'ReportAgent', 
                     'description': 'Create summary report', 'outputs': ['report']}
                ]
                interpretation = "ERT inversion completed. Resistivity model generated without water content conversion."
            else:
                # Convert to water content
                update_progress("Converting to water content", 0.65, "Running Monte Carlo petrophysics")
                
                # Get mesh and cell markers from inversion results
                mesh = inversion_results.get('mesh')
                cell_markers = np.array(mesh.cellMarkers()) if mesh else np.zeros(len(inversion_results.get('resistivity_model', [])))
                
                petro_input = {
                    'resistivity_model': inversion_results.get('resistivity_model'),
                    'mesh': mesh,
                    'cell_markers': cell_markers,
                    'petrophysical_params': workflow_config.get('petrophysical_params', {}),
                    'n_realizations': workflow_config.get('n_realizations', 100),
                    'geological_context': workflow_config.get('geological_context', 'generic watershed'),
                    'output_dir': str(output_dir / 'petrophysics')
                }

                petro_results = petrophysics_agent.execute(petro_input)

                if petro_results.get('status') != 'success':
                    raise ValueError(f'Petrophysics conversion failed: {petro_results.get("error")}')
                
                update_progress("Petrophysics complete", 0.75, "Water content model generated")

                # Combine results
                results = {
                    'status': 'success',
                    'ert_data': ert_data,
                    'inversion_results': inversion_results,
                    'evaluation_results': evaluation_results,
                    'petrophysics_results': petro_results,
                    'petrophysical_params': workflow_config.get('petrophysical_params', {}),
                    'water_content_mean': petro_results.get('water_content_mean'),
                    'water_content_std': petro_results.get('water_content_std'),
                    'skip_petrophysics': False
                }
                
                # Set execution plan for full workflow
                execution_plan = [
                    {'step': 'Load ERT Data', 'agent': 'ERTLoaderAgent', 
                     'description': 'Load ERT data file', 'outputs': ['ert_data']},
                    {'step': 'Run Inversion', 'agent': 'ERTInversionAgent', 
                     'description': 'Invert for resistivity model', 'outputs': ['resistivity_model', 'mesh']},
                    {'step': 'Evaluate Quality', 'agent': 'InversionEvaluationAgent', 
                     'description': 'Assess inversion quality', 'outputs': ['quality_score']},
                    {'step': 'Convert to Water Content', 'agent': 'PetrophysicsAgent', 
                     'description': 'Apply petrophysics with Monte Carlo', 'outputs': ['water_content', 'uncertainty']},
                    {'step': 'Generate Report', 'agent': 'ReportAgent', 
                     'description': 'Create comprehensive report', 'outputs': ['report']}
                ]
                
                # Build detailed interpretation including petrophysical parameters used
                layer_params_used = petro_results.get('layer_params_used', {})
                params_summary = ""
                if layer_params_used:
                    for layer_name, params in layer_params_used.items():
                        params_summary += f"\n  - {layer_name}: "
                        if 'rho_sat' in params:
                            params_summary += f"ρ_sat={params['rho_sat']:.1f}Ωm, "
                        if 'porosity' in params:
                            params_summary += f"φ={params['porosity']:.2f}, "
                        if 'n' in params:
                            params_summary += f"n={params['n']:.2f}, "
                        if 'm' in params:
                            params_summary += f"m={params['m']:.2f}"
                
                interpretation = f"""ERT inversion and petrophysical conversion completed successfully.

**Inversion Results:**
- Chi-squared misfit: {inversion_results.get('chi2', 'N/A'):.3f}
- Iterations: {inversion_results.get('iterations', 'N/A')}

**Petrophysical Conversion:**
- Monte Carlo realizations: {workflow_config.get('n_realizations', 100)}
- Parameters used:{params_summary if params_summary else ' Default Archie parameters'}

**Water Content Statistics:**
- Mean water content range: {np.nanmin(petro_results.get('water_content_mean', [0])):.3f} - {np.nanmax(petro_results.get('water_content_mean', [0])):.3f}
- Uncertainty (std): {np.nanmean(petro_results.get('water_content_std', [0])):.3f}
"""

            # Generate comprehensive report
            if results.get('status') == 'success':
                update_progress("Generating report", 0.85, "Creating visualizations and summary")
                from .report_agent import ReportAgent
                report_agent = ReportAgent(api_key=api_key, model=llm_model, llm_provider=llm_provider)
                
                # Build workflow_data based on whether petrophysics was run
                workflow_data = {
                    'ert_data': {
                        'n_electrodes': len(ert_data.electrodes),
                        'num_electrodes': len(ert_data.electrodes),
                        'n_measurements': len(ert_data.observations),
                        'num_measurements': len(ert_data.observations),
                        'instrument': workflow_config.get('instrument', 'DAS-1')
                    },
                    'inversion_results': {
                        'chi2': inversion_results.get('chi2'),
                        'iterations': inversion_results.get('iterations'),
                        'resistivity_model': inversion_results.get('resistivity_model'),
                        'mesh': inversion_results.get('mesh'),
                        'coverage': inversion_results.get('coverage')
                    },
                    'evaluation_results': {
                        'quality_score': evaluation_results.get('quality_score') if evaluation_results else None
                    },
                    'skip_petrophysics': skip_petrophysics
                }
                
                # Only include water content and petrophysics data if conversion was performed
                if not skip_petrophysics:
                    petro_results = results.get('petrophysics_results', {})
                    workflow_data['water_content'] = {
                        'mesh': inversion_results['mesh'],
                        'water_content_mean': petro_results.get('water_content_mean'),
                        'water_content_std': petro_results.get('water_content_std'),
                        'layer_params_used': petro_results.get('layer_params_used', {}),
                        'layer_params': petro_results.get('layer_params', {}),
                        'petrophysical_params': workflow_config.get('petrophysical_params', {}),
                        'n_realizations': workflow_config.get('n_realizations', 200)
                    }
                    workflow_data['petrophysics_results'] = petro_results
                    workflow_data['petrophysical_params'] = workflow_config.get('petrophysical_params', {})
                
                report_input = {
                    'workflow_data': workflow_data,
                    'config': workflow_config,
                    'output_dir': str(output_dir)
                }
                report_results = report_agent.execute(report_input)
                
                update_progress("Report complete", 0.95, "Saving files")
                
                if report_results.get('status') == 'success':
                    report_files = {}
                    if report_results.get('report_file'):
                        report_files['report_markdown'] = report_results['report_file']
                    if report_results.get('html_file'):
                        report_files['report_html'] = report_results['html_file']
                    if report_results.get('pdf_file'):
                        report_files['report_pdf'] = report_results['pdf_file']
                    for vis_name, vis_path in (report_results.get('visualization_files') or {}).items():
                        report_files[f'visualization_{vis_name}'] = vis_path

        elif workflow_type == 'tdem':
            # Use TDEMAgent for Time-Domain Electromagnetic workflow
            from .tdem_agent import TDEMAgent
            tdem_agent = TDEMAgent(api_key=api_key, model=llm_model, llm_provider=llm_provider)
            
            update_progress("Starting TDEM workflow", 0.15, "Configuring electromagnetic inversion")
            print('Running TDEM workflow...')
            
            # Set execution plan for TDEM workflow
            execution_plan = [
                {'step': 'Load TDEM Data', 'agent': 'TDEMAgent', 
                 'description': 'Load time-domain electromagnetic sounding data', 
                 'outputs': ['times', 'dobs', 'uncertainties']},
                {'step': 'Run TDEM Inversion', 'agent': 'TDEMAgent', 
                 'description': 'Invert for 1D conductivity model using SimPEG', 
                 'outputs': ['conductivity_model', 'chi2']},
                {'step': 'Generate Visualization', 'agent': 'TDEMAgent', 
                 'description': 'Create result plots and interpretation', 
                 'outputs': ['visualization_files', 'interpretation']},
            ]
            
            interpretation = (
                "TDEM (Time-Domain Electromagnetic) workflow processes electromagnetic sounding data "
                "to recover subsurface conductivity structure. Uses SimPEG for 1D layered Earth inversion."
            )
            
            # Get TDEM data file with path resolution
            tdem_file = workflow_config.get('tdem_file') or workflow_config.get('data_file')
            if tdem_file:
                tdem_file_path = Path(tdem_file)
                if not tdem_file_path.exists():
                    # Try to find in uploaded_files
                    uploaded_files = workflow_config.get('uploaded_files', {})
                    if tdem_file_path.name in uploaded_files:
                        tdem_file = uploaded_files[tdem_file_path.name]
                        tdem_file_path = Path(tdem_file)
                        print(f'  → Found TDEM file in uploads: {tdem_file_path.name}')
                    else:
                        # Try project_dir
                        project_dir = workflow_config.get('project_dir', '.')
                        if project_dir and project_dir != '.':
                            combined_path = Path(project_dir) / tdem_file_path.name
                            if combined_path.exists():
                                tdem_file_path = combined_path
                                tdem_file = str(tdem_file_path)
                
                if not tdem_file_path.exists():
                    raise ValueError(f"TDEM data file not found: {tdem_file}")
                
                print(f'  → TDEM file: {tdem_file_path.name}')
                tdem_file = str(tdem_file_path)
            
            # Prepare TDEM input
            tdem_input = {
                'mode': workflow_config.get('tdem_mode', 'inversion'),
                'data_file': tdem_file,
                'source_radius': workflow_config.get('source_radius', 10.0),
                'n_layers': workflow_config.get('n_layers', 20),
                'min_thickness': workflow_config.get('min_thickness', 0.5),
                'max_thickness': workflow_config.get('max_thickness', 10.0),
                'starting_conductivity': workflow_config.get('starting_conductivity', 0.001),
                'use_irls': workflow_config.get('use_irls', True),
                'max_iterations': workflow_config.get('max_iterations', 50),
                'output_dir': str(output_dir / 'tdem'),
                'verbose': True
            }
            
            # Handle forward modeling mode
            if tdem_input['mode'] == 'forward':
                tdem_input['thicknesses'] = workflow_config.get('thicknesses')
                tdem_input['conductivity'] = workflow_config.get('conductivity')
                tdem_input['times'] = workflow_config.get('times')
                tdem_input['noise_level'] = workflow_config.get('noise_level', 0.05)
            
            # Handle hydro-to-tdem mode
            if tdem_input['mode'] == 'hydro_to_tdem':
                tdem_input['water_content'] = workflow_config.get('water_content')
                tdem_input['porosity'] = workflow_config.get('porosity')
                tdem_input['layer_thicknesses'] = workflow_config.get('layer_thicknesses')
                tdem_input['petrophysical_params'] = workflow_config.get('petrophysical_params', {})
            
            update_progress("Running TDEM processing", 0.30, "Loading data and running inversion")
            
            # Execute TDEM workflow
            results = tdem_agent.execute(tdem_input)
            
            if results.get('status') == 'success':
                update_progress("TDEM complete", 0.80, f"Chi² = {results.get('chi2', 'N/A')}")
                
                # Generate report
                update_progress("Generating report", 0.90, "Creating TDEM report")
                from .report_agent import ReportAgent
                report_agent = ReportAgent(api_key=api_key, model=llm_model, llm_provider=llm_provider)
                
                # Create TDEM-specific report
                tdem_report = f"""# TDEM Inversion Report

**Generated by:** PyHydroGeophysX TDEMAgent
**Mode:** {results.get('mode', 'inversion')}

## Executive Summary

{results.get('interpretation', 'TDEM processing completed successfully.')}

## Inversion Results

### Model Statistics
- **Number of Layers:** {results.get('n_layers', 'N/A')}
- **Chi-squared Misfit:** {results.get('chi2', 'N/A'):.3f}
- **Conductivity Range:** {results.get('conductivity_range', ['N/A', 'N/A'])[0]:.4f} - {results.get('conductivity_range', ['N/A', 'N/A'])[1]:.4f} S/m
- **Resistivity Range:** {results.get('resistivity_range', ['N/A', 'N/A'])[0]:.1f} - {results.get('resistivity_range', ['N/A', 'N/A'])[1]:.1f} Ωm

## Visualization

![TDEM Result]({Path(results.get('visualization_file', '')).name})

## Output Files

- Output directory: `{results.get('output_dir', 'N/A')}`
- Recovered conductivity: `recovered_conductivity.npy`
- Layer thicknesses: `inv_thicknesses.npy`
- Predicted data: `predicted_data.npy`

---
*Report generated by PyHydroGeophysX TDEMAgent using SimPEG*
"""
                # Save report
                report_file = output_dir / 'tdem_report.md'
                with open(report_file, 'w', encoding='utf-8') as f:
                    f.write(tdem_report)
                
                report_files = {'report_markdown': str(report_file)}
                
                # Save HTML version
                html_file = report_agent._save_html_report(tdem_report, str(output_dir), 'tdem_report')
                if html_file:
                    report_files['report_html'] = html_file
                
                # Save PDF version
                pdf_file = report_agent._save_pdf_report(tdem_report, str(output_dir), 
                                                         visualization_files={'tdem_result': results.get('visualization_file', '')},
                                                         filename='tdem_report')
                if pdf_file:
                    report_files['report_pdf'] = pdf_file
                
                if results.get('visualization_file'):
                    report_files['visualization_tdem'] = results['visualization_file']
                
                interpretation = results.get('interpretation', interpretation)
            else:
                update_progress("TDEM failed", 1.0, results.get('error', 'Unknown error'))
                raise ValueError(f"TDEM processing failed: {results.get('error')}")

        elif workflow_type == 'seismic':
            # Use SeismicAgent for standalone seismic refraction tomography
            from .seismic_agent import SeismicAgent
            seismic_agent = SeismicAgent(api_key=api_key, model=llm_model, llm_provider=llm_provider)
            
            update_progress("Starting seismic workflow", 0.15, "Configuring seismic refraction tomography")
            print('Running seismic refraction tomography workflow...')
            
            # Set execution plan for seismic workflow
            execution_plan = [
                {'step': 'Load Seismic Data', 'agent': 'SeismicAgent', 
                 'description': 'Load travel time data from .dat file', 
                 'outputs': ['seismic_data']},
                {'step': 'Run SRT Inversion', 'agent': 'SeismicAgent', 
                 'description': 'Invert for P-wave velocity model using PyGIMLI', 
                 'outputs': ['velocity_model', 'mesh', 'coverage']},
                {'step': 'Extract Interfaces', 'agent': 'SeismicAgent', 
                 'description': 'Extract geological interfaces from velocity thresholds', 
                 'outputs': ['interface_coords']},
                {'step': 'Generate Visualization', 'agent': 'SeismicAgent', 
                 'description': 'Create velocity tomogram and interface plots', 
                 'outputs': ['visualization_files', 'interpretation']},
            ]
            
            interpretation = (
                "Seismic refraction tomography (SRT) workflow inverts travel time data to "
                "recover subsurface P-wave velocity structure. Velocity interfaces are "
                "extracted for geological interpretation and hydrogeological modeling."
            )
            
            # Get seismic file
            seismic_file = workflow_config.get('seismic_file')
            if not seismic_file:
                raise ValueError('No seismic file specified in configuration. '
                               'Please provide seismic_file path.')
            
            # Normalize seismic file path
            seismic_file_path = Path(seismic_file)
            project_dir = workflow_config.get('project_dir', '.')
            
            if not seismic_file_path.exists():
                # Try combining with project_dir
                if project_dir and project_dir != '.':
                    combined_path = Path(project_dir) / seismic_file_path.name
                    if combined_path.exists():
                        seismic_file_path = combined_path
                    elif combined_path.parts[0] == 'examples':
                        combined_path = Path(*combined_path.parts[1:])
                        if combined_path.exists():
                            seismic_file_path = combined_path
                
                # Try removing 'examples/' prefix
                if not seismic_file_path.exists() and len(seismic_file_path.parts) > 0:
                    if seismic_file_path.parts[0] == 'examples':
                        alt_path = Path(*seismic_file_path.parts[1:])
                        if alt_path.exists():
                            seismic_file_path = alt_path
            
            seismic_file = str(seismic_file_path)
            print(f'  → Seismic file: {seismic_file_path.name}')
            
            # Get velocity thresholds for interface extraction
            velocity_thresholds = workflow_config.get('velocity_thresholds', [1200])
            if isinstance(velocity_thresholds, (int, float)):
                velocity_thresholds = [velocity_thresholds]
            
            # Add additional threshold from user request if specified
            velocity_threshold = workflow_config.get('velocity_threshold')
            if velocity_threshold and velocity_threshold not in velocity_thresholds:
                velocity_thresholds.append(velocity_threshold)
            
            print(f'  → Velocity thresholds: {velocity_thresholds} m/s')
            
            # Prepare inversion parameters
            inversion_params = workflow_config.get('inversion_params', {})
            if not inversion_params:
                inversion_params = {
                    'lam': workflow_config.get('lambda', 50),
                    'zWeight': workflow_config.get('z_weight', 0.2),
                    'vTop': workflow_config.get('v_top', 500),
                    'vBottom': workflow_config.get('v_bottom', 5000),
                    'paraDepth': workflow_config.get('para_depth', 30.0),
                    'limits': workflow_config.get('velocity_limits', [300., 8000.])
                }
            
            # Prepare seismic input
            seismic_input = {
                'seismic_file': seismic_file,
                'velocity_threshold': velocity_thresholds[0] if velocity_thresholds else 1200,
                'velocity_thresholds': velocity_thresholds,
                'inversion_params': inversion_params,
                'extract_interfaces': workflow_config.get('extract_interfaces', True),
                'output_dir': str(output_dir / 'seismic')
            }
            
            update_progress("Running seismic inversion", 0.30, "Loading data and inverting velocity model")
            
            # Execute seismic workflow
            results = seismic_agent.execute(seismic_input)
            
            if results.get('status') == 'success':
                vel_range = results.get('velocity_range', [0, 0])
                update_progress("Seismic inversion complete", 0.80, 
                              f"Velocity: {vel_range[0]:.0f} - {vel_range[1]:.0f} m/s")
                
                # Generate report
                update_progress("Generating report", 0.90, "Creating seismic report")
                from .report_agent import ReportAgent
                report_agent = ReportAgent(api_key=api_key, model=llm_model, llm_provider=llm_provider)
                
                # Format interface information
                interfaces_info = ""
                for threshold, data in results.get('interfaces', {}).items():
                    z_min = min(data['z']) if len(data['z']) > 0 else 'N/A'
                    z_max = max(data['z']) if len(data['z']) > 0 else 'N/A'
                    interfaces_info += f"- **{threshold} m/s interface:** Depth range {z_min:.1f} to {z_max:.1f} m\n"
                
                # Create seismic-specific report
                seismic_report = f"""# Seismic Refraction Tomography Report

**Generated by:** PyHydroGeophysX SeismicAgent

## Executive Summary

{results.get('interpretation', 'Seismic refraction tomography completed successfully.')}

## Survey Information

- **Data File:** `{seismic_file_path.name}`
- **Number of Shots:** {results.get('n_shots', 'N/A')}
- **Number of Receivers:** {results.get('n_receivers', 'N/A')}
- **Total Travel Times:** {results.get('n_data', 'N/A')}

## Inversion Results

### Velocity Model Statistics
- **Velocity Range:** {vel_range[0]:.0f} - {vel_range[1]:.0f} m/s
- **Mesh Cells:** {results.get('mesh').cellCount() if results.get('mesh') else 'N/A'}

### Inversion Parameters
- **Lambda (regularization):** {inversion_params.get('lam', 50)}
- **Z-Weight:** {inversion_params.get('zWeight', 0.2)}
- **Velocity Constraints:** {inversion_params.get('vTop', 500)} - {inversion_params.get('vBottom', 5000)} m/s

## Extracted Interfaces

{interfaces_info if interfaces_info else 'No interfaces extracted.'}

### Geological Interpretation

Based on typical velocity-depth relationships:
- **< 1200 m/s:** Weathered soil/regolith
- **1200-3000 m/s:** Fractured rock
- **> 3000 m/s:** Competent bedrock

## Visualization

![Seismic Velocity Model]({Path(results.get('visualization_file', '')).name})

## Output Files

- **Output directory:** `{results.get('output_dir', 'N/A')}`
- **Velocity model:** `velocity_model.npy`
- **Coverage:** `coverage.npy`
- **Mesh:** `seismic_mesh.bms`
- **Interface files:** `interface_*ms.txt`

---
*Report generated by PyHydroGeophysX SeismicAgent using PyGIMLI*
"""
                # Save report
                report_file = output_dir / 'seismic_report.md'
                with open(report_file, 'w', encoding='utf-8') as f:
                    f.write(seismic_report)
                
                report_files = {'report_markdown': str(report_file)}
                
                # Save HTML version
                html_file = report_agent._save_html_report(seismic_report, str(output_dir), 'seismic_report')
                if html_file:
                    report_files['report_html'] = html_file
                
                # Save PDF version
                pdf_file = report_agent._save_pdf_report(seismic_report, str(output_dir),
                                                         visualization_files={'velocity_model': results.get('visualization_file', '')},
                                                         filename='seismic_report')
                if pdf_file:
                    report_files['report_pdf'] = pdf_file
                
                if results.get('visualization_file'):
                    report_files['visualization_seismic'] = results['visualization_file']
                
                interpretation = results.get('interpretation', interpretation)
            else:
                update_progress("Seismic inversion failed", 1.0, results.get('error', 'Unknown error'))
                raise ValueError(f"Seismic processing failed: {results.get('error')}")

        elif workflow_type == 'custom':
            # Handle out-of-scope requests with CodeGenerationAgent
            from .code_generation_agent import CodeGenerationAgent
            code_agent = CodeGenerationAgent(api_key=api_key, model=llm_model, llm_provider=llm_provider)
            
            update_progress("Analyzing custom request", 0.20, "Checking if standard workflow applies")
            
            # First, check if this is truly out of scope
            scope_check = code_agent.check_request_scope(
                workflow_config.get('user_request', ''),
                workflow_config
            )
            
            if scope_check.get('in_scope', True) and not scope_check.get('out_of_scope_parts'):
                # Request is in scope but we couldn't detect the workflow type
                # This likely means missing data files
                raise ValueError(
                    'Could not infer workflow type from config! '
                    'Please provide at least ert_file or data_file.\n'
                    f"Recommendation: {scope_check.get('recommendation', 'Check your input files')}"
                )
            
            # Request is out of scope - use code generation
            print(f"  → Out of scope parts: {scope_check.get('out_of_scope_parts', [])}")
            print(f"  → Recommendation: {scope_check.get('recommendation', 'Using code generation')}")
            
            update_progress("Generating custom code", 0.40, "Using LLM to write analysis code")
            
            # Prepare available data for code generation
            available_data = {
                'workflow_config': workflow_config,
                'output_dir': str(output_dir)
            }
            
            # Add any file paths from config
            for key in ['ert_file', 'data_file', 'seismic_file', 'electrode_file']:
                if workflow_config.get(key):
                    available_data[key] = workflow_config[key]
            
            code_input = {
                'user_request': workflow_config.get('user_request', ''),
                'available_data': available_data,
                'output_dir': str(output_dir / 'custom'),
                'context': f"Out of scope parts: {scope_check.get('out_of_scope_parts', [])}"
            }
            
            code_results = code_agent.execute(code_input)
            
            update_progress("Custom analysis complete", 0.80, 
                          "Success" if code_results.get('status') == 'success' else "Failed")
            
            # Set execution plan for custom workflow
            execution_plan = [
                {'step': 'Check Request Scope', 'agent': 'CodeGenerationAgent', 
                 'description': 'Analyze if request is within standard capabilities', 
                 'outputs': ['scope_check']},
                {'step': 'Generate Custom Code', 'agent': 'CodeGenerationAgent', 
                 'description': 'Use LLM to write analysis code', 
                 'outputs': ['python_code']},
                {'step': 'Execute Code', 'agent': 'CodeGenerationAgent', 
                 'description': 'Run generated code safely', 
                 'outputs': ['results', 'outputs']},
            ]
            
            interpretation = code_results.get('interpretation', 
                'Custom analysis attempted. See code output for details.')
            
            results = {
                'status': code_results.get('status', 'failed'),
                'custom_analysis': True,
                'code_results': code_results,
                'out_of_scope_parts': scope_check.get('out_of_scope_parts', []),
                'recommendation': scope_check.get('recommendation', '')
            }
            
            # Generate report with custom analysis results
            if code_results.get('status') == 'success':
                update_progress("Generating report", 0.90, "Documenting custom analysis")
                from .report_agent import ReportAgent
                report_agent = ReportAgent(api_key=api_key, model=llm_model, llm_provider=llm_provider)
                
                # Create custom report
                custom_report = f"""# Custom Analysis Report

**Generated by:** PyHydroGeophysX CodeGenerationAgent
**Request:** {workflow_config.get('user_request', 'Not specified')}

## Analysis Summary

{interpretation}

## Out-of-Scope Components

The following parts of your request were not covered by standard workflows:
{chr(10).join('- ' + part for part in scope_check.get('out_of_scope_parts', ['Custom analysis required']))}

## Generated Code

The following Python code was generated to address your request:

```python
{code_results.get('code', 'No code generated')}
```

## Execution Output

```
{code_results.get('output', 'No output captured')}
```

## Files Generated

- Code file: {code_results.get('code_file', 'N/A')}
- Output directory: {code_results.get('output_dir', str(output_dir))}

---
*Report generated automatically by PyHydroGeophysX CodeGenerationAgent*
"""
                # Save report
                report_file = output_dir / 'custom_analysis_report.md'
                with open(report_file, 'w', encoding='utf-8') as f:
                    f.write(custom_report)
                
                report_files = {'report_markdown': str(report_file)}
                
                # Try to save HTML version
                html_file = report_agent._save_html_report(custom_report, str(output_dir), 'custom_analysis_report')
                if html_file:
                    report_files['report_html'] = html_file
                
                # Save PDF version
                pdf_file = report_agent._save_pdf_report(custom_report, str(output_dir), 
                                                         filename='custom_analysis_report')
                if pdf_file:
                    report_files['report_pdf'] = pdf_file
        
        else:
            raise ValueError('Unknown workflow type!')

        return results, execution_plan, interpretation, report_files
