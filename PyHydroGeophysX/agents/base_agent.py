"""
Base Agent Class for Multi-Agent System

Provides the foundation for all specialized agents in the workflow.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import os
import json
from pathlib import Path


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
        with open(output_file, 'w') as f:
            json.dump({
                'agent': self.name,
                'context': {k: str(v) for k, v in self.context.items()},
                'results': {k: str(v) for k, v in self.results.items()}
            }, f, indent=2)
        
        return output_file
    
    @staticmethod
    def run_unified_agent_workflow(workflow_config, api_key, llm_model, llm_provider, output_dir):
        """
        Unified agent workflow: infers task type from config and runs the appropriate pipeline.
        Supported: data fusion, time-lapse, direct ERT conversion.
        Returns: results dict, execution plan, interpretation, report files
        """
        # 1. Infer workflow type from configuration keys
        fusion_keys = {'fusion_pattern', 'methods', 'seismic_file', 'ert_file', 'velocity_threshold'}
        time_lapse_keys = {'timelapse', 'timelapse_files', 'timelapse_params', 'climate_config'}
        direct_ert_keys = {'ert_file', 'petrophysics_only', 'direct_ert', 'petrophysical_params'}

        config_keys = set(workflow_config.keys())

        if fusion_keys & config_keys:
            workflow_type = 'data_fusion'
        elif time_lapse_keys & config_keys:
            workflow_type = 'time_lapse'
        elif direct_ert_keys & config_keys:
            workflow_type = 'direct_ert'
        else:
            # Default to data fusion if we have ERT and seismic files
            if workflow_config.get('ert_file') and workflow_config.get('seismic_file'):
                workflow_type = 'data_fusion'
            elif workflow_config.get('ert_file'):
                workflow_type = 'direct_ert'
            else:
                raise ValueError('Could not infer workflow type from config!')

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

            print('\nExecuting complete data fusion workflow...')
            results = fusion_agent.execute_full_workflow(fusion_input)

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
                    report_files = report_results.get('visualization_files', {})

        elif workflow_type == 'time_lapse':
            # Use ERT agents for time-lapse workflow
            from .ert_loader_agent import ERTLoaderAgent
            from .ert_inversion_agent import ERTInversionAgent
            from .inversion_evaluation_agent import InversionEvaluationAgent
            ert_loader = ERTLoaderAgent(api_key=api_key, model=llm_model, llm_provider=llm_provider)
            ert_inversion = ERTInversionAgent(api_key=api_key, model=llm_model, llm_provider=llm_provider)
            eval_agent = InversionEvaluationAgent(api_key=api_key, model=llm_model, llm_provider=llm_provider)

            print('Running time-lapse ERT workflow...')

            # Load time-lapse data files
            time_lapse_files = workflow_config.get('timelapse_files', [])
            if not time_lapse_files:
                raise ValueError('No time-lapse files specified in configuration')

            time_lapse_data = []
            for i, data_file in enumerate(time_lapse_files):
                print(f'Loading dataset {i+1}/{len(time_lapse_files)}: {Path(data_file).name}')
                result = ert_loader.execute({
                    'data_file': data_file,
                    'instrument': workflow_config.get('instrument', 'E4D'),
                    'project_dir': workflow_config.get('project_dir', '.'),
                    'crs': workflow_config.get('crs', 'local')
                })
                if result['status'] != 'success':
                    print(f'Failed to load {data_file}: {result.get("error")}')
                    continue
                time_lapse_data.append(result['ert_data'])

            if len(time_lapse_data) < 2:
                raise ValueError(f'Need at least 2 datasets for time-lapse, got {len(time_lapse_data)}')

            # Run time-lapse inversion
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

            results = ert_inversion.execute(inversion_input)

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
                    'output_dir': str(output_dir)
                }
                report_results = report_agent.generate_timelapse_report(report_input)
                if report_results.get('status') == 'success':
                    report_files = report_results.get('visualization_files', {})

        elif workflow_type == 'direct_ert':
            # Use ERT agents for direct ERT to water content conversion
            from .ert_loader_agent import ERTLoaderAgent
            from .ert_inversion_agent import ERTInversionAgent
            from .water_content_agent import WaterContentAgent
            from .inversion_evaluation_agent import InversionEvaluationAgent
            ert_loader = ERTLoaderAgent(api_key=api_key, model=llm_model, llm_provider=llm_provider)
            ert_inversion = ERTInversionAgent(api_key=api_key, model=llm_model, llm_provider=llm_provider)
            water_content_agent = WaterContentAgent(api_key=api_key, model=llm_model, llm_provider=llm_provider)
            eval_agent = InversionEvaluationAgent(api_key=api_key, model=llm_model, llm_provider=llm_provider)

            print('Running direct ERT to water content workflow...')

            # Load ERT data
            data_file = workflow_config.get('ert_file')
            if not data_file:
                raise ValueError('No ERT file specified in configuration')

            print(f'Loading ERT data: {Path(data_file).name}')
            load_result = ert_loader.execute({
                'data_file': data_file,
                'instrument': workflow_config.get('instrument', 'DAS-1'),
                'project_dir': workflow_config.get('project_dir', '.'),
                'crs': workflow_config.get('crs', 'local')
            })

            if load_result['status'] != 'success':
                raise ValueError(f'Failed to load ERT data: {load_result.get("error")}')

            ert_data = load_result['ert_data']

            # Run inversion
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

            # Evaluate inversion quality
            evaluation_results = None
            if inversion_results.get('status') == 'success':
                print('Evaluating inversion quality...')
                eval_input = {
                    'inversion_results': inversion_results,
                    'ert_data': ert_data,
                    'quality_threshold': 0.7
                }
                evaluation_results = eval_agent.execute(eval_input)

            # Convert to water content
            petro_input = {
                'resistivity_model': inversion_results['resistivity_model'],
                'mesh': inversion_results['mesh'],
                'cell_markers': inversion_results.get('cell_markers'),
                'petrophysical_params': workflow_config.get('petrophysical_params', {}),
                'geological_context': workflow_config.get('geological_context', 'generic subsurface materials'),
                'n_realizations': workflow_config.get('n_realizations', 200),
                'output_dir': str(output_dir / 'petrophysics')
            }

            petro_results = water_content_agent.execute(petro_input)

            if petro_results.get('status') != 'success':
                raise ValueError(f'Petrophysics conversion failed: {petro_results.get("error")}')

            # Combine results
            results = {
                'status': 'success',
                'ert_data': ert_data,
                'inversion_results': inversion_results,
                'evaluation_results': evaluation_results,
                'petrophysics_results': petro_results,
                'water_content_mean': petro_results.get('water_content_mean'),
                'water_content_std': petro_results.get('water_content_std')
            }

            # Generate comprehensive report
            if results.get('status') == 'success':
                from .report_agent import ReportAgent
                report_agent = ReportAgent(api_key=api_key, model=llm_model, llm_provider=llm_provider)
                workflow_data = {
                    'ert_data': {
                        'n_electrodes': len(ert_data.electrodes),
                        'n_measurements': len(ert_data.observations),
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
                    'water_content': {
                        'mesh': inversion_results['mesh'],
                        'water_content_mean': petro_results.get('water_content_mean'),
                        'water_content_std': petro_results.get('water_content_std'),
                        'layer_params_used': petro_results.get('layer_params_used', {}),
                        'n_realizations': workflow_config.get('n_realizations', 200)
                    }
                }
                report_input = {
                    'workflow_data': workflow_data,
                    'config': workflow_config,
                    'output_dir': str(output_dir)
                }
                report_results = report_agent.execute(report_input)
                if report_results.get('status') == 'success':
                    report_files = report_results.get('visualization_files', {})

        else:
            raise ValueError('Unknown workflow type!')

        return results, execution_plan, interpretation, report_files
