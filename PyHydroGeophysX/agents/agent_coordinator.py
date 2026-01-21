"""
Agent Coordinator for Multi-Agent Workflow

Coordinates the execution of multiple specialized agents to complete
the full geophysical processing workflow. Supports cross-modal geophysical
data processing (ERT, seismic, and more) with multiple LLM API providers
(GPT, Gemini, Claude).
"""

from typing import Dict, Any, List, Optional
import os
import json
from datetime import datetime


class AgentCoordinator:
    """
    Coordinates multiple agents to execute a complete workflow.
    
    The coordinator manages cross-modal geophysical workflows such as: 
    "load geophysical data → process → invert → convert to hydrologic parameters → report"
    with support for multiple data types (ERT, seismic, etc.) and LLM providers (GPT, Gemini, Claude).
    """
    
    def __init__(self, api_key: Optional[str] = None, output_dir: str = "results/agents",
                 llm_provider: str = "openai"):
        """
        Initialize the agent coordinator.
        
        Args:
            api_key: LLM API key for agents
            output_dir: Directory for saving results
            llm_provider: LLM provider to use ('openai', 'gemini', or 'claude')
        """
        self.api_key = api_key or self._get_default_api_key(llm_provider)
        self.output_dir = output_dir
        self.llm_provider = llm_provider.lower()
        self.agents = {}
        self.workflow_state = {
            'status': 'initialized',
            'current_step': None,
            'completed_steps': [],
            'data': {}
        }
        self.execution_log = []
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def _get_default_api_key(self, provider: str) -> Optional[str]:
        """Get default API key based on provider."""
        provider_env_map = {
            'openai': 'OPENAI_API_KEY',
            'gemini': 'GEMINI_API_KEY',
            'claude': 'ANTHROPIC_API_KEY'
        }
        env_var = provider_env_map.get(provider.lower())
        return os.getenv(env_var) if env_var else None
    
    def register_agent(self, agent_name: str, agent_instance):
        """
        Register an agent with the coordinator.
        
        Args:
            agent_name: Unique identifier for the agent
            agent_instance: Agent instance to register
        """
        self.agents[agent_name] = agent_instance
        self._log(f"Registered agent: {agent_name}")
    
    def execute_workflow(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the complete workflow with registered agents.
        
        Args:
            config: Configuration dictionary containing:
                - data_file: Path to ERT data file
                - instrument: Instrument type (E4D, Syscal, etc.)
                - inversion_params: Parameters for inversion
                - petrophysical_params: Parameters for water content conversion
                - use_seismic: Whether to include seismic processing (default: False)
                - seismic_data: Optional seismic data file
                - use_climate: Whether to include climate data (default: False)
                - climate_config: Climate data configuration (coords/geometry, dates, etc.)
                - ert_timestamps: Timestamps for ERT acquisitions (for climate alignment)
                
        Returns:
            Dictionary containing workflow results
        """
        self._log("Starting workflow execution")
        self.workflow_state['status'] = 'running'
        
        try:
            # Step 0 (Optional): Fetch climate data if requested
            if config.get('use_climate', False) and 'climate_config' in config:
                self._log("Step 0: Fetching climate data")
                self.workflow_state['current_step'] = 'fetch_climate'
                
                climate_config = config['climate_config']
                # Add ERT timestamps if provided
                if 'ert_timestamps' in config:
                    climate_config['ert_timestamps'] = config['ert_timestamps']
                
                climate_results = self._execute_agent('climate_data', climate_config)
                self.workflow_state['data']['climate_data'] = climate_results
                self.workflow_state['completed_steps'].append('fetch_climate')
            
            # Step 1: Load ERT data
            self._log("Step 1: Loading ERT data")
            self.workflow_state['current_step'] = 'load_ert'
            ert_data = self._execute_agent('ert_loader', {
                'data_file': config.get('data_file'),
                'instrument': config.get('instrument', 'E4D'),
                'project_dir': config.get('project_dir', '.'),
                'crs': config.get('crs', 'local')
            })
            self.workflow_state['data']['ert_data'] = ert_data
            self.workflow_state['completed_steps'].append('load_ert')
            
            # Step 1b (Optional): Process seismic data if available
            if config.get('use_seismic', False) and 'seismic_data' in config:
                self._log("Step 1b: Processing seismic data")
                self.workflow_state['current_step'] = 'process_seismic'
                seismic_results = self._execute_agent('seismic_processor', {
                    'seismic_data': config['seismic_data'],
                    'velocity_threshold': config.get('velocity_threshold', 1200)
                })
                self.workflow_state['data']['seismic_structure'] = seismic_results
                self.workflow_state['completed_steps'].append('process_seismic')
            
            # Step 2: ERT Inversion
            self._log("Step 2: Running ERT inversion")
            self.workflow_state['current_step'] = 'invert_ert'
            
            inversion_input = {
                'ert_data': ert_data['ert_data'],  # Extract the actual ERT data object from result dict
                'inversion_params': config.get('inversion_params', {}),
            }
            
            # Add seismic structure if available
            if 'seismic_structure' in self.workflow_state['data']:
                inversion_input['seismic_structure'] = self.workflow_state['data']['seismic_structure']
                inversion_input['use_structure_constraint'] = True
            
            inversion_results = self._execute_agent('ert_inversion', inversion_input)
            self.workflow_state['data']['inversion_results'] = inversion_results
            self.workflow_state['completed_steps'].append('invert_ert')
            
            # Step 3: Convert to water content
            self._log("Step 3: Converting to water content")
            self.workflow_state['current_step'] = 'convert_to_wc'
            wc_results = self._execute_agent('water_content', {
                'inversion_results': inversion_results,
                'petrophysical_params': config.get('petrophysical_params', {}),
                'uncertainty_analysis': config.get('run_uncertainty', False),
                'n_realizations': config.get('n_realizations', 100)
            })
            self.workflow_state['data']['water_content'] = wc_results
            self.workflow_state['completed_steps'].append('convert_to_wc')
            
            # Step 4: Generate report
            self._log("Step 4: Generating report")
            self.workflow_state['current_step'] = 'generate_report'
            report = self._execute_agent('report', {
                'workflow_data': self.workflow_state['data'],
                'config': config,
                'output_dir': self.output_dir
            })
            self.workflow_state['data']['report'] = report
            self.workflow_state['completed_steps'].append('generate_report')
            
            # Mark workflow as complete
            self.workflow_state['status'] = 'completed'
            self.workflow_state['current_step'] = None
            self._log("Workflow completed successfully")
            
            # Save workflow state
            self._save_workflow_state()
            
            return {
                'status': 'success',
                'results': self.workflow_state['data'],
                'log': self.execution_log
            }
            
        except Exception as e:
            self.workflow_state['status'] = 'failed'
            self._log(f"Workflow failed: {str(e)}", level='ERROR')
            self._save_workflow_state()
            
            return {
                'status': 'failed',
                'error': str(e),
                'log': self.execution_log,
                'partial_results': self.workflow_state['data']
            }
    
    def _execute_agent(self, agent_name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a specific agent.
        
        Args:
            agent_name: Name of the agent to execute
            input_data: Input data for the agent
            
        Returns:
            Agent execution results
        """
        if agent_name not in self.agents:
            raise ValueError(f"Agent '{agent_name}' not registered")
        
        agent = self.agents[agent_name]
        self._log(f"Executing agent: {agent_name}")
        
        try:
            result = agent.execute(input_data)
            self._log(f"Agent {agent_name} completed successfully")
            return result
        except Exception as e:
            self._log(f"Agent {agent_name} failed: {str(e)}", level='ERROR')
            raise
    
    def _log(self, message: str, level: str = 'INFO'):
        """Add entry to execution log."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'message': message
        }
        self.execution_log.append(log_entry)
        print(f"[{level}] {message}")
    
    def _save_workflow_state(self):
        """Save current workflow state to file."""
        state_file = os.path.join(self.output_dir, 'workflow_state.json')
        
        # Convert non-serializable objects to strings
        serializable_state = {
            'status': self.workflow_state['status'],
            'current_step': self.workflow_state['current_step'],
            'completed_steps': self.workflow_state['completed_steps'],
            'data_keys': list(self.workflow_state['data'].keys())
        }
        
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_state, f, indent=2)
        
        # Save execution log
        log_file = os.path.join(self.output_dir, 'execution_log.json')
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(self.execution_log, f, indent=2)
    
    def get_workflow_summary(self) -> Dict[str, Any]:
        """Get summary of workflow execution."""
        return {
            'status': self.workflow_state['status'],
            'completed_steps': self.workflow_state['completed_steps'],
            'total_steps': len(self.workflow_state['completed_steps']),
            'current_step': self.workflow_state['current_step'],
            'available_results': list(self.workflow_state['data'].keys())
        }
