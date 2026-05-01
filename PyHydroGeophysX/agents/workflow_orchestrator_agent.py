"""
Workflow Orchestrator Agent

Intelligent agent that automatically detects workflow type from configuration
and routes to appropriate specialized agents. This provides a unified interface
for all PyHydroGeophysX workflows.

Supported Workflows:
    - Standard ERT inversion with water content conversion
    - Time-lapse ERT monitoring with climate integration
    - Multi-method data fusion (seismic + ERT)
"""

from typing import Any, Dict, List, Optional

import numpy as np

from .base_agent import BaseAgent


# ---------------------------------------------------------------------------
# Workflow Orchestrator Agent
# ---------------------------------------------------------------------------
class WorkflowOrchestratorAgent(BaseAgent):
    """
    Master orchestrator that detects workflow type and coordinates execution.
    
    This agent provides a single entry point for all workflows:
    1. Analyzes the configuration to determine workflow type
    2. Generates a workflow description and agent plan
    3. Routes to appropriate specialized agents
    4. Aggregates and returns results
    
    The agent can handle:
    - Simple ERT → water content workflows
    - Complex time-lapse + climate workflows
    - Multi-method data fusion workflows
    """
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None,
                 llm_provider: str = "openai"):
        """Initialize Workflow Orchestrator Agent."""
        super().__init__("workflow_orchestrator", api_key, model, llm_provider)
        self.system_message = """You are an expert workflow orchestrator for geophysical data processing.
You analyze configurations and determine the appropriate workflow type and required agents.
You provide clear descriptions of what will be done and which agents will be used."""
        
        # Track registered agents
        self.registered_agents = {}
    
    def register_agent(self, name: str, agent: Any):
        """
        Register a specialized agent for use in workflows.
        
        Args:
            name: Agent identifier (e.g., 'ert_loader', 'ert_inversion')
            agent: Agent instance
        """
        self.registered_agents[name] = agent
        self._log_execution(f"Registered agent: {name}")
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze config, determine workflow, and execute.
        
        Args:
            input_data: Dictionary containing:
                - workflow_config: Configuration from ContextInputAgent
                - coordinator: Optional AgentCoordinator instance (if available)
                - output_dir: Base output directory
                
        Returns:
            Dictionary containing:
                - status: 'success' or 'failed'
                - workflow_type: Detected workflow type
                - workflow_description: Human-readable description
                - agent_plan: List of agents that will be used
                - results: Workflow execution results
        """
        self._log_execution("Starting workflow orchestration")
        
        try:
            workflow_config = input_data.get('workflow_config', {})
            coordinator = input_data.get('coordinator')
            output_dir = input_data.get('output_dir', 'results/orchestrated_workflow')
            
            # Detect workflow type
            workflow_type = self._detect_workflow_type(workflow_config)
            self._log_execution(f"Detected workflow type: {workflow_type}")
            
            # Generate workflow description and agent plan
            workflow_description, agent_plan = self._generate_workflow_plan(
                workflow_type, workflow_config
            )
            
            self._log_execution(f"Workflow description: {workflow_description}")
            self._log_execution(f"Agent plan: {', '.join(agent_plan)}")
            
            # Get LLM interpretation if available
            interpretation = None
            if self.api_key:
                interpretation = self._interpret_workflow(
                    workflow_type,
                    workflow_description,
                    agent_plan,
                    workflow_config
                )
            
            # Execute workflow (if coordinator provided)
            execution_results = None
            if coordinator:
                self._log_execution("Executing workflow with coordinator...")
                execution_results = self._execute_workflow(
                    workflow_type,
                    workflow_config,
                    coordinator,
                    output_dir
                )
            else:
                self._log_execution("No coordinator provided - returning plan only")
            
            self.results = {
                'status': 'success',
                'workflow_type': workflow_type,
                'workflow_description': workflow_description,
                'agent_plan': agent_plan,
                'interpretation': interpretation,
                'execution_results': execution_results,
                'output_dir': output_dir
            }
            
            self._log_execution("Workflow orchestration completed successfully")
            return self.results
            
        except Exception as e:
            self._log_execution(f"Error during orchestration: {str(e)}", level='ERROR')
            self.results = {
                'status': 'failed',
                'error': str(e)
            }
            raise
    
    def _detect_workflow_type(self, config: Dict[str, Any]) -> str:
        """
        Detect workflow type from configuration.

        This is the single authoritative implementation used both by this agent
        and by ``BaseAgent.run_unified_agent_workflow``.

        Args:
            config: Workflow configuration dictionary

        Returns:
            Workflow type string.  Possible values:
            'tdem', 'seismic', 'model_output', 'time_lapse', 'data_fusion',
            'ert_data_process', 'direct_ert', 'standard_ert', 'custom'
        """
        config_keys = set(config.keys())
        user_request_lower = config.get('user_request', '').lower()

        # ── Intent flags ────────────────────────────────────────────────────
        mentions_ert = ('ert' in user_request_lower) or bool(config.get('ert_file'))
        mentions_seismic = (
            ('seismic' in user_request_lower)
            or bool(config.get('seismic_file') or config.get('raw_seismic_file'))
        )
        mentions_tdem = ('tdem' in user_request_lower) or bool(config.get('tdem_file'))
        inversion_keywords = ['invert', 'inversion', 'tomography', 'forward']
        mentions_inversion = any(kw in user_request_lower for kw in inversion_keywords)

        hydro_keywords = [
            'modflow', 'parflow', 'par flow', 'hydrological model',
            'watercontent', 'saturation', 'porosity',
        ]
        mentions_hydro = (
            any(kw in user_request_lower for kw in hydro_keywords)
            or bool(config.get('hydro_model'))
        )
        hydro_only = mentions_hydro and not (
            mentions_ert or mentions_seismic or mentions_tdem or mentions_inversion
        )

        processing_keywords = [
            'data processing', 'quality control', 'qc', 'preprocess', 'export', 'resipy',
        ]
        ert_processing = bool(config.get('ert_data_processing')) or (
            bool(config.get('ert_file'))
            and any(kw in user_request_lower for kw in processing_keywords)
            and not any(kw in user_request_lower for kw in inversion_keywords)
        )

        # ── Priority order ───────────────────────────────────────────────────
        # 1. TDEM
        if (
            config.get('tdem_file')
            or config.get('tdem_data')
            or 'tdem' in user_request_lower
            or 'tem ' in user_request_lower
            or 'electromagnetic' in user_request_lower
        ):
            return 'tdem'

        # 2. Standalone seismic
        if (
            (config.get('raw_seismic_file') and not config.get('ert_file'))
            or (config.get('seismic_file') and not config.get('ert_file'))
            or config.get('seismic_only', False)
            or (
                'seismic' in user_request_lower
                and 'ert' not in user_request_lower
                and 'resistivity' not in user_request_lower
                and 'fusion' not in user_request_lower
            )
            or 'srt inversion' in user_request_lower
            or 'seismic refraction' in user_request_lower
            or 'travel time' in user_request_lower
        ):
            return 'seismic'

        # 3. Hydrological model output (MODFLOW / ParFlow)
        if config.get('hydro_model') in ['modflow', 'parflow', 'both'] or hydro_only:
            return 'model_output'

        # 4. Time-lapse ERT
        if not hydro_only and (
            'timelapse_files' in config_keys
            or 'time_lapse_files' in config_keys
            or 'timelapse_params' in config_keys
            or config.get('inversion_mode') == 'time-lapse'
        ):
            return 'time_lapse'

        # 5. Data fusion (ERT + seismic or explicit fusion config)
        if (
            config.get('velocity_threshold')
            or (config.get('ert_file') and (config.get('seismic_file') or config.get('raw_seismic_file')))
            or (
                config.get('fusion_pattern')
                and config.get('fusion_pattern') not in [None, 'None', '']
            )
            or (
                config.get('methods')
                and len(config.get('methods', [])) > 1
                and 'seismic' in config.get('methods', [])
            )
            or config.get('use_seismic')
        ):
            return 'data_fusion'

        # 6. ERT data processing (QC / export, no inversion)
        if ert_processing:
            return 'ert_data_process'

        # 7. Standard / direct ERT
        if config.get('ert_file'):
            return 'direct_ert'

        # 8. Unknown — attempt custom code generation
        return 'custom'
    
    def _generate_workflow_plan(self, workflow_type: str, 
                                config: Dict[str, Any]) -> tuple:
        """
        Generate workflow description and agent execution plan.
        
        Args:
            workflow_type: Type of workflow
            config: Workflow configuration
            
        Returns:
            Tuple of (description, agent_list)
        """
        if workflow_type == 'standard_ert':
            description = "Standard ERT inversion with water content conversion"
            agents = ['ert_loader', 'ert_inversion', 'water_content', 'report']
            
            if config.get('use_climate'):
                agents.insert(0, 'climate_data')
                description += " and climate integration"
        
        elif workflow_type == 'time_lapse':
            n_files = len(config.get('time_lapse_files', []))
            method = config.get('time_lapse_method', 'difference')
            
            description = f"Time-lapse ERT monitoring ({method} method) with {n_files} timesteps"
            agents = ['ert_loader', 'ert_inversion', 'report']
            
            if config.get('use_climate'):
                agents.insert(0, 'climate_data')
                description += " and climate correlation"
            
            if config.get('run_uncertainty'):
                agents.insert(-1, 'water_content')
                description += " with water content conversion"
        
        elif workflow_type == 'data_fusion':
            fusion_pattern = config.get('fusion_pattern', 'structure_constraint')
            methods = config.get('methods', ['seismic', 'ert'])
            
            description = f"Multi-method data fusion ({', '.join(methods)})"
            
            if fusion_pattern == 'full_integration':
                agents = ['seismic_processor', 'data_fusion', 'water_content', 'report']
                description += " with full structure-constrained hydrogeophysical workflow"
            elif fusion_pattern == 'structure_constraint':
                agents = ['seismic_processor', 'data_fusion', 'ert_inversion', 'report']
                description += " with structure-constrained ERT inversion"
            else:
                agents = ['data_fusion', 'report']
        else:
            description = "Unknown workflow type"
            agents = []
        
        return description, agents
    
    def _interpret_workflow(self, workflow_type: str, description: str,
                           agent_plan: List[str], config: Dict) -> str:
        """
        Generate LLM interpretation of the workflow.
        
        Args:
            workflow_type: Type of workflow
            description: Workflow description
            agent_plan: List of agents to be used
            config: Configuration dictionary
            
        Returns:
            Interpretation string
        """
        try:
            config_summary = self._summarize_config(config)
            
            prompt = f"""Explain this geophysical workflow in simple terms:

Workflow Type: {workflow_type}
Description: {description}
Agents: {', '.join(agent_plan)}

Configuration highlights:
{config_summary}

Provide a 2-3 sentence explanation suitable for a user about:
1. What data will be processed
2. What methods will be applied
3. What results they will get"""
            
            interpretation = self.query_llm(prompt, self.system_message,
                                          temperature=0.5, max_tokens=200)
            return interpretation
        except:
            return description
    
    def _summarize_config(self, config: Dict) -> str:
        """Create human-readable summary of key config parameters."""
        summary_parts = []
        
        # Data source
        if config.get('data_file'):
            summary_parts.append(f"- Data: {config['data_file']}")
        elif config.get('time_lapse_files'):
            n_files = len(config['time_lapse_files'])
            summary_parts.append(f"- Data: {n_files} time-lapse files")
        
        # Instrument
        if config.get('instrument'):
            summary_parts.append(f"- Instrument: {config['instrument']}")
        
        # Inversion parameters
        inv_params = config.get('inversion_params', {})
        if inv_params:
            lambda_val = inv_params.get('lambda', 20)
            summary_parts.append(f"- Regularization: λ={lambda_val}")
        
        # Special features
        if config.get('use_climate'):
            summary_parts.append("- Climate integration: enabled")
        
        if config.get('use_seismic'):
            summary_parts.append("- Seismic constraints: enabled")
        
        if config.get('run_uncertainty'):
            n_real = config.get('n_realizations', 100)
            summary_parts.append(f"- Uncertainty: {n_real} MC realizations")
        
        return '\n'.join(summary_parts) if summary_parts else "No key parameters identified"
    
    def _execute_workflow(self, workflow_type: str, config: Dict,
                         coordinator: Any, output_dir: str) -> Dict:
        """
        Execute the workflow using the coordinator.
        
        Args:
            workflow_type: Type of workflow
            config: Configuration dictionary
            coordinator: AgentCoordinator instance
            output_dir: Output directory
            
        Returns:
            Execution results dictionary
        """
        try:
            # Update config with output directory
            config['output_dir'] = output_dir
            
            # Execute using coordinator
            results = coordinator.execute_workflow(config)
            
            return results
        except Exception as e:
            self._log_execution(f"Workflow execution error: {str(e)}", level='ERROR')
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    def get_workflow_summary(self) -> Dict[str, Any]:
        """
        Get summary of orchestrated workflow.
        
        Returns:
            Summary dictionary
        """
        if not self.results:
            return {'error': 'No workflow has been orchestrated yet'}
        
        return {
            'workflow_type': self.results.get('workflow_type'),
            'description': self.results.get('workflow_description'),
            'agents': self.results.get('agent_plan'),
            'status': self.results.get('status'),
            'interpretation': self.results.get('interpretation')
        }
    
    def _log_execution(self, message: str, level: str = 'INFO'):
        """Log execution message."""
        print(f"[{self.name}] [{level}] {message}")

