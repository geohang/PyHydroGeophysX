"""
Data Fusion Agent

Intelligent coordinator for multi-method geophysical workflows. This agent understands
which geophysical methods should work together and orchestrates complex data fusion
workflows like seismic-constrained ERT inversion.

The DataFusionAgent is designed to be extensible for future multi-method combinations.
"""

from typing import Dict, Any, Optional, List
import numpy as np
from .base_agent import BaseAgent


class DataFusionAgent(BaseAgent):
    """
    Agent for intelligent coordination of multi-method geophysical workflows.
    
    This agent understands common geophysical data fusion patterns and can
    recommend and execute appropriate multi-method workflows based on user
    requirements.
    
    Supported Fusion Patterns:
        - ERT + Seismic: Structure-constrained resistivity inversion
        - ERT + Gravity: Density-constrained models
        - Multiple Time-Lapse: Joint temporal inversion
        - (Extensible for future methods)
    """
    
    # Define known fusion patterns
    FUSION_PATTERNS = {
        'structure_constraint': {
            'methods': ['seismic', 'ert'],
            'description': 'Use seismic velocity interfaces to constrain ERT inversion',
            'workflow': ['seismic_inversion', 'interface_extraction', 'constrained_ert'],
            'benefits': 'Improved layer boundary resolution and reduced artifacts'
        },
        'petrophysics_integration': {
            'methods': ['ert', 'petrophysics'],
            'description': 'Convert resistivity to hydrological properties',
            'workflow': ['ert_inversion', 'petrophysics_conversion'],
            'benefits': 'Direct hydrological interpretation from geophysical data'
        },
        'full_integration': {
            'methods': ['seismic', 'ert', 'petrophysics'],
            'description': 'Structure-constrained ERT with hydrological conversion',
            'workflow': ['seismic_inversion', 'interface_extraction', 
                        'constrained_ert', 'petrophysics_conversion'],
            'benefits': 'Complete geological-to-hydrological workflow with constraints'
        }
    }
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None,
                 llm_provider: str = "openai"):
        """Initialize Data Fusion Agent."""
        super().__init__("data_fusion", api_key, model, llm_provider)
        self.system_message = """You are an expert in multi-method geophysical data fusion.
You understand how different geophysical methods complement each other and can recommend
optimal workflows for integrating multiple datasets. You know when to apply structural
constraints, joint inversions, and petrophysical transformations."""
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute multi-method data fusion workflow.
        
        Args:
            input_data: Dictionary containing:
                - fusion_pattern: Name of fusion pattern or 'auto' for LLM recommendation
                - methods: List of available methods (e.g., ['seismic', 'ert'])
                - workflow_config: Configuration for the fusion workflow
                - data: Dictionary of data for each method
                - output_dir: Directory for saving results
                
        Returns:
            Dictionary containing fused results and workflow metadata
        """
        self._log_execution("Starting multi-method data fusion")
        
        try:
            # Extract parameters
            fusion_pattern = input_data.get('fusion_pattern', 'auto')
            available_methods = input_data.get('methods', [])
            workflow_config = input_data.get('workflow_config', {})
            output_dir = input_data.get('output_dir', 'results/data_fusion')
            
            # Determine fusion pattern
            if fusion_pattern == 'auto' and self.api_key:
                self._log_execution("Requesting LLM to recommend fusion pattern")
                fusion_pattern = self._recommend_fusion_pattern(available_methods, workflow_config)
            elif fusion_pattern == 'auto':
                # Default pattern based on available methods
                fusion_pattern = self._default_pattern(available_methods)
            
            self._log_execution(f"Using fusion pattern: {fusion_pattern}")
            
            # Validate pattern
            if fusion_pattern not in self.FUSION_PATTERNS:
                raise ValueError(f"Unknown fusion pattern: {fusion_pattern}")
            
            pattern_info = self.FUSION_PATTERNS[fusion_pattern]
            
            # Check if all required methods are available
            required_methods = pattern_info['methods']
            missing = [m for m in required_methods if m not in available_methods]
            if missing:
                raise ValueError(f"Fusion pattern '{fusion_pattern}' requires methods: {missing}")
            
            self._log_execution(f"Pattern description: {pattern_info['description']}")
            self._log_execution(f"Workflow steps: {pattern_info['workflow']}")
            
            # Get workflow execution plan
            execution_plan = self._create_execution_plan(
                fusion_pattern,
                workflow_config,
                input_data
            )
            
            self._log_execution(f"Execution plan created with {len(execution_plan)} steps")
            
            # Get LLM interpretation if available
            interpretation = None
            if self.api_key:
                interpretation = self._interpret_fusion_strategy(
                    fusion_pattern,
                    pattern_info,
                    workflow_config
                )
            
            self.results = {
                'status': 'success',
                'fusion_pattern': fusion_pattern,
                'pattern_info': pattern_info,
                'execution_plan': execution_plan,
                'interpretation': interpretation,
                'output_dir': output_dir
            }
            
            self._log_execution("Data fusion planning completed successfully")
            
            return self.results
            
        except Exception as e:
            self._log_execution(f"Error during data fusion: {str(e)}", level='ERROR')
            self.results = {
                'status': 'failed',
                'error': str(e)
            }
            raise
    
    def _recommend_fusion_pattern(self, available_methods: List[str], 
                                  workflow_config: Dict) -> str:
        """
        Use LLM to recommend optimal fusion pattern.
        
        Args:
            available_methods: List of available geophysical methods
            workflow_config: User's workflow configuration
            
        Returns:
            Recommended fusion pattern name
        """
        try:
            methods_desc = ', '.join(available_methods)
            config_desc = '\n'.join([f"- {k}: {v}" for k, v in workflow_config.items()])
            
            patterns_desc = '\n'.join([
                f"- {name}: {info['description']}"
                for name, info in self.FUSION_PATTERNS.items()
            ])
            
            prompt = f"""You have the following geophysical methods available: {methods_desc}

User workflow configuration:
{config_desc}

Available fusion patterns:
{patterns_desc}

Which fusion pattern is most appropriate for this workflow? 
Respond with ONLY the pattern name (e.g., 'structure_constraint', 'full_integration')."""
            
            response = self.query_llm(prompt, self.system_message, 
                                     temperature=0.3, max_tokens=50)
            
            # Extract pattern name from response
            pattern = response.strip().lower()
            
            # Validate it's a known pattern
            for known_pattern in self.FUSION_PATTERNS.keys():
                if known_pattern in pattern:
                    return known_pattern
            
            # Default fallback
            self._log_execution("Could not parse LLM response, using default pattern")
            return self._default_pattern(available_methods)
            
        except Exception as e:
            self._log_execution(f"LLM recommendation failed: {e}, using default")
            return self._default_pattern(available_methods)
    
    def _default_pattern(self, available_methods: List[str]) -> str:
        """
        Determine default fusion pattern based on available methods.
        
        Args:
            available_methods: List of available methods
            
        Returns:
            Default pattern name
        """
        methods_set = set(available_methods)
        
        # Check for full integration
        if {'seismic', 'ert', 'petrophysics'}.issubset(methods_set):
            return 'full_integration'
        
        # Check for structure constraint
        if {'seismic', 'ert'}.issubset(methods_set):
            return 'structure_constraint'
        
        # Check for petrophysics only
        if {'ert', 'petrophysics'}.issubset(methods_set):
            return 'petrophysics_integration'
        
        raise ValueError(f"No suitable fusion pattern for methods: {available_methods}")
    
    def _create_execution_plan(self, pattern: str, workflow_config: Dict,
                              input_data: Dict) -> List[Dict[str, Any]]:
        """
        Create detailed execution plan for the fusion workflow.
        
        Args:
            pattern: Fusion pattern name
            workflow_config: Workflow configuration
            input_data: Input data dictionary
            
        Returns:
            List of execution steps
        """
        pattern_info = self.FUSION_PATTERNS[pattern]
        workflow_steps = pattern_info['workflow']
        
        execution_plan = []
        
        for step in workflow_steps:
            if step == 'seismic_inversion':
                execution_plan.append({
                    'step': 'seismic_inversion',
                    'agent': 'SeismicAgent',
                    'description': 'Invert seismic travel time data to obtain velocity model',
                    'inputs': {
                        'seismic_data': input_data.get('data', {}).get('seismic'),
                        'velocity_threshold': workflow_config.get('velocity_threshold', 1200),
                        'inversion_params': workflow_config.get('seismic_params', {})
                    },
                    'outputs': ['velocity_model', 'mesh', 'interface_coords']
                })
            
            elif step == 'interface_extraction':
                execution_plan.append({
                    'step': 'interface_extraction',
                    'agent': 'SeismicAgent',
                    'description': 'Extract velocity interface for structural constraints',
                    'inputs': {
                        'velocity_model': 'from:seismic_inversion',
                        'threshold': workflow_config.get('velocity_threshold', 1200)
                    },
                    'outputs': ['interface_x', 'interface_z']
                })
            
            elif step == 'constrained_ert':
                execution_plan.append({
                    'step': 'constrained_ert',
                    'agent': 'StructureConstraintAgent',
                    'description': 'Run ERT inversion with seismic structural constraints',
                    'inputs': {
                        'ert_data': input_data.get('data', {}).get('ert'),
                        'interface_coords': 'from:interface_extraction',
                        'inversion_params': workflow_config.get('ert_params', {})
                    },
                    'outputs': ['resistivity_model', 'constrained_mesh', 'coverage']
                })
            
            elif step == 'ert_inversion':
                execution_plan.append({
                    'step': 'ert_inversion',
                    'agent': 'ERTInversionAgent',
                    'description': 'Run standard ERT inversion',
                    'inputs': {
                        'ert_data': input_data.get('data', {}).get('ert'),
                        'inversion_params': workflow_config.get('ert_params', {})
                    },
                    'outputs': ['resistivity_model', 'mesh', 'coverage']
                })
            
            elif step == 'petrophysics_conversion':
                execution_plan.append({
                    'step': 'petrophysics_conversion',
                    'agent': 'PetrophysicsAgent',
                    'description': 'Convert resistivity to water content with uncertainty',
                    'inputs': {
                        'resistivity_model': 'from:constrained_ert' if 'constrained_ert' in workflow_steps else 'from:ert_inversion',
                        'mesh': 'from:constrained_ert' if 'constrained_ert' in workflow_steps else 'from:ert_inversion',
                        'layer_markers': 'from:mesh',
                        'petrophysics_params': workflow_config.get('petrophysics_params', {}),
                        'use_monte_carlo': workflow_config.get('uncertainty_analysis', True)
                    },
                    'outputs': ['water_content_mean', 'water_content_std', 
                               'saturation_mean', 'porosity']
                })
        
        return execution_plan
    
    def _interpret_fusion_strategy(self, pattern: str, pattern_info: Dict,
                                   workflow_config: Dict) -> str:
        """
        Generate LLM interpretation of the fusion strategy.
        
        Args:
            pattern: Fusion pattern name
            pattern_info: Pattern information
            workflow_config: Workflow configuration
            
        Returns:
            Interpretation string
        """
        try:
            prompt = f"""Explain the benefits and workflow of this geophysical data fusion strategy:

Pattern: {pattern}
Description: {pattern_info['description']}
Methods: {', '.join(pattern_info['methods'])}
Workflow: {' → '.join(pattern_info['workflow'])}

Provide a brief explanation (3-4 sentences) suitable for a user about:
1. Why this fusion approach is valuable
2. What advantages it provides over single-method approaches
3. What to expect from the results"""
            
            interpretation = self.query_llm(prompt, self.system_message,
                                           temperature=0.5, max_tokens=250)
            return interpretation
        except:
            return pattern_info['benefits']
    
    def get_available_patterns(self) -> Dict[str, Dict]:
        """
        Get information about all available fusion patterns.
        
        Returns:
            Dictionary of fusion patterns with descriptions
        """
        return self.FUSION_PATTERNS.copy()
    
    def validate_workflow(self, available_methods: List[str], 
                         desired_pattern: str) -> Dict[str, Any]:
        """
        Validate if a desired fusion pattern can be executed with available methods.
        
        Args:
            available_methods: List of available methods
            desired_pattern: Desired fusion pattern name
            
        Returns:
            Validation result dictionary
        """
        if desired_pattern not in self.FUSION_PATTERNS:
            return {
                'valid': False,
                'error': f"Unknown fusion pattern: {desired_pattern}",
                'available_patterns': list(self.FUSION_PATTERNS.keys())
            }
        
        pattern_info = self.FUSION_PATTERNS[desired_pattern]
        required = set(pattern_info['methods'])
        available = set(available_methods)
        
        missing = required - available
        
        if missing:
            return {
                'valid': False,
                'error': f"Missing required methods: {list(missing)}",
                'required': list(required),
                'available': list(available)
            }
        
        return {
            'valid': True,
            'pattern': desired_pattern,
            'description': pattern_info['description'],
            'workflow': pattern_info['workflow']
        }
    
    def _log_execution(self, message: str, level: str = 'INFO'):
        """Log execution message."""
        print(f"[{self.name}] [{level}] {message}")
