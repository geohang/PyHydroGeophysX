"""
Data Fusion Agent

Intelligent coordinator for multi-method geophysical workflows. This agent understands
which geophysical methods should work together and orchestrates complex data fusion
workflows like seismic-constrained ERT inversion.

The DataFusionAgent is designed to be extensible for future multi-method combinations.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from .base_agent import BaseAgent


# ---------------------------------------------------------------------------
# Data Fusion Agent
# ---------------------------------------------------------------------------
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
                        'velocity_threshold': workflow_config.get('velocity_threshold', 1000),
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
                        'threshold': workflow_config.get('velocity_threshold', 1000)
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
    
    def execute_full_workflow(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the complete multi-method data fusion workflow, not just planning.
        
        This method actually runs the agents and produces results, unlike execute()
        which only creates a plan.
        
        Args:
            input_data: Dictionary containing:
                - fusion_pattern: Name of fusion pattern
                - methods: List of available methods
                - workflow_config: Configuration for the fusion workflow
                - data: Dictionary of data for each method
                - output_dir: Directory for saving results
                
        Returns:
            Dictionary containing complete workflow results
        """
        self._log_execution("Starting complete multi-method data fusion workflow")
        
        try:
            # First get the execution plan
            plan_result = self.execute(input_data)
            if plan_result['status'] != 'success':
                return plan_result
            
            fusion_pattern = plan_result['fusion_pattern']
            execution_plan = plan_result['execution_plan']
            workflow_config = input_data.get('workflow_config', {})
            output_dir = input_data.get('output_dir', 'results/data_fusion')
            
            self._log_execution(f"Executing {len(execution_plan)} workflow steps")
            
            # Initialize results storage
            workflow_results = {
                'status': 'success',
                'fusion_pattern': fusion_pattern,
                'execution_plan': execution_plan,
                'interpretation': plan_result.get('interpretation'),
                'output_dir': output_dir,
                'step_results': {}
            }
            
            # Execute each step in the plan
            # IMPORTANT: workflow_results accumulates results from each step so later steps
            # can access outputs from earlier steps (e.g., interface_extraction needs results
            # from seismic_inversion)
            for i, step in enumerate(execution_plan, 1):
                step_name = step['step']
                agent_name = step['agent']
                
                self._log_execution(f"Step {i}/{len(execution_plan)}: {step_name} using {agent_name}")
                
                try:
                    # Pass workflow_results to methods that need access to previous step results
                    if agent_name == 'SeismicAgent':
                        step_result = self._execute_seismic_step(step, workflow_config, input_data, 
                                                                workflow_results, output_dir)
                    elif agent_name == 'StructureConstraintAgent':
                        step_result = self._execute_structure_constraint_step(step, workflow_config, input_data, 
                                                                             workflow_results, output_dir)
                    elif agent_name == 'ERTInversionAgent':
                        step_result = self._execute_ert_inversion_step(step, workflow_config, input_data, output_dir)
                    elif agent_name == 'PetrophysicsAgent':
                        step_result = self._execute_petrophysics_step(step, workflow_config, workflow_results, output_dir)
                    else:
                        raise ValueError(f"Unknown agent: {agent_name}")
                    
                    # Store this step's results so subsequent steps can access them
                    workflow_results['step_results'][step_name] = step_result
                    
                    if step_result.get('status') != 'success':
                        workflow_results['status'] = 'failed'
                        workflow_results['error'] = f"Step {step_name} failed: {step_result.get('error', 'Unknown error')}"
                        break
                    
                except Exception as e:
                    self._log_execution(f"Error in step {step_name}: {str(e)}", level='ERROR')
                    workflow_results['status'] = 'failed'
                    workflow_results['error'] = f"Step {step_name} failed: {str(e)}"
                    break
            
            # If all steps succeeded, add final results
            if workflow_results['status'] == 'success':
                self._log_execution("All workflow steps completed successfully")
                workflow_results.update(self._compile_final_results(workflow_results))
            
            return workflow_results
            
        except Exception as e:
            self._log_execution(f"Error during full workflow execution: {str(e)}", level='ERROR')
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    def _execute_seismic_step(self, step: Dict, workflow_config: Dict, 
                             input_data: Dict, workflow_results: Dict, 
                             output_dir: str) -> Dict[str, Any]:
        """Execute seismic-related steps using SeismicAgent."""
        from .seismic_agent import SeismicAgent
        
        seismic_agent = SeismicAgent(
            api_key=self.api_key,
            model=self.model,
            llm_provider=self.llm_provider
        )
        
        step_name = step['step']
        
        if step_name == 'seismic_inversion':
            # Load seismic data
            seismic_file = input_data.get('data', {}).get('seismic')
            if not seismic_file:
                raise ValueError("Seismic data file not provided")
            
            from pathlib import Path

            import pygimli.physics.traveltime as tt
            
            seismic_path = Path(seismic_file)
            if not seismic_path.exists():
                # Try relative to examples directory
                seismic_path = Path('examples') / seismic_path
                if not seismic_path.exists():
                    raise FileNotFoundError(f"Seismic file not found: {seismic_file}")
            
            ttData = tt.load(str(seismic_path))
            
            # IMPORTANT: Get velocity_threshold from user's config (required parameter)
            velocity_threshold = workflow_config.get('velocity_threshold')
            if velocity_threshold is None:
                raise ValueError("velocity_threshold is required in workflow_config")
            self._log_execution(f"Running seismic inversion with velocity_threshold={velocity_threshold} m/s from user config")
            
            seismic_input = {
                'seismic_data': ttData,
                'velocity_threshold': velocity_threshold,  # Must pass this!
                'inversion_params': workflow_config.get('seismic_params', {}),
                'output_dir': f"{output_dir}/seismic"
            }
            
            return seismic_agent.execute(seismic_input)
            
        elif step_name == 'interface_extraction':
            # Get interface from previous seismic step results
            # CRITICAL: SeismicAgent.execute() already does interface extraction!
            # We just need to extract the interface_coords from the previous results
            step_results = workflow_results.get('step_results', {})
            seismic_results = step_results.get('seismic_inversion')
            
            if not seismic_results or seismic_results.get('status') != 'success':
                raise ValueError("Seismic inversion results not available for interface extraction")
            
            # Interface coordinates were already extracted by SeismicAgent.execute()
            interface_coords = seismic_results.get('interface_coords')
            # Get velocity_threshold from seismic results (already used in inversion)
            velocity_threshold = seismic_results.get('velocity_threshold')
            
            if interface_coords is None:
                raise ValueError("Interface coordinates not found in seismic results")
            
            self._log_execution(f"Interface extracted: {len(interface_coords[0])} points at {velocity_threshold} m/s")
            
            # Return the interface extraction results
            return {
                'status': 'success',
                'interface_coords': interface_coords,
                'velocity_threshold': velocity_threshold,
                'output_dir': seismic_results.get('output_dir')
            }
        
        else:
            raise ValueError(f"Unknown seismic step: {step_name}")
    
    def _execute_structure_constraint_step(self, step: Dict, workflow_config: Dict,
                                          input_data: Dict, workflow_results: Dict,
                                          output_dir: str) -> Dict[str, Any]:
        """Execute structure constraint steps using StructureConstraintAgent."""
        try:
            from .structure_constraint_agent import StructureConstraintAgent
            
            structure_agent = StructureConstraintAgent(
                api_key=self.api_key,
                model=self.model,
                llm_provider=self.llm_provider
            )
            
            # Get ERT data
            ert_file = input_data.get('data', {}).get('ert')
            if not ert_file:
                raise ValueError("ERT data file not provided")
            
            self._log_execution(f"Loading ERT data from: {ert_file}")
            
            from pathlib import Path

            from pygimli.physics import ert as pygimli_ert
            
            ert_path = Path(ert_file)
            if not ert_path.exists():
                ert_path = Path('examples') / ert_path
                if not ert_path.exists():
                    raise FileNotFoundError(f"ERT file not found: {ert_file}")
            
            # Use PyGIMLi's direct ERT loader (more robust than RESIPY for BERT format)
            try:
                ertData = pygimli_ert.load(str(ert_path))
                self._log_execution(f"  Loaded using PyGIMLi's ert.load()")
            except Exception as e:
                # Fallback to RESIPY loader if PyGIMLi fails
                self._log_execution(f"  PyGIMLi load failed, trying RESIPY loader...")
                from PyHydroGeophysX.data_processing.ert_data_agent import load_ert_resipy
                ertData = load_ert_resipy(
                    project_dir=str(ert_path.parent),
                    data_file=str(ert_path),
                    instrument=workflow_config.get('instrument', 'BERT'),
                    electrode_file=None
                )
            
            self._log_execution(f"ERT data loaded: {ertData.sensorCount()} electrodes, {ertData.size()} measurements")
            
            # Get interface coordinates from interface_extraction step
            interface_coords = None
            if 'step_results' in workflow_results:
                interface_step = workflow_results['step_results'].get('interface_extraction')
                if interface_step and interface_step.get('status') == 'success':
                    interface_coords = interface_step.get('interface_coords')
                    self._log_execution(f"Got interface_coords from previous step: {len(interface_coords[0])} points")
                    self._log_execution(f"  Interface X range: {min(interface_coords[0]):.1f} - {max(interface_coords[0]):.1f}")
                    self._log_execution(f"  Interface Z range: {min(interface_coords[1]):.1f} - {max(interface_coords[1]):.1f}")
            
            if interface_coords is None:
                raise ValueError("Interface coordinates not available from previous step")
            
            # Get velocity_threshold from user's config
            velocity_threshold = workflow_config.get('velocity_threshold')
            if velocity_threshold is None:
                raise ValueError("velocity_threshold is required in workflow_config")
            
            # Since we already have interface_coords, we DON'T need seismic_data
            # (passing a file path string would cause issues)
            structure_input = {
                'ert_data': ertData,
                'seismic_data': None,  # Not needed - we have interface_coords
                'velocity_threshold': velocity_threshold,
                'interface_coords': interface_coords,
                'seismic_params': workflow_config.get('seismic_params', {}),
                'inversion_params': workflow_config.get('ert_params', {}),
                'output_dir': f"{output_dir}/structure_constrained",
                'mesh_quality': workflow_config.get('mesh_quality', 34),
                'mesh_params': workflow_config.get('mesh_params', {})
            }
            
            self._log_execution("Calling StructureConstraintAgent.execute()...")
            result = structure_agent.execute(structure_input)
            self._log_execution(f"StructureConstraintAgent returned status: {result.get('status')}")
            
            return result
            
        except Exception as e:
            import traceback
            self._log_execution(f"ERROR in structure constraint step: {str(e)}", level='ERROR')
            self._log_execution(f"Full traceback:\n{traceback.format_exc()}", level='ERROR')
            raise
    
    def _execute_ert_inversion_step(self, step: Dict, workflow_config: Dict,
                                   input_data: Dict, output_dir: str) -> Dict[str, Any]:
        """Execute ERT inversion steps using ERTInversionAgent."""
        from .ert_inversion_agent import ERTInversionAgent
        
        ert_agent = ERTInversionAgent(
            api_key=self.api_key,
            model=self.model,
            llm_provider=self.llm_provider
        )
        
        # Get ERT data
        ert_file = input_data.get('data', {}).get('ert')
        if not ert_file:
            raise ValueError("ERT data file not provided")
        
        from pathlib import Path

        from pygimli.physics import ert as pygimli_ert
        
        ert_path = Path(ert_file)
        if not ert_path.exists():
            ert_path = Path('examples') / ert_path
            if not ert_path.exists():
                raise FileNotFoundError(f"ERT file not found: {ert_file}")
        
        # Use PyGIMLi's direct ERT loader (more robust than RESIPY for BERT format)
        try:
            ertData = pygimli_ert.load(str(ert_path))
        except Exception as e:
            # Fallback to RESIPY loader if PyGIMLi fails
            from PyHydroGeophysX.data_processing.ert_data_agent import load_ert_resipy
            ertData = load_ert_resipy(
                project_dir=str(ert_path.parent),
                data_file=str(ert_path),
                instrument=workflow_config.get('instrument', 'BERT'),
                electrode_file=None
            )
        
        ert_input = {
            'ert_data': ertData,
            'inversion_params': workflow_config.get('ert_params', {}),
            'output_dir': f"{output_dir}/ert_inversion"
        }
        
        return ert_agent.execute(ert_input)
    
    def _execute_petrophysics_step(self, step: Dict, workflow_config: Dict,
                                  workflow_results: Dict, output_dir: str) -> Dict[str, Any]:
        """Execute petrophysics steps using PetrophysicsAgent."""
        from .petrophysics_agent import PetrophysicsAgent
        
        petro_agent = PetrophysicsAgent(
            api_key=self.api_key,
            model=self.model,
            llm_provider=self.llm_provider
        )
        
        # Get resistivity model and mesh from previous steps
        resistivity_model = None
        mesh = None
        cell_markers = None
        
        step_results = workflow_results.get('step_results', {})
        
        # Try constrained ERT first, then regular ERT
        if 'constrained_ert' in step_results:
            constrained_results = step_results['constrained_ert']
            if constrained_results.get('status') == 'success':
                resistivity_model = constrained_results.get('resistivity_model')
                mesh = constrained_results.get('mesh')
                cell_markers = constrained_results.get('cell_markers')
        
        if resistivity_model is None and 'ert_inversion' in step_results:
            ert_results = step_results['ert_inversion']
            if ert_results.get('status') == 'success':
                resistivity_model = ert_results.get('resistivity_model')
                mesh = ert_results.get('mesh')
                cell_markers = np.array(mesh.cellMarkers()) if mesh else None
        
        if resistivity_model is None:
            raise ValueError("No resistivity model available for petrophysics conversion")
        
        petro_input = {
            'resistivity_model': resistivity_model,
            'mesh': mesh,
            'cell_markers': cell_markers,
            'n_realizations': workflow_config.get('n_realizations', 100),
            'layer_params': workflow_config.get('layer_params', {}),
            'output_dir': f"{output_dir}/petrophysics"
        }
        
        return petro_agent.execute(petro_input)
    
    def _compile_final_results(self, workflow_results: Dict) -> Dict[str, Any]:
        """Compile final results from all workflow steps for report generation."""
        final_results = {
            'status': 'success'  # Required for report generation
        }
        step_results = workflow_results.get('step_results', {})
        
        # Merge statistics from all steps
        combined_statistics = {}
        
        # Extract key results from different steps
        if 'constrained_ert' in step_results:
            ert_results = step_results['constrained_ert']
            final_results.update({
                'resistivity_model': ert_results.get('resistivity_model'),
                'mesh': ert_results.get('mesh'),
                'cell_markers': ert_results.get('cell_markers'),
                'coverage': ert_results.get('coverage')
            })
            # Add ERT statistics
            if 'statistics' in ert_results:
                combined_statistics.update(ert_results['statistics'])
        
        elif 'ert_inversion' in step_results:
            ert_results = step_results['ert_inversion']
            final_results.update({
                'resistivity_model': ert_results.get('resistivity_model'),
                'mesh': ert_results.get('mesh'),
                'coverage': ert_results.get('coverage')
            })
            # Add ERT statistics
            if 'statistics' in ert_results:
                combined_statistics.update(ert_results['statistics'])
        
        if 'petrophysics_conversion' in step_results:
            petro_results = step_results['petrophysics_conversion']
            final_results.update({
                'water_content_mean': petro_results.get('water_content_mean'),
                'water_content_std': petro_results.get('water_content_std')
            })
            # Add petrophysics statistics
            if 'statistics' in petro_results:
                combined_statistics.update(petro_results['statistics'])
        
        # IMPORTANT: Seismic results need to be properly structured for visualization
        if 'seismic_inversion' in step_results:
            seismic_results = step_results['seismic_inversion']
            # Ensure seismic_results has coverage field (required for visualization)
            if 'coverage' not in seismic_results and 'mesh' in seismic_results:
                # Create a default coverage array if missing
                import numpy as np
                mesh = seismic_results['mesh']
                seismic_results['coverage'] = np.ones(mesh.cellCount())
            
            final_results['seismic_results'] = seismic_results
            # Add seismic statistics if available
            if 'statistics' in seismic_results:
                combined_statistics.update(seismic_results['statistics'])
        
        if 'interface_extraction' in step_results:
            interface_results = step_results['interface_extraction']
            final_results['interface_coords'] = interface_results.get('interface_coords')
            final_results['velocity_threshold'] = interface_results.get('velocity_threshold')
        
        # Add combined statistics
        if combined_statistics:
            final_results['statistics'] = combined_statistics
        
        return final_results
