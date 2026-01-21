"""
Data Fusion Agent

Intelligent coordinator for multi-method geophysical workflows. This agent understands
which geophysical methods should work together and orchestrates complex data fusion
workflows like seismic-constrained ERT inversion.

The DataFusionAgent is designed to be extensible for future multi-method combinations.
"""

from typing import Dict, Any, Optional, List
import numpy as np
import os
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
    
    def execute_full_workflow(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute complete multi-method data fusion workflow internally.
        
        This method handles the full workflow by calling specialized agents internally:
        1. Load/prepare data (seismic and ERT)
        2. Run seismic inversion (via SeismicAgent)
        3. Extract velocity interface
        4. Run structure-constrained ERT (via StructureConstraintAgent)
        5. Optionally convert to water content (via PetrophysicsAgent)
        
        Args:
            input_data: Dictionary containing:
                - seismic_data or seismic_file: Seismic travel time data
                - ert_data or ert_file: ERT measurement data
                - velocity_threshold: Velocity threshold for interface (default: 1000 m/s)
                - fusion_pattern: Workflow pattern to use
                - workflow_config: Full configuration dictionary
                - agents: Dictionary of registered agents {'seismic': agent, 'structure': agent, etc.}
                - output_dir: Base output directory
                
        Returns:
            Dictionary containing complete workflow results
        """
        self._log_execution("Starting full data fusion workflow execution")
        
        try:
            # Import specialized agents if not provided
            from .seismic_agent import SeismicAgent
            from .structure_constraint_agent import StructureConstraintAgent
            
            # Extract configuration
            workflow_config = input_data.get('workflow_config', {})
            fusion_pattern = input_data.get('fusion_pattern', 
                                           workflow_config.get('fusion_pattern', 'structure_constraint'))
            output_dir = input_data.get('output_dir', 'results/data_fusion')
            os.makedirs(output_dir, exist_ok=True)
            
            # Get or create specialized agents
            agents = input_data.get('agents', {})
            seismic_agent = agents.get('seismic')
            if not seismic_agent:
                seismic_agent = SeismicAgent(
                    api_key=self.api_key,
                    model=self.model,
                    llm_provider=self.llm_provider
                )
            
            structure_agent = agents.get('structure')
            if not structure_agent:
                structure_agent = StructureConstraintAgent(
                    api_key=self.api_key,
                    model=self.model,
                    llm_provider=self.llm_provider
                )
            
            # Extract data
            seismic_data = input_data.get('seismic_data')
            ert_data = input_data.get('ert_data')
            
            # Handle file loading if needed
            if seismic_data is None and 'seismic_file' in workflow_config:
                self._log_execution("Loading seismic data from file")
                seismic_data = self._load_seismic_file(workflow_config['seismic_file'])
            
            if ert_data is None and 'ert_file' in workflow_config:
                self._log_execution("Loading ERT data from file")
                ert_data = self._load_ert_file(workflow_config)
            
            if seismic_data is None or ert_data is None:
                raise ValueError("Both seismic_data and ert_data are required")
            
            # Get parameters
            velocity_threshold = workflow_config.get('velocity_threshold', 1000)
            seismic_params = workflow_config.get('seismic_params', {})
            ert_params = workflow_config.get('ert_params', {})
            mesh_params = workflow_config.get('mesh_params', {})
            
            # STEP 1: Run seismic inversion (if not using pre-computed interface)
            interface_coords = input_data.get('interface_coords')
            
            if interface_coords is None:
                self._log_execution("=" * 70)
                self._log_execution("STEP 1: Seismic Inversion (via SeismicAgent)")
                self._log_execution("=" * 70)
                
                seismic_input = {
                    'seismic_data': seismic_data,
                    'velocity_threshold': velocity_threshold,
                    'inversion_params': seismic_params,
                    'output_dir': os.path.join(output_dir, 'seismic')
                }
                
                seismic_results = seismic_agent.execute(seismic_input)
                
                if seismic_results['status'] != 'success':
                    raise RuntimeError(f"Seismic inversion failed: {seismic_results.get('error')}")
                
                interface_coords = seismic_results['interface_coords']
                velocity_model = seismic_results['velocity_model']
                
                self._log_execution(f"✓ Seismic inversion complete")
                self._log_execution(f"  Interface extracted: {len(interface_coords[0])} points")
            else:
                self._log_execution("Using pre-computed interface coordinates")
                seismic_results = None
                velocity_model = None
            
            # STEP 2: Structure-constrained ERT inversion (via StructureConstraintAgent)
            self._log_execution("\n" + "=" * 70)
            self._log_execution("STEP 2: Structure-Constrained ERT (via StructureConstraintAgent)")
            self._log_execution("=" * 70)
            
            structure_input = {
                'ert_data': ert_data,
                'interface_coords': interface_coords,
                'velocity_threshold': velocity_threshold,
                'inversion_params': ert_params,
                'mesh_params': mesh_params,
                'output_dir': os.path.join(output_dir, 'structure_constrained'),
                'mesh_quality': workflow_config.get('mesh_quality', 31)
            }
            
            structure_results = structure_agent.execute(structure_input)
            
            if structure_results['status'] != 'success':
                raise RuntimeError(f"Structure-constrained inversion failed: {structure_results.get('error')}")
            
            self._log_execution(f"✓ Structure-constrained ERT complete")
            self._log_execution(f"  Resistivity range: {structure_results['statistics']['resistivity_range']}")
            
            # STEP 3: Optional water content conversion
            petro_results = None
            if fusion_pattern == 'full_integration' and 'petrophysics' in agents:
                self._log_execution("\n" + "=" * 70)
                self._log_execution("STEP 3: Water Content Conversion (via PetrophysicsAgent)")
                self._log_execution("=" * 70)
                
                petro_agent = agents['petrophysics']
                petro_input = {
                    'resistivity_model': structure_results['resistivity_model'],
                    'mesh': structure_results['mesh'],
                    'cell_markers': structure_results['cell_markers'],
                    'layer_params': workflow_config.get('layer_params', {}),
                    'n_realizations': workflow_config.get('n_realizations', 100),
                    'output_dir': os.path.join(output_dir, 'petrophysics')
                }
                
                petro_results = petro_agent.execute(petro_input)
                
                if petro_results['status'] == 'success':
                    self._log_execution(f"✓ Water content conversion complete")
                else:
                    self._log_execution(f"⚠ Water content conversion failed", level='WARN')
            
            # Get LLM interpretation of complete workflow
            interpretation = None
            if self.api_key:
                interpretation = self._interpret_fusion_workflow(
                    seismic_results,
                    structure_results,
                    petro_results,
                    fusion_pattern
                )
            
            # Compile results
            self.results = {
                'status': 'success',
                'fusion_pattern': fusion_pattern,
                'seismic_results': seismic_results,
                'structure_results': structure_results,
                'petro_results': petro_results,
                'interpretation': interpretation,
                'output_dir': output_dir
            }
            
            self._log_execution("\n" + "=" * 70)
            self._log_execution("✓ Complete data fusion workflow finished successfully")
            self._log_execution("=" * 70)
            
            return self.results
            
        except Exception as e:
            self._log_execution(f"Error during full workflow execution: {str(e)}", level='ERROR')
            self.results = {
                'status': 'failed',
                'error': str(e)
            }
            raise
    
    def _load_seismic_file(self, file_path: str):
        """Load seismic data from file."""
        try:
            import pygimli.physics.traveltime as tt
            self._log_execution(f"Loading seismic data from: {file_path}")
            return tt.load(file_path)
        except Exception as e:
            self._log_execution(f"Failed to load seismic file: {e}", level='ERROR')
            raise
    
    def _load_ert_file(self, config: Dict) -> Any:
        """Load ERT data from file using configuration."""
        try:
            from pygimli.physics import ert
            file_path = config.get('ert_file')
            instrument = config.get('instrument', 'BERT')
            
            self._log_execution(f"Loading ERT data from: {file_path}")
            self._log_execution(f"  Instrument: {instrument}")
            
            return ert.load(file_path)
        except Exception as e:
            self._log_execution(f"Failed to load ERT file: {e}", level='ERROR')
            raise
    
    def _interpret_fusion_workflow(self, seismic_results, structure_results, 
                                   petro_results, pattern: str) -> str:
        """
        Generate LLM interpretation of complete fusion workflow.
        
        Args:
            seismic_results: Results from seismic inversion
            structure_results: Results from structure-constrained ERT
            petro_results: Results from petrophysics (or None)
            pattern: Fusion pattern used
            
        Returns:
            Interpretation string
        """
        try:
            workflow_summary = f"Fusion Pattern: {pattern}\n\n"
            
            if seismic_results:
                velocity_range = [np.min(seismic_results['velocity_model']),
                                 np.max(seismic_results['velocity_model'])]
                workflow_summary += f"Seismic Results:\n"
                workflow_summary += f"  - Velocity range: {velocity_range[0]:.0f} - {velocity_range[1]:.0f} m/s\n"
                workflow_summary += f"  - Interface points: {len(seismic_results['interface_coords'][0])}\n\n"
            
            if structure_results:
                stats = structure_results['statistics']
                workflow_summary += f"Structure-Constrained ERT Results:\n"
                workflow_summary += f"  - Resistivity: {stats['resistivity_range'][0]:.1f} - {stats['resistivity_range'][1]:.1f} Ωm\n"
                workflow_summary += f"  - Layers: {stats['num_layers']}\n\n"
            
            if petro_results:
                petro_stats = petro_results.get('statistics', {})
                workflow_summary += f"Water Content Results:\n"
                workflow_summary += f"  - Range: {petro_stats.get('wc_range', 'N/A')}\n"
                workflow_summary += f"  - Mean: {petro_stats.get('mean_water_content', 'N/A')}\n\n"
            
            prompt = f"""Provide a comprehensive interpretation of this multi-method geophysical data fusion workflow:

{workflow_summary}

Explain in 3-4 sentences:
1. How the methods complemented each other
2. Key findings from the integrated analysis
3. Confidence in the results due to multi-method constraints"""
            
            interpretation = self.query_llm(prompt, self.system_message,
                                          temperature=0.5, max_tokens=300)
            return interpretation
        except:
            return "Multi-method data fusion workflow completed successfully"
    
    def _log_execution(self, message: str, level: str = 'INFO'):
        """Log execution message."""
        print(f"[{self.name}] [{level}] {message}")
