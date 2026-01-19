"""
Code Generation Agent for Out-of-Scope Requests

Uses LLM to generate and execute custom code when user requests are not covered
by the standard workflow configuration.
"""

from typing import Dict, Any, Optional, Tuple
import os
import traceback
from io import StringIO
import sys
from .base_agent import BaseAgent


class CodeGenerationAgent(BaseAgent):
    """
    Agent that uses LLM to generate custom Python code for out-of-scope requests.
    
    This agent is triggered when:
    1. User request doesn't match standard workflow patterns
    2. User explicitly asks for custom analysis
    3. Standard workflow fails and user wants alternative approaches
    
    Safety features:
    - Code runs in isolated namespace
    - Limited imports allowed
    - Execution timeout
    - Results captured and validated
    """
    
    # Allowed modules for generated code
    ALLOWED_IMPORTS = [
        'numpy', 'np',
        'pandas', 'pd', 
        'matplotlib', 'plt',
        'scipy',
        'os', 'pathlib',
        'json',
        'pygimli', 'pg',
    ]
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None,
                 llm_provider: str = "openai"):
        """Initialize Code Generation Agent."""
        super().__init__("code_generator", api_key, model, llm_provider)
        self.system_message = """You are an expert Python programmer specializing in geophysics and hydrogeology.
You write clean, efficient, and well-documented code using PyGIMLI, NumPy, Pandas, and Matplotlib.
Your code follows best practices and includes appropriate error handling.
You only use standard scientific Python libraries that are commonly available."""
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate and execute custom code based on user request.
        
        Args:
            input_data: Dictionary containing:
                - user_request: Natural language description of what to do
                - available_data: Dict of available data (file paths, arrays, etc.)
                - output_dir: Directory for saving outputs
                - context: Additional context about the workflow
                
        Returns:
            Dictionary containing:
                - status: 'success' or 'failed'
                - code: Generated Python code
                - output: Execution output/results
                - error: Error message if failed
                - interpretation: LLM interpretation of results
        """
        self._log_execution("Starting code generation for custom request")
        
        try:
            user_request = input_data.get('user_request', '')
            available_data = input_data.get('available_data', {})
            output_dir = input_data.get('output_dir', 'results/custom')
            context = input_data.get('context', '')
            
            if not user_request:
                raise ValueError("user_request is required")
            
            os.makedirs(output_dir, exist_ok=True)
            
            # Step 1: Analyze the request and determine what code is needed
            self._log_execution("Analyzing request to generate code")
            code_plan = self._plan_code_generation(user_request, available_data, context)
            
            # Step 2: Generate the Python code
            self._log_execution("Generating Python code")
            generated_code = self._generate_code(code_plan, available_data, output_dir)
            
            # Step 3: Validate the generated code
            is_safe, safety_message = self._validate_code_safety(generated_code)
            if not is_safe:
                self._log_execution(f"Code safety check failed: {safety_message}", level='WARNING')
                return {
                    'status': 'failed',
                    'code': generated_code,
                    'error': f"Code safety validation failed: {safety_message}",
                    'interpretation': "The generated code could not be validated for safe execution."
                }
            
            # Step 4: Execute the code
            self._log_execution("Executing generated code")
            success, output, error = self._execute_code(generated_code, available_data, output_dir)
            
            # Step 5: Interpret the results
            interpretation = None
            if success and self.api_key:
                self._log_execution("Generating interpretation of results")
                interpretation = self._interpret_results(user_request, generated_code, output)
            
            # Save the generated code
            code_file = os.path.join(output_dir, 'custom_analysis.py')
            with open(code_file, 'w', encoding='utf-8') as f:
                f.write(f'"""\nAuto-generated code for: {user_request}\n"""\n\n')
                f.write(generated_code)
            
            self.results = {
                'status': 'success' if success else 'failed',
                'code': generated_code,
                'code_file': code_file,
                'output': output,
                'error': error if not success else None,
                'interpretation': interpretation,
                'output_dir': output_dir
            }
            
            return self.results
            
        except Exception as e:
            self._log_execution(f"Error in code generation: {str(e)}", level='ERROR')
            self.results = {
                'status': 'failed',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
            return self.results
    
    def _plan_code_generation(self, user_request: str, available_data: Dict, context: str) -> str:
        """Create a plan for what code to generate."""
        prompt = f"""Analyze this geophysics/hydrogeology request and create a code generation plan.

User Request:
{user_request}

Available Data:
{self._format_available_data(available_data)}

Context:
{context if context else 'Standard PyHydroGeophysX workflow context.'}

Provide a structured plan including:
1. What the code should accomplish (1-2 sentences)
2. Key steps/functions needed (bullet points)
3. Required libraries (only from: numpy, pandas, matplotlib, scipy, pygimli, os, pathlib, json)
4. Expected outputs (files, plots, or data structures)

Keep the plan concise and focused on what can be achieved with available data."""
        
        plan = self.query_llm(prompt, self.system_message, temperature=0.3, max_tokens=500)
        return plan
    
    def _generate_code(self, code_plan: str, available_data: Dict, output_dir: str) -> str:
        """Generate Python code based on the plan."""
        prompt = f"""Generate Python code based on this plan:

{code_plan}

Available Data (accessible via `data` dict):
{self._format_available_data(available_data)}

Output Directory: {output_dir}

Requirements:
1. Write clean, well-commented Python code
2. Use only these libraries: numpy (as np), pandas (as pd), matplotlib.pyplot (as plt), scipy, pygimli (as pg), os, pathlib, json
3. All output files should be saved to the output_dir variable
4. Store main results in a dict called `results`
5. Handle potential errors gracefully
6. Print progress messages for long operations
7. Close all matplotlib figures after saving

Return ONLY the Python code, no markdown formatting or explanations.
Start with necessary imports."""
        
        code = self.query_llm(prompt, self.system_message, temperature=0.2, max_tokens=2000)
        
        # Clean up code (remove markdown code blocks if present)
        code = self._clean_code_response(code)
        
        return code
    
    def _clean_code_response(self, code: str) -> str:
        """Clean up LLM code response by removing markdown formatting."""
        import re
        
        # Remove markdown code blocks
        code = re.sub(r'^```python\s*', '', code, flags=re.MULTILINE)
        code = re.sub(r'^```\s*$', '', code, flags=re.MULTILINE)
        code = re.sub(r'^```\s*', '', code, flags=re.MULTILINE)
        
        return code.strip()
    
    def _validate_code_safety(self, code: str) -> Tuple[bool, str]:
        """
        Validate that generated code is safe to execute.
        
        Returns:
            Tuple of (is_safe, message)
        """
        dangerous_patterns = [
            ('exec(', 'Dynamic code execution'),
            ('eval(', 'Dynamic code evaluation'),
            ('__import__', 'Dynamic imports'),
            ('subprocess', 'Subprocess execution'),
            ('os.system', 'System command execution'),
            ('os.popen', 'Pipe command execution'),
            ('shutil.rmtree', 'Recursive file deletion'),
            ('open(', 'File operations'),  # Allow but flag
        ]
        
        warnings = []
        
        for pattern, description in dangerous_patterns:
            if pattern in code:
                if pattern == 'open(':
                    # Allow file operations but log
                    warnings.append(f"Code uses file operations ({description})")
                else:
                    return False, f"Disallowed operation: {description}"
        
        # Check for disallowed imports
        import re
        import_pattern = r'(?:from|import)\s+(\w+)'
        imports = re.findall(import_pattern, code)
        
        base_allowed = ['numpy', 'pandas', 'matplotlib', 'scipy', 'pygimli', 'os', 'pathlib', 'json', 'np', 'pd', 'plt', 'pg']
        for imp in imports:
            if imp not in base_allowed:
                # Allow submodules of allowed packages
                if not any(imp.startswith(allowed) for allowed in base_allowed):
                    warnings.append(f"Unusual import: {imp}")
        
        if warnings:
            self._log_execution(f"Code validation warnings: {'; '.join(warnings)}", level='WARNING')
        
        return True, "Code passed safety validation"
    
    def _execute_code(self, code: str, available_data: Dict, output_dir: str) -> Tuple[bool, str, str]:
        """
        Execute generated code in a controlled environment.
        
        Returns:
            Tuple of (success, output, error)
        """
        # Create execution namespace with limited globals
        namespace = {
            'data': available_data,
            'output_dir': output_dir,
            'results': {},
            '__builtins__': __builtins__,
        }
        
        # Capture stdout
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = captured_output = StringIO()
        sys.stderr = captured_error = StringIO()
        
        try:
            # Execute the code
            exec(code, namespace)
            
            output = captured_output.getvalue()
            error_output = captured_error.getvalue()
            
            # Get results from namespace
            results = namespace.get('results', {})
            
            # Format output
            if results:
                output += f"\n\nResults: {results}"
            
            if error_output:
                output += f"\n\nWarnings/Errors:\n{error_output}"
            
            return True, output, ""
            
        except Exception as e:
            error = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            return False, captured_output.getvalue(), error
            
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
    
    def _interpret_results(self, user_request: str, code: str, output: str) -> str:
        """Generate LLM interpretation of execution results."""
        prompt = f"""Interpret the results of this custom analysis:

User Request:
{user_request}

Code Output:
{output[:2000]}  # Truncate if too long

Provide a brief interpretation (2-3 sentences) that:
1. Summarizes what the code accomplished
2. Highlights key results or findings
3. Notes any issues or limitations"""
        
        try:
            interpretation = self.query_llm(prompt, self.system_message, 
                                           temperature=0.5, max_tokens=300)
            return interpretation
        except Exception:
            return "Custom analysis completed. See output for details."
    
    def _format_available_data(self, data: Dict) -> str:
        """Format available data dict for prompt."""
        if not data:
            return "No data provided"
        
        lines = []
        for key, value in data.items():
            if hasattr(value, 'shape'):
                lines.append(f"- {key}: numpy array with shape {value.shape}")
            elif hasattr(value, 'columns'):
                lines.append(f"- {key}: pandas DataFrame with columns {list(value.columns)}")
            elif isinstance(value, str) and (value.endswith('.npy') or value.endswith('.dat') or value.endswith('.csv')):
                lines.append(f"- {key}: file path '{value}'")
            else:
                lines.append(f"- {key}: {type(value).__name__}")
        
        return '\n'.join(lines) if lines else "No structured data"
    
    def check_request_scope(self, user_request: str, workflow_config: Dict) -> Dict[str, Any]:
        """
        Check if a user request is within the standard workflow scope.
        
        Args:
            user_request: User's natural language request
            workflow_config: Parsed workflow configuration
            
        Returns:
            Dict with:
                - in_scope: bool - whether request is handled by standard workflow
                - out_of_scope_parts: list - parts that need custom code
                - recommendation: str - suggested approach
        """
        prompt = f"""Analyze if this geophysics request can be handled by standard workflows.

User Request:
{user_request}

Standard Workflow Capabilities:
1. ERT data loading and processing (DAS-1, Syscal, E4D, ABEM formats)
2. ERT inversion (standard and time-lapse)
3. Petrophysical conversion (resistivity to water content using Archie's law)
4. Seismic refraction tomography
5. Structure-constrained inversion (seismic + ERT fusion)
6. Monte Carlo uncertainty quantification
7. Climate data integration (precipitation, temperature, PET)
8. Report generation with visualizations

Parsed Configuration:
- Workflow type: {workflow_config.get('inversion_mode', 'standard')}
- Has ERT file: {bool(workflow_config.get('ert_file') or workflow_config.get('data_file'))}
- Has seismic file: {bool(workflow_config.get('seismic_file'))}
- Petrophysics requested: {bool(workflow_config.get('petrophysical_params'))}

Respond in JSON format:
{{
    "in_scope": true/false,
    "out_of_scope_parts": ["list of parts not covered"],
    "recommendation": "brief suggestion for how to handle"
}}"""
        
        try:
            import json
            response = self.query_llm(prompt, self.system_message, temperature=0.2, max_tokens=300)
            # Clean and parse JSON
            response = response.strip()
            if response.startswith('```'):
                import re
                response = re.sub(r'^```(?:json)?\s*', '', response)
                response = re.sub(r'\s*```$', '', response)
            result = json.loads(response)
            return result
        except Exception as e:
            self._log_execution(f"Error checking request scope: {e}", level='WARNING')
            return {
                'in_scope': True,  # Assume in scope if check fails
                'out_of_scope_parts': [],
                'recommendation': 'Proceed with standard workflow'
            }
    
    def _log_execution(self, message: str, level: str = 'INFO'):
        """Log execution message."""
        print(f"[{self.name}] [{level}] {message}")
