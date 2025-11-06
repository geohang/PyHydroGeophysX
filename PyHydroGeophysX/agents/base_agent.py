"""
Base Agent Class for Multi-Agent System

Provides the foundation for all specialized agents in the workflow.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import os
import json


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the multi-agent system.
    
    Each agent is specialized for a specific task and can communicate
    with other agents through the coordinator.
    """
    
    def __init__(self, name: str, api_key: Optional[str] = None, model: str = "gpt-4", 
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
        
        # Set API key based on provider
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
