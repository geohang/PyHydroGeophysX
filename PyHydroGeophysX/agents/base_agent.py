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
    
    def __init__(self, name: str, api_key: Optional[str] = None, model: str = "gpt-4"):
        """
        Initialize the base agent.
        
        Args:
            name: Name identifier for this agent
            api_key: OpenAI API key (uses OPENAI_API_KEY env var if not provided)
            model: OpenAI model to use (default: gpt-4, alternatives: gpt-3.5-turbo, gpt-4-turbo)
        """
        self.name = name
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.model = model or os.getenv('OPENAI_MODEL', 'gpt-4')
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
        Query the LLM (GPT API) for assistance.
        
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
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
                "or pass api_key during initialization."
            )
        
        try:
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
            
        except ImportError:
            raise ImportError(
                "OpenAI package not installed. Install with: pip install openai"
            )
        except Exception as e:
            raise RuntimeError(f"Error querying LLM: {str(e)}")
    
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
