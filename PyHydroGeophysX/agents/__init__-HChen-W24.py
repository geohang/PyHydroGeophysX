"""
Multi-Agent System for Automated Geophysical Workflows

This module provides an automatic cross-modal geophysics agent system for 
subsurface hydrology, supporting multiple LLM APIs (GPT, Gemini, Claude) to 
automate workflows that process geophysical data (ERT, seismic, and more) 
into hydrologic information.

The workflow example: "load geophysical data → process → invert → convert to 
hydrologic parameters → report" with optional cross-modal integration.

Each agent is specialized for a specific task and communicates through
a coordinator to execute the complete workflow.
"""

from .agent_coordinator import AgentCoordinator
from .context_input_agent import ContextInputAgent
from .workflow_orchestrator_agent import WorkflowOrchestratorAgent
from .ert_loader_agent import ERTLoaderAgent
from .ert_inversion_agent import ERTInversionAgent
from .inversion_evaluation_agent import InversionEvaluationAgent
from .water_content_agent import WaterContentAgent
from .report_agent import ReportAgent
from .seismic_agent import SeismicAgent
from .climate_data_agent import ClimateDataAgent
from .data_fusion_agent import DataFusionAgent
from .structure_constraint_agent import StructureConstraintAgent
from .petrophysics_agent import PetrophysicsAgent

__all__ = [
    'AgentCoordinator',
    'ContextInputAgent',
    'WorkflowOrchestratorAgent',
    'ERTLoaderAgent',
    'ERTInversionAgent',
    'InversionEvaluationAgent',
    'WaterContentAgent',
    'ReportAgent',
    'SeismicAgent',
    'ClimateDataAgent',
    'DataFusionAgent',
    'StructureConstraintAgent',
    'PetrophysicsAgent'
]
