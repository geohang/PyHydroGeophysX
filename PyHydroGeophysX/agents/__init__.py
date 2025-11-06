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
from .ert_loader_agent import ERTLoaderAgent
from .ert_inversion_agent import ERTInversionAgent
from .water_content_agent import WaterContentAgent
from .report_agent import ReportAgent
from .seismic_agent import SeismicAgent
from .climate_data_agent import ClimateDataAgent

__all__ = [
    'AgentCoordinator',
    'ERTLoaderAgent',
    'ERTInversionAgent',
    'WaterContentAgent',
    'ReportAgent',
    'SeismicAgent',
    'ClimateDataAgent'
]
