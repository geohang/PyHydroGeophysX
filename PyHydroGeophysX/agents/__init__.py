"""
Multi-Agent System for Automated Geophysical Workflows

This module provides a GPT API-based multi-agent system for automating
the workflow: "load ERT → invert → convert to water content → report"
with optional seismic data integration.

Each agent is specialized for a specific task and communicates through
a coordinator to execute the complete workflow.
"""

from .agent_coordinator import AgentCoordinator
from .ert_loader_agent import ERTLoaderAgent
from .ert_inversion_agent import ERTInversionAgent
from .water_content_agent import WaterContentAgent
from .report_agent import ReportAgent
from .seismic_agent import SeismicAgent

__all__ = [
    'AgentCoordinator',
    'ERTLoaderAgent',
    'ERTInversionAgent',
    'WaterContentAgent',
    'ReportAgent',
    'SeismicAgent'
]
