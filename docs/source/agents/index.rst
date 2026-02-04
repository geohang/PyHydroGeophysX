Multi-Agent AI System
=====================

PyHydroGeophysX includes a powerful multi-agent system for automating geophysical
workflows using Large Language Models (LLMs). The system supports multiple providers
including OpenAI GPT, Google Gemini, and Anthropic Claude.

.. image:: /_static/agent_workflow.png
   :alt: Multi-Agent System Overview
   :align: center
   :width: 600px

|

Key Capabilities
----------------

* **Natural Language Interface**: Describe workflows in plain English
* **Automated Workflow Orchestration**: Intelligent agent coordination
* **Multi-Method Data Fusion**: Combine ERT, seismic, and TDEM data
* **Uncertainty Quantification**: Monte Carlo methods with layer-specific parameters
* **Quality Control**: Automatic inversion evaluation and parameter optimization
* **Comprehensive Reporting**: Generate publication-ready reports

Agent Categories
----------------

**Input/Configuration Agents**
    Parse natural language requests and load data from various instruments

**Processing/Inversion Agents**
    Perform ERT, seismic, and TDEM inversions with quality control

**Conversion/Analysis Agents**
    Convert geophysical properties to hydrological parameters

**Output/Reporting Agents**
    Generate comprehensive reports and visualizations

.. toctree::
   :maxdepth: 2
   :caption: Contents

   overview
   quick_start
   architecture
   agent_reference
   workflows
