System Overview
===============

The PyHydroGeophysX multi-agent system provides an intelligent framework for
automating complex geophysical workflows in subsurface hydrology. By leveraging
Large Language Models (LLMs), the system enables users to describe their
analysis goals in natural language and have the system automatically orchestrate
the necessary processing steps.

Design Principles
-----------------

**Specialization**
    Each agent handles one specific task (data loading, inversion, conversion, etc.)
    with expert-level knowledge encoded in its system prompt.

**Coordination**
    The AgentCoordinator orchestrates multi-agent workflows, managing state and
    data flow between agents.

**Extensibility**
    New agents can be easily added by inheriting from BaseAgent and implementing
    the execute() method.

**LLM Integration**
    Natural language interfaces allow non-expert users to describe complex
    workflows without programming knowledge.

**Uncertainty Quantification**
    Monte Carlo methods are built into conversion agents to propagate parameter
    uncertainties through the analysis pipeline.

Supported LLM Providers
-----------------------

The system supports multiple LLM providers:

* **OpenAI GPT** (GPT-4, GPT-3.5-turbo)
* **Google Gemini** (Gemini Pro)
* **Anthropic Claude** (Claude 3)

Each provider can be configured via API keys, and the system gracefully handles
cases where LLM features are not available.

Agent Hierarchy
---------------

.. code-block:: text

    ┌─────────────────────────────────────────────────────────────────┐
    │                      AgentCoordinator                           │
    │  (Orchestrates multi-agent workflows, manages state)            │
    └──────────────────────┬──────────────────────────────────────────┘
                           │
             ┌─────────────┼─────────────┐
             │             │             │
        ┌────▼────┐   ┌────▼────┐   ┌───▼────┐
        │ Input   │   │ Process │   │ Output │
        │ Agents  │   │ Agents  │   │ Agents │
        └─────────┘   └─────────┘   └────────┘

Agent Categories
----------------

Input/Configuration Agents
^^^^^^^^^^^^^^^^^^^^^^^^^^

* ``ContextInputAgent`` - Parses natural language requests into structured configurations
* ``ERTLoaderAgent`` - Loads ERT field data from various commercial instruments
* ``SeismicAgent`` - Processes seismic refraction data
* ``ClimateDataAgent`` - Fetches climate data for temporal analysis

Processing/Inversion Agents
^^^^^^^^^^^^^^^^^^^^^^^^^^^

* ``ERTInversionAgent`` - Performs standard and time-lapse ERT inversions
* ``InversionEvaluationAgent`` - Evaluates quality and optimizes parameters
* ``TDEMAgent`` - Performs TDEM forward modeling and inversion
* ``DataFusionAgent`` - Coordinates multi-method data fusion
* ``StructureConstraintAgent`` - Applies seismic constraints to ERT inversion

Conversion/Analysis Agents
^^^^^^^^^^^^^^^^^^^^^^^^^^

* ``PetrophysicsAgent`` - Layer-specific resistivity to water content conversion
* ``WaterContentAgent`` - General petrophysical conversion

Output/Reporting Agents
^^^^^^^^^^^^^^^^^^^^^^^

* ``ReportAgent`` - Generates comprehensive reports with visualizations

Benefits
--------

**For Researchers**
    Automate repetitive processing tasks while maintaining full control over
    parameters and methods.

**For Practitioners**
    Access expert-level analysis without deep programming expertise through
    natural language interfaces.

**For Educators**
    Demonstrate complete geophysical workflows with clear, reproducible steps
    and comprehensive documentation.
