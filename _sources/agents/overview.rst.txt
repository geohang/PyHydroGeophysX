System Overview
===============

The PyHydroGeophysX multi-agent system provides an intelligent framework for
automating complex geophysical workflows in subsurface hydrology. By leveraging
Large Language Models (LLMs), the system enables users to describe their
analysis goals in natural language and have the system automatically orchestrate
the necessary processing steps.

Who This Is For
---------------

* **Hydrogeophysics researchers** who know the science but may not know this
  codebase should start with :doc:`quick_start` and the Python API examples.
* **Graduate students and non-experts** running field data should start with
  the :doc:`webapp`, especially demo mode before using their own uploads.
* **Reviewers and collaborators** evaluating the project should start with
  :doc:`webapp`, then use :doc:`workflows` and :doc:`agent_reference` to inspect
  the supported workflow surface.

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

* **OpenAI GPT** (the GPT-5.6 family, and the GPT-4.x names)
* **Google Gemini** (Gemini 2.5)
* **Anthropic Claude** (Claude Haiku 4.5, Sonnet 5, Opus 5)
* Any **OpenAI-compatible** endpoint (DeepSeek, OpenRouter, a local server)

Each provider can be configured via API keys, and the system gracefully handles
cases where LLM features are not available.

For OpenAI and Anthropic the chat surfaces do not ask you to name a model. They
offer a three-step **request level** instead — level 1 for the simple majority of
requests, level 2 for coding, reasoning, and agent work, level 3 for what the
levels below could not finish — and resolve the level to a model per provider.
The ladder lives in :data:`PyHydroGeophysX.llm.providers.MODEL_TIERS`, so the
desktop studio and the web app always offer the same levels; see
:ref:`choosing-a-request-level` for the models and their prices.

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
