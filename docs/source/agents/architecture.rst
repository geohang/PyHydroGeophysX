System Architecture
===================

This document provides a comprehensive overview of the PyHydroGeophysX multi-agent
system architecture, including agent relationships, data flow, and communication
protocols.

This page is for contributors. If you just want to use the system, start with
:doc:`quick_start` or the :doc:`webapp`.

Agent Communication Protocol
----------------------------

Standard Input/Output Format
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All agents follow a standardized I/O pattern:

.. code-block:: python

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        Standard agent execution method.

        Args:
            input_data: Dictionary with agent-specific inputs

        Returns:
            AgentResult containing user-facing status, summary, data,
            warnings, optional AI interpretation, timing, and fix hints.
        """

Standard Output Keys
^^^^^^^^^^^^^^^^^^^^

All agents include these common output keys:

* ``status`` (str): ``success``, ``failed``, or ``needs_review``
* ``summary`` (str): one-sentence human-readable summary, always populated
* ``data`` (dict): numerical results, file paths, models, or other structured outputs
* ``warnings`` (list[str]): non-fatal issues the user should review
* ``next_suggested_action`` (str, optional): a practical next step
* ``llm_interpretation`` (str, optional): AI-generated text that must be labeled at render time
* ``elapsed_seconds`` (float): wall-clock execution time for the agent
* ``cost_estimate_usd`` (float, optional): approximate LLM cost when known
* ``error`` (str, optional): error message when failed
* ``error_fix_hint`` (str, optional): specific guidance for how to fix the error

``AgentResult`` remains dict-like for older calling code, so expressions such as
``result["status"]`` and ``result.get("summary")`` still work. Legacy dictionary
returns are wrapped by ``AgentCoordinator`` with a ``DeprecationWarning``.

LLM Cost Ledger
^^^^^^^^^^^^^^^

Agents that call an LLM record ``provider``, ``model``, ``prompt_tokens``,
``completion_tokens``, and ``cost_estimate_usd`` in ``llm_usage_ledger``. The
coordinator and unified workflow expose ``total_llm_cost_estimate_usd`` as an
approximate value. Pricing rates live in ``PyHydroGeophysX/agents/_pricing.py``
and should be checked before public cost claims.

Data Flow Patterns
------------------

Natural Language to Structured Workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

    User Natural Language Request
            │
            ▼
    ContextInputAgent (LLM Processing)
            │
            ▼
    Structured Configuration (JSON)
            │
            ▼
    DataFusionAgent (Create Execution Plan)
            │
            ▼
    Sequential Agent Execution

Structure-Constrained ERT Data Path
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

    Seismic Data          ERT Data
        │                     │
        ▼                     ▼
    SeismicAgent      ERTLoaderAgent
        │                     │
        ▼                     │
    velocity_model        ert_data
        │                     │
        └───────┬─────────────┘
                │
                ▼
    StructureConstraintAgent
        │
        ├─ Extract interfaces
        ├─ Create constrained mesh
        └─ Constrained ERT inversion
        │
        ▼
    PetrophysicsAgent
        │
        └─ Layer-specific conversion
        │
        ▼
    water_content ± uncertainty

Inversion Quality Optimization Loop
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

    Initial ERT Inversion (lambda = 20)
            │
            ▼
    InversionEvaluationAgent
        │
        ├─ Calculate chi-squared
        ├─ Evaluate smoothness
        ├─ Check physical plausibility
        └─ Compute quality score
        │
        ▼
    Quality Score < 70?
        │
    ┌───┴───┐
    │ Yes   │ No
    │       │
    ▼       ▼
    Adjust     Accept
    Parameters Results
        │
        ▼
    Re-run Inversion
        │
        └────────────────────┐
                             │
                             ▼
                    (repeat until quality >= 70
                     or max_attempts reached)

Monte Carlo Uncertainty Propagation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

    Layer-Specific Parameters
        │
        ├─ Layer 1: porosity ± σ, n ± σ, m ± σ
        ├─ Layer 2: porosity ± σ, n ± σ, m ± σ
        └─ Layer 3: porosity ± σ, n ± σ, m ± σ
        │
        ▼
    PetrophysicsAgent
        │
        For each cell:
          For i = 1 to N_realizations:
            Sample parameters from distributions
            Calculate WC(i) = f(resistivity, params_i)
        │
        ▼
    Output:
        - water_content_mean
        - water_content_std
        - 95% confidence intervals

Workflow State Management
-------------------------

The ``AgentCoordinator`` maintains workflow state and supports **checkpoint /
resume** so a run that fails mid-way can continue from the last completed step:

.. code-block:: python

    from PyHydroGeophysX.agents import AgentCoordinator

    coordinator = AgentCoordinator(api_key=api_key, output_dir='./results')

    # First run — saves a pickle checkpoint after each step
    results = coordinator.execute_workflow(config)

    # If a step fails, restart with resume=True to skip completed steps
    results = coordinator.execute_workflow(config, resume=True)

Checkpoint files are stored as ``<output_dir>/checkpoints/<step>.pkl`` with a
``<step>.json`` sidecar for quick inspection.  The state dict structure is:

.. code-block:: python

    {
        'status': 'initialized' | 'running' | 'completed' | 'failed',
        'current_step': 'agent_name',
        'completed_steps': ['fetch_climate', 'load_ert', 'invert_ert', ...],
        'data': {
            'ert_data': ...,
            'inversion_results': ...,
        }
    }

LLM Usage Ledger and Cost Tracking
-----------------------------------

Every LLM call appends an entry to ``agent.llm_usage_ledger``:

.. code-block:: python

    {
        'agent': 'ert_inversion',
        'provider': 'openai',
        'model': 'gpt-4o-mini',
        'prompt_tokens': 320,
        'completion_tokens': 85,
        'total_tokens': 405,
        'cost_estimate_usd': 0.000121,
        'timestamp': 1714500000.0,
    }

The ``AgentCoordinator`` aggregates entries from all agents.  After a run,
call ``get_workflow_summary()`` to see totals:

.. code-block:: python

    summary = coordinator.get_workflow_summary()
    # {
    #   'status': 'completed',
    #   'completed_steps': [...],
    #   'total_llm_cost_estimate_usd': 0.0034,
    #   'total_llm_tokens': 8200,
    #   'llm_calls': 12,
    # }

Pricing rates are defined in ``PyHydroGeophysX/agents/_pricing.py`` and should
be checked before reporting exact figures in publications.

LLM Retry with Exponential Back-off
------------------------------------

All three LLM providers (OpenAI, Gemini, Claude) automatically retry on
transient rate-limit errors via ``BaseAgent._retry_llm_call``:

.. code-block:: text

    Attempt 1 → rate-limit? → wait 1 s → Attempt 2
    Attempt 2 → rate-limit? → wait 2 s → Attempt 3
    Attempt 3 → rate-limit? → raise (propagate to caller)
    Non-transient error → raise immediately (no retry)

Errors matching ``rate limit``, ``resource exhausted``, ``quota``,
``too many requests``, or ``429`` trigger a retry.  All other exceptions
propagate immediately.

Agent System Prompt Augmentation
---------------------------------

On the first ``query_llm()`` call each agent lazily loads its ``.agent.md`` file
from ``.github/agents/<name>.agent.md`` (relative to the repository root) and
appends the Markdown body to ``self.system_message``.  This keeps agent personas
editable as plain Markdown without modifying Python source.

``ContextInputAgent`` initialises ``self.system_message`` explicitly in
``__init__`` so the augmentation hook fires correctly on the first call.

Fusion Patterns
---------------

The ``DataFusionAgent`` supports several pre-defined fusion patterns:

Pattern 1: structure_constraint
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* **Methods**: Seismic → ERT
* **Description**: Use seismic velocity interfaces to constrain ERT inversion
* **Workflow**: ``seismic_inversion`` → ``interface_extraction`` → ``constrained_ert``
* **Benefits**: Improved layer boundary resolution, reduced artifacts

Pattern 2: petrophysics_integration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* **Methods**: ERT → Petrophysics
* **Description**: Convert resistivity to hydrological properties
* **Workflow**: ``ert_inversion`` → ``petrophysics_conversion``
* **Benefits**: Direct hydrological interpretation

Pattern 3: full_integration
^^^^^^^^^^^^^^^^^^^^^^^^^^^

* **Methods**: Seismic → ERT → Petrophysics
* **Description**: Complete geological-to-hydrological workflow
* **Workflow**: ``seismic_inversion`` → ``interface_extraction`` → ``constrained_ert`` → ``petrophysics_conversion``
* **Benefits**: Comprehensive subsurface characterization with constraints

Extension Points
----------------

Adding New Agents
^^^^^^^^^^^^^^^^^

To add a new agent to the system:

1. **Inherit from BaseAgent**:

.. code-block:: python

    from PyHydroGeophysX.agents.base_agent import BaseAgent

    class NewMethodAgent(BaseAgent):
        def __init__(self, api_key, model='gpt-4', llm_provider='openai'):
            super().__init__("new_method", api_key, model, llm_provider)
            self.system_message = "Your expert role description"

2. **Implement execute() method**:

.. code-block:: python

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        # Your processing logic
        return {
            'status': 'success',
            'output_key': output_value
        }

3. **Register in __init__.py**:

.. code-block:: python

    from .new_method_agent import NewMethodAgent
    __all__ = [..., 'NewMethodAgent']

4. **Update DataFusionAgent patterns** (if multi-method):

.. code-block:: python

    FUSION_PATTERNS = {
        'new_pattern': {
            'methods': ['method1', 'new_method'],
            'workflow': ['step1', 'step2'],
            ...
        }
    }

Error Handling
--------------

The agent system implements robust error handling:

* **Input Validation**: All agents validate inputs before processing
* **Graceful Degradation**: LLM features are optional; system works without them
* **Retry Logic**: Failed operations can be retried with adjusted parameters
* **Comprehensive Logging**: All agent actions are logged for debugging

Performance Considerations
--------------------------

* **Parallel Execution**: Independent agents can run in parallel
* **Caching**: Intermediate results are cached to avoid recomputation
* **Memory Management**: Large arrays are processed in chunks when necessary
* **GPU Acceleration**: TDEM forward modeling supports GPU via SimPEG
