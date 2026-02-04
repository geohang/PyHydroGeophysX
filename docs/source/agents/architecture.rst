System Architecture
===================

This document provides a comprehensive overview of the PyHydroGeophysX multi-agent
system architecture, including agent relationships, data flow, and communication
protocols.

Agent Communication Protocol
----------------------------

Standard Input/Output Format
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All agents follow a standardized I/O pattern:

.. code-block:: python

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Standard agent execution method.

        Args:
            input_data: Dictionary with agent-specific inputs

        Returns:
            Dictionary containing:
                - status: 'success', 'failed', or 'needs_improvement'
                - [agent-specific outputs]
                - error: Error message (if status='failed')
                - interpretation: AI interpretation (if LLM available)
        """

Standard Output Keys
^^^^^^^^^^^^^^^^^^^^

All agents include these common output keys:

* ``status`` (str): 'success', 'failed', or 'needs_improvement'
* ``error`` (str): Error message if failed
* ``interpretation`` (str): AI interpretation (if LLM available)

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

The AgentCoordinator maintains workflow state:

.. code-block:: python

    {
        'status': 'initialized' | 'running' | 'completed' | 'failed',
        'current_step': 'agent_name',
        'completed_steps': ['agent1', 'agent2', ...],
        'data': {
            'agent1': {...},
            'agent2': {...}
        }
    }

Fusion Patterns
---------------

The DataFusionAgent supports several pre-defined fusion patterns:

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
