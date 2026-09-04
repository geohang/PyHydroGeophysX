Agent Workflows
===============

PyHydroGeophysX includes an agent-based system for natural-language orchestration
of hydrogeophysical tasks.  These step-by-step tutorials walk through the most
common use cases.

.. contents:: On this page
   :local:
   :depth: 2

Quick Reference
---------------

+-------------------------------------------+-----------------------------------------------------+
| Task                                      | Where to go                                         |
+===========================================+=====================================================+
| Web app (no install)                      | `pyhydrogeophysx.streamlit.app`_                    |
+-------------------------------------------+-----------------------------------------------------+
| 3D Mesh Builder GUI                       | :ref:`tut-mesh3d`                                   |
+-------------------------------------------+-----------------------------------------------------+
| First ERT workflow in code                | :ref:`tut-basic-ert`                                |
+-------------------------------------------+-----------------------------------------------------+
| Recover a failed/crashed workflow         | :ref:`tut-resume`                                   |
+-------------------------------------------+-----------------------------------------------------+
| Track LLM cost                            | :ref:`tut-cost`                                     |
+-------------------------------------------+-----------------------------------------------------+
| Time-lapse monitoring                     | :doc:`/agents/workflows`                            |
+-------------------------------------------+-----------------------------------------------------+
| Architecture deep-dive                    | :doc:`/agents/architecture`                         |
+-------------------------------------------+-----------------------------------------------------+
| Agent API reference                       | :doc:`/agents/agent_reference`                      |
+-------------------------------------------+-----------------------------------------------------+

.. _pyhydrogeophysx.streamlit.app: https://pyhydrogeophysx.streamlit.app/

.. _tut-mesh3d:

Tutorial 1 — Interactive 3D Mesh Builder
-----------------------------------------

The 3D Mesh Builder is a standalone Streamlit app.  No API key is required.

**Launch**

.. code-block:: bash

    # via the package launcher (recommended)
    python -m PyHydroGeophysX.gui_mesh3d

    # or directly
    streamlit run examples/app_mesh3d.py

**Step-by-step: surface grid mesh**

1. Open the sidebar and choose **Surface Grid** as the Electrode Array Type.
2. Set **Grid Nx = 10**, **Grid Ny = 5**, **Spacing = 5 m**.
3. Select **Linear Tilt** topography to simulate gently sloping terrain.
4. Click the **Electrode View** tab — a 3D scatter plot confirms electrode positions.
5. Increase **Max Cell Size** to 5 m for a quick first mesh (lower values = finer).
6. Click the **Generate Mesh** tab, then press **Generate 3D Mesh**.
   Cell count, node count, and a quality histogram appear below.
7. Click the **Export** tab and download the mesh as ``.bms`` (PyGIMLi) or ``.vtk``
   (ParaView / any VTK reader).

**Step-by-step: crosshole mesh**

1. Choose **Crosshole** array type.
2. Set borehole depth and per-electrode spacing.
3. Set **Max Cell Size = 2 m** for crosshole resolution.
4. Confirm both boreholes in the **Electrode View** tab.
5. Generate and export as ``.bms`` for use with PyGIMLi crosshole inversion.

**Requirements:** ``pygimli``, ``gmsh`` on PATH, ``plotly``, ``streamlit``.
See :ref:`agents/troubleshooting:3D Mesh Generation Fails (gmsh not found)` if
GMSH is missing.

.. _tut-basic-ert:

Tutorial 2 — First ERT Workflow in Code
-----------------------------------------

This tutorial runs a complete ERT inversion and water-content conversion without
the web app.

**Prerequisites**

.. code-block:: bash

    conda activate pg        # or your PyHydroGeophysX environment
    export OPENAI_API_KEY="sk-..."

**Code walkthrough**

.. code-block:: python

    import os
    from PyHydroGeophysX.agents import (
        AgentCoordinator,
        ContextInputAgent,
        ERTLoaderAgent,
        ERTInversionAgent,
        InversionEvaluationAgent,
        WaterContentAgent,
        ReportAgent,
    )

    api_key = os.environ['OPENAI_API_KEY']

    # 1. Create coordinator with a results directory
    coordinator = AgentCoordinator(api_key=api_key, output_dir='./results')

    # 2. Register agents
    coordinator.register_agent('context',       ContextInputAgent(api_key))
    coordinator.register_agent('ert_loader',    ERTLoaderAgent(api_key))
    coordinator.register_agent('ert_inversion', ERTInversionAgent(api_key))
    coordinator.register_agent('evaluation',    InversionEvaluationAgent(api_key))
    coordinator.register_agent('water_content', WaterContentAgent(api_key))
    coordinator.register_agent('report',        ReportAgent(api_key))

    # 3. Define workflow configuration
    config = {
        'data_file': 'examples/data/sample_ert.ohm',
        'instrument': 'E4D',
        'inversion_params': {'lambda': 20, 'max_iter': 10},
        'petrophysical_params': {
            'layer1': {'porosity': 0.35, 'n': 2.0, 'm': 1.5},
        },
    }

    # 4. (Optional) Dry-run — validates files and estimates cost without running
    preview = coordinator.execute_workflow(config, dry_run=True)
    print("Execution plan:   ", preview['data']['execution_plan'])
    print("Dep. warnings:    ", preview['data']['validation_warnings'])
    print(f"Est. cost:        ${preview['cost_estimate_usd']:.4f}")

    # 5. Run the full workflow
    results = coordinator.execute_workflow(config)

    # 6. Check LLM cost summary
    summary = coordinator.get_workflow_summary()
    print(f"Total cost: ${summary['total_llm_cost_estimate_usd']:.4f}")
    print(f"Tokens:     {summary['total_llm_tokens']}")
    print(f"LLM calls:  {summary['llm_calls']}")

    # 7. Persist outputs (numpy → .npy, meshes → .bms, DataFrames → .csv)
    coordinator.save_workflow_results()

.. _tut-resume:

Tutorial 3 — Resuming a Failed Workflow
-----------------------------------------

Long workflows can be interrupted (network drop, out-of-memory, server restart).
Since v0.3 every step is checkpointed automatically.  Restart with ``resume=True``
and only the failed/skipped steps run again.

.. code-block:: python

    # First run — crashes at step 3 of 5
    try:
        results = coordinator.execute_workflow(config)
    except Exception as exc:
        print(f"Failed: {exc}")

    # Restart — steps 1 & 2 loaded from ./results/checkpoints/
    results = coordinator.execute_workflow(config, resume=True)

Checkpoint files are stored as:

.. code-block:: text

    results/
    └── checkpoints/
        ├── fetch_climate.pkl
        ├── fetch_climate.json        ← human-readable sidecar
        ├── load_ert.pkl
        └── load_ert.json

To force a clean restart delete the checkpoints folder:

.. code-block:: bash

    rm -rf ./results/checkpoints/

See also: :ref:`agents/troubleshooting:Resuming a Failed or Interrupted Workflow`

.. _tut-cost:

Tutorial 4 — Tracking LLM Cost and Token Usage
------------------------------------------------

Every LLM call is recorded in each agent's ``llm_usage_ledger`` and aggregated
by ``AgentCoordinator``.

**Per-agent ledger** (inspect raw entries):

.. code-block:: python

    for entry in coordinator.agents['ert_inversion'].llm_usage_ledger:
        print(entry)
    # {
    #   'agent': 'ert_inversion',
    #   'provider': 'openai',
    #   'model': 'gpt-4o-mini',
    #   'prompt_tokens': 320,
    #   'completion_tokens': 85,
    #   'total_tokens': 405,
    #   'cost_estimate_usd': 0.000121,
    #   'timestamp': 1714500000.0,
    # }

**Workflow-level summary** (after ``execute_workflow``):

.. code-block:: python

    summary = coordinator.get_workflow_summary()
    print(f"Status:       {summary['status']}")
    print(f"Steps done:   {summary['completed_steps']}")
    print(f"Total cost:   ${summary['total_llm_cost_estimate_usd']:.4f}")
    print(f"Total tokens: {summary['total_llm_tokens']}")
    print(f"LLM calls:    {summary['llm_calls']}")

Cost rates are defined in ``PyHydroGeophysX/agents/_pricing.py``.

Tutorial 5 — Natural Language to Workflow
------------------------------------------

Skip manual config dictionaries and let ``ContextInputAgent`` parse a plain
English description:

.. code-block:: python

    from PyHydroGeophysX.agents import ContextInputAgent
    import os

    context = ContextInputAgent(api_key=os.environ['OPENAI_API_KEY'])

    result = context.execute({
        'user_request': (
            "I have Syscal Pro ERT data at data/field.bin. "
            "Run a standard inversion with lambda=20, then convert "
            "resistivity to water content assuming porosity 0.35."
        )
    })
    config = result['workflow_config']
    # Pass config directly to coordinator.execute_workflow(config)

The agent is guided by a detailed system message set at initialisation and
augmented with ``.github/agents/context_input.agent.md`` on first call.

Further Reading
---------------

* Full workflow patterns: :doc:`/agents/workflows`
* Agent API: :doc:`/agents/agent_reference`
* Architecture: :doc:`/agents/architecture`
* Hosted app & 3D Mesh Builder: :doc:`/agents/webapp`
* Troubleshooting: :doc:`/agents/troubleshooting`

