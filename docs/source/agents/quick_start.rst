Quick Start Guide
=================

This guide will help you get started with the PyHydroGeophysX multi-agent system
in just a few minutes.

.. contents:: On this page
   :local:
   :depth: 1

Installation
------------

The agent system requires an API key from your preferred LLM provider:

.. code-block:: bash

    # Set your API key as an environment variable
    export OPENAI_API_KEY="your-api-key-here"
    # or
    export GEMINI_API_KEY="your-api-key-here"
    # or
    export ANTHROPIC_API_KEY="your-api-key-here"

Try Without an API Key
----------------------

The hosted Streamlit app starts in demo mode by default. Demo mode loads bundled
cached results, makes no LLM calls, and lets first-time users inspect the expected
outputs before uploading their own data:

`https://pyhydrogeophysx.streamlit.app/ <https://pyhydrogeophysx.streamlit.app/>`_

To preview a local agent workflow without running an inversion or interpretation
step, use the dry-run mode:

.. code-block:: python

    from PyHydroGeophysX.agents import AgentCoordinator
    coordinator = AgentCoordinator(api_key=None)
    result = coordinator.execute_workflow(
        {"user_request": "Run ERT inversion on examples/data/sample.ohm"},
        dry_run=True,
    )
    # result contains: execution_plan, validation_warnings, cost_estimate_usd

For a local no-LLM smoke test that runs a short ERT inversion on bundled data:

.. code-block:: bash

    python examples/Ex_hello_agent.py

Launch the 3D Mesh Builder
--------------------------

A dedicated interactive GUI for building 3D ERT meshes ships with the package:

.. code-block:: bash

    # Recommended — via the package launcher
    python -m PyHydroGeophysX.gui_mesh3d

    # Or run directly with Streamlit
    streamlit run examples/app_mesh3d.py

The app opens in your browser with three tabs: **Electrode View** (interactive
3D preview), **Generate Mesh** (runs ``Mesh3DCreator``), and **Export** (.bms /
.vtk download). See :ref:`agents/webapp:3D Mesh Builder App` for the full
step-by-step guide.

Dry-Run / Preview Workflow
--------------------------

Before committing to a full inversion run, use ``dry_run=True`` (or
``resume=True`` to skip completed steps after a failure):

.. code-block:: python

    from PyHydroGeophysX.agents import AgentCoordinator

    coordinator = AgentCoordinator(api_key=None, output_dir='./results')

    # Preview: validates files, checks dependencies, estimates LLM cost
    preview = coordinator.execute_workflow(
        {"data_file": "field_ert.ohm", "instrument": "E4D"},
        dry_run=True,
    )
    print(preview["data"]["execution_plan"])
    print(preview["data"]["validation_warnings"])
    print(f"Est. cost: ${preview['cost_estimate_usd']:.4f}")

    # Resume a previously interrupted run (skips checkpointed steps)
    results = coordinator.execute_workflow(config, resume=True)

Basic Usage
-----------

The simplest way to use the agent system is through the ``AgentCoordinator``:

.. code-block:: python

    from PyHydroGeophysX.agents import (
        AgentCoordinator,
        ContextInputAgent,
        ERTLoaderAgent,
        ERTInversionAgent,
        WaterContentAgent,
        ReportAgent
    )
    import os

    api_key = os.environ.get('OPENAI_API_KEY')
    coordinator = AgentCoordinator(api_key=api_key, output_dir='./results')

    coordinator.register_agent('context', ContextInputAgent(api_key))
    coordinator.register_agent('ert_loader', ERTLoaderAgent(api_key))
    coordinator.register_agent('ert_inversion', ERTInversionAgent(api_key))
    coordinator.register_agent('water_content', WaterContentAgent(api_key))
    coordinator.register_agent('report', ReportAgent(api_key))

    config = {
        'data_file': 'data/field_ert.ohm',
        'instrument': 'E4D',
        'inversion_params': {'lambda': 20, 'max_iter': 10},
    }

    results = coordinator.execute_workflow(config)

    # Check LLM cost after the run
    summary = coordinator.get_workflow_summary()
    print(f"Total LLM cost: ${summary['total_llm_cost_estimate_usd']:.4f}")
    print(f"Tokens used:    {summary['total_llm_tokens']}")
    print(f"LLM calls:      {summary['llm_calls']}")

Natural Language Interface
--------------------------

Describe your workflow in plain English and let ``ContextInputAgent`` translate it:

.. code-block:: python

    from PyHydroGeophysX.agents import ContextInputAgent
    import os

    context = ContextInputAgent(api_key=os.environ['OPENAI_API_KEY'])

    request = """
    I have ERT data from a Syscal Pro instrument.
    The data file is data/field_data.bin. I want to perform
    a standard inversion with moderate smoothing and then convert the
    resistivity to water content using Archie's law with porosity 0.35.
    """

    result = context.execute({'user_request': request})
    workflow_config = result['workflow_config']

Example: Standard ERT Workflow
-------------------------------

.. code-block:: python

    from PyHydroGeophysX.agents import (
        ERTLoaderAgent, ERTInversionAgent,
        InversionEvaluationAgent, WaterContentAgent,
    )
    import os

    api_key = os.environ.get('OPENAI_API_KEY')

    # Step 1: Load ERT data
    loader = ERTLoaderAgent(api_key)
    data_result = loader.execute({
        'data_file': 'data/field_ert.ohm',
        'instrument': 'E4D',
        'quality_check': True,
    })

    # Step 2: Invert
    inverter = ERTInversionAgent(api_key)
    inv_result = inverter.execute({
        'ert_data': data_result['ert_data'],
        'inversion_mode': 'standard',
        'inversion_params': {'lambda': 20, 'max_iter': 10},
    })

    # Step 3: Evaluate and auto-optimize if chi² is poor
    evaluator = InversionEvaluationAgent(api_key)
    eval_result = evaluator.execute({
        'inversion_results': inv_result,
        'ert_data': data_result['ert_data'],
        'auto_adjust': True,
    })

    # Step 4: Convert to water content
    converter = WaterContentAgent(api_key)
    wc_result = converter.execute({
        'inversion_results': eval_result['final_results'],
        'petrophysical_params': {
            'layer1': {'porosity': 0.35, 'n': 2.0, 'm': 1.5},
        },
    })

    print(f"Water content range: {wc_result['statistics']}")

    # Step 5: Save results (numpy arrays → .npy, meshes → .bms)
    converter.save_results('./results/water_content')

Example: Multi-Method Fusion
-----------------------------

Combining seismic and ERT data with structural constraints:

.. code-block:: python

    from PyHydroGeophysX.agents import DataFusionAgent
    import os

    fusion = DataFusionAgent(api_key=os.environ.get('OPENAI_API_KEY'))

    pattern_result = fusion.execute({
        'fusion_pattern': 'structure_constraint',
        'methods': ['seismic', 'ert'],
        'data': {'seismic': 'data/seismic.sgt', 'ert': 'data/ert.ohm'},
    })
    # DataFusionAgent creates a structured execution plan and runs it end-to-end.


Next Steps
----------

* Read the :doc:`architecture` document for detailed system design
* Explore :doc:`agent_reference` for individual agent documentation
* See :doc:`workflows` for common workflow patterns
