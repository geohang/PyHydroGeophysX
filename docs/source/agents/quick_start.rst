Quick Start Guide
=================

This guide will help you get started with the PyHydroGeophysX multi-agent system
in just a few minutes.

Installation
------------

The agent system requires an API key from your preferred LLM provider:

.. code-block:: bash

    # Set your API key as an environment variable
    export OPENAI_API_KEY="your-api-key-here"
    # or
    export GOOGLE_API_KEY="your-api-key-here"
    # or
    export ANTHROPIC_API_KEY="your-api-key-here"

Basic Usage
-----------

The simplest way to use the agent system is through the AgentCoordinator:

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

    # Initialize coordinator
    api_key = os.environ.get('OPENAI_API_KEY')
    coordinator = AgentCoordinator(
        api_key=api_key,
        output_dir='./results'
    )

    # Register agents
    coordinator.register_agent('context', ContextInputAgent(api_key))
    coordinator.register_agent('ert_loader', ERTLoaderAgent(api_key))
    coordinator.register_agent('ert_inversion', ERTInversionAgent(api_key))
    coordinator.register_agent('water_content', WaterContentAgent(api_key))
    coordinator.register_agent('report', ReportAgent(api_key))

    # Define workflow configuration
    config = {
        'data_file': 'data/field_ert.ohm',
        'instrument': 'E4D',
        'inversion_params': {
            'lambda': 20,
            'max_iter': 10
        }
    }

    # Execute workflow
    results = coordinator.execute_workflow(config)

Natural Language Interface
--------------------------

You can also describe your workflow in plain English:

.. code-block:: python

    from PyHydroGeophysX.agents import ContextInputAgent

    # Initialize context agent
    context = ContextInputAgent(api_key)

    # Describe workflow in natural language
    request = """
    I have ERT data from a Syscal Pro instrument that I need to process.
    The data file is located at data/field_data.bin. I want to perform
    a standard inversion with moderate smoothing and then convert the
    resistivity to water content using Archie's law with porosity 0.35.
    """

    # Parse to structured configuration
    result = context.execute({'user_request': request})
    workflow_config = result['workflow_config']

    # workflow_config now contains:
    # {
    #     'data_file': 'data/field_data.bin',
    #     'instrument': 'Syscal',
    #     'inversion_params': {'lambda': 20, ...},
    #     'petrophysical_params': {'porosity': 0.35, ...}
    # }

Example: Standard ERT Workflow
------------------------------

Here's a complete example of processing ERT data:

.. code-block:: python

    from PyHydroGeophysX.agents import (
        ERTLoaderAgent,
        ERTInversionAgent,
        InversionEvaluationAgent,
        WaterContentAgent
    )

    # Step 1: Load ERT data
    loader = ERTLoaderAgent(api_key)
    data_result = loader.execute({
        'data_file': 'data/field_ert.ohm',
        'instrument': 'E4D',
        'quality_check': True
    })

    # Step 2: Perform inversion
    inverter = ERTInversionAgent(api_key)
    inv_result = inverter.execute({
        'ert_data': data_result['ert_data'],
        'inversion_mode': 'standard',
        'inversion_params': {
            'lambda': 20,
            'max_iter': 10
        }
    })

    # Step 3: Evaluate quality
    evaluator = InversionEvaluationAgent(api_key)
    eval_result = evaluator.execute({
        'inversion_results': inv_result,
        'ert_data': data_result['ert_data'],
        'auto_adjust': True
    })

    # Step 4: Convert to water content
    converter = WaterContentAgent(api_key)
    wc_result = converter.execute({
        'inversion_results': eval_result['final_results'],
        'petrophysical_params': {
            'layer1': {'porosity': 0.35, 'n': 2.0, 'm': 1.5}
        }
    })

    print(f"Water content range: {wc_result['statistics']}")

Example: Multi-Method Fusion
----------------------------

Combining seismic and ERT data with structural constraints:

.. code-block:: python

    from PyHydroGeophysX.agents import (
        DataFusionAgent,
        SeismicAgent,
        StructureConstraintAgent,
        PetrophysicsAgent
    )

    # Initialize fusion agent
    fusion = DataFusionAgent(api_key)

    # Recommend fusion pattern
    pattern_result = fusion.execute({
        'fusion_pattern': 'auto',
        'methods': ['seismic', 'ert'],
        'data': {
            'seismic': 'data/seismic.sgt',
            'ert': 'data/ert.ohm'
        }
    })

    # The DataFusionAgent will recommend 'structure_constraint' pattern
    # and create an execution plan

Next Steps
----------

* Read the :doc:`architecture` document for detailed system design
* Explore :doc:`agent_reference` for individual agent documentation
* See :doc:`workflows` for common workflow patterns
