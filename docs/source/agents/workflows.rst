Common Workflow Patterns
========================

This document describes common workflow patterns for using the PyHydroGeophysX
multi-agent system, from basic ERT processing to advanced multi-method fusion.

Pattern A: Basic ERT Processing
-------------------------------

**Use Case**: Single ERT dataset, no constraints

**Agents**: Context → Loader → Inversion → Evaluation → WaterContent → Report

**Workflow Diagram**:

.. code-block:: text

    User Input
        │
        ▼
    ContextInputAgent (parse natural language)
        │
        ▼
    ERTLoaderAgent (load data)
        │
        ▼
    ERTInversionAgent (invert)
        │
        ▼
    InversionEvaluationAgent (evaluate & optimize)
        │
        ▼
    WaterContentAgent (convert to WC)
        │
        ▼
    ReportAgent (generate report)

**Example Code**:

.. code-block:: python

    from PyHydroGeophysX.agents import (
        AgentCoordinator, ERTLoaderAgent, ERTInversionAgent,
        InversionEvaluationAgent, WaterContentAgent, ReportAgent,
    )
    import os

    api_key = os.environ.get('OPENAI_API_KEY')
    coordinator = AgentCoordinator(api_key=api_key, output_dir='./results')
    coordinator.register_agent('ert_loader', ERTLoaderAgent(api_key))
    coordinator.register_agent('ert_inversion', ERTInversionAgent(api_key))
    coordinator.register_agent('evaluation', InversionEvaluationAgent(api_key))
    coordinator.register_agent('water_content', WaterContentAgent(api_key))
    coordinator.register_agent('report', ReportAgent(api_key))

    config = {
        'data_file': 'field_ert.ohm',
        'instrument': 'E4D',
        'inversion_params': {'lambda': 20},
    }

    # Run — checkpoints saved after each step
    results = coordinator.execute_workflow(config)

    # Review cost and token usage
    summary = coordinator.get_workflow_summary()
    print(f"Total cost:   ${summary['total_llm_cost_estimate_usd']:.4f}")
    print(f"Total tokens: {summary['total_llm_tokens']}")

    # Save all agent outputs (numpy arrays → .npy, meshes → .bms)
    coordinator.save_workflow_results()

**Recovering from a failed run**:

.. code-block:: python

    # If the workflow crashes at, say, the inversion step, restart with:
    results = coordinator.execute_workflow(config, resume=True)
    # Steps already saved as checkpoints are skipped automatically.

Pattern B: Time-Lapse Monitoring
--------------------------------

**Use Case**: Multiple ERT datasets over time

**Agents**: Context → Loader (xN) → Climate → Inversion (time-lapse) → Evaluation → WaterContent → Report

**Special Features**:

* Temporal regularization
* Climate data integration
* Difference/ratio/joint methods

**Workflow Diagram**:

.. code-block:: text

    User Input
        │
        ▼
    ContextInputAgent (identify time-lapse)
        │
        ▼
    ERTLoaderAgent (load multiple datasets)
        │
        ▼
    ClimateDataAgent (fetch climate data) [optional]
        │
        ▼
    ERTInversionAgent (time-lapse inversion)
        │
        ├─ difference method
        ├─ ratio method
        └─ joint inversion
        │
        ▼
    InversionEvaluationAgent (evaluate quality)
        │
        ▼
    WaterContentAgent (temporal conversion)
        │
        ▼
    ReportAgent (temporal analysis + climate integration)

**Example Code**:

.. code-block:: python

    from PyHydroGeophysX.agents import (
        ERTLoaderAgent,
        ClimateDataAgent,
        ERTInversionAgent,
        WaterContentAgent
    )

    # Load multiple time-lapse datasets
    loader = ERTLoaderAgent(api_key)
    datasets = []
    for file in ['ert_t1.ohm', 'ert_t2.ohm', 'ert_t3.ohm']:
        result = loader.execute({'data_file': file, 'instrument': 'E4D'})
        datasets.append(result['ert_data'])

    # Fetch climate data
    climate = ClimateDataAgent(api_key)
    climate_data = climate.execute({
        'geometry': {'lat': 41.5, 'lon': -91.5},
        'start_date': '2024-01-01',
        'end_date': '2024-03-31',
        'variables': ['precipitation', 'temperature']
    })

    # Time-lapse inversion
    inverter = ERTInversionAgent(api_key)
    inv_result = inverter.execute({
        'time_lapse_data': datasets,
        'inversion_mode': 'time-lapse',
        'time_lapse_method': 'ratio',
        'temporal_regularization': 10.0
    })

Pattern C: Structure-Constrained Fusion
---------------------------------------

**Use Case**: Seismic + ERT with layer identification

**Agents**: Context → DataFusion → Seismic → Structure → Petrophysics → Report

**Special Features**:

* Interface extraction from seismic velocity
* Layer-specific petrophysical parameters
* Monte Carlo uncertainty (10,000 realizations)

**Workflow Diagram**:

.. code-block:: text

    User Input
        │
        ▼
    ContextInputAgent (parse multi-method request)
        │
        ▼
    DataFusionAgent (recommend fusion pattern)
        │
        ├─────────────────┬─────────────────┐
        │                 │                 │
        ▼                 ▼                 ▼
    SeismicAgent    ERTLoaderAgent     (other methods)
    (velocity)          │
        │               │
        └───────┬───────┘
                │
                ▼
      StructureConstraintAgent
      (extract interfaces → constrained mesh → constrained inversion)
                │
                ▼
        PetrophysicsAgent
        (layer-specific conversion with MC uncertainty)
                │
                ▼
          ReportAgent

**Example Code**:

.. code-block:: python

    from PyHydroGeophysX.agents import (
        DataFusionAgent,
        SeismicAgent,
        ERTLoaderAgent,
        StructureConstraintAgent,
        PetrophysicsAgent
    )

    # Recommend fusion pattern
    fusion = DataFusionAgent(api_key)
    plan = fusion.execute({
        'fusion_pattern': 'auto',
        'methods': ['seismic', 'ert'],
        'data': {
            'seismic': 'seismic.sgt',
            'ert': 'ert.ohm'
        }
    })

    # Process seismic data
    seismic = SeismicAgent(api_key)
    seis_result = seismic.execute({
        'seismic_data': 'seismic.sgt',
        'velocity_threshold': 1200
    })

    # Load ERT data
    ert_loader = ERTLoaderAgent(api_key)
    ert_data = ert_loader.execute({
        'data_file': 'ert.ohm',
        'instrument': 'E4D'
    })

    # Structure-constrained inversion
    structure = StructureConstraintAgent(api_key)
    constrained = structure.execute({
        'ert_data': ert_data['ert_data'],
        'velocity_model': seis_result['velocity_model'],
        'velocity_thresholds': [1000, 1950],
        'lambda': 20
    })

    # Layer-specific petrophysics
    petro = PetrophysicsAgent(api_key)
    wc = petro.execute({
        'resistivity_model': constrained['resistivity_model'],
        'mesh': constrained['mesh'],
        'cell_markers': constrained['cell_markers'],
        'n_realizations': 10000
    })

Pattern D: Full Field-Scale Analysis
------------------------------------

**Use Case**: Complete hydrogeophysical characterization

**Agents**: Context → DataFusion → Seismic → ERTLoader → Structure → Evaluation → Petrophysics → Report

**Special Features**:

* Multi-method integration
* Quality optimization loop
* Comprehensive uncertainty quantification
* Publication-ready reports

**Workflow Diagram**:

.. code-block:: text

    User Natural Language Request
            │
            ▼
    ContextInputAgent
            │
            ▼
    DataFusionAgent
            │
            ├─────────────────────────────┐
            │                             │
            ▼                             ▼
    SeismicAgent                    ERTLoaderAgent
            │                             │
            └───────────┬─────────────────┘
                        │
                        ▼
            StructureConstraintAgent
                        │
                        ▼
            InversionEvaluationAgent
                        │
                    ┌───┴───┐
                    │       │
                    ▼       ▼
                Adjust   Accept
                        │
                        ▼
            PetrophysicsAgent (MC uncertainty)
                        │
                        ▼
                ReportAgent

TDEM Workflow
-------------

**Use Case**: Time-Domain Electromagnetic sounding from hydrological models

**Workflow Diagram**:

.. code-block:: text

    MODFLOW/ParFlow Data
            │
            ▼
    Extract 1D Vertical Profile
            │
            ▼
    WS_Model (Waxman-Smits Petrophysics)
            │
            ▼
    TDEMForwardModeling
            │
            ▼
    Add Synthetic Noise
            │
            ▼
    TDEMInversion
            │
            ▼
    Compare True vs Recovered

**Example Code**:

.. code-block:: python

    from PyHydroGeophysX.forward.tdem_forward import (
        TDEMForwardModeling,
        TDEMSurveyConfig
    )
    from PyHydroGeophysX.inversion.tdem_inversion import TDEMInversion
    from PyHydroGeophysX.petrophysics.resistivity_models import WS_Model
    import numpy as np

    # Convert water content to conductivity
    conductivity = 1.0 / WS_Model(
        saturation, porosity, sigma_w=0.05, m=1.5, n=2.0, sigma_s=0.001
    )

    # Configure TDEM survey
    times = np.logspace(-5, -2, 31)
    survey = TDEMSurveyConfig(
        source_location=np.array([0.0, 0.0, 0.0]),
        source_radius=10.0,
        times=times
    )

    # Forward modeling
    fwd = TDEMForwardModeling(thicknesses, survey)
    dobs, dpred, uncertainties = fwd.forward_with_noise(
        conductivity, noise_level=0.05
    )

    # Inversion
    inv = TDEMInversion(
        times=times,
        dobs=dobs,
        uncertainties=uncertainties,
        source_radius=10.0,
        n_layers=20,
        use_irls=True
    )
    result = inv.run()

3D ERT Workflow
---------------

**Use Case**: 3D ERT forward modeling with MODFLOW integration

**Workflow Diagram**:

.. code-block:: text

    MODFLOW 3D Data
            │
            ▼
    Analyze Active Cells (find optimal domain)
            │
            ▼
    Create Surface Electrode Array
            │
            ▼
    Apply Topography to Electrodes
            │
            ▼
    Create 3D Mesh with Topography
            │
            ▼
    Interpolate Hydrological Properties
            │
            ▼
    Waxman-Smits Conversion
            │
            ▼
    3D ERT Forward Modeling
            │
            ▼
    PyVista Visualization

**Example Code**:

.. code-block:: python

    from PyHydroGeophysX.core.mesh_3d import (
        Mesh3DCreator,
        create_3d_ert_data_container
    )
    from PyHydroGeophysX.petrophysics.resistivity_models import WS_Model
    from pygimli.physics import ert

    # Initialize mesh creator
    creator = Mesh3DCreator(
        mesh_directory='./meshes',
        elec_refinement=0.3
    )

    # Create electrodes with topography
    electrodes = creator.create_surface_electrode_array(
        nx=5, ny=5, dx=4.0, dy=4.0
    )
    electrodes = creator.apply_topography_to_electrodes(
        electrodes, topography_function
    )

    # Create 3D mesh
    mesh = creator.create_3d_mesh_with_topography(
        electrode_positions=electrodes,
        topography_func=topography_function,
        para_depth=14.0,
        use_prism_mesh=True
    )

    # Create data container
    data = create_3d_ert_data_container(
        electrodes, scheme='dd', dimension=3
    )

    # Forward modeling
    fop = ert.ERTModelling()
    fop.setData(data)
    fop.setMesh(mesh)
    response = fop.response(resistivity_mesh)

Best Practices
--------------

1. **Start Simple**: Begin with Pattern A before moving to more complex workflows

2. **Validate Data**: Always enable quality_check=True in ERTLoaderAgent

3. **Iterate on Inversion**: Use InversionEvaluationAgent with auto_adjust=True

4. **Quantify Uncertainty**: Use PetrophysicsAgent with n_realizations >= 1000

5. **Document Results**: Always include ReportAgent for reproducibility

6. **Check Chi-Squared**: Target chi2 between 0.8 and 1.5 for good data fit

7. **Use Appropriate Constraints**: Structure constraints improve resolution but require seismic data
