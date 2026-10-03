Agent Reference
===============

This document provides detailed documentation for each agent in the PyHydroGeophysX
multi-agent system, including their inputs, outputs, and responsibilities.

BaseAgent
---------

The abstract base class that all agents inherit from.

.. code-block:: python

    class BaseAgent:
        def __init__(self, name, api_key, model, llm_provider):
            self.name = name
            self.api_key = api_key
            self.model = model
            self.llm_provider = llm_provider
            self.llm_usage_ledger = []   # one dict per LLM call

        def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
            raise NotImplementedError

**Key utility methods** added in v0.3:

``query_llm(prompt, system_message=None)``
    Sends a prompt to the configured LLM provider.  On the **first call**,
    the agent lazily loads ``.github/agents/<name>.agent.md`` and appends its
    body to the system message (YAML frontmatter is stripped automatically).

``save_results(output_dir)``
    Persists all agent outputs to *output_dir*.  Type-aware serialisation:

    * NumPy arrays → ``<name>_<key>.npy``
    * PyGIMLi meshes / ``DataContainer`` → ``<name>_<key>.bms``
    * Pandas DataFrames → ``<name>_<key>.csv``
    * Plain JSON-serialisable values → ``results.json``
    * Anything else → stub entry in ``results.json`` with ``{__type__, repr}``

``_retry_llm_call(fn, max_retries=3)`` *(static)*
    Wraps any callable that performs one LLM call.  Retries up to
    ``max_retries`` times with exponential back-off (sleep 2^attempt seconds)
    for transient rate-limit errors.  Non-transient errors are propagated
    immediately without retry.

``_load_agent_md_for_name(name)`` *(static)*
    Reads ``.github/agents/<name>.agent.md`` relative to the repo root and
    returns the Markdown body with YAML frontmatter stripped.  Returns ``""``
    if the file is missing.

AgentCoordinator
----------------

**Purpose**: Orchestrates multi-agent workflows and manages execution state.

This is not a processing agent but an orchestration layer that:

* Registers agents
* Manages workflow state
* Coordinates agent execution (with optional checkpoint / resume)
* Aggregates LLM cost across all registered agents
* Validates environment dependencies before running

**Key Methods** (v0.3):

.. code-block:: python

    register_agent(name, instance)
    # Add an agent to the workflow.

    execute_workflow(config, dry_run=False, resume=False)
    # Run the complete workflow.
    # dry_run=True: validate, plan, and estimate cost without running agents.
    # resume=True: skip steps for which a checkpoint already exists.
    # config['user_request'] fills in the ERT files, instrument, petrophysical
    # parameters and seismic file it names where config leaves them out;
    # anything it asks for that the run does not do is listed in
    # result['warnings'] (and in the preview's validation_warnings).

    preview_workflow(config)
    # Equivalent to execute_workflow(..., dry_run=True).
    # Returns validation_warnings (including dependency checks), execution_plan,
    # and cost_estimate_usd.

    get_workflow_state()
    # Return current status dict.

    get_workflow_summary()
    # Return aggregated statistics after (or during) a run:
    # {
    #   'status': ...,
    #   'completed_steps': [...],
    #   'total_steps': N,
    #   'current_step': ...,
    #   'available_results': [...],
    #   'total_llm_cost_estimate_usd': 0.0034,
    #   'total_llm_tokens': 8200,
    #   'llm_calls': 12,
    # }

    save_workflow_results()
    # Persist all agent outputs via each agent's save_results().

**Checkpoint / Resume Example**:

.. code-block:: python

    from PyHydroGeophysX.agents import AgentCoordinator

    coordinator = AgentCoordinator(api_key=api_key, output_dir='./results')
    # ... register agents ...

    # First attempt: may fail at step 3 of 5
    try:
        results = coordinator.execute_workflow(config)
    except Exception as exc:
        print(f"Workflow failed: {exc}")

    # Resume: steps 1 and 2 are loaded from checkpoints
    results = coordinator.execute_workflow(config, resume=True)

**Dependency Pre-check**:

``preview_workflow()`` automatically calls ``_check_dependencies(plan)`` to
test whether required packages (``pygimli``, ``gmsh``, ``anthropic``, and
``google-genai`` or the older ``google-generativeai``) are importable **before**
the workflow runs.  Missing
dependencies appear in ``validation_warnings``.

ContextInputAgent
-----------------

**Purpose**: Translates natural language workflow descriptions into structured configurations.

``ContextInputAgent`` always sets a detailed ``self.system_message`` in
``__init__`` so that it correctly guides the LLM from the very first call,
even before the ``.agent.md`` augmentation hook fires:

.. code-block:: python

    # excerpt from __init__
    self.system_message = (
        "You are an expert workflow configuration interpreter for "
        "PyHydroGeophysX.  Translate natural-language geophysical workflow "
        "requests into structured JSON configuration dictionaries..."
    )

**Inputs**:

* ``user_request`` (str): Natural language workflow description
* ``available_data`` (dict, optional): Available files/instruments

**Outputs**:

* ``workflow_config`` (dict): Structured configuration
* ``explanation`` (str): Human-readable explanation

ERTLoaderAgent
--------------

**Purpose**: Loads and validates ERT field data from various instruments.

**System Prompt**:
    You are an expert in electrical resistivity tomography (ERT) data processing.
    Your role is to load and validate ERT field data from various commercial instruments.

**Inputs**:

* ``data_file`` (str): Path to ERT data file
* ``instrument`` (str): Instrument type (E4D, Syscal, ABEM, BERT)
* ``project_dir`` (str): Project directory
* ``crs`` (str): Coordinate reference system
* ``quality_check`` (bool): Whether to perform QC

**Outputs**:

* ``ert_data`` (object): Loaded ERT dataset (PyGIMLi DataContainer)
* ``num_electrodes`` (int): Number of electrodes
* ``num_measurements`` (int): Number of measurements
* ``quality_metrics`` (dict): Data quality statistics

ERTInversionAgent
-----------------

**Purpose**: Performs ERT inversion (standard or time-lapse).

**System Prompt**:
    You are an expert in electrical resistivity tomography (ERT) inversion. Your role
    is to configure and execute ERT inversions, select appropriate regularization
    parameters, and interpret inversion results.

**Inputs**:

* ``ert_data`` (object): ERT data (for standard inversion)
* ``time_lapse_data`` (list): List of ERT datasets (for time-lapse)
* ``inversion_mode`` (str): 'standard' or 'time-lapse'
* ``time_lapse_method`` (str): 'difference', 'ratio', or 'joint'
* ``temporal_regularization`` (float): Temporal smoothing weight
* ``inversion_params`` (dict): Lambda, max_iter, method
* ``use_structure_constraint`` (bool): Whether to use seismic structure
* ``seismic_structure`` (object): Optional seismic structure data

**Outputs**:

* ``resistivity_model`` (array): Inverted resistivity model
* ``mesh`` (object): PyGIMLi mesh
* ``chi2_values`` (list): Chi-squared fit statistics
* ``coverage`` (array): Model coverage/sensitivity
* ``final_models`` (array): Time-series models (for time-lapse)

InversionEvaluationAgent
------------------------

**Purpose**: Evaluates inversion quality and automatically optimizes parameters.

**System Prompt**:
    You are an expert in geophysical inversion quality assessment. Your role is to
    evaluate ERT inversion results based on data fit, model smoothness, and physical
    plausibility.

**Inputs**:

* ``inversion_results`` (dict): Results from ERTInversionAgent
* ``ert_data`` (object): Original ERT data
* ``inversion_params`` (dict): Current parameters
* ``auto_adjust`` (bool): Whether to auto-adjust parameters
* ``max_attempts`` (int): Maximum re-inversion attempts

**Outputs**:

* ``quality_score`` (float): Overall quality (0-100)
* ``quality_metrics`` (dict): Detailed metrics
* ``component_scores`` (dict): Individual component scores
* ``recommendations`` (list): Improvement suggestions
* ``adjusted_params`` (dict): Optimized parameters
* ``final_results`` (dict): Best inversion results

**Quality Metrics**:

1. **Data Fit**: Chi-squared target (0.8-1.5 acceptable)
2. **Smoothness**: Model roughness evaluation
3. **Physical Plausibility**: Resistivity range (1-10,000 Ohm-m)
4. **Convergence**: Iteration stability
5. **Coverage**: Model sensitivity

DataFusionAgent
---------------

**Purpose**: Intelligent coordinator for multi-method geophysical workflows.

**System Prompt**:
    You are an expert in multi-method geophysical data fusion. You understand how
    different geophysical methods complement each other and can recommend optimal
    workflows for integrating multiple datasets.

**Inputs**:

* ``fusion_pattern`` (str): Pattern name or 'auto'
* ``methods`` (list): Available methods
* ``workflow_config`` (dict): Configuration for fusion
* ``data`` (dict): Data for each method
* ``output_dir`` (str): Results directory

**Outputs**:

* ``fusion_pattern`` (str): Selected pattern
* ``execution_plan`` (list): Step-by-step plan
* ``status`` (str): Success/failure
* ``interpretation`` (str): AI interpretation of results

StructureConstraintAgent
------------------------

**Purpose**: Applies seismic velocity interfaces as structural constraints to ERT inversion.

**System Prompt**:
    You are an expert in structure-constrained geophysical inversion. You understand
    how to incorporate a priori geological information from seismic data into ERT
    inversions.

**Inputs**:

* ``ert_data`` (object): ERT measurement data
* ``seismic_data`` (object): Seismic travel time data (optional)
* ``velocity_model`` (array): Velocity model from seismic inversion
* ``mesh`` (object): PyGIMLi mesh
* ``velocity_thresholds`` (list): Thresholds for interface extraction
* ``mesh_quality`` (int): Constrained mesh quality
* ``lambda`` (float): ERT regularization parameter
* ``limits`` (list): Resistivity bounds [min, max]

**Outputs**:

* ``resistivity_model`` (array): Constrained resistivity model
* ``mesh`` (object): Constrained mesh with layer markers
* ``cell_markers`` (array): Cell layer identifications
* ``coverage`` (array): Model coverage
* ``interfaces`` (list): Extracted velocity interfaces
* ``statistics`` (dict): Resistivity range, chi2, data fit, n_layers

PetrophysicsAgent
-----------------

**Purpose**: Converts resistivity to water content using layer-specific petrophysical
models with Monte Carlo uncertainty quantification.

**System Prompt**:
    You are an expert in petrophysical modeling and hydrogeophysics. You understand
    how to convert electrical resistivity to water content using Archie's law and
    modified petrophysical relationships.

**Petrophysical Model**:

.. code-block:: text

    Archie's Law (modified with surface conductivity):
    sigma_bulk = sigma_fluid * phi^m * S^n + sigma_surface

    Where:
    - sigma_bulk: Bulk conductivity (1/resistivity)
    - sigma_fluid: Fluid conductivity (1/rho_fluid)
    - phi: Porosity
    - S: Saturation (water content / porosity)
    - m: Cementation exponent
    - n: Saturation exponent
    - sigma_surface: Surface conductivity (clay effect)

**Default Layer Parameters**:

+------------+---------------+-----+-----+-----------------+--------------+
| Layer Type | Porosity (phi)| m   | n   | sigma_surface   | rho_fluid    |
+============+===============+=====+=====+=================+==============+
| Regolith   | 0.42 +/- 0.05 | 1.3 | 2.1 | 1/200 +/- 1/200 | 20 Ohm-m     |
+------------+---------------+-----+-----+-----------------+--------------+
| Bedrock    | 0.25 +/- 0.15 | 1.9 | 1.7 | 0.0 +/- 0.0     | 20 Ohm-m     |
+------------+---------------+-----+-----+-----------------+--------------+

**Inputs**:

* ``resistivity_model`` (array): Resistivity values
* ``mesh`` (object): PyGIMLi mesh
* ``cell_markers`` (array): Layer identifications
* ``layer_params`` (dict): Parameters for each layer
* ``n_realizations`` (int): Monte Carlo samples (default: 100)

**Outputs**:

* ``water_content_mean`` (array): Mean water content per cell
* ``water_content_std`` (array): Standard deviation (uncertainty)
* ``saturation_mean`` (array): Mean saturation
* ``saturation_std`` (array): Saturation uncertainty
* ``statistics`` (dict): WC range, mean WC, mean uncertainty

WaterContentAgent
-----------------

**Purpose**: General resistivity to water content conversion (simpler than PetrophysicsAgent).

**System Prompt**:
    You are an expert in petrophysical relationships and rock physics. Your role is
    to convert electrical resistivity to water content using appropriate models.

**Inputs**:

* ``inversion_results`` (dict): ERT inversion results
* ``petrophysical_params`` (dict): Parameters for each layer
* ``uncertainty_analysis`` (bool): Whether to run Monte Carlo
* ``n_realizations`` (int): MC realizations (default: 100)

**Outputs**:

* ``water_content`` (array): Water content estimates
* ``uncertainties`` (array): Uncertainty estimates (if MC enabled)
* ``statistics`` (dict): Summary statistics

SeismicAgent
------------

**Purpose**: Processes seismic refraction data and extracts velocity structures.

**System Prompt**:
    You are an expert in seismic refraction tomography (SRT). Your role is to process
    seismic travel time data, perform velocity inversions, and extract geological
    structure interfaces.

**Inputs**:

* ``seismic_data`` (object or str): Seismic travel time data, loaded or as a
  file path (the same as ``seismic_file``)
* ``velocity_threshold`` (float): Threshold for interface detection
* ``inversion_params`` (dict): Seismic inversion parameters
* ``output_dir`` (str): Results directory

**Outputs**:

* ``velocity_model`` (array): Velocity distribution
* ``interface_coords`` (tuple): (x, z) coordinates of interface
* ``mesh`` (object): Seismic inversion mesh
* ``statistics`` (dict): Velocity range, chi2, data fit

TDEMAgent
---------

**Purpose**: Performs Time-Domain Electromagnetic forward modeling and inversion.

**Inputs**:

* ``layer_thicknesses`` (array): Layer thicknesses for 1D model
* ``conductivity`` (array): Layer conductivities
* ``survey_config`` (TDEMSurveyConfig): Survey parameters
* ``inversion_params`` (dict): Inversion configuration

**Outputs**:

* ``forward_response`` (array): TDEM response
* ``recovered_model`` (array): Inverted conductivity model
* ``chi2`` (float): Data misfit
* ``statistics`` (dict): Inversion statistics

ClimateDataAgent
----------------

**Purpose**: Retrieves the daily weather at the survey site - precipitation,
minimum and maximum air temperature and reference evapotranspiration - from the
Open-Meteo historical-weather API (the ERA5 reanalysis; one HTTPS request, no API
key, global coverage), with the antecedent-moisture features ERT is read against.

**System Prompt**:
    You are an expert in climate data analysis for hydrogeophysical studies. You
    understand how precipitation, evapotranspiration, and temperature affect subsurface
    moisture and resistivity measurements.

**Inputs**:

* ``coords`` (tuple): Site longitude and latitude, in ``crs``; a list of points
  is averaged, since an ERT line spans far less than one reanalysis cell
* ``dates`` (tuple): ``(start_date, end_date)`` as YYYY-MM-DD, or a list of years
* ``crs`` (int or str): Coordinate reference system of ``coords`` (default 4326)
* ``ert_timestamps`` (list, optional): Survey times to align the series to
* ``antecedent_days`` (list, optional): Windows for antecedent precipitation
  totals (default 1, 3 and 7 days)
* ``output_dir`` (str, optional): Folder to save ``climate_data.csv`` and its
  metadata in
* ``csv_file`` (str, optional): A series saved earlier, read instead of fetching

**Outputs**:

* ``climate_data`` (DataFrame): Daily ``prcp`` (mm), ``tmin`` and ``tmax``
  (degC) and ``pet`` (FAO-56 reference evapotranspiration, mm)
* ``derived_features`` (dict): Antecedent precipitation totals and P - PET
* ``ert_alignment`` (dict): The rows of the survey days, when
  ``ert_timestamps`` were given
* ``metadata`` (dict): The period, the source and its attribution, the site
  and the grid cell the values come from
* ``notes`` (list): Anything asked for that the source does not provide

ReportAgent
-----------

**Purpose**: Generates comprehensive reports from workflow results.

**System Prompt**:
    You are an expert in technical report writing for geophysical and hydrological
    studies. Your role is to synthesize results from ERT data processing, inversion,
    water content analysis, and climate data into clear, informative reports.

**Report Sections**:

1. Executive Summary
2. Data Processing Summary
3. Climate Data Summary (if available)
4. Inversion Results
5. Water Content Analysis
6. Climate-Resistivity Analysis (if climate data available)
7. Quality Assessment
8. Conclusions & Recommendations

**Inputs**:

* ``workflow_data`` (dict): All data from workflow steps
* ``config`` (dict): Original workflow configuration
* ``output_dir`` (str): Report output directory

**Outputs**:

* ``report_path`` (str): Path to generated report
* ``figures`` (list): Generated figure paths
* ``summary_stats`` (dict): Key statistics

Workflow Controller
-------------------

**Purpose**: Chooses each step of a ``BaseAgent.run_unified_agent_workflow()``
run.

The entry point hands the parsed configuration to the controller in
``PyHydroGeophysX.agents.runtime``. Before each step the controller lists the
tools whose inputs the run already holds (``load_ert_surveys``,
``fetch_climate``, ``invert_ert``, ``invert_time_lapse``,
``evaluate_inversion``, ``convert_water_content``, ``invert_seismic``,
``derive_structure``, ``fuse_methods``, ``invert_tdem``,
``convert_tdem_water_content``, ``invert_mt``, ``convert_mt_water_content``,
``load_model_output``, ``write_report``), asks the model which one to run and
reads the result before choosing again. Without an API key it takes the first
runnable tool in registration order, which is dependency order.

``invert_mt`` runs when the configuration names MT sites (``mt_files``: EDI,
EMTF XML, Z- or J-files, or folders of them; a request that names ``.edi``
files sets it). It inverts each site in 1D by Occam's method, with the options
of ``mt_params``, and a line of three or more located sites also in 2D, with
``mt_profile_params``; ``convert_mt_water_content`` converts the layered
models when the request asks for water content and no ERT or TDEM model is
there to carry it.

**Inputs**: ``workflow_config`` dict, as ``ContextInputAgent.parse_request``
produces it.

**Outputs**: ``(results, execution_plan, interpretation, report_files)``. The
execution plan lists the steps that ran, each with the reason it was chosen.

Set ``PHGX_LEGACY_WORKFLOW=1`` to run the previous pipeline instead. It sorts
the request into one of eight workflow types (``tdem``, ``seismic``,
``model_output``, ``time_lapse``, ``data_fusion``, ``ert_data_process``,
``direct_ert``, ``custom``) and runs that type's fixed sequence. Its
classifier, once ``WorkflowOrchestratorAgent._detect_workflow_type``, lives in
``PyHydroGeophysX.agents._legacy_workflow``.

Removed agents
--------------

Three agents were removed in 0.5.0. Importing one still works, with a
``DeprecationWarning`` that names its replacement; creating one raises the
same message.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Removed
     - Use instead
   * - ``WorkflowOrchestratorAgent``
     - ``BaseAgent.run_unified_agent_workflow``, whose controller chooses each
       step of a run, or ``AgentCoordinator`` for the fixed ERT pipeline
   * - ``CodeGenerationAgent``
     - ``PyHydroGeophysX.workflows.export_workflow_bundle`` to export a
       workflow's code
   * - ``GeophysicalInversionAgent``
     - ``ERTInversionAgent``, ``SeismicAgent`` or ``TDEMAgent``, or the
       ``SRTInversion``, ``TimeLapseSRTInversion``, ``FDEMInversion`` and
       ``JointERTSRTInversion`` classes directly

3D Mesh Builder
---------------

**Purpose**: Builds and exports 3D meshes for ERT forward modeling and
inversion. It is not an agent: the **3D Mesh Builder Streamlit app**
(``python -m PyHydroGeophysX.gui_mesh3d``) and the desktop studio's 3D mesh page
call ``generate_mesh`` and ``save_outputs`` from
``PyHydroGeophysX.core.mesh_3d``, which build on ``Mesh3DCreator``. The same
functions can be called directly:

.. code-block:: python

    from pathlib import Path

    from PyHydroGeophysX.core.mesh_3d import generate_mesh, mesh_summary, save_outputs

    config = {
        "output_dir": "ert_mesh",
        "array_type": "Surface grid",
        "mesh_type": "Surface with topography",
        "nx": 10, "ny": 6, "dx": 5.0, "dy": 5.0, "x_offset": 0.0, "y_offset": 0.0,
        "topography_type": "Linear tilt", "z_base": 100.0, "tilt_x": 0.05, "tilt_y": 0.0,
        "electrode_refinement": 0.5,    # cell size at the electrodes (m)
        "boundary_refinement": 2.0,     # cell size at the domain boundary (m)
        "attractor_distance": 5.0,      # distance over which the refinement fades (m)
        "mesh_engine": "PyGIMLi prism",
        "para_depth": 30.0, "dz_fine": 0.5, "dz_coarse": 2.0, "boundary_extension": 1.4,
    }
    result = generate_mesh(config, log=print)
    files = save_outputs(result["mesh"], result["electrodes"], Path("ert_mesh"), "ert_mesh",
                         ["BMS mesh (.bms)", "VTK mesh (.vtk)", "Sensor positions (.csv)"])
    print(mesh_summary(result["mesh"]))   # cells, nodes, boundaries, dimension

**Array types** (``array_type``): ``"Surface grid"`` (``nx``, ``ny``, ``dx``,
``dy``, ``x_offset``, ``y_offset``), ``"Single borehole"`` and ``"Crosshole"``.

**Mesh types** (``mesh_type``): ``"Surface with topography"`` or ``"Box mesh"``.

**Topography types** (``topography_type``): ``"Flat"`` (``z_flat``),
``"Linear tilt"`` (``z_base``, ``tilt_x``, ``tilt_y``), ``"Gaussian hill"``
(``hill_base``, ``hill_amp``, ``hill_sigma``, ``hill_cx``, ``hill_cy``) and
``"Custom expression"`` (``topography_expr``, a NumPy expression in ``x`` and
``y``).

**Mesh engines** (``mesh_engine``): ``"Auto"``, ``"Gmsh (tetrahedral)"``,
``"PyGIMLi prism"``, ``"Structured grid"`` and ``"E4D (Triangle + TetGen)"``. Gmsh falls
back to the structured grid when it fails.

**Returns**: ``generate_mesh`` returns a dictionary with ``mesh`` (the PyGIMLi
mesh), ``electrodes`` (a DataFrame with columns ``n``, ``x``, ``y``, ``z``),
``generator`` (the mesh engine that ran) and ``zones``. ``save_outputs`` returns
the written paths under ``bms``, ``mesh_structure``, ``vtk`` and
``sensors_csv``.
