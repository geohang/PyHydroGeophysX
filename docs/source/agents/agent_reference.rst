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

        def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
            raise NotImplementedError

AgentCoordinator
----------------

**Purpose**: Orchestrates multi-agent workflows and manages execution state.

This is not a processing agent but an orchestration layer that:

* Registers agents
* Manages workflow state
* Coordinates agent execution
* Handles data flow between agents
* Logs execution history

**Key Methods**:

.. code-block:: python

    register_agent(name, instance)  # Add agent to workflow
    execute_workflow(config)        # Run complete workflow
    get_workflow_state()           # Get current state
    save_workflow_results()        # Persist results

ContextInputAgent
-----------------

**Purpose**: Translates natural language workflow descriptions into structured configurations.

**System Prompt**:
    You are an expert in geophysical workflow design. You translate natural language
    descriptions of geophysical workflows into structured JSON configurations.

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

* ``seismic_data`` (object): Seismic travel time data
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

**Purpose**: Fetches and processes climate data for temporal analysis.

**System Prompt**:
    You are an expert in climate data analysis for hydrogeophysical studies. You
    understand how precipitation, evapotranspiration, and temperature affect subsurface
    moisture and resistivity measurements.

**Inputs**:

* ``geometry`` (dict): Site coordinates (lat, lon)
* ``start_date`` (str): Start date (YYYY-MM-DD)
* ``end_date`` (str): End date (YYYY-MM-DD)
* ``variables`` (list): Climate variables
* ``source`` (str): Data source (default: 'daymet')

**Outputs**:

* ``climate_data`` (DataFrame): Time-series climate data
* ``precipitation`` (Series): Daily precipitation (mm)
* ``temperature`` (Series): Daily temperature (C)
* ``pet`` (Series): Potential evapotranspiration (mm)
* ``statistics`` (dict): Summary statistics

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
