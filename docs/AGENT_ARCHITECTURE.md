# PyHydroGeophysX Multi-Agent System Architecture

**Version:** 1.0  
**Date:** November 6, 2025  
**Purpose:** Comprehensive documentation of all agents, their roles, system prompts, and relationships

---

## Table of Contents
1. [System Overview](#system-overview)
2. [Agent Hierarchy](#agent-hierarchy)
3. [Individual Agent Details](#individual-agent-details)
4. [Agent Relationships & Workflows](#agent-relationships--workflows)
5. [Data Flow Diagrams](#data-flow-diagrams)
6. [Common Workflow Patterns](#common-workflow-patterns)

---

## System Overview

PyHydroGeophysX implements a **multi-agent system** for automated geophysical workflows in subsurface hydrology. The system supports multiple LLM providers (OpenAI GPT, Google Gemini, Anthropic Claude) to automate workflows from raw geophysical data (ERT, seismic) to hydrologic parameters (water content, saturation).

### Core Design Principles
- **Specialization**: Each agent handles one specific task
- **Coordination**: AgentCoordinator orchestrates multi-agent workflows
- **Extensibility**: Easy to add new agents and methods
- **LLM Integration**: Natural language interfaces for non-experts
- **Uncertainty Quantification**: Monte Carlo methods built-in

---

## Agent Hierarchy

```
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
```

### Agent Categories

#### 1. **Input/Configuration Agents**
- `ContextInputAgent` - Parses natural language requests
- `ERTLoaderAgent` - Loads ERT field data
- `SeismicAgent` - Processes seismic data
- `ClimateDataAgent` - Fetches climate data

#### 2. **Processing/Inversion Agents**
- `ERTInversionAgent` - Performs ERT inversion (standard & time-lapse)
- `InversionEvaluationAgent` - Evaluates and optimizes inversion quality
- `DataFusionAgent` - Coordinates multi-method fusion
- `StructureConstraintAgent` - Applies seismic constraints to ERT

#### 3. **Conversion/Analysis Agents**
- `PetrophysicsAgent` - Converts resistivity to water content (layer-specific)
- `WaterContentAgent` - Converts resistivity to water content (general)

#### 4. **Output/Reporting Agents**
- `ReportAgent` - Generates comprehensive reports

---

## Individual Agent Details

### 1. ContextInputAgent

**Purpose:** Translates natural language workflow descriptions into structured configurations

**System Prompt:**
```
You are an expert in geophysical workflow design. You translate natural language 
descriptions of geophysical workflows into structured JSON configurations. You understand:
- ERT inversion parameters (regularization, mesh quality, convergence)
- Time-lapse methods (difference, ratio, joint)
- Seismic constraints and interface extraction
- Petrophysical parameters (Archie's law, porosity, saturation)
- Climate data integration for temporal analysis
```

**Inputs:**
- `user_request` (str): Natural language workflow description
- `available_data` (dict, optional): Available files/instruments

**Outputs:**
- `workflow_config` (dict): Structured configuration
- `explanation` (str): Human-readable explanation

**Key Responsibilities:**
- Parse natural language requests
- Identify workflow type (standard ERT, time-lapse, multi-method)
- Extract parameters (files, methods, thresholds)
- Generate structured JSON configuration

**Relationships:**
- **Used by:** User input → workflow execution
- **Feeds into:** `AgentCoordinator`, `DataFusionAgent`

---

### 2. ERTLoaderAgent

**Purpose:** Loads and validates ERT field data from various instruments

**System Prompt:**
```
You are an expert in electrical resistivity tomography (ERT) data processing. 
Your role is to load and validate ERT field data from various commercial instruments, 
perform quality control, and prepare data for inversion. You understand different 
data formats, coordinate systems, and common data quality issues.
```

**Inputs:**
- `data_file` (str): Path to ERT data file
- `instrument` (str): Instrument type (E4D, Syscal, ABEM, BERT)
- `project_dir` (str): Project directory
- `crs` (str): Coordinate reference system ('local' or EPSG code)
- `quality_check` (bool): Whether to perform QC

**Outputs:**
- `ert_data` (object): Loaded ERT dataset (PyGIMLi DataContainer)
- `num_electrodes` (int): Number of electrodes
- `num_measurements` (int): Number of measurements
- `quality_metrics` (dict): Data quality statistics

**Key Responsibilities:**
- Load ERT data from multiple formats
- Validate data quality (outlier detection)
- Transform coordinate systems
- Export for inversion

**Relationships:**
- **Preceded by:** `ContextInputAgent`
- **Feeds into:** `ERTInversionAgent`, `StructureConstraintAgent`

---

### 3. ERTInversionAgent

**Purpose:** Performs ERT inversion (standard or time-lapse)

**System Prompt:**
```
You are an expert in electrical resistivity tomography (ERT) inversion. Your role 
is to configure and execute ERT inversions, select appropriate regularization 
parameters, and interpret inversion results. You understand smoothness constraints, 
structural constraints, and convergence criteria.
```

**Inputs:**
- `ert_data` (object): ERT data (for standard inversion)
- `time_lapse_data` (list): List of ERT datasets (for time-lapse)
- `inversion_mode` (str): 'standard' or 'time-lapse'
- `time_lapse_method` (str): 'difference', 'ratio', or 'joint'
- `temporal_regularization` (float): Temporal smoothing weight
- `inversion_params` (dict): Lambda, max_iter, method (cgls/lsqr)
- `use_structure_constraint` (bool): Whether to use seismic structure
- `seismic_structure` (object): Optional seismic structure data

**Outputs:**
- `resistivity_model` (array): Inverted resistivity model
- `mesh` (object): PyGIMLI mesh
- `chi2_values` (list): Chi-squared fit statistics
- `coverage` (array): Model coverage/sensitivity
- `final_models` (array): Time-series models (for time-lapse)

**Key Responsibilities:**
- Configure inversion parameters
- Perform standard or time-lapse inversion
- Calculate chi-squared and convergence statistics
- Apply structural constraints (if available)

**Relationships:**
- **Preceded by:** `ERTLoaderAgent`, `StructureConstraintAgent` (optional)
- **Feeds into:** `InversionEvaluationAgent`, `PetrophysicsAgent`, `WaterContentAgent`

---

### 4. InversionEvaluationAgent

**Purpose:** Evaluates inversion quality and automatically optimizes parameters

**System Prompt:**
```
You are an expert in geophysical inversion quality assessment. Your role is to 
evaluate ERT inversion results based on data fit, model smoothness, and physical 
plausibility. You understand chi-squared statistics, L-curves, and optimal 
regularization parameter selection.
```

**Inputs:**
- `inversion_results` (dict): Results from `ERTInversionAgent`
- `ert_data` (object): Original ERT data
- `inversion_params` (dict): Current parameters
- `auto_adjust` (bool): Whether to auto-adjust parameters
- `max_attempts` (int): Maximum re-inversion attempts

**Outputs:**
- `quality_score` (float): Overall quality (0-100)
- `quality_metrics` (dict): Detailed metrics (data_fit, smoothness, physics, convergence, coverage)
- `component_scores` (dict): Individual component scores
- `recommendations` (list): Improvement suggestions
- `adjusted_params` (dict): Optimized parameters
- `final_results` (dict): Best inversion results
- `evaluation_history` (list): All attempts

**Quality Metrics:**
1. **Data Fit**: Chi-squared target (0.8-1.5 acceptable)
2. **Smoothness**: Model roughness evaluation
3. **Physical Plausibility**: Resistivity range (1-10,000 Ωm)
4. **Convergence**: Iteration stability
5. **Coverage**: Model sensitivity

**Adjustment Strategy:**
- **Underfit** (chi2 < 0.8): Reduce lambda by 50%
- **Overfit** (chi2 > 2.0): Increase lambda by 100%
- **Fine-tune**: Adjust lambda by ±20%

**Key Responsibilities:**
- Evaluate inversion quality (5 metrics)
- Determine if results are acceptable (score > 70)
- Automatically adjust regularization parameters
- Trigger re-inversion if needed (up to max_attempts)
- Track optimization history

**Relationships:**
- **Preceded by:** `ERTInversionAgent`
- **Feeds into:** Improved `ERTInversionAgent` results, `ReportAgent`

---

### 5. DataFusionAgent

**Purpose:** Intelligent coordinator for multi-method geophysical workflows

**System Prompt:**
```
You are an expert in multi-method geophysical data fusion. You understand how 
different geophysical methods complement each other and can recommend optimal 
workflows for integrating multiple datasets. You know when to apply structural 
constraints, joint inversions, and petrophysical transformations.
```

**Fusion Patterns:**

#### Pattern 1: `structure_constraint`
- **Methods:** Seismic → ERT
- **Description:** Use seismic velocity interfaces to constrain ERT inversion
- **Workflow:** `seismic_inversion` → `interface_extraction` → `constrained_ert`
- **Benefits:** Improved layer boundary resolution, reduced artifacts

#### Pattern 2: `petrophysics_integration`
- **Methods:** ERT → Petrophysics
- **Description:** Convert resistivity to hydrological properties
- **Workflow:** `ert_inversion` → `petrophysics_conversion`
- **Benefits:** Direct hydrological interpretation

#### Pattern 3: `full_integration`
- **Methods:** Seismic → ERT → Petrophysics
- **Description:** Complete geological-to-hydrological workflow
- **Workflow:** `seismic_inversion` → `interface_extraction` → `constrained_ert` → `petrophysics_conversion`
- **Benefits:** Comprehensive subsurface characterization with constraints

**Inputs:**
- `fusion_pattern` (str): Pattern name or 'auto'
- `methods` (list): Available methods
- `workflow_config` (dict): Configuration for fusion
- `data` (dict): Data for each method
- `output_dir` (str): Results directory

**Outputs:**
- `fusion_pattern` (str): Selected pattern
- `execution_plan` (list): Step-by-step plan
- `status` (str): Success/failure
- `interpretation` (str): AI interpretation of results

**Key Responsibilities:**
- Recommend fusion patterns based on available methods
- Create execution plans for multi-method workflows
- Coordinate StructureConstraintAgent + PetrophysicsAgent
- Validate method compatibility

**Relationships:**
- **Preceded by:** `ContextInputAgent`
- **Coordinates:** `SeismicAgent`, `StructureConstraintAgent`, `PetrophysicsAgent`
- **Feeds into:** `ReportAgent`

---

### 6. StructureConstraintAgent

**Purpose:** Applies seismic velocity interfaces as structural constraints to ERT inversion

**System Prompt:**
```
You are an expert in structure-constrained geophysical inversion. You understand 
how to incorporate a priori geological information from seismic data into ERT 
inversions to improve layer boundary resolution and reduce artifacts.
```

**Inputs:**
- `ert_data` (object): ERT measurement data
- `seismic_data` (object): Seismic travel time data (optional)
- `velocity_model` (array): Velocity model from seismic inversion
- `mesh` (object): PyGIMLI mesh
- `velocity_thresholds` (list): Thresholds for interface extraction (e.g., [1000, 1950] m/s)
- `mesh_quality` (int): Constrained mesh quality
- `lambda` (float): ERT regularization parameter
- `limits` (list): Resistivity bounds [min, max]
- `data_qc_threshold` (float): Data quality control threshold
- `run_comparison` (bool): Also run unconstrained for comparison

**Outputs:**
- `resistivity_model` (array): Constrained resistivity model
- `mesh` (object): Constrained mesh with layer markers
- `cell_markers` (array): Cell layer identifications
- `coverage` (array): Model coverage
- `interfaces` (list): Extracted velocity interfaces
- `statistics` (dict): Resistivity range, chi2, data fit, n_layers
- `unconstrained_results` (dict): Comparison results (if requested)

**Key Responsibilities:**
- Extract velocity interfaces from seismic model
- Create mesh with geological boundaries
- Perform structure-constrained ERT inversion
- Compare constrained vs unconstrained results

**Relationships:**
- **Preceded by:** `SeismicAgent` (for velocity model)
- **Works with:** `ERTLoaderAgent` (for ERT data)
- **Feeds into:** `PetrophysicsAgent`, `DataFusionAgent`

---

### 7. PetrophysicsAgent

**Purpose:** Converts resistivity to water content using layer-specific petrophysical models with Monte Carlo uncertainty quantification

**System Prompt:**
```
You are an expert in petrophysical modeling and hydrogeophysics. You understand 
how to convert electrical resistivity to water content using Archie's law and 
modified petrophysical relationships. You can recommend appropriate parameters 
for different geological materials and quantify uncertainties.
```

**Petrophysical Model:**
```
Archie's Law (modified with surface conductivity):
σ_bulk = σ_fluid * φ^m * S^n + σ_surface

Where:
- σ_bulk: Bulk conductivity (1/resistivity)
- σ_fluid: Fluid conductivity (1/rho_fluid)
- φ: Porosity
- S: Saturation (water content / porosity)
- m: Cementation exponent
- n: Saturation exponent
- σ_surface: Surface conductivity (clay effect)
```

**Default Layer Parameters:**

| Layer Type | Porosity (φ) | m | n | σ_surface (S/m) | ρ_fluid (Ωm) |
|-----------|-------------|---|---|----------------|--------------|
| Regolith | 0.42 ± 0.05 | 1.3 ± 0.1 | 2.1 ± 0.1 | 1/200 ± 1/200 | 20 |
| Bedrock | 0.25 ± 0.15 | 1.9 ± 0.2 | 1.7 ± 0.2 | 0.0 ± 0.0 | 20 |

**Inputs:**
- `resistivity_model` (array): Resistivity values (1D or 2D for time-lapse)
- `mesh` (object): PyGIMLI mesh
- `cell_markers` (array): Layer identifications
- `layer_params` (dict): Parameters for each layer (optional, uses defaults if not provided)
- `n_realizations` (int): Monte Carlo samples (default: 100)
- `output_dir` (str): Results directory

**Outputs:**
- `water_content_mean` (array): Mean water content per cell
- `water_content_std` (array): Standard deviation (uncertainty)
- `saturation_mean` (array): Mean saturation
- `saturation_std` (array): Saturation uncertainty
- `statistics` (dict): WC range, mean WC, mean uncertainty, n_realizations
- `layer_statistics` (dict): Statistics by geological layer

**Key Responsibilities:**
- Layer-specific petrophysical conversion
- Monte Carlo uncertainty quantification
- Handle time-lapse resistivity series
- Generate water content distributions

**Relationships:**
- **Preceded by:** `StructureConstraintAgent` or `ERTInversionAgent`
- **Requires:** Cell markers for layer identification
- **Feeds into:** `ReportAgent`

---

### 8. WaterContentAgent

**Purpose:** General resistivity to water content conversion (simpler than PetrophysicsAgent)

**System Prompt:**
```
You are an expert in petrophysical relationships and rock physics. Your role is 
to convert electrical resistivity to water content using appropriate models 
(Archie's law, Waxman-Smits), select suitable parameters for different geological 
layers, and quantify uncertainties.
```

**Inputs:**
- `inversion_results` (dict): ERT inversion results
- `petrophysical_params` (dict): Parameters for each layer
- `uncertainty_analysis` (bool): Whether to run Monte Carlo
- `n_realizations` (int): MC realizations (default: 100)
- `output_dir` (str): Results directory

**Outputs:**
- `water_content` (array): Water content estimates
- `uncertainties` (array): Uncertainty estimates (if MC enabled)
- `statistics` (dict): Summary statistics

**Key Responsibilities:**
- General petrophysical conversion
- Optional uncertainty analysis
- Simpler interface than PetrophysicsAgent

**Relationships:**
- **Alternative to:** `PetrophysicsAgent` (use PetrophysicsAgent for layer-specific, structure-constrained workflows)
- **Preceded by:** `ERTInversionAgent`
- **Feeds into:** `ReportAgent`

---

### 9. SeismicAgent

**Purpose:** Processes seismic refraction data and extracts velocity structures

**System Prompt:**
```
You are an expert in seismic refraction tomography (SRT). Your role is to process 
seismic travel time data, perform velocity inversions, and extract geological 
structure interfaces. You understand velocity-depth relationships and how to 
identify layer boundaries.
```

**Inputs:**
- `seismic_data` (object): Seismic travel time data
- `velocity_threshold` (float): Threshold for interface detection (default: 1200 m/s)
- `inversion_params` (dict): Seismic inversion parameters
  - `lam`: Regularization (default: 50)
  - `zWeight`: Depth weighting (default: 0.2)
  - `vTop`: Top velocity bound (default: 500 m/s)
  - `vBottom`: Bottom velocity bound (default: 5000 m/s)
- `output_dir` (str): Results directory

**Outputs:**
- `velocity_model` (array): Velocity distribution
- `interface_coords` (tuple): (x, z) coordinates of interface
- `mesh` (object): Seismic inversion mesh
- `statistics` (dict): Velocity range, chi2, data fit

**Key Responsibilities:**
- Seismic travel time inversion
- Extract velocity interfaces
- Identify geological boundaries
- Provide structure for ERT constraints

**Relationships:**
- **Preceded by:** `ContextInputAgent` or manual data loading
- **Feeds into:** `StructureConstraintAgent`, `DataFusionAgent`

---

### 10. ClimateDataAgent

**Purpose:** Fetches and processes climate data for temporal analysis

**System Prompt:**
```
You are an expert in climate data analysis for hydrogeophysical studies. You 
understand how precipitation, evapotranspiration, and temperature affect subsurface 
moisture and resistivity measurements.
```

**Inputs:**
- `geometry` (dict): Site coordinates (lat, lon) or bounding box
- `start_date` (str): Start date (YYYY-MM-DD)
- `end_date` (str): End date (YYYY-MM-DD)
- `variables` (list): Climate variables ['precipitation', 'temperature', 'pet']
- `source` (str): Data source (default: 'daymet')
- `output_dir` (str): Results directory

**Outputs:**
- `climate_data` (DataFrame): Time-series climate data
- `precipitation` (Series): Daily precipitation (mm)
- `temperature` (Series): Daily temperature (°C)
- `pet` (Series): Potential evapotranspiration (mm)
- `statistics` (dict): Summary statistics

**Key Responsibilities:**
- Fetch climate data from APIs (Daymet, PRISM, etc.)
- Process time-series data
- Calculate antecedent moisture indices
- Align with ERT measurement timestamps

**Relationships:**
- **Preceded by:** `ContextInputAgent`
- **Feeds into:** `ReportAgent` (for temporal interpretation)
- **Used with:** Time-lapse ERT workflows

---

### 11. ReportAgent

**Purpose:** Generates comprehensive reports from workflow results

**System Prompt:**
```
You are an expert in technical report writing for geophysical and hydrological 
studies. Your role is to synthesize results from ERT data processing, inversion, 
water content analysis, and climate data into clear, informative reports suitable 
for scientists and engineers. You should integrate climate insights (precipitation, 
PET, temperature) to explain resistivity changes and provide data quality caveats.
```

**Inputs:**
- `workflow_data` (dict): All data from workflow steps
- `config` (dict): Original workflow configuration
- `output_dir` (str): Report output directory

**Outputs:**
- `report_path` (str): Path to generated report
- `figures` (list): Generated figure paths
- `summary_stats` (dict): Key statistics

**Report Sections:**
1. **Executive Summary**: High-level findings
2. **Data Processing Summary**: Data loading and QC
3. **Climate Data Summary**: Precipitation, temperature, PET trends (if available)
4. **Inversion Results**: Resistivity models, convergence, chi-squared
5. **Water Content Analysis**: Hydrological interpretation
6. **Climate-Resistivity Analysis**: Cross-modal temporal patterns (if climate data available)
7. **Quality Assessment**: Data quality, inversion quality, uncertainties
8. **Conclusions & Recommendations**: Key findings and next steps

**Key Responsibilities:**
- Aggregate results from all agents
- Generate visualizations
- Write technical summaries
- Provide AI interpretations (if LLM available)

**Relationships:**
- **Preceded by:** All other agents (final step)
- **Uses:** All workflow outputs

---

### 12. AgentCoordinator

**Purpose:** Orchestrates multi-agent workflows and manages execution state

**Not a processing agent** - instead, it's the **orchestration layer** that:
- Registers agents
- Manages workflow state
- Coordinates agent execution
- Handles data flow between agents
- Logs execution history

**Key Methods:**
- `register_agent(name, instance)`: Add agent to workflow
- `execute_workflow(config)`: Run complete workflow
- `get_workflow_state()`: Get current state
- `save_workflow_results()`: Persist results

**Typical Workflow Execution:**
```python
coordinator = AgentCoordinator(api_key, output_dir)

# Register agents
coordinator.register_agent('context', ContextInputAgent())
coordinator.register_agent('ert_loader', ERTLoaderAgent())
coordinator.register_agent('ert_inversion', ERTInversionAgent())
coordinator.register_agent('water_content', WaterContentAgent())
coordinator.register_agent('report', ReportAgent())

# Execute workflow
results = coordinator.execute_workflow(config)
```

**Workflow State Tracking:**
```python
{
    'status': 'initialized' | 'running' | 'completed' | 'failed',
    'current_step': 'agent_name',
    'completed_steps': ['agent1', 'agent2', ...],
    'data': {
        'agent1': {...},
        'agent2': {...}
    }
}
```

---

## Agent Relationships & Workflows

### Workflow 1: Standard ERT Workflow

```
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
```

**Data Flow:**
1. User: "I need to process ERT data from file X.ohm"
2. Context: Parse → `{data_file: X.ohm, instrument: E4D, inversion_params: {...}}`
3. Loader: Load data → `ert_data` object
4. Inversion: Invert → `resistivity_model`, `mesh`, `chi2`
5. Evaluation: Check quality → adjust parameters if needed → re-invert if quality < 70
6. Water Content: Convert → `water_content`, `uncertainties`
7. Report: Generate → PDF/HTML report with figures

---

### Workflow 2: Time-Lapse ERT Workflow

```
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
```

**Key Differences from Standard:**
- Multiple `ert_data` objects loaded
- Time-lapse specific inversion methods
- Climate data for temporal interpretation
- Temporal regularization parameter

---

### Workflow 3: Structure-Constrained Multi-Method Fusion

```
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
```

**Execution Plan (from DataFusionAgent):**
```json
{
  "fusion_pattern": "full_integration",
  "execution_plan": [
    {
      "step": "seismic_inversion",
      "agent": "SeismicAgent",
      "description": "Invert seismic travel times",
      "outputs": ["velocity_model"]
    },
    {
      "step": "interface_extraction",
      "agent": "StructureConstraintAgent",
      "description": "Extract velocity interfaces at [1000, 1950] m/s",
      "outputs": ["interface_coords", "cell_markers"]
    },
    {
      "step": "constrained_ert",
      "agent": "StructureConstraintAgent",
      "description": "ERT inversion with seismic constraints",
      "outputs": ["resistivity_model", "mesh"]
    },
    {
      "step": "petrophysics_conversion",
      "agent": "PetrophysicsAgent",
      "description": "Layer-specific resistivity → water content",
      "outputs": ["water_content_mean", "water_content_std"]
    }
  ]
}
```

---

## Data Flow Diagrams

### Flow 1: Natural Language to Structured Workflow

```
┌──────────────────────────────────────────────────────────────┐
│ User Natural Language Request                                │
│ "I need to analyze ERT data with seismic constraints         │
│  and convert to water content"                               │
└────────────────────┬─────────────────────────────────────────┘
                     │
                     ▼
          ┌──────────────────────┐
          │ ContextInputAgent    │
          │ (LLM Processing)     │
          └──────────┬───────────┘
                     │
                     ▼
          ┌──────────────────────┐
          │ Structured Config    │
          │ {                    │
          │   fusion_pattern:    │
          │     "full_integration│
          │   methods: [         │
          │     "seismic", "ert" │
          │   ],                 │
          │   velocity_thresholds│
          │   layer_params: {...}│
          │ }                    │
          └──────────┬───────────┘
                     │
                     ▼
          ┌──────────────────────┐
          │ DataFusionAgent      │
          │ (Create Plan)        │
          └──────────┬───────────┘
                     │
                     ▼
          ┌──────────────────────┐
          │ Execution Plan       │
          │ Step 1: Seismic      │
          │ Step 2: Interface    │
          │ Step 3: ERT          │
          │ Step 4: Petrophysics │
          └──────────────────────┘
```

---

### Flow 2: Structure-Constrained ERT Data Path

```
Seismic Data          ERT Data
    │                     │
    ▼                     ▼
┌──────────┐      ┌──────────────┐
│ Seismic  │      │ ERT Loader   │
│ Agent    │      │ Agent        │
└────┬─────┘      └──────┬───────┘
     │                   │
     ▼                   │
velocity_model          ert_data
     │                   │
     └───────┬───────────┘
             │
             ▼
    ┌────────────────────┐
    │ Structure          │
    │ Constraint Agent   │
    └────────┬───────────┘
             │
             ├─ Extract interfaces (1000, 1950 m/s)
             ├─ Create mesh with boundaries
             ├─ Mark cells by layer (1, 2, 3)
             └─ Constrained ERT inversion
             │
             ▼
    ┌────────────────────┐
    │ Results:           │
    │ - resistivity      │
    │ - mesh             │
    │ - cell_markers     │
    │ - 3 geological     │
    │   units            │
    └────────┬───────────┘
             │
             ▼
    ┌────────────────────┐
    │ Petrophysics       │
    │ Agent              │
    └────────┬───────────┘
             │
             ├─ Layer 1 params (regolith)
             ├─ Layer 2 params (fractured bedrock)
             └─ Layer 3 params (fresh bedrock)
             │
             ▼
    water_content_mean ± std
```

---

### Flow 3: Inversion Quality Optimization Loop

```
Initial ERT Inversion
   (lambda = 20)
         │
         ▼
┌─────────────────────────┐
│ InversionEvaluation     │
│ Agent                   │
│                         │
│ Calculate:              │
│ - Chi-squared: 2.5      │
│ - Smoothness: OK        │
│ - Physics: OK           │
│ - Coverage: OK          │
│                         │
│ Quality Score: 45/100   │
│ Status: OVERFIT         │
└──────────┬──────────────┘
           │
           ▼
    ┌──────────────┐
    │ Adjust       │
    │ Parameters   │
    │              │
    │ chi2 > 2.0   │
    │ → Increase   │
    │   lambda     │
    │ λ = 20 * 2.0 │
    │ λ = 40       │
    └──────┬───────┘
           │
           ▼
Re-run ERT Inversion
   (lambda = 40)
         │
         ▼
┌─────────────────────────┐
│ InversionEvaluation     │
│ Agent                   │
│                         │
│ Calculate:              │
│ - Chi-squared: 1.1      │
│ - Smoothness: Good      │
│ - Physics: Good         │
│ - Coverage: Good        │
│                         │
│ Quality Score: 82/100   │
│ Status: ACCEPTABLE      │
└──────────┬──────────────┘
           │
           ▼
    Accept Results
    (Return best model)
```

---

### Flow 4: Monte Carlo Uncertainty Propagation

```
Layer-Specific Parameters
    │
    ├─ Layer 1 (Regolith)
    │   ├─ porosity: 0.50 ± 0.05
    │   ├─ n: 2.2 ± 0.1
    │   ├─ m: 1.5 ± 0.15
    │   └─ σ_sur: 1/400 ± 1/800
    │
    ├─ Layer 2 (Fractured Bedrock)
    │   ├─ porosity: 0.25 ± 0.05
    │   └─ (...)
    │
    └─ Layer 3 (Fresh Bedrock)
        └─ (...)
         │
         ▼
┌────────────────────────────┐
│ PetrophysicsAgent          │
│                            │
│ For each cell:             │
│   For i = 1 to 10,000:     │
│     Sample params from     │
│     distributions          │
│     Calculate WC(i)        │
│                            │
│ Aggregate:                 │
│   WC_mean = mean(WC)       │
│   WC_std = std(WC)         │
└──────────┬─────────────────┘
           │
           ▼
    ┌──────────────────┐
    │ Output:          │
    │ - water_content  │
    │   (mean)         │
    │ - uncertainty    │
    │   (std)          │
    │ - 95% CI         │
    └──────────────────┘
```

---

## Common Workflow Patterns

### Pattern A: Basic ERT Processing
**Use Case:** Single ERT dataset, no constraints  
**Agents:** Context → Loader → Inversion → Evaluation → WaterContent → Report  
**Timeline:** ~5-10 minutes

---

### Pattern B: Time-Lapse Monitoring
**Use Case:** Multiple ERT datasets over time  
**Agents:** Context → Loader (×N) → Climate → Inversion (time-lapse) → Evaluation → WaterContent → Report  
**Timeline:** ~15-30 minutes  
**Special Features:**
- Temporal regularization
- Climate data integration
- Difference/ratio/joint methods

---

### Pattern C: Structure-Constrained Fusion
**Use Case:** Seismic + ERT with layer identification  
**Agents:** Context → DataFusion → Seismic → Structure → Petrophysics → Report  
**Timeline:** ~20-40 minutes  
**Special Features:**
- Interface extraction
- Layer-specific parameters
- Monte Carlo uncertainty (10,000 realizations)

---

### Pattern D: Full Field-Scale Analysis
**Use Case:** Complete hydrogeophysical characterization  
**Agents:** Context → DataFusion → Seismic → ERTLoader → Structure → Evaluation → Petrophysics → Report  
**Timeline:** ~30-60 minutes  
**Special Features:**
- Multi-method integration
- Quality optimization
- Comprehensive uncertainty quantification
- Publication-ready reports

---

## Agent Communication Protocol

### Input/Output Standards

All agents follow this I/O pattern:

```python
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
```

### Standard Output Keys

All agents include:
- `status` (str): 'success', 'failed', 'needs_improvement'
- `error` (str): Error message if failed
- `interpretation` (str): AI interpretation (if LLM available)

Agent-specific outputs vary (see individual agent sections).

---

## System Prompts Summary Table

| Agent | Core Expertise | Key Responsibilities |
|-------|----------------|---------------------|
| **ContextInputAgent** | Workflow design | Parse NL → JSON config |
| **ERTLoaderAgent** | Data processing | Load ERT, QC, validate |
| **ERTInversionAgent** | Inversion theory | Standard & time-lapse inversion |
| **InversionEvaluationAgent** | Quality assessment | Evaluate, optimize, re-run |
| **DataFusionAgent** | Multi-method integration | Recommend patterns, coordinate |
| **StructureConstraintAgent** | Constrained inversion | Extract interfaces, apply constraints |
| **PetrophysicsAgent** | Petrophysical modeling | Layer-specific WC + MC uncertainty |
| **WaterContentAgent** | Rock physics | General WC conversion |
| **SeismicAgent** | Seismic tomography | SRT inversion, interface extraction |
| **ClimateDataAgent** | Climate analysis | Fetch climate, temporal alignment |
| **ReportAgent** | Technical writing | Synthesize results, generate reports |

---

## Extension Points for New Agents

To add a new agent:

1. **Inherit from BaseAgent**
```python
from .base_agent import BaseAgent

class NewMethodAgent(BaseAgent):
    def __init__(self, api_key, model, llm_provider):
        super().__init__("new_method", api_key, model, llm_provider)
        self.system_message = "Your expert role description"
```

2. **Implement execute() method**
```python
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        # Your processing logic
        return {
            'status': 'success',
            'output_key': output_value
        }
```

3. **Register in `__init__.py`**
```python
from .new_method_agent import NewMethodAgent

__all__ = [..., 'NewMethodAgent']
```

4. **Update DataFusionAgent patterns** (if multi-method)
```python
FUSION_PATTERNS = {
    'new_pattern': {
        'methods': ['method1', 'new_method'],
        'workflow': ['step1', 'step2'],
        ...
    }
}
```

---

## References

- **Code Location:** `PyHydroGeophysX/agents/`
- **Example Notebooks:** `examples/Ex_*_NaturalLanguage.ipynb`
- **Documentation:** `PyHydroGeophysX/agents/README.md`
- **Guide:** `PyHydroGeophysX/agents/DATA_FUSION_README.md`

---

**End of Document**

*For visualization, consider using this structure to create:*
- *Agent hierarchy tree diagram*
- *Data flow network diagram*
- *Workflow sequence diagrams*
- *Agent relationship graph*
