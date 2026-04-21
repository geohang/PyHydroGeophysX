[![tests](https://github.com/geohang/PyHydroGeophysX/actions/workflows/tests.yml/badge.svg)](https://github.com/geohang/PyHydroGeophysX/actions/workflows/tests.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17025139.svg)](https://doi.org/10.5281/zenodo.17025139)

<div align="center">
  <img src="logo.png" alt="PyHydroGeophysX Logo" width="400">
</div>

# PyHydroGeophysX

A comprehensive Python package for integrating hydrological model outputs with geophysical forward modeling and inversion, specializing in electrical resistivity tomography (ERT) and seismic refraction tomography (SRT) for watershed monitoring applications.

<div align="center">
  <img src="frame.png" alt="HydroGeophysX Framework" width="600">
</div>

*HydroGeophysX bridges the strengths and limitations of hydrologic modeling and geophysical monitoring, enabling process-based survey design, inversion constraints, and model calibration. This integrative approach supports advanced watershed studies and critical zone science.*

## Quick Links

- Documentation: https://geohang.github.io/PyHydroGeophysX/
- Examples gallery: https://geohang.github.io/PyHydroGeophysX/auto_examples/index.html
- Agent web app: https://pyhydrogeophysx.streamlit.app/

## 🌟 Key Features

- 🌊 **Hydrological Model Integration:** Seamless loading and processing of MODFLOW and ParFlow outputs  
- 📊 **ERT Data Processing:** Standardized loading, quality control, and export of ERT field data with RESIPY integration  
- 🤖 **Multi-Agent AI System:** Automatic cross-modal geophysics agent supporting multiple LLM APIs (GPT, Gemini, Claude) for automated workflows processing ERT, seismic, and other geophysical data into hydrologic information
- 🧊 **3D ERT Modeling:** Complete 3D mesh creation, forward modeling, and visualization with topography and MODFLOW integration **NEW**
- 📡 **TDEM Forward & Inversion:** Time-Domain Electromagnetic (TDEM) forward modeling and inversion using SimPEG **NEW**
- 📶 **FDEM Forward & Inversion:** Frequency-Domain EM modeling and inversion for 1D layered Earth using SimPEG **NEW**
- 🧭 **SRT Inversion Module:** Dedicated SRT inversion class with log-slowness Gauss-Newton optimization **NEW**
- 🔗 **Joint ERT+SRT Inversion:** Structure-constrained and geostatistical joint inversion workflows **NEW**
- 🧱 **Cross-Constraint Utilities:** Cross-gradient and structural constraint builders for cooperative inversion **NEW**
- 🧩 **Unified Inversion Interface:** Single factory API for ERT/SRT/TDEM/FDEM/joint inversion dispatch **NEW**
- 🪨 **Petrophysical Relationships:** Advanced models for converting between water content, saturation, resistivity, and seismic velocity  
- ⚡ **Forward Modeling:** Complete ERT, SRT, TDEM, and FDEM forward modeling capabilities with synthetic data generation  
- 🔄 **Time-Lapse Inversion:** Sophisticated algorithms for time-lapse ERT inversion with temporal regularization  
- 🏔️ **Structure-Constrained Inversion:** Integration of seismic velocity interfaces for constrained ERT inversion  
- 🔬 **Uncertainty Quantification:** Monte Carlo methods for parameter uncertainty assessment  
- 🚀 **High Performance:** GPU acceleration support (CUDA/CuPy) and parallel processing capabilities  
- 💡 **Advanced Solvers:** Multiple linear solvers (CGLS, LSQR, RRLS) with optional GPU acceleration

## Key Workflows

- Field ERT ingestion, QC, visualization, and export for inversion (RESIPY-compatible).
- HM -> GM: MODFLOW/ParFlow outputs -> petrophysics -> 2D/3D forward responses for ERT/SRT/TDEM, including time-lapse monitoring and survey sensitivity.
- GM -> HM: single-time, time-lapse, and windowed inversions with temporal regularization; extract seismic interfaces and velocity structure for constrained meshes and hydrologic parameterization.
- Iterative ModEx loop: use model results to guide survey design and use geophysical inversions to calibrate and parameterize hydrologic models.
- Multi-physics and uncertainty workflows: combine ERT/SRT/EM and Monte Carlo to quantify parameter uncertainty.

## New Inversion Interfaces (v0.2)

```python
from PyHydroGeophysX.inversion import GeophysicalInversion, JointERTSRTInversion

# Unified method dispatch: "ert", "srt", "tdem", "fdem", "joint_ert_srt"
inv = GeophysicalInversion("srt", data_file="path/to/survey.sgt")
srt_result = inv.run()

# Joint ERT + SRT inversion with structure/geostatistical options
joint = JointERTSRTInversion(
    ert_data="path/to/ert.dat",
    srt_data="path/to/srt.sgt",
    regularization_mode="geostat",
    cross_gradient_mode="direct",
)
joint_result = joint.run()
```

## 📋 Requirements

- Python 3.8 or higher
- Core: NumPy, SciPy, Matplotlib, tqdm
- Optional geophysics engines: PyGIMLi, SimPEG, RESIPY, FloPy
- Optional acceleration: CuPy, joblib

**Tested version ranges:** PyGIMLi ≥ 1.5.5 (<2.0), SimPEG ≥ 0.24, NumPy ≥ 1.21, SciPy ≥ 1.8, Matplotlib ≥ 3.5. Full pins are specified in `setup.py`, `requirements.txt`, and `environment.yml`.

## 🛠️ Installation
### From PyPI (Recommended)

```bash
pip install pyhydrogeophysx
```

Install geophysics features (recommended for inversion/forward workflows):

```bash
pip install "pyhydrogeophysx[geophysics]"
```

Install geophysics + agents:

```bash
pip install "pyhydrogeophysx[geophysics,agents]"
```

### From Source

```bash
git clone https://github.com/geohang/PyHydroGeophysX.git
cd PyHydroGeophysX
pip install -e .
```

For GPU support (optional):

```bash
pip install cupy-cuda11x  # Replace with your CUDA version
```




## 📚 Documentation

Comprehensive documentation is available at https://geohang.github.io/PyHydroGeophysX/.

To build documentation locally:

```bash
cd docs
make html  # On Windows: .\\make.bat html
```

## 🗂️ Package Structure

```
PyHydroGeophysX/
├── core/               # Core utilities
│   ├── interpolation.py    # Profile interpolation tools
│   ├── mesh_utils.py       # 2D mesh creation and manipulation
│   └── mesh_3d.py          # 3D mesh creation with GMSH/PyGIMLi **NEW**
├── agents/             # Multi-agent AI system **NEW**
│   ├── agent_coordinator.py    # Workflow orchestration
│   ├── ert_loader_agent.py     # ERT data loading agent
│   ├── ert_inversion_agent.py  # Inversion agent
│   ├── water_content_agent.py  # Water content conversion agent
│   ├── seismic_agent.py        # Seismic processing agent
│   ├── climate_data_agent.py   # Climate data and PET agent **NEW**
│   └── report_agent.py         # Report generation agent
├── data_processing/    # Geophysical data processing
│   └── ert_data_agent.py   # ERT data loading, QC, and export
├── model_output/       # Hydrological model interfaces
│   ├── modflow_output.py   # MODFLOW data loading
│   └── parflow_output.py   # ParFlow data loading
├── petrophysics/       # Rock physics models
│   ├── resistivity_models.py  # Waxman-Smits, Archie models
│   └── velocity_models.py     # DEM, Hertz-Mindlin models
├── forward/            # Forward modeling
│   ├── ert_forward.py      # 2D/3D ERT forward modeling
│   ├── srt_forward.py      # Seismic forward modeling
│   ├── tdem_forward.py     # TDEM forward modeling
│   └── fdem_forward.py     # FDEM forward modeling **NEW**
├── inversion/          # Inverse modeling
│   ├── ert_inversion.py    # Single-time ERT inversion
│   ├── srt_inversion.py    # Single-time SRT inversion **NEW**
│   ├── srt_time_lapse.py   # Time-lapse SRT inversion **NEW**
│   ├── time_lapse.py       # Time-lapse inversion
│   ├── windowed.py         # Windowed time-lapse for large datasets
│   ├── tdem_inversion.py   # TDEM inversion with SimPEG
│   ├── fdem_inversion.py   # FDEM inversion with SimPEG **NEW**
│   ├── joint_ert_srt.py    # Joint ERT+SRT inversion **NEW**
│   ├── cross_constraints.py # Cross-gradient/structural constraints **NEW**
│   └── multi_method.py     # Unified inversion interface **NEW**
├── solvers/            # Linear algebra solvers
│   └── linear_solvers.py   # CGLS, LSQR, RRLS with GPU support
├── Hydro_modular/      # Direct hydro-to-geophysics conversion
└── Geophy_modular/     # Geophysical data processing tools
```

## 📖 Examples

The examples folder provides paired Jupyter notebooks (.ipynb) and Python scripts (.py) for each workflow. Data used by examples is under examples/data, with outputs written to examples/results.

- **Ex_ERT_data_process**: Loading, quality control, and export of field ERT data with RESIPY integration (notebook: Ex_ERT_data_process.ipynb).
- **Ex_multi_agent_workflow**: Automated multi-agent workflow for ERT processing with optional seismic integration (script: Ex_multi_agent_workflow.py). **NEW**
- Ex_model_output: Loading and processing hydrological model outputs (MODFLOW/ParFlow) (notebook: Ex_model_output.ipynb, script: Ex_model_output.py).
- Ex_ERT_workflow: End‑to‑end ERT modeling and inversion workflow (notebook: Ex_ERT_workflow.ipynb, script: Ex_ERT_workflow.py).
- Ex_Time_lapse_measurement: Generate synthetic time‑lapse ERT measurements and schedules (notebook: Ex_Time_lapse_measurement.ipynb, script: Ex_Time_lapse_measurement.py).
- Ex_TL_inversion: Time‑lapse ERT inversion with temporal regularization and windowed processing (notebook: Ex_TL_inversion.ipynb, script: Ex_TL_inversion.py).
- Ex_Structure_resinv: Structure‑constrained resistivity inversion using seismic interfaces (notebook: Ex_Structure_resinv.ipynb, script: Ex_Structure_resinv.py).
- Ex_structure_TLresinv: Structure‑constrained time‑lapse resistivity inversion (notebook: Ex_structure_TLresinv.ipynb, script: Ex_structure_TLresinv.py).
- EX_SRT_forward: Seismic refraction tomography forward modeling and synthetic travel times (notebook: EX_SRT_forward.ipynb, script: EX_SRT_forward.py).
- Ex_SRT_inv: Seismic refraction tomography inversion workflow (notebook: Ex_SRT_inv.ipynb, script: Ex_SRT_inv.py).
- Ex_SRT_inv now includes both PyGIMLi direct inversion and packaged `SRTInversion` side-by-side comparison (notebook: Ex_SRT_inv.ipynb, script: Ex_SRT_inv.py).
- Ex_MC_Hydro: Monte Carlo uncertainty quantification for hydro‑to‑resistivity conversion (notebook: Ex_MC_Hydro.ipynb, script: Ex_MC_Hydro.py).
- **Ex_3D_ERT_forward**: 3D ERT forward modeling with MODFLOW integration using Mesh3DCreator and PyVista visualization (notebook: Ex_3D_ERT_forward.ipynb). **NEW**
- **Ex_TDEM_workflow**: Time-Domain Electromagnetic (TDEM) forward modeling and inversion from hydrological models using SimPEG (notebook: Ex_TDEM_workflow.ipynb). **NEW**
- **Ex_FDEM_workflow**: Frequency-domain EM forward modeling and inversion (script: Ex_FDEM_workflow.py). **NEW**
- **Ex_cross_constraints**: Structural and cross-gradient constraints across methods (script: Ex_cross_constraints.py). **NEW**
- **Ex_hydro_to_multigeophys**: Hydro -> petrophysics -> multi-method forward/inversion workflow (script: Ex_hydro_to_multigeophys.py). **NEW**


## 🚀 Quick Start

## 0. Multi-Agent AI Workflow (NEW)

Automatic cross-modal geophysics agent for subsurface hydrology. Automate geophysical data processing workflows (ERT, seismic, and more) using AI agents with support for multiple LLM APIs (GPT, Gemini, Claude):

```python
from PyHydroGeophysX.agents import (
    AgentCoordinator, ERTLoaderAgent, ERTInversionAgent,
    WaterContentAgent, ReportAgent, SeismicAgent, ClimateDataAgent
)

# Initialize coordinator with your LLM API key (supports OpenAI GPT, Google Gemini, Anthropic Claude)
coordinator = AgentCoordinator(api_key='your-api-key', llm_provider='openai')  # or 'gemini', 'claude'

# Register specialized agents
coordinator.register_agent('ert_loader', ERTLoaderAgent())
coordinator.register_agent('ert_inversion', ERTInversionAgent())
coordinator.register_agent('water_content', WaterContentAgent())
coordinator.register_agent('report', ReportAgent())

# Optional: Add seismic agent for structure-constrained inversion
coordinator.register_agent('seismic_processor', SeismicAgent())

# Optional: Add climate agent for meteorological data integration **NEW**
coordinator.register_agent('climate_data', ClimateDataAgent())

# Configure and execute workflow
config = {
    'data_file': 'data/ERT/survey.dat',
    'instrument': 'E4D',
    'inversion_params': {'lambda': 20.0, 'max_iterations': 10},
    'run_uncertainty': True,
    'n_realizations': 100,
    # Optional seismic integration
    'use_seismic': True,
    'seismic_data': travel_time_data,
    'velocity_threshold': 1200,
    # Optional climate data integration **NEW**
    'use_climate': True,
    'climate_config': {
        'coords': (-105.3, 40.0),  # Site location (lon, lat)
        'dates': ('2023-06-01', '2023-09-30'),
        'pet_method': 'penman_monteith',
        'antecedent_days': [1, 3, 7]
    },
    'ert_timestamps': ['2023-06-15', '2023-07-15', '2023-08-15']
}

# Run complete workflow
results = coordinator.execute_workflow(config)

# Access results
if results['status'] == 'success':
    print(f"Report: {results['results']['report']['report_file']}")
    print(f"Water content: {results['results']['water_content']['water_content_mean']}")
    if 'climate_data' in results['results']:
        print(f"Climate data: {results['results']['climate_data']['metadata']}")
```

**Key Features:**
- 🤖 AI-powered parameter selection and interpretation with multiple LLM API support (GPT, Gemini, Claude)
- 🔄 Fully automated workflow execution for cross-modal geophysical data
- 📊 Automatic quality control and uncertainty quantification
- 📝 Comprehensive report generation with visualizations
- 🌊 Cross-modal integration (ERT, seismic, and other geophysical methods)
- ☁️ Climate data integration with PET computation for hydrologic context **NEW**

See `examples/Ex_multi_agent_workflow.py` and `examples/Ex_climate_ert_integration.py` for complete examples.

## 1. ERT Field Data Processing

Load, quality control, and export field ERT data with RESIPY integration:

```python
from PyHydroGeophysX.data_processing.ert_data_agent import (
    load_ert_resipy, qc_and_visualize, export_for_inversion, LocalRef
)

# Load ERT field data from various instruments (E4D, Syscal, ABEM, etc.)
ert = load_ert_resipy(
    project_dir="data/ERT/E4D",
    data_file="data/ERT/E4D/2021-10-08_1400.ohm",
    instrument="E4D",
    crs="local",
    local_ref=LocalRef(origin_x=0.0, origin_y=0.0, azimuth_deg=90.0)
)

# Run quality control and generate diagnostic plots
artifacts = qc_and_visualize(ert, outdir="results/ert_data_process")
# Generates: rhoa_hist.png, pseudosection.png, data_summary.json

# Export to pyGIMLi/BERT format for inversion
bert_path = export_for_inversion(ert, outdir="results/ert_data_process", fmt="pgimli")
# Creates: bert_data.dat (with electrode coordinates and measurements)
```

## 1.5 Climate Data Integration (NEW)

Fetch meteorological data and compute potential evapotranspiration (PET) for integration with ERT analysis:

```python
from PyHydroGeophysX.agents import ClimateDataAgent

# Initialize climate agent
climate_agent = ClimateDataAgent()

# Fetch climate data for ERT site
climate_config = {
    'coords': (-105.3, 40.0),  # Site location (lon, lat)
    'dates': ('2023-06-01', '2023-09-30'),  # Campaign period
    'crs': 4326,
    'variables': ['prcp', 'tmin', 'tmax', 'srad', 'vp', 'dayl'],
    'pet_method': 'penman_monteith',  # or 'priestley_taylor', 'hargreaves_samani'
    'pet_params': {
        'arid_correction': True,  # For arid regions
        'soil_heat_flux': 0,
        'albedo': 0.23
    },
    'time_scale': 'daily',
    'region': 'na',
    'ert_timestamps': ['2023-06-15', '2023-07-15', '2023-08-15'],
    'antecedent_days': [1, 3, 7]  # Compute antecedent precipitation
}

# Execute data retrieval
results = climate_agent.execute(climate_config)

# Access climate data
climate_data = results['climate_data']  # Daily climate variables + PET
derived_features = results['derived_features']  # Antecedent totals, P-PET
ert_aligned = results['ert_alignment']  # Climate data at ERT timestamps

# Get summary
print(climate_agent.get_climate_summary())
```

**Climate Data Features:**
- 🌡️ Daily meteorological variables (prcp, tmin, tmax, srad, vp, dayl)
- 💧 PET computation with multiple methods (Penman-Monteith, Priestley-Taylor, Hargreaves-Samani)
- 📊 Derived features: antecedent precipitation totals, P-PET water balance
- 🎯 Temporal alignment with ERT acquisition timestamps
- 🔬 PET method comparison for robustness assessment
- 🏜️ Arid region corrections for improved estimates

See `examples/Ex_climate_ert_integration.py` for complete examples.

## 2. Hydrological Model Integration

Load and process outputs from various hydrological models:

```python
# MODFLOW
from PyHydroGeophysX import MODFLOWWaterContent, MODFLOWPorosity

processor = MODFLOWWaterContent("sim_workspace", idomain)
water_content = processor.load_time_range(start_idx=0, end_idx=10)

# ParFlow
from PyHydroGeophysX import ParflowSaturation, ParflowPorosity

saturation_proc = ParflowSaturation("model_dir", "run_name")
saturation = saturation_proc.load_timestep(100)
```

## 3. Petrophysical Modeling

Convert between hydrological and geophysical properties:

```python
from PyHydroGeophysX.petrophysics import (
    water_content_to_resistivity,
    HertzMindlinModel,
    DEMModel
)

# Water content to resistivity (Waxman-Smits model)
resistivity = water_content_to_resistivity(
    water_content=wc, rhos=100, n=2.2, porosity=0.3, sigma_sur=0.002
)

# Water content to seismic velocity (rock physics models)
hm_model = HertzMindlinModel()
vp_high, vp_low = hm_model.calculate_velocity(
    porosity=porosity, saturation=saturation,
    bulk_modulus=30.0, shear_modulus=20.0, mineral_density=2650
)
```

## 4. Forward Modeling

Generate synthetic geophysical data:

```python
from PyHydroGeophysX.forward import ERTForwardModeling, SeismicForwardModeling

# ERT forward modeling
ert_fwd = ERTForwardModeling(mesh, data)
synthetic_data = ert_fwd.create_synthetic_data(
    xpos=electrode_positions, res_models=resistivity_model
)

# Seismic forward modeling
srt_fwd = SeismicForwardModeling(mesh, scheme)
travel_times = srt_fwd.create_synthetic_data(
    sensor_x=geophone_positions, velocity_model=velocity_model
)
```

## 4b. 3D ERT Forward Modeling (NEW)

Create complete 3D meshes and forward models with MODFLOW integration:

```python
from PyHydroGeophysX.core.mesh_3d import Mesh3DCreator
import pygimli as pg

# Define domain and electrode array
domain_x, domain_y, domain_z = 20.0, 20.0, 14.0  # meters
nx_elec, ny_elec = 5, 5  # 5x5 electrode grid
spacing = 4.0  # 4m between electrodes

# Create mesh with electrodes
mesh_creator = Mesh3DCreator(domain_x, domain_y, domain_z, 
                             max_element_size=2.0, quality=1.2)
electrodes = mesh_creator.create_electrode_grid(
    nx=nx_elec, ny=ny_elec, spacing=spacing, z_offset=0.0
)
mesh = mesh_creator.create_3d_mesh_with_topography(
    electrodes=electrodes, topography=None
)

# Load MODFLOW data and interpolate to mesh
from PyHydroGeophysX.model_output.modflow_output import MODFLOWWaterContent
wc_processor = MODFLOWWaterContent(model_dir, idomain, cell_size=1.0)
water_content = wc_processor.load_timestep(0)

# Forward model with PyGIMLi
from PyHydroGeophysX.forward.ert_forward import ERTForwardModeling
ert_fwd = ERTForwardModeling(mesh, data)
synthetic_data = ert_fwd.create_synthetic_3d_data(
    electrodes=electrodes, res_model=resistivity_model
)
```

## 4c. TDEM Forward Modeling (NEW)

Time-Domain Electromagnetic forward modeling using SimPEG:

```python
from PyHydroGeophysX.forward.tdem_forward import TDEMForwardModeling, TDEMSurveyConfig
import numpy as np

# Define survey configuration
config = TDEMSurveyConfig(
    source_type='VMD',              # Vertical magnetic dipole
    source_location=[0, 0, 1],      # Source at 1m height
    source_radius=5.0,              # 5m loop radius
    receiver_location=[0, 0, 1],    # Co-located receiver
    receiver_type='dBdt',           # Measure dB/dt
    times=np.logspace(-5, -2, 31)   # Time gates
)

# Define 1D layered Earth model
layer_thicknesses = [0.5, 1.0, 2.0, 5.0]  # meters
conductivities = [0.01, 0.1, 0.05, 0.02]  # S/m

# Create forward model and compute response
tdem_fwd = TDEMForwardModeling(config)
response = tdem_fwd.forward(layer_thicknesses, conductivities)

# For hydro-geophysics integration with MODFLOW
wc_profile = water_content[:, row, col]  # 1D profile from MODFLOW
sigma = waxman_smits(wc_profile, rhos=100, n=2.0, porosity=0.35)
response = tdem_fwd.forward(layer_thicknesses, sigma)
```

## 5. Time-Lapse Inversion

Perform sophisticated time-lapse ERT inversions:

```python
from PyHydroGeophysX.inversion import TimeLapseERTInversion, WindowedTimeLapseERTInversion

# Full time-lapse inversion
inversion = TimeLapseERTInversion(
    data_files=ert_files,
    measurement_times=times,
    lambda_val=50.0,        # Spatial regularization
    alpha=10.0,             # Temporal regularization
    inversion_type="L2"     # L1, L2, or L1L2
)
result = inversion.run()

# Windowed inversion for large datasets
windowed_inv = WindowedTimeLapseERTInversion(
    data_dir="data/", ert_files=files, window_size=3
)
result = windowed_inv.run(window_parallel=True)
```

## 6. Uncertainty Quantification

Quantify uncertainty in water content estimates:

```python
from PyHydroGeophysX.Geophy_modular import ERTtoWC

# Set up Monte Carlo analysis
converter = ERTtoWC(mesh, resistivity_values, cell_markers, coverage)

# Define parameter distributions for different geological layers
layer_distributions = {
    3: {  # Top layer
        'rhos': {'mean': 100.0, 'std': 20.0},
        'n': {'mean': 2.2, 'std': 0.2},
        'porosity': {'mean': 0.40, 'std': 0.05}
    },
    2: {  # Bottom layer
        'rhos': {'mean': 500.0, 'std': 100.0},
        'n': {'mean': 1.8, 'std': 0.2},
        'porosity': {'mean': 0.35, 'std': 0.1}
    }
}

converter.setup_layer_distributions(layer_distributions)
wc_all, sat_all, params = converter.run_monte_carlo(n_realizations=100)
stats = converter.get_statistics()  # mean, std, percentiles
```

## 📊 Example Workflows

### Complete Workflow: Hydrology to Geophysics

```python
from PyHydroGeophysX import *

# 1. Load hydrological data
processor = MODFLOWWaterContent("modflow_dir", idomain)
water_content = processor.load_timestep(timestep=50)

# 2. Set up 2D profile interpolation
interpolator = ProfileInterpolator(
    point1=[115, 70], point2=[95, 180], 
    surface_data=surface_elevation
)

# 3. Create mesh with geological structure
mesh_creator = MeshCreator(quality=32)
mesh, _ = mesh_creator.create_from_layers(
    surface=surface_line, layers=[layer1, layer2]
)

# 4. Convert to resistivity
resistivity = water_content_to_resistivity(
    water_content, rhos=100, n=2.2, porosity=0.3
)

# 5. Forward model synthetic ERT data
synthetic_data, _ = ERTForwardModeling.create_synthetic_data(
    xpos=electrode_positions, mesh=mesh, res_models=resistivity
)

# 6. Invert synthetic data
inversion = ERTInversion(data_file="synthetic_data.dat")
result = inversion.run()
```

### Structure-Constrained Inversion

```python
# 1. Process seismic data to extract velocity structure
from PyHydroGeophysX.Geophy_modular import process_seismic_tomography, extract_velocity_structure

TT_manager = process_seismic_tomography(travel_time_data, lam=50)
interface_x, interface_z, _ = extract_velocity_structure(
    TT_manager.paraDomain, TT_manager.model.array(), threshold=1200
)

# 2. Create ERT mesh with velocity interface constraints
from PyHydroGeophysX.Geophy_modular import create_ert_mesh_with_structure

constrained_mesh, markers, regions = create_ert_mesh_with_structure(
    ert_data, (interface_x, interface_z)
)

# 3. Run constrained inversion
inversion = TimeLapseERTInversion(
    data_files=ert_files, mesh=constrained_mesh
)
result = inversion.run()
```

## 🛠 Advanced Features

### GPU Acceleration

Enable GPU acceleration for large-scale inversions:

```python
inversion = TimeLapseERTInversion(
    data_files=files,
    use_gpu=True,           # Requires CuPy
    parallel=True,          # CPU parallelization
    n_jobs=-1               # Use all available cores
)
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for full guidelines, including how to add new petrophysical models, hydrological-model loaders, and geophysical backends.

- Fork the repository  
- Create your feature branch (`git checkout -b feature/AmazingFeature`)  
- Commit your changes (`git commit -m 'Add some AmazingFeature'`)  
- Push to the branch (`git push origin feature/AmazingFeature`)  
- Open a Pull Request  

## 📝 Citation

If you use PyHydroGeophysX in your research, please cite:

```bibtex
@article{chen2026pyhydrogeophysx,
  author  = {Chen, Hang and Niu, Qifei and Wu, Yuxin},
  title   = {PyHydroGeophysX: An Extensible Open-Source Platform for Integrating Hydrological Models with Geophysical Measurements},
  journal = {SoftwareX},
  year    = {2026},
  note    = {In press},
  url     = {https://github.com/geohang/PyHydroGeophysX}
}
```

```bibtex
@article{chen2026agentworkflow,
  author = {Chen, Hang},
  title = {A Generalizable Automated Geophysical Agent Workflow for Accessible Subsurface Hydrology Analysis},
  journal = {Big Data and Earth System},
  pages = {100042},
  year = {2026}
}
```

### References

- Chen, H., Niu, Q., Mendieta, A., Bradford, J., & McNamara, J. (2023). Geophysics-informed hydrologic modeling of a mountain headwater catchment for studying hydrological partitioning in the critical zone. *Water Resources Research, 59*(12), e2023WR035280. https://doi.org/10.1029/2023WR035280

Additionally, please cite the underlying libraries you use:

**For ERT data processing (RESIPY):**
```bibtex
@article{blanchy2020resipy,
  title={ResIPy, an intuitive open source software for complex geoelectrical inversion/modeling},
  author={Blanchy, Guillaume and Saneiyan, Sina and Boyd, Jimmy and McLachlan, Paul and Binley, Andrew},
  journal={Computers \& Geosciences},
  volume={137},
  pages={104423},
  year={2020},
  publisher={Elsevier},
  doi={10.1016/j.cageo.2020.104423}
}
```

**For geophysical modeling (pyGIMLi):**
```bibtex
@article{rucker2017pygimli,
  title={pyGIMLi: An open-source library for modelling and inversion in geophysics},
  author={R{\"u}cker, Carsten and G{\"u}nther, Thomas and Wagner, Florian M},
  journal={Computers \& Geosciences},
  volume={109},
  pages={106--123},
  year={2017},
  publisher={Elsevier},
  doi={10.1016/j.cageo.2017.07.011}
}
```

**For hydrological modeling (ParFlow):**
```bibtex
@article{maxwell2015parflow,
  title={A high-resolution simulation of groundwater and surface water over most of the continental US with the integrated hydrologic model ParFlow v3},
  author={Maxwell, Reed M and Condon, Laura E and Kollet, Stefan J},
  journal={Geoscientific Model Development},
  volume={8},
  number={3},
  pages={923--937},
  year={2015},
  publisher={Copernicus GmbH},
  doi={10.5194/gmd-8-923-2015}
}
```

**For MODFLOW scripting (FloPy):**
```bibtex
@article{bakker2016flopy,
  author={Bakker, Mark and Post, Vincent and Langevin, Christian D and Hughes, Joseph D and White, Jeremy T and Starn, Jeffrey J and Fienen, Michael N},
  title={Scripting MODFLOW model development using Python and FloPy},
  journal={Groundwater},
  volume={54},
  number={5},
  pages={733--739},
  year={2016},
  doi={10.1111/gwat.12413},
  url={https://ngwa.onlinelibrary.wiley.com/doi/full/10.1111/gwat.12413}
}
```

**For electromagnetic modeling (SimPEG):**
```bibtex
@article{cockett2015simpeg,
  title={SimPEG: An open source framework for simulation and gradient based parameter estimation in geophysical applications},
  author={Cockett, Rowan and Kang, Seogi and Heagy, Lindsey J and Pidlisecky, Adam and Oldenburg, Douglas W},
  journal={Computers \& Geosciences},
  volume={85},
  pages={142--154},
  year={2015},
  publisher={Elsevier},
  doi={10.1016/j.cageo.2015.09.015}
}
```

## 📄 License

This project is licensed under the Apache-2.0 license - see the LICENSE file for details.

## 🙏 Acknowledgments

- **RESIPY** developers (Blanchy, Guillaume and Saneiyan, Sina and Boyd, Jimmy and McLachlan, Paul and Binley, Andrew and contributors) for the excellent ERT data processing library that powers our field data workflows
- **pyGIMLi** team (Carsten Rücker, Thomas Günther, Florian Wagner, and contributors) for the outstanding geophysical modeling and inversion framework  
- **SimPEG** team (Rowan Cockett, Seogi Kang, Lindsey Heagy, Adam Pidlisecky, Douglas Oldenburg, and contributors) for the powerful simulation and parameter estimation framework enabling TDEM forward modeling and inversion
- **ParFlow** development team (Reed Maxwell, Laura Condon, Stefan Kollet, and contributors) for the integrated surface-subsurface hydrologic model
- **MODFLOW** community (USGS and contributors) for the groundwater modeling standard
- All open-source contributors and users providing valuable feedback  

## 📧 Contact

Author: Hang Chen  
Email: hchen117@uiowa.edu
Issues: GitHub Issues  

---

PyHydroGeophysX - Bridging the gap between hydrological models and geophysical monitoring

Note: This package is under active development. Please report issues and feature requests through the GitHub issue tracker.

