# PyHydroGeophysX

A comprehensive Python package for integrating hydrological model outputs with geophysical forward modeling and inversion, specializing in electrical resistivity tomography (ERT) and seismic refraction tomography (SRT) for watershed monitoring applications.

## 🌟 Key Features

- 🌊 **Hydrological Model Integration:** Seamless loading and processing of MODFLOW and ParFlow outputs  
- 📊 **ERT Data Processing:** Standardized loading, quality control, and export of ERT field data with RESIPY integration  
- 🪨 **Petrophysical Relationships:** Advanced models for converting between water content, saturation, resistivity, and seismic velocity  
- ⚡ **Forward Modeling:** Complete ERT and SRT forward modeling capabilities with synthetic data generation  
- 🔄 **Time-Lapse Inversion:** Sophisticated algorithms for time-lapse ERT inversion with temporal regularization  
- 🏔️ **Structure-Constrained Inversion:** Integration of seismic velocity interfaces for constrained ERT inversion  
- � **Uncertainty Quantification:** Monte Carlo methods for parameter uncertainty assessment  
- 🚀 **High Performance:** GPU acceleration support (CUDA/CuPy) and parallel processing capabilities  
- � **Advanced Solvers:** Multiple linear solvers (CGLS, LSQR, RRLS) with optional GPU acceleration

## 📋 Requirements

- Python 3.8 or higher  
- NumPy, SciPy, Matplotlib  
- PyGIMLi (for geophysical modeling)  
- Optional: CuPy (for GPU acceleration), joblib (for parallel processing)

## 🛠️ Installation
### From PyPI (Recommended)

pip install pyhydrogeophysx

### From Source

```bash
git clone https://github.com/yourusername/PyHydroGeophysX.git
cd PyHydroGeophysX
pip install -e .
```

### Dependencies

```bash
pip install numpy scipy matplotlib pygimli joblib tqdm
```

For GPU support (optional):

```bash
pip install cupy-cuda11x  # Replace with your CUDA version
```




## 📚 Documentation

Comprehensive documentation is available at Read the Docs.

To build documentation locally:

```bash
cd docs
make html
```

## 🗂️ Package Structure

```
PyHydroGeophysX/
├── core/               # Core utilities
│   ├── interpolation.py    # Profile interpolation tools
│   └── mesh_utils.py       # Mesh creation and manipulation
├── data_processing/    # Geophysical data processing **NEW**
│   └── ert_data_agent.py   # ERT data loading, QC, and export
├── model_output/       # Hydrological model interfaces
│   ├── modflow_output.py   # MODFLOW data loading
│   └── parflow_output.py   # ParFlow data loading
├── petrophysics/       # Rock physics models
│   ├── resistivity_models.py  # Waxman-Smits, Archie models
│   └── velocity_models.py     # DEM, Hertz-Mindlin models
├── forward/            # Forward modeling
│   ├── ert_forward.py      # ERT forward modeling
│   └── srt_forward.py      # Seismic forward modeling
├── inversion/          # Inverse modeling
│   ├── ert_inversion.py    # Single-time ERT inversion
│   ├── time_lapse.py       # Time-lapse inversion
│   └── windowed.py         # Windowed time-lapse for large datasets
├── solvers/            # Linear algebra solvers
│   └── linear_solvers.py   # CGLS, LSQR, RRLS with GPU support
├── Hydro_modular/      # Direct hydro-to-geophysics conversion
└── Geophy_modular/     # Geophysical data processing tools
```

## 📖 Examples

The examples folder provides paired Jupyter notebooks (.ipynb) and Python scripts (.py) for each workflow. Data used by examples is under examples/data, with outputs written to examples/results.

- **Ex_ERT_data_process**: Loading, quality control, and export of field ERT data with RESIPY integration (notebook: Ex_ERT_data_process.ipynb).
- Ex_model_output: Loading and processing hydrological model outputs (MODFLOW/ParFlow) (notebook: Ex_model_output.ipynb, script: Ex_model_output.py).
- Ex_ERT_workflow: End‑to‑end ERT modeling and inversion workflow (notebook: Ex_ERT_workflow.ipynb, script: Ex_ERT_workflow.py).
- Ex_Time_lapse_measurement: Generate synthetic time‑lapse ERT measurements and schedules (notebook: Ex_Time_lapse_measurement.ipynb, script: Ex_Time_lapse_measurement.py).
- Ex_TL_inversion: Time‑lapse ERT inversion with temporal regularization and windowed processing (notebook: Ex_TL_inversion.ipynb, script: Ex_TL_inversion.py).
- Ex_Structure_resinv: Structure‑constrained resistivity inversion using seismic interfaces (notebook: Ex_Structure_resinv.ipynb, script: Ex_Structure_resinv.py).
- Ex_structure_TLresinv: Structure‑constrained time‑lapse resistivity inversion (notebook: Ex_structure_TLresinv.ipynb, script: Ex_structure_TLresinv.py).
- EX_SRT_forward: Seismic refraction tomography forward modeling and synthetic travel times (notebook: EX_SRT_forward.ipynb, script: EX_SRT_forward.py).
- Ex_SRT_inv: Seismic refraction tomography inversion workflow (notebook: Ex_SRT_inv.ipynb, script: Ex_SRT_inv.py).
- Ex_MC_Hydro: Monte Carlo uncertainty quantification for hydro‑to‑resistivity conversion (notebook: Ex_MC_Hydro.ipynb, script: Ex_MC_Hydro.py).


## 🚀 Quick Start

## 0. ERT Field Data Processing

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

## 1. Hydrological Model Integration

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

## 2. Petrophysical Modeling

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

## 3. Forward Modeling

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

## 4. Time-Lapse Inversion

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

## 5. Uncertainty Quantification

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

We welcome contributions! Please see our Contributing Guidelines for details.

- Fork the repository  
- Create your feature branch (`git checkout -b feature/AmazingFeature`)  
- Commit your changes (`git commit -m 'Add some AmazingFeature'`)  
- Push to the branch (`git push origin feature/AmazingFeature`)  
- Open a Pull Request  

## 📝 Citation

If you use PyHydroGeophysX in your research, please cite:

```bibtex
@software{chen2025pyhydrogeophysx,
  author = {Chen, Hang and Niu, Qifei and Wu, Yuxin},
  title = {PyHydroGeophysX: An Extensible Open-Source Platform for Bridging Hydrological Models and Geophysical Measurements},
  year = {2025},
  publisher = {Water Resources Research (under review)},
  url = {https://github.com/geohang/PyHydroGeophysX}
}
```

Additionally, if you use the ERT data processing module, please cite RESIPY:

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

## 📄 License

This project is licensed under the Apache-2.0 license - see the LICENSE file for details.

## 🙏 Acknowledgments

- **RESIPY** developers (Guillaume Blanchy, Jimmy Boyd, and contributors) for the excellent ERT data processing library that powers our field data workflows
- **PyGIMLi** team for the outstanding geophysical modeling framework  
- **MODFLOW** and **ParFlow** communities for hydrologic modeling tools  
- All open-source contributors and users providing valuable feedback  

## 📧 Contact

Author: Hang Chen  
Email: hangchen.work@gmail.com
Issues: GitHub Issues  

---

PyHydroGeophysX - Bridging the gap between hydrological models and geophysical monitoring

Note: This package is under active development. Please report issues and feature requests through the GitHub issue tracker.
