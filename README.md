# PyHydroGeophysX

A comprehensive Python package for integrating hydrological model outputs with geophysical forward modeling and inversion, specializing in electrical resistivity tomography (ERT) and seismic refraction tomography (SRT) for watershed monitoring applications.

## 🌟 Key Features

- 🌊 **Hydrological Model Integration:** Seamless loading and processing of MODFLOW and ParFlow outputs.
- 🪨 **Petrophysical Relationships:** Advanced models for converting between water content, saturation, resistivity, and seismic velocity (e.g., Archie's, Waxman-Smits, DEM, Hertz-Mindlin).
- ⚡ **Forward Modeling:** Complete ERT and SRT forward modeling capabilities with synthetic data generation using PyGIMLi.
- 🔄 **Time-Lapse Inversion:** Sophisticated algorithms for time-lapse ERT inversion with temporal regularization (L1, L2, and L1L2 norms supported).
- 🏔️ **Structure-Constrained Inversion:** Integration of seismic velocity interfaces for constrained ERT inversion.
- 📊 **Uncertainty Quantification:** Monte Carlo methods for assessing uncertainty in water content estimates derived from ERT.
- 🚀 **High Performance (Optional):** GPU acceleration support (CUDA/CuPy) and parallel processing capabilities (Joblib) for selected solvers.
- 📈 **Advanced Solvers:** A suite of iterative linear solvers (CGLS, LSQR, RRLS, RRLSQR) and direct solvers, with Tikhonov regularization and iterative refinement options.

## 📋 Requirements

- Python 3.8 or higher
- NumPy, SciPy, Matplotlib
- PyGIMLi (core geophysical modeling and meshing)
- tqdm (primarily for progress bars in examples and potentially long computations)
- Optional:
    - CuPy (for GPU acceleration in specific solvers)
    - joblib (for parallel processing in specific solvers)

## 🛠️ Installation

### From Source

The primary method for installation is from source:

```bash
git clone https://github.com/your-org-or-username/PyHydroGeophysX.git
cd PyHydroGeophysX
pip install -e .
```
This installs the package in "editable" mode, which is useful for development. For a standard installation, you can use:
```bash
pip install .
```
from within the `PyHydroGeophysX` directory.

### Dependencies

The core dependencies can be installed via pip:
```bash
pip install numpy scipy matplotlib pygimli tqdm
```

For optional GPU support (requires a compatible NVIDIA GPU and CUDA toolkit):
```bash
pip install cupy-cuda11x  # Replace '11x' with your specific CUDA version (e.g., cupy-cuda110, cupy-cuda120)
```

For optional parallel processing features in some solvers:
```bash
pip install joblib
```

## 📚 Documentation

Comprehensive documentation is built locally using Sphinx.

To build documentation:
```bash
cd docs
make html
```
Then open `docs/build/html/index.html` in your web browser.
(Note: If a Read the Docs link is set up in the future, it will be provided here.)


## 🗂️ Package Structure

```
PyHydroGeophysX/
├── core/                   # Core utilities
│   ├── interpolation.py    # Profile interpolation tools
│   └── mesh_utils.py       # Mesh creation and manipulation
├── model_output/           # Hydrological model interfaces
│   ├── base.py             # Base class for model output processing
│   ├── modflow_output.py   # MODFLOW data loading
│   ├── parflow_output.py   # ParFlow data loading
│   └── water_content.py    # Water content calculation utilities
├── petrophysics/           # Rock physics models
│   ├── resistivity_models.py  # Waxman-Smits, Archie models
│   └── velocity_models.py     # DEM, Hertz-Mindlin models
├── forward/                # Forward modeling
│   ├── ert_forward.py      # ERT forward modeling
│   └── srt_forward.py      # Seismic forward modeling
├── inversion/              # Inverse modeling
│   ├── base.py             # Base classes for inversion
│   ├── ert_inversion.py    # Single-time ERT inversion
│   ├── time_lapse.py       # Time-lapse inversion
│   └── windowed.py         # Windowed time-lapse for large datasets
├── solvers/                # Linear algebra solvers
│   ├── linear_solvers.py   # CGLS, LSQR, RRLS etc. with GPU/parallel options
│   └── solver.py           # (Note: This may be a legacy/shim module)
├── Hydro_modular/          # Modules for direct hydro-to-geophysics conversion
│   ├── hydro_to_ert.py     # Converts hydro model outputs to ERT data
│   └── hydro_to_srt.py     # Converts hydro model outputs to SRT data
└── Geophy_modular/         # Modules for geophysical data processing & integration
    ├── ERT_to_WC.py        # Converts ERT resistivity to water content with UQ
    ├── seismic_processor.py # Processes seismic data to identify structures
    └── structure_integration.py # Integrates structural info into meshes
```

## 📖 Examples

The `examples/` directory contains comprehensive tutorials demonstrating various functionalities:

- `Ex1_model_output.py`: Loading and processing outputs from hydrological models like MODFLOW and ParFlow.
- `Ex2_workflow.py`: Illustrates a complete workflow, from hydrological model data to geophysical inversion.
- `Ex3_Time_lapse_measurement.py`: Focuses on creating synthetic time-lapse ERT data.
- `Ex4_TL_inversion.py`: Demonstrates time-lapse ERT inversion techniques.
- `Ex5_SRT.py`: Covers the seismic refraction tomography (SRT) workflow, including forward modeling and inversion.
- `Ex6_Structure_resinv.py`: Shows how to perform structure-constrained ERT inversion using information derived from seismic data.
- `Ex7_structure_TLresinv.py`: Extends structural constraints to time-lapse ERT inversion.
- `Ex8_MC_WC.py`: Details Monte Carlo uncertainty quantification for converting ERT results to water content.

*(Note: Prerequisite variables in example snippets below, such as `mesh`, `data`, `electrode_positions`, etc., are typically defined in earlier parts of the respective example scripts or assumed to be loaded/generated as shown in other examples like `Ex2_workflow.py`.)*

## 🚀 Quick Start

### 1. Hydrological Model Integration

Load and process outputs from various hydrological models:
*(Assumes `idomain` is defined for MODFLOW, and `model_dir`, `run_name` for Parflow as per examples.)*
```python
# MODFLOW
from PyHydroGeophysX.model_output import MODFLOWWaterContent, MODFLOWPorosity

# Assuming 'idomain' (active model domain array) is defined
processor = MODFLOWWaterContent("path/to/modflow_workspace", idomain)
water_content = processor.load_time_range(start_idx=0, end_idx=10)

# ParFlow
from PyHydroGeophysX.model_output import ParflowSaturation, ParflowPorosity

saturation_proc = ParflowSaturation("path/to/parflow_model_dir", "parflow_run_name")
saturation = saturation_proc.load_timestep(100)
```

### 2. Petrophysical Modeling

Convert between hydrological and geophysical properties:
*(Assumes `wc`, `porosity`, `saturation` variables are obtained from previous steps or defined.)*
```python
from PyHydroGeophysX.petrophysics import (
    water_content_to_resistivity,
    HertzMindlinModel,
    DEMModel
)

# Water content to resistivity (e.g., using Waxman-Smits like parameters)
# Example parameters; these should be calibrated or site-specific.
resistivity = water_content_to_resistivity(
    water_content=wc, rho_saturated=100.0, saturation_exponent_n=2.0,
    porosity=porosity, sigma_surface=0.002
)

# Water content to seismic velocity (rock physics models)
hm_model = HertzMindlinModel() # Uses default critical_porosity, coordination_number
# Example parameters; these should be site-specific.
vp_high, vp_low = hm_model.calculate_velocity(
    porosity=porosity, saturation=saturation,
    matrix_bulk_modulus=30.0, matrix_shear_modulus=20.0, matrix_density=2650.0
    # Other parameters like depth, fluid properties use defaults if not specified
)
```

### 3. Forward Modeling

Generate synthetic geophysical data:
*(Assumes `mesh`, `data` (for ERT scheme), `electrode_positions`, `resistivity_model`,
`scheme` (for SRT), `geophone_positions`, `velocity_model` are defined, e.g., from previous steps or examples.)*
```python
from PyHydroGeophysX.forward import ERTForwardModeling, SeismicForwardModeling
import numpy as np # For example arrays
import pygimli as pg # For mesh/data creation examples

# Example: Create a basic mesh and ERT data scheme if not loaded
# mesh = pg.createGrid(x=np.linspace(0,10,11), y=np.linspace(0,-5,6))
# electrode_positions = np.array([[x,0] for x in np.linspace(0,10,11)])
# data = pg.physics.ert.createData(elecs=electrode_positions, schemeName='dd')
# resistivity_model = np.ones(mesh.cellCount()) * 100.0


# ERT forward modeling
ert_fwd = ERTForwardModeling(mesh=mesh, data_scheme=data)
synthetic_data_ert = ert_fwd.forward(resistivity_model=resistivity_model) # Returns modeled data

# Or create synthetic data with noise
# synthetic_data_container_ert, _ = ERTForwardModeling.create_synthetic_data(
#     xpos_electrodes=electrode_positions[:,0], ypos_electrodes=electrode_positions[:,1],
#     fwd_mesh=mesh, resistivity_values=resistivity_model
# )


# Seismic forward modeling (assuming 'scheme' and 'velocity_model' are defined)
# srt_fwd = SeismicForwardModeling(mesh=mesh, scheme=scheme)
# travel_times = srt_fwd.forward(velocity_model=velocity_model, is_slowness=False)

# Or create synthetic data with noise
# geophone_positions = np.linspace(0, 20, 21) # Example
# velocity_model_example = np.ones(mesh.cellCount()) * 1000.0 # Example
# synthetic_travel_times, _ = SeismicForwardModeling.create_synthetic_data(
#     sensor_x_coords=geophone_positions, fwd_mesh=mesh, velocity_model_values=velocity_model_example
# )
```

### 4. Time-Lapse Inversion

Perform sophisticated time-lapse ERT inversions:
*(Assumes `ert_files` is a list of data file paths and `times` is a list of measurement times.)*
```python
from PyHydroGeophysX.inversion import TimeLapseERTInversion, WindowedTimeLapseERTInversion

# Full time-lapse inversion (assuming 'ert_files' and 'times' are defined)
# inversion = TimeLapseERTInversion(
#     data_files=ert_files,
#     measurement_times=times,
#     lambda_val=50.0,        # Spatial regularization
#     alpha=10.0,             # Temporal regularization
#     inversion_type="L2"     # L1, L2, or L1L2
# )
# result = inversion.run() # This would run the inversion

# Windowed inversion for large datasets
# windowed_inv = WindowedTimeLapseERTInversion(
#     data_dir="path/to/data_directory/", ert_files=list_of_filenames,
#     measurement_times=list_of_times, window_size=3
# )
# result_windowed = windowed_inv.run(process_windows_in_parallel=True)
```

### 5. Uncertainty Quantification

Quantify uncertainty in water content estimates:
*(Assumes `mesh`, `resistivity_values` (e.g., from inversion), `cell_markers`, `coverage` are defined.)*
```python
from PyHydroGeophysX.Geophy_modular import ERTtoWC

# converter = ERTtoWC(mesh, resistivity_values, cell_markers, coverage)

# Define parameter distributions for different geological layers
# layer_distributions = {
#     10: {  # Example marker for Top layer
#         'rhos': {'mean': 100.0, 'std': 20.0}, # Saturated resistivity
#         'n': {'mean': 2.0, 'std': 0.1},      # Saturation exponent
#         'sigma_s': {'mean': 0.001, 'std': 0.0005}, # Surface conductivity
#         'porosity': {'mean': 0.30, 'std': 0.05}
#     },
#     20: {  # Example marker for Bottom layer
#         'rhos': {'mean': 500.0, 'std': 50.0},
#         'n': {'mean': 1.8, 'std': 0.1},
#         'sigma_s': {'mean': 0.0001, 'std': 0.00005},
#         'porosity': {'mean': 0.25, 'std': 0.03}
#     }
# }
# converter.setup_layer_distributions(layer_distributions)
# wc_all_realizations, sat_all_realizations, params_used = converter.run_monte_carlo(n_realizations=100)
# stats_wc = converter.get_statistics()  # mean, std, percentiles of water content
```

## 📊 Example Workflows

*(Note: These are conceptual. Refer to the `examples/` directory for runnable scripts.
Prerequisite variables need to be defined as in those examples.)*

### Complete Workflow: Hydrology to Geophysics

```python
# Conceptual:
# 1. Load hydrological data (e.g., water content from MODFLOW)
#    wc_data_3d, porosity_data_3d, idomain_data = load_modflow_outputs(...)
#    surface_elevation_data = load_surface_dem(...)

# 2. Define a 2D profile and interpolate hydro data
#    profile = ProfileInterpolator(start_point, end_point, surface_elevation_data)
#    wc_profile = profile.interpolate_3d_data(wc_data_3d)
#    porosity_profile = profile.interpolate_3d_data(porosity_data_3d)
#    layer_structure_profile = profile.get_profile_structure(...) # if layered

# 3. Create a PyGIMLi mesh for the profile, possibly with layers
#    mesh = create_mesh_with_layers(profile.L_profile, layer_structure_profile, ...)
#    wc_on_mesh = profile.interpolate_to_mesh(wc_profile, ...)
#    porosity_on_mesh = profile.interpolate_to_mesh(porosity_profile, ...)

# 4. Convert to resistivity model using petrophysics
#    resistivity_on_mesh = water_content_to_resistivity(wc_on_mesh, ..., porosity_on_mesh, ...)

# 5. Define ERT survey and generate synthetic data
#    electrode_coords = define_electrode_positions(profile.L_profile, profile.surface_profile, ...)
#    synthetic_ert_data, _ = ERTForwardModeling.create_synthetic_data(
#        xpos_electrodes=electrode_coords[:,0], ypos_electrodes=electrode_coords[:,1],
#        fwd_mesh=mesh, resistivity_values=resistivity_on_mesh
#    )

# 6. Invert synthetic data
#    ert_inverter = ERTInversion(data_file="path_to_synthetic_data.dat", mesh=mesh) # Or pass DataContainer directly
#    inversion_result = ert_inverter.run()
```

### Structure-Constrained Inversion

```python
# Conceptual:
# 1. Perform SRT inversion to get a velocity model
#    srt_data = load_srt_data(...)
#    srt_inversion = SRTInversion(srt_data, srt_mesh) # Assuming an SRTInversion class
#    velocity_model = srt_inversion.run().final_model

# 2. Extract structural interface (e.g., bedrock) from velocity model
#    interface_x, interface_z, _ = extract_velocity_structure(
#        srt_mesh, velocity_model, threshold=1500.0
#    )

# 3. Create ERT mesh incorporating this structural interface
#    ert_data_for_constrained_inv = load_ert_data(...)
#    constrained_mesh, _, _ = create_ert_mesh_with_structure(
#        ert_data_for_constrained_inv, (interface_x, interface_z)
#    )

# 4. Run ERT inversion using the constrained mesh
#    constrained_ert_inverter = ERTInversion(data=ert_data_for_constrained_inv, mesh=constrained_mesh)
#    constrained_result = constrained_ert_inverter.run()
```

## 🛠 Advanced Features

### GPU Acceleration

Enable GPU acceleration for supported solvers (requires CuPy and compatible hardware):

```python
# Example for a solver that supports GPU
# from PyHydroGeophysX.solvers import CGLSSolver
# solver = CGLSSolver(use_gpu=True)

# Inversion classes might also take use_gpu flags:
# inversion = TimeLapseERTInversion(
#     data_files=list_of_filenames, measurement_times=list_of_times,
#     use_gpu=True, # This flag would be passed down to the solver
#     parallel=True, # For CPU parallel parts if any
#     n_jobs=-1
# )
# result = inversion.run()
```

## 🤝 Contributing

We welcome contributions! Please see our Contributing Guidelines (to be created) for details.

- Fork the repository.
- Create your feature branch (`git checkout -b feature/AmazingFeature`).
- Commit your changes (`git commit -m 'Add some AmazingFeature'`).
- Push to the branch (`git push origin feature/AmazingFeature`).
- Open a Pull Request.

## 📝 Citation

If you use PyHydroGeophysX in your research, please cite:

```bibtex
@software{chen2024pyhydrogeophysx,
  author = {Chen, Hang and {others}},
  title = {PyHydroGeophysX: A Python Package for Integrated Hydrological and Geophysical Modeling and Inversion},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/your-org-or-username/PyHydroGeophysX}
}
```
*(Please update with actual authors and details when available.)*

## 📄 License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## 🙏 Acknowledgments

- The PyGIMLi community for their excellent open-source geophysical modeling framework.
- Developers and communities of MODFLOW, ParFlow, and other tools that inspire or integrate with this work.

## 📧 Contact

Author: Hang Chen  
Email: hchen8@lbl.gov
Issues: [GitHub Issues for PyHydroGeophysX](https://github.com/your-org-or-username/PyHydroGeophysX/issues)

---

PyHydroGeophysX - Bridging the gap between hydrological models and geophysical monitoring.

*Note: This package is under active development. Please report issues and feature requests through the GitHub issue tracker.*
