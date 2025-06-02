# PyHydroGeophysX
<!-- Main Title and brief description of the package -->
A comprehensive Python package for integrating hydrological model outputs with geophysical forward modeling and inversion, specializing in electrical resistivity tomography (ERT) and seismic refraction tomography (SRT) for watershed monitoring applications.

## 🌟 Key Features
<!-- This section highlights the main capabilities of the package. -->
- 🌊 **Hydrological Model Integration:** Seamless loading and processing of MODFLOW and ParFlow outputs.
- 🪨 **Petrophysical Relationships:** Advanced models for converting between water content, saturation, resistivity, and seismic velocity.
- ⚡ **Forward Modeling:** Complete ERT and SRT forward modeling capabilities, including synthetic data generation.
- 🔄 **Time-Lapse Inversion:** Sophisticated algorithms for time-lapse ERT inversion with temporal regularization.
- 🏔️ **Structure-Constrained Inversion:** Integration of seismic velocity interfaces (or other structural information) for constrained ERT inversion.
- 📊 **Uncertainty Quantification:** Monte Carlo methods for assessing parameter uncertainty in petrophysical conversions.
- 🚀 **High Performance:** Support for GPU acceleration (via CUDA/CuPy) and parallel CPU processing for demanding computations.
- 📈 **Advanced Solvers:** A selection of multiple linear solvers (e.g., CGLS, LSQR, RRLS) with options for GPU acceleration.

## 📋 Requirements
<!-- Lists essential and optional dependencies. -->
- Python 3.8 or higher
- NumPy, SciPy, Matplotlib (core scientific Python libraries)
- PyGIMLi (essential for geophysical modeling and inversion)
- Optional:
    - CuPy (for GPU acceleration in solvers)
    - joblib (for parallel processing in some modules)
    - tqdm (for progress bars in Monte Carlo simulations and other iterative processes)

## 🛠️ Installation
<!-- Provides instructions for installing the package. -->

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

<!-- Link to detailed documentation -->
Comprehensive documentation is available at [Read the Docs](https://your-rtd-link.readthedocs.io/en/latest/). <!-- TODO: Update this link when available -->

To build documentation locally:
<!-- Instructions for developers to build docs -->
```bash
cd docs
make html
```

## 🗂️ Package Structure
<!-- Overview of the main directories and their purpose -->
```
PyHydroGeophysX/
├── core/               # Core utilities (e.g., interpolation, mesh tools)
│   ├── interpolation.py    # Profile interpolation tools
│   └── mesh_utils.py       # Mesh creation and manipulation utilities
├── model_output/       # Interfaces for hydrological model outputs
│   ├── modflow_output.py   # MODFLOW data loading classes
│   └── parflow_output.py   # ParFlow data loading classes
├── petrophysics/       # Petrophysical (rock physics) models
│   ├── resistivity_models.py  # Models like Waxman-Smits, Archie for resistivity
│   └── velocity_models.py     # Models like DEM, Hertz-Mindlin for seismic velocity
├── forward/            # Geophysical forward modeling tools
│   ├── ert_forward.py      # ERT forward modeling and synthetic data generation
│   └── srt_forward.py      # Seismic forward modeling and synthetic data generation
├── inversion/          # Geophysical inverse modeling tools
│   ├── ert_inversion.py    # Single-time ERT inversion
│   ├── time_lapse.py       # Time-lapse ERT inversion with temporal regularization
│   └── windowed.py         # Windowed time-lapse inversion for large temporal datasets
├── solvers/            # Linear algebra solvers for inversion
│   └── linear_solvers.py   # Implementations of CGLS, LSQR, RRLS with optional GPU support
├── Hydro_modular/      # Modules for direct conversion from hydrological parameters to geophysical inputs
│   ├── hydro_to_ert.py     # Converts hydro outputs to ERT model/data
│   └── hydro_to_srt.py     # Converts hydro outputs to SRT model/data
└── Geophy_modular/     # Modules for specialized geophysical data processing and integration
    ├── ERT_to_WC.py        # Converts ERT results to Water Content with uncertainty
    ├── seismic_processor.py # Processes seismic data to extract structural information
    └── structure_integration.py # Integrates structural info (e.g., from seismic) into ERT meshes
```
<!-- TODO: Add a note about the examples/ and tests/ directories if they exist -->

## 📖 Examples
<!-- Points users to example scripts demonstrating usage. -->
The `examples/` directory contains comprehensive Jupyter notebooks or Python scripts for various workflows:

- `Ex1_model_output.py`: Demonstrates loading and basic processing of hydrological model outputs.
- `Ex2_workflow.py`: Illustrates a complete workflow from hydrological models to geophysical inversion.
- `Ex3_Time_lapse_measurement.py`: Shows how to create synthetic time-lapse ERT data.
- `Ex4_TL_inversion.py`: Details time-lapse ERT inversion techniques.
- `Ex5_SRT.py`: Covers a seismic refraction tomography workflow.
- `Ex6_Structure_resinv.py`: Example of structure-constrained ERT inversion.
- `Ex7_structure_TLresinv.py`: Example of structure-constrained time-lapse ERT inversion.
- `Ex8_MC_WC.py`: Demonstrates Monte Carlo uncertainty quantification for water content estimation.
<!-- Ensure these example filenames are accurate and cover the main functionalities. -->

## 🚀 Quick Start
<!-- Provides minimal, runnable code snippets for core functionalities. -->

### 1. Hydrological Model Integration
<!-- Example for loading data from hydrological models like MODFLOW and ParFlow. -->
Load and process outputs from various hydrological models:

```python
# MODFLOW Example
from PyHydroGeophysX.model_output import MODFLOWWaterContent, MODFLOWPorosity # Corrected import path

# Assuming 'idomain' is a numpy array defining the active model domain
# processor = MODFLOWWaterContent("path/to/sim_workspace", idomain)
# water_content = processor.load_time_range(start_idx=0, end_idx=10)

# ParFlow Example
from PyHydroGeophysX.model_output import ParflowSaturation, ParflowPorosity # Corrected import path

# saturation_proc = ParflowSaturation("path/to/model_dir", "run_name")
# saturation = saturation_proc.load_timestep(100)
```
<!-- TODO: Ensure placeholder paths like "sim_workspace" are clear.
     Actual runnable examples should be in the examples/ directory. -->

### 2. Petrophysical Modeling
<!-- Example for using petrophysical models to link hydro and geophy properties. -->
Convert between hydrological and geophysical properties:

```python
from PyHydroGeophysX.petrophysics import (
    water_content_to_resistivity,
    HertzMindlinModel,
    DEMModel
)

# Example: Water content to resistivity using Waxman-Smits model
# wc = np.array([...]) # example water content
# porosity_map = np.array([...]) # example porosity
# resistivity = water_content_to_resistivity(
#     water_content=wc, rhos=100.0, n=2.0, porosity=porosity_map, sigma_sur=0.002
# )

# Example: Estimating seismic velocity using Hertz-Mindlin model
# porosity_array = np.array([...])
# saturation_array = np.array([...])
# hm_model = HertzMindlinModel()
# vp_high, vp_low = hm_model.calculate_velocity(
#     porosity=porosity_array, saturation=saturation_array,
#     bulk_modulus=30.0, shear_modulus=20.0, mineral_density=2650.0
# )
```

### 3. Forward Modeling
<!-- Example for generating synthetic ERT or SRT data. -->
Generate synthetic geophysical data:

```python
from PyHydroGeophysX.forward import ERTForwardModeling, SeismicForwardModeling
# Assuming 'mesh', 'data' (for ERT scheme), 'electrode_positions', 'resistivity_model' are predefined
# ert_fwd = ERTForwardModeling(mesh, data)
# synthetic_ert_data, _ = ert_fwd.create_synthetic_data( # Corrected: returns tuple
#     xpos=electrode_positions, res_models=resistivity_model
# )

# Assuming 'scheme' (for SRT), 'geophone_positions', 'velocity_model' are predefined
# srt_fwd = SeismicForwardModeling(mesh, scheme)
# synthetic_srt_data, _ = srt_fwd.create_synthetic_data( # Corrected: returns tuple
#     sensor_x=geophone_positions, velocity_model=velocity_model
# )
```

### 4. Time-Lapse Inversion
<!-- Example for performing time-lapse ERT inversion. -->
Perform sophisticated time-lapse ERT inversions:

```python
from PyHydroGeophysX.inversion import TimeLapseERTInversion, WindowedTimeLapseERTInversion

# Example: Full time-lapse inversion
# ert_files_list = ["data_t1.dat", "data_t2.dat", ...]
# measurement_times_list = [0.0, 1.0, ...]
# inversion = TimeLapseERTInversion(
#     data_files=ert_files_list,
#     measurement_times=measurement_times_list,
#     lambda_val=50.0,        # Spatial regularization strength
#     alpha=10.0,             # Temporal regularization strength
#     inversion_type="L2"     # Norm type: L1, L2, or L1L2
# )
# result = inversion.run()

# Example: Windowed inversion for very large datasets
# windowed_inv = WindowedTimeLapseERTInversion(
#     data_dir="path/to/data_directory/", ert_files=list_of_filenames,
#     measurement_times=list_of_times, window_size=3
# )
# result_windowed = windowed_inv.run(window_parallel=True) # Optionally run windows in parallel
```

### 5. Uncertainty Quantification
<!-- Example for Monte Carlo based uncertainty analysis. -->
Quantify uncertainty in water content estimates derived from ERT:

```python
from PyHydroGeophysX.Geophy_modular import ERTtoWC # Corrected import path

# Assuming 'mesh', 'resistivity_values_timelapse', 'cell_layer_markers', 'coverage_map' are predefined
# converter = ERTtoWC(mesh, resistivity_values_timelapse, cell_layer_markers, coverage_map)

# Define parameter distributions for different geological layers (mean and standard deviation)
layer_distributions_example = {
    1: {  # Example for layer/marker 1
        'rhos': {'mean': 100.0, 'std': 20.0},      # Saturated resistivity (Ω·m)
        'n': {'mean': 2.0, 'std': 0.1},            # Saturation exponent
        'sigma_sur': {'mean': 0.001, 'std': 0.0005},# Surface conductivity (S/m)
        'porosity': {'mean': 0.35, 'std': 0.05}    # Porosity (φ, fraction)
    },
    # Define for other layers as needed
}

# converter.setup_layer_distributions(layer_distributions_example)
# wc_realizations, sat_realizations, params_sampled = converter.run_monte_carlo(n_realizations=100)
# statistics = converter.get_statistics()  # e.g., mean, std, percentiles of water content
```
<!-- Note: Parameter names in layer_distributions example were slightly adjusted for consistency with typical petrophysical symbols. -->

## 📊 Example Workflows
<!-- High-level overview of combining functionalities. -->

### Complete Workflow: Hydrology to Geophysics
<!-- Illustrates an end-to-end simulation. -->
```python
# Import necessary modules from PyHydroGeophysX
# from PyHydroGeophysX.model_output import MODFLOWWaterContent
# from PyHydroGeophysX.core import ProfileInterpolator, MeshCreator
# from PyHydroGeophysX.petrophysics import water_content_to_resistivity
# from PyHydroGeophysX.forward import ERTForwardModeling
# from PyHydroGeophysX.inversion import ERTInversion

# 1. Load hydrological data (e.g., MODFLOW water content)
# idomain_data = np.loadtxt('idomain.txt') # Example idomain
# processor = MODFLOWWaterContent("path/to/modflow_dir", idomain_data)
# water_content_map = processor.load_timestep(timestep_idx=50) # Corrected arg name

# 2. Set up 2D profile for interpolation (if converting 3D hydro to 2D geophy)
# surface_elevation_data = np.loadtxt('surface_elevation.txt') # Example surface data
# interpolator = ProfileInterpolator(
#     point1=[115, 70], point2=[95, 180], # Example profile start/end indices
#     surface_data=surface_elevation_data
# )

# 3. Create mesh, possibly with geological structure
# surface_line_coords = ... # Define surface polyline
# layer1_boundary_coords = ... # Define first layer boundary
# layer2_boundary_coords = ... # Define second layer boundary
# mesh_creator = MeshCreator(quality=32.0) # Quality usually float
# # Assuming water_content_map and porosity_map are now on the cells of this new mesh
# # This step usually involves interpolating hydro output to this specific mesh.
# # The example implies direct use, which needs careful data mapping.
# geo_mesh, _ = mesh_creator.create_from_layers(
#      surface=surface_line_coords, layers=[layer1_boundary_coords, layer2_boundary_coords]
# )


# 4. Convert hydrological property (e.g., water content) to geophysical property (e.g., resistivity)
# This step requires water_content and porosity on the cells of 'geo_mesh'.
# resistivity_model_on_mesh = water_content_to_resistivity(
#     water_content_on_mesh, rhos=100.0, n=2.0, porosity=porosity_on_mesh
# )

# 5. Define electrode positions and generate synthetic ERT data
# electrode_positions_array = ...
# synthetic_ert_data, _ = ERTForwardModeling.create_synthetic_data(
#     xpos=electrode_positions_array[:,0], ypos=electrode_positions_array[:,1], # Assuming 2D positions
#     mesh=geo_mesh, res_models=resistivity_model_on_mesh
# )
# synthetic_ert_data.save("synthetic_ert_data.dat")


# 6. Invert synthetic ERT data
# inversion_setup = ERTInversion(data_file="synthetic_ert_data.dat", mesh=geo_mesh)
# inversion_result = inversion_setup.run()
```
<!-- Clarified that hydro properties need to be on the specific mesh. -->

### Structure-Constrained Inversion
<!-- Shows integration of seismic-derived structure into ERT inversion. -->
```python
# 1. Process seismic data to extract velocity structure
from PyHydroGeophysX.Geophy_modular import process_seismic_tomography, extract_velocity_structure # Corrected import

# travel_time_data_container = ... # Load or create pg.DataContainer for SRT
# seismic_inversion_manager = process_seismic_tomography(travel_time_data_container, lam=50.0)
# Assuming seismic_inversion_manager.model holds slowness, convert to velocity:
# velocity_model_srt = 1.0 / seismic_inversion_manager.model.array()
# interface_x_coords, interface_z_coords, _ = extract_velocity_structure(
#     seismic_inversion_manager.paraDomain, velocity_model_srt, threshold=1200.0
# )

# 2. Create ERT mesh incorporating the velocity interface as a structural constraint
from PyHydroGeophysX.Geophy_modular import create_ert_mesh_with_structure # Corrected import

# ert_data_container = ... # Load ERT data for mesh creation geometry
# constrained_ert_mesh, cell_markers, region_map = create_ert_mesh_with_structure(
#     ert_data_container, (interface_x_coords, interface_z_coords)
# )

# 3. Run ERT inversion using the structurally constrained mesh
# (Can be single time or time-lapse)
# from PyHydroGeophysX.inversion import ERTInversion # Or TimeLapseERTInversion
# list_of_ert_data_files = ["ert_data_t1.dat"] # Example for single time
# constrained_inversion = ERTInversion(
#     data_file=list_of_ert_data_files[0], mesh=constrained_ert_mesh
#     # Add other inversion parameters as needed, e.g., using region_map for regularization per region
# )
# result_constrained = constrained_inversion.run()
```
<!-- Clarified model type (velocity vs slowness) for extract_velocity_structure. -->

## 🛠 Advanced Features
<!-- Highlights features for more complex scenarios. -->

### GPU Acceleration
<!-- Example of enabling GPU for solvers. -->
Enable GPU acceleration for large-scale inversions (requires CuPy and compatible solvers):

```python
# from PyHydroGeophysX.inversion import TimeLapseERTInversion # Example
# Assuming 'list_of_files' and 'list_of_times' are defined
# inversion_gpu = TimeLapseERTInversion(
#     data_files=list_of_files,
#     measurement_times=list_of_times,
#     use_gpu=True,           # Attempt to use GPU acceleration
#     parallel=True,          # Also enable parallel CPU for parts that support it
#     n_jobs=-1               # Use all available CPU cores for parallel sections
# )
# result_gpu = inversion_gpu.run()
```

## 🤝 Contributing
<!-- Standard contribution guidelines. -->
We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details. <!-- TODO: Create CONTRIBUTING.md -->

- Fork the repository
- Create your feature branch (`git checkout -b feature/AmazingFeature`)
- Commit your changes (`git commit -m 'Add some AmazingFeature'`)
- Push to the branch (`git push origin feature/AmazingFeature`)
- Open a Pull Request

## 📝 Citation
<!-- How to cite the software if used in research. -->
If you use PyHydroGeophysX in your research, please cite:

```bibtex
@software{chen2025pyhydrogeophysx,
  author = {Chen, Hang and {Your Name if you contribute}},
  title = {PyHydroGeophysX: A Python Package for Integrating Hydrological and Geophysical Modeling},
  year = {2025}, <!-- TODO: Update year as appropriate -->
  publisher = {GitHub},
  url = {https://github.com/yourusername/PyHydroGeophysX} <!-- TODO: Update URL -->
}
```

## 📄 License
<!-- License information. -->
This project is licensed under the MIT License - see the `LICENSE` file for details.

## 🙏 Acknowledgments
<!-- Credits to other projects or communities. -->
- The PyGIMLi team for their excellent and comprehensive geophysical modeling framework.
- The communities behind MODFLOW, ParFlow, and other open-source tools that inspire and enable integrated modeling.

## 📧 Contact
<!-- Contact information and issue reporting. -->
Author: Hang Chen  
Email: hchen8@lbl.gov <!-- Verify or update email -->
Issues: Please report bugs or request features via [GitHub Issues](https://github.com/yourusername/PyHydroGeophysX/issues). <!-- TODO: Update URL -->

---

PyHydroGeophysX - Bridging the gap between hydrological models and geophysical monitoring.

*Note: This package is under active development. Please report any issues and feature requests through the GitHub issue tracker.*
<!-- TODO: Add a "Current Status" or "Roadmap" section if useful. -->
