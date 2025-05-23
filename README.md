# PyHydroGeophysX

A comprehensive Python package for integrating hydrological model outputs with geophysical forward modeling and inversion, specializing in electrical resistivity tomography (ERT) and seismic refraction tomography (SRT) for watershed monitoring applications.

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)  
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

## 🌟 Key Features

- **Hydrological Model Integration**: Seamless loading and processing of MODFLOW and ParFlow outputs  
- **Petrophysical Relationships**: Advanced models for converting between water content, saturation, resistivity, and seismic velocity  
- **Forward Modeling**: Complete ERT and SRT forward modeling capabilities with synthetic data generation  
- **Time-Lapse Inversion**: Sophisticated algorithms for time-lapse ERT inversion with temporal regularization  
- **Structure-Constrained Inversion**: Integration of seismic velocity interfaces for constrained ERT inversion  
- **Uncertainty Quantification**: Monte Carlo methods for parameter uncertainty assessment  
- **High-Performance Computing**: GPU acceleration and parallel processing support  

## 🔧 Main Components

### 1. Hydrological Model Integration  
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
2. Petrophysical Modeling
Convert between hydrological and geophysical properties:

python
Copy
Edit
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
3. Forward Modeling
Generate synthetic geophysical data:

python
Copy
Edit
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
4. Time-Lapse Inversion
Perform sophisticated time-lapse ERT inversions:

python
Copy
Edit
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
5. Uncertainty Quantification
Quantify uncertainty in water content estimates:

python
Copy
Edit
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
📊 Example Workflows
Complete Workflow: Hydrology to Geophysics
python
Copy
Edit
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
Structure-Constrained Inversion
python
Copy
Edit
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
🛠 Advanced Features
GPU Acceleration
python
Copy
Edit
# Enable GPU acceleration for large-scale inversions
inversion = TimeLapseERTInversion(
    data_files=files,
    use_gpu=True,           # Requires CuPy
    parallel=True,          # CPU parallelization
    n_jobs=-1               # Use all available cores
)
Custom Solver Configuration
python
Copy
Edit
from PyHydroGeophysX.solvers import CGLSSolver, TikhonvRegularization

# Configure custom solver
solver = CGLSSolver(
    max_iterations=200,
    tolerance=1e-8,
    use_gpu=True,
    damping=0.1
)

# Apply Tikhonov regularization
tikhonov = TikhonvRegularization(
    alpha=1e-3, 
    regularization_type='gradient'
)
📚 Documentation
Installation Guide: See docs/installation.rst

API Reference: Full API documentation available in docs/

Examples: Comprehensive examples in examples/

Tutorials: Step-by-step tutorials for common workflows

🧪 Examples
The examples/ directory contains comprehensive tutorials:

Ex1_model_output.py: Loading hydrological model outputs

Ex2_workflow.py: Complete workflow from hydrology to geophysics

Ex3_Time_lapse_measurement.py: Creating synthetic time-lapse data

Ex4_TL_inversion.py: Time-lapse inversion techniques

Ex5_SRT.py: Seismic refraction tomography

Ex6_Structure_resinv.py: Structure-constrained inversion

Ex7_structure_TLresinv.py: Structure-constrained time-lapse inversion

Ex8_MC_WC.py: Monte Carlo uncertainty quantification

🔗 Dependencies
Required

Python ≥ 3.8

NumPy

SciPy

matplotlib

PyGIMLI

joblib

tqdm

Optional

CuPy (for GPU acceleration)

flopy (for MODFLOW support)

parflow (for ParFlow support)

🤝 Contributing
We welcome contributions! Please see our Contributing Guidelines for details.

Fork the repository

Create a feature branch (git checkout -b feature/amazing-feature)

Commit your changes (git commit -m 'Add amazing feature')

Push to the branch (git push origin feature/amazing-feature)

Open a Pull Request

📄 License
This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

📞 Contact
Author: Hang Chen
Email: [your.email@domain.com]
Institution: [Your Institution]

🙏 Acknowledgments
PyGIMLI team for the excellent geophysical modeling framework

MODFLOW and ParFlow communities for hydrological modeling tools

Contributors and beta testers

📈 Citation
If you use PyHydroGeophysX in your research, please cite:

bibtex
Copy
Edit
@software{chen2025pyhydrogeophysx,
  title={PyHydroGeophysX: Integrated Hydrological-Geophysical Modeling for Watershed Monitoring},
  author={Chen, Hang},
  year={2025},
  url={https://github.com/yourusername/PyHydroGeophysX},
  version={0.1.0}
}
Note: This package is under active development. Please report issues and feature requests through the GitHub issue tracker.

yaml
Copy
Edit

---

Just select all, copy, and paste into your README.md. It will preserve formatting and code blocks perfectly.

If you want me to split it into smaller sections or add a TOC, just say!