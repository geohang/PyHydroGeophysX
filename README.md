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

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/PyHydroGeophysX.git
cd PyHydroGeophysX

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .# PyHydroGeophysX

'''


### 📦 Package Structure



PyHydroGeophysX/
├── core/                    # Core utilities
│   ├── interpolation.py    # Profile and mesh interpolation
│   └── mesh_utils.py       # Mesh creation and management
├── model_output/            # Hydrological model processors
│   ├── modflow_output.py   # MODFLOW data handling
│   └── parflow_output.py   # ParFlow data handling
├── petrophysics/           # Rock physics models
│   ├── resistivity_models.py  # Archie's law, Waxman-Smits
│   └── velocity_models.py     # DEM, Hertz-Mindlin models
├── forward/                # Forward modeling
│   ├── ert_forward.py      # ERT forward modeling
│   └── srt_forward.py      # Seismic forward modeling
├── inversion/              # Inversion algorithms
│   ├── base.py            # Base inversion classes
│   ├── ert_inversion.py   # Single-time ERT inversion
│   ├── time_lapse.py      # Time-lapse ERT inversion
│   └── windowed.py        # Windowed time-lapse inversion
├── solvers/               # Linear equation solvers
│   └── linear_solvers.py  # CGLS, LSQR, GPU-accelerated solvers
└── Geophy_modular/        # Geophysical processing modules
    ├── ERT_to_WC.py       # ERT to water content conversion
    └── seismic_processor.py # Seismic data processing