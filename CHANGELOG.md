# Changelog

All notable changes to PyHydroGeophysX will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-11-06

### Added
- **New `data_processing` module** for field ERT data processing
  - `ert_data_agent.py`: Standardized interface for loading, QC, and exporting ERT field data
  - `load_ert_resipy()`: Load ERT data from 14+ commercial instruments via RESIPY integration
    - Supported instruments: E4D, Syscal, ABEM-Lund, Sting, ARES, Protocol DC/IP, BERT, and more
    - Flexible coordinate reference systems: local, projected (EPSG), geographic (WGS84)
    - Automatic handling of Windows/OneDrive permission issues
    - Unix-style path support on Windows
    - Flexible column name detection for different instrument outputs
  - `qc_and_visualize()`: Generate diagnostic plots and summary statistics
    - Apparent resistivity histograms (log-scale)
    - Pseudosection visualizations
    - JSON summary with statistical metrics
  - `export_for_inversion()`: Export to pyGIMLi/BERT format for inversion
    - Unified data format with electrode coordinates
    - Complete measurement data (13 columns including geometric factors, resistance, validity)
  - New data structures: `LocalRef`, `Electrode`, `Quadruplet`, `Observation`, `ERTDataset`
  - Complete example notebook: `Ex_ERT_data_process.ipynb`

### Changed
- Updated package structure to include `data_processing/` module
- Enhanced README with ERT data processing workflow examples
- Expanded documentation with new API reference for data processing module
- Updated quickstart guide with field data processing examples

### Documentation
- Added comprehensive API documentation for `data_processing` module
- Created `docs/source/api/data_processing.rst` with detailed function reference
- Updated main documentation index to include data processing
- Added workflow examples for time-lapse ERT surveys
- Documented all 14+ supported instrument types with usage examples
- Added RESIPY citation in README and documentation

### Dependencies
- Added `resipy>=3.4.0` as optional dependency under `geophysics` extras
- Maintained backward compatibility with existing installations

### Fixed
- RESIPY API compatibility across different versions
- Permission errors with OneDrive directories (automatic tempfile fallback)
- Path handling for Unix-style paths on Windows
- Flexible column detection for apparent resistivity and error measurements

## [0.1.0] - 2024

### Added
- Initial release of PyHydroGeophysX
- Hydrological model integration (MODFLOW, ParFlow)
- Petrophysical relationships (Archie, Waxman-Smits, DEM, Hertz-Mindlin)
- Forward modeling capabilities (ERT, SRT)
- Time-lapse ERT inversion with temporal regularization
- Structure-constrained inversion using seismic interfaces
- Monte Carlo uncertainty quantification
- GPU acceleration support (CUDA/CuPy)
- Parallel processing capabilities
- Advanced linear solvers (CGLS, LSQR, RRLS)
- Comprehensive example notebooks and scripts
- Documentation website with Sphinx

### Core Modules
- `core/`: Interpolation, kriging, mesh utilities, plotting tools
- `model_output/`: MODFLOW and ParFlow data loaders
- `petrophysics/`: Resistivity and velocity models
- `forward/`: ERT and SRT forward modeling
- `inversion/`: Single-time, time-lapse, and windowed inversion
- `solvers/`: Linear algebra solvers with GPU support
- `Hydro_modular/`: Direct hydro-to-geophysics conversion
- `Geophy_modular/`: Geophysical data processing and structure integration

[0.2.0]: https://github.com/geohang/PyHydroGeophysX/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/geohang/PyHydroGeophysX/releases/tag/v0.1.0
