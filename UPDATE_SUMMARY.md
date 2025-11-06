# PyHydroGeophysX v0.2.0 Update Summary

## Overview

PyHydroGeophysX has been updated to **version 0.2.0** with major new functionality for ERT field data processing. The package now provides a complete workflow from field data acquisition to inversion-ready formats.

## What's New in v0.2.0

### 🆕 New Module: `data_processing`

A complete pipeline for handling field ERT data with RESIPY integration:

**Key Functions:**
- `load_ert_resipy()`: Load data from 14+ commercial ERT instruments
- `qc_and_visualize()`: Automated quality control with diagnostic plots
- `export_for_inversion()`: Export to pyGIMLi/BERT format

**Supported Instruments:**
- E4D, Syscal, ABEM-Lund, Sting, ARES
- Protocol DC/IP, BERT, DAS-1, Electra
- ResInv, PRIME/RESIMGR, Lippmann
- Custom and merged datasets

**Features:**
- ✅ Flexible coordinate systems (local, projected, geographic)
- ✅ Automatic error handling for Windows/OneDrive
- ✅ Cross-platform path support
- ✅ Robust column detection for different instruments
- ✅ Complete metadata preservation

## Files Updated

### Documentation
1. **README.md**
   - Added ERT data processing section in Quick Start
   - Updated package structure diagram
   - Added Ex_ERT_data_process to examples list
   - Highlighted new functionality in Key Features

2. **docs/source/api/data_processing.rst** (NEW)
   - Comprehensive API reference for ert_data_agent module
   - Detailed function documentation
   - Code examples and workflows
   - Time-lapse survey processing guide

3. **docs/source/api/index.rst**
   - Added data_processing to module list

4. **docs/source/quickstart.rst**
   - Added ERT field data processing example
   - Installation instructions for RESIPY
   - Quick workflow demonstration

5. **CHANGELOG.md** (NEW)
   - Complete version history
   - Detailed list of changes for v0.2.0
   - Breaking changes and migration guide

6. **RELEASE.md** (NEW)
   - Step-by-step PyPI release guide
   - Testing procedures
   - Troubleshooting tips

### Configuration
7. **setup.py**
   - Version: 0.1.0 → **0.2.0**
   - Added `resipy>=3.4.0` to geophysics extras

8. **pyproject.toml**
   - Version: 0.1.0 → **0.2.0**
   - Added `resipy>=3.4.0` to geophysics extras

## Usage Example

```python
from PyHydroGeophysX.data_processing.ert_data_agent import (
    load_ert_resipy, qc_and_visualize, export_for_inversion, LocalRef
)

# 1. Load field data
ert = load_ert_resipy(
    project_dir="data/ERT/E4D",
    data_file="data/ERT/E4D/2021-10-08_1400.ohm",
    instrument="E4D",
    crs="local",
    local_ref=LocalRef(origin_x=0.0, origin_y=0.0, azimuth_deg=90.0)
)

# 2. Generate QC plots
artifacts = qc_and_visualize(ert, outdir="results/qc")
# Creates: rhoa_hist.png, pseudosection.png, data_summary.json

# 3. Export for inversion
bert_path = export_for_inversion(ert, outdir="results/inversion", fmt="pgimli")
# Creates: bert_data.dat (pyGIMLi/BERT format)

print(f"Processed {len(ert.observations)} measurements from {len(ert.electrodes)} electrodes")
```

## Installation

### For Users

**Basic installation:**
```bash
pip install PyHydroGeophysX
```

**With ERT data processing:**
```bash
pip install PyHydroGeophysX[geophysics]
```

**With all features:**
```bash
pip install PyHydroGeophysX[all]
```

### For Developers

```bash
git clone https://github.com/geohang/PyHydroGeophysX.git
cd PyHydroGeophysX
pip install -e .[all]
```

## Publishing to PyPI

Follow the steps in `RELEASE.md`:

1. **Build distribution:**
   ```powershell
   python -m build
   ```

2. **Test on TestPyPI:**
   ```powershell
   python -m twine upload --repository testpypi dist/*
   ```

3. **Upload to PyPI:**
   ```powershell
   python -m twine upload dist/*
   ```

4. **Create GitHub Release:**
   ```powershell
   git tag -a v0.2.0 -m "Version 0.2.0"
   git push origin v0.2.0
   ```

## Documentation Website

### Building Locally

```powershell
cd docs
.\make.bat clean
.\make.bat html
```

Open `docs\build\html\index.html` in browser.

### Deploying to GitHub Pages

```powershell
cd docs
.\make.bat clean
.\make.bat html

git checkout gh-pages
Copy-Item -Recurse -Force docs\build\html\* .
git add .
git commit -m "Update docs for v0.2.0"
git push origin gh-pages
git checkout main
```

## Testing Checklist

Before release, verify:

- [ ] All imports work correctly
  ```python
  from PyHydroGeophysX.data_processing.ert_data_agent import load_ert_resipy
  ```

- [ ] Example notebook runs successfully
  ```powershell
  jupyter notebook examples/Ex_ERT_data_process.ipynb
  ```

- [ ] Documentation builds without errors
  ```powershell
  cd docs; .\make.bat html
  ```

- [ ] Tests pass (if available)
  ```powershell
  pytest tests/
  ```

- [ ] Package installs from TestPyPI
  ```powershell
  pip install --index-url https://test.pypi.org/simple/ PyHydroGeophysX
  ```

## Migration Guide

### For Existing Users

No breaking changes! All existing code continues to work. The new `data_processing` module is an addition, not a replacement.

To use new features, install RESIPY:
```bash
pip install resipy
```

### New Dependencies

- `resipy>=3.4.0` (optional, for ERT data processing)

## Community

- **Issues:** https://github.com/geohang/PyHydroGeophysX/issues
- **Discussions:** https://github.com/geohang/PyHydroGeophysX/discussions
- **Documentation:** https://geohang.github.io/PyHydroGeophysX/
- **Email:** hangchen.work@gmail.com

## Acknowledgments

This update builds on the excellent RESIPY library for ERT data processing. Special thanks to:
- **RESIPY development team** (Guillaume Blanchy, Jimmy Boyd, and contributors) for creating an intuitive and powerful ERT data processing framework
- **PyGIMLi community** for the geophysical modeling infrastructure
- All contributors and users providing feedback

**RESIPY Citation:**

Blanchy, G., Saneiyan, S., Boyd, J., McLachlan, P., & Binley, A. (2020). ResIPy, an intuitive open source software for complex geoelectrical inversion/modeling. *Computers & Geosciences*, 137, 104423. https://doi.org/10.1016/j.cageo.2020.104423

## Next Steps

1. **Publish to PyPI** following RELEASE.md guide
2. **Update GitHub repository** with new tag and release
3. **Deploy documentation** to GitHub Pages
4. **Announce release** to community

---

**Version:** 0.2.0  
**Release Date:** November 6, 2025  
**Maintainer:** Hang Chen (hangchen.work@gmail.com)
