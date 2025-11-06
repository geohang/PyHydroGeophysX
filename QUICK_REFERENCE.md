# PyHydroGeophysX v0.2.0 - Quick Reference

## New ERT Data Processing Module

### Import
```python
from PyHydroGeophysX.data_processing.ert_data_agent import (
    load_ert_resipy, qc_and_visualize, export_for_inversion, LocalRef
)
```

### Load ERT Data

**Local Coordinates:**
```python
ert = load_ert_resipy(
    project_dir="data/ERT/E4D",
    data_file="data/ERT/E4D/survey.ohm",
    instrument="E4D",
    crs="local",
    local_ref=LocalRef(origin_x=0.0, origin_y=0.0, azimuth_deg=90.0)
)
```

**UTM Coordinates:**
```python
ert = load_ert_resipy(
    project_dir="data/ERT/Syscal",
    data_file="data/ERT/Syscal/survey.txt",
    instrument="Syscal",
    crs="EPSG:32615",  # UTM Zone 15N
    epsg=32615
)
```

**Geographic Coordinates:**
```python
ert = load_ert_resipy(
    project_dir="data/ERT",
    data_file="data/ERT/survey.dat",
    instrument="ABEM-Lund",
    crs="WGS84"
)
```

### Quality Control

```python
artifacts = qc_and_visualize(ert, outdir="results/qc")
# Creates:
# - rhoa_hist.png: Apparent resistivity histogram
# - pseudosection.png: 2D data visualization
# - data_summary.json: Statistical summary
```

### Export for Inversion

```python
bert_path = export_for_inversion(
    ert,
    outdir="results/inversion",
    fmt="pgimli",
    filename="survey_data.dat"
)
# Creates pyGIMLi/BERT format file for inversion
```

## Supported Instruments

| Instrument | Type | Format |
|------------|------|--------|
| E4D | Resistivity | .ohm |
| Syscal | Resistivity/IP | .txt |
| ABEM-Lund | Resistivity | Various |
| Sting | Resistivity/IP | .stg |
| ARES | Resistivity/IP | .ares |
| Protocol DC | Resistivity | .pro |
| Protocol IP | IP | .pro |
| BERT | Standard | .dat |
| DAS-1 | Resistivity | .dat |
| Electra | Resistivity | .dat |
| ResInv | Standard | .inv |
| PRIME/RESIMGR | Standard | Various |
| Lippmann | Resistivity | .dat |
| Custom | User-defined | Various |
| Merged | Combined | Various |

## Data Structures

### LocalRef
```python
LocalRef(
    origin_x=0.0,      # X-origin in world coords
    origin_y=0.0,      # Y-origin in world coords
    azimuth_deg=0.0    # Profile azimuth (deg from N)
)
```

### ERTDataset
```python
ert.electrodes         # List[Electrode]: Electrode positions
ert.observations       # List[Observation]: Measurements
ert.crs                # str: Coordinate system
ert.local_ref          # LocalRef | None
ert.metadata           # Dict: Survey metadata
```

### Access Data
```python
# Number of electrodes and measurements
n_electrodes = len(ert.electrodes)
n_measurements = len(ert.observations)

# First electrode
e = ert.electrodes[0]
print(f"Electrode {e.id}: ({e.x}, {e.y}, {e.z})")

# First observation
obs = ert.observations[0]
print(f"A={obs.quad.A}, B={obs.quad.B}, M={obs.quad.M}, N={obs.quad.N}")
print(f"Apparent resistivity: {obs.app_res} Ω·m")
print(f"Error: {obs.err}")
```

## Complete Workflow

```python
# 1. Import
from PyHydroGeophysX.data_processing.ert_data_agent import *

# 2. Load
ert = load_ert_resipy(
    project_dir="data/ERT/E4D",
    data_file="data/ERT/E4D/2021-10-08_1400.ohm",
    instrument="E4D",
    crs="local",
    local_ref=LocalRef(0, 0, 90)
)

# 3. QC
artifacts = qc_and_visualize(ert, outdir="results/qc")

# 4. Export
bert_path = export_for_inversion(ert, outdir="results", fmt="pgimli")

# 5. Verify
print(f"Processed {len(ert.observations)} measurements")
print(f"Exported to: {bert_path}")
```

## Time-Lapse Processing

```python
from pathlib import Path

data_files = [
    "2021-10-08_1400.ohm",
    "2021-10-09_1400.ohm",
    "2021-10-10_1400.ohm",
]

for i, data_file in enumerate(data_files):
    ert = load_ert_resipy(
        project_dir="data/ERT/E4D",
        data_file=f"data/ERT/E4D/{data_file}",
        instrument="E4D",
        crs="local",
        local_ref=LocalRef(0, 0, 90)
    )
    
    bert_path = export_for_inversion(
        ert,
        outdir="results/time_lapse",
        filename=f"t{i:03d}_{Path(data_file).stem}.dat"
    )
    
    print(f"Timestep {i}: {bert_path}")
```

## Installation

**Standard:**
```bash
pip install PyHydroGeophysX
```

**With ERT processing (includes RESIPY):**
```bash
pip install PyHydroGeophysX[geophysics]
```

**Development:**
```bash
git clone https://github.com/geohang/PyHydroGeophysX.git
cd PyHydroGeophysX
pip install -e .[all]
```

## Common Issues

### RESIPY not found
```bash
pip install resipy
```

### Permission denied (OneDrive)
Function handles automatically with tempfile fallback

### Column not found (app/rhoa)
Function detects flexibly: 'app', 'rhoa', 'Rho'

### Unix paths on Windows
Function handles automatically: `data/ERT/file.ohm`

## Documentation

- **Full API:** https://geohang.github.io/PyHydroGeophysX/api/data_processing.html
- **Examples:** `examples/Ex_ERT_data_process.ipynb`
- **GitHub:** https://github.com/geohang/PyHydroGeophysX

## Support

- **Issues:** https://github.com/geohang/PyHydroGeophysX/issues
- **Email:** hangchen.work@gmail.com

---
**Version:** 0.2.0 | **Date:** 2025-11-06
