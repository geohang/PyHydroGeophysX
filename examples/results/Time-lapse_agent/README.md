# Time-Lapse Agent Results Directory

This directory contains all outputs from the time-lapse ERT inversion workflow with climate data integration.

## Workflow Source

Generated from: `Ex_TimeLapse_NaturalLanguage.ipynb`

## Contents

### Inversion Results
- **Resistivity models**: Time-lapse resistivity models for each survey date
- **Mesh files**: PyGIMLi mesh data
- **Inversion parameters**: Lambda values, iterations, convergence data

### Visualizations
- `time_lapse_changes.png` - Resistivity change maps (2×2 grid showing temporal changes)
- `climate_resistivity_comparison.png` - Climate-ERT correlation plots (4-panel comparison)
- `optimization_history.png` - Inversion quality improvement over optimization attempts

### Reports
- `report.md` - Comprehensive markdown report with:
  - Time-lapse inversion results
  - Climate data analysis
  - Correlation findings
  - AI-powered interpretation
  - Recommendations

- `report.html` - Interactive HTML version (recommended for viewing)
  - Better formatting and styling
  - Embedded figures
  - Easy to share and present

### Analysis Data
- **Correlation data**: Resistivity vs climate variable correlations
- **Statistical summaries**: Mean changes, extrema, temporal evolution
- **Quality metrics**: Chi-squared values, RMS errors, convergence info

## Study Site

**Location**: Mt. Snodgrass, near Crested Butte, Colorado  
**Coordinates**: 38.92584°N, -106.97998°W (WGS84)  
**Elevation**: ~3,150 m  
**Study Period**: October 2021 - February 2022

## Survey Details

**ERT Surveys** (5 time steps):
1. 2021-10-08 (Baseline)
2. 2021-11-08
3. 2021-12-08
4. 2022-01-08
5. 2022-02-08

**Instrument**: E4D  
**Inversion Method**: Difference method with temporal regularization  
**Climate Data**: Precipitation, temperature, PET from Daymet

## Key Analyses

### 1. Time-Lapse Resistivity Changes
- Baseline comparison approach
- Temporal regularization to reduce noise
- Spatial patterns of resistivity changes

### 2. Climate Data Integration
- Daily precipitation events
- 7-day antecedent precipitation
- Temperature variations
- Potential evapotranspiration (PET)
- Moisture balance (P-PET)

### 3. Correlation Analysis
- Resistivity change vs. precipitation
- Resistivity change vs. temperature
- Resistivity change vs. moisture balance
- Lag effects and antecedent conditions

### 4. AI-Powered Interpretation
- Automated analysis of patterns
- Physical process interpretation
- Recommendations for further study

## How to View Results

### Best Option: HTML Report
```bash
# Open in browser
start report.html  # Windows
open report.html   # macOS
xdg-open report.html  # Linux
```

### Visualizations
Open PNG files in any image viewer. High-resolution (300 DPI) suitable for publications.

### Raw Data
- Use Python/pandas to load CSV files
- Use PyGIMLi to load mesh and model files
- Use matplotlib to recreate/modify plots

## Regenerating Results

To regenerate these results:

1. Open `Ex_TimeLapse_NaturalLanguage.ipynb`
2. Run all cells in order
3. Ensure climate data is available (see climate fetching instructions in notebook)
4. Results will be saved in this directory

## File Organization

```
Time-lapse_agent/
├── README.md (this file)
├── report.md
├── report.html
├── time_lapse_changes.png
├── climate_resistivity_comparison.png
├── optimization_history.png (if optimization was performed)
├── models/
│   ├── baseline_model.dat
│   ├── timestep_1_model.dat
│   ├── ...
├── mesh/
│   └── mesh.bms
└── data/
    ├── correlation_analysis.csv
    └── temporal_statistics.csv
```

## Citations

If you use these results in publications, please cite:

- PyHydroGeophysX package
- PyGIMLi (geophysical inversion)
- Daymet (climate data)
- Relevant publication describing your study

## Questions or Issues?

- Check the notebook for detailed workflow steps
- See `NOTEBOOK_ENHANCEMENTS.md` for feature documentation
- Review `CLIMATE_DATA_WORKFLOW.md` for climate data setup

---

**Generated**: Run date will be in report files  
**Notebook Version**: Ex_TimeLapse_NaturalLanguage.ipynb  
**PyHydroGeophysX**: Cross-modal geophysics agent system
