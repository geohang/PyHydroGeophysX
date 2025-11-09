# Time-Lapse Report Integration Summary

**Date:** 2024
**Feature:** Consolidated time-lapse reporting in ReportAgent

---

## Overview

This document summarizes the integration of time-lapse ERT reporting functionality into the `ReportAgent`, moving visualization and analysis code from notebooks into a reusable, modular reporting system.

## Changes Made

### 1. ReportAgent Enhancement (report_agent.py)

Added **NEW** time-lapse reporting module without modifying existing code:

#### New Method: `generate_timelapse_report()`
Main entry point for time-lapse report generation. Coordinates all report sections and visualizations.

**Input Parameters:**
```python
{
    'inversion_results': Dict,        # Time-lapse inversion results with final_models, mesh, coverage, chi2_values
    'climate_data': Dict,             # Optional: Climate data results
    'site_info': Dict,                # Site information (name, location, coordinates, etc.)
    'comparison_data': DataFrame,     # Optional: Climate-resistivity comparison data
    'output_dir': str,                # Output directory (default: 'results/Time-lapse_agent')
    'inversion_mode': 'time-lapse',
    'time_lapse_method': str          # 'difference', 'joint', or 'ratio'
}
```

**Returns:**
```python
{
    'status': 'success' or 'failed',
    'report_file': str,               # Path to markdown report
    'html_file': str,                 # Path to HTML report
    'visualization_files': Dict,      # Paths to generated plots
    'executive_summary': str,         # Summary text
    'output_dir': str
}
```

#### Supporting Methods

1. **`_generate_timelapse_executive_summary()`**
   - Site information summary
   - Monitoring objectives
   - Method and configuration details
   - Overall quality metrics

2. **`_generate_timelapse_inversion_section()`**
   - Methodology explanation (difference/joint/ratio)
   - Inversion parameters
   - Convergence and chi-squared values
   - Temporal resistivity statistics
   - Resistivity changes by time step

3. **`_generate_timelapse_climate_section()`**
   - Climate data metadata
   - Date range and variables
   - ERT survey alignment
   - Climate conditions at survey times

4. **`_generate_timelapse_correlation_section()`**
   - Correlation coefficients calculation
   - Climate variable relationships
   - Interpretation guidelines
   - Key findings summary

5. **`_generate_timelapse_visualizations()`**
   - Time-lapse resistivity change maps
   - Climate-resistivity comparison plots
   - Uses PyGIMLi `pg.show()` with parameters:
     - Colormap: `RdBu_r` for changes
     - Coverage masking enabled
     - Vertical orientation
     - Professional formatting (Arial font, 300 DPI)

6. **`_generate_timelapse_narrative()`**
   - LLM-enhanced interpretation
   - Integrates all sections
   - Provides professional narrative

7. **`_compile_timelapse_report()`**
   - Combines all sections
   - Adds visualizations
   - Generates recommendations
   - Creates markdown report

#### Visualization Details

**Time-Lapse Resistivity Changes:**
- 2x2 subplot grid showing up to 4 time steps
- Difference from baseline (Δρ in Ω·m)
- Colormap: `RdBu_r` (red=increase, blue=decrease)
- Range: ±50 Ω·m
- Coverage masking applied
- Labels: Distance (m) vs. Elevation (m)

**Climate-Resistivity Correlation:**
- 4-panel figure:
  1. **Top-left:** Mean resistivity change over time
  2. **Top-right:** Daily and 7-day antecedent precipitation
  3. **Bottom-left:** Temperature and PET
  4. **Bottom-right:** Moisture balance (P-PET) vs. resistivity
- Dual y-axes for related variables
- Grid enabled, professional formatting

### 2. Notebook Updates (Ex_TimeLapse_NaturalLanguage.ipynb)

#### Removed Cells
- **Cell 18 (former #VSC-41e5b4e6):** Time-lapse visualization code
- **Cell 20 (former #VSC-c4cc998e):** Temporal statistics calculations
- **Cell 25 (former #VSC-a060553c):** Climate-resistivity correlation analysis

#### Added/Modified Cells

**New Section 7: Generate Comprehensive Time-Lapse Report**
```python
from PyHydroGeophysX.agents import ReportAgent

# Initialize report agent
report_agent = ReportAgent(api_key=api_key, llm_provider=llm_provider, llm_model=llm_model)

# Prepare input
report_input = {
    'inversion_results': results_diff,
    'climate_data': workflow_config_diff.get('climate_data'),
    'site_info': {...},
    'output_dir': 'results/Time-lapse_agent',
    'inversion_mode': 'time-lapse',
    'time_lapse_method': workflow_config_diff.get('time_lapse_method', 'difference')
}

# Generate report
report_result = report_agent.generate_timelapse_report(report_input)
```

**Updated Section 9: Optional Climate-Resistivity Update**
- If climate data manually fetched, can regenerate report
- Calculates comparison DataFrame
- Calls `generate_timelapse_report()` again with climate integration

**Updated Final Summary:**
- Reflects new workflow structure
- Documents all features and outputs
- Provides usage tips and advanced customization

---

## Key Benefits

### 1. Modularity
- **Reusable:** Report generation logic separated from notebooks
- **Maintainable:** Single source of truth for report generation
- **Extensible:** Easy to add new analysis modules

### 2. Professional Output
- **Consistent Formatting:** Arial font, 300 DPI, standardized layouts
- **Publication-Ready:** High-quality figures and reports
- **LLM-Enhanced:** Intelligent interpretation and recommendations

### 3. Workflow Efficiency
- **Automated:** Single method call generates complete report
- **Comprehensive:** All visualizations, statistics, and analysis included
- **Flexible:** Optional climate integration, customizable parameters

### 4. Backward Compatibility
- **Existing Code Unchanged:** Standard ERT workflows unaffected
- **New Module Only:** Time-lapse module is completely separate
- **No Breaking Changes:** All existing examples continue to work

---

## Usage Example

### Basic Time-Lapse Report (No Climate)
```python
from PyHydroGeophysX.agents import ReportAgent

report_agent = ReportAgent(api_key=api_key, llm_provider='openai', llm_model='gpt-4')

report_input = {
    'inversion_results': {
        'final_models': final_models,  # numpy array (n_cells, n_timesteps)
        'mesh': mesh,                  # PyGIMLi mesh object
        'coverage': coverage,           # numpy array
        'chi2_values': [1.2, 1.1, 1.0, 0.9],
        'n_timesteps': 4,
        'temporal_regularization': 1.0,
        'lambda': 30,
        'max_iterations': 20,
        'method': 'lsqr'
    },
    'site_info': {
        'name': 'Mt. Snodgrass',
        'location': 'Crested Butte, Colorado',
        'coordinates': '38.869°N, 106.964°W',
        'elevation': '3,291 m',
        'study_period': '2021-10-08 to 2022-02-08',
        'description': 'High-elevation time-lapse ERT monitoring'
    },
    'output_dir': 'results/Time-lapse_agent',
    'inversion_mode': 'time-lapse',
    'time_lapse_method': 'difference'
}

result = report_agent.generate_timelapse_report(report_input)
print(f"Report: {result['report_file']}")
```

### With Climate Integration
```python
# Add climate data and comparison DataFrame
report_input['climate_data'] = climate_results
report_input['comparison_data'] = comparison_df  # pandas DataFrame with Date, resistivity changes, climate vars

result = report_agent.generate_timelapse_report(report_input)
# Now includes climate-resistivity correlation analysis
```

---

## Output Structure

```
results/Time-lapse_agent/
├── time_lapse_report.md                       # Markdown report
├── time_lapse_report.html                     # HTML report (recommended)
├── timelapse_resistivity_changes.png          # 2x2 resistivity change maps
└── climate_resistivity_correlation.png        # 4-panel climate analysis (if available)
```

### Report Sections

1. **Executive Summary**
   - Site information
   - Monitoring objectives
   - Method and configuration
   - Overall quality metrics

2. **Integrated Analysis (LLM-Enhanced)**
   - Cohesive narrative interpretation
   - Key findings synthesis
   - Recommendations

3. **Time-Lapse Inversion Results**
   - Methodology explanation
   - Inversion parameters
   - Convergence metrics
   - Temporal statistics

4. **Climate Data Integration** (if available)
   - Climate metadata
   - Survey alignment
   - Climate conditions summary

5. **Climate-Resistivity Correlation** (if available)
   - Correlation coefficients
   - Interpretation guidelines
   - Key findings

6. **Visualizations**
   - Embedded figures with captions

7. **Summary and Recommendations**
   - Key findings recap
   - Future monitoring suggestions
   - Validation recommendations

---

## Comparison DataFrame Format

For climate-resistivity correlation analysis:

```python
import pandas as pd

comparison_df = pd.DataFrame({
    'Date': ['2021-11-08', '2021-12-08', '2022-01-08', '2022-02-08'],
    'Mean_Resistivity_Change_Ohm_m': [-15.2, -8.5, 12.3, 5.7],
    'Precipitation_mm': [25.4, 12.1, 5.3, 18.9],
    'Precip_7d_mm': [65.2, 45.8, 28.3, 52.1],
    'Temp_Mean_C': [2.5, -5.2, -8.1, -2.3],
    'PET_mm': [0.8, 0.5, 0.3, 0.6],
    'P_minus_PET_mm': [24.6, 11.6, 5.0, 18.3]
})
```

---

## Technical Details

### Dependencies
- **PyGIMLi:** For mesh visualization with `pg.show()`
- **Matplotlib:** For multi-panel plots and figure generation
- **NumPy:** For statistical calculations
- **Pandas:** For tabular data handling
- **LLM (Optional):** For narrative generation

### Visualization Parameters

**Time-Lapse Changes:**
```python
pg.show(
    mesh,
    change,
    cMap='RdBu_r',
    cMin=-50,
    cMax=50,
    label=r'$\Delta\rho$ ($\Omega \cdot m$)',
    orientation='vertical',
    coverage=coverage_mask
)
```

**Figure Settings:**
```python
matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['font.size'] = 12
fig.savefig(..., dpi=300, bbox_inches='tight')
```

### Statistical Metrics

Calculated for each time step:
- Mean resistivity change
- Maximum decrease (moisture increase indicator)
- Maximum increase (drying/freezing indicator)
- Standard deviation
- Range

Correlation coefficients computed for:
- Daily precipitation vs. resistivity
- Antecedent precipitation (7-day) vs. resistivity
- Temperature vs. resistivity
- Moisture balance (P-PET) vs. resistivity

---

## Future Enhancements

### Potential Additions
1. **Depth-Dependent Analysis:** Layer-by-layer statistics
2. **Animation Generation:** GIF/video of temporal changes
3. **Advanced Correlation:** Lag analysis, cross-correlation
4. **Model Comparison:** Compare different inversion methods
5. **Uncertainty Quantification:** Bootstrap confidence intervals
6. **3D Visualization:** For 3D ERT datasets

### Customization Options
- Custom colormap selection
- User-defined resistivity ranges
- Alternative statistical metrics
- Additional climate variables
- Custom correlation methods

---

## Testing Recommendations

### Test Cases
1. **Basic report** (no climate): Verify all sections generated
2. **With climate:** Verify correlation analysis included
3. **Multiple time steps:** Test with 2, 4, 8+ time steps
4. **Different methods:** Test difference, joint, ratio inversions
5. **Error handling:** Test with missing data, invalid inputs

### Validation Checks
- [ ] All visualizations generated
- [ ] Statistics calculated correctly
- [ ] Correlations computed accurately
- [ ] HTML report renders properly
- [ ] File paths correct and accessible
- [ ] LLM narrative coherent (if enabled)
- [ ] No errors in console output

---

## Documentation References

Related documentation files:
- `CONTEXT_AGENT_ENHANCEMENT.md` - Parameter extraction
- `CLIMATE_AUTO_EXTRACTION.md` - Climate configuration
- `NATURAL_LANGUAGE_REQUEST_GUIDE.md` - Request format
- `ENHANCEMENT_COMPLETE_SUMMARY.md` - Overall summary

---

## Backward Compatibility

### Existing Code Unaffected
✅ **Standard ERT workflows:** `execute()` method unchanged  
✅ **Other examples:** All existing notebooks work as before  
✅ **Report generation:** Original `execute()` still generates standard reports  
✅ **API:** No breaking changes to existing methods

### New Code Only
The time-lapse reporting is a **completely separate module**:
- New method: `generate_timelapse_report()`
- New supporting methods: All prefixed with `_generate_timelapse_*`
- New visualization logic: Specific to time-lapse workflows
- New report format: Tailored for temporal monitoring

Users can continue using existing workflows without any changes. The time-lapse module is **opt-in only**.

---

## Summary

This integration provides a **professional, reusable, and modular** solution for time-lapse ERT reporting. By moving visualization and analysis logic from notebooks to the ReportAgent, we achieve:

✅ **Consistency:** Single implementation across all projects  
✅ **Quality:** Publication-ready figures and reports  
✅ **Efficiency:** One method call generates everything  
✅ **Maintainability:** Easy to update and extend  
✅ **Flexibility:** Optional climate integration  
✅ **Compatibility:** Existing code unaffected

The ReportAgent now serves as a **comprehensive reporting engine** for both standard ERT inversions and time-lapse monitoring workflows, providing users with automated, professional-quality reports with minimal effort.

---

**Implementation Complete!** 🎉

For questions or issues, refer to the comprehensive inline documentation in `report_agent.py` or contact the development team.
