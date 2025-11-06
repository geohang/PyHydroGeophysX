# Multi-Agent System for Automated Geophysical Workflows

This module provides a GPT API-based multi-agent system for automating geophysical data processing workflows in PyHydroGeophysX.

## Overview

The multi-agent system coordinates specialized AI agents to execute the complete workflow:

**"load ERT → invert → convert to water content → report"**

With optional seismic data integration for structure-constrained inversion and climate data integration for hydrologic context.

## Architecture

### Agent Coordinator
- `AgentCoordinator`: Orchestrates the complete workflow and manages agent communication

### Specialized Agents
1. **ERTLoaderAgent**: Loads and quality-checks ERT field data from various instruments
2. **ERTInversionAgent**: Performs ERT inversion with optional structural constraints
3. **WaterContentAgent**: Converts resistivity to water content using petrophysical models
4. **ReportAgent**: Generates comprehensive reports with visualizations
5. **SeismicAgent** (optional): Processes seismic refraction data for structural constraints
6. **ClimateDataAgent** (optional): Fetches meteorological data and computes PET for hydrologic analysis

## Key Features

- 🤖 **AI-Enhanced**: Each agent uses LLM API (GPT/Gemini/Claude) for intelligent parameter selection and interpretation
- 🔄 **Automated Workflow**: Complete end-to-end processing with minimal user input
- 📊 **Quality Control**: Automatic data validation and quality metrics
- 🧪 **Uncertainty Quantification**: Monte Carlo analysis for water content estimates
- 📈 **Comprehensive Reports**: Automatic generation of reports with visualizations
- 🌊 **Seismic Integration**: Optional structure-constrained inversion using seismic data
- ☁️ **Climate Integration**: Optional meteorological data and PET for hydrologic context

## Installation

The multi-agent system requires the LLM API packages:

```bash
# For OpenAI GPT
pip install openai>=1.0.0

# For Google Gemini
pip install google-generativeai>=0.3.0

# For Anthropic Claude
pip install anthropic>=0.18.0

# For climate data integration
pip install pydaymet>=0.16.0 pandas>=1.3.0 xarray>=0.19.0
```

For full functionality, install PyHydroGeophysX with all dependencies:

```bash
pip install pyhydrogeophysx[all]
```

## Configuration

Set your LLM API key as an environment variable based on the provider you want to use:

```bash
# For OpenAI GPT
export OPENAI_API_KEY='your-api-key-here'

# For Google Gemini
export GEMINI_API_KEY='your-api-key-here'

# For Anthropic Claude
export ANTHROPIC_API_KEY='your-api-key-here'
```

Or pass it directly when creating agents:

```python
# Choose your LLM provider
coordinator = AgentCoordinator(
    api_key='your-api-key',
    llm_provider='openai'  # or 'gemini', 'claude'
)
```

**Note**: The system works without an API key, but LLM-enhanced features (parameter recommendations, interpretations) will be disabled.

## Usage

### Basic Workflow

```python
from PyHydroGeophysX.agents import (
    AgentCoordinator,
    ERTLoaderAgent,
    ERTInversionAgent,
    WaterContentAgent,
    ReportAgent
)

# Initialize coordinator
coordinator = AgentCoordinator(output_dir="results/agents")

# Register agents
coordinator.register_agent('ert_loader', ERTLoaderAgent())
coordinator.register_agent('ert_inversion', ERTInversionAgent())
coordinator.register_agent('water_content', WaterContentAgent())
coordinator.register_agent('report', ReportAgent())

# Configure workflow
config = {
    'data_file': 'data/ERT/survey.dat',
    'instrument': 'E4D',
    'inversion_params': {
        'lambda': 20.0,
        'max_iterations': 10
    },
    'run_uncertainty': True,
    'n_realizations': 100
}

# Execute workflow
results = coordinator.execute_workflow(config)

# Access results
if results['status'] == 'success':
    wc = results['results']['water_content']
    report = results['results']['report']
    print(f"Report saved to: {report['report_file']}")
```

### With Seismic Integration

```python
# Add seismic agent
coordinator.register_agent('seismic_processor', SeismicAgent())

# Configure with seismic data
config = {
    'data_file': 'data/ERT/survey.dat',
    'instrument': 'E4D',
    'use_seismic': True,
    'seismic_data': travel_time_data,  # PyGIMLi travel time data
    'velocity_threshold': 1200,  # m/s
    'inversion_params': {'lambda': 20.0}
}

results = coordinator.execute_workflow(config)
```

### With Climate Data Integration

```python
from PyHydroGeophysX.agents import ClimateDataAgent

# Add climate agent
coordinator.register_agent('climate_data', ClimateDataAgent())

# Configure with climate data
config = {
    'data_file': 'data/ERT/survey.dat',
    'instrument': 'E4D',
    'use_climate': True,
    'climate_config': {
        'coords': (-105.3, 40.0),  # Site location (lon, lat)
        'dates': ('2023-06-01', '2023-09-30'),  # Campaign period
        'pet_method': 'penman_monteith',
        'antecedent_days': [1, 3, 7]
    },
    'ert_timestamps': ['2023-06-15', '2023-07-15', '2023-08-15'],
    'inversion_params': {'lambda': 20.0}
}

results = coordinator.execute_workflow(config)

# Access climate data
if results['status'] == 'success':
    climate_data = results['results']['climate_data']
    print(f"Climate features: {climate_data['derived_features'].keys()}")
```

## Workflow Steps

0. **Fetch Climate Data** (optional)
   - Retrieves meteorological data from PyDaymet
   - Computes PET using multiple methods
   - Aligns climate data with ERT timestamps
   - Generates derived features (antecedent precipitation, P-PET)
   - LLM provides hydrologic context

1. **Load ERT Data**
   - Loads data from commercial instruments (E4D, Syscal, ABEM, etc.)
   - Performs automatic quality control
   - LLM provides data quality insights

2. **Process Seismic Data** (optional)
   - Inverts seismic travel time data
   - Extracts velocity interfaces
   - LLM interprets velocity structure

3. **ERT Inversion**
   - Performs resistivity inversion
   - Optionally applies seismic structural constraints
   - LLM recommends inversion parameters

4. **Water Content Conversion**
   - Converts resistivity to water content
   - Applies petrophysical models (Archie, Waxman-Smits)
   - Runs Monte Carlo uncertainty analysis
   - LLM suggests parameter distributions

5. **Generate Report**
   - Creates comprehensive markdown/HTML report
   - Generates visualization plots
   - Includes climate context if available
   - LLM provides narrative summary

## Configuration Options

### Workflow Configuration

```python
config = {
    # Data source
    'data_file': str,           # Path to ERT data file
    'project_dir': str,         # Project directory
    'instrument': str,          # Instrument type (E4D, Syscal, etc.)
    'crs': str,                 # Coordinate system ('local' or EPSG code)
    
    # Inversion parameters
    'inversion_params': {
        'lambda': float,        # Regularization (default: 20.0)
        'max_iterations': int,  # Max iterations (default: 10)
        'method': str,          # Solver method (default: 'cgls')
        'use_gpu': bool,        # GPU acceleration (default: False)
    },
    
    # Petrophysical parameters
    'petrophysical_params': {
        # layer_marker: {
        #     'rhos': {'mean': float, 'std': float},
        #     'n': {'mean': float, 'std': float},
        #     'sigma_sur': {'mean': float, 'std': float},
        #     'porosity': {'mean': float, 'std': float}
        # }
    },
    
    # Uncertainty quantification
    'run_uncertainty': bool,    # Run Monte Carlo (default: False)
    'n_realizations': int,      # MC realizations (default: 100)
    
    # Seismic integration (optional)
    'use_seismic': bool,        # Enable seismic (default: False)
    'seismic_data': object,     # PyGIMLi travel time data
    'velocity_threshold': float,# Interface threshold m/s (default: 1200)
    
    # Climate data integration (optional)
    'use_climate': bool,        # Enable climate data (default: False)
    'climate_config': {
        'coords': tuple,        # (lon, lat) or list of tuples
        'geometry': object,     # Polygon or bbox for gridded data
        'dates': tuple,         # (start_date, end_date) or list of years
        'crs': int,             # Coordinate system (default: 4326)
        'variables': list,      # Variables to retrieve (default: all)
        'pet_method': str,      # PET method or list of methods
        'pet_params': dict,     # PET parameters (arid_correction, etc.)
        'time_scale': str,      # 'daily', 'monthly', or 'annual'
        'antecedent_days': list # Days for antecedent totals (default: [1,3,7])
    },
    'ert_timestamps': list      # ERT acquisition times for alignment
}
```

## Agent Details

### ERTLoaderAgent
- Loads data from 14+ instrument formats
- Performs automatic quality control
- Generates diagnostic plots
- Exports to inversion format

### ERTInversionAgent
- Configures and runs ERT inversion
- Supports structure-constrained inversion
- Monitors convergence
- Interprets results

### WaterContentAgent
- Applies petrophysical models
- Runs Monte Carlo uncertainty analysis
- Handles multiple geological layers
- Calculates statistics (mean, std, percentiles)

### SeismicAgent
- Inverts seismic travel time data
- Extracts velocity interfaces
- Provides structural constraints for ERT
- Interprets velocity structure

### ClimateDataAgent
- Fetches daily climate data from PyDaymet
- Computes PET using multiple methods (Penman-Monteith, Priestley-Taylor, Hargreaves-Samani)
- Aligns climate data with ERT timestamps
- Generates derived features (antecedent precipitation, P-PET)
- Supports both point and gridded data
- Provides climate context for resistivity interpretation

### ReportAgent
- Generates markdown and HTML reports
- Creates visualization plots
- Compiles workflow summary
- Provides narrative interpretation

## Output Structure

```
results/agents/
├── workflow_state.json          # Workflow status
├── execution_log.json           # Execution log
├── ert_loader_results.json      # ERT loading results
├── ert_inversion/               # Inversion outputs
│   ├── ert_data_for_inversion.dat
│   └── inversion_results.dat
├── water_content/               # Water content results
│   ├── water_content_mean.npy
│   ├── water_content_std.npy
│   └── water_content_p50.npy
└── reports/                     # Generated reports
    ├── workflow_report.md
    ├── workflow_report.html
    ├── resistivity_model.png
    └── water_content.png
```

## Examples

See example scripts for complete working examples:

```bash
# Standard ERT workflow
python examples/Ex_multi_agent_workflow.py --mode ert

# With seismic integration
python examples/Ex_multi_agent_workflow.py --mode seismic

# Climate data integration with ERT
python examples/Ex_climate_ert_integration.py
```

## LLM Features

When an LLM API key is provided (OpenAI/Gemini/Claude), agents can:

1. **Recommend Parameters**: Suggest optimal inversion and petrophysical parameters
2. **Interpret Results**: Provide expert interpretation of results
3. **Quality Assessment**: Assess data quality and identify issues
4. **Generate Reports**: Create narrative summaries and recommendations
5. **Climate Context**: Explain resistivity changes in context of climate events

## Dependencies

**Core:**
- `numpy`, `scipy`, `matplotlib` - Standard scientific Python
- `tqdm` - Progress bars

**LLM APIs (optional but recommended):**
- `openai>=1.0.0` - For OpenAI GPT models
- `google-generativeai>=0.3.0` - For Google Gemini models
- `anthropic>=0.18.0` - For Anthropic Claude models

**Geophysics:**
- `pygimli>=1.5` - For geophysical modeling
- `resipy>=3.4.0` - For ERT data processing

**Climate data (optional):**
- `pydaymet>=0.16.0` - For meteorological data retrieval
- `pandas>=1.3.0` - For data manipulation
- `xarray>=0.19.0` - For gridded data handling

**Other:**
- `markdown>=3.0` - HTML report generation (optional)

## Notes

- The system is designed to work with or without LLM features
- Each agent can be used independently if needed
- Results are saved at each step for debugging and inspection
- Failed workflows save partial results for recovery

## Troubleshooting

### API Key Issues
```python
# Check if API key is set
import os
print(os.getenv('OPENAI_API_KEY'))

# Set temporarily in code (not recommended for production)
coordinator = AgentCoordinator(api_key='your-key')
```

### Missing Dependencies
```bash
# Install all required packages
pip install pyhydrogeophysx[geophysics]
pip install openai markdown
```

### GPU Support
```bash
# For GPU-accelerated inversion
pip install cupy-cuda11x  # Match your CUDA version
```

## Citation

If you use the multi-agent system in your research, please cite PyHydroGeophysX:

```bibtex
@software{chen2025pyhydrogeophysx,
  author = {Chen, Hang and Niu, Qifei and Wu, Yuxin},
  title = {PyHydroGeophysX: An Extensible Open-Source Platform for 
           Bridging Hydrological Models and Geophysical Measurements},
  year = {2025},
  publisher = {Water Resources Research (under review)},
  url = {https://github.com/geohang/PyHydroGeophysX}
}
```

## License

Apache-2.0 License - see LICENSE file for details.
