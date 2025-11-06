# Multi-Agent System for Automated Geophysical Workflows

This module provides a GPT API-based multi-agent system for automating geophysical data processing workflows in PyHydroGeophysX.

## Overview

The multi-agent system coordinates specialized AI agents to execute the complete workflow:

**"load ERT → invert → convert to water content → report"**

With optional seismic data integration for structure-constrained inversion.

## Architecture

### Agent Coordinator
- `AgentCoordinator`: Orchestrates the complete workflow and manages agent communication

### Specialized Agents
1. **ERTLoaderAgent**: Loads and quality-checks ERT field data from various instruments
2. **ERTInversionAgent**: Performs ERT inversion with optional structural constraints
3. **WaterContentAgent**: Converts resistivity to water content using petrophysical models
4. **ReportAgent**: Generates comprehensive reports with visualizations
5. **SeismicAgent** (optional): Processes seismic refraction data for structural constraints

## Key Features

- 🤖 **AI-Enhanced**: Each agent uses GPT API for intelligent parameter selection and interpretation
- 🔄 **Automated Workflow**: Complete end-to-end processing with minimal user input
- 📊 **Quality Control**: Automatic data validation and quality metrics
- 🧪 **Uncertainty Quantification**: Monte Carlo analysis for water content estimates
- 📈 **Comprehensive Reports**: Automatic generation of reports with visualizations
- 🌊 **Seismic Integration**: Optional structure-constrained inversion using seismic data

## Installation

The multi-agent system requires the OpenAI package:

```bash
pip install openai>=1.0.0
```

For full functionality, install PyHydroGeophysX with geophysics dependencies:

```bash
pip install pyhydrogeophysx[geophysics]
```

## Configuration

Set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY='your-api-key-here'
```

Or pass it directly when creating agents:

```python
coordinator = AgentCoordinator(api_key='your-api-key')
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

## Workflow Steps

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
    'velocity_threshold': float # Interface threshold m/s (default: 1200)
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

See `examples/Ex_multi_agent_workflow.py` for complete working examples:

```bash
# Run standard ERT workflow
python examples/Ex_multi_agent_workflow.py --mode ert

# Run with seismic integration
python examples/Ex_multi_agent_workflow.py --mode seismic
```

## LLM Features

When an OpenAI API key is provided, agents can:

1. **Recommend Parameters**: Suggest optimal inversion and petrophysical parameters
2. **Interpret Results**: Provide expert interpretation of results
3. **Quality Assessment**: Assess data quality and identify issues
4. **Generate Reports**: Create narrative summaries and recommendations

## Dependencies

- `openai>=1.0.0` - For GPT API access (optional but recommended)
- `pygimli>=1.5` - For geophysical modeling
- `numpy`, `scipy`, `matplotlib` - Standard scientific Python
- `tqdm` - Progress bars
- `markdown` - HTML report generation (optional)

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
