# Time-Lapse ERT Inversion

## Overview

Time-lapse ERT inversion allows monitoring of temporal changes in subsurface electrical resistivity, which can indicate:
- Moisture infiltration and redistribution
- Groundwater table fluctuations
- Contaminant plume movement
- Seasonal variations in water content
- Thermal changes

## Features

### Inversion Methods

1. **Difference Method**: Inverts each dataset independently, then calculates differences
   - Best for: Large temporal changes, independent datasets
   - Formula: `Δρ(t) = ρ(t) - ρ(baseline)`

2. **Ratio Method**: Calculates ratios between time steps
   - Best for: Percentage changes, normalized comparisons
   - Formula: `Ratio(t) = ρ(t) / ρ(baseline)`

3. **Joint Method**: Coupled inversion with temporal constraints
   - Best for: Small changes, noisy data, temporal smoothness
   - Includes temporal regularization parameter

## Usage

### 1. Natural Language Configuration

```python
from PyHydroGeophysX.agents import ContextInputAgent

context_agent = ContextInputAgent(api_key=api_key, model='gpt-4')

request = """
Run time-lapse ERT inversion on 5 datasets:
- 2021-10-08_1400.ohm (baseline)
- 2021-11-15_1400.ohm
- 2021-12-20_1400.ohm
- 2022-01-25_1400.ohm
- 2022-02-28_1400.ohm

Use difference method with temporal regularization of 15.
"""

config = context_agent.parse_request(request)
```

### 2. Manual Configuration

```python
workflow_config = {
    'inversion_mode': 'time-lapse',
    'time_lapse_files': [
        'data/ERT/E4D/2021-10-08_1400.ohm',
        'data/ERT/E4D/2021-11-15_1400.ohm',
        'data/ERT/E4D/2021-12-20_1400.ohm',
        'data/ERT/E4D/2022-01-25_1400.ohm',
        'data/ERT/E4D/2022-02-28_1400.ohm'
    ],
    'baseline_file': 'data/ERT/E4D/2021-10-08_1400.ohm',
    'time_lapse_method': 'difference',  # or 'ratio', 'joint'
    'temporal_regularization': 15.0,
    'project_dir': 'data/ERT/E4D',
    'instrument': 'E4D',
    'inversion_params': {
        'lambda': 20.0,
        'max_iterations': 10,
        'method': 'cgls'
    }
}
```

### 3. Execute Workflow

```python
from PyHydroGeophysX.agents import AgentCoordinator, ERTInversionAgent

coordinator = AgentCoordinator(api_key=api_key, output_dir='results/time_lapse')
coordinator.register_agent('ert_inversion', ERTInversionAgent(api_key=api_key))

results = coordinator.execute_workflow(workflow_config)
```

## Results Structure

Time-lapse inversion returns:

```python
{
    'status': 'success',
    'inversion_mode': 'time-lapse',
    'baseline_model': array(...),  # Initial resistivity model
    'time_lapse_models': [array(...), ...],  # Models for each time step
    'changes': [array(...), ...],  # Resistivity changes from baseline
    'mesh': Mesh object,
    'method': 'difference',
    'temporal_regularization': 15.0,
    'n_timesteps': 5,
    'chi2_values': [13.2, 12.8, 13.5, ...],  # Fit quality for each time step
    'interpretation': "LLM-generated interpretation...",
    'output_dir': 'results/time_lapse'
}
```

## Parameters

### Required Parameters

- `inversion_mode`: Must be `'time-lapse'`
- `time_lapse_files`: List of ERT data files in temporal order (minimum 2)

### Optional Parameters

- `time_lapse_method`: `'difference'` (default), `'ratio'`, or `'joint'`
- `temporal_regularization`: Smoothing weight for temporal changes (default: 10.0)
  - Higher values → smoother temporal evolution
  - Lower values → allow larger time steps
  - Typical range: 5-50
- `baseline_index`: Index of baseline dataset (default: 0 = first file)
- `inversion_params`: Standard inversion parameters (lambda, max_iterations, etc.)

## Best Practices

### 1. Data Preparation
- Ensure consistent electrode positions across all surveys
- Use same acquisition protocol for all time steps
- Check data quality and remove outliers
- Consider reciprocal measurements for error estimation

### 2. Parameter Selection

**For large expected changes (>20% resistivity change):**
```python
{
    'time_lapse_method': 'difference',
    'temporal_regularization': 5.0,  # Allow larger changes
    'inversion_params': {'lambda': 10.0}  # Less spatial smoothing
}
```

**For small changes or noisy data:**
```python
{
    'time_lapse_method': 'joint',
    'temporal_regularization': 30.0,  # Strong temporal coupling
    'inversion_params': {'lambda': 50.0}  # More spatial smoothing
}
```

**For percentage-based analysis:**
```python
{
    'time_lapse_method': 'ratio',
    'temporal_regularization': 15.0
}
```

### 3. Quality Control

Check these indicators:
- `chi2_values`: Should be consistent across time steps (within 2x of each other)
- Large sudden changes may indicate data quality issues
- Smooth temporal evolution suggests good temporal regularization

### 4. Interpretation

Time-lapse resistivity changes can indicate:

**Decreasing resistivity:**
- Moisture infiltration
- Groundwater recharge
- Saturation increase
- Temperature increase

**Increasing resistivity:**
- Drying/evaporation
- Water table decline
- Desaturation
- Temperature decrease
- Salt crystallization

## Examples

### Example 1: Monitoring Rainfall Infiltration

```python
request = """
Monitor rainfall infiltration using 3 ERT surveys:
- Before rain: 2021-08-01_1000.ohm
- During rain: 2021-08-02_1400.ohm
- After rain: 2021-08-03_1000.ohm

Use difference method to see moisture changes.
Lambda=15, 12 iterations.
"""

config = context_agent.parse_request(request)
results = coordinator.execute_workflow(config)
```

### Example 2: Seasonal Groundwater Monitoring

```python
workflow_config = {
    'inversion_mode': 'time-lapse',
    'time_lapse_files': [
        'data/winter.ohm',  # Dry season
        'data/spring.ohm',  # Recharge
        'data/summer.ohm',  # Peak water
        'data/fall.ohm'     # Decline
    ],
    'time_lapse_method': 'joint',  # Smooth seasonal transitions
    'temporal_regularization': 25.0,
    'inversion_params': {'lambda': 30.0, 'max_iterations': 15}
}
```

## Troubleshooting

### Issue: Very large changes between time steps
**Solution**: Increase temporal_regularization or check data quality

### Issue: No visible changes
**Solution**: 
- Decrease temporal_regularization
- Check if changes are smaller than expected
- Verify baseline is correct

### Issue: Inconsistent chi2 values
**Solution**:
- Check for data quality differences between surveys
- Consider normalizing data
- Adjust lambda parameter

## References

- Daily, W., et al. (1992). Electrical resistivity tomography of vadose water movement. Water Resources Research.
- Singha, K., & Gorelick, S. M. (2005). Saline tracer visualized with electrical resistivity tomography. Geophysics.
- Rings, J., & Hauck, C. (2009). Time-lapse refraction seismic tomography for the detection of ground ice. The Cryosphere.

## See Also

- [ERT Inversion Agent](ert_inversion_agent.py)
- [Context Input Agent](context_input_agent.py)
- [Example Notebook](../examples/Ex_multi_agent_workflow.ipynb)
