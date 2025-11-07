# New Features Summary

## 1. Context Input Agent (Natural Language Configuration)

### Location
`PyHydroGeophysX/agents/context_input_agent.py`

### Purpose
Translates natural language workflow requests into structured configuration dictionaries.

### Key Features
- **Natural Language Parsing**: Convert descriptions like "I want to run a time-lapse ERT inversion..." into valid config
- **Automatic Time-Lapse Detection**: Recognizes keywords (time-lapse, temporal, monitoring, 4D, repeated)
- **Smart Defaults**: Fills in missing parameters with reasonable values
- **Configuration Explanation**: Generates human-readable explanations of configs
- **Expert Suggestions**: Provides recommendations based on site conditions

### Methods
```python
ContextInputAgent(api_key, model, llm_provider)
  .parse_request(user_request, available_data) → workflow_config
  .explain_config(config) → explanation_text
  .suggest_improvements(config, site_conditions) → suggestions_text
```

### Example Usage
```python
from PyHydroGeophysX.agents import ContextInputAgent

agent = ContextInputAgent(api_key=api_key, model='gpt-4.1-nano')

request = """
Run time-lapse ERT inversion on 5 datasets collected monthly.
Use difference method with temporal regularization of 15.
Include water content conversion with 100 MC realizations.
"""

config = agent.parse_request(request)
explanation = agent.explain_config(config)
suggestions = agent.suggest_improvements(config, "Sandy soil, shallow water table")
```

## 2. Time-Lapse ERT Inversion

### Location
`PyHydroGeophysX/agents/ert_inversion_agent.py` (enhanced)

### Purpose
Monitor temporal changes in subsurface resistivity for applications like:
- Groundwater monitoring
- Moisture infiltration tracking
- Contaminant plume movement
- Seasonal variation studies

### Key Features
- **Three Inversion Methods**:
  - **Difference**: Independent inversions, then calculate Δρ = ρ(t) - ρ(baseline)
  - **Ratio**: Percentage changes, Ratio = ρ(t) / ρ(baseline)
  - **Joint**: Coupled inversion with temporal smoothness constraints
  
- **Temporal Regularization**: Control smoothness between time steps
- **Baseline Selection**: Choose reference dataset
- **LLM Interpretation**: Automatic interpretation of temporal changes

### Configuration Parameters
```python
{
    'inversion_mode': 'time-lapse',  # NEW: triggers time-lapse workflow
    'time_lapse_files': [              # NEW: list of data files
        'data1.ohm',
        'data2.ohm',
        'data3.ohm'
    ],
    'time_lapse_method': 'difference',  # NEW: 'difference', 'ratio', or 'joint'
    'temporal_regularization': 10.0,    # NEW: temporal smoothing weight
    'baseline_index': 0,                # NEW: which dataset is baseline
    # Standard parameters still work:
    'inversion_params': {'lambda': 20.0, 'max_iterations': 10}
}
```

### Results Structure
```python
{
    'status': 'success',
    'inversion_mode': 'time-lapse',
    'baseline_model': array(...),           # Initial resistivity
    'time_lapse_models': [array(...), ...], # All time steps
    'changes': [array(...), ...],           # Changes from baseline
    'mesh': mesh_object,
    'n_timesteps': 5,
    'chi2_values': [13.2, 12.8, ...],      # Quality for each step
    'interpretation': "LLM analysis of temporal changes..."
}
```

## 3. Enhanced Workflow Integration

### Updated Files
1. `PyHydroGeophysX/agents/__init__.py` - Added ContextInputAgent export
2. `PyHydroGeophysX/agents/ert_inversion_agent.py` - Added time-lapse support
3. `examples/Ex_multi_agent_workflow.ipynb` - Added demonstration cells

### Workflow Modes

**Standard Mode (existing):**
```python
config = {'inversion_mode': 'standard', ...}
results = coordinator.execute_workflow(config)
# Returns single inversion result
```

**Time-Lapse Mode (new):**
```python
config = {'inversion_mode': 'time-lapse', 'time_lapse_files': [...], ...}
results = coordinator.execute_workflow(config)
# Returns temporal series of results
```

## 4. Documentation

### New Files
- `PyHydroGeophysX/agents/context_input_agent.py` - Full implementation
- `PyHydroGeophysX/agents/TIME_LAPSE_GUIDE.md` - Comprehensive guide

### Updated Files
- `PyHydroGeophysX/agents/__init__.py` - Export new agent
- `examples/Ex_multi_agent_workflow.ipynb` - 5 new example cells

## Usage Examples in Notebook

### Cell 1: Import New Agent
```python
from PyHydroGeophysX.agents import ContextInputAgent
```

### Cell 2: Standard NL Configuration
```python
context_agent = ContextInputAgent(api_key=api_key, model=llm_model)
request = "Run standard ERT with lambda=25, 12 iterations..."
config = context_agent.parse_request(request)
```

### Cell 3: Time-Lapse NL Configuration
```python
request_tl = """
Time-lapse ERT on 5 monthly datasets...
Use difference method, temporal_reg=15...
"""
config_tl = context_agent.parse_request(request_tl)
```

### Cell 4: Get Expert Suggestions
```python
suggestions = context_agent.suggest_improvements(
    config, 
    "Sandy soil, seasonal monitoring"
)
```

### Cell 5: Execute with NL Config
```python
results = coordinator.execute_workflow(config_nl)
```

## Benefits

### 1. Ease of Use
- No need to manually write complex JSON configurations
- Natural language descriptions → automatic config generation
- Reduces configuration errors

### 2. Time-Lapse Capabilities
- Monitor subsurface changes over time
- Three inversion methods for different scenarios
- Automatic temporal smoothing

### 3. LLM Integration
- Smart parameter suggestions based on site conditions
- Automatic interpretation of results
- Configuration explanations for transparency

### 4. Flexibility
- Works with existing standard workflow
- Backward compatible (standard mode still works)
- Can mix manual and NL configuration

## Parameter Guidance

### Temporal Regularization
- **5-10**: Large, rapid changes expected
- **10-20**: Moderate changes, typical monitoring
- **20-50**: Small changes, noisy data, smooth evolution

### Time-Lapse Method Selection
- **Difference**: Best for large changes, independent datasets
- **Ratio**: Best for percentage analysis, normalized comparison
- **Joint**: Best for small changes, noisy data, coupled inversion

### Lambda (Spatial Regularization)
- **10-20**: High resolution, low noise data
- **20-50**: Moderate smoothing, typical field data
- **50-100**: Heavy smoothing, very noisy data

## Next Steps

1. **Test Natural Language Configuration**:
   - Run cell with `context_agent.parse_request()`
   - Verify generated config matches intent

2. **Test Time-Lapse Detection**:
   - Use keywords: "time-lapse", "monitoring", "temporal"
   - Check if `inversion_mode` set to 'time-lapse'

3. **Execute Time-Lapse Workflow** (when ready):
   - Prepare multiple ERT datasets
   - Configure time-lapse parameters
   - Run through AgentCoordinator

4. **Experiment with Parameters**:
   - Try different temporal_regularization values
   - Compare difference vs. ratio vs. joint methods
   - Adjust lambda based on data quality

## Migration Guide

### From Old to New System

**Before (Manual Configuration):**
```python
workflow_config = {
    'data_file': 'data.ohm',
    'instrument': 'E4D',
    'inversion_params': {'lambda': 20.0, 'max_iterations': 10},
    # ... many more fields ...
}
```

**After (Natural Language):**
```python
request = "Run standard ERT inversion on E4D data with moderate regularization"
workflow_config = context_agent.parse_request(request)
```

### Adding Time-Lapse to Existing Workflow

**Step 1**: Add time-lapse files to config
```python
config['inversion_mode'] = 'time-lapse'
config['time_lapse_files'] = ['file1.ohm', 'file2.ohm', 'file3.ohm']
config['time_lapse_method'] = 'difference'
config['temporal_regularization'] = 15.0
```

**Step 2**: Execute (same as before)
```python
results = coordinator.execute_workflow(config)
```

**Step 3**: Access time-lapse results
```python
baseline = results['baseline_model']
changes = results['changes']  # List of change arrays
```

## Troubleshooting

### Issue: Context agent not generating correct config
**Solution**: Provide more details in natural language request, include file paths and specific parameters

### Issue: Time-lapse mode not detected
**Solution**: Use explicit keywords: "time-lapse", "temporal", "monitoring", "4D", or set `inversion_mode='time-lapse'` manually

### Issue: Time-lapse inversion fails
**Solution**: 
- Check that all files exist
- Verify at least 2 datasets provided
- Ensure consistent electrode positions across datasets

### Issue: Results show no temporal changes
**Solution**:
- Decrease temporal_regularization
- Check data quality
- Verify baseline is correct

## Contact & Support

For issues or questions:
1. Check documentation: `TIME_LAPSE_GUIDE.md`
2. Review example notebook cells
3. Check error messages from LLM responses
4. Verify API key is valid for chosen LLM provider
