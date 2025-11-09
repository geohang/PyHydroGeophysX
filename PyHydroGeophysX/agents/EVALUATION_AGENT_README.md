# Inversion Evaluation Agent

## Overview

The `InversionEvaluationAgent` is an intelligent agent that automatically assesses ERT inversion quality and optimizes regularization parameters to achieve the best possible results. It uses multiple quality metrics and can automatically re-run inversions with adjusted parameters until acceptable quality is achieved.

## Features

### 1. Comprehensive Quality Evaluation
The agent evaluates inversions using four key metrics:

- **Data Fit (40% weight)**: Chi-squared statistics to assess how well the model fits observed data
- **Model Smoothness (25% weight)**: Gradient analysis to ensure realistic model structure
- **Physical Plausibility (25% weight)**: Validates resistivity values are within reasonable bounds
- **Convergence Quality (10% weight)**: Checks if the inversion converged properly

### 2. Automatic Parameter Optimization
- Detects overfitting (chi² too high) → increases lambda
- Detects underfitting (chi² too low) → decreases lambda
- Adjusts maximum iterations if convergence is poor
- Can make multiple attempts to find optimal parameters

### 3. Quality Scoring
- Overall score: 0-100 (weighted average of component scores)
- Acceptable threshold: 70/100
- Provides detailed breakdown by component
- Tracks improvement across optimization attempts

## Usage

### Basic Usage

```python
from PyHydroGeophysX.agents import InversionEvaluationAgent

# Initialize agent
eval_agent = InversionEvaluationAgent(
    api_key='your-openai-api-key',  # Optional, for AI interpretation
    model='gpt-4o-mini',
    llm_provider='openai'
)

# Evaluate inversion results
eval_input = {
    'inversion_results': inversion_results,  # From ERTInversionAgent
    'ert_data': ert_data,                   # Original ERT data
    'inversion_params': {                    # Current parameters
        'lambda': 20.0,
        'max_iterations': 10,
        'method': 'cgls'
    },
    'auto_adjust': True,                     # Enable optimization
    'max_attempts': 5                        # Max re-inversion attempts
}

eval_results = eval_agent.execute(eval_input)

# Check results
print(f"Quality Score: {eval_results['quality_score']:.1f}/100")
print(f"Status: {eval_results['status']}")
print(f"Attempts: {eval_results['attempts']}")
```

### Time-Lapse Inversion Evaluation

```python
# For time-lapse inversions
eval_input = {
    'inversion_results': time_lapse_results,
    'time_lapse_data': [ert_data1, ert_data2, ...],  # List of ERT datasets
    'inversion_mode': 'time-lapse',
    'inversion_params': {
        'lambda': 50.0,
        'alpha': 10.0,
        'max_iterations': 15
    },
    'auto_adjust': True,
    'max_attempts': 3,
    'project_dir': 'data/ERT/E4D',
    'instrument': 'E4D'
}

eval_results = eval_agent.execute(eval_input)
```

### Custom Quality Thresholds

```python
# Override default thresholds
eval_input = {
    'inversion_results': results,
    'ert_data': data,
    'inversion_params': params,
    'custom_thresholds': {
        'chi2_target': 0.9,           # Target chi-squared
        'chi2_acceptable_range': (0.7, 1.3),  # Acceptable range
        'max_gradient': 50.0,         # Max resistivity gradient
        'min_resistivity': 5.0,       # Min physical resistivity
        'max_resistivity': 5000.0     # Max physical resistivity
    }
}
```

## Output Structure

```python
{
    'status': 'success',              # 'success', 'needs_improvement', or 'error'
    'quality_score': 85.2,            # Overall score (0-100)
    'component_scores': {             # Individual component scores
        'data_fit': 90.5,
        'smoothness': 82.3,
        'physical_plausibility': 88.1,
        'convergence': 95.0
    },
    'quality_metrics': {              # Detailed metrics
        'data_fit': {
            'final_chi2': 0.95,
            'target_chi2': 1.0,
            'status': 'good'
        },
        'physical_plausibility': {
            'min_resistivity': 10.5,
            'max_resistivity': 2500.0,
            'violations': 0
        },
        # ... more metrics
    },
    'recommendations': [              # List of improvement suggestions
        "Results meet quality criteria. No adjustments needed."
    ],
    'adjusted_params': {              # Optimized parameters (if auto_adjust=True)
        'lambda': 45.0,
        'max_iterations': 15
    },
    'final_results': {...},           # Best inversion results
    'evaluation_history': [...],      # History of all attempts
    'attempts': 3,                    # Number of optimization attempts
    'interpretation': "..."           # AI-powered interpretation (if API key provided)
}
```

## Quality Thresholds (Default)

| Metric | Target | Acceptable Range | Action if Outside |
|--------|--------|------------------|-------------------|
| Chi-squared | 1.0 | 0.8 - 1.5 | Adjust lambda |
| Resistivity | - | 1 - 10,000 Ωm | Flag violations |
| Max Gradient | - | < 100 Ωm | Increase smoothing |

## Parameter Adjustment Strategy

### Underfitting (Chi² < 0.8)
- **Problem**: Model is too smooth, not fitting data well
- **Action**: Reduce lambda by 50%
- **Example**: λ = 20 → λ = 10

### Overfitting (Chi² > 1.5)
- **Problem**: Model is too rough, fitting noise
- **Action**: Increase lambda by 100%
- **Example**: λ = 20 → λ = 40

### Fine-tuning
- **Problem**: Close to target but not optimal
- **Action**: Adjust lambda by ±20%
- **Example**: λ = 20 → λ = 24 (if chi² slightly high)

## Integration with Workflow

### Complete Workflow Example

```python
from PyHydroGeophysX.agents import (
    ERTLoaderAgent,
    ERTInversionAgent,
    InversionEvaluationAgent
)

# 1. Load data
loader = ERTLoaderAgent(api_key=api_key)
data_result = loader.execute({
    'data_file': 'data.ohm',
    'instrument': 'E4D'
})

# 2. Run inversion
inverter = ERTInversionAgent(api_key=api_key)
inv_result = inverter.execute({
    'ert_data': data_result['ert_data'],
    'inversion_params': {'lambda': 20.0}
})

# 3. Evaluate and optimize
evaluator = InversionEvaluationAgent(api_key=api_key)
eval_result = evaluator.execute({
    'inversion_results': inv_result,
    'ert_data': data_result['ert_data'],
    'inversion_params': {'lambda': 20.0},
    'auto_adjust': True,
    'max_attempts': 5
})

# 4. Use optimized results
if eval_result['status'] == 'success':
    final_results = eval_result['final_results']
    print(f"Achieved quality score: {eval_result['quality_score']:.1f}/100")
```

## Visualization

The agent provides detailed evaluation history that can be visualized:

```python
history = eval_result['evaluation_history']

# Plot quality score improvement
attempts = range(1, len(history) + 1)
scores = [h['quality_score'] for h in history]

plt.plot(attempts, scores, 'o-')
plt.axhline(y=70, color='g', linestyle='--', label='Acceptable')
plt.xlabel('Attempt')
plt.ylabel('Quality Score')
plt.title('Optimization Progress')
plt.legend()
plt.show()
```

## Advanced Features

### 1. LLM-Powered Interpretation
When an API key is provided, the agent generates natural language interpretations:

```python
if eval_result['interpretation']:
    print(eval_result['interpretation'])
# Output: "The inversion shows excellent data fit (chi²=0.95) and 
#          physically reasonable resistivity values. The model is 
#          suitable for hydrogeophysical interpretation."
```

### 2. Convergence Monitoring
Tracks chi-squared evolution across iterations to detect:
- Proper convergence (< 1% improvement in last iterations)
- Premature stopping (still improving significantly)
- Oscillation or divergence

### 3. Spatial Quality Assessment
Evaluates model gradients to identify:
- Unrealistic sharp boundaries
- Over-smoothed features
- Optimal balance between fit and smoothness

## Best Practices

1. **Initial Parameters**: Start with moderate regularization (lambda=20-50)
2. **Max Attempts**: Use 3-5 attempts for automatic optimization
3. **Custom Thresholds**: Adjust based on your specific application
4. **Time-Lapse**: Use higher lambda (50-100) for temporal stability
5. **Monitor History**: Review optimization history to understand convergence
6. **Physical Constraints**: Set appropriate resistivity bounds for your site

## Examples

See `examples/Ex_Inversion_Evaluation.py` for a complete working example.

See `examples/Ex_TimeLapse_NaturalLanguage.ipynb` for integration with time-lapse workflow.

## Troubleshooting

### Issue: Agent keeps increasing lambda but quality doesn't improve
**Solution**: Check data quality - may have bad measurements or electrode issues

### Issue: Physical plausibility score is low
**Solution**: Set model constraints in inversion_params:
```python
inversion_params = {
    'lambda': 20.0,
    'model_constraints': (10.0, 1000.0)  # Min/max resistivity
}
```

## API Reference

### Class: InversionEvaluationAgent

**Constructor Parameters:**
- `api_key` (Optional[str]): LLM API key for interpretations
- `model` (Optional[str]): LLM model name
- `llm_provider` (str): 'openai', 'gemini', or 'claude'

**Methods:**
- `execute(input_data)`: Main evaluation method
- `_evaluate_quality(results, params)`: Comprehensive quality assessment
- `_adjust_parameters(params, metrics, recommendations)`: Parameter optimization
- `_rerun_inversion(input, adjusted_params)`: Re-run with new parameters

**Attributes:**
- `quality_thresholds`: Dictionary of quality thresholds
- `adjustment_factors`: Parameter adjustment multipliers
- `max_iterations`: Maximum optimization attempts
- `history`: List of evaluation attempts

## Citation

If you use the InversionEvaluationAgent in your research, please cite:

```
PyHydroGeophysX: An automated multi-agent system for hydrogeophysical workflows
[Add full citation when published]
```
