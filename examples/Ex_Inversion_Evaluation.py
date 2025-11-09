"""
Example: Automatic Inversion Quality Evaluation and Parameter Optimization

This example demonstrates how to use the InversionEvaluationAgent to:
1. Assess ERT inversion quality using multiple metrics
2. Automatically adjust regularization parameters
3. Re-run inversions until acceptable quality is achieved

The agent evaluates:
- Data fit (chi-squared statistics)
- Model smoothness
- Physical plausibility
- Convergence quality
- Model coverage

And automatically adjusts lambda (regularization) and max_iterations based on the evaluation.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Setup package path
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    current_dir = os.getcwd()

parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import PyHydroGeophysX modules
from PyHydroGeophysX.agents import (
    ERTLoaderAgent,
    ERTInversionAgent,
    InversionEvaluationAgent
)

# %%
print("="*70)
print("EXAMPLE: Automatic Inversion Quality Evaluation")
print("="*70)

# Set API key for LLM-powered interpretation (optional)
api_key = os.getenv('OPENAI_API_KEY')  # or set directly
llm_model = 'gpt-4o-mini'
llm_provider = 'openai'

# %%
# Step 1: Load ERT data
print("\nStep 1: Loading ERT data...")

data_dir = Path('data/ERT/E4D')
data_file = data_dir / '2021-10-08_1400.ohm'

loader_agent = ERTLoaderAgent(api_key=api_key, model=llm_model, llm_provider=llm_provider)

load_result = loader_agent.execute({
    'data_file': str(data_file),
    'instrument': 'E4D',
    'project_dir': str(data_dir),
    'crs': 'local'
})

if load_result['status'] != 'success':
    print(f"✗ Failed to load data: {load_result.get('error')}")
    sys.exit(1)

print(f"✓ Loaded {load_result['num_electrodes']} electrodes, "
      f"{load_result['num_measurements']} measurements")

# %%
# Step 2: Run initial inversion with sub-optimal parameters
print("\nStep 2: Running initial inversion with intentionally poor parameters...")
print("(Using lambda=5.0 - too low, will likely overfit)")

inversion_agent = ERTInversionAgent(api_key=api_key, model=llm_model, llm_provider=llm_provider)

# Use poor parameters to demonstrate optimization
initial_params = {
    'lambda': 5.0,  # Too low - will overfit
    'max_iterations': 5,  # Too few
    'method': 'cgls'
}

inversion_result = inversion_agent.execute({
    'ert_data': load_result['ert_data'],
    'inversion_mode': 'standard',
    'inversion_params': initial_params,
    'output_dir': 'results/evaluation_example'
})

if inversion_result['status'] != 'success':
    print(f"✗ Inversion failed: {inversion_result.get('error')}")
    sys.exit(1)

print("✓ Initial inversion completed")

# %%
# Step 3: Evaluate and automatically optimize
print("\nStep 3: Evaluating inversion quality and optimizing parameters...")
print("="*70)

eval_agent = InversionEvaluationAgent(api_key=api_key, model=llm_model, llm_provider=llm_provider)

eval_input = {
    'inversion_results': inversion_result,
    'ert_data': load_result['ert_data'],
    'inversion_params': initial_params,
    'inversion_mode': 'standard',
    'auto_adjust': True,  # Enable automatic optimization
    'max_attempts': 5,    # Try up to 5 times
    'project_dir': str(data_dir),
    'instrument': 'E4D'
}

eval_result = eval_agent.execute(eval_input)

# %%
# Step 4: Display evaluation results
print("\n" + "="*70)
print("EVALUATION RESULTS")
print("="*70)

print(f"\nStatus: {eval_result['status']}")
print(f"Overall Quality Score: {eval_result['quality_score']:.1f}/100")
print(f"Optimization Attempts: {eval_result.get('attempts', 1)}")

print("\nComponent Scores:")
component_scores = eval_result.get('component_scores', {})
for component, score in component_scores.items():
    status = "✓" if score >= 70 else "⚠" if score >= 50 else "✗"
    print(f"  {status} {component.replace('_', ' ').title()}: {score:.1f}/100")

print("\nKey Metrics:")
metrics = eval_result.get('quality_metrics', {})
if 'data_fit' in metrics:
    chi2 = metrics['data_fit'].get('final_chi2', 'N/A')
    print(f"  - Final chi²: {chi2}")
    print(f"  - Data fit status: {metrics['data_fit'].get('status', 'unknown')}")

if 'physical_plausibility' in metrics:
    min_res = metrics['physical_plausibility'].get('min_resistivity', 'N/A')
    max_res = metrics['physical_plausibility'].get('max_resistivity', 'N/A')
    print(f"  - Resistivity range: {min_res:.1f} - {max_res:.1f} Ωm")

print("\nRecommendations:")
for rec in eval_result.get('recommendations', []):
    print(f"  • {rec}")

if eval_result.get('interpretation'):
    print(f"\nAI Interpretation:")
    print(f"  {eval_result['interpretation']}")

# %%
# Step 5: Visualize optimization history
if eval_result.get('attempts', 1) > 1:
    history = eval_result.get('evaluation_history', [])
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Quality scores
    ax1 = axes[0, 0]
    attempts = list(range(1, len(history) + 1))
    quality_scores = [h['quality_score'] for h in history]
    
    ax1.plot(attempts, quality_scores, 'o-', linewidth=2, markersize=10, color='steelblue')
    ax1.axhline(y=70, color='green', linestyle='--', linewidth=2, label='Acceptable (70)')
    ax1.fill_between(attempts, 70, 100, alpha=0.2, color='green')
    ax1.set_xlabel('Optimization Attempt', fontsize=12)
    ax1.set_ylabel('Quality Score', fontsize=12)
    ax1.set_title('Quality Score Evolution', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim([0, 105])
    
    # Plot 2: Lambda evolution
    ax2 = axes[0, 1]
    lambda_values = [h['parameters'].get('lambda', 0) for h in history]
    
    ax2.plot(attempts, lambda_values, 's-', linewidth=2, markersize=10, color='coral')
    ax2.set_xlabel('Optimization Attempt', fontsize=12)
    ax2.set_ylabel('Lambda (Regularization)', fontsize=12)
    ax2.set_title('Regularization Parameter Evolution', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Chi-squared evolution
    ax3 = axes[1, 0]
    chi2_values = []
    for h in history:
        chi2 = h['metrics'].get('data_fit', {}).get('final_chi2')
        chi2_values.append(chi2 if chi2 is not None else np.nan)
    
    ax3.plot(attempts, chi2_values, '^-', linewidth=2, markersize=10, color='purple')
    ax3.axhline(y=1.0, color='green', linestyle='--', linewidth=2, label='Target (1.0)')
    ax3.axhspan(0.8, 1.5, alpha=0.2, color='green', label='Acceptable range')
    ax3.set_xlabel('Optimization Attempt', fontsize=12)
    ax3.set_ylabel('Chi-squared (χ²)', fontsize=12)
    ax3.set_title('Data Fit Evolution', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot 4: Component scores comparison
    ax4 = axes[1, 1]
    components = ['Data\nFit', 'Smooth\nness', 'Physical\nPlausibility', 'Conver\ngence']
    comp_keys = ['data_fit', 'smoothness', 'physical_plausibility', 'convergence']
    
    first_scores = [history[0]['component_scores'][k] for k in comp_keys]
    best_idx = np.argmax([h['quality_score'] for h in history])
    best_scores = [history[best_idx]['component_scores'][k] for k in comp_keys]
    
    x = np.arange(len(components))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, first_scores, width, label='Initial', alpha=0.8, color='lightcoral')
    bars2 = ax4.bar(x + width/2, best_scores, width, label=f'Optimized', alpha=0.8, color='lightgreen')
    
    ax4.set_ylabel('Score', fontsize=12)
    ax4.set_title('Component Score Improvement', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(components, fontsize=9)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_ylim([0, 105])
    ax4.axhline(y=70, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('results/evaluation_example/optimization_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n✓ Visualization saved to: results/evaluation_example/optimization_history.png")
    
    # Print parameter evolution table
    print("\nParameter Evolution:")
    print("-" * 70)
    print(f"{'Attempt':<10} {'Lambda':<12} {'Max Iter':<12} {'Chi²':<12} {'Quality':<12}")
    print("-" * 70)
    for i, h in enumerate(history):
        params = h['parameters']
        chi2 = h['metrics'].get('data_fit', {}).get('final_chi2', 'N/A')
        chi2_str = f"{chi2:.3f}" if isinstance(chi2, (int, float)) else str(chi2)
        print(f"{i+1:<10} {params.get('lambda', 'N/A'):<12.2f} "
              f"{params.get('max_iterations', 'N/A'):<12} "
              f"{chi2_str:<12} {h['quality_score']:<12.1f}")
    print("-" * 70)

print("\n" + "="*70)
print("EXAMPLE COMPLETE")
print("="*70)
print(f"\nFinal Status: {eval_result['status'].upper()}")
print(f"Quality Improvement: {history[0]['quality_score']:.1f} → {eval_result['quality_score']:.1f}")
print(f"\nBest parameters found:")
print(f"  - Lambda: {eval_result.get('adjusted_params', {}).get('lambda', 'N/A')}")
print(f"  - Max iterations: {eval_result.get('adjusted_params', {}).get('max_iterations', 'N/A')}")
