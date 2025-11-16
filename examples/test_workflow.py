#!/usr/bin/env python
"""Test script for debugging unified workflow"""

import os
import sys
from pathlib import Path
import json

# Import agents
from PyHydroGeophysX.agents import BaseAgent, ContextInputAgent

# Setup
llm_provider = 'openai'
llm_model = 'gpt-4o-mini'
api_key = os.getenv('OPENAI_API_KEY')

if not api_key:
    print('ERROR: OPENAI_API_KEY not set')
    sys.exit(1)

print("="*70)
print("TESTING UNIFIED DATA FUSION WORKFLOW")
print("="*70)

# Initialize
context_agent = ContextInputAgent(api_key=api_key, model=llm_model, llm_provider=llm_provider)

user_request = """I need to characterize subsurface water content using a multi-method approach with field data:

1. First, use field seismic refraction data to identify the boundary between regolith and fractured bedrock.
   The seismic data is in 'data/Seismic/srtfieldline2.dat' (BERT format)
   Use a velocity threshold of 1000 m/s to extract the interface for regolith and fractured bedrock.

2. Then, use this seismic structure to constrain ERT inversion with field ERT data.
   The ERT data is in 'data/ERT/Bert/fielddataline2.dat' (BERT format).
   Apply moderate regularization (lambda=20) since we have structural constraints and field data.

3. Finally, convert the resistivity model to water content using layer-specific petrophysical parameters.
   Use Monte Carlo uncertainty analysis with 100 realizations.
   Account for different petrophysical properties in regolith vs fractured bedrock layers:
   - Regolith layer: rho_sat (50-250 Ωm), n (1.3-2.2), porosity (0.25-0.5)
   - Fractured bedrock layer: rho_sat (165-350 Ωm), n (2.0-2.2), porosity (0.2-0.3)

This is a full structure-constrained hydrogeophysical workflow for field data analysis."""

print('\nParsing natural language request...')
config = context_agent.parse_request(user_request)

print(f'\n✓ Configuration extracted:')
print(f'  - Seismic file: {config.get("seismic_file")}')
print(f'  - ERT file: {config.get("ert_file")}')
print(f'  - Velocity threshold: {config.get("velocity_threshold")} m/s')
print(f'  - Layer params: {len(config.get("layer_params", {}))} layers')

# Run workflow
output_dir = Path('results/unified_workflow/example3_test')
output_dir.mkdir(parents=True, exist_ok=True)

print(f'\n🚀 Running unified workflow...')
print(f'Output directory: {output_dir}')
print("="*70)

try:
    results, execution_plan, interpretation, report_files = BaseAgent.run_unified_agent_workflow(
        config, api_key, llm_model, llm_provider, output_dir
    )
    
    print("\n" + "="*70)
    print("WORKFLOW RESULTS")
    print("="*70)
    print(f"Status: {results.get('status')}")
    
    if results.get('status') == 'success':
        print("\n✅ Workflow completed successfully!")
        if 'statistics' in results:
            print(f"\nStatistics:")
            for key, value in results['statistics'].items():
                print(f"  {key}: {value}")
    else:
        print(f"\n❌ Workflow failed: {results.get('error')}")
        
except Exception as e:
    print(f'\n❌ ERROR: {str(e)}')
    import traceback
    print("\nFull traceback:")
    traceback.print_exc()
    sys.exit(1)

