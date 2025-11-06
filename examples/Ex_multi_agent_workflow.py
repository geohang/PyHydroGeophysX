"""
Example: Multi-Agent Workflow for ERT Processing
=================================================

This example demonstrates how to use the GPT API-based multi-agent system
to automate the complete workflow:
"load ERT → invert → convert to water content → report"

The system can optionally integrate seismic data for structure-constrained inversion.

Each agent is specialized for a specific task:
1. ERTLoaderAgent: Loads and quality-checks ERT data
2. SeismicAgent (optional): Processes seismic data for structural constraints
3. ERTInversionAgent: Performs ERT inversion
4. WaterContentAgent: Converts resistivity to water content with uncertainty
5. ReportAgent: Generates comprehensive reports

The AgentCoordinator manages the workflow and ensures proper data flow
between agents.
"""

import os
import sys

# Setup package path
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    current_dir = os.getcwd()

parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from PyHydroGeophysX.agents import (
    AgentCoordinator,
    ERTLoaderAgent,
    ERTInversionAgent,
    WaterContentAgent,
    ReportAgent,
    SeismicAgent
)


def run_ert_workflow_example():
    """
    Run the complete ERT workflow using multi-agent system.
    
    This example demonstrates:
    1. Setting up agents with OpenAI API
    2. Configuring workflow parameters
    3. Executing the complete workflow
    4. Accessing results
    """
    
    print("=" * 80)
    print("Multi-Agent ERT Workflow Example")
    print("=" * 80)
    
    # Note: Set your OpenAI API key as environment variable
    # export OPENAI_API_KEY='your-api-key-here'
    # Or pass it directly when creating the coordinator
    
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        print("\n⚠️  Warning: OPENAI_API_KEY not found in environment variables.")
        print("    The system will work but LLM-enhanced features will be disabled.")
        print("    Set it with: export OPENAI_API_KEY='your-key'\n")
    
    # Step 1: Initialize the coordinator
    print("\n[1/5] Initializing agent coordinator...")
    coordinator = AgentCoordinator(
        api_key=api_key,
        output_dir="results/agents_workflow"
    )
    
    # Step 2: Register specialized agents
    print("[2/5] Registering specialized agents...")
    
    # Register ERT loader agent
    ert_loader = ERTLoaderAgent(api_key=api_key)
    coordinator.register_agent('ert_loader', ert_loader)
    
    # Register ERT inversion agent
    ert_inversion = ERTInversionAgent(api_key=api_key)
    coordinator.register_agent('ert_inversion', ert_inversion)
    
    # Register water content conversion agent
    water_content = WaterContentAgent(api_key=api_key)
    coordinator.register_agent('water_content', water_content)
    
    # Register report generation agent
    report_gen = ReportAgent(api_key=api_key)
    coordinator.register_agent('report', report_gen)
    
    # Register seismic agent (optional)
    seismic = SeismicAgent(api_key=api_key)
    coordinator.register_agent('seismic_processor', seismic)
    
    print("   ✓ All agents registered")
    
    # Step 3: Configure workflow
    print("[3/5] Configuring workflow...")
    
    workflow_config = {
        # Data source configuration
        'data_file': 'data/ERT/E4D/2021-10-08_1400.ohm',
        'project_dir': 'data/ERT/E4D',
        'instrument': 'E4D',
        'crs': 'local',
        
        # Inversion parameters
        'inversion_params': {
            'lambda': 20.0,          # Regularization parameter
            'max_iterations': 10,     # Maximum inversion iterations
            'method': 'cgls',         # Solver method
            'use_gpu': False          # GPU acceleration (requires CuPy)
        },
        
        # Petrophysical parameters for water content conversion
        'petrophysical_params': {
            # Parameters will be auto-suggested by LLM if not provided
            # or can be manually specified per layer
            # Example:
            # 0: {  # Layer marker 0 (top layer)
            #     'rhos': {'mean': 100.0, 'std': 20.0},
            #     'n': {'mean': 2.2, 'std': 0.2},
            #     'sigma_sur': {'mean': 0.002, 'std': 0.0005},
            #     'porosity': {'mean': 0.40, 'std': 0.05}
            # }
        },
        
        # Uncertainty quantification
        'run_uncertainty': True,      # Run Monte Carlo analysis
        'n_realizations': 100,        # Number of MC realizations
        
        # Optional: Seismic integration
        'use_seismic': False,         # Set to True to use seismic constraints
        # 'seismic_data': seismic_travel_time_data,  # Provide seismic data if available
        # 'velocity_threshold': 1200,  # m/s threshold for interface detection
    }
    
    print("   ✓ Workflow configured")
    
    # Step 4: Execute workflow
    print("[4/5] Executing workflow...")
    print("-" * 80)
    
    try:
        results = coordinator.execute_workflow(workflow_config)
        
        print("-" * 80)
        print("[5/5] Workflow completed!")
        print(f"\n✓ Status: {results['status']}")
        
        # Display summary
        if results['status'] == 'success':
            print("\n📊 Workflow Summary:")
            summary = coordinator.get_workflow_summary()
            print(f"   - Completed steps: {', '.join(summary['completed_steps'])}")
            print(f"   - Available results: {', '.join(summary['available_results'])}")
            
            # Access specific results
            if 'water_content' in results['results']:
                wc_results = results['results']['water_content']
                print(f"\n💧 Water Content Results:")
                print(f"   - Output directory: {wc_results['output_dir']}")
                if wc_results.get('interpretation'):
                    print(f"   - Interpretation: {wc_results['interpretation']}")
            
            if 'report' in results['results']:
                report_results = results['results']['report']
                print(f"\n📄 Report Generated:")
                print(f"   - Report file: {report_results['report_file']}")
                if report_results.get('html_file'):
                    print(f"   - HTML report: {report_results['html_file']}")
        
        else:
            print(f"\n❌ Workflow failed: {results.get('error', 'Unknown error')}")
            if 'partial_results' in results:
                print(f"   Partial results available for: {list(results['partial_results'].keys())}")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Error executing workflow: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def run_ert_with_seismic_example():
    """
    Example with seismic data integration.
    
    This example shows how to include seismic refraction data
    for structure-constrained ERT inversion.
    """
    
    print("=" * 80)
    print("Multi-Agent ERT Workflow with Seismic Integration")
    print("=" * 80)
    
    # Initialize coordinator
    api_key = os.getenv('OPENAI_API_KEY')
    coordinator = AgentCoordinator(api_key=api_key)
    
    # Register all agents (including seismic)
    coordinator.register_agent('ert_loader', ERTLoaderAgent(api_key))
    coordinator.register_agent('seismic_processor', SeismicAgent(api_key))
    coordinator.register_agent('ert_inversion', ERTInversionAgent(api_key))
    coordinator.register_agent('water_content', WaterContentAgent(api_key))
    coordinator.register_agent('report', ReportAgent(api_key))
    
    # Configure workflow with seismic integration
    workflow_config = {
        'data_file': 'data/ERT/E4D/2021-10-08_1400.ohm',
        'project_dir': 'data/ERT/E4D',
        'instrument': 'E4D',
        'crs': 'local',
        
        # Enable seismic integration
        'use_seismic': True,
        'seismic_data': None,  # Provide seismic travel time data object
        'velocity_threshold': 1200,  # m/s
        
        'inversion_params': {
            'lambda': 20.0,
            'max_iterations': 10,
        },
        
        'run_uncertainty': True,
        'n_realizations': 50,
    }
    
    print("\n🔬 Running workflow with seismic constraints...")
    
    try:
        results = coordinator.execute_workflow(workflow_config)
        
        if results['status'] == 'success':
            print("\n✓ Workflow with seismic integration completed successfully!")
            
            # Check if seismic results are available
            if 'seismic_structure' in results['results']:
                seis = results['results']['seismic_structure']
                print(f"\n🌊 Seismic Results:")
                print(f"   - Interface extracted: Yes")
                print(f"   - Velocity threshold: {seis['velocity_threshold']} m/s")
                if seis.get('interpretation'):
                    print(f"   - Interpretation: {seis['interpretation']}")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return None


if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Multi-Agent Geophysical Workflow Example"
    )
    parser.add_argument(
        '--mode',
        choices=['ert', 'seismic'],
        default='ert',
        help='Workflow mode: ert (standard) or seismic (with seismic integration)'
    )
    
    args = parser.parse_args()
    
    # Run appropriate example
    if args.mode == 'seismic':
        results = run_ert_with_seismic_example()
    else:
        results = run_ert_workflow_example()
    
    print("\n" + "=" * 80)
    print("Example completed!")
    print("=" * 80)
