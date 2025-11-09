"""
Test script to verify the climate data workflow
Run this to check if everything is set up correctly
"""

import sys
import os
from pathlib import Path

def check_files():
    """Check if all required files exist"""
    print("=" * 70)
    print("Checking Climate Data Workflow Files")
    print("=" * 70)
    
    required_files = [
        'fetch_climate_data.py',
        'setup_climate_env.bat',
        'fetch_climate.bat',
        'climate_config_example.json',
        'CLIMATE_DATA_WORKFLOW.md',
        'CLIMATE_QUICK_REF.md',
        'CLIMATE_WORKFLOW_DIAGRAM.txt'
    ]
    
    all_exist = True
    for file in required_files:
        exists = Path(file).exists()
        status = "✓" if exists else "✗"
        print(f"  {status} {file}")
        if not exists:
            all_exist = False
    
    print()
    return all_exist

def check_agent():
    """Check if ClimateDataAgent has CSV loading capability"""
    print("=" * 70)
    print("Checking ClimateDataAgent")
    print("=" * 70)
    
    try:
        # Try to import
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from PyHydroGeophysX.agents import ClimateDataAgent
        print("  ✓ ClimateDataAgent imported successfully")
        
        # Check for _load_from_csv method
        if hasattr(ClimateDataAgent, '_load_from_csv'):
            print("  ✓ CSV loading capability available")
            return True
        else:
            print("  ✗ CSV loading capability not found")
            return False
            
    except ImportError as e:
        print(f"  ✗ Failed to import ClimateDataAgent: {e}")
        return False

def check_example_config():
    """Check if example config is valid JSON"""
    print()
    print("=" * 70)
    print("Checking Example Configuration")
    print("=" * 70)
    
    try:
        import json
        with open('climate_config_example.json', 'r') as f:
            config = json.load(f)
        
        required_keys = ['coords', 'dates', 'output']
        missing = [key for key in required_keys if key not in config]
        
        if missing:
            print(f"  ✗ Missing required keys: {missing}")
            return False
        else:
            print(f"  ✓ Valid configuration")
            print(f"    - Coords: {config['coords']}")
            print(f"    - Dates: {config['dates']}")
            print(f"    - Output: {config['output']}")
            return True
            
    except Exception as e:
        print(f"  ✗ Invalid configuration: {e}")
        return False

def check_environment():
    """Check current Python environment"""
    print()
    print("=" * 70)
    print("Checking Current Environment")
    print("=" * 70)
    
    import numpy as np
    import pandas as pd
    
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  NumPy: {np.__version__}")
    print(f"  pandas: {pd.__version__}")
    
    # Check if PyDaymet is available (not expected in main env)
    try:
        import pydaymet
        print(f"  PyDaymet: {pydaymet.__version__} (warning: should be in separate env)")
        return False
    except ImportError:
        print(f"  PyDaymet: Not installed (correct for main environment)")
    
    # Check NumPy version
    if np.__version__.startswith('1'):
        print(f"  ✓ NumPy 1.x detected (correct for PyGIMLi)")
        return True
    elif np.__version__.startswith('2'):
        print(f"  ✗ NumPy 2.x detected (incompatible with PyGIMLi)")
        print(f"    Run: pip install 'numpy<2.0' --force-reinstall")
        return False

def print_next_steps():
    """Print next steps for user"""
    print()
    print("=" * 70)
    print("Next Steps")
    print("=" * 70)
    print()
    print("1. Setup climate fetch environment (one-time):")
    print("   .\\setup_climate_env.bat")
    print()
    print("2. Edit climate_config_example.json with your site info")
    print()
    print("3. Fetch climate data:")
    print("   .\\fetch_climate.bat climate_config_example.json")
    print()
    print("4. Use in your workflow:")
    print("   workflow_config = {")
    print("       'use_climate': True,")
    print("       'climate_config': {")
    print("           'csv_file': 'data/climate/climate_data.csv',")
    print("       },")
    print("       'ert_timestamps': ['2021-10-08'],")
    print("       ...")
    print("   }")
    print()
    print("See CLIMATE_DATA_WORKFLOW.md for detailed instructions.")
    print()

def main():
    """Run all checks"""
    print()
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "CLIMATE DATA WORKFLOW TEST" + " " * 27 + "║")
    print("╚" + "=" * 68 + "╝")
    print()
    
    results = []
    
    # Run checks
    results.append(("Files", check_files()))
    results.append(("Agent", check_agent()))
    results.append(("Config", check_example_config()))
    results.append(("Environment", check_environment()))
    
    # Summary
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    
    all_passed = all(result[1] for result in results)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
    
    print()
    
    if all_passed:
        print("✓ All checks passed! Ready to use climate data workflow.")
        print_next_steps()
        return 0
    else:
        print("✗ Some checks failed. Please review the errors above.")
        print()
        print("For help, see:")
        print("  - CLIMATE_DATA_WORKFLOW.md (detailed guide)")
        print("  - CLIMATE_QUICK_REF.md (quick reference)")
        print("  - CLIMATE_WORKFLOW_DIAGRAM.txt (visual workflow)")
        return 1

if __name__ == '__main__':
    sys.exit(main())
