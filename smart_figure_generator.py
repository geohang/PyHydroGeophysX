#!/usr/bin/env python3
"""
Script to pre-generate figures from examples for documentation.
Run this locally where you have all the data files and dependencies.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def run_example_and_extract_figures(example_path, output_dir):
    """Run an example script and extract any generated figures."""
    
    print(f"Processing {example_path}...")
    
    # Create output directory for this example
    example_name = Path(example_path).stem
    fig_dir = os.path.join(output_dir, example_name)
    os.makedirs(fig_dir, exist_ok=True)
    
    # Change to examples directory
    original_dir = os.getcwd()
    examples_dir = os.path.dirname(example_path)
    os.chdir(examples_dir)
    
    try:
        # Run the example script
        result = subprocess.run([
            sys.executable, os.path.basename(example_path)
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print(f"✓ {example_name} executed successfully")
            
            # Look for any .png, .jpg, .svg files that were created
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.svg', '*.pdf']:
                for fig_file in Path('.').glob(f"**/{ext}"):
                    if fig_file.is_file():
                        # Copy to output directory
                        dest = os.path.join(fig_dir, fig_file.name)
                        shutil.copy2(fig_file, dest)
                        print(f"  Copied figure: {fig_file.name}")
            
            # Also check if matplotlib saved any figures
            import matplotlib.pyplot as plt
            if plt.get_fignums():  # If there are open figures
                for i, fignum in enumerate(plt.get_fignums()):
                    fig = plt.figure(fignum)
                    fig_path = os.path.join(fig_dir, f"{example_name}_fig_{i+1}.png")
                    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
                    print(f"  Saved figure: {example_name}_fig_{i+1}.png")
                plt.close('all')
                
        else:
            print(f"✗ {example_name} failed with error:")
            print(result.stderr)
            
    except subprocess.TimeoutExpired:
        print(f"✗ {example_name} timed out after 5 minutes")
    except Exception as e:
        print(f"✗ {example_name} failed with exception: {e}")
    finally:
        os.chdir(original_dir)

def main():
    """Generate figures for all examples."""
    
    # Find all example files
    examples_dir = "examples"
    example_files = []
    
    for file in Path(examples_dir).glob("Ex*.py"):
        example_files.append(str(file))
    
    if not example_files:
        print("No example files found!")
        return
    
    # Create output directory
    output_dir = "docs/source/auto_examples/images"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Found {len(example_files)} example files")
    print("Generating figures...")
    
    for example_file in sorted(example_files):
        run_example_and_extract_figures(example_file, output_dir)
    
    print(f"\nFigures saved to: {output_dir}")
    print("Add these files to git and commit them to include in documentation.")

if __name__ == "__main__":
    main()