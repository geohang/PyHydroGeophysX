#!/usr/bin/env python3
"""
Extract figures from Jupyter notebooks for web documentation.
"""

import json
import base64
from pathlib import Path

def extract_figures_from_notebook(notebook_path, output_dir):
    """Extract all figures from a Jupyter notebook."""
    
    notebook_name = Path(notebook_path).stem
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    figure_count = 0
    
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code' and 'outputs' in cell:
            for output in cell['outputs']:
                if 'data' in output:
                    # Check for image data
                    for mime_type, data in output['data'].items():
                        if mime_type.startswith('image/'):
                            figure_count += 1
                            
                            # Determine file extension
                            if 'png' in mime_type:
                                ext = 'png'
                            elif 'jpeg' in mime_type or 'jpg' in mime_type:
                                ext = 'jpg'
                            elif 'svg' in mime_type:
                                ext = 'svg'
                            else:
                                ext = 'png'  # default
                            
                            filename = f"{notebook_name}_fig_{figure_count:02d}.{ext}"
                            filepath = output_path / filename
                            
                            # Decode and save
                            if isinstance(data, str):
                                # Base64 encoded
                                img_data = base64.b64decode(data)
                                with open(filepath, 'wb') as img_file:
                                    img_file.write(img_data)
                            elif isinstance(data, list):
                                # SVG text data
                                with open(filepath, 'w') as img_file:
                                    img_file.write(''.join(data))
                            
                            print(f"✅ Extracted: {filename}")
    
    return figure_count

def main():
    """Extract figures from all notebooks."""
    
    # Find all notebook files
    notebooks = list(Path('.').glob('*.ipynb')) + list(Path('examples').glob('*.ipynb'))
    
    if not notebooks:
        print("❌ No .ipynb files found!")
        return
    
    output_dir = "docs/source/auto_examples/images"
    total_figures = 0
    
    print(f"🔍 Found {len(notebooks)} notebooks")
    
    for notebook in notebooks:
        print(f"\n📔 Processing {notebook.name}...")
        figures = extract_figures_from_notebook(notebook, output_dir)
        total_figures += figures
        print(f"   Extracted {figures} figures")
    
    print(f"\n🎉 Total: {total_figures} figures extracted to {output_dir}")
    print("💡 Next: git add docs/source/auto_examples/images/* && git commit -m 'Add notebook figures'")

if __name__ == "__main__":
    main()