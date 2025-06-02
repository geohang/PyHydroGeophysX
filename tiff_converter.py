#!/usr/bin/env python3
"""
Convert TIFF figures to PNG for web documentation.
"""

import os
from pathlib import Path
import shutil

def convert_tiff_to_png(input_dir, output_dir):
    """Convert all TIFF files to PNG for web compatibility."""
    
    try:
        from PIL import Image
        pil_available = True
    except ImportError:
        print("⚠️  PIL/Pillow not available. Installing...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "Pillow"])
        from PIL import Image
        pil_available = True
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all TIFF files
    tiff_files = list(input_path.glob("*.tiff")) + list(input_path.glob("*.tif"))
    
    if not tiff_files:
        print(f"No TIFF files found in {input_dir}")
        return
    
    print(f"🔄 Converting {len(tiff_files)} TIFF files to PNG...")
    
    for tiff_file in tiff_files:
        try:
            # Open TIFF and convert to PNG
            img = Image.open(tiff_file)
            png_name = tiff_file.stem + ".png"
            png_path = output_path / png_name
            
            # Convert to RGB if necessary (removes transparency issues)
            if img.mode in ('RGBA', 'LA', 'P'):
                rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                rgb_img.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                img = rgb_img
            
            img.save(png_path, 'PNG', dpi=(150, 150), optimize=True)
            print(f"  ✅ {tiff_file.name} → {png_name}")
            
        except Exception as e:
            print(f"  ❌ Failed to convert {tiff_file.name}: {e}")

def main():
    """Convert TIFF files from examples results to PNG for docs."""
    
    # Common directories where your examples might save TIFF files
    search_dirs = [
        "examples",
        "examples/results", 
        ".",
    ]
    
    output_dir = "docs/source/auto_examples/images"
    
    for search_dir in search_dirs:
        if Path(search_dir).exists():
            print(f"🔍 Searching for TIFF files in {search_dir}...")
            convert_tiff_to_png(search_dir, output_dir)
            
            # Also search subdirectories
            for subdir in Path(search_dir).glob("**/"):
                if any(subdir.glob("*.tiff")) or any(subdir.glob("*.tif")):
                    convert_tiff_to_png(subdir, output_dir)

if __name__ == "__main__":
    main()