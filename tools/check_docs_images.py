#!/usr/bin/env python
"""
check_docs_images.py - Verify all image references in documentation exist.

This script scans docs/source/**/*.rst files for image and figure directives,
resolves the paths to actual files, and reports any missing images.

Usage:
    python tools/check_docs_images.py

Exit codes:
    0 - All images found
    1 - Missing images detected

This script should be run before building documentation to catch broken
image references early. It's designed to be used in CI pipelines.
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Tuple, Set


def find_rst_files(docs_source: Path) -> List[Path]:
    """Find all RST files in docs/source directory."""
    return list(docs_source.rglob("*.rst"))


def extract_image_references(rst_file: Path) -> List[Tuple[str, int]]:
    """
    Extract image/figure references from an RST file.
    
    Returns list of (image_path, line_number) tuples.
    """
    image_refs = []
    
    # Patterns to match image and figure directives
    patterns = [
        r'^\s*\.\.\s+image::\s+(.+)$',
        r'^\s*\.\.\s+figure::\s+(.+)$',
    ]
    
    try:
        with open(rst_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Warning: Could not read {rst_file}: {e}")
        return []
    
    for line_num, line in enumerate(lines, 1):
        for pattern in patterns:
            match = re.match(pattern, line)
            if match:
                image_path = match.group(1).strip()
                # Skip external URLs
                if not image_path.startswith(('http://', 'https://')):
                    image_refs.append((image_path, line_num))
    
    return image_refs


def resolve_image_path(image_ref: str, rst_file: Path, docs_source: Path) -> Path:
    """
    Resolve an image reference to an actual file path.
    
    Handles both absolute (starting with /) and relative paths.
    """
    if image_ref.startswith('/'):
        # Absolute path from docs/source root
        return docs_source / image_ref.lstrip('/')
    else:
        # Relative path from RST file location
        return rst_file.parent / image_ref


def check_image_exists(image_path: Path) -> bool:
    """Check if an image file exists."""
    return image_path.exists() and image_path.is_file()


def check_all_images(docs_source: Path) -> Tuple[int, int, List[Tuple[Path, str, int]]]:
    """
    Check all image references in documentation.
    
    Returns:
        (total_images, missing_count, list of (rst_file, image_path, line_num) for missing)
    """
    rst_files = find_rst_files(docs_source)
    
    total_images = 0
    missing_images = []
    checked_paths: Set[str] = set()  # Avoid duplicate checks
    
    for rst_file in rst_files:
        image_refs = extract_image_references(rst_file)
        
        for image_path, line_num in image_refs:
            total_images += 1
            
            resolved_path = resolve_image_path(image_path, rst_file, docs_source)
            
            # Create a unique key for this reference
            check_key = f"{rst_file}:{image_path}"
            if check_key in checked_paths:
                continue
            checked_paths.add(check_key)
            
            if not check_image_exists(resolved_path):
                missing_images.append((rst_file, image_path, line_num))
    
    return total_images, len(missing_images), missing_images


def main():
    """Main entry point."""
    # Find docs/source directory
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    docs_source = repo_root / "docs" / "source"
    
    if not docs_source.exists():
        print(f"Error: docs/source directory not found at {docs_source}")
        sys.exit(1)
    
    print("=" * 60)
    print("PyHydroGeophysX Documentation Image Check")
    print("=" * 60)
    print(f"Scanning: {docs_source}")
    print()
    
    total, missing_count, missing_images = check_all_images(docs_source)
    
    print(f"Total image references found: {total}")
    print(f"Missing images: {missing_count}")
    print()
    
    if missing_images:
        print("MISSING IMAGES:")
        print("-" * 60)
        for rst_file, image_path, line_num in missing_images:
            rel_rst = rst_file.relative_to(docs_source)
            print(f"  {rel_rst}:{line_num}")
            print(f"    -> {image_path}")
            print()
        
        print("-" * 60)
        print(f"ERROR: {missing_count} missing image(s) detected!")
        print("Please add the missing images or fix the references.")
        sys.exit(1)
    else:
        print("✓ All image references are valid!")
        sys.exit(0)


if __name__ == "__main__":
    main()
