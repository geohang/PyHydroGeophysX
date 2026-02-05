# Final working Sphinx configuration for PyHydroGeophysX

import os
import sys

# Add project to path
sys.path.insert(0, os.path.abspath('../../'))
sys.path.insert(0, os.path.abspath('../../PyHydroGeophysX'))

# Project information
project = 'PyHydroGeophysX'
copyright = '2025, Hang Chen'
author = 'Hang Chen'
release = '0.1.0'

# Extensions
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx_gallery.gen_gallery',
    'sphinx_copybutton',
    'sphinx_design',
]

# Sphinx Gallery configuration - FINAL WORKING VERSION
sphinx_gallery_conf = {
    'examples_dirs': '../../examples',           # Path to example scripts
    'gallery_dirs': 'auto_examples',             # Output gallery directory
    'filename_pattern': '/(Ex|EX).*\.py$',
    'plot_gallery': False,                       # Don't execute scripts (use pre-generated figures)
    'download_all_examples': True,               # Allow downloading scripts
    'abort_on_example_error': False,             # Continue on errors
    'remove_config_comments': True,              # Clean up code display
    'show_memory': False,                        # Don't show memory usage
    'expected_failing_examples': [],             # No failing examples
    'image_scrapers': (),                        # Don't try to scrape images from execution
    'first_notebook_cell': '# PyHydroGeophysX Example\n# Figures are pre-generated',
    'show_signature': False,                     # Don't show function signatures
    'backreferences_dir': None,                  # Disable backreferences
}

# HTML theme
html_theme = 'pydata_sphinx_theme'
html_title = 'PyHydroGeophysX Documentation'
html_logo = '_static/logo.png'

html_theme_options = {
    'navbar_start': ['navbar-logo'],
    'navbar_center': ['navbar-nav'],
    'navbar_end': ['search-button', 'navbar-icon-links'],
    'icon_links': [
        {
            'name': 'GitHub',
            'url': 'https://github.com/geohang/PyHydroGeophysX',
            'icon': 'fa-brands fa-github',
        },
    ],
    'use_edit_page_button': True,
    'show_toc_level': 2,
}

html_context = {
    'github_user': 'geohang',
    'github_repo': 'PyHydroGeophysX',
    'github_version': 'main',
    'doc_path': 'docs/source',
}

# Static files
html_static_path = ['_static']
templates_path = ['_templates']

# Create directories
os.makedirs(os.path.join(os.path.dirname(__file__), '_static'), exist_ok=True)

# Mock imports for documentation build
autodoc_mock_imports = [
    'pygimli', 'flopy', 'parflow', 'cupy', 'joblib', 'meshop',
    'tqdm', 'matplotlib', 'scipy', 'numpy'
]

# GitHub Pages
html_baseurl = 'https://geohang.github.io/PyHydroGeophysX/'
