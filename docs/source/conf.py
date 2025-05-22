# Configuration file for the Sphinx documentation builder.

import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath('../../'))

# -- Project information -----------------------------------------------------
project = 'PyHydroGeophysX'
copyright = '2025, Hang Chen'
author = 'Hang Chen'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.githubpages',
    'myst_parser',
    'sphinx_copybutton',
    'sphinx_gallery.gen_gallery',
]

# Napoleon settings for Google/NumPy style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# Sphinx Gallery configuration - show examples but don't execute them
sphinx_gallery_conf = {
    'examples_dirs': '../../examples',
    'gallery_dirs': 'auto_examples',
    'filename_pattern': '/Ex.*\.py$',
    'ignore_pattern': '__pycache__|\.ipynb$|\.ipynb_checkpoints',
    'download_all_examples': False,
    'plot_gallery': False,  # Don't generate plots
    'run_stale_examples': False,  # Don't run examples
    'abort_on_example_error': False,  # Continue on errors
    'capture_repr': (),  # Don't capture output
    'show_memory': False,  # Don't show memory usage
    'remove_config_comments': True,  # Clean up the display
    'expected_failing_examples': [],
    'gallery_dirs': 'auto_examples',
    'mod_example_dir': False,  # Don't modify example directory
    'subsection_order': 'ExplicitOrder',
}

# Templates path
templates_path = ['_templates']
exclude_patterns = []

# Source file suffixes
source_suffix = {
    '.rst': None,
    '.md': 'myst_parser',
}

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'

html_theme_options = {
    'canonical_url': '',
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}

html_static_path = ['_static']