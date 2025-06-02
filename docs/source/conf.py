# Configuration file for the Sphinx documentation builder.

import os
import sys
import unittest.mock as mock
import logging
logging.getLogger().setLevel(logging.WARNING)

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath('../../'))
sys.path.insert(0, os.path.abspath('../../PyHydroGeophysX'))

# Mock problematic imports for building docs
mock_modules = [
    'pygimli', 'pygimli.physics', 'pygimli.physics.ert', 
    'pygimli.physics.traveltime', 'pygimli.meshtools',
    'pygimli.viewer', 'pygimli.viewer.mpl', 'pygimli.utils',
    'pygimli.core', 'pygimli.matrix', 'flopy', 'parflow',
    'cupy', 'cupyx', 'cupyx.scipy', 'cupyx.scipy.sparse',
    'joblib', 'meshop'
]

for mod_name in mock_modules:
    sys.modules[mod_name] = mock.MagicMock()

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
    'nbsphinx',
    'sphinx_copybutton',
    'sphinx_gallery.gen_gallery',
]

# Napoleon settings
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

autodoc_mock_imports = mock_modules
nbsphinx_allow_errors = True

# Sphinx Gallery configuration
sphinx_gallery_conf = {
    'examples_dirs': '../../examples',
    'gallery_dirs': 'auto_examples',
    'plot_gallery': False,
    'download_all_examples': False,
    'filename_pattern': '/Ex.*\.py$',
    'ignore_pattern': '__pycache__|\.ipynb$',
    'expected_failing_examples': [],
    'capture_repr': (),
    'abort_on_example_error': False,
    'run_stale_examples': False,
}

templates_path = ['_templates']
exclude_patterns = []

source_suffix = {
    '.rst': None,
    '.md': 'myst_parser',
}

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'

# REPLACE 'YOUR_GITHUB_USERNAME' with your actual GitHub username
html_baseurl = 'https://YOUR_GITHUB_USERNAME.github.io/PyHydroGeophysX/'
html_title = 'PyHydroGeophysX Documentation'

html_theme_options = {
    'canonical_url': 'https://YOUR_GITHUB_USERNAME.github.io/PyHydroGeophysX/',
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

# GitHub integration - REPLACE 'YOUR_GITHUB_USERNAME' with your actual username
html_context = {
    "display_github": True,
    "github_user": "YOUR_GITHUB_USERNAME",
    "github_repo": "PyHydroGeophysX",
    "github_version": "main",
    "conf_py_path": "/docs/source/",
}

html_static_path = ['_static']