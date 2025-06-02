# Configuration file for the Sphinx documentation builder.

import os
import sys
import unittest.mock as mock
import logging
logging.getLogger().setLevel(logging.WARNING)

# Add project root and main package to sys.path
sys.path.insert(0, os.path.abspath('../../'))
sys.path.insert(0, os.path.abspath('../../PyHydroGeophysX'))

# List of heavy/optional modules to mock for doc build
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
    'examples_dirs': '../../examples',         # path to your example scripts
    'gallery_dirs': 'auto_examples',           # path to output gallery
    'plot_gallery': False,                     # Don't run example scripts
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

# UPDATE these with your GitHub username!
html_baseurl = 'https://geohang.github.io/PyHydroGeophysX/'
html_title = 'PyHydroGeophysX Documentation'

html_theme_options = {
    'canonical_url': 'https://geohang.github.io/PyHydroGeophysX/',
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

html_context = {
    "display_github": True,
    "github_user": "YOUR_GITHUB_USERNAME",
    "github_repo": "PyHydroGeophysX",
    "github_version": "main",
    "conf_py_path": "/docs/source/",
}

html_static_path = ['_static']
