# Configuration file for the Sphinx documentation builder.

import os
import sys
import unittest.mock as mock
import logging
logging.getLogger().setLevel(logging.WARNING)

# Add project root and main package to sys.path
sys.path.insert(0, os.path.abspath('../../'))
sys.path.insert(0, os.path.abspath('../../PyHydroGeophysX'))

# Create more sophisticated mocks for critical classes
class MockDataContainer:
    """Mock for pg.DataContainer that works with isinstance"""
    pass

class MockMesh:
    """Mock for pg.Mesh that works with isinstance"""
    def cellCenters(self):
        # Return a 2D array to avoid indexing errors
        return [[0, 0], [1, 1]]  # Mock 2D coordinates
    
    def cellMarkers(self):
        return [1, 2]
    
    def cellCount(self):
        return 2

# Create mock modules with proper class structure
def create_pygimli_mock():
    pygimli_mock = mock.MagicMock()
    pygimli_mock.DataContainer = MockDataContainer
    pygimli_mock.Mesh = MockMesh
    
    # Add other commonly used pygimli components
    pygimli_mock.physics = mock.MagicMock()
    pygimli_mock.physics.ert = mock.MagicMock()
    pygimli_mock.physics.traveltime = mock.MagicMock()
    pygimli_mock.meshtools = mock.MagicMock()
    pygimli_mock.viewer = mock.MagicMock()
    pygimli_mock.viewer.mpl = mock.MagicMock()
    pygimli_mock.utils = mock.MagicMock()
    pygimli_mock.core = mock.MagicMock()
    pygimli_mock.matrix = mock.MagicMock()
    
    return pygimli_mock

# List of heavy/optional modules to mock
mock_modules = [
    'flopy', 'parflow', 'cupy', 'cupyx', 'cupyx.scipy', 'cupyx.scipy.sparse',
    'joblib', 'meshop'
]

# Mock standard modules
for mod_name in mock_modules:
    sys.modules[mod_name] = mock.MagicMock()

# Special handling for pygimli
sys.modules['pygimli'] = create_pygimli_mock()
sys.modules['pygimli.physics'] = sys.modules['pygimli'].physics
sys.modules['pygimli.physics.ert'] = sys.modules['pygimli'].physics.ert
sys.modules['pygimli.physics.traveltime'] = sys.modules['pygimli'].physics.traveltime
sys.modules['pygimli.meshtools'] = sys.modules['pygimli'].meshtools
sys.modules['pygimli.viewer'] = sys.modules['pygimli'].viewer
sys.modules['pygimli.viewer.mpl'] = sys.modules['pygimli'].viewer.mpl
sys.modules['pygimli.utils'] = sys.modules['pygimli'].utils
sys.modules['pygimli.core'] = sys.modules['pygimli'].core
sys.modules['pygimli.matrix'] = sys.modules['pygimli'].matrix

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
autodoc_mock_imports = mock_modules + ['pygimli']
nbsphinx_allow_errors = True

# Sphinx Gallery configuration - DISABLE execution for docs build
sphinx_gallery_conf = {
    'examples_dirs': '../../examples',         # path to your example scripts
    'gallery_dirs': 'auto_examples',           # path to output gallery
    'plot_gallery': 'True',                   # Generate gallery but don't execute
    'download_all_examples': True,            # Allow download of example files
    'filename_pattern': '/Ex.*\.py$',
    'ignore_pattern': '__pycache__|\.ipynb$',
    'expected_failing_examples': [            # Mark all examples as expected to fail
        '../../examples/Ex1_model_output.py',
        '../../examples/Ex2_workflow.py',
        '../../examples/Ex3_Time_lapse_measurement.py',
        '../../examples/Ex4_TL_inversion.py',
        '../../examples/Ex5_SRT.py',
        '../../examples/Ex6_Structure_resinv.py',
        '../../examples/Ex7_structure_TLresinv.py',
        '../../examples/Ex8_MC_WC.py',
    ],
    'capture_repr': (),
    'abort_on_example_error': False,
    'run_stale_examples': False,
    'first_notebook_cell': '%matplotlib inline\n'
                          '# This example requires data files and dependencies\n'
                          '# that are not available in the documentation build\n'
                          'print("Example code shown for reference only")',
}

templates_path = ['_templates']
exclude_patterns = []

source_suffix = {
    '.rst': None,
    '.md': 'myst_parser',
}

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'

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
    "github_user": "geohang",
    "github_repo": "PyHydroGeophysX",
    "github_version": "main",
    "conf_py_path": "/docs/source/",
}

html_static_path = ['_static']

# Create _static directory if it doesn't exist
static_dir = os.path.join(os.path.dirname(__file__), '_static')
if not os.path.exists(static_dir):
    os.makedirs(static_dir)

# Add custom CSS if needed
html_css_files = []

# Suppress warnings about missing static path
html_static_path = ['_static'] if os.path.exists(static_dir) else []

# Add custom roles for PyGIMLi documentation references
def setup(app):
    """Custom setup function for Sphinx."""
    # Define custom role for gimliapi references
    def gimliapi_role(name, rawtext, text, lineno, inliner, options={}, content=[]):
        """Custom role for GIMLI API references."""
        from docutils import nodes
        node = nodes.literal(rawtext, text)
        return [node], []
    
    app.add_role('gimliapi', gimliapi_role)
    return {'version': '0.1'}