# Configuration file for the Sphinx documentation builder.

import os
import sys
import unittest.mock as mock
import logging
logging.getLogger().setLevel(logging.WARNING)

# Add the project root to the Python path
project_root = os.path.abspath('../../')
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'PyHydroGeophysX'))

print(f"Python path: {sys.path[:3]}")  # Debug info

# Mock ALL problematic imports before anything else
mock_modules = [
    # PyGIMLI and related
    'pygimli', 'pygimli.physics', 'pygimli.physics.ert', 
    'pygimli.physics.traveltime', 'pygimli.meshtools',
    'pygimli.viewer', 'pygimli.viewer.mpl', 'pygimli.utils',
    'pygimli.core', 'pygimli.matrix', 'pygimli.rrng', 
    'pygimli.rrng.randpin', 'pygimli.manager',
    
    # Hydrological modeling
    'flopy', 'flopy.mf6', 'flopy.modflow', 'flopy.utils',
    'parflow', 'parflow.tools', 'parflow.tools.io',
    
    # GPU and parallel computing
    'cupy', 'cupyx', 'cupyx.scipy', 'cupyx.scipy.sparse',
    'joblib', 'multiprocessing',
    
    # Other optional dependencies
    'meshop', 'pygeostat', 'gempy',
    
    # Color palettes
    'palettable', 'palettable.lightbartlein', 
    'palettable.lightbartlein.diverging',
    'palettable.cartocolors', 'palettable.cartocolors.diverging',
]

# Create comprehensive mocks
class EnhancedMock(mock.MagicMock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Return self for chained calls
        self.__getitem__ = mock.MagicMock(return_value=self)
        self.__setitem__ = mock.MagicMock()

for mod_name in mock_modules:
    mock_obj = EnhancedMock()
    
    # Add specific attributes for known modules
    if mod_name == 'pygimli':
        # Add all commonly used pygimli functions and classes
        attrs = ['show', 'load', 'save', 'Mesh', 'Vector', 'DataContainer', 
                'randn', 'x', 'y', 'z', 'utils', 'meshtools', 'core', 
                'matrix', 'physics', 'viewer', 'rrng', 'createGrid',
                'createData', 'meshtools']
        for attr in attrs:
            setattr(mock_obj, attr, EnhancedMock())
            
    elif 'palettable' in mod_name:
        mock_obj.mpl_colormap = EnhancedMock()
        
    sys.modules[mod_name] = mock_obj

# Now safe to set up Sphinx
project = 'PyHydroGeophysX'
copyright = '2025, Hang Chen'
author = 'Hang Chen'
release = '0.1.0'

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

# Simplified gallery config - no execution
sphinx_gallery_conf = {
    'examples_dirs': '../../examples',
    'gallery_dirs': 'auto_examples',
    'plot_gallery': False,
    'download_all_examples': True,
    'filename_pattern': '/Ex.*\.py$',
    'ignore_pattern': '__pycache__|\.ipynb$|download_small_data\.py|create_example_data\.py',
    'abort_on_example_error': False,
    'run_stale_examples': False,
}

templates_path = ['_templates']
exclude_patterns = []

source_suffix = {
    '.rst': None,
    '.md': 'myst_parser',
}

# HTML output options
html_theme = 'sphinx_rtd_theme'
html_title = 'PyHydroGeophysX Documentation'

html_theme_options = {
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

# Suppress warnings
suppress_warnings = ['autodoc', 'autodoc.import_object']