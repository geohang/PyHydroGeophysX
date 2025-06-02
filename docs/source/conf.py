# Enhanced mock configuration for documentation
import os
import sys
import unittest.mock as mock

# Add project paths
sys.path.insert(0, os.path.abspath('../../'))
sys.path.insert(0, os.path.abspath('../../PyHydroGeophysX'))

# Comprehensive mock list - include ALL problematic packages
mock_modules = [
    'pygimli', 'pygimli.physics', 'pygimli.physics.ert', 
    'pygimli.physics.traveltime', 'pygimli.meshtools',
    'pygimli.viewer', 'pygimli.viewer.mpl', 'pygimli.utils',
    'pygimli.core', 'pygimli.matrix', 'pygimli.rrng', 'pygimli.rrng.randpin',
    'flopy', 'flopy.mf6', 'flopy.modflow',
    'parflow', 'parflow.tools', 'parflow.tools.io',
    'cupy', 'cupyx', 'cupyx.scipy', 'cupyx.scipy.sparse',
    'joblib', 'meshop', 
    'palettable', 'palettable.lightbartlein', 'palettable.lightbartlein.diverging',
    'palettable.cartocolors', 'palettable.cartocolors.diverging'
]

# Create comprehensive mocks
for mod_name in mock_modules:
    mock_obj = mock.MagicMock()
    
    # Add specific attributes for known modules
    if 'palettable' in mod_name:
        mock_obj.mpl_colormap = mock.MagicMock()
    elif mod_name == 'pygimli':
        # Add commonly used pygimli functions
        mock_obj.show = mock.MagicMock()
        mock_obj.load = mock.MagicMock()
        mock_obj.save = mock.MagicMock()
        mock_obj.Mesh = mock.MagicMock()
        mock_obj.Vector = mock.MagicMock()
        mock_obj.DataContainer = mock.MagicMock()
        mock_obj.randn = mock.MagicMock(return_value=mock.MagicMock())
        mock_obj.x = mock.MagicMock()
        mock_obj.y = mock.MagicMock()
        mock_obj.z = mock.MagicMock()
        mock_obj.utils = mock.MagicMock()
        mock_obj.utils.gmat2numpy = mock.MagicMock()
        mock_obj.utils.cMap = mock.MagicMock()
        mock_obj.utils.sparseMatrix2coo = mock.MagicMock()
        mock_obj.meshtools = mock.MagicMock()
        mock_obj.core = mock.MagicMock()
        mock_obj.core.coverageDCtrans = mock.MagicMock()
        mock_obj.matrix = mock.MagicMock()
        mock_obj.matrix.RSparseMapMatrix = mock.MagicMock()
        mock_obj.matrix.RVector = mock.MagicMock()
    
    sys.modules[mod_name] = mock_obj

# Rest of your configuration...
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

# Disable example execution for now to avoid dependency issues
sphinx_gallery_conf = {
    'examples_dirs': '../../examples',
    'gallery_dirs': 'auto_examples',
    'plot_gallery': False,  # Disable execution
    'download_all_examples': True,
    'filename_pattern': '/Ex.*\.py$',
    'ignore_pattern': '__pycache__|\.ipynb$',
    'expected_failing_examples': [],
    'abort_on_example_error': False,
    'run_stale_examples': False,
}

# ... rest of your configuration
html_theme = 'sphinx_rtd_theme'