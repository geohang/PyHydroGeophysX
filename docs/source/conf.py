# Final working Sphinx configuration for PyHydroGeophysX

import os
import sys
import importlib.util

# Add project to path
sys.path.insert(0, os.path.abspath('../../'))
sys.path.insert(0, os.path.abspath('../../PyHydroGeophysX'))

# Project information
project = 'PyHydroGeophysX'
copyright = '2025, Hang Chen'
author = 'Hang Chen'
release = '0.3.0'

# Extensions
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx_gallery.gen_gallery',
]

# Optional extensions (build should still work if unavailable)
for _ext in ['sphinx_copybutton', 'sphinx_design']:
    if importlib.util.find_spec(_ext):
        extensions.append(_ext)

# Sphinx Gallery configuration - FINAL WORKING VERSION
sphinx_gallery_conf = {
    'examples_dirs': '../../examples',           # Path to example scripts
    'gallery_dirs': 'auto_examples',             # Output gallery directory
    'filename_pattern': r'/(Ex|EX)_.*\.py$',
    'ignore_pattern': r'(app_.*|aquah_web|generate_synthetic_examples)\.py$',
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

# What the site does not build.
#
# documentation/ holds an older topic-by-topic guide whose pages are thin stubs
# pointing at the gallery. Tutorials now covers the same ground by user task, so
# building both publishes two guides competing in search while only one is
# maintained. The files stay on disk.
#
# The api/ directory carries two parallel sets: hand-curated pages that
# api/index.rst organises, and a sphinx-apidoc set named PyHydroGeophysX.*.rst
# reachable only through modules.rst. Building both documents every symbol
# twice, which is where the bulk of the duplicate-object warnings came from.
# Only the apidoc pages that have a curated counterpart are dropped, so a module
# documented in one place only, such as llm and workflows, keeps its page.
_api_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'api')
_curated = {
    name[:-4] for name in os.listdir(_api_dir)
    if name.endswith('.rst') and not name.startswith('PyHydroGeophysX')
}
exclude_patterns = ['documentation/**', 'api/modules.rst', 'api/PyHydroGeophysX.rst']
exclude_patterns += [
    'api/PyHydroGeophysX.%s.rst' % module
    for module in (
        name[len('PyHydroGeophysX.'):-4] for name in os.listdir(_api_dir)
        if name.startswith('PyHydroGeophysX.') and name.endswith('.rst')
    )
    if module in _curated
]

# HTML theme
if importlib.util.find_spec('pydata_sphinx_theme'):
    html_theme = 'pydata_sphinx_theme'
else:
    html_theme = 'alabaster'
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
        {
            'name': 'Environmental Geophysics Course',
            'url': 'https://geohang.github.io/environmental-geophysics/',
            'icon': 'fa-solid fa-graduation-cap',
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
    'tqdm', 'matplotlib', 'scipy', 'numpy', 'simpeg', 'discretize',
    'resipy', 'openai', 'google', 'google.generativeai', 'anthropic'
]

# GitHub Pages
html_baseurl = 'https://geohang.github.io/PyHydroGeophysX/'

# Usage and Downloads dashboard (docs/source/usage.rst). The assets read
# _static/stats/*.json, which tools/fetch_usage_stats.py regenerates.
html_css_files = ['usage-stats.css', 'site.css']
html_js_files = ['usage-stats.js']

# That page has no child documents, so its left sidebar renders an empty
# "Section Navigation" block. Dropping it gives the world map the width it
# needs; the right-hand "On this page" list still carries the in-page nav.
html_sidebars = {'usage': [], 'index': []}

# Optional privacy-friendly analytics for the visitor map on that page.
# Set this to the GoatCounter subdomain (for example 'pyhydrogeophysx') to start
# collecting page views; leave it empty and no tracking script is emitted at all.
# The GOATCOUNTER_SITE environment variable overrides it, and the matching
# GOATCOUNTER_TOKEN secret lets the stats workflow read the counts back.
GOATCOUNTER_SITE = ''

goatcounter_site = os.environ.get('GOATCOUNTER_SITE', GOATCOUNTER_SITE).strip()

if goatcounter_site:
    html_js_files.append(
        (
            'https://gc.zgo.at/count.js',
            {
                'data-goatcounter': f'https://{goatcounter_site}.goatcounter.com/count',
                'async': 'async',
            },
        )
    )

# External links known to return 403 to automated linkcheck clients
linkcheck_ignore = [
    r'https://ssrn\.com/abstract=6238293',
    r'https://doi\.org/10\.2139/ssrn\.6238293',
    r'https://papers\.ssrn\.com/sol3/papers\.cfm\?abstract_id=6238293',
]
