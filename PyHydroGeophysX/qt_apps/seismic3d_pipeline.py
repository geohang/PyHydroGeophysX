"""Compatibility alias for the promoted Seismic3D pipeline.

Aliasing the module object preserves legacy monkeypatch behavior without
keeping a second function implementation in ``qt_apps``.
"""

import sys as _sys

from PyHydroGeophysX._internal.deprecations import warn_legacy_path as _warn_legacy_path
from PyHydroGeophysX.Geophy_modular import structure_integration as _canonical

_warn_legacy_path("qt_apps.seismic3d_pipeline", "Geophy_modular.structure_integration")
_sys.modules[__name__] = _canonical
