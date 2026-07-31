"""Compatibility shim: the provider layer moved to :mod:`PyHydroGeophysX.llm.providers`.

The implementation is Qt-free and shared by the desktop chat panel and the
Streamlit AQUAH chat, so it now lives outside ``qt_apps``. Importing this
module hands back the relocated module object itself, which keeps every
existing name (public and private) working. New code should import from
``PyHydroGeophysX.llm.providers`` directly.
"""

import sys as _sys

from PyHydroGeophysX.llm import providers as _providers

_sys.modules[__name__] = _providers
