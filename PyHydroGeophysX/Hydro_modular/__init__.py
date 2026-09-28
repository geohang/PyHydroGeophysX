"""
Hydro_modular package for hydrologic to geophysical conversion utilities.

hydro_to_ert and hydro_to_srt need PyGIMLi and are imported on first use, so
this package and its PyGIMLi-free helpers (the profile tools in
hydro_to_geophysics) import without it; asking for either of the two without
PyGIMLi raises an ImportError that says what is missing. hydro_to_tdem,
hydro_to_fdem and hydro_to_gravity need SimPEG as well; without it they are
placeholders that say so when called. Imported unguarded, a missing SimPEG made
this package, and so hydro_to_ert, fail to import at all.
"""

import importlib
import sys
import types

from PyHydroGeophysX._internal.optional_dependencies import optional_import_error

#: Exports that need PyGIMLi, each defined in the submodule of the same name.
_PYGIMLI_EXPORTS = ("hydro_to_ert", "hydro_to_srt")


def _unavailable(name, error):
    """A stand-in for ``name`` that raises why it could not be imported."""
    def unavailable(*args, **kwargs):
        raise optional_import_error(name, error) from error

    unavailable.__name__ = unavailable.__qualname__ = name
    unavailable.__doc__ = f"{name} could not be imported: {error}"
    # For callers that check before they start work (run_hydro_forward).
    unavailable.unavailable_because = error
    return unavailable


try:
    from PyHydroGeophysX.Hydro_modular.hydro_to_tdem import hydro_to_tdem
except ImportError as error:
    hydro_to_tdem = _unavailable("hydro_to_tdem", error)
try:
    from PyHydroGeophysX.Hydro_modular.hydro_to_fdem import hydro_to_fdem
except ImportError as error:
    hydro_to_fdem = _unavailable("hydro_to_fdem", error)
try:
    from PyHydroGeophysX.Hydro_modular.hydro_to_gravity import hydro_to_gravity
except ImportError as error:
    hydro_to_gravity = _unavailable("hydro_to_gravity", error)

__all__ = [
    'hydro_to_ert',
    'hydro_to_srt',
    'hydro_to_tdem',
    'hydro_to_fdem',
    'hydro_to_gravity'
]


def __getattr__(name):
    if name not in _PYGIMLI_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        module = importlib.import_module(f"{__name__}.{name}")
    except ImportError as exc:
        raise optional_import_error(name, exc) from exc
    # Importing the submodule has bound the function here too; see _Package.
    return getattr(module, name)


def __dir__():
    return sorted(set(globals()) | set(__all__))


class _Package(types.ModuleType):
    """Keeps the PyGIMLi exports bound to the functions, not their submodules.

    Importing a submodule binds it onto this package under its own name, which
    for hydro_to_ert and hydro_to_srt is also the function it defines. Without
    this, an ``import PyHydroGeophysX.Hydro_modular.hydro_to_ert`` anywhere
    would make ``from PyHydroGeophysX.Hydro_modular import hydro_to_ert``
    return the module; the eager imports this replaces never allowed that.
    """

    def __setattr__(self, name, value):
        if name in _PYGIMLI_EXPORTS and isinstance(value, types.ModuleType):
            value = getattr(value, name)
        super().__setattr__(name, value)


sys.modules[__name__].__class__ = _Package
