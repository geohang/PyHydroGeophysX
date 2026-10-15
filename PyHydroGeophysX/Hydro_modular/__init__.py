"""
Hydro_modular package for hydrologic to geophysical conversion utilities.

Conversion functions load their optional numerical engines only when requested.
PyGIMLi exports raise an informative ImportError when unavailable; the SimPEG
exports retain their callable placeholders with ``unavailable_because``.
"""

import importlib
import sys
import types

from PyHydroGeophysX._internal.optional_dependencies import optional_import_error

#: Each function is defined in the submodule of the same name.
_PYGIMLI_EXPORTS = ("hydro_to_ert", "hydro_to_srt")
_SIMPEG_EXPORTS = ("hydro_to_tdem", "hydro_to_fdem", "hydro_to_gravity")
_EXPORTS = _PYGIMLI_EXPORTS + _SIMPEG_EXPORTS


def _unavailable(name, error):
    """A stand-in for ``name`` that raises why it could not be imported."""
    def unavailable(*args, **kwargs):
        raise optional_import_error(name, error) from error

    unavailable.__name__ = unavailable.__qualname__ = name
    unavailable.__doc__ = f"{name} could not be imported: {error}"
    # For callers that check before they start work (run_hydro_forward).
    unavailable.unavailable_because = error
    return unavailable


__all__ = [
    'hydro_to_ert',
    'hydro_to_srt',
    'hydro_to_tdem',
    'hydro_to_fdem',
    'hydro_to_gravity'
]


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        module = importlib.import_module(f"{__name__}.{name}")
    except ImportError as exc:
        if name in _PYGIMLI_EXPORTS:
            raise optional_import_error(name, exc) from exc
        value = _unavailable(name, exc)
        globals()[name] = value
        return value
    # Importing the submodule has bound the function here too; see _Package.
    return getattr(module, name)


def __dir__():
    return sorted(set(globals()) | set(__all__))


class _Package(types.ModuleType):
    """Keeps the conversion exports bound to the functions, not their submodules.

    Importing a submodule binds it onto this package under its own name, which
    for hydro_to_ert and hydro_to_srt is also the function it defines. Without
    this, an ``import PyHydroGeophysX.Hydro_modular.hydro_to_ert`` anywhere
    would make ``from PyHydroGeophysX.Hydro_modular import hydro_to_ert``
    return the module; the eager imports this replaces never allowed that.
    """

    def __setattr__(self, name, value):
        if name in _EXPORTS and isinstance(value, types.ModuleType):
            value = getattr(value, name)
        super().__setattr__(name, value)


sys.modules[__name__].__class__ = _Package
