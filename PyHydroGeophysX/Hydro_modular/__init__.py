"""
Hydro_modular package for hydrologic to geophysical conversion utilities.

hydro_to_ert and hydro_to_srt need PyGIMLi. hydro_to_tdem, hydro_to_fdem and
hydro_to_gravity need SimPEG as well; without it they are placeholders that say
so when called. Imported unguarded, a missing SimPEG made this package, and so
hydro_to_ert, fail to import at all.
"""

from PyHydroGeophysX._internal.optional_dependencies import optional_import_error
from PyHydroGeophysX.Hydro_modular.hydro_to_ert import hydro_to_ert
from PyHydroGeophysX.Hydro_modular.hydro_to_srt import hydro_to_srt


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
