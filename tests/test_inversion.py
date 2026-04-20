import pytest

try:
    import pygimli
    _ = pygimli.RVector3  # only available when the C extension loaded successfully
    _PYGIMLI_OK = True
except (ImportError, AttributeError):
    _PYGIMLI_OK = False

pytestmark = pytest.mark.skipif(not _PYGIMLI_OK, reason="pygimli C extension not functional")


def test_ert_inversion_import():
    from PyHydroGeophysX.inversion.ert_inversion import ERTInversion  # noqa: F401


def test_time_lapse_inversion_import():
    from PyHydroGeophysX.inversion import TimeLapseERTInversion  # noqa: F401
