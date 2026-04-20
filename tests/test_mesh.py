import pytest

try:
    import pygimli
    _ = pygimli.RVector3  # only available when the C extension loaded successfully
    _PYGIMLI_OK = True
except (ImportError, AttributeError):
    _PYGIMLI_OK = False

pytestmark = pytest.mark.skipif(not _PYGIMLI_OK, reason="pygimli C extension not functional")


def test_mesh_creator_import():
    from PyHydroGeophysX.core import mesh_utils  # noqa: F401


def test_profile_interpolator_import():
    from PyHydroGeophysX.core.interpolation import ProfileInterpolator  # noqa: F401
