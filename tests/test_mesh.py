import pytest

pygimli = pytest.importorskip("pygimli", reason="pygimli not installed")


def test_mesh_creator_import():
    from PyHydroGeophysX.core import mesh_utils  # noqa: F401


def test_profile_interpolator_import():
    from PyHydroGeophysX.core.interpolation import ProfileInterpolator  # noqa: F401
