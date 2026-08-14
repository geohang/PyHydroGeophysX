"""When the basemap may be registered to a survey, and when it may not.

The fit decides whether the Basemap control is available at all, so a rule that
is too strict does not degrade the picture, it removes the feature: the checkbox
greys out with no way for the reader to see why.
"""

import numpy as np
import pytest

from PyHydroGeophysX.visualization.basemap import (
    fit_local_transform,
    web_mercator,
)


def _survey(lon0: float, lat0: float, east_m: float, north_m: float, n: int = 200):
    """A survey of ``n`` stations, with projected metres and longitude/latitude.

    The projected pair is built from the geographic one through the same
    Mercator relation, then unscaled by the local convergence, which is what a
    UTM grid does to first order. Any residual left over is the departure the
    fit has to tolerate.
    """
    t = np.linspace(0.0, 1.0, n)
    metres_per_degree = 111320.0
    lon = lon0 + (east_m * t) / (metres_per_degree * np.cos(np.radians(lat0)))
    lat = lat0 + (north_m * t) / metres_per_degree
    mx, my = web_mercator(lon, lat)
    scale = 1.0 / np.cos(np.radians(lat0))
    return mx / scale, my / scale, lon, lat


def test_accepts_a_survey_whose_coordinates_disagree_by_a_metre():
    """A datum difference is not a reason to withhold imagery.

    NAD83 against WGS84 is one to two metres across the United States, and an
    instrument that writes one pair in each shows exactly that. Satellite
    imagery is georeferenced to several metres itself, so refusing here would
    reject an error smaller than the basemap's own.
    """
    x, y, lon, lat = _survey(-106.61, 38.91, 650.0, 890.0)
    shifted_x, shifted_y = x + 1.4, y - 0.9
    assert fit_local_transform(shifted_x, shifted_y, lon, lat) is not None


def test_accepts_a_survey_with_a_few_bad_fixes():
    """One station with a dropped fix must not disable the whole survey."""
    x, y, lon, lat = _survey(-91.58, 41.66, 120.0, 300.0)
    x = x.copy()
    x[5] += 60.0        # a bad fix, far outside the survey's own scatter
    x[73] -= 45.0
    assert fit_local_transform(x, y, lon, lat) is not None


def test_rejects_coordinates_of_a_different_shape():
    """Coordinates that trace a different figure cannot be registered.

    A similarity is free to rotate, scale and shift, so it maps any straight
    line onto any other straight line exactly; a single line offers nothing to
    check against and the fit rightly accepts it. What a survey with genuine
    two-dimensional spread does give is a shape, and pairing it with the
    coordinates of a different shape is what the test has to catch.
    """
    x, y, lon, lat = _survey(-106.61, 38.91, 650.0, 890.0, n=200)
    corner = np.arange(lon.size) > lon.size // 2
    bent_lon, bent_lat = lon.copy(), lat.copy()
    bent_lon[corner] += 0.004        # roughly 350 m east, a right-angle turn
    assert fit_local_transform(x, y, bent_lon, bent_lat) is None


def test_rejects_a_swapped_coordinate_pair_on_a_survey_with_shape():
    """Swapping longitude and latitude reflects the figure.

    Only visible on a survey that has a figure: for a straight line a
    reflection is also a rotation, which a similarity is entitled to apply,
    so the check has nothing to catch there.
    """
    x, y, lon, lat = _survey(-106.61, 38.91, 650.0, 890.0, n=200)
    corner = np.arange(lon.size) > lon.size // 2
    x = x.copy()
    x[corner] += 350.0
    assert fit_local_transform(x, y, lon, lat) is None      # shapes differ
    assert fit_local_transform(x, y, lat, lon) is None      # and swapped


def test_rejects_too_few_or_unusable_stations():
    x, y, lon, lat = _survey(-106.61, 38.91, 650.0, 890.0, n=2)
    assert fit_local_transform(x, y, lon, lat) is None
    x, y, lon, lat = _survey(-106.61, 38.91, 650.0, 890.0, n=20)
    assert fit_local_transform(np.full(20, np.nan), y, lon, lat) is None


def test_the_transform_places_stations_where_they_belong():
    """An accepted fit has to be usable, not merely non-None."""
    x, y, lon, lat = _survey(-106.61, 38.91, 650.0, 890.0)
    fit = fit_local_transform(x, y, lon, lat)
    assert fit is not None
    a, b = fit
    mx, my = web_mercator(lon, lat)
    placed = a * (x + 1j * y) + b
    error = np.abs(placed - (mx + 1j * my))
    # In Web Mercator metres, which at this latitude are about 1.3 times shorter
    # than ground metres, so this is well inside a pixel of the imagery.
    assert float(error.max()) < 8.0


@pytest.mark.parametrize("n", [3, 9, 500])
def test_survey_size_does_not_change_the_verdict(n):
    x, y, lon, lat = _survey(-106.61, 38.91, 650.0, 890.0, n=n)
    assert fit_local_transform(x, y, lon, lat) is not None
