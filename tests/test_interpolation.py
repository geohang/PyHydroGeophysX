"""Profile interpolation tests for the regular-grid fast path and fallbacks."""

import numpy as np
import pytest

from PyHydroGeophysX.core.interpolation import (
    interpolate_to_profile,
    prepare_2D_profile_data,
    setup_profile_coordinates,
)


@pytest.fixture()
def regular_grid():
    x = np.linspace(0.0, 9.0, 10)
    y = np.linspace(0.0, 4.0, 5)
    return np.meshgrid(x, y)


def test_linear_field_is_exact_on_regular_grid(regular_grid):
    XX, YY = regular_grid
    data = 2.0 * XX + 3.0 * YY + 1.0
    x_pro = np.linspace(0.5, 8.5, 17)
    y_pro = np.linspace(0.2, 3.8, 17)
    sampled = interpolate_to_profile(data, XX, YY, x_pro, y_pro)
    assert np.allclose(sampled, 2.0 * x_pro + 3.0 * y_pro + 1.0, atol=1e-9)


def test_nearest_method(regular_grid):
    XX, YY = regular_grid
    data = XX + 10.0 * YY
    sampled = interpolate_to_profile(
        data, XX, YY, np.array([3.1]), np.array([2.1]), method="nearest"
    )
    assert sampled[0] == pytest.approx(23.0)


def test_outside_grid_yields_nan(regular_grid):
    XX, YY = regular_grid
    data = np.ones_like(XX)
    sampled = interpolate_to_profile(data, XX, YY, np.array([-5.0]), np.array([2.0]))
    assert np.isnan(sampled[0])


def test_irregular_grid_falls_back_to_griddata(regular_grid):
    XX, YY = regular_grid
    rng = np.random.default_rng(42)
    XXj = XX + rng.uniform(-0.01, 0.01, XX.shape)
    data = 2.0 * XXj + 3.0 * YY
    sampled = interpolate_to_profile(data, XXj, YY, np.array([4.5]), np.array([2.0]))
    assert sampled[0] == pytest.approx(2.0 * 4.5 + 3.0 * 2.0, abs=0.05)


def test_prepare_2d_profile_stacks_layers(regular_grid):
    XX, YY = regular_grid
    layers = np.stack([XX, YY, XX + YY])
    out = prepare_2D_profile_data(
        layers, XX, YY, np.array([1.0, 2.0]), np.array([1.0, 3.0])
    )
    assert out.shape == (3, 2)
    assert np.allclose(out[0], [1.0, 2.0])
    assert np.allclose(out[1], [1.0, 3.0])
    assert np.allclose(out[2], [2.0, 5.0])


def test_setup_profile_coordinates_shapes():
    surface = np.ones((5, 10))
    x_pro, y_pro, l_profile, XX, YY = setup_profile_coordinates(
        [1, 1], [8, 3], surface, num_points=50
    )
    assert x_pro.shape == y_pro.shape == l_profile.shape == (49,)
    assert l_profile[0] == pytest.approx(0.0)
    assert np.all(np.diff(l_profile) > 0.0)
    assert XX.shape == surface.shape and YY.shape == surface.shape
