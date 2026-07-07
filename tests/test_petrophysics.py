"""Petrophysics rock-physics model tests (numpy/scipy only)."""

import numpy as np
import pytest

from PyHydroGeophysX.petrophysics.resistivity_models import (
    WS_Model,
    resistivity_to_water_content,
    water_content_to_resistivity,
)
from PyHydroGeophysX.petrophysics.velocity_models import water_content_to_velocity


def test_water_content_resistivity_round_trip():
    porosity = np.full(5, 0.3)
    wc = np.linspace(0.05, 0.28, 5)
    res = water_content_to_resistivity(wc, rhos=100.0, n=2.0, porosity=porosity)
    back = resistivity_to_water_content(res, rhos=100.0, n=2.0, porosity=porosity)
    assert np.allclose(back, wc, rtol=1e-6)


def test_more_water_means_lower_resistivity():
    res = water_content_to_resistivity(
        np.array([0.05, 0.15, 0.29]), rhos=100.0, n=2.0, porosity=0.3
    )
    assert res[0] > res[1] > res[2]


def test_ws_model_resistivity_drops_with_saturation():
    res = WS_Model(
        np.array([0.2, 0.5, 1.0]), porosity=0.3, sigma_w=0.05, m=1.5, n=2.0
    )
    assert res[0] > res[1] > res[2]


def test_full_saturation_recovers_rhos():
    res = water_content_to_resistivity(
        np.array([0.3]), rhos=250.0, n=2.0, porosity=0.3
    )
    assert np.allclose(res, 250.0)


def test_velocity_models_bounded_and_linear_endpoints():
    wc = np.linspace(0.0, 0.3, 7)
    for model in ("linear", "wyllie", "raymer"):
        vel = water_content_to_velocity(
            wc, v_dry=3500.0, v_sat=4500.0, porosity=0.3, model=model
        )
        assert np.all(vel >= 200.0) and np.all(vel <= 8000.0)
    linear = water_content_to_velocity(wc, porosity=0.3, model="linear")
    assert linear[0] == pytest.approx(3500.0)
    assert linear[-1] == pytest.approx(4500.0)
    assert np.all(np.diff(linear) >= 0.0)


def test_velocity_unknown_model_raises():
    with pytest.raises(ValueError):
        water_content_to_velocity(np.array([0.1]), model="nope")
