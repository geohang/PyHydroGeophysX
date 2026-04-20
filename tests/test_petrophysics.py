import numpy as np


def test_water_content_to_resistivity_signature():
    from PyHydroGeophysX.petrophysics import water_content_to_resistivity
    wc = np.array([[0.22, 0.28], [0.31, 0.35]])
    rho = water_content_to_resistivity(wc, rhos=100.0, n=2.0, porosity=0.3)
    assert rho.shape == wc.shape
    assert np.all(rho > 0)
    # Wetter soil -> lower resistivity (monotonicity sanity check)
    assert rho[1, 1] < rho[0, 0]


def test_hertz_mindlin_import():
    from PyHydroGeophysX.petrophysics import HertzMindlinModel
    m = HertzMindlinModel()
    assert hasattr(m, "calculate_velocity")
