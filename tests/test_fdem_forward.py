import numpy as np
import pytest

pytest.importorskip("simpeg")

from PyHydroGeophysX.forward.fdem_forward import FDEMForwardModeling, FDEMSurveyConfig


def test_fdem_forward_complex_response():
    config = FDEMSurveyConfig(
        frequencies=np.array([100.0, 1000.0]),
        receiver_component="secondary",
        waveform_type="dipole",
    )

    modeler = FDEMForwardModeling(thicknesses=np.array([10.0]), survey_config=config)
    response = modeler.forward(conductivity=np.array([0.01, 0.05]))

    assert response.size == config.frequencies.size
    assert np.iscomplexobj(response)


def test_fdem_forward_with_noise_shapes():
    config = FDEMSurveyConfig(
        frequencies=np.array([200.0, 500.0, 1000.0]),
        receiver_component="secondary",
        waveform_type="dipole",
    )

    modeler = FDEMForwardModeling(thicknesses=np.array([8.0, 12.0]), survey_config=config)
    noisy, clean, unc = modeler.forward_with_noise(
        conductivity=np.array([0.02, 0.05, 0.1]),
        noise_level=0.03,
        seed=0,
    )

    assert noisy.shape == clean.shape
    assert unc.shape == clean.shape
    assert np.all(np.isfinite(np.abs(noisy)))
