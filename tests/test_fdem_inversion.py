import numpy as np
import pytest

pytest.importorskip("simpeg")

import PyHydroGeophysX.inversion.fdem_inversion as fdem_inv


def test_fdem_vector_helpers():
    arr = np.array([1.0 + 2.0j, 3.0 + 4.0j])
    vec = fdem_inv.FDEMInversion._to_simpeg_vector(arr)
    packed = fdem_inv.FDEMInversion._pack_complex(vec)

    assert vec.shape == (4,)
    assert packed.shape == (2,)
    assert np.allclose(packed, arr)


def test_fdem_inversion_run_with_mocked_baseinversion(monkeypatch):
    class _DummyBaseInversion:
        def __init__(self, inv_prob, directives_list):
            self.inv_prob = inv_prob

        def run(self, starting_model):
            self.inv_prob.l2model = np.asarray(starting_model).copy()
            return np.asarray(starting_model).copy()

    monkeypatch.setattr(fdem_inv.inversion, "BaseInversion", _DummyBaseInversion)

    frequencies = np.array([100.0, 1000.0])
    dobs = np.array([1e-9 + 1e-10j, 8e-10 + 9e-11j])
    unc = np.array([1e-10, 1e-10])

    inversion = fdem_inv.FDEMInversion(
        frequencies=frequencies,
        dobs=dobs,
        uncertainties=unc,
        thicknesses=np.array([10.0]),
        max_iterations=2,
        use_irls=False,
        verbose=False,
    )

    result = inversion.run()

    assert result.recovered_model is not None
    assert result.recovered_conductivity is not None
    assert result.predicted_data is not None
    assert result.frequencies.shape == frequencies.shape
    assert np.isfinite(result.chi2)
