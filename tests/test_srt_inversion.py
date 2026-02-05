import numpy as np
import pytest
import scipy.sparse as sp

pg = pytest.importorskip("pygimli")

from PyHydroGeophysX.inversion.srt_inversion import SRTInversion


class _DummyParaDomain:
    def __init__(self, n_cells: int):
        self._n_cells = n_cells

    def cellCount(self):
        return self._n_cells


class _DummyFop:
    def __init__(self):
        self.paraDomain = _DummyParaDomain(3)
        self._jac = np.eye(3)

    def response(self, slowness):
        s = np.asarray(slowness, dtype=float).ravel()
        return np.array([1000.0 * s[0], 1000.0 * s[1], 1000.0 * s[2]])

    def createJacobian(self, slowness):
        self._jac = np.diag([1000.0, 1000.0, 1000.0])

    def jacobian(self):
        return self._jac


def test_srt_inversion_constructor_defaults(monkeypatch):
    data = pg.DataContainer()
    monkeypatch.setattr(pg, "load", lambda _: data)

    inv = SRTInversion("dummy.sgt")

    assert inv.parameters["lambda_val"] == 50.0
    assert inv.parameters["method"] == "cgls"
    assert inv.parameters["zWeight"] == 0.2
    assert inv.parameters["vTop"] == 500.0
    assert inv.parameters["vBottom"] == 5000.0


def test_srt_inversion_run_with_mocked_setup(monkeypatch):
    data = pg.DataContainer()
    monkeypatch.setattr(pg, "load", lambda _: data)
    monkeypatch.setattr(pg.utils, "gmat2numpy", lambda x: np.asarray(x))

    def _fake_setup(self):
        self.fop = _DummyFop()
        self.t_obs = np.array([[1.0], [1.2], [1.5]], dtype=float)
        self.Wd_diag = np.ones(3, dtype=float)
        self.Wd = sp.diags(self.Wd_diag)
        self.Wd_sq = sp.diags(self.Wd_diag**2)
        self.Wm = sp.eye(3, format="csr")
        self.mesh = None
        self._setup_complete = True

    monkeypatch.setattr(SRTInversion, "setup", _fake_setup)

    inv = SRTInversion("dummy.sgt", max_iterations=4)
    result = inv.run(initial_model=np.array([1000.0, 900.0, 800.0]))

    assert result.final_model is not None
    assert result.final_model.size == 3
    assert np.all(np.isfinite(result.final_model))
    assert result.predicted_data is not None
    assert len(result.iteration_chi2) > 0
