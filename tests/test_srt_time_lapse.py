import numpy as np
import pytest
import scipy.sparse as sp

pg = pytest.importorskip("pygimli")

from PyHydroGeophysX.inversion.srt_time_lapse import TimeLapseSRTInversion


class _DummyParaDomain:
    def __init__(self, n_cells: int):
        self._n_cells = n_cells

    def cellCount(self):
        return self._n_cells


class _DummyFop:
    def __init__(self):
        self.paraDomain = _DummyParaDomain(2)
        self._jac = np.eye(2)

    def response(self, slowness):
        s = np.asarray(slowness, dtype=float).ravel()
        return np.array([1000.0 * s[0], 1000.0 * s[1]])

    def createJacobian(self, slowness):
        self._jac = np.diag([1000.0, 1000.0])

    def jacobian(self):
        return self._jac


def test_srt_time_lapse_constructor_defaults(monkeypatch):
    data = pg.DataContainer()
    monkeypatch.setattr(pg, "load", lambda _: data)

    inv = TimeLapseSRTInversion(
        data_files=["t0.sgt", "t1.sgt"],
        measurement_times=[0.0, 1.0],
    )

    assert inv.parameters["lambda_val"] == 50.0
    assert inv.parameters["alpha"] == 10.0
    assert inv.parameters["method"] == "cgls"


def test_srt_time_lapse_run_with_mocked_setup(monkeypatch):
    data = pg.DataContainer()
    monkeypatch.setattr(pg, "load", lambda _: data)
    monkeypatch.setattr(pg.utils, "gmat2numpy", lambda x: np.asarray(x))

    def _fake_setup(self):
        self.n_cells = 2
        self.n_times = 2
        self.fops = [_DummyFop(), _DummyFop()]
        self.t_obs = np.array([[1.0], [1.1], [1.0], [1.1]], dtype=float)
        self.Wd_diag = np.ones(4, dtype=float)
        self.Wd = sp.diags(self.Wd_diag)
        self.Wd_sq = sp.diags(self.Wd_diag**2)
        self.Wm = sp.eye(4, format="csr")
        self.Wt = sp.csr_matrix([[1.0, 0.0, -1.0, 0.0], [0.0, 1.0, 0.0, -1.0]])
        self._setup_complete = True

    monkeypatch.setattr(TimeLapseSRTInversion, "setup", _fake_setup)

    inv = TimeLapseSRTInversion(
        data_files=["t0.sgt", "t1.sgt"],
        measurement_times=[0.0, 1.0],
        max_iterations=4,
    )
    result = inv.run(initial_model=np.array([1000.0, 900.0]))

    assert result.final_models is not None
    assert result.final_models.shape == (2, 2)
    assert len(result.all_chi2) > 0
    assert result.predicted_data is not None
