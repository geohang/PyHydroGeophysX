import numpy as np
import pytest
import scipy.sparse as sp

pg = pytest.importorskip("pygimli")
tt = pytest.importorskip("pygimli.physics.traveltime")
pytest.importorskip("pygimli.physics.ert")

from PyHydroGeophysX.inversion.joint_ert_srt import JointERTSRTInversion


def test_joint_ert_srt_constructor_defaults(monkeypatch):
    dc = pg.DataContainer()
    monkeypatch.setattr(tt, "load", lambda _: dc)

    from pygimli.physics import ert as ert_module

    monkeypatch.setattr(ert_module, "load", lambda _: dc)

    inv = JointERTSRTInversion("dummy_ert.dat", "dummy_srt.sgt")
    assert inv.parameters["cross_gradient_mode"] == "direct"
    assert inv.parameters["regularization_mode"] == "smoothness"
    assert inv.parameters["lambda_cg_ert"] > 0
    assert inv.parameters["lambda_cg_srt"] > 0


def test_joint_ert_srt_run_with_mocked_setup(monkeypatch):
    dc = pg.DataContainer()
    monkeypatch.setattr(tt, "load", lambda _: dc)

    from pygimli.physics import ert as ert_module

    monkeypatch.setattr(ert_module, "load", lambda _: dc)

    def _fake_setup(self):
        self.mesh = object()
        self.Wd_ert = sp.eye(3, format="csr")
        self.Wd_srt = sp.eye(3, format="csr")
        self.Wm_ert = sp.eye(3, format="csr")
        self.Wm_srt = sp.eye(3, format="csr")
        self.dobs_ert = np.array([[1.0], [1.2], [0.8]])
        self.dobs_srt = np.array([[0.5], [0.4], [0.45]])
        self.mr = np.zeros(3, dtype=float)
        self.mv = np.zeros(3, dtype=float)
        self.mr_ref = self.mr.copy()
        self.mv_ref = self.mv.copy()
        self.X = np.array(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [2.0, 0.0, 1.0],
            ]
        )
        self.RCM = np.ones((3, 3), dtype=float)
        self._setup_complete = True

    def _fake_ert_forward_and_jac(self, model):
        m = np.asarray(model, dtype=float).reshape(-1, 1)
        return m, np.eye(m.shape[0], dtype=float)

    def _fake_srt_forward_and_jac(self, model):
        m = np.asarray(model, dtype=float).reshape(-1, 1)
        return m, np.eye(m.shape[0], dtype=float)

    monkeypatch.setattr(JointERTSRTInversion, "setup", _fake_setup)
    monkeypatch.setattr(JointERTSRTInversion, "_ert_forward_and_jac", _fake_ert_forward_and_jac)
    monkeypatch.setattr(JointERTSRTInversion, "_srt_forward_and_jac", _fake_srt_forward_and_jac)

    inv = JointERTSRTInversion(
        "dummy_ert.dat",
        "dummy_srt.sgt",
        max_iterations=4,
        solver="scipy_lsqr",
        lambda_ert=1e-6,
        lambda_srt=1e-6,
        lambda_cg_ert=0.0,
        lambda_cg_srt=0.0,
        ert_bounds=(0.1, 10.0),
        srt_velocity_bounds=(0.1, 10.0),
    )
    result = inv.run()

    assert result.ert_resistivity is not None
    assert result.srt_velocity is not None
    assert result.chi2_ert is not None
    assert result.chi2_srt is not None
    assert np.isfinite(result.chi2_ert)
    assert np.isfinite(result.chi2_srt)
    assert len(result.iteration_history) > 0
