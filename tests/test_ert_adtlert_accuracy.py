"""Pin PyHydroGeophysX to the published AD-TLERT inversion settings."""

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("adtlert")

from PyHydroGeophysX.inversion.ert_inversion import (  # noqa: E402
    _ADTLertEngine,
    _diverged,
    _enable_adtlert_float64,
)


def test_single_inversion_uses_paper_sensitivity_configuration(monkeypatch):
    """The public backend must not silently drift from the paper workflow."""
    import adtlert.inversion as adtlert_inversion

    captured = {}

    def fake_invert(forward, observed, initial, *, reference_model, config):
        captured["config"] = config
        return SimpleNamespace(
            iteration_chi2=[2.0, 1.2],
            final_model=np.asarray(initial, dtype=float),
            predicted_data=np.asarray(observed, dtype=float),
            coverage=np.ones_like(initial, dtype=float),
        )

    monkeypatch.setattr(
        adtlert_inversion, "invert_single_log_resistivity", fake_invert
    )
    engine = _ADTLertEngine.__new__(_ADTLertEngine)
    engine._forward = object()
    engine._observed = np.array([100.0, 120.0])
    engine._errors = np.array([0.05, 0.05])
    engine._initial_model = np.array([110.0, 110.0, 110.0])
    engine._model_constraints = (1.0, 1.0e5)
    engine._solver = "gpu_cgls"
    engine._adtlert_version = "test"
    engine.mesh = object()
    engine.container = SimpleNamespace(size=lambda: 2)

    result = engine.fit(
        lam=50.0,
        max_iterations=5,
        plateau_tolerance=1.0e-4,
        target_chi2=1.0,
    )
    config = captured["config"]
    assert config.normal_sensitivity is True
    assert config.include_robin_boundary_derivative is False
    assert config.linearized_solver == "gpu_cgls"
    assert config.max_log_step == 1.0
    assert config.line_search is True
    assert result.metrics["sensitivity_profile"] == "paper"


def test_a_climbing_misfit_is_reported_as_divergence():
    class Run:
        convergence = [10.0, 40.0, 90.0]
        chi2 = 90.0

    class Falling:
        convergence = [10.0, 4.0, 2.0]
        chi2 = 2.0

    assert _diverged(Run()) is True
    assert _diverged(Falling()) is False
    assert _diverged(type("Empty", (), {"convergence": []})()) is False


def test_float64_is_forced_not_merely_defaulted(monkeypatch):
    monkeypatch.setenv("ADTLERT_ENABLE_FLOAT64", "0")
    _enable_adtlert_float64()
    import os
    assert os.environ["ADTLERT_ENABLE_FLOAT64"] == "1"
