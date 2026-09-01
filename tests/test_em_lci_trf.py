"""Bound-aware LCI: full objective, explicit budgets and unchanged data errors."""
import numpy as np
import pytest
from PyHydroGeophysX.inversion.em1d_lci import SoundingBlock, solve_lci, invert_lci_with_robust_errors


def linear_block(A, d, **kwargs):
    A = np.asarray(A, float)
    return SoundingBlock(
        forward=lambda sigma: A @ -np.log10(sigma),
        jacobian=lambda sigma: -A/(np.log(10)*sigma),
        dobs=np.asarray(d, float), uncertainty=np.ones(len(d)), **kwargs)


def test_bound_aware_step_finds_constrained_solution():
    # The unconstrained Newton optimum is outside x>=0. Its clipped direction
    # cannot leave the constrained corner, though x[0] is free to improve.
    b = linear_block([[1., 2.], [0., 1.]], [1., -2.])
    out = solve_lci([b], 2, solver="trf", initial_model=np.ones((1, 2)),
                    smoothness=0, lateral_smoothness=0, bounds=(0, 4),
                    target_chi2=0, verbose=False, trf_ftol=1e-10, trf_xtol=1e-10)
    np.testing.assert_allclose(np.log10(out.models[0]), [1., 0.], atol=1e-4)
    assert out.diagnostics["solver_converged"]
    assert out.diagnostics["projected_gradient_inf"] < 1e-4
    assert out.chi2 == pytest.approx(2.)
    np.testing.assert_array_equal(b.uncertainty, [1., 1.])


def test_formal_lci_default_is_balanced_trf():
    out = solve_lci([linear_block(np.eye(2), [1., 2.])], 2,
                    smoothness=0., lateral_smoothness=0., target_chi2=0.,
                    verbose=False)
    assert out.diagnostics["solver"] == "trf"
    assert out.diagnostics["max_nfev"] == 90
    assert out.diagnostics["ftol"] == pytest.approx(1e-4)


def test_shallow_hinge_and_regularization_share_the_exact_objective():
    b = linear_block(np.eye(2), [2., 2.], prior_lower=np.array([3., 3.]),
                     prior_weights=np.array([2., 0.]))
    out = solve_lci([b], 2, solver="trf", smoothness=0, lateral_smoothness=0,
                    target_chi2=0, verbose=False)
    np.testing.assert_allclose(np.log10(out.models[0]), [2.8, 2.], atol=1e-5)
    assert out.diagnostics["objective"] == pytest.approx(.8)
    assert out.n_data == 2
    assert np.all(np.diff(out.diagnostics["objective_history"]) <= 1e-10)


def test_full_objective_can_improve_even_when_data_fit_starts_perfect():
    b = linear_block(np.eye(2), [1., 3.])
    out = solve_lci([b], 2, solver="trf", initial_model=np.array([[10., 1000.]]),
                    smoothness=10, target_chi2=0, verbose=False)
    assert out.chi2 > 0
    assert out.diagnostics["objective"] < 400
    assert np.all(np.diff(out.diagnostics["objective_history"]) <= 1e-10)


def test_budget_exhaustion_is_not_convergence_or_a_reason_to_inflate_errors():
    b = linear_block(np.eye(2), [0., 4.])
    out, blocks, info = invert_lci_with_robust_errors(
        [b], 2, solver="trf", trf_max_nfev=1, auto_lambda=True,
        error_target_chi2=1.75, threshold=1., starting_resistivity=10000., verbose=False)
    assert out.stop_reason == "max_nfev"
    assert not out.diagnostics["solver_converged"]
    assert out.lambda_search["status"] == "solver_incomplete"
    assert info["downweighted"] == info["dropped"] == 0
    assert not info["target_reached"]
    assert "inner solver incomplete" in info["stopped_because"]
    np.testing.assert_array_equal(blocks[0].uncertainty, b.uncertainty)


def test_separate_station_heights_are_not_a_common_qt_override():
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from PyHydroGeophysX.qt_apps.modules.em_processing import EMProcessingModule
    app = QApplication.instance() or QApplication([])
    panel = EMProcessingModule(None, lambda *args, **kwargs: None)
    try:
        panel._data = {"heights": np.array([1., 1.]), "n_soundings": 2,
                       "system": {"tx_height": .9, "rx_height": 1.}}
        panel._geom_heights = np.array([30., 30.])
        panel._apply_temcompany_metadata()
        assert panel._geom_heights is None
        geom = panel._collect_geom()
        assert geom["tx_height"] == .9
        assert geom["rx_height"] == 1.
    finally:
        panel.close()


def test_invalid_solver_is_not_silently_ignored():
    with pytest.raises(ValueError, match="solver"):
        solve_lci([linear_block(np.eye(2), [1., 2.])], 2, solver="typo")


def test_trf_stacked_jacobian_matches_finite_difference(monkeypatch):
    import scipy.optimize
    real = scipy.optimize.least_squares
    checked = []

    def verify(fun, x0, jac, **kwargs):
        x = np.array([1.1, 2.2, 2.1, 1.2])
        direction = np.array([.1, -.2, .3, -.4])
        eps = 1e-6
        analytic = jac(x) @ direction
        fd = (fun(x+eps*direction)-fun(x-eps*direction))/(2*eps)
        np.testing.assert_allclose(analytic, fd, rtol=1e-7, atol=1e-8)
        checked.append(True)
        return real(fun, x0, jac=jac, **kwargs)

    monkeypatch.setattr(scipy.optimize, "least_squares", verify)
    blocks = [linear_block([[1., 2.], [3., 1.]], [2., 3.], position=i,
                          prior_lower=np.array([2., 2.]), prior_weights=np.array([.8, .3]))
              for i in range(2)]
    solve_lci(blocks, 2, solver="trf", smoothness=1.7, lateral_smoothness=.6,
              target_chi2=0, verbose=False)
    assert checked == [True]


def test_trf_threading_preserves_result():
    blocks = [linear_block([[1., 2.], [3., 1.]], [2.+i/10, 3.], position=i) for i in range(4)]
    a = solve_lci(blocks, 2, solver="trf", target_chi2=0, parallel_workers=1, verbose=False)
    b = solve_lci(blocks, 2, solver="trf", target_chi2=0, parallel_workers=3, verbose=False)
    np.testing.assert_array_equal(a.models, b.models)
    assert a.chi2 == b.chi2


@pytest.mark.parametrize("solver", ["trf", "gauss_newton"])
def test_median_history_is_equal_sounding_not_equal_gate_and_excludes_empty(solver):
    blocks = [linear_block([[.5, .5]], [1.], position=0),
              linear_block(np.full((3, 2), .5), [0., 0., 0.], position=1),
              linear_block(np.empty((0, 2)), [], position=2)]
    out = solve_lci(blocks, 2, solver=solver, starting_resistivity=100.,
                    target_chi2=0, max_iterations=2, trf_max_nfev=6, verbose=False)
    assert out.chi2_history[0] == pytest.approx(3.25)
    assert out.chi2_median_history[0] == pytest.approx(2.5)
    assert len(out.chi2_history) == len(out.chi2_median_history)
    assert out.chi2_median_history[-1] == pytest.approx(np.nanmedian(out.chi2_per_sounding))
    assert out.as_dict()["chi2_median_history"] == out.chi2_median_history
