"""Bounded error targets and the empirical resistive-background prior."""
import numpy as np
import pytest

from PyHydroGeophysX.inversion.robust_errors import select_error_factors, reweight_errors
from PyHydroGeophysX.inversion.em1d_priors import (
    resistive_prior_target,
    shallow_prior_scores,
    shallow_prior_terms,
)
from PyHydroGeophysX.inversion.em1d_lci import SoundingBlock, solve_lci


@pytest.mark.parametrize("n", [1, 2, 3, 4, 9, 10, 4884])
def test_unchanged_quota_rounds_up_and_is_never_spent_to_hit_the_target(n):
    r = np.linspace(4, 100, n)
    f, info = select_error_factors(r, min_unchanged_fraction=.7, target_chi2=1.75)
    assert sum(f == 1) >= int(np.ceil(.7*n))
    assert np.all((f >= 1) & (f <= 10))
    np.testing.assert_array_equal(f[:int(np.ceil(.7*n))], 1.)
    assert info["fixed_model_min_chi2"] > 1.75


def test_reachable_target_changes_only_the_eligible_thirty_percent():
    r = np.r_[np.full(7, .5), 10., 12., 15.]
    f, info = select_error_factors(r, min_unchanged_fraction=.7, target_chi2=1.75)
    assert info["status"] == "target_matched_at_current_model"
    assert np.mean((r/f)**2) == pytest.approx(1.75)
    np.testing.assert_array_equal(f[:7], 1.)
    assert np.all(f[7:] > 1)


def test_target_does_not_inflate_gates_below_threshold_or_all_gate_errors():
    r = np.r_[np.full(7, 2.), 10., 12., 15.]
    f, info = select_error_factors(r, min_unchanged_fraction=0., target_chi2=1.75)
    np.testing.assert_array_equal(f[:7], 1.)
    assert info["fixed_model_min_chi2"] > 1.75
    assert np.mean((r/f)**2) > 1.75


@pytest.mark.parametrize("options", [{"min_unchanged_fraction": -.1},
    {"min_unchanged_fraction": 1.1}, {"target_chi2": -1.}, {"target_chi2": np.nan}])
def test_bad_target_or_quota_is_rejected(options):
    with pytest.raises(ValueError):
        select_error_factors(np.ones(10), **options)


def test_target_mode_refits_with_actual_errors_and_reports_raw_chi2():
    calls = []
    r = np.r_[np.full(7, .5), 10., 12., 15.]
    def solve(errors, previous):
        calls.append(errors.copy())
        return r.copy()
    _, effective, info = reweight_errors(
        np.zeros(10), np.ones(10), solve, lambda x: x,
        min_unchanged_fraction=.7, target_chi2=1.75, target_tolerance=.25)
    assert len(calls) == 2 and info["target_reached"]
    assert info["kept"] == 10 and info["unchanged"] == 7
    np.testing.assert_array_equal(calls[-1], effective)
    assert info["chi2_original"] == np.mean(r*r)
    assert info["chi2_effective"] == pytest.approx(1.75)


def lm_data(std):
    return {"moments": {"LM": {"times": np.array([5., 7., 9.])*1e-6,
                               "response": np.ones(3)*1e-8,
                               "relative_std": np.full(3, std)}}}


def test_prior_needs_sustained_quality_loss_and_resets_at_each_line():
    data = [lm_data(.05) for _ in range(20)] + [lm_data(.2) for _ in range(20)]
    data[12] = lm_data(.4)  # isolated contamination should not trigger
    inv = {"shallow_prior_enabled": True, "shallow_prior_window": 5,
           "rel_error": 0.0, "noise_floor": 1e-18}
    scores, report = shallow_prior_scores(data, np.arange(40), np.ones(40), inv)
    assert not scores[:20].any()
    assert np.all(scores[25:] > 0.)
    assert report["active_soundings"] > 0
    # The same low-quality observations on a separate line form their own baseline.
    scores, _ = shallow_prior_scores(data, np.arange(40), np.r_[np.ones(20), np.ones(20)*2], inv)
    assert not scores.any()


def test_no_lm_baseline_does_not_automatically_assert_high_resistivity():
    data = [{"moments": {"HM": {}}} for _ in range(25)]
    scores, _ = shallow_prior_scores(data, np.arange(25), np.ones(25), {"shallow_prior_enabled": True})
    assert not scores.any()


def test_manual_prior_is_limited_to_requested_line_and_distance():
    data = [lm_data(.05)]*12
    inv = {"shallow_prior_enabled": True, "shallow_prior_mode": "manual",
           "shallow_prior_lines": [2], "shallow_prior_distance_min_m": 20.,
           "shallow_prior_distance_max_m": 40.}
    scores, _ = shallow_prior_scores(data, np.arange(12)*10., np.repeat([1, 2], 6), inv)
    np.testing.assert_array_equal(np.flatnonzero(scores), [8, 9, 10])


def test_background_penalty_spans_every_layer_and_is_grid_normalised():
    inv = {"shallow_prior_enabled": True, "shallow_prior_mode": "manual",
           "shallow_prior_depth_m": 10., "shallow_prior_min_resistivity": 1000.,
           "shallow_prior_weight": 2.}
    a, wa = shallow_prior_terms(inv, [5., 10., 20.])
    b, wb = shallow_prior_terms(inv, [2.5, 2.5, 5., 5., 20.])
    assert np.sum(wa**2) == pytest.approx(4.)
    assert np.sum(wb**2) == pytest.approx(4.)
    assert np.sum((wa*np.maximum(a-2., 0))**2) == pytest.approx(np.sum((wb*np.maximum(b-2., 0))**2))
    assert np.all(wa > 0) and np.all(wb > 0)
    assert np.sum((wa*np.maximum(a-4., 0))**2) == 0  # no penalty above lower tendency
    _, disabled = shallow_prior_terms({**inv, "shallow_prior_enabled": False}, [5., 10., 20.])
    assert not disabled.any()


def test_lci_accepts_a_layer_specific_prior_vector_without_adding_pseudo_data():
    a = np.eye(2)
    args = dict(forward=lambda sigma: -np.log10(sigma),
                jacobian=lambda sigma: -a/(np.log(10)*sigma),
                dobs=np.array([2., 2.]), uncertainty=np.ones(2))
    plain = SoundingBlock(**args)
    prior = SoundingBlock(**args, prior_lower=np.array([3., 3.]), prior_weights=np.array([2., 0.]))
    settings = dict(smoothness=0., lateral_smoothness=0., max_iterations=30,
                    target_chi2=0., convergence_metric="objective", verbose=False)
    base = solve_lci([plain], 2, **settings)
    fitted = solve_lci([prior], 2, **settings)
    assert fitted.models[0, 0] > base.models[0, 0]
    assert fitted.models[0, 1] == pytest.approx(base.models[0, 1])
    assert fitted.n_data == 2
    expected = np.mean((-np.log10(1/fitted.models[0]) - plain.dobs)**2)
    assert fitted.chi2 == pytest.approx(expected)


@pytest.mark.parametrize("analytic", [True, False])
def test_independent_occam_prior_matches_its_exact_linear_optimum(analytic):
    from PyHydroGeophysX.inversion.em1d import _occam_1d
    inv = {"n_layers": 2, "layer_thicknesses": [10.], "smoothness": 0.,
           "starting_resistivity": 100., "max_iterations": 30,
           "shallow_prior_enabled": True, "shallow_prior_mode": "manual",
           "shallow_prior_depth_m": 10., "shallow_prior_min_resistivity": 1000.,
           "shallow_prior_weight": 2.}
    jac = (lambda sigma: -np.eye(2)/(np.log(10)*sigma)) if analytic else None
    model, chi2, _, _ = _occam_1d(lambda sigma: -np.log10(sigma), np.array([2., 2.]),
                                 np.ones(2), 2, inv, lambda message: None, jac)
    np.testing.assert_allclose(np.log10(model), [8/3, 8/3], atol=1e-6)
    assert chi2 == pytest.approx(4/9, abs=1e-6)


def test_automatic_background_target_is_twice_the_effective_start_and_bounded():
    inv = {"starting_resistivity": 800., "shallow_prior_resistivity_factor": 2.,
           "shallow_prior_min_resistivity": 0., "rho_max": 10000.}
    reference, target, factor, source = resistive_prior_target(inv)
    assert (reference, target, factor) == pytest.approx((800., 1600., 2.))
    assert source == "starting_model_factor"
    reference, target, factor, source = resistive_prior_target({
        **inv, "_resistive_prior_reference_resistivity": 6000.})
    assert (reference, target, factor) == pytest.approx((6000., 10000., 2.))
    assert source == "starting_model_factor"


def test_reweighting_plateau_uses_the_objective_not_only_data_chi2(monkeypatch):
    import PyHydroGeophysX.inversion.em1d_lci as lci
    original = lci._solve_normal_equations
    monkeypatch.setattr(lci, "_solve_normal_equations", lambda a, b: .1*original(a, b))
    b = SoundingBlock(forward=lambda sigma: -np.log10(sigma),
                      jacobian=lambda sigma: -np.eye(2)/(np.log(10)*sigma),
                      dobs=np.array([1., 3.]), uncertainty=np.ones(2))
    settings = dict(initial_model=np.array([[10., 1000.]]), smoothness=10.,
                    target_chi2=0., max_iterations=8,
                    solver="gauss_newton", verbose=False)
    data_stop = solve_lci([b], 2, **settings)
    objective_stop = solve_lci([b], 2, convergence_metric="objective", **settings)
    assert data_stop.iterations == 2
    assert objective_stop.iterations == 8


@pytest.mark.parametrize("late_signal,late_noise,late_time,activated", [
    (0.2, .05, 7e-6, True),   # weaker signal, unchanged uncertainty
    (1., .25, 7e-6, False),   # increased noise alone is not resistive evidence
    (.2, .15, 7e-6, False),   # veto large uncertainty growth, even if signal drops
    (.2, .05, 9e-6, False),   # different physical reference gates cannot be compared
])
def test_raw_quality_prior_separates_signal_decline_from_noise_growth(
        late_signal, late_noise, late_time, activated):
    data = [lm_data(.05)]*30
    first = {"LM_signal": 1., "LM_noise": .05, "LM_reference_time": 7e-6}
    last = {"LM_signal": late_signal, "LM_noise": late_noise, "LM_reference_time": late_time}
    scores, report = shallow_prior_scores(
        data, np.arange(30), np.ones(30),
        {"shallow_prior_enabled": True, "shallow_prior_window": 5},
        [first]*15 + [last]*15)
    assert bool(scores[-5:].any()) is activated
    assert not scores[:15].any()
    assert report["quality_source"] == "fixed_raw_LM_gate"


def test_raw_diagnostics_survive_qt_input_container(tmp_path, monkeypatch):
    from PyHydroGeophysX.data_processing import em1d as reader
    from PyHydroGeophysX.inversion.em1d_priors import raw_lm_quality_rows
    raw = reader._temcompany_raw_lm_quality(
        {"LM_VoltageValues": "[1e-7, 2e-8, 3e-9]",
         "LM_VoltageValues_STD": "[0.05, 0.1, 0.2]"},
        {"LM_GateCentreTime": [5e-6, 7e-6, 9e-6]})
    source = {**lm_data(.05), "raw_lm_quality": raw, "n_soundings": 1}
    monkeypatch.setattr(reader, "load_sounding", lambda *args, **kwargs: source)
    path = reader.save_sounding_container(tmp_path / "input.npz", "source", "TDEM", moment="LM+HM")
    restored = reader.load_sounding_container(path, "TDEM")
    assert raw_lm_quality_rows([restored]) == raw_lm_quality_rows([source])
    assert raw_lm_quality_rows([restored], 1)[0]["LM_signal"] == pytest.approx(2e-8)
    np.testing.assert_array_equal(restored["moments"]["LM"]["response"], source["moments"]["LM"]["response"])


def test_ground_preset_enables_requested_prior_and_bounded_error_target():
    from PyHydroGeophysX.inversion.em1d import INVERSION_PRESETS
    preset = INVERSION_PRESETS["ground_tem"]
    assert preset["shallow_prior_enabled"] is True
    assert preset["shallow_prior_mode"] == "signal_threshold"
    assert preset["shallow_prior_resistivity_factor"] == 2.
    assert preset["robust_min_unchanged_fraction"] == .70
    assert preset["robust_target_chi2"] == 1.75


def test_absolute_signal_prior_needs_sustained_weak_signal_not_an_snr_drop():
    data = [lm_data(.05)]*30
    inv = {"shallow_prior_enabled": True, "shallow_prior_mode": "signal_threshold",
           "shallow_prior_signal_threshold": 3e-9, "shallow_prior_window": 5}
    rows = [{"LM_signal": 6e-9, "LM_noise": 1e-9, "LM_reference_time": 9e-6}]*15
    rows += [{"LM_signal": 1e-9, "LM_noise": 2e-10, "LM_reference_time": 9e-6}]*15
    score, report = shallow_prior_scores(data, np.arange(30), np.ones(30), inv, rows)
    assert not score[:18].any()
    assert np.all(score[18:] > 0)
    assert report["signal_to_threshold"][-1] == pytest.approx(1/3)
    # A low-signal line need not begin with a strong baseline.
    score, _ = shallow_prior_scores(data, np.arange(30), np.ones(30), inv, rows[15:]*2)
    assert score[4] > 0
    # Strong SNR decline alone does not trigger the absolute-signal option.
    rows = [{"LM_signal": 6e-9, "LM_noise": 1e-7, "LM_reference_time": 9e-6}]*30
    score, _ = shallow_prior_scores(data, np.arange(30), np.ones(30), inv, rows)
    assert not score.any()


def test_absolute_signal_prior_missing_raw_input_cannot_become_zero_signal():
    inv = {"shallow_prior_enabled": True, "shallow_prior_mode": "signal_threshold",
           "shallow_prior_signal_threshold": 3e-9}
    score, _ = shallow_prior_scores([lm_data(.05)]*30, np.arange(30), np.ones(30), inv)
    assert not score.any()


def test_signal_calibration_uses_raw_units_and_actual_instrument(monkeypatch):
    from PyHydroGeophysX.inversion import em1d
    from PyHydroGeophysX.inversion.em1d_priors import shallow_signal_thresholds
    recorded = []
    def blocks(data, geometry, inv, thickness):
        recorded.append((data, geometry, inv, thickness))
        return data
    monkeypatch.setattr(em1d, "tdem_moment_blocks", blocks)
    monkeypatch.setattr(em1d, "_moment_forward", lambda blocks: lambda sigma: sigma[:1]*3e-6)
    data = {"raw_lm_quality": {"times": [5e-6, 7e-6, 9e-6],
                               "transmitter": {"waveform_times": [0., 1e-6]}}}
    geom = {"tx_rx_sep": 14.75}
    inv = {"shallow_prior_min_resistivity": 1000., "data_scale": 123.}
    assert shallow_signal_thresholds([data], [geom], inv)[0] == pytest.approx(3e-9)
    assert recorded[0][1] == geom
    assert recorded[0][2]["data_scale"] == 1.
    assert recorded[0][0]["moments"]["LM"]["times"][0] == 9e-6
    manual = shallow_signal_thresholds([data], [geom], {**inv, "shallow_prior_signal_threshold": 4e-9})
    assert manual[0] == 4e-9 and len(recorded) == 1
