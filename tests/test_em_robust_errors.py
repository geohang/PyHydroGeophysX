"""Robust errors retain data, bound influence and keep raw fit metrics honest."""

from pathlib import Path
from types import SimpleNamespace
import csv
import json

import numpy as np
import pytest

from PyHydroGeophysX.inversion.robust_errors import huber_error_factor, reweight_errors
from PyHydroGeophysX.inversion.em1d import _occam_with_optional_rejection
from PyHydroGeophysX.inversion.em1d_lci import (
    SoundingBlock, invert_lci, invert_lci_with_robust_errors,
)


def test_huber_formula_sign_symmetry_cap_and_no_zero_weights():
    residual = np.array([0., 2., 3., 12., -12., 300., 1e6])
    factors = huber_error_factor(residual, 3, 10)
    np.testing.assert_allclose(factors, [1, 1, 1, 2, 2, 10, 10])
    assert np.min(1 / factors**2) == pytest.approx(.01)


@pytest.mark.parametrize("options", [
    {"threshold": 0}, {"threshold": float("nan")}, {"max_error_factor": .9},
    {"max_error_factor": float("inf")}, {"passes": 0}, {"passes": 1.5},
    {"passes": True}, {"passes": None}, {"passes": float("inf")},
])
def test_invalid_settings_fail_before_solving(options):
    def never(*args):
        pytest.fail("invalid settings must not start a costly fit")
    with pytest.raises(ValueError):
        reweight_errors([1.], [1.], never, never, **options)


def test_errors_can_recover_and_do_not_compound_or_mutate_the_input():
    observed, base = np.zeros(2), np.array([1., 2.])
    calls = []
    # First one residual is 12 sigma, then it becomes good, then stays good.
    predictions = [np.array([12., 0.]), np.array([0., 0.]), np.array([0., 0.])]
    def solve(errors, previous):
        calls.append(errors.copy())
        assert (previous is None) == (len(calls) == 1)
        errors[:] = 999  # even a misbehaving solver cannot overwrite the baseline
        return predictions[len(calls) - 1]
    _, errors, info = reweight_errors(observed, base, solve, lambda result: result)
    np.testing.assert_array_equal(base, [1, 2])
    np.testing.assert_array_equal(observed, [0, 0])
    np.testing.assert_allclose(calls, [[1, 2], [2, 2], [1, 2]])
    np.testing.assert_array_equal(errors, base)
    assert info["kept"] == 2 and info["dropped"] == 0
    assert info["downweighted"] == 0


@pytest.mark.parametrize("count", [1, 2, 3, 4, 17])
def test_even_sparse_soundings_keep_every_gate(count):
    calls = []
    def solve(errors, previous):
        calls.append(errors.copy())
        return np.full(count, 12.)
    _, errors, info = reweight_errors(np.zeros(count), np.ones(count), solve, lambda x: x)
    # Persistent large residuals do not repeatedly multiply the error by two.
    assert len(calls) == 2
    np.testing.assert_array_equal(errors, np.full(count, 2.))
    assert info["n_start"] == info["kept"] == count
    assert info["dropped"] == 0
    assert info["chi2_original"] == 144.
    assert info["chi2_effective"] == 36.


def block(outlier=True, position=0.):
    # Linear in log10(rho), nonlinear in the conductivity the callback receives.
    # Nine observations of one parameter, one badly contaminated observation.
    A = np.ones((10, 2)) / 2
    observed = np.full(10, 2.)
    if outlier:
        observed[-1] += 1.
    return SoundingBlock(
        forward=lambda sigma: A @ -np.log10(sigma),
        jacobian=lambda sigma: -A / (np.log(10) * sigma[None, :]),
        dobs=observed, uncertainty=np.full(10, .02), position=position, line=1)


def test_occam_robust_fit_is_less_biased_and_overrides_hard_rejection():
    b = block()
    inv = {"smoothness": 1., "max_iterations": 30, "starting_resistivity": 90.}
    args = (b.forward, b.dobs, b.uncertainty, 2)
    plain = _occam_with_optional_rejection(*args, inv, lambda m: None, b.jacobian)
    fitted = _occam_with_optional_rejection(
        *args, {**inv, "robust_errors": True, "reject_outliers": True}, lambda m: None, b.jacobian)
    info = fitted[-1]["robust"]
    assert abs(np.log10(fitted[0]).mean() - 2) < abs(np.log10(plain[0]).mean() - 2) / 3
    assert fitted[-2].all() and fitted[-2].size == 10
    assert not fitted[-1]["enabled"] and info["dropped"] == 0
    r = (b.forward(1/fitted[0]) - b.dobs) / b.uncertainty
    assert fitted[1] == pytest.approx(np.mean(r*r))
    assert info["chi2_effective"] == pytest.approx(np.mean((r/np.array(info["error_factor"]))**2))
    assert fitted[1] >= plain[1]  # changing weights must not be advertised as lower raw chi2


def test_clean_data_and_factor_one_preserve_the_original_solver():
    b = block(outlier=False)
    inv = {"smoothness": 1., "max_iterations": 20}
    args = (b.forward, b.dobs, b.uncertainty, 2)
    plain = _occam_with_optional_rejection(*args, inv, lambda m: None, b.jacobian)
    robust = _occam_with_optional_rejection(
        *args, {**inv, "robust_errors": True}, lambda m: None, b.jacobian)
    np.testing.assert_array_equal(plain[0], robust[0])
    assert robust[-1]["robust"]["passes"] == []
    b = block()
    plain = _occam_with_optional_rejection(b.forward, b.dobs, b.uncertainty, 2, inv, lambda m: None)
    capped = _occam_with_optional_rejection(
        b.forward, b.dobs, b.uncertainty, 2,
        {**inv, "robust_errors": True, "robust_max_error_factor": 1.}, lambda m: None)
    np.testing.assert_array_equal(plain[0], capped[0])


def test_lci_retains_operators_and_gates_with_parallel_equivalence():
    blocks = [block(position=0), block(position=5)]
    settings = dict(smoothness=1., lateral_smoothness=1., max_iterations=25,
                    auto_lambda=False, target_chi2=0., chi2_tolerance=0., verbose=False)
    plain = invert_lci(blocks, 2, parallel_workers=1, **settings)
    serial, effective, info = invert_lci_with_robust_errors(
        blocks, 2, parallel_workers=1, **settings)
    parallel, _, parallel_info = invert_lci_with_robust_errors(
        blocks, 2, parallel_workers=2, **settings)
    np.testing.assert_array_equal(serial.models, parallel.models)
    assert info["weights"] == parallel_info["weights"]
    assert abs(np.log10(serial.models).mean() - 2) < abs(np.log10(plain.models).mean() - 2) / 3
    assert serial.n_data == plain.n_data == info["kept"] == 20
    assert info["block_offsets"] == [0, 10, 20]
    assert serial.chi2 == pytest.approx(info["chi2_effective"])
    for old, new in zip(blocks, effective):
        assert old.forward is new.forward and old.jacobian is new.jacobian
        np.testing.assert_array_equal(old.dobs, new.dobs)
        np.testing.assert_array_equal(old.uncertainty, np.full(10, .02))
        assert new.uncertainty[-1] > old.uncertainty[-1]


def test_reweighting_freezes_lambda_and_warm_starts(monkeypatch):
    import PyHydroGeophysX.inversion.em1d_lci as lci
    calls = []
    def fake(blocks, n_layers, **kwargs):
        calls.append(kwargs.copy())
        return SimpleNamespace(models=np.full((len(blocks), n_layers), 100.),
                               smoothness_scale=1.7, lambda_search={"status": "initial"},
                               chi2_history=[1.], seconds=1., iterations=2)
    monkeypatch.setattr(lci, "invert_lci", fake)
    _, _, info = lci.invert_lci_with_robust_errors(
        [block()], 2, smoothness=2., lateral_smoothness=1.3, auto_lambda=True)
    assert len(calls) == 2
    assert calls[0]["auto_lambda"] and not calls[1]["auto_lambda"]
    assert calls[1]["smoothness_scale"] == 1.7
    np.testing.assert_array_equal(calls[1]["initial_model"], [[100, 100]])
    assert calls[0]["smoothness"] == calls[1]["smoothness"] == 2.
    assert calls[0]["lateral_smoothness"] == calls[1]["lateral_smoothness"] == 1.3
    assert info["total_iterations"] == 4


PROJECT = Path(__file__).resolve().parents[2] / "TEM2go_data" / "trailcreek"


@pytest.mark.skipif(not (PROJECT / "project.db").exists(), reason="optional real TEM project")
@pytest.mark.parametrize("mode,count", [("simultaneous", 4), ("sequential", 3), ("off", 3),
                                        ("simultaneous", 1)])
def test_real_joint_line_keeps_all_gates_and_exports_auditable_errors(tmp_path, mode, count):
    from PyHydroGeophysX.workflows import em1d as workflow
    head = workflow.load_sounding(str(PROJECT), "TDEM", sounding=0, moment="LM+HM",
                                  max_relative_std=None, reject_negative=False)
    geom = {**head["system"], "tem_moment": "LM+HM", "tail_max_relative_std": None}
    inv = {**workflow.preset_inversion("ground_tem"), "n_layers": 6,
           "max_iterations": 3, "parallel_workers": 2, "lci_mode": mode,
           "robust_passes": 2}
    result = workflow.invert_line(str(PROJECT), "TDEM", geom, inv, max_soundings=count,
                                  lines=[3], doi_blank=False, out_dir=tmp_path)
    info = result["robust"]
    assert info["enabled"] and info["kept"] > 0 and info["dropped"] == 0
    assert not result["outliers"]["enabled"]
    if mode == "off" or count == 1:
        assert not result["lci"]
    # Reload exactly the same stations and independently assemble their original errors.
    from PyHydroGeophysX.inversion.em1d import build_sounding_block
    start, _ = workflow._line_block(head, [3])
    errors = []
    for i in range(count):
        data = workflow.load_sounding(str(PROJECT), "TDEM", sounding=start+i, moment="LM+HM",
                                      max_relative_std=None, reject_negative=False)
        b = build_sounding_block(data, workflow._station_geometry(geom, data), inv)
        errors.extend(b.uncertainty)
    assert info["kept"] == len(errors) == sum(result["data_count_list"])
    with (tmp_path / "robust_gate_errors.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    np.testing.assert_allclose([float(r["error_original"]) for r in rows], errors)
    assert len(rows) == len(errors)
    assert {int(r["line"]) for r in rows} == {3}
    assert {r["station"] for r in rows} == set(map(str, result["station_ids"]))
    r = np.array([float(row["residual_original"]) for row in rows])
    f = np.array([float(row["error_factor"]) for row in rows])
    assert result["chi2"] == pytest.approx(np.mean(r*r))
    assert result["chi2_effective"] == pytest.approx(np.mean((r/f)**2))
    per_item = np.asarray(result["chi2_effective_list"], float)
    counts = np.asarray(result["data_count_list"])
    assert np.average(per_item[counts > 0], weights=counts[counts > 0]) == pytest.approx(result["chi2_effective"])
    assert np.all((f >= 1) & (f <= 10))
    assert np.isfinite(result["sensitivity"]).all()
    stored = json.loads((tmp_path / "robust_errors.json").read_text())
    assert stored["kept"] == info["kept"]


@pytest.mark.skipif(not (PROJECT / "project.db").exists(), reason="optional real TEM project")
@pytest.mark.parametrize("moment", ["LM", "HM", "LM+HM"])
def test_persisted_single_sounding_saves_weights_and_respects_import_qc(tmp_path, moment):
    from PyHydroGeophysX.workflows.domain import run_em_inversion
    from PyHydroGeophysX.workflows.models import ArtifactRef, RunContext, WorkflowSpec
    from PyHydroGeophysX.workflows import em1d as workflow
    geom = {"use_project_flags": True, "tail_max_relative_std": None,
            "reject_negative": False, "gate_rejection": "individual", "tem_moment": moment}
    inv = {**workflow.preset_inversion("ground_tem"), "n_layers": 6,
           "max_iterations": 4, "robust_passes": 1}
    spec = WorkflowSpec(workflow_id="em.inversion", inputs={
        "data": ArtifactRef.from_path(PROJECT, artifact_id="project", kind="em_sounding")},
        parameters={**inv, "method": "TDEM", "moment": moment, "sounding": 120, "geometry": geom})
    result = run_em_inversion(spec, RunContext(project_root=tmp_path, output_dir=tmp_path))
    result.to_dict()  # all audit information is serializable across the worker boundary
    fitted = result.legacy_payload()
    assert fitted["robust"]["enabled"] and fitted["fit_mask"].all()
    with (tmp_path / "robust_gate_errors.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == fitted["n_data"]
    assert len(result.artifacts) == 5
    assert fitted["chi2_effective"] == fitted["robust"]["chi2_effective"]
    loaded = workflow.load_sounding(str(PROJECT), "TDEM", sounding=120, moment=moment,
                                    max_relative_std=None, reject_negative=False)
    if moment == "LM+HM":
        assert sum(len(m["times"]) for m in loaded["moments"].values()) == len(rows)
        assert {r["moment"] for r in rows} == set(loaded["moments"])
    else:
        assert len(loaded["times"]) == len(rows)
