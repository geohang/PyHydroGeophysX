"""The inversion presets, the survey summary and the per-gate report.

These three exist to put on screen what the pipeline is about to do, so what
they say has to be what the pipeline does. The tests below check exactly that
correspondence: a preset lands on the framework's own defaults, the gate report
names the same gates the loader returns, and the verdicts it attaches are the
verdicts the selection acted on.
"""

from __future__ import annotations

import inspect
import itertools
import re
import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from PyHydroGeophysX.inversion import em1d as inversion_em1d
from PyHydroGeophysX.workflows import em1d as workflows_em1d
from PyHydroGeophysX.data_processing.em1d import (
    GATE_STATUS,
    _gate_disposition,
    _temcompany_json_array,
    _temcompany_valid_channels,
    gate_report,
    load_sounding,
    survey_summary,
)
from PyHydroGeophysX.inversion.em1d import (
    DEFAULT_INVERSION,
    INVERSION_PRESETS,
    preset_inversion,
)

#: A ground TDEM project with 929 stations across three lines. Skipped where the
#: data is not checked out, since it is far too large to live in the repository.
PROJECT = (Path(__file__).resolve().parents[2] / "TEM2go_data" / "trailcreek")

needs_project = pytest.mark.skipif(
    not (PROJECT / "project.db").exists(),
    reason="the TEM2Go trailcreek project is not present")


# -- presets -----------------------------------------------------------------
def test_every_preset_only_names_settings_something_reads():
    """A preset key nothing reads would silently do nothing.

    The panel writes a preset over its controls by key and the runner reads the
    settings by key, so a typo is invisible at both ends: the control is never
    touched and the run quietly uses its own fallback. Catching it here is the
    only place it shows.

    ``DEFAULT_INVERSION`` is not the whole vocabulary. It carries what a single
    sounding needs, and :func:`invert_line` reads six more (outlier rejection
    and the starting model) with their fallbacks written at the point of use.
    So the test is that a preset key is read somewhere, not that it is declared
    in one particular dictionary.

    The read sites are matched rather than the bare name. Scanning for the name
    anywhere in the source would find the preset's own definition and pass on
    any typo, which is the failure this is here to catch.
    """
    source = "".join(
        inspect.getsource(module)
        for module in (inversion_em1d, workflows_em1d))
    for name, preset in INVERSION_PRESETS.items():
        for key in preset:
            read = (key in DEFAULT_INVERSION
                    or re.search(r"""\.get\(\s*["']%s["']|\[["']%s["']\]"""
                                 % (re.escape(key), re.escape(key)), source))
            assert read, f"preset {name!r} sets {key!r}, which nothing reads"


def test_a_preset_is_the_defaults_with_its_own_settings_on_top():
    generic = preset_inversion("generic")
    assert generic == DEFAULT_INVERSION
    ground = preset_inversion("ground_tem")
    for key, value in INVERSION_PRESETS["ground_tem"].items():
        assert ground[key] == value
    for key in set(DEFAULT_INVERSION) - set(INVERSION_PRESETS["ground_tem"]):
        assert ground[key] == DEFAULT_INVERSION[key]


def test_a_preset_does_not_hand_back_its_own_dictionary():
    """A caller that edits what it got must not move the preset for everyone."""
    first = preset_inversion("ground_tem")
    first["n_layers"] = 999
    assert preset_inversion("ground_tem")["n_layers"] != 999


def test_an_unknown_preset_says_which_ones_exist():
    with pytest.raises(ValueError, match="ground_tem"):
        preset_inversion("no_such_survey")


def test_the_doi_threshold_default_matches_the_one_the_run_uses():
    """``inversion.em1d`` repeats the constant rather than importing it.

    The import is deliberately avoided: ``em1d_lci`` pulls in SciPy sparse, and
    the module holding the defaults is imported by the CLI and by the Qt panel,
    neither of which should pay for that. The cost of repeating it is that the
    two can drift, so they are compared here.
    """
    from PyHydroGeophysX.inversion.em1d_lci import DOI_SENSITIVITY_THRESHOLD

    assert DEFAULT_INVERSION["doi_threshold"] == DOI_SENSITIVITY_THRESHOLD


def test_formal_line_solver_defaults_are_the_validated_trf_settings():
    assert DEFAULT_INVERSION["lci_solver"] == "trf"
    assert DEFAULT_INVERSION["lci_max_nfev"] == 90
    assert DEFAULT_INVERSION["lci_ftol"] == pytest.approx(1e-4)
    assert DEFAULT_INVERSION["lci_xtol"] == pytest.approx(1e-6)
    assert DEFAULT_INVERSION["lci_gtol"] == pytest.approx(1e-5)


# -- the gate verdicts -------------------------------------------------------
def test_a_verdict_is_one_of_the_named_ones():
    times = np.array([1e-5, 2e-5, 3e-5, 4e-5, np.nan])
    values = np.array([1e-7, -2e-8, 3e-9, 4e-10, 1e-10])
    std = np.array([0.02, 0.05, 0.50, 0.04, 0.01])
    flags = np.array([1, 1, 1, 0, 1])
    status, _, _ = _gate_disposition(times, values, std, flags, True, 0.30,
                                     "individual", True)
    assert set(status) <= set(GATE_STATUS)
    assert list(status) == ["kept", "reversed sign", "noisy", "flagged out",
                            "dummy"]


def test_truncation_carries_the_gates_after_a_noisy_one():
    times = np.array([1e-5, 2e-5, 3e-5, 4e-5])
    values = np.array([1e-7, 2e-8, 3e-9, 4e-10])
    std = np.array([0.02, 0.50, 0.03, 0.02])
    flags = np.ones(4)
    carried, _, _ = _gate_disposition(times, values, std, flags, True, 0.30,
                                      "truncate", False)
    assert list(carried) == ["kept", "noisy", "after a noisy one",
                             "after a noisy one"]
    alone, _, _ = _gate_disposition(times, values, std, flags, True, 0.30,
                                    "individual", False)
    assert list(alone) == ["kept", "noisy", "kept", "kept"]


def test_a_gate_that_is_both_reversed_and_noisy_reads_as_reversed():
    """Of the two, the sign is the more specific thing to tell a reader."""
    status, _, _ = _gate_disposition(
        np.array([1e-5]), np.array([-1e-9]), np.array([0.9]), np.array([1]),
        True, 0.30, "individual", True)
    assert list(status) == ["reversed sign"]


def _selection_before_the_verdicts_existed(times, response, std, flags,
                                           use_flags, cut, mode, reject_negative):
    """The gate selection exactly as it stood before it grew per-gate reasons.

    Frozen here on purpose. The verdicts and the selection now share one
    implementation, so asserting that they agree asserts nothing. What is worth
    holding is that adding the reasons did not move which gates a run sees, and
    that needs a reference the refactor cannot have changed.
    """
    n = min(np.size(times), np.size(response))
    t = np.asarray(times, dtype=float).ravel()[:n]
    d = np.asarray(response, dtype=float).ravel()[:n]
    mask = np.isfinite(t) & (t > 0.0) & np.isfinite(d) & (np.abs(d) < 9_000.0)
    if use_flags and flags is not None and np.size(flags):
        use = np.asarray(flags, dtype=float).ravel()
        mask &= np.pad(use[:n] > 0, (0, max(0, n - use.size)),
                       constant_values=False)[:n]
    kept_std = None
    if std is not None and np.size(std):
        raw = np.asarray(std, dtype=float).ravel()
        raw = np.pad(raw[:n], (0, max(0, n - raw.size)),
                     constant_values=np.nan)[:n]
        kept_std = raw[mask]
        kept_std[~np.isfinite(kept_std) | (kept_std < 0.0)
                 | (kept_std >= 9_000.0)] = np.nan
    if cut is not None and np.any(mask):
        kept = np.flatnonzero(mask)
        errors = kept_std if kept_std is not None else np.full(kept.size, np.nan)
        noisy = np.isfinite(errors) & (errors > float(cut))
        reversed_sign = ((d[kept] < 0.0) if reject_negative
                         else np.zeros(kept.size, dtype=bool))
        if mode == "truncate" and noisy.any():
            noisy[int(np.argmax(noisy)):] = True
        spoiled = noisy | reversed_sign
        if spoiled.any():
            mask[kept[spoiled]] = False
            if kept_std is not None:
                kept_std = kept_std[~spoiled]
    return mask, kept_std


def test_adding_the_verdicts_did_not_move_which_gates_a_run_sees():
    """Every combination of the four switches, against the frozen rule.

    The same comparison over one 929-station project agreed on all 44,592
    station-moment-setting combinations it holds; the random cases here cover
    the corners that survey has no example of, such as a noisy gate the flags
    left in.
    """
    rng = np.random.default_rng(20260829)
    times = np.logspace(-5, -3.5, 20)
    compared = 0
    for _ in range(50):
        values = rng.normal(0.0, 1.0, 20) * np.logspace(-7, -11, 20)
        std = np.abs(rng.normal(0.15, 0.2, 20))
        flags = rng.integers(0, 2, 20)
        for use_flags, cut, mode, negative in itertools.product(
            (True, False), (None, 0.3, 0.1), ("truncate", "individual"),
            (True, False),
        ):
            reference, reference_std = _selection_before_the_verdicts_existed(
                times, values, std, flags, use_flags, cut, mode, negative)
            status, _, _ = _gate_disposition(times, values, std, flags,
                                             use_flags, cut, mode, negative)
            mask = status == "kept"
            compared += 1
            assert np.array_equal(mask, reference)
            if not mask.any():
                with pytest.raises(ValueError):
                    _temcompany_valid_channels(times, values, std, flags,
                                               use_flags, cut, mode, negative)
                continue
            kept_t, kept_d, kept_std = _temcompany_valid_channels(
                times, values, std, flags, use_flags, cut, mode, negative)
            assert np.allclose(kept_t, times[reference])
            assert np.allclose(kept_d, values[reference])
            assert np.allclose(kept_std, reference_std, equal_nan=True)
    assert compared == 50 * 2 * 3 * 2 * 2


# -- the report against the loader -------------------------------------------
@needs_project
def test_a_station_number_means_the_same_station_in_both_readers():
    for index in (0, 7, 400, 881):
        loaded = load_sounding(str(PROJECT), "TDEM", sounding=index,
                               moment="LM+HM", max_relative_std=None,
                               reject_negative=False)
        report = gate_report(str(PROJECT), index)
        assert report["n_soundings"] == loaded["n_soundings"]
        assert report["station"] == str(loaded["station_ids"][loaded["sounding"]])


@needs_project
def test_the_report_holds_every_gate_and_marks_the_ones_the_loader_returns():
    loaded = load_sounding(str(PROJECT), "TDEM", sounding=31, moment="LM+HM",
                           max_relative_std=None, reject_negative=False)
    report = gate_report(str(PROJECT), 31)
    for name, moment in report["moments"].items():
        kept = moment["status"] == "kept"
        assert moment["held"] == moment["status"].size
        assert moment["kept"] == int(kept.sum())
        # Every gate the file holds is present, not only the survivors.
        assert moment["held"] > moment["kept"]
        assert np.allclose(loaded["moments"][name]["times"],
                           moment["times"][kept])
        assert np.allclose(loaded["moments"][name]["response"],
                           moment["values"][kept])
        # A gate window brackets its own centre.
        good = np.isfinite(moment["open"]) & np.isfinite(moment["close"])
        assert np.all(moment["open"][good] < moment["times"][good])
        assert np.all(moment["times"][good] < moment["close"][good])


@needs_project
def test_a_stricter_cut_keeps_fewer_gates_and_never_more():
    loose = gate_report(str(PROJECT), 12, max_relative_std=None)
    tight = gate_report(str(PROJECT), 12, max_relative_std=0.10,
                        reject_negative=True)
    for name in loose["moments"]:
        assert (tight["moments"][name]["kept"]
                <= loose["moments"][name]["kept"])
        assert (tight["moments"][name]["held"]
                == loose["moments"][name]["held"])


@needs_project
def test_the_survey_totals_add_up_to_the_per_station_rows():
    summary = survey_summary(str(PROJECT))
    rows, totals = summary["rows"], summary["totals"]
    assert totals["stations"] == len(rows)
    assert totals["gates_kept"] == sum(int(r["gates_kept"]) for r in rows)
    assert totals["stations_with_data"] == sum(
        1 for r in rows if r["gates_kept"])
    assert (totals["stations_emptied"]
            == totals["stations"] - totals["stations_with_data"])


@needs_project
def test_the_survey_summary_counts_what_the_loader_would_load():
    """The table's station count is the number an inversion would run."""
    summary = survey_summary(str(PROJECT), max_relative_std=None,
                             reject_negative=False)
    loaded = load_sounding(str(PROJECT), "TDEM", sounding=0, moment="LM+HM",
                           max_relative_std=None, reject_negative=False)
    assert summary["totals"]["stations_with_data"] == loaded["n_soundings"]


@needs_project
def test_the_report_reads_the_gates_the_database_actually_stores():
    """One station checked against the raw JSON, so the decode is not assumed."""
    report = gate_report(str(PROJECT), 0)
    con = sqlite3.connect(str(PROJECT / "project.db"))
    con.row_factory = sqlite3.Row
    try:
        specs = {r["RxTxSpecsId"]: json.loads(r["RxTxSpecsJson"])
                 for r in con.execute("SELECT * FROM RxTxSpecs")
                 if r["RxTxSpecsJson"]}
        row = next(r for r in con.execute(
            "SELECT * FROM StationStackData ORDER BY LineNumber, AveragedDataId")
            if str(r["StationId"]) == report["station"])
        spec = specs.get(row["RxTxSpecsId"], next(iter(specs.values())))
        for name, moment in report["moments"].items():
            stored = _temcompany_json_array(row[f"{name}_VoltageValues"])
            assert np.allclose(moment["values"], stored[:moment["held"]])
            assert np.allclose(
                moment["times"],
                np.asarray(spec[f"{name}_GateCentreTime"],
                           dtype=float)[:moment["held"]])
    finally:
        con.close()


# -- the two lateral knobs ---------------------------------------------------
@needs_project
def test_the_two_lateral_knobs_are_one_knob():
    """``lateral_smoothness`` and ``lateral_weight_scale`` multiply.

    They meet as a product before the solver sees either, and the product is
    squared into the penalty weight, so 1.3 x 2.0 and 2.6 x 1.0 must give the
    same section. Both settings stay in the framework because a reader who meets
    one of them in a config file should find the other documented; the
    convention is that ``lateral_smoothness`` is the one to move, and this holds
    the property that convention rests on.
    """
    from PyHydroGeophysX.workflows import em1d as workflows_em1d

    head = workflows_em1d.load_sounding(
        str(PROJECT), "TDEM", sounding=0, moment="LM+HM",
        max_relative_std=None, reject_negative=False)
    geom = {**head["system"], "tem_moment": "LM+HM"}
    base = {**workflows_em1d.DEFAULT_INVERSION, "n_layers": 6,
            "max_iterations": 2, "auto_lambda": False, "data_scale": 1.0}
    positions = np.asarray(head["positions"], dtype=float)

    def section(smoothness, scale):
        result = workflows_em1d.invert_line(
            str(PROJECT), "TDEM", geom,
            {**base, "lateral_smoothness": smoothness,
             "lateral_weight_scale": scale},
            positions=positions, max_soundings=4, lines=[2], doi_blank=False)
        return np.asarray(result["model3d"][:, 0, :], dtype=float)

    doubled_smoothness = section(2.6, 1.0)
    doubled_scale = section(1.3, 2.0)
    assert np.array_equal(doubled_smoothness, doubled_scale, equal_nan=True), (
        "the two knobs stopped multiplying; the panel tooltip and the "
        "DEFAULT_INVERSION comment both promise that they do")
    # And that the product matters at all, so the test above is not comparing
    # two runs that ignore both settings.
    assert not np.allclose(doubled_smoothness, section(1.3, 1.0),
                           rtol=1e-3, atol=0)
