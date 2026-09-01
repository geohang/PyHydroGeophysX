"""The EM panel's data views: presets on the controls, and the new tabs.

The panel is where a preset becomes a run and where the reader's verdicts become
a picture, so both directions are checked here: a preset reaches every control
it names and comes back out of :meth:`_collect_inv` unchanged, and the views
survive being handed a dataset, an empty one, and nothing at all.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from PySide6.QtWidgets import QApplication  # noqa: E402

from PyHydroGeophysX.inversion.em1d import (  # noqa: E402
    INVERSION_PRESETS,
    preset_inversion,
)
from PyHydroGeophysX.qt_apps.modules.em_processing import (  # noqa: E402
    EMProcessingModule,
    _modelled_gates,
)
from PyHydroGeophysX.qt_apps.widgets.em_gate_view import EMGateView  # noqa: E402
from PyHydroGeophysX.qt_apps.widgets.em_survey_view import (  # noqa: E402
    EMMetadataView,
    EMSurveyView,
)

PROJECT = Path(__file__).resolve().parents[2] / "TEM2go_data" / "trailcreek"

needs_project = pytest.mark.skipif(
    not (PROJECT / "project.db").exists(),
    reason="the TEM2Go trailcreek project is not present")

#: Preset settings the panel deliberately does not expose. They drive the
#: legacy hard-rejection path, which robust error weighting replaced; they stay
#: in the framework so a saved configuration can reproduce an older run, and the
#: panel states the path is off rather than offering it. A key added here should
#: come with the reason it is unreachable.
UNEXPOSED = {"reject_outliers", "outlier_threshold", "outlier_passes",
             "min_data_fraction", "min_gates_per_sounding"}


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def panel(app):
    module = EMProcessingModule(None, lambda *a, **k: None)
    yield module
    module.close()


# -- presets on the controls -------------------------------------------------
def test_the_panel_opens_on_the_ground_preset(panel):
    """A ground TDEM survey is what this panel is used for, so it is the default.

    Selecting the combo entry is not enough: adding the first item to an empty
    combo box sets its index without emitting ``currentIndexChanged``, so the
    settings have to be checked rather than the label. A panel that showed
    "ground_tem" and ran the framework defaults is the failure here.
    """
    assert panel._preset.currentData() == "ground_tem"
    opened = panel._collect_inv()
    for key, value in INVERSION_PRESETS["ground_tem"].items():
        if key in UNEXPOSED:
            continue
        assert opened[key] == value, f"the panel opened without {key}"


def test_a_preset_reaches_every_control_it_names(panel):
    """Setting a preset must move the controls, not just the combo box.

    A preset that named a setting the panel had a control for, and did not move
    it, would look applied and run with the framework default. That is the
    failure this catches. Settings the panel does not expose at all are a
    different case and are listed in ``UNEXPOSED`` with their reason.
    """
    for name in INVERSION_PRESETS:
        panel._apply_preset(name)
        wanted = preset_inversion(name)
        got = panel._collect_inv()
        for key in set(INVERSION_PRESETS[name]) - UNEXPOSED:
            assert key in got, f"{name}: the panel does not carry {key!r}"
            expected = wanted[key]
            if isinstance(expected, float):
                expected = pytest.approx(expected)
            assert got[key] == expected, f"{name}: {key} did not reach the panel"


def test_the_panel_keeps_the_legacy_rejection_path_switched_off(panel):
    """It is stated rather than omitted, so nothing else can decide it.

    A key the panel leaves out takes whatever fallback the runner carries, and a
    configuration loaded from elsewhere would then choose it with nothing on
    screen to show that it had.
    """
    for name in INVERSION_PRESETS:
        panel._apply_preset(name)
        assert panel._collect_inv()["reject_outliers"] is False
    assert not hasattr(panel, "_reject"), (
        "the legacy rejection control is back on the panel; if that is "
        "deliberate, UNEXPOSED and this test need to say so")


def test_switching_presets_goes_both_ways(panel):
    """A preset that could not be left would be a trap rather than a default."""
    panel._apply_preset("ground_tem")
    ground = panel._collect_inv()
    panel._apply_preset("generic")
    generic = panel._collect_inv()
    assert generic["n_layers"] != ground["n_layers"]
    panel._apply_preset("ground_tem")
    assert panel._collect_inv()["n_layers"] == ground["n_layers"]


def test_panel_defaults_to_formal_trf_and_keeps_legacy_selectable(panel):
    inv = panel._collect_inv()
    assert inv["lci_solver"] == "trf"
    assert inv["lci_max_nfev"] == 90
    assert inv["lci_ftol"] == pytest.approx(1e-4)
    panel._set_lci_solver("gauss_newton")
    assert panel._collect_inv()["lci_solver"] == "gauss_newton"
    assert not panel._trf_nfev.isEnabled()
    panel._set_lci_solver("trf")
    assert panel._trf_nfev.isEnabled()


def test_empirical_background_prior_is_collapsed_and_uses_start_factor(panel):
    panel._apply_preset("ground_tem")
    inv = panel._collect_inv()
    assert not panel._prior_advanced.isChecked()
    assert inv["shallow_prior_enabled"] is True
    assert inv["shallow_prior_depth_m"] == 0.
    assert inv["shallow_prior_min_resistivity"] == 0.
    assert inv["shallow_prior_resistivity_factor"] == pytest.approx(2.)


def test_robust_controls_preserve_imported_gates_by_default(panel):
    inv = panel._collect_inv()
    assert inv["robust_errors"] and not inv["reject_outliers"]
    assert inv["robust_threshold"] == 3.
    assert inv["robust_passes"] == 3
    assert inv["robust_max_error_factor"] == 10.
    assert panel._gate_qc()["max_relative_std"] is None
    assert panel._gate_qc()["use_flags"]  # not a request to restore flagged/dummy data
    assert not panel._gate_qc()["reject_negative"]


def test_robust_weighting_is_the_only_path_the_panel_offers(panel):
    """Its settings must reach the run in every coupling mode.

    Robust weighting inflates the uncertainty of a gate the model cannot
    explain rather than deleting it, which is what a station carrying three or
    four gates needs: deletion spends data it does not have.
    """
    panel._robust.setChecked(True)
    assert panel._collect_inv()["reject_outliers"] is False
    panel._robust_sigma.setValue(4.)
    panel._robust_passes.setValue(2)
    panel._robust_max_factor.setValue(5.)
    for mode in ("off", "sequential", "simultaneous"):
        panel._set_lci_mode(mode)
        inv = panel._collect_inv()
        assert inv["robust_errors"] and panel._robust.isEnabled()
        assert (inv["robust_threshold"], inv["robust_passes"], inv["robust_max_error_factor"]) == (4., 2, 5.)


@needs_project
def test_loaded_project_retains_robust_defaults_and_every_gate(panel):
    from PyHydroGeophysX.workflows import em1d as workflow
    panel._source_path = PROJECT
    panel._method.setCurrentText("TDEM")
    panel._data_format.setCurrentText("TEM2Go project (folder)")
    panel._load_sounding(120)
    inv = panel._collect_inv()
    assert inv["robust_errors"] and not inv["reject_outliers"]
    result = workflow.tdem_joint_invert(panel._data, panel._collect_geom(), inv)
    info = result["robust"]
    count = sum(len(item["times"]) for item in panel._data["moments"].values())
    assert result["n_data"] == info["kept"] == info["n_start"] == count
    assert info["downweighted"] > 0 and info["dropped"] == 0
    assert result["fit_mask"].all()
    for name, item in result["moments"].items():
        assert item["fit_mask"].all()
        assert item["weights"].size == len(panel._data["moments"][name]["times"])


# -- the views hold up on nothing --------------------------------------------
def test_the_survey_view_clears_without_a_survey(app):
    view = EMSurveyView()
    view.set_summary(None)
    assert view._table.rowCount() == 0
    assert view._totals.text() == ""
    view.set_summary({"rows": [], "totals": {}})
    assert view._table.rowCount() == 0


def test_the_metadata_view_clears_without_metadata(app):
    view = EMMetadataView()
    view.set_metadata(None)
    assert view.topLevelItemCount() == 0
    view.set_metadata({"system": {"tx_area": 0.4, "gates": np.arange(30)}})
    assert view.topLevelItemCount() == 1


def test_the_gate_view_clears_without_a_report(app):
    view = EMGateView()
    view.set_report(None)
    assert view._table.rowCount() == 0
    assert view._caption.text() == ""


def test_the_gate_view_draws_a_report_with_no_survivors(app):
    """A station the selection empties still has gates worth showing."""
    view = EMGateView()
    view.set_report({
        "station": "1_00001", "line": 1,
        "moments": {"HM": {
            "times": np.logspace(-5, -4, 4),
            "open": np.logspace(-5, -4, 4) * 0.9,
            "close": np.logspace(-5, -4, 4) * 1.1,
            "values": np.array([1e-9, -2e-10, 3e-11, np.nan]),
            "relative_std": np.array([0.4, 0.5, 0.6, np.nan]),
            "flags": np.zeros(4),
            "status": np.array(["flagged out", "reversed sign", "noisy",
                                "dummy"], dtype=object),
            "held": 4, "kept": 0,
        }},
    })
    assert view._table.rowCount() == 4
    assert "0 of 4 gates kept" in view._caption.text()
    view._show_windows.setChecked(True)     # the window bars must not raise
    view._show_dropped.setChecked(False)    # nor must hiding everything


def test_the_gate_view_takes_a_model_overlay_and_gives_it_back(app):
    view = EMGateView()
    view.set_report({"station": "s", "line": 1, "moments": {"HM": {
        "times": np.logspace(-5, -4, 3), "open": np.logspace(-5, -4, 3) * 0.9,
        "close": np.logspace(-5, -4, 3) * 1.1,
        "values": np.array([1e-9, 5e-10, 1e-10]),
        "relative_std": np.array([0.05, 0.06, 0.07]), "flags": np.ones(3),
        "status": np.array(["kept"] * 3, dtype=object), "held": 3, "kept": 3}}})
    before = len(view._plot.getPlotItem().items)
    view.set_model({"HM": {"times": np.logspace(-5, -4, 20),
                           "pred": np.logspace(-9, -10, 20)}})
    assert len(view._plot.getPlotItem().items) > before
    view.set_model(None)
    assert len(view._plot.getPlotItem().items) == before


def test_a_result_becomes_a_gate_overlay():
    joint = _modelled_gates({
        "joint_moments": True,
        "moments": {"LM": {"times": np.array([1e-5]), "obs": np.array([1.0]),
                           "pred": np.array([1.1]),
                           "fit_mask": np.array([True])}},
    })
    assert set(joint) == {"LM"}
    assert list(joint["LM"]["fit_mask"]) == [True]
    single = _modelled_gates({"times": np.array([1e-5]),
                              "pred": np.array([1.1]), "tem_moment": "LM",
                              "fit_mask": np.array([False])})
    assert set(single) == {"LM"}
    assert list(single["LM"]["fit_mask"]) == [False]
    assert _modelled_gates({"frequencies": np.array([1e3]),
                            "pred_real": np.array([1.0])}) is None
    assert _modelled_gates({}) is None


def _three_kept_report():
    """One moment, three gates, all of them surviving the loader's selection."""
    times = np.logspace(-5, -4, 3)
    return {"station": "s", "line": 1, "moments": {"HM": {
        "times": times, "open": times * 0.9, "close": times * 1.1,
        "values": np.array([1e-9, 5e-10, 1e-10]),
        "relative_std": np.array([0.05, 0.06, 0.07]), "flags": np.ones(3),
        "status": np.array(["kept"] * 3, dtype=object), "held": 3, "kept": 3}}}


def test_the_view_marks_the_gates_the_fit_threw_out(app):
    """Outlier rejection runs after the loader's selection, so the view must
    say so; otherwise the picture claims the fit used a gate it discarded."""
    view = EMGateView()
    view.set_report(_three_kept_report())
    assert "The fit rejected" not in view._caption.text()
    before = len(view._plot.getPlotItem().items)
    view.set_model({"HM": {"times": np.logspace(-5, -4, 3),
                           "pred": np.array([1e-9, 5e-10, 1e-10]),
                           "fit_mask": np.array([True, False, True])}})
    assert "The fit rejected 1 of the 3 it was given." in view._caption.text()
    assert len(view._plot.getPlotItem().items) > before


def test_a_fit_mask_of_the_wrong_length_is_not_drawn(app):
    """A mask from a different selection cannot be mapped onto these gates.

    Drawing it anyway would put the marker on whichever gate happened to sit at
    that index, which is worse than not drawing it.
    """
    view = EMGateView()
    view.set_report(_three_kept_report())
    view.set_model({"HM": {"times": np.logspace(-5, -4, 3),
                           "pred": np.array([1e-9, 5e-10, 1e-10]),
                           "fit_mask": np.array([True, False])}})
    assert "The fit rejected" not in view._caption.text()


def test_a_fit_that_kept_everything_says_nothing(app):
    view = EMGateView()
    view.set_report(_three_kept_report())
    view.set_model({"HM": {"times": np.logspace(-5, -4, 3),
                           "pred": np.array([1e-9, 5e-10, 1e-10]),
                           "fit_mask": np.ones(3, dtype=bool)}})
    assert "The fit rejected" not in view._caption.text()


@needs_project
def test_the_marked_gates_are_the_ones_the_run_actually_dropped(panel):
    """End to end: a real fit on a station whose rejection reaches the floor."""
    from PyHydroGeophysX.workflows import em1d as workflows_em1d

    panel._source_path = PROJECT
    panel._method.setCurrentText("TDEM")
    panel._data_format.setCurrentText("TEM2Go project (folder)")
    panel._tail_cut.setValue(0.30)
    panel._load_sounding(120)
    # The panel no longer offers the legacy path, so the opt-in is made in the
    # settings the run is handed. The overlay has to keep mapping a fit_mask
    # onto the right gates for a configuration that still uses it.
    inv = {**panel._collect_inv(), "robust_errors": False,
           "reject_outliers": True, "outlier_threshold": 3.0,
           "outlier_passes": 2, "min_data_fraction": 0.8,
           "min_gates_per_sounding": 3}
    result = workflows_em1d.tdem_joint_invert(
        panel._data, panel._collect_geom(), inv)
    report = result["outliers"]
    assert report["enabled"] and report["dropped"] > 0
    assert report["kept"] >= report["floor"]        # the floor is honoured
    assert result["n_data"] == report["kept"]       # and reported consistently
    # This station has more outliers than the floor allows, so the run has to
    # stop at the floor and say that it did rather than keep cutting.
    assert report["kept"] == report["floor"]
    assert report["stopped_because"] == "at the 80% floor"
    assert sum(p["dropped"] for p in report["passes"]) == report["dropped"]

    overlay = _modelled_gates(result)
    dropped = sum(int((~np.asarray(item["fit_mask"], dtype=bool)).sum())
                  for item in overlay.values())
    assert dropped == report["dropped"]
    # And the mask lines up with the gates the report calls kept.
    for name, item in overlay.items():
        held = panel._gate_view._report["moments"][name]
        assert np.asarray(item["fit_mask"]).size == int(
            (held["status"] == "kept").sum())


# -- the views on a real survey ----------------------------------------------
@needs_project
def test_loading_a_project_fills_every_data_view(panel):
    panel._source_path = PROJECT
    panel._method.setCurrentText("TDEM")
    panel._data_format.setCurrentText("TEM2Go project (folder)")
    panel._load_sounding(0)

    assert panel._survey_view._table.rowCount() == 929
    assert "stations" in panel._survey_view._totals.text()
    assert panel._metadata_view.topLevelItemCount() >= 3
    assert panel._gate_view._table.rowCount() > 0
    assert "gates kept" in panel._gate_view._caption.text()


@needs_project
def test_the_station_table_opens_in_file_order(panel):
    """Enabling sorting applies the header's indicator, which would reorder it."""
    panel._source_path = PROJECT
    panel._method.setCurrentText("TDEM")
    panel._data_format.setCurrentText("TEM2Go project (folder)")
    panel._load_sounding(0)
    table = panel._survey_view._table
    shown = [table.item(row, 0).text() for row in range(5)]
    assert shown == [r["station"] for r in panel._survey_view._rows[:5]]


@needs_project
def test_picking_a_station_loads_that_station(panel):
    panel._source_path = PROJECT
    panel._method.setCurrentText("TDEM")
    panel._data_format.setCurrentText("TEM2Go project (folder)")
    panel._load_sounding(0)
    picked = []
    panel._survey_view.stationPicked.connect(picked.append)
    panel._survey_view._table.selectRow(5)
    assert picked == [5]


def _fake_summary(n, offset=0):
    """A survey of ``n`` identical stations, enough for the table to fill."""
    rows = [{"station": "s%04d" % (i + offset), "line": 1, "x": float(i),
             "y": 0.0, "elevation": 0.0, "rx_tx_distance": 15.0,
             "gates_kept": 4, "LM_gates_kept": 2, "HM_gates_kept": 2,
             "LM_gates_held": 9, "HM_gates_held": 16,
             "LM_median_std": 0.05, "HM_median_std": 0.05}
            for i in range(n)]
    return {"rows": rows,
            "totals": {"stations": n, "stations_with_data": n,
                       "stations_emptied": 0, "gates_kept": 4 * n,
                       "gates_held": 25 * n}}


def test_a_refill_that_drops_the_selected_row_is_not_a_pick(app):
    """Tightening the error cut empties stations, which shortens the table.

    Qt moves the selection when the row under it goes, and that reads back
    through ``itemSelectionChanged`` as though the user had clicked. Without the
    guard the panel loads a station nobody asked for, and does so in the middle
    of the refresh that shortened the table.
    """
    view = EMSurveyView()
    view.set_summary(_fake_summary(20))
    view._table.selectRow(15)
    picked = []
    view.stationPicked.connect(picked.append)
    view.set_summary(_fake_summary(5))      # the selected row no longer exists
    assert picked == []
    view.set_summary(_fake_summary(20))     # nor does growing it back
    assert picked == []
    view.set_summary(None)
    assert picked == []


def test_a_pick_still_reaches_the_panel_after_a_refill(app):
    """The guard must not be left on, or the table would stop responding."""
    view = EMSurveyView()
    view.set_summary(_fake_summary(20))
    view.set_summary(_fake_summary(5))
    picked = []
    view.stationPicked.connect(picked.append)
    view._table.selectRow(2)
    assert picked == [2]


@needs_project
def test_a_refresh_does_not_read_back_as_a_pick(panel):
    """The same, through the panel, where the refresh is what refills."""
    panel._source_path = PROJECT
    panel._method.setCurrentText("TDEM")
    panel._data_format.setCurrentText("TEM2Go project (folder)")
    panel._load_sounding(0)
    panel._survey_view._table.selectRow(3)
    picked = []
    panel._survey_view.stationPicked.connect(picked.append)
    panel._refresh_survey_views()
    assert picked == []


# -- one sounding tab, two views ---------------------------------------------
def test_the_sounding_tab_opens_on_the_curve_viewer(panel):
    """With nothing loaded the gate view has nothing to draw."""
    assert panel._sounding_stack.currentWidget() is panel._curve
    titles = [panel._tabs.tabText(i) for i in range(panel._tabs.count())]
    assert "Gates" not in titles, "the gate view has its own tab again"
    assert titles.count("Sounding") == 1


@needs_project
def test_a_project_raises_the_gate_view_and_anything_else_lowers_it(panel):
    """Only a project records the gates a station dropped.

    A text sounding or a frequency sweep holds the survivors and nothing else,
    so the gate view would be blank and the curve viewer is the only view its
    data supports.
    """
    panel._source_path = PROJECT
    panel._method.setCurrentText("TDEM")
    panel._data_format.setCurrentText("TEM2Go project (folder)")
    panel._load_sounding(0)
    assert panel._sounding_stack.currentWidget() is panel._gate_view

    panel._data = {**panel._data, "temcompany": False}
    panel._refresh_gate_view()
    assert panel._sounding_stack.currentWidget() is panel._curve


@needs_project
def test_the_gate_view_exports_every_gate_not_only_the_survivors(panel, tmp_path):
    """A file holding only what survived cannot be told from a thin station."""
    import csv

    panel._source_path = PROJECT
    panel._method.setCurrentText("TDEM")
    panel._data_format.setCurrentText("TEM2Go project (folder)")
    panel._load_sounding(120)
    target = tmp_path / "gates.csv"
    panel._gate_view.export_csv(str(target))

    rows = list(csv.reader(target.open(encoding="utf-8")))
    header, data = rows[0], rows[1:]
    assert header[:2] == ["station", "line"] and header[-1] == "Verdict"
    report = panel._gate_view._report
    held = sum(int(m["held"]) for m in report["moments"].values())
    kept = sum(int(m["kept"]) for m in report["moments"].values())
    assert len(data) == held, "the export dropped the gates the selection did"
    assert sum(1 for r in data if r[-1] == "kept") == kept
    assert kept < held, "this station should have dropped gates to export"
    assert {r[0] for r in data} == {report["station"]}


@needs_project
def test_the_gate_view_writes_a_png(panel, tmp_path):
    panel._source_path = PROJECT
    panel._method.setCurrentText("TDEM")
    panel._data_format.setCurrentText("TEM2Go project (folder)")
    panel._load_sounding(120)
    target = tmp_path / "gates.png"
    panel._gate_view.export_png(str(target))
    assert target.stat().st_size > 1000


# -- the acquisition description, off the tab bar and onto the agent ---------
def test_the_acquisition_tab_is_gone_but_the_answer_is_not(panel):
    """It was reference rather than a working view, so it moved to the agent.

    The tab may go only because the answer is reachable without it. Without the
    action those fields would exist nowhere a user could get at them, which is
    worse than a tab nobody opens.
    """
    titles = [panel._tabs.tabText(i) for i in range(panel._tabs.count())]
    assert "Acquisition" not in titles
    assert "get_acquisition" in panel.agent_apply("nope", {})["valid_actions"]


def test_asking_before_anything_is_loaded_says_so(panel):
    answer = panel.agent_apply("get_acquisition", {})
    assert answer["status"] == "failed"
    assert "No data" in answer["error"]


def test_an_unknown_section_lists_the_ones_that_exist(panel):
    answer = panel.agent_apply("get_acquisition", {"section": "waveform"})
    assert answer["status"] == "failed"
    assert "all" in answer["valid"] and "instrument" in answer["valid"]


@needs_project
def test_the_agent_gets_what_the_tab_used_to_show(panel):
    """Whole arrays rather than the summaries the tree drew.

    A reader skimming a column wants to know a waveform has twenty-two nodes.
    Something answering a question about it needs the nodes.
    """
    import json

    panel._source_path = PROJECT
    panel._method.setCurrentText("TDEM")
    panel._data_format.setCurrentText("TEM2Go project (folder)")
    panel._load_sounding(0)

    answer = panel.agent_apply("get_acquisition", {})
    assert answer["status"] == "ok"
    assert set(answer) >= {"instrument", "system", "protocol", "inversion_defaults"}
    # It has to survive being sent, which the reader's numpy arrays do not.
    json.dumps(answer)

    instrument = answer["instrument"]
    assert instrument["tx_rx_sep"] > 0
    assert instrument["data_factor_applied"] is False
    assert len(instrument["analog_lowpass_hz"]) == 2
    waveform = instrument["moments"]["LM"]["waveform_times"]
    assert len(waveform) > 10 and all(isinstance(v, float) for v in waveform)

    one = panel.agent_apply("get_acquisition", {"section": "protocol"})
    assert set(one) - {"status", "source"} == {"protocol"}


@needs_project
def test_the_description_matches_what_the_reader_returned(panel):
    """The action reports the file, not a copy that could drift from it."""
    panel._source_path = PROJECT
    panel._method.setCurrentText("TDEM")
    panel._data_format.setCurrentText("TEM2Go project (folder)")
    panel._load_sounding(0)
    answer = panel.agent_apply("get_acquisition", {"section": "system"})
    for key, value in panel._data["system"].items():
        got = answer["system"][key]
        if isinstance(value, float) and np.isfinite(value):
            assert got == pytest.approx(value), key
        elif isinstance(value, (str, bool, int)):
            assert got == value, key


# -- survey geometry ---------------------------------------------------------
def test_the_two_sensor_heights_are_separate_controls(panel):
    """The forward places the loop and the coil separately.

    One field for both could only ever set them equal, which is exactly what a
    frame mounting them differently would need to override and could not.
    """
    panel._tx_height.setValue(2.0)
    panel._rx_height.setValue(0.5)
    geom = panel._collect_geom()
    assert geom["tx_height"] == pytest.approx(2.0)
    assert geom["rx_height"] == pytest.approx(0.5)
    # The general key travels too, or a reader that prefers it would be handed
    # a value nothing on the panel chose.
    assert geom["height"] == pytest.approx(0.5)


@needs_project
def test_a_project_geometry_wins_until_the_switch_is_cleared(panel):
    """Editing the fields used to do nothing on a project, silently.

    Every station records its own separation and heights, and those are overlaid
    on whatever the panel sent, so a user could type forty metres and watch the
    run use fourteen with nothing to say why.
    """
    from PyHydroGeophysX.workflows.em1d import _station_geometry

    panel._source_path = PROJECT
    panel._method.setCurrentText("TDEM")
    panel._data_format.setCurrentText("TEM2Go project (folder)")
    panel._load_sounding(0)
    assert panel._per_station_geometry.isChecked(), "the recorded geometry is the default"

    panel._tx_rx.setValue(40.0)
    panel._tx_height.setValue(5.0)
    panel._rx_height.setValue(3.0)

    kept = _station_geometry(panel._collect_geom(), panel._data)
    assert kept["tx_rx_sep"] != pytest.approx(40.0), "the station should win"
    assert kept["tx_height"] == pytest.approx(
        panel._data["system"]["tx_height"])

    panel._per_station_geometry.setChecked(False)
    imposed = _station_geometry(panel._collect_geom(), panel._data)
    assert imposed["tx_rx_sep"] == pytest.approx(40.0)
    assert imposed["tx_height"] == pytest.approx(5.0)
    assert imposed["rx_height"] == pytest.approx(3.0)


@needs_project
def test_loading_a_project_fills_both_heights_from_the_file(panel):
    panel._source_path = PROJECT
    panel._method.setCurrentText("TDEM")
    panel._data_format.setCurrentText("TEM2Go project (folder)")
    panel._load_sounding(0)
    system = panel._data["system"]
    assert panel._tx_height.value() == pytest.approx(system["tx_height"], abs=1e-6)
    assert panel._rx_height.value() == pytest.approx(system["rx_height"], abs=1e-6)


def test_the_agent_can_still_set_a_single_height(panel):
    """The old key set one number; it now has to reach both."""
    panel.agent_apply("set_params", {"params": {"height": 4.0}})
    geom = panel._collect_geom()
    assert geom["tx_height"] == pytest.approx(4.0)
    assert geom["rx_height"] == pytest.approx(4.0)
    panel.agent_apply("set_params", {"params": {"rx_height": 1.5}})
    assert panel._collect_geom()["rx_height"] == pytest.approx(1.5)
    assert panel._collect_geom()["tx_height"] == pytest.approx(4.0)
