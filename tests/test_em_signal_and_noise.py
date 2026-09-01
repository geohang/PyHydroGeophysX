"""Signal and absolute noise, recorded per station and drawn along a line.

A station returns fewer usable gates for two reasons that call for opposite
readings. A smaller signal over a steady noise floor is evidence about the
ground, since a resistive half-space returns dB/dt going as ``rho ** -1.5`` and
is genuinely quieter. A steady signal under a risen floor is an instrument or an
environment and says nothing about the ground. The relative error the file
records is the second divided by the first, so it rises either way; these tests
hold the property that makes the pair separable, which is that both are recorded
rather than one being derived from the other after the fact.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from PyHydroGeophysX.data_processing.em1d import (
    _reference_gate_signal,
    gate_report,
    survey_summary,
)

PROJECT = Path(__file__).resolve().parents[2] / "TEM2go_data" / "trailcreek"

needs_project = pytest.mark.skipif(
    not (PROJECT / "project.db").exists(),
    reason="the TEM2Go trailcreek project is not present")


# -- the reference gate ------------------------------------------------------
def test_the_reference_gate_is_the_voltage_and_its_own_scatter():
    signal, noise, at = _reference_gate_signal(
        [1e-7, 2e-8, 3e-9], [0.05, 0.10, 0.20], [5e-6, 9e-6, 30e-6], 1)
    assert signal == pytest.approx(2e-8)
    assert noise == pytest.approx(2e-8 * 0.10)
    assert at == pytest.approx(9e-6)


def test_a_negative_gate_still_reports_its_magnitude():
    """An offset loop reverses sign at early time; that is not a missing gate."""
    signal, noise, _ = _reference_gate_signal([-2e-8], [0.10], [5e-6], 0)
    assert signal == pytest.approx(2e-8)
    assert noise == pytest.approx(2e-9)


@pytest.mark.parametrize("values,why", [
    ([1e-7, np.nan], "a non-finite value"),
    ([1e-7, 9_999.0], "a dummy fill value"),
])
def test_a_gate_without_a_measurement_reports_nothing(values, why):
    """A substitute here would read as a measurement, so it stays NaN."""
    signal, noise, at = _reference_gate_signal(
        values, [0.05, 0.05], [5e-6, 9e-6], 1)
    assert np.isnan(signal) and np.isnan(noise) and np.isnan(at), why


def test_an_index_past_the_table_reports_nothing():
    assert all(np.isnan(v)
               for v in _reference_gate_signal([1e-7], [0.05], [5e-6], 7))


def test_a_missing_error_leaves_the_signal_readable():
    """The voltage is still a measurement when its scatter was not recorded."""
    signal, noise, _ = _reference_gate_signal([1e-7], [np.nan], [5e-6], 0)
    assert signal == pytest.approx(1e-7)
    assert np.isnan(noise)


# -- against the real survey -------------------------------------------------
@needs_project
def test_the_summary_records_both_halves_for_every_station():
    summary = survey_summary(str(PROJECT), max_relative_std=None,
                             reject_negative=False)
    rows = summary["rows"]
    assert rows
    for name in ("LM", "HM"):
        signal = np.asarray([r[f"{name}_signal"] for r in rows], dtype=float)
        noise = np.asarray([r[f"{name}_noise"] for r in rows], dtype=float)
        at = np.asarray([r[f"{name}_reference_time"] for r in rows], dtype=float)
        assert np.isfinite(signal).all(), f"{name} lost a station"
        assert (signal > 0).all()
        assert (noise[np.isfinite(noise)] >= 0).all()
        # One index is one physical time, which is what makes the column
        # comparable from station to station along a line.
        assert np.allclose(at, at[0])


@needs_project
def test_the_recorded_pair_matches_the_gate_the_file_holds():
    """The summary reads its own columns; this checks them against the file."""
    summary = survey_summary(str(PROJECT), max_relative_std=None,
                             reject_negative=False)
    checked = 0
    for index in (0, 200, 600):
        row = summary["rows"][index]
        report = gate_report(str(PROJECT), index, use_flags=False,
                             max_relative_std=None, reject_negative=False)
        # gate_report indexes the stations an inversion sees, which is a subset,
        # so only compare where the two land on the same station.
        if report["station"] != row["station"]:
            continue
        for name, moment in report["moments"].items():
            values = np.asarray(moment["values"], dtype=float)
            std = np.asarray(moment["relative_std"], dtype=float)
            assert row[f"{name}_signal"] == pytest.approx(abs(values[2]))
            assert row[f"{name}_noise"] == pytest.approx(abs(values[2]) * std[2])
            checked += 1
    assert checked, "no station lined up, so nothing was actually compared"


@needs_project
def test_the_quiet_lines_are_the_ones_that_lose_gates():
    """The survey's own trend, which is why the columns are worth recording.

    Resistive ground returns a smaller response and therefore fewer gates above
    the noise floor. If the two moved independently the pair would not be
    telling one story, and the view built on it would not be worth drawing.
    """
    summary = survey_summary(str(PROJECT), max_relative_std=None,
                             reject_negative=False)
    rows = [r for r in summary["rows"] if r["gates_kept"]]
    ratio, gates = [], []
    for line in sorted({int(r["line"]) for r in rows}):
        on_line = [r for r in rows if int(r["line"]) == line]
        signal = np.nanmedian([r["LM_signal"] for r in on_line])
        noise = np.nanmedian([r["LM_noise"] for r in on_line])
        ratio.append(signal / noise)
        gates.append(float(np.mean([r["gates_kept"] for r in on_line])))
    assert np.corrcoef(ratio, gates)[0, 1] > 0.9, (
        f"signal-to-noise and gate count disagree across lines: {ratio}, {gates}")


# -- the figure --------------------------------------------------------------
@needs_project
def test_the_figure_draws_one_panel_per_moment():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from PyHydroGeophysX.visualization import plot_signal_and_noise

    summary = survey_summary(str(PROJECT), max_relative_std=None,
                             reject_negative=False)
    fig, axes = plot_signal_and_noise(summary, line=2)
    try:
        assert len(axes) == 2
        assert all(len(ax.lines) == 2 for ax in axes)   # signal and noise
        assert "line 2" in axes[-1].get_xlabel()
    finally:
        plt.close(fig)


@needs_project
def test_the_figure_refuses_a_line_the_survey_does_not_have():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from PyHydroGeophysX.visualization import plot_signal_and_noise

    summary = survey_summary(str(PROJECT), max_relative_std=None,
                             reject_negative=False)
    with pytest.raises(ValueError, match=r"\[1, 2, 3\]"):
        plot_signal_and_noise(summary, line=9)


def test_the_figure_refuses_an_empty_summary():
    pytest.importorskip("matplotlib")
    from PyHydroGeophysX.visualization import plot_signal_and_noise

    with pytest.raises(ValueError, match="no stations"):
        plot_signal_and_noise({"rows": []})


# -- the Qt view -------------------------------------------------------------
def test_the_view_clears_without_a_survey():
    pytest.importorskip("PySide6")
    pytest.importorskip("pyqtgraph")
    from PySide6.QtWidgets import QApplication

    from PyHydroGeophysX.qt_apps.widgets.em_survey_view import EMSignalNoiseView

    QApplication.instance() or QApplication([])
    view = EMSignalNoiseView()
    view.set_summary(None)
    assert view._line.count() == 0
    assert view._caption.text() == ""


@needs_project
def test_the_view_offers_the_lines_and_keeps_the_chosen_one():
    pytest.importorskip("PySide6")
    pytest.importorskip("pyqtgraph")
    from PySide6.QtWidgets import QApplication

    from PyHydroGeophysX.qt_apps.widgets.em_survey_view import EMSignalNoiseView

    QApplication.instance() or QApplication([])
    view = EMSignalNoiseView()
    summary = survey_summary(str(PROJECT), max_relative_std=None,
                             reject_negative=False)
    view.set_summary(summary)
    assert [view._line.itemData(i)
            for i in range(view._line.count())] == [1, 2, 3]
    view._line.setCurrentIndex(2)
    assert "323 stations" in view._caption.text()
    # Re-reading the same survey under another gate setting must not send the
    # view back to the first line.
    view.set_summary(summary)
    assert view._line.currentData() == 3
