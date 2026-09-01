"""Running one survey line rather than a prefix of the whole file.

A survey whose lines differ in data quality cannot be served by one set of gate
and rejection settings. The lateral constraint never crosses a line, so a line
inverted on its own is tied exactly as it would be inside a whole-survey run;
what changes is which settings reach it. These tests hold the selection itself:
that a named line resolves to that line's stations and no others, and that a
request which cannot be honoured is refused rather than quietly widened.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from PyHydroGeophysX.workflows.em1d import _line_block

PROJECT = Path(__file__).resolve().parents[2] / "TEM2go_data" / "trailcreek"

needs_project = pytest.mark.skipif(
    not (PROJECT / "project.db").exists(),
    reason="the TEM2Go trailcreek project is not present")


def _head(lines, n=None):
    """The little a line selection reads out of a loaded survey."""
    numbers = np.asarray(lines, dtype=int)
    return {"n_soundings": int(n if n is not None else numbers.size),
            "line_numbers": numbers}


# -- the resolver ------------------------------------------------------------
def test_no_selection_runs_the_whole_survey_from_its_first_station():
    assert _line_block(_head([1, 1, 2, 2, 2]), None) == (0, 5)


def test_a_named_line_resolves_to_its_own_block():
    head = _head([1, 1, 2, 2, 2, 3])
    assert _line_block(head, [1]) == (0, 2)
    assert _line_block(head, [2]) == (2, 3)
    assert _line_block(head, [3]) == (5, 1)


def test_adjacent_lines_may_be_run_together():
    head = _head([1, 1, 2, 2, 2, 3])
    assert _line_block(head, [1, 2]) == (0, 5)
    assert _line_block(head, [2, 3]) == (2, 4)


def test_lines_that_are_not_adjacent_are_refused():
    """Widening to the span would invert a line nobody asked for.

    Lines 1 and 3 enclose line 2, and running it under settings chosen for its
    neighbours is worse than declining, because nothing downstream would record
    that it happened.
    """
    with pytest.raises(ValueError, match="not adjacent"):
        _line_block(_head([1, 1, 2, 2, 3]), [1, 3])


def test_a_line_the_survey_does_not_have_is_refused_and_says_what_it_has():
    with pytest.raises(ValueError, match=r"\[1, 2\]"):
        _line_block(_head([1, 1, 2]), [7])


def test_an_empty_selection_is_refused():
    with pytest.raises(ValueError, match="at least one"):
        _line_block(_head([1, 1, 2]), [])


def test_a_source_with_no_line_numbers_is_refused():
    """A plain text sounding file records no line, so it cannot be split."""
    with pytest.raises(ValueError, match="line number per station"):
        _line_block({"n_soundings": 4, "line_numbers": []}, [1])


def test_a_repeated_line_number_is_not_counted_twice():
    assert _line_block(_head([1, 1, 2, 2, 2]), [2, 2, 2]) == (2, 3)


# -- against the real survey -------------------------------------------------
@needs_project
def test_each_trailcreek_line_resolves_to_only_its_own_stations():
    from PyHydroGeophysX.workflows import em1d as workflows_em1d

    head = workflows_em1d.load_sounding(
        str(PROJECT), "TDEM", sounding=0, moment="LM+HM",
        max_relative_std=None, reject_negative=False)
    numbers = np.asarray(head["line_numbers"], dtype=int)
    seen = 0
    for line in sorted(set(numbers.tolist())):
        offset, count = _line_block(head, [line])
        block = numbers[offset:offset + count]
        assert np.all(block == line), f"line {line} picked up other lines"
        assert count == int(np.count_nonzero(numbers == line))
        seen += count
    assert seen == numbers.size, "the lines together must cover the survey"


@needs_project
def test_inverting_one_line_returns_only_that_line():
    """The whole path, not just the resolver: three stations off line 2."""
    from PyHydroGeophysX.workflows import em1d as workflows_em1d

    head = workflows_em1d.load_sounding(
        str(PROJECT), "TDEM", sounding=0, moment="LM+HM",
        max_relative_std=None, reject_negative=False)
    geom = {**head["system"], "tem_moment": "LM+HM"}
    inv = {**workflows_em1d.DEFAULT_INVERSION, "n_layers": 8,
           "max_iterations": 2, "auto_lambda": False, "data_scale": 1.0}
    result = workflows_em1d.invert_line(
        str(PROJECT), "TDEM", geom, inv,
        positions=np.asarray(head["positions"], dtype=float),
        max_soundings=3, lines=[2], doi_blank=False)
    assert np.all(np.asarray(result["line_numbers"], dtype=int) == 2)
    # And they are line 2's own first stations, not the file's.
    expected = np.asarray(head["station_ids"], dtype=str)[
        _line_block(head, [2])[0]:][:3]
    assert list(np.asarray(result["station_ids"], dtype=str)) == list(expected)
    # The positions travel with the stations rather than restarting at zero.
    offset = _line_block(head, [2])[0]
    assert np.allclose(
        np.asarray(result["positions"], dtype=float),
        np.asarray(head["positions"], dtype=float)[offset:offset + 3])


# -- the panel control -------------------------------------------------------
@needs_project
def test_the_panel_offers_the_survey_s_lines_and_keeps_the_choice():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from PyHydroGeophysX.qt_apps.modules.em_processing import EMProcessingModule

    app = QApplication.instance() or QApplication([])
    panel = EMProcessingModule(None, lambda *a, **k: None)
    try:
        assert panel._selected_lines() is None      # nothing loaded yet
        panel._source_path = PROJECT
        panel._method.setCurrentText("TDEM")
        panel._data_format.setCurrentText("TEM2Go project (folder)")
        panel._load_sounding(0)
        offered = [panel._line_pick.itemData(i)
                   for i in range(panel._line_pick.count())]
        assert offered == [None, 1, 2, 3]
        panel._line_pick.setCurrentIndex(offered.index(3))
        assert panel._selected_lines() == [3]
        # Re-reading the same survey under another gate setting must not widen
        # the run back to every line.
        panel._load_sounding(0)
        assert panel._selected_lines() == [3]
    finally:
        panel.close()
