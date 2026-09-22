"""Acquisition times read off a monitoring sequence."""

import datetime as dt
import os

import pytest

from PyHydroGeophysX.data_processing.survey_timing import (
    format_duration,
    parse_timestamp,
    survey_timing,
    timestamp_from_header,
)


def test_iso_names_give_dates_intervals_and_elapsed_days():
    timing = survey_timing([
        "line1_2021-10-08_1400.ohm",
        "line1_2021-10-08_1500.ohm",
        "line1_2021-10-09_1400.ohm",
    ])
    assert timing.dated and timing.source == "file names"
    assert timing.timestamps[0] == dt.datetime(2021, 10, 8, 14, 0)
    assert timing.times == pytest.approx([0.0, 1 / 24, 1.0])
    assert [format_duration(g) for g in timing.intervals] == ["1 h 00 min", "23 h 00 min"]
    # Labels keep the clock time because two surveys share a date.
    assert timing.labels[0] == "2021-10-08 14:00"


def test_compact_names_parse_and_report_the_span():
    timing = survey_timing(["20171105_1418.Data", "20171109_1417.Data"])
    assert timing.dated
    assert timing.timestamps[-1] == dt.datetime(2017, 11, 9, 14, 17)
    assert "4 d 00 h" in timing.summary()


def test_a_pattern_has_to_read_the_whole_set():
    """One dated file among undated ones is not a time axis."""
    timing = survey_timing(["2021-10-08_1400.ohm", "second_survey.ohm"])
    assert not timing.dated
    assert timing.source == "index"
    assert timing.times == [1.0, 2.0]
    assert "no readable acquisition time" in timing.summary()


def test_us_style_names_are_read_when_the_whole_set_agrees():
    timing = survey_timing(["site_06-12-2024 14-30.dat", "site_06-13-2024 09-15.dat"])
    assert timing.dated
    assert timing.timestamps[0] == dt.datetime(2024, 6, 12, 14, 30)


def test_repeated_timestamps_cannot_order_a_sequence():
    timing = survey_timing(["a_2024-06-12.dat", "b_2024-06-12.dat"])
    assert not timing.dated and timing.source == "index"


def test_header_timestamp_is_the_fallback_for_renamed_files(tmp_path):
    for index, stamp in enumerate(("2024-03-01 08:00:00", "2024-03-01 12:30:00")):
        path = tmp_path / f"export_{index}.dat"
        path.write_text(f"Instrument: demo\nDate/Time: {stamp}\n4\n1 2 3 4 10.0\n")
    files = sorted(str(p) for p in tmp_path.glob("export_*.dat"))
    timing = survey_timing(files)
    assert timing.dated and timing.source == "file headers"
    assert format_duration(timing.intervals[0]) == "4 h 30 min"
    assert timestamp_from_header(files[0]) == dt.datetime(2024, 3, 1, 8, 0)


def test_modification_times_are_opt_in(tmp_path):
    files = []
    for index in range(2):
        path = tmp_path / f"plain_{index}.dat"
        path.write_text("1 2 3 4 10.0\n")
        stamp = dt.datetime(2024, 5, 1, 9 + index).timestamp()
        os.utime(path, (stamp, stamp))
        files.append(str(path))
    assert not survey_timing(files).dated
    opted_in = survey_timing(files, allow_mtime=True)
    assert opted_in.source == "file modification times"
    assert format_duration(opted_in.intervals[0]) == "1 h 00 min"


def test_supplied_times_win_over_the_names():
    stamps = [dt.datetime(2020, 1, 1, 0, 0), dt.datetime(2020, 1, 1, 6, 0)]
    timing = survey_timing(["2021-10-08_1400.ohm", "2021-10-09_1400.ohm"],
                           timestamps=stamps)
    assert timing.timestamps == stamps
    assert timing.times == pytest.approx([0.0, 0.25])


@pytest.mark.parametrize("seconds, text", [
    (45, "45 s"),
    (95, "1.6 min"),
    (1800, "30 min"),
    (3600, "1 h 00 min"),
    (3660, "1 h 01 min"),
    (86400 * 2 + 3600 * 6, "2 d 06 h"),
    (None, ""),
    (float("nan"), ""),
])
def test_durations_are_reported_in_units_a_reader_recognises(seconds, text):
    assert format_duration(seconds) == text


def test_an_impossible_date_does_not_stop_the_scan():
    assert parse_timestamp("run_2021-13-45_more_2021-10-08")[0] == dt.datetime(2021, 10, 8)


def test_the_table_carries_the_interval_in_both_forms():
    timing = survey_timing(["a_2024-06-12_0900.dat", "b_2024-06-12_1200.dat"])
    rows = timing.rows()
    assert timing.csv_header()[0] == "index"
    assert rows[0][4] == "" and rows[0][5] == ""          # nothing before the first
    assert rows[1][4] == pytest.approx(0.125) and rows[1][5] == "3 h 00 min"
    assert timing.to_dict()["median_interval"] == "3 h 00 min"
