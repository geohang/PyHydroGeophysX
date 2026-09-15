"""Regression coverage for field-data input edge cases."""
import datetime as dt
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from PyHydroGeophysX.data_processing import ert_io


@pytest.mark.parametrize("stamp", [
    "2026_01_12_14_10_49", "2026-01-12_141049",
    "20260112141049", "2026-01-12T14:10:49",
])
def test_filename_timestamp_includes_seconds(stamp):
    assert ert_io._parse_date(f"results_{stamp}_pygimli") == dt.datetime(2026, 1, 12, 14, 10, 49)


def test_second_spaced_surveys_keep_elapsed_time_and_distinct_labels():
    times, labels = ert_io.measurement_times_for([
        "results_2026_01_12_14_10_49.dat",
        "results_2026_01_12_14_10_50.dat",
    ])
    assert times == pytest.approx([0., 1. / 86400.])
    assert labels == ["2026-01-12 14:10:49", "2026-01-12 14:10:50"]


def test_date_only_and_unparseable_names_keep_existing_behavior():
    assert ert_io.measurement_times_for(["2021-10-08.dat", "2021-10-10.dat"])[0] == [0., 2.]
    assert ert_io.measurement_times_for(["survey_a.dat", "survey_b.dat"]) == ([1., 2.], ["1", "2"])
    assert ert_io._parse_date("2026-02-30") is None


@pytest.mark.parametrize("fail", [False, True])
def test_instrument_loader_cleans_temporary_project(monkeypatch, tmp_path, fail):
    seen = []
    real_temporary_directory = ert_io.tempfile.TemporaryDirectory
    monkeypatch.setattr(ert_io.tempfile, "TemporaryDirectory", lambda **kw:
                        real_temporary_directory(dir=tmp_path, **kw))
    def load(**kwargs):
        project = Path(kwargs["project_dir"])
        seen.append(project)
        (project / "reader_output.txt").write_text("temporary")
        if fail:
            raise ValueError("bad input")
        return object()
    monkeypatch.setitem(sys.modules, "PyHydroGeophysX.data_processing.ert_data_agent",
                        SimpleNamespace(load_ert_resipy=load))
    sentinel = object()
    monkeypatch.setattr(ert_io, "standard_to_pg", lambda std: sentinel)
    result = ert_io._via_instrument("survey.dat", "BERT", None, None, lambda msg: None)
    assert result is (None if fail else sentinel)
    assert seen and not seen[0].exists()


@pytest.mark.parametrize("timelapse", [False, True])
@pytest.mark.parametrize("errors", [[.002, 0., -.1, np.nan, np.inf], [.002]*5])
def test_srt_replaces_only_invalid_error_entries(timelapse, errors):
    pg = pytest.importorskip("pygimli")
    from PyHydroGeophysX.inversion.srt_inversion import SRTInversion
    from PyHydroGeophysX.inversion.srt_time_lapse import TimeLapseSRTInversion
    data = pg.DataContainer()
    data.resize(5)
    data["err"] = errors
    observed = np.linspace(.01, .05, 5)
    params = {"relativeError": .03, "absoluteError": .001}
    if timelapse:
        inv = object.__new__(TimeLapseSRTInversion)
        inv.parameters = params
        actual = inv._estimate_errors(data, observed)
    else:
        inv = object.__new__(SRTInversion)
        inv.data, inv.parameters = data, params
        actual = inv._estimate_data_errors(observed)
    expected = np.sqrt((.03 * observed)**2 + .001**2)
    valid = np.isfinite(errors) & (np.asarray(errors) > 0)
    expected[valid] = np.asarray(errors)[valid]
    np.testing.assert_allclose(actual, expected)
