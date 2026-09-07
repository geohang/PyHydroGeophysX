"""The two TEMcompany project layouts read as one survey.

TEMImage 3 replaced ``project.db`` with ``project.tiw``. The tables were renamed
and reshaped, one flag array changed sense, and the gate centre times were
dropped. :mod:`PyHydroGeophysX.data_processing.temcompany_project` absorbs all
of that so the readers keep one code path, and these tests hold the pieces of it
that a later release could quietly break.

The fixtures build both files from the same numbers, so a test that passes on
one and fails on the other is naming a translation error rather than a
difference between two surveys. They are written from the layouts of two real
projects, an Ashton Prairie survey under the newer file and a Trail Creek survey
under the older, and they carry only the columns the readers touch.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import struct
from pathlib import Path

import numpy as np
import pytest

from PyHydroGeophysX.data_processing import temcompany_project as tcp

# A gate table shaped like a TEM2Go's: ten per decade, half a decade of overlap,
# so a gate closes where the gate two later opens.
LM_OPEN = [4.4999995e-06, 5.658838e-06, 7.1160994e-06, 8.948634e-06]
LM_CLOSE = [7.1160994e-06, 8.948634e-06, 1.1253082e-05, 1.4150971e-05]
HM_OPEN = [1.0e-05, 1.2531894e-05, 1.5704836e-05, 1.9681133e-05]
HM_CLOSE = [1.5704836e-05, 1.9681133e-05, 2.4664186e-05, 3.0908897e-05]

LM_VALUES = [3.4577570e-06, 1.7779225e-06, 9.0401818e-07, 4.7152795e-07]
HM_VALUES = [3.7917107e-07, 1.9335903e-07, 9.6930478e-08, 4.7841777e-08]
LM_STD = [0.03502774, 0.03123142, 0.03093182, 0.03145365]
HM_STD = [0.03029386, 0.03020152, 0.03017556, 0.03024479]

#: The high moment's last gate was dropped; the low moment kept all four.
HM_FILTERED = [0, 0, 0, 256]
LM_FILTERED = [0, 0, 0, 0]

WAVEFORM_TIME = [-3.0e-4, -1.0e-4, 0.0, 1.0e-6]
WAVEFORM_AMPLITUDE = [0.0, 0.9, 1.0, 0.0]


def _centres(open_times, close_times):
    return [math.sqrt(a * b) for a, b in zip(open_times, close_times)]


def _packed(*values: float) -> bytes:
    return struct.pack("<%dd" % len(values), *values)


def _model_data() -> str:
    """``StationModelTable.ModelData`` for the one station, both moments."""
    return json.dumps([
        {"MomentType": 0, "StationId": 1,
         "Time_Open": LM_OPEN, "Time_Close": LM_CLOSE,
         "dBdt_Inp": LM_VALUES, "Std_Inp": LM_STD,
         "dBdt_Fwr": [v * 1.01 for v in LM_VALUES]},
        {"MomentType": 1, "StationId": 2,
         "Time_Open": HM_OPEN[:3], "Time_Close": HM_CLOSE[:3],
         "dBdt_Inp": HM_VALUES[:3], "Std_Inp": HM_STD[:3],
         "dBdt_Fwr": [v * 0.99 for v in HM_VALUES[:3]]},
    ])


def _lupus_input() -> bytes:
    """The solver input TEMImage 3 stores beside a run."""
    return json.dumps({
        "Lupus_Settings": {
            "LogData": False, "SCI_Contraints": False, "SCI_MaxDistance": 200,
            "LC_RefDistance": 20, "LC_AutoScale": True,
            "LC_AutoScalePower": 0.75, "NormType_VerticalCon": 2,
            "NormType_LateralCon": 2, "DOIdepthMax": 200, "IteMax": 30,
        },
        "N_Layer": 4,
        "N_Model_position": 1,
        "Smooth_model": True,
        "EPSG": 32615,
        "Model_position": [{
            "XYZ": [617769.8, 4613550.9, 202.1],
            "LineNum": 1, "StaNum": 1, "LineId": 1,
            "Model_start": {"Res": [22.0] * 4, "Thk": [1.0, 1.16, 1.35]},
            "VCon_Res": [2.0, 2.0, 2.0],
            "LCon_Res": [1.3] * 4,
            "UsedStationIds": [1, 2],
        }],
    }).encode("utf-8")


def write_workspace_project(folder: Path) -> Path:
    """A one-station ``project.tiw`` in the TEMImage 3 layout."""
    path = folder / "project.tiw"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE Project (ProjectId INTEGER PRIMARY KEY, "
                "Name TEXT, EPSG INTEGER)")
    con.execute("INSERT INTO Project VALUES (1, 'project', 32615)")
    con.execute(
        "CREATE TABLE ProtocolTable (ProtocolId INTEGER PRIMARY KEY, "
        "ProjectId INTEGER, InstrumentType INTEGER, UniStd REAL, "
        "HighMomentGateOpenTimes TEXT, HighMomentGateCenterTimes TEXT, "
        "HighMomentGateCloseTimes TEXT, LowMomentGateOpenTimes TEXT, "
        "LowMomentGateCenterTimes TEXT, LowMomentGateCloseTimes TEXT, "
        "TargetCurrents TEXT, HighMomentWaveformTime TEXT, "
        "LowMomentWaveformTime TEXT, HighMomentWaveformAmplitude TEXT, "
        "LowMomentWaveformAmplitude TEXT, HighMomentPeriodTime REAL, "
        "LowMomentPeriodTime REAL, RxCoilPosition BLOB, TxCoilPosition BLOB, "
        "TxXYLength BLOB, FilterCutOffs BLOB, GateShapeChannelA INTEGER, "
        "GateShapeChannelAParameter1 REAL, TxCoilArea REAL, "
        "TxLoopTurns INTEGER, HMCalibrationFactor REAL, "
        "HMCalibrationFactorOverride REAL, LMCalibrationFactor REAL, "
        "LMCalibrationFactorOverride REAL, LMGateTimeShift REAL, "
        "LMGateTimeShiftOverride REAL, HMGateTimeShift REAL, "
        "HMGateTimeShiftOverride REAL)")
    con.execute(
        "INSERT INTO ProtocolTable VALUES (1, 1, 3, 0.0, ?, ?, ?, ?, ?, ?, ?, "
        "?, ?, ?, ?, 8.0e-4, 4.0e-4, ?, ?, ?, ?, 1, 0.667, 0.3969, 4, "
        "1.08, NULL, 1.05, NULL, 0.0, NULL, 0.0, NULL)",
        (json.dumps(HM_OPEN), json.dumps(_centres(HM_OPEN, HM_CLOSE)),
         json.dumps(HM_CLOSE), json.dumps(LM_OPEN),
         json.dumps(_centres(LM_OPEN, LM_CLOSE)), json.dumps(LM_CLOSE),
         json.dumps([1, 10]), json.dumps(WAVEFORM_TIME),
         json.dumps(WAVEFORM_TIME), json.dumps(WAVEFORM_AMPLITUDE),
         json.dumps(WAVEFORM_AMPLITUDE),
         _packed(-15.0, 0.0, 1.0), _packed(0.0, 0.0, 1.0),
         _packed(0.63, 0.63), _packed(450000.0, 800000.0)))
    con.execute("CREATE TABLE LineTable (LineId INTEGER PRIMARY KEY, "
                "ProjectId INTEGER, LineNumber INTEGER)")
    con.execute("INSERT INTO LineTable VALUES (1, 1, 1)")
    con.execute(
        "CREATE TABLE StationStacks (StationId INTEGER PRIMARY KEY, "
        "FileId INTEGER, StationNumber INTEGER, LineNumber INTEGER, "
        "TransmitterMoment INTEGER, TxCurrent REAL, Time TEXT, "
        "Latitude REAL, Longitude REAL, Altitude REAL, UtmX REAL, UtmY REAL, "
        "UtmZ REAL, DigitalElevation REAL, RxTxDistance REAL, "
        "LineProfileCoordinate REAL, GlobalProfileCoordinate REAL, "
        "ProjectId INTEGER, LineId INTEGER, dBdt TEXT, Std TEXT, "
        "Filtered TEXT)")
    for station_id, moment, values, std, filtered, current in (
        (1, 0, LM_VALUES, LM_STD, LM_FILTERED, 1.0),
        (2, 1, HM_VALUES, HM_STD, HM_FILTERED, 10.0),
    ):
        con.execute(
            "INSERT INTO StationStacks VALUES (?, 1, 1, 1, ?, ?, "
            "'2026-09-06T17:01:01.099Z', 41.665, -91.585, 202.1, 617769.8, "
            "4613550.9, 202.1, NULL, 6.16, 1.88, 3.45, 1, 1, ?, ?, ?)",
            (station_id, moment, current, json.dumps(values),
             json.dumps(std), json.dumps(filtered)))
    con.execute(
        "CREATE TABLE InversionRunTable (ProjectId INTEGER, "
        "InversionRunId INTEGER PRIMARY KEY, InversionRunCreateAt TEXT, "
        "InversionRunFinishedAt TEXT, InversionRunName TEXT, "
        "InversionInput BLOB, InversionOutput BLOB, InversionLog BLOB, "
        "SortOrder INTEGER)")
    con.execute(
        "INSERT INTO InversionRunTable VALUES (1, 1, "
        "'2026-09-06T23:47:09Z', '2026-09-06T23:47:15Z', 'LCI_project', "
        "?, NULL, ?, 0)",
        (_lupus_input(), b"Total run time : 5.3 s"))
    con.execute(
        "CREATE TABLE StationModelTable (ModelId INTEGER PRIMARY KEY, "
        "ProjectId INTEGER, InversionId INTEGER, LineId INTEGER, "
        "StationNumber INTEGER, LineNumber INTEGER, UtmX REAL, UtmY REAL, "
        "UtmZ REAL, DigitalElevation REAL, UsedStations TEXT, "
        "LineProfileCoordinate REAL, GlobalProfileCoordinate REAL, "
        "Resistivities TEXT, Thicknesses TEXT, Depths TEXT, DOI REAL, "
        "DataFit REAL, Time TEXT, ModelData TEXT)")
    con.execute(
        "INSERT INTO StationModelTable VALUES (1, 1, 1, 1, 1, 1, 617769.8, "
        "4613550.9, 202.1, NULL, ?, 1.88, 3.45, ?, ?, ?, 50.0, 5.63, "
        "'2026-09-06T17:01:01.099', ?)",
        (json.dumps([1, 2]), json.dumps([5.0, 16.0, 20.0, 17.0]),
         json.dumps([1.0, 1.16, 1.35]), json.dumps([1.0, 2.16, 3.51]),
         _model_data()))
    con.commit()
    con.close()
    return path


def write_legacy_project(folder: Path) -> Path:
    """The same station in the layout earlier TEMImage releases wrote."""
    path = folder / "project.db"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE RxTxSpecs (RxTxSpecsId INTEGER PRIMARY KEY, "
                "RxTxSpecsJson TEXT)")
    spec = {
        "LM_GateOpenTime": LM_OPEN,
        "LM_GateCentreTime": _centres(LM_OPEN, LM_CLOSE),
        "LM_GateCloseTime": LM_CLOSE,
        "HM_GateOpenTime": HM_OPEN,
        "HM_GateCentreTime": _centres(HM_OPEN, HM_CLOSE),
        "HM_GateCloseTime": HM_CLOSE,
        "LMWaveformTime": WAVEFORM_TIME, "HMWaveformTime": WAVEFORM_TIME,
        "LMWaveformAmplitude": WAVEFORM_AMPLITUDE,
        "HMWaveformAmplitude": WAVEFORM_AMPLITUDE,
        "LMWaveformPeriod": 4.0e-4, "HMWaveformPeriod": 8.0e-4,
        "LM_GateTimeShift": 0.0, "HM_GateTimeShift": 0.0,
        "LM_DataFactor": 1.05, "HM_DataFactor": 1.08,
        "LM_Tx_TargetCurrent": 1.0, "HM_Tx_TargetCurrent": 10.0,
        "GateShape": 1, "GateShapePar1": 0.667,
        "LPFilter_1order": [450000.0, 800000.0],
        "LowPassFilterRxCoil": [], "LowPassFilterRxInst": [],
        "TxLoopArea": 0.3969, "TxLoopXYlength": [0.63, 0.63],
        "TxLoopXYZPos": [0.0, 0.0, 1.0], "RxCoilXYZPos": [-15.0, 0.0, 1.0],
        "NTurnsTxLoop": 4, "InstrumentType": "TEM2Go",
        "UTMZone": 15, "UTMZoneLetter": "T",
    }
    con.execute("INSERT INTO RxTxSpecs VALUES (1, ?)", (json.dumps(spec),))
    con.execute(
        "CREATE TABLE StationStackData (AveragedDataId INTEGER PRIMARY KEY, "
        "LineNumber INTEGER, StationId TEXT, UtmX REAL, UtmY REAL, UtmZ REAL, "
        "Latitude REAL, Longitude REAL, Elevation REAL, Timestamp TEXT, "
        "LM_VoltageValues TEXT, HM_VoltageValues TEXT, "
        "LM_VoltageValues_STD TEXT, HM_VoltageValues_STD TEXT, "
        "LM_InUseFlags TEXT, HM_InUseFlags TEXT, ProfileDistance REAL, "
        "TotalSurveyDistance REAL, RxTxDistance REAL, RxTxSpecsId INTEGER)")
    con.execute(
        "INSERT INTO StationStackData VALUES (1, 1, '1_00001', 617769.8, "
        "4613550.9, 202.1, 41.665, -91.585, 202.1, "
        "'2026-09-06T17:01:01.099Z', ?, ?, ?, ?, ?, ?, 1.88, 3.45, 6.16, 1)",
        (json.dumps(LM_VALUES), json.dumps(HM_VALUES), json.dumps(LM_STD),
         json.dumps(HM_STD), json.dumps([1, 1, 1, 1]),
         json.dumps([1, 1, 1, 0])))
    con.commit()
    con.close()
    return path


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    folder = tmp_path / "workspace"
    folder.mkdir()
    write_workspace_project(folder)
    return folder


@pytest.fixture()
def legacy(tmp_path: Path) -> Path:
    folder = tmp_path / "legacy"
    folder.mkdir()
    write_legacy_project(folder)
    return folder


# ---------------------------------------------------------------------------
# Which file a folder is read from
# ---------------------------------------------------------------------------
def test_the_workspace_file_wins_when_both_are_present(tmp_path: Path) -> None:
    """A migrated project keeps a stale ``project.db`` beside the new file.

    On the Ashton Prairie project the leftover ``project.db`` carries the
    transients but no inversion rows at all, so reading it would report an
    inverted survey as never inverted.
    """
    folder = tmp_path / "both"
    folder.mkdir()
    write_legacy_project(folder)
    write_workspace_project(folder)

    assert tcp.project_file(folder).name == "project.tiw"


def test_a_folder_with_no_project_file_says_so(tmp_path: Path) -> None:
    """A raw acquisition folder looks like a project from outside."""
    folder = tmp_path / "raw"
    folder.mkdir()
    (folder / "Protocol_TEM2Go.sts").write_text("UniStd = 0.03\n")

    with pytest.raises(ValueError, match="no project file"):
        tcp.project_file(folder)


def test_the_layout_is_read_off_the_tables_not_the_suffix(
        workspace: Path, legacy: Path) -> None:
    for folder, expected in ((workspace, tcp.WORKSPACE_SCHEMA),
                             (legacy, tcp.LEGACY_SCHEMA)):
        con = tcp.open_project(folder)
        try:
            assert tcp.schema_of(con) == expected
            assert tcp.missing_survey_tables(con) == []
        finally:
            con.close()


def test_an_unrelated_sqlite_file_is_not_a_project(tmp_path: Path) -> None:
    """``.db`` is a common name for any SQLite file."""
    from PyHydroGeophysX.data_processing.em1d import is_temcompany_source

    path = tmp_path / "notes.db"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE notes (body TEXT)")
    con.commit()
    con.close()

    assert not is_temcompany_source(str(path))


def test_a_workspace_folder_is_recognised(workspace: Path) -> None:
    from PyHydroGeophysX.data_processing.em1d import is_temcompany_source

    assert is_temcompany_source(str(workspace))
    assert is_temcompany_source(str(workspace / "project.tiw"))


# ---------------------------------------------------------------------------
# The translation itself
# ---------------------------------------------------------------------------
def test_both_layouts_give_the_same_acquisition_spec(
        workspace: Path, legacy: Path) -> None:
    """Every key the forward and the gate selection read, from either file.

    ``RxCoilAreaChA`` is the one field the newer layout drops, and it is
    descriptive only, so it is excluded here rather than filled in.
    """
    con = tcp.open_project(workspace)
    try:
        new = next(iter(tcp.read_specs(con).values()))
    finally:
        con.close()
    con = tcp.open_project(legacy)
    try:
        old = next(iter(tcp.read_specs(con).values()))
    finally:
        con.close()

    for key in sorted(old):
        assert key in new, f"{key} missing from the workspace spec"
        if isinstance(old[key], str):
            assert new[key] == old[key], key
        else:
            np.testing.assert_allclose(
                np.asarray(new[key], dtype=float),
                np.asarray(old[key], dtype=float), rtol=1e-6, err_msg=key)
    assert new["InstrumentType"] == "TEM2Go"
    assert new["UTMZone"] == 15 and new["UTMZoneLetter"] == "T"


def test_the_packed_geometry_decodes(workspace: Path) -> None:
    """``ProtocolTable`` stores the loop and coil geometry as raw doubles."""
    con = tcp.open_project(workspace)
    try:
        spec = next(iter(tcp.read_specs(con).values()))
    finally:
        con.close()

    assert spec["RxCoilXYZPos"] == [-15.0, 0.0, 1.0]
    assert spec["TxLoopXYZPos"] == [0.0, 0.0, 1.0]
    assert spec["LPFilter_1order"] == [450000.0, 800000.0]
    np.testing.assert_allclose(spec["TxLoopXYlength"], [0.63, 0.63])


def test_the_two_moment_rows_pivot_back_into_one_station(
        workspace: Path, legacy: Path) -> None:
    """``StationStacks`` splits a station in two; the readers want it whole."""
    con = tcp.open_project(workspace)
    try:
        new = tcp.read_station_stacks(con)
    finally:
        con.close()
    con = tcp.open_project(legacy)
    try:
        old = tcp.read_station_stacks(con)
    finally:
        con.close()

    assert len(new) == len(old) == 1
    for key in ("StationId", "LineNumber", "UtmX", "UtmY", "RxTxDistance",
                "RxTxSpecsId"):
        assert new[0][key] == old[0][key], key
    for moment in ("LM", "HM"):
        for column in ("_VoltageValues", "_VoltageValues_STD", "_InUseFlags"):
            np.testing.assert_allclose(
                json.loads(new[0][moment + column]),
                json.loads(old[0][moment + column]),
                rtol=1e-6, err_msg=moment + column)


def test_the_filtered_bitmask_becomes_the_older_in_use_flags(
        workspace: Path) -> None:
    """``Filtered`` is 0 for a kept gate; ``InUseFlags`` was 1 for one.

    The reason a gate was dropped is new information the older array could not
    carry, so it is kept beside the flags rather than discarded.
    """
    con = tcp.open_project(workspace)
    try:
        row = tcp.read_station_stacks(con)[0]
    finally:
        con.close()

    assert json.loads(row["HM_InUseFlags"]) == [1, 1, 1, 0]
    assert json.loads(row["LM_InUseFlags"]) == [1, 1, 1, 1]
    assert json.loads(row["HM_FilterReason"]) == HM_FILTERED


def test_a_short_filter_array_keeps_the_remaining_gates() -> None:
    """A truncated mask must not silently switch off the gates past its end."""
    assert tcp._use_flags_from_filtered([0, 256], 4) == [1, 0, 1, 1]
    assert tcp._use_flags_from_filtered([], 3) == [1, 1, 1]


def test_the_gate_centre_is_the_geometric_mean_of_open_and_close(
        workspace: Path) -> None:
    """``ModelData`` drops the centre, so it has to be recomputed.

    Against the centres ``ProtocolTable`` stores for the same gates, the
    geometric mean agrees to 7e-8 relative on both moments of a real project.
    """
    con = tcp.open_project(workspace)
    try:
        spec = next(iter(tcp.read_specs(con).values()))
        models = tcp.read_inversion_models(con)
    finally:
        con.close()

    datasets = {entry["MomentType"]: entry
                for entry in json.loads(models[0]["Datasets"])}
    np.testing.assert_allclose(
        datasets["LM"]["Time_Centre"], spec["LM_GateCentreTime"], rtol=1e-7)
    np.testing.assert_allclose(
        datasets["HM"]["Time_Centre"], spec["HM_GateCentreTime"][:3], rtol=1e-7)


def test_the_model_arrays_are_renamed_not_reordered(workspace: Path) -> None:
    con = tcp.open_project(workspace)
    try:
        models = tcp.read_inversion_models(con)
    finally:
        con.close()

    assert len(models) == 1
    model = models[0]
    assert model["InversionName"] == "LCI_project"
    assert model["DOI"] == 50.0 and model["DataFit"] == 5.63
    # ``UsedStations`` names the two StationStacks rows the model was fitted to,
    # and the smaller of them is the id the older layout carried.
    assert model["AverageDataID"] == 1

    datasets = {entry["MomentType"]: entry
                for entry in json.loads(model["Datasets"])}
    np.testing.assert_allclose(datasets["LM"]["InputData"], LM_VALUES)
    np.testing.assert_allclose(datasets["LM"]["InputSTD"], LM_STD)
    np.testing.assert_allclose(datasets["HM"]["InputData"], HM_VALUES[:3])
    np.testing.assert_allclose(
        datasets["HM"]["ForwardData"], [v * 0.99 for v in HM_VALUES[:3]])


# ---------------------------------------------------------------------------
# The solver's own record
# ---------------------------------------------------------------------------
def test_the_settings_come_from_the_solver_input(workspace: Path) -> None:
    """The newer file stores what the run was given, not what a panel showed.

    Under the older layout a setting reading "Auto" left the stored number inert
    and a default applied instead, so one project recorded a 10 m reference
    distance for a run made at 50 m. Here the layer grid, the starting model and
    both constraint weights are read off the input the solver received.
    """
    con = tcp.open_project(workspace)
    try:
        defaults = tcp.read_inversion_defaults(con)
    finally:
        con.close()

    assert defaults["n_layers"] == 4
    np.testing.assert_allclose(defaults["layer_thicknesses"], [1.0, 1.16, 1.35])
    assert defaults["starting_resistivity"] == 22.0
    assert defaults["smoothness"] == 2.0
    assert defaults["lateral_smoothness"] == 1.3
    assert defaults["constraint"] == "LCI"
    assert defaults["reference_distance"] == 20.0
    assert defaults["lateral_distance_power"] == 0.75
    assert defaults["max_iterations"] == 30


def test_a_legacy_project_has_no_solver_record(legacy: Path) -> None:
    """Which leaves the older settings path in place rather than emptying it."""
    con = tcp.open_project(legacy)
    try:
        assert tcp.read_run_record(con) is None
        assert tcp.read_inversion_defaults(con) == {}
    finally:
        con.close()


def test_the_run_record_carries_the_input_output_and_log(
        workspace: Path) -> None:
    con = tcp.open_project(workspace)
    try:
        record = tcp.read_run_record(con)
    finally:
        con.close()

    assert record is not None
    assert record["name"] == "LCI_project"
    assert record["input"]["N_Layer"] == 4
    assert record["output"] is None
    assert "Total run time" in record["log"]


# ---------------------------------------------------------------------------
# Coordinate metadata
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("latitude,band", [
    (41.665, "T"),   # Iowa, and the Ashton Prairie project records 15T
    (39.0, "S"),     # Colorado, and the Trail Creek project records 13S
    (-33.9, "H"),
    (100.0, ""),     # outside the banded range
])
def test_the_zone_letter_is_a_latitude_band(latitude, band) -> None:
    """It is not a hemisphere, which is what an EPSG code alone would give."""
    assert tcp._mgrs_band(latitude) == band


@pytest.mark.parametrize("epsg,zone,northern", [
    (32615, 15, True),
    (32613, 13, True),
    (32733, 33, False),
    (4326, None, True),
    (None, None, True),
])
def test_the_utm_zone_comes_out_of_the_epsg_code(epsg, zone, northern) -> None:
    assert tcp._utm_from_epsg(epsg) == (zone, northern)


def test_a_zero_uniform_error_is_reported_as_absent(workspace: Path) -> None:
    """The migrated project writes 0.0 where its protocol states 0.03.

    A zero therefore means the importer did not carry the value across, so the
    ``.sts`` beside the project stays the authority.
    """
    con = tcp.open_project(workspace)
    try:
        assert tcp.protocol_uniform_error(con) is None
    finally:
        con.close()


# ---------------------------------------------------------------------------
# End to end through the readers
# ---------------------------------------------------------------------------
def test_a_sounding_reads_the_same_out_of_either_layout(
        workspace: Path, legacy: Path) -> None:
    """The gate selection, the geometry and the values, all through em1d."""
    from PyHydroGeophysX.data_processing.em1d import load_sounding

    new = load_sounding(str(workspace), "TDEM", moment="LM+HM",
                        max_relative_std=None, reject_negative=False)
    old = load_sounding(str(legacy), "TDEM", moment="LM+HM",
                        max_relative_std=None, reject_negative=False)

    assert sorted(new["moments"]) == sorted(old["moments"]) == ["HM", "LM"]
    for name in ("LM", "HM"):
        np.testing.assert_allclose(new["moments"][name]["times"],
                                   old["moments"][name]["times"], rtol=1e-7)
        np.testing.assert_allclose(new["moments"][name]["response"],
                                   old["moments"][name]["response"], rtol=1e-6)
    # The high moment's fourth gate is switched off in both files.
    assert new["moments"]["HM"]["times"].size == 3
    for key in ("tx_rx_sep", "loop_area", "loop_turns", "height"):
        assert new["system"][key] == old["system"][key], key


def test_the_stored_inversion_reads_out_of_the_workspace_layout(
        workspace: Path) -> None:
    from PyHydroGeophysX.data_processing import temcompany_reference as reference

    assert reference.has_reference_models(workspace)
    assert reference.reference_inversion_names(workspace) == ["LCI_project"]

    loaded = reference.load_reference_models(workspace)
    assert loaded["n_stations"] == 1
    station = loaded["stations"][0]
    assert station["doi"] == 50.0
    assert station["elevation"] == 202.1
    np.testing.assert_allclose(station["thickness"], [1.0, 1.16, 1.35])

    high = station["moments"]["HM"]
    # Three gates were used out of four, and the reader has to say which three.
    np.testing.assert_array_equal(high["spec_gate_indices"], [0, 1, 2])
    np.testing.assert_allclose(high["observed"], HM_VALUES[:3])


# ---------------------------------------------------------------------------
# Reading the survey once per file
# ---------------------------------------------------------------------------
def test_the_gate_arrays_travel_as_the_text_the_file_holds(
        workspace: Path) -> None:
    """No decode-and-re-encode round trip on the way to the reader.

    The pivot has to build the in-use flags, which means decoding ``Filtered``.
    It does not have to touch the measured arrays, and decoding them to floats
    only to serialise them again for the reader to decode was 5 s of an 89 s
    line inversion.
    """
    con = tcp.open_project(workspace)
    try:
        row = tcp.read_station_stacks(con)[0]
        stored = {
            r["TransmitterMoment"]: r for r in
            con.execute("SELECT TransmitterMoment, dBdt, Std FROM StationStacks")
        }
    finally:
        con.close()

    for moment, index in (("LM", 0), ("HM", 1)):
        assert row[moment + "_VoltageValues"] == stored[index]["dBdt"]
        assert row[moment + "_VoltageValues_STD"] == stored[index]["Std"]


@pytest.mark.parametrize("text,length", [
    ("[]", 0), ("[1]", 1), ("[1,2,3]", 3), ("  [ 1 , 2 ]  ", 2),
    ("", 0), (None, 0),
])
def test_a_flat_json_array_is_counted_without_decoding(text, length) -> None:
    assert tcp._json_array_length(text) == length


def test_a_survey_is_parsed_once_per_file(workspace: Path) -> None:
    """A line inversion asks for the same stations once per station."""
    tcp.clear_survey_cache()
    first = tcp.read_survey(workspace)
    second = tcp.read_survey(workspace)
    assert first is second
    assert first[1] is second[1]


def test_a_rewritten_project_is_read_again(tmp_path: Path) -> None:
    """The cache key is name, modification time and size, so a rewrite shows."""
    folder = tmp_path / "rewritten"
    folder.mkdir()
    write_workspace_project(folder)
    tcp.clear_survey_cache()
    before = tcp.read_survey(folder)

    path = folder / "project.tiw"
    con = sqlite3.connect(path)
    con.execute("UPDATE StationStacks SET UtmX = 999.0")
    con.commit()
    con.close()
    # A rewrite inside the filesystem's timestamp resolution would be missed,
    # which is the documented limit; nudge the stamp so the test is about the
    # cache rather than about the clock.
    os.utime(path, (0, 0))

    after = tcp.read_survey(folder)
    assert after is not before
    assert after[1][0]["UtmX"] == 999.0


def test_clearing_the_caches_reaches_both_layers(workspace: Path) -> None:
    """``em1d`` caches the saved settings and the gate selection on top."""
    from PyHydroGeophysX.data_processing import em1d

    em1d.clear_temcompany_caches()
    em1d.load_sounding(str(workspace), "TDEM", moment="LM+HM",
                       max_relative_std=None, reject_negative=False)
    assert tcp._SURVEY_CACHE
    assert em1d._TEMCOMPANY_SELECTION_CACHE

    em1d.clear_temcompany_caches()
    assert not tcp._SURVEY_CACHE
    assert not em1d._TEMCOMPANY_SELECTION_CACHE
    assert not em1d._TEMCOMPANY_DEFAULTS_CACHE


def test_the_gate_selection_is_cached_per_setting_not_per_station(
        workspace: Path) -> None:
    """Two settings must not share one answer, however often either is asked."""
    from PyHydroGeophysX.data_processing import em1d

    em1d.clear_temcompany_caches()
    loose = em1d.load_sounding(str(workspace), "TDEM", moment="LM+HM",
                               max_relative_std=None, reject_negative=False)
    strict = em1d.load_sounding(str(workspace), "TDEM", moment="LM+HM",
                                max_relative_std=0.031,
                                gate_rejection="individual",
                                reject_negative=False)
    assert len(em1d._TEMCOMPANY_SELECTION_CACHE) == 2
    assert (strict["moments"]["LM"]["times"].size
            < loose["moments"]["LM"]["times"].size)


def test_a_project_file_can_be_named_directly(tmp_path: Path) -> None:
    """A reprocessing sits beside the original under its own name.

    ``project2.tiw`` next to ``project.tiw`` is what a second pass through the
    same survey produces, and the folder lookup takes the standard name, so the
    only way to reach the second one is to name it.
    """
    from PyHydroGeophysX.data_processing import em1d

    folder = tmp_path / "reprocessed"
    folder.mkdir()
    write_workspace_project(folder)
    second = folder / "project2.tiw"
    second.write_bytes((folder / "project.tiw").read_bytes())

    for moment in ("LM+HM", "HM"):
        loaded = em1d.load_sounding(str(second), "TDEM", moment=moment,
                                    max_relative_std=None,
                                    reject_negative=False)
        assert loaded["n_soundings"] == 1
    assert em1d.is_temcompany_source(str(second))


# ---------------------------------------------------------------------------
# Choosing between several projects in one folder
# ---------------------------------------------------------------------------
def test_every_project_in_a_folder_is_listed(tmp_path: Path) -> None:
    """A folder holds more than one more often than not.

    A migrated project keeps its ``project.db``, and a reprocessing of the same
    survey lands beside it under its own name.
    """
    folder = tmp_path / "several"
    folder.mkdir()
    write_workspace_project(folder)
    write_legacy_project(folder)
    (folder / "project2.tiw").write_bytes((folder / "project.tiw").read_bytes())

    names = [item.name for item in tcp.project_files(folder)]
    assert names == ["project.tiw", "project.db", "project2.tiw"]
    # The reader's own rule is unchanged, so a caller that cannot ask is not
    # silently moved onto a different survey.
    assert tcp.project_file(folder).name == "project.tiw"


def test_a_database_that_is_not_a_project_is_not_offered(tmp_path: Path) -> None:
    """One survey folder keeps an acquisition database beside its project."""
    folder = tmp_path / "mixed"
    folder.mkdir()
    write_workspace_project(folder)
    other = sqlite3.connect(folder / "notes.db")
    other.execute("CREATE TABLE notes (body TEXT)")
    other.commit()
    other.close()

    assert [item.name for item in tcp.project_files(folder)] == [
        "project.tiw", "notes.db"]
    assert [item["name"] for item in tcp.choosable_projects(folder)] == [
        "project.tiw"]
    assert "not a TEMcompany project" in tcp.describe_project(folder / "notes.db")


def test_a_project_is_described_by_what_it_holds(workspace: Path) -> None:
    """Two files in a folder differ in content, not in size."""
    summary = tcp.project_summary(workspace / "project.tiw")
    assert summary["is_project"] and summary["layout"] == tcp.WORKSPACE_SCHEMA
    assert summary["stations"] == 1
    assert summary["inversion"] == "LCI_project"

    label = tcp.describe_project(workspace / "project.tiw")
    assert "project.tiw" in label and "1 stations" in label
    assert "LCI_project" in label


def test_an_unreadable_file_is_described_rather_than_raising(
        tmp_path: Path) -> None:
    """A chooser builds a list; one bad entry must not empty it.

    SQLite opens a file lazily, so a folder holding something that is not a
    database at all surfaces at the first query rather than at the open, and
    reads as a file carrying no survey tables. Either way the entry describes
    itself and is left out of the offer.
    """
    broken = tmp_path / "broken.tiw"
    broken.write_bytes(b"not a database at all")

    label = tcp.describe_project(broken)
    assert broken.name in label
    assert "not a TEMcompany project" in label or "unreadable" in label
    assert tcp.choosable_projects(tmp_path) == []


# ---------------------------------------------------------------------------
# A folder with no project at all
# ---------------------------------------------------------------------------
def _xyz_numbers(values) -> str:
    return " ".join("%.6E" % float(v) for v in values)


def write_station_xyz(folder: Path, with_protocol: bool = True) -> Path:
    """A two-station ``*_StationData.xyz``, shaped like a real export.

    The comment block is the part that matters: it states both waveforms, every
    gate window and the receiver's filter corners, which is what lets an export
    reach the same forward a project does.
    """
    header = (["Project", "Date", "Time", "Line", "Station", "Latitude",
               "Longitude", "Elevation"]
              + ["LMgate%03d" % (i + 1) for i in range(len(LM_VALUES))]
              + ["HMgate%03d" % (i + 1) for i in range(len(HM_VALUES))]
              + ["LMstd%03d" % (i + 1) for i in range(len(LM_STD))]
              + ["HMstd%03d" % (i + 1) for i in range(len(HM_STD))])
    lines = [
        "/TEMImage Station Stacked by TEMcompany, version: 2026.1.6.0",
        "/Unit dB/dt: [V/(A*m4)]",
        "/Dummy dB/dt: 99999, stdF: 99999",
        "/[Geometry Info]",
        "/LoopX (m): 0.630",
        "/LoopY (m): 0.630",
        "/LoopZ (m): 1.000",
        "/LoopArea (m2): 0.397",
        "/LoopTurns : 4",
        "/RXcoil X-Position (m): -15.000",
        "/[Waveform Info]",
        "/LM_WaveformTime: " + _xyz_numbers(WAVEFORM_TIME),
        "/LM_WaveformAmplitude: " + _xyz_numbers(WAVEFORM_AMPLITUDE),
        "/HM_WaveformTime: " + _xyz_numbers(WAVEFORM_TIME),
        "/HM_WaveformAmplitude: " + _xyz_numbers(WAVEFORM_AMPLITUDE),
        "/[Gate Times]",
        "/LM_GateOpenTime: " + _xyz_numbers(LM_OPEN),
        "/LM_GateCloseTime: " + _xyz_numbers(LM_CLOSE),
        "/LM_GateCentreTime: " + _xyz_numbers(_centres(LM_OPEN, LM_CLOSE)),
        "/HM_GateOpenTime: " + _xyz_numbers(HM_OPEN),
        "/HM_GateCloseTime: " + _xyz_numbers(HM_CLOSE),
        "/HM_GateCentreTime: " + _xyz_numbers(_centres(HM_OPEN, HM_CLOSE)),
        "/[1st Order Filters]",
        "/Cutoff Frequencies: 450000,800000",
        "/" + " ".join(header),
    ]
    for station in (1, 2):
        row = ["AP", "20260906", "17.01.01.099", "001", "1_%05d" % station,
               "%.7f" % (41.665 + 1e-5 * station), "%.7f" % -91.585, "202.1"]
        row += ["%.6E" % v for v in LM_VALUES]
        row += ["%.6E" % v for v in HM_VALUES]
        row += ["%.6E" % v for v in LM_STD]
        row += ["%.6E" % v for v in HM_STD]
        lines.append(" ".join(row))

    path = folder / "AP_StationData.xyz"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if with_protocol:
        (folder / "Protocol_TEM2Go.sts").write_text(
            "UniStd = 0.03\nLM_PeriodTime = 400\nHM_PeriodTime = 800\n"
            "ChA_WinFunc = 1\nChA_WinFuncPar1 = 0.667\n", encoding="utf-8")
    return path


@pytest.fixture()
def export_only(tmp_path: Path) -> Path:
    folder = tmp_path / "exported"
    folder.mkdir()
    write_station_xyz(folder)
    return folder


def test_a_folder_with_no_project_reads_its_export(export_only: Path) -> None:
    """Not every survey has been imported, and an export is enough to invert."""
    from PyHydroGeophysX.data_processing import em1d

    assert tcp.project_files(export_only) == []
    assert em1d.is_temcompany_source(str(export_only))

    loaded = em1d.load_sounding(str(export_only), "TDEM", moment="LM+HM",
                                max_relative_std=None, reject_negative=False)
    assert loaded["n_soundings"] == 2
    assert sorted(loaded["moments"]) == ["HM", "LM"]
    assert loaded["source_format"] == "TEMcompany station XYZ"
    # An export writes seven significant digits, so it is compared at the
    # precision it stores rather than at double precision.
    np.testing.assert_allclose(loaded["moments"]["LM"]["response"],
                               LM_VALUES, rtol=1e-6)
    np.testing.assert_allclose(loaded["moments"]["HM"]["response"],
                               HM_VALUES, rtol=1e-6)


def test_an_export_carries_the_whole_instrument(export_only: Path) -> None:
    """The comment block states the waveform, the gates and the filter.

    Reading only the gate centres, which is what this used to do, models a bare
    step-off: the analog filter alone is 16 to 28 percent of the low moment's
    early amplitude.
    """
    from PyHydroGeophysX.data_processing import em1d

    loaded = em1d.load_sounding(str(export_only), "TDEM", moment="LM+HM",
                                max_relative_std=None, reject_negative=False)
    for name, opens, closes in (("LM", LM_OPEN, LM_CLOSE),
                                ("HM", HM_OPEN, HM_CLOSE)):
        transmitter = loaded["moments"][name]["transmitter"]
        np.testing.assert_allclose(transmitter["waveform_times"],
                                   WAVEFORM_TIME, rtol=1e-6)
        np.testing.assert_allclose(transmitter["waveform_currents"],
                                   WAVEFORM_AMPLITUDE, rtol=1e-6)
        windows = transmitter["gate_windows"]
        np.testing.assert_allclose(windows["open"], opens, rtol=1e-6)
        np.testing.assert_allclose(windows["close"], closes, rtol=1e-6)
        assert transmitter["analog_lowpass"] == {
            "first_order_cutoffs_hz": (450000.0, 800000.0)}
        # Only the protocol beside the export states these two.
        assert transmitter["gate_window_shape"] == 1.0
        assert transmitter["gate_window_par"] == 0.667
    assert loaded["moments"]["LM"]["transmitter"]["waveform_period"] == pytest.approx(4.0e-4)
    assert loaded["moments"]["HM"]["transmitter"]["waveform_period"] == pytest.approx(8.0e-4)


def test_an_export_without_its_protocol_still_reads(tmp_path: Path) -> None:
    """With the repetition correction and the gate window absent, and said so."""
    from PyHydroGeophysX.data_processing import em1d

    folder = tmp_path / "bare"
    folder.mkdir()
    write_station_xyz(folder, with_protocol=False)

    loaded = em1d.load_sounding(str(folder), "TDEM", moment="LM+HM",
                                max_relative_std=None, reject_negative=False)
    transmitter = loaded["moments"]["HM"]["transmitter"]
    assert transmitter["waveform_period"] is None
    assert transmitter["gate_window_shape"] is None
    assert loaded["forward_metadata"]["gate_window_shape"] is None


def test_an_acquisition_folder_without_a_raw_stream_says_what_is_missing(
        tmp_path: Path) -> None:
    """A protocol and nothing readable beside it: name what is absent.

    The raw stream is what makes such a folder readable, so a folder holding a
    protocol without one has to say that rather than fall through to the
    generic table reader, which would complain about the extension instead.
    """
    from PyHydroGeophysX.data_processing import em1d

    folder = tmp_path / "raw"
    folder.mkdir()
    (folder / "Protocol_TEM2Go.sts").write_text("UniStd = 0.03\n")
    (folder / "StationStacks.stk").write_bytes(b"\x00\x01\x02")

    with pytest.raises(ValueError, match="no .stb raw stream"):
        em1d.load_sounding(str(folder), "TDEM", moment="LM+HM")
