"""Tests for reading a TEM2Go acquisition folder straight from its raw stream.

The fixtures build a raw file byte by byte from the format specification, so a
change to the decode shows up as a failure here rather than as a wrong sounding
much later. The layout under test is the one the format note records: a
length-prefixed version string, the protocol as text, then one record per raw
transient with fixed telemetry offsets, a self-describing auxiliary block, and
the two measured arrays.

The last two tests run only where a real acquisition folder is present. They
check the claims that matter for a reader, that the decoded records and the
generated gate table agree with what the survey's own project holds.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import struct
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from PyHydroGeophysX.data_processing import temcompany_stb as stb

EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)

PROTOCOL = """;[ProtocolDesigner] SampleFrequencyMHz = 4.0, MinimumGateTimeLM = 4.5, MaximumGateTimeLM = 44.0, MinimumGateTimeHM = 10.0, MaximumGateTimeHM = 600.0, NumberOfGatesPerDecade = 800, UseFrontGateLM = False, UseFrontGateHM = True
;
[Timing]
LM_StackSize = 500
LM_OnTime = 200
LM_PeriodTime = 400
HM_StackSize = 1000
HM_OnTime = 300
HM_PeriodTime = 800

[Gate]
LM_ChA_GateNum = 4
LM_ChA_GateIndex0 = 8, 819, 820, 821
LM_ChA_GateIndex1 = 209, 820, 821, 979
HM_ChA_GateNum = 5
HM_ChA_GateIndex0 = 8, 160, 1241, 1242, 1243
HM_ChA_GateIndex1 = 159, 200, 1242, 1243, 3056

[Processing]
UniStd = 0.03

[Tapering]
LM_ChA_TapStartTime = 4.5
HM_ChA_TapStartTime = 10.0
ChA_GatePerDecade = 10
ChA_GateOverlap = 0.5
ChA_WinFunc = 1
ChA_WinFuncPar1 = 0.667

[RxTxSpecs]
RxCoil_XYZPos = -15.0, 0.0, 1.00
RxCoil_AreaChA = 441.0
TxLoop_XYZPos = 0.0, 0.0, 1.00
TxLoop_XYLength = 0.63, 0.63
TxLoop_NTurns = 4
LPFilter_1order = 450000.0, 800000.0
LM_Waveform_Time = -200.0, -193.79, -100.0, 0.0
LM_Waveform_Amplitude = 0.0, 0.2311, 1.0, 0.0
HM_Waveform_Time = -300.0, -200.0, -100.0, 0.0
HM_Waveform_Amplitude = 0.0, 0.5, 1.0, 0.0
LM_Tx_TargetCurrent = 1.0
HM_Tx_TargetCurrent = 10.0

[ProcessedGateTimes]
LM_ChA_OpenTime=4.4999995,5.658838,7.1160994
LM_ChA_CenterTime=5.6588373,7.1160994,8.948634
LM_ChA_CloseTime=7.1160994,8.948634,11.253082
HM_ChA_OpenTime=10,12.531894,15.704836
HM_ChA_CenterTime=12.531894,15.704836,19.681133
HM_ChA_CloseTime=15.704836,19.681133,24.664186

[InstrumentInfo]
LM_GateTimeShift=+000.00
LM_DataFactor=01.050
HM_GateTimeShift=+000.00
HM_DataFactor=01.080
InstrumentID=TEM2Go-0031
InstrumentType=TEM2Go
SampleRateInMHz=4
"""

# The stored table above fixes three gates a moment, which is what the record
# payloads below carry. The design parameters describe the instrument's full
# table, so the generated table is longer, and one test checks each.
GATES = {0: 3, 1: 3}


def build_record(*, moment: int, channel: int, gps_us: int, latitude: float,
                 longitude: float, current: float, stored: list,
                 relative_std: list, aux: dict, temperature: int = 21,
                 stack: int = 500, speed: float = 1.5,
                 altitude: float = 230.0) -> bytes:
    """One record laid out as the format specification describes it."""
    prefix = bytearray(191)

    def put(offset: int, fmt: str, value) -> None:
        # offset is relative to the marker, so -191 lands on prefix[0]
        struct.pack_into(fmt, prefix, offset + 191, value)

    put(-191, "<B", moment)
    put(-190, "<B", channel)
    put(-185, "<i", 1)
    put(-176, "<Q", gps_us)
    put(-168, "<f", speed)
    put(-158, "<d", latitude)
    put(-150, "<d", longitude)
    put(-142, "<f", altitude)

    # The auxiliary block opens with the ``CoilDistance`` entry, and that
    # entry's own length prefix and name are what marks the record, so the
    # marker is part of the block rather than something written before it.
    assert next(iter(aux)) == "CoilDistance", "CoilDistance marks the record"
    block = bytearray()
    for name, value in aux.items():
        raw = name.encode("ascii")
        block += struct.pack("<I", len(raw)) + raw + struct.pack("<d", value)

    tail = bytearray(44)
    struct.pack_into("<B", tail, 22, temperature)
    struct.pack_into("<f", tail, 36, current)
    struct.pack_into("<i", tail, 40, stack)
    for value in stored:
        tail += struct.pack("<f", value)
    for value in relative_std:
        tail += struct.pack("<f", value)
    return bytes(prefix) + bytes(block) + bytes(tail)


def build_stb(path: Path, records: list, protocol: str = PROTOCOL) -> Path:
    """Assemble a raw file from a protocol and a list of record payloads."""
    label = b"VERSION 4.5"
    body = bytearray(b"\x02\x04")
    body += struct.pack("<I", len(label)) + label
    body += protocol.encode("ascii") + b"\x00\x00\x00\x00"
    for record in records:
        body += record
    path.write_bytes(bytes(body))
    return path


@pytest.fixture()
def folder(tmp_path: Path) -> Path:
    """An acquisition folder with a protocol, a line file and a raw stream."""
    root = tmp_path / "Survey"
    (root / "Data" / "2026_0906").mkdir(parents=True)
    (root / "Protocol_TEM2Go.sts").write_text(PROTOCOL)
    start = datetime(2026, 9, 6, 17, 0, 0, tzinfo=timezone.utc)
    (root / "Survey.lin").write_text(
        "06-09-2026 17:00:00 0001 41.6650000 -91.5850000 ! Start 000140\n"
        "06-09-2026 17:30:00 0001 41.6659000 -91.5852000 ! End\n")
    records = []
    # Ten stations' worth of records, the two moments alternating, walking
    # north so the station grouping has travel to split on.
    for step in range(20):
        offset_m = step * 0.6
        latitude = 41.665 + offset_m / 111_320.0
        stamp = int((start - EPOCH).total_seconds() * 1e6) + step * 1_000_000
        for moment in (0, 1):
            current = 1.0 if moment == 0 else 10.0
            records.append(build_record(
                moment=moment, channel=0, gps_us=stamp, latitude=latitude,
                longitude=-91.585, current=current,
                stored=[1e-3, 5e-4, 2e-4], relative_std=[0.03, 0.04, 0.05],
                aux={"CoilDistance": 14.5},
                stack=500 if moment == 0 else 1000))
    build_stb(root / "Data" / "2026_0906" / "raw.stb", records)
    return root


# ------------------------------------------------------------------ protocol


def test_the_protocol_is_read_from_the_raw_file_itself(folder: Path) -> None:
    """A raw file carries its own protocol, so no side file is needed."""
    path = next(folder.rglob("*.stb"))
    protocol = stb.read_protocol(path)
    assert protocol["instrument_type"] == "TEM2Go"
    assert protocol["instrument_id"] == "TEM2Go-0031"
    assert protocol["sample_rate_hz"] == pytest.approx(4e6)
    assert protocol["tx_area"] == pytest.approx(0.63 * 0.63)
    assert protocol["tx_turns"] == pytest.approx(4.0)
    assert protocol["uniform_std"] == pytest.approx(0.03)
    assert protocol["LM_data_factor"] == pytest.approx(1.05)
    assert protocol["HM_data_factor"] == pytest.approx(1.08)
    assert protocol["LM_target_current"] == pytest.approx(1.0)
    assert protocol["HM_target_current"] == pytest.approx(10.0)
    assert protocol["nominal_rx_tx_distance"] == pytest.approx(15.0)


def test_a_standalone_sts_protocol_reads_the_same(folder: Path) -> None:
    """The side file and the embedded copy are the same text."""
    embedded = stb.read_protocol(next(folder.rglob("*.stb")))
    side = stb.read_protocol(folder / "Protocol_TEM2Go.sts")
    assert side["tx_area"] == pytest.approx(embedded["tx_area"])
    assert side["LM_data_factor"] == pytest.approx(embedded["LM_data_factor"])


def test_a_file_that_is_not_a_raw_stream_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "not-raw.stb"
    path.write_bytes(b"\x00" * 64)
    with pytest.raises(ValueError, match="VERSION"):
        stb.read_protocol(path)


# ---------------------------------------------------------------- gate table


def test_the_gate_table_is_generated_from_the_design_parameters(
        folder: Path) -> None:
    """Gate ``k`` opens on series point ``k`` and closes on ``k + 2``.

    The overlap of one half is what makes a gate straddle two steps, and the
    number of steps follows from the span of the acquisition window in decades
    times the gates per decade.
    """
    protocol = stb.read_protocol(folder / "Protocol_TEM2Go.sts")
    table = stb.generate_gate_table(protocol, "LM")
    assert table is not None
    # window is samples 819 to 979 at 4 MHz, anchored so 819 is 4.5 us
    assert table["open"][0] == pytest.approx(4.5e-6)
    assert table["close"][-1] == pytest.approx(44.5e-6)
    # centre is the geometric mean of the gate's own edges
    assert np.allclose(table["centre"],
                       np.sqrt(table["open"] * table["close"]), rtol=1e-12)
    # the series is geometric
    ratios = table["open"][1:] / table["open"][:-1]
    assert np.ptp(ratios) < 1e-9


def test_a_stored_gate_table_wins_over_a_generated_one(folder: Path) -> None:
    """A protocol carrying explicit gate times is believed rather than refitted."""
    text = PROTOCOL + (
        "\n[ProcessedGateTimes]\n"
        "LM_ChA_OpenTime=1.0,2.0,4.0\n"
        "LM_ChA_CenterTime=2.0,4.0,8.0\n"
        "LM_ChA_CloseTime=4.0,8.0,16.0\n")
    path = build_stb(folder / "Data" / "stored.stb", [], protocol=text)
    protocol = stb.read_protocol(path)
    table = stb.gate_table(protocol, "LM")
    assert np.allclose(table["centre"], [2e-6, 4e-6, 8e-6])


def test_a_gate_time_shift_moves_every_gate(folder: Path) -> None:
    text = PROTOCOL.replace("LM_GateTimeShift=+000.00",
                            "LM_GateTimeShift=+002.50")
    path = build_stb(folder / "Data" / "shifted.stb", [], protocol=text)
    plain = stb.gate_table(stb.read_protocol(folder / "Protocol_TEM2Go.sts"), "LM")
    shifted = stb.gate_table(stb.read_protocol(path), "LM")
    assert np.allclose(shifted["centre"] - plain["centre"], 2.5e-6)


# ------------------------------------------------------------------- records


def test_records_decode_to_telemetry_and_both_measured_arrays(
        folder: Path) -> None:
    records, protocol = stb.read_records(next(folder.rglob("*.stb")))
    assert len(records) == 40
    first = records[0]
    assert first.moment == 0 and first.moment_name == "LM"
    assert first.channel == 0
    assert first.tx_current == pytest.approx(1.0)
    assert first.temperature == pytest.approx(21.0)
    assert first.stack_size == 500
    assert first.aux == {"CoilDistance": pytest.approx(14.5)}
    assert first.rx_tx_distance == pytest.approx(14.5)
    assert first.relative_std == pytest.approx([0.03, 0.04, 0.05])
    # the stored voltages are normalised by the transmitter moment and scaled
    # by the moment's data factor
    expected = np.asarray([1e-3, 5e-4, 2e-4]) * 1.05 / (1.0 * 4 * 0.63 * 0.63)
    assert first.dbdt == pytest.approx(expected, rel=1e-6)


def test_the_stack_centre_is_when_the_position_belongs(folder: Path) -> None:
    """The reported time is the middle of the stacking window, not its start."""
    records, protocol = stb.read_records(next(folder.rglob("*.stb")))
    low = next(r for r in records if r.moment == 0)
    half = 0.5 * 500 * 400e-6
    assert low.utm_time - low.gps_time == timedelta(seconds=half)


def test_a_failed_range_reading_is_not_taken_as_a_distance(
        folder: Path) -> None:
    """The sentinel means "no measurement" and must not reach the forward."""
    record = build_record(
        moment=0, channel=0, gps_us=0, latitude=41.0, longitude=-91.0,
        current=1.0, stored=[1e-3, 5e-4, 2e-4],
        relative_std=[0.03, 0.04, 0.05],
        aux={"CoilDistance": stb.COIL_DISTANCE_SENTINEL})
    path = build_stb(folder / "Data" / "sentinel.stb", [record])
    records, _ = stb.read_records(path)
    assert math.isnan(records[0].rx_tx_distance)


def test_the_field_estimate_stands_in_for_a_failed_reading(
        folder: Path) -> None:
    record = build_record(
        moment=0, channel=0, gps_us=0, latitude=41.0, longitude=-91.0,
        current=1.0, stored=[1e-3, 5e-4, 2e-4],
        relative_std=[0.03, 0.04, 0.05],
        aux={"CoilDistance": stb.COIL_DISTANCE_SENTINEL,
             "CoilDistanceBField": 12.25})
    path = build_stb(folder / "Data" / "bfield.stb", [record])
    records, _ = stb.read_records(path)
    assert records[0].rx_tx_distance == pytest.approx(12.25)


def test_a_transmitter_that_is_off_normalises_by_one_amp(
        folder: Path) -> None:
    """Dividing by a current near zero would blow the voltages up."""
    record = build_record(
        moment=0, channel=0, gps_us=0, latitude=41.0, longitude=-91.0,
        current=0.0005, stored=[1e-3, 5e-4, 2e-4],
        relative_std=[0.03, 0.04, 0.05], aux={"CoilDistance": 14.5})
    path = build_stb(folder / "Data" / "off.stb", [record])
    records, _ = stb.read_records(path)
    expected = np.asarray([1e-3, 5e-4, 2e-4]) * 1.05 / (1.0 * 4 * 0.63 * 0.63)
    assert records[0].dbdt == pytest.approx(expected, rel=1e-6)


# --------------------------------------------------------------------- lines


def test_line_windows_come_from_the_line_file(folder: Path) -> None:
    windows = stb.read_line_windows(folder)
    assert len(windows) == 1
    assert windows[0]["line_number"] == 1
    assert windows[0]["start"] == datetime(2026, 9, 6, 17, 0, 0,
                                           tzinfo=timezone.utc)
    assert windows[0]["end"] == datetime(2026, 9, 6, 17, 30, 0,
                                         tzinfo=timezone.utc)


def test_a_line_left_open_stays_open(tmp_path: Path) -> None:
    """An interrupted acquisition writes a start with no end."""
    (tmp_path / "x.lin").write_text(
        "06-09-2026 17:00:00 0001 41.0 -91.0 ! Start 000140\n")
    windows = stb.read_line_windows(tmp_path)
    assert windows[0]["end"] is None


def test_records_outside_every_line_take_no_part(folder: Path) -> None:
    """The stream recorded between lines is not survey data."""
    records, protocol = stb.read_records(next(folder.rglob("*.stb")))
    windows = [{"line_number": 1,
                "start": datetime(2030, 1, 1, tzinfo=timezone.utc),
                "end": datetime(2030, 1, 2, tzinfo=timezone.utc)}]
    assert stb.assign_lines(records, windows) == 0
    assert all(r.line_number is None for r in records)
    stations = stb.stack_stations(records, protocol)
    assert stations == []


# ------------------------------------------------------------------ filters


def test_a_drive_current_outside_its_tolerance_condemns_the_record(
        folder: Path) -> None:
    """The tolerance is a percentage of the moment's own target current."""
    good = build_record(moment=1, channel=0, gps_us=0, latitude=41.0,
                        longitude=-91.0, current=10.5,
                        stored=[1e-3, 5e-4, 2e-4],
                        relative_std=[0.03, 0.04, 0.05],
                        aux={"CoilDistance": 14.5}, stack=1000)
    bad = build_record(moment=1, channel=0, gps_us=0, latitude=41.0,
                       longitude=-91.0, current=6.0,
                       stored=[1e-3, 5e-4, 2e-4],
                       relative_std=[0.03, 0.04, 0.05],
                       aux={"CoilDistance": 14.5}, stack=1000)
    path = build_stb(folder / "Data" / "current.stb", [good, bad])
    records, protocol = stb.read_records(path)
    settings = stb.ProcessingSettings(spike_enabled=False)
    stb.apply_raw_filters(records, protocol, settings)
    assert not (records[0].filtered & stb.FILTER_TX_CURRENT).any()
    # a whole-record test, so every gate carries it
    assert (records[1].filtered & stb.FILTER_TX_CURRENT).all()


def test_an_amplitude_outlier_is_flagged_and_its_neighbours_are_not(
        folder: Path) -> None:
    """The window is centred on the record under test, so one spike is local."""
    records = []
    for step in range(12):
        value = 1e-3 if step != 6 else 5e-2
        records.append(build_record(
            moment=0, channel=0, gps_us=step * 1_000_000, latitude=41.0,
            longitude=-91.0, current=1.0, stored=[value, 5e-4, 2e-4],
            relative_std=[0.03, 0.04, 0.05], aux={"CoilDistance": 14.5}))
    path = build_stb(folder / "Data" / "spike.stb", records)
    decoded, protocol = stb.read_records(path)
    stb.apply_raw_filters(decoded, protocol)
    flagged = [i for i, r in enumerate(decoded)
               if r.filtered[0] & stb.FILTER_SPIKE]
    assert flagged == [6]
    # the other two gates are flat, so nothing there is an outlier
    assert not any(r.filtered[1] & stb.FILTER_SPIKE for r in decoded)


def test_the_spike_test_can_be_switched_off(folder: Path) -> None:
    records = []
    for step in range(12):
        value = 1e-3 if step != 6 else 5e-2
        records.append(build_record(
            moment=0, channel=0, gps_us=step * 1_000_000, latitude=41.0,
            longitude=-91.0, current=1.0, stored=[value, 5e-4, 2e-4],
            relative_std=[0.03, 0.04, 0.05], aux={"CoilDistance": 14.5}))
    path = build_stb(folder / "Data" / "nospike.stb", records)
    decoded, protocol = stb.read_records(path)
    stb.apply_raw_filters(decoded, protocol,
                          stb.ProcessingSettings(spike_enabled=False))
    assert not any((r.filtered & stb.FILTER_SPIKE).any() for r in decoded)


def test_the_modified_z_score_uses_the_iglewicz_hoaglin_scaling() -> None:
    """A deviation of 3.5 modified-Z units is the boundary.

    The score is ``0.6745 * |x - median| / MAD``, so the deviation that trips a
    threshold of 3.5 is ``3.5 / 0.6745`` median absolute deviations. Pinning it
    here keeps the constant from drifting to the 1.4826 scaling, which is a
    consistent estimate of the standard deviation and a different test.
    """
    assert 3.5 / 0.6745 == pytest.approx(5.1890, abs=1e-4)


# ----------------------------------------------------------------- stacking


def test_a_station_averages_its_members_and_reports_how_many(
        folder: Path) -> None:
    records, protocol = stb.read_records(next(folder.rglob("*.stb")))
    stb.assign_lines(records, stb.read_line_windows(folder))
    stb.apply_raw_filters(records, protocol)
    stations = stb.stack_stations(records, protocol)
    assert stations
    assert {s["moment_name"] for s in stations} == {"LM", "HM"}
    for station in stations:
        assert station["n_members"] >= 2
        assert station["used_points"].max() <= station["n_members"]
        assert station["dbdt"].size == station["times"].size
        assert station["relative_std"].size == station["times"].size


def test_a_single_member_gate_takes_the_uniform_error_twice() -> None:
    """One member has no scatter to measure, so only the floor is left."""
    mean, error = stb._mean_and_stderr(np.asarray([2.0], dtype=np.float32), 0.03)
    assert mean == pytest.approx(2.0)
    assert error == pytest.approx(math.sqrt(2.0) * 0.03)


def test_an_empty_gate_returns_zero_rather_than_a_nan() -> None:
    mean, error = stb._mean_and_stderr(np.asarray([], dtype=np.float32), 0.03)
    assert (mean, error) == (0.0, 0.0)


def test_a_mean_at_the_noise_floor_is_dropped() -> None:
    mean, error = stb._mean_and_stderr(
        np.asarray([1e-30, -1e-30], dtype=np.float32), 0.03)
    assert (mean, error) == (0.0, 0.0)


def test_the_relative_error_combines_scatter_with_the_floor() -> None:
    values = np.asarray([1.0, 1.1, 0.9, 1.05], dtype=np.float32)
    mean, error = stb._mean_and_stderr(values, 0.03)
    data = values.astype(float)
    scatter = math.sqrt(float(np.sum((data - data.mean()) ** 2)) / data.size)
    expected = math.hypot(scatter / math.sqrt(data.size) / abs(data.mean()), 0.03)
    assert error == pytest.approx(expected)


def test_a_station_gate_with_too_few_members_is_flagged(folder: Path) -> None:
    """The count test is inclusive, so a lone survivor is flagged too."""
    dbdt = np.asarray([1e-6, 5e-7, 2e-7])
    std = np.asarray([0.03, 0.03, 0.03])
    used = np.asarray([4, 1, 0])
    times = np.asarray([5e-6, 1e-5, 2e-5])
    flags = stb._filter_station_gates(
        dbdt, std, used, times, "LM",
        stb.ProcessingSettings(std_enabled=False, slope_enabled=False))
    assert not flags[0] & stb.FILTER_TOO_FEW_POINTS
    assert flags[1] & stb.FILTER_TOO_FEW_POINTS
    assert flags[2] & stb.FILTER_TOO_FEW_POINTS


def test_a_noisy_late_gate_disables_the_rest_of_the_curve() -> None:
    """The error test latches: one bad gate condemns everything after it."""
    times = np.asarray([1e-5, 2e-5, 4e-5, 8e-5, 1.6e-4])
    dbdt = np.asarray([1e-6, 5e-7, 2e-7, 1e-7, 5e-8])
    std = np.asarray([0.03, 0.03, 0.9, 0.05, 0.05])
    used = np.full(5, 4)
    flags = stb._filter_station_gates(
        dbdt, std, used, times, "LM",
        stb.ProcessingSettings(slope_enabled=False))
    assert not flags[0] & stb.FILTER_STD
    assert not flags[1] & stb.FILTER_STD
    assert (flags[2:] & stb.FILTER_STD).all()


def test_a_break_in_the_log_log_slope_disables_the_rest(folder: Path) -> None:
    times = np.asarray([1e-5, 2e-5, 4e-5, 8e-5, 1.6e-4])
    # a decay that reverses, which is a slope change no ground response makes
    dbdt = np.asarray([1e-6, 5e-7, 2e-7, 5e-3, 1e-7])
    std = np.full(5, 0.03)
    used = np.full(5, 4)
    flags = stb._filter_station_gates(
        dbdt, std, used, times, "LM",
        stb.ProcessingSettings(std_enabled=False))
    assert (flags & stb.FILTER_SLOPE).any()
    assert flags[-1] & stb.FILTER_SLOPE


# -------------------------------------------------------------------- folder


def test_reading_a_folder_returns_every_stage(folder: Path) -> None:
    survey = stb.read_acquisition_folder(folder)
    assert survey["n_records"] == 40
    assert survey["n_records_on_a_line"] == 40
    assert survey["n_stations"] > 0
    assert len(survey["lines"]) == 1
    assert survey["records"] and survey["stations"]
    assert survey["protocol"]["instrument_type"] == "TEM2Go"


def test_a_folder_with_no_raw_stream_is_refused(tmp_path: Path) -> None:
    (tmp_path / "Protocol.sts").write_text(PROTOCOL)
    assert not stb.is_stb_folder(tmp_path)
    with pytest.raises(FileNotFoundError, match="no .stb"):
        stb.read_acquisition_folder(tmp_path)


def test_the_companion_streams_are_not_mistaken_for_the_record_stream(
        folder: Path) -> None:
    """Only the stream whose records are soundings should be read."""
    data = (folder / "Data" / "2026_0906")
    (data / "raw_Rx.stb").write_bytes(b"junk")
    (data / "raw_Tx.stb").write_bytes(b"junk")
    assert [p.name for p in stb.find_stb_files(folder)] == ["raw.stb"]


def test_the_legacy_spec_carries_what_the_forward_needs(folder: Path) -> None:
    """A raw read reaches the forward through the same names a project does."""
    protocol = stb.read_protocol(next(folder.rglob("*.stb")))
    spec = stb.legacy_spec(protocol)
    for moment in ("LM", "HM"):
        assert spec[f"{moment}_GateCentreTime"].size
        assert spec[f"{moment}_GateOpenTime"].size
        assert spec[f"{moment}_GateCloseTime"].size
        assert spec[f"{moment}WaveformTime"].size == \
            spec[f"{moment}WaveformAmplitude"].size
        assert spec[f"{moment}WaveformPeriod"] > 0
    assert spec["GateShape"] == pytest.approx(1.0)
    assert spec["GateShapePar1"] == pytest.approx(0.667)
    assert spec["TxLoopArea"] == pytest.approx(0.63 * 0.63)
    assert list(spec["LPFilter_1order"]) == [450000.0, 800000.0]


def test_a_leading_period_boundary_node_is_dropped_from_the_waveform(
        folder: Path) -> None:
    """A zero-current node half a period before turn-off marks the cycle.

    Carrying it into the forward would prepend a segment of zero current to the
    ramp. The low moment's protocol opens with such a node and the high
    moment's does not, so both cases are covered by one survey.
    """
    protocol = stb.read_protocol(folder / "Protocol_TEM2Go.sts")
    spec = stb.legacy_spec(protocol)
    # LM opens at -200 us with zero amplitude, which is -PeriodTime/2
    assert spec["LMWaveformTime"][0] == pytest.approx(-193.79e-6)
    assert spec["LMWaveformAmplitude"][0] == pytest.approx(0.2311)
    # HM opens at -300 us, inside its 800 us period, so it is a real node
    assert spec["HMWaveformTime"][0] == pytest.approx(-300e-6)


def test_the_package_entry_point_reads_a_folder_with_no_project(
        folder: Path) -> None:
    """The whole point: a folder off the instrument loads like any other source."""
    from PyHydroGeophysX.data_processing.em1d import (
        is_temcompany_source, load_temcompany_sounding)

    assert is_temcompany_source(str(folder))
    data = load_temcompany_sounding(str(folder), sounding=0, moment="LM+HM")
    assert data["temcompany"] is True
    assert data["source_format"] == "TEM2Go acquisition folder (.stb)"
    assert data["n_soundings"] > 0
    assert set(data["available_moments"]) <= {"LM", "HM"}
    assert data["uniform_error"] == pytest.approx(0.03)
    assert data["times"].size == data["response"].size
    assert data["system"]["source_radius"] > 0
    assert data["stb_survey"]["n_records"] == 40


# ------------------------------------------------- against a real acquisition

REAL = Path(os.environ.get(
    "TEMCOMPANY_PROJECT_DIR",
    Path.home() / "OneDrive - University of Iowa" / "Teaching" / "APLL data"
    / "TEM" / "Sep06"))
HAS_REAL = (REAL / "project2.tiw").is_file() and any(REAL.rglob("*.stb"))
real_only = pytest.mark.skipif(
    not HAS_REAL, reason="needs a real TEM2Go acquisition folder and its project")


@real_only
def test_every_record_column_reproduces_from_the_raw_file() -> None:
    """The decode is exact, so assert equality rather than a tolerance.

    ``dBdt`` and ``Std`` are stored as float32 in both the raw file and the
    project, so they are compared at that precision rather than at double.
    """
    path = next(REAL.rglob("*.stb"))
    records, protocol = stb.read_records(path)
    uri = ("file:///" + str(REAL / "project2.tiw").replace("\\", "/")
           .replace(" ", "%20") + "?mode=ro")
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            "SELECT TransmitterMoment, TxCurrent, dBdt, Std, GpsLatitude, "
            "GpsLongitude FROM Data ORDER BY RawDataId").fetchall()
    finally:
        connection.close()

    assert len(records) == len(rows)
    for record, row in zip(records, rows):
        assert record.moment == row["TransmitterMoment"]
        assert record.tx_current == pytest.approx(row["TxCurrent"], rel=2e-6)
        assert record.latitude == pytest.approx(row["GpsLatitude"], abs=1e-12)
        assert record.longitude == pytest.approx(row["GpsLongitude"], abs=1e-12)
        assert record.dbdt == pytest.approx(
            np.asarray(json.loads(row["dBdt"]), dtype=float), rel=2e-6)
        assert record.relative_std == pytest.approx(
            np.asarray(json.loads(row["Std"]), dtype=float), rel=2e-6)


@real_only
def test_the_generated_gate_table_matches_the_one_the_project_holds() -> None:
    """Both the stored table and the regenerated one are checked.

    The stored table is exact. The regenerated one agrees to about one part in
    ten million, which is the precision the project keeps its own copy at.
    """
    protocol = stb.read_protocol(next(REAL.rglob("*.stb")))
    uri = ("file:///" + str(REAL / "project2.tiw").replace("\\", "/")
           .replace(" ", "%20") + "?mode=ro")
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    try:
        row = connection.execute("SELECT * FROM ProtocolTable").fetchone()
    finally:
        connection.close()

    for moment, key in (("LM", "LowMoment"), ("HM", "HighMoment")):
        want = np.asarray(json.loads(row[key + "GateCenterTimes"]), dtype=float)
        stored = stb.gate_table(protocol, moment)
        assert stored["centre"].size == want.size
        assert np.allclose(stored["centre"], want, rtol=1e-12)
        generated = stb.generate_gate_table(protocol, moment)
        assert generated is not None
        assert generated["centre"].size == want.size
        assert np.allclose(generated["centre"], want, rtol=1e-6)


@real_only
def test_the_raw_filter_mask_reproduces_what_the_project_recorded() -> None:
    """Both raw-stage tests together, over every gate of every record."""
    path = next(REAL.rglob("*.stb"))
    records, protocol = stb.read_records(path)
    stb.apply_raw_filters(records, protocol)
    uri = ("file:///" + str(REAL / "project2.tiw").replace("\\", "/")
           .replace(" ", "%20") + "?mode=ro")
    connection = sqlite3.connect(uri, uri=True)
    try:
        want = [np.asarray(json.loads(value), dtype=int) for (value,) in
                connection.execute("SELECT Filtered FROM Data "
                                   "ORDER BY RawDataId")]
    finally:
        connection.close()

    assert len(records) == len(want)
    for record, expected in zip(records, want):
        assert np.array_equal(record.filtered, expected)
