import pytest

from PyHydroGeophysX.data_processing.seismic import (
    FirstBreakPick,
    first_breaks_to_traveltime,
)


def _read_sensor_coords(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    count = int(lines[0])
    coords = []
    for line in lines[2 : 2 + count]:
        x, z = line.split()[:2]
        coords.append((float(x), float(z)))
    return coords


def test_traveltime_export_preserves_explicit_zero_source_coordinate(tmp_path):
    path = tmp_path / "traveltime.dat"
    picks = [
        FirstBreakPick(
            source_id=2,
            receiver_id=1,
            time_s=0.02,
            source_x=48.0,
            source_z=212.131,
            receiver_x=2.0,
            receiver_z=212.185,
            field_record=2,
            trace_number=1,
            trace_index=0,
            amplitude=0.0,
        ),
        FirstBreakPick(
            source_id=14,
            receiver_id=2,
            time_s=0.03,
            source_x=0.0,
            source_z=212.185,
            receiver_x=4.0,
            receiver_z=211.249,
            field_record=14,
            trace_number=2,
            trace_index=1,
            amplitude=0.0,
        ),
    ]

    first_breaks_to_traveltime(picks, str(path), receiver_spacing=2.0)

    coords = _read_sensor_coords(path)
    assert (0.0, 212.185) in coords
    assert (24.0, 212.185) not in coords


def test_traveltime_export_falls_back_for_missing_dict_coordinates(tmp_path):
    path = tmp_path / "traveltime.dat"
    rows = [
        {"source_id": 2, "receiver_id": 2, "time_s": 0.02},
        {"source_id": 14, "receiver_id": 3, "time_s": 0.03},
    ]

    first_breaks_to_traveltime(rows, str(path), receiver_spacing=2.0)

    coords = _read_sensor_coords(path)
    assert (0.0, 0.0) in coords
    assert (24.0, 0.0) in coords
    assert (2.0, 0.0) in coords


def test_bundled_ap411_has_a_real_zero_meter_last_shot():
    pytest.importorskip("numpy")
    from pathlib import Path

    from PyHydroGeophysX.data_processing.seismic import read_segy

    path = Path(__file__).resolve().parents[1] / "examples" / "data" / "Seismic" / "AP_411.sgy"
    dataset = read_segy(str(path), max_traces=4000)
    shot14 = dataset.get_gather(14)

    assert {header.source_x for header in shot14.headers} == {0.0}
