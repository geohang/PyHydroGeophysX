from pathlib import Path

import numpy as np
import pytest

from PyHydroGeophysX.data_processing.seismic import (
    SeismicTraceHeader,
    apply_agc,
    bandpass_filter,
    export_first_breaks,
    fit_velocity_traveltime_model,
    first_breaks_to_traveltime,
    normalize_traces,
    pick_first_breaks,
    predict_velocity_traveltimes,
    read_geometrics_dat,
    read_segy,
    tukey_taper,
)


def _geometrics_tag(text: str) -> bytes:
    payload = text.encode("latin1") + b"\x00"
    return (len(payload) + 2).to_bytes(2, "little") + payload


def _write_geometrics_dat(path: Path, shot_number: int, source_x: float, traces: np.ndarray, dt: float) -> None:
    n_samples, n_traces = traces.shape
    global_header_size = 128
    blocks = []
    offsets = []
    cursor = global_header_size
    for trace_index in range(n_traces):
        tags = b"".join(
            [
                _geometrics_tag(f"CHANNEL_NUMBER {trace_index + 1}"),
                _geometrics_tag(f"RECEIVER_LOCATION {2.0 * trace_index:.2f}"),
                _geometrics_tag(f"SAMPLE_INTERVAL {dt:.6f}"),
                _geometrics_tag(f"SHOT_SEQUENCE_NUMBER {shot_number}"),
                _geometrics_tag(f"SOURCE_LOCATION {source_x:.2f}"),
            ]
        )
        header_length = 32 + len(tags)
        data = np.asarray(traces[:, trace_index], dtype="<f4").tobytes()
        header = bytearray(32)
        header[0:2] = (0x4422).to_bytes(2, "little")
        header[2:4] = int(header_length).to_bytes(2, "little")
        header[4:8] = len(data).to_bytes(4, "little")
        header[8:12] = int(n_samples).to_bytes(4, "little")
        header[12:16] = (4).to_bytes(4, "little")
        block = bytes(header) + tags + data
        offsets.append(cursor)
        blocks.append(block)
        cursor += len(block)

    global_header = bytearray(global_header_size)
    global_header[6:8] = int(n_traces).to_bytes(2, "little")
    for index, offset in enumerate(offsets):
        global_header[32 + 4 * index : 36 + 4 * index] = int(offset).to_bytes(4, "little")
    path.write_bytes(bytes(global_header) + b"".join(blocks))


def test_read_segy_example_metadata():
    segy_path = Path("example/example/example_data.sgy")
    if not segy_path.exists():
        pytest.skip("Bundled raw SEG-Y example is not present.")

    dataset = read_segy(str(segy_path), max_traces=4)

    assert dataset.metadata.sample_interval_us == 125
    assert dataset.metadata.samples_per_trace == 4000
    assert dataset.metadata.format_code == 1
    assert dataset.traces.shape == (4000, 4)
    assert len(dataset.headers) == 4
    assert np.isfinite(dataset.traces).all()


def test_read_geometrics_dat_directory(tmp_path):
    traces_1 = np.arange(15, dtype=np.float32).reshape(5, 3)
    traces_2 = traces_1 + 100
    _write_geometrics_dat(tmp_path / "1.dat", shot_number=1, source_x=10.0, traces=traces_1, dt=0.001)
    _write_geometrics_dat(tmp_path / "2.dat", shot_number=2, source_x=8.0, traces=traces_2, dt=0.001)

    dataset = read_geometrics_dat(str(tmp_path))

    assert dataset.metadata.sample_interval_us == 1000
    assert dataset.metadata.samples_per_trace == 5
    assert dataset.metadata.format_code == 200
    assert dataset.traces.shape == (5, 6)
    assert dataset.field_records == [1, 2]
    assert dataset.get_gather(1).traces.shape == (5, 3)
    assert dataset.get_gather(2).headers[0].source_x == pytest.approx(8.0)
    assert dataset.get_gather(2).headers[-1].receiver_x == pytest.approx(4.0)


def test_preprocessing_helpers_on_synthetic_traces():
    dt = 0.001
    t = np.arange(400) * dt
    traces = np.column_stack(
        [
            np.sin(2 * np.pi * 20 * t),
            0.5 * np.sin(2 * np.pi * 35 * t),
            np.zeros_like(t),
        ]
    )

    gained = apply_agc(traces, dt=dt, window=0.05)
    normalized = normalize_traces(gained)
    tapered = tukey_taper(64, 8)
    filtered = bandpass_filter(traces[:, :2], dt=dt, f1=5, f2=10, f3=80, f4=120)

    assert gained.shape == traces.shape
    assert np.isfinite(gained).all()
    assert np.max(np.abs(normalized[:, 0])) == pytest.approx(1.0)
    assert tapered[0] == pytest.approx(0.0)
    assert tapered[-1] == pytest.approx(0.0)
    assert filtered.shape == (400, 2)


def test_first_break_export_and_traveltime_file(tmp_path):
    dt = 0.001
    traces = np.zeros((120, 3), dtype=float)
    traces[20, 0] = 1.0
    traces[30, 1] = 1.0
    traces[40, 2] = 1.0
    headers = [
        SeismicTraceHeader(1, 1, 1, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0),
        SeismicTraceHeader(1, 2, 1, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 2.0),
        SeismicTraceHeader(1, 3, 1, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 3.0),
    ]

    picks = pick_first_breaks(
        traces,
        dt=dt,
        headers=headers,
        threshold=0.5,
        noise_multiplier=0.0,
        max_time=0.08,
    )
    csv_path = export_first_breaks(picks, str(tmp_path / "picks.csv"))
    dat_path = first_breaks_to_traveltime(picks, str(tmp_path / "travel_time.dat"))

    assert [pick.time_s for pick in picks] == pytest.approx([0.02, 0.03, 0.04])
    assert Path(csv_path).read_text().splitlines()[0].startswith("source_id,receiver_id,time_s")
    dat_lines = Path(dat_path).read_text().splitlines()
    assert dat_lines[1] == "# x y"
    assert "# s g t" in dat_lines


def test_velocity_model_one_layer_recovers_apparent_velocity():
    receiver_x = np.linspace(5.0, 55.0, 11)
    source_x = np.zeros_like(receiver_x)
    true_times = 0.003 + receiver_x / 500.0

    model = fit_velocity_traveltime_model(source_x, receiver_x, true_times, max_segments=3)
    predicted = predict_velocity_traveltimes(model, source_x, receiver_x)

    assert len(model.segments) == 1
    assert model.segments[0].apparent_velocity_m_s == pytest.approx(500.0, rel=0.03)
    assert predicted == pytest.approx(true_times, abs=1e-6)


def test_velocity_model_two_segment_prefers_faster_deeper_branch():
    receiver_x = np.linspace(5.0, 85.0, 17)
    source_x = np.zeros_like(receiver_x)
    crossover = 35.0
    first_branch = 0.004 + receiver_x / 450.0
    second_intercept = (0.004 + crossover / 450.0) - crossover / 1450.0
    second_branch = second_intercept + receiver_x / 1450.0
    true_times = np.where(receiver_x <= crossover, first_branch, second_branch)

    model = fit_velocity_traveltime_model(source_x, receiver_x, true_times, max_segments=3)
    predicted = predict_velocity_traveltimes(model, source_x, receiver_x)
    velocities = [segment.apparent_velocity_m_s for segment in model.segments]

    assert len(model.segments) >= 2
    assert velocities[0] < velocities[-1]
    assert velocities[0] == pytest.approx(450.0, rel=0.12)
    assert velocities[-1] == pytest.approx(1450.0, rel=0.12)
    assert predicted == pytest.approx(true_times, abs=0.002)


def test_velocity_model_manual_anchors_can_override_predictions():
    receiver_x = np.linspace(5.0, 55.0, 11)
    source_x = np.zeros_like(receiver_x)
    auto_times = 0.005 + receiver_x / 700.0
    manual_indices = np.array([0, 5, 10], dtype=int)
    manual_times = auto_times[manual_indices] + np.array([0.001, -0.002, 0.001])

    model = fit_velocity_traveltime_model(
        np.r_[source_x[manual_indices], source_x],
        np.r_[receiver_x[manual_indices], receiver_x],
        np.r_[manual_times, auto_times],
        weights=np.r_[np.ones(manual_indices.size), np.full(receiver_x.size, 0.04)],
        max_segments=3,
    )
    final_times = predict_velocity_traveltimes(model, source_x, receiver_x)
    final_times[manual_indices] = manual_times

    assert final_times[manual_indices] == pytest.approx(manual_times)


def test_velocity_model_anchor_mask_passes_through_manual_points():
    receiver_x = np.array([5.0, 10.0, 20.0, 30.0, 40.0, 50.0])
    source_x = np.zeros_like(receiver_x)
    auto_hints = 0.006 + receiver_x / 900.0
    manual_indices = np.array([0, 2, 5], dtype=int)
    manual_times = np.array([0.010, 0.035, 0.055])
    fit_times = auto_hints.copy()
    fit_times[manual_indices] = manual_times
    anchor_mask = np.zeros_like(fit_times, dtype=bool)
    anchor_mask[manual_indices] = True

    model = fit_velocity_traveltime_model(
        source_x,
        receiver_x,
        fit_times,
        weights=np.where(anchor_mask, 1.0, 0.04),
        anchor_mask=anchor_mask,
        max_segments=3,
    )
    predicted = predict_velocity_traveltimes(model, source_x, receiver_x)

    assert len(model.segments) == 2
    assert predicted[manual_indices] == pytest.approx(manual_times, abs=1e-12)
    assert "anchor-exact" in model.message


def test_velocity_model_does_not_jump_to_late_high_amplitude_phase():
    receiver_x = np.linspace(5.0, 75.0, 15)
    source_x = np.zeros_like(receiver_x)
    first_arrival = 0.004 + receiver_x / 600.0
    late_phase = first_arrival + 0.045

    model = fit_velocity_traveltime_model(source_x[::4], receiver_x[::4], first_arrival[::4], max_segments=3)
    predicted = predict_velocity_traveltimes(model, source_x, receiver_x)

    assert np.max(np.abs(predicted - first_arrival)) < 0.002
    assert np.min(np.abs(predicted - late_phase)) > 0.020


def test_velocity_model_source_inside_spread_fits_left_and_right_branches():
    receiver_x = np.array([-30.0, -20.0, -10.0, 10.0, 20.0, 30.0])
    source_x = np.zeros_like(receiver_x)
    true_times = 0.004 + np.abs(receiver_x) / 650.0

    model = fit_velocity_traveltime_model(source_x, receiver_x, true_times, max_segments=1)
    predicted = predict_velocity_traveltimes(model, source_x, receiver_x)

    assert set(model.branch_ids) == {"left", "right"}
    assert predicted == pytest.approx(true_times, abs=1e-6)
