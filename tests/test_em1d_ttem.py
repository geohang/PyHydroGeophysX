from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import struct

import numpy as np
import pytest

from PyHydroGeophysX.data_processing.em1d import load_sounding
from PyHydroGeophysX.data_processing.ttem import is_ttem_source


def _header() -> bytes:
    lines = [
        "[SOFTWAREID_1]", "HARDCHID=0", "MOMENTID=2",
        "[SOFTWAREID_2]", "HARDCHID=0", "MOMENTID=1",
    ]
    for moment, n_gates, ontime, front in ((1, 6, 450e-6, 454e-6),
                                            (2, 5, 200e-6, 202e-6)):
        lines.extend([f"[MOMENTID_{moment}]", f"ONTIME={ontime}",
                      f"FRONTGATETIME={front}", f"NINDEX={n_gates}"])
        for gate in range(1, n_gates + 1):
            centre = (2 * gate + 1) * 1e-6
            lines.extend([f"[MOMENTID_{moment}_SAMPLE_{gate}]",
                          f"SAMPLEINDEX={2 * gate + 2}",
                          "SAMPLEFACTOR=1e-8",
                          f"SAMPLECENTERTIME={centre}"])
    return ("\r\n".join(lines) + "\r\n").encode("ascii")


def _delphi_days(when: datetime) -> float:
    return (when - datetime(1899, 12, 30)).total_seconds() / 86400.0


def _make_project(folder: Path) -> Path:
    run = folder / "tTEMLog" / "20260811_120000" / "Run001"
    run.mkdir(parents=True)
    raw_path = run / "20260811_120000_000_tTEM_Rawdata.skb"
    chunks = [b"VERSION 2 ", _header(), b"program", b"hardware"]
    start = datetime(2026, 8, 11, 12, 0, 0)
    payload = bytearray(struct.pack("<I", 2))
    for chunk in chunks:
        payload.extend(struct.pack("<I", len(chunk)))
        payload.extend(chunk)
    for cycle in range(4):
        for software_id, n_gates, delay in ((1, 5, 0.0), (2, 6, 0.25)):
            when = start + timedelta(seconds=0.6 * cycle + delay)
            values = np.empty((4, n_gates), dtype="<i2")
            values[0::2] = -100
            values[1::2] = 100
            payload.extend(struct.pack("<BddIII", 1, _delphi_days(when),
                                       _delphi_days(when + timedelta(seconds=.1)),
                                       software_id, 4, n_gates))
            payload.extend(values.tobytes())
    raw_path.write_bytes(payload)

    gps = run / "20260811_120000_000_tTEM_GPS.sps"
    gps.write_text(
        "G12 2026 08 11 12 00 00 000 "
        "$GPGGA,120000.00,4100.0000,N,09100.0000,W,2,14,0.7,200.0,M,0,M,,*00;\n"
        "G12 2026 08 11 12 00 03 000 "
        "$GPGGA,120003.00,4100.0060,N,09059.9940,W,2,14,0.7,201.0,M,0,M,,*00;\n",
        encoding="ascii",
    )
    tx = run / "20260811_120000_000_tTEM_Rawdata_TX.sps"
    lines = []
    for cycle in range(4):
        when = start + timedelta(seconds=0.6 * cycle)
        stamp = when.strftime("%Y %m %d %H %M %S")
        lines.append(f"TXD {stamp} 000 0 0 0 0 0 0 0 0 0 1 0 30.0 0 0 0")
        lines.append(f"TXD {stamp} 250 0 0 0 0 0 0 0 0 0 1 0 3.0 0 0 0")
    tx.write_text("\n".join(lines), encoding="ascii")
    return folder


def test_ttem_raw_detection_and_joint_loading(tmp_path: Path) -> None:
    source = _make_project(tmp_path / "survey")
    assert is_ttem_source(str(source))

    result = load_sounding(str(source), "TDEM", moment="LM+HM")

    assert result["ttem"] is True
    assert result["temcompany"] is True
    assert result["n_soundings"] == 2
    assert set(result["moments"]) == {"LM", "HM"}
    np.testing.assert_allclose(result["moments"]["LM"]["response"], 100e-8 / (3.0 * 8.0))
    np.testing.assert_allclose(result["moments"]["HM"]["response"], 100e-8 / (30.0 * 8.0))
    assert result["line_numbers"].tolist() == [1, 1]
    assert np.all(np.diff(result["positions"]) > 0)
    assert result["system"]["tx_rx_sep"] == 9.28


def test_ttem_raw_is_tdem_only(tmp_path: Path) -> None:
    source = _make_project(tmp_path / "survey")
    try:
        load_sounding(str(source), "FDEM")
    except ValueError as exc:
        assert "time-domain" in str(exc)
    else:
        raise AssertionError("Raw tTEM input should reject FDEM loading")


def test_ttem_loop_area_controls_normalization(tmp_path: Path) -> None:
    source = _make_project(tmp_path / "survey")
    area_8 = load_sounding(
        str(source), "TDEM", moment="HM", ttem_loop_area=8.0
    )
    area_4 = load_sounding(
        str(source), "TDEM", moment="HM", ttem_loop_area=4.0
    )

    np.testing.assert_allclose(area_4["response"], 2.0 * area_8["response"])
    assert area_4["system"]["loop_area"] == 4.0
    np.testing.assert_allclose(area_4["system"]["source_radius"], np.sqrt(4.0 / np.pi))


def test_ttem_applies_gex_and_tfi(tmp_path: Path) -> None:
    source = _make_project(tmp_path / "survey")
    gex = source / "system.gex"
    gate_lines = "\n".join(
        f"GateTime{i:02d}={2*i+1}e-6 {2*i}e-6 {2*i+2}e-6"
        for i in range(1, 7)
    )
    gex.write_text(
        "[General]\n"
        "RxCoilPosition1=-7 0 -0.5\nTxCoilPosition1=0 0 -0.7\n"
        "TxLoopArea=4\nRxCoilLPFilter1=.86 420e3\n"
        "WaveformLMPoint01=-2e-4 1\nWaveformLMPoint02=2e-6 0\n"
        "WaveformHMPoint01=-4.5e-4 1\nWaveformHMPoint02=4e-6 0\n"
        f"{gate_lines}\n"
        "[Channel1]\nTransmitterMoment=LM\nNoGates=5\nGateFactor=2\n"
        "RemoveInitialGates=1\nRemoveGatesFrom=5\nUniformDataSTD=.04\n"
        "TiBLowPassFilter=1 679e3\n"
        "[Channel2]\nTransmitterMoment=HM\nNoGates=6\nGateFactor=1\n"
        "RemoveInitialGates=2\nUniformDataSTD=.04\n"
        "TiBLowPassFilter=1 679e3\n",
        encoding="ascii",
    )
    tfi = source / "system.tfi"
    tfi.write_text(
        "[FilterSwCh1]\nPeriodtime=.000496\nFilter=.5 .5\n"
        "[FilterSwCh2]\nPeriodtime=.001773\nFilter=.5 .5\n",
        encoding="ascii",
    )

    result = load_sounding(str(source), "TDEM", moment="LM+HM")

    assert result["calibration"]["gex_applied"] is True
    assert result["calibration"]["tfi_applied"] is True
    assert result["system"]["loop_area"] == 4.0
    assert result["system"]["tx_rx_sep"] == 7.0
    assert result["system"]["height"] == 0.5
    assert result["protocol"]["uniform_std"] == 0.04
    assert result["protocol"]["tfi_channels"] == [1, 2]
    assert result["calibration"]["analog_lowpass_modelled"] is True
    assert result["protocol"]["analog_lowpass"]["LM"] == {
        "receiver_damping": 0.86,
        "receiver_cutoff_hz": 420e3,
        "tib_order": 1,
        "tib_cutoff_hz": 679e3,
    }
    assert result["moments"]["HM"]["transmitter"]["analog_lowpass"] == (
        result["protocol"]["analog_lowpass"]["HM"]
    )
    assert result["moments"]["LM"]["times"].size == 3
    assert result["moments"]["HM"]["times"].size == 4


def test_gex_analog_filter_is_causal_and_has_unity_dc_gain() -> None:
    # The filter itself is plain NumPy, but it lives beside the SimPEG survey
    # builders and ``tdem_forward`` imports SimPEG at module scope.
    pytest.importorskip("simpeg")

    from PyHydroGeophysX.forward.tdem_forward import _filter_operator

    times = np.linspace(0.0, 10e-6, 101)
    response = _filter_operator(
        times, np.ones((times.size, 1)), (0.86, 420e3, 1, 679e3, ())
    ).ravel()

    assert response[0] == 0.0
    assert abs(response[-1] - 1.0) < 1e-8
    assert np.max(response) < 1.01


def test_cascaded_first_order_stages_keep_unity_dc_gain() -> None:
    """Some systems list first-order corner frequencies instead of a damped pair."""
    pytest.importorskip("simpeg")

    from PyHydroGeophysX.forward.tdem_forward import _filter_operator

    times = np.linspace(0.0, 10e-6, 201)
    response = _filter_operator(
        times, np.ones((times.size, 1)), (0.0, 0.0, 0, 0.0, (450e3, 800e3))
    ).ravel()

    assert response[0] == 0.0                      # causal
    assert abs(response[-1] - 1.0) < 1e-6          # unity DC gain
    assert np.all(np.diff(response) >= -1e-12)     # real poles, so no ringing


def test_the_receiver_filter_lifts_a_steeply_decaying_early_time() -> None:
    """Why the filter matters for a transient, and in which direction.

    A causal low-pass output is a weighted average of the input over the
    preceding time constant. On a steeply decaying signal that window sits where
    the signal was larger, so filtering *raises* the modelled early-time value,
    and the lift dies away once the signal changes little across a time constant.
    Leaving the filter out therefore tilts the predicted decay, and an inversion
    absorbs a tilt by inventing a vertical resistivity gradient near the surface.
    """
    pytest.importorskip("simpeg")

    from PyHydroGeophysX.forward.tdem_forward import _filter_operator

    times = np.linspace(1e-7, 4e-5, 4000)
    decay = times ** -2.5                          # a typical early-time slope
    filtered = _filter_operator(
        times, decay[:, None], (0.0, 0.0, 0, 0.0, (450e3, 800e3))).ravel()

    at_early = int(np.searchsorted(times, 5.7e-6))
    at_late = int(np.searchsorted(times, 3.5e-5))
    early = filtered[at_early] / decay[at_early]
    late = filtered[at_late] / decay[at_late]

    # The lift is what decays, so compare the excess over unity rather than the
    # ratios themselves: both stay above 1, and it is the gap that shrinks.
    assert early > 1.10                            # a real lift at the first gate
    assert (late - 1.0) < (early - 1.0) / 3.0      # and largely gone by the late ones


def test_first_order_corners_are_read_and_validated() -> None:
    from PyHydroGeophysX.forward.tdem_forward import _analog_parameters

    assert _analog_parameters(None)[4] == ()
    assert _analog_parameters({"first_order_cutoffs_hz": (450e3, 800e3)})[4] == (450e3, 800e3)
    # Zero and negative corners describe no filter and must not create a stage.
    assert _analog_parameters({"first_order_cutoffs_hz": (0.0, -1.0)})[4] == ()


def test_the_filter_grid_is_uniform_in_log_across_the_whole_span() -> None:
    """The reconstruction error is what sets the density, and it is relative.

    An earlier grid was dense in linear time only until the filter had settled,
    then fell back to the receiver times themselves. That reads as if the filter
    sets the requirement, and it does not: the first-order-hold step is exact
    for an input that is linear over it, so the error comes from how far the
    decay departs from a straight line between nodes. That departure is
    relative and roughly constant per log step, so the density belongs in log
    time and the same density serves every decade. The old grid left steps
    twenty times the filter's time constant past the settling point, and the
    lag that leaves does not die away with time the way the operator does.
    """
    pytest.importorskip("simpeg")

    from PyHydroGeophysX.forward.em1d import _tdem_config
    from PyHydroGeophysX.forward.tdem_forward import _analog_sampling

    gates = np.geomspace(5.7e-6, 3.5e-5, 9)
    geom = {
        "source_radius": 0.355, "tx_rx_sep": 15.0, "height": 0.9,
        "orientation": "z", "receiver_type": "dbdt", "waveform": "step_off",
        "analog_lowpass": {"first_order_cutoffs_hz": (450e3, 800e3)},
        "analog_model_points_per_decade": 40,
    }
    times, _ = _analog_sampling(_tdem_config(geom, gates))

    steps = np.diff(np.log10(times))
    assert times.size > 100
    # Forty per decade, so no step wider than a fortieth of a decade. The gate
    # times are unioned in, which can only make a step smaller.
    assert np.max(steps) <= 1.0 / 40.0 + 1e-9
    # And the density holds out to the last gate rather than stopping early.
    assert times[-1] >= gates[-1] * (1.0 - 1e-9)


def test_a_slow_filter_does_not_pay_for_a_fast_one() -> None:
    """The rule scales with the corner, so a gentle filter stays cheap."""
    pytest.importorskip("simpeg")

    from PyHydroGeophysX.forward.em1d import _tdem_config
    from PyHydroGeophysX.forward.tdem_forward import _analog_sampling

    gates = np.geomspace(5.7e-6, 3.5e-5, 9)
    base = {"source_radius": 0.355, "tx_rx_sep": 15.0, "height": 0.9,
            "orientation": "z", "receiver_type": "dbdt", "waveform": "step_off"}
    fast, _ = _analog_sampling(_tdem_config(
        {**base, "analog_lowpass": {"first_order_cutoffs_hz": (450e3, 800e3)}}, gates))
    slow, _ = _analog_sampling(_tdem_config(
        {**base, "analog_lowpass": {"first_order_cutoffs_hz": (45e3, 80e3)}}, gates))

    # A ten-fold slower filter settles ten times later but needs the same number
    # of nodes per time constant, so the counts stay the same order.
    assert slow.size < fast.size * 3
