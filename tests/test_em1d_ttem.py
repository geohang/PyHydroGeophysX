from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import struct

import numpy as np

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
        "TxLoopArea=4\n"
        "WaveformLMPoint01=-2e-4 1\nWaveformLMPoint02=2e-6 0\n"
        "WaveformHMPoint01=-4.5e-4 1\nWaveformHMPoint02=4e-6 0\n"
        f"{gate_lines}\n"
        "[Channel1]\nTransmitterMoment=LM\nNoGates=5\nGateFactor=2\n"
        "RemoveInitialGates=1\nRemoveGatesFrom=5\nUniformDataSTD=.04\n"
        "[Channel2]\nTransmitterMoment=HM\nNoGates=6\nGateFactor=1\n"
        "RemoveInitialGates=2\nUniformDataSTD=.04\n",
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
    assert result["moments"]["LM"]["times"].size == 3
    assert result["moments"]["HM"]["times"].size == 4
