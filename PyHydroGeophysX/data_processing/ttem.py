"""Reader for TEMcompany tTEM ``SKB``/``SPS`` raw acquisitions.

The acquisition logger writes alternating-polarity transient series to SKB and
navigation/transmitter telemetry to SPS.  This module indexes those files
lazily, stacks a few consecutive cycles into a sounding, and exposes the same
dictionary contract as the TEM2Go reader.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from functools import lru_cache
import math
from pathlib import Path
import re
import struct
from typing import Any, Dict, Iterable, Optional

import numpy as np

_DELPHI_EPOCH = datetime(1899, 12, 30)
_EARTH_RADIUS = 6_371_000.0


def is_ttem_source(path: str) -> bool:
    """Return whether *path* contains a TEMcompany tTEM raw acquisition."""
    source = Path(path)
    if source.is_file():
        return source.suffix.lower() == ".skb"
    if not source.is_dir():
        return False
    return next(source.rglob("*_tTEM_Rawdata.skb"), None) is not None


def _number(section: str, name: str, default: float = 0.0) -> float:
    match = re.search(rf"(?im)^{re.escape(name)}\s*=\s*([^\r\n]+)", section)
    if match is None:
        return float(default)
    try:
        return float(match.group(1).strip())
    except ValueError:
        return float(default)


def _section(text: str, name: str) -> str:
    match = re.search(
        rf"(?ims)^\[{re.escape(name)}\]\s*(.*?)(?=^\[|\Z)", text
    )
    return match.group(1) if match else ""


def _text_value(section: str, name: str, default: str = "") -> str:
    match = re.search(rf"(?im)^{re.escape(name)}\s*=\s*([^\r\n]*)", section)
    if match is None:
        return default
    return match.group(1).split("/", 1)[0].strip()


def _float_values(section: str, name: str) -> np.ndarray:
    raw = _text_value(section, name)
    try:
        return np.asarray([float(value) for value in raw.split()], dtype=float)
    except ValueError:
        return np.array([], dtype=float)


@lru_cache(maxsize=8)
def _read_gex(resolved_path: str) -> Dict[str, Any]:
    """Parse the tTEM GEX fields used by import and 1-D forward modelling."""
    path = Path(resolved_path)
    text = path.read_text(encoding="utf-8-sig", errors="replace").replace("\r", "")
    general = _section(text, "General")
    rx = _float_values(general, "RxCoilPosition1")
    tx = _float_values(general, "TxCoilPosition1")
    loop_area = _number(general, "TxLoopArea", 8.0)
    rx_lowpasses = {
        number: values
        for number in range(1, 9)
        if (values := _float_values(general, f"RxCoilLPFilter{number}")).size
    }
    result: Dict[str, Any] = {
        "path": str(path), "loop_area": loop_area,
        "tx_rx_sep": (float(np.linalg.norm(rx[:2] - tx[:2]))
                      if rx.size >= 2 and tx.size >= 2 else 9.28),
        "height": abs(float(rx[2])) if rx.size >= 3 else 0.43,
        "tx_height": abs(float(tx[2])) if tx.size >= 3 else np.nan,
        # The suffix identifies the receiver coil, not another cascaded stage.
        "rx_lowpasses": rx_lowpasses,
        "rx_lowpass": rx_lowpasses.get(1, np.array([], dtype=float)),
        "front_gate_delay": _number(general, "FrontGateDelay", 0.0),
        "moments": {},
    }
    gate_rows = []
    for match in re.finditer(r"(?im)^GateTime(\d+)\s*=\s*([^/\r\n]+)", general):
        try:
            values = [float(value) for value in match.group(2).split()[:3]]
        except ValueError:
            continue
        if len(values) == 3:
            gate_rows.append((int(match.group(1)), *values))
    gate_rows.sort()
    gates = np.asarray([row[1:] for row in gate_rows], dtype=float)
    for channel_number in range(1, 9):
        channel = _section(text, f"Channel{channel_number}")
        if not channel:
            continue
        moment = _text_value(channel, "TransmitterMoment").upper()
        if moment not in {"LM", "HM"}:
            continue
        waveform_rows = []
        for match in re.finditer(
            rf"(?im)^Waveform{moment}Point(\d+)\s*=\s*([^/\r\n]+)", general
        ):
            try:
                values = [float(value) for value in match.group(2).split()[:2]]
            except ValueError:
                continue
            if len(values) == 2:
                waveform_rows.append((int(match.group(1)), *values))
        waveform_rows.sort()
        waveform = np.asarray([row[1:] for row in waveform_rows], dtype=float)
        n_gates = int(_number(channel, "NoGates", gates.shape[0]))
        selected_gates = gates[:n_gates].copy()
        shift = _number(channel, "GateTimeShift", 0.0)
        if selected_gates.size:
            selected_gates += shift
        rx_coil = int(_number(channel, "RxCoilNumber", 1))
        result["moments"][moment] = {
            "times": selected_gates[:, 0] if selected_gates.size else np.array([]),
            "gate_open": selected_gates[:, 1] if selected_gates.size else np.array([]),
            "gate_close": selected_gates[:, 2] if selected_gates.size else np.array([]),
            "waveform_times": waveform[:, 0] if waveform.size else np.array([]),
            "waveform_currents": waveform[:, 1] if waveform.size else np.array([]),
            "gate_factor": _number(channel, "GateFactor", 1.0),
            "loop_turns": int(_number(general, f"NumberOfTurns{moment}", 1)),
            "remove_initial": int(_number(channel, "RemoveInitialGates", 0)),
            "remove_from": int(_number(channel, "RemoveGatesFrom", n_gates + 1)),
            "uniform_std": _number(channel, "UniformDataSTD", 0.03),
            "rx_coil": rx_coil,
            "rx_lowpass": rx_lowpasses.get(rx_coil, np.array([], dtype=float)),
            "tib_lowpass": _float_values(channel, "TiBLowPassFilter"),
        }
    if not result["moments"]:
        raise ValueError(f"No LM/HM channel definitions were found in GEX: {path}")
    return result


@lru_cache(maxsize=8)
def _read_tfi(resolved_path: str) -> Dict[int, Dict[str, Any]]:
    """Read FIR coefficients for the software channels in a tTEM TFI file."""
    path = Path(resolved_path)
    text = path.read_text(encoding="utf-8-sig", errors="replace").replace("\r", "")
    result: Dict[int, Dict[str, Any]] = {}
    for channel in range(1, 9):
        block = _section(text, f"FilterSwCh{channel}")
        coefficients = _float_values(block, "Filter") if block else np.array([])
        if coefficients.size:
            result[channel] = {
                "coefficients": coefficients,
                "period": _number(block, "Periodtime", np.nan),
            }
    if not result:
        raise ValueError(f"No FilterSwCh FIR definitions were found in TFI: {path}")
    return result


def _calibration_path(source: Path, explicit: Optional[str], suffix: str) -> Optional[Path]:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"Selected {suffix.upper()} file not found: {path}")
        if path.suffix.lower() != f".{suffix}":
            raise ValueError(f"Selected calibration file must end in .{suffix}: {path}")
        return path
    root = source if source.is_dir() else source.parent
    matches = sorted(root.rglob(f"*.{suffix}"))
    return matches[0].resolve() if len(matches) == 1 else None


def _timestamp(parts: list[str], start: int = 1) -> float:
    values = [int(parts[start + i]) for i in range(6)]
    milliseconds = int(parts[start + 6])
    return (datetime(*values) + timedelta(milliseconds=milliseconds)).timestamp()


def _nmea_coordinate(value: str, hemisphere: str) -> float:
    raw = float(value)
    degrees = math.floor(raw / 100.0)
    result = degrees + (raw - 100.0 * degrees) / 60.0
    return -result if hemisphere.upper() in {"S", "W"} else result


def _read_gps(path: Optional[Path]) -> Dict[str, np.ndarray]:
    rows = []
    if path is None or not path.exists():
        return {key: np.array([], dtype=float) for key in ("time", "lat", "lon", "elev")}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("G12 ") or "$GPGGA," not in line:
            continue
        parts = line.split()
        try:
            date = datetime(int(parts[1]), int(parts[2]), int(parts[3]))
            gga = line.split("$GPGGA,", 1)[1].split(";", 1)[0].split(",")
            clock = gga[0]
            hour, minute = int(clock[:2]), int(clock[2:4])
            second = float(clock[4:])
            whole = int(second)
            when = date.replace(hour=hour, minute=minute, second=whole)
            when += timedelta(seconds=second - whole)
            lat = _nmea_coordinate(gga[1], gga[2])
            lon = _nmea_coordinate(gga[3], gga[4])
            elev = float(gga[8])
            rows.append((when.timestamp(), lat, lon, elev))
        except (IndexError, TypeError, ValueError):
            continue
    if not rows:
        return {key: np.array([], dtype=float) for key in ("time", "lat", "lon", "elev")}
    values = np.asarray(rows, dtype=float)
    order = np.argsort(values[:, 0])
    values = values[order]
    # The logger can repeat a PC-time packet.  np.interp needs ascending unique times.
    _, unique = np.unique(values[:, 0], return_index=True)
    values = values[np.sort(unique)]
    return {"time": values[:, 0], "lat": values[:, 1],
            "lon": values[:, 2], "elev": values[:, 3]}


def _read_tx(path: Optional[Path]) -> Dict[str, np.ndarray]:
    rows = []
    if path is not None and path.exists():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.startswith("TXD "):
                continue
            parts = line.split()
            try:
                rows.append((_timestamp(parts), float(parts[19])))
            except (IndexError, TypeError, ValueError):
                continue
    values = np.asarray(rows, dtype=float).reshape((-1, 2)) if rows else np.empty((0, 2))
    return {"time": values[:, 0], "current": values[:, 1]}


def _nearest_file(folder: Path, pattern: str) -> Optional[Path]:
    found = sorted(folder.glob(pattern))
    return found[0] if found else None


def _parse_header(text: str) -> Dict[str, Any]:
    software: Dict[int, str] = {}
    moments: Dict[str, Dict[str, Any]] = {}
    for software_id in (1, 2):
        block = _section(text, f"SOFTWAREID_{software_id}")
        moment_id = int(_number(block, "MOMENTID", 0))
        name = "HM" if moment_id == 1 else "LM"
        software[software_id] = name
        moment_block = _section(text, f"MOMENTID_{moment_id}")
        n_gates = int(_number(moment_block, "NINDEX", 0))
        centres, factors, indices = [], [], []
        for gate in range(1, n_gates + 1):
            sample = _section(text, f"MOMENTID_{moment_id}_SAMPLE_{gate}")
            centres.append(_number(sample, "SAMPLECENTERTIME", np.nan))
            factors.append(_number(sample, "SAMPLEFACTOR", np.nan))
            indices.append(_number(sample, "SAMPLEINDEX", np.nan))
        centre = np.asarray(centres, dtype=float)
        close = np.asarray(indices, dtype=float) * 1e-6
        open_ = 2.0 * centre - close
        on_time = _number(moment_block, "ONTIME", 0.0)
        turnoff = max(_number(moment_block, "FRONTGATETIME", on_time) - on_time, 1e-7)
        moments[name] = {
            "times": centre,
            "factors": np.asarray(factors, dtype=float),
            "gate_open": open_,
            "gate_close": close,
            "waveform_times": np.asarray([-on_time, 0.0, turnoff]),
            "waveform_currents": np.asarray([1.0, 1.0, 0.0]),
        }
    return {"software": software, "moments": moments}


def _raw_files(source: Path) -> Iterable[Path]:
    if source.is_file():
        return [source]
    return sorted(source.rglob("*_tTEM_Rawdata.skb"))


def _index_file(path: Path) -> Dict[str, Any]:
    records = []
    with path.open("rb") as handle:
        raw = handle.read(4)
        if len(raw) != 4 or struct.unpack("<I", raw)[0] != 2:
            raise ValueError(f"Unsupported tTEM SKB version in {path.name}.")
        chunks = []
        for _ in range(4):
            size_raw = handle.read(4)
            if len(size_raw) != 4:
                raise ValueError(f"Truncated tTEM SKB header in {path.name}.")
            size = struct.unpack("<I", size_raw)[0]
            chunks.append(handle.read(size))
        header = _parse_header(chunks[1].decode("latin1", errors="replace"))
        while True:
            record_header = handle.read(29)
            if len(record_header) < 29:
                break
            hard_channel = record_header[0]
            start_day, end_day = struct.unpack_from("<dd", record_header, 1)
            software_id, n_series, n_gates = struct.unpack_from("<III", record_header, 17)
            payload = int(2 * n_series * n_gates)
            if hard_channel > 16 or software_id not in header["software"] or payload <= 0:
                break
            offset = handle.tell()
            handle.seek(payload, 1)
            when = (_DELPHI_EPOCH + timedelta(days=0.5 * (start_day + end_day))).timestamp()
            records.append({"path": path, "offset": offset, "software_id": software_id,
                            "moment": header["software"][software_id], "n_series": n_series,
                            "n_gates": n_gates, "time": when})
    gps = _read_gps(_nearest_file(path.parent, "*_tTEM_GPS.sps"))
    tx = _read_tx(_nearest_file(path.parent, "*_tTEM_Rawdata_TX.sps"))
    return {"path": path, "records": records, "header": header, "gps": gps, "tx": tx}


def _cycle_groups(records: list[Dict[str, Any]], stack_seconds: float) -> list[list[Dict[str, Any]]]:
    cycles: list[list[Dict[str, Any]]] = []
    active: list[Dict[str, Any]] = []
    for record in records:
        if record["moment"] == "LM":
            if active:
                cycles.append(active)
            active = [record]
        elif active:
            active.append(record)
    if active:
        cycles.append(active)
    complete = [cycle for cycle in cycles if {r["moment"] for r in cycle} == {"LM", "HM"}]
    if not complete:
        return []
    cycle_times = np.asarray([np.mean([r["time"] for r in cycle]) for cycle in complete])
    interval = float(np.nanmedian(np.diff(cycle_times))) if cycle_times.size > 1 else stack_seconds
    per_stack = max(1, int(round(float(stack_seconds) / max(interval, 1e-3))))
    return [sum(complete[i:i + per_stack], []) for i in range(0, len(complete), per_stack)]


@lru_cache(maxsize=4)
def _project_index(resolved_path: str, stack_seconds: float = 2.0) -> Dict[str, Any]:
    source = Path(resolved_path)
    files = [_index_file(path) for path in _raw_files(source)]
    soundings = []
    line_number = 0
    for item in files:
        groups = _cycle_groups(item["records"], stack_seconds)
        if not groups:
            continue
        line_number += 1
        for group_number, records in enumerate(groups):
            when = float(np.mean([record["time"] for record in records]))
            soundings.append({"records": records, "file": item, "time": when,
                              "line": line_number, "station": group_number + 1})
    if not soundings:
        raise ValueError("No complete alternating LM/HM cycles were found in the tTEM SKB files.")

    all_gps = [item["gps"] for item in files if item["gps"]["time"].size]
    if all_gps:
        gps = {key: np.concatenate([item[key] for item in all_gps])
               for key in ("time", "lat", "lon", "elev")}
        order = np.argsort(gps["time"])
        gps = {key: value[order] for key, value in gps.items()}
        times = np.asarray([item["time"] for item in soundings])
        lat = np.interp(times, gps["time"], gps["lat"])
        lon = np.interp(times, gps["time"], gps["lon"])
        elev = np.interp(times, gps["time"], gps["elev"])
    else:
        lat = lon = elev = np.full(len(soundings), np.nan)
    if np.any(np.isfinite(lat)) and np.any(np.isfinite(lon)):
        lat0, lon0 = float(np.nanmedian(lat)), float(np.nanmedian(lon))
        x = _EARTH_RADIUS * np.cos(np.deg2rad(lat0)) * np.deg2rad(lon - lon0)
        y = _EARTH_RADIUS * np.deg2rad(lat - lat0)
    else:
        x = np.arange(len(soundings), dtype=float) * 10.0
        y = np.zeros(len(soundings), dtype=float)
    positions = np.zeros(len(soundings), dtype=float)
    cursor = 0.0
    for line in sorted({int(item["line"]) for item in soundings}):
        indices = np.asarray([i for i, item in enumerate(soundings) if item["line"] == line])
        distance = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x[indices]), np.diff(y[indices])))])
        positions[indices] = cursor + distance
        cursor = float(positions[indices[-1]] + 10.0)
    return {"files": files, "soundings": soundings, "lat": lat, "lon": lon,
            "elev": elev, "x": x, "y": y, "positions": positions}


def _current(record: Dict[str, Any], file_index: Dict[str, Any]) -> float:
    tx = file_index["tx"]
    if not tx["time"].size:
        return 30.0 if record["moment"] == "HM" else 3.0
    # Alternating telemetry contains both moments.  Filter before choosing the
    # nearest sample so an HM record never receives the adjacent LM current.
    positive = tx["current"][tx["current"] > 0.0]
    low, high = (np.nanpercentile(positive, [25.0, 75.0])
                 if positive.size >= 2 else (3.0, 30.0))
    threshold = math.sqrt(low * high) if high > 1.5 * low else float(np.nanmedian(positive))
    mask = (tx["current"] > threshold if record["moment"] == "HM"
            else (tx["current"] > 0.0) & (tx["current"] <= threshold))
    times, currents = tx["time"][mask], tx["current"][mask]
    if not times.size:
        return 30.0 if record["moment"] == "HM" else 3.0
    return float(currents[int(np.argmin(np.abs(times - record["time"])))])


def _stack_record(
    record: Dict[str, Any], file_index: Dict[str, Any], loop_area: float,
    loop_turns: int, filters: Dict[int, Dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    with record["path"].open("rb") as handle:
        handle.seek(record["offset"])
        count = int(record["n_series"] * record["n_gates"])
        raw = np.frombuffer(handle.read(2 * count), dtype="<i2", count=count)
    values = raw.reshape((record["n_series"], record["n_gates"])).astype(float)
    signs = np.where(np.arange(record["n_series"]) % 2 == 0, 1.0, -1.0)
    signed = -(values * signs[:, None])
    filter_spec = filters.get(int(record["software_id"]), {})
    coefficients = np.asarray(filter_spec.get("coefficients", []), dtype=float).ravel()
    if 1 < coefficients.size <= signed.shape[0]:
        # The TFI is an FIR over the sign-corrected transient sequence.  Decimate
        # valid filtered outputs by one filter length for a conservative number
        # of independent samples when estimating the stack error.
        filtered = np.apply_along_axis(
            lambda values_: np.convolve(values_, coefficients, mode="valid"),
            0, signed,
        )
        mean = np.mean(filtered, axis=0)
        independent = filtered[::coefficients.size]
        sem = (np.std(independent, axis=0, ddof=1) / math.sqrt(independent.shape[0])
               if independent.shape[0] > 1 else np.zeros(mean.size))
    else:
        n_pairs = signed.shape[0] // 2
        if n_pairs < 1:
            raise ValueError("A tTEM transient record contains no polarity pair.")
        pairs = 0.5 * (signed[:2 * n_pairs:2] + signed[1:2 * n_pairs:2])
        mean = np.mean(pairs, axis=0)
        sem = (np.std(pairs, axis=0, ddof=1) / math.sqrt(pairs.shape[0])
               if pairs.shape[0] > 1 else np.zeros(mean.size))
    config = file_index["header"]["moments"][record["moment"]]
    scale = config["factors"] / max(
        abs(_current(record, file_index)) * float(loop_area) * max(1, int(loop_turns)),
        1e-12,
    )
    return mean * scale, sem * np.abs(scale)


def _moment_data(
    sounding: Dict[str, Any], moment: str, loop_area: float,
    gex: Optional[Dict[str, Any]], filters: Dict[int, Dict[str, Any]],
) -> Dict[str, Any]:
    file_index = sounding["file"]
    records = [record for record in sounding["records"] if record["moment"] == moment]
    calibrated = (gex or {}).get("moments", {}).get(moment)
    loop_turns = int((calibrated or {}).get("loop_turns", 1))
    stacked = [
        _stack_record(record, file_index, loop_area, loop_turns, filters)
        for record in records
    ]
    responses = np.asarray([item[0] for item in stacked])
    within = np.asarray([item[1] for item in stacked])
    response = np.mean(responses, axis=0)
    between = (np.std(responses, axis=0, ddof=1) / math.sqrt(responses.shape[0])
               if responses.shape[0] > 1 else np.zeros(response.size))
    absolute_std = np.sqrt(between ** 2 + np.mean(within ** 2, axis=0))
    relative_std = absolute_std / np.maximum(np.abs(response), 1e-30)
    config = file_index["header"]["moments"][moment]
    if calibrated:
        response *= float(calibrated.get("gate_factor", 1.0))
        absolute_std *= abs(float(calibrated.get("gate_factor", 1.0)))
        times = np.asarray(calibrated.get("times", []), dtype=float)
        gate_open = np.asarray(calibrated.get("gate_open", []), dtype=float)
        gate_close = np.asarray(calibrated.get("gate_close", []), dtype=float)
        waveform_times = np.asarray(calibrated.get("waveform_times", []), dtype=float)
        waveform_currents = np.asarray(calibrated.get("waveform_currents", []), dtype=float)
    else:
        times = np.asarray(config["times"], dtype=float)
        gate_open = np.asarray(config["gate_open"], dtype=float)
        gate_close = np.asarray(config["gate_close"], dtype=float)
        waveform_times = np.asarray(config["waveform_times"], dtype=float)
        waveform_currents = np.asarray(config["waveform_currents"], dtype=float)
    # Earliest raw channels are transmitter/electronics recovery.  Keep the
    # first consistently positive part of the decay, while never discarding
    # more than six gates automatically when no GEX RemoveInitialGates exists.
    first, stop = 0, min(response.size, times.size)
    if calibrated:
        first = max(0, int(calibrated.get("remove_initial", 0)))
        stop = min(stop, max(first, int(calibrated.get("remove_from", stop + 1)) - 1))
    else:
        candidates = [i for i in range(min(6, max(1, response.size - 2)))
                      if np.all(np.isfinite(response[i:i + 3]))
                      and np.all(response[i:i + 3] > 0.0)]
        if candidates:
            first = candidates[0]
    selection = slice(first, stop)
    relative_std = absolute_std / np.maximum(np.abs(response), 1e-30)
    analog_lowpass: Dict[str, Any] = {}
    receiver_filter = np.asarray(
        (calibrated or {}).get("rx_lowpass", (gex or {}).get("rx_lowpass", [])),
        dtype=float,
    )
    tib_filter = np.asarray((calibrated or {}).get("tib_lowpass", []), dtype=float)
    if (receiver_filter.size >= 2 and np.all(np.isfinite(receiver_filter[:2]))
            and np.all(receiver_filter[:2] > 0.0)):
        analog_lowpass.update({
            "receiver_damping": float(receiver_filter[0]),
            "receiver_cutoff_hz": float(receiver_filter[1]),
        })
    if (tib_filter.size >= 2 and np.all(np.isfinite(tib_filter[:2]))
            and np.all(tib_filter[:2] > 0.0)):
        analog_lowpass.update({
            "tib_order": max(1, int(round(float(tib_filter[0])))),
            "tib_cutoff_hz": float(tib_filter[1]),
        })
    transmitter = {
        "waveform": "custom", "waveform_times": waveform_times,
        "waveform_currents": waveform_currents,
        "loop_turns": loop_turns,
        "gate_windows": {"open": gate_open[selection],
                         "close": gate_close[selection],
                         "centre": times[selection]},
    }
    if analog_lowpass:
        transmitter["analog_lowpass"] = analog_lowpass
    return {"times": times[selection], "response": response[selection],
            "relative_std": relative_std[selection], "transmitter": transmitter}


def load_ttem_sounding(
    path: str, sounding: int = 0, moment: str = "LM+HM", *,
    max_relative_std: Optional[float] = None, stack_seconds: float = 2.0,
    loop_area: Optional[float] = None, gex_path: Optional[str] = None,
    tfi_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Load and stack one sounding from a raw TEMcompany tTEM survey.

    Data are normalized by measured transmitter current and the loop area from
    the selected GEX (or an explicit UI override).  A selected TFI is applied to
    each sign-corrected transient sequence before stacking.
    """
    from PyHydroGeophysX.data_processing.em1d import (
        _normalise_temcompany_moment, _temcompany_valid_channels,
    )

    selected = _normalise_temcompany_moment(moment)
    source = Path(path).resolve()
    selected_gex = _calibration_path(source, gex_path, "gex")
    selected_tfi = _calibration_path(source, tfi_path, "tfi")
    gex = _read_gex(str(selected_gex)) if selected_gex else None
    filters = _read_tfi(str(selected_tfi)) if selected_tfi else {}
    effective_area = max(
        float(loop_area if loop_area is not None
              else (gex or {}).get("loop_area", 8.0)),
        1e-6,
    )
    index = _project_index(str(source), float(stack_seconds))
    n_soundings = len(index["soundings"])
    selected_index = max(0, min(int(sounding), n_soundings - 1))
    item = index["soundings"][selected_index]
    available = [selected] if selected != "LM+HM" else ["LM", "HM"]
    moments: Dict[str, Dict[str, Any]] = {}
    for name in available:
        data = _moment_data(item, name, effective_area, gex, filters)
        try:
            times, response, rel = _temcompany_valid_channels(
                data["times"], data["response"], data["relative_std"],
                use_flags=False, max_relative_std=max_relative_std,
            )
        except ValueError:
            if selected != "LM+HM":
                raise
            continue
        transmitter = dict(data["transmitter"])
        gates = transmitter["gate_windows"]
        picked = [int(np.argmin(np.abs(gates["centre"] - value))) for value in times]
        transmitter["gate_windows"] = {
            key: np.asarray(value)[picked] for key, value in gates.items()
        }
        moments[name] = {"times": times, "response": response,
                         "relative_std": rel, "transmitter": transmitter}
    if not moments:
        raise ValueError(
            "The selected raw tTEM sounding has no usable LM or HM gates after QC."
        )
    analog_filters = {
        name: dict(item["transmitter"].get("analog_lowpass", {}))
        for name, item in moments.items()
        if item["transmitter"].get("analog_lowpass")
    }
    analog_lowpass_modelled = bool(analog_filters)
    system = {
        "source_radius": math.sqrt(effective_area / math.pi),
        "tx_rx_sep": float((gex or {}).get("tx_rx_sep", 9.28)),
        "height": float((gex or {}).get("height", 0.43)),
        "tx_height": float((gex or {}).get("tx_height", np.nan)),
        "orientation": "z", "receiver_type": "dbdt",
        "response_sign": -1.0, "loop_area": effective_area, "loop_turns": 1,
        "source_moment": 1.0, "data_scale": 1.0, "auto_scale": False,
        "gate_samples": 3,
    }
    result: Dict[str, Any] = {
        "n_soundings": n_soundings, "sounding": selected_index,
        "positions": index["positions"], "x": index["x"], "y": index["y"],
        "latitude": index["lat"], "longitude": index["lon"],
        "elevation": index["elev"],
        "heights": np.full(n_soundings, float(system["height"])),
        "line_numbers": np.asarray([row["line"] for row in index["soundings"]], dtype=int),
        "station_ids": np.asarray([f"Run{row['line']:03d}-{row['station']:05d}"
                                   for row in index["soundings"]]),
        "temcompany": True, "ttem": True, "tem_moment": selected,
        "source_format": "TEMcompany tTEM raw (SKB/SPS)",
        "coordinate_system": "local tangent plane (WGS84 GPS)",
        "system": system,
        "inversion_defaults": {"tem_moment": "LM+HM", "reference_distance": 10.0},
        "protocol": {
            "stack_seconds": float(stack_seconds), "loop_area": effective_area,
            "uniform_std": float(np.nanmedian([
                item.get("uniform_std", 0.03)
                for item in (gex or {}).get("moments", {}).values()
            ])) if (gex or {}).get("moments") else None,
            "geometry_source": (f"GEX: {selected_gex.name}" if selected_gex
                                else "standard defaults; GEX not present"),
            "gex_file": str(selected_gex) if selected_gex else "",
            "tfi_file": str(selected_tfi) if selected_tfi else "",
            "tfi_channels": sorted(filters),
            "analog_lowpass": analog_filters,
            "analog_lowpass_modelled": analog_lowpass_modelled,
        },
        "calibration": {
            "gex_applied": bool(selected_gex), "tfi_applied": bool(selected_tfi),
            "analog_lowpass": analog_filters,
            "analog_lowpass_modelled": analog_lowpass_modelled,
            "gex_path": str(selected_gex) if selected_gex else "",
            "tfi_path": str(selected_tfi) if selected_tfi else "",
        },
    }
    if selected == "LM+HM":
        result["moments"] = moments
        result["transmitter"] = {name: data["transmitter"] for name, data in moments.items()}
        reference = np.unique(np.concatenate([data["times"] for data in moments.values()]))
        result["times"] = reference
        result["response"] = np.full(reference.size, np.nan)
        result["relative_std"] = np.full(reference.size, np.nan)
    else:
        result.update(moments[selected])
        result["transmitter"] = moments[selected]["transmitter"]
    return result


__all__ = ["is_ttem_source", "load_ttem_sounding"]
