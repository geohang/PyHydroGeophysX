"""Read a TEM2Go acquisition folder directly from its ``.stb`` raw file.

A folder copied straight off the instrument holds no project database and no
XYZ export. It holds the raw stream (``.stb``), the acquisition protocol
(``.sts``), and a line file (``.lin``) recording when each survey line started
and ended. This module turns that folder into station stacks, so a survey can
be inverted before it has been imported anywhere.

The raw file is self-describing. It opens with a length-prefixed version
string, then the protocol as plain text, and the text carries three sections
this reader depends on: ``[ProcessedGateTimes]`` for the gate table,
``[InstrumentInfo]`` for the per-moment data factor, and ``[RxTxSpecs]`` for
the loop geometry. The binary body that follows is one record per raw
transient. The instrument gates before it writes, so a record already holds one
value per gate rather than a sampled waveform.

Three stages turn those records into stacks, and each is configurable through
:class:`ProcessingSettings`:

``read_records``
    decodes the binary body into telemetry and the two measured arrays.
``apply_raw_filters``
    marks gates the transmitter current or an amplitude outlier condemns.
``stack_stations``
    groups records into stations and averages them with an error estimate.

The gate table is read from ``[ProcessedGateTimes]`` when present. A protocol
that predates that section is handled by :func:`generate_gate_table`, which
rebuilds the same table from the gate design parameters and agrees with the
stored one to about one part in ten million on the surveys checked so far.
"""

from __future__ import annotations

import math
import re
import struct
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "ProcessingSettings",
    "StbRecord",
    "apply_raw_filters",
    "find_stb_files",
    "generate_gate_table",
    "gate_table",
    "legacy_spec",
    "is_stb_folder",
    "read_acquisition_folder",
    "read_line_windows",
    "read_protocol",
    "read_records",
    "stack_stations",
]

# The name marker that begins every record's auxiliary block.
_COIL_DISTANCE = b"CoilDistance"
_MARKER = struct.pack("<I", len(_COIL_DISTANCE)) + _COIL_DISTANCE

# Offsets of the fixed telemetry fields, relative to the marker. The moment is
# read as a single byte rather than as the wider field it sits in, because the
# byte above it carries the receiver channel and only two moments exist.
_TELEMETRY = (
    ("moment", -191, "<B", 1),
    ("channel", -190, "<B", 1),
    ("transmitter_on", -185, "<i", 4),
    ("gps_time_us", -176, "<Q", 8),
    ("gps_speed", -168, "<f", 4),
    ("latitude", -158, "<d", 8),
    ("longitude", -150, "<d", 8),
    ("altitude", -142, "<f", 4),
)

_MOMENT_NAME = {0: "LM", 1: "HM"}
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)

#: Value of ``CoilDistance`` meaning the range sensor returned no reading.
COIL_DISTANCE_SENTINEL = -1.0

#: A transmitter current at or below this is treated as the transmitter being
#: off, and the recorded voltages are normalised by 1 A instead.
MIN_TX_CURRENT = 0.01

#: Bit values of the per-gate filter mask this reader writes.
FILTER_NONE = 0
FILTER_TX_CURRENT = 2
FILTER_DBDT = 4
FILTER_STD = 8
FILTER_SPIKE = 16
FILTER_TOO_FEW_POINTS = 64
FILTER_SLOPE = 128


@dataclass
class ProcessingSettings:
    """Thresholds for the raw and station stages.

    The defaults match the acquisition protocol's own processing configuration
    for this instrument. Every value is a keyword argument so a survey that
    needs different treatment can say so without editing the reader.
    """

    tx_enabled: bool = True
    tx_tolerance: Mapping[str, float] = field(
        default_factory=lambda: {"LM": 22.0, "HM": 11.0})   # percent of target
    spike_enabled: bool = True
    spike_threshold: float = 3.5
    spike_window: int = 20
    dbdt_minimum: float = 1e-25
    std_enabled: bool = True
    std_maximum: Mapping[str, float] = field(
        default_factory=lambda: {"LM": 0.4, "HM": 0.5})
    std_from_time: Mapping[str, float] = field(
        default_factory=lambda: {"LM": 3e-5, "HM": 5e-5})
    slope_enabled: bool = True
    slope_threshold: float = 1.5
    slope_from_time: Mapping[str, float] = field(
        default_factory=lambda: {"LM": 2e-5, "HM": 3e-5})
    slope_back_step: Mapping[str, int] = field(
        default_factory=lambda: {"LM": 1, "HM": 0})
    minimum_points: int = 1
    station_distance: float = 2.0         # metres of travel per station
    min_raw_per_station: int = 2
    minimum_relative_error: Optional[float] = None


@dataclass
class StbRecord:
    """One raw transient, decoded."""

    index: int
    moment: int
    channel: int
    transmitter_on: int
    gps_time: datetime
    gps_speed: float
    latitude: float
    longitude: float
    altitude: float
    temperature: float
    tx_current: float
    stack_size: int
    aux: Dict[str, float]
    dbdt: np.ndarray
    relative_std: np.ndarray
    filtered: np.ndarray
    rx_tx_distance: float = float("nan")
    utm_time: Optional[datetime] = None
    x: float = float("nan")
    y: float = float("nan")
    line_number: Optional[int] = None

    @property
    def moment_name(self) -> str:
        return _MOMENT_NAME.get(self.moment, str(self.moment))


# ---------------------------------------------------------------- protocol


def _protocol_text(data: bytes) -> str:
    """The plain-text protocol that opens a raw file.

    A length-prefixed version string comes first, preceded on some releases by
    a pair of version bytes, so the prefix is located rather than assumed.
    """
    if len(data) < 32:
        raise ValueError("file is too short to be a TEM2Go raw file")
    for offset in range(0, 12):
        length = struct.unpack_from("<I", data, offset)[0]
        if not 0 < length < 64:
            continue
        label = data[offset + 4:offset + 4 + length]
        if label[:7] != b"VERSION":
            continue
        body = data[offset + 4 + length:]
        stop = body.find(b"\x00\x00\x00")
        return body[:stop if stop > 0 else 200_000].decode(
            "latin-1", errors="replace")
    raise ValueError("no length-prefixed VERSION string near the start")


def _ini_scalars(text: str) -> Dict[str, str]:
    """Every ``key = value`` line, last one winning."""
    out: Dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith((";", "[")):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        out[key.strip()] = value.strip()
    return out


def _designer(text: str) -> Dict[str, Any]:
    """The ``[ProtocolDesigner]`` comment line, parsed into a mapping."""
    line = next((l for l in text.splitlines() if "ProtocolDesigner" in l), "")
    out: Dict[str, Any] = {}
    for key, value in re.findall(r"(\w+)\s*=\s*([^,]+)", line):
        value = value.strip()
        if value in ("True", "False"):
            out[key] = value == "True"
            continue
        try:
            out[key] = float(value)
        except ValueError:
            out[key] = value
    return out


def _floats(value: Optional[str]) -> List[float]:
    if not value:
        return []
    out = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            out.append(float(item))
        except ValueError:
            pass
    return out


def read_protocol(source: Any) -> Dict[str, Any]:
    """Parse the acquisition protocol from a raw file, a ``.sts`` or text."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        data = path.read_bytes()
        text = (_protocol_text(data) if path.suffix.lower() in (".stb", ".rxstb",
                                                               ".txstb")
                else data.decode("latin-1", errors="replace"))
    elif isinstance(source, bytes):
        text = _protocol_text(source)
    else:
        text = str(source)

    scalars = _ini_scalars(text)
    designer = _designer(text)

    def number(key: str, default: float = float("nan")) -> float:
        try:
            return float(scalars[key])
        except (KeyError, ValueError):
            return default

    spec: Dict[str, Any] = {
        "text": text,
        "designer": designer,
        "scalars": scalars,
        "instrument_type": scalars.get("InstrumentType", "TEM2Go"),
        "instrument_id": scalars.get("InstrumentID", ""),
        "sample_rate_hz": number("SampleRateInMHz",
                                 designer.get("SampleFrequencyMHz", 4.0)) * 1e6,
        "uniform_std": number("UniStd", 0.03),
        "tx_area": math.prod(_floats(scalars.get("TxLoop_XYLength")) or [0.0]),
        "tx_turns": number("TxLoop_NTurns", 1.0),
        "rx_area": number("RxCoil_AreaChA"),
        "low_pass_hz": tuple(_floats(scalars.get("LPFilter_1order"))),
        "gate_shape": number("ChA_WinFunc", 1.0),
        "gate_shape_parameter": number("ChA_WinFuncPar1", 0.667),
        "gates_per_decade": number("ChA_GatePerDecade", 10.0),
        "gate_overlap": number("ChA_GateOverlap", 0.5),
    }
    for moment in ("LM", "HM"):
        spec[f"{moment}_stack_size"] = number(f"{moment}_StackSize")
        spec[f"{moment}_period_s"] = number(f"{moment}_PeriodTime") * 1e-6
        spec[f"{moment}_on_time_s"] = number(f"{moment}_OnTime") * 1e-6
        spec[f"{moment}_target_current"] = number(f"{moment}_Tx_TargetCurrent")
        spec[f"{moment}_data_factor"] = number(f"{moment}_DataFactor", 1.0)
        spec[f"{moment}_gate_time_shift"] = number(
            f"{moment}_GateTimeShift", 0.0) * 1e-6
        spec[f"{moment}_waveform_amplitude"] = np.asarray(
            _floats(scalars.get(f"{moment}_Waveform_Amplitude")), dtype=float)

    rx = _floats(scalars.get("RxCoil_XYZPos"))
    tx = _floats(scalars.get("TxLoop_XYZPos"))
    spec["rx_position"] = tuple(rx) if len(rx) == 3 else (0.0, 0.0, 0.0)
    spec["tx_position"] = tuple(tx) if len(tx) == 3 else (0.0, 0.0, 0.0)
    spec["nominal_rx_tx_distance"] = (
        math.dist(spec["rx_position"][:2], spec["tx_position"][:2])
        if rx and tx else float("nan"))

    for moment in ("LM", "HM"):
        stored = {}
        for role, key in (("open", "OpenTime"), ("centre", "CenterTime"),
                          ("close", "CloseTime")):
            values = _floats(scalars.get(f"{moment}_ChA_{key}"))
            if values:
                stored[role] = np.asarray(values, dtype=float) * 1e-6
        spec[f"{moment}_gates"] = stored or None
    return spec


def _acquisition_window(text: str, moment: str) -> Optional[Tuple[int, int]]:
    """Sample-index span of the measurement gates, skipping sign/front gates.

    The gate index arrays list the sign gate, then the front gate when the
    moment uses one, then a contiguous chain of measurement gates. The chain is
    what the output table spans, and it is found by following ``start[k] ==
    end[k-1]`` rather than by assuming how many leading entries to drop.
    """
    scalars = _ini_scalars(text)
    starts = [int(v) for v in _floats(scalars.get(f"{moment}_ChA_GateIndex0"))]
    ends = [int(v) for v in _floats(scalars.get(f"{moment}_ChA_GateIndex1"))]
    if len(starts) < 2 or len(starts) != len(ends):
        return None
    first = None
    for k in range(1, len(starts)):
        if starts[k] == ends[k - 1]:
            first = k - 1
            break
    if first is None:
        return None
    last = first
    while last + 1 < len(starts) and starts[last + 1] == ends[last]:
        last += 1
    return starts[first], ends[last]


def generate_gate_table(protocol: Mapping[str, Any],
                        moment: str) -> Optional[Dict[str, np.ndarray]]:
    """Rebuild the gate table from the gate design parameters.

    The table is a geometric series anchored on the moment's first gate time
    and ending where the acquisition window ends. Gate ``k`` opens at series
    point ``k``, is centred on ``k + 1`` and closes on ``k + span``, where
    ``span`` follows from the gate overlap. The number of steps is the span of
    the window in decades times the gates per decade, rounded.
    """
    designer = protocol.get("designer") or {}
    t_start = designer.get(f"MinimumGateTime{moment}")
    if t_start is None:
        t_start = protocol["scalars"].get(f"{moment}_ChA_TapStartTime")
    try:
        t_start = float(t_start) * 1e-6
    except (TypeError, ValueError):
        return None
    window = _acquisition_window(protocol.get("text", ""), moment)
    if not window or not t_start > 0:
        return None
    fs = float(protocol.get("sample_rate_hz") or 0.0)
    if fs <= 0:
        return None
    first_idx, last_idx = window
    origin = first_idx - t_start * fs
    t_end = (last_idx - origin) / fs
    if not t_end > t_start:
        return None
    per_decade = float(protocol.get("gates_per_decade") or 10.0)
    overlap = float(protocol.get("gate_overlap") or 0.5)
    span = max(int(round(1.0 / max(1.0 - overlap, 1e-9))), 1)
    steps = int(round(math.log10(t_end / t_start) * per_decade))
    if steps <= span:
        return None
    ratio = (t_end / t_start) ** (1.0 / steps)
    series = t_start * ratio ** np.arange(steps + 1)
    n = steps + 1 - span
    return {"open": series[:n], "centre": series[1:n + 1],
            "close": series[span:span + n]}


def gate_table(protocol: Mapping[str, Any], moment: str) -> Dict[str, np.ndarray]:
    """The moment's gate open, centre and close times, in seconds.

    The stored table wins when the protocol carries one. Otherwise the table is
    rebuilt from the design parameters.
    """
    stored = protocol.get(f"{moment}_gates")
    table = dict(stored) if stored else generate_gate_table(protocol, moment)
    if not table:
        raise ValueError(f"protocol carries no gate table for the {moment} moment")
    shift = float(protocol.get(f"{moment}_gate_time_shift") or 0.0)
    if shift:
        table = {k: v + shift for k, v in table.items()}
    if "centre" not in table and {"open", "close"} <= set(table):
        table["centre"] = np.sqrt(table["open"] * table["close"])
    return table


def legacy_spec(protocol: Mapping[str, Any]) -> Dict[str, Any]:
    """Re-key a parsed protocol into the namespace the rest of the reader uses.

    The project and XYZ paths both hand the forward layer a flat mapping whose
    names follow the acquisition protocol's own spelling. Producing the same
    mapping here means a survey read from a raw folder reaches the forward
    through exactly the code a project does, rather than through a parallel
    path that could drift away from it.
    """
    scalars = protocol.get("scalars", {})
    spec: Dict[str, Any] = {
        "GateShape": protocol.get("gate_shape"),
        "GateShapePar1": protocol.get("gate_shape_parameter"),
        "UniStd": protocol.get("uniform_std"),
        "TxLoopArea": protocol.get("tx_area"),
        "TxLoopNTurns": protocol.get("tx_turns"),
        "TxLoopXYlength": list(_floats(scalars.get("TxLoop_XYLength"))),
        "TxLoopXYZPos": list(protocol.get("tx_position", (0.0, 0.0, 0.0))),
        "RxCoilXYZPos": list(protocol.get("rx_position", (0.0, 0.0, 0.0))),
        "RxCoilArea": protocol.get("rx_area"),
        "LPFilter_1order": list(protocol.get("low_pass_hz", ())),
        "InstrumentType": protocol.get("instrument_type"),
    }
    for moment in ("LM", "HM"):
        table = gate_table(protocol, moment)
        spec[f"{moment}_GateCentreTime"] = table["centre"]
        spec[f"{moment}_GateOpenTime"] = table["open"]
        spec[f"{moment}_GateCloseTime"] = table["close"]
        spec[f"{moment}_GateTimeShift"] = protocol.get(
            f"{moment}_gate_time_shift", 0.0)
        spec[f"{moment}_DataFactor"] = protocol.get(f"{moment}_data_factor")
        spec[f"{moment}_Tx_TargetCurrent"] = protocol.get(
            f"{moment}_target_current")
        spec[f"{moment}_StackSize"] = protocol.get(f"{moment}_stack_size")
        spec[f"{moment}_PeriodTime"] = protocol.get(f"{moment}_period_s")
        # The forward reads the half-period under the name the project uses.
        spec[f"{moment}WaveformPeriod"] = protocol.get(f"{moment}_period_s")
        times = np.asarray(
            _floats(scalars.get(f"{moment}_Waveform_Time")), dtype=float) * 1e-6
        amplitudes = np.asarray(
            _floats(scalars.get(f"{moment}_Waveform_Amplitude")), dtype=float)
        times, amplitudes = _trim_waveform(
            times, amplitudes, protocol.get(f"{moment}_period_s"))
        spec[f"{moment}WaveformTime"] = times
        spec[f"{moment}WaveformAmplitude"] = amplitudes
    return spec


def _trim_waveform(times: np.ndarray, amplitudes: np.ndarray,
                   period: Optional[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Drop a leading node that only marks the start of the half cycle.

    A protocol may open the ramp with a zero-current node placed exactly half a
    period before turn-off. That node marks where the cycle begins rather than
    describing the ramp, and carrying it into the forward would add a segment
    of zero current before the real waveform starts.
    """
    if times.size != amplitudes.size or times.size < 2 or not period:
        return times, amplitudes
    boundary = -0.5 * float(period)
    if abs(times[0] - boundary) <= 1e-12 and amplitudes[0] == 0.0:
        return times[1:], amplitudes[1:]
    return times, amplitudes


# ------------------------------------------------------------------ records


def _parse_aux(data: bytes, pos: int) -> Tuple[Dict[str, float], int]:
    """Read the self-describing auxiliary block, returning it and its end."""
    entries: Dict[str, float] = {}
    while pos + 4 <= len(data):
        length = struct.unpack_from("<I", data, pos)[0]
        if not 2 <= length <= 40 or pos + 4 + length + 8 > len(data):
            break
        name = data[pos + 4:pos + 4 + length]
        if not all(65 <= b <= 122 for b in name):
            break
        entries[name.decode("ascii")] = struct.unpack_from(
            "<d", data, pos + 4 + length)[0]
        pos += 4 + length + 8
    return entries, pos


def read_records(source: Any,
                 protocol: Optional[Mapping[str, Any]] = None
                 ) -> Tuple[List[StbRecord], Dict[str, Any]]:
    """Decode every raw transient in a ``.stb`` file.

    Returns the records in file order together with the protocol they were
    read against.
    """
    data = Path(source).read_bytes() if isinstance(source, (str, Path)) else source
    spec = dict(protocol) if protocol else read_protocol(data)

    gates = {}
    for moment, name in _MOMENT_NAME.items():
        try:
            gates[moment] = len(gate_table(spec, name)["centre"])
        except ValueError:
            gates[moment] = 0

    offsets = []
    pos = data.find(_MARKER)
    while pos != -1:
        offsets.append(pos)
        pos = data.find(_MARKER, pos + 1)

    records: List[StbRecord] = []
    for index, marker in enumerate(offsets):
        values = {}
        ok = True
        for name, offset, fmt, size in _TELEMETRY:
            at = marker + offset
            if at < 0 or at + size > len(data):
                ok = False
                break
            values[name] = struct.unpack_from(fmt, data, at)[0]
        if not ok:
            continue
        moment = int(values["moment"])
        n = gates.get(moment, 0)
        if n <= 0:
            continue
        aux, end = _parse_aux(data, marker)
        if end + 44 + 8 * n > len(data):
            continue
        temperature = struct.unpack_from("<B", data, end + 22)[0]
        current = struct.unpack_from("<f", data, end + 36)[0]
        stack = struct.unpack_from("<i", data, end + 40)[0]
        stored = np.frombuffer(data, "<f4", n, end + 44).astype(np.float64)
        rel_std = np.frombuffer(data, "<f4", n, end + 44 + 4 * n).astype(np.float64)

        name = _MOMENT_NAME[moment]
        factor = float(spec.get(f"{name}_data_factor") or 1.0)
        area = float(spec.get("tx_area") or 0.0)
        turns = float(spec.get("tx_turns") or 1.0)
        drive = current if abs(current) >= MIN_TX_CURRENT else 1.0
        denominator = drive * turns * area
        dbdt = (stored * factor / denominator if denominator
                else np.full(n, np.nan))

        gps_time = _EPOCH + timedelta(microseconds=int(values["gps_time_us"]))
        half_window = 0.5 * float(spec.get(f"{name}_stack_size") or 0.0) * \
            float(spec.get(f"{name}_period_s") or 0.0)

        distance = aux.get("CoilDistance", float("nan"))
        if distance == COIL_DISTANCE_SENTINEL:
            distance = aux.get("CoilDistanceBField", float("nan"))
            if distance == COIL_DISTANCE_SENTINEL:
                distance = float("nan")

        records.append(StbRecord(
            index=index, moment=moment, channel=int(values["channel"]),
            transmitter_on=int(values["transmitter_on"]), gps_time=gps_time,
            gps_speed=float(values["gps_speed"]),
            latitude=float(values["latitude"]),
            longitude=float(values["longitude"]),
            altitude=float(values["altitude"]), temperature=float(temperature),
            tx_current=float(current), stack_size=int(stack), aux=aux,
            dbdt=dbdt.astype(np.float32), relative_std=rel_std,
            filtered=np.zeros(n, dtype=int), rx_tx_distance=float(distance),
            utm_time=gps_time + timedelta(seconds=half_window)))

    _add_local_coordinates(records)
    return records, spec


def _add_local_coordinates(records: Sequence[StbRecord]) -> None:
    """Attach local metric coordinates from a tangent plane at the centroid.

    The raw file carries geographic coordinates and no projection. A tangent
    plane keeps plan distances metric, which is all the station grouping and
    the along-line coordinate need, without adding a projection dependency.
    """
    if not records:
        return
    lat = np.asarray([r.latitude for r in records], dtype=float)
    lon = np.asarray([r.longitude for r in records], dtype=float)
    good = np.isfinite(lat) & np.isfinite(lon) & ((lat != 0) | (lon != 0))
    if not good.any():
        return
    lat0 = float(np.mean(lat[good]))
    lon0 = float(np.mean(lon[good]))
    radius = 6_371_000.0
    x = np.deg2rad(lon - lon0) * radius * math.cos(math.radians(lat0))
    y = np.deg2rad(lat - lat0) * radius
    for record, xi, yi in zip(records, x, y):
        record.x = float(xi)
        record.y = float(yi)


# -------------------------------------------------------------------- lines


def read_line_windows(folder: Any) -> List[Dict[str, Any]]:
    """Survey lines and their time windows, from the folder's ``.lin`` file.

    Each line contributes a start and an end row. A line whose end is missing
    stays open, which is what an interrupted acquisition leaves behind.
    """
    folder = Path(folder)
    files = sorted(folder.glob("*.lin"))
    if not files:
        return []
    pattern = re.compile(
        r"^(\d{2})-(\d{2})-(\d{4})\s+(\d{2}):(\d{2}):(\d{2})\s+(\d+)\s+"
        r"([-\d.]+)\s+([-\d.]+)\s*!\s*(\w+)")
    opened: Dict[int, Dict[str, Any]] = {}
    lines: List[Dict[str, Any]] = []
    for path in files:
        for raw in path.read_text(errors="replace").splitlines():
            match = pattern.match(raw.strip())
            if not match:
                continue
            day, month, year, hh, mm, ss, number, lat, lon, kind = match.groups()
            stamp = datetime(int(year), int(month), int(day), int(hh), int(mm),
                             int(ss), tzinfo=timezone.utc)
            number = int(number)
            if kind.lower() == "start":
                entry = {"line_number": number, "start": stamp, "end": None,
                         "latitude": float(lat), "longitude": float(lon)}
                lines.append(entry)
                opened[number] = entry
            elif number in opened:
                opened.pop(number)["end"] = stamp
    lines.sort(key=lambda item: item["start"])
    return lines


def assign_lines(records: Sequence[StbRecord],
                 windows: Sequence[Mapping[str, Any]]) -> int:
    """Tag each record with the survey line whose window contains its time.

    Returns the number of records that fell inside a line. Records outside
    every window keep ``line_number`` of ``None`` and take no part in stacking,
    which is what happens to the instrument idling between lines.
    """
    tagged = 0
    for record in records:
        stamp = record.utm_time or record.gps_time
        for window in windows:
            end = window.get("end")
            if window["start"] <= stamp and (end is None or stamp <= end):
                record.line_number = int(window["line_number"])
                tagged += 1
                break
    return tagged


# ------------------------------------------------------------------ filters


def _median(values: np.ndarray) -> float:
    """Midpoint of the sorted values, and zero for an empty set."""
    n = values.size
    if n == 0:
        return 0.0
    ordered = np.sort(values)
    half = n // 2
    if n % 2:
        return float(ordered[half])
    return 0.5 * (float(ordered[half - 1]) + float(ordered[half]))


def _filter_tx_current(records: Sequence[StbRecord], protocol: Mapping[str, Any],
                       settings: ProcessingSettings) -> None:
    """Condemn a whole record whose drive current missed its target.

    The tolerance is a percentage of the moment's target current and applies
    either side of it.
    """
    if not settings.tx_enabled:
        for record in records:
            record.filtered &= ~FILTER_TX_CURRENT
        return
    for record in records:
        name = record.moment_name
        target = protocol.get(f"{name}_target_current")
        tolerance = settings.tx_tolerance.get(name)
        if not target or not np.isfinite(target) or tolerance is None:
            continue
        allowed = abs(target) * float(tolerance) / 100.0
        if abs(record.tx_current - target) > allowed:
            record.filtered |= FILTER_TX_CURRENT
        else:
            record.filtered &= ~FILTER_TX_CURRENT


def _filter_spikes(records: Sequence[StbRecord],
                   settings: ProcessingSettings) -> None:
    """Flag a gate whose value stands out from its neighbours in the stream.

    For each gate the comparison runs over a window of records centred on the
    one being tested, using the modified Z score of Iglewicz and Hoaglin,
    ``0.6745 * (x - median) / MAD``. Only records whose gate is still usable
    enter the median and the deviation, and the pass writes into the same flags
    it reads, so a record dropped at one gate leaves the windows of the records
    that follow it at that gate.
    """
    if not settings.spike_enabled or len(records) < 2:
        for record in records:
            record.filtered &= ~FILTER_SPIKE
        return
    n = len(records)
    width = min(max(2, int(settings.spike_window)), n)
    values = np.vstack([r.dbdt for r in records]).astype(np.float32)
    flags = np.vstack([r.filtered for r in records])
    n_gates = values.shape[1]
    for gate in range(n_gates):
        column = values[:, gate]
        for j in range(n):
            start = min(max(j - width // 2, 0), n - width)
            block = slice(start, start + width)
            usable = flags[block, gate] == 0
            sample = column[block][usable]
            centre = _median(sample)
            spread = _median(np.abs(sample.astype(np.float64)
                                    - centre).astype(np.float32))
            if not spread > 0.0:
                spread = 1e-12
            score = abs(abs(float(column[j]) - centre) / spread * 0.6745)
            if score > settings.spike_threshold:
                flags[j, gate] |= FILTER_SPIKE
            else:
                flags[j, gate] &= ~FILTER_SPIKE
    for record, row in zip(records, flags):
        record.filtered = row


def apply_raw_filters(records: Sequence[StbRecord], protocol: Mapping[str, Any],
                      settings: Optional[ProcessingSettings] = None) -> None:
    """Mark the gates that the raw-stage tests condemn, in place.

    The spike test runs per receiver channel and transmitter moment over the
    whole stream, because that is the sequence a moving instrument samples.
    """
    settings = settings or ProcessingSettings()
    _filter_tx_current(records, protocol, settings)
    for record in records:
        below = np.abs(record.dbdt) < settings.dbdt_minimum
        record.filtered = np.where(below, record.filtered | FILTER_DBDT,
                                   record.filtered & ~FILTER_DBDT)
    groups: Dict[Tuple[int, int], List[StbRecord]] = {}
    for record in records:
        groups.setdefault((record.channel, record.moment), []).append(record)
    for group in groups.values():
        _filter_spikes(group, settings)


# ----------------------------------------------------------------- stacking


def _nanmean(values: Iterable[float], fallback: float = float("nan")) -> float:
    """Mean of the finite values, or the fallback when none are finite.

    Every reading of the range sensor can fail on a station, and the recorded
    failure is a sentinel rather than a distance, so a station with no valid
    reading falls back to the protocol's nominal separation.
    """
    data = np.asarray([v for v in values], dtype=float)
    good = data[np.isfinite(data)]
    return float(good.mean()) if good.size else float(fallback)


def _mean_and_stderr(values: np.ndarray, minimum_relative: float
                     ) -> Tuple[float, float]:
    """Mean of the kept members and its relative standard error.

    A single member has no scatter to measure, so its error is the uniform term
    taken twice in quadrature. A mean that is not finite, or is at the noise
    floor, returns zero for both so the gate can be dropped downstream.
    """
    n = values.size
    if n == 0:
        return 0.0, 0.0
    if n == 1:
        return float(values[0]), math.sqrt(minimum_relative ** 2 * 2.0)
    data = values.astype(np.float64)
    mean = float(np.mean(data))
    if not np.isfinite(mean) or abs(mean) <= 1e-25:
        return 0.0, 0.0
    scatter = math.sqrt(float(np.sum((data - mean) ** 2)) / n)
    relative = scatter / math.sqrt(n) / abs(mean)
    return mean, math.sqrt(relative ** 2 + minimum_relative ** 2)


def _group_by_travel(records: Sequence[StbRecord], threshold: float,
                     minimum: int) -> List[List[StbRecord]]:
    """Split a moment's records into stations by distance travelled.

    A station collects records until the instrument has moved further than the
    threshold from where the station began. This reproduces the station count
    and spacing of an imported survey closely, but the boundary between two
    stations is placed independently, so membership near a boundary can differ
    by a record from what an import of the same folder would produce.
    """
    stations: List[List[StbRecord]] = []
    current: List[StbRecord] = []
    for record in records:
        if not current:
            current = [record]
            continue
        first = current[0]
        moved = math.hypot(record.x - first.x, record.y - first.y)
        if np.isfinite(moved) and moved > threshold:
            stations.append(current)
            current = [record]
        else:
            current.append(record)
    if current:
        stations.append(current)
    return [s for s in stations if len(s) >= minimum]


def _filter_station_gates(dbdt: np.ndarray, std: np.ndarray, used: np.ndarray,
                          times: np.ndarray, moment: str,
                          settings: ProcessingSettings) -> np.ndarray:
    """The per-gate mask for one stacked curve."""
    n = dbdt.size
    flags = np.zeros(n, dtype=int)

    if settings.std_enabled:
        start = max(0.0, float(settings.std_from_time.get(moment, 0.0)))
        maximum = float(settings.std_maximum.get(moment, np.inf))
        first = next((i for i in range(n) if times[i] > start), n)
        latched = False
        for gate in range(first, n):
            if latched or std[gate] > maximum:
                flags[gate] |= FILTER_STD
                latched = True

    if settings.slope_enabled and n >= 3:
        with np.errstate(divide="ignore", invalid="ignore"):
            log_y = np.log10(np.abs(dbdt))
            log_t = np.log10(np.abs(times))
            slope = np.diff(log_y) / np.diff(log_t)
            change = np.abs(np.diff(slope))
        start = max(0.0, float(settings.slope_from_time.get(moment, 0.0)))
        first = 0
        while first < n and times[first] <= start:
            first += 1
        back = int(settings.slope_back_step.get(moment, 0))
        cut = -1
        for k in range(max(0, first - 1), change.size):
            if np.isfinite(change[k]) and change[k] > settings.slope_threshold:
                cut = max(0, (k + 1) - back)
                break
        if cut >= 0:
            flags[cut + 1:] |= FILTER_SLOPE

    flags |= np.where(used <= settings.minimum_points, FILTER_TOO_FEW_POINTS, 0)
    flags |= np.where(np.abs(dbdt) < settings.dbdt_minimum, FILTER_DBDT, 0)
    return flags


def stack_stations(records: Sequence[StbRecord], protocol: Mapping[str, Any],
                   settings: Optional[ProcessingSettings] = None
                   ) -> List[Dict[str, Any]]:
    """Group filtered records into stations and average them.

    Records outside every survey line are skipped, matching what an import
    does with the stream recorded between lines.
    """
    settings = settings or ProcessingSettings()
    floor = settings.minimum_relative_error
    if floor is None or not np.isfinite(floor) or floor <= 0:
        floor = float(protocol.get("uniform_std") or 0.03)
    floor = max(floor, float(protocol.get("uniform_std") or 0.03))

    buckets: Dict[Tuple[Any, int, int], List[StbRecord]] = {}
    for record in records:
        if record.line_number is None:
            continue
        buckets.setdefault(
            (record.line_number, record.channel, record.moment), []).append(record)

    stations: List[Dict[str, Any]] = []
    for (line, channel, moment), group in sorted(buckets.items(),
                                                 key=lambda kv: kv[0][:2]):
        group.sort(key=lambda r: r.index)
        name = _MOMENT_NAME.get(moment, str(moment))
        times = gate_table(protocol, name)["centre"]
        for number, members in enumerate(
                _group_by_travel(group, settings.station_distance,
                                 settings.min_raw_per_station), start=1):
            n_gates = members[0].dbdt.size
            dbdt = np.zeros(n_gates)
            std = np.zeros(n_gates)
            used = np.zeros(n_gates, dtype=int)
            for gate in range(n_gates):
                kept = np.asarray([m.dbdt[gate] for m in members
                                   if m.filtered[gate] == 0], dtype=np.float32)
                mean, error = _mean_and_stderr(kept, floor)
                dbdt[gate] = mean
                std[gate] = error
                used[gate] = kept.size
            flags = _filter_station_gates(dbdt, std, used, times, name, settings)
            stations.append({
                "line_number": line, "station_number": number, "channel": channel,
                "moment": moment, "moment_name": name,
                "times": times, "dbdt": dbdt, "relative_std": std,
                "used_points": used, "filtered": flags,
                "n_members": len(members),
                "member_indices": [m.index for m in members],
                "tx_current": float(np.mean([m.tx_current for m in members])),
                "x": float(np.mean([m.x for m in members])),
                "y": float(np.mean([m.y for m in members])),
                "latitude": float(np.mean([m.latitude for m in members])),
                "longitude": float(np.mean([m.longitude for m in members])),
                "elevation": float(np.mean([m.altitude for m in members])),
                "rx_tx_distance": _nanmean(
                    [m.rx_tx_distance for m in members],
                    protocol.get("nominal_rx_tx_distance", float("nan"))),
                "time": members[len(members) // 2].utm_time,
            })
    return stations


# ------------------------------------------------------------------- folder


def find_stb_files(folder: Any) -> List[Path]:
    """Every raw stream file under an acquisition folder, in name order.

    The receiver and transmitter companions are excluded; only the stream whose
    records are the soundings is returned.
    """
    folder = Path(folder)
    if folder.is_file():
        return [folder] if folder.suffix.lower() == ".stb" else []
    return sorted(p for p in folder.rglob("*.stb")
                  if not p.name.lower().endswith(("_rx.stb", "_tx.stb")))


def is_stb_folder(path: Any) -> bool:
    """True when the path is, or contains, a readable raw stream."""
    try:
        return bool(find_stb_files(path))
    except OSError:
        return False


def read_acquisition_folder(folder: Any,
                            settings: Optional[ProcessingSettings] = None
                            ) -> Dict[str, Any]:
    """Read a folder straight off the instrument into station stacks.

    Returns the protocol, the decoded records, the survey lines and the
    stations, so a caller can inspect any stage rather than only the result.
    """
    settings = settings or ProcessingSettings()
    folder = Path(folder)
    files = find_stb_files(folder)
    if not files:
        raise FileNotFoundError(f"no .stb raw file under {folder}")

    root = folder if folder.is_dir() else folder.parent
    for candidate in (root, *root.parents[:3]):
        windows = read_line_windows(candidate)
        if windows:
            break
    else:
        windows = []

    records: List[StbRecord] = []
    protocol: Optional[Dict[str, Any]] = None
    for path in files:
        part, spec = read_records(path)
        protocol = protocol or spec
        offset = len(records)
        for record in part:
            record.index += offset
        records.extend(part)
    assert protocol is not None

    _add_local_coordinates(records)
    tagged = assign_lines(records, windows)
    apply_raw_filters(records, protocol, settings)
    stations = stack_stations(records, protocol, settings)
    return {
        "folder": str(folder),
        "files": [str(p) for p in files],
        "protocol": protocol,
        "records": records,
        "lines": windows,
        "stations": stations,
        "n_records": len(records),
        "n_records_on_a_line": tagged,
        "n_stations": len(stations),
    }
