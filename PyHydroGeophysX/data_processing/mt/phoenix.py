# The bit assignments and gain values in _board_gain and _low_pass are those of
# PhoenixGeoPy and of mth5's Phoenix reader, which implement the same rules.
# Both are distributed under the MIT licence, below. PyHydroGeophysX's changes
# are under the package's Apache-2.0 licence.
#
# MIT License
#
# Copyright (c) 2021 Phoenix Geophysics Ltd. (PhoenixGeoPy)
# Copyright (c) 2020 JP (mth5)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Phoenix Geophysics MTU-5C, MTU-5P, MTU-8A, RXU-8A and MTU-2C recordings.

A recording is a folder ``<serial>_<yyyy-mm-dd-hhmmss>`` holding
``recmeta.json`` and ``config.json`` and one numbered folder per channel; the
channel map in ``recmeta.json`` says which connector (E1, E2, H1, H2, H3...)
each folder holds. Each folder has the channel's files, fragmented in time:

- ``.bin`` - the native 24 kS/s stream: 64-byte frames of twenty 24-bit
  big-endian samples and a little-endian footer whose 28-bit counter numbers
  the frame from the start of the recording;
- ``.td_150``, ``.td_30`` - decimated continuous float32 volts;
- ``.td_24k``, ``.td_2400`` and other rates - decimated segmented files,
  bursts each with its own 32-byte subheader (GPS time stamp, sample count).

All of them start with the 128-byte header of Phoenix's specification, which
this reader follows. The hardware-configuration bytes of that header set a
board's gains and low-pass corner; :func:`_board_gain` and :func:`_low_pass`
read them with the rules of PhoenixGeoPy (Phoenix Geophysics, 2021) and mth5.

Samples come out in volts at the instrument input: native counts are scaled
by the A/D converter's +-5 V over 2^23 and divided by the board's total gain,
which is what the decimated files already hold. The responses come from
Phoenix's calibration files: ``<receiver>_rxcal.json`` (each channel's
anti-alias filter, one table per low-pass setting) and
``<coil>_*.scal.json`` (each coil, mV/nT).

Phoenix time stamps count GPS seconds from 1970; they are converted to UTC
with the receiver's own leap-second count. Files written before firmware 2.0
(native version < 4, decimated < 3) stamp one second early, which is undone.
The stamps do not include the decimation filters' delay (about 15 ms at
150 S/s against the 24 kS/s bursts): every channel of a stream shares it, so
it cancels in transfer functions, but streams of different rates are offset
by it.

Phoenix Geophysics (2021). Time series file specifications for MTU-8A,
RXU-8A, MTU-5C, MTU-2C and MTU-5D. Document DAA09, version 210915.
"""

from __future__ import annotations

import json
import re
import struct
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from .timeseries import (
    Channel,
    ResponseStage,
    TimeSeriesRun,
    as_datetime64,
    assemble_runs,
    contiguous_blocks,
    dipole_stage,
    gps_leap_seconds,
    normalize_component,
    shifted,
    unit_stage,
)

_FILE_NAME = re.compile(r"^(\d+)_([0-9A-Fa-f]{8})_([0-9A-Fa-f]+)_([0-9A-Fa-f]{8})\.(bin|td_\w+)$")
#: Decimated files that are continuous; every other ``td_`` rate is segmented.
_CONTINUOUS = {"td_150", "td_30"}
_AD_RANGE_VOLTS = 5.0
_SEGMENT_HEADER = 32


def _text(raw: bytes) -> str:
    return raw.split(b"\x00")[0].decode("latin-1").strip()


def read_phoenix_header(path: Any) -> Dict[str, Any]:
    """The 128-byte header of a Phoenix time-series file, decoded."""
    with Path(path).open("rb") as stream:
        raw = stream.read(128)
    if len(raw) < 128:
        raise ValueError(f"{Path(path).name} is shorter than a Phoenix header")
    u = lambda fmt, offset: struct.unpack_from("<" + fmt, raw, offset)[0]
    header = {
        "file_type": raw[0], "file_version": raw[1], "header_length": u("H", 2),
        "instrument_type": _text(raw[4:12]), "instrument_serial": _text(raw[12:20]),
        "recording_id": u("I", 20), "channel_id": raw[24], "file_sequence": u("I", 25),
        "fragment_period": u("H", 29), "board_model": raw[31:39].decode("latin-1"),
        "board_serial": _text(raw[39:47]), "hardware": list(raw[51:59]),
        "sample_rate": u("H", 59) * 10.0 ** struct.unpack_from("<b", raw, 61)[0],
        "bytes_per_sample": raw[62], "longitude": u("f", 71), "latitude": u("f", 75),
        "elevation": u("f", 79), "battery_volts": u("H", 105) / 1000.0,
    }
    if header["file_type"] == 1:
        frame = u("I", 63)
        header.update(frame_size=frame & 0x00FFFFFF, footer_size=frame >> 24,
                      frame_rollovers=u("H", 69), saturated_frames=u("H", 101),
                      missing_frames=u("H", 103))
    else:
        header["decimation_scheme"] = u("I", 119)
    return header


def _board_gain(header: Dict[str, Any]) -> Tuple[float, str]:
    """The board's total gain from instrument input to A/D, and the channel type."""
    hw = header["hardware"]
    model = header["board_model"]
    main_model, revision = model[:5], model[6:7]
    old_banks = main_model in ("BCM01", "BCM03") or model[:7] == "BCM05-A"
    kind = "E" if hw[1] & 0x08 else "H"
    preamp = 1.0
    if kind == "E" and hw[0] & 0x10:
        if main_model in ("BCM01", "BCM03"):
            preamp = 8.0 if revision == "L" else 4.0
        else:
            preamp = 4.0 if model[:7] == "BCM05-A" else 8.0
    main = {0x00: 1.0, 0x04: 4.0, 0x08: 16.0 if old_banks else 6.0,
            0x0C: 32.0 if old_banks else 8.0}[hw[0] & 0x0C]
    intrinsic = 1.0 if (kind == "H" and hw[1] & 0x01) else 0.5
    attenuator = 1.0
    if kind == "E" and hw[4] & 0x01:
        attenuator = 0.1 if old_banks else 523.0 / 5223.0
    return main * preamp * attenuator * intrinsic, kind


def _low_pass(header: Dict[str, Any]) -> float:
    hw0, board = header["hardware"][0], header["board_model"][:5]
    fast = board in ("BCM03", "BCM06")
    if not hw0 & 0x80:
        return 17800.0 if fast else 10000.0
    return {3: 10.0, 2: 1000.0 if fast else 100.0, 1: 10000.0 if fast else 1000.0}.get(hw0 & 0x03, float("nan"))


def _native_blocks(path: Path, header: Dict[str, Any], start_utc: np.datetime64,
                   scale: float) -> Tuple[List[Tuple[np.datetime64, np.ndarray]], int]:
    """Gap-free blocks of a ``.bin`` file, placed by their frame counters."""
    frame_size = header["frame_size"] or 64
    raw = np.frombuffer(path.read_bytes(), dtype=np.uint8, offset=header["header_length"])
    n_frames = raw.size // frame_size
    frames = raw[: n_frames * frame_size].reshape(n_frames, frame_size)
    footer_size = header["footer_size"] or 4
    per_frame = (frame_size - footer_size) // 3
    b = frames[:, : per_frame * 3].reshape(-1, 3).astype(np.int32)
    counts = (b[:, 0] << 16) | (b[:, 1] << 8) | b[:, 2]
    counts -= (counts & 0x800000) << 1
    data = (counts * scale).astype(np.float32).reshape(n_frames, per_frame)
    footer = np.ascontiguousarray(frames[:, frame_size - 4:]).view("<u4").ravel()
    counter = (footer & 0x0FFFFFFF).astype(np.int64)
    counter += np.int64(header["frame_rollovers"]) << 28
    counter += np.concatenate([[0], np.cumsum(np.diff(counter) < 0)]) << 28
    saturated = int(np.count_nonzero((footer >> 28) & 0x7))
    breaks = np.flatnonzero(np.diff(counter) != 1) + 1
    blocks = []
    rate = header["sample_rate"]
    for first, last in zip(np.r_[0, breaks], np.r_[breaks, n_frames]):
        if last > first:
            offset = counter[first] * per_frame / rate
            blocks.append((shifted(start_utc, offset), data[first:last].ravel()))
    return blocks, saturated


def _decimated_blocks(path: Path, header: Dict[str, Any], start_utc: np.datetime64,
                      continuous: bool, leap: int, late: float) -> List[Tuple[np.datetime64, np.ndarray]]:
    raw = path.read_bytes()
    offset = header["header_length"]
    if continuous:
        data = np.frombuffer(raw, dtype="<f4", offset=offset, count=(len(raw) - offset) // 4)
        # Decimation filters need the recording's first second; files then follow each other.
        sequence = max(int(header["file_sequence"]), 1)
        seconds = 1.0 if sequence == 1 else (sequence - 1) * float(header["fragment_period"])
        return [(shifted(start_utc, seconds), data)]
    blocks = []
    while offset + _SEGMENT_HEADER <= len(raw):
        stamp, n = struct.unpack_from("<II", raw, offset)
        offset += _SEGMENT_HEADER
        n = min(int(n), (len(raw) - offset) // 4)
        if n <= 0:
            break
        when = np.datetime64("1970-01-01T00:00:00", "ns") + np.timedelta64(int((stamp + late - leap) * 1e9), "ns")
        blocks.append((when, np.frombuffer(raw, dtype="<f4", offset=offset, count=n)))
        offset += 4 * n
    return blocks


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, ValueError):
        return {}


def read_phoenix_calibration(path: Any) -> Dict[str, Any]:
    """A Phoenix ``rxcal.json``, ``scal.json`` or binary ``.scal``: each channel's response tables.

    Returns ``{"kind": "receiver" | "sensor", "instrument_serial", "sensor_serial",
    "model", "channels": {tag: [ResponseStage, ...]}}``; a receiver file has one
    table per low-pass setting.
    """
    source = Path(path)
    raw = source.read_bytes()
    if source.suffix.lower() == ".scal" and not raw.lstrip()[:1] == b"{":
        return _read_binary_scal(source, raw)
    content = _load_json(source)
    kind = "sensor" if "sensor" in str(content.get("file_type", "")).lower() else "receiver"
    channels: Dict[str, List[ResponseStage]] = {}
    for entry in content.get("cal_data", []):
        tag = str(entry.get("tag", "")).upper()
        for table in entry.get("chan_data", []):
            stage = ResponseStage.from_amplitude_phase(
                f"phoenix_{kind}_{tag.lower()}", "nT" if kind == "sensor" else "V",
                "mV" if kind == "sensor" else "V", table["freq_Hz"], table["magnitude"],
                table["phs_deg"], metadata={"source_file": str(source),
                                            "max_frequency": float(np.max(table["freq_Hz"]))})
            channels.setdefault(tag, []).append(stage)
    return {"kind": kind, "instrument_serial": str(content.get("inst_serial", "")),
            "sensor_serial": str(content.get("sensor_serial", "")),
            "model": str(content.get("sensor_type", "") or content.get("instrument_type", "")),
            "channels": channels, "source_file": str(source)}


#: A binary ``.scal`` record: frequency (Hz), the response's real and
#: imaginary parts (mV/nT), and 20 bytes the files leave unset.
_SCAL_RECORD = 32


def _read_binary_scal(source: Path, raw: bytes) -> Dict[str, Any]:
    """A coil calibration in Phoenix's binary ``.scal`` layout.

    Phoenix documents only the JSON form; this layout was read off the files
    (``<serial>_<time>.scal``): the number of frequencies is a uint32 at byte
    8, the sensor's serial is text at byte 59 and its model at byte 133, and
    the file ends with one 32-byte record per frequency - float32 frequency,
    then the real and imaginary parts of the response in mV/nT. A coil's table
    shows which part is which: below its corner the response is imaginary and
    grows with frequency, above it real and flat.
    """
    count = struct.unpack_from("<I", raw, 8)[0] if len(raw) >= 12 else 0
    start = len(raw) - _SCAL_RECORD * count
    if count < 2 or start < 64:
        raise ValueError(f"{source.name} is not a Phoenix sensor calibration")
    table = np.frombuffer(raw, dtype="<f4", count=count * _SCAL_RECORD // 4, offset=start)
    table = table.reshape(count, _SCAL_RECORD // 4).astype(float)
    frequency = table[:, 0]
    if not (np.all(np.isfinite(table[:, :3])) and np.all(frequency > 0) and np.all(np.diff(frequency) > 0)):
        raise ValueError(f"{source.name}: the calibration table does not read as one "
                         "(frequencies not positive and increasing)")
    serial = _text(raw[59:69])
    if not re.fullmatch(r"[0-9A-Za-z-]+", serial):
        serial = source.name.split("_")[0].split(".")[0]
    stage = ResponseStage("phoenix_sensor_h", "nT", "mV", frequency=frequency,
                          values=table[:, 1] + 1j * table[:, 2],
                          metadata={"source_file": str(source), "max_frequency": float(frequency.max())})
    return {"kind": "sensor", "instrument_serial": "", "sensor_serial": serial,
            "model": _text(raw[133:149]), "channels": {"H": [stage]}, "source_file": str(source)}


def _calibration_time(cal: Dict[str, Any]) -> int:
    """When a calibration was made, from its file name ``<serial>_<hex time>``; 0 if unknown."""
    parts = Path(cal["source_file"]).name.split(".")[0].split("_")
    try:
        return int(parts[1], 16) if len(parts) > 1 else 0
    except ValueError:
        return 0


#: Calibration files: Phoenix's JSON and its binary coil tables.
_CALIBRATION_PATTERNS = ("*rxcal.json", "*scal.json", "*.scal")


def _calibration_files(recording: Path, calibration: Any) -> List[Path]:
    """Calibration files given (files, or folders searched through), or found near the recording.

    Without ``calibration``, the recording's folder and the two above it are
    searched, with any folder in them named for calibrations (``Calibration``,
    ``cal_files``...), which is where a survey keeps its coils' tables.
    """
    if calibration is not None:
        items = [calibration] if isinstance(calibration, (str, Path)) else list(calibration)
        found = []
        for item in map(Path, items):
            found += ([p for pattern in _CALIBRATION_PATTERNS for p in sorted(item.rglob(pattern))]
                      if item.is_dir() else [item])
        return list(dict.fromkeys(found))
    found = []
    for folder in (recording, recording.parent, recording.parent.parent):
        if not folder.is_dir():
            continue
        found += [p for pattern in _CALIBRATION_PATTERNS for p in sorted(folder.glob(pattern))]
        for sub in sorted(p for p in folder.iterdir()
                          if p.is_dir() and re.search(r"(^|[^a-z])cal", p.name.lower())):
            found += [p for pattern in _CALIBRATION_PATTERNS for p in sorted(sub.rglob(pattern))]
    return list(dict.fromkeys(found))


def phoenix_coil_calibrations(path: Any, calibration: Any = None) -> List[Dict[str, Any]]:
    """The coil calibrations a Phoenix recording would use: ``{"serial", "model", "file"}`` each.

    ``calibration`` is as :func:`read_phoenix` takes it. One entry per serial,
    its newest file. Files that do not read as calibrations are left out.
    """
    newest: Dict[str, Dict[str, Any]] = {}
    for file in _calibration_files(_recording_folder(Path(path)), calibration):
        try:
            cal = read_phoenix_calibration(file)
        except (OSError, ValueError, KeyError, TypeError):
            continue
        serial = cal["sensor_serial"] or Path(file).name.split("_")[0].split(".")[0]
        if cal["kind"] != "sensor" or not serial:
            continue
        if serial not in newest or _calibration_time(cal) > _calibration_time(newest[serial]["cal"]):
            newest[serial] = {"serial": serial, "model": cal.get("model", ""), "file": str(file), "cal": cal}
    return [{k: v for k, v in item.items() if k != "cal"} for _, item in sorted(newest.items())]


def _match_low_pass(stages: List[ResponseStage], low_pass: float) -> Optional[ResponseStage]:
    """The table for a low-pass setting: the narrowest one reaching past it."""
    if not stages:
        return None
    ordered = sorted(stages, key=lambda s: s.metadata["max_frequency"])
    if not np.isfinite(low_pass):
        return ordered[-1]
    for stage in ordered:
        if stage.metadata["max_frequency"] >= low_pass:
            return stage
    return ordered[-1]


def _recording_folder(path: Path) -> Path:
    if path.is_file():
        return path.parent.parent
    if path.name.isdigit() and not (path / "recmeta.json").exists():
        return path.parent
    return path


def is_phoenix_recording(path: Any) -> bool:
    source = Path(path)
    if source.is_file():
        return bool(_FILE_NAME.match(source.name))
    if not source.is_dir():
        return False
    folder = _recording_folder(source)
    if (folder / "recmeta.json").exists():
        return True
    return any(_FILE_NAME.match(p.name) for sub in folder.iterdir() if sub.is_dir()
               for p in list(sub.iterdir())[:3])


def _channel_layout(recmeta: Dict[str, Any], config: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    """Each channel folder's tag and settings, from the receiver's metadata."""
    chans = recmeta.get("chconfig", {}).get("chans", [])
    by_tag = {str(c.get("tag", "")).upper(): c for c in chans}
    mapping = recmeta.get("channel_map", {}).get("mapping", [])
    layout = {}
    if mapping:
        for entry in mapping:
            tag = str(entry.get("tag", "")).upper()
            layout[int(entry["idx"])] = dict(by_tag.get(tag, {}), tag=tag)
    else:
        for index, c in enumerate(chans):
            layout[index] = dict(c, tag=str(c.get("tag", "")).upper())
    for block in config.get("config", []):
        for c in block.get("chanconfig", []):
            tag = str(c.get("tag", "")).upper()
            for entry in layout.values():
                if entry["tag"] == tag and "dipole" in c:
                    entry["dipole"] = c["dipole"]
    return layout


def read_phoenix(path: Any, *, sample_rate: Union[None, float, Sequence[float]] = None,
                 components: Optional[Iterable[str]] = None, start: Any = None, end: Any = None,
                 calibration: Any = None, coils: Optional[Dict[str, str]] = None) -> List[TimeSeriesRun]:
    """Read a Phoenix MTU-5C-family recording into runs.

    ``path`` is the recording folder, one of its channel folders, or a file
    (which reads that file's stream, every channel of it). ``sample_rate``
    picks the stream(s) - 24000 for the native ``.bin`` or a decimated rate;
    by default every rate present is read, highest first.
    ``start`` and ``end`` (UTC) limit what is read: files wholly outside them
    are not loaded, which matters for long recordings (a day of 24 kHz bursts
    is gigabytes). ``calibration`` is a calibration file, a folder of them
    (searched through) or a list; by default ``*rxcal.json``, ``*scal.json``
    and binary ``*.scal`` files are looked for in the recording folder and the
    two above it, and in any folder there named for calibrations.

    ``coils`` says which coil was on which magnetic channel, ``{"H1": "4684",
    ...}`` (or ``"hx"``...), by serial or calibration file. A receiver whose
    coils were not detected records serial 0, so without it their responses
    are not known and the magnetic channels stay in volts.

    Continuous streams come out as one run per gap-free stretch, segmented
    ones as one run per burst.
    """
    recording = _recording_folder(Path(path))
    only_stream = None
    if Path(path).is_file():
        only_stream = _FILE_NAME.match(Path(path).name).group(5).lower()
        if sample_rate is None:
            sample_rate = read_phoenix_header(path)["sample_rate"]
    recmeta = _load_json(recording / "recmeta.json")
    config = _load_json(recording / "config.json")
    layout = _channel_layout(recmeta, config)
    wanted = None if components is None else {normalize_component(c) for c in components}
    rates = None if sample_rate is None else {float(r) for r in np.atleast_1d(sample_rate)}
    start_limit, end_limit = as_datetime64(start), as_datetime64(end)
    timing = recmeta.get("timing", {})

    files: Dict[Tuple[str, float], Dict[str, List[Path]]] = {}
    headers: Dict[Path, Dict[str, Any]] = {}
    for folder in sorted(p for p in recording.iterdir() if p.is_dir() and p.name.isdigit()):
        entry = layout.get(int(folder.name), {"tag": f"CH{folder.name}"})
        component = normalize_component(entry["tag"])
        if wanted is not None and component not in wanted:
            continue
        for item in sorted(folder.iterdir()):
            match = _FILE_NAME.match(item.name)
            if not match or item.stat().st_size <= 128:
                continue
            if only_stream is not None and match.group(5).lower() != only_stream:
                continue
            header = read_phoenix_header(item)
            if rates is not None and header["sample_rate"] not in rates:
                continue
            headers[item] = header
            key = (match.group(5).lower(), header["sample_rate"])
            files.setdefault(key, {}).setdefault(component, []).append(item)
    if not files:
        raise ValueError(f"no Phoenix time series found in {recording}"
                         + (f" at {sorted(rates)} Hz" if rates else ""))
    recorded = {component for by_component in files.values() for component in by_component}
    not_recorded = _not_recorded(layout, recorded, wanted)

    calibrations = [read_phoenix_calibration(p) for p in _calibration_files(recording, calibration)]
    receiver_cal = next((c for c in calibrations if c["kind"] == "receiver"
                         and c["instrument_serial"] in ("", str(recmeta.get("instid", "")))), None)
    sensor_cals = [c for c in calibrations if c["kind"] == "sensor"]
    coil_of = _coil_choices(layout, recorded, coils, sensor_cals)

    runs: List[TimeSeriesRun] = []
    for (extension, rate), by_component in sorted(files.items(), key=lambda item: -item[0][1]):
        pieces: Dict[str, List[Tuple[np.datetime64, np.ndarray]]] = {}
        info: Dict[str, Dict[str, Any]] = {}
        for component, paths in by_component.items():
            first = headers[paths[0]]
            native = first["file_type"] == 1
            late = 1.0 if first["file_version"] < (4 if native else 3) else 0.0
            recording_gps = np.datetime64("1970-01-01T00:00:00", "ns") + np.timedelta64(
                int((first["recording_id"] + late) * 1e9), "ns")
            leap = int(timing.get("tm_leapSecAtRecStart", gps_leap_seconds(recording_gps)))
            start_utc = recording_gps - np.timedelta64(leap, "s")
            gain, kind = _board_gain(first)
            blocks, saturated = [], 0
            for item in paths:
                header = headers[item]
                period = float(header["fragment_period"] or 0)
                if period and (start_limit is not None or end_limit is not None):
                    # A native file's sequence counts from 0, a decimated one's from 1; the
                    # minute's margin each side covers the leap seconds and stamp offsets.
                    sequence = int(header["file_sequence"]) - (0 if native else 1)
                    begin = shifted(start_utc, sequence * period - 60.0)
                    if (end_limit is not None and begin >= end_limit) or (
                            start_limit is not None and shifted(begin, period + 120.0) <= start_limit):
                        continue
                if native:
                    found, count = _native_blocks(item, header, start_utc, _AD_RANGE_VOLTS / 2**23 / gain)
                    blocks += found
                    saturated += count
                else:
                    blocks += _decimated_blocks(item, header, start_utc, extension in _CONTINUOUS, leap, late)
            pieces[component] = contiguous_blocks(blocks, rate)
            # Both kinds of file hold volts at the input, where the A/D's +-5 V is +-5 V / gain.
            info[component] = {"header": first, "gain": gain if native else 1.0, "kind": kind,
                               "full_scale": _AD_RANGE_VOLTS / gain,
                               "saturated_frames": saturated, "files": [str(p) for p in paths],
                               "late_stamp_corrected": bool(late), "leap_seconds": leap}

        def make_channel(component: str, data: np.ndarray) -> Channel:
            return _phoenix_channel(component, data, layout, info[component], receiver_cal,
                                    coil_of.get(component))

        layout_meta = recmeta.get("layout", {})
        found = assemble_runs(
            pieces, rate, make_channel,
            station=str(layout_meta.get("Station_Name", recording.name)),
            latitude=float(timing.get("gps_lat", headers[next(iter(headers))]["latitude"])),
            longitude=float(timing.get("gps_lon", headers[next(iter(headers))]["longitude"])),
            elevation=float(timing.get("gps_alt", headers[next(iter(headers))]["elevation"])),
            instrument=f"Phoenix {recmeta.get('receiver_commercial_name', first['instrument_type'])}",
            serial=str(recmeta.get("instid", first["instrument_serial"])),
            metadata={"source_format": "phoenix", "recording": str(recording),
                      "stream": extension, "survey": layout_meta.get("Survey_Name", ""),
                      "not_recorded": not_recorded,
                      "notes": _recording_notes(not_recorded, layout)
                      + _saturation_notes(layout, recorded, recmeta.get("acqtime"))
                      + _calibration_notes(info, receiver_cal, sensor_cals, coil_of)})
        for run in found:
            if start_limit is not None or end_limit is not None:
                run = run.window(start_limit, end_limit)
            if run.n_samples:
                runs.append(run)
    return runs


def _not_recorded(layout: Dict[int, Dict[str, Any]], recorded: set,
                  wanted: Optional[set]) -> Dict[str, str]:
    """The receiver's channels with no data, and why: switched off, or no files.

    A receiver set to record only the coils still lists E1 and E2 in its
    metadata; saying so keeps a recording without its electric field from
    passing for a complete one.
    """
    missing = {}
    for entry in layout.values():
        component = normalize_component(entry["tag"])
        if component in recorded or (wanted is not None and component not in wanted):
            continue
        missing[component] = ("switched off on the receiver" if entry.get("on", 1) in (0, "0")
                              else "no files in the recording")
    return missing


def _recording_notes(not_recorded: Dict[str, str], layout: Dict[int, Dict[str, Any]]) -> List[str]:
    tags = {normalize_component(e["tag"]): e["tag"] for e in layout.values()}
    by_reason: Dict[str, List[str]] = {}
    for component, reason in not_recorded.items():
        by_reason.setdefault(reason, []).append(f"{tags.get(component, component.upper())} ({component})")
    return [f"not recorded, {reason}: {', '.join(names)}" for reason, names in by_reason.items()]


def _saturation_notes(layout: Dict[int, Dict[str, Any]], recorded: set, acquisition_s: Any) -> List[str]:
    """The receiver's own count of native frames (20 samples) at full scale, per channel.

    It covers the whole recording, whichever streams were kept, so it says
    whether a channel clipped even when only decimated files are read.
    """
    parts = []
    for entry in layout.values():
        component = normalize_component(entry["tag"])
        count = int(entry.get("frameSat", 0) or 0)
        if component not in recorded or count <= 0:
            continue
        frames = float(acquisition_s or 0) * float(entry.get("sampleRate", 0) or 0) / 20.0
        share = f" ({100.0 * count / frames:.2g} % of the recording)" if frames > 0 else ""
        parts.append(f"{entry['tag']} ({component}) {count} frames{share}")
    return [f"the receiver counted analog saturation: {', '.join(parts)}"] if parts else []


def _calibration_notes(info, receiver_cal, sensor_cals, coil_of) -> List[str]:
    notes = []
    if receiver_cal is None:
        notes.append("no receiver calibration (rxcal.json): the anti-alias filters are not removed")
    if any(item["kind"] == "H" for item in info.values()) and not sensor_cals:
        notes.append("no coil calibration (scal.json or .scal): magnetic channels stay in volts")
    unknown = [coil["tag"] for coil in coil_of.values() if coil["stage"] is None and not coil["serial"]]
    if unknown and sensor_cals:
        notes.append(f"the receiver did not record which coil was on {', '.join(unknown)}: choose "
                     "each one's calibration (coils); until then they stay in volts")
    missing = [f"{coil['tag']} (coil {coil['serial']})" for coil in coil_of.values()
               if coil["stage"] is None and coil["serial"]]
    if missing:
        notes.append(f"no calibration found for {', '.join(missing)}: it stays in volts")
    saturated = {c: item["saturated_frames"] for c, item in info.items() if item["saturated_frames"]}
    if saturated:
        notes.append(f"frames with analog saturation: {saturated}")
    return notes


def _phoenix_channel(component, data, layout, info, receiver_cal, coil) -> Channel:
    header = info["header"]
    entry = next((e for e in layout.values() if normalize_component(e["tag"]) == component), {})
    tag = entry.get("tag", component.upper())
    response: List[ResponseStage] = []
    azimuth, dipole = float("nan"), float("nan")
    sensor, serial = "", ""
    if info["kind"] == "E":
        geometry = entry.get("dipole")
        if geometry:
            north = float(geometry.get("PosNorth", 0)) - float(geometry.get("NegNorth", 0))
            east = float(geometry.get("PosEast", 0)) - float(geometry.get("NegEast", 0))
            if north or east:
                dipole = float(np.hypot(north, east))
                azimuth = float(np.degrees(np.arctan2(east, north)) % 360.0)
        if not np.isfinite(dipole):
            dipole = float(entry.get("length1", 0) or 0) + float(entry.get("length2", 0) or 0)
        if dipole > 0:
            response += [dipole_stage(dipole), unit_stage("mV", "V")]
        sensor = "electrode dipole"
    else:
        sensor = (coil or {}).get("model") or str(entry.get("type_name", ""))
        serial = (coil or {}).get("serial") or str(entry.get("serial", ""))
        if coil is not None and coil["stage"] is not None:
            response += [coil["stage"], unit_stage("mV", "V")]
    low_pass = float(entry.get("lp", _low_pass(header)))
    if receiver_cal is not None:
        stage = _match_low_pass(receiver_cal["channels"].get(tag, []), low_pass)
        if stage is not None:
            response.append(stage)
    return Channel(component=component, data=data, units="V", response=response,
                   azimuth=azimuth, dipole_length=dipole, sensor=sensor, sensor_serial=serial,
                   metadata={"tag": tag, "board": header["board_model"].strip(),
                             "board_gain": info["gain"], "low_pass_hz": low_pass,
                      "full_scale": info["full_scale"],
                      "receiver_saturated_frames": int(entry.get("frameSat", 0) or 0),
                             "saturated_frames": info["saturated_frames"], "files": info["files"],
                             "late_stamp_corrected": info["late_stamp_corrected"],
                             "leap_seconds": info["leap_seconds"]})


def _same_serial(a: str, b: str) -> bool:
    """Serials that differ only in leading zeros are one: field notes write coil 447, files 0447."""
    return str(a).lstrip("0") == str(b).lstrip("0") != ""


def _coil_calibration(serial: str, sensor_cals: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The newest calibration named for coil ``serial`` (``<serial>_...``), else the newest stating it."""
    if not serial or serial == "0":
        return None
    named = [c for c in sensor_cals
             if _same_serial(Path(c["source_file"]).name.split("_")[0].split(".")[0], serial)]
    stated = [c for c in sensor_cals if _same_serial(c["sensor_serial"], serial)]
    for group in (named, stated):
        usable = [c for c in group if any(c["channels"].values())]
        if usable:
            return max(usable, key=_calibration_time)
    return None


def _coil_choices(layout: Dict[int, Dict[str, Any]], recorded: set, coils: Optional[Dict[str, str]],
                  sensor_cals: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Each recorded coil's serial, model and response, by component.

    A coil named in ``coils`` (by serial, or by its calibration file) is that
    one; any other is the serial the receiver recorded. The response is found
    once for the recording and shared by all its runs.
    """
    chosen: Dict[str, str] = {}
    for key, value in (coils or {}).items():
        if value in (None, ""):
            continue
        value = str(value)
        if Path(value).is_file():
            cal = read_phoenix_calibration(value)
            sensor_cals = sensor_cals + [cal]
            value = cal["sensor_serial"] or Path(value).name.split("_")[0].split(".")[0]
        chosen[normalize_component(key)] = value
    found = {}
    for entry in layout.values():
        component = normalize_component(entry["tag"])
        if not component.startswith("h") or component not in recorded:
            continue
        serial = chosen.get(component, str(entry.get("serial", "") or ""))
        cal = _coil_calibration(serial, sensor_cals)
        if component in chosen and cal is None:
            known = sorted({c["sensor_serial"] for c in sensor_cals if c["sensor_serial"]})
            raise ValueError(f"no calibration for coil {serial}, given for {entry['tag']}; "
                             + (f"the coil calibrations found are for {', '.join(known)}" if known
                                else "no coil calibration was found"))
        stage = None
        if cal is not None:
            stages = cal["channels"].get(entry["tag"]) or next(iter(cal["channels"].values()), [])
            stage = replace(stages[0], name=f"phoenix_coil_{serial}") if stages else None
        found[component] = {"tag": entry["tag"], "serial": "" if serial == "0" else serial,
                            "model": (cal or {}).get("model", ""), "stage": stage,
                            "file": (cal or {}).get("source_file", "")}
    return found
