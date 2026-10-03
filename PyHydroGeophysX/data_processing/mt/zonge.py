"""Zonge International ZEN receivers: Z3D files, one per channel.

A Z3D file starts with 512-byte text records: the GPS board's header
(``A/D Rate``, ``Ch.Factor``, ``GpsWeek``, ``Lat``/``Long`` in radians...), the
schedule, and metadata records of ``|KEY=VALUE|`` pairs - ``CH.CMP``,
``CH.OFFSET.XYZ1``/``XYZ2`` (electrode offsets, m), ``CH.ANTSN`` (coil serial),
``RX.XAZIMUTH`` and, in magnetic files, the coil calibration ``CAL.ANT``
(frequency in Hz : amplitude in mV/nT : phase in mrad - read as radians per
second instead, the frequencies give impedance phases outside their
quadrants on Zonge's sample site, as Hz they give a smooth sounding whose
4096 and 256 S/s runs agree where they overlap). Then come int32
samples, one second per block, each block led by a 64-byte GPS stamp that
starts with the words 0x7FFFFFFF, 0x80000000 and gives the time in 1/1024 s
of the GPS week. Samples times ``Ch.Factor`` are volts at the input (divided
by the channel gain).

Files of one station, sample rate and schedule start form a run; the first
two seconds are dropped, because the ZEN's SD-card buffer can corrupt them,
and a block that does not hold one second of samples ends a run.

Zonge International (2014). ZEN receiver and Z3D file format notes, in
ZenACQ documentation. Zonge International, Tucson, AZ.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

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
    read_response_table,
    unit_stage,
)

_RECORD = 512
_FLAG = b"\xff\xff\xff\x7f\x00\x00\x00\x80"
_STAMP_WORDS = 16
_WEEK = 604800
_GPS_EPOCH = np.datetime64("1980-01-06T00:00:00", "ns")


def is_z3d_file(path: Any) -> bool:
    source = Path(path)
    return source.is_file() and source.suffix.lower() == ".z3d"


def read_z3d_header(path: Any) -> Dict[str, Any]:
    """The text records of a Z3D file: header, schedule and metadata, and the data offset."""
    raw = Path(path).read_bytes()
    first = raw.find(_FLAG, _RECORD)
    if first < 0:
        raise ValueError(f"{Path(path).name}: no GPS stamp found (old Z3D firmware is not supported)")
    header: Dict[str, str] = {}
    schedule: Dict[str, str] = {}
    pairs: List[str] = []
    for offset in range(0, first - _RECORD + 1, _RECORD):
        text = raw[offset:offset + _RECORD].replace(b"\x00", b"").decode("latin-1")
        lines = [line for line in text.splitlines() if line.strip()]
        title = lines[0] if lines else ""
        body = lines[1:]
        if "Schedule" in title:
            for line in body:
                key, _, value = line.partition("=")
                schedule[key.strip().replace("Schedule.", "")] = value.strip()
        elif "Metadata" in title or "Caldata" in title:
            pairs.append("".join(line.strip() for line in body))
        elif "=" in text:
            for line in lines:
                key, sep, value = line.partition("=")
                if sep:
                    header[key.strip()] = value.strip()
    metadata: Dict[str, str] = {}
    for item in "".join(pairs).split("|"):
        key, sep, value = item.partition("=")
        if sep:
            name = key.strip().upper()
            metadata[name] = value.strip() if name not in metadata else metadata[name] + "|" + value.strip()
    return {"header": header, "schedule": schedule, "metadata": metadata, "data_offset": first}


def _float(mapping: Dict[str, str], key: str, default: float = float("nan")) -> float:
    try:
        return float(mapping.get(key, default))
    except (TypeError, ValueError):
        return default


def _coil_table(text: str) -> Optional[ResponseStage]:
    """``CAL.ANT``: serial, then ``frequency:amplitude:phase`` entries (Hz, mV/nT, mrad)."""
    serial, _, rest = text.partition(",")
    rows = []
    for entry in rest.split(","):
        parts = entry.strip().split(":")
        if len(parts) == 3:
            try:
                rows.append([float(p) for p in parts])
            except ValueError:
                continue
    if not rows:
        return None
    table = np.asarray(rows)
    return ResponseStage.from_amplitude_phase(
        f"zonge_ant_{serial.strip()}", "nT", "mV", table[:, 0], table[:, 1], table[:, 2],
        phase_units="mrad", metadata={"source": "Z3D CAL.ANT"})


def _blocks(path: Path, info: Dict[str, Any], rate: float,
            skip_seconds: int) -> Tuple[List[Tuple[np.datetime64, np.ndarray]], List[str]]:
    raw = path.read_bytes()
    offset = info["data_offset"]
    words = np.frombuffer(raw, dtype="<i4", offset=offset, count=(len(raw) - offset) // 4)
    candidates = np.flatnonzero((words[:-1] == 0x7FFFFFFF) & (words[1:] == -0x80000000))
    stamps = [int(candidates[0])] if candidates.size else []
    for index in candidates[1:]:
        if index - stamps[-1] >= _STAMP_WORDS:
            stamps.append(int(index))
    notes = []
    week = int(_float(info["header"], "GpsWeek", 0))
    blocks, previous = [], None
    for k, index in enumerate(stamps):
        stop = stamps[k + 1] if k + 1 < len(stamps) else words.size
        seconds = words[index + 2] / 1024.0
        if previous is not None and seconds < previous - _WEEK / 2:
            week += 1
        previous = seconds
        if k < skip_seconds:
            continue
        gps = _GPS_EPOCH + np.timedelta64(int(round(week * _WEEK + round(seconds))), "s")
        when = gps - np.timedelta64(gps_leap_seconds(gps), "s")
        samples = words[index + _STAMP_WORDS:stop]
        if k + 1 == len(stamps) and samples.size > rate:
            samples = samples[: int(rate)]
        blocks.append((when, samples))
    short = sum(1 for _, s in blocks[:-1] if s.size != int(rate))
    if short:
        notes.append(f"{path.name}: {short} one-second blocks without {int(rate)} samples split the run")
    return blocks, notes


def read_zonge(path: Any, *, components: Optional[Iterable[str]] = None,
               calibration: Optional[Dict[str, Any]] = None, start: Any = None, end: Any = None,
               skip_seconds: int = 2) -> List[TimeSeriesRun]:
    """Read Z3D files - one file or a folder - into runs.

    ``calibration`` maps a component (or a coil serial) to a
    :class:`ResponseStage` or a text table (frequency Hz, amplitude mV/nT,
    phase in degrees) that replaces the coil table stored in the file.
    """
    source = Path(path)
    files = [source] if source.is_file() else sorted({*source.glob("*.Z3D"), *source.glob("*.z3d")})
    if not files:
        raise ValueError(f"no Z3D files found in {source}")
    wanted = None if components is None else {normalize_component(c) for c in components}
    overrides = {str(k).lower(): v for k, v in (calibration or {}).items()}

    groups: Dict[Tuple[str, float, str], List[Tuple[Path, Dict[str, Any]]]] = {}
    for item in files:
        info = read_z3d_header(item)
        meta, header, schedule = info["metadata"], info["header"], info["schedule"]
        component = normalize_component(meta.get("CH.CMP", item.stem.split("_")[-1]))
        if wanted is not None and component not in wanted:
            continue
        info["component"] = component
        rate = _float(header, "A/D Rate", _float(schedule, "S/R"))
        key = (meta.get("RX.STN", item.stem.split("_")[0]), rate,
               f"{schedule.get('Date', '')}T{schedule.get('Time', '')}")
        groups.setdefault(key, []).append((item, info))

    runs: List[TimeSeriesRun] = []
    for (station, rate, scheduled), members in sorted(groups.items(), key=lambda kv: (kv[0][2], -kv[0][1])):
        pieces, infos, notes = {}, {}, []
        for item, info in members:
            blocks, found_notes = _blocks(item, info, rate, skip_seconds)
            notes += found_notes
            pieces[info["component"]] = contiguous_blocks(blocks, rate)
            infos[info["component"]] = (item, info)

        def make_channel(component: str, counts: np.ndarray) -> Channel:
            item, info = infos[component]
            channel, note = _zonge_channel(component, counts, item, info, overrides)
            if note:
                notes.append(note)
            return channel

        header = members[0][1]["header"]
        found = assemble_runs(
            pieces, rate, make_channel, station=str(station),
            latitude=float(np.degrees(_float(header, "Lat"))),
            longitude=float(np.degrees(_float(header, "Long"))),
            elevation=_float(header, "Alt"), instrument="Zonge ZEN",
            serial=str(header.get("Box number", "")),
            metadata={"source_format": "zonge_z3d", "schedule_start": scheduled,
                      "files": [str(item) for item, _ in members], "notes": notes})
        start_limit, end_limit = as_datetime64(start), as_datetime64(end)
        for run in found:
            run.metadata["notes"] = list(dict.fromkeys(run.metadata["notes"] + notes))
            if start_limit is not None or end_limit is not None:
                run = run.window(start_limit, end_limit)
            if run.n_samples:
                runs.append(run)
    return runs


def _zonge_channel(component, counts, item, info, overrides) -> Tuple[Channel, str]:
    header, meta = info["header"], info["metadata"]
    factor = _float(header, "Ch.Factor", 9.536743164062e-10)
    gain = _float(header, "ChannelGain", _float(header, "A/D Gain", 1.0)) or 1.0
    data = counts.astype(float) * (factor / gain)
    x_azimuth = _float(meta, "RX.XAZIMUTH", 0.0)
    response: List[ResponseStage] = []
    note = ""
    dipole = float("nan")
    if component.startswith("e"):
        try:
            p1 = np.array([float(v) for v in meta["CH.OFFSET.XYZ1"].split(":")])
            p2 = np.array([float(v) for v in meta["CH.OFFSET.XYZ2"].split(":")])
            dipole = float(np.linalg.norm(p2 - p1))
        except (KeyError, ValueError):
            dipole = _float(meta, "CH.LENGTH")
        azimuth = x_azimuth + (90.0 if component == "ey" else 0.0)
        if dipole > 0:
            response = [dipole_stage(dipole), unit_stage("mV", "V")]
        else:
            note = f"{item.name}: no electrode offsets, so {component} stays in volts"
        sensor, serial = "electrode dipole", ""
    else:
        azimuth = _float(meta, "CH.AZIMUTH", x_azimuth + (90.0 if component == "hy" else 0.0))
        serial = meta.get("CH.ANTSN", "")
        sensor = "Zonge ANT/4 induction coil"
        coil = overrides.get(component, overrides.get(serial.lower()))
        if isinstance(coil, (str, Path)):
            coil = read_response_table(coil, name=f"zonge_ant_{serial}")
        if coil is None and "CAL.ANT" in meta:
            coil = _coil_table(meta["CAL.ANT"].split("|")[0])
        if coil is not None:
            response = [coil, unit_stage("mV", "V")]
        else:
            note = f"{item.name}: no coil calibration for {serial or component}; it stays in volts"
    channel = Channel(component=component, data=data, units="V", response=response,
                      azimuth=azimuth, tilt=_float(meta, "CH.INCL", 0.0) if component != "hz" else 90.0,
                      dipole_length=dipole, sensor=sensor, sensor_serial=serial,
                      metadata={"file": str(item), "ch_factor": factor, "gain": gain,
                                "board_calibration": meta.get("CAL.BRD", "")})
    return channel, note
