"""Metronix ADU-06/07/08 recordings: ATS files, the measurement XML and coil calibrations.

An ADU writes one ``.ats`` file per channel - ``085_V01_C02_R001_THx_BL_2048H.ats``,
system 085, channel 2, run 1, Hx, band 2048 Hz - into a ``meas_<start>``
folder with a measurement XML. An ATS file (header versions 80 and 81) is a
1024-byte little-endian header followed by int32 samples, which times the
header's least significant bit give millivolts at the channel input (the
gains are already in it).

The header fields read here: header length (0), version (2), samples (4),
sample rate (8, float32), start in Unix seconds (12), LSB in mV (16, double),
UTC offset (24), system serial (32), channel number (36), chopper (37),
channel type (38, ``Ex``...), sensor type (40) and serial (46), electrode
positions x1, y1, z1, x2, y2, z2 in m (48-68, x north, y east), dipole length
and angle (72, 76), latitude and longitude in milliseconds of arc (96, 100),
elevation in cm (104) and the system type (132).

Coils (MFS-06e, MFS-07e) are calibrated as V/(nT Hz) against frequency, with
the chopper on and off; the table for the channel's chopper setting is used,
from the measurement XML's ``calibration_sensors`` or a Metronix calibration
text file (``MFS06E0005.txt``), given as ``calibration`` or found in a
``cal`` folder next to the measurements.

Metronix Geophysics (2009). ADU-07 system manual, appendix: ATS file format.
"""

from __future__ import annotations

import re
import struct
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from .timeseries import (
    Channel,
    ResponseStage,
    TimeSeriesRun,
    as_datetime64,
    assemble_runs,
    dipole_stage,
    normalize_component,
)

_NAME = re.compile(r"^(\d+)_V(\d+)_C(\d+)_R(\d+)_T(\w+?)_B(\w)_(\d+)([HS])\.ats$", re.IGNORECASE)


def read_ats_header(path: Any) -> Dict[str, Any]:
    raw = Path(path).read_bytes()[:1024]
    if len(raw) < 160:
        raise ValueError(f"{Path(path).name} is too short for an ATS header")
    u = lambda fmt, offset: struct.unpack_from("<" + fmt, raw, offset)[0]
    text = lambda a, b: raw[a:b].split(b"\x00")[0].decode("latin-1").strip()
    return {
        "header_length": u("H", 0), "version": u("h", 2), "n_samples": u("I", 4),
        "sample_rate": float(u("f", 8)), "start": u("i", 12), "lsb_mv": u("d", 16),
        "utc_offset": u("i", 24), "system_serial": u("H", 32), "channel_number": raw[36],
        "chopper": raw[37], "channel_type": text(38, 40), "sensor_type": text(40, 46),
        "sensor_serial": u("h", 46), "positions": [u("f", 48 + 4 * i) for i in range(6)],
        "dipole_length": float(u("f", 72)), "angle": float(u("f", 76)),
        "latitude": u("i", 96) / 3.6e6, "longitude": u("i", 100) / 3.6e6,
        "elevation": u("i", 104) / 100.0, "system_type": text(132, 144),
    }


def is_ats_file(path: Any) -> bool:
    source = Path(path)
    return source.is_file() and source.suffix.lower() == ".ats"


def _caldata_stage(rows: List[Tuple[float, float, float]], name: str, source: str) -> Optional[ResponseStage]:
    if not rows:
        return None
    table = np.asarray(sorted(rows))
    values = table[:, 1] * table[:, 0] * 1e3      # V/(nT Hz) * Hz -> mV/nT
    return ResponseStage.from_amplitude_phase(name, "nT", "mV", table[:, 0], values, table[:, 2],
                                              metadata={"source_file": source})


def read_metronix_calibration(path: Any) -> Dict[str, Any]:
    """A Metronix coil calibration text file: ``{"sensor", "serial", "on", "off"}`` stages."""
    source = Path(path)
    lines = source.read_text(encoding="latin-1", errors="replace").splitlines()
    sensor, serial = "", ""
    tables: Dict[str, List[Tuple[float, float, float]]] = {"on": [], "off": []}
    current = "on"
    for line in lines:
        match = re.search(r"Magnetometer:\s*([\w-]+)\s*#\s*(\d+)", line)
        if match:
            sensor, serial = match.group(1), match.group(2)
        lower = line.strip().lower()
        if lower.startswith("chopper on"):
            current = "on"
        elif lower.startswith("chopper off"):
            current = "off"
        parts = line.split()
        if len(parts) >= 3:
            try:
                tables[current].append(tuple(float(v) for v in parts[:3]))
            except ValueError:
                pass
    if not serial:
        digits = re.findall(r"\d+", source.stem)
        serial = digits[-1].lstrip("0") if digits else ""
    return {"sensor": sensor, "serial": str(int(serial)) if serial.isdigit() else serial,
            "source_file": str(source),
            **{mode: _caldata_stage(rows, f"metronix_{sensor or 'coil'}_{serial}_chopper_{mode}", str(source))
               for mode, rows in tables.items()}}


def _xml_calibrations(xml_path: Optional[Path]) -> Dict[int, Dict[str, Optional[ResponseStage]]]:
    """Each channel's coil tables from a measurement XML's ``calibration_sensors``."""
    if xml_path is None:
        return {}
    try:
        root = ET.parse(xml_path).getroot()
    except (ET.ParseError, OSError):
        return {}
    found: Dict[int, Dict[str, Optional[ResponseStage]]] = {}
    for block in root.iter("calibration_sensors"):
        for channel in block.findall("channel"):
            rows: Dict[str, List[Tuple[float, float, float]]] = {"on": [], "off": []}
            for entry in channel.iter("caldata"):
                mode = "on" if entry.get("chopper", "on").lower() == "on" else "off"
                values = {c.get("unit", ""): c.text for c in entry}
                try:
                    f = float(values.get("Hz"))
                    m = float(values.get("V/(nT*Hz)"))
                    p = float(values.get("deg"))
                except (TypeError, ValueError):
                    continue
                rows[mode].append((f, m, p))
            serial = channel.findtext(".//ci_serial_number", default="")
            name = f"metronix_coil_{serial}"
            found[int(channel.get("id", -1))] = {
                mode: _caldata_stage(r, f"{name}_chopper_{mode}", str(xml_path)) for mode, r in rows.items()}
    return found


def _coil_stage(header: Dict[str, Any], xml_cals, file_cals) -> Tuple[Optional[ResponseStage], str]:
    mode = "on" if header["chopper"] else "off"
    other = "off" if mode == "on" else "on"
    serial = str(header["sensor_serial"])
    candidates = []
    for cal in file_cals:
        if cal["serial"] == serial and serial not in ("", "0"):
            candidates.append(cal)
    candidates.append(xml_cals.get(header["channel_number"], {}))
    for cal in candidates:
        if cal.get(mode) is not None:
            return cal[mode], ""
        if cal.get(other) is not None:
            return cal[other], f"no chopper-{mode} table for {header['channel_type']}; chopper-{other} used"
    return None, ""


def read_metronix(path: Any, *, components: Optional[Iterable[str]] = None,
                  calibration: Any = None, start: Any = None, end: Any = None) -> List[TimeSeriesRun]:
    """Read Metronix ATS files - one file, a ``meas_`` folder or a site folder - into runs.

    Files of one measurement folder with the same run number and sample rate
    form a run. ``calibration`` is a coil calibration text file, a folder of
    them, or a list; ``cal`` folders beside and above the measurements are
    searched too.
    """
    source = Path(path)
    files = [source] if source.is_file() else sorted(source.rglob("*.ats")) + sorted(source.rglob("*.ATS"))
    files = [f for f in dict.fromkeys(files) if f.stat().st_size > 1024]
    if not files:
        raise ValueError(f"no non-empty ATS files found in {source}")
    wanted = None if components is None else {normalize_component(c) for c in components}
    file_cals = [read_metronix_calibration(p) for p in _calibration_files(files, calibration)]

    groups: Dict[Tuple[Path, str, float], List[Tuple[Path, Dict[str, Any]]]] = {}
    for item in files:
        header = read_ats_header(item)
        if header["version"] >= 1000:
            raise ValueError(f"{item.name}: sliced ATS (header version {header['version']}) is not supported")
        match = _NAME.match(item.name)
        run_id = match.group(4) if match else "0"
        groups.setdefault((item.parent, run_id, header["sample_rate"]), []).append((item, header))

    runs: List[TimeSeriesRun] = []
    for (folder, run_id, rate), members in sorted(groups.items(), key=lambda kv: (str(kv[0][0]), -kv[0][2])):
        xml_path = next(iter(sorted(folder.glob("*.xml"))), None)
        xml_cals = _xml_calibrations(xml_path)
        pieces, headers, notes = {}, {}, []
        for item, header in members:
            component = normalize_component(header["channel_type"] or (_NAME.match(item.name).group(5)
                                                                       if _NAME.match(item.name) else "ch"))
            if wanted is not None and component not in wanted:
                continue
            count = min(header["n_samples"], (item.stat().st_size - header["header_length"]) // 4)
            counts = np.fromfile(item, dtype="<i4", count=count, offset=header["header_length"])
            begin = np.datetime64(int(header["start"]), "s").astype("datetime64[ns]")
            pieces[component] = [(begin, counts.astype(float) * header["lsb_mv"])]
            headers[component] = (item, header)
            if header["utc_offset"]:
                notes.append(f"{item.name}: header UTC offset {header['utc_offset']} s not applied")
        if not pieces:
            continue

        def make_channel(component: str, data: np.ndarray) -> Channel:
            item, header = headers[component]
            channel, note = _metronix_channel(component, data, header, xml_cals, file_cals, item)
            if note:
                notes.append(note)
            return channel

        first = next(iter(headers.values()))[1]
        found = assemble_runs(
            pieces, rate, make_channel, station=folder.parent.name if folder.name.startswith("meas") else folder.name,
            latitude=first["latitude"], longitude=first["longitude"], elevation=first["elevation"],
            instrument=f"Metronix {first['system_type'] or 'ADU'}", serial=str(first["system_serial"]),
            metadata={"source_format": "metronix_ats", "folder": str(folder), "run": run_id,
                      "xml": str(xml_path or ""), "notes": notes})
        start_limit, end_limit = as_datetime64(start), as_datetime64(end)
        for run in found:
            run.metadata["notes"] = list(dict.fromkeys(run.metadata["notes"] + notes))
            if start_limit is not None or end_limit is not None:
                run = run.window(start_limit, end_limit)
            if run.n_samples:
                runs.append(run)
    return runs


def _calibration_files(files: List[Path], calibration: Any) -> List[Path]:
    places: List[Path] = []
    if calibration is not None:
        places += [Path(p) for p in ([calibration] if isinstance(calibration, (str, Path)) else calibration)]
    for item in files[:1]:
        for folder in list(item.parents)[:4]:
            places.append(folder / "cal")
    found: List[Path] = []
    for place in places:
        if place.is_dir():
            found += sorted(place.glob("*.txt")) + sorted(place.glob("*.TXT"))
        elif place.is_file():
            found.append(place)
    return list(dict.fromkeys(found))


def _metronix_channel(component, data, header, xml_cals, file_cals, item) -> Tuple[Channel, str]:
    x1, y1, z1, x2, y2, z2 = header["positions"]
    response: List[ResponseStage] = []
    note = ""
    azimuth, dipole = float("nan"), float("nan")
    if component.startswith("e"):
        north, east, down = x2 - x1, y2 - y1, z2 - z1
        if north or east:
            dipole = float(np.sqrt(north**2 + east**2 + down**2))
            azimuth = float(np.degrees(np.arctan2(east, north)) % 360.0)
        elif header["dipole_length"] > 0:
            dipole, azimuth = header["dipole_length"], header["angle"]
        if dipole > 0:
            response = [dipole_stage(dipole)]
        sensor = header["sensor_type"] or "electrode dipole"
    else:
        sensor = header["sensor_type"]
        coil, note = _coil_stage(header, xml_cals, file_cals)
        if coil is not None:
            response = [coil]
        else:
            note = f"no calibration for {sensor} #{header['sensor_serial']} ({component}): it stays in mV"
    channel = Channel(component=component, data=data, units="mV", response=response,
                      azimuth=azimuth, dipole_length=dipole, sensor=sensor,
                      sensor_serial=str(header["sensor_serial"]),
                      metadata={"file": str(item), "lsb_mv": header["lsb_mv"],
                                "chopper": bool(header["chopper"]),
                                "channel_number": header["channel_number"]})
    return channel, note
