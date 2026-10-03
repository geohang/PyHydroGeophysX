"""Phoenix Geophysics legacy MTU-5, MTU-5A and V8 recordings: TS2-TS5 and TBL.

A site's recording is a table file ``<name>.TBL`` and one time-series file per
band, ``<name>.TS2`` ... ``<name>.TS5`` (MTU-5A: 2400, 150 and 15 S/s for
TS3-TS5; TS2 for AMT). A TS file is a sequence of records, each a 32-byte tag
- the UTC time of its first scan (second, minute, hour, day, month, year,
weekday, century), serial number, scans in the record, channels per scan,
tag length, status and saturation flags, sample length, sample rate - followed
by the scans: per channel a 24-bit signed little-endian integer. Records that
follow each other in time form one run; the high bands are bursts.

The TBL file holds 25-byte records: a 12-byte name and a 13-byte value
(integer, double, text or time). Its gains turn counts into volts at the box
input, ``counts * FSCV / 2^23 / gain`` with gain ``EGN`` for E and
``HGN * HATT`` for H, and its ``EXLN``/``EYLN`` dipole lengths and ``HNOM``
coil sensitivity (mV/nT) give a flat response; Phoenix's per-frequency box
and coil calibrations (CLB/CLC) are not decoded, but any table can be given
as ``calibration`` (see :func:`read_response_table`). Without them the
magnetic channels are right only in the coil's flat band, and only up to the
coil's phase there: on Phoenix's sample site the impedance phases come out
180 degrees from their usual quadrants and the apparent resistivity climbs
at long periods, where an induction coil's output falls with frequency.

Phoenix Geophysics (2005). Instrument and sensor calibration: concepts and
utility programs. Phoenix Geophysics Ltd., Toronto.
"""

from __future__ import annotations

import struct
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
    normalize_component,
    read_response_table,
    unit_stage,
)

_TAG = 32
_DOUBLES = {"EXLN", "EYLN", "EAZM", "HAZM", "HATT", "HNOM", "HNUM", "FSCV", "DECL", "TSTV",
            "EXAC", "EXDC", "EYAC", "EYDC", "HXAC", "HXDC", "HYAC", "HYDC", "HZAC", "HZDC",
            "CFMN", "CFMX", "CCMN", "CCMX", "HAMP"}
_TEXTS = {"SITE", "FILE", "CMPY", "SRVY", "LATG", "LNGG", "HXSN", "HYSN", "HZSN", "VER", "HW",
          "CPTH", "EPTH", "DPTH", "SPTH"}
_TIMES = {"STIM", "ETIM", "HTIM", "LTIM", "FTIM", "NUTC", "LFIX", "TSYN", "ETMH"}
_CHANNEL_KEYS = (("CHEX", "ex"), ("CHEY", "ey"), ("CHHX", "hx"), ("CHHY", "hy"), ("CHHZ", "hz"))


def _time(raw: bytes) -> Optional[np.datetime64]:
    second, minute, hour, day, month, year, _, century = raw[:8]
    try:
        return np.datetime64(f"{century * 100 + year:04d}-{month:02d}-{day:02d}"
                             f"T{hour:02d}:{minute:02d}:{second:02d}", "ns")
    except ValueError:
        return None


def read_phoenix_table(path: Any) -> Dict[str, Any]:
    """The records of a legacy Phoenix ``.TBL`` file, decoded by name."""
    raw = Path(path).read_bytes()
    table: Dict[str, Any] = {}
    for offset in range(0, len(raw) - 24, 25):
        key = raw[offset:offset + 12].split(b"\x00")[0].decode("latin-1").strip()
        value = raw[offset + 12:offset + 25]
        if not key:
            continue
        if key in _DOUBLES:
            decoded: Any = struct.unpack_from("<d", value)[0]
        elif key in _TEXTS:
            decoded = value.split(b"\x00")[0].decode("latin-1").strip()
        elif key in _TIMES:
            decoded = _time(value)
        else:
            decoded = struct.unpack_from("<i", value)[0]
        table[key] = decoded
    return table


def _degrees(text: str) -> float:
    """``ddmm.mmmm,H`` (or ``dddmm.mmmm,H``) as signed decimal degrees."""
    try:
        number, _, hemisphere = str(text).partition(",")
        value = float(number)
    except ValueError:
        return float("nan")
    degrees = int(value / 100)
    result = degrees + (value - 100 * degrees) / 60.0
    return -result if hemisphere.strip().upper()[:1] in ("S", "W") else result


def is_phoenix_legacy_file(path: Any) -> bool:
    source = Path(path)
    suffix = source.suffix.upper()
    return source.is_file() and (suffix in (".TS2", ".TS3", ".TS4", ".TS5", ".TSL", ".TSH")
                                 or (suffix == ".TBL" and any(source.with_suffix(s).exists()
                                                             for s in (".TS3", ".TS4", ".TS5", ".TS2"))))


def _records(raw: bytes) -> Tuple[List[Tuple[np.datetime64, int, int]], int, float, int]:
    """Each record's (time, byte offset, scans), with the file's channels and rate."""
    records, offset = [], 0
    n_channels = rate = None
    saturated = 0
    while offset + _TAG <= len(raw):
        tag = raw[offset:offset + _TAG]
        scans, channels = struct.unpack_from("<HB", tag, 10)
        width = tag[17] or 3
        size = scans * channels * width
        if channels == 0 or offset + _TAG + size > len(raw):
            break
        if n_channels is None:
            n_channels = channels
            rate = float(struct.unpack_from("<H", tag, 18)[0]) if tag[20] == 0 else float("nan")
        when = _time(tag)
        if when is not None and channels == n_channels:
            records.append((when, offset + _TAG, scans))
            saturated += tag[15] != 0
        offset += _TAG + size
    return records, n_channels or 0, rate or float("nan"), saturated


def _band_data(raw: bytes, records, n_channels: int,
               rate: float) -> List[Tuple[np.datetime64, np.ndarray]]:
    """The counts of each stretch of back-to-back records, ``(time, (scans, channels) int32)``.

    Records that follow each other in the file and in time, with equal scan
    counts, are decoded together.
    """
    buffer = np.frombuffer(raw, dtype=np.uint8)
    times = np.array([when for when, _, _ in records], dtype="datetime64[ns]")
    offsets = np.array([offset for _, offset, _ in records])
    scans = np.array([count for _, _, count in records])
    step = (scans * 1e9 / rate).round().astype("timedelta64[ns]")
    size = scans * n_channels * 3
    joined = ((times[1:] == times[:-1] + step[:-1]) & (scans[1:] == scans[:-1])
              & (offsets[1:] == offsets[:-1] + size[:-1] + _TAG))
    breaks = np.flatnonzero(~joined) + 1
    blocks = []
    for first, last in zip(np.r_[0, breaks], np.r_[breaks, len(records)]):
        count, width = last - first, int(size[first]) + _TAG
        begin = int(offsets[first]) - _TAG
        rows = buffer[begin:begin + count * width].reshape(count, width)[:, _TAG:]
        b = rows.reshape(count * int(scans[first]), n_channels, 3).astype(np.int32)
        counts = b[..., 0] | (b[..., 1] << 8) | (b[..., 2] << 16)
        counts -= (counts & 0x800000) << 1
        blocks.append((times[first], counts))
    return blocks


def read_phoenix_legacy(path: Any, *, bands: Optional[Iterable[str]] = None,
                        components: Optional[Iterable[str]] = None, start: Any = None,
                        end: Any = None,
                        calibration: Optional[Dict[str, Any]] = None) -> List[TimeSeriesRun]:
    """Read a legacy Phoenix site (``.TBL`` with ``.TS2``-``.TS5``) into runs.

    ``path`` is one of the site's files. ``bands`` picks ``"TS3"``, ``"TS4"``...
    (default: every TS file of the site). ``calibration`` maps a component to
    a :class:`ResponseStage` or a text table of frequency, amplitude (mV/nT for
    coils), phase in degrees, which replaces the flat coil sensitivity.
    """
    source = Path(path)
    stem = source.with_suffix("")
    table_path = next((p for p in (stem.with_suffix(".TBL"), stem.with_suffix(".tbl")) if p.exists()), None)
    table = read_phoenix_table(table_path) if table_path else {}
    if source.suffix.upper().startswith(".TS") and bands is None:
        chosen = [source]
    else:
        names = [b.upper().lstrip(".") for b in bands] if bands else ["TS2", "TS3", "TS4", "TS5"]
        chosen = [p for name in names for p in (stem.with_suffix("." + name), stem.with_suffix("." + name.lower()))
                  if p.exists()]
    chosen = list(dict.fromkeys(chosen))
    if not chosen:
        raise ValueError(f"no TS files found for {stem.name}")

    order = [name for key, name in sorted(_CHANNEL_KEYS, key=lambda kv: table.get(kv[0], 99))
             if table.get(key, 0)]
    if not order:
        order = ["ex", "ey", "hx", "hy", "hz"]
    wanted = None if components is None else {normalize_component(c) for c in components}
    notes: List[str] = []
    fscv = table.get("FSCV")
    if fscv is None:
        notes.append("no TBL file: samples stay in counts and the response is unknown")
    calibration = {normalize_component(k): v for k, v in (calibration or {}).items()}

    runs: List[TimeSeriesRun] = []
    for ts_path in chosen:
        raw = ts_path.read_bytes()
        records, n_channels, rate, saturated = _records(raw)
        band = ts_path.suffix.upper().lstrip(".")
        if not np.isfinite(rate):
            rate = float(table.get(f"SRL{band[-1]}", float("nan")))
        if not records or not np.isfinite(rate):
            continue
        names = (order + [f"ch{i}" for i in range(len(order), n_channels)])[:n_channels]
        counts = _band_data(raw, records, n_channels, rate)
        pieces = {}
        for column, name in enumerate(names):
            if wanted is not None and name not in wanted:
                continue
            blocks = [(when, data[:, column]) for when, data in counts]
            pieces[name] = contiguous_blocks(blocks, rate)

        def make_channel(component: str, data: np.ndarray) -> Channel:
            return _legacy_channel(component, data, table, calibration)

        run_notes = notes + ([f"{saturated} records flagged saturated"] if saturated else [])
        found = assemble_runs(
            pieces, rate, make_channel, station=str(table.get("SITE", stem.name)),
            latitude=_degrees(table.get("LATG", "")), longitude=_degrees(table.get("LNGG", "")),
            elevation=float(table.get("ELEV", float("nan"))), declination=float(table.get("DECL", 0.0)),
            instrument=f"Phoenix {table.get('HW', 'MTU-5')}", serial=str(table.get("SNUM", "")),
            metadata={"source_format": "phoenix_legacy", "band": band, "file": str(ts_path),
                      "survey": table.get("SRVY", ""), "table": str(table_path or ""),
                      "notes": run_notes})
        start_limit, end_limit = as_datetime64(start), as_datetime64(end)
        for run in found:
            if start_limit is not None or end_limit is not None:
                run = run.window(start_limit, end_limit)
            if run.n_samples:
                runs.append(run)
    return runs


def _legacy_channel(component: str, counts: np.ndarray, table: Dict[str, Any],
                    calibration: Dict[str, Any]) -> Channel:
    fscv = table.get("FSCV")
    electric = component.startswith("e")
    response: List[ResponseStage] = []
    dipole = float("nan")
    if fscv is None:
        data, units = counts.astype(np.float32), "counts"
    else:
        gain = float(table.get("EGN", 1)) if electric else float(table.get("HGN", 1)) * float(table.get("HATT", 1.0))
        data, units = (counts * (float(fscv) / 2**23 / gain)).astype(np.float32), "V"
    if electric:
        dipole = float(table.get("EXLN" if component == "ex" else "EYLN", 0.0) or 0.0)
        azimuth = float(table.get("EAZM", 0.0)) + (90.0 if component == "ey" else 0.0)
        if units == "V" and dipole > 0:
            response = [dipole_stage(dipole), unit_stage("mV", "V")]
        sensor, serial = "electrode dipole", ""
    else:
        azimuth = float(table.get("HAZM", 0.0)) + {"hy": 90.0}.get(component, 0.0)
        serial = str(table.get(f"{component.upper()}SN", ""))
        sensor = "induction coil"
        coil = calibration.get(component)
        if isinstance(coil, (str, Path)):
            coil = read_response_table(coil, name=f"coil_{component}")
        if coil is None:
            sensitivity = float(table.get("HNOM", table.get("HNUM", 0.0)) or 0.0)
            if sensitivity > 0:
                coil = ResponseStage(f"coil_{serial or component}_flat", "nT", "mV", gain=sensitivity,
                                     metadata={"note": "flat sensitivity HNOM from the TBL file"})
        if units == "V" and coil is not None:
            response = [coil, unit_stage("mV", "V")]
    if component == "hz":
        azimuth = 0.0
    return Channel(component=component, data=data, units=units, response=response,
                   azimuth=azimuth, dipole_length=dipole, sensor=sensor, sensor_serial=serial,
                   metadata={"gain_egn": table.get("EGN"), "gain_hgn": table.get("HGN"),
                             "hatt": table.get("HATT"), "fscv": fscv})
