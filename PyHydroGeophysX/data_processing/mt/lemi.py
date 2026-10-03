"""LEMI-424 long-period MT stations: the logger's 1 S/s text files.

Each line is one second: year, month, day, hour, minute, second (UTC); Bx,
By, Bz in nT from the fluxgate; the electronics' and the sensor's
temperatures; E1-E4 in mV (the potential differences of up to four
dipoles); battery voltage; elevation; latitude and longitude as
``ddmm.mmmm`` with a hemisphere letter; satellites; fix flag; time
difference. Files usually hold a day (or an hour) and are joined in time.

A fluxgate measures B itself, flat across the MT band, so the magnetic
channels are already in nT. The files do not hold the dipole lengths: pass
``dipole_lengths`` (m) to bring E1/E2 to mV/km; E1 is taken as Ex and E2 as
Ey, E3 and E4 keep their names unless ``components`` renames them.

LEMI LLC (2009). LEMI-424 long-period magnetotelluric station: user manual.
Lviv Centre of Institute of Space Research, Lviv, Ukraine.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from .timeseries import (
    Channel,
    TimeSeriesRun,
    as_datetime64,
    assemble_runs,
    contiguous_blocks,
    dipole_stage,
    read_response_table,
)

_COLUMNS = ("bx", "by", "bz", "temperature_e", "temperature_h", "e1", "e2", "e3", "e4",
            "battery", "elevation", "latitude", "lat_hemisphere", "longitude", "lon_hemisphere",
            "n_satellites", "gps_fix", "time_diff")
#: LEMI's names for the channels, and the components they become.
_RENAME = {"bx": "hx", "by": "hy", "bz": "hz", "e1": "ex", "e2": "ey", "e3": "e3", "e4": "e4"}


def is_lemi424_file(path: Any) -> bool:
    source = Path(path)
    if not source.is_file() or source.suffix.lower() != ".txt":
        return False
    try:
        with source.open("r", encoding="latin-1") as handle:
            parts = handle.readline().split()
    except OSError:
        return False
    return len(parts) >= 24 and parts[18] in ("N", "S") and parts[20] in ("E", "W")


def _degrees(value: float, hemisphere: str) -> float:
    degrees = int(value / 100)
    result = degrees + (value - 100 * degrees) / 60.0
    return -result if hemisphere in ("S", "W") else result


def _read_file(path: Path) -> Dict[str, Any]:
    rows = [line.split() for line in path.read_text(encoding="latin-1").splitlines() if line.strip()]
    rows = [r for r in rows if len(r) >= 24]
    if not rows:
        raise ValueError(f"{path.name} has no LEMI-424 records")
    numbers = np.array([[float(v) for v in r[:18]] + [float(r[19]), float(r[21]), float(r[22]),
                                                      float(r[23])] for r in rows])
    stamps = np.array([f"{int(r[0]):04d}-{int(r[1]):02d}-{int(r[2]):02d}T{int(r[3]):02d}:"
                       f"{int(r[4]):02d}:{int(r[5]):02d}" for r in rows], dtype="datetime64[ns]")
    return {"times": stamps, "data": {name: numbers[:, 6 + i] for i, name in enumerate(_COLUMNS[:11])},
            "latitude": _degrees(numbers[0, 17], rows[0][18]),
            "longitude": _degrees(numbers[0, 18], rows[0][20])}


def read_lemi424(path: Any, *, dipole_lengths: Optional[Dict[str, float]] = None,
                 components: Optional[Iterable[str]] = None, station: str = "",
                 calibration: Optional[Dict[str, Any]] = None, start: Any = None,
                 end: Any = None) -> List[TimeSeriesRun]:
    """Read LEMI-424 text files - one, a folder of them, or a list - into runs.

    ``dipole_lengths`` maps ``"ex"``/``"ey"`` (or ``"e1"``...) to metres.
    ``components`` lists the channels to keep, by LEMI name or component
    (default: hx, hy, hz, ex, ey). ``calibration`` maps a component to a
    text table (frequency, amplitude, phase in degrees) of its response.
    """
    source = Path(path) if isinstance(path, (str, Path)) else None
    if source is not None and source.is_dir():
        files = sorted(p for p in source.iterdir() if is_lemi424_file(p))
    elif source is not None:
        files = [source]
    else:
        files = [Path(p) for p in path]
    if not files:
        raise ValueError(f"no LEMI-424 files found in {path}")
    lengths = {_RENAME.get(str(k).lower(), str(k).lower()): float(v) for k, v in (dipole_lengths or {}).items()}
    keep = ["hx", "hy", "hz", "ex", "ey"] if components is None else [
        _RENAME.get(str(c).lower(), str(c).lower()) for c in components]
    reverse = {v: k for k, v in _RENAME.items()}
    tables = {str(k).lower(): v for k, v in (calibration or {}).items()}

    parsed = [_read_file(p) for p in files]
    pieces: Dict[str, List] = {}
    for component in keep:
        column = reverse.get(component, component)
        blocks = []
        for item in parsed:
            times, values = item["times"], item["data"][column]
            steps = np.diff(times) != np.timedelta64(1, "s")
            for first, last in zip(np.r_[0, np.flatnonzero(steps) + 1], np.r_[np.flatnonzero(steps) + 1, times.size]):
                blocks.append((times[first], values[first:last]))
        pieces[component] = contiguous_blocks(blocks, 1.0)

    notes: List[str] = []

    def make_channel(component: str, data: np.ndarray) -> Channel:
        response = []
        units = "nT" if component.startswith("h") else "mV"
        if component.startswith("e"):
            length = lengths.get(component, 0.0)
            if length > 0:
                response = [dipole_stage(length)]
            else:
                notes.append(f"no dipole length for {component}: it stays in mV")
        table = tables.get(component)
        if table is not None:
            stage = read_response_table(table, name=f"lemi_{component}",
                                        input_units="nT" if units == "nT" else "mV/km", output_units=units)
            response = response + [stage] if component.startswith("e") else [stage]
        return Channel(component=component, data=data, units=units, response=response,
                       dipole_length=lengths.get(component, float("nan")),
                       sensor="LEMI-424 fluxgate" if units == "nT" else "electrode dipole")

    found = assemble_runs(pieces, 1.0, make_channel, station=station or files[0].parent.name,
                          latitude=parsed[0]["latitude"], longitude=parsed[0]["longitude"],
                          elevation=float(np.median(parsed[0]["data"]["elevation"])),
                          instrument="LEMI-424",
                          metadata={"source_format": "lemi424", "files": [str(p) for p in files],
                                    "notes": notes})
    runs = []
    start_limit, end_limit = as_datetime64(start), as_datetime64(end)
    for run in found:
        run.metadata["notes"] = list(dict.fromkeys(notes))
        if start_limit is not None or end_limit is not None:
            run = run.window(start_limit, end_limit)
        if run.n_samples:
            runs.append(run)
    return runs
