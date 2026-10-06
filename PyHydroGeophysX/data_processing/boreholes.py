"""Boreholes along a geophysical profile: wells, lithology logs and water levels.

A section is read against the ground truth beside it - the cores, the CMT and
monitoring-well logs, the water levels - and that is only fair when each well
is put where it belongs: projected onto the profile, with the distance it sits
off the line written next to it, its depths on the section's own vertical
datum, and its water level from the date the survey was made. This module holds
the data and the geometry; :func:`draw_borehole_columns` draws them on a
matplotlib section, and the studio's Boreholes page
(``qt_apps/modules/boreholes.py``) loads and edits them.

Nothing here knows about any site: every well, unit and colour comes from the
user's files and the legend table.

Column templates
----------------
CSV (any delimiter) or XLSX (first sheet), one header row. Header names are
matched without regard to case, spaces or underscores, and the aliases in
brackets are accepted. :func:`write_templates` writes all three with an example
row.

**Wells** - one row per well:

``well_id`` [id, well, name, borehole], ``easting`` [x, east], ``northing``
[y, north], ``ground_elevation`` [elevation, z, ground, surface_elevation]
(m, optional), ``crs`` [epsg, srs] (e.g. ``EPSG:26915``, optional),
``datum_offset`` [vertical_offset, datum_shift] (m, optional: added to the
ground elevation and to heads to put them on the section's vertical datum -
e.g. a surveyed elevation against a lidar surface), ``source`` [reference,
data_source] (optional, shown under the column).

**Lithology** - one row per interval:

``well_id``, ``top`` [from, depth_top, top_depth], ``bottom`` [to,
depth_bottom, bottom_depth] (depths below ground), ``unit`` [lithology, class,
code], ``description`` [desc, notes] (optional), ``length_unit`` [units]
(``m`` or ``ft``, optional; the file's unit otherwise).

**Geophysical logs** - borehole geophysics (natural gamma, conductivity,
resistivity, temperature, ...), each a curve against depth:

a table with ``well_id``, ``depth`` [md, depth_m, dept] (below ground) and one
column per curve, the unit in brackets after the name (``gamma [API]``,
``EC (mS/m)``), optional ``length_unit``; or a LAS 2.0 file (Canadian Well
Logging Society), the first curve the depth and ``WELL`` in its ``~W`` section
naming the well (:func:`read_las`).

**Water levels** - one row per measurement:

``well_id``, ``datetime`` [date, time, timestamp], and either
``depth_below_ground`` [depth, dtw, depth_to_water] or ``head_elevation``
[head, water_elevation]; optional ``reference`` [reference_point,
measuring_point] (text) and ``reference_height`` [stickup, mp_height] (height of
the measuring point above ground: a depth measured from it is reduced to depth
below ground), ``length_unit``.

Depths and heights are stored in metres; a file in feet is converted on reading.
"""

from __future__ import annotations

import datetime as _dt
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "Well",
    "LithologyInterval",
    "WaterLevel",
    "GeophysicalLog",
    "BoreholeData",
    "ProfileLine",
    "ProjectedWell",
    "read_table",
    "read_wells",
    "read_lithology",
    "read_water_levels",
    "read_log_table",
    "read_las",
    "read_geophysical_logs",
    "load_boreholes",
    "write_templates",
    "default_legend",
    "project_wells",
    "column_water",
    "column_label",
    "draw_borehole_columns",
    "nearby_groups",
    "draw_well_map",
    "draw_well_logs",
    "draw_water_levels",
    "draw_well_comparison",
    "FEET",
]

#: Metres in one international foot.
FEET = 0.3048

_ALIASES = {
    "well_id": ("wellid", "id", "well", "name", "borehole", "boreholeid", "wellname"),
    "easting": ("easting", "x", "east", "e", "xcoord"),
    "northing": ("northing", "y", "north", "n", "ycoord"),
    "ground_elevation": ("groundelevation", "elevation", "z", "ground", "surfaceelevation",
                         "elev", "groundelev", "landsurface"),
    "crs": ("crs", "epsg", "srs", "coordinatesystem"),
    "datum_offset": ("datumoffset", "verticaloffset", "datumshift", "verticaldatumoffset"),
    "source": ("source", "reference", "datasource", "ref"),
    "top": ("top", "from", "depthtop", "topdepth", "depthfrom"),
    "bottom": ("bottom", "to", "depthbottom", "bottomdepth", "depthto", "base"),
    "unit": ("unit", "lithology", "class", "code", "lith", "lithologyunit"),
    "description": ("description", "desc", "notes", "comment"),
    "length_unit": ("lengthunit", "units", "unitsoflength", "depthunit"),
    "datetime": ("datetime", "date", "time", "timestamp", "measured", "datemeasured"),
    "depth_below_ground": ("depthbelowground", "depth", "dtw", "depthtowater", "waterdepth",
                           "depthbgs"),
    "head_elevation": ("headelevation", "head", "waterelevation", "wlelevation",
                       "watertableelevation", "hydraulichead"),
    "reference": ("reference", "referencepoint", "measuringpoint", "mp"),
    "reference_height": ("referenceheight", "stickup", "mpheight", "measuringpointheight"),
}

#: Fill colours handed out to lithology units in the order they are first met.
#: Muted earth and grey tones that read on a light plot canvas.
_LITHOLOGY_COLORS = (
    "#e8d7a8", "#c9c9c9", "#f2c14e", "#6b4f3a", "#a7a7a7", "#d9c58a", "#b9b4a6",
    "#8c6d46", "#d4b483", "#9fb7a3", "#c97b63", "#7d8c99",
)


def _norm(name: Any) -> str:
    return "".join(ch for ch in str(name).strip().lower() if ch.isalnum())


def _unit_factor(unit: Any, default: str = "m") -> float:
    text = _norm(unit) if unit is not None and str(unit).strip() and str(unit) != "nan" else default
    if text in ("m", "meter", "meters", "metre", "metres"):
        return 1.0
    if text in ("ft", "feet", "foot", "usft"):
        return FEET
    raise ValueError(f"Unknown length unit {unit!r}; use m or ft.")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class Well:
    """A well's location; ``ground_elevation`` and ``datum_offset`` in metres."""

    well_id: str
    easting: float
    northing: float
    ground_elevation: float = float("nan")
    crs: str = ""
    datum_offset: float = 0.0
    source: str = ""

    @property
    def section_elevation(self) -> float:
        """Ground elevation on the section's datum: surveyed plus the datum offset."""
        return float(self.ground_elevation) + float(self.datum_offset or 0.0)


@dataclass
class LithologyInterval:
    """One logged interval, ``top`` and ``bottom`` in metres below ground."""

    well_id: str
    top: float
    bottom: float
    unit: str
    description: str = ""


@dataclass
class WaterLevel:
    """One water-level measurement: a depth below ground (m) or a head elevation (m)."""

    well_id: str
    time: Optional[_dt.datetime]
    depth: float = float("nan")
    head: float = float("nan")
    reference: str = ""

    def depth_below_ground(self, well: Optional[Well] = None) -> float:
        """Depth to water below ground, from the depth or from the head and the well's ground."""
        if np.isfinite(self.depth):
            return float(self.depth)
        if well is not None and np.isfinite(self.head) and np.isfinite(well.ground_elevation):
            return float(well.ground_elevation) - float(self.head)
        return float("nan")


@dataclass
class GeophysicalLog:
    """One borehole geophysical curve: ``values`` against ``depth`` (m below ground)."""

    well_id: str
    name: str
    unit: str
    depth: np.ndarray
    values: np.ndarray
    source: str = ""

    def __post_init__(self) -> None:
        depth = np.asarray(self.depth, dtype=float).ravel()
        values = np.asarray(self.values, dtype=float).ravel()
        if depth.size != values.size:
            raise ValueError(f"Log {self.name!r} of {self.well_id}: {depth.size} depths, "
                             f"{values.size} values.")
        keep = np.isfinite(depth)
        order = np.argsort(depth[keep], kind="stable")
        self.depth, self.values = depth[keep][order], values[keep][order]

    @property
    def label(self) -> str:
        return f"{self.name} ({self.unit})" if self.unit else self.name


@dataclass
class BoreholeData:
    """Wells, their lithology and water levels, and the legend the units are drawn with.

    ``legend`` maps a unit name to ``(colour, label)``; units not in it are
    added with :func:`default_legend` colours when the data are loaded.
    """

    wells: Dict[str, Well] = field(default_factory=dict)
    lithology: List[LithologyInterval] = field(default_factory=list)
    water_levels: List[WaterLevel] = field(default_factory=list)
    legend: Dict[str, Tuple[str, str]] = field(default_factory=dict)
    logs: List[GeophysicalLog] = field(default_factory=list)

    def well_logs(self, well_id: str) -> List[GeophysicalLog]:
        return [log for log in self.logs if log.well_id == well_id]

    def intervals(self, well_id: str) -> List[LithologyInterval]:
        return sorted((i for i in self.lithology if i.well_id == well_id), key=lambda i: i.top)

    def levels(self, well_id: str) -> List[WaterLevel]:
        return [w for w in self.water_levels if w.well_id == well_id]

    def units(self) -> List[str]:
        seen: List[str] = []
        for interval in self.lithology:
            if interval.unit not in seen:
                seen.append(interval.unit)
        return seen

    def complete_legend(self) -> None:
        """Give every unit in the logs a legend entry, keeping the ones already set."""
        self.legend = default_legend(self.units(), self.legend)

    def water_level(self, well_id: str, when: Optional[_dt.datetime] = None,
                    tolerance_days: Optional[float] = None) -> Optional[WaterLevel]:
        """The measurement nearest ``when`` (the latest without one), or None.

        With ``tolerance_days`` a measurement farther than that from ``when``
        does not count.
        """
        levels = [w for w in self.levels(well_id)
                  if np.isfinite(w.depth) or np.isfinite(w.head)]
        if not levels:
            return None
        dated = [w for w in levels if w.time is not None]
        if when is None:
            return max(dated, key=lambda w: w.time) if dated else levels[-1]
        if not dated:
            return None
        best = min(dated, key=lambda w: abs((w.time - when).total_seconds()))
        if tolerance_days is not None and abs((best.time - when).total_seconds()) > tolerance_days * 86400:
            return None
        return best

    def water_range(self, well_id: str) -> Optional[Tuple[float, float, int]]:
        """``(shallowest, deepest, count)`` of the depths to water below ground."""
        well = self.wells.get(well_id)
        depths = [w.depth_below_ground(well) for w in self.levels(well_id)]
        depths = [d for d in depths if np.isfinite(d)]
        if not depths:
            return None
        return float(min(depths)), float(max(depths)), len(depths)


def default_legend(units: Iterable[str],
                   existing: Optional[Dict[str, Tuple[str, str]]] = None) -> Dict[str, Tuple[str, str]]:
    """A ``unit -> (colour, label)`` table: ``existing`` entries kept, new units coloured in turn."""
    legend = dict(existing or {})
    used = {colour for colour, _ in legend.values()}
    palette = [c for c in _LITHOLOGY_COLORS if c not in used] or list(_LITHOLOGY_COLORS)
    k = 0
    for unit in units:
        if unit in legend:
            continue
        legend[unit] = (palette[k % len(palette)], str(unit))
        k += 1
    return legend


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def read_table(source: Any, sheet: Any = 0):
    """A pandas DataFrame from a CSV/TXT (delimiter sniffed) or XLSX/XLS path, or a DataFrame."""
    import pandas as pd

    if isinstance(source, pd.DataFrame):
        return source.copy()
    path = Path(source)
    if path.suffix.lower() in (".xlsx", ".xlsm", ".xls"):
        return pd.read_excel(path, sheet_name=sheet)
    return pd.read_csv(path, sep=None, engine="python", comment="#", skipinitialspace=True)


def _columns(frame, required: Sequence[str], optional: Sequence[str], what: str) -> Dict[str, str]:
    """``canonical -> actual`` column names, raising with the template when one is missing."""
    lookup = {_norm(c): c for c in frame.columns}
    found: Dict[str, str] = {}
    for key in list(required) + list(optional):
        for alias in (_norm(key),) + _ALIASES.get(key, ()):
            if alias in lookup:
                found[key] = lookup[alias]
                break
    missing = [k for k in required if k not in found]
    if missing:
        raise ValueError(
            f"The {what} table has no {', '.join(missing)} column. Expected columns: "
            f"{', '.join(required)} (and optionally {', '.join(optional)}); found "
            f"{', '.join(map(str, frame.columns))}.")
    return found


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def _number(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out


def _well_key(value: Any) -> str:
    text = _text(value)
    # 12.0 read by pandas from an integer id column is well "12".
    try:
        number = float(text)
        if number.is_integer():
            return str(int(number))
    except ValueError:
        pass
    return text


def read_wells(source: Any, *, default_crs: str = "") -> List[Well]:
    """Wells from a table (see the module's column template)."""
    frame = read_table(source)
    cols = _columns(frame, ("well_id", "easting", "northing"),
                    ("ground_elevation", "crs", "datum_offset", "source"), "wells")
    wells: List[Well] = []
    for _, row in frame.iterrows():
        well_id = _well_key(row[cols["well_id"]])
        if not well_id:
            continue
        crs = _text(row[cols["crs"]]) if "crs" in cols else ""
        if crs and crs.isdigit():
            crs = f"EPSG:{crs}"
        offset = _number(row[cols["datum_offset"]]) if "datum_offset" in cols else 0.0
        wells.append(Well(
            well_id=well_id, easting=_number(row[cols["easting"]]),
            northing=_number(row[cols["northing"]]),
            ground_elevation=_number(row[cols["ground_elevation"]]) if "ground_elevation" in cols else float("nan"),
            crs=crs or default_crs, datum_offset=offset if np.isfinite(offset) else 0.0,
            source=_text(row[cols["source"]]) if "source" in cols else ""))
    return wells


def read_lithology(source: Any, *, length_unit: str = "m") -> List[LithologyInterval]:
    """Lithology intervals from a table; depths converted to metres below ground."""
    frame = read_table(source)
    cols = _columns(frame, ("well_id", "top", "bottom", "unit"), ("description", "length_unit"),
                    "lithology")
    out: List[LithologyInterval] = []
    for _, row in frame.iterrows():
        well_id = _well_key(row[cols["well_id"]])
        if not well_id:
            continue
        factor = _unit_factor(row[cols["length_unit"]] if "length_unit" in cols else None, length_unit)
        top, bottom = _number(row[cols["top"]]) * factor, _number(row[cols["bottom"]]) * factor
        if not (np.isfinite(top) and np.isfinite(bottom)):
            continue
        if bottom < top:
            top, bottom = bottom, top
        out.append(LithologyInterval(well_id, float(top), float(bottom),
                                     _text(row[cols["unit"]]) or "unknown",
                                     _text(row[cols["description"]]) if "description" in cols else ""))
    return out


def _parse_time(value: Any) -> Optional[_dt.datetime]:
    import pandas as pd

    if value is None or (isinstance(value, float) and math.isnan(value)) or _text(value) == "":
        return None
    try:
        stamp = pd.to_datetime(value)
    except (ValueError, TypeError):
        return None
    if pd.isna(stamp):
        return None
    return stamp.to_pydatetime().replace(tzinfo=None)


def read_water_levels(source: Any, *, length_unit: str = "m") -> List[WaterLevel]:
    """Water levels from a table; depths and heads converted to metres.

    A depth measured from a reference point ``reference_height`` above ground
    is reduced to a depth below ground.
    """
    frame = read_table(source)
    cols = _columns(frame, ("well_id",), ("datetime", "depth_below_ground", "head_elevation",
                                          "reference", "reference_height", "length_unit"),
                    "water-level")
    if "depth_below_ground" not in cols and "head_elevation" not in cols:
        raise ValueError("The water-level table needs a depth_below_ground or a head_elevation "
                         f"column; found {', '.join(map(str, frame.columns))}.")
    out: List[WaterLevel] = []
    for _, row in frame.iterrows():
        well_id = _well_key(row[cols["well_id"]])
        if not well_id:
            continue
        factor = _unit_factor(row[cols["length_unit"]] if "length_unit" in cols else None, length_unit)
        depth = _number(row[cols["depth_below_ground"]]) * factor if "depth_below_ground" in cols else float("nan")
        head = _number(row[cols["head_elevation"]]) * factor if "head_elevation" in cols else float("nan")
        lift = _number(row[cols["reference_height"]]) * factor if "reference_height" in cols else float("nan")
        if np.isfinite(depth) and np.isfinite(lift):
            depth -= lift
        if not (np.isfinite(depth) or np.isfinite(head)):
            continue
        out.append(WaterLevel(well_id, _parse_time(row[cols["datetime"]]) if "datetime" in cols else None,
                              float(depth), float(head),
                              _text(row[cols["reference"]]) if "reference" in cols else ""))
    return out


def _curve_name(header: Any) -> Tuple[str, str]:
    """``'gamma [API]'`` / ``'EC (mS/m)'`` -> ``('gamma', 'API')``; no brackets, no unit."""
    import re

    text = str(header).strip()
    match = re.match(r"^(.*?)\s*[\[(]\s*([^\])]*)\s*[\])]\s*$", text)
    return (match.group(1).strip(), match.group(2).strip()) if match else (text, "")


def read_log_table(source: Any, *, length_unit: str = "m", source_name: str = "") -> List[GeophysicalLog]:
    """Geophysical logs from a table: ``well_id``, ``depth`` and one column per curve."""
    frame = read_table(source)
    depth_aliases = ("depth", "md", "depthm", "dept", "measureddepth")
    norm = {_norm(c): c for c in frame.columns}
    well_col = next((norm[a] for a in _ALIASES["well_id"] + ("wellid",) if a in norm), None)
    depth_col = next((norm[a] for a in depth_aliases if a in norm), None)
    if well_col is None or depth_col is None:
        raise ValueError("The log table needs a well_id and a depth column; found "
                         f"{', '.join(map(str, frame.columns))}.")
    unit_col = next((norm[a] for a in _ALIASES["length_unit"] if a in norm), None)
    skip = {well_col, depth_col, unit_col}
    curves = [c for c in frame.columns if c not in skip
              and np.issubdtype(frame[c].dtype, np.number)]
    if not curves:
        raise ValueError("The log table has no numeric curve columns besides the depth.")
    out: List[GeophysicalLog] = []
    for well, rows in frame.groupby(frame[well_col].map(_well_key)):
        if not well:
            continue
        unit = rows[unit_col].iloc[0] if unit_col is not None else None
        depth = rows[depth_col].to_numpy(dtype=float) * _unit_factor(unit, length_unit)
        for column in curves:
            name, curve_unit = _curve_name(column)
            out.append(GeophysicalLog(well, name, curve_unit, depth,
                                      rows[column].to_numpy(dtype=float), source_name))
    return out


def read_las(path: Any, *, well_id: Optional[str] = None) -> List[GeophysicalLog]:
    """The curves of a LAS 2.0 file (CWLS Log ASCII Standard), depths in metres.

    The first curve of ``~C`` is the depth (``M``, ``F`` or ``FT``); the
    ``NULL`` value of ``~W`` becomes NaN; ``WELL`` names the well unless
    ``well_id`` is given. Wrapped files (``WRAP. YES``) are not read.
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    section, info, curves, rows = "", {}, [], []
    for raw in text:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("~"):
            section = line[1:2].upper()
            continue
        if section in ("V", "W", "C"):
            mnemonic, _, rest = line.partition(".")
            unit, _, rest = rest.partition(" ")
            value = rest.split(":", 1)[0].strip()
            key = mnemonic.strip().upper()
            if section == "C":
                curves.append((mnemonic.strip(), unit.strip()))
            else:
                info[key] = value
        elif section == "A":
            rows.extend(float(v) for v in line.split())
    if info.get("WRAP", "NO").upper().startswith("Y"):
        raise ValueError("Wrapped LAS files are not read; save it unwrapped (WRAP. NO).")
    if len(curves) < 2 or not rows:
        raise ValueError("The LAS file has no curves besides the depth, or no data.")
    data = np.asarray(rows, dtype=float)
    if data.size % len(curves):
        raise ValueError(f"The LAS data do not divide into {len(curves)} curves.")
    data = data.reshape(-1, len(curves))
    null = info.get("NULL")
    if null:
        try:
            data[np.isclose(data, float(null))] = np.nan
        except ValueError:
            pass
    depth_unit = curves[0][1].upper()
    factor = FEET if depth_unit in ("F", "FT", "FEET") else 1.0
    well = _well_key(well_id or info.get("WELL") or Path(path).stem)
    return [GeophysicalLog(well, name, unit, data[:, 0] * factor, data[:, k], Path(path).name)
            for k, (name, unit) in enumerate(curves) if k > 0]


def read_geophysical_logs(path: Any, *, well_id: Optional[str] = None,
                          length_unit: str = "m") -> List[GeophysicalLog]:
    """A LAS file or a log table, by its extension."""
    if str(path).lower().endswith(".las"):
        return read_las(path, well_id=well_id)
    return read_log_table(path, length_unit=length_unit, source_name=Path(str(path)).name)


def load_boreholes(wells: Any = None, lithology: Any = None, water_levels: Any = None, *,
                   length_unit: str = "m",
                   legend: Optional[Dict[str, Tuple[str, str]]] = None) -> BoreholeData:
    """Read any of the three tables into one :class:`BoreholeData`."""
    data = BoreholeData(legend=dict(legend or {}))
    if wells is not None:
        data.wells = {w.well_id: w for w in read_wells(wells)}
    if lithology is not None:
        data.lithology = read_lithology(lithology, length_unit=length_unit)
    if water_levels is not None:
        data.water_levels = read_water_levels(water_levels, length_unit=length_unit)
    data.complete_legend()
    return data


def write_templates(folder: Any) -> List[Path]:
    """Write ``wells_template.csv``, ``lithology_template.csv`` and ``water_levels_template.csv``.

    Each has the header row and one example row to overwrite.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    files = {
        "wells_template.csv": (
            "well_id,easting,northing,ground_elevation,crs,datum_offset,source\n"
            "MW-1,500000.0,4600000.0,250.00,EPSG:26915,0.0,driller's log 2021\n"),
        "lithology_template.csv": (
            "well_id,top,bottom,unit,description,length_unit\n"
            "MW-1,0.0,1.2,soil,topsoil,m\n"),
        "water_levels_template.csv": (
            "well_id,datetime,depth_below_ground,head_elevation,reference,reference_height,length_unit\n"
            "MW-1,2024-05-01 10:00,3.45,,top of casing,0.6,m\n"),
    }
    paths = []
    for name, text in files.items():
        path = folder / name
        path.write_text(text, encoding="utf-8")
        paths.append(path)
    return paths


# ---------------------------------------------------------------------------
# Projection onto a profile
# ---------------------------------------------------------------------------

_COMPASS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")


@dataclass
class ProfileLine:
    """A profile in map coordinates: a polyline, distance ``start_distance`` at its first vertex.

    For a section drawn in local along-line metres (geophone 1 at 0 m, say),
    give the line's map position with :meth:`from_bearing`; for wells already
    in along-line coordinates use :meth:`local`, which puts the line on the
    easting axis.
    """

    points: np.ndarray
    start_distance: float = 0.0

    def __post_init__(self) -> None:
        pts = np.asarray(self.points, dtype=float).reshape(-1, 2)
        if pts.shape[0] < 2:
            raise ValueError("A profile line needs at least two points.")
        keep = np.r_[True, np.any(np.diff(pts, axis=0) != 0, axis=1)]
        pts = pts[keep]
        if pts.shape[0] < 2:
            raise ValueError("A profile line needs two distinct points.")
        self.points = pts

    @classmethod
    def from_bearing(cls, easting: float, northing: float, bearing_deg: float,
                     length: float = 100.0, start_distance: float = 0.0) -> "ProfileLine":
        """A straight line from ``(easting, northing)`` toward ``bearing_deg`` (clockwise from north)."""
        b = math.radians(float(bearing_deg))
        end = (easting + length * math.sin(b), northing + length * math.cos(b))
        return cls(np.array([[easting, northing], end]), start_distance)

    @classmethod
    def local(cls, length: float = 100.0, start_distance: float = 0.0) -> "ProfileLine":
        """The line along the easting axis: easting is the along-line distance."""
        return cls(np.array([[start_distance, 0.0], [start_distance + length, 0.0]]), start_distance)

    @classmethod
    def from_file(cls, path: Any, start_distance: float = 0.0) -> "ProfileLine":
        """A polyline from a table with easting and northing columns (vertices in order)."""
        frame = read_table(path)
        cols = _columns(frame, ("easting", "northing"), (), "profile line")
        pts = frame[[cols["easting"], cols["northing"]]].to_numpy(dtype=float)
        return cls(pts[np.all(np.isfinite(pts), axis=1)], start_distance)

    @property
    def length(self) -> float:
        return float(np.sum(np.hypot(*np.diff(self.points, axis=0).T)))

    def project(self, easting: Any, northing: Any) -> Dict[str, np.ndarray]:
        """Along-line distance, signed offset and distance to the line of map points.

        ``along``: distance along the line to the nearest point on it (extended
        straight past either end, so a well beyond the end gets a distance off
        the section). ``offset``: perpendicular distance, positive to the right
        looking along the line. ``distance``: distance to the line itself, not
        extended. ``side``: compass direction from the line to the point.
        """
        e = np.atleast_1d(np.asarray(easting, dtype=float))
        n = np.atleast_1d(np.asarray(northing, dtype=float))
        p = np.column_stack([e, n])
        a, b = self.points[:-1], self.points[1:]
        seg = b - a
        seg_len = np.hypot(seg[:, 0], seg[:, 1])
        cum = np.r_[0.0, np.cumsum(seg_len)]
        n_seg = seg.shape[0]
        along = np.empty(p.shape[0])
        offset = np.empty(p.shape[0])
        dist = np.empty(p.shape[0])
        side = []
        for k, q in enumerate(p):
            t = ((q - a) * seg).sum(axis=1) / seg_len ** 2
            tc = np.clip(t, 0.0, 1.0)
            foot = a + tc[:, None] * seg
            d = np.hypot(*(q - foot).T)
            j = int(np.argmin(d))
            dist[k] = d[j]
            tj = t[j]
            # Extend the end segments, not the inner ones, past their vertices.
            if not ((j == 0 and tj < 0) or (j == n_seg - 1 and tj > 1)):
                tj = tc[j]
            foot_j = a[j] + tj * seg[j]
            along[k] = self.start_distance + cum[j] + tj * seg_len[j]
            u = seg[j] / seg_len[j]
            rel = q - foot_j
            offset[k] = rel[0] * u[1] - rel[1] * u[0]   # right of the direction of travel
            if np.hypot(*rel) > 0:
                bearing = (math.degrees(math.atan2(rel[0], rel[1])) + 360.0) % 360.0
                side.append(_COMPASS[int((bearing + 22.5) // 45) % 8])
            else:
                side.append("")
        return {"along": along, "offset": offset, "distance": dist, "side": np.array(side)}


@dataclass
class ProjectedWell:
    """A well placed on a profile: its distance along it and off it (m)."""

    well: Well
    along: float
    offset: float
    distance: float
    side: str


def project_wells(wells: Iterable[Well], profile: ProfileLine,
                  max_distance: Optional[float] = None) -> List[ProjectedWell]:
    """The wells within ``max_distance`` of the profile, in order along it."""
    wells = [w for w in wells if np.isfinite(w.easting) and np.isfinite(w.northing)]
    if not wells:
        return []
    geo = profile.project([w.easting for w in wells], [w.northing for w in wells])
    out = [ProjectedWell(w, float(geo["along"][k]), float(geo["offset"][k]),
                         float(geo["distance"][k]), str(geo["side"][k]))
           for k, w in enumerate(wells)]
    if max_distance is not None:
        out = [p for p in out if p.distance <= float(max_distance)]
    return sorted(out, key=lambda p: p.along)


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def column_water(data: BoreholeData, well: Well, *, ground: float = float("nan"),
                 when: Optional[_dt.datetime] = None,
                 tolerance_days: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """What a well's column shows of its water levels, in metres below the column's top.

    With ``when``, the reading nearest that date (within ``tolerance_days``):
    ``{"kind": "date", "depth", "text"}``. Without, the range of every
    reading: ``{"kind": "range", "shallow", "deep", "count", "text"}``. None
    when there is nothing to show. ``ground`` is the elevation of the column's
    top on the section's datum, which turns a head into a depth.

    Every renderer of the columns - matplotlib sections and the studio's
    pyqtgraph views - reads the water level here, so they cannot disagree.
    """
    def depth_of(level: WaterLevel) -> float:
        if np.isfinite(level.depth):
            return float(level.depth)
        if np.isfinite(level.head) and np.isfinite(ground):
            return float(ground) - (float(level.head) + float(well.datum_offset or 0.0))
        return float("nan")

    if when is not None:
        level = data.water_level(well.well_id, when, tolerance_days)
        if level is None or not np.isfinite(depth_of(level)):
            return None
        stamp = level.time.strftime("%Y-%m-%d") if level.time else "undated"
        return {"kind": "date", "depth": depth_of(level), "text": f"WT {stamp}"}
    depths = [d for d in (depth_of(w) for w in data.levels(well.well_id)) if np.isfinite(d)]
    if not depths:
        return None
    count = len(depths)
    return {"kind": "range", "shallow": min(depths), "deep": max(depths), "count": count,
            "text": f"WT range, {count} reading{'s' if count != 1 else ''}"}


def column_label(item: ProjectedWell, water: Optional[Dict[str, Any]] = None,
                 length_unit: Optional[str] = None) -> List[str]:
    """The lines under a column's well name: distance off the line, source, water shown."""
    from PyHydroGeophysX.visualization.axis_units import normalize_length_unit, to_display_length

    unit = normalize_length_unit(length_unit)
    off = to_display_length(abs(item.offset), unit)
    lines = [f"{off:.1f} {unit} {item.side} of line".replace("  ", " ")
             if round(off, 1) > 0 else "on line"]
    if item.well.source:
        lines.append(item.well.source)
    if water:
        lines.append(water["text"])
    return lines


def draw_borehole_columns(ax: Any, data: BoreholeData, projected: Sequence[ProjectedWell], *,
                          vertical: str = "depth", depth_reference: Optional[float] = None,
                          surface: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
                          width: Optional[float] = None, when: Optional[_dt.datetime] = None,
                          tolerance_days: Optional[float] = None, lithology: bool = True,
                          water: bool = True, labels: bool = True, legend: bool = True,
                          length_unit: Optional[str] = None, water_color: str = "#007aff",
                          edge_color: str = "#1d1d1f", z_order: float = 6.0,
                          depth_positive_down: bool = False) -> Dict[str, Any]:
    """Draw lithology columns and water levels on a section at the wells' positions.

    ``vertical='depth'``: the section's y is ``depth_reference - depth`` (an
    elevation-like coordinate whose axis reads depth, as
    :func:`~PyHydroGeophysX.visualization.axis_units.set_section_axes` draws a
    section without elevations; ``depth_reference`` defaults to 0). With
    ``depth_positive_down`` the section's y is the depth itself, for a view
    that plots depths as positive numbers on an inverted axis.
    ``vertical='elevation'``: y is the well's ground elevation plus its datum
    offset, minus the depth; a well without a ground elevation is hung from
    ``surface`` (``(x, z)`` of the section's ground) at its position, and is
    left out when there is none.

    Water: with ``when``, the measurement nearest that date (within
    ``tolerance_days``) is drawn as a marker; without it, the range of all of
    a well's measurements as a bar. Each column is labelled with the well, its
    distance off the line (in ``length_unit``, the plots' unit by default) and
    its data source. Returns ``{"drawn": [...], "skipped": [...], "handles":
    legend handles}``.
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle

    if width is None:
        lo, hi = ax.get_xlim()
        width = 0.03 * abs(hi - lo) or 1.0
    ref = 0.0 if depth_reference is None else float(depth_reference)
    drawn, skipped, units_used = [], [], []
    water_drawn = False
    # Labels of wells close together are staggered upward instead of overprinting.
    label_slots: List[Tuple[float, int]] = []
    for item in projected:
        well = item.well
        if vertical == "elevation":
            top = well.section_elevation
            if not np.isfinite(top) and surface is not None:
                sx, sz = (np.asarray(v, dtype=float) for v in surface)
                if sx.size:
                    top = float(np.interp(item.along, sx, sz))
            if not np.isfinite(top):
                skipped.append(f"{well.well_id}: no ground elevation and no section surface")
                continue

            def y_of(depth: float, top: float = top) -> float:
                return top - depth
        elif depth_positive_down:
            def y_of(depth: float) -> float:
                return ref + depth
        else:
            def y_of(depth: float) -> float:
                return ref - depth
        x = item.along
        ground = (top if vertical == "elevation" else well.section_elevation)
        intervals = data.intervals(well.well_id) if lithology else []
        for interval in intervals:
            colour = data.legend.get(interval.unit, ("#d9d9d9", interval.unit))[0]
            y0, y1 = y_of(interval.bottom), y_of(interval.top)
            ax.add_patch(Rectangle((x - width / 2, min(y0, y1)), width, abs(y1 - y0),
                                   facecolor=colour, edgecolor=edge_color, linewidth=0.6,
                                   zorder=z_order))
            if interval.unit not in units_used:
                units_used.append(interval.unit)
        y_top = y_of(0.0)
        bottom = max((i.bottom for i in intervals), default=0.0)
        if not intervals:
            ax.plot([x, x], [y_top, y_of(max(bottom, 0.0))], color=edge_color, lw=1.0, zorder=z_order)
        shown = (column_water(data, well, ground=ground, when=when,
                              tolerance_days=tolerance_days) if water else None)
        if shown is not None and shown["kind"] == "date":
            ax.plot([x + 0.75 * width], [y_of(shown["depth"])], marker="v", markersize=8,
                    color=water_color, markeredgecolor=edge_color, markeredgewidth=0.5,
                    linestyle="none", zorder=z_order + 1)
            water_drawn = True
        elif shown is not None:
            y_a, y_b = y_of(shown["shallow"]), y_of(shown["deep"])
            bar = max(abs(y_a - y_b), 0.02 * width)
            ax.add_patch(Rectangle((x + 0.55 * width, min(y_a, y_b)), 0.35 * width, bar,
                                   facecolor=water_color, edgecolor="none", alpha=0.85,
                                   zorder=z_order + 1))
            water_drawn = True
        if labels:
            lines = column_label(item, shown, length_unit)
            px = float(ax.transData.transform((x, y_top))[0])
            level = 0
            for other_px, other_level in label_slots:
                if abs(px - other_px) < 80.0 and other_level >= level:
                    level = other_level + 1
            label_slots.append((px, level))
            lift = 44 * level
            ax.annotate(well.well_id, (x, y_top), xytext=(0, 4 + lift), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, fontweight="bold", zorder=z_order + 2,
                        annotation_clip=False)
            ax.annotate("\n".join(lines), (x, y_top), xytext=(0, 15 + lift), textcoords="offset points",
                        ha="center", va="bottom", fontsize=6.5, zorder=z_order + 2,
                        annotation_clip=False)
        drawn.append(well.well_id)
    handles = [Patch(facecolor=data.legend.get(u, ("#d9d9d9", u))[0], edgecolor=edge_color,
                     label=data.legend.get(u, ("#d9d9d9", u))[1]) for u in units_used]
    if water_drawn:
        handles.append(Line2D([], [], marker="v" if when is not None else "s", linestyle="none",
                              color=water_color, label="water level" if when is not None
                              else "water-level range"))
    if legend and handles:
        # Under the section and its axis label, where it covers neither the model nor a log.
        from matplotlib.transforms import offset_copy

        below = offset_copy(ax.transAxes, fig=ax.figure, y=-34, units="points")
        ax.legend(handles=handles, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 0.0),
                  bbox_transform=below, ncol=min(4, len(handles)), frameon=False)
    return {"drawn": drawn, "skipped": skipped, "handles": handles,
            "label_rows": 1 + max((lvl for _, lvl in label_slots), default=0)}


def nearby_groups(xy: Any, fraction: float = 0.02) -> List[List[int]]:
    """Indices of points closer together than ``fraction`` of their extent, grouped.

    A nest of monitoring wells beside a CMT is one group, so a map labels it
    once, listing them, instead of printing the names over each other.
    """
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    if not len(xy):
        return []
    near = fraction * max(float(np.ptp(xy[:, 0])), float(np.ptp(xy[:, 1])), 1.0)
    groups: List[List[int]] = []
    for k in range(len(xy)):
        for group in groups:
            if np.hypot(*(xy[group].mean(axis=0) - xy[k])) <= near:
                group.append(k)
                break
        else:
            groups.append([k])
    return groups


def draw_well_map(ax: Any, data: BoreholeData, *, selected: Sequence[str] = (),
                  length_unit: Optional[str] = None, color: str = "#007aff",
                  ink: str = "#1d1d1f") -> List[str]:
    """The wells in plan view, labelled; ``selected`` ones filled, the rest hollow.

    Coordinates are drawn as the wells give them (easting and northing, metres),
    ticked in ``length_unit``. Wells closer together than 2 % of the map's
    extent - a nest of monitoring wells beside a CMT - share one label listing
    them, rather than printing over each other. Returns the wells drawn.
    """
    from PyHydroGeophysX.visualization.axis_units import set_length_axis

    wells = [w for w in data.wells.values() if np.isfinite(w.easting) and np.isfinite(w.northing)]
    chosen = set(selected)
    if not wells:
        return []
    xy = np.array([(w.easting, w.northing) for w in wells])
    groups = nearby_groups(xy)
    for well in wells:
        filled = well.well_id in chosen
        ax.plot([well.easting], [well.northing], marker="o", markersize=8 if filled else 6,
                markerfacecolor=color if filled else "none", markeredgecolor=color,
                markeredgewidth=1.4, linestyle="none", zorder=3)
    for group in groups:
        names = sorted(wells[k].well_id for k in group)
        at = xy[group].max(axis=0)
        ax.annotate("\n".join(names), (at[0], at[1]), xytext=(6, 2), textcoords="offset points",
                    fontsize=8, color=ink, va="top",
                    fontweight="bold" if any(n in chosen for n in names) else "normal")
    ax.set_aspect("equal", adjustable="datalim")
    set_length_axis(ax, "x", "Easting", unit=length_unit)
    set_length_axis(ax, "y", "Northing", unit=length_unit)
    ax.ticklabel_format(useOffset=False, style="plain")
    ax.grid(True, alpha=0.3)
    return [w.well_id for w in wells]


def draw_well_logs(fig: Any, data: BoreholeData, well_ids: Sequence[str], *,
                   when: Optional[_dt.datetime] = None, tolerance_days: Optional[float] = None,
                   length_unit: Optional[str] = None, water_color: str = "#007aff",
                   edge_color: str = "#1d1d1f", max_depth: Optional[float] = None) -> List[Any]:
    """The wells side by side: for each, its lithology column and then one track per curve.

    All tracks share one depth axis, positive down, ticked in ``length_unit``.
    The water level is drawn on the lithology column as :func:`column_water`
    gives it (the reading nearest ``when``, or the range of all readings).
    Returns the axes.
    """
    from matplotlib.patches import Patch, Rectangle

    from PyHydroGeophysX.visualization.axis_units import set_length_axis

    fig.clear()
    wells = [w for w in well_ids if w in data.wells or data.intervals(w) or data.well_logs(w)]
    if not wells:
        return []
    ratios: List[float] = []
    tracks: List[Tuple[str, Optional[GeophysicalLog]]] = []
    for well in wells:
        tracks.append((well, None))
        ratios.append(0.6)
        for log in data.well_logs(well):
            tracks.append((well, log))
            ratios.append(1.0)
    deepest = max([i.bottom for w in wells for i in data.intervals(w)]
                  + [float(np.nanmax(log.depth)) for w in wells for log in data.well_logs(w)]
                  + [r[1] for w in wells if (r := data.water_range(w)) is not None] + [1.0])
    bottom = float(max_depth) if max_depth else deepest * 1.05
    axes = fig.subplots(1, len(tracks), sharey=True, gridspec_kw={"width_ratios": ratios,
                                                                    "wspace": 0.08})
    axes = np.atleast_1d(axes)
    units_used: List[str] = []
    for k, (ax, (well, log)) in enumerate(zip(axes, tracks)):
        if log is None:
            for interval in data.intervals(well):
                colour = data.legend.get(interval.unit, ("#d9d9d9", interval.unit))[0]
                ax.add_patch(Rectangle((0.0, interval.top), 1.0, interval.bottom - interval.top,
                                       facecolor=colour, edgecolor=edge_color, linewidth=0.6))
                if interval.unit not in units_used:
                    units_used.append(interval.unit)
            shown = column_water(data, data.wells.get(well, Well(well, np.nan, np.nan)),
                                 ground=data.wells[well].section_elevation if well in data.wells
                                 else float("nan"), when=when, tolerance_days=tolerance_days)
            if shown is not None and shown["kind"] == "date":
                ax.plot([0.5], [shown["depth"]], marker="v", markersize=10, color=water_color,
                        markeredgecolor=edge_color, linestyle="none", zorder=4)
                ax.axhline(shown["depth"], color=water_color, lw=1.2, zorder=3)
            elif shown is not None:
                ax.axhspan(shown["shallow"], shown["deep"], color=water_color, alpha=0.35,
                           lw=0, zorder=3)
                for depth in (shown["shallow"], shown["deep"]):
                    ax.axhline(depth, color=water_color, lw=1.2, zorder=3)
            ax.set_xlim(0.0, 1.0)
            ax.set_xticks([])
            ax.set_title(well, fontsize=9, fontweight="bold")
            if not data.intervals(well):
                ax.text(0.5, 0.02 * bottom, "no log", ha="center", va="top", fontsize=7,
                        color="#8e8e93")
        else:
            ax.plot(log.values, log.depth, color=edge_color, lw=1.0)
            ax.set_title(log.label, fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis="x", labelsize=7)
        if k:
            ax.tick_params(axis="y", left=False)
    axes[0].set_ylim(bottom, 0.0)
    set_length_axis(axes[0], "y", "Depth below ground", unit=length_unit)
    handles = [Patch(facecolor=data.legend.get(u, ("#d9d9d9", u))[0], edgecolor=edge_color,
                     label=data.legend.get(u, ("#d9d9d9", u))[1]) for u in units_used]
    if any(data.levels(w) for w in wells):
        handles.append(Patch(facecolor=water_color, alpha=0.35 if when is None else 1.0,
                             label="water level" if when is not None else "water-level range"))
    if handles:
        fig.legend(handles=handles, loc="lower center", ncol=min(6, len(handles)), fontsize=8,
                   frameon=False)
    return list(axes)


def draw_water_levels(ax: Any, data: BoreholeData, well_ids: Optional[Sequence[str]] = None, *,
                      length_unit: Optional[str] = None,
                      when: Optional[_dt.datetime] = None) -> List[str]:
    """Depth to water against time, one line per well, depth positive down.

    A head is turned into a depth with the well's ground elevation; readings
    without a date are left out. ``when`` marks a date (a survey's, say).
    Returns the wells drawn.
    """
    from PyHydroGeophysX.visualization.axis_units import set_length_axis

    ids = list(well_ids) if well_ids is not None else sorted({w.well_id for w in data.water_levels})
    drawn = []
    for well_id in ids:
        well = data.wells.get(well_id)
        points = sorted((w.time, w.depth_below_ground(well)) for w in data.levels(well_id)
                        if w.time is not None and np.isfinite(w.depth_below_ground(well)))
        if not points:
            continue
        times, depths = zip(*points)
        ax.plot(times, depths, lw=1.2, label=well_id, marker="o" if len(points) < 30 else None,
                markersize=3)
        drawn.append(well_id)
    if when is not None:
        ax.axvline(when, color="#8e8e93", lw=1.0, ls="--")
    ax.invert_yaxis()
    set_length_axis(ax, "y", "Depth to water below ground", unit=length_unit)
    ax.grid(True, alpha=0.3)
    if drawn:
        ax.legend(fontsize=8, frameon=False)
    return drawn


def draw_well_comparison(fig: Any, data: BoreholeData, well_id: str,
                         surveys: Sequence[Dict[str, Any]], *,
                         when: Optional[_dt.datetime] = None,
                         tolerance_days: Optional[float] = None,
                         length_unit: Optional[str] = None, max_depth: Optional[float] = None,
                         water_color: str = "#007aff", edge_color: str = "#1d1d1f") -> List[Any]:
    """One well beside the surveys around it, on one depth axis.

    The well's tracks come first - its lithology column, then one track per
    geophysical curve - then one track per survey read down the vertical near
    the well (a dict with ``name``, ``units``, ``log_scale``, ``top``,
    ``bottom``, ``values``, optionally ``faded`` - below the depth of
    investigation - ``distance`` in metres from the well, ``where``,
    ``color`` and ``trace``, which draws a seismic trace as a wiggle). The water level :func:`column_water` gives is drawn across
    every track, so a change in a survey can be read against it. Depths are
    metres below ground, ticked in ``length_unit``. Returns the axes.
    """
    from matplotlib.patches import Patch, Rectangle

    from PyHydroGeophysX.visualization.axis_units import (normalize_length_unit, set_length_axis,
                                                          to_display_length)

    fig.clear()
    well = data.wells.get(well_id, Well(well_id, np.nan, np.nan))
    intervals = data.intervals(well_id)
    logs = data.well_logs(well_id)
    deepest = max([i.bottom for i in intervals] + [float(np.nanmax(g.depth)) for g in logs]
                  + [r[1] for r in [data.water_range(well_id)] if r is not None] + [0.0])
    bottom = float(max_depth) if max_depth else max(1.5 * deepest, 5.0)
    ratios = [0.55] + [1.0] * (len(logs) + len(surveys))
    axes = np.atleast_1d(fig.subplots(1, len(ratios), sharey=True,
                                      gridspec_kw={"width_ratios": ratios, "wspace": 0.12}))
    water = column_water(data, well, ground=well.section_elevation, when=when,
                         tolerance_days=tolerance_days)

    def mark_water(ax: Any) -> None:
        if water is None:
            return
        if water["kind"] == "date":
            ax.axhline(water["depth"], color=water_color, lw=1.2, zorder=4)
        else:
            ax.axhspan(water["shallow"], water["deep"], color=water_color, alpha=0.18, lw=0,
                       zorder=0)

    lith = axes[0]
    for interval in intervals:
        colour = data.legend.get(interval.unit, ("#d9d9d9", interval.unit))[0]
        lith.add_patch(Rectangle((0.0, interval.top), 1.0, interval.bottom - interval.top,
                                 facecolor=colour, edgecolor=edge_color, linewidth=0.6))
    if not intervals:
        lith.text(0.5, 0.03 * bottom, "no lithology", ha="center", va="top", fontsize=7,
                  color="#8e8e93")
    mark_water(lith)
    lith.set_xlim(0.0, 1.0)
    lith.set_xticks([])
    lith.set_title(well_id, fontsize=9, fontweight="bold")
    for ax, log in zip(axes[1:], logs):
        ax.plot(log.values, log.depth, color=edge_color, lw=1.0)
        ax.set_title(log.label, fontsize=8)
        mark_water(ax)
    unit = normalize_length_unit(length_unit)
    for ax, survey in zip(axes[1 + len(logs):], surveys):
        top = np.asarray(survey["top"], dtype=float)
        base = np.asarray(survey["bottom"], dtype=float)
        values = np.asarray(survey["values"], dtype=float)
        faded = survey.get("faded")
        faded = np.zeros(values.size, bool) if faded is None else np.asarray(faded, bool)
        good = np.isfinite(values) & (values > 0 if survey.get("log_scale") else True)
        colour = survey.get("color", edge_color)
        if survey.get("trace"):
            # A seismic trace reads as a wiggle, its positive half filled.
            mid = 0.5 * (top + base)
            ax.plot(np.where(good, values, np.nan), mid, color=colour, lw=1.0)
            ax.fill_betweenx(mid, 0.0, np.where(good, values, 0.0),
                             where=good & (values > 0), color=colour, alpha=0.5, lw=0)
            ax.axvline(0.0, color="#aeaeb2", lw=0.8)
            good = np.zeros_like(good)
        for k in np.flatnonzero(good):
            style = dict(color="#aeaeb2", lw=1.0, ls="--") if faded[k] else dict(color=colour, lw=1.6)
            ax.plot([values[k], values[k]], [top[k], base[k]], **style)
            if k + 1 < values.size and good[k + 1] and abs(base[k] - top[k + 1]) < 1e-6:
                ax.plot([values[k], values[k + 1]], [base[k], base[k]], **style)
        if survey.get("log_scale"):
            ax.set_xscale("log")
        mark_water(ax)
        away = survey.get("distance")
        where = (f"{to_display_length(away, unit):.1f} {unit} from the well"
                 if away is not None and np.isfinite(away) else "")
        ax.set_title(f"{survey['name']}\n{where}", fontsize=8)
        ax.set_xlabel(survey.get("units", ""), fontsize=8)
    for k, ax in enumerate(axes):
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", labelsize=7)
        if k:
            ax.tick_params(axis="y", left=False)
    axes[0].set_ylim(bottom, 0.0)
    set_length_axis(axes[0], "y", "Depth below ground", unit=length_unit)
    handles = [Patch(facecolor=data.legend.get(i.unit, ("#d9d9d9", i.unit))[0],
                     edgecolor=edge_color, label=data.legend.get(i.unit, ("#d9d9d9", i.unit))[1])
               for i in {i.unit: i for i in intervals}.values()]
    if water is not None:
        handles.append(Patch(facecolor=water_color, alpha=1.0 if water["kind"] == "date" else 0.3,
                             label=water["text"].replace("WT", "water level")))
    if any(s.get("faded") is not None and np.any(s["faded"]) for s in surveys):
        handles.append(Patch(facecolor="white", edgecolor="#aeaeb2", linestyle="--",
                             label="below the depth of investigation"))
    if handles:
        fig.legend(handles=handles, loc="lower center", ncol=min(6, len(handles)), fontsize=8,
                   frameon=False)
    return list(axes)
