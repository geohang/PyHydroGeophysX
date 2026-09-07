"""Reading a TEMcompany project, whichever of its two file layouts it uses.

TEMImage rewrote its project file. Releases up to 2026 wrote ``project.db``,
whose tables are ``RawStackData``, ``StationStackData``, ``InversionModel``,
``RxTxSpecs``, ``SurveyLines`` and ``UserSettingsJson``. TEMImage 3 writes
``project.tiw`` instead, with ``Data``, ``StationStacks``, ``StationModelTable``,
``ProtocolTable``, ``LineTable`` and ``InversionRunTable``. Both are SQLite, and
they carry the same survey; almost every difference is a renaming, a
normalisation of a JSON blob into columns, or a change of sign convention on one
flag array.

This module is the one place that knows about the difference. It reads either
file and returns the survey in the older shape: a ``RxTxSpecs``-style spec dict
per protocol, and one station-stack row per station carrying both moments. Every
reader in :mod:`PyHydroGeophysX.data_processing.em1d` and
:mod:`PyHydroGeophysX.data_processing.temcompany_reference` goes through it, so
those modules keep one code path rather than two.

Translating towards the older shape rather than the newer one is deliberate. The
older shape is what the gate-selection, forward and inversion code was measured
against, over 882 inverted stations of one survey, and that measurement is
recorded in ``local_plans/temcompany_forward_alignment.md``. Moving the whole
pipeline onto new key names would put that evidence out of reach for no gain,
since the two describe the same instrument.

Four differences are more than a renaming and are worth stating.

``StationStacks`` holds one row per station and moment, keyed by
``TransmitterMoment`` (0 for the low moment, 1 for the high), where
``StationStackData`` held one row per station with ``LM_`` and ``HM_`` column
pairs. The rows are pivoted back into pairs here.

The in-use flag changed sense and meaning. ``InUseFlags`` was 1 for a gate the
inversion used. ``Filtered`` is 0 for a gate the inversion used and otherwise a
bitmask naming why it was dropped, so a reason survives where before only the
verdict did. On the Ashton Prairie survey the two agree on every gate once the
sense is flipped: 2,824 gates with ``Filtered == 0`` against the 2,824 data
points the inversion log reports. The mask value is kept under ``_FilterReason``
for a caller that wants it.

Gate centre times are gone. ``ProtocolTable`` stores open, centre and close, but
``StationModelTable.ModelData`` stores only open and close. The centre is the
geometric mean of the two, which reproduces the stored ``ProtocolTable`` centres
to 7e-8 relative on both moments of this instrument, so it is computed rather
than looked up.

The saved inversion settings moved from a free-form ``UserSettingsJson`` blob to
the actual solver input. ``InversionRunTable.InversionInput`` is the JSON that
was handed to the Lupus inversion, layer grid, starting model, vertical and
lateral constraint weights and all, and ``InversionOutput`` is what came back.
That removes the guesswork in the older path, where a setting reading "Auto"
meant the stored number was inert and a default applied instead. See
:func:`read_inversion_defaults`.

Nothing here executes anything. It parses a database the acquisition software
wrote.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import struct
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: Project file names, in the order a directory is searched.
#:
#: The workspace file comes first because a project migrated from the older
#: release keeps its ``project.db`` beside the new one, and that copy is stale:
#: on the Ashton Prairie project the leftover ``project.db`` carries the
#: transients but zero rows in ``InversionModel``, while ``project.tiw`` holds
#: the inversion. Preferring the older file would silently report a survey as
#: never inverted.
PROJECT_FILE_NAMES: Tuple[str, ...] = ("project.tiw", "project.db")

#: Project file suffixes, same order and same reason.
PROJECT_SUFFIXES: Tuple[str, ...] = (".tiw", ".db")

#: Schema names :func:`schema_of` returns.
WORKSPACE_SCHEMA = "workspace"
LEGACY_SCHEMA = "legacy"

#: ``TransmitterMoment`` as ``StationStacks`` and ``ModelData`` record it.
MOMENT_BY_INDEX: Dict[int, str] = {0: "LM", 1: "HM"}

#: Instrument names by ``ProtocolTable.InstrumentType``.
#:
#: The older file wrote the name; the newer one writes a code. Only the TEM2Go
#: has been seen, and an unknown code is reported as ``"TEMcompany type <n>"``
#: rather than guessed at, so a reader can see that the file said something the
#: mapping does not cover.
INSTRUMENT_BY_CODE: Dict[int, str] = {3: "TEM2Go"}

#: MGRS latitude bands, which is what ``RxTxSpecs.UTMZoneLetter`` holds.
#:
#: The letter is not a hemisphere: one project records zone 13 band S at 39 N in
#: Colorado, another zone 15 band T at 41.7 N in Iowa. Bands run 8 degrees from
#: 80 S, skipping I and O, and X covers the last 12.
_MGRS_BANDS = "CDEFGHJKLMNPQRSTUVWX"


def _mgrs_band(latitude: Any) -> str:
    """The MGRS latitude band a latitude falls in, or an empty string."""
    try:
        value = float(latitude)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(value) or not (-80.0 <= value <= 84.0):
        return ""
    index = min(int((value + 80.0) // 8.0), len(_MGRS_BANDS) - 1)
    return _MGRS_BANDS[index]


def _utm_from_epsg(epsg: Any) -> Tuple[Optional[int], bool]:
    """UTM zone number and northern-hemisphere flag from a WGS84 UTM EPSG code."""
    try:
        code = int(epsg)
    except (TypeError, ValueError):
        return None, True
    if 32601 <= code <= 32660:
        return code - 32600, True
    if 32701 <= code <= 32760:
        return code - 32700, False
    return None, True


def _doubles(blob: Any) -> List[float]:
    """A little-endian double array out of a BLOB column.

    ``ProtocolTable`` stores the loop and coil geometry as packed doubles rather
    than as JSON, so the positions, the loop side lengths and the filter corners
    all arrive this way.
    """
    if not isinstance(blob, (bytes, bytearray, memoryview)):
        return []
    raw = bytes(blob)
    if not raw or len(raw) % 8:
        return []
    return list(struct.unpack("<%dd" % (len(raw) // 8), raw))


def _json_list(value: Any) -> List[float]:
    """A float list out of a JSON text column, empty where there is none."""
    if value in (None, "", "[]"):
        return []
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    if not isinstance(parsed, Sequence) or isinstance(parsed, (str, bytes)):
        return []
    out: List[float] = []
    for item in parsed:
        try:
            out.append(float(item))
        except (TypeError, ValueError):
            out.append(float("nan"))
    return out


def _json_array_length(value: Any) -> int:
    """How many numbers a flat JSON numeric array holds, without decoding it.

    The gate arrays in a project file are flat lists of numbers, so the count is
    one more than the number of commas. Counting rather than decoding matters
    because a line inversion asks for the same station stacks once per station:
    on a 140-station survey the decode-and-re-encode round trip this replaces
    was 5 s of an 89 s run.
    """
    if not isinstance(value, str):
        return len(_json_list(value))
    body = value.strip()
    if len(body) < 2 or body[0] != "[":
        return 0
    inner = body[1:-1].strip()
    return 0 if not inner else inner.count(",") + 1


def _row_get(row: Any, key: str, default: Any = None) -> Any:
    """One column of a row, or *default* where the row has no such column.

    ``sqlite3.Row`` raises ``IndexError`` for a name it does not hold and a dict
    raises ``KeyError``; both mean the same thing here.
    """
    try:
        return row[key]
    except (IndexError, KeyError, TypeError):
        return default


def project_files(path: str | Path) -> List[Path]:
    """Every project file in *path*, the one :func:`project_file` picks first.

    A folder often holds more than one. A project migrated from an earlier
    release keeps its ``project.db`` beside the new ``project.tiw``, and a
    reprocessing of the same survey is written under its own name, so
    ``project2.tiw`` sits beside ``project.tiw`` describing the same day at a
    different stacking. Neither is wrong and only the operator knows which one
    is wanted, so a caller with a user to ask should list them and ask.
    """
    source = Path(path)
    if source.is_file():
        return [source]
    found: List[Path] = []
    seen = set()
    for name in PROJECT_FILE_NAMES:
        candidate = source / name
        if candidate.is_file():
            found.append(candidate)
            seen.add(name.lower())
    for suffix in PROJECT_SUFFIXES:
        try:
            items = sorted(source.glob("*" + suffix))
        except OSError:
            continue
        for item in items:
            if item.is_file() and item.name.lower() not in seen:
                found.append(item)
                seen.add(item.name.lower())
    return found


def project_file(path: str | Path) -> Path:
    """The project file inside *path*, or *path* itself when it is one.

    Where a folder holds several, the standard name wins and the newer layout
    wins within it, which is the right default for a caller that cannot ask.
    Use :func:`project_files` to offer the choice, and pass the chosen file
    here.

    Raises ``ValueError`` when *path* is a directory holding no project file,
    which is what a raw acquisition folder looks like: it carries the ``.sts``
    protocol and the line files but nothing the software has imported yet.
    """
    source = Path(path)
    if source.is_file():
        return source
    found = project_files(source)
    if not found:
        raise ValueError(
            f"{source} holds no project file. A TEMcompany project has a "
            "project.tiw (TEMImage 3) or a project.db (earlier releases) beside "
            "its protocol; a raw acquisition folder does not.")
    standard = {name.lower() for name in PROJECT_FILE_NAMES}
    if len(found) == 1 or found[0].name.lower() in standard:
        return found[0]
    raise ValueError(
        f"{source} holds no project.tiw or project.db and more than one other "
        "database; pass the file directly.")


def project_summary(path: str | Path) -> Dict[str, Any]:
    """Enough about one project file to choose between two of them.

    Station and inversion counts rather than a file size, because two files in
    a folder differ in what they hold rather than in how big they are: on one
    survey ``project.tiw`` carries 140 stations and ``project2.tiw`` a
    reprocessing of the same day at 529. A file that cannot be opened is
    reported with its error rather than raising, since the caller is building a
    list and one bad entry should not empty it.
    """
    target = Path(path)
    summary: Dict[str, Any] = {
        "path": target, "name": target.name, "stations": None,
        "inversion": None, "layout": None, "is_project": False, "error": None,
    }
    try:
        connection = open_project(target)
    except (OSError, ValueError, sqlite3.Error) as error:
        summary["error"] = str(error)
        return summary
    try:
        summary["layout"] = schema_of(connection)
        # A project folder can hold databases that are not projects: one survey
        # keeps a per-day acquisition file beside project.db. The table check is
        # what separates them, so a chooser can leave those out.
        summary["is_project"] = not missing_survey_tables(connection)
        if summary["is_project"]:
            table = ("StationStacks" if summary["layout"] == WORKSPACE_SCHEMA
                     else "StationStackData")
            column = ("StationNumber" if table == "StationStacks"
                      else "StationId")
            rows = connection.execute(
                f"SELECT COUNT(DISTINCT LineNumber || '_' || {column}) "
                f"FROM {table}").fetchone()
            summary["stations"] = int(rows[0]) if rows else None
            summary["inversion"] = last_inversion_name(connection)
    except sqlite3.Error as error:
        summary["error"] = str(error)
    finally:
        connection.close()
    return summary


def describe_project(path: str | Path) -> str:
    """One line naming a project file and what it holds, for a chooser."""
    summary = project_summary(path)
    if summary["error"]:
        return f"{summary['name']}  (unreadable: {summary['error']})"
    if not summary["is_project"]:
        return f"{summary['name']}  (not a TEMcompany project)"
    parts = [summary["name"]]
    if summary["stations"] is not None:
        parts.append(f"{summary['stations']} stations")
    parts.append(summary["inversion"] or "no inversion")
    return "  -  ".join(parts)


def choosable_projects(path: str | Path) -> List[Dict[str, Any]]:
    """Summaries of the project files in *path* that are worth offering.

    Filtered to the ones that carry a survey, so a folder holding an
    acquisition database beside its project does not offer both. The order is
    :func:`project_files` order, so the first entry is what a caller that does
    not ask would have read.
    """
    summaries = [project_summary(item) for item in project_files(path)]
    return [item for item in summaries if item["is_project"]]


def open_project(path: str | Path) -> sqlite3.Connection:
    """Open a project file read-only, so reading cannot alter a survey."""
    database = project_file(path)
    connection = sqlite3.connect(
        database.resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def table_names(con: sqlite3.Connection) -> set:
    """Every table the open project holds."""
    try:
        return {row[0] for row in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
    except sqlite3.Error:
        return set()


def schema_of(con: sqlite3.Connection) -> str:
    """Which of the two layouts an open project uses.

    Keys on ``StationStacks``, which only TEMImage 3 writes, rather than on the
    file suffix, so a workspace saved under another name still reads.
    """
    names = table_names(con)
    if "StationStacks" in names and "ProtocolTable" in names:
        return WORKSPACE_SCHEMA
    return LEGACY_SCHEMA


def missing_survey_tables(con: sqlite3.Connection) -> List[str]:
    """The tables a project needs for a sounding read and does not have.

    A file that is SQLite but not a TEMcompany project reads as the older layout
    and then turns up empty, so a caller checks this first and can say which
    layout it was checked against.
    """
    names = table_names(con)
    required = (("StationStacks", "ProtocolTable")
                if schema_of(con) == WORKSPACE_SCHEMA
                else ("StationStackData", "RxTxSpecs"))
    return sorted(name for name in required if name not in names)


# -- protocol and acquisition spec -------------------------------------------

def _workspace_spec(row: Any, epsg: Any, latitude: Any) -> Dict[str, Any]:
    """One ``ProtocolTable`` row in the older ``RxTxSpecsJson`` key namespace."""
    spec: Dict[str, Any] = {}
    for moment, prefix in (("LM", "LowMoment"), ("HM", "HighMoment")):
        spec[f"{moment}_GateOpenTime"] = _json_list(
            _row_get(row, prefix + "GateOpenTimes"))
        spec[f"{moment}_GateCentreTime"] = _json_list(
            _row_get(row, prefix + "GateCenterTimes"))
        spec[f"{moment}_GateCloseTime"] = _json_list(
            _row_get(row, prefix + "GateCloseTimes"))
        spec[f"{moment}WaveformTime"] = _json_list(
            _row_get(row, prefix + "WaveformTime"))
        spec[f"{moment}WaveformAmplitude"] = _json_list(
            _row_get(row, prefix + "WaveformAmplitude"))
        spec[f"{moment}WaveformPeriod"] = _row_get(row, prefix + "PeriodTime")
        # The override column wins where the operator set one, which is what the
        # application shows and what the inversion was run with.
        shift = _row_get(row, f"{moment}GateTimeShiftOverride")
        spec[f"{moment}_GateTimeShift"] = (
            _row_get(row, f"{moment}GateTimeShift") if shift is None else shift)
        factor = _row_get(row, f"{moment}CalibrationFactorOverride")
        spec[f"{moment}_DataFactor"] = (
            _row_get(row, f"{moment}CalibrationFactor")
            if factor is None else factor)

    currents = _json_list(_row_get(row, "TargetCurrents"))
    if len(currents) >= 2:
        spec["LM_Tx_TargetCurrent"] = currents[0]
        spec["HM_Tx_TargetCurrent"] = currents[1]

    spec["GateShape"] = _row_get(row, "GateShapeChannelA")
    spec["GateShapePar1"] = _row_get(row, "GateShapeChannelAParameter1")
    spec["LPFilter_1order"] = _doubles(_row_get(row, "FilterCutOffs"))
    # Recorded empty in every project seen under either layout, and kept so a
    # caller reading the spec finds the same keys it found before.
    spec["LowPassFilterRxCoil"] = []
    spec["LowPassFilterRxInst"] = []
    spec["TxLoopArea"] = _row_get(row, "TxCoilArea")
    spec["TxLoopXYlength"] = _doubles(_row_get(row, "TxXYLength"))
    spec["TxLoopXYZPos"] = _doubles(_row_get(row, "TxCoilPosition"))
    spec["RxCoilXYZPos"] = _doubles(_row_get(row, "RxCoilPosition"))
    spec["NTurnsTxLoop"] = _row_get(row, "TxLoopTurns")
    code = _row_get(row, "InstrumentType")
    spec["InstrumentType"] = INSTRUMENT_BY_CODE.get(
        code, f"TEMcompany type {code}" if code is not None else "")

    zone, northern = _utm_from_epsg(epsg)
    if zone is not None:
        spec["UTMZone"] = zone
        band = _mgrs_band(latitude)
        spec["UTMZoneLetter"] = band or ("N" if northern else "S")
    spec["EPSG"] = epsg
    # ``RxCoilAreaChA`` has no column in the newer layout. It is descriptive
    # only, reported in forward metadata and never read by the forward, so it is
    # left absent rather than filled with an instrument constant that the file
    # does not state.
    return spec


def read_specs(con: sqlite3.Connection) -> Dict[Any, Dict[str, Any]]:
    """Every acquisition spec the project holds, keyed as the stations key them.

    The key is ``RxTxSpecsId`` under the older layout and ``ProtocolId`` under
    the newer one; :func:`read_station_stacks` writes the matching value into
    each row's ``RxTxSpecsId``, so a caller joins the two the same way in both
    cases.
    """
    if schema_of(con) == LEGACY_SCHEMA:
        return {
            row["RxTxSpecsId"]: json.loads(row["RxTxSpecsJson"])
            for row in con.execute("SELECT * FROM RxTxSpecs")
            if row["RxTxSpecsJson"]
        }
    try:
        project = con.execute("SELECT EPSG FROM Project LIMIT 1").fetchone()
        epsg = project[0] if project is not None else None
    except sqlite3.Error:
        epsg = None
    try:
        found = con.execute(
            "SELECT Latitude FROM StationStacks WHERE Latitude IS NOT NULL "
            "LIMIT 1").fetchone()
        latitude = found[0] if found is not None else None
    except sqlite3.Error:
        latitude = None
    return {
        row["ProtocolId"]: _workspace_spec(row, epsg, latitude)
        for row in con.execute("SELECT * FROM ProtocolTable")
    }


def protocol_uniform_error(con: sqlite3.Connection) -> Optional[float]:
    """``UniStd`` as the project file records it, where it records a usable one.

    The newer layout has a ``ProtocolTable.UniStd`` column, and on the one
    migrated project seen it holds 0.0 while the ``.sts`` beside it states 0.03
    and the smallest stored error is 0.030000098. A zero therefore means the
    importer did not carry the value across, not that the survey ran without a
    uniform term, so it is reported as absent and the ``.sts`` stays the
    authority. See ``_temcompany_uniform_error`` in ``em1d``.
    """
    if schema_of(con) != WORKSPACE_SCHEMA:
        return None
    try:
        row = con.execute("SELECT UniStd FROM ProtocolTable LIMIT 1").fetchone()
    except sqlite3.Error:
        return None
    if row is None or row[0] is None:
        return None
    try:
        value = float(row[0])
    except (TypeError, ValueError):
        return None
    return value if 0.0 < value < 1.0 else None


# -- station stacks ----------------------------------------------------------

def _use_flags_from_filtered(filtered: Sequence[float],
                             n_gates: int) -> List[int]:
    """The older 1-means-used flags out of the newer 0-means-used bitmask."""
    if not filtered:
        return [1] * n_gates
    flags = [1 if int(value) == 0 else 0 for value in filtered[:n_gates]]
    flags.extend([1] * max(0, n_gates - len(flags)))
    return flags


def read_station_stacks(con: sqlite3.Connection) -> List[Any]:
    """Every station stack, one entry per station, in line then station order.

    Under the older layout these are the ``StationStackData`` rows themselves.
    Under the newer one the two ``StationStacks`` rows of a station, one per
    moment, are pivoted into a single mapping carrying ``LM_`` and ``HM_``
    columns, so a caller reads both layouts with one set of column names.

    The result holds ``sqlite3.Row`` or ``dict`` depending on the layout. Both
    support ``row[name]`` and ``row.keys()``, and the readers already catch the
    ``IndexError`` a ``Row`` raises for an absent name beside the ``KeyError`` a
    ``dict`` raises, so the difference does not reach a call site.
    """
    if schema_of(con) == LEGACY_SCHEMA:
        return list(con.execute(
            "SELECT * FROM StationStackData "
            "ORDER BY LineNumber, AveragedDataId"))

    try:
        found = con.execute(
            "SELECT ProtocolId FROM ProtocolTable LIMIT 1").fetchone()
        protocol_id = found[0] if found is not None else None
    except sqlite3.Error:
        protocol_id = None

    grouped: Dict[Tuple[Any, Any], Dict[str, Any]] = {}
    order: List[Tuple[Any, Any]] = []
    for row in con.execute(
        "SELECT * FROM StationStacks "
        "ORDER BY LineNumber, StationNumber, StationId"
    ):
        moment = MOMENT_BY_INDEX.get(row["TransmitterMoment"])
        if moment is None:
            continue
        line = row["LineNumber"]
        number = row["StationNumber"]
        key = (line, number)
        entry = grouped.get(key)
        if entry is None:
            entry = {
                # The older layout names a station "<line>_<number:05d>"; the
                # survey summary and the tests both surface that string, so it
                # is reproduced rather than replaced.
                "StationId": (f"{int(line)}_{int(number):05d}"
                              if line is not None and number is not None
                              else str(number)),
                "StationNumber": number,
                "LineNumber": line,
                "AveragedDataId": row["StationId"],
                "UtmX": row["UtmX"],
                "UtmY": row["UtmY"],
                "UtmZ": row["UtmZ"],
                "Latitude": row["Latitude"],
                "Longitude": row["Longitude"],
                "Elevation": row["Altitude"],
                "DigitalElevation": row["DigitalElevation"],
                "Timestamp": row["Time"],
                "ProfileDistance": row["LineProfileCoordinate"],
                "TotalSurveyDistance": row["GlobalProfileCoordinate"],
                "RxTxDistance": row["RxTxDistance"],
                "RxTxSpecsId": protocol_id,
                "LineId": row["LineId"],
                "_StationIds": {},
            }
            grouped[key] = entry
            order.append(key)
        # The measured arrays travel as the JSON text the file holds, unchanged.
        # Decoding them here and re-encoding them for the reader to decode again
        # is what the older layout never asked for, and it dominated the read.
        filtered = _json_list(row["Filtered"])
        entry[f"{moment}_VoltageValues"] = row["dBdt"]
        entry[f"{moment}_VoltageValues_STD"] = row["Std"]
        entry[f"{moment}_InUseFlags"] = json.dumps(
            _use_flags_from_filtered(filtered, _json_array_length(row["dBdt"])))
        # The reason a gate was dropped, which the older flag array could not
        # carry. Kept so a caller that wants it does not have to reopen the file.
        entry[f"{moment}_FilterReason"] = row["Filtered"]
        entry[f"{moment}_TxCurrent"] = row["TxCurrent"]
        entry["_StationIds"][moment] = row["StationId"]

    stacks: List[Any] = []
    for key in order:
        entry = grouped[key]
        for moment in MOMENT_BY_INDEX.values():
            entry.setdefault(f"{moment}_VoltageValues", "[]")
            entry.setdefault(f"{moment}_VoltageValues_STD", "[]")
            entry.setdefault(f"{moment}_InUseFlags", "[]")
        ids = [value for value in entry["_StationIds"].values()
               if value is not None]
        if ids:
            entry["AveragedDataId"] = min(ids)
        stacks.append(entry)
    return stacks


# -- reading a survey once per file ------------------------------------------

#: How many project files :func:`read_survey` keeps parsed at a time.
#:
#: One is enough for a line inversion and two for a comparison between surveys.
#: The entries hold the station stacks as the file's own JSON text, so a
#: 929-station project costs a few megabytes.
SURVEY_CACHE_LIMIT = 4

_SURVEY_CACHE: "OrderedDict[Tuple[str, int, int], Tuple[Dict[Any, Dict[str, Any]], List[Any]]]" = OrderedDict()
_SURVEY_CACHE_LOCK = threading.Lock()


def file_key(path: str | Path) -> Tuple[str, int, int]:
    """Identity of a project file for caching: resolved name, mtime and size.

    Content hashing would be stronger and would cost a read of the whole file,
    which is the thing being avoided. A project written twice within the
    filesystem's timestamp resolution and to exactly the same length is the case
    this misses, and an acquisition program does not produce it.
    """
    resolved = Path(path).resolve()
    stat = resolved.stat()
    name = str(resolved)
    if os.name == "nt":
        name = name.lower()
    return name, stat.st_mtime_ns, stat.st_size


def read_survey(path: str | Path
                ) -> Tuple[Dict[Any, Dict[str, Any]], List[Any]]:
    """The specs and station stacks of a project, parsed once per file.

    A line inversion reads one station at a time, and every one of those reads
    used to reopen the project and parse every station in it. On a 140-station
    survey that was 7 s of an 89 s run, and the cost grows with the square of
    the survey, so a 929-station project pays it in minutes.

    The result is cached on :func:`file_key`, so a project rewritten between
    calls is reparsed. Callers must treat what comes back as read-only: they all
    share it.
    """
    key = file_key(project_file(path))
    cached = _SURVEY_CACHE.get(key)
    if cached is not None:
        return cached
    connection = open_project(path)
    try:
        missing = missing_survey_tables(connection)
        if missing:
            raise ValueError(
                f"{Path(key[0]).name} is not a supported TEMcompany project "
                f"(missing {', '.join(missing)}).")
        value = (read_specs(connection), read_station_stacks(connection))
    finally:
        connection.close()
    with _SURVEY_CACHE_LOCK:
        _SURVEY_CACHE[key] = value
        _SURVEY_CACHE.move_to_end(key)
        while len(_SURVEY_CACHE) > SURVEY_CACHE_LIMIT:
            _SURVEY_CACHE.popitem(last=False)
    return value


def clear_survey_cache() -> None:
    """Forget every parsed project. For tests, and for a caller that rewrote one."""
    with _SURVEY_CACHE_LOCK:
        _SURVEY_CACHE.clear()


# -- inversion results -------------------------------------------------------

def inversion_names(con: sqlite3.Connection) -> List[str]:
    """Every named inversion the project holds, the largest run first."""
    if schema_of(con) == WORKSPACE_SCHEMA:
        try:
            rows = con.execute(
                "SELECT r.InversionRunName, COUNT(m.ModelId) AS n "
                "FROM InversionRunTable r LEFT JOIN StationModelTable m "
                "ON m.InversionId = r.InversionRunId "
                "GROUP BY r.InversionRunId ORDER BY n DESC").fetchall()
        except sqlite3.Error:
            return []
        return [str(row[0]) for row in rows]
    try:
        rows = con.execute(
            "SELECT InversionName, COUNT(*) AS n FROM InversionModel "
            "GROUP BY InversionName ORDER BY n DESC").fetchall()
    except sqlite3.Error:
        return []
    return [str(row[0]) for row in rows]


def last_inversion_name(con: sqlite3.Connection) -> Optional[str]:
    """The inversion the project's own record points at, if it names one.

    The newer layout timestamps every run, so the latest finished run is the one
    the application would show. The older layout has no timestamp and instead
    records ``LastInversionName`` inside its settings blob.
    """
    if schema_of(con) == WORKSPACE_SCHEMA:
        try:
            row = con.execute(
                "SELECT InversionRunName FROM InversionRunTable "
                "ORDER BY InversionRunFinishedAt DESC, InversionRunId DESC "
                "LIMIT 1").fetchone()
        except sqlite3.Error:
            return None
        return str(row[0]) if row is not None else None
    try:
        row = con.execute(
            "SELECT * FROM UserSettingsJson ORDER BY 1 DESC LIMIT 1").fetchone()
    except sqlite3.Error:
        return None
    if row is None:
        return None
    raw = next((value for value in row
                if isinstance(value, str) and value.lstrip().startswith("{")),
               None)
    if not raw:
        return None
    try:
        settings = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    name = dict(settings.get("InverseSettings", {})).get("LastInversionName")
    return str(name) if name else None


def has_inversion_models(con: sqlite3.Connection) -> bool:
    """Whether the project holds inversion results as well as soundings."""
    table = ("StationModelTable" if schema_of(con) == WORKSPACE_SCHEMA
             else "InversionModel")
    if table not in table_names(con):
        return False
    try:
        return con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] > 0
    except sqlite3.Error:
        return False


def _model_data_to_datasets(raw: Any) -> str:
    """``StationModelTable.ModelData`` in the older ``Datasets`` key namespace.

    The newer blob renames three arrays, numbers the moment instead of naming
    it, and drops the gate centre. The centre is the geometric mean of open and
    close, which matches the centres ``ProtocolTable`` stores to 7e-8 relative
    on both moments.
    """
    if raw in (None, "", "[]"):
        return "[]"
    try:
        entries = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError, json.JSONDecodeError):
        return "[]"
    out: List[Dict[str, Any]] = []
    for entry in entries or []:
        if not isinstance(entry, dict):
            continue
        moment = MOMENT_BY_INDEX.get(entry.get("MomentType"))
        if moment is None:
            continue
        opens = _json_list(entry.get("Time_Open"))
        closes = _json_list(entry.get("Time_Close"))
        centres = [math.sqrt(a * b) if (a > 0.0 and b > 0.0) else float("nan")
                   for a, b in zip(opens, closes)]
        out.append({
            "MomentType": moment,
            "Time_Centre": centres,
            "Time_Open": opens,
            "Time_Close": closes,
            "InputData": _json_list(entry.get("dBdt_Inp")),
            "InputSTD": _json_list(entry.get("Std_Inp")),
            "ForwardData": _json_list(entry.get("dBdt_Fwr")),
        })
    return json.dumps(out)


def read_inversion_models(con: sqlite3.Connection,
                          inversion_name: Optional[str] = None) -> List[Any]:
    """Every station of one stored inversion, in the older row shape.

    ``inversion_name`` selects among several saved runs; the default is the run
    :func:`last_inversion_name` points at, and failing that the one with the
    most stations.
    """
    if inversion_name is None:
        inversion_name = last_inversion_name(con)
    if inversion_name is None:
        names = inversion_names(con)
        inversion_name = names[0] if names else None

    if schema_of(con) == LEGACY_SCHEMA:
        query = "SELECT * FROM InversionModel"
        parameters: Tuple[Any, ...] = ()
        if inversion_name is not None:
            query += " WHERE InversionName = ?"
            parameters = (str(inversion_name),)
        query += " ORDER BY LineNumber, AverageDataID"
        try:
            return list(con.execute(query, parameters))
        except sqlite3.Error:
            return []

    stacks = {(row["LineNumber"], row["StationNumber"]): row
              for row in read_station_stacks(con)}
    query = ("SELECT m.*, r.InversionRunName FROM StationModelTable m "
             "JOIN InversionRunTable r ON r.InversionRunId = m.InversionId")
    parameters = ()
    if inversion_name is not None:
        query += " WHERE r.InversionRunName = ?"
        parameters = (str(inversion_name),)
    query += " ORDER BY m.LineNumber, m.StationNumber, m.ModelId"
    try:
        rows = list(con.execute(query, parameters))
    except sqlite3.Error:
        return []

    models: List[Dict[str, Any]] = []
    for row in rows:
        stack = stacks.get((row["LineNumber"], row["StationNumber"]))
        used = _json_list(row["UsedStations"])
        if used:
            average_id: Any = int(min(used))
        elif stack is not None:
            average_id = stack["AveragedDataId"]
        else:
            average_id = row["ModelId"]
        elevation = row["UtmZ"]
        if row["DigitalElevation"] is not None:
            elevation = row["DigitalElevation"]
        models.append({
            "InversionID": row["ModelId"],
            "InversionName": str(row["InversionRunName"]),
            "LineNumber": row["LineNumber"],
            "AverageDataID": average_id,
            "UTMx": row["UtmX"],
            "UTMy": row["UtmY"],
            "UTMz": row["UtmZ"],
            "Elevation": elevation,
            "Latitude": stack["Latitude"] if stack is not None else None,
            "Longitude": stack["Longitude"] if stack is not None else None,
            "DOI": row["DOI"],
            "DataFit": row["DataFit"],
            "Resistivity": row["Resistivities"],
            "Thickness": row["Thicknesses"],
            "Datasets": _model_data_to_datasets(row["ModelData"]),
            "ProfileLocalDistance": row["LineProfileCoordinate"],
            "CreatedAt": row["Time"],
            "StationNumber": row["StationNumber"],
            "Depths": row["Depths"],
        })
    return models


def read_stored_thicknesses(con: sqlite3.Connection,
                            inversion_name: Optional[str]) -> Optional[str]:
    """The layer grid one stored inversion ran on, as its JSON text."""
    if schema_of(con) == LEGACY_SCHEMA:
        try:
            row = None
            if inversion_name:
                row = con.execute(
                    "SELECT Thickness FROM InversionModel "
                    "WHERE InversionName = ? LIMIT 1",
                    (str(inversion_name),)).fetchone()
            if row is None:
                row = con.execute(
                    "SELECT Thickness FROM InversionModel LIMIT 1").fetchone()
        except sqlite3.Error:
            return None
        return row[0] if row is not None else None
    try:
        row = None
        if inversion_name:
            row = con.execute(
                "SELECT m.Thicknesses FROM StationModelTable m "
                "JOIN InversionRunTable r ON r.InversionRunId = m.InversionId "
                "WHERE r.InversionRunName = ? LIMIT 1",
                (str(inversion_name),)).fetchone()
        if row is None:
            row = con.execute(
                "SELECT Thicknesses FROM StationModelTable LIMIT 1").fetchone()
    except sqlite3.Error:
        return None
    return row[0] if row is not None else None


# -- the solver's own record -------------------------------------------------

def read_run_record(con: sqlite3.Connection,
                    inversion_name: Optional[str] = None
                    ) -> Optional[Dict[str, Any]]:
    """The input, output and log of one stored inversion run.

    Only TEMImage 3 keeps this. ``InversionRunTable`` stores the JSON handed to
    the Lupus inversion, the JSON it returned, and its console log, which
    together state the layer grid, the starting model, the per-layer vertical
    and lateral constraint weights, the waveform and gate description of every
    dataset, the norms at every iteration and the wall time each stage took.

    That is a stronger reference than the older ``Datasets`` blob, which held
    the data and the forward response but nothing about the constraints. Returns
    ``None`` for a project written by an earlier release.
    """
    if schema_of(con) != WORKSPACE_SCHEMA:
        return None
    query = ("SELECT InversionRunName, InversionRunCreateAt, "
             "InversionRunFinishedAt, InversionInput, InversionOutput, "
             "InversionLog FROM InversionRunTable")
    parameters: Tuple[Any, ...] = ()
    if inversion_name is not None:
        query += " WHERE InversionRunName = ?"
        parameters = (str(inversion_name),)
    query += " ORDER BY InversionRunFinishedAt DESC, InversionRunId DESC LIMIT 1"
    try:
        row = con.execute(query, parameters).fetchone()
    except sqlite3.Error:
        return None
    if row is None:
        return None

    def _decode(blob: Any) -> str:
        if blob is None:
            return ""
        if isinstance(blob, (bytes, bytearray, memoryview)):
            return bytes(blob).decode("utf-8-sig", errors="replace")
        return str(blob)

    def _parsed(text: str) -> Optional[Dict[str, Any]]:
        if not text.strip():
            return None
        try:
            parsed = json.loads(text)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
        return parsed if isinstance(parsed, dict) else None

    return {
        "name": str(row["InversionRunName"]),
        "started": row["InversionRunCreateAt"],
        "finished": row["InversionRunFinishedAt"],
        "input": _parsed(_decode(row["InversionInput"])),
        "output": _parsed(_decode(row["InversionOutput"])),
        "log": _decode(row["InversionLog"]),
    }


def read_inversion_defaults(con: sqlite3.Connection) -> Dict[str, Any]:
    """The settings a stored run actually used, read from the solver input.

    The older layout stored a settings blob whose numbers could be inert: a
    setting reading "Auto" meant a default applied instead, and one project
    stored a reference distance of 10 m for a run made at 50 m. The newer layout
    stores the solver input itself, so nothing has to be inferred. The layer
    grid, the starting resistivity and the two constraint weights are read off
    the first model position, and the rest off the ``Lupus_Settings`` block that
    travelled with it.

    Returns an empty dict for a project written by an earlier release, or for
    one holding no inversion, which leaves the caller's existing fallback in
    place.
    """
    record = read_run_record(con)
    if record is None or not isinstance(record.get("input"), dict):
        return {}
    payload = record["input"]
    settings = dict(payload.get("Lupus_Settings", {}))
    positions = payload.get("Model_position") or []
    if not positions or not isinstance(positions[0], dict):
        return {}
    first = positions[0]
    start = dict(first.get("Model_start", {}))
    thickness = _json_list(start.get("Thk"))
    resistivity = _json_list(start.get("Res"))
    try:
        n_layers = int(payload.get("N_Layer", len(resistivity)))
    except (TypeError, ValueError):
        n_layers = len(resistivity)
    if n_layers < 2 or len(thickness) != n_layers - 1:
        return {}
    vertical = _json_list(first.get("VCon_Res"))
    lateral = _json_list(first.get("LCon_Res"))

    def _number(key: str, fallback: float) -> float:
        try:
            return float(settings[key])
        except (KeyError, TypeError, ValueError):
            return float(fallback)

    auto_scale = bool(settings.get("LC_AutoScale", False))
    return {
        "n_layers": n_layers,
        "min_thickness": thickness[0],
        "max_thickness": thickness[-1],
        "layer_thicknesses": list(thickness),
        "last_depth": float(sum(thickness)),
        "starting_resistivity": resistivity[0] if resistivity else 40.0,
        # The newer release picks the starting model from the data on every run
        # seen, which is what "Auto" meant in the older settings blob. Reported
        # under the same key so a caller can switch on it either way.
        "start_model_mode": "Auto",
        "smoothness": vertical[0] if vertical else 2.0,
        "lateral_smoothness": lateral[0] if lateral else 1.3,
        "tem_moment": "LM+HM",
        "constraint": "SCI" if settings.get("SCI_Contraints") else "LCI",
        "norm": "L%d" % int(_number("NormType_VerticalCon", 2)),
        "reference_distance": _number("LC_RefDistance", 50.0),
        "reference_distance_stored": None,
        "sci_max_distance": _number("SCI_MaxDistance", 300.0),
        "sci_max_distance_stored": None,
        "lateral_weight_scale": 1.0,
        "lateral_distance_power": (_number("LC_AutoScalePower", 0.75)
                                   if auto_scale else 0.0),
        "lateral_distance_power_stored": None,
        "max_iterations": int(_number("IteMax", 30)),
        "doi_max_depth": _number("DOIdepthMax", 300.0),
        "log_data": bool(settings.get("LogData", False)),
    }


__all__ = [
    "INSTRUMENT_BY_CODE",
    "LEGACY_SCHEMA",
    "MOMENT_BY_INDEX",
    "PROJECT_FILE_NAMES",
    "PROJECT_SUFFIXES",
    "SURVEY_CACHE_LIMIT",
    "WORKSPACE_SCHEMA",
    "clear_survey_cache",
    "describe_project",
    "file_key",
    "has_inversion_models",
    "inversion_names",
    "last_inversion_name",
    "missing_survey_tables",
    "open_project",
    "project_file",
    "project_files",
    "project_summary",
    "protocol_uniform_error",
    "read_inversion_defaults",
    "read_inversion_models",
    "read_run_record",
    "read_specs",
    "read_station_stacks",
    "read_stored_thicknesses",
    "read_survey",
    "schema_of",
    "table_names",
]
