"""Read inversion results stored inside a TEMcompany project file.

``project.db`` keeps more than the recorded transients. Where a survey has
already been inverted, ``InversionModel`` holds the recovered model, the depth
of investigation, the data fit, and a ``Datasets`` blob carrying, per moment,
the gate windows used, the data the inversion was given (``InputData``), the
uncertainty assigned to it (``InputSTD``) and the response computed for the
recovered model (``ForwardData``).

This module reads those records so that results produced earlier can be loaded
alongside the soundings they came from: plotted on a section, compared with a
new inversion, or used to check that this package's own reader returns the same
gates, values and errors the file holds. Measured over the 882 inverted
stations of one survey, ``InputData`` is identical to ``LM/HM_VoltageValues``
and ``InputSTD`` to ``LM/HM_VoltageValues_STD`` to the last bit, which is the
property :mod:`tests.test_temcompany_alignment` asserts.

Nothing here executes anything. It parses a database the acquisition software
wrote.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

from PyHydroGeophysX.data_processing.em1d import (
    _temcompany_json_array,
    _temcompany_protocol,
)

#: Moments a ``Datasets`` blob can name.
REFERENCE_MOMENTS: Tuple[str, ...] = ("LM", "HM")


def _project_database(path: str | Path) -> Path:
    """The ``project.db`` inside *path*, or *path* itself when it is one."""
    source = Path(path)
    if source.is_file():
        return source
    databases = sorted(source.glob("*.db"))
    named = [item for item in databases if item.name.lower() == "project.db"]
    if named:
        return named[0]
    if len(databases) == 1:
        return databases[0]
    if not databases:
        # A raw acquisition folder rather than an imported project. The two look
        # alike from outside: both carry the .sts protocol and the line files.
        raise ValueError(
            f"{source} holds no database. A TEMcompany project has a project.db "
            "beside its protocol; a raw acquisition folder does not, and so "
            "holds no inversion results.")
    raise ValueError(
        f"{source} holds no project.db and more than one other database; "
        "pass the file directly.")


def _read_only(database: Path) -> sqlite3.Connection:
    """Open *database* read-only, so reading cannot alter a survey."""
    connection = sqlite3.connect(
        database.resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def has_reference_models(path: str | Path) -> bool:
    """Whether *path* holds inversion results as well as soundings."""
    try:
        database = _project_database(path)
    except (OSError, ValueError):
        return False
    if not database.is_file():
        return False
    try:
        connection = _read_only(database)
    except sqlite3.Error:
        return False
    try:
        names = {row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        if "InversionModel" not in names:
            return False
        return connection.execute(
            "SELECT COUNT(*) FROM InversionModel").fetchone()[0] > 0
    except sqlite3.Error:
        return False
    finally:
        connection.close()


def reference_inversion_names(path: str | Path) -> List[str]:
    """Every named inversion the project holds, the current one first.

    "Current" is the project's own answer where it has one: ``InverseSettings``
    records ``LastInversionName``, which is the run the application would show.
    Only where that is missing does the order fall back to a guess, and the
    guess is row count rather than timestamp because a later run is often a
    re-inversion of a few stations while the one worth comparing against covers
    the survey.
    """
    database = _project_database(path)
    connection = _read_only(database)
    try:
        rows = connection.execute(
            "SELECT InversionName, COUNT(*) AS n FROM InversionModel "
            "GROUP BY InversionName ORDER BY n DESC").fetchall()
    except sqlite3.Error:
        return []
    finally:
        connection.close()
    names = [str(row[0]) for row in rows]
    current = _last_inversion_name(database)
    if current in names:
        names.remove(current)
        names.insert(0, current)
    return names


def _last_inversion_name(database: Path) -> Optional[str]:
    """The inversion the project's settings point at, if it names one."""
    try:
        connection = _read_only(database)
    except sqlite3.Error:
        return None
    try:
        row = connection.execute(
            "SELECT * FROM UserSettingsJson ORDER BY 1 DESC LIMIT 1").fetchone()
    except sqlite3.Error:
        return None
    finally:
        connection.close()
    if row is None:
        return None
    raw = next((value for value in row
                if isinstance(value, str) and value.lstrip().startswith("{")), None)
    if not raw:
        return None
    try:
        settings = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    name = dict(settings.get("InverseSettings", {})).get("LastInversionName")
    return str(name) if name else None


def _station_geometry(row: Optional[sqlite3.Row]) -> Dict[str, float]:
    """Per-station transmitter and receiver geometry.

    The three distance columns disagree by up to a metre and they fail
    differently: ``RxTxDistanceBField`` carries zeros where the field estimate
    did not converge, and ``RxTxDistanceGPSBased`` inherits GPS scatter. The
    edited ``RxTxDistance`` is the operative one, so it is reported as
    ``rx_tx_distance`` and the other two travel beside it for anyone who wants
    to compare them.
    """
    result: Dict[str, float] = {}
    if row is None:
        return result
    for key, name in (
        ("rx_tx_distance", "RxTxDistance"),
        ("rx_tx_distance_bfield", "RxTxDistanceBField"),
        ("rx_tx_distance_gps", "RxTxDistanceGPSBased"),
        ("rx_coil_height", "RxCoilHeight"),
        ("tx_coil_height", "TxCoilHeight"),
        ("elevation", "Elevation"),
    ):
        try:
            value = row[name]
        except (IndexError, KeyError):
            continue
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            result[key] = number
    return result


def _parse_datasets(raw: Any) -> Dict[str, Dict[str, np.ndarray]]:
    """The per-moment arrays inside one ``Datasets`` blob."""
    if raw in (None, "", "[]"):
        return {}
    try:
        entries = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    moments: Dict[str, Dict[str, np.ndarray]] = {}
    for entry in entries or []:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("MomentType", "")).upper()
        if name not in REFERENCE_MOMENTS:
            continue
        block = {
            "times": np.asarray(entry.get("Time_Centre", []), dtype=float).ravel(),
            "gate_open": np.asarray(entry.get("Time_Open", []), dtype=float).ravel(),
            "gate_close": np.asarray(entry.get("Time_Close", []), dtype=float).ravel(),
            "observed": np.asarray(entry.get("InputData", []), dtype=float).ravel(),
            "relative_std": np.asarray(entry.get("InputSTD", []), dtype=float).ravel(),
            "forward": np.asarray(entry.get("ForwardData", []), dtype=float).ravel(),
        }
        sizes = {value.size for value in block.values()}
        if len(sizes) != 1 or not block["times"].size:
            # A blob whose columns disagree describes no gate set. Skipping it
            # is better than returning arrays that cannot be zipped.
            continue
        moments[name] = block
    return moments


def _spec_gate_indices(centres: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Which of the instrument's gates a reference block used.

    Matching on the centre time rather than on position, because the stored
    selection is not contiguous: over one survey only 36 percent of the HM
    selections were a single run of gates.
    """
    centres = np.asarray(centres, dtype=float).ravel()
    times = np.asarray(times, dtype=float).ravel()
    if not centres.size or not times.size:
        return np.array([], dtype=int)
    picked = np.asarray(
        [int(np.argmin(np.abs(centres - value))) for value in times], dtype=int)
    if not np.allclose(centres[picked], times, rtol=1e-6, atol=0.0):
        return np.array([], dtype=int)
    return picked


def iter_reference_stations(
    path: str | Path, inversion_name: Optional[str] = None,
) -> Iterator[Dict[str, Any]]:
    """Yield one dict per station of a stored inversion, in file order.

    ``inversion_name`` selects among several saved runs; the default is
    whichever name carries the most rows, which is the full-survey inversion in
    every project seen so far.
    """
    database = _project_database(path)
    connection = _read_only(database)
    try:
        specs = {
            item["RxTxSpecsId"]: json.loads(item["RxTxSpecsJson"])
            for item in connection.execute("SELECT * FROM RxTxSpecs")
            if item["RxTxSpecsJson"]
        }
        stations = {
            item["AveragedDataId"]: item
            for item in connection.execute("SELECT * FROM StationStackData")
        }
        if inversion_name is None:
            # The project's own pointer first; see reference_inversion_names.
            inversion_name = _last_inversion_name(database)
        if inversion_name is None:
            available = connection.execute(
                "SELECT InversionName, COUNT(*) AS n FROM InversionModel "
                "GROUP BY InversionName ORDER BY n DESC LIMIT 1").fetchone()
            inversion_name = str(available[0]) if available else None
        query = "SELECT * FROM InversionModel"
        parameters: Tuple[Any, ...] = ()
        if inversion_name is not None:
            query += " WHERE InversionName = ?"
            parameters = (str(inversion_name),)
        query += " ORDER BY LineNumber, AverageDataID"
        for row in connection.execute(query, parameters):
            moments = _parse_datasets(row["Datasets"])
            if not moments:
                continue
            station = stations.get(row["AverageDataID"])
            spec: Dict[str, Any] = {}
            if station is not None:
                spec = specs.get(
                    station["RxTxSpecsId"], next(iter(specs.values()), {}))
            elif specs:
                spec = next(iter(specs.values()))
            for name, block in moments.items():
                centres = np.asarray(
                    spec.get(f"{name}_GateCentreTime", []), dtype=float)
                block["spec_gate_indices"] = _spec_gate_indices(
                    centres, block["times"])
                if station is not None:
                    block["stored_response"] = _temcompany_json_array(
                        station[f"{name}_VoltageValues"])
                    block["stored_std"] = _temcompany_json_array(
                        station[f"{name}_VoltageValues_STD"])
                    block["stored_flags"] = _temcompany_json_array(
                        station[f"{name}_InUseFlags"])
            yield {
                "inversion_name": str(row["InversionName"]),
                "average_data_id": int(row["AverageDataID"]),
                "line_number": int(row["LineNumber"]),
                "x": float(row["UTMx"]),
                "y": float(row["UTMy"]),
                "elevation": float(row["Elevation"]),
                "doi": (float(row["DOI"]) if row["DOI"] is not None
                        else float("nan")),
                "data_fit": (float(row["DataFit"]) if row["DataFit"] is not None
                             else float("nan")),
                "resistivity": np.asarray(
                    json.loads(row["Resistivity"]), dtype=float).ravel(),
                "thickness": np.asarray(
                    json.loads(row["Thickness"]), dtype=float).ravel(),
                "moments": moments,
                "geometry": _station_geometry(station),
                "spec": spec,
            }
    finally:
        connection.close()


def load_reference_models(
    path: str | Path, inversion_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Every station of one stored inversion, with the project's own settings.

    Returns ``stations`` alongside the ``RxTxSpecs`` block and the acquisition
    ``protocol``, so a caller has the full geometry, waveform and gate
    description that produced the stored responses.
    """
    stations = list(iter_reference_stations(path, inversion_name))
    database = _project_database(path)
    spec = stations[0]["spec"] if stations else {}
    return {
        "stations": stations,
        "n_stations": len(stations),
        "inversion_name": stations[0]["inversion_name"] if stations else None,
        "spec": spec,
        "protocol": _temcompany_protocol(database.parent),
        "source": str(database),
    }


__all__ = [
    "REFERENCE_MOMENTS",
    "has_reference_models",
    "iter_reference_stations",
    "load_reference_models",
    "reference_inversion_names",
]
