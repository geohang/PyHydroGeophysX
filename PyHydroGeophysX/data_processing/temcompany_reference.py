"""Read inversion results stored inside a TEMcompany project file.

A project file keeps more than the recorded transients. Where a survey has
already been inverted, it holds the recovered model, the depth of
investigation, the data fit, and a per-moment record of the gate windows used,
the data the inversion was given (``InputData``), the uncertainty assigned to it
(``InputSTD``) and the response computed for the recovered model
(``ForwardData``). Earlier releases of TEMImage kept those in
``InversionModel.Datasets`` inside ``project.db``; TEMImage 3 keeps them in
``StationModelTable.ModelData`` inside ``project.tiw``.
:mod:`PyHydroGeophysX.data_processing.temcompany_project` reads either and
returns the older shape, which is the shape this module works in.

TEMImage 3 also keeps the solver's own input, output and log, which state the
constraints a run used rather than only its result. Those are reached through
:func:`PyHydroGeophysX.data_processing.temcompany_project.read_run_record`.

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

from PyHydroGeophysX.data_processing import temcompany_project
from PyHydroGeophysX.data_processing.em1d import (
    _temcompany_json_array,
    _temcompany_protocol,
)

#: Moments a reference block can name.
REFERENCE_MOMENTS: Tuple[str, ...] = ("LM", "HM")

#: Kept under the older name so callers that imported it keep working.
_project_database = temcompany_project.project_file
_read_only = temcompany_project.open_project


def has_reference_models(path: str | Path) -> bool:
    """Whether *path* holds inversion results as well as soundings."""
    try:
        connection = temcompany_project.open_project(path)
    except (OSError, ValueError, sqlite3.Error):
        return False
    try:
        return temcompany_project.has_inversion_models(connection)
    except sqlite3.Error:
        return False
    finally:
        connection.close()


def reference_inversion_names(path: str | Path) -> List[str]:
    """Every named inversion the project holds, the current one first.

    "Current" is the project's own answer where it has one. TEMImage 3
    timestamps every run in ``InversionRunTable``, so the latest finished run is
    the one the application would show; earlier releases record
    ``LastInversionName`` in ``InverseSettings``. Only where neither is present
    does the order fall back to a guess, and the guess is row count rather than
    timestamp because a later run is often a re-inversion of a few stations
    while the one worth comparing against covers the survey.
    """
    connection = temcompany_project.open_project(path)
    try:
        names = temcompany_project.inversion_names(connection)
        current = temcompany_project.last_inversion_name(connection)
    finally:
        connection.close()
    if current in names:
        names.remove(current)
        names.insert(0, current)
    return names


def _last_inversion_name(database: Path) -> Optional[str]:
    """The inversion the project's own record points at, if it names one."""
    try:
        connection = temcompany_project.open_project(database)
    except (OSError, ValueError, sqlite3.Error):
        return None
    try:
        return temcompany_project.last_inversion_name(connection)
    finally:
        connection.close()


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
    database = temcompany_project.project_file(path)
    connection = temcompany_project.open_project(database)
    try:
        specs = temcompany_project.read_specs(connection)
        stations = {
            item["AveragedDataId"]: item
            for item in temcompany_project.read_station_stacks(connection)
        }
        for row in temcompany_project.read_inversion_models(
                connection, inversion_name):
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
                # A project that never had a digital elevation model applied
                # records the GPS altitude here and nothing under Elevation, so
                # the fallback keeps a station rather than dropping it.
                "elevation": float(row["Elevation"] if row["Elevation"]
                                   is not None else row["UTMz"]),
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
