"""1D electromagnetic sounding and line-geometry readers."""

from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from PyHydroGeophysX.data_processing import run_inputs, table_io
from PyHydroGeophysX.data_processing.ttem import is_ttem_source, load_ttem_sounding

TEMCOMPANY_MOMENTS = ("LM+HM", "HM", "LM")

#: What :func:`save_sounding_container` writes and :func:`load_sounding` reads
#: back, so a recorded run carries the soundings it imported rather than the
#: acquisition folder they were parsed out of.
SOUNDING_CONTAINER_KIND = "em_soundings"


def is_temcompany_source(path: str) -> bool:
    """Return whether *path* looks like a TEMcompany/TEM2Go export.

    Both complete project directories and the self-describing ``*.xyz`` exports
    written by TEMImage are accepted.
    """
    source = Path(path)
    if source.is_dir():
        names = {item.name.lower() for item in source.iterdir() if item.is_file()}
        return ("project.db" in names
                or any(name.endswith("_stationdata.xyz") for name in names)
                or any(name.endswith("_rawdata.xyz") for name in names))
    if source.suffix.lower() != ".xyz" or not source.is_file():
        return False
    try:
        with source.open("r", encoding="utf-8-sig", errors="replace") as handle:
            return "temcompany" in "".join(handle.readline() for _ in range(4)).lower()
    except OSError:
        return False


def _temcompany_json_array(value: Any, *, dtype=float) -> np.ndarray:
    """Decode a numeric array stored by TEMcompany in SQLite."""
    if value in (None, ""):
        return np.array([], dtype=dtype)
    parsed = json.loads(value) if isinstance(value, str) else value
    return np.asarray(parsed, dtype=dtype).ravel()


def _temcompany_valid_channels(
    times: np.ndarray,
    response: np.ndarray,
    relative_std: Optional[np.ndarray] = None,
    flags: Optional[np.ndarray] = None,
    use_flags: bool = True,
    max_relative_std: Optional[float] = None,
) -> "tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]":
    """Apply TEMcompany dummy values and, unless waived, its in-use flags.

    ``use_flags=False`` keeps every gate the file holds a finite, non-dummy value
    for, including the ones the acquisition software's own QC switched off. That
    is a deliberate choice to re-do the editing here rather than inherit it: on a
    noisy ground survey the project may leave only a quarter of the gates in use,
    and the inversion's own outlier rejection can be a better judge of which of
    the rest are usable. The gates come back with their recorded stack errors, so
    a gate that was switched off for being noisy still carries that noise in its
    weight.

    ``max_relative_std`` truncates the decay tail: walking the surviving gates
    from early to late, the first one that is negative or noisier than this cut
    ends the sounding, and it and everything after it are dropped. The vendor's
    own documentation states the rule as "data points with uncertainty exceeding
    30 % are excluded from the analysis/inversion", and the in-use flags do not
    record it, so a reader that honours only the flags takes in late gates the
    acquisition software itself discarded. Measured against one project's stored
    inversion inputs, no truncation reproduces its gate selection at 53 of 116
    stations; the documented 0.30 reaches 81, and 0.25 reaches 90. ``None`` keeps
    every flagged gate.

    Truncating rather than dropping the bad gates individually is the point: past
    the first gate the decay has fallen into the noise, a later gate that happens
    to look clean is chance, and keeping it invites the inversion to fit noise at
    the depth that gate appears to probe. The vendor applies a slope filter on
    import that disables oscillating gates outright, which is what the in-use
    flags carry; this is the separate cut its inversion applies on top.

    The sign half of the test needs care. A sign change is not automatically an
    error: over a very conductive near-surface the response can genuinely start
    negative and cross over, and offset-loop systems like this one see reversals
    from 3D and ramp effects too, which is why the vendor documents sign changes
    as an ordinary feature rather than a fault. In the projects measured here its
    inversion nonetheless kept essentially no negative gate (0 of 912 LM, 1 of
    1109 HM), so cutting at the first one reproduces what it did. On a survey
    where the reversal is real, turn the cut off (``None``) rather than let it
    delete the signal.
    """
    n = min(np.size(times), np.size(response))
    t = np.asarray(times, dtype=float).ravel()[:n]
    d = np.asarray(response, dtype=float).ravel()[:n]
    mask = np.isfinite(t) & (t > 0.0) & np.isfinite(d) & (np.abs(d) < 9_000.0)
    if use_flags and flags is not None and np.size(flags):
        use = np.asarray(flags, dtype=float).ravel()
        mask &= np.pad(use[:n] > 0, (0, max(0, n - use.size)), constant_values=False)[:n]
    std = None
    if relative_std is not None and np.size(relative_std):
        raw_std = np.asarray(relative_std, dtype=float).ravel()
        raw_std = np.pad(raw_std[:n], (0, max(0, n - raw_std.size)),
                         constant_values=np.nan)[:n]
        std = raw_std[mask]
        std[~np.isfinite(std) | (std < 0.0) | (std >= 9_000.0)] = np.nan
    if max_relative_std is not None and np.any(mask):
        kept = np.flatnonzero(mask)                     # gate centres ascend
        errors = std if std is not None else np.full(kept.size, np.nan)
        spoiled = (d[kept] < 0.0) | (np.isfinite(errors)
                                     & (errors > float(max_relative_std)))
        if spoiled.any():
            stop = int(np.argmax(spoiled))
            mask[kept[stop:]] = False
            if std is not None:
                std = std[:stop]
    if not np.any(mask):
        raise ValueError("The selected TEMcompany sounding has no enabled, finite time gates.")
    return t[mask], d[mask], std


def _temcompany_positions(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return cumulative survey distance for map coordinates."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size < 2:
        return np.zeros(x.size, dtype=float)
    steps = np.hypot(np.diff(x), np.diff(y))
    steps[~np.isfinite(steps)] = 0.0
    return np.concatenate([[0.0], np.cumsum(steps)])


def _normalise_temcompany_moment(moment: str) -> str:
    selected = str(moment).upper().replace(" ", "")
    aliases = {"BOTH": "LM+HM", "JOINT": "LM+HM", "HM+LM": "LM+HM"}
    selected = aliases.get(selected, selected)
    if selected not in TEMCOMPANY_MOMENTS:
        raise ValueError(
            f"TEMcompany moment must be one of {TEMCOMPANY_MOMENTS}, got {moment!r}.")
    return selected


def _geometric_layer_thicknesses(
    n_layers: int, first_layer: float, last_depth: float
) -> np.ndarray:
    """Build the finite-layer grid used by TEMcompany's fixed-layer inversion."""
    count = max(1, int(n_layers) - 1)
    first = max(float(first_layer), 1e-6)
    target = max(float(last_depth), first * count)
    if count == 1:
        return np.asarray([target], dtype=float)
    if np.isclose(target, first * count):
        return np.full(count, first, dtype=float)
    lo, hi = 1.0, 10.0
    for _ in range(80):
        ratio = 0.5 * (lo + hi)
        total = first * (ratio ** count - 1.0) / (ratio - 1.0)
        if total < target:
            lo = ratio
        else:
            hi = ratio
    ratio = 0.5 * (lo + hi)
    return first * ratio ** np.arange(count, dtype=float)


def _temcompany_inversion_defaults(con: sqlite3.Connection) -> Dict[str, Any]:
    """Read the inversion settings saved inside a TEMcompany project."""
    try:
        row = con.execute("SELECT * FROM UserSettingsJson ORDER BY 1 DESC LIMIT 1").fetchone()
        if row is None:
            return {}
        raw = next(
            (value for value in row if isinstance(value, str) and value.lstrip().startswith("{")),
            None,
        )
        settings = json.loads(raw) if raw else {}
        inverse = dict(settings.get("InverseSettings", {}))
        saved = dict(settings.get("SavedInversions", {}))
        last_name = inverse.get("LastInversionName")
        if last_name in saved:
            inverse = {**inverse, **dict(saved[last_name])}
        n_layers = int(inverse.get("Nlayers", 0) or 0)
        first = float(inverse.get("FirstLayer", 0.0) or 0.0)
        last_depth = float(inverse.get("LastDepth", 0.0) or 0.0)
        if n_layers < 2 or first <= 0.0 or last_depth <= 0.0:
            return {}
        thickness = _geometric_layer_thicknesses(n_layers, first, last_depth)
        moment = str(inverse.get("InversionMoment", "Both"))
        return {
            "n_layers": n_layers,
            "min_thickness": float(thickness[0]),
            "max_thickness": float(thickness[-1]),
            "layer_thicknesses": thickness,
            "last_depth": last_depth,
            "starting_resistivity": float(
                inverse.get("ManualStartModelResistivity", 40.0) or 40.0),
            "smoothness": float(inverse.get("VerticalSmoothness", 2.0)),
            "lateral_smoothness": float(inverse.get("LateralSmoothness", 1.3)),
            "tem_moment": _normalise_temcompany_moment(moment),
            "constraint": str(inverse.get("InversionConstraint", "LCI")),
            "norm": str(inverse.get("InversionType", "L2")),
            "reference_distance": float(inverse.get("LcRefDistance", 10.0) or 10.0),
            # TEMcompany stores LcAutoScale rather than its internal numerical
            # multiplier. Normalize a layer-vector constraint by sqrt(N), the
            # standard L2 dimension scaling, when that switch is enabled.
            "lateral_weight_scale": (
                math.sqrt(n_layers)
                if bool(inverse.get("LcAutoScale", False)) else 1.0
            ),
        }
    except (sqlite3.Error, TypeError, ValueError, json.JSONDecodeError):
        return {}


def _response_on_times(data: Dict[str, Any], reference_times: np.ndarray) -> np.ndarray:
    """Place a TDEM response on a reference gate vector, filling absent gates."""
    reference = np.asarray(reference_times, dtype=float).ravel()
    times = np.asarray(data["times"], dtype=float).ravel()
    response = np.asarray(data["response"], dtype=float).ravel()
    if times.size == reference.size and np.allclose(times, reference, rtol=1e-7, atol=0.0):
        return response
    aligned = np.full(reference.size, np.nan, dtype=float)
    for index, value in enumerate(reference):
        matches = np.flatnonzero(np.isclose(times, value, rtol=1e-7, atol=1e-15))
        if matches.size:
            aligned[index] = response[matches[0]]
    return aligned


def _temcompany_protocol(folder: Path) -> Dict[str, Any]:
    """Acquisition settings from the ``.sts`` protocol beside a project.

    Almost everything the forward needs is duplicated in ``project.db``: loop
    geometry, turn-off waveform, gate windows, filter corners, currents. Three
    things are only here.

    ``UniStd`` is the uniform relative uncertainty the instrument adds to every
    gate on top of its stack error, the "3 % uniform uncertainty" the vendor
    documentation describes. Reading it means the number comes from the protocol
    that was actually run rather than from a default that happens to match this
    one. ``LM_StackSize`` / ``HM_StackSize`` are the transient counts, which an
    absolute noise model would need. ``ChA_WinFunc`` and ``ChA_GateOverlap``
    describe the gate taper; the gate averaging here uses a flat window, so a
    shaped one is a small refinement still on the table.

    Returns an empty dict when no protocol file is present, which is the normal
    case for a project copied without it.
    """
    try:
        files = sorted(Path(folder).glob("*.sts"))
    except OSError:
        return {}
    if not files:
        return {}
    values: Dict[str, str] = {}
    try:
        for line in files[0].read_text(encoding="utf-8-sig",
                                       errors="replace").splitlines():
            body = line.split(";", 1)[0].strip()
            key, sep, value = body.partition("=")
            if sep and key.strip():
                values[key.strip()] = value.strip()
    except OSError:
        return {}

    def number(name: str) -> Optional[float]:
        try:
            return float(values[name].split(",")[0])
        except (KeyError, ValueError):
            return None

    result: Dict[str, Any] = {"protocol_file": files[0].name}
    for key, name in (("uniform_std", "UniStd"),
                      ("stack_size_lm", "LM_StackSize"),
                      ("stack_size_hm", "HM_StackSize"),
                      ("gate_overlap", "ChA_GateOverlap"),
                      ("gates_per_decade", "ChA_GatePerDecade"),
                      ("gate_window_shape", "ChA_WinFunc"),
                      ("gate_window_par", "ChA_WinFuncPar1"),
                      ("powerline_hz", "PowerLineMonitorFreq")):
        value = number(name)
        if value is not None:
            result[key] = value
    if "AutoSignDetection" in values:
        result["auto_sign_detection"] = values["AutoSignDetection"]
    return result


def _temcompany_transmitter(spec: Dict[str, Any], moment: str) -> Dict[str, Any]:
    """Turn-off waveform and gate windows for one moment, as the file records them.

    TEMcompany stores the measured ramp as (time, amplitude) nodes and every
    gate's open and close time. Both matter for a ground system: the first gate
    opens a microsecond or two after the ramp ends, where a step-off stand-in is
    at its worst, and the late gates are wide enough that the centre value is not
    the window average.
    """
    times = np.asarray(spec.get(f"{moment}WaveformTime", []), dtype=float).ravel()
    currents = np.asarray(spec.get(f"{moment}WaveformAmplitude", []),
                          dtype=float).ravel()
    usable = times.size >= 2 and times.size == currents.size
    centre = np.asarray(spec.get(f"{moment}_GateCentreTime", []), dtype=float).ravel()
    opens = np.asarray(spec.get(f"{moment}_GateOpenTime", []), dtype=float).ravel()
    closes = np.asarray(spec.get(f"{moment}_GateCloseTime", []), dtype=float).ravel()
    windows = (centre.size and centre.size == opens.size == closes.size)
    return {
        "waveform_times": times if usable else None,
        "waveform_currents": currents if usable else None,
        "gate_windows": ({"centre": centre, "open": opens, "close": closes}
                         if windows else None),
    }


def _temcompany_column(rows, key: str, dtype=float, default=np.nan) -> np.ndarray:
    """One column of ``StationStackData``, or the default where it is absent.

    The schema has grown across TEMcompany releases, and the synthetic exporter
    writes only what its example needs, so a project may carry no Longitude or
    Latitude at all. ``sqlite3.Row`` raises ``IndexError`` for a name it does
    not hold rather than returning None, which turns a missing optional column
    into a failure to open the survey.
    """
    values = []
    for row in rows:
        try:
            value = row[key]
        except (IndexError, KeyError):
            value = None
        values.append(default if value is None else value)
    return np.asarray(values, dtype=dtype)


def _temcompany_system(spec: Dict[str, Any], row: Optional[sqlite3.Row] = None) -> Dict[str, Any]:
    """Map TEMcompany loop/receiver metadata to the studio geometry."""
    area = float(spec.get("TxLoopArea", 0.0) or 0.0)
    if area <= 0.0:
        xy = np.asarray(spec.get("TxLoopXYlength", []), dtype=float).ravel()
        if xy.size >= 2:
            area = float(abs(xy[0] * xy[1]))
    radius = math.sqrt(area / math.pi) if area > 0.0 else 10.0
    tx = np.asarray(spec.get("TxLoopXYZPos", [0.0, 0.0, 0.0]), dtype=float).ravel()
    rx = np.asarray(spec.get("RxCoilXYZPos", [0.0, 0.0, 0.0]), dtype=float).ravel()
    sep = float(np.linalg.norm(rx[:2] - tx[:2])) if tx.size >= 2 and rx.size >= 2 else 0.0
    height = None
    if row is not None:
        for key in ("RxCoilHeight", "TxCoilHeight"):
            try:
                value = row[key]
            except (IndexError, KeyError):
                value = None
            if value is not None and np.isfinite(float(value)):
                height = float(value)
                break
    if height is None:
        zs = [arr[2] for arr in (tx, rx) if arr.size >= 3 and np.isfinite(arr[2])]
        height = float(np.mean(zs)) if zs else 0.0
    return {
        "source_radius": radius,
        "tx_rx_sep": sep,
        "height": height,
        "orientation": "z",
        "waveform": "step_off",
        "receiver_type": "dbdt",
        "response_sign": -1.0,
        "data_scale": 1.0,
        "auto_scale": False,
        "loop_area": area,
        "loop_turns": int(spec.get("NTurnsTxLoop", 1) or 1),
        # The export is dB/dt already divided by the transmitter moment
        # (V/A/m^4), so the forward has to model a UNIT moment. Modelling the
        # real moment on top counts it twice: on this instrument that is a
        # factor turns * area = 1.59, and the inversion pays for it by raising
        # every recovered resistivity to bring the amplitude back down.
        "source_moment": 1.0,
    }


def _load_temcompany_database(path: Path, sounding: int, moment: str,
                              use_flags: bool = True,
                              max_relative_std: Optional[float] = None) -> Dict[str, Any]:
    """Load one stacked sounding and project geometry from ``project.db``."""
    uri = path.resolve().as_uri() + "?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    con.row_factory = sqlite3.Row
    try:
        tables = {row[0] for row in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        required = {"StationStackData", "RxTxSpecs"}
        if not required.issubset(tables):
            raise ValueError(
                f"{path.name} is not a supported TEMcompany database "
                f"(missing {', '.join(sorted(required - tables))}).")
        specs = {
            item["RxTxSpecsId"]: json.loads(item["RxTxSpecsJson"])
            for item in con.execute("SELECT * FROM RxTxSpecs")
            if item["RxTxSpecsJson"]
        }
        all_rows = list(con.execute(
            "SELECT * FROM StationStackData "
            "ORDER BY LineNumber, AveragedDataId"))
        value_key = f"{moment}_VoltageValues"
        rows = []
        for candidate in all_rows:
            if candidate[value_key] in (None, "", "[]"):
                continue
            candidate_spec = specs.get(
                candidate["RxTxSpecsId"], next(iter(specs.values()), {}))
            try:
                _temcompany_valid_channels(
                    np.asarray(candidate_spec.get(f"{moment}_GateCentreTime", []), dtype=float),
                    _temcompany_json_array(candidate[value_key]),
                    _temcompany_json_array(candidate[f"{moment}_VoltageValues_STD"]),
                    _temcompany_json_array(candidate[f"{moment}_InUseFlags"]),
                    use_flags, max_relative_std,
                )
            except (ValueError, TypeError, json.JSONDecodeError):
                continue
            rows.append(candidate)
        if not rows:
            raise ValueError(f"No enabled {moment} station stacks were found in {path.name}.")
        s = max(0, min(int(sounding), len(rows) - 1))
        row = rows[s]
        spec = specs.get(row["RxTxSpecsId"], next(iter(specs.values()), {}))
        times = np.asarray(spec.get(f"{moment}_GateCentreTime", []), dtype=float)
        response = _temcompany_json_array(row[value_key])
        std = _temcompany_json_array(row[f"{moment}_VoltageValues_STD"])
        flags = _temcompany_json_array(row[f"{moment}_InUseFlags"])
        times, response, std = _temcompany_valid_channels(
            times, response, std, flags, use_flags, max_relative_std)

        x = np.asarray([item["UtmX"] for item in rows], dtype=float)
        y = np.asarray([item["UtmY"] for item in rows], dtype=float)
        elevation = _temcompany_column(rows, "Elevation")
        heights = _temcompany_column(rows, "RxCoilHeight")
        zone = spec.get("UTMZone")
        zone_letter = str(spec.get("UTMZoneLetter", "") or "").strip()
        coordinate_system = (
            f"UTM zone {zone:g}{zone_letter}"
            if isinstance(zone, (int, float)) else
            f"UTM zone {zone}{zone_letter}" if zone not in (None, "") else "UTM"
        )
        result: Dict[str, Any] = {
            "times": times,
            "response": response,
            "n_soundings": len(rows),
            "sounding": s,
            "relative_std": std,
            "positions": _temcompany_positions(x, y),
            "x": x,
            "y": y,
            # Kept alongside the UTM pair so a map view can place imagery
            # without a projection library (see visualization.basemap).
            "longitude": _temcompany_column(rows, "Longitude"),
            "latitude": _temcompany_column(rows, "Latitude"),
            "elevation": elevation,
            "heights": heights,
            "line_numbers": np.asarray([item["LineNumber"] for item in rows], dtype=int),
            "station_ids": np.asarray([str(item["StationId"]) for item in rows]),
            "transmitter": _temcompany_transmitter(spec, moment),
            "temcompany": True,
            "tem_moment": moment,
            "source_format": "TEMcompany project database",
            "coordinate_system": coordinate_system,
            "system": _temcompany_system(spec, row),
            "inversion_defaults": _temcompany_inversion_defaults(con),
            "protocol": _temcompany_protocol(path.parent),
        }
        return result
    finally:
        con.close()


def _load_temcompany_joint_database(path: Path, sounding: int,
                                    use_flags: bool = True,
                                    max_relative_std: Optional[float] = None) -> Dict[str, Any]:
    """Load all usable LM/HM gates for one station, preserving common station order."""
    uri = path.resolve().as_uri() + "?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    con.row_factory = sqlite3.Row
    try:
        specs = {
            item["RxTxSpecsId"]: json.loads(item["RxTxSpecsJson"])
            for item in con.execute("SELECT * FROM RxTxSpecs")
            if item["RxTxSpecsJson"]
        }
        entries: List["tuple[sqlite3.Row, Dict[str, Dict[str, np.ndarray]]]"] = []
        for row in con.execute(
            "SELECT * FROM StationStackData ORDER BY LineNumber, AveragedDataId"
        ):
            spec = specs.get(row["RxTxSpecsId"], next(iter(specs.values()), {}))
            moments: Dict[str, Dict[str, np.ndarray]] = {}
            for selected in ("LM", "HM"):
                try:
                    times, response, std = _temcompany_valid_channels(
                        np.asarray(spec.get(f"{selected}_GateCentreTime", []), dtype=float),
                        _temcompany_json_array(row[f"{selected}_VoltageValues"]),
                        _temcompany_json_array(row[f"{selected}_VoltageValues_STD"]),
                        _temcompany_json_array(row[f"{selected}_InUseFlags"]),
                        use_flags, max_relative_std,
                    )
                except (ValueError, TypeError, json.JSONDecodeError):
                    continue
                moments[selected] = {
                    "times": times,
                    "response": response,
                    "transmitter": _temcompany_transmitter(spec, selected),
                    "relative_std": (
                        np.asarray(std, dtype=float)
                        if std is not None else np.array([], dtype=float)
                    ),
                }
            if moments:
                entries.append((row, moments))
        if not entries:
            raise ValueError(f"No enabled LM/HM station stacks were found in {path.name}.")

        index = max(0, min(int(sounding), len(entries) - 1))
        row, moments = entries[index]
        spec = specs.get(row["RxTxSpecsId"], next(iter(specs.values()), {}))
        preview_name = "HM" if "HM" in moments else "LM"
        preview = moments[preview_name]
        rows = [entry[0] for entry in entries]
        x = np.asarray([item["UtmX"] for item in rows], dtype=float)
        y = np.asarray([item["UtmY"] for item in rows], dtype=float)
        zone = spec.get("UTMZone")
        zone_letter = str(spec.get("UTMZoneLetter", "") or "").strip()
        coordinate_system = (
            f"UTM zone {zone:g}{zone_letter}"
            if isinstance(zone, (int, float)) else
            f"UTM zone {zone}{zone_letter}" if zone not in (None, "") else "UTM"
        )
        return {
            "times": preview["times"],
            "response": preview["response"],
            "relative_std": preview["relative_std"],
            "moments": moments,
            "available_moments": tuple(moments),
            "n_soundings": len(entries),
            "sounding": index,
            "positions": _temcompany_positions(x, y),
            "x": x,
            "y": y,
            "longitude": _temcompany_column(rows, "Longitude"),
            "latitude": _temcompany_column(rows, "Latitude"),
            "elevation": _temcompany_column(rows, "Elevation"),
            "heights": _temcompany_column(rows, "RxCoilHeight"),
            "line_numbers": np.asarray([item["LineNumber"] for item in rows], dtype=int),
            "station_ids": np.asarray([str(item["StationId"]) for item in rows]),
            "average_data_ids": np.asarray(
                [item["AveragedDataId"] for item in rows], dtype=int),
            "temcompany": True,
            "tem_moment": "LM+HM",
            "source_format": "TEMcompany project database",
            "coordinate_system": coordinate_system,
            "system": _temcompany_system(spec, row),
            "inversion_defaults": _temcompany_inversion_defaults(con),
            "protocol": _temcompany_protocol(path.parent),
        }
    finally:
        con.close()


def _temcompany_comment_values(comments: List[str], key: str) -> np.ndarray:
    wanted = key.lower()
    for line in comments:
        body = line.lstrip("/").strip()
        label, separator, values = body.partition(":")
        if separator and label.strip().lower() == wanted:
            return np.fromstring(values.replace(",", " "), sep=" ", dtype=float)
    return np.array([], dtype=float)


def _temcompany_comment_scalar(comments: List[str], key: str, default: float = 0.0) -> float:
    values = _temcompany_comment_values(comments, key)
    return float(values[0]) if values.size else float(default)


def _load_temcompany_xyz(path: Path, sounding: int, moment: str) -> Dict[str, Any]:
    """Load TEMcompany station-stacked or raw ``*.xyz`` text exports."""
    lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    comments = [line.strip() for line in lines if line.lstrip().startswith("/")]
    header: Optional[List[str]] = None
    for line in comments:
        fields = line.lstrip("/").split()
        lowered = {field.lower() for field in fields}
        if "project" in lowered and "date" in lowered and "line" in lowered:
            header = fields
    if not header:
        raise ValueError(f"Could not find the TEMcompany column header in {path.name}.")
    records = [
        line.split() for line in lines
        if line.strip() and not line.lstrip().startswith("/")
    ]
    records = [row for row in records if len(row) >= len(header)]
    if not records:
        raise ValueError(f"No TEMcompany data records were found in {path.name}.")
    columns = {name.lower(): index for index, name in enumerate(header)}
    raw_export = "channel" in columns
    if raw_export:
        channel_index = columns["channel"]
        records = [row for row in records if row[channel_index].upper() == moment]
        gate_names = [name for name in header if name.lower().startswith("gate")]
        std_names: List[str] = []
        station_name = "station-id"
    else:
        gate_names = [name for name in header if name.lower().startswith(moment.lower() + "gate")]
        std_names = [name for name in header if name.lower().startswith(moment.lower() + "std")]
        station_name = "station"
    if not records or not gate_names:
        raise ValueError(f"No {moment} gates were found in {path.name}.")
    gate_names.sort(key=lambda name: int("".join(filter(str.isdigit, name)) or 0))
    std_names.sort(key=lambda name: int("".join(filter(str.isdigit, name)) or 0))
    times = _temcompany_comment_values(comments, f"{moment}_GateCentreTime")
    if not times.size:
        raise ValueError(f"{moment} gate centre times are missing from {path.name}.")

    def numeric(name: str, row: List[str], default: float = np.nan) -> float:
        index = columns.get(name.lower())
        try:
            return float(row[index]) if index is not None else float(default)
        except (TypeError, ValueError):
            return float(default)

    lat = np.asarray([numeric("latitude", row) for row in records])
    lon = np.asarray([numeric("longitude", row) for row in records])
    elevation = np.asarray([numeric("elevation", row) for row in records])
    # Standalone XYZ exports carry geographic coordinates but no CRS transform.
    # Use a local tangent-plane approximation so plan/section distances remain
    # metric without introducing a pyproj dependency.
    lat0 = float(np.nanmean(lat))
    lon0 = float(np.nanmean(lon))
    earth_radius = 6_371_000.0
    x = np.deg2rad(lon - lon0) * earth_radius * math.cos(math.radians(lat0))
    y = np.deg2rad(lat - lat0) * earth_radius
    s = max(0, min(int(sounding), len(records) - 1))
    row = records[s]
    response = np.asarray([numeric(name, row) for name in gate_names], dtype=float)
    std = (np.asarray([numeric(name, row) for name in std_names], dtype=float)
           if std_names else None)
    times, response, std = _temcompany_valid_channels(times, response, std)

    area = _temcompany_comment_scalar(comments, "LoopArea (m2)", 0.0)
    loop_x = _temcompany_comment_scalar(comments, "LoopX (m)", 0.0)
    loop_y = _temcompany_comment_scalar(comments, "LoopY (m)", 0.0)
    if area <= 0.0:
        area = abs(loop_x * loop_y)
    rx_x = _temcompany_comment_scalar(comments, "RXcoil X-Position (m)", 0.0)
    height = _temcompany_comment_scalar(comments, "LoopZ (m)", 0.0)
    system = {
        "source_radius": math.sqrt(area / math.pi) if area > 0.0 else 10.0,
        "tx_rx_sep": abs(rx_x),
        "height": height,
        "orientation": "z",
        "waveform": "step_off",
        "receiver_type": "dbdt",
        "response_sign": -1.0,
        "data_scale": 1.0,
        "auto_scale": False,
        "loop_area": area,
        "loop_turns": int(_temcompany_comment_scalar(comments, "LoopTurns", 1.0)),
        # The export is dB/dt already divided by the transmitter moment
        # (V/A/m^4), so the forward has to model a UNIT moment. Modelling the
        # real moment on top counts it twice: on this instrument that is a
        # factor turns * area = 1.59, and the inversion pays for it by raising
        # every recovered resistivity to bring the amplitude back down.
        "source_moment": 1.0,
    }
    station_index = columns.get(station_name)
    return {
        "times": times,
        "response": response,
        "n_soundings": len(records),
        "sounding": s,
        "relative_std": std,
        "positions": _temcompany_positions(x, y),
        "x": x,
        "y": y,
        "longitude": lon,
        "latitude": lat,
        "elevation": elevation,
        "heights": np.full(len(records), height, dtype=float),
        "line_numbers": np.asarray(
            [int(numeric("line", item, 0)) for item in records], dtype=int),
        "station_ids": np.asarray([
            item[station_index] if station_index is not None else str(index + 1)
            for index, item in enumerate(records)
        ]),
        "temcompany": True,
        "tem_moment": moment,
        "source_format": "TEMcompany raw XYZ" if raw_export else "TEMcompany station XYZ",
        "coordinate_system": "local metric (from latitude/longitude)",
        "system": system,
    }


def load_temcompany_sounding(
    path: str, sounding: int = 0, moment: str = "HM", *, use_flags: bool = True,
    max_relative_std: Optional[float] = None,
) -> Dict[str, Any]:
    """Load a TEMcompany/TEM2Go sounding from a project folder or XYZ export.

    ``use_flags=False`` ignores the project's in-use flags and returns every gate
    with a finite, non-dummy value. Only a project database records those flags,
    so it makes no difference to an XYZ export.
    """
    selected = _normalise_temcompany_moment(moment)
    source = Path(path)
    if source.is_dir():
        databases = sorted(source.glob("*.db"))
        project_db = next(
            (item for item in databases if item.name.lower() == "project.db"),
            databases[0] if len(databases) == 1 else None,
        )
        if project_db is not None:
            if selected == "LM+HM":
                return _load_temcompany_joint_database(
                    project_db, sounding, use_flags, max_relative_std)
            return _load_temcompany_database(
                project_db, sounding, selected, use_flags, max_relative_std)
        xyz = sorted(source.glob("*_StationData.xyz"))
        if not xyz:
            xyz = sorted(source.glob("*_RawData.xyz"))
        if len(xyz) != 1:
            raise ValueError(
                "Select a TEMcompany project directory containing project.db or "
                "one *_StationData.xyz file.")
        source = xyz[0]
    if not source.exists():
        raise ValueError(f"File not found: {source}")
    if selected == "LM+HM":
        raise ValueError(
            "Joint LM+HM loading currently requires a TEMcompany project folder "
            "containing project.db; select HM or LM for a standalone XYZ export.")
    return _load_temcompany_xyz(source, sounding, selected)


def load_sounding(
    path: str, method: str, sounding: int = 0, *, moment: str = "HM",
    use_flags: bool = True, max_relative_std: Optional[float] = None,
    ttem_loop_area: Optional[float] = None, ttem_gex_path: Optional[str] = None,
    ttem_tfi_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Load one sounding from a sounding file.

    The first column is the abscissa (FDEM: frequency Hz; TDEM: time s). The
    remaining columns hold the response(s) — a single sounding, or several stacked
    side by side so one file can carry a whole survey line (common for airborne EM
    exports). ``sounding`` picks which one (0-based):

    - **TDEM**: each extra column is one sounding's response → column ``1 + sounding``.
    - **FDEM**: response columns come in ``(real, imag)`` pairs → one sounding is the
      pair starting at ``1 + 2*sounding``; a lone trailing real column gives imag = 0.

    The returned dict also reports ``n_soundings`` so the caller can offer a picker.
    ``use_flags`` applies only to TEMcompany project databases; see
    :func:`load_temcompany_sounding`.
    """
    if run_inputs.is_container(path):
        return load_sounding_container(path, method, sounding=sounding)
    if is_ttem_source(path):
        if method != "TDEM":
            raise ValueError("TEMcompany tTEM raw data are time-domain EM data; select TDEM.")
        return load_ttem_sounding(path, sounding=sounding, moment=moment,
                                  max_relative_std=max_relative_std,
                                  loop_area=ttem_loop_area,
                                  gex_path=ttem_gex_path,
                                  tfi_path=ttem_tfi_path)
    if is_temcompany_source(path):
        if method != "TDEM":
            raise ValueError("TEMcompany/TEM2Go exports are time-domain EM data; select TDEM.")
        return load_temcompany_sounding(path, sounding=sounding, moment=moment,
                                        use_flags=use_flags,
                                        max_relative_std=max_relative_std)
    table = np.atleast_2d(table_io.load_2d_array(path)).astype(float)
    if table.shape[1] < 2:
        raise ValueError(f"Expected >= 2 columns, got shape {table.shape}.")
    x = table[:, 0]
    n_resp = table.shape[1] - 1
    if method == "FDEM":
        n_soundings = max(1, (n_resp + 1) // 2)
        s = max(0, min(int(sounding), n_soundings - 1))
        ri = 1 + 2 * s
        real = table[:, ri]
        imag = table[:, ri + 1] if ri + 1 < table.shape[1] else np.zeros_like(x)
        return {"frequencies": x, "real": real, "imag": imag,
                "n_soundings": n_soundings, "sounding": s}
    n_soundings = n_resp
    s = max(0, min(int(sounding), n_soundings - 1))
    return {"times": x, "response": table[:, 1 + s],
            "n_soundings": n_soundings, "sounding": s}


def save_sounding_container(
    destination: str | Path, path: str, method: str, *, moment: str = "HM",
    use_flags: bool = True, max_relative_std: Optional[float] = None,
    ttem_loop_area: Optional[float] = None, ttem_gex_path: Optional[str] = None,
    ttem_tfi_path: Optional[str] = None, progress=None,
) -> Path:
    """Materialize every sounding in ``path`` into one compressed container.

    A recorded run used to keep its input by copying the acquisition folder,
    which for a TEMcompany project is hundreds of megabytes per inversion. The
    soundings themselves are a few hundred kilobytes, and they are the only part
    the inversion reads, so this parses the survey once and stores the result.

    The load settings are baked in, because they decide what the arrays contain:
    a container written for ``LM+HM`` holds joint moments, and re-reading it
    under another moment would silently return the wrong gates. They travel in
    the manifest so a reader can report what it was given.
    """
    settings = dict(
        moment=moment, use_flags=use_flags, max_relative_std=max_relative_std,
        ttem_loop_area=ttem_loop_area, ttem_gex_path=ttem_gex_path,
        ttem_tfi_path=ttem_tfi_path,
    )
    head = load_sounding(path, method, sounding=0, **settings)
    total = max(1, int(head.get("n_soundings", 1)))
    soundings: List[Dict[str, Any]] = [head]
    for index in range(1, total):
        soundings.append(load_sounding(path, method, sounding=index, **settings))
        if progress is not None and index % 100 == 0:
            progress(f"  materialized {index + 1}/{total} soundings")
    return run_inputs.save_sequence_container(
        destination,
        soundings,
        kind=SOUNDING_CONTAINER_KIND,
        meta={
            "method": str(method).upper(),
            "n_soundings": total,
            "source_name": Path(path).name,
            "source_format": str(head.get("source_format", "")),
            **{key: value for key, value in settings.items() if value is not None},
        },
    )


def load_sounding_container(
    path: str | Path, method: str, sounding: int = 0
) -> Dict[str, Any]:
    """Return one sounding from a container written by :func:`save_sounding_container`.

    The moment and flag settings are not arguments here: they were applied when
    the container was written, and the stored arrays are what they produced.
    """
    manifest = run_inputs.read_manifest(path)
    stored_method = str(manifest.get("meta", {}).get("method", "")).upper()
    if stored_method and str(method).upper() != stored_method:
        raise ValueError(
            f"{Path(path).name} holds {stored_method} soundings; {str(method).upper()} "
            "was requested. Re-import the survey to invert it as the other method."
        )
    if run_inputs.sequence_length(path) < 1:
        raise ValueError(f"{Path(path).name} holds no soundings.")
    return dict(
        run_inputs.load_sequence_item(
            path, int(sounding), kind=SOUNDING_CONTAINER_KIND
        )
    )


def load_line_geometry(path: str) -> Dict[str, Any]:
    """Load per-sounding line geometry: along-line ``positions`` (m), optional
    sensor ``heights`` (m), and optional map coordinates ``x``/``y`` (e.g.
    easting/northing) for plan-view depth slices. Recognizes header names
    (distance/position for the position; alt/height for the height;
    easting/northing for the map coordinates, which also derive the distance when
    no distance column is present). A header-less file is read by column order
    (1 column = position; 2+ = position, height). ``positions`` is shifted to
    start at 0.
    """
    positions = heights = xs = ys = None
    try:
        import pandas as pd
        df = pd.read_csv(path)
        if df.shape[1] >= 1 and any(not str(c).replace(".", "").lstrip("-").isdigit()
                                    for c in df.columns):
            cols = {str(c).lower().strip(): c for c in df.columns}

            def pick(names):
                for nm in names:
                    if nm in cols:
                        return np.asarray(df[cols[nm]], dtype=float).ravel()
                return None

            positions = pick(["dist_m", "distance", "position", "dist", "offset"])
            heights = pick(["sensor_alt_m", "alt", "altitude", "height", "sensor_height", "clearance"])
            xs = pick(["e_utm13n", "easting_m", "easting", "east", "e", "x_utm", "x_m", "x"])
            ys = pick(["n_utm13n", "northing_m", "northing", "north", "n", "y_utm", "y_m", "y"])
            if positions is None and xs is not None and ys is not None:
                positions = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(xs), np.diff(ys)))])
            elif positions is None and xs is not None:
                positions = xs
    except Exception:  # noqa: BLE001 - fall back to a plain numeric table
        positions = None
    if positions is None:
        arr = np.atleast_2d(table_io.load_2d_array(path)).astype(float)
        positions = arr[:, 0]
        heights = arr[:, 1] if arr.shape[1] >= 2 else None
    positions = np.asarray(positions, dtype=float).ravel()
    positions = positions - float(np.nanmin(positions))
    if heights is not None:
        heights = np.asarray(heights, dtype=float).ravel()
    has_xy = xs is not None and ys is not None
    return {"positions": positions, "heights": heights,
            "x": np.asarray(xs, dtype=float).ravel() if xs is not None else None,
            "y": np.asarray(ys, dtype=float).ravel() if ys is not None else None,
            "n": int(positions.size), "has_heights": heights is not None, "has_xy": has_xy}

__all__ = [
    "SOUNDING_CONTAINER_KIND",
    "TEMCOMPANY_MOMENTS",
    "is_temcompany_source",
    "is_ttem_source",
    "load_temcompany_sounding",
    "load_ttem_sounding",
    "load_sounding",
    "load_sounding_container",
    "load_line_geometry",
    "save_sounding_container",
]
