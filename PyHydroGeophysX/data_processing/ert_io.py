"""Robust ERT file loading for the desktop studio (Qt-free).

A single source of truth for turning an ERT data file into a pygimli
``DataContainerERT`` with correct geometry, topography, and apparent
resistivity. Both the single-inversion ERT module and the time-lapse pipeline
use this so they behave identically.

Why this exists: pygimli's native ``ert.load`` cannot parse several common
field formats. The E4D survey export (used here with a ``.ohm`` extension), for
example, carries a leading index column and no ``# a b m n`` token header, so
``ert.load`` misreads the index column as coordinates, drops the topography,
and discards every measurement (``size() == 0``). The device-specific parsers in
:mod:`PyHydroGeophysX.data_processing.ert_data_agent` (resipy or the embedded
fallback) handle those layouts; this module wires them to pygimli and falls back
across loaders so a file never silently loads empty.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np

from PyHydroGeophysX._internal.utils import noop as _noop

LogFn = Callable[[str], None]


# ---------------------------------------------------------------------------
# StandardERT -> pygimli DataContainerERT
# ---------------------------------------------------------------------------
def electrode_elevation(electrodes) -> np.ndarray:
    """Per-electrode elevation aligned with ``electrodes``.

    ``load_ert_resipy`` carries 2D elevation in ``y`` (its
    ``_normalize_elevation_axis`` moves a flat-y/varying-z profile so elevation
    lives in ``y``); fall back to ``z`` if ``y`` is flat.
    """
    if not electrodes:
        return np.zeros(0)
    ys = np.array([float(e.y) for e in electrodes])
    zs = np.array([float(e.z) for e in electrodes])
    return ys if ys.std() >= zs.std() else zs


def standard_to_pg(std):
    """Build a pygimli ``DataContainerERT`` from a ``StandardERT`` for inversion.

    Most instrument loaders report transfer resistance (V/I), not apparent
    resistivity, so apparent resistivity is recovered with geometric factors:
    ``rhoa = R * k``. When the source already provides apparent resistivity it is
    used as-is. pygimli's forward operator uses the same ``data["k"]``, so the
    inversion stays self-consistent. Returns ``None`` if pygimli is unavailable
    or no measurement maps onto the electrode set.
    """
    try:
        import pygimli as pg
        from pygimli.physics import ert as pg_ert
    except Exception:  # noqa: BLE001
        return None
    data = pg.DataContainerERT()
    id_to_idx = {}
    elecs = std.electrodes or []
    elev = electrode_elevation(elecs)
    for i, e in enumerate(elecs):
        # Elevation goes in y, which is the vertical axis for a 2D problem.
        #
        # PyGIMLi's analytic geometric factor mirrors each electrode across a free
        # surface fixed at z = 0. An electrode at z = 0 coincides with its image
        # and gets the half-space factor 2*pi*a; one at z = 211 has its image 422 m
        # away, contributing nothing, so it gets the whole-space 4*pi*a instead.
        # Putting elevation in z therefore doubles k exactly. Since the forward
        # response is (U/I)*k, the inversion halves the model to compensate and
        # chi2 never moves, so nothing else in the pipeline notices: createMesh
        # reads the same container as a 2D section with topography and builds the
        # right mesh. Only k is affected, and only silently.
        data.createSensor(pg.Pos(float(e.x), float(elev[i])))
        id_to_idx[int(e.id)] = i
    keys = ("A", "B", "M", "N")
    valid = [
        o for o in (std.observations or [])
        if o.app_res is not None and all(int(getattr(o.quad, k)) in id_to_idx for k in keys)
    ]
    if not valid:
        return None
    data.resize(len(valid))
    for name, qk in zip(("a", "b", "m", "n"), keys):
        data.set(name, [id_to_idx[int(getattr(o.quad, qk))] for o in valid])
    vals = np.array([float(o.app_res) for o in valid], dtype=float)
    data.set("err", [float(o.rel_err) if o.rel_err else 0.05 for o in valid])
    try:
        data["k"] = pg_ert.createGeometricFactors(data, numerical=False)
        k = np.asarray(data["k"], dtype=float)
    except Exception:  # noqa: BLE001
        k = np.ones(len(valid))
    source = str((std.metadata or {}).get("app_res_source", "")).lower()
    if source == "resistance":
        resistance, rhoa = vals, vals * k
    else:
        # The source gave apparent resistivity, so the transfer resistance is
        # recovered from it. Storing rhoa in "r" as well would make the two
        # disagree by a factor of k, which quietly breaks any error model with an
        # absolute term (relative + absolute/|R|).
        rhoa = vals
        resistance = vals / np.where(np.abs(k) > 1e-12, k, np.nan)
    data.set("r", np.nan_to_num(resistance, nan=0.0, posinf=0.0, neginf=0.0))
    data.set("rhoa", rhoa)
    # The potential and current, where the reader gave them for every reading and
    # said they are in volts and amperes - PyGIMLi's units for these tokens. Other
    # routes fill the same fields in their reader's own units (ResIPy's current
    # is not in amperes), and those are left out rather than mislabelled.
    if (std.metadata or {}).get("potential_current_units") == "V, A":
        for token, attribute in (("u", "dV"), ("i", "I")):
            values = [getattr(o, attribute, None) for o in valid]
            if all(value is not None for value in values):
                data.set(token, np.asarray(values, dtype=float))
    # Transmitter contact resistance in ohm, under a token of its own (PyGIMLi has
    # none for it). All or nothing: a reading with no value would otherwise fail
    # a contact-resistance check it never took.
    contacts = [getattr(o, "contact_r", None) for o in valid]
    if contacts and all(value is not None for value in contacts):
        data.set("rc", np.asarray(contacts, dtype=float))

    # Keep the instrument's own geometric factors under a separate token when the
    # file reported apparent resistivity, because then rhoa was formed with *those*
    # factors. k above is recomputed from the electrode geometry, and if the two
    # disagree the pair (rhoa, k) no longer describes one measurement: the forward
    # response would use one convention and the observation the other, which
    # rescales the whole section without touching chi2. Nothing here can tell which
    # is right, so both are carried and the inversion reconciles them.
    if source != "resistance":
        file_k = np.array(
            [float(o.K) if o.K is not None else np.nan for o in valid], dtype=float
        )
        if np.all(np.isfinite(file_k)) and not np.allclose(file_k, 1.0):
            data.set("k_file", file_k)

    data.markValid(data("rhoa") > 0)
    return data


# ---------------------------------------------------------------------------
# Robust single-file loader
# ---------------------------------------------------------------------------
_AUTO_NAMES = ("", "auto", "none", "auto-detect", "auto-detect (pygimli)")
#: Count-prefixed / device formats tried when native pygimli load comes back empty.
_RECOVERY_INSTRUMENTS = ("E4D", "BERT", "Syscal")
#: Picks whose reader, having refused a file, may still be answered by pygimli's
#: own loader: it reads the unified format (BERT, and E4D or ARES files written
#: in it), Res2DInv, and ASCII column tables such as Terrameter LS exports.
_NATIVE_RETRY = ("BERT", "E4D", "ARES", "ResInv", "ABEM-Lund", "Custom")
#: Picks for which the recovery sweep's count-prefixed readers are the same
#: family of layouts rather than a guess.
_SWEEP_RETRY = ("BERT", "E4D", "ARES", "Custom")


def _usable(data) -> bool:
    try:
        return data is not None and int(data.size()) > 0
    except Exception:  # noqa: BLE001
        return False


def _ensure_rhoa(data, log: LogFn = _noop) -> None:
    """Make sure a natively-loaded container has positive apparent resistivity."""
    try:
        from pygimli.physics import ert as pg_ert
        if data is None:
            return
        have_rhoa = data.haveData("rhoa") and np.any(np.asarray(data["rhoa"], dtype=float) > 0)
        if have_rhoa:
            return
        if not data.haveData("k") or not np.any(np.asarray(data["k"], dtype=float) != 0):
            data["k"] = pg_ert.createGeometricFactors(data, numerical=False)
        k = np.asarray(data["k"], dtype=float)
        if data.haveData("r"):
            data["rhoa"] = np.asarray(data["r"], dtype=float) * k
        elif data.haveData("u") and data.haveData("i"):
            data["rhoa"] = np.asarray(data["u"], dtype=float) / np.asarray(data["i"], dtype=float) * k
    except Exception as exc:  # noqa: BLE001
        log(f"Apparent-resistivity computation skipped: {exc}")


def _via_instrument(path: str, instrument: str, electrode_file: Optional[str],
                    spacing: Optional[float], log: LogFn,
                    failures: Optional[List[Exception]] = None):
    try:
        from PyHydroGeophysX.data_processing.ert_data_agent import load_ert_resipy
    except Exception as exc:  # noqa: BLE001
        log(f"ert_data_agent unavailable ({exc}).")
        return None
    try:
        with tempfile.TemporaryDirectory(prefix="phgx_resipy_") as proj:
            std = load_ert_resipy(project_dir=proj, data_file=str(path), instrument=instrument,
                                  spacing=spacing, electrode_file=electrode_file)
            return standard_to_pg(std)
    except Exception as exc:  # noqa: BLE001
        log(f"Instrument '{instrument}' loader error: {exc}")
        if failures is not None:
            failures.append(exc)
        return None


def _canonical_instrument(instrument: str) -> str:
    try:
        from PyHydroGeophysX.data_processing.ert_data_agent import _normalize_instrument_name
        return _normalize_instrument_name(instrument)
    except Exception:  # noqa: BLE001
        return instrument


def _via_native(path: str, log: LogFn):
    try:
        from pygimli.physics import ert as pg_ert
        data = pg_ert.load(str(path), verbose=False)
    except Exception as exc:  # noqa: BLE001
        log(f"pygimli native load failed: {exc}")
        return None
    _ensure_rhoa(data, log)
    return data


def load_ert_container(path: str, instrument: Optional[str] = None,
                       electrode_file: Optional[str] = None,
                       spacing: Optional[float] = None, log: LogFn = _noop):
    """Load one ERT file into a pygimli ``DataContainerERT``, robustly.

    An explicit ``instrument`` uses the device parsers (handles index-prefixed
    E4D, BERT topography, Syscal, etc.); ``None``/``"auto"`` uses pygimli's own
    reader. If the chosen path yields no measurements the loader falls back: an
    explicit instrument retries native pygimli; a still-empty result triggers a
    short sweep of count-prefixed device formats so an E4D-style file never
    loads empty. Raises ``ValueError`` if nothing parses.

    When the explicit instrument's reader refused the file outright, the retries
    run only where they read the same family of layouts (``_NATIVE_RETRY``,
    ``_SWEEP_RETRY``); otherwise that reader's error is raised. Picked as DAS-1,
    a DAS-1 file missing its data marker used to be "auto-recovered" by the
    sweep as one reading on one electrode.
    """
    path = str(path)
    inst = str(instrument).strip() if instrument else ""
    is_auto = inst.lower() in _AUTO_NAMES
    failures: List[Exception] = []
    retry_native = retry_sweep = True

    # 1. explicit instrument
    if not is_auto:
        data = _via_instrument(path, inst, electrode_file, spacing, log, failures)
        if _usable(data):
            return data
        if failures:
            picked = _canonical_instrument(inst)
            retry_native = picked in _NATIVE_RETRY
            retry_sweep = picked in _SWEEP_RETRY
            if not (retry_native or retry_sweep):
                raise ValueError(
                    f"'{Path(path).name}' could not be read as {inst}: {failures[0]}"
                ) from failures[0]
        log(f"Instrument '{inst}' parsed no usable measurements; trying pygimli auto-detect.")

    # 2. native pygimli
    if retry_native:
        data = _via_native(path, log)
        if _usable(data):
            return data

    # 3. recovery sweep across count-prefixed device formats
    for cand in (_RECOVERY_INSTRUMENTS if retry_sweep else ()):
        if cand.lower() == inst.lower():
            continue
        data = _via_instrument(path, cand, electrode_file, spacing, log, failures)
        if _usable(data):
            log(f"Auto-recovered '{Path(path).name}' using instrument='{cand}'.")
            return data

    # The first reader's error is the one worth reading: it ran on the pick, or on
    # the file's own header when a detector redirected it.
    first = f" First error: {failures[0]}" if failures else ""
    raise ValueError(
        f"No ERT measurements could be parsed from '{Path(path).name}'. "
        f"Pick the matching instrument/format in the loader.{first}")


# ---------------------------------------------------------------------------
# Measurement times from filenames
# ---------------------------------------------------------------------------
def survey_timing_for(files: Sequence[str], *, allow_header: bool = True,
                      allow_mtime: bool = False):
    """Full acquisition timing of a monitoring sequence.

    Thin re-export of :func:`PyHydroGeophysX.data_processing.survey_timing.survey_timing`
    so callers already holding this module do not need a second import. Prefer it
    over :func:`measurement_times_for` whenever the interval between surveys is
    worth reporting, which for a time-lapse survey is always.
    """
    from PyHydroGeophysX.data_processing.survey_timing import survey_timing

    return survey_timing(files, allow_header=allow_header, allow_mtime=allow_mtime)


def measurement_times_for(files: Sequence[str]) -> Tuple[List[float], List[str]]:
    """Derive numeric measurement times + display labels from filenames.

    When every filename embeds a distinct timestamp, times are elapsed days from
    the earliest acquisition and labels are the dates; otherwise it falls back to
    a sequential ``1..n`` with index labels. :func:`survey_timing_for` returns the
    same times plus the absolute timestamps and the intervals between them.
    """
    timing = survey_timing_for(files)
    return list(timing.times), list(timing.labels)


def save_edited_ert_container(
    data: Any,
    destination: str | Path,
    electrodes: Sequence[dict],
) -> str:
    """Persist QC-filtered ERT data with the current electrode edits applied.

    ``original_index`` is zero-based for retained sensors and ``None`` for a
    newly added sensor. Deleting an original sensor removes measurements that
    reference it through PyGIMLi's own ``removeSensorIdx`` implementation.
    Retained sensors must preserve their original relative order; the current
    Qt editor moves/adds/deletes but does not expose arbitrary reordering.
    """
    import pygimli as pg

    edited = pg.DataContainerERT(data)
    original_count = int(edited.sensorCount())
    retained = [
        int(item["original_index"])
        for item in electrodes
        if item.get("original_index") is not None
    ]
    if retained != sorted(retained) or len(retained) != len(set(retained)):
        raise ValueError(
            "Edited ERT electrode order is ambiguous; retained original indices "
            "must be unique and remain in their original order."
        )
    if any(index < 0 or index >= original_count for index in retained):
        raise ValueError("Edited ERT electrode metadata contains an invalid original index.")

    for index in reversed([i for i in range(original_count) if i not in set(retained)]):
        edited.removeSensorIdx(index)

    retained_rows = [
        item for item in electrodes if item.get("original_index") is not None
    ]
    for current_index, item in enumerate(retained_rows):
        edited.setSensorPosition(
            current_index,
            pg.Pos(float(item["x"]), float(item["z"])),
        )
    for item in electrodes:
        if item.get("original_index") is None:
            edited.createSensor(pg.Pos(float(item["x"]), float(item["z"])))

    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    edited.save(str(target))
    return str(target)


# ---------------------------------------------------------------------------
# Normalize a sequence into clean pygimli files for the core inversion
# ---------------------------------------------------------------------------
def align_timelapse_abmn(containers, log: LogFn = _noop):
    """Align the ABMN union for ADTLERT; missing rows get relative error 1.0.

    Electrode numbering must describe the same positions. Exact ABMN tuples
    are matched, without merging reciprocal or reversed-polarity measurements.
    Missing rhoa uses the median of available values for that tuple. A 100%
    error downweights these placeholders; it does not give them zero weight.
    """
    import pygimli as pg

    if not containers:
        raise ValueError("Time-lapse alignment needs at least one ERT dataset.")
    reference = np.asarray(containers[0].sensorPositions(), dtype=float)
    layouts, lookup, samples = [], {}, {}
    for index, data in enumerate(containers):
        sensors = np.asarray(data.sensorPositions(), dtype=float)
        if sensors.shape != reference.shape or not np.allclose(
            sensors, reference, rtol=0.0, atol=1.e-8
        ):
            raise ValueError(
                f"ADTLERT requires identical electrode positions; step {index} differs."
            )
        rows = [tuple(row) for row in np.column_stack([
            np.asarray(data[key], dtype=np.int64) for key in ("a", "b", "m", "n")
        ])]
        if len(set(rows)) != len(rows):
            raise ValueError(f"Duplicate ABMN measurements in step {index}; alignment is ambiguous.")
        layouts.append({key: row for row, key in enumerate(rows)})
        for row, key in enumerate(rows):
            lookup.setdefault(key, (index, row))
            value = float(data["rhoa"][row])
            if np.isfinite(value) and value > 0:
                samples.setdefault(key, []).append(value)
    keys = list(lookup)
    if any(key not in samples for key in keys):
        raise ValueError("Cannot fill ABMN without any positive finite apparent resistivity.")
    fields = [{str(token): np.asarray(data[str(token)]).copy()
               for token in data.dataMap().keys()} for data in containers]
    tokens = set().union(*(field.keys() for field in fields))
    aligned = []
    for index, data in enumerate(containers):
        result = pg.DataContainerERT(data)
        result.resize(len(keys))
        # Copy every field using the same row mapping, including auxiliary data.
        for token in tokens:
            values = []
            for key in keys:
                src_index, row = (index, layouts[index][key]) if key in layouts[index] else lookup[key]
                source = fields[src_index]
                values.append(float(source[token][row]) if token in source else 0.0)
            result[token] = values
        missing = np.asarray([key not in layouts[index] for key in keys])
        rhoa = np.asarray(result["rhoa"]).copy()
        errors = np.asarray(result["err"]).copy()
        for row in np.flatnonzero(missing):
            rhoa[row] = np.median(samples[keys[row]])
        errors[missing] = 1.0
        result["rhoa"], result["err"] = rhoa, errors
        result["valid"] = np.ones(len(keys))
        if any(source.haveData("r") for source in containers):
            resistance = np.asarray(result["r"]).copy()
            factors = np.asarray(result["k"])
            safe = missing & (factors != 0)
            resistance[safe] = rhoa[safe] / factors[safe]
            result["r"] = resistance
        aligned.append(result)
        log(f"ADTLERT alignment step {index}: {len(keys)} ABMN rows, "
            f"{int(missing.sum())} filled with 100% relative error")
    return aligned


def normalize_for_timelapse(files: Sequence[str], instrument: Optional[str],
                            out_dir: str, log: LogFn = _noop,
                            max_error: Optional[float] = None,
                            engine: str = "pyhydro"):
    """Write native files, filtering each survey independently.

    PyHydro keeps each survey's own measurement count and ordering. ADTLERT
    aligns the union of surviving ABMN rows and fills missing rows at 100% error.
    """
    if engine not in ("pyhydro", "adtlert"):
        raise ValueError(f"Unsupported time-lapse engine: {engine}")
    base = Path(out_dir) / "qt_ert_timelapse" / "normalized"
    base.mkdir(parents=True, exist_ok=True)
    containers = [load_ert_container(f, instrument=instrument, log=log) for f in files]
    if not containers:
        raise ValueError("Time-lapse normalization needs at least one ERT file.")
    for index, data in enumerate(containers):
        rhoa = np.asarray(data["rhoa"], dtype=float)
        quality = np.isfinite(rhoa) & (rhoa > 0.0)
        if data.haveData("valid"):
            quality &= np.asarray(data["valid"], dtype=float) > 0.0
        if data.haveData("err"):
            errors = np.asarray(data["err"], dtype=float)
            quality &= np.isfinite(errors) & (errors > 0.0)
            if max_error is not None:
                quality &= errors <= float(max_error)
        if int(quality.sum()) < 4:
            raise ValueError(f"Need at least four valid measurements in time-lapse step {index}.")
        if not quality.all():
            data.remove(~quality)
        log(f"Time-lapse quality step {index}: {int(quality.sum())}/{len(quality)} retained")
    if engine == "adtlert":
        containers = align_timelapse_abmn(containers, log=log)
    for stale in base.glob("step_*.dat"):
        stale.unlink()
    basenames: List[str] = []
    for i, (f, data) in enumerate(zip(files, containers)):
        name = f"step_{i:03d}.dat"
        data.save(str(base / name))
        basenames.append(name)
        log(f"Prepared {i + 1}/{len(files)}: {Path(f).name} -> "
            f"{int(data.size())} data, {int(data.sensorCount())} electrodes")
    return str(base), basenames, containers
