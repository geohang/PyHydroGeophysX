"""Robust ERT file loading for the desktop workbench (Qt-free).

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

import datetime as _dt
import re
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
                    spacing: Optional[float], log: LogFn):
    try:
        from PyHydroGeophysX.data_processing.ert_data_agent import load_ert_resipy
    except Exception as exc:  # noqa: BLE001
        log(f"ert_data_agent unavailable ({exc}).")
        return None
    proj = tempfile.mkdtemp(prefix="phgx_resipy_")
    try:
        std = load_ert_resipy(project_dir=proj, data_file=str(path), instrument=instrument,
                              spacing=spacing, electrode_file=electrode_file)
    except Exception as exc:  # noqa: BLE001
        log(f"Instrument '{instrument}' loader error: {exc}")
        return None
    return standard_to_pg(std)


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
    """
    path = str(path)
    inst = str(instrument).strip() if instrument else ""
    is_auto = inst.lower() in _AUTO_NAMES

    # 1. explicit instrument
    if not is_auto:
        data = _via_instrument(path, inst, electrode_file, spacing, log)
        if _usable(data):
            return data
        log(f"Instrument '{inst}' parsed no usable measurements; trying pygimli auto-detect.")

    # 2. native pygimli
    data = _via_native(path, log)
    if _usable(data):
        return data

    # 3. recovery sweep across count-prefixed device formats
    for cand in _RECOVERY_INSTRUMENTS:
        if cand.lower() == inst.lower():
            continue
        data = _via_instrument(path, cand, electrode_file, spacing, log)
        if _usable(data):
            log(f"Auto-recovered '{Path(path).name}' using instrument='{cand}'.")
            return data

    raise ValueError(
        f"No ERT measurements could be parsed from '{Path(path).name}'. "
        f"Pick the matching instrument/format in the loader.")


# ---------------------------------------------------------------------------
# Measurement times from filenames
# ---------------------------------------------------------------------------
_DATE_PATTERNS: List[Tuple[re.Pattern, bool]] = [
    (re.compile(r"(\d{4})[-_]?(\d{2})[-_]?(\d{2})[-_ ]?(\d{2})(\d{2})"), True),   # ...YYYY-MM-DD_HHMM
    (re.compile(r"(\d{4})[-_]?(\d{2})[-_]?(\d{2})"), False),                      # ...YYYY-MM-DD
]


def _parse_date(stem: str) -> Optional[_dt.datetime]:
    for pattern, has_time in _DATE_PATTERNS:
        m = pattern.search(stem)
        if not m:
            continue
        try:
            if has_time:
                return _dt.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                                    int(m.group(4)), int(m.group(5)))
            return _dt.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            continue
    return None


def measurement_times_for(files: Sequence[str]) -> Tuple[List[float], List[str]]:
    """Derive numeric measurement times + display labels from filenames.

    When every filename embeds a distinct date, times are elapsed days from the
    first acquisition and labels are ``YYYY-MM-DD``. Otherwise falls back to a
    sequential ``1..n`` with index labels.
    """
    dates = [_parse_date(Path(f).stem) for f in files]
    if files and all(d is not None for d in dates):
        t0 = min(dates)
        times = [round((d - t0).total_seconds() / 86400.0, 4) for d in dates]
        labels = [d.strftime("%Y-%m-%d") for d in dates]
        if len(set(times)) == len(times):  # distinct -> usable as a time axis
            return times, labels
    n = len(files)
    return [float(i + 1) for i in range(n)], [str(i + 1) for i in range(n)]


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
def normalize_for_timelapse(files: Sequence[str], instrument: Optional[str],
                            out_dir: str, log: LogFn = _noop):
    """Load each file robustly and write a clean pygimli ``.dat`` into one folder.

    The core :class:`TimeLapseERTInversion` reloads files with ``ert.load``; the
    normalized files are written in pygimli's native unified format (proper token
    headers, geometric factors, ``rhoa = R*k``, topography in ``z``) so that
    reload is correct. Returns ``(clean_dir, basenames, containers)``; all clean
    files share one folder so the windowed inversion (which takes a directory +
    filenames) works directly.
    """
    base = Path(out_dir) / "qt_ert_timelapse" / "normalized"
    base.mkdir(parents=True, exist_ok=True)
    for stale in base.glob("step_*.dat"):
        try:
            stale.unlink()
        except OSError:
            pass
    basenames: List[str] = []
    containers: List[Any] = []
    for i, f in enumerate(files):
        data = load_ert_container(f, instrument=instrument, log=log)
        name = f"step_{i:03d}.dat"
        data.save(str(base / name))
        basenames.append(name)
        containers.append(data)
        log(f"Prepared {i + 1}/{len(files)}: {Path(f).name} -> "
            f"{int(data.size())} data, {int(data.sensorCount())} electrodes")
    return str(base), basenames, containers
