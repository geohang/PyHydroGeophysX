"""Independently written ERT instrument-format readers and reciprocal analysis.

Every routine here was written from the file format itself and from the standard
DC-resistivity literature, not from any other implementation. The DAS-1 reader
works off the column map the format writes into its own header, so it reads the
layout out of each file rather than assuming one; the sample acquisitions under
``examples/data/ERT/DAS/`` were the reference while writing it. The reciprocal
analysis implements the definition given in the induced-polarisation literature
(Slater et al. 2000; Binley and Kemna 2005; Binley and Slater 2020, ch. 6).

The return contract matches the other readers in
:mod:`PyHydroGeophysX.data_processing.ert_data_agent`: ``(elec, df)`` where
``elec`` is an ``(n_electrodes, 3)`` array of x, y, z and ``df`` carries 1-based
``a``, ``b``, ``m``, ``n`` electrode indices alongside the measured columns.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

__all__ = ["parse_das1", "parse_res2dinv_general", "parse_sting", "parse_tx0",
           "reciprocal_errors"]


# The DAS-1 header writes its own layout as ``#key= value`` directives, so the
# column positions below are only the fallback for a file that omits them. They
# are the positions the column-name comment above the data block describes.
_DAS1_DEFAULT_COLUMNS: Dict[str, int] = {
    "elec_cable_col": 1, "elec_id_col": 2,
    "elec_x_col": 3, "elec_y_col": 4, "elec_z_col": 5,
    "data_id_col": 1,
    "data_a_cable_col": 2, "data_a_elec_col": 3,
    "data_b_cable_col": 4, "data_b_elec_col": 5,
    "data_m_cable_col": 6, "data_m_elec_col": 7,
    "data_n_cable_col": 8, "data_n_elec_col": 9,
    "data_res_col": 10, "data_std_res_col": 11,
    "data_amp_col": 12,
    "data_ip_wind_col": 14, "data_std_ip_col": 15,
    "data_i_curr_col": 18, "data_contact_r_col": 19,
}

#: Coordinates the instrument writes for an electrode that is not on the line.
#: A remote (infinity) electrode is recorded as a large sentinel or as a
#: non-finite value; either way it has no real position and must not be treated
#: as one when the survey extent is computed.
_REMOTE_SENTINELS = (-9999999.0, -999999.0, -99999.0, -9999.0, -999.0,
                     9999.0, 99999.0, 999999.0, 9999999.0)

_DIRECTIVE = re.compile(r"^\s*#\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(\S+)")


def _das1_tokens(line: str) -> List[str]:
    """Split a DAS-1 record into the fields its column map counts.

    An electrode is written as ``cable,number`` in one whitespace-delimited
    field, and the header's column numbers count the two halves separately, so
    the comma is a field separator exactly like a space.
    """
    return [tok for tok in re.split(r"[\s,]+", line.strip()) if tok]


def _pick(tokens: List[str], column: Optional[int]) -> Optional[str]:
    """One 1-based column from a token list, or None when it is absent."""
    if column is None or column < 1 or column > len(tokens):
        return None
    return tokens[column - 1]


def _as_float(text: Optional[str]) -> float:
    """One numeric field as a float, NaN when the field is not numeric.

    Shared by the readers in this module because every instrument here writes
    numbers ``float`` already accepts -- DAS-1 writes an explicit sign and a
    leading decimal point (``+.0018633``), SuperSting writes padded scientific
    notation (`` 1.88352E-01``) -- and every one of them writes a text message
    in place of the whole numeric run when a reading failed.
    """
    if text is None:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def _columns_from_mode(lines: List[str]) -> Dict[str, int]:
    """Column map inferred from the acquisition-mode flags in the header.

    Used only when the file omits its ``#..._col`` directives. The data record
    grows with what the acquisition recorded, so the position of every field
    after the quadrupole depends on the mode:

    * ``#SAprs`` adds an apparent-resistivity column ahead of the resistance;
    * each ``#TW`` time window adds an IP value and its standard deviation;
    * ``#SCltSP`` adds a self-potential column.

    The four electrode fields sit at fixed positions, so only the tail moves.
    """
    apparent = any(line.lstrip().startswith("#SAprs") for line in lines)
    windows = sum(1 for line in lines if line.lstrip().startswith("#TW"))
    self_potential = 1 if any(line.lstrip().startswith("#SCltSP") for line in lines) else 0

    columns = {key: value for key, value in _DAS1_DEFAULT_COLUMNS.items()
               if not key.startswith("data_") or key.endswith(("_cable_col", "_elec_col"))}
    columns["data_id_col"] = 1
    base = 10 + (1 if apparent else 0)           # 1-based: after a,b,m,n
    columns["data_res_col"] = base
    columns["data_std_res_col"] = base + 1
    columns["data_amp_col"] = base + 2
    tail = base + 4 + 2 * windows + self_potential
    columns["data_i_curr_col"] = tail
    columns["data_contact_r_col"] = tail + 1
    if windows:
        columns["data_ip_wind_col"] = base + 4
        columns["data_std_ip_col"] = base + 5
    return columns


def _read_directives(lines: List[str]) -> Dict[str, int]:
    """Column map for a DAS-1 file, from its directives where it writes them.

    The format can write its own layout as ``#elec_*_col`` and ``#data_*_col``
    directives, which is the reliable route because it removes the guesswork
    entirely. The directives sit between the electrode block and the data block
    in the files seen here, so the whole file is scanned rather than its head.
    A column of -1 marks a field the acquisition did not record.

    Where a file omits them the layout has to be inferred from the mode flags
    instead, which is what :func:`_columns_from_mode` does.
    """
    found: Dict[str, int] = {}
    for line in lines:
        match = _DIRECTIVE.match(line)
        if match is None:
            continue
        key, value = match.group(1), match.group(2)
        if not key.endswith("_col"):
            continue
        try:
            index = int(value)
        except ValueError:
            continue
        found[key] = index if index > 0 else -1

    columns = _columns_from_mode(lines)
    columns.update(found)                        # a written layout always wins
    return columns


def _block(lines: List[str], start: str, end: str) -> List[str]:
    """The records between two DAS-1 section markers, comments dropped.

    ``!`` opens a comment in this format, and the column-name legend above each
    block is written as one.
    """
    try:
        first = next(i for i, line in enumerate(lines) if line.strip().startswith(start))
        last = next(i for i, line in enumerate(lines) if line.strip().startswith(end))
    except StopIteration:
        raise ValueError(f"DAS-1 file has no {start} ... {end} section.")
    if last <= first:
        raise ValueError(f"DAS-1 section {start} closes before it opens.")
    return [line for line in lines[first + 1:last]
            if line.strip() and not line.lstrip().startswith("!")]


def _electrode_table(rows: List[str], columns: Dict[str, int]):
    """Electrode coordinates and the ``(cable, number) -> index`` lookup.

    The lookup is what the data block needs: it addresses electrodes by cable
    and position on that cable, not by a global number.
    """
    positions: List[Tuple[float, float, float]] = []
    lookup: Dict[Tuple[str, str], int] = {}
    for row in rows:
        tokens = _das1_tokens(row)
        cable = _pick(tokens, columns.get("elec_cable_col"))
        number = _pick(tokens, columns.get("elec_id_col"))
        if cable is None or number is None:
            continue
        x = _as_float(_pick(tokens, columns.get("elec_x_col")))
        y = _as_float(_pick(tokens, columns.get("elec_y_col")))
        z = _as_float(_pick(tokens, columns.get("elec_z_col")))
        key = (cable.lstrip("0") or "0", number.lstrip("0") or "0")
        if key in lookup:
            continue
        lookup[key] = len(positions) + 1        # 1-based, as the readers agree
        positions.append((x, y, z))
    if not positions:
        raise ValueError("DAS-1 file has no readable electrode rows.")
    return np.asarray(positions, dtype=float), lookup


def _mark_remote(elec: np.ndarray) -> np.ndarray:
    """True where an electrode carries a sentinel or non-finite coordinate."""
    remote = ~np.isfinite(elec).all(axis=1)
    for sentinel in _REMOTE_SENTINELS:
        remote |= np.isclose(elec[:, 0], sentinel)
    return remote


def parse_das1(path: str | Path) -> Tuple[np.ndarray, pd.DataFrame]:
    """Read a DAS-1 ``.Data`` acquisition into electrodes and measurements.

    The format is self-describing: it writes ``#elec_*_col`` and ``#data_*_col``
    directives that give the column of every field, so the layout is read from
    each file instead of assumed. Records whose reading failed carry a text
    message where the numbers belong (a contact-resistance or compliance
    failure, which is the majority of rows in a survey run against dry ground);
    those rows parse to NaN and are dropped.

    Returns ``(elec, df)``. ``elec`` is ``(n_electrodes, 3)`` in the file's own
    coordinates. ``df`` carries 1-based ``a``, ``b``, ``m``, ``n``, the transfer
    resistance ``resist`` in ohm, its standard deviation ``dev`` in ohm, the same
    quantity as a fraction of the reading in ``error``, and, where the
    acquisition recorded them, ``ip`` in mV/V, the injected current ``current``
    in mA, and the transmitter ``contact_r`` in ohm.

    Both forms of the stacking error are reported on purpose. ``dev`` is what the
    file holds; ``error`` is ``dev / |resist|``. A consumer handed only the
    absolute form has to guess its units from the magnitudes, and that guess
    fails on this instrument: its stacking standard deviations run to a few
    tenths of an ohm, which a magnitude test reads as a fraction and so as a
    several-percent error, where the true relative error is under 0.1 %. The
    units are known here, so the division belongs here.
    """
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    columns = _read_directives(lines)

    elec, lookup = _electrode_table(_block(lines, "#elec_start", "#elec_end"), columns)

    records: List[Dict[str, Any]] = []
    for row in _block(lines, "#data_start", "#data_end"):
        tokens = _das1_tokens(row)
        quad: List[int] = []
        for role in ("a", "b", "m", "n"):
            cable = _pick(tokens, columns.get(f"data_{role}_cable_col"))
            number = _pick(tokens, columns.get(f"data_{role}_elec_col"))
            if cable is None or number is None:
                break
            index = lookup.get((cable.lstrip("0") or "0", number.lstrip("0") or "0"))
            if index is None:
                break
            quad.append(index)
        if len(quad) != 4:
            continue
        resist = _as_float(_pick(tokens, columns.get("data_res_col")))
        if not np.isfinite(resist):
            continue                            # the reading failed; no value here
        record = {
            "a": quad[0], "b": quad[1], "m": quad[2], "n": quad[3],
            "resist": resist,
            "dev": _as_float(_pick(tokens, columns.get("data_std_res_col"))),
        }
        for name, key in (("ip", "data_ip_wind_col"),
                          ("current", "data_i_curr_col"),
                          ("contact_r", "data_contact_r_col")):
            value = _as_float(_pick(tokens, columns.get(key)))
            if np.isfinite(value):
                record[name] = value
        records.append(record)

    if not records:
        raise ValueError(
            f"{Path(path).name}: every DAS-1 record failed its reading, so the "
            "file holds no usable measurement.")

    df = pd.DataFrame.from_records(records)
    if "ip" not in df.columns:
        df["ip"] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        relative = df["dev"].abs() / df["resist"].abs()
    df["error"] = relative.where(np.isfinite(relative))
    df.attrs["remote_electrodes"] = _mark_remote(elec)
    return elec, df


_TX0_ELECTRODE = re.compile(
    r"^\*\s*Electrode\s*\[\s*(\d+)\s*\]\s*x\s*y\s*z\s*\(m\)\s*=\s*"
    r"([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)")

#: Column names in the ``* num`` legend, mapped to what this package calls them.
_TX0_COLUMNS = {
    "a": "a", "b": "b", "m": "m", "n": "n",
    "rho": "rhoa",          # apparent resistivity, ohm m
    "phi": "ip",            # phase, mrad
    "i": "current",         # mA
    "u": "voltage",         # mV
    "du": "error_percent",  # relative error on U, already a percentage
}


def parse_tx0(path: str | Path) -> Tuple[np.ndarray, pd.DataFrame]:
    """Read a ``.tx0`` acquisition (Lippmann 4-Point Light / GeoTest export).

    Two things make this format pleasant to read, and both are used here rather
    than assumed away. It writes an explicit electrode table, one ``* Electrode
    [n] x y z (m) =`` line per electrode, so the geometry needs no separate file
    and survives an irregular layout. And it names its data columns in a
    ``* num A B M N ...`` legend, so the columns are located by name instead of
    by position, which is what keeps the reader working when an acquisition
    records a different set of them.

    The quadrupole is given as electrode numbers, so it maps onto the electrode
    table directly. ``dU`` is the relative error on the measured voltage and is
    already a percentage; it is converted to a fraction and returned as
    ``error``, because the units are known here and a consumer would otherwise
    have to guess them from the magnitudes.

    Returns ``(elec, df)`` with 1-based ``a``, ``b``, ``m``, ``n``, apparent
    resistivity ``rhoa``, phase ``ip`` in mrad, and ``error`` as a fraction.
    """
    lines = Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()

    numbered: Dict[int, Tuple[float, float, float]] = {}
    for line in lines:
        found = _TX0_ELECTRODE.match(line)
        if found:
            numbered[int(found.group(1))] = (
                float(found.group(2)), float(found.group(3)), float(found.group(4)))
    if not numbered:
        raise ValueError(
            f"{Path(path).name}: no '* Electrode [n] x y z' block, so the "
            "electrode geometry cannot be recovered from this file alone.")
    order = sorted(numbered)
    elec = np.asarray([numbered[k] for k in order], dtype=float)
    index = {number: position for position, number in enumerate(order, start=1)}

    try:
        legend = next(i for i, line in enumerate(lines)
                      if line.lstrip().startswith("* num"))
    except StopIteration:
        raise ValueError(f"{Path(path).name}: no '* num' column legend.")
    names = [tok.lower() for tok in lines[legend].lstrip("* ").split()]
    # First occurrence wins. The legend reuses "n": once for the N electrode and
    # again, ten columns later, for the stack count. The quadrupole is written
    # first, so keeping the earlier column is what reads N rather than a count of
    # stacks, which is otherwise a silent and very plausible-looking error.
    wanted: Dict[int, str] = {}
    claimed: set = set()
    for position, name in enumerate(names):
        key = _TX0_COLUMNS.get(name)
        if key is None or key in claimed:
            continue
        wanted[position] = key
        claimed.add(key)
    if not {"a", "b", "m", "n"} <= set(wanted.values()):
        raise ValueError(f"{Path(path).name}: the column legend names no quadrupole.")

    records: List[Dict[str, Any]] = []
    for line in lines[legend + 1:]:
        stripped = line.strip()
        # The units row and the acquisition's own warnings both start with '*'.
        if not stripped or stripped.startswith("*"):
            continue
        tokens = stripped.split()
        if len(tokens) <= max(wanted):
            continue
        row: Dict[str, Any] = {}
        for position, key in wanted.items():
            row[key] = _as_float(tokens[position])
        quad = [index.get(int(row[role])) for role in ("a", "b", "m", "n")
                if np.isfinite(row.get(role, float("nan")))]
        if len(quad) != 4 or any(q is None for q in quad):
            continue
        row.update(zip(("a", "b", "m", "n"), quad))
        records.append(row)

    if not records:
        raise ValueError(f"{Path(path).name}: the data block holds no readable rows.")

    df = pd.DataFrame.from_records(records)
    if "error_percent" in df.columns:
        df["error"] = df.pop("error_percent") / 100.0
    if "ip" not in df.columns:
        df["ip"] = np.nan
    return elec, df


def parse_res2dinv_general(path: str | Path) -> Tuple[np.ndarray, pd.DataFrame]:
    """Read a Res2DInv general-array ``.dat`` file.

    The general array format is positional: a short fixed header, then one
    record per measurement holding the number of electrodes in the array, the
    (x, z) of each of the four in turn, and the apparent resistivity.

    The format gives electrode *positions* rather than indices, so the electrode
    table has to be recovered from the measurements: every distinct x that
    appears in any role is an electrode, and sorting them left to right gives
    the numbering the rest of the package expects. That works because a surface
    line has one electrode per position; it would not survive a borehole layout,
    where two electrodes share an x, which is why z is checked and the file is
    rejected rather than silently flattened when it is not a surface survey.
    """
    lines = [line.strip() for line in
             Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()]
    if len(lines) < 10:
        raise ValueError("Res2DInv general-array file is too short to hold a header.")
    try:
        spacing = float(lines[1])
        n_measurements = int(float(lines[6]))
    except (ValueError, IndexError) as error:
        raise ValueError(f"Res2DInv header is not in the general-array layout: {error}")
    if n_measurements <= 0:
        raise ValueError("Res2DInv header reports no measurements.")

    records = []
    for line in lines[9:]:
        if not line:
            continue
        parts = re.split(r"[\s,]+", line)
        if len(parts) < 10:
            continue
        try:
            values = [float(p) for p in parts[:10]]
        except ValueError:
            continue                              # a trailing topography block
        records.append(values)
        if len(records) >= n_measurements:
            break
    if not records:
        raise ValueError("Res2DInv file holds no readable measurement rows.")

    table = np.asarray(records, dtype=float)
    xs = table[:, [1, 3, 5, 7]]                   # x of A, B, M, N
    zs = table[:, [2, 4, 6, 8]]                   # z of the same four
    rhoa = table[:, 9]
    if not np.allclose(zs, zs.flat[0]):
        raise ValueError(
            "This Res2DInv file places electrodes at more than one elevation, so "
            "position alone does not identify them. Read it with ResIPy instead.")

    positions = np.unique(xs)
    index = {value: number for number, value in enumerate(positions, start=1)}
    elec = np.column_stack([positions,
                            np.zeros_like(positions),
                            np.full_like(positions, float(zs.flat[0]))])
    df = pd.DataFrame({
        "a": [index[v] for v in xs[:, 0]],
        "b": [index[v] for v in xs[:, 1]],
        "m": [index[v] for v in xs[:, 2]],
        "n": [index[v] for v in xs[:, 3]],
        "rhoa": rhoa,
        "ip": np.nan,
    })
    df.attrs["electrode_spacing"] = spacing
    return elec, df


# A ``.stg`` record writes its measurement before its geometry, always in this
# order and always comma separated: record number, measurement mode, date, time,
# transfer resistance, stacking error, injected current, apparent resistivity,
# command-file name, and then the four electrodes as x/y/z triplets in A, B, M,
# N order. The instrument writes no header row above the data block, so these
# positions are the format itself rather than an assumption about one.
_STG_COLUMNS = {
    "resist": 4,          # V/I, ohm
    "error_percent": 5,   # spread across the measurement cycles, %
    "current": 6,         # injected current, mA
    "rhoa": 7,            # apparent resistivity, in the file's own length unit
    "command": 8,         # the command file, which is the operator's line name
}
_STG_FIRST_COORD = 9
_STG_N_FIELDS = _STG_FIRST_COORD + 12   # the shortest record the format writes

#: The header declares the length unit. Coordinates, geometric factors and
#: apparent resistivities are all written in it, so it is recorded rather than
#: converted: a file in feet is internally consistent in feet.
_STG_UNIT = re.compile(r"^\s*Unit\s*:\s*(\S+)", re.IGNORECASE)
_STG_RECORDS = re.compile(r"Records\s*:\s*(\d+)", re.IGNORECASE)
_STG_SETTING = re.compile(r"^([A-Za-z][A-Za-z0-9_]*)=(.*)$")


def _stg_record(fields: List[str]) -> Optional[Dict[str, Any]]:
    """One ``.stg`` data line as a record, or None when the line is not one.

    The header carries no comment marker, so "is this a measurement?" has to be
    answered from the line itself: a record starts with an integer and holds
    twelve readable coordinates after the command-file name. Header lines, blank
    lines and a truncated final record all fail that test.
    """
    if len(fields) < _STG_N_FIELDS:
        return None
    try:
        record_no = int(fields[0])
    except ValueError:
        return None

    coords = np.array([_as_float(f) for f in
                       fields[_STG_FIRST_COORD:_STG_FIRST_COORD + 12]], dtype=float)
    if not np.isfinite(coords).all():
        return None

    resist = _as_float(fields[_STG_COLUMNS["resist"]])
    if not np.isfinite(resist):
        return None                    # the cycle failed; there is no reading here

    record: Dict[str, Any] = {
        "record": record_no,
        "command": fields[_STG_COLUMNS["command"]].strip(),
        "resist": resist,
        "error_percent": _as_float(fields[_STG_COLUMNS["error_percent"]]),
        "current": _as_float(fields[_STG_COLUMNS["current"]]),
        "rhoa": _as_float(fields[_STG_COLUMNS["rhoa"]]),
        "coords": coords.reshape(4, 3),
    }

    # Whatever follows the geometry is annotation. An IP acquisition writes its
    # decay windows there as bare numbers; the firmware writes the acquisition
    # settings as ``key=value``. Neither sits at a fixed column, so they are
    # told apart by shape instead of by position.
    windows: List[float] = []
    settings: Dict[str, str] = {}
    for field in fields[_STG_FIRST_COORD + 12:]:
        field = field.strip()
        if not field:
            continue
        setting = _STG_SETTING.match(field)
        if setting:
            settings[setting.group(1)] = setting.group(2).strip()
            continue
        value = _as_float(field)
        if np.isfinite(value):
            windows.append(value)
    record["ip_windows"] = windows
    record["settings"] = settings
    return record


def _stg_electrodes(coords: np.ndarray) -> Tuple[np.ndarray, Dict[tuple, int]]:
    """The electrode table implied by every position the records mention.

    A ``.stg`` file addresses electrodes by position and never writes an
    electrode table, so the table has to be recovered from the measurements.
    Positions are rounded to a micrometre first, because they reach this point
    as text and AGI does not spell them one way: the raw export writes
    ``6.18000E+02`` where a rescaled one writes ``618.0000``. A difference far
    below any survey's precision must not become a second electrode.

    Numbering runs along the axis the array is longest in, so a straight line
    gets its along-line order whichever compass direction it was laid out in,
    and a genuinely three-dimensional array still gets a stable, reproducible
    one.
    """
    unique = np.unique(np.round(coords.reshape(-1, 3), 6), axis=0)
    extent = unique.max(axis=0) - unique.min(axis=0)
    primary, secondary, tertiary = np.argsort(extent)[::-1]
    order = np.lexsort((unique[:, tertiary], unique[:, secondary], unique[:, primary]))
    elec = unique[order]
    lookup = {tuple(row): i + 1 for i, row in enumerate(elec)}  # 1-based, as the readers agree
    return elec, lookup


def parse_sting(path: str | Path) -> Tuple[np.ndarray, pd.DataFrame]:
    """Read an AGI SuperSting / Sting R1 ``.stg`` export.

    The format is a short header followed by one comma-separated record per
    measurement. Two of its properties shape this reader:

    * Electrodes are identified by their coordinates, not by a number, and the
      file holds no electrode table. The table returned here is therefore built
      from the positions the records use, and ``a``/``b``/``m``/``n`` are 1-based
      indices into it. An electrode that was laid out but never energised or read
      does not appear in it, because the file never mentions it.
    * The instrument writes both the transfer resistance and the apparent
      resistivity it derived from it, so their ratio is the geometric factor the
      instrument itself used. It is carried in ``k`` rather than recomputed,
      which lets a caller see whether the file's own geometry agrees with the
      electrode positions it wrote.

    Column 6 is the stacking error in percent -- the spread across the
    measurement cycles, not a reciprocal error -- and is reported both as written
    (``error_percent``) and as the fraction (``error``) the rest of this package
    expects. Negative resistances are kept: a reversed or noisy reading is a
    quality-control decision for the caller, not a parse failure.

    Lengths come back in whatever unit the header declares, which is reported in
    ``df.attrs["unit"]``. A survey recorded in feet is returned in feet, with
    apparent resistivity in ohm-feet to match, because converting the coordinates
    without also rescaling the resistivities would leave the two disagreeing.

    Returns ``(elec, df)``: an ``(n_electrodes, 3)`` array of x, y, z, and a
    frame carrying ``a``, ``b``, ``m``, ``n``, ``resist``, ``rhoa``, ``k``,
    ``error``, ``error_percent``, ``current`` and ``ip``.
    """
    lines = Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()

    unit = "meter"
    declared: Optional[int] = None
    for line in lines[:8]:
        found_unit = _STG_UNIT.search(line)
        if found_unit:
            unit = found_unit.group(1).strip().lower()
        found_records = _STG_RECORDS.search(line)
        if found_records and declared is None:
            declared = int(found_records.group(1))

    records: List[Dict[str, Any]] = []
    skipped = 0
    for line in lines:
        if not line.strip():
            continue
        record = _stg_record(line.split(","))
        if record is None:
            skipped += 1
            continue
        records.append(record)

    if not records:
        raise ValueError(
            f"{Path(path).name}: no SuperSting records could be read. A .stg record "
            "is a comma-separated line that starts with a record number and carries "
            "four x/y/z electrode positions.")

    coords = np.round(np.array([r["coords"] for r in records], dtype=float), 6)
    elec, lookup = _stg_electrodes(coords)
    quads = np.array([[lookup[tuple(row[role])] for role in range(4)] for row in coords],
                     dtype=int)

    resist = np.array([r["resist"] for r in records], dtype=float)
    rhoa = np.array([r["rhoa"] for r in records], dtype=float)
    error_percent = np.array([r["error_percent"] for r in records], dtype=float)

    # The instrument's own geometric factor, recovered from the two numbers it
    # wrote. A reading of exactly zero resistance carries no factor, so it stays
    # NaN rather than becoming an infinity the inversion would have to filter.
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.where(np.abs(resist) > 0, rhoa / resist, np.nan)

    df = pd.DataFrame({
        "a": quads[:, 0], "b": quads[:, 1], "m": quads[:, 2], "n": quads[:, 3],
        "resist": resist,
        "rhoa": rhoa,
        "k": k,
        "error": error_percent / 100.0,
        "error_percent": error_percent,
        "current": np.array([r["current"] for r in records], dtype=float),
        "record": np.array([r["record"] for r in records], dtype=int),
        "command": [r["command"] for r in records],
    })

    # An IP acquisition writes one column per decay window and a DC acquisition
    # writes none, but ``ip`` has to exist either way because the callers read it.
    n_windows = max((len(r["ip_windows"]) for r in records), default=0)
    for window in range(n_windows):
        df[f"ip_{window + 1}"] = [
            r["ip_windows"][window] if window < len(r["ip_windows"]) else np.nan
            for r in records]
    df["ip"] = df["ip_1"] if n_windows else np.nan

    df.attrs["remote_electrodes"] = _mark_remote(elec)
    df.attrs["unit"] = unit
    df.attrs["records_declared"] = declared
    df.attrs["lines_skipped"] = skipped
    df.attrs["acquisition_settings"] = records[0]["settings"]
    return elec, df


def _reciprocal_key(frame: pd.DataFrame) -> Tuple[pd.Series, np.ndarray]:
    """A label shared by a measurement and its reciprocal, plus a sign correction.

    Reciprocity says that exchanging the current pair with the potential pair
    leaves the transfer resistance unchanged, so the label has to be invariant
    under that exchange. It also has to survive a pair being written in either
    order, since an operator or an instrument may record (B, A) for (A, B).

    Order within a pair is not free, though. ``R = V_MN / I_AB``, so reversing
    the current pair flips the sign of the current, reversing the potential pair
    flips the sign of the voltage, and either one alone flips the sign of R.
    Sorting the pairs to build the label therefore has to report what it did:
    when an odd number of reversals was applied, the recorded resistance belongs
    to the opposite sign convention and must be negated before two rows are
    compared. Ignoring that turns a perfectly good pair holding R and -R into an
    error of 200 %, and rejects it.

    Returns the label and the multiplier (+1 or -1) that brings each row into the
    canonical convention.
    """
    current = frame[["a", "b"]].to_numpy(dtype=np.int64)
    potential = frame[["m", "n"]].to_numpy(dtype=np.int64)

    flip_current = current[:, 0] > current[:, 1]
    flip_potential = potential[:, 0] > potential[:, 1]
    current = np.sort(current, axis=1)
    potential = np.sort(potential, axis=1)

    # An odd number of reversals flips the sign; two reversals cancel.
    sign = np.where(flip_current ^ flip_potential, -1.0, 1.0)

    # Exchanging the two pairs is the reciprocal itself and does not touch sign.
    swap = (current[:, 0] > potential[:, 0]) | (
        (current[:, 0] == potential[:, 0]) & (current[:, 1] > potential[:, 1]))
    low = np.where(swap[:, None], potential, current)
    high = np.where(swap[:, None], current, potential)
    quad = np.column_stack([low, high])
    return pd.Series([tuple(row) for row in quad], index=frame.index), sign


def reciprocal_errors(
    df: pd.DataFrame,
    max_reciprocal_error: float = 0.05,
    *,
    drop_failed: bool = True,
) -> pd.DataFrame:
    """Pair each measurement with its reciprocal and score the disagreement.

    For a quadrupole measured both ways, the relative reciprocal error is

        e = |R_reciprocal - R_normal| / |(R_reciprocal + R_normal) / 2|

    which is the standard definition in the DC and IP literature (Slater et al.
    2000; Binley and Kemna 2005; Binley and Slater 2020, ch. 6). It is the most
    useful single error estimate a DC survey produces, because it carries
    everything that is not repeatable about the measurement, contact resistance
    and telluric noise included, where a stacking standard deviation carries
    only the scatter within one reading.

    The error is computed on transfer resistance rather than on apparent
    resistivity on purpose: the geometric factor is identical for a pair and for
    its reciprocal, so it cancels, and computing the error before any geometry
    is applied keeps a large geometric factor from inflating the difference.

    Both members of a pair receive the same error, and both receive
    ``reciprocalMean``, the average of the two resistances. That average is the
    better value to invert: the two readings are independent measurements of the
    same quantity, so averaging them halves the variance.

    A measurement with no reciprocal keeps its row and takes NaN, so a survey
    that was never measured reciprocally passes through unchanged. With
    ``drop_failed`` set, pairs above the threshold are removed; the caller keeps
    the returned frame's index, so a parallel column can be realigned to it.
    """
    if df.empty or not {"a", "b", "m", "n", "resist"}.issubset(df.columns):
        out = df.copy()
        out["reciprocalErrRel"] = np.nan
        out["reciprocalMean"] = pd.to_numeric(out.get("resist"), errors="coerce")
        return out

    out = df.copy()
    key, sign = _reciprocal_key(out)
    # Compare in one sign convention, then report in the row's own convention.
    canonical = pd.to_numeric(out["resist"], errors="coerce") * sign

    grouped = canonical.groupby(key)
    mean = grouped.transform("mean")
    span = grouped.transform(lambda s: s.max() - s.min())
    counts = grouped.transform("size")

    with np.errstate(divide="ignore", invalid="ignore"):
        error = np.abs(span) / np.abs(mean)
    # A pair straddling zero has a mean near zero and an error that says nothing
    # about the measurement, so it is left unscored rather than rejected.
    error = error.where(np.isfinite(error) & (counts >= 2))
    out["reciprocalErrRel"] = error
    out["reciprocalMean"] = np.where(counts >= 2, mean * sign, canonical * sign)

    if drop_failed and np.isfinite(error).any():
        keep = ~(error > float(max_reciprocal_error))    # NaN stays, being unpaired
        out = out.loc[keep]
    return out
