"""Jones's J-format (``.j``) transfer functions, as BIRRP writes them: read.

Comment lines start with ``#`` and keyword lines with ``>`` (``>LATITUDE =``,
``>AZIMUTH =``); the first other line is the station name. Then come blocks,
each a component name - ``ZXX``..``ZYY`` impedances, ``RXX``..``RYY`` apparent
resistivities and phases, ``TZX``/``TZY`` tipper - with an optional units word,
a line with the number of rows, and the rows: period, real part, imaginary
part, error, ... for Z and T; period, resistivity, phase, ... for R. Missing
values are written -999; a negative period is a frequency.

The units word is unreliable: BIRRP labels field-unit impedances ``S.I.``. When
the file also has R blocks, the impedance units are those that reproduce the
file's own apparent resistivities; otherwise field units, (mV/km)/nT, are
assumed unless ``z_units`` says otherwise.

Jones, A. G. (1994). J-format: a format for the exchange of magnetotelluric
transfer functions. MTNet, https://www.mtnet.info/docs/jformat.txt
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .transfer_function import FIELD_TO_OHM, MU0, TransferFunction, resolve_sign_convention

_Z = {"ZXX": (0, 0), "ZXY": (0, 1), "ZYX": (1, 0), "ZYY": (1, 1)}
_R = {"RXX": (0, 0), "RXY": (0, 1), "RYX": (1, 0), "RYY": (1, 1)}
_T = {"TZX": 0, "TZY": 1}


def is_jfile(path: Any) -> bool:
    source = Path(path)
    return source.is_file() and source.suffix.lower() == ".j"


def _float(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        return float("nan")
    return float("nan") if value == -999 else value


def read_jfile(path: Any, *, z_units: str = "auto", sign_convention: str = "auto") -> TransferFunction:
    """Read a J-format file; ``z_units`` is ``"auto"``, ``"field"`` or ``"ohm"``.

    BIRRP writes e^{-i omega t}; ``sign_convention`` is resolved as for EDI files
    (:func:`resolve_sign_convention`), so such a file is conjugated.
    """
    source = Path(path)
    lines = source.read_text(encoding="latin-1", errors="replace").splitlines()
    header: Dict[str, str] = {}
    body: List[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith(">"):
            key, _, value = stripped[1:].partition("=")
            header[key.strip().upper()] = value.strip()
            continue
        body.append(stripped)
    if not body:
        raise ValueError(f"{source.name} has no data")
    station = body[0].split()[0]
    blocks: Dict[str, List[List[float]]] = {}
    k = 1
    while k < len(body):
        name = body[k].split()[0].upper()
        if name in _Z or name in _R or name in _T:
            count = int(float(body[k + 1].split()[0]))
            rows = [[_float(v) for v in body[j].split()] for j in range(k + 2, k + 2 + count)]
            blocks[name] = rows
            k += 2 + count
        else:
            k += 1

    def period_of(value: float) -> float:
        return 1.0 / abs(value) if value < 0 else value

    periods = sorted({round(period_of(row[0]), 10) for name in list(_Z) + list(_T)
                      for row in blocks.get(name, []) if np.isfinite(row[0]) and row[0] != 0})
    if not periods:
        raise ValueError(f"{source.name} has no impedance or tipper blocks")
    index = {p: i for i, p in enumerate(periods)}
    n = len(periods)
    z = np.full((n, 2, 2), np.nan + 1j * np.nan)
    z_err = np.full((n, 2, 2), np.nan)
    for name, (i, j) in _Z.items():
        for row in blocks.get(name, []):
            k = index.get(round(period_of(row[0]), 10))
            if k is not None and len(row) >= 4:
                z[k, i, j] = complex(row[1], row[2])
                z_err[k, i, j] = row[3]
    tipper = tipper_err = None
    if any(name in blocks for name in _T):
        tipper = np.full((n, 1, 2), np.nan + 1j * np.nan)
        tipper_err = np.full((n, 1, 2), np.nan)
        for name, j in _T.items():
            for row in blocks.get(name, []):
                k = index.get(round(period_of(row[0]), 10))
                if k is not None and len(row) >= 4:
                    tipper[k, 0, j] = complex(row[1], row[2])
                    tipper_err[k, 0, j] = row[3]

    notes = []
    scale = FIELD_TO_OHM
    if z_units == "ohm":
        scale = 1.0
    elif z_units == "auto" and any(name in blocks for name in _R):
        scale, note = _units_from_resistivity(blocks, index, z, np.asarray(periods))
        notes.append(note)
    elif z_units not in ("auto", "field"):
        raise ValueError("z_units must be 'auto', 'field' or 'ohm'")
    azimuth = _float(header.get("AZIMUTH", "0")) if header.get("AZIMUTH") else 0.0

    def coordinate(key: str) -> float:
        return _float(header.get(key, "")) if header.get(key) else float("nan")

    tf = TransferFunction(
        frequency=1.0 / np.asarray(periods), z=z * scale, z_err=z_err * scale,
        tipper=tipper, tipper_err=tipper_err,
        rotation=np.full(n, azimuth if np.isfinite(azimuth) else 0.0),
        station=station, latitude=coordinate("LATITUDE"), longitude=coordinate("LONGITUDE"),
        elevation=coordinate("ELEVATION"),
        metadata={"source_format": "jfile", "source_file": str(source), "header": header,
                  "notes": notes},
    )
    return resolve_sign_convention(tf, sign_convention)


def _units_from_resistivity(blocks, index, z, periods) -> tuple:
    """The impedance scale (ohms per file unit) that reproduces the file's R blocks."""
    misfit = {}
    for label, scale in (("field units", FIELD_TO_OHM), ("ohms", 1.0)):
        logs = []
        for name, (i, j) in _R.items():
            for row in blocks.get(name, []):
                k = index.get(round(1.0 / abs(row[0]) if row[0] < 0 else row[0], 10))
                if k is None or len(row) < 2 or not np.isfinite(row[1]) or row[1] <= 0:
                    continue
                rho = abs(z[k, i, j] * scale) ** 2 * periods[k] / (2 * np.pi * MU0)
                if np.isfinite(rho) and rho > 0:
                    logs.append(abs(np.log10(rho / row[1])))
        misfit[label] = (np.median(logs) if logs else np.inf, scale)
    label = min(misfit, key=lambda key: misfit[key][0])
    return misfit[label][1], (f"Impedance units taken as {label}: they reproduce the "
                              f"file's apparent resistivities (median log10 misfit "
                              f"{misfit[label][0]:.3f}).")
