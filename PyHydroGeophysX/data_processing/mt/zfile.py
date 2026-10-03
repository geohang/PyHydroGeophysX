"""Egbert's EMTF Z-files (``.zmm``, ``.zrr``, ``.zss``): read.

The text output of Egbert's EMTF processing: a header with the site
coordinates, the declination and each channel's azimuth and tilt, then, per
period, the transfer functions of every output channel (Hz, Ex, Ey, ...) on
the inputs Hx, Hy, the inverse coherent signal power matrix ``S`` (2 x 2) and
the residual covariance (outputs x outputs), both stored as lower triangles of
complex numbers. The variance of an element is ``residual_ii S_jj``; this
reader keeps ``S`` and the covariance, so errors rotate correctly.

The transfer functions are in measurement coordinates, field units
((mV/km)/nT) and e^{+i omega t}. Channels laid out orthogonally keep that
frame, with the rotation the Hx azimuth; otherwise the tensor is transformed
to the orthogonal frame of the azimuths' north, ``Z = U_e^-1 Z_m U_h``.

Egbert, G. D. (1997). Robust multiple-station magnetotelluric data processing.
Geophysical Journal International, 130(2), 475-496.
https://doi.org/10.1111/j.1365-246X.1997.tb05663.x
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Dict, List

import numpy as np

from .transfer_function import FIELD_TO_OHM, TransferFunction

_SUFFIXES = (".zmm", ".zrr", ".zss")


def is_zfile(path: Any) -> bool:
    source = Path(path)
    return source.is_file() and source.suffix.lower() in _SUFFIXES


def _numbers(line: str) -> List[float]:
    return [float(v) for v in re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", line)]


def _lower_triangle(values: List[float], size: int) -> np.ndarray:
    """A Hermitian matrix from its lower triangle, written row by row as re, im pairs."""
    pairs = np.asarray(values, dtype=float).reshape(-1, 2)
    matrix = np.zeros((size, size), dtype=complex)
    k = 0
    for i in range(size):
        for j in range(i + 1):
            matrix[i, j] = complex(*pairs[k])
            matrix[j, i] = np.conj(matrix[i, j])
            k += 1
    return matrix


def read_zfile(path: Any) -> TransferFunction:
    """Read an EMTF Z-file into a :class:`TransferFunction`."""
    source = Path(path)
    lines = source.read_text(encoding="latin-1", errors="replace").splitlines()
    latitude = longitude = declination = float("nan")
    channels: List[Dict[str, Any]] = []
    station = source.stem
    k = 0
    while k < len(lines):
        line = lines[k]
        low = line.strip().lower()
        if low.startswith("station"):
            station = line.split(":", 1)[-1].strip() or station
        elif low.startswith("coordinate"):
            numbers = _numbers(line)
            if len(numbers) >= 3:
                latitude, longitude, declination = numbers[0], numbers[1], numbers[2]
        elif low.startswith("number of channels"):
            n_channels = int(_numbers(line)[0])
        elif low.startswith("orientations"):
            for row in lines[k + 1:k + 1 + n_channels]:
                parts = row.split()
                channels.append({"index": int(parts[0]), "azimuth": float(parts[1]),
                                 "tilt": float(parts[2]), "name": parts[-1].lower()})
            k += n_channels
        elif low.startswith("period"):
            break
        k += 1
    if not channels:
        raise ValueError(f"{source.name}: no channel orientations found")
    names = [c["name"] for c in channels]
    if names[:2] != ["hx", "hy"]:
        raise ValueError(f"{source.name}: expected the inputs Hx, Hy first, found {names[:2]}")
    outputs = names[2:]
    n_out = len(outputs)

    periods, tfs, sigs, ress = [], [], [], []
    while k < len(lines):
        line = lines[k]
        if line.strip().lower().startswith("period"):
            periods.append(_numbers(line.split(":", 1)[1])[0])
            k += 1
            block: Dict[str, List[float]] = {"tf": [], "sig": [], "res": []}
            current = None
            while k < len(lines) and not lines[k].strip().lower().startswith("period"):
                low = lines[k].strip().lower()
                if low.startswith("transfer functions"):
                    current = "tf"
                elif low.startswith("inverse coherent"):
                    current = "sig"
                elif low.startswith("residual covariance"):
                    current = "res"
                elif low.startswith("number of data") or not low:
                    pass
                elif current:
                    block[current].extend(_numbers(lines[k]))
                k += 1
            tf_values = np.asarray(block["tf"], dtype=float).reshape(n_out, 2, 2)
            tfs.append(tf_values[..., 0] + 1j * tf_values[..., 1])
            sigs.append(_lower_triangle(block["sig"], 2))
            ress.append(_lower_triangle(block["res"], n_out))
        else:
            k += 1
    tf_all = np.asarray(tfs)                  # (n, n_out, 2)
    sig_all = np.asarray(sigs)                # (n, 2, 2)
    res_all = np.asarray(ress)                # (n, n_out, n_out)
    azimuth = {c["name"]: c["azimuth"] for c in channels}

    e_rows = [outputs.index(c) for c in ("ex", "ey") if c in outputs]
    z = z_res = None
    if len(e_rows) == 2:
        z = tf_all[:, e_rows, :] * FIELD_TO_OHM
        z_res = res_all[:, e_rows][:, :, e_rows] * FIELD_TO_OHM ** 2
    tipper = t_res = None
    if "hz" in outputs:
        h = outputs.index("hz")
        tipper = tf_all[:, [h], :]
        t_res = res_all[:, [h]][:, :, [h]]

    rotation = azimuth["hx"]
    orthogonal = (abs(((azimuth["hy"] - azimuth["hx"]) % 360.0) - 90.0) < 0.5
                  and (z is None or (abs(azimuth["ex"] - azimuth["hx"]) < 0.5
                                     and abs(azimuth["ey"] - azimuth["hy"]) < 0.5)))
    notes = []
    if not orthogonal:
        def unit(a):
            return np.array([np.cos(np.radians(a)), np.sin(np.radians(a))])
        U_h = np.stack([unit(azimuth["hx"]), unit(azimuth["hy"])])
        if z is not None:
            U_e_inv = np.linalg.inv(np.stack([unit(azimuth["ex"]), unit(azimuth["ey"])]))
            z = U_e_inv @ z @ U_h
            z_res = U_e_inv @ z_res @ U_e_inv.T
        if tipper is not None:
            tipper = tipper @ U_h
        sig_all = U_h.T @ sig_all @ U_h
        rotation = 0.0
        notes.append("Channels not orthogonal: transformed to the azimuths' orthogonal frame.")
    n = len(periods)
    return TransferFunction(
        frequency=1.0 / np.asarray(periods, dtype=float),
        z=z, tipper=tipper, rotation=np.full(n, rotation),
        inverse_signal_power=sig_all, z_residual_covariance=z_res,
        tipper_residual_covariance=t_res,
        station=station, latitude=latitude, longitude=longitude, declination=declination,
        metadata={"source_format": source.suffix.lower().lstrip("."), "source_file": str(source),
                  "channels": channels, "sign_convention": "+", "notes": notes},
    )
