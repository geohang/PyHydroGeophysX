"""SEG MT/EMAP Electrical Data Interchange (EDI) files: read and write.

The format is Wight's 1987 SEG standard. A file is a sequence of blocks, each a
``>KEYWORD``, an option list ``NAME=value`` and, for data blocks, a data set
``//count`` followed by the values; ``>=KEYWORD`` opens a section and
``>!...!`` is a comment. Files written by real instruments and programs bend
it, so this reader takes keywords and option names in any case, options with
spaces after ``=`` (``ID=    11.001``), ``// 80`` with a space, and values
written with a Fortran ``D`` exponent.

What a file can hold, in the order the reader prefers it:

- the impedance ``>ZXXR``/``>ZXXI`` ..., with ``>ZXX.VAR`` (or ``.R.VAR`` and
  ``.I.VAR``) and the tipper ``>TXR.EXP`` ... with ``>TXVAR.EXP``;
- apparent resistivities and phases only, ``>RHOXY``/``>PHSXY`` with ``.ERR``,
  turned into the impedance they imply. Programs commonly write PHSYX folded
  into the first quadrant; it is moved back to the third, and the site says so;
- power spectra (``>=SPECTRASECT``), from which the impedance and tipper are
  estimated, ``Z = <E R*> <H R*>^-1`` with the section's reference channels.

Impedances are in (mV/km)/nT and are converted to ohms. Values at or beyond
the file's EMPTY (1.0E32) are missing. The standard's time dependence is
e^{+i omega t}; a file whose phases sit in the quadrants of the opposite
convention throughout is conjugated, and the site records it.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .transfer_function import (
    FIELD_TO_OHM,
    TransferFunction,
    from_apparent_resistivity,
    resolve_sign_convention,
)

_EMPTY = 1.0e32
_COMMENT = re.compile(r">!.*?!", re.DOTALL)
_BLOCK_START = re.compile(r"(?m)^[ \t]*>(?!!)")
_OPTION = re.compile(r'([A-Za-z][\w.]*)\s*=\s*("[^"]*"|[^\s"]+)?')
_NUMBER = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eEdD][-+]?\d+)?")
#: Blocks whose text is free and never holds a data set.
_FREE_TEXT = {"INFO"}
#: Blocks with options only: a ``//`` in them (a URL in >HEAD) is not a data set.
_NO_DATA = _FREE_TEXT | {"HEAD", "=DEFINEMEAS", "HMEAS", "EMEAS", "=MTSECT", "END"}


def is_edi_file(path: Any) -> bool:
    """Whether ``path`` is an EDI file: a ``.edi`` name, or a text that opens with >HEAD."""
    source = Path(path)
    if not source.is_file():
        return False
    if source.suffix.lower() == ".edi":
        return True
    try:
        head = source.read_text(encoding="latin-1", errors="replace")[:2000]
    except OSError:
        return False
    return bool(re.match(r"\s*>HEAD\b", head, re.IGNORECASE))


# ---------------------------------------------------------------------------
# tokenizing
# ---------------------------------------------------------------------------
def _blocks(text: str) -> List[Tuple[str, Dict[str, str], Optional[np.ndarray], str]]:
    """Split an EDI text into ``(KEYWORD, options, data or None, raw body)``."""
    text = _COMMENT.sub(" ", text)
    out = []
    starts = [m.end() for m in _BLOCK_START.finditer(text)]
    for k, start in enumerate(starts):
        end = starts[k + 1] - 1 if k + 1 < len(starts) else len(text)
        chunk = text[start:end]
        match = re.match(r"\s*(=?[A-Za-z][\w.]*)", chunk)
        if match is None:
            continue
        keyword = match.group(1).upper()
        body = chunk[match.end():]
        data = None
        if keyword not in _NO_DATA and "//" in body:
            head, _, rest = body.partition("//")
            numbers = _NUMBER.findall(rest.replace("D", "E").replace("d", "e"))
            if numbers:
                count = int(float(numbers[0]))
                values = np.asarray([float(v) for v in numbers[1:1 + count]], dtype=float)
                data = values
            body = head
        options = {}
        if keyword not in _FREE_TEXT:
            for name, value in _OPTION.findall(body):
                options[name.upper()] = (value or "").strip('"')
        out.append((keyword, options, data, chunk[match.end():]))
    return out


def _as_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(str(value).replace("D", "E").replace("d", "e"))
    except (TypeError, ValueError):
        return default


def _angle(value: Any) -> float:
    """A latitude or longitude written ``dd:mm:ss.s``, ``dd:mm.mm`` or decimal."""
    text = str(value or "").strip()
    if not text:
        return float("nan")
    if ":" not in text:
        return _as_float(text)
    sign = -1.0 if text.startswith("-") else 1.0
    parts = [abs(_as_float(p, 0.0)) for p in text.lstrip("+-").split(":")]
    degrees = sum(p / 60.0 ** i for i, p in enumerate(parts))
    return sign * degrees


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------
def read_edi(path: Any, *, sign_convention: str = "auto") -> TransferFunction:
    """Read an EDI file into a :class:`TransferFunction`.

    ``sign_convention`` is ``"+"`` (the standard's e^{+i omega t}), ``"-"``
    (conjugate everything), or ``"auto"``: ``"+"`` unless the Zxy and Zyx
    phases lie in the quadrants of e^{-i omega t} at four frequencies in five.
    """
    source = Path(path)
    text = source.read_text(encoding="latin-1", errors="replace")
    blocks = _blocks(text)
    if not blocks or blocks[0][0] != "HEAD":
        raise ValueError(f"{source.name} is not an EDI file: it does not start with >HEAD")

    head: Dict[str, str] = {}
    info: List[str] = []
    define: Dict[str, str] = {}
    measurements: Dict[str, Dict[str, str]] = {}
    sections: List[Tuple[str, Dict[str, str], Optional[np.ndarray]]] = []
    data: Dict[str, Tuple[Dict[str, str], np.ndarray]] = {}
    spectra: List[Tuple[Dict[str, str], np.ndarray]] = []
    for keyword, options, values, raw in blocks:
        if keyword == "HEAD":
            head.update(options)
        elif keyword == "INFO":
            info.append(raw.strip("\r\n"))
        elif keyword == "=DEFINEMEAS":
            define.update(options)
        elif keyword in ("HMEAS", "EMEAS"):
            ident = options.get("ID", "").strip()
            measurements[ident] = dict(options, KIND=keyword[0])
        elif keyword.startswith("="):
            sections.append((keyword, options, values))
        elif keyword == "SPECTRA":
            if values is not None:
                spectra.append((options, values))
        elif keyword == "END":
            break
        elif values is not None:
            data.setdefault(keyword, (options, values))

    empty = abs(_as_float(head.get("EMPTY", _EMPTY), _EMPTY)) or _EMPTY

    n_freq = data["FREQ"][1].size if "FREQ" in data else None

    def block(name: str) -> Optional[np.ndarray]:
        entry = data.get(name)
        if entry is None:
            return None
        values = np.asarray(entry[1], dtype=float).copy()
        values[np.abs(values) >= 0.999 * empty] = np.nan
        if n_freq is not None and values.size != n_freq:
            raise ValueError(f"{source.name}: >{name} has {values.size} values for "
                             f"{n_freq} frequencies")
        return values

    metadata: Dict[str, Any] = {
        "source_format": "edi", "source_file": str(source), "head": dict(head),
        "info": "\n".join(info).strip(), "define_measurements": dict(define),
        "measurements": measurements, "notes": [],
    }
    elevation = _as_float(head.get("ELEV", define.get("REFELEV")))
    if str(head.get("UNITS", "")).strip().upper() in ("FT", "FEET"):
        elevation *= 0.3048
    site = dict(
        station=str(head.get("DATAID", source.stem)).strip() or source.stem,
        latitude=_angle(head.get("LAT", define.get("REFLAT"))),
        longitude=_angle(head.get("LONG", head.get("LON", define.get("REFLONG")))),
        elevation=elevation,
        metadata=metadata,
    )

    spectra_section = next((s for s in sections if s[0] == "=SPECTRASECT"), None)
    frequency = block("FREQ")
    if frequency is not None and _has_any(data, "ZXYR", "ZYXR", "ZXXR", "ZYYR"):
        tf = _impedance_blocks(block, frequency, data, measurements, site)
    elif frequency is not None and _has_any(data, "RHOXY", "RHOYX"):
        tf = _resistivity_blocks(block, frequency, data, site)
    elif spectra_section is not None and spectra:
        tf = _spectra_blocks(spectra_section, spectra, measurements, site, empty)
    else:
        raise ValueError(f"{source.name} has no impedance, apparent resistivity or "
                         "spectra data that this reader can use")
    return resolve_sign_convention(tf, sign_convention)


def _has_any(data: Dict[str, Any], *names: str) -> bool:
    return any(name in data for name in names)


def _rotation_for(data, block, frequency, rot_option: str, measurements) -> np.ndarray:
    """The x-axis azimuth of a block, from its ROT option, the rotation block or the hx sensor."""
    rot = rot_option.strip().upper()
    if rot and rot not in ("NONE", "NORTH"):
        angles = block(rot)
        if angles is not None and angles.size == frequency.size:
            return np.where(np.isfinite(angles), angles, 0.0)
    if rot == "NORTH":
        return np.zeros(frequency.size)
    hx = next((m for m in measurements.values()
               if str(m.get("CHTYPE", "")).upper() == "HX"), None)
    return np.full(frequency.size, _as_float(hx.get("AZM", 0.0), 0.0) if hx else 0.0)


def _impedance_blocks(block, frequency, data, measurements, site) -> TransferFunction:
    n = frequency.size
    z = np.full((n, 2, 2), np.nan + 1j * np.nan)
    z_err = np.full((n, 2, 2), np.nan)
    rot_option = ""
    for i, a in enumerate("XY"):
        for j, b in enumerate("XY"):
            name = f"Z{a}{b}"
            real, imag = block(name + "R"), block(name + "I")
            if real is None or imag is None:
                continue
            rot_option = rot_option or data[name + "R"][0].get("ROT", "")
            z[:, i, j] = (real + 1j * imag) * FIELD_TO_OHM
            var = block(name + ".VAR")
            if var is None:
                var_r, var_i = block(name + "R.VAR"), block(name + "I.VAR")
                if var_r is None:
                    var_r, var_i = block(name + ".R.VAR"), block(name + ".I.VAR")
                if var_r is not None and var_i is not None:
                    var = 0.5 * (var_r + var_i)
            if var is not None:
                z_err[:, i, j] = np.sqrt(np.abs(var)) * FIELD_TO_OHM
    tipper, tipper_err, t_rot = _tipper_blocks(block, data, n)
    rotation = _rotation_for(data, block, frequency, rot_option, measurements)
    tf = TransferFunction(frequency=frequency, z=z, z_err=z_err, tipper=tipper,
                          tipper_err=tipper_err, rotation=rotation, **site)
    if t_rot is not None and tipper is not None:
        t_rotation = _rotation_for(data, block, frequency, t_rot, measurements)
        if not np.allclose(t_rotation, rotation):
            # The tipper is expressed in its own frame; bring it to the impedance's.
            moved = TransferFunction(frequency=frequency, tipper=tipper, tipper_err=tipper_err,
                                     rotation=t_rotation).rotated(rotation)
            tf.tipper, tf.tipper_err = moved.tipper, moved.tipper_err
    return tf


def _tipper_blocks(block, data, n):
    tx_r, tx_i = block("TXR.EXP"), block("TXI.EXP")
    ty_r, ty_i = block("TYR.EXP"), block("TYI.EXP")
    if tx_r is None or ty_r is None or tx_i is None or ty_i is None:
        return None, None, None
    tipper = np.stack([tx_r + 1j * tx_i, ty_r + 1j * ty_i], axis=-1).reshape(n, 1, 2)
    err = np.full((n, 1, 2), np.nan)
    for k, name in enumerate(("TXVAR.EXP", "TYVAR.EXP")):
        var = block(name)
        if var is not None:
            err[:, 0, k] = np.sqrt(np.abs(var))
    rot = data["TXR.EXP"][0].get("ROT", "")
    return tipper, err, rot


def _resistivity_blocks(block, frequency, data, site) -> TransferFunction:
    rho, phase, rho_err, phase_err = {}, {}, {}, {}
    rot_option = ""
    for component in ("xx", "xy", "yx", "yy"):
        r = block(f"RHO{component.upper()}")
        p = block(f"PHS{component.upper()}")
        if r is None or p is None:
            continue
        rot_option = rot_option or data[f"RHO{component.upper()}"][0].get("ROT", "")
        if component == "yx" and np.nanmedian(np.abs(p)) < 90.0:
            # Folded into the first quadrant, as most writers do; the standard
            # puts Zyx near -135 for a half-space.
            p = p - 180.0
            site["metadata"]["notes"].append(
                "PHSYX was written in the first quadrant; it was moved to the third.")
        rho[component], phase[component] = r, p
        for store, suffix in ((rho_err, "RHO"), (phase_err, "PHS")):
            err = block(f"{suffix}{component.upper()}.ERR")
            if err is not None:
                store[component] = err
    rotation = _rotation_for(data, block, frequency, rot_option, site["metadata"]["measurements"])
    site["metadata"]["notes"].append("Impedance computed from apparent resistivity and phase.")
    return from_apparent_resistivity(frequency, rho, phase, rho_err=rho_err or None,
                                     phase_err_deg=phase_err or None, rotation=rotation, **site)


def _spectra_blocks(section, spectra, measurements, site, empty) -> TransferFunction:
    """The impedance and tipper a spectra section implies, with first-order errors.

    The cross-power matrix of each block is unpacked as the standard describes
    it - the real part of <A B*> below the diagonal, the imaginary part above
    it - and ``Z = <E R*> <H R*>^-1`` with the reference channels, which are the
    section's remote channels when it has them and the local H otherwise. The
    variance of each element is ``r S_jj / n``, with ``r`` the output's residual
    power and ``S = <H R*>^-H <R R*> <H R*>^-1``, the remote-reference estimator's
    covariance, and ``n`` the block's AVGT, as EMTF computes it; its square root
    is the error, the EMTF convention the covariance-based errors here follow.
    (AVGF is not multiplied in: writers put the same count in both.)

    The cross powers, packed as the standard describes, give the complex
    conjugate of the impedance in its own e^{+i omega t} convention: read as
    they are, the phases come out in the quadrants of e^{-i omega t} (Zxy near
    -45, Zyx near +135). So the estimate is conjugated.
    """
    _, options, ids = section
    if ids is None:
        raise ValueError("the >=SPECTRASECT block does not list its channels")
    order = [f"{value:.6g}" for value in ids]
    by_id = {f"{_as_float(key):.6g}": value for key, value in measurements.items()}
    kinds = [str(by_id.get(key, {}).get("CHTYPE", "")).upper() for key in order]

    def first(name: str, start: int = 0) -> Optional[int]:
        for index in range(start, len(kinds)):
            if kinds[index] == name:
                return index
        return None

    ex, ey, hx, hy, hz = (first(name) for name in ("EX", "EY", "HX", "HY", "HZ"))
    if None in (ex, ey, hx, hy):
        raise ValueError(f"the spectra section needs Ex, Ey, Hx and Hy, found {kinds}")
    # Remote channels: an R-type, or a second HX/HY after the local pair.
    rx = next((i for i, k in enumerate(kinds) if k in ("RX", "RHX")), None)
    ry = next((i for i, k in enumerate(kinds) if k in ("RY", "RHY")), None)
    if rx is None:
        rx = first("HX", hx + 1)
    if ry is None:
        ry = first("HY", hy + 1)
    remote = rx is not None and ry is not None
    ref = (rx, ry) if remote else (hx, hy)

    frequency, z, z_err, tipper, tipper_err, rotation = [], [], [], [], [], []
    n_chan = len(order)
    for opts, values in spectra:
        if values.size != n_chan * n_chan:
            continue
        packed = values.reshape(n_chan, n_chan)
        if np.any(np.abs(packed) >= 0.999 * empty):
            continue
        lower = np.tril(packed, -1)
        upper = np.triu(packed, 1)
        C = np.diag(np.diag(packed)).astype(complex) + (lower.T + 1j * upper)
        C = C + np.conj(np.triu(C, 1)).T
        HR = C[np.ix_((hx, hy), ref)]
        try:
            HR_inv = np.linalg.inv(HR)
        except np.linalg.LinAlgError:
            continue
        RR = C[np.ix_(ref, ref)]
        # Cov(z) = sigma^2 / n  <H R*>^-H <R R*> <H R*>^-1 for a row of Z; with
        # the local reference it is the familiar <H H*>^-1.
        S = np.conj(HR_inv).T @ RR @ HR_inv
        n_avg = max(_as_float(opts.get("AVGT", 1), 1.0), 1.0)
        rows = []
        errs = []
        for out in (ex, ey) + ((hz,) if hz is not None else ()):
            ER = C[out, list(ref)]
            row = ER @ HR_inv          # conjugated below, with the errors unchanged
            HH = C[np.ix_((hx, hy), (hx, hy))]
            He = C[[hx, hy], out]
            residual = (C[out, out] - row @ He - np.conj(He) @ np.conj(row)
                        + row @ HH @ np.conj(row))
            rows.append(row)
            errs.append(np.sqrt(np.abs(np.real(residual)) * np.abs(np.real(np.diag(S)))
                                / n_avg))
        rows = [np.conj(row) for row in rows]
        frequency.append(_as_float(opts.get("FREQ")))
        z.append(np.array(rows[:2]) * FIELD_TO_OHM)
        z_err.append(np.array(errs[:2]) * FIELD_TO_OHM)
        if hz is not None:
            tipper.append(np.array(rows[2:3]))
            tipper_err.append(np.array(errs[2:3]))
        rotation.append(_as_float(opts.get("ROTSPEC", 0.0), 0.0))
    if not frequency:
        raise ValueError("no spectra block could be turned into an impedance")
    site["metadata"]["notes"].append(
        "Impedance estimated from the power spectra with "
        + ("remote" if remote else "local") + " reference channels.")
    site["metadata"]["spectra_reference"] = "remote" if remote else "local"
    return TransferFunction(
        frequency=np.asarray(frequency), z=np.asarray(z), z_err=np.asarray(z_err),
        tipper=np.asarray(tipper) if tipper else None,
        tipper_err=np.asarray(tipper_err) if tipper_err else None,
        rotation=np.asarray(rotation), **site)


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------
def _format_block(name: str, values: np.ndarray, options: str = "") -> List[str]:
    values = np.asarray(values, dtype=float)
    values = np.where(np.isfinite(values), values, _EMPTY)
    lines = [f">{name}{(' ' + options) if options else ''} //{values.size}"]
    for start in range(0, values.size, 6):
        lines.append(" ".join(f"{v: .6E}" for v in values[start:start + 6]))
    return lines


def _format_angle(value: float) -> str:
    if not np.isfinite(value):
        return "0:00:00.00"
    sign = "-" if value < 0 else ""
    value = abs(value)
    d = int(value)
    m = int((value - d) * 60)
    s = (value - d - m / 60.0) * 3600
    return f"{sign}{d}:{m:02d}:{s:05.2f}"


def write_edi(tf: TransferFunction, path: Any) -> Path:
    """Write ``tf`` as a standard EDI file, impedances in (mV/km)/nT, e^{+i omega t}.

    The impedance blocks carry ``Z**.VAR`` as the squared standard error, the
    tipper the experimental ``T*.EXP`` blocks every program reads, and both
    rotation blocks the site's rotation.
    """
    target = Path(path)
    station = tf.station or target.stem
    lines = [
        ">HEAD",
        f'  DATAID="{station}"',
        '  ACQBY="unknown"',
        '  FILEBY="PyHydroGeophysX"',
        f"  FILEDATE={date.today().isoformat()}",
        f"  LAT={_format_angle(tf.latitude)}",
        f"  LONG={_format_angle(tf.longitude)}",
        f"  ELEV={tf.elevation if np.isfinite(tf.elevation) else 0.0:.3f}",
        "  UNITS=M",
        '  STDVERS="SEG 1.0"',
        "  EMPTY=1.0E32",
        "",
        ">INFO",
        f"  Written by PyHydroGeophysX from {tf.metadata.get('source_format', 'a transfer function')}.",
        "",
        ">=DEFINEMEAS",
        "  MAXCHAN=7",
        "  MAXRUN=999",
        "  MAXMEAS=9999",
        "  UNITS=M",
        f"  REFLAT={_format_angle(tf.latitude)}",
        f"  REFLONG={_format_angle(tf.longitude)}",
        f"  REFELEV={tf.elevation if np.isfinite(tf.elevation) else 0.0:.3f}",
        "",
        ">HMEAS ID=1001.001 CHTYPE=HX X=0.0 Y=0.0 Z=0.0 AZM=0.0",
        ">HMEAS ID=1002.001 CHTYPE=HY X=0.0 Y=0.0 Z=0.0 AZM=90.0",
        ">HMEAS ID=1003.001 CHTYPE=HZ X=0.0 Y=0.0 Z=0.0 AZM=0.0",
        ">EMEAS ID=1004.001 CHTYPE=EX X=0.0 Y=0.0 Z=0.0 X2=0.0 Y2=0.0",
        ">EMEAS ID=1005.001 CHTYPE=EY X=0.0 Y=0.0 Z=0.0 X2=0.0 Y2=0.0",
        "",
        ">=MTSECT",
        f'  SECTID="{station}"',
        f"  NFREQ={tf.n_frequencies}",
        "  HX=1001.001",
        "  HY=1002.001",
        "  HZ=1003.001",
        "  EX=1004.001",
        "  EY=1005.001",
        "",
    ]
    lines += _format_block("FREQ", tf.frequency)
    lines += _format_block("ZROT", tf.rotation)
    if tf.z is not None:
        z = tf.z_field_units
        err = (tf.z_err if tf.z_err is not None else np.full(tf.z.shape, np.nan)) / FIELD_TO_OHM
        for i, a in enumerate("XY"):
            for j, b in enumerate("XY"):
                lines += _format_block(f"Z{a}{b}R", z[:, i, j].real, "ROT=ZROT")
                lines += _format_block(f"Z{a}{b}I", z[:, i, j].imag, "ROT=ZROT")
                lines += _format_block(f"Z{a}{b}.VAR", err[:, i, j] ** 2, "ROT=ZROT")
    if tf.tipper is not None:
        lines += _format_block("TROT", tf.rotation)
        terr = tf.tipper_err if tf.tipper_err is not None else np.full(tf.tipper.shape, np.nan)
        for k, a in enumerate("XY"):
            lines += _format_block(f"T{a}R.EXP", tf.tipper[:, 0, k].real, "ROT=TROT")
            lines += _format_block(f"T{a}I.EXP", tf.tipper[:, 0, k].imag, "ROT=TROT")
            lines += _format_block(f"T{a}VAR.EXP", terr[:, 0, k] ** 2, "ROT=TROT")
    lines.append(">END")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="ascii", errors="replace")
    return target
