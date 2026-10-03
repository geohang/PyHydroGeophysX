"""EMTF XML transfer functions (Kelbert 2020): read and write.

The format of the EarthScope/IRIS SPUD archive. Each ``<Period>`` holds the
impedance ``<Z>``, the tipper ``<T>``, their variances ``<Z.VAR>``/``<T.VAR>``
and, from EMTF, the full error model: the inverse signal power
``<Z.INVSIGCOV>`` (inputs Hx, Hy) and the residual covariance
``<Z.RESIDCOV>`` (outputs). The file states its sign convention
(``<SignConvention>exp(+ i\\omega t)</SignConvention>``) and its units
(``[mV/km]/[nT]``, ``[V/m]/[T]`` or ohms); both are converted to this package's.
The data are in the frame ``<Orientation angle_to_geographic_north=...>``.

Kelbert, A. (2020). EMTF XML: New data interchange format and conversion tools
for electromagnetic transfer functions. Geophysics, 85(1), F1-F17.
https://doi.org/10.1190/geo2018-0679.1

Kelbert, A., Erofeeva, S., Trabant, C., Karstens, R., & Van Fossen, M. (2018).
Taking magnetotelluric data out of the drawer. Eos, 99.
https://doi.org/10.1029/2018EO112859
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Dict, Optional
import xml.etree.ElementTree as ET

import numpy as np

from .transfer_function import FIELD_TO_OHM, MU0, TransferFunction

_INPUTS = {"hx": 0, "hy": 1}
_OUTPUTS_Z = {"ex": 0, "ey": 1}


def is_emtf_xml_file(path: Any) -> bool:
    source = Path(path)
    if not source.is_file() or source.suffix.lower() != ".xml":
        return False
    try:
        head = source.read_text(encoding="utf-8", errors="replace")[:4000]
    except OSError:
        return False
    return "<EM_TF" in head or "<Data count" in head


def _text(node: Optional[ET.Element], default: str = "") -> str:
    return (node.text or "").strip() if node is not None and node.text else default


def _float(node: Optional[ET.Element]) -> float:
    try:
        return float(_text(node))
    except ValueError:
        return float("nan")


def _impedance_scale(units: str) -> float:
    """Ohms per unit of the file's impedance."""
    text = (units or "").replace(" ", "").lower()
    if "mv/km" in text:
        return FIELD_TO_OHM
    if "[v/m]/[t]" in text or text == "v/m/t":
        return MU0
    if "ohm" in text:
        return 1.0
    return FIELD_TO_OHM


_EMPTY = 1.0e32
_BARE_AMPERSAND = re.compile(r"&(?!(?:amp|lt|gt|quot|apos|#\d+|#x[0-9a-fA-F]+);)")


def _parse(source: Path) -> ET.Element:
    """The XML root, tolerating the bare ``&`` of citations that archive files carry."""
    text = source.read_text(encoding="utf-8", errors="replace")
    return ET.fromstring(_BARE_AMPERSAND.sub("&amp;", text))


def _child(node: Optional[ET.Element], name: str) -> Optional[ET.Element]:
    """The first child called ``name``, in any case (``Z.var`` as well as ``Z.VAR``)."""
    if node is None:
        return None
    wanted = name.lower()
    return next((child for child in node if child.tag.lower() == wanted), None)


def _number(text: str) -> float:
    value = float(text)
    return float("nan") if abs(value) >= 0.999 * _EMPTY else value


def _complex(text: str) -> complex:
    parts = [_number(v) for v in text.split()]
    return complex(parts[0], parts[1] if len(parts) > 1 else 0.0)


def _fill(block: Optional[ET.Element], rows: Dict[str, int], cols: Dict[str, int],
          shape, complex_values: bool) -> Optional[np.ndarray]:
    if block is None:
        return None
    out = np.full(shape, np.nan + (1j * np.nan if complex_values else 0.0),
                  dtype=complex if complex_values else float)
    for value in block:
        output = (value.get("output") or "").lower()
        inp = (value.get("input") or "").lower()
        if output not in rows or inp not in cols or value.text is None:
            continue
        number = _complex(value.text) if complex_values else _number(value.text.split()[0])
        out[rows[output], cols[inp]] = number
    return out


def read_emtf_xml(path: Any) -> TransferFunction:
    """Read an EMTF XML file into a :class:`TransferFunction`."""
    source = Path(path)
    root = _parse(source)
    site_node = root.find("Site")
    location = site_node.find("Location") if site_node is not None else None
    sign = _text(root.find("ProcessingInfo/SignConvention"), "exp(+ i\\omega t)")
    orientation = site_node.find("Orientation") if site_node is not None else None
    angle = float((orientation.get("angle_to_geographic_north") if orientation is not None
                   else None) or 0.0)

    frequency, z, z_var, sig, z_res, t, t_var, t_res = ([] for _ in range(8))
    scale = FIELD_TO_OHM
    data = root.find("Data")
    if data is None:
        raise ValueError(f"{source.name} has no <Data> section")
    for period in data.findall("Period"):
        value = float(period.get("value"))
        units = (period.get("units") or "secs").lower()
        seconds = value if units.startswith("s") else 1.0 / value
        frequency.append(1.0 / seconds)
        z_node = _child(period, "Z")
        if z_node is not None:
            scale = _impedance_scale(z_node.get("units", ""))
        z.append(_fill(z_node, _OUTPUTS_Z, _INPUTS, (2, 2), True))
        z_var.append(_fill(_child(period, "Z.VAR"), _OUTPUTS_Z, _INPUTS, (2, 2), False))
        sig.append(_fill(_child(period, "Z.INVSIGCOV"), _INPUTS, _INPUTS, (2, 2), True))
        z_res.append(_fill(_child(period, "Z.RESIDCOV"), _OUTPUTS_Z, _OUTPUTS_Z, (2, 2), True))
        t.append(_fill(_child(period, "T"), {"hz": 0}, _INPUTS, (1, 2), True))
        t_var.append(_fill(_child(period, "T.VAR"), {"hz": 0}, _INPUTS, (1, 2), False))
        t_res.append(_fill(_child(period, "T.RESIDCOV"), {"hz": 0}, {"hz": 0}, (1, 1), True))
        if sig[-1] is None:
            sig[-1] = _fill(_child(period, "T.INVSIGCOV"), _INPUTS, _INPUTS, (2, 2), True)

    def stack(items, shape, complex_values):
        if all(item is None for item in items):
            return None
        blank = np.full(shape, np.nan + (1j * np.nan if complex_values else 0.0))
        return np.stack([blank if item is None else item for item in items])

    n = len(frequency)
    z_arr = stack(z, (2, 2), True)
    zvar = stack(z_var, (2, 2), False)
    sig_arr = stack(sig, (2, 2), True)
    zres = stack(z_res, (2, 2), True)
    t_arr = stack(t, (1, 2), True)
    tvar = stack(t_var, (1, 2), False)
    tres = stack(t_res, (1, 1), True)
    if z_arr is not None:
        z_arr = z_arr * scale
        if zres is not None:
            zres = zres * scale ** 2
    metadata: Dict[str, Any] = {
        "source_format": "emtf_xml", "source_file": str(source), "notes": [],
        "sign_convention_in_file": sign,
        "product_id": _text(root.find("ProductId")),
        "name": _text(site_node.find("Name")) if site_node is not None else "",
    }
    tf = TransferFunction(
        frequency=np.asarray(frequency),
        z=z_arr,
        z_err=None if zvar is None else np.sqrt(np.abs(zvar)) * scale,
        tipper=t_arr,
        tipper_err=None if tvar is None else np.sqrt(np.abs(tvar)),
        rotation=np.full(n, angle),
        inverse_signal_power=sig_arr if (zres is not None or tres is not None) else None,
        z_residual_covariance=zres if sig_arr is not None else None,
        tipper_residual_covariance=tres if sig_arr is not None else None,
        station=_text(site_node.find("Id")) if site_node is not None else source.stem,
        latitude=_float(location.find("Latitude")) if location is not None else float("nan"),
        longitude=_float(location.find("Longitude")) if location is not None else float("nan"),
        elevation=_float(location.find("Elevation")) if location is not None else float("nan"),
        declination=_float(location.find("Declination")) if location is not None else float("nan"),
        metadata=metadata,
    )
    if "-" in sign.replace("exp(", "").split("i")[0]:
        tf = tf.conjugated()
        tf.metadata["notes"].append("Conjugated from the file's e^{-i omega t}.")
    tf.metadata["sign_convention"] = "+"
    return tf


def _value(parent: ET.Element, text: str, **attributes: str) -> None:
    node = ET.SubElement(parent, "Value", {k: v for k, v in attributes.items() if v})
    node.text = text


def write_emtf_xml(tf: TransferFunction, path: Any) -> Path:
    """Write ``tf`` as an EMTF XML file, impedances in (mV/km)/nT, e^{+i omega t}.

    The full error model is written when the site carries it; otherwise only
    the variances, ``err^2``. One rotation angle is written for the whole site,
    so a site whose frequencies are in different frames is first rotated to
    the frame of its first frequency.
    """
    if np.ptp(tf.rotation) > 1e-6:
        tf = tf.rotated(float(tf.rotation[0]))
    root = ET.Element("EM_TF")
    ET.SubElement(root, "Description").text = "Magnetotelluric transfer functions"
    ET.SubElement(root, "ProductId").text = tf.station or Path(path).stem
    site = ET.SubElement(root, "Site")
    ET.SubElement(site, "Id").text = tf.station or Path(path).stem
    location = ET.SubElement(site, "Location", {"datum": "WGS84"})
    for tag, value in (("Latitude", tf.latitude), ("Longitude", tf.longitude),
                       ("Elevation", tf.elevation)):
        node = ET.SubElement(location, tag, {"units": "meters"} if tag == "Elevation" else {})
        node.text = f"{value:.6f}" if np.isfinite(value) else "0.0"
    if np.isfinite(tf.declination):
        ET.SubElement(location, "Declination").text = f"{tf.declination:.3f}"
    ET.SubElement(site, "Orientation",
                  {"angle_to_geographic_north": f"{tf.rotation[0]:.3f}"}).text = "orthogonal"
    processing = ET.SubElement(root, "ProcessingInfo")
    ET.SubElement(processing, "SignConvention").text = "exp(+ i\\omega t)"
    data = ET.SubElement(root, "Data", {"count": str(tf.n_frequencies)})
    names_z = (("Zxx", "Ex", "Hx"), ("Zxy", "Ex", "Hy"), ("Zyx", "Ey", "Hx"), ("Zyy", "Ey", "Hy"))
    z = tf.z_field_units
    for k in np.argsort(tf.period):
        period = ET.SubElement(data, "Period", {"value": f"{tf.period[k]:.6e}", "units": "secs"})
        if z is not None:
            block = ET.SubElement(period, "Z", {"type": "complex", "size": "2 2",
                                                "units": "[mV/km]/[nT]"})
            var = ET.SubElement(period, "Z.VAR", {"type": "real", "size": "2 2"})
            err = tf.z_err if tf.z_err is not None else np.full(tf.z.shape, np.nan)
            for (name, out, inp), (i, j) in zip(names_z, ((0, 0), (0, 1), (1, 0), (1, 1))):
                _value(block, f"{z[k, i, j].real:.6e} {z[k, i, j].imag:.6e}",
                       name=name, output=out, input=inp)
                _value(var, f"{(err[k, i, j] / FIELD_TO_OHM) ** 2:.6e}",
                       name=name, output=out, input=inp)
            if tf.inverse_signal_power is not None and tf.z_residual_covariance is not None:
                _matrix(period, "Z.INVSIGCOV", tf.inverse_signal_power[k], ("Hx", "Hy"), ("Hx", "Hy"))
                _matrix(period, "Z.RESIDCOV", tf.z_residual_covariance[k] / FIELD_TO_OHM ** 2,
                        ("Ex", "Ey"), ("Ex", "Ey"))
        if tf.tipper is not None:
            block = ET.SubElement(period, "T", {"type": "complex", "size": "1 2", "units": "[]"})
            var = ET.SubElement(period, "T.VAR", {"type": "real", "size": "1 2"})
            err = tf.tipper_err if tf.tipper_err is not None else np.full(tf.tipper.shape, np.nan)
            for j, (name, inp) in enumerate((("Tx", "Hx"), ("Ty", "Hy"))):
                value = tf.tipper[k, 0, j]
                _value(block, f"{value.real:.6e} {value.imag:.6e}", name=name, output="Hz", input=inp)
                _value(var, f"{err[k, 0, j] ** 2:.6e}", name=name, output="Hz", input=inp)
            if tf.inverse_signal_power is not None and tf.tipper_residual_covariance is not None:
                _matrix(period, "T.INVSIGCOV", tf.inverse_signal_power[k], ("Hx", "Hy"), ("Hx", "Hy"))
                _matrix(period, "T.RESIDCOV", tf.tipper_residual_covariance[k], ("Hz",), ("Hz",))
    ET.indent(root)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(target, encoding="utf-8", xml_declaration=True)
    return target


def _matrix(parent: ET.Element, tag: str, values: np.ndarray, rows, cols) -> None:
    block = ET.SubElement(parent, tag, {"type": "complex", "size": f"{len(rows)} {len(cols)}"})
    for i, out in enumerate(rows):
        for j, inp in enumerate(cols):
            _value(block, f"{values[i, j].real:.6e} {values[i, j].imag:.6e}", output=out, input=inp)
