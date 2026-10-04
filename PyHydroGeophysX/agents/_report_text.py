"""Small pieces of report text shared by the per-method report sections.

Lengths are shown in the unit the run's figures use
(``config['figure_style']['length_unit']``); every result stays in metres.
"""
from __future__ import annotations

import os
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from ..visualization.axis_units import get_length_unit, normalize_length_unit, to_display_length


def length_unit(config: Optional[Mapping[str, Any]] = None) -> str:
    """``'m'`` or ``'ft'``: the unit the run's figures, and so its tables, show."""
    unit = ((config or {}).get("figure_style") or {}).get("length_unit")
    return normalize_length_unit(unit or get_length_unit())


def length(value: float, unit: str) -> str:
    """A length in metres, written in ``unit`` as a reader would write it.

    Three significant figures; kilometres from 10 km, and whole numbers with
    separators from a thousand, rather than "4.75e+04 m".

    >>> length(10.0, 'm'), length(10.0, 'ft'), length(2345.0, 'm'), length(47500.0, 'm')
    ('10 m', '32.8 ft', '2,345 m', '47.5 km')
    """
    shown = float(to_display_length(float(value), unit))
    if unit == "m" and abs(shown) >= 10000:
        return f"{shown / 1000:.3g} km"
    if abs(shown) >= 1000:
        return f"{shown:,.0f} {unit}"
    return f"{shown:.3g} {unit}"


def number(value: float, digits: int = 4) -> str:
    """A value to ``digits`` significant figures, as a table prints it.

    Rounding noise is shown as 0, and a value too large for ``digits`` figures
    as a whole number with separators rather than in exponent form.

    >>> number(3.988e-15), number(20.913), number(-111.29), number(1093.4, 3), number(0.000125)
    ('0', '20.91', '-111.3', '1,093', '0.000125')
    """
    value = float(value)
    if abs(value) < 1e-9:
        return "0"
    if abs(value) >= 10 ** digits:
        return f"{value:,.0f}"
    return f"{value:.{digits}g}"


def band(top: float, bottom: float, unit: str) -> str:
    """A depth band as a table cell, ``'2-5 m'`` or ``'below 160 m'``.

    >>> band(2.0, 5.0, 'm'), band(160.0, float('inf'), 'm'), band(3000.0, 10000.0, 'm')
    ('2-5 m', 'below 160 m', '3,000 m - 10 km')
    """
    if not np.isfinite(bottom):
        return f"below {length(top, unit)}"
    low, high = length(top, unit), length(bottom, unit)
    low_value, _, low_unit = low.partition(" ")
    if low_unit == high.partition(" ")[2]:
        return f"{low_value}-{high}"
    return f"{low} - {high}"


def spread(values: Any, digits: int = 3) -> Optional[str]:
    """``'p10 - p90'`` of the finite values, or None when there are none.

    >>> spread([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    '2 - 10'
    """
    values = np.asarray(values, dtype=float).ravel()
    values = values[np.isfinite(values)]
    if not values.size:
        return None
    low, high = np.percentile(values, [10, 90])
    return f"{number(low, digits)} - {number(high, digits)}"


def relative(path: Any, output_dir: Optional[str]) -> str:
    """``path`` relative to the report, as a link in it is written."""
    if not output_dir:
        return os.path.basename(str(path))
    try:
        return os.path.relpath(str(path), str(output_dir)).replace(os.sep, "/")
    except ValueError:      # another drive
        return str(path).replace(os.sep, "/")


def source_name(path: Any) -> Optional[str]:
    """A data file with the folder it is in: ``'Sep06/project.tiw'``.

    The folder is half the name of an instrument's project - every TEM2Go
    survey's file is called ``project.tiw`` - so it is kept.

    >>> source_name('/data/TEM/Sep06/project.tiw')
    'Sep06/project.tiw'
    """
    text = str(path or "").replace("\\", "/").rstrip("/")
    if not text:
        return None
    parts = text.split("/")
    return "/".join(parts[-2:]) if len(parts) > 1 else parts[-1]


def plural(count: int, noun: str, many: Optional[str] = None) -> str:
    """``'1 site'``, ``'3 sites'``.

    >>> plural(1, 'site'), plural(3, 'site'), plural(2, 'sounding')
    ('1 site', '3 sites', '2 soundings')
    """
    return f"{count} {noun if count == 1 else (many or noun + 's')}"


def given(rows: Sequence[Sequence[Any]]) -> list:
    """The ``(label, value)`` rows that have a value to print."""
    return [row for row in rows if row[1] not in (None, "", "N/A")]
