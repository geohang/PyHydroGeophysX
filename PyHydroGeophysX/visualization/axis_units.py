"""Length units and the vertical axis of a section: metres or feet, elevation or depth.

Every model in the package is computed in metres, and it stays in metres: the
meshes, the contour grids, the clip polygons and every exported file. What this
module changes is only what an axis *shows*. A figure drawn "in feet" keeps its
data in metres and gets a tick locator and formatter that place round numbers
of feet along the axis, so a pyGIMLi mesh drawing, a filled contour and a clip
path all line up exactly as before - nothing is re-projected.

The same mechanism turns the vertical axis of a section into depth. A survey
read without elevations puts every electrode at z = 0 and the model below it at
negative z, and labelling that axis "Elevation" presents a made-up datum as a
measured one. When the ground surface is flat at zero, :func:`set_section_axes`
labels the axis "Depth" and shows the values positive downward, which is what
pyGIMLi itself does for such a mesh. A section with real topography keeps
"Elevation".

The unit is a package-wide preference, like a matplotlib rcParam, so one choice
reaches every figure: :func:`set_length_unit` sets it, :func:`length_unit`
changes it for a block, and every plotting function also takes ``length_unit``
to override it for one call.

Examples
--------
>>> from PyHydroGeophysX.visualization.axis_units import length_label, length_unit
>>> length_label("Distance")
'Distance (m)'
>>> with length_unit("feet"):
...     length_label("Distance")
'Distance (ft)'
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Optional, Tuple

import numpy as np

__all__ = [
    "FEET_PER_METRE",
    "LENGTH_UNITS",
    "normalize_length_unit",
    "get_length_unit",
    "set_length_unit",
    "length_unit",
    "length_factor",
    "to_display_length",
    "length_label",
    "section_has_elevation",
    "resolve_vertical",
    "vertical_axis",
    "set_length_axis",
    "set_section_axes",
    "section_label_pair",
    "pyqtgraph_length_axis",
]

#: International foot: exactly 0.3048 m.
FEET_PER_METRE = 1.0 / 0.3048

#: The units an axis can be shown in.
LENGTH_UNITS: Tuple[str, ...] = ("m", "ft")

_ALIASES = {
    "m": "m", "meter": "m", "meters": "m", "metre": "m", "metres": "m",
    "ft": "ft", "feet": "ft", "foot": "ft",
}

_FACTORS = {"m": 1.0, "ft": FEET_PER_METRE}

_VERTICAL = ("auto", "elevation", "depth")

_default_unit = "m"


def normalize_length_unit(unit: Optional[str] = None) -> str:
    """``'m'`` or ``'ft'`` for any spelling of either; None is the current default.

    Examples
    --------
    >>> normalize_length_unit("Feet")
    'ft'
    >>> normalize_length_unit(" metres ")
    'm'
    """
    if unit is None:
        return _default_unit
    key = str(unit).strip().lower()
    if key not in _ALIASES:
        raise ValueError(
            f"Unknown length unit {unit!r}; use one of {', '.join(LENGTH_UNITS)}.")
    return _ALIASES[key]


def get_length_unit() -> str:
    """The unit figures are drawn in when a call does not name one."""
    return _default_unit


def set_length_unit(unit: str) -> str:
    """Draw every figure in ``unit`` from now on; returns the previous unit."""
    global _default_unit
    previous = _default_unit
    _default_unit = normalize_length_unit(unit)
    return previous


@contextmanager
def length_unit(unit: Optional[str]) -> Iterator[str]:
    """Draw in ``unit`` inside the block, then go back. None leaves it as it is."""
    if unit is None:
        yield _default_unit
        return
    previous = set_length_unit(unit)
    try:
        yield _default_unit
    finally:
        set_length_unit(previous)


def length_factor(unit: Optional[str] = None) -> float:
    """Displayed units per metre: 1 for metres, 3.28084 for feet."""
    return _FACTORS[normalize_length_unit(unit)]


def to_display_length(values: Any, unit: Optional[str] = None) -> Any:
    """A length in metres - scalar or array - converted to ``unit``.

    For numbers written into titles and annotations, which no axis formatter
    reaches.

    Examples
    --------
    >>> round(to_display_length(10.0, "ft"), 3)
    32.808
    """
    factor = length_factor(unit)
    if np.isscalar(values):
        return float(values) * factor
    return np.asarray(values, dtype=float) * factor


def length_label(name: str, unit: Optional[str] = None) -> str:
    """``'Distance (ft)'``: an axis name with the unit it is shown in."""
    return f"{name} ({normalize_length_unit(unit)})"


# ---------------------------------------------------------------------------
# Elevation or depth
# ---------------------------------------------------------------------------

def _surface_z(mesh: Any = None, surface: Any = None, z: Any = None) -> Optional[np.ndarray]:
    """Ground-surface elevations from whichever of the three the caller has."""
    if surface is not None:
        arr = np.asarray(surface[1] if isinstance(surface, tuple) else surface, dtype=float)
        if arr.ndim == 2 and arr.shape[1] >= 2:
            arr = arr[:, 1]
        return arr.ravel()
    if mesh is not None:
        try:
            from PyHydroGeophysX.core.section_geometry import surface_line

            return np.asarray(surface_line(mesh)[1], dtype=float)
        except Exception:  # noqa: BLE001 - an unreadable mesh keeps "Elevation"
            return None
    if z is not None:
        # Only the section's coordinates: the top of them is all that says where
        # the ground is.
        arr = np.asarray(z, dtype=float).ravel()
        arr = arr[np.isfinite(arr)]
        return arr[[int(np.argmax(arr))]] if arr.size else None
    return None


def section_has_elevation(mesh: Any = None, *, surface: Any = None, z: Any = None) -> bool:
    """Whether a section's vertical coordinate is a real elevation.

    False when the ground surface is flat at zero - the geometry a survey gets
    when it was read without elevations. Give the ``mesh``, the ``surface``
    (``(x, z)`` tuple, ``(n, 2)`` array or the surface z values) or just the
    section's ``z`` coordinates, whose top is then taken as the surface. When
    none of them can be read the answer is True, which keeps the label a
    section has always had.

    Examples
    --------
    >>> section_has_elevation(surface=[0.0, 0.0, 0.0])
    False
    >>> section_has_elevation(surface=[312.4, 310.9, 309.0])
    True
    >>> section_has_elevation(z=[0.0, -5.0, -20.0])
    False
    """
    return _has_elevation(_surface_z(mesh, surface, z))


def _has_elevation(top: Optional[np.ndarray]) -> bool:
    if top is None:
        return True
    top = top[np.isfinite(top)]
    if top.size == 0:
        return True
    return not bool(np.all(np.abs(top) <= 1.0e-6))


def _check_vertical(vertical: Optional[str]) -> str:
    mode = str(vertical or "auto").strip().lower()
    if mode not in _VERTICAL:
        raise ValueError(f"vertical must be one of {', '.join(_VERTICAL)}, not {vertical!r}.")
    return mode


def resolve_vertical(vertical: str = "auto", *, mesh: Any = None, surface: Any = None,
                     z: Any = None) -> str:
    """``'elevation'`` or ``'depth'``: what the vertical axis of a section shows.

    ``'auto'`` gives depth when :func:`section_has_elevation` says the section
    has no elevation, and elevation otherwise.
    """
    return vertical_axis(vertical, mesh=mesh, surface=surface, z=z)[0]


def vertical_axis(vertical: str = "auto", *, mesh: Any = None, surface: Any = None,
                  z: Any = None) -> Tuple[str, Optional[float]]:
    """``(mode, depth_reference)`` for a section's vertical axis.

    ``mode`` is as :func:`resolve_vertical` decides it. For depth,
    ``depth_reference`` is the level depth is measured down from - the top of
    the ground surface, 0 when it cannot be read; for elevation it is None.
    The surface is read once, so a view can keep the pair and pass it back to
    :func:`set_section_axes` on every redraw.

    Examples
    --------
    >>> vertical_axis(z=[0.0, -4.0, -12.0])
    ('depth', 0.0)
    >>> vertical_axis(surface=[102.0, 101.5])
    ('elevation', None)
    """
    mode = _check_vertical(vertical)
    top = None
    if mode != "elevation":
        top = _surface_z(mesh, surface, z)
    if mode == "auto":
        mode = "elevation" if _has_elevation(top) else "depth"
    if mode == "elevation":
        return mode, None
    finite = None if top is None else top[np.isfinite(top)]
    return mode, float(np.max(finite)) if finite is not None and finite.size else 0.0


# ---------------------------------------------------------------------------
# Axis ticks shown in another unit or as depth
# ---------------------------------------------------------------------------

def _transforms(factor: float, depth_reference: Optional[float]):
    """``(to_display, from_display)`` between data metres and what the axis shows."""
    if depth_reference is None:
        return (lambda v: np.asarray(v, dtype=float) * factor,
                lambda d: np.asarray(d, dtype=float) / factor)
    ref = float(depth_reference)
    return (lambda v: (ref - np.asarray(v, dtype=float)) * factor,
            lambda d: ref - np.asarray(d, dtype=float) / factor)


def _display_ticker(to_display, from_display):
    """A matplotlib locator and formatter that tick in displayed units.

    Built on first use so that importing this module never imports matplotlib.
    """
    from matplotlib.ticker import Formatter, Locator, MaxNLocator

    class DisplayLocator(Locator):
        """Round numbers of the displayed unit, placed at their data positions."""

        phgx_display = True

        def __init__(self) -> None:
            self._base = MaxNLocator(nbins="auto", steps=[1, 2, 2.5, 5, 10])

        def set_axis(self, axis) -> None:
            super().set_axis(axis)
            self._base.set_axis(axis)

        def __call__(self):
            vmin, vmax = self.axis.get_view_interval()
            return self.tick_values(vmin, vmax)

        def tick_values(self, vmin, vmax):
            lo, hi = sorted(float(v) for v in to_display([vmin, vmax]))
            ticks = np.asarray(self._base.tick_values(lo, hi), dtype=float)
            return np.sort(from_display(ticks))

    class DisplayFormatter(Formatter):
        """Tick labels in the displayed unit, with as many decimals as they need."""

        phgx_display = True

        def __init__(self) -> None:
            self._decimals = 0

        def set_locs(self, locs) -> None:
            super().set_locs(locs)
            shown = np.asarray(to_display(locs), dtype=float)
            shown = shown[np.isfinite(shown)]
            self._decimals = 0
            if shown.size:
                scale = max(float(np.max(np.abs(shown))), 1.0)
                for decimals in range(7):
                    if np.allclose(np.round(shown, decimals), shown,
                                   rtol=0.0, atol=scale * 1.0e-9):
                        self._decimals = decimals
                        break
                else:
                    self._decimals = 6

        def _text(self, value: float, decimals: int) -> str:
            shown = float(to_display(value))
            if abs(shown) < 0.5 * 10.0 ** (-decimals):
                shown = 0.0          # no "-0" on the surface line
            return self.fix_minus(f"{shown:.{decimals}f}")

        def __call__(self, x, pos=None) -> str:
            return self._text(x, self._decimals)

        def format_data_short(self, value) -> str:
            # The cursor readout: finer than the ticks, in the same unit.
            return self._text(value, max(self._decimals, 2))

        def format_data(self, value) -> str:
            return self._text(value, max(self._decimals, 3))

    return DisplayLocator(), DisplayFormatter()


def _axis(ax: Any, which: str):
    which = str(which).lower()
    if which not in ("x", "y", "z") or (which == "z" and not hasattr(ax, "zaxis")):
        raise ValueError(f"axis must be 'x', 'y' or, on 3-D axes, 'z'; not {which!r}")
    return getattr(ax, f"{which}axis")


def set_length_axis(ax: Any, which: str = "x", name: str = "Distance", *,
                    unit: Optional[str] = None,
                    depth_reference: Optional[float] = None,
                    labelled: bool = True, **label_kw: Any) -> str:
    """Label one axis of ``ax`` as a length and tick it in ``unit``.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    which : ``'x'``, ``'y'`` or, on 3-D axes, ``'z'``
    name : str
        The quantity, e.g. ``'Distance'`` or ``'Depth'``; the unit is appended.
    unit : str, optional
        ``'m'`` or ``'ft'`` (any spelling). Defaults to :func:`get_length_unit`.
    depth_reference : float, optional
        Show ``depth_reference - value`` instead of the value, positive
        downward: the data are elevations and the axis reads depth below this
        level.
    labelled : bool
        Set the label. False leaves it alone - an inner panel of a grid, whose
        ticks still have to be in the right unit.
    **label_kw
        Passed to ``set_xlabel``/``set_ylabel`` (font size, family).

    Returns
    -------
    str
        The label text.
    """
    resolved = normalize_length_unit(unit)
    axis = _axis(ax, which)
    label = length_label(name, resolved)
    factor = _FACTORS[resolved]
    if factor != 1.0 or depth_reference is not None:
        locator, formatter = _display_ticker(*_transforms(factor, depth_reference))
        axis.set_major_locator(locator)
        axis.set_major_formatter(formatter)
    else:
        # Plain metres. An axis drawn in feet or as depth before - a view that
        # reuses its axes - gets matplotlib's own ticks back, and so does one
        # pyGIMLi relabelled for what it took to be a depth section, so the
        # numbers agree with the label.
        from matplotlib.ticker import AutoLocator, FuncFormatter, ScalarFormatter

        if getattr(axis.get_major_locator(), "phgx_display", False):
            axis.set_major_locator(AutoLocator())
        formatter = axis.get_major_formatter()
        if getattr(formatter, "phgx_display", False) or isinstance(formatter, FuncFormatter):
            axis.set_major_formatter(ScalarFormatter())
    if labelled:
        getattr(ax, f"set_{which.lower()}label")(label, **label_kw)
    return label


def set_section_axes(ax: Any, *, mesh: Any = None, surface: Any = None, z: Any = None,
                     vertical: str = "auto", unit: Optional[str] = None,
                     xlabel: str = "Distance",
                     depth_reference: Optional[float] = None,
                     label_x: bool = True, label_y: bool = True,
                     elevation_name: str = "Elevation", depth_name: str = "Depth",
                     **label_kw: Any) -> str:
    """Label and tick both axes of a 2-D section; returns ``'elevation'`` or ``'depth'``.

    The horizontal axis becomes ``xlabel`` in ``unit``. The vertical one is
    ``elevation_name`` or, when the section has no elevation (see
    :func:`resolve_vertical`), ``depth_name`` shown positive downward from
    ``depth_reference`` - the top of the ground surface unless given.

    ``label_x``/``label_y`` False keep the ticks in the right unit but leave
    that label off, for the inner panels of a grid.
    """
    if _check_vertical(vertical) == "depth" and depth_reference is not None:
        mode = "depth"               # nothing left to read off the geometry
    else:
        mode, reference = vertical_axis(vertical, mesh=mesh, surface=surface, z=z)
        if depth_reference is None:
            depth_reference = reference
    set_length_axis(ax, "x", xlabel, unit=unit, labelled=label_x, **label_kw)
    if mode == "depth":
        set_length_axis(ax, "y", depth_name, unit=unit, depth_reference=depth_reference,
                        labelled=label_y, **label_kw)
    else:
        set_length_axis(ax, "y", elevation_name, unit=unit, labelled=label_y, **label_kw)
    return mode


def pyqtgraph_length_axis(plot: Any, side: str, name: str, *,
                          unit: Optional[str] = None) -> str:
    """Label a pyqtgraph ``PlotItem``/``PlotWidget`` axis as a length in ``unit``.

    pyqtgraph scales an axis's tick values itself (``AxisItem.setScale``) and
    picks round numbers in the scaled unit, so the data stay in metres here as
    well. Returns the label text.
    """
    resolved = normalize_length_unit(unit)
    label = length_label(name, resolved)
    axis = plot.getAxis(side)
    axis.setScale(_FACTORS[resolved])
    plot.setLabel(side, label)
    return label


def section_label_pair(vertical: str = "auto", *, unit: Optional[str] = None,
                       xlabel: str = "Distance", mesh: Any = None, surface: Any = None,
                       z: Any = None) -> Tuple[str, str]:
    """``(xlabel, ylabel)`` text for a section, for code that only sets labels."""
    mode = resolve_vertical(vertical, mesh=mesh, surface=surface, z=z)
    return (length_label(xlabel, unit),
            length_label("Depth" if mode == "depth" else "Elevation", unit))

