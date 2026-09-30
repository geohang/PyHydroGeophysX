"""One length unit for every plot in the studio: metres or feet.

The choice lives in the View menu and applies to every section, map and profile
the studio draws, the way the colour map of a quantity does - but for the whole
studio at once, since nobody wants distances in feet on one page and metres on
the next. It is remembered between sessions.

The data never change unit. Every model, mesh and exported file stays in
metres; a view drawn "in feet" only ticks and labels its axes in feet (see
:mod:`PyHydroGeophysX.visualization.axis_units`), so switching back and forth
cannot round anything off.

A view that draws a length axis connects its redraw to :func:`notifier`'s
``changed`` signal and asks :func:`current` - or simply leaves the unit out of
its ``axis_units`` calls, which then use the same preference.
"""

from __future__ import annotations

from typing import Any, Optional

from PySide6.QtCore import QObject, QSettings, Signal

from PyHydroGeophysX.visualization import axis_units

__all__ = ["notifier", "current", "set_unit", "restore", "pyqtgraph_axis"]

_SETTINGS_KEY = "main/lengthUnit"


class LengthUnitNotifier(QObject):
    """Emits ``changed(unit)`` when the studio's length unit changes."""

    changed = Signal(str)


_NOTIFIER: Optional[LengthUnitNotifier] = None


def notifier() -> LengthUnitNotifier:
    """The one notifier every view connects its redraw to."""
    global _NOTIFIER
    if _NOTIFIER is None:
        _NOTIFIER = LengthUnitNotifier()
    return _NOTIFIER


def current() -> str:
    """``'m'`` or ``'ft'``: the unit plots are drawn in now."""
    return axis_units.get_length_unit()


def set_unit(unit: str, *, remember: bool = True) -> str:
    """Draw every plot in ``unit`` from now on; returns the unit in force.

    Views are told only when the unit actually changes, so choosing the one
    already in force redraws nothing.
    """
    resolved = axis_units.normalize_length_unit(unit)
    if remember:
        QSettings("PyHydroGeophysX", "Studio").setValue(_SETTINGS_KEY, resolved)
    if resolved != axis_units.get_length_unit():
        axis_units.set_length_unit(resolved)
        notifier().changed.emit(resolved)
    return resolved


def restore() -> str:
    """Put back the unit chosen in an earlier session; metres if there is none."""
    saved = QSettings("PyHydroGeophysX", "Studio").value(_SETTINGS_KEY)
    try:
        return set_unit(str(saved), remember=False) if saved else current()
    except ValueError:        # a value this version does not know
        return current()


def pyqtgraph_axis(plot: Any, side: str, name: str) -> str:
    """Label a pyqtgraph axis as a length in the studio's unit, ticking in it.

    Call it again from the view's handler for :func:`notifier`'s ``changed``
    signal; the data stay in metres either way.
    """
    return axis_units.pyqtgraph_length_axis(plot, side, name)
