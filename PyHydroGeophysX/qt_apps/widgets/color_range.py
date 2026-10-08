"""The colour limits of a plot, typed in: "Lock range" and a lower and upper value.

Every colour-mapped view in the studio offers the same control, the one the
Resistivity model view started with. Unticked, the plot scales itself to what
it shows and the two boxes follow, so they always say what the colours mean;
ticked, the plot keeps the limits in the boxes, which can then be typed in.
Locking is what lets two surveys, time steps or lines be compared on one
scale: rescaled each to its own extremes, the difference between them is
exactly what the colours stop showing.

A view asks :meth:`ColorRange.limits` for the limits to draw with, handing it
the ones it would choose itself, and redraws on :attr:`ColorRange.changed`.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QWidget

from PyHydroGeophysX.qt_apps.qt_utils import PlainDoubleSpinBox

__all__ = ["ColorRange"]


class ColorRange(QWidget):
    """"Lock range" with the lower and upper colour limit beside it.

    ``positive`` keeps both limits above zero, for a quantity drawn on a log
    scale (resistivity, conductivity), where a limit of zero or below has no
    colour.
    """

    #: The limits to draw with changed: locked or unlocked, or a limit typed
    #: while locked.
    changed = Signal()

    def __init__(self, parent: Optional[QWidget] = None, *, positive: bool = False,
                 decimals: int = 3, what: str = "the colours") -> None:
        super().__init__(parent)
        self._positive = bool(positive)
        self._decimals = int(decimals)
        self._lock = QCheckBox("Lock range")
        self._lock.setToolTip(
            f"Keep {what} on the limits typed beside this instead of rescaling to "
            "each plot. Two surveys, time steps or lines can only be compared on "
            "one scale. Unticked, the boxes show the limits in use, so ticking it "
            "keeps what is on screen.")
        self._lock.toggled.connect(self._on_locked)
        self._lower = PlainDoubleSpinBox()
        self._upper = PlainDoubleSpinBox()
        floor = 1.0e-9 if self._positive else -1.0e12
        for box, name in ((self._lower, "Lower"), (self._upper, "Upper")):
            box.setRange(floor, 1.0e12)
            box.setDecimals(int(decimals))
            box.setMaximumWidth(96)
            box.setKeyboardTracking(False)      # redraw on commit, not per keystroke
            box.setEnabled(False)
            box.setToolTip(f"{name} colour limit, used while “Lock range” is ticked.")
            box.valueChanged.connect(self._on_limit_changed)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self._lock)
        layout.addWidget(self._lower)
        layout.addWidget(QLabel("–"))
        layout.addWidget(self._upper)

    # -- what the view asks ---------------------------------------------------
    def is_locked(self) -> bool:
        return self._lock.isChecked()

    def limits(self, low: float, high: float) -> Tuple[float, float]:
        """The limits to draw with, given the ones the view would choose itself.

        Locked with a usable pair typed in, those; otherwise ``(low, high)``,
        which the boxes then show.
        """
        if self.is_locked():
            lo, hi = float(self._lower.value()), float(self._upper.value())
            if hi > lo and (lo > 0.0 or not self._positive):
                return lo, hi
        self.track(low, high)
        return float(low), float(high)

    def track(self, low: float, high: float) -> None:
        """Show ``(low, high)`` in the boxes while unlocked, without a redraw."""
        if self.is_locked():
            return
        try:
            lo, hi = float(low), float(high)
        except (TypeError, ValueError):
            return
        if not (hi > lo):
            return
        self._write(lo, hi)

    def set_range(self, low: float, high: float, lock: bool = True) -> None:
        """Put ``(low, high)`` in the boxes and lock them (or not)."""
        lo, hi = float(low), float(high)
        if not (hi > lo):
            return
        self._write(lo, hi)
        if self._lock.isChecked() != bool(lock):
            self._lock.setChecked(bool(lock))   # its handler emits changed
        elif lock:
            self.changed.emit()

    def unlock(self) -> None:
        self._lock.setChecked(False)

    # -- internals ------------------------------------------------------------
    def _write(self, lo: float, hi: float) -> None:
        span = abs(hi - lo)
        # Decimals enough for the values at hand - raw amplitudes of 1e-6
        # showed as 0 at three - set before the values, which they round.
        if span > 0.0:
            needed = 3 - int(math.floor(math.log10(span)))
            decimals = min(max(self._decimals, needed), 12)
        else:
            decimals = self._decimals
        step = max(span / 50.0, 10.0 ** -decimals)
        for box, value in ((self._lower, lo), (self._upper, hi)):
            box.blockSignals(True)
            box.setDecimals(decimals)
            box.setSingleStep(step)
            box.setValue(value)
            box.blockSignals(False)

    def _on_locked(self, locked: bool) -> None:
        self._lower.setEnabled(locked)
        self._upper.setEnabled(locked)
        self.changed.emit()

    def _on_limit_changed(self, _value: float) -> None:
        if self.is_locked():
            self.changed.emit()
