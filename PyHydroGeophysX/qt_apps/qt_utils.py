"""Small Qt UI helpers shared across workbench modules."""

from __future__ import annotations

from typing import Callable

from PySide6.QtCore import QObject, QTimer


class Debouncer(QObject):
    """Coalesce rapid signals into a single delayed callback.

    Connect a widget's ``valueChanged`` / ``toggled`` signal to :meth:`trigger`;
    the wrapped ``callback`` runs once, ``interval_ms`` after the last trigger.
    A slider drag or fast spin therefore stops firing the heavy recompute on
    every step and only runs it when the user pauses.
    """

    def __init__(self, callback: Callable[[], None], interval_ms: int = 80, parent=None) -> None:
        super().__init__(parent)
        self._callback = callback
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(int(interval_ms))
        self._timer.timeout.connect(self._fire)

    def trigger(self, *_args) -> None:
        """(Re)start the timer. Extra signal arguments are ignored."""
        self._timer.start()

    def flush(self) -> None:
        """Run the callback immediately if a call is pending."""
        if self._timer.isActive():
            self._timer.stop()
            self._fire()

    def _fire(self) -> None:
        self._callback()
