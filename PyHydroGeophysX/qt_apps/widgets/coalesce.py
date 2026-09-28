"""One call per turn of the event loop, however many times it was asked for.

A view that redraws on every signal redraws as often as the signals come: a
slider dragged across ten depths rebuilds its figure ten times, and a caller
setting the colour range and then the data rebuilds it twice. ``draw_idle``
does not help there - it merges only the final paint, while the interpolation,
the statistics and the rebuilt artists before it ran once per signal. Asking
through a :class:`Coalesced` instead runs the work once, after the events that
asked for it have all been handled.
"""

from __future__ import annotations

from typing import Callable

from PySide6.QtCore import QObject, QTimer


class Coalesced(QObject):
    """Run ``fn`` once the current event has been handled, however often asked.

    ``request()`` schedules the call; every further request before it runs
    folds into the same one. ``flush()`` runs a pending call now, for code
    that reads the result straight away - a capture of the view.
    """

    def __init__(self, fn: Callable[[], None], parent: QObject) -> None:
        super().__init__(parent)
        self._fn = fn
        self._pending = False
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(0)
        self._timer.timeout.connect(self.flush)

    def request(self) -> None:
        self._pending = True
        self._timer.start()

    def flush(self) -> None:
        self._timer.stop()
        if self._pending:
            self._pending = False
            self._fn()

    @property
    def pending(self) -> bool:
        return self._pending
