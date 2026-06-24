"""Generic background worker for the workbench.

``TaskWorker`` runs an arbitrary callable off the UI thread and reports the
result through Qt signals, so any blocking step (file parsing, profile
extraction) can move off the main thread with the same pattern the inversion
modules already use. The bespoke inversion workers can be refactored onto this
later; it is not required.
"""

from __future__ import annotations

from typing import Any, Callable

from PySide6.QtCore import QThread, Signal


class TaskWorker(QThread):
    """Run ``fn(*args, **kwargs)`` off the UI thread.

    ``succeeded`` carries the return value, ``failed`` the error text, and
    ``logged`` optional progress strings. When ``with_log=True`` the callable is
    given a ``log`` keyword (a function taking one string) so it can report
    progress. :meth:`cancel` requests a cooperative stop: short tasks finish
    naturally and their result is simply dropped, so the UI is never updated
    from a cancelled run.
    """

    succeeded = Signal(object)
    failed = Signal(str)
    logged = Signal(str)

    def __init__(self, fn: Callable[..., Any], *args: Any, with_log: bool = False, **kwargs: Any) -> None:
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs
        self._with_log = with_log
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True
        self.requestInterruption()

    def is_cancelled(self) -> bool:
        return self._cancelled or self.isInterruptionRequested()

    def run(self) -> None:  # noqa: D401 - QThread entry point
        try:
            kwargs = dict(self._kwargs)
            if self._with_log:
                kwargs.setdefault("log", lambda m: self.logged.emit(str(m)))
            result = self._fn(*self._args, **kwargs)
            if self._cancelled:
                return
            self.succeeded.emit(result)
        except Exception as exc:  # noqa: BLE001
            if not self._cancelled:
                self.failed.emit(str(exc))
