"""Controls for a workflow run in progress: stopping it.

A long inversion ties up the machine. A workflow that runs in its own process
(``ProcessWorkflowWorker``) can be ended at once - inside a native solve as much
as in Python - and the page is free for the next run straight away.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from PySide6.QtWidgets import QHBoxLayout, QProgressBar, QPushButton, QWidget

from PyHydroGeophysX.qt_apps import theme

__all__ = ["StopButton", "progress_with_stop"]


class StopButton(QPushButton):
    """Stop for the run :meth:`attach` hands it.

    Hidden until a run that can be stopped is attached; pressing it ends the
    run's process and every process it started, and it hides again when the
    run finishes. ``on_stop`` is told first, with the operation the run was
    attached under, so the page can record the run as stopped by the user
    rather than failed (``BaseModule.stop_button`` wires that up).
    """

    def __init__(self, parent: Optional[QWidget] = None, *,
                 log: Optional[Callable[[str], Any]] = None,
                 on_stop: Optional[Callable[[str], Any]] = None,
                 what: str = "The inversion") -> None:
        super().__init__("Stop", parent)
        self._worker: Any = None
        self._operation = ""
        self._log = log
        self._on_stop = on_stop
        self._default_what = self._what = what
        self.setIcon(theme.icon("fa5s.stop"))
        self.setToolTip(
            "End the running computation now. Nothing it computed so far is kept; "
            "run it again to start over.")
        self.clicked.connect(self.stop)
        self.setVisible(False)

    def attach(self, worker: Any, operation: str = "", what: str = "") -> None:
        """Offer Stop for ``worker``'s run, if it is one that can be stopped.

        ``operation`` is the page's name for the run (its persisted-run
        operation id); ``what`` names this run in the log, in place of the
        name the button was made with.
        """
        self._worker = worker
        self._operation = str(operation)
        self._what = what or self._default_what
        if not callable(getattr(worker, "cancel", None)):
            self.setVisible(False)
            return
        # Only its own end hides the button: a run in stages (the time-lapse
        # QC, then the inversion) attaches the next stage before the first
        # one's thread has reported finishing.
        worker.finished.connect(lambda w=worker: self.detach() if self._worker is w else None)
        self.setEnabled(True)
        self.setVisible(True)

    def detach(self) -> None:
        """The run is over: nothing left to stop."""
        self._worker = None
        self.setVisible(False)

    def stop(self) -> None:
        worker = self._worker
        if worker is None or worker.is_cancelled():
            return
        self.setEnabled(False)           # the process may take a moment to die
        if self._on_stop is not None:
            self._on_stop(self._operation)
        worker.cancel()
        if self._log is not None:
            self._log(f"{self._what} was stopped.")


def progress_with_stop(progress: QProgressBar, stop: StopButton) -> QWidget:
    """A progress bar with its Stop button beside it, as one row."""
    row = QWidget()
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(progress, 1)
    layout.addWidget(stop)
    return row
