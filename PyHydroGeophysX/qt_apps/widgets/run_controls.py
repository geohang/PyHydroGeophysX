"""Controls for a workflow run in progress: pausing it, and saying that it is paused.

A long inversion ties up the machine, and the only way to get it back used to be
cancelling the run and starting it again from the beginning. A workflow that runs
in its own process (``ProcessWorkflowWorker``) can instead be frozen where it
stands and continued later from the same instruction, with nothing recomputed.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from PySide6.QtWidgets import QHBoxLayout, QProgressBar, QPushButton, QWidget

from PyHydroGeophysX.qt_apps import theme

__all__ = ["PauseButton", "progress_with_pause"]


class PauseButton(QPushButton):
    """Pause / Resume for the run :meth:`attach` hands it.

    Hidden until a run that can pause is attached, it follows the run's
    ``pausedChanged`` signal and hides again when the run finishes. While the
    run is paused the progress bar says so, rather than animating as though
    the computation were still going.
    """

    def __init__(self, parent: Optional[QWidget] = None, *,
                 progress: Optional[QProgressBar] = None,
                 log: Optional[Callable[[str], Any]] = None,
                 what: str = "The inversion") -> None:
        super().__init__("Pause", parent)
        self._worker: Any = None
        self._progress = progress
        self._log = log
        self._what = what
        self._saved: Optional[tuple] = None
        self.setToolTip(
            "Freeze the running computation where it stands, and continue it later "
            "from the same point; nothing is recomputed. It keeps its memory while "
            "paused, and closing the studio still ends it.")
        self.clicked.connect(self._toggle)
        self._show(False)
        self.setVisible(False)

    def attach(self, worker: Any) -> None:
        """Offer Pause for ``worker``'s run, if it is one that can be paused."""
        self._worker = worker
        if not callable(getattr(worker, "pause", None)):
            self.setVisible(False)
            return
        worker.pausedChanged.connect(self._on_paused)
        worker.finished.connect(self.detach)
        self._show(False)
        self.setVisible(True)

    def detach(self) -> None:
        """The run is over: nothing to pause, and nothing left paused on screen."""
        self._worker = None
        self._restore_progress()
        self._show(False)
        self.setVisible(False)

    def _toggle(self) -> None:
        worker = self._worker
        if worker is None:
            return
        if worker.is_paused():
            worker.resume()
        else:
            worker.pause()

    def _on_paused(self, paused: bool) -> None:
        if self._worker is None:
            return
        self._show(paused)
        if paused:
            self._hold_progress()
        else:
            self._restore_progress()
        if self._log is not None:
            self._log(f"{self._what} is paused; press Resume to continue it from the "
                      "same point." if paused else f"{self._what} continues.")

    def _show(self, paused: bool) -> None:
        self.setText("Resume" if paused else "Pause")
        self.setIcon(theme.icon("fa5s.play" if paused else "fa5s.pause"))

    def _hold_progress(self) -> None:
        bar = self._progress
        if bar is None or self._saved is not None:
            return
        self._saved = (bar.minimum(), bar.maximum(), bar.value(), bar.format())
        if bar.maximum() == 0:
            # A busy indicator keeps animating; a paused run is not busy.
            bar.setRange(0, 1)
            bar.setValue(0)
            bar.setFormat("Paused")
        else:
            bar.setFormat(f"Paused - {bar.format()}")

    def _restore_progress(self) -> None:
        bar, saved = self._progress, self._saved
        self._saved = None
        if bar is None or saved is None:
            return
        minimum, maximum, value, text = saved
        # Output written just before the pause can still arrive after it and
        # move the bar on; that newer state is kept, not rolled back.
        if maximum == 0:
            if (bar.minimum(), bar.maximum(), bar.format()) == (0, 1, "Paused"):
                bar.setRange(minimum, maximum)
                bar.setValue(value)
                bar.setFormat(text)
        elif bar.format().startswith("Paused - "):
            bar.setFormat(bar.format()[len("Paused - "):])


def progress_with_pause(progress: QProgressBar, pause: PauseButton) -> QWidget:
    """A progress bar with its Pause button beside it, as one form row."""
    row = QWidget()
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(progress, 1)
    layout.addWidget(pause)
    return row
