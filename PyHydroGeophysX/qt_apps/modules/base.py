"""Base classes shared by all workbench module pages."""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional, Tuple

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QLabel, QTextEdit, QVBoxLayout, QWidget

LogFn = Callable[..., None]
#: One offer in a module's export menu: what it writes, and how to write it.
ExportAction = Tuple[str, Callable[[], Any]]


class BaseModule(QWidget):
    """A page in the central stacked widget.

    Subclasses get ``self.state`` (the shared :class:`WorkbenchState`) and a
    ``self.log(message, level)`` helper wired to the bottom log panel. They call
    ``self.report_result(dict)`` to publish their result into the bridge state.
    """

    resultsUpdated = Signal()
    #: Ask the main window to open the Mesh 3D module and load a mesh/volume file.
    viewMeshRequested = Signal(str)
    #: Ask the main window to switch to another module by key (cross-module handoff).
    navigateRequested = Signal(str)
    module_key = "base"
    module_title = "Module"

    def __init__(self, state: Any, log: LogFn, parent=None) -> None:
        super().__init__(parent)
        self.state = state
        self._log_fn = log
        self._workers: list = []

    def log(self, message: str, level: str = "info") -> None:
        try:
            self._log_fn(message, level)
        except Exception:
            pass

    def report_result(self, data: Dict[str, Any]) -> None:
        self.state.update_module_result(self.module_key, data)
        self.resultsUpdated.emit()

    def export_actions(self) -> List[ExportAction]:
        """What ``File > Export Results…`` should offer while this page is open.

        Every module writes its own products in its own formats, and before this
        hook the only way to reach them was to know which button on which tab
        did it. A module returns the exports that make sense *right now* — an
        entry it leaves out is one there is no result for yet — and the window
        runs the single offer directly or asks when there is more than one.
        """
        return []

    def begin_persisted_run(
        self, operation_id: str, workflow_id: str = "", *, label: str = ""
    ):
        """Allocate the sole durable directory for a module operation."""
        return self.state.begin_run(
            self.module_key, operation_id, workflow_id, label=label
        )

    def finish_persisted_run(self, result: Any, operation_id: str = "") -> None:
        self.state.finish_run(self.module_key, result, operation_id)

    def fail_persisted_run(self, error: str, operation_id: str = "") -> None:
        self.state.fail_run(self.module_key, error, operation_id)

    def cancel_persisted_run(self, error: str = "", operation_id: str = "") -> None:
        self.state.cancel_run(self.module_key, error, operation_id)

    # -- agent command interface --------------------------------------------
    def agent_describe(self) -> Dict[str, Any]:
        """Describe this module for the AQUAH assistant.

        Subclasses override this to advertise the actions they accept and their
        current parameter values. The default reports no actions, so the agent
        can still navigate to the module and read state, but cannot drive it.
        """
        return {
            "module": self.module_key,
            "title": self.module_title,
            "actions": [],
            "note": "This module has no agent actions yet; navigation and status only.",
        }

    def agent_apply(self, action: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Run an agent action on this module. Override in subclasses."""
        return {
            "status": "failed",
            "error": f"Module '{self.module_key}' does not support action '{action}'.",
        }

    def agent_view_context(self, view: str) -> Optional[Dict[str, Any]]:
        """Numbers that belong with a captured picture of ``view``, or ``None``.

        A model reading a plot has to match a marker against an axis label far
        away from it, and it gets the index wrong often enough to matter. The
        module already holds those values exactly, so a view that plots
        identifiable per-item values should return them here. The picture then
        answers what only vision can ("is this pick on the first arrival or on
        noise") while the numbers answer which item it was.
        """
        return None

    # -- worker lifecycle ----------------------------------------------------
    def register_worker(self, worker: Any) -> Any:
        """Track a QThread so the window can join it on shutdown.

        The reference is kept until the thread finishes, which both prevents the
        worker from being garbage-collected mid-run and lets :meth:`stop_workers`
        cancel/join anything still running when the app closes.
        """
        self._workers.append(worker)
        worker.finished.connect(lambda: self._drop_worker(worker))
        return worker

    def _drop_worker(self, worker: Any) -> None:
        if worker in self._workers:
            self._workers.remove(worker)

    def stop_workers(self, wait_ms: int = 5000) -> None:
        """Cancel cooperatively-interruptible workers and join running threads."""
        had_workers = bool(self._workers)
        for worker in list(self._workers):
            try:
                if hasattr(worker, "cancel"):
                    worker.cancel()
                if worker.isRunning():
                    worker.quit()
                    worker.wait(wait_ms)
            except Exception:  # noqa: BLE001 - shutdown best effort
                pass
        if had_workers:
            self.state.cancel_module_runs(
                self.module_key, "Workbench closed during computation"
            )


class HomePage(BaseModule):
    """Landing page shown when no specific module is selected."""

    module_key = "home"
    module_title = "Home"

    def __init__(self, state: Any, log: LogFn, parent=None) -> None:
        super().__init__(state, log, parent)
        layout = QVBoxLayout(self)
        title = QLabel("<h2>PyHydroGeophysX Professional Workbench</h2>")
        intro = QLabel(
            "Select a module from the project tree on the left.<br><br>"
            "<b>Geophysical Data Processing</b>: Seismic, ERT, Mesh 3D, EM, "
            "Gravity / Magnetics, and Joint Inversion.<br>"
            "<b>Hydro → Geophysics</b>: load hydrologic model outputs, pick a "
            "profile, set survey geometry, and run forward modeling.<br>"
            "<b>Geophy → Hydrology</b>: derive subsurface structure and "
            "hydrology from geophysics. <b>Seismic → Structure</b> builds a 3D "
            "structure (bedrock interface + velocity volume) from velocity "
            "sections; that structure can be handed to <b>ERT → Water Content</b>, "
            "which estimates water content and porosity per layer with Monte "
            "Carlo uncertainty."
        )
        intro.setWordWrap(True)
        self._summary = QTextEdit()
        self._summary.setReadOnly(True)
        layout.addWidget(title)
        layout.addWidget(intro)
        layout.addWidget(QLabel("<b>Session context</b>"))
        layout.addWidget(self._summary, stretch=1)
        self.refresh()

    def refresh(self) -> None:
        try:
            summary = self.state.context_summary()
        except Exception:
            summary = {}
        self._summary.setPlainText(json.dumps(summary, indent=2, default=str))


class PlaceholderModule(BaseModule):
    """A clean page used for not-yet-implemented modules and import failures."""

    def __init__(
        self,
        state: Any,
        log: LogFn,
        title: str,
        message: str,
        key: str = "placeholder",
        parent=None,
    ) -> None:
        super().__init__(state, log, parent)
        self.module_key = key
        self.module_title = title
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignTop)
        layout.addWidget(QLabel(f"<h3>{title}</h3>"))
        body = QLabel(message)
        body.setWordWrap(True)
        body.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(body)
        layout.addStretch(1)
