"""Off-thread model-call worker for AQUAH.

``LlmCallWorker`` runs one :meth:`Provider.complete` on a ``QThread`` so the
network round-trip never blocks the UI. Because the user confirms every tool
call, the rest of the orchestration stays on the main thread in the chat panel.
Provider configuration and the actual SDK calls live in :mod:`providers`.
"""

from __future__ import annotations

from typing import Any, Dict, List

from PySide6.QtCore import QThread, Signal


class LlmCallWorker(QThread):
    """Run one ``provider.complete(system, messages, specs)`` off the UI thread."""

    succeeded = Signal(dict)
    failed = Signal(str)

    def __init__(self, provider: Any, system: str, messages: List[Dict[str, Any]], specs) -> None:
        super().__init__()
        self._provider = provider
        self._system = system
        self._messages = list(messages)
        self._specs = specs

    def run(self) -> None:  # noqa: D401 - QThread entry point
        try:
            out = self._provider.complete(self._system, self._messages, self._specs)
            self.succeeded.emit(out)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"{type(exc).__name__}: {exc}")
