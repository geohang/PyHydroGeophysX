"""Regressions for EM line-inversion worker registration."""

from __future__ import annotations

import os

import pytest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
try:
    import PySide6.QtGui  # noqa: F401
except ImportError as exc:  # pragma: no cover - environment dependent
    pytest.skip(f"Qt stack unavailable: {exc}", allow_module_level=True)

from PySide6.QtCore import QThread
from PySide6.QtWidgets import QApplication, QSpinBox

from PyHydroGeophysX.qt_apps.modules.em_processing import EMProcessingModule


def test_cpu_thread_control_does_not_replace_worker_registry() -> None:
    app = QApplication.instance() or QApplication([])
    module = EMProcessingModule(None, lambda *_args: None)
    worker = QThread()

    assert isinstance(module._workers, list)
    assert isinstance(module._parallel_workers_spin, QSpinBox)
    module.register_worker(worker)
    assert worker in module._workers

    module._drop_worker(worker)
    module.close()
    app.processEvents()
