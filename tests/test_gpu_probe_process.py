import json
import inspect
from pathlib import Path

import pytest


def _qt_probe_dependencies():
    try:
        from PySide6.QtCore import QCoreApplication, QEventLoop, QTimer

        from PyHydroGeophysX.qt_apps.workers import ProcessProbeWorker
    except ImportError as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Qt stack unavailable: {exc}")
    return ProcessProbeWorker, QCoreApplication, QEventLoop, QTimer


def test_process_probe_worker_runs_json_module_in_fresh_interpreter() -> None:
    ProcessProbeWorker, QCoreApplication, QEventLoop, QTimer = (
        _qt_probe_dependencies()
    )
    module_name = "tests._gpu_probe_fixture"
    project_root = Path(__file__).resolve().parents[1]

    app = QCoreApplication.instance() or QCoreApplication([])
    loop = QEventLoop()
    results = []
    errors = []
    worker = ProcessProbeWorker(
        module_name,
        timeout_ms=5000,
        working_directory=project_root,
    )
    worker.succeeded.connect(results.append)
    worker.failed.connect(errors.append)
    worker.finished.connect(loop.quit)
    QTimer.singleShot(7000, loop.quit)
    worker.start()
    loop.exec()
    app.processEvents()

    if errors and errors[0].startswith("Could not start probe process: pipe:"):
        pytest.skip("This test sandbox blocks QProcess pipe creation")
    assert errors == []
    assert len(results) == 1
    assert results[0]["device"] == "test GPU"
    assert int(results[0]["pid"]) != 0
    assert Path(worker.process.program()).name.lower().startswith("python")
    assert worker.process.arguments() == ["-m", module_name]
    assert worker.process.processEnvironment().value("PYTHONIOENCODING") == "utf-8"


def test_process_probe_worker_reports_structured_child_failure() -> None:
    ProcessProbeWorker, _QCoreApplication, _QEventLoop, _QTimer = (
        _qt_probe_dependencies()
    )
    from PySide6.QtCore import QProcess

    worker = ProcessProbeWorker("unused.probe")
    worker._stdout.extend(
        b'{"ok": false, "error": "OSError: shm.dll WinError 127"}\n'
    )
    errors = []
    worker.failed.connect(errors.append)

    worker._on_finished(0, QProcess.ExitStatus.NormalExit)

    assert errors == ["OSError: shm.dll WinError 127"]


def test_adtlert_probe_cli_emits_success_json(monkeypatch, capsys) -> None:
    from PyHydroGeophysX.qt_apps import adtlert_probe

    expected = {
        "device": "NVIDIA RTX test",
        "forward_solver": "cudss",
        "linearized_solver": "gpu_cgls",
    }
    monkeypatch.setattr(adtlert_probe, "probe_adtlert_runtime", lambda: expected)

    assert adtlert_probe.main() == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload == {"ok": True, "result": expected}


def test_adtlert_probe_cli_serializes_native_import_error(monkeypatch, capsys) -> None:
    from PyHydroGeophysX.qt_apps import adtlert_probe

    def fail_probe():
        raise OSError("shm.dll WinError 127")

    monkeypatch.setattr(adtlert_probe, "probe_adtlert_runtime", fail_probe)

    assert adtlert_probe.main() == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "ok": False,
        "error": "OSError: shm.dll WinError 127",
    }


def test_ert_gpu_preflight_no_longer_imports_torch_in_gui_module() -> None:
    from PyHydroGeophysX.qt_apps.modules.ert_processing import ERTProcessingModule

    assert not hasattr(ERTProcessingModule, "_probe_adtlert_runtime")
    source = inspect.getsource(ERTProcessingModule._on_engine_changed)
    assert "ProcessProbeWorker" in source
    assert "TaskWorker" not in source
