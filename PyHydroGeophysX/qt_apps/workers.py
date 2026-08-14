"""Generic background worker for the workbench.

``TaskWorker`` handles non-workflow support tasks such as file parsing and
preview generation. ``WorkflowWorker`` runs cooperative workflows in a Qt
thread, while ``ProcessProbeWorker`` and ``ProcessWorkflowWorker`` isolate
native-library probes and workflows in separate Python processes.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys
from typing import Any, Callable

from PySide6.QtCore import (
    QObject,
    QProcess,
    QProcessEnvironment,
    QThread,
    QTimer,
    Signal,
)

from PyHydroGeophysX._internal.optional_dependencies import BackendUnavailable
from PyHydroGeophysX.workflows import (
    RunContext,
    WorkflowRunResult,
    WorkflowSpec,
    run_workflow,
)


def _error_message(exc: Exception) -> str:
    if isinstance(exc, BackendUnavailable):
        return f"Backend unavailable: {exc}"
    return str(exc)


def _active_console_python() -> Path:
    """Return the console Python launcher belonging to the active environment."""
    prefix_python = Path(sys.prefix) / "Scripts" / "python.exe"
    executable = prefix_python if prefix_python.is_file() else Path(sys.executable)
    if executable.name.lower().startswith("pythonw"):
        console_python = executable.with_name(
            executable.name.replace("pythonw", "python", 1)
        )
        if console_python.is_file():
            executable = console_python
    return executable


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
                self.failed.emit(_error_message(exc))


class WorkflowWorker(QThread):
    """Execute a serializable workflow without coupling its handler to Qt."""

    succeeded = Signal(object)
    failed = Signal(str)
    logged = Signal(str)

    def __init__(self, spec: WorkflowSpec, context: RunContext) -> None:
        super().__init__()
        self._spec = spec
        self._context = context
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True
        self.requestInterruption()

    def is_cancelled(self) -> bool:
        return self._cancelled or self.isInterruptionRequested()

    def run(self) -> None:  # noqa: D401 - QThread entry point
        try:
            self._context.progress = lambda message: self.logged.emit(str(message))
            self._context.cancelled = self.is_cancelled
            result = run_workflow(self._spec, self._context)
            if not self.is_cancelled():
                self.succeeded.emit(result)
        except Exception as exc:  # noqa: BLE001
            if not self.is_cancelled():
                self.failed.emit(_error_message(exc))


class ProcessProbeWorker(QObject):
    """Run a JSON-emitting diagnostic module in a clean Python process.

    GPU libraries load several native Windows DLLs. Importing them from a
    ``QThread`` still uses the workbench process and can therefore inherit a
    conflicting OpenMP DLL that another visualization dependency loaded first.
    A fresh interpreter has its own DLL namespace and matches the isolation used
    by the actual ERT inversion workflow.

    The child module must print a final JSON object with either
    ``{"ok": true, "result": {...}}`` or ``{"ok": false, "error": "..."}``.
    Earlier stdout/stderr is retained only as a diagnostic if the contract is
    not satisfied.
    """

    succeeded = Signal(object)
    failed = Signal(str)
    finished = Signal()

    def __init__(
        self,
        module: str,
        *,
        timeout_ms: int = 60000,
        working_directory: str | Path | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.module = str(module)
        self.timeout_ms = max(1000, int(timeout_ms))
        self._cancelled = False
        self._finished = False
        self._timed_out = False
        self._stdout = bytearray()
        self._stderr = bytearray()

        self.process = QProcess(self)
        if working_directory is not None:
            self.process.setWorkingDirectory(str(Path(working_directory).resolve()))
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("PYTHONUTF8", "1")
        environment.insert("PYTHONIOENCODING", "utf-8")
        self.process.setProcessEnvironment(environment)
        self.process.setProgram(str(_active_console_python()))
        self.process.setArguments(["-m", self.module])
        self.process.readyReadStandardOutput.connect(self._read_stdout)
        self.process.readyReadStandardError.connect(self._read_stderr)
        self.process.errorOccurred.connect(self._on_process_error)
        self.process.finished.connect(self._on_finished)

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._on_timeout)

    def start(self) -> None:
        self._timer.start(self.timeout_ms)
        self.process.start()

    def cancel(self) -> None:
        self._cancelled = True
        self._timer.stop()
        if self.isRunning():
            self.process.terminate()
            QTimer.singleShot(2000, self._kill_if_running)

    def quit(self) -> None:
        """QThread-compatible shutdown hook used by ``BaseModule``."""
        self.cancel()

    def wait(self, timeout_ms: int = 30000) -> bool:
        return bool(self.process.waitForFinished(int(timeout_ms)))

    def isRunning(self) -> bool:  # noqa: N802 - mirror QThread's public API
        return self.process.state() != QProcess.ProcessState.NotRunning

    def is_cancelled(self) -> bool:
        return self._cancelled

    def _kill_if_running(self) -> None:
        if self.isRunning():
            self.process.kill()

    def _read_stdout(self) -> None:
        self._stdout.extend(bytes(self.process.readAllStandardOutput()))

    def _read_stderr(self) -> None:
        self._stderr.extend(bytes(self.process.readAllStandardError()))

    def _on_timeout(self) -> None:
        if self._finished:
            return
        self._timed_out = True
        if self.isRunning():
            self.process.kill()
        else:
            self._finish_with_error(
                f"Probe process timed out after {self.timeout_ms / 1000:g} seconds."
            )

    def _on_process_error(self, error: QProcess.ProcessError) -> None:
        if error == QProcess.ProcessError.FailedToStart and not self._finished:
            self._finish_with_error(
                f"Could not start probe process: {self.process.errorString()}"
            )

    def _finish_with_error(self, message: str) -> None:
        if self._finished:
            return
        self._finished = True
        self._timer.stop()
        if not self._cancelled:
            self.failed.emit(message)
        self.finished.emit()

    @staticmethod
    def _last_json_object(raw: bytes) -> dict[str, Any]:
        text = raw.decode("utf-8", errors="replace")
        for line in reversed(text.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict) and "ok" in payload:
                return payload
        raise ValueError("probe did not return its JSON result")

    def _on_finished(
        self, exit_code: int, _exit_status: QProcess.ExitStatus
    ) -> None:
        if self._finished:
            return
        self._timer.stop()
        self._read_stdout()
        self._read_stderr()
        if self._cancelled:
            self._finished = True
            self.finished.emit()
            return
        if self._timed_out:
            self._finish_with_error(
                f"Probe process timed out after {self.timeout_ms / 1000:g} seconds."
            )
            return
        if int(exit_code) != 0:
            unsigned = int(exit_code) & 0xFFFFFFFF
            detail = self._stderr.decode("utf-8", errors="replace").strip()
            suffix = f" {detail[-2000:]}" if detail else ""
            self._finish_with_error(
                f"Probe process exited with code {int(exit_code)} "
                f"(0x{unsigned:08X}).{suffix}"
            )
            return
        try:
            payload = self._last_json_object(bytes(self._stdout))
        except Exception as exc:  # noqa: BLE001 - malformed child output
            detail = self._stderr.decode("utf-8", errors="replace").strip()
            suffix = f" Stderr: {detail[-2000:]}" if detail else ""
            self._finish_with_error(f"Could not read probe result: {exc}.{suffix}")
            return
        if not bool(payload.get("ok")):
            self._finish_with_error(str(payload.get("error") or "GPU probe failed."))
            return
        result = payload.get("result")
        if not isinstance(result, dict):
            self._finish_with_error("GPU probe returned an invalid result object.")
            return
        self._finished = True
        self.succeeded.emit(result)
        self.finished.emit()


class ProcessWorkflowWorker(QObject):
    """Execute a recipe in an isolated Python process.

    A ``QThread`` still shares the interpreter and GIL with Qt's event loop.
    Some long-running extension calls do not release that GIL often enough, so
    the window stops painting even though the workflow is nominally off-thread.
    ``QProcess`` gives the workflow its own interpreter and makes cancellation
    enforceable without terminating the workbench.
    """

    succeeded = Signal(object)
    failed = Signal(str)
    logged = Signal(str)
    progressed = Signal(int, int, str)
    finished = Signal()

    def __init__(
        self,
        recipe_path: str | Path,
        project_root: str | Path,
        output_dir: str | Path,
        result_path: str | Path,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.recipe_path = Path(recipe_path).resolve()
        self.project_root = Path(project_root).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.result_path = Path(result_path).resolve()
        self._cancelled = False
        self._finished = False

        self.process = QProcess(self)
        self.process.setWorkingDirectory(str(self.project_root))
        # A GUI process on Chinese Windows can give redirected Python streams
        # the system GBK encoding.  Workflow logs legitimately contain Unicode
        # text such as ``chi²`` and an ellipsis; printing either would then raise
        # UnicodeEncodeError after a successful inversion.  Make the child's
        # encoder match the UTF-8 decoder used by ``_emit_output`` below.
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("PYTHONUTF8", "1")
        environment.insert("PYTHONIOENCODING", "utf-8")
        self.process.setProcessEnvironment(environment)
        # ``sys.executable`` can resolve to the Windows Store base executable
        # even when the workbench was launched through ``.venv\Scripts``.  A
        # direct QProcess launch of that MSIX executable bypasses the venv
        # launcher and has crashed in extension DLL initialization (notably
        # pyarrow/arrow.dll).  Prefer the console launcher belonging to the
        # active prefix; QProcess supplies pipes so it does not open a console.
        self.process.setProgram(str(_active_console_python()))
        self.process.setArguments([
            "-m",
            "PyHydroGeophysX.workflows.cli",
            "run",
            str(self.recipe_path),
            "--project-root",
            str(self.project_root),
            "--output-dir",
            str(self.output_dir),
            "--result-file",
            str(self.result_path),
        ])
        self.process.readyReadStandardOutput.connect(self._read_stdout)
        self.process.readyReadStandardError.connect(self._read_stderr)
        self.process.errorOccurred.connect(self._on_process_error)
        self.process.finished.connect(self._on_finished)

    def start(self) -> None:
        self.result_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.result_path.unlink()
        except FileNotFoundError:
            pass
        self.process.start()

    def cancel(self) -> None:
        self._cancelled = True
        if self.isRunning():
            self.process.terminate()
            QTimer.singleShot(2000, self._kill_if_running)

    def quit(self) -> None:
        """QThread-compatible shutdown hook used by ``BaseModule``."""
        self.cancel()

    def wait(self, timeout_ms: int = 30000) -> bool:
        return bool(self.process.waitForFinished(int(timeout_ms)))

    def isRunning(self) -> bool:  # noqa: N802 - mirror QThread's public API
        return self.process.state() != QProcess.ProcessState.NotRunning

    def is_cancelled(self) -> bool:
        return self._cancelled

    def _kill_if_running(self) -> None:
        if self.isRunning():
            self.process.kill()

    def _emit_output(self, raw: bytes) -> None:
        text = raw.decode("utf-8", errors="replace")
        for line in text.splitlines():
            rendered = line.rstrip()
            if not rendered.strip():
                continue
            match = re.match(
                r"^\[progress\s+(\d+)/(\d+)\]\s*(.*)$", rendered.strip()
            )
            if match is not None:
                current, total = int(match.group(1)), int(match.group(2))
                label = match.group(3).strip()
                if total > 0 and 0 <= current <= total:
                    self.progressed.emit(current, total, label)
                rendered = label or rendered
            self.logged.emit(rendered)

    def _read_stdout(self) -> None:
        self._emit_output(bytes(self.process.readAllStandardOutput()))

    def _read_stderr(self) -> None:
        self._emit_output(bytes(self.process.readAllStandardError()))

    def _on_process_error(self, error: QProcess.ProcessError) -> None:
        if error == QProcess.ProcessError.FailedToStart and not self._finished:
            self._finish_with_error(
                f"Could not start workflow process: {self.process.errorString()}"
            )

    def _finish_with_error(self, message: str) -> None:
        if self._finished:
            return
        self._finished = True
        if not self._cancelled:
            self.failed.emit(message)
        self.finished.emit()

    def _on_finished(
        self, exit_code: int, _exit_status: QProcess.ExitStatus
    ) -> None:
        if self._finished:
            return
        self._read_stdout()
        self._read_stderr()
        if self._cancelled:
            self._finished = True
            self.finished.emit()
            return
        if int(exit_code) != 0:
            unsigned = int(exit_code) & 0xFFFFFFFF
            self._finish_with_error(
                f"Workflow process exited with code {int(exit_code)} "
                f"(0x{unsigned:08X})."
            )
            return
        try:
            payload = json.loads(self.result_path.read_text(encoding="utf-8"))
            result = WorkflowRunResult.from_dict(payload)
        except Exception as exc:  # noqa: BLE001 - report malformed child output
            self._finish_with_error(
                f"Could not read workflow result {self.result_path}: {exc}"
            )
            return
        self._finished = True
        self.succeeded.emit(result)
        self.finished.emit()


__all__ = [
    "ProcessProbeWorker",
    "ProcessWorkflowWorker",
    "TaskWorker",
    "WorkflowWorker",
]
