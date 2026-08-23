"""Qt-free run history for the desktop studio.

Each saved run owns a small ``run.json`` record.  The root index is deliberately
a cache: it can always be rebuilt by scanning those records, so an interrupted
OneDrive update cannot make the expensive scientific outputs disappear from
the studio.

**A finished computation is not part of the history until the user saves it.**
A solver has to write its outputs somewhere, so a run still gets a directory
under ``runs/`` while it computes, and that directory is marked with an
``UNSAVED`` file.  What it does not get is ``run.json``: without that record the
run is invisible to :meth:`ResultsStore.rebuild_index`, to the Model Viewer, and
to any later session that opens the Project.  :meth:`ResultsStore.save_run`
writes the record and clears the marker; :meth:`ResultsStore.discard_run`
removes the directory.  Nothing moves on disk in either case, so a path a module
captured while computing still resolves afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional
import uuid


STORE_SCHEMA_VERSION = "1"
RUN_SCHEMA_VERSION = "1"
INDEX_FILENAME = "phgx_results_index.json"
RUN_FILENAME = "run.json"
IMPORTS_DIRNAME = "imports"
#: Written into a run directory while it is unsaved, and removed on save. The
#: absence of ``run.json`` is what actually keeps a run out of the history; this
#: names the state for anyone browsing the folder, and lets a later session tell
#: an abandoned run from a directory it does not recognize.
UNSAVED_MARKER = "UNSAVED"
_UNSAVED_NOTE = (
    "This run has not been saved to the Project.\n"
    "It is invisible to the studio's run history until you save it there,\n"
    "and deleting this folder discards it.\n"
)
_AUTO_DISCOVER_LIMIT = 200
_AUTO_DISCOVER_FORMATS = {
    "bms", "csv", "dat", "json", "jpg", "jpeg", "npy", "npz", "png",
    "ply", "stl", "tif", "tiff", "tsv", "txt", "vtk", "vtp", "vtu",
}

_SUCCESS = {"ok", "success", "saved", "completed", "complete", "succeeded"}
_FAILED = {"failed", "failure", "error"}
_CANCELLED = {"cancelled", "canceled"}
_SHORT_CODES = {
    "ert.single_inversion": "ert",
    "ert.timelapse_inversion": "erttl",
    "seismic.srt_inversion": "srt",
    "em.inversion": "em",
    "em.line_inversion": "emline",
    "gravmag.process": "grav",
    "gravmag.forward_bodies": "gravfwd",
    "gravmag.invert": "gminv",
    "joint_inversion.run": "joint",
    "mesh3d.build": "mesh",
    "ert3d.forward": "ert3d",
    "hydro_geophysics.forward": "hydro",
    "geo_hydrology.ert_to_wc": "geo",
    "seismic3d.build": "seis3d",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize_status(value: Any) -> str:
    """Map legacy workflow status spellings onto the Result Store vocabulary."""
    raw = str(value or "").strip().lower()
    if raw in _SUCCESS:
        return "success"
    if raw in _FAILED:
        return "failed"
    if raw in _CANCELLED:
        return "cancelled"
    if raw in {"running", "interrupted"}:
        return raw
    return "unknown"


def _json_safe(value: Any) -> Any:
    """Return a conservative JSON view without importing NumPy eagerly."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        try:
            shape = [int(part) for part in value.shape]
            size = int(value.size)
            if size <= 2048 and hasattr(value, "tolist"):
                return _json_safe(value.tolist())
            return {
                "$array": {
                    "shape": shape,
                    "dtype": str(value.dtype),
                    "size": size,
                }
            }
        except Exception:
            pass
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:
            pass
    return str(value)


def _atomic_write_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    attempts: int = 5,
) -> Path:
    """Atomically replace *path*, retrying transient Windows/OneDrive locks."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(_json_safe(payload), indent=2, allow_nan=False),
            encoding="utf-8",
        )
        last_error: Optional[OSError] = None
        for index in range(max(1, int(attempts))):
            try:
                os.replace(temporary, path)
                return path
            except OSError as exc:
                last_error = exc
                if index + 1 < attempts:
                    time.sleep(0.05 * (2 ** index))
        assert last_error is not None
        raise last_error
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


@dataclass
class RunRecord:
    run_id: str
    run_dir: Path
    module_key: str
    operation_id: str
    workflow_id: str = ""
    status: str = "running"
    raw_status: str = "running"
    created_at: str = field(default_factory=_utc_now)
    finished_at: str = ""
    label: str = ""
    notes: str = ""
    recipe_path: str = ""
    result_path: str = "result.json"
    artifacts: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)
    error: str = ""
    managed: bool = True
    imported: bool = False

    @property
    def metadata_path(self) -> Path:
        return self.run_dir / RUN_FILENAME

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": RUN_SCHEMA_VERSION,
            "run_id": self.run_id,
            "module_key": self.module_key,
            "operation_id": self.operation_id,
            "workflow_id": self.workflow_id,
            "status": self.status,
            "raw_status": self.raw_status,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "label": self.label,
            "notes": self.notes,
            "recipe_path": self.recipe_path,
            "result_path": self.result_path,
            "artifacts": _json_safe(self.artifacts),
            "metrics": _json_safe(self.metrics),
            "summary": _json_safe(self.summary),
            "warnings": _json_safe(self.warnings),
            "provenance": _json_safe(self.provenance),
            "error": self.error,
            "managed": bool(self.managed),
            "imported": bool(self.imported),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], run_dir: Path) -> "RunRecord":
        return cls(
            run_id=str(payload.get("run_id") or run_dir.name),
            run_dir=Path(run_dir).resolve(),
            module_key=str(payload.get("module_key") or "unknown"),
            operation_id=str(payload.get("operation_id") or payload.get("workflow_id") or "unknown"),
            workflow_id=str(payload.get("workflow_id") or ""),
            status=normalize_status(payload.get("status")),
            raw_status=str(payload.get("raw_status") or payload.get("status") or ""),
            created_at=str(payload.get("created_at") or ""),
            finished_at=str(payload.get("finished_at") or ""),
            label=str(payload.get("label") or ""),
            notes=str(payload.get("notes") or ""),
            recipe_path=str(payload.get("recipe_path") or ""),
            result_path=str(payload.get("result_path") or "result.json"),
            artifacts=[dict(item) for item in payload.get("artifacts") or [] if isinstance(item, Mapping)],
            metrics=dict(payload.get("metrics") or {}),
            summary=dict(payload.get("summary") or {}),
            warnings=[str(item) for item in payload.get("warnings") or []],
            provenance=dict(payload.get("provenance") or {}),
            error=str(payload.get("error") or ""),
            managed=bool(payload.get("managed", True)),
            imported=bool(payload.get("imported", False)),
        )


@dataclass(frozen=True)
class RunHandle:
    record: RunRecord

    @property
    def run_id(self) -> str:
        return self.record.run_id

    @property
    def run_dir(self) -> Path:
        return self.record.run_dir

    @property
    def inputs_dir(self) -> Path:
        return self.run_dir / "inputs"

    @property
    def outputs_dir(self) -> Path:
        return self.run_dir / "outputs"

    @property
    def logs_dir(self) -> Path:
        return self.run_dir / "logs"

    @property
    def recipe_path(self) -> Path:
        if self.record.recipe_path:
            return self.run_dir / self.record.recipe_path
        return self.run_dir / "recipe.json"

    @property
    def result_path(self) -> Path:
        return self.run_dir / (self.record.result_path or "result.json")


class ResultsStore:
    """Folder-backed history of studio operations and workflow outputs."""

    def __init__(self, root: Path, *, read_only: bool = False) -> None:
        self.root = Path(root).expanduser().resolve()
        self.read_only = bool(read_only)
        self.runs_dir = self.root / "runs"
        self.imports_dir = self.root / IMPORTS_DIRNAME
        self.index_path = self.root / INDEX_FILENAME
        self.last_warning = ""
        self._records: Dict[str, RunRecord] = {}
        #: Runs computed this session that the user has not saved. Held here and
        #: not on disk as ``run.json``, which is what keeps them out of the
        #: history; the payload is kept so a later save writes the same
        #: ``result.json`` the run produced.
        self._unsaved: Dict[str, RunRecord] = {}
        self._unsaved_payloads: Dict[str, Dict[str, Any]] = {}
        if not self.read_only:
            self.runs_dir.mkdir(parents=True, exist_ok=True)
            self.imports_dir.mkdir(parents=True, exist_ok=True)
            (self.root / "scratch").mkdir(parents=True, exist_ok=True)
        self.rebuild_index(recover_running=not self.read_only)

    @classmethod
    def open_or_create(cls, root: str | Path, *, read_only: bool = False) -> "ResultsStore":
        return cls(Path(root), read_only=read_only)

    def _short_code(self, operation_id: str, module_key: str) -> str:
        candidate = _SHORT_CODES.get(str(operation_id), str(module_key or "run").lower())
        cleaned = "".join(ch for ch in candidate if ch.isalnum())[:8]
        return cleaned or "run"

    def begin_run(
        self,
        module_key: str,
        operation_id: str,
        workflow_id: str = "",
        *,
        label: str = "",
    ) -> RunHandle:
        if self.read_only:
            raise PermissionError("This Result Store is read-only.")
        now = datetime.now(timezone.utc)
        code = self._short_code(operation_id, module_key)
        for _ in range(20):
            run_id = f"{now:%Y%m%d-%H%M%S}_{code}_{uuid.uuid4().hex[:4]}"
            run_dir = self.runs_dir / run_id
            try:
                run_dir.mkdir(parents=False, exist_ok=False)
                break
            except FileExistsError:
                continue
        else:
            raise FileExistsError("Could not allocate a unique Result Store run directory.")
        for child in ("inputs", "outputs", "logs"):
            (run_dir / child).mkdir()
        display = label or f"{operation_id or module_key} · {now.astimezone():%Y-%m-%d %H:%M}"
        record = RunRecord(
            run_id=run_id,
            run_dir=run_dir.resolve(),
            module_key=str(module_key),
            operation_id=str(operation_id),
            workflow_id=str(workflow_id),
            label=display,
        )
        # No run.json: the solver needs a directory to write into, but the run
        # does not join the Project's history until the user saves it. The marker
        # says so in the folder itself, where someone browsing will look.
        try:
            (run_dir / UNSAVED_MARKER).write_text(_UNSAVED_NOTE, encoding="utf-8")
        except OSError as exc:
            self.last_warning = f"Could not mark the run folder unsaved: {exc}"
        self._unsaved[record.run_id] = record
        return RunHandle(record)

    @staticmethod
    def _result_payload(result: Any) -> Dict[str, Any]:
        if hasattr(result, "to_dict") and callable(result.to_dict):
            return dict(result.to_dict())
        if isinstance(result, Mapping):
            return dict(result)
        raise TypeError(f"Unsupported run result type: {type(result).__name__}")

    def finish_run(self, handle: RunHandle, result: Any) -> RunRecord:
        payload = self._result_payload(result)
        record = handle.record
        raw_status = str(payload.get("status") or "success")
        record.raw_status = raw_status
        record.status = normalize_status(raw_status)
        if record.status == "unknown":
            record.status = "success"
        record.finished_at = _utc_now()
        record.summary = dict(payload.get("summary") or {})
        record.metrics = dict(payload.get("metrics") or {})
        record.warnings = [str(item) for item in payload.get("warnings") or []]
        record.provenance = dict(payload.get("provenance") or {})
        record.artifacts = [
            dict(item.get("$artifact", item))
            for item in payload.get("artifacts") or []
            if isinstance(item, Mapping)
        ]
        known_paths = {str(item.get("path") or "") for item in record.artifacts}
        candidates = sorted(
            path for path in handle.outputs_dir.rglob("*")
            if path.is_file() and not path.is_symlink()
            and path.suffix.lstrip(".").lower() in _AUTO_DISCOVER_FORMATS
        )
        undiscovered = 0
        added = 0
        for path in candidates:
            if not path.is_file() or path.is_symlink():
                continue
            relative = path.relative_to(record.run_dir).as_posix()
            if relative in known_paths or str(path) in known_paths:
                continue
            if added >= _AUTO_DISCOVER_LIMIT:
                undiscovered += 1
                continue
            record.artifacts.append({
                "artifact_id": f"output:{relative}",
                "kind": "attachment",
                "path": relative,
                "format": path.suffix.lstrip(".").lower(),
                "checksum": "",
                "metadata": {"auto_discovered": True},
            })
            added += 1
        if undiscovered:
            record.warnings.append(
                f"{undiscovered} additional output files were not individually listed."
            )
        self._stage(record, payload)
        return record

    def _stage(self, record: RunRecord, payload: Optional[Mapping[str, Any]] = None) -> None:
        """Hold a closed run in memory, awaiting the user's decision to save."""
        self._unsaved[record.run_id] = record
        if payload is not None:
            self._unsaved_payloads[record.run_id] = dict(payload)

    def fail_run(self, handle: RunHandle, error: str, *, raw_status: str = "failed") -> RunRecord:
        return self._close_unsuccessful(handle, "failed", raw_status, error)

    def cancel_run(self, handle: RunHandle, error: str = "") -> RunRecord:
        return self._close_unsuccessful(handle, "cancelled", "cancelled", error)

    def _close_unsuccessful(
        self, handle: RunHandle, status: str, raw_status: str, error: str
    ) -> RunRecord:
        record = handle.record
        record.status = status
        record.raw_status = raw_status
        record.error = str(error)
        record.finished_at = _utc_now()
        # A failed run is staged like any other. Its inputs and log are often
        # worth keeping to diagnose the failure, and that is the user's call.
        self._stage(record)
        return record

    # -- saving --------------------------------------------------------------
    def has_unsaved(self) -> bool:
        return bool(self._unsaved)

    def list_unsaved_runs(self) -> List[RunRecord]:
        return sorted(
            self._unsaved.values(),
            key=lambda item: (item.created_at, item.run_id),
            reverse=True,
        )

    def save_run(self, run_id: str) -> RunRecord:
        """Write a staged run's record, putting it into the Project's history."""
        if self.read_only:
            raise PermissionError("This Result Store is read-only.")
        key = str(run_id)
        record = self._unsaved.get(key)
        if record is None:
            existing = self._records.get(key)
            if existing is not None:
                return existing          # already saved; saving twice is a no-op
            raise KeyError(run_id)
        if record.status == "running":
            raise RuntimeError("A running computation cannot be saved yet.")
        payload = self._unsaved_payloads.get(key)
        try:
            if payload is not None:
                _atomic_write_json(record.run_dir / (record.result_path or "result.json"),
                                   payload)
            _atomic_write_json(record.metadata_path, record.to_dict())
        except (OSError, TypeError, ValueError) as exc:
            # The calculation succeeded and the user asked to keep it. Preserve a
            # uniquely named recovery record rather than losing the run to a
            # metadata lock, and leave it staged so the save can be retried.
            self.last_warning = f"Run metadata update is pending recovery: {exc}"
            recovery = record.run_dir / f"run.recovery.{uuid.uuid4().hex[:8]}.json"
            recovery.write_text(
                json.dumps(_json_safe(record.to_dict()), indent=2, allow_nan=False),
                encoding="utf-8",
            )
            raise
        try:
            (record.run_dir / UNSAVED_MARKER).unlink(missing_ok=True)
        except OSError:
            pass                          # the record is what decides; the marker is a label
        self._unsaved.pop(key, None)
        self._unsaved_payloads.pop(key, None)
        self._records[record.run_id] = record
        self._write_index_nonfatal(self.list_runs())
        return record

    def save_all_unsaved(self) -> List[RunRecord]:
        """Save every staged run that has finished. Returns the ones saved."""
        saved: List[RunRecord] = []
        for record in self.list_unsaved_runs():
            if record.status == "running":
                continue
            saved.append(self.save_run(record.run_id))
        return saved

    def discard_run(self, run_id: str) -> None:
        """Delete a staged run's folder. Saved runs go through delete_run."""
        if self.read_only:
            raise PermissionError("This Result Store is read-only.")
        key = str(run_id)
        record = self._unsaved.get(key)
        if record is None:
            raise KeyError(run_id)
        if record.status == "running":
            raise RuntimeError("A running computation cannot be discarded.")
        self._remove_run_directory(record.run_dir)
        self._unsaved.pop(key, None)
        self._unsaved_payloads.pop(key, None)

    def discard_all_unsaved(self) -> int:
        """Delete every staged run that has finished. Returns how many went."""
        removed = 0
        for record in self.list_unsaved_runs():
            if record.status == "running":
                continue
            self.discard_run(record.run_id)
            removed += 1
        return removed

    def abandoned_run_dirs(self) -> List[Path]:
        """Run folders left unsaved by an earlier session.

        A crash or a forced quit leaves a marked directory with no ``run.json``.
        Nothing reads it, so it would sit in the Project consuming space without
        appearing anywhere; this is how the studio offers to clear it.
        """
        if not self.runs_dir.is_dir():
            return []
        live = {record.run_dir.resolve() for record in self._unsaved.values()}
        found = []
        for candidate in sorted(self.runs_dir.iterdir()):
            if not candidate.is_dir() or candidate.is_symlink():
                continue
            if (candidate / RUN_FILENAME).exists():
                continue
            if candidate.resolve() in live:
                continue
            if any(candidate.glob("run.recovery.*.json")):
                # A save the user asked for that could not write run.json. It is
                # recovered on the next scan, so it is not abandoned.
                continue
            if (candidate / UNSAVED_MARKER).exists():
                found.append(candidate)
        return found

    def clear_abandoned_runs(self) -> int:
        """Delete the folders :meth:`abandoned_run_dirs` reports."""
        if self.read_only:
            raise PermissionError("This Result Store is read-only.")
        removed = 0
        for path in self.abandoned_run_dirs():
            self._remove_run_directory(path)
            removed += 1
        return removed

    def _remove_run_directory(self, run_dir: Path) -> None:
        """Delete a run folder, refusing anything that is not one."""
        target = Path(run_dir).resolve()
        runs_root = self.runs_dir.resolve()
        try:
            relative = target.relative_to(runs_root)
        except ValueError as exc:
            raise ValueError("Refusing to delete a run outside this Result Store.") from exc
        if len(relative.parts) != 1 or target == runs_root:
            raise ValueError("Refusing to delete an invalid run directory.")
        shutil.rmtree(target)

    def update_run(self, run_id: str, *, label: Optional[str] = None, notes: Optional[str] = None) -> RunRecord:
        record = self.get_run(run_id)
        if record is None:
            raise KeyError(run_id)
        if self.read_only:
            raise PermissionError("This Result Store is read-only.")
        if label is not None:
            record.label = str(label).strip() or record.label
        if notes is not None:
            record.notes = str(notes)
        if record.run_id in self._unsaved:
            # Naming a run before deciding to keep it is normal. Writing run.json
            # here would save it as a side effect of typing a label.
            return record
        _atomic_write_json(record.metadata_path, record.to_dict())
        self._records[record.run_id] = record
        self._write_index_nonfatal(self.list_runs())
        return record

    def _read_record(self, path: Path, *, recover_running: bool = False) -> Optional[RunRecord]:
        record: Optional[RunRecord] = None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            record = RunRecord.from_dict(payload, path.parent)
        except (OSError, ValueError, TypeError):
            pass

        # A successful calculation may have written this after OneDrive locked
        # result.json or run.json. Prefer a newer recovery over the stale
        # ``running`` record, then repair the truth file when the store is writable.
        recoveries = sorted(
            path.parent.glob("run.recovery.*.json"),
            key=lambda item: item.stat().st_mtime_ns,
            reverse=True,
        )
        if recoveries:
            try:
                recovery = recoveries[0]
                use_recovery = record is None or recovery.stat().st_mtime_ns >= path.stat().st_mtime_ns
                if use_recovery:
                    payload = json.loads(recovery.read_text(encoding="utf-8"))
                    record = RunRecord.from_dict(payload, path.parent)
                    if not self.read_only:
                        try:
                            _atomic_write_json(path, record.to_dict())
                        except OSError as exc:
                            self.last_warning = f"Could not repair recovered run metadata: {exc}"
            except (OSError, ValueError, TypeError):
                pass
        if record is None:
            return None
        if recover_running and record.status == "running":
            record.status = "interrupted"
            record.raw_status = record.raw_status or "running"
            record.finished_at = record.finished_at or _utc_now()
            try:
                _atomic_write_json(record.metadata_path, record.to_dict())
            except OSError as exc:
                self.last_warning = f"Could not persist interrupted run state: {exc}"
        return record

    def _scan_records(self, *, recover_running: bool = False) -> List[RunRecord]:
        if not self.root.exists():
            return []
        records: Dict[str, RunRecord] = {}
        paths = list(self.runs_dir.glob(f"*/{RUN_FILENAME}"))
        # A save that hit a locked run.json leaves a recovery record beside it.
        # The run was one the user asked to keep, so it is found by the name that
        # did get written; ``_read_record`` repairs run.json from it.
        paths.extend(
            recovery.parent / RUN_FILENAME
            for recovery in self.runs_dir.glob("*/run.recovery.*.json")
            if not (recovery.parent / RUN_FILENAME).exists()
        )
        # Read-only browsing of a legacy tree recognizes sidecars in place. A
        # writable Project stores small import pointers, so routine refreshes do
        # not walk every scientific output directory.
        if self.read_only:
            paths.extend(path for path in self.root.rglob(RUN_FILENAME) if path not in paths)
        for pointer in self.imports_dir.glob("*.json"):
            try:
                payload = json.loads(pointer.read_text(encoding="utf-8"))
                linked = Path(str(payload.get("run_json") or "")).resolve()
            except (OSError, TypeError, ValueError):
                continue
            if linked.name == RUN_FILENAME:
                paths.append(linked)
        for path in paths:
            if path.is_symlink():
                continue
            record = self._read_record(path, recover_running=recover_running)
            if record is not None:
                records[record.run_id] = record
        return sorted(
            records.values(),
            key=lambda item: (item.created_at, item.run_id),
            reverse=True,
        )

    def list_runs(self) -> List[RunRecord]:
        return sorted(
            self._records.values(),
            key=lambda item: (item.created_at, item.run_id),
            reverse=True,
        )

    def get_run(self, run_id: str) -> Optional[RunRecord]:
        key = str(run_id)
        return self._records.get(key) or self._unsaved.get(key)

    def is_unsaved(self, run_id: str) -> bool:
        return str(run_id) in self._unsaved

    def rebuild_index(self, *, recover_running: bool = False) -> List[RunRecord]:
        records = self._scan_records(recover_running=recover_running)
        self._records = {record.run_id: record for record in records}
        if not self.read_only:
            self._write_index_nonfatal(records)
        return records

    def _write_index_nonfatal(self, records: Iterable[RunRecord]) -> None:
        if self.read_only:
            return
        payload = {
            "schema_version": STORE_SCHEMA_VERSION,
            "updated_at": _utc_now(),
            "runs": [
                {
                    "run_id": item.run_id,
                    "run_path": self._index_run_path(item),
                    "module_key": item.module_key,
                    "operation_id": item.operation_id,
                    "workflow_id": item.workflow_id,
                    "status": item.status,
                    "created_at": item.created_at,
                    "finished_at": item.finished_at,
                    "label": item.label,
                    "managed": item.managed,
                    "imported": item.imported,
                }
                for item in records
            ],
        }
        try:
            _atomic_write_json(self.index_path, payload)
        except OSError as exc:
            self.last_warning = f"Result Store index will be rebuilt later: {exc}"

    def _index_run_path(self, record: RunRecord) -> str:
        try:
            return record.run_dir.relative_to(self.root).as_posix()
        except ValueError:
            return str(record.run_dir)

    def locate_run_artifact(self, record: RunRecord, artifact: Any) -> Path:
        if hasattr(artifact, "path"):
            value = artifact.path
        elif isinstance(artifact, Mapping):
            value = artifact.get("path", "")
        else:
            value = artifact
        candidate = Path(str(value))
        if candidate.is_absolute():
            resolved = candidate.resolve()
        else:
            resolved = (record.run_dir / candidate).resolve()
        base = record.run_dir.resolve()
        try:
            resolved.relative_to(base)
        except ValueError as exc:
            raise ValueError(f"Artifact path escapes run directory: {value}") from exc
        return resolved

    def scratch_dir(self, module_key: str) -> Path:
        if self.read_only:
            raise PermissionError("This Result Store is read-only.")
        safe = "".join(ch for ch in str(module_key) if ch.isalnum() or ch in "-_") or "module"
        target = self.root / "scratch" / safe
        target.mkdir(parents=True, exist_ok=True)
        return target

    def run_size(self, record: RunRecord) -> int:
        total = 0
        for path in record.run_dir.rglob("*"):
            try:
                if path.is_file() and not path.is_symlink():
                    total += path.stat().st_size
            except OSError:
                pass
        return total

    def delete_run(self, run_id: str) -> None:
        if self.read_only:
            raise PermissionError("This Result Store is read-only.")
        record = self.get_run(run_id)
        if record is None:
            raise KeyError(run_id)
        if self.is_unsaved(record.run_id):
            self.discard_run(record.run_id)
            return
        if record.status == "running":
            raise RuntimeError("A running operation cannot be deleted.")
        if not record.managed or record.imported:
            raise PermissionError("Imported runs are not managed and cannot be deleted here.")
        self._remove_run_directory(record.run_dir)
        self._records.pop(record.run_id, None)
        self._write_index_nonfatal(self.list_runs())

    @staticmethod
    def preview_legacy(root: str | Path) -> List[Dict[str, Any]]:
        """Conservatively find directories containing one recipe/result pair."""
        base = Path(root).expanduser().resolve()
        found: List[Dict[str, Any]] = []
        if not base.is_dir():
            return found
        for recipe in base.rglob("*_recipe.json"):
            if recipe.is_symlink():
                continue
            try:
                recipe.resolve().relative_to(base)
            except ValueError:
                continue
            stem = recipe.name[:-len("_recipe.json")]
            candidates = [
                recipe.with_name(f"{stem}_process_result.json"),
                recipe.with_name("result.json"),
            ]
            result = next((path for path in candidates if path.is_file()), None)
            if result is None:
                continue
            found.append({"directory": recipe.parent, "recipe": recipe, "result": result})
        return found

    def import_legacy(self, source: str | Path) -> List[RunRecord]:
        """Add sidecars in place and link them into the current Project."""
        if self.read_only:
            raise PermissionError("This Result Store is read-only.")
        imported: List[RunRecord] = []
        seen_dirs = set()
        for item in self.preview_legacy(source):
            directory = Path(item["directory"]).resolve()
            if directory in seen_dirs:
                continue
            seen_dirs.add(directory)
            existing_path = directory / RUN_FILENAME
            if existing_path.exists():
                existing = self._read_record(existing_path)
                if existing is not None and existing.imported:
                    self._link_imported_record(existing)
                    self._records[existing.run_id] = existing
                continue
            try:
                recipe_payload = json.loads(Path(item["recipe"]).read_text(encoding="utf-8"))
                result_payload = json.loads(Path(item["result"]).read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                continue
            workflow_id = str(recipe_payload.get("workflow_id") or "legacy.import")
            module_key = workflow_id.partition(".")[0] or "legacy"
            artifacts = [
                dict(value.get("$artifact", value))
                for value in result_payload.get("artifacts") or []
                if isinstance(value, Mapping)
            ]
            known = {Path(str(value.get("path") or "")).as_posix() for value in artifacts}
            for path in directory.iterdir():
                if path.is_file() and not path.is_symlink() and path.name not in {
                    RUN_FILENAME, Path(item["recipe"]).name, Path(item["result"]).name,
                } and path.name not in known:
                    artifacts.append({
                        "artifact_id": f"attachment:{path.name}",
                        "kind": "attachment",
                        "path": path.name,
                        "format": path.suffix.lstrip(".").lower(),
                        "checksum": "",
                        "metadata": {"imported_attachment": True},
                    })
            created = datetime.fromtimestamp(
                Path(item["result"]).stat().st_mtime, timezone.utc
            ).isoformat(timespec="seconds")
            record = RunRecord(
                run_id=f"legacy_{uuid.uuid5(uuid.NAMESPACE_URL, str(directory)).hex[:12]}",
                run_dir=directory,
                module_key=module_key,
                operation_id=workflow_id,
                workflow_id=workflow_id,
                status=normalize_status(result_payload.get("status")),
                raw_status=str(result_payload.get("status") or ""),
                created_at=created,
                finished_at=created,
                label=f"Imported {workflow_id}",
                recipe_path=Path(item["recipe"]).name,
                result_path=Path(item["result"]).name,
                artifacts=artifacts,
                metrics=dict(result_payload.get("metrics") or {}),
                summary=dict(result_payload.get("summary") or {}),
                warnings=[str(value) for value in result_payload.get("warnings") or []],
                provenance=dict(result_payload.get("provenance") or {}),
                managed=False,
                imported=True,
            )
            _atomic_write_json(record.metadata_path, record.to_dict())
            self._link_imported_record(record)
            self._records[record.run_id] = record
            imported.append(record)
        self._write_index_nonfatal(self.list_runs())
        return imported

    def _link_imported_record(self, record: RunRecord) -> None:
        pointer = self.imports_dir / f"{record.run_id}.json"
        _atomic_write_json(pointer, {
            "schema_version": STORE_SCHEMA_VERSION,
            "run_id": record.run_id,
            "run_json": str(record.metadata_path),
        })


__all__ = [
    "INDEX_FILENAME",
    "RUN_FILENAME",
    "UNSAVED_MARKER",
    "RunHandle",
    "RunRecord",
    "ResultsStore",
    "normalize_status",
]
