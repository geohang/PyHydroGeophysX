"""Studio state and the Streamlit <-> Qt JSON bridge (Qt-free).

``StudioState`` owns the context that Streamlit handed us (where the project
lives, where to read hydro data, where to write results) and accumulates
per-module results. It can persist those results back to the bridge file that
Streamlit polls. Nothing here imports PySide6 so it stays easy to test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from PyHydroGeophysX.qt_apps import io_utils
from PyHydroGeophysX.qt_apps.results_store import ResultsStore, RunHandle
from PyHydroGeophysX._internal.utils import utc_now as _utc_now

RESULT_FILENAME = "full_studio_result.json"
APP_NAME = "PyHydroGeophysX Professional Studio"


@dataclass
class StudioState:
    """Mutable state shared across the studio main window and its modules."""

    context_path: Optional[Path] = None
    result_path: Optional[Path] = None
    context: Dict[str, Any] = field(default_factory=dict)
    selected_module: str = "home"
    project_root: Optional[Path] = None
    output_dir: Optional[Path] = None
    #: User-facing Project storage.  Kept distinct from ``project_root``, which
    #: is the Streamlit bridge/source-project context.
    results_store_root: Optional[Path] = None
    results_store: Optional[ResultsStore] = field(default=None, repr=False)
    hydro_data_dir: Optional[Path] = None
    hydro_output_dir: Optional[Path] = None
    selected_points: List[List[float]] = field(default_factory=list)
    active_dataset: Optional[str] = None
    module_results: Dict[str, Any] = field(default_factory=dict)
    workflow_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    active_workflow_id: str = ""
    active_recipe_path: str = ""
    #: In-memory observations, models, meshes, and interfaces shared by the
    #: geophysical processing pages. Payloads can contain NumPy/PyGIMLi objects
    #: and are deliberately excluded from the Streamlit bridge JSON.
    geophysical_resources: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    #: In-session handoff payload, e.g. a seismic-derived bedrock interface passed
    #: from the Seismic structure module to the ERT -> hydrology module. Not
    #: serialized into the bridge result file.
    shared_structure: Optional[Dict[str, Any]] = None
    _active_run_handles: Dict[Tuple[str, str], RunHandle] = field(
        default_factory=dict, repr=False
    )
    #: Called with no arguments whenever the set of runs awaiting a save may have
    #: changed. The window uses it to keep its unsaved-run indicator honest. A
    #: plain callable rather than a signal, so this module stays Qt-free.
    on_runs_changed: Optional[Callable[[], None]] = field(default=None, repr=False)

    # -- construction --------------------------------------------------------
    @classmethod
    def from_context(cls, context_path: Optional[str]) -> "StudioState":
        """Build a state object, loading the context JSON when one is given."""
        state = cls()
        if context_path:
            state.context_path = Path(context_path)
            state.load_context()
        else:
            # No bridge context: still pick a sensible, stable place to write
            # results so "Save Result" never crashes when the app is launched
            # standalone. Prefer the repository results dir when the current
            # directory looks like a PyHydroGeophysX checkout; otherwise use a
            # per-user location instead of scattering qt_bridge dirs around
            # whatever directory the app happened to start in.
            cwd = Path.cwd()
            if (cwd / "PyHydroGeophysX").is_dir() and (cwd / "examples").is_dir():
                base = cwd / "results" / "streamlit_workflow"
            else:
                base = Path.home() / ".pyhydrogeophysx"
            state.output_dir = base
            state.result_path = base / "qt_bridge" / RESULT_FILENAME
        return state

    # -- bridge IO -----------------------------------------------------------
    def load_context(self) -> Dict[str, Any]:
        """Load the bridge context JSON. Missing/invalid files yield ``{}``."""
        context = io_utils.read_json(self.context_path) if self.context_path else None
        self.context = context or {}

        def _as_path(value: Any) -> Optional[Path]:
            return Path(value) if value else None

        self.project_root = _as_path(self.context.get("project_root"))
        self.output_dir = _as_path(self.context.get("output_dir"))
        self.hydro_data_dir = _as_path(self.context.get("hydro_data_dir"))
        self.hydro_output_dir = _as_path(self.context.get("hydro_output_dir"))

        initial_module = self.context.get("initial_module") or "home"
        if initial_module:
            self.selected_module = str(initial_module)

        # The result file lives next to the context file in the qt_bridge dir.
        if self.context_path is not None:
            self.result_path = self.context_path.parent / RESULT_FILENAME
        elif self.output_dir is not None:
            self.result_path = self.output_dir / "qt_bridge" / RESULT_FILENAME
        return self.context

    def update_module_result(self, module_name: str, result_dict: Dict[str, Any]) -> None:
        """Record (or overwrite) the result payload for a single module."""
        self.module_results[str(module_name)] = result_dict

    def update_workflow_result(
        self,
        module_name: str,
        workflow_id: str,
        result_dict: Dict[str, Any],
        *,
        recipe_path: str = "",
    ) -> None:
        """Record a canonical workflow result while preserving the legacy bridge key."""
        module_key = str(module_name)
        key = (module_key, str(workflow_id))
        self.active_workflow_id = str(workflow_id)
        self.active_recipe_path = str(recipe_path)
        self.workflow_results[module_key] = {
            "workflow_id": self.active_workflow_id,
            "recipe_path": self.active_recipe_path,
            "result": dict(result_dict),
        }
        handle = self._active_run_handles.pop(key, None)
        if handle is not None and self.results_store is not None:
            if recipe_path:
                try:
                    handle.record.recipe_path = Path(recipe_path).resolve().relative_to(
                        handle.run_dir.resolve()
                    ).as_posix()
                except ValueError:
                    # A public API caller may keep its recipe elsewhere; managed Qt
                    # runs deliberately never persist paths outside their run root.
                    handle.record.recipe_path = ""
            self.results_store.finish_run(handle, result_dict)
            self._notify_runs_changed()

    # -- durable Result Store ----------------------------------------------
    def set_results_store(self, root: str | Path, *, read_only: bool = False) -> ResultsStore:
        """Open/create the folder that owns durable Qt run history."""
        requested = Path(root).expanduser().resolve()
        if (
            self.results_store is not None
            and self.results_store.root == requested
            and self.results_store.read_only == bool(read_only)
        ):
            return self.results_store
        running = [
            handle.run_id for handle in self._active_run_handles.values()
            if handle.record.status == "running"
        ]
        if running:
            raise RuntimeError(
                "Cannot switch Project while a computation is running: "
                + ", ".join(running)
            )
        store = ResultsStore.open_or_create(requested, read_only=read_only)
        self.results_store_root = store.root
        self.results_store = store
        if not read_only:
            # Existing module forms still read this bridge-compatible field for
            # defaults.  New persistent writes use RunHandle directories.
            self.output_dir = store.root
            if self.context_path is None:
                self.result_path = store.root / "qt_bridge" / RESULT_FILENAME
        return store

    def ensure_results_store(self) -> ResultsStore:
        if self.results_store is None:
            fallback = Path.home() / ".pyhydrogeophysx" / "results"
            self.set_results_store(self.output_dir or fallback)
        assert self.results_store is not None
        return self.results_store

    def begin_run(
        self,
        module_name: str,
        operation_id: str,
        workflow_id: str = "",
        *,
        label: str = "",
    ) -> RunHandle:
        module_key = str(module_name)
        operation_key = str(operation_id)
        key = (module_key, operation_key)
        existing = self._active_run_handles.get(key)
        if existing is not None and existing.record.status == "running":
            raise RuntimeError(
                f"Operation {module_key!r}/{operation_key!r} already has an active persisted run."
            )
        handle = self.ensure_results_store().begin_run(
            module_key, operation_key, workflow_id, label=label
        )
        self._active_run_handles[key] = handle
        return handle

    def active_run(
        self, module_name: str, operation_id: str = ""
    ) -> Optional[RunHandle]:
        module_key = str(module_name)
        if operation_id:
            return self._active_run_handles.get((module_key, str(operation_id)))
        matches = [
            handle for (module, _operation), handle in self._active_run_handles.items()
            if module == module_key
        ]
        return matches[0] if len(matches) == 1 else None

    def _pop_active_run(
        self, module_name: str, operation_id: str = ""
    ) -> Optional[RunHandle]:
        module_key = str(module_name)
        if operation_id:
            return self._active_run_handles.pop((module_key, str(operation_id)), None)
        keys = [key for key in self._active_run_handles if key[0] == module_key]
        if len(keys) > 1:
            raise RuntimeError(
                f"Module {module_key!r} has multiple active operations; operation_id is required."
            )
        return self._active_run_handles.pop(keys[0], None) if keys else None

    def _notify_runs_changed(self) -> None:
        if self.on_runs_changed is None:
            return
        try:
            self.on_runs_changed()
        except Exception:  # noqa: BLE001 - an indicator must not break a run
            pass

    def finish_run(
        self, module_name: str, result: Any, operation_id: str = ""
    ) -> None:
        handle = self._pop_active_run(module_name, operation_id)
        if handle is not None:
            self.ensure_results_store().finish_run(handle, result)
            self._notify_runs_changed()

    def fail_run(self, module_name: str, error: str, operation_id: str = "") -> None:
        handle = self._pop_active_run(module_name, operation_id)
        if handle is not None:
            self.ensure_results_store().fail_run(handle, error)
            self._notify_runs_changed()

    def cancel_run(
        self, module_name: str, error: str = "", operation_id: str = ""
    ) -> None:
        handle = self._pop_active_run(module_name, operation_id)
        if handle is not None:
            self.ensure_results_store().cancel_run(handle, error)
            self._notify_runs_changed()

    def cancel_module_runs(self, module_name: str, error: str = "") -> None:
        module_key = str(module_name)
        for key in [key for key in self._active_run_handles if key[0] == module_key]:
            handle = self._active_run_handles.pop(key)
            self.ensure_results_store().cancel_run(handle, error)
            self._notify_runs_changed()

    # -- saving --------------------------------------------------------------
    def has_unsaved_runs(self) -> bool:
        """Whether any finished run is still waiting on the user's decision."""
        store = self.results_store
        return bool(store is not None and store.has_unsaved())

    def unsaved_runs(self) -> List[Any]:
        store = self.results_store
        return store.list_unsaved_runs() if store is not None else []

    def save_all_runs(self) -> List[Any]:
        """Put every finished-but-unsaved run into the Project's history."""
        store = self.results_store
        if store is None:
            return []
        try:
            return store.save_all_unsaved()
        finally:
            self._notify_runs_changed()

    def discard_all_runs(self) -> int:
        store = self.results_store
        if store is None:
            return 0
        try:
            return store.discard_all_unsaved()
        finally:
            self._notify_runs_changed()

    def clear_project_session(self) -> None:
        """Drop process-local results when the active Project changes."""
        if self._active_run_handles:
            raise RuntimeError("Cannot clear Project state while computations are running.")
        self.module_results.clear()
        self.workflow_results.clear()
        self.geophysical_resources.clear()
        self.shared_structure = None
        self.selected_points.clear()
        self.active_dataset = None
        self.active_workflow_id = ""
        self.active_recipe_path = ""

    def register_geophysical_resource(
        self,
        method: str,
        role: str,
        payload: Any,
        *,
        label: str = "",
        path: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        resource_id: Optional[str] = None,
    ) -> str:
        """Register a process-local geophysical observation or inversion product."""
        method_key = str(method).strip().upper()
        role_key = str(role).strip().lower()
        if not method_key or not role_key:
            raise ValueError("method and role are required for a geophysical resource.")
        if payload is None:
            raise ValueError("A geophysical resource payload cannot be None.")
        if resource_id is None:
            prefix = f"{method_key.lower()}:{role_key}"
            index = 1
            resource_id = f"{prefix}:{index}"
            while resource_id in self.geophysical_resources:
                index += 1
                resource_id = f"{prefix}:{index}"
        resource_id = str(resource_id)
        self.geophysical_resources[resource_id] = {
            "id": resource_id,
            "method": method_key,
            "role": role_key,
            "label": label or f"{method_key} {role_key}",
            "path": str(path or ""),
            "metadata": dict(metadata or {}),
            "payload": payload,
        }
        return resource_id

    def list_geophysical_resources(
        self,
        method: Optional[str] = None,
        role: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List resource descriptors without exposing non-serializable payloads."""
        method_key = str(method).strip().upper() if method else None
        role_key = str(role).strip().lower() if role else None
        resources: List[Dict[str, Any]] = []
        for item in self.geophysical_resources.values():
            if method_key and item["method"] != method_key:
                continue
            if role_key and item["role"] != role_key:
                continue
            resources.append({key: value for key, value in item.items() if key != "payload"})
        return resources

    def get_geophysical_resource(self, resource_id: str) -> Optional[Dict[str, Any]]:
        """Return a registered resource including its in-memory payload."""
        return self.geophysical_resources.get(str(resource_id))

    def build_result(self) -> Dict[str, Any]:
        """Assemble the result document in the schema Streamlit expects."""
        return {
            "app": APP_NAME,
            "status": "saved",
            "selected_module": self.selected_module,
            "created_time": _utc_now(),
            "context_path": str(self.context_path) if self.context_path else "",
            "output_dir": str(self.output_dir) if self.output_dir else "",
            "module_results": self.module_results,
            "workflow_id": self.active_workflow_id,
            "recipe_path": self.active_recipe_path,
            "workflow_results": self.workflow_results,
        }

    def save_result(self) -> Path:
        """Persist the current results to the bridge result file."""
        if self.result_path is None:
            base = self.output_dir or Path.cwd()
            self.result_path = base / "qt_bridge" / RESULT_FILENAME
        return io_utils.write_json(self.result_path, self.build_result())

    # -- convenience ---------------------------------------------------------
    def context_summary(self) -> Dict[str, Any]:
        """Return a compact dict for the idle properties panel."""
        return {
            "app_key": self.context.get("app_key", ""),
            "selected_module": self.selected_module,
            "project_root": str(self.project_root or ""),
            "output_dir": str(self.output_dir or ""),
            "hydro_data_dir": str(self.hydro_data_dir or ""),
            "hydro_output_dir": str(self.hydro_output_dir or ""),
            "context_path": str(self.context_path or ""),
            "result_path": str(self.result_path or ""),
            "results_store_root": str(self.results_store_root or ""),
            "demo_mode": self.context.get("demo_mode", False),
        }
