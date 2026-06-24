"""Workbench state and the Streamlit <-> Qt JSON bridge (Qt-free).

``WorkbenchState`` owns the context that Streamlit handed us (where the project
lives, where to read hydro data, where to write results) and accumulates
per-module results. It can persist those results back to the bridge file that
Streamlit polls. Nothing here imports PySide6 so it stays easy to test.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from PyHydroGeophysX.qt_apps import io_utils

RESULT_FILENAME = "full_workbench_result.json"
APP_NAME = "PyHydroGeophysX Professional Workbench"


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class WorkbenchState:
    """Mutable state shared across the workbench main window and its modules."""

    context_path: Optional[Path] = None
    result_path: Optional[Path] = None
    context: Dict[str, Any] = field(default_factory=dict)
    selected_module: str = "home"
    project_root: Optional[Path] = None
    output_dir: Optional[Path] = None
    hydro_data_dir: Optional[Path] = None
    hydro_output_dir: Optional[Path] = None
    selected_points: List[List[float]] = field(default_factory=list)
    active_dataset: Optional[str] = None
    module_results: Dict[str, Any] = field(default_factory=dict)
    #: In-session handoff payload, e.g. a seismic-derived bedrock interface passed
    #: from the Seismic structure module to the ERT -> hydrology module. Not
    #: serialized into the bridge result file.
    shared_structure: Optional[Dict[str, Any]] = None

    # -- construction --------------------------------------------------------
    @classmethod
    def from_context(cls, context_path: Optional[str]) -> "WorkbenchState":
        """Build a state object, loading the context JSON when one is given."""
        state = cls()
        if context_path:
            state.context_path = Path(context_path)
            state.load_context()
        else:
            # No bridge context: still pick a sensible place to write results so
            # "Save Result" never crashes when the app is launched standalone.
            state.output_dir = Path.cwd()
            state.result_path = state.output_dir / "qt_bridge" / RESULT_FILENAME
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
            "demo_mode": self.context.get("demo_mode", False),
        }
