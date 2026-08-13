"""Guided Qt workflow for registered two-method joint inversions."""

from __future__ import annotations

import html
import json
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from PyHydroGeophysX.inversion.joint_api import (
    METHODS,
    JointInversionRequest,
    JointInversionResult,
    get_joint_capability,
    get_joint_capabilities,
    pair_joint_soundings,
    split_joint_soundings,
    validate_profile_interface,
)
from PyHydroGeophysX.data_processing.joint_io import save_joint_observations
from PyHydroGeophysX.workflows import em1d as em_pipeline
from PyHydroGeophysX.workflows import gravmag as gravmag_pipeline
from PyHydroGeophysX.qt_apps import io_utils, theme
from PyHydroGeophysX.qt_apps.modules.base import BaseModule, LogFn
from PyHydroGeophysX.qt_apps.qt_utils import (
    BusyStateController,
    ReproduceBar,
    WizardNavigator,
    make_double_spinbox,
)
from PyHydroGeophysX.qt_apps.workers import WorkflowWorker
from PyHydroGeophysX.workflows import (
    ArtifactRef,
    RunContext,
    WorkflowRunResult,
    WorkflowSpec,
    export_workflow_bundle,
)

_STEPS = ["Methods", "Data", "Compatibility", "Parameters", "Run", "Results"]
_ICONS = ["fa5s.project-diagram", "fa5s.database", "fa5s.check-double",
          "fa5s.sliders-h", "fa5s.play", "fa5s.chart-area"]
_P = theme.PALETTE


class JointInversionModule(BaseModule):
    """Six-step workbench page for joint and cooperative inversion."""

    module_key = "joint_inversion"
    module_title = "Joint Inversion"

    def __init__(self, state: Any, log: LogFn, parent=None) -> None:
        super().__init__(state, log, parent)
        self._current = 0
        self._data: Dict[str, Any] = {}
        self._paths: Dict[str, str] = {}
        self._validated = False
        self._pairing_table: Optional[List[Dict[str, int]]] = None
        self._result: Optional[JointInversionResult] = None
        self._worker: Optional[WorkflowWorker] = None
        self._run_busy: Optional[BusyStateController] = None
        self._workflow_recipe_path = ""

        root = QVBoxLayout(self)
        root.addWidget(self._build_strip())
        self._stack = QStackedWidget()
        self._stack.addWidget(self._build_methods_step())
        self._stack.addWidget(self._build_data_step())
        self._stack.addWidget(self._build_compatibility_step())
        self._stack.addWidget(self._build_parameters_step())
        self._stack.addWidget(self._build_run_step())
        self._stack.addWidget(self._build_results_step())
        root.addWidget(self._stack, stretch=1)
        self._reproduce = ReproduceBar()
        root.addWidget(self._reproduce)
        root.addLayout(self._build_nav())
        self._navigator = WizardNavigator(
            self._stack,
            previous_button=self._back,
            next_button=self._next,
            on_changed=self._on_wizard_changed,
            parent=self,
        )
        self._pair_changed()
        self._go_to(0)

    @staticmethod
    def _dspin(value: float, lo: float, hi: float, step: float, decimals: int = 3) -> QDoubleSpinBox:
        return make_double_spinbox(value, lo, hi, step, decimals)

    @staticmethod
    def _ispin(value: int, lo: int, hi: int) -> QSpinBox:
        widget = QSpinBox()
        widget.setRange(lo, hi)
        widget.setValue(value)
        return widget

    def _build_strip(self) -> QWidget:
        bar = QWidget()
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(4, 2, 4, 6)
        layout.setSpacing(5)
        self._chips: List[QPushButton] = []
        for index, name in enumerate(_STEPS):
            chip = QPushButton(f"  {index + 1}. {name}")
            chip.setIcon(theme.icon(_ICONS[index], color="#ffffff"))
            chip.clicked.connect(lambda _checked=False, target=index: self._go_to(target))
            self._chips.append(chip)
            layout.addWidget(chip)
            if index < len(_STEPS) - 1:
                layout.addWidget(QLabel("›"))
        layout.addStretch(1)
        return bar

    def _chip_style(self, state: str) -> str:
        if state == "active":
            return (f"QPushButton{{background:{_P['primary']};color:#fff;border:none;"
                    "border-radius:14px;padding:6px 12px;font-weight:700;}")
        if state == "done":
            return (f"QPushButton{{background:{_P['green']};color:#fff;border:none;"
                    "border-radius:14px;padding:6px 12px;font-weight:600;}")
        return (f"QPushButton{{background:{_P['card']};color:{_P['muted']};"
                f"border:1px solid {_P['border']};border-radius:14px;padding:6px 12px;}}")

    def _status(self) -> List[bool]:
        try:
            capability = get_joint_capability(self._method_a.currentText(), self._method_b.currentText())
        except ValueError:
            return [False] * len(_STEPS)
        methods_ok = (capability.implemented and self._backend_error(capability) == ""
                      and bool(self._strategy.currentData()))
        data_ok = methods_ok and all(method in self._data for method in capability.methods)
        return [methods_ok, data_ok, data_ok and self._validated,
                data_ok and self._validated, self._result is not None, self._result is not None]

    def _update_strip(self) -> None:
        done = self._status()
        for index, chip in enumerate(self._chips):
            state = "active" if index == self._current else "done" if done[index] else "inactive"
            chip.setStyleSheet(self._chip_style(state))
            icon = "fa5s.check" if state == "done" else _ICONS[index]
            color = "#ffffff" if state != "inactive" else _P["muted"]
            chip.setIcon(theme.icon(icon, color=color))

    def _build_nav(self) -> QHBoxLayout:
        layout = QHBoxLayout()
        self._back = QPushButton("Back")
        self._back.setIcon(theme.icon("fa5s.arrow-left"))
        self._next = QPushButton("Next")
        self._next.setIcon(theme.icon("fa5s.arrow-right"))
        layout.addWidget(self._back)
        layout.addStretch(1)
        layout.addWidget(self._next)
        return layout

    def _go_to(self, index: int) -> None:
        self._navigator.go_to(index)

    def _on_wizard_changed(self, index: int) -> None:
        self._current = index
        if self._current == 1:
            self._refresh_resource_choices()
        elif self._current == 2:
            self._draw_alignment()
        elif self._current == 4:
            self._refresh_review()
        self._next.setEnabled(self._current < len(_STEPS) - 1 and self._status()[self._current])
        self._update_strip()

    # -- Step 1 -------------------------------------------------------------
    def _build_methods_step(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(QLabel("<h3>Step 1 · Select two geophysical methods</h3>"))
        layout.addWidget(QLabel(
            "The compatibility registry distinguishes simultaneous joint inversion from "
            "cooperative/sequential constraints. Planned combinations are visible but cannot run."))
        form = QFormLayout()
        self._method_a = QComboBox(); self._method_a.addItems(METHODS)
        self._method_b = QComboBox(); self._method_b.addItems(METHODS)
        self._method_b.setCurrentText("SRT")
        self._strategy = QComboBox()
        form.addRow("Method A", self._method_a)
        form.addRow("Method B", self._method_b)
        form.addRow("Joint strategy", self._strategy)
        layout.addLayout(form)
        self._capability = QLabel(); self._capability.setWordWrap(True)
        layout.addWidget(self._capability)
        table = QTextBrowser()
        lines = ["<b>Available in this release</b><ul>"]
        for item in get_joint_capabilities(False):
            lines.append(f"<li>{item.methods[0]} + {item.methods[1]}: "
                         f"{', '.join(item.strategies.values())}</li>")
        lines.append("</ul><b>Other mixed-dimensional pairs:</b> Planned")
        table.setHtml("".join(lines))
        layout.addWidget(table, stretch=1)
        self._method_a.currentTextChanged.connect(self._pair_changed)
        self._method_b.currentTextChanged.connect(self._pair_changed)
        self._strategy.currentIndexChanged.connect(self._strategy_changed)
        return page

    def _pair_changed(self) -> None:
        try:
            capability = get_joint_capability(self._method_a.currentText(), self._method_b.currentText())
        except ValueError as exc:
            self._capability.setText(f"<span style='color:#b42318'><b>Unavailable:</b> {html.escape(str(exc))}</span>")
            self._strategy.blockSignals(True); self._strategy.clear(); self._strategy.blockSignals(False)
            self._update_strip()
            return
        self._strategy.blockSignals(True)
        self._strategy.clear()
        for strategy_id, label in capability.strategies.items():
            self._strategy.addItem(label, strategy_id)
        self._strategy.blockSignals(False)
        backend_error = self._backend_error(capability)
        if capability.implemented and not backend_error:
            backend_text = ", ".join(capability.backends) or "registered runner"
            self._capability.setText(
                f"<span style='color:{_P['green']}'><b>Available</b></span> · "
                f"{capability.dimension} · {capability.model_parameter} · "
                f"backend: {html.escape(backend_text)}<br>{capability.description}")
        elif backend_error:
            self._capability.setText(
                f"<span style='color:#b42318'><b>Backend unavailable:</b> "
                f"{html.escape(backend_error)}</span>")
        else:
            self._capability.setText(
                "<span style='color:#b42318'><b>Planned:</b> no scientifically validated "
                "joint runner is registered for this pair.</span>")
        self._validated = False
        self._result = None
        self._update_data_labels()
        self._strategy_changed()

    @staticmethod
    def _backend_error(capability: Any) -> str:
        if not capability.implemented:
            return ""
        try:
            if capability.methods == ("ERT", "SRT"):
                import importlib.util
                return "" if importlib.util.find_spec("pygimli") is not None else (
                    "PyGIMLi is required. Install pyhydrogeophysx[geophysics]."
                )
            if capability.methods == ("FDEM", "TDEM"):
                statuses = [em_pipeline.backend_status(method) for method in capability.methods]
                errors = [item["error"] for item in statuses if not item["available"]]
                return "" if not errors else "; ".join(errors)
            if capability.methods == ("Gravity", "Magnetics"):
                status = gravmag_pipeline.backend_status()
                return "" if status["available"] else (
                    f"{status['error']}. Install pyhydrogeophysx[geophysics]."
                )
        except Exception as exc:  # noqa: BLE001
            return str(exc)
        return ""

    def _strategy_changed(self) -> None:
        if hasattr(self, "_preset"):
            try:
                self._refresh_parameter_visibility()
            except ValueError:
                pass
        self._validated = False
        self._update_strip()

    # -- Step 2 -------------------------------------------------------------
    def _build_data_step(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(QLabel("<h3>Step 2 · Load or reuse observed data</h3>"))
        layout.addWidget(QLabel(
            "Joint inversion always requires observations. Existing recovered models may be "
            "used later as starting/reference models, but never replace observed data."))
        self._data_groups: List[QGroupBox] = []
        self._resource_combos: List[QComboBox] = []
        for slot in range(2):
            group = QGroupBox()
            row = QHBoxLayout(group)
            combo = QComboBox()
            combo.currentIndexChanged.connect(lambda _index, target=slot: self._resource_selected(target))
            browse = QPushButton("Upload file…")
            browse.setIcon(theme.icon("fa5s.folder-open"))
            browse.clicked.connect(lambda _checked=False, target=slot: self._browse_data(target))
            row.addWidget(combo, stretch=1); row.addWidget(browse)
            self._data_groups.append(group); self._resource_combos.append(combo)
            layout.addWidget(group)
        example = QPushButton("Use bundled example for selected pair")
        example.setIcon(theme.icon("fa5s.flask"))
        example.clicked.connect(self._use_example)
        layout.addWidget(example)
        self._data_status = QTextBrowser()
        layout.addWidget(self._data_status, stretch=1)
        return page

    def _selected_methods(self) -> Tuple[str, str]:
        capability = get_joint_capability(self._method_a.currentText(), self._method_b.currentText())
        return capability.methods

    def _update_data_labels(self) -> None:
        if not hasattr(self, "_data_groups"):
            return
        try:
            methods = self._selected_methods()
        except ValueError:
            methods = (self._method_a.currentText(), self._method_b.currentText())
        for slot, method in enumerate(methods):
            self._data_groups[slot].setTitle(f"{method} observations")
        self._refresh_resource_choices()

    def _refresh_resource_choices(self) -> None:
        if not hasattr(self, "_resource_combos"):
            return
        try:
            methods = self._selected_methods()
        except ValueError:
            return
        for slot, method in enumerate(methods):
            combo = self._resource_combos[slot]
            selected = combo.currentData()
            combo.blockSignals(True); combo.clear()
            combo.addItem("Choose a project resource or upload a file", None)
            if hasattr(self.state, "list_geophysical_resources"):
                for resource in self.state.list_geophysical_resources(method, "observed_data"):
                    combo.addItem(resource["label"], resource["id"])
            if selected:
                index = combo.findData(selected)
                if index >= 0:
                    combo.setCurrentIndex(index)
            combo.blockSignals(False)
        self._render_data_status()

    def _resource_selected(self, slot: int) -> None:
        resource_id = self._resource_combos[slot].currentData()
        if not resource_id or not hasattr(self.state, "get_geophysical_resource"):
            return
        resource = self.state.get_geophysical_resource(resource_id)
        if resource is None:
            return
        method = self._selected_methods()[slot]
        payload = resource["payload"]
        path = resource.get("path", "")
        # EM resources from the processing page represent the active sounding.
        # Reload all soundings from their source file when available.
        if (method in {"FDEM", "TDEM"} and path
                and not (isinstance(payload, dict) and "soundings" in payload)):
            try:
                payload = self._load_em_file(path, method)
            except Exception as exc:
                self.log(f"Could not reload full {method} resource: {exc}", "warn")
        self._data[method] = payload
        self._paths[method] = path
        if method == "Magnetics":
            field = resource.get("metadata", {}).get("field", {})
            if field:
                self._pf_strength.setValue(float(field.get("strength_nT", self._pf_strength.value())))
                self._pf_inclination.setValue(float(field.get("inclination", self._pf_inclination.value())))
                self._pf_declination.setValue(float(field.get("declination", self._pf_declination.value())))
        self._validated = False
        self._render_data_status(); self._update_strip()

    def _browse_data(self, slot: int) -> None:
        method = self._selected_methods()[slot]
        if method == "ERT":
            filters = "ERT data (*.dat *.ohm *.txt);;All files (*)"
        elif method == "SRT":
            filters = "Travel-time data (*.dat *.sgt *.tt *.gtt *.txt);;All files (*)"
        elif method in {"FDEM", "TDEM"}:
            filters = "EM tables (*.csv *.txt *.dat);;All files (*)"
        elif method in {"Gravity", "Magnetics"}:
            filters = "Station data (*.csv *.txt *.dat);;All files (*)"
        else:
            return
        path, _ = QFileDialog.getOpenFileName(self, f"Load {method} observations", "", filters)
        if not path:
            return
        try:
            if method == "ERT":
                from pygimli.physics import ert
                payload = ert.load(path)
            elif method == "SRT":
                import pygimli.physics.traveltime as tt
                payload = tt.load(path)
            elif method in {"Gravity", "Magnetics"}:
                table = io_utils.load_xyz_table(path, min_cols=3)
                payload = {
                    "x": np.asarray(table[:, 0], dtype=float),
                    "y": np.asarray(table[:, 1], dtype=float),
                    "value": np.asarray(table[:, 2], dtype=float),
                    "z": (np.asarray(table[:, 3], dtype=float) if table.shape[1] >= 4
                          else np.ones(table.shape[0], dtype=float)),
                }
            else:
                payload = self._load_em_file(path, method)
        except Exception as exc:  # noqa: BLE001
            self.log(f"Could not load {method} data: {exc}", "error")
            return
        self._data[method] = payload; self._paths[method] = path
        if hasattr(self.state, "register_geophysical_resource"):
            self.state.register_geophysical_resource(
                method, "observed_data", payload, label=f"{method} observations · {Path(path).name}",
                path=path, resource_id=f"{method.lower()}:observed_data:joint",
            )
        self._validated = False
        self._refresh_resource_choices(); self._render_data_status(); self._update_strip()

    @staticmethod
    def _load_em_file(path: str, method: str) -> Any:
        first = em_pipeline.load_sounding(path, method, sounding=0)
        count = int(first.get("n_soundings", 1))
        if count == 1:
            return first
        return {"soundings": [em_pipeline.load_sounding(path, method, sounding=index)
                              for index in range(count)]}

    def _use_example(self) -> None:
        pair = self._selected_methods()
        root = Path(__file__).resolve().parents[3] / "examples" / "data"
        try:
            if pair == ("ERT", "SRT"):
                ert_path = root / "ERT" / "Bert" / "fielddataline2.dat"
                srt_path = root / "Seismic" / "srtfieldline2.dat"
                from pygimli.physics import ert
                import pygimli.physics.traveltime as tt
                self._data["ERT"] = ert.load(str(ert_path))
                self._data["SRT"] = tt.load(str(srt_path))
                self._paths.update({"ERT": str(ert_path), "SRT": str(srt_path)})
            elif pair == ("FDEM", "TDEM"):
                f_path = root / "EM" / "joint_synthetic_fdem.csv"
                t_path = root / "EM" / "joint_synthetic_tdem.csv"
                self._data["FDEM"] = self._load_em_file(str(f_path), "FDEM")
                self._data["TDEM"] = self._load_em_file(str(t_path), "TDEM")
                self._paths.update({"FDEM": str(f_path), "TDEM": str(t_path)})
            elif pair == ("Gravity", "Magnetics"):
                from PyHydroGeophysX.forward.gravmag import (
                    gravity_sphere,
                    magnetic_dipole,
                )
                grid_x, grid_y = np.meshgrid(
                    np.linspace(-100.0, 100.0, 15), np.linspace(-100.0, 100.0, 15)
                )
                x = grid_x.ravel(); y = grid_y.ravel(); z = np.ones(x.size)
                body = {
                    "x0": 0.0, "y0": 0.0, "z0": 45.0, "radius": 28.0,
                    "density": 0.3, "susceptibility": 0.035,
                }
                field = {"strength": 50000.0, "inclination": 60.0, "declination": 0.0}
                self._data["Gravity"] = {
                    "x": x, "y": y, "z": z,
                    "value": gravity_sphere(x, y, body),
                }
                self._data["Magnetics"] = {
                    "x": x, "y": y, "z": z,
                    "value": magnetic_dipole(x, y, body, field),
                }
                self._paths.update({
                    "Gravity": "bundled paired synthetic",
                    "Magnetics": "bundled paired synthetic",
                })
            else:
                raise ValueError("No runnable bundled example exists for this planned pair.")
        except Exception as exc:  # noqa: BLE001
            self.log(f"Could not load joint example: {exc}", "error")
            return
        self._validated = False
        self._render_data_status(); self._update_strip()
        self.log(f"Loaded bundled {pair[0]} + {pair[1]} joint example.", "success")

    def _render_data_status(self) -> None:
        if not hasattr(self, "_data_status"):
            return
        lines = ["<b>Current observations</b><ul>"]
        for method in self._selected_methods():
            value = self._data.get(method)
            if value is None:
                lines.append(f"<li>{method}: <span style='color:#b42318'>not loaded</span></li>")
                continue
            if method in {"ERT", "SRT"}:
                count = int(value.size())
                text = f"{count} measurements"
            elif method in {"Gravity", "Magnetics"}:
                text = f"{np.asarray(value['value']).size} stations"
            else:
                soundings = value.get("soundings") if isinstance(value, dict) else None
                text = f"{len(soundings) if soundings else 1} sounding(s)"
            lines.append(f"<li>{method}: {text} · {html.escape(self._paths.get(method, 'project resource'))}</li>")
        lines.append("</ul>")
        self._data_status.setHtml("".join(lines))

    # -- Step 3 -------------------------------------------------------------
    def _build_compatibility_step(self) -> QWidget:
        page = QWidget(); layout = QVBoxLayout(page)
        layout.addWidget(QLabel("<h3>Step 3 · Check geometry and model compatibility</h3>"))
        self._alignment_note = QLabel(); self._alignment_note.setWordWrap(True)
        layout.addWidget(self._alignment_note)
        offsets = QGroupBox("Explicit profile-coordinate offsets (ERT + SRT only)")
        form = QFormLayout(offsets)
        self._ert_x = self._dspin(0, -1e6, 1e6, 1, 2)
        self._ert_z = self._dspin(0, -1e6, 1e6, 1, 2)
        self._srt_x = self._dspin(0, -1e6, 1e6, 1, 2)
        self._srt_z = self._dspin(0, -1e6, 1e6, 1, 2)
        form.addRow("ERT x offset (m)", self._ert_x); form.addRow("ERT z offset (m)", self._ert_z)
        form.addRow("SRT x offset (m)", self._srt_x); form.addRow("SRT z offset (m)", self._srt_z)
        self._offset_group = offsets; layout.addWidget(offsets)
        pairing_group = QGroupBox("Explicit EM sounding pairing")
        pairing_row = QHBoxLayout(pairing_group)
        pairing_button = QPushButton("Load pairing CSV…")
        pairing_button.clicked.connect(self._load_pairing_table)
        self._pairing_label = QLabel("No pairing table loaded.")
        pairing_row.addWidget(pairing_button); pairing_row.addWidget(self._pairing_label, stretch=1)
        self._pairing_group = pairing_group; layout.addWidget(pairing_group)
        self._alignment_figure = Figure(figsize=(7, 3), tight_layout=True)
        self._alignment_canvas = FigureCanvas(self._alignment_figure)
        layout.addWidget(self._alignment_canvas, stretch=1)
        validate = QPushButton("Validate compatibility")
        validate.setProperty("primary", True); validate.clicked.connect(self._validate)
        layout.addWidget(validate)
        return page

    def _draw_alignment(self) -> None:
        pair = self._selected_methods()
        self._offset_group.setVisible(pair == ("ERT", "SRT"))
        self._pairing_group.setVisible(pair == ("FDEM", "TDEM"))
        self._alignment_figure.clear(); axis = self._alignment_figure.add_subplot(111)
        if pair == ("ERT", "SRT") and all(method in self._data for method in pair):
            for method, color in (("ERT", "#d95f02"), ("SRT", "#1b9e77")):
                positions = self._data[method].sensorPositions()
                x = np.asarray([float(position.x()) for position in positions])
                z = np.asarray([float(position.y()) for position in positions])
                x += self._ert_x.value() if method == "ERT" else self._srt_x.value()
                z += self._ert_z.value() if method == "ERT" else self._srt_z.value()
                axis.scatter(x, z, s=22, label=method, color=color)
            axis.set_xlabel("Profile distance (m)"); axis.set_ylabel("Elevation / z (m)")
            axis.legend(); axis.grid(alpha=0.25)
            self._alignment_note.setText(
                "Offsets are never inferred or applied automatically. Confirm that both sensor arrays "
                "belong to the same profile before validation.")
        elif pair == ("FDEM", "TDEM"):
            counts = []
            for method in pair:
                value = self._data.get(method, {})
                counts.append(len(value.get("soundings", [])) or (1 if value else 0))
            axis.bar(pair, counts, color=("#4c78a8", "#f58518")); axis.set_ylabel("Soundings")
            self._alignment_note.setText(
                "With coordinates, soundings are matched one-to-one by distance. Without coordinates, "
                "equal-sized lines are paired by index; unequal counts require a pairing table.")
        elif pair == ("Gravity", "Magnetics") and all(method in self._data for method in pair):
            for method, color, marker in (
                ("Gravity", "#4c78a8", "o"), ("Magnetics", "#e45756", "+")
            ):
                value = self._data[method]
                axis.scatter(
                    np.asarray(value["x"], dtype=float),
                    np.asarray(value["y"], dtype=float),
                    s=16, alpha=0.65, label=method, color=color, marker=marker,
                )
            axis.set_xlabel("Easting / x (m)"); axis.set_ylabel("Northing / y (m)")
            axis.legend(); axis.grid(alpha=0.25); axis.set_aspect("equal", adjustable="datalim")
            self._alignment_note.setText(
                "Both surveys must use the same projected coordinate system and overlap in x and y. "
                "The workflow does not infer a reprojection or coordinate offset."
            )
        else:
            axis.text(0.5, 0.5, "Load both observations first.", ha="center", va="center")
        self._alignment_canvas.draw_idle()

    def _load_pairing_table(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load FDEM/TDEM pairing table", "", "CSV (*.csv);;All files (*)"
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            pairing = [
                {"fdem_index": int(row["fdem_index"]), "tdem_index": int(row["tdem_index"])}
                for row in rows
            ]
            if not pairing:
                raise ValueError("The table contains no pairs.")
        except Exception as exc:  # noqa: BLE001
            self.log(
                "Could not load pairing table. Expected zero-based columns "
                f"fdem_index,tdem_index: {exc}", "error"
            )
            return
        self._pairing_table = pairing
        self._pairing_label.setText(f"{Path(path).name} · {len(pairing)} pair(s)")
        self._validated = False; self._update_strip()

    def _validate(self) -> None:
        pair = self._selected_methods()
        if not all(method in self._data for method in pair):
            self._validated = False; self.log("Load both observed datasets first.", "warn"); return
        try:
            if pair == ("ERT", "SRT"):
                extents = []
                for method in pair:
                    positions = self._data[method].sensorPositions()
                    offset = self._ert_x.value() if method == "ERT" else self._srt_x.value()
                    xs = np.asarray([float(position.x()) + offset for position in positions])
                    extents.append((float(xs.min()), float(xs.max())))
                if min(extents[0][1], extents[1][1]) <= max(extents[0][0], extents[1][0]):
                    raise ValueError("ERT and SRT profile ranges do not overlap.")
                parameters = self._collect_parameters()
                interface = parameters.get("interface_coords")
                if self._strategy.currentData() == "sequential_structure" and interface is not None:
                    positions = self._data["ERT"].sensorPositions()
                    validate_profile_interface(
                        interface,
                        [float(position.x()) + self._ert_x.value() for position in positions],
                        [float(position.y()) + self._ert_z.value() for position in positions],
                    )
            elif pair == ("FDEM", "TDEM"):
                pairs = pair_joint_soundings(
                    self._data["FDEM"], self._data["TDEM"], self._collect_parameters()
                )
                self.log(f"Validated {len(pairs)} FDEM/TDEM sounding pair(s).", "info")
            elif pair == ("Gravity", "Magnetics"):
                extents = {}
                for method in pair:
                    value = self._data[method]
                    x = np.asarray(value["x"], dtype=float)
                    y = np.asarray(value["y"], dtype=float)
                    observed = np.asarray(value["value"], dtype=float)
                    if not (x.size == y.size == observed.size) or x.size < 20:
                        raise ValueError(f"{method} needs at least 20 matching x/y/value stations.")
                    if not np.all(np.isfinite(np.r_[x, y, observed])):
                        raise ValueError(f"{method} station data contain non-finite values.")
                    extents[method] = (x.min(), x.max(), y.min(), y.max())
                gravity_extent = extents["Gravity"]
                magnetic_extent = extents["Magnetics"]
                x_overlap = min(gravity_extent[1], magnetic_extent[1]) > max(
                    gravity_extent[0], magnetic_extent[0]
                )
                y_overlap = min(gravity_extent[3], magnetic_extent[3]) > max(
                    gravity_extent[2], magnetic_extent[2]
                )
                if not (x_overlap and y_overlap):
                    raise ValueError("Gravity and Magnetics station footprints do not overlap in x/y.")
                self.log("Validated overlapping Gravity/Magnetics station footprints.", "info")
        except Exception as exc:
            self._validated = False; self.log(f"Compatibility check failed: {exc}", "error")
        else:
            self._validated = True; self.log("Joint data compatibility check passed.", "success")
        self._update_strip(); self._go_to(self._current)

    # -- Step 4 -------------------------------------------------------------
    def _build_parameters_step(self) -> QWidget:
        page = QWidget(); outer = QVBoxLayout(page)
        outer.addWidget(QLabel("<h3>Step 4 · Choose a preset and tune joint parameters</h3>"))
        form = QFormLayout()
        self._preset = QComboBox(); self._preset.currentIndexChanged.connect(self._apply_preset)
        self._baseline = QCheckBox("Run independent single-method inversions for comparison")
        self._baseline.setChecked(True)
        self._max_iter = self._ispin(20, 1, 200)
        form.addRow("Preset", self._preset); form.addRow("Maximum iterations", self._max_iter)
        form.addRow("Baseline comparison", self._baseline)
        outer.addLayout(form)
        advanced = QGroupBox("Advanced")
        advanced.setCheckable(True); advanced.setChecked(False)
        advanced_inner = QWidget()
        advanced_form = QFormLayout(advanced_inner)
        advanced_layout = QVBoxLayout(advanced)
        advanced_layout.setContentsMargins(8, 4, 8, 8)
        advanced_layout.addWidget(advanced_inner)
        advanced_inner.setVisible(False)
        advanced.toggled.connect(advanced_inner.setVisible)
        self._lambda_a = self._dspin(10, 0, 1e7, 1)
        self._lambda_b = self._dspin(10, 0, 1e7, 1)
        self._lambda_cg_a = self._dspin(120, 0, 1e7, 10)
        self._lambda_cg_b = self._dspin(80, 0, 1e7, 10)
        self._relative_error = self._dspin(0.05, 0.001, 1.0, 0.01, 3)
        self._velocity_threshold = self._dspin(1000, 50, 20000, 50, 0)
        self._use_interface = QCheckBox("Use current project seismic interface when available")
        self._use_interface.setChecked(True)
        self._n_layers = self._ispin(15, 2, 100)
        self._smoothness = self._dspin(0.3, 0, 1000, 0.1)
        self._backend = QComboBox()
        self._backend.addItem("Auto (SimPEG native, SciPy fallback)", "auto")
        self._backend.addItem("SimPEG native", "simpeg")
        self._backend.addItem("SciPy fallback", "scipy")
        self._backend.setToolTip(
            "Auto uses native SimPEG joint inversion when SimPEG >= 0.25.2 is available."
        )
        self._weight_a = self._dspin(1, 0.001, 1000, 0.1)
        self._weight_b = self._dspin(1, 0.001, 1000, 0.1)
        self._pair_tolerance = self._dspin(0, 0, 1e6, 1, 2)
        self._pf_nxy = self._ispin(12, 4, 40)
        self._pf_nz = self._ispin(8, 3, 30)
        self._pf_cross = self._dspin(2e12, 0, 1e15, 1e11, 0)
        self._pf_gravity_floor = self._dspin(0.5, 0, 1e6, 0.1, 3)
        self._pf_magnetics_floor = self._dspin(2.0, 0, 1e6, 0.5, 3)
        self._pf_detrend = self._ispin(0, 0, 3)
        self._pf_strength = self._dspin(50000, 100, 100000, 100, 0)
        self._pf_inclination = self._dspin(60, -90, 90, 1, 1)
        self._pf_declination = self._dspin(0, -180, 180, 1, 1)
        self._advanced_rows = {
            "lambda_a": ("λ method A", self._lambda_a), "lambda_b": ("λ method B", self._lambda_b),
            "lambda_cg_a": ("Cross-gradient λ A", self._lambda_cg_a),
            "lambda_cg_b": ("Cross-gradient λ B", self._lambda_cg_b),
            "relative_error": ("Relative error", self._relative_error),
            "velocity_threshold": ("Velocity interface (m/s)", self._velocity_threshold),
            "use_interface": ("Existing interface", self._use_interface),
            "backend": ("Solver backend", self._backend),
            "n_layers": ("EM layers", self._n_layers), "smoothness": ("EM smoothness", self._smoothness),
            "weight_a": ("FDEM weight", self._weight_a), "weight_b": ("TDEM weight", self._weight_b),
            "pair_tolerance": ("Pairing tolerance (m; 0=auto)", self._pair_tolerance),
            "pf_nxy": ("Potential-field lateral cells", self._pf_nxy),
            "pf_nz": ("Potential-field depth cells", self._pf_nz),
            "pf_cross": ("Cross-gradient weight", self._pf_cross),
            "pf_gravity_floor": ("Gravity noise floor (mGal)", self._pf_gravity_floor),
            "pf_magnetics_floor": ("Magnetics noise floor (nT)", self._pf_magnetics_floor),
            "pf_detrend": ("Regional detrend degree", self._pf_detrend),
            "pf_strength": ("Inducing field strength (nT)", self._pf_strength),
            "pf_inclination": ("Inducing field inclination (°)", self._pf_inclination),
            "pf_declination": ("Inducing field declination (°)", self._pf_declination),
        }
        for label, widget in self._advanced_rows.values():
            advanced_form.addRow(label, widget)
        outer.addWidget(advanced)
        outer.addStretch(1)
        return page

    def _refresh_parameter_visibility(self) -> None:
        pair = self._selected_methods(); strategy = self._strategy.currentData()
        if pair == ("ERT", "SRT"):
            if strategy == "cross_gradient_geostatistical":
                presets = [("Spatial / geostatistical", "geostat")]
            elif strategy == "sequential_structure":
                presets = [("Sequential structure constraint", "sequential")]
            else:
                presets = [("Standard direct", "direct")]
        elif pair == ("FDEM", "TDEM"):
            presets = [("Balanced shared conductivity", "em")]
        else:
            presets = [("SimPEG cross-gradient", "potential_field")]
        current = self._preset.currentData()
        self._preset.blockSignals(True); self._preset.clear()
        for label, value in presets:
            self._preset.addItem(label, value)
        index = self._preset.findData(current)
        self._preset.setCurrentIndex(max(index, 0)); self._preset.blockSignals(False)
        ert_joint = pair == ("ERT", "SRT") and strategy != "sequential_structure"
        sequential = strategy == "sequential_structure"
        em = pair == ("FDEM", "TDEM")
        potential_field = pair == ("Gravity", "Magnetics")
        visibility = {
            "lambda_a": ert_joint, "lambda_b": ert_joint,
            "lambda_cg_a": ert_joint, "lambda_cg_b": ert_joint,
            "relative_error": ert_joint or em or potential_field, "velocity_threshold": sequential,
            "use_interface": sequential,
            "backend": em, "n_layers": em, "smoothness": em,
            "weight_a": em or potential_field, "weight_b": em or potential_field,
            "pair_tolerance": em,
            "pf_nxy": potential_field, "pf_nz": potential_field,
            "pf_cross": potential_field, "pf_gravity_floor": potential_field,
            "pf_magnetics_floor": potential_field, "pf_detrend": potential_field,
            "pf_strength": potential_field, "pf_inclination": potential_field,
            "pf_declination": potential_field,
        }
        weight_labels = (
            ("Gravity weight", "Magnetics weight") if potential_field
            else ("FDEM weight", "TDEM weight")
        )
        for widget, label in ((self._weight_a, weight_labels[0]), (self._weight_b, weight_labels[1])):
            label_widget = widget.parentWidget().layout().labelForField(widget)
            if label_widget is not None:
                label_widget.setText(label)
            widget.setVisible(em or potential_field)
            if label_widget is not None:
                label_widget.setVisible(em or potential_field)
        for key, (_label, widget) in self._advanced_rows.items():
            widget.setVisible(visibility[key])
            label_widget = widget.parentWidget().layout().labelForField(widget)
            if label_widget is not None:
                label_widget.setVisible(visibility[key])
        self._apply_preset()

    def _apply_preset(self) -> None:
        preset = self._preset.currentData()
        if preset == "direct":
            self._lambda_a.setValue(10); self._lambda_b.setValue(10)
            self._lambda_cg_a.setValue(120); self._lambda_cg_b.setValue(80)
        elif preset == "geostat":
            self._lambda_a.setValue(10); self._lambda_b.setValue(10)
            self._lambda_cg_a.setValue(5000); self._lambda_cg_b.setValue(5000)
        elif preset == "em":
            self._n_layers.setValue(15); self._smoothness.setValue(0.3)
            self._relative_error.setValue(0.05)
            self._weight_a.setValue(1); self._weight_b.setValue(1)
        elif preset == "potential_field":
            self._pf_nxy.setValue(12); self._pf_nz.setValue(8)
            self._pf_cross.setValue(2e12); self._relative_error.setValue(0.03)
            self._pf_gravity_floor.setValue(0.5); self._pf_magnetics_floor.setValue(2.0)
            self._weight_a.setValue(1); self._weight_b.setValue(1)

    def _collect_parameters(self) -> Dict[str, Any]:
        pair = self._selected_methods(); strategy = self._strategy.currentData()
        if pair == ("ERT", "SRT"):
            params: Dict[str, Any] = {
                "max_iterations": self._max_iter.value(),
                "ert_x_offset": self._ert_x.value(), "ert_z_offset": self._ert_z.value(),
                "srt_x_offset": self._srt_x.value(), "srt_z_offset": self._srt_z.value(),
            }
            if strategy == "sequential_structure":
                params["velocity_threshold"] = self._velocity_threshold.value()
                if self._use_interface.isChecked():
                    shared = getattr(self.state, "shared_structure", None) or {}
                    interface = shared.get("interface_xz")
                    if interface is None and hasattr(self.state, "list_geophysical_resources"):
                        resources = self.state.list_geophysical_resources("SRT", "interface")
                        if resources:
                            resource = self.state.get_geophysical_resource(resources[-1]["id"])
                            interface = resource.get("payload") if resource else None
                    if interface is not None and len(interface):
                        points = np.asarray(interface, dtype=float)
                        if points.ndim == 2 and points.shape[0] == 2 and points.shape[1] > 2:
                            params["interface_coords"] = (points[0], points[1])
                        elif points.ndim == 2 and points.shape[1] >= 2:
                            params["interface_coords"] = (points[:, 0], points[:, 1])
            else:
                params.update({
                    "lambda_ert": self._lambda_a.value(), "lambda_srt": self._lambda_b.value(),
                    "lambda_cg_ert": self._lambda_cg_a.value(), "lambda_cg_srt": self._lambda_cg_b.value(),
                    "ert_relative_error": self._relative_error.value(),
                    "srt_relative_error": self._relative_error.value(), "verbose": False,
                })
            return params
        if pair == ("Gravity", "Magnetics"):
            return {
                "n_xy": self._pf_nxy.value(), "n_z": self._pf_nz.value(),
                "max_iterations": self._max_iter.value(),
                "gravity_relative_error": self._relative_error.value(),
                "magnetics_relative_error": self._relative_error.value(),
                "gravity_noise_floor": self._pf_gravity_floor.value(),
                "magnetics_noise_floor": self._pf_magnetics_floor.value(),
                "gravity_weight": self._weight_a.value(),
                "magnetics_weight": self._weight_b.value(),
                "cross_gradient_weight": self._pf_cross.value(),
                "gravity_detrend": self._pf_detrend.value(),
                "magnetics_detrend": self._pf_detrend.value(),
                "field": {
                    "strength_nT": self._pf_strength.value(),
                    "inclination": self._pf_inclination.value(),
                    "declination": self._pf_declination.value(),
                },
            }
        tolerance = self._pair_tolerance.value()
        return {
            "n_layers": self._n_layers.value(), "max_iterations": self._max_iter.value(),
            "backend": self._backend.currentData(),
            "smoothness": self._smoothness.value(), "fdem_weight": self._weight_a.value(),
            "tdem_weight": self._weight_b.value(),
            "fdem_relative_error": self._relative_error.value(),
            "tdem_relative_error": self._relative_error.value(),
            **({"pairing_tolerance": tolerance} if tolerance > 0 else {}),
            **({"pairing_table": list(self._pairing_table)} if self._pairing_table else {}),
        }

    # -- Step 5 -------------------------------------------------------------
    def _build_run_step(self) -> QWidget:
        page = QWidget(); layout = QVBoxLayout(page)
        layout.addWidget(QLabel("<h3>Step 5 · Review and run</h3>"))
        self._review = QTextBrowser(); layout.addWidget(self._review, stretch=1)
        self._run_status = QLabel("Ready."); self._run_status.setWordWrap(True)
        layout.addWidget(self._run_status)
        self._progress = QProgressBar(); self._progress.setVisible(False)
        layout.addWidget(self._progress)
        row = QHBoxLayout()
        self._run_button = QPushButton("Run joint inversion")
        self._run_button.setProperty("primary", True); self._run_button.clicked.connect(self._run)
        self._cancel_button = QPushButton("Cancel"); self._cancel_button.setEnabled(False)
        self._cancel_button.clicked.connect(self._cancel)
        row.addWidget(self._run_button); row.addWidget(self._cancel_button); row.addStretch(1)
        layout.addLayout(row)
        return page

    def _refresh_review(self) -> None:
        pair = self._selected_methods(); strategy = self._strategy.currentText()
        payload = {
            "methods": pair, "strategy": strategy, "baseline": self._baseline.isChecked(),
            "data": {method: self._paths.get(method, "project resource") for method in pair},
            "parameters": self._collect_parameters(),
        }
        self._review.setPlainText(json.dumps(payload, indent=2, default=str))
        self._run_button.setEnabled(self._validated and self._worker is None)

    def _request(self) -> JointInversionRequest:
        pair = self._selected_methods()
        out = Path(self.state.output_dir or Path.cwd()) / "joint_inversion" / (
            f"{pair[0].lower()}_{pair[1].lower()}_{self._strategy.currentData()}")
        return JointInversionRequest(
            method_a=pair[0], method_b=pair[1], strategy=str(self._strategy.currentData()),
            data={method: self._data[method] for method in pair},
            parameters=self._collect_parameters(), output_dir=out,
            run_baseline=self._baseline.isChecked(),
        )

    def _run(self) -> None:
        if not self._validated:
            self.log("Validate data compatibility before running.", "warn"); return
        request = self._request()
        try:
            run = self.begin_persisted_run(
                "joint_inversion.run",
                workflow_id="joint_inversion.run",
            )
        except Exception as exc:  # noqa: BLE001
            self.log(f"Could not prepare Project run: {exc}", "error")
            return
        input_dir = run.inputs_dir
        references: Dict[str, ArtifactRef] = {}
        for method, payload in request.data.items():
            try:
                stored = save_joint_observations(
                    method,
                    payload,
                    input_dir / str(method).lower(),
                )
            except Exception as exc:  # noqa: BLE001
                self.fail_persisted_run(str(exc))
                self.log(f"Could not persist joint-inversion inputs: {exc}", "error")
                return
            references[method] = ArtifactRef.from_path(
                stored,
                artifact_id=f"joint-input:{method.lower()}",
                kind="joint_observations",
                base_dir=run.run_dir,
                metadata={"method": method},
            )
        parameters = dict(request.parameters)
        interface = parameters.get("interface_coords")
        if interface is not None:
            parameters["interface_coords"] = [
                np.asarray(axis, dtype=float).tolist() for axis in interface
            ]
        spec = WorkflowSpec(
            workflow_id="joint_inversion.run",
            inputs={"data": references},
            parameters={
                "method_a": request.method_a,
                "method_b": request.method_b,
                "strategy": request.strategy,
                "run_baseline": request.run_baseline,
                **parameters,
            },
            metadata={"source": "qt", "input_order": list(request.data)},
        )
        recipe_path, script_path = export_workflow_bundle(
            spec, run.run_dir, stem="joint_inversion"
        )
        self._reproduce.set_bundle(recipe_path, script_path)
        self._workflow_recipe_path = str(recipe_path)
        self._run_busy = BusyStateController(
            [self._run_button, self._cancel_button]
        )
        self._run_busy.start(enabled_while_busy=[self._cancel_button])
        self._progress.setVisible(True); self._progress.setRange(0, 0)
        self._run_status.setText("Starting joint inversion…")
        worker = WorkflowWorker(
            spec,
            RunContext(project_root=run.run_dir, output_dir=run.outputs_dir),
        )
        worker.logged.connect(self._on_workflow_progress)
        worker.succeeded.connect(self._on_workflow_success)
        worker.failed.connect(self._on_failure)
        worker.finished.connect(self._on_finished)
        self._worker = self.register_worker(worker); worker.start()

    def _on_workflow_progress(self, message: str) -> None:
        try:
            record = json.loads(message)
        except (TypeError, ValueError):
            self._run_status.setText(str(message))
        else:
            self._on_progress(record if isinstance(record, dict) else {"message": message})

    def _on_workflow_success(self, result: WorkflowRunResult) -> None:
        domain_result = result.objects.get("domain_result")
        if not isinstance(domain_result, JointInversionResult):
            self._on_failure("Joint workflow did not return its in-process domain result.")
            return
        if hasattr(self.state, "update_workflow_result"):
            self.state.update_workflow_result(
                self.module_key,
                "joint_inversion.run",
                result.to_dict(),
                recipe_path=self._workflow_recipe_path,
            )
        self._on_success(domain_result)

    def _cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel(); self._run_status.setText("Cancelling after the current solver step…")
            self.cancel_persisted_run("Cancelled by user", "joint_inversion.run")

    def _on_progress(self, record: Dict[str, Any]) -> None:
        if "message" in record:
            text = str(record["message"])
        elif "sounding" in record:
            text = f"Sounding {record['sounding']}"
            if "soundings_total" in record:
                text += f" / {record['soundings_total']}"
        else:
            text = f"Iteration {record.get('iteration', len(record))}"
        metrics = [f"{key}={value:.3g}" for key, value in record.items()
                   if key.startswith("chi2") and isinstance(value, (int, float))]
        self._run_status.setText(text + (" · " + ", ".join(metrics) if metrics else ""))

    def _on_success(self, result: JointInversionResult) -> None:
        self._result = result
        self._populate_results(result)
        self.report_result(result.summary())
        if hasattr(self.state, "register_geophysical_resource"):
            for method, model in result.models.items():
                resource_method = method if method in METHODS else "+".join(result.methods)
                self.state.register_geophysical_resource(
                    resource_method, "model", model,
                    label=f"Joint {resource_method} model", path=result.artifacts.get("arrays", ""),
                    metadata={"strategy": result.strategy, "chi2": result.chi2},
                    resource_id=f"joint:{str(resource_method).lower()}:latest",
                )
        self.log(f"{result.methods[0]} + {result.methods[1]} joint inversion complete.", "success")
        self._go_to(5)

    def _on_failure(self, message: str) -> None:
        self.fail_persisted_run(message)
        self._run_status.setText(f"Failed: {message}"); self.log(f"Joint inversion failed: {message}", "error")

    def _on_finished(self) -> None:
        self._worker = None; self._progress.setVisible(False)
        if self._run_busy is not None:
            self._run_busy.finish()
            self._run_busy = None
        self._cancel_button.setEnabled(False)
        self._run_button.setEnabled(self._validated)

    # -- Step 6 -------------------------------------------------------------
    def _build_results_step(self) -> QWidget:
        page = QWidget(); layout = QVBoxLayout(page)
        header = QHBoxLayout(); header.addWidget(QLabel("<h3>Step 6 · Joint inversion results</h3>"))
        header.addStretch(1)
        open_button = QPushButton("Open output folder"); open_button.clicked.connect(self._open_output)
        header.addWidget(open_button); layout.addLayout(header)
        self._result_tabs = QTabWidget(); layout.addWidget(self._result_tabs, stretch=1)
        self._overview = QTextBrowser(); self._result_tabs.addTab(self._overview, "Overview")
        self._models_figure = Figure(figsize=(8, 4), tight_layout=True)
        self._models_canvas = FigureCanvas(self._models_figure); self._result_tabs.addTab(self._models_canvas, "Models")
        self._fit_figure = Figure(figsize=(8, 4), tight_layout=True)
        self._fit_canvas = FigureCanvas(self._fit_figure); self._result_tabs.addTab(self._fit_canvas, "Data Fit")
        baseline_page = QWidget(); baseline_layout = QVBoxLayout(baseline_page)
        self._baseline_view = QTextBrowser(); baseline_layout.addWidget(self._baseline_view)
        self._baseline_figure = Figure(figsize=(8, 3), tight_layout=True)
        self._baseline_canvas = FigureCanvas(self._baseline_figure)
        baseline_layout.addWidget(self._baseline_canvas, stretch=1)
        self._result_tabs.addTab(baseline_page, "Baseline Comparison")
        self._files = QTextBrowser(); self._result_tabs.addTab(self._files, "Files")
        return page

    def _populate_results(self, result: JointInversionResult) -> None:
        chi = "".join(f"<li>{method}: χ² = {value:.4g}</li>" for method, value in result.chi2.items())
        counts = "".join(
            f"<li>{html.escape(str(method))}: {int(value)} data</li>"
            for method, value in result.meta.get("data_counts", {}).items()
        )
        warnings = "".join(f"<li>{html.escape(item)}</li>" for item in result.warnings)
        backend = html.escape(str(result.meta.get("backend", "")))
        backend_version = html.escape(str(result.meta.get("backend_version", "")))
        backend_label = f"{backend} {backend_version}".strip()
        backend_line = f"<b>Backend:</b> {backend_label}<br>" if backend_label else ""
        self._overview.setHtml(
            f"<h3>{result.methods[0]} + {result.methods[1]}</h3>"
            f"<p><b>Strategy:</b> {html.escape(result.strategy)}<br>"
            f"{backend_line}"
            f"<b>Status:</b> {html.escape(result.status)}<br>"
            f"<b>Iterations:</b> {len(result.history)}</p>"
            f"<b>Data</b><ul>{counts}</ul><b>Fit</b><ul>{chi}</ul>"
            + (f"<b>Warnings</b><ul>{warnings}</ul>" if warnings else ""))
        self._plot_models(result); self._plot_fits(result)
        if result.baseline:
            rows = ["<h3>Independent baseline comparison</h3><ul>"]
            for name, value in result.baseline.items():
                if isinstance(value, dict) and "chi2" in value:
                    rows.append(f"<li>{html.escape(str(name))}: χ²={float(value['chi2']):.4g}</li>")
                elif isinstance(value, dict):
                    details = []
                    for method, method_result in value.items():
                        if isinstance(method_result, dict) and "chi2" in method_result:
                            details.append(f"{method} χ²={float(method_result['chi2']):.4g}")
                    rows.append(
                        f"<li>Sounding {html.escape(str(name))}: "
                        f"{html.escape(', '.join(details) if details else 'available')}</li>"
                    )
                else:
                    rows.append(f"<li>{html.escape(str(name))}: available</li>")
            rows.append("</ul><p>Joint models are shown in the Models tab.</p>")
            self._baseline_view.setHtml("".join(rows))
        else:
            self._baseline_view.setHtml("<p>Independent baseline was not requested.</p>")
        self._plot_baseline(result)
        files = "".join(f"<li><b>{html.escape(name)}</b>: {html.escape(path)}</li>"
                        for name, path in result.artifacts.items())
        self._files.setHtml(f"<ul>{files}</ul>")

    def _plot_models(self, result: JointInversionResult) -> None:
        self._models_figure.clear()
        if result.methods == ("ERT", "SRT"):
            mesh = result.meta.get("mesh")
            for index, method in enumerate(result.methods, start=1):
                axis = self._models_figure.add_subplot(1, 2, index)
                values = np.asarray(result.models.get(method, []), dtype=float).ravel()
                if method == "SRT" and values.size == 0 and result.meta.get("interface_coords") is not None:
                    interface_x, interface_z = result.meta["interface_coords"]
                    axis.plot(interface_x, interface_z, "k-", lw=2, label="SRT-derived interface")
                    axis.set_xlabel("Profile distance (m)"); axis.set_ylabel("Elevation / z (m)")
                    axis.legend(); axis.set_title("Sequential structural constraint")
                    continue
                try:
                    import pygimli as pg
                    pg.show(mesh, values, ax=axis, label="Resistivity (Ω m)" if method == "ERT" else "Velocity (m/s)")
                except Exception:
                    axis.plot(values); axis.set_xlabel("Cell index")
                axis.set_title(f"{method} joint model")
        elif result.methods == ("Gravity", "Magnetics"):
            edges = tuple(np.asarray(item, dtype=float) for item in result.meta.get("edges", ()))
            shape = tuple(int(value) for value in result.meta.get("model_shape", ()))
            for index, method in enumerate(result.methods, start=1):
                axis = self._models_figure.add_subplot(1, 2, index)
                values = np.asarray(result.models.get(method, []), dtype=float)
                if len(edges) == 3 and len(shape) == 3 and values.size == int(np.prod(shape)):
                    model3d = values.reshape(shape, order="F")
                    y_index = shape[1] // 2
                    image = axis.pcolormesh(
                        edges[0], edges[2], model3d[:, y_index, :].T,
                        shading="auto", cmap="RdBu_r" if method == "Gravity" else "viridis",
                    )
                    self._models_figure.colorbar(
                        image, ax=axis,
                        label="Density contrast (g/cc)" if method == "Gravity"
                        else "Susceptibility (SI)",
                    )
                    axis.set_xlabel("x (m)"); axis.set_ylabel("Elevation / z (m)")
                    axis.set_title(f"{method} · middle-y slice")
                else:
                    axis.plot(values.ravel()); axis.set_xlabel("Cell index")
                    axis.set_title(f"{method} joint model")
        else:
            axis = self._models_figure.add_subplot(111)
            model = np.asarray(result.models["resistivity"], dtype=float)
            thickness = np.asarray(result.meta.get("thicknesses", []), dtype=float)
            if model.ndim == 1:
                depths = np.r_[0.0, np.cumsum(thickness)]
                axis.step(model, depths, where="post"); axis.invert_yaxis()
                axis.set_xscale("log"); axis.set_xlabel("Resistivity (Ω m)"); axis.set_ylabel("Depth (m)")
            else:
                image = axis.imshow(model.T, aspect="auto", origin="upper", cmap="viridis")
                self._models_figure.colorbar(image, ax=axis, label="Resistivity (Ω m)")
                axis.set_xlabel("Matched sounding"); axis.set_ylabel("Layer")
            axis.set_title("Shared FDEM–TDEM resistivity model")
        self._models_canvas.draw_idle()

    def _plot_fits(self, result: JointInversionResult) -> None:
        self._fit_figure.clear()
        pair = result.methods
        for index, method in enumerate(pair, start=1):
            axis = self._fit_figure.add_subplot(1, 2, index)
            predicted = np.asarray(result.predicted.get(method, []), dtype=float).ravel()
            observed: np.ndarray
            data = self._data.get(method)
            if method == "ERT":
                observed = np.asarray(data["rhoa"], dtype=float).ravel()
                predicted = np.exp(predicted)
            elif method == "SRT":
                observed = np.asarray(data["t"], dtype=float).ravel()
            elif method == "FDEM":
                soundings, _coordinates = split_joint_soundings(data)
                indices = [int(item["fdem_index"]) for item in result.meta.get("pairing_manifest", [])]
                selected = [soundings[index] for index in indices] if indices else soundings[:1]
                observed = np.concatenate([
                    np.r_[np.asarray(sounding["real"]), np.asarray(sounding["imag"])]
                    for sounding in selected
                ])
            elif method == "TDEM":
                soundings, _coordinates = split_joint_soundings(data)
                indices = [int(item["tdem_index"]) for item in result.meta.get("pairing_manifest", [])]
                selected = [soundings[index] for index in indices] if indices else soundings[:1]
                observed = np.concatenate([
                    np.asarray(sounding["response"]) for sounding in selected
                ])
            elif method in {"Gravity", "Magnetics"}:
                meta_key = "observed_gravity" if method == "Gravity" else "observed_magnetics"
                observed = np.asarray(result.meta.get(meta_key, data["value"]), dtype=float).ravel()
            else:
                observed = np.asarray([], dtype=float)
            count = min(observed.size, predicted.size)
            axis.plot(observed[:count], "o", ms=3, label="Observed")
            axis.plot(predicted[:count], "-", lw=1.3, label="Predicted")
            axis.set_title(f"{method} data fit"); axis.set_xlabel("Datum"); axis.legend(); axis.grid(alpha=0.2)
        self._fit_canvas.draw_idle()

    def _plot_baseline(self, result: JointInversionResult) -> None:
        self._baseline_figure.clear()
        if not result.baseline:
            self._baseline_canvas.draw_idle()
            return
        comparisons: List[Tuple[str, np.ndarray, np.ndarray]] = []
        if result.methods == ("ERT", "SRT"):
            for method in result.methods:
                baseline = result.baseline.get(method)
                if not isinstance(baseline, dict) or baseline.get("model") is None:
                    continue
                joint_model = np.asarray(result.models.get(method, []), dtype=float).ravel()
                baseline_model = np.asarray(baseline["model"], dtype=float).ravel()
                if joint_model.size and baseline_model.size:
                    comparisons.append((method, joint_model, baseline_model))
        elif result.methods == ("Gravity", "Magnetics"):
            for method in result.methods:
                baseline = result.baseline.get(method)
                if not isinstance(baseline, dict) or baseline.get("model") is None:
                    continue
                joint_model = np.asarray(result.models.get(method, []), dtype=float).ravel()
                baseline_model = np.asarray(baseline["model"], dtype=float).ravel()
                if joint_model.size and baseline_model.size:
                    comparisons.append((method, joint_model, baseline_model))
        else:
            first = result.baseline.get("1", {})
            joint_models = np.asarray(result.models.get("resistivity", []), dtype=float)
            joint_model = joint_models if joint_models.ndim == 1 else joint_models[0]
            for method in result.methods:
                baseline = first.get(method) if isinstance(first, dict) else None
                if isinstance(baseline, dict) and baseline.get("resistivity") is not None:
                    comparisons.append((
                        method,
                        np.asarray(joint_model, dtype=float).ravel(),
                        np.asarray(baseline["resistivity"], dtype=float).ravel(),
                    ))
        if not comparisons:
            axis = self._baseline_figure.add_subplot(111)
            axis.text(0.5, 0.5, "Baseline metrics are available; model arrays were not returned.",
                      ha="center", va="center")
            axis.set_axis_off()
        for index, (method, joint_model, baseline_model) in enumerate(comparisons, start=1):
            axis = self._baseline_figure.add_subplot(1, len(comparisons), index)
            count = min(joint_model.size, baseline_model.size)
            cells = np.arange(count)
            axis.plot(cells, joint_model[:count], label="Joint", lw=1.5)
            axis.plot(cells, baseline_model[:count], label="Independent", lw=1.2)
            axis.plot(cells, joint_model[:count] - baseline_model[:count],
                      label="Difference", lw=1.0, alpha=0.8)
            axis.set_title(method); axis.set_xlabel("Layer / cell")
            axis.grid(alpha=0.2); axis.legend(fontsize=8)
        self._baseline_canvas.draw_idle()

    def _open_output(self) -> None:
        if self._result is None:
            return
        summary = self._result.artifacts.get("summary")
        if summary:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(summary).parent)))

    # -- AQUAH --------------------------------------------------------------
    def agent_describe(self) -> Dict[str, Any]:
        return {
            "module": self.module_key,
            "title": self.module_title,
            "state": {"step": _STEPS[self._current], "methods": self._selected_methods(),
                      "strategy": self._strategy.currentData(), "validated": self._validated,
                      "has_results": self._result is not None},
            "capabilities": [
                {"methods": item.methods, "strategies": dict(item.strategies),
                 "implemented": item.implemented, "backends": item.backends}
                for item in get_joint_capabilities()
            ],
            "actions": [
                {"name": "select_pair", "args": {"method_a": "str", "method_b": "str"}},
                {"name": "set_strategy", "args": {"strategy": "str"}},
                {"name": "use_example_data", "args": {}},
                {"name": "validate", "args": {}},
                {"name": "run_joint_inversion", "args": {}},
                {"name": "get_status", "args": {}},
            ],
        }

    def agent_apply(self, action: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if action == "get_status":
            return self.agent_describe()["state"]
        if action == "select_pair":
            self._method_a.setCurrentText(str(args.get("method_a", "")))
            self._method_b.setCurrentText(str(args.get("method_b", "")))
            return {"status": "ok", "methods": self._selected_methods()}
        if action == "set_strategy":
            index = self._strategy.findData(str(args.get("strategy", "")))
            if index < 0:
                return {"status": "failed", "error": "Strategy is unavailable for the selected pair."}
            self._strategy.setCurrentIndex(index); return {"status": "ok"}
        if action == "use_example_data":
            self._use_example(); return {"status": "ok" if all(m in self._data for m in self._selected_methods()) else "failed"}
        if action == "validate":
            self._validate(); return {"status": "ok" if self._validated else "failed"}
        if action == "run_joint_inversion":
            self._run(); return {"status": "started" if self._worker is not None else "failed"}
        return {"status": "failed", "error": f"Unknown action {action!r}."}


__all__ = ["JointInversionModule"]
