"""Hydro -> Geophysics guided wizard.

A six-step workflow (Data -> Profile -> Methods -> Parameters -> Run -> Results)
that mirrors the Streamlit guided experience: a clickable status strip, inline
validation, a two-click profile picker with a live preview, per-method parameter
panels with an Advanced section, a progress-tracked forward run, and an inline
results gallery. The profile is picked before any method parameters are set so the
forward problem is defined first. The heavy lifting is reused from
``PyHydroGeophysX.qt_apps.hydro_pipeline`` (fast profile extraction + the real
pygimli forward run with a config-export fallback).
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QStackedWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from PyHydroGeophysX.Hydro_modular import hydro_to_geophysics as hydro_pipeline
from PyHydroGeophysX.qt_apps import io_utils, theme
from PyHydroGeophysX.qt_apps.modules.base import BaseModule, LogFn
from PyHydroGeophysX.qt_apps.qt_utils import (
    BusyStateController,
    Debouncer,
    ReproduceBar,
    WizardNavigator,
    make_double_spinbox,
    select_directory,
)
from PyHydroGeophysX.qt_apps.widgets.array_viewer import ArrayViewer
from PyHydroGeophysX.qt_apps.widgets.image_view import ZoomableImageView
from PyHydroGeophysX.qt_apps.workers import TaskWorker, WorkflowWorker
from PyHydroGeophysX.workflows import (
    ArtifactRef,
    RunContext,
    WorkflowRunResult,
    WorkflowSpec,
    export_workflow_bundle,
)

_METHODS = ["ERT", "SRT", "TDEM", "FDEM", "Gravity"]
_DISPLAY_VARS = ["Water content", "Porosity", "Top", "Bottom"]
_STEPS = ["Data", "Profile", "Methods", "Parameters", "Run", "Results"]
_STEP_ICONS = [
    "fa5s.database",
    "fa5s.route",
    "fa5s.layer-group",
    "fa5s.sliders-h",
    "fa5s.play",
    "fa5s.chart-area",
]
_P = theme.PALETTE

_INPUT_FORMAT_FALLBACK = (
    "# Hydro input data format\n\n"
    "Place these four files in one folder:\n\n"
    "- `Watercontent.npy` - (n_time, n_layers, ny, nx) or (n_layers, ny, nx)\n"
    "- `Porosity.npy` - (n_layers, ny, nx)\n"
    "- `top.npy` - surface grid (ny, nx), NumPy\n"
    "- `bot.npy` - layer bottoms (n_layers, ny, nx)\n\n"
    "See hydro_input_format.md for details."
)


class HydroGeophysicsModule(BaseModule):
    module_key = "hydro_geophysics"
    module_title = "Hydro → Geophysics"

    def __init__(self, state: Any, log: LogFn, parent=None) -> None:
        super().__init__(state, log, parent)
        self._manual_dir: Optional[Path] = None
        self._wc = self._por = self._top = self._bot = None
        self._point1: Optional[List[float]] = None
        self._point2: Optional[List[float]] = None
        self._profile: Optional[Dict[str, Any]] = None
        self._worker: Optional[WorkflowWorker] = None
        self._run_busy: Optional[BusyStateController] = None
        self._workflow_recipe_path = ""
        self._preview_worker: Optional[TaskWorker] = None
        self._preview_debounced = Debouncer(self._update_preview, 120)
        self._param_panels: Dict[str, QWidget] = {}
        self._current = 0
        self._agent_data_source_confirmed = False
        self._agent_data_source = ""
        self._agent_parameters_confirmed = False
        self._agent_parameter_mode = ""

        root = QVBoxLayout(self)
        root.addWidget(self._build_strip())
        self._stack = QStackedWidget()
        self._stack.addWidget(self._build_data_step())        # 0 Data
        self._stack.addWidget(self._build_profile_step())     # 1 Profile
        self._stack.addWidget(self._build_methods_step())     # 2 Methods
        self._stack.addWidget(self._build_parameters_step())  # 3 Parameters
        self._stack.addWidget(self._build_run_step())         # 4 Run
        self._stack.addWidget(self._build_results_step())     # 5 Results
        root.addWidget(self._stack, stretch=1)
        self._reproduce = ReproduceBar()
        root.addWidget(self._reproduce)
        root.addLayout(self._build_nav())
        self._navigator = WizardNavigator(
            self._stack,
            previous_button=self._back_btn,
            next_button=self._next_btn,
            on_changed=self._on_wizard_changed,
            parent=self,
        )

        self._reload()
        self._go_to(0)

    # -- small widget helpers -----------------------------------------------
    @staticmethod
    def _dspin(value: float, lo: float, hi: float, step: float, decimals: int) -> QDoubleSpinBox:
        return make_double_spinbox(value, lo, hi, step, decimals)

    @staticmethod
    def _ispin(value: int, lo: int, hi: int) -> QSpinBox:
        s = QSpinBox()
        s.setRange(lo, hi)
        s.setValue(value)
        return s

    def _advanced_box(self, title: str = "Advanced") -> Tuple[QGroupBox, QFormLayout]:
        """A collapsible group: toggling the title check shows/hides the form."""
        box = QGroupBox(title)
        box.setCheckable(True)
        box.setChecked(False)
        inner = QWidget()
        form = QFormLayout(inner)
        form.setContentsMargins(8, 4, 8, 4)
        outer = QVBoxLayout(box)
        outer.setContentsMargins(8, 4, 8, 8)
        outer.addWidget(inner)
        inner.setVisible(False)
        box.toggled.connect(inner.setVisible)
        return box, form

    # -- status strip --------------------------------------------------------
    def _build_strip(self) -> QWidget:
        bar = QWidget()
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(4, 2, 4, 6)
        layout.setSpacing(6)
        self._chips: List[QPushButton] = []
        for i, name in enumerate(_STEPS):
            chip = QPushButton(f"  {i + 1}. {name}")
            chip.setIcon(theme.icon(_STEP_ICONS[i], color="#ffffff"))
            chip.clicked.connect(lambda _=False, idx=i: self._go_to(idx))
            self._chips.append(chip)
            layout.addWidget(chip)
            if i < len(_STEPS) - 1:
                arrow = QLabel("›")
                arrow.setStyleSheet(f"color:{_P['muted']}; font-size:16pt;")
                layout.addWidget(arrow)
        layout.addStretch(1)
        return bar

    def _chip_style(self, kind: str) -> str:
        if kind == "active":
            return f"QPushButton {{ background:{_P['primary']}; color:#fff; border:none; border-radius:14px; padding:6px 14px; font-weight:700; }}"
        if kind == "done":
            return f"QPushButton {{ background:{_P['green']}; color:#fff; border:none; border-radius:14px; padding:6px 14px; font-weight:600; }}"
        return f"QPushButton {{ background:{_P['card']}; color:{_P['muted']}; border:1px solid {_P['border']}; border-radius:14px; padding:6px 14px; }}"

    def _update_strip(self) -> None:
        status = self._step_status()
        # Chip order: Data, Profile, Methods, Parameters, Run, Results.
        done = [
            status["data"],
            status["profile"],
            status["methods"],
            status["methods"],
            status["all"],
            status["results"],
        ]
        for i, chip in enumerate(self._chips):
            if i == self._current:
                chip.setStyleSheet(self._chip_style("active"))
                chip.setIcon(theme.icon(_STEP_ICONS[i], color="#ffffff"))
            elif done[i]:
                chip.setStyleSheet(self._chip_style("done"))
                chip.setIcon(theme.icon("fa5s.check", color="#ffffff"))
            else:
                chip.setStyleSheet(self._chip_style("inactive"))
                chip.setIcon(theme.icon(_STEP_ICONS[i], color=_P["muted"]))

    def _build_nav(self) -> QHBoxLayout:
        nav = QHBoxLayout()
        self._back_btn = QPushButton("Back")
        self._back_btn.setIcon(theme.icon("fa5s.arrow-left"))
        self._next_btn = QPushButton("Next")
        self._next_btn.setIcon(theme.icon("fa5s.arrow-right"))
        nav.addWidget(self._back_btn)
        nav.addStretch(1)
        nav.addWidget(self._next_btn)
        return nav

    def _go_to(self, step: int) -> None:
        self._navigator.go_to(step)

    def _on_wizard_changed(self, step: int) -> None:
        self._current = step
        if self._current == 1:  # Profile
            self._update_display()
        if self._current == 3:  # Parameters
            self._refresh_parameter_panels()
        if self._current == 4:  # Run
            self._refresh_run_checklist()
        self._update_strip()

    def _step_status(self) -> Dict[str, bool]:
        data_ok = self._top is not None or self._wc is not None
        methods_ok = len(self._collect_methods()) > 0
        profile_ok = bool(self._point1 and self._point2)
        ran = self.state.module_results.get(self.module_key, {}).get("status") == "ok"
        return {"data": data_ok, "methods": methods_ok, "profile": profile_ok,
                "all": data_ok and methods_ok and profile_ok, "results": ran}

    # -- Step 1: Data --------------------------------------------------------
    def _build_data_step(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(QLabel("<h3>Step 1 · Load hydrologic model outputs</h3>"))
        layout.addWidget(QLabel(
            "Choose a folder containing <code>Watercontent.npy</code>, "
            "<code>Porosity.npy</code>, <code>top.npy</code>, <code>bot.npy</code>."))
        row = QHBoxLayout()
        ex_btn = QPushButton("Use example / context data")
        ex_btn.setIcon(theme.icon("fa5s.box-open"))
        ex_btn.clicked.connect(self._use_context_data)
        sel_btn = QPushButton("Select folder…")
        sel_btn.setIcon(theme.icon("fa5s.folder-open"))
        sel_btn.clicked.connect(self._select_folder)
        reload_btn = QPushButton("Reload")
        reload_btn.setIcon(theme.icon("fa5s.sync"))
        reload_btn.clicked.connect(self._reload)
        help_btn = QPushButton("Data format")
        help_btn.setIcon(theme.icon("fa5s.file-alt"))
        help_btn.clicked.connect(self._show_format_help)
        row.addWidget(ex_btn)
        row.addWidget(sel_btn)
        row.addWidget(reload_btn)
        row.addWidget(help_btn)
        row.addStretch(1)
        layout.addLayout(row)

        self._checklist = QGroupBox("Required files")
        self._checklist_layout = QVBoxLayout(self._checklist)
        layout.addWidget(self._checklist)
        self._data_status = QLabel("No folder selected.")
        self._data_status.setWordWrap(True)
        layout.addWidget(self._data_status)
        layout.addStretch(1)
        return page

    def _refresh_checklist(self) -> None:
        while self._checklist_layout.count():
            item = self._checklist_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        data_dir = self._data_dir()
        if data_dir is None:
            self._checklist_layout.addWidget(QLabel("Select a data folder to validate."))
            return
        files = hydro_pipeline.find_hydro_files(Path(data_dir))
        for key, name in hydro_pipeline.HYDRO_FILES.items():
            present = files.get(key) is not None
            icon = "fa5s.check-circle" if present else "fa5s.times-circle"
            color = _P["green"] if present else _P["red"]
            line = QHBoxLayout()
            badge = QLabel()
            badge.setPixmap(theme.icon(icon, color=color).pixmap(16, 16))
            text = QLabel(name)
            text.setStyleSheet(f"color:{color if not present else _P['text']};")
            line.addWidget(badge)
            line.addWidget(text)
            line.addStretch(1)
            wrap = QWidget(); wrap.setLayout(line)
            self._checklist_layout.addWidget(wrap)

    # -- Step 2: Profile -----------------------------------------------------
    def _build_profile_step(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.addWidget(QLabel("<h3>Step 2 · Pick a profile on the map</h3>"))
        outer.addWidget(QLabel(
            "Click two points to define the 2D cross-section. The profile is "
            "extracted first; method parameters come afterwards."))

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Variable"))
        self._variable = QComboBox(); self._variable.addItems(_DISPLAY_VARS)
        self._variable.currentTextChanged.connect(self._update_display)
        controls.addWidget(self._variable)
        controls.addWidget(QLabel("Snapshot"))
        self._snapshot = QSpinBox(); self._snapshot.setRange(0, 0); self._snapshot.valueChanged.connect(self._update_display)
        controls.addWidget(self._snapshot)
        controls.addWidget(QLabel("Layer"))
        self._layer = QSpinBox(); self._layer.setRange(0, 0); self._layer.valueChanged.connect(self._update_display)
        controls.addWidget(self._layer)
        controls.addWidget(QLabel("Samples"))
        self._num_samples = QSpinBox(); self._num_samples.setRange(50, 1000); self._num_samples.setValue(220)
        self._num_samples.valueChanged.connect(lambda *_: self._preview_debounced.trigger())
        controls.addWidget(self._num_samples)
        self._profile_mode = QCheckBox("Pick mode (two clicks)")
        self._profile_mode.setChecked(True)
        controls.addWidget(self._profile_mode)
        controls.addStretch(1)
        outer.addLayout(controls)

        split = QHBoxLayout()
        self._map = ArrayViewer()
        self._map.profileSelected.connect(self._on_profile_selected)
        self._profile_mode.toggled.connect(self._map.set_profile_mode)
        self._map.set_profile_mode(True)
        split.addWidget(self._map, stretch=3)

        side = QVBoxLayout()
        manual = QGroupBox("Manual coordinates [col, row]")
        mform = QFormLayout(manual)
        self._p1x = QSpinBox(); self._p1x.setRange(0, 100000)
        self._p1y = QSpinBox(); self._p1y.setRange(0, 100000)
        self._p2x = QSpinBox(); self._p2x.setRange(0, 100000)
        self._p2y = QSpinBox(); self._p2y.setRange(0, 100000)
        mform.addRow("P1 col", self._p1x); mform.addRow("P1 row", self._p1y)
        mform.addRow("P2 col", self._p2x); mform.addRow("P2 row", self._p2y)
        apply_btn = QPushButton("Apply coordinates")
        apply_btn.clicked.connect(self._apply_manual_points)
        mform.addRow(apply_btn)
        clear_btn = QPushButton("Clear profile")
        clear_btn.setIcon(theme.icon("fa5s.eraser"))
        clear_btn.clicked.connect(self._clear_profile)
        mform.addRow(clear_btn)
        side.addWidget(manual)
        self._profile_label = QLabel("No profile selected.")
        self._profile_label.setWordWrap(True)
        side.addWidget(self._profile_label)
        side.addWidget(QLabel("<b>Live profile preview</b>"))
        self._preview = ArrayViewer()
        self._preview.set_colormap("viridis")
        side.addWidget(self._preview, stretch=1)
        wrap = QWidget(); wrap.setLayout(side); wrap.setMaximumWidth(360)
        split.addWidget(wrap)
        outer.addLayout(split, stretch=1)
        return page

    # -- Step 3: Methods -----------------------------------------------------
    def _build_methods_step(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.addWidget(QLabel("<h3>Step 3 · Choose geophysical methods</h3>"))
        outer.addWidget(QLabel(
            "Pick the forward methods to simulate along the profile. "
            "Set their parameters in the next step."))

        presets = QHBoxLayout()
        for label, sel in (
            ("Select all", list(_METHODS)),
            ("ERT + SRT", ["ERT", "SRT"]),
            ("EM + Gravity", ["TDEM", "FDEM", "Gravity"]),
            ("Clear", []),
        ):
            btn = QPushButton(label)
            btn.clicked.connect(lambda _=False, s=sel: self._apply_preset(s))
            presets.addWidget(btn)
        presets.addStretch(1)
        outer.addLayout(presets)

        methods_box = QGroupBox("Methods")
        grid = QGridLayout(methods_box)
        self._method_boxes: Dict[str, QCheckBox] = {}
        for i, name in enumerate(_METHODS):
            box = QCheckBox(name)
            if name in ("ERT", "SRT"):
                box.setChecked(True)
            box.toggled.connect(self._update_strip)
            self._method_boxes[name] = box
            grid.addWidget(box, i // 3, i % 3)
        outer.addWidget(methods_box)
        outer.addStretch(1)
        return page

    def _apply_preset(self, selection: List[str]) -> None:
        for name, box in self._method_boxes.items():
            box.setChecked(name in selection)
        self._update_strip()

    # -- Step 4: Parameters --------------------------------------------------
    def _build_parameters_step(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.addWidget(QLabel("<h3>Step 4 · Set parameters</h3>"))
        outer.addWidget(QLabel(
            "Only the panels for the methods you selected are shown. "
            "Open <b>Advanced</b> for the less-common settings."))

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        ilayout = QVBoxLayout(inner)
        ilayout.addWidget(self._build_global_group())
        self._param_panels = {
            "ERT": self._build_ert_panel(),
            "SRT": self._build_srt_panel(),
            "TDEM": self._build_tdem_panel(),
            "FDEM": self._build_fdem_panel(),
            "Gravity": self._build_gravity_panel(),
        }
        for panel in self._param_panels.values():
            ilayout.addWidget(panel)
        ilayout.addStretch(1)
        scroll.setWidget(inner)
        outer.addWidget(scroll, stretch=1)
        return page

    def _refresh_parameter_panels(self) -> None:
        selected = set(self._collect_methods())
        for name, panel in self._param_panels.items():
            panel.setVisible(name in selected)

    def _build_global_group(self) -> QGroupBox:
        box = QGroupBox("Shared settings")
        form = QFormLayout(box)
        self._seed = self._ispin(7, 0, 10_000_000)
        form.addRow("Random seed", self._seed)
        adv, aform = self._advanced_box()
        self._mesh_quality = self._dspin(32.0, 1.0, 40.0, 1.0, 0)
        self._mesh_area = self._dspin(1.0, 0.01, 100.0, 0.1, 2)
        self._em_stations = self._ispin(24, 4, 200)
        aform.addRow("Mesh quality", self._mesh_quality)
        aform.addRow("Mesh max area", self._mesh_area)
        aform.addRow("EM stations", self._em_stations)
        form.addRow(adv)
        return box

    def _build_ert_panel(self) -> QGroupBox:
        panel = QGroupBox("ERT parameters")
        v = QVBoxLayout(panel)
        v.addWidget(self._build_petro_group())
        v.addWidget(self._build_ert_group())
        adv, form = self._advanced_box()
        self._ert_noise = self._dspin(0.03, 0.0, 1.0, 0.005, 3)
        self._ert_rel = self._dspin(0.03, 0.0, 1.0, 0.005, 3)
        self._ert_abs = self._dspin(0.0, 0.0, 100.0, 0.1, 3)
        form.addRow("Noise level", self._ert_noise)
        form.addRow("Relative error", self._ert_rel)
        form.addRow("Absolute error", self._ert_abs)
        v.addWidget(adv)
        return panel

    def _build_ert_group(self) -> QGroupBox:
        box = QGroupBox("ERT acquisition")
        form = QFormLayout(box)
        self._ert_count = QSpinBox(); self._ert_count.setRange(4, 512); self._ert_count.setValue(72)
        self._ert_spacing = QDoubleSpinBox(); self._ert_spacing.setRange(0.1, 50.0); self._ert_spacing.setValue(1.0)
        self._ert_start = QDoubleSpinBox(); self._ert_start.setRange(0.0, 1000.0); self._ert_start.setValue(0.0)
        self._ert_scheme = QComboBox(); self._ert_scheme.addItems(["wa", "dd", "slm", "dp"])
        form.addRow("Electrodes", self._ert_count)
        form.addRow("Spacing (m)", self._ert_spacing)
        form.addRow("Start x (m)", self._ert_start)
        form.addRow("Array type", self._ert_scheme)
        return box

    def _build_petro_group(self) -> QGroupBox:
        box = QGroupBox("Resistivity petrophysics (top / mid / bot)")
        grid = QGridLayout(box)

        self._rho = [self._dspin(100.0, 1.0, 1e5, 10.0, 1), self._dspin(500.0, 1.0, 1e5, 10.0, 1), self._dspin(2400.0, 1.0, 1e5, 10.0, 1)]
        self._archie = [self._dspin(2.2, 1.0, 3.0, 0.1, 2), self._dspin(1.8, 1.0, 3.0, 0.1, 2), self._dspin(2.5, 1.0, 3.0, 0.1, 2)]
        self._sigma_s = [self._dspin(0.002, 0.0, 1.0, 0.001, 4), self._dspin(0.0, 0.0, 1.0, 0.001, 4), self._dspin(0.0, 0.0, 1.0, 0.001, 4)]
        for col, head in enumerate(["", "top", "mid", "bot"]):
            grid.addWidget(QLabel(f"<b>{head}</b>"), 0, col)
        for r, (label, spins) in enumerate(
            [("rho_sat", self._rho), ("archie_n", self._archie), ("sigma_s", self._sigma_s)], start=1
        ):
            grid.addWidget(QLabel(label), r, 0)
            for col, spin in enumerate(spins, start=1):
                grid.addWidget(spin, r, col)
        return box

    def _build_srt_panel(self) -> QGroupBox:
        panel = QGroupBox("SRT parameters")
        v = QVBoxLayout(panel)
        v.addWidget(self._build_srt_group())
        v.addWidget(self._build_velocity_group())
        adv, form = self._advanced_box()
        self._srt_noise = self._dspin(0.01, 0.0, 1.0, 0.005, 3)
        self._srt_noise_abs = self._dspin(1e-5, 0.0, 1.0, 1e-5, 6)
        form.addRow("Noise level", self._srt_noise)
        form.addRow("Absolute noise (s)", self._srt_noise_abs)
        v.addWidget(adv)
        return panel

    def _build_srt_group(self) -> QGroupBox:
        box = QGroupBox("SRT acquisition")
        form = QFormLayout(box)
        self._srt_count = QSpinBox(); self._srt_count.setRange(4, 512); self._srt_count.setValue(72)
        self._srt_spacing = QDoubleSpinBox(); self._srt_spacing.setRange(0.1, 50.0); self._srt_spacing.setValue(1.0)
        self._srt_shot = QSpinBox(); self._srt_shot.setRange(1, 100); self._srt_shot.setValue(5)
        form.addRow("Sensors", self._srt_count)
        form.addRow("Spacing (m)", self._srt_spacing)
        form.addRow("Shot spacing (sensors)", self._srt_shot)
        return box

    def _build_velocity_group(self) -> QGroupBox:
        box = QGroupBox("Velocity model (top / mid / bot)")
        grid = QGridLayout(box)
        self._vel_bulk = [self._dspin(30.0, 1.0, 200.0, 1.0, 1), self._dspin(50.0, 1.0, 200.0, 1.0, 1), self._dspin(55.0, 1.0, 200.0, 1.0, 1)]
        self._vel_shear = [self._dspin(20.0, 1.0, 200.0, 1.0, 1), self._dspin(35.0, 1.0, 200.0, 1.0, 1), self._dspin(50.0, 1.0, 200.0, 1.0, 1)]
        self._vel_rho = [self._dspin(2650.0, 1000.0, 4000.0, 10.0, 0), self._dspin(2670.0, 1000.0, 4000.0, 10.0, 0), self._dspin(2680.0, 1000.0, 4000.0, 10.0, 0)]
        self._vel_top_depth = self._dspin(1.0, 0.1, 100.0, 0.5, 2)
        self._vel_mid_aspect = self._dspin(0.05, 0.001, 1.0, 0.01, 3)
        self._vel_bot_aspect = self._dspin(0.03, 0.001, 1.0, 0.01, 3)
        for col, head in enumerate(["", "top", "mid", "bot"]):
            grid.addWidget(QLabel(f"<b>{head}</b>"), 0, col)
        for r, (label, spins) in enumerate(
            [("bulk_modulus (GPa)", self._vel_bulk), ("shear_modulus (GPa)", self._vel_shear), ("mineral_density", self._vel_rho)], start=1
        ):
            grid.addWidget(QLabel(label), r, 0)
            for col, spin in enumerate(spins, start=1):
                grid.addWidget(spin, r, col)
        grid.addWidget(QLabel("depth(top) / aspect(mid,bot)"), 4, 0)
        grid.addWidget(self._vel_top_depth, 4, 1)
        grid.addWidget(self._vel_mid_aspect, 4, 2)
        grid.addWidget(self._vel_bot_aspect, 4, 3)
        return box

    def _build_tdem_panel(self) -> QGroupBox:
        panel = QGroupBox("TDEM parameters")
        form = QFormLayout(panel)
        self._tdem_tmin = self._dspin(1e-5, 1e-7, 1.0, 1e-5, 7)
        self._tdem_tmax = self._dspin(1e-2, 1e-5, 10.0, 1e-3, 5)
        self._tdem_ngates = self._ispin(28, 4, 200)
        self._tdem_sigma_w = self._dspin(0.05, 1e-4, 10.0, 0.01, 4)
        self._tdem_m = self._dspin(1.5, 1.0, 3.0, 0.1, 2)
        self._tdem_n = self._dspin(2.0, 1.0, 3.0, 0.1, 2)
        self._tdem_sigma_s = self._dspin(0.0, 0.0, 1.0, 0.001, 4)
        self._tdem_noise = self._dspin(0.03, 0.0, 1.0, 0.005, 3)
        form.addRow("Min time (s)", self._tdem_tmin)
        form.addRow("Max time (s)", self._tdem_tmax)
        form.addRow("Time gates", self._tdem_ngates)
        form.addRow("sigma_w (S/m)", self._tdem_sigma_w)
        form.addRow("m", self._tdem_m)
        form.addRow("n", self._tdem_n)
        form.addRow("sigma_s (S/m)", self._tdem_sigma_s)
        form.addRow("Noise level", self._tdem_noise)
        adv, aform = self._advanced_box()
        self._tdem_radius = self._dspin(10.0, 0.5, 200.0, 0.5, 1)
        aform.addRow("Source radius (m)", self._tdem_radius)
        form.addRow(adv)
        return panel

    def _build_fdem_panel(self) -> QGroupBox:
        panel = QGroupBox("FDEM parameters")
        form = QFormLayout(panel)
        self._fdem_fmin = self._dspin(10.0, 0.1, 1e6, 10.0, 2)
        self._fdem_fmax = self._dspin(1e4, 1.0, 1e7, 100.0, 1)
        self._fdem_nfreq = self._ispin(18, 4, 200)
        self._fdem_sigma_w = self._dspin(0.05, 1e-4, 10.0, 0.01, 4)
        self._fdem_m = self._dspin(1.5, 1.0, 3.0, 0.1, 2)
        self._fdem_n = self._dspin(2.0, 1.0, 3.0, 0.1, 2)
        self._fdem_sigma_s = self._dspin(0.0, 0.0, 1.0, 0.001, 4)
        self._fdem_noise = self._dspin(0.03, 0.0, 1.0, 0.005, 3)
        form.addRow("Min freq (Hz)", self._fdem_fmin)
        form.addRow("Max freq (Hz)", self._fdem_fmax)
        form.addRow("Frequencies", self._fdem_nfreq)
        form.addRow("sigma_w (S/m)", self._fdem_sigma_w)
        form.addRow("m", self._fdem_m)
        form.addRow("n", self._fdem_n)
        form.addRow("sigma_s (S/m)", self._fdem_sigma_s)
        form.addRow("Noise level", self._fdem_noise)
        adv, aform = self._advanced_box()
        self._fdem_src_x = self._dspin(0.0, -100.0, 1000.0, 1.0, 1)
        self._fdem_rec_x = self._dspin(12.0, -100.0, 1000.0, 1.0, 1)
        self._fdem_radius = self._dspin(10.0, 0.5, 200.0, 0.5, 1)
        self._fdem_orient = QComboBox(); self._fdem_orient.addItems(["z", "x", "y"])
        self._fdem_comp = QComboBox(); self._fdem_comp.addItems(["secondary", "total"])
        self._fdem_wave = QComboBox(); self._fdem_wave.addItems(["dipole", "loop"])
        aform.addRow("Source x (m)", self._fdem_src_x)
        aform.addRow("Receiver x (m)", self._fdem_rec_x)
        aform.addRow("Source radius (m)", self._fdem_radius)
        aform.addRow("Receiver orientation", self._fdem_orient)
        aform.addRow("Receiver component", self._fdem_comp)
        aform.addRow("Waveform", self._fdem_wave)
        form.addRow(adv)
        return panel

    def _build_gravity_panel(self) -> QGroupBox:
        panel = QGroupBox("Gravity parameters")
        form = QFormLayout(panel)
        self._grav_rho_matrix = self._dspin(2650.0, 1000.0, 4000.0, 10.0, 0)
        self._grav_rho_water = self._dspin(1000.0, 500.0, 1500.0, 10.0, 0)
        self._grav_rho_air = self._dspin(1.225, 0.0, 100.0, 0.1, 3)
        self._grav_sensor_h = self._dspin(1.0, 0.0, 100.0, 0.5, 2)
        self._grav_noise = self._dspin(0.02, 0.0, 1.0, 0.005, 3)
        form.addRow("rho_matrix", self._grav_rho_matrix)
        form.addRow("rho_water", self._grav_rho_water)
        form.addRow("rho_air", self._grav_rho_air)
        form.addRow("Sensor height (m)", self._grav_sensor_h)
        form.addRow("Noise level", self._grav_noise)
        adv, aform = self._advanced_box()
        self._grav_mesh_nx = self._ispin(80, 10, 400)
        self._grav_mesh_nz = self._ispin(60, 10, 400)
        self._grav_width_y = self._dspin(12.0, 1.0, 200.0, 1.0, 1)
        aform.addRow("Mesh nx", self._grav_mesh_nx)
        aform.addRow("Mesh nz", self._grav_mesh_nz)
        aform.addRow("Model width y (m)", self._grav_width_y)
        form.addRow(adv)
        return panel

    # -- Step 5: Run ---------------------------------------------------------
    def _build_run_step(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(QLabel("<h3>Step 5 · Review and run forward modeling</h3>"))
        self._run_checklist = QGroupBox("Readiness")
        self._run_checklist_layout = QVBoxLayout(self._run_checklist)
        layout.addWidget(self._run_checklist)

        self._run_btn = QPushButton("Run forward modeling")
        self._run_btn.setProperty("primary", True)
        self._run_btn.setIcon(theme.icon("fa5s.play", color="#ffffff"))
        self._run_btn.clicked.connect(self._run_forward)
        layout.addWidget(self._run_btn)
        cfg_btn = QPushButton("Export survey config JSON (no backend needed)")
        cfg_btn.setIcon(theme.icon("fa5s.file-export"))
        cfg_btn.clicked.connect(self._export_config)
        layout.addWidget(cfg_btn)

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        layout.addWidget(self._progress)
        self._run_status = QLabel("")
        self._run_status.setWordWrap(True)
        layout.addWidget(self._run_status)
        layout.addStretch(1)
        return page

    def _refresh_run_checklist(self) -> None:
        while self._run_checklist_layout.count():
            item = self._run_checklist_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        status = self._step_status()
        rows = [
            ("Data loaded", status["data"], str(self._data_dir() or "—")),
            ("Profile picked", status["profile"],
             f"{self._point1} → {self._point2}" if status["profile"] else "pick two points in Step 2"),
            ("Methods selected", status["methods"], ", ".join(self._collect_methods()) or "none"),
        ]
        for label, ok, detail in rows:
            icon = "fa5s.check-circle" if ok else "fa5s.times-circle"
            color = _P["green"] if ok else _P["red"]
            line = QHBoxLayout()
            badge = QLabel(); badge.setPixmap(theme.icon(icon, color=color).pixmap(16, 16))
            text = QLabel(f"<b>{label}</b> — {detail}")
            text.setWordWrap(True)
            line.addWidget(badge); line.addWidget(text, stretch=1)
            wrap = QWidget(); wrap.setLayout(line)
            self._run_checklist_layout.addWidget(wrap)
        self._run_btn.setEnabled(status["all"])

    # -- Step 6: Results -----------------------------------------------------
    def _build_results_step(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        header = QHBoxLayout()
        header.addWidget(QLabel("<h3>Step 6 · Results</h3>"))
        header.addStretch(1)
        open_btn = QPushButton("Open output folder")
        open_btn.setIcon(theme.icon("fa5s.folder-open"))
        open_btn.clicked.connect(self._open_output_folder)
        header.addWidget(open_btn)
        layout.addLayout(header)

        self._result_summary = QLabel("No results yet. Complete Steps 1–5 and run forward modeling.")
        self._result_summary.setWordWrap(True)
        layout.addWidget(self._result_summary)
        picker = QHBoxLayout()
        picker.addWidget(QLabel("Method:"))
        self._result_method = QComboBox()
        self._result_method.currentIndexChanged.connect(self._show_selected_result)
        picker.addWidget(self._result_method)
        picker.addStretch(1)
        layout.addLayout(picker)

        panels = QHBoxLayout()
        model_box = QGroupBox("Model")
        model_layout = QVBoxLayout(model_box)
        self._result_model_caption = QLabel("Select a completed method to view its model.")
        self._result_model_caption.setWordWrap(True)
        self._result_model_view = ZoomableImageView()
        self._result_model_view.setMinimumHeight(440)
        model_layout.addWidget(self._result_model_caption)
        model_layout.addWidget(self._result_model_view, stretch=1)
        measurement_box = QGroupBox("Measurements")
        measurement_layout = QVBoxLayout(measurement_box)
        self._result_measurement_caption = QLabel("Select a completed method to view synthetic measurements.")
        self._result_measurement_caption.setWordWrap(True)
        self._result_measurement_view = ZoomableImageView()
        self._result_measurement_view.setMinimumHeight(440)
        measurement_layout.addWidget(self._result_measurement_caption)
        measurement_layout.addWidget(self._result_measurement_view, stretch=1)
        panels.addWidget(model_box, stretch=1)
        panels.addWidget(measurement_box, stretch=1)
        layout.addLayout(panels, stretch=1)
        self._result_files = QLabel("")
        self._result_files.setWordWrap(True)
        layout.addWidget(self._result_files)
        self._result_paths: Dict[str, Dict[str, str]] = {}
        return page

    def _populate_results(self, result: Dict[str, Any]) -> None:
        self._result_summary.setText(
            f"<b>Status:</b> {result.get('status')} &nbsp; "
            f"<b>Methods:</b> {', '.join(result.get('methods', []))} &nbsp; "
            f"<b>Mesh cells:</b> {result.get('mesh_cells', '—')}")
        paths = result.get("display_paths", {})
        self._result_paths = {
            str(method): {str(panel): str(path) for panel, path in panels.items()}
            for method, panels in paths.items() if isinstance(panels, dict)
        }
        # Backward-compatible fallback for old results written before separate
        # model/measurement figures were available.
        legacy = [str(path) for path in result.get("figure_paths", []) if Path(path).exists()]
        if not self._result_paths and legacy:
            self._result_paths = {
                method: {"model": legacy[0], "measurement": legacy[0]}
                for method in result.get("methods", [])
            }
        self._result_method.blockSignals(True)
        self._result_method.clear()
        for method in result.get("methods", []):
            if method in self._result_paths:
                label = "Seismic (SRT)" if method == "SRT" else str(method)
                self._result_method.addItem(label, method)
        self._result_method.blockSignals(False)
        if self._result_method.count():
            self._result_method.setCurrentIndex(0)
            self._show_selected_result()
        else:
            self._show_selected_result()
        files = [Path(p).name for p in result.get("data_paths", [])]
        cfg = result.get("config_path")
        bits = [f"<b>Synthetic data:</b> {', '.join(files)}" if files else ""]
        if cfg:
            bits.append(f"Survey config: <code>{cfg}</code>")
        self._result_files.setText("<br>".join(bit for bit in bits if bit))

    def _show_selected_result(self) -> None:
        """Load the selected method into the side-by-side model/data panels."""
        method = self._result_method.currentData()
        paths = self._result_paths.get(str(method), {}) if method else {}
        for panel, view, caption in (
            ("model", self._result_model_view, self._result_model_caption),
            ("measurement", self._result_measurement_view, self._result_measurement_caption),
        ):
            path = paths.get(panel)
            if path and Path(path).exists() and view.set_image_file(path):
                caption.setText(f"{str(method)} {panel}: <code>{Path(path).name}</code>")
            else:
                view.clear()
                caption.setText(f"No {panel} figure is available for this method.")

    def _open_output_folder(self) -> None:
        out = self.state.module_results.get(self.module_key, {}).get("output_dir")
        out = out or str(self.state.hydro_output_dir or self.state.output_dir or "")
        if out and Path(out).exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(out))
        else:
            self.log("Output folder not available yet.", "warn")

    # -- data loading --------------------------------------------------------
    def _data_dir(self) -> Optional[Path]:
        if self._manual_dir is not None:
            return self._manual_dir
        if self.state.hydro_data_dir is not None:
            return Path(self.state.hydro_data_dir)
        return None

    def _use_context_data(self) -> None:
        self._agent_data_source_confirmed = False
        self._agent_data_source = ""
        self._manual_dir = None
        if self.state.hydro_data_dir is None:
            self._manual_dir = self._example_data_dir()
        self._reload()
        if self._top is not None or self._wc is not None:
            self._agent_data_source_confirmed = True
            self._agent_data_source = (
                "project_context" if self.state.hydro_data_dir is not None else "example"
            )

    def _example_data_dir(self) -> Optional[Path]:
        """Locate the bundled synthetic hydrology grids for standalone launches.

        Points at the synthetic ``timelapse_infiltration`` dataset produced by
        ``examples/generate_synthetic_examples.py``. The real Treeline demo in
        ``examples/data`` is preserved and can still be opened via "Select
        folder...".
        """
        rel = ("examples", "data", "synthetic", "timelapse_infiltration")
        candidates: List[Path] = []
        if self.state.project_root:
            candidates.append(Path(self.state.project_root).joinpath(*rel))
        here = Path(__file__).resolve()
        candidates.extend(p.joinpath(*rel) for p in here.parents)
        for cand in candidates:
            files = hydro_pipeline.find_hydro_files(cand)
            if files.get("top") is not None or files.get("water_content") is not None:
                return cand
        return None

    def _show_format_help(self) -> None:
        """Pop a dialog rendering the input-data-format documentation."""
        doc_path = Path(__file__).with_name("hydro_input_format.md")
        try:
            text = doc_path.read_text(encoding="utf-8")
        except Exception:  # noqa: BLE001
            text = _INPUT_FORMAT_FALLBACK
        dlg = QDialog(self)
        dlg.setWindowTitle("Hydro input data format")
        dlg.resize(760, 660)
        lay = QVBoxLayout(dlg)
        browser = QTextBrowser()
        browser.setOpenExternalLinks(True)
        try:
            browser.setMarkdown(text)
        except Exception:  # noqa: BLE001 - very old Qt without setMarkdown
            browser.setPlainText(text)
        lay.addWidget(browser)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        lay.addWidget(close_btn)
        dlg.exec()

    def _select_folder(self) -> None:
        start = str(self._data_dir() or Path.cwd())
        path = select_directory(self, "Select hydro data folder", start)
        if path:
            self._agent_data_source_confirmed = False
            self._agent_data_source = ""
            self._manual_dir = path
            self._reload()
            if self._top is not None or self._wc is not None:
                self._agent_data_source_confirmed = True
                self._agent_data_source = "user_data"

    def _reload(self) -> None:
        data_dir = self._data_dir()
        self._wc = self._por = self._top = self._bot = None
        if hasattr(self, "_checklist_layout"):
            self._refresh_checklist()
        if data_dir is None or not Path(data_dir).exists():
            if hasattr(self, "_data_status"):
                self._data_status.setText("No hydro data folder set. Use 'Select folder…'.")
            self._update_strip()
            return
        files = hydro_pipeline.find_hydro_files(Path(data_dir))
        # These stay memory-mapped for as long as the module holds them, which is
        # what keeps a multi-gigabyte water-content array off the heap. The cost
        # is that Windows will not let anything overwrite a mapped file, so a
        # regenerated folder cannot be reloaded until the old maps are dropped.
        # Releasing them here makes reloading the same folder work; the reference
        # count is what closes the map, so this has to happen before reassigning.
        self._release_hydro_arrays()
        self._wc = self._safe_load(files["water_content"], mmap=True)
        self._por = self._safe_load(files["porosity"], mmap=True)
        # top is a NumPy array (top.npy), same as the other three; load with
        # np.load, not np.loadtxt (text=True fails on a binary .npy and left
        # _top as None).
        self._top = self._safe_load(files["top"], mmap=True)
        self._bot = self._safe_load(files["bot"], mmap=True)
        if self._wc is not None and self._wc.ndim == 4:
            self._snapshot.setRange(0, self._wc.shape[0] - 1)
        max_layers = max((arr.shape[-3] if getattr(arr, "ndim", 0) >= 3 else 1)
                         for arr in (self._wc, self._por, self._bot) if arr is not None) if any(
            a is not None for a in (self._wc, self._por, self._bot)) else 1
        self._layer.setRange(0, max(0, max_layers - 1))
        missing = [hydro_pipeline.HYDRO_FILES[k] for k, v in files.items() if v is None]
        wc_shape = getattr(self._wc, "shape", None)
        if hasattr(self, "_data_status"):
            msg = [f"Folder: {data_dir}", f"Watercontent: {tuple(wc_shape)}" if wc_shape else "Watercontent: —"]
            if missing:
                msg.append(f"Missing: {', '.join(missing)}")
            self._data_status.setText("<br>".join(msg))
        self.log(f"Loaded hydro data from {data_dir}", "success" if not missing else "warn")
        self._update_display()
        self._update_strip()

    def _release_hydro_arrays(self) -> None:
        """Drop the memory-mapped hydro arrays so their file handles close.

        A numpy memmap closes when its last reference goes, so clearing the
        attributes is the release. gc.collect covers the case where a plot or a
        slice still holds one, which would otherwise keep the file locked.
        """
        import gc

        for name in ("_wc", "_por", "_top", "_bot"):
            if getattr(self, name, None) is not None:
                setattr(self, name, None)
        gc.collect()

    def _safe_load(self, path, mmap: bool = False, text: bool = False):
        if path is None:
            return None
        try:
            if text:
                return np.loadtxt(path)
            return np.load(path, mmap_mode="r" if mmap else None)
        except Exception as exc:  # noqa: BLE001
            self.log(f"Failed to load {Path(path).name}: {exc}", "error")
            return None

    def _update_display(self) -> None:
        if not hasattr(self, "_map"):
            return
        var = self._variable.currentText()
        arr2d = None
        layer = self._layer.value()
        try:
            if var == "Water content" and self._wc is not None:
                w = self._wc[self._snapshot.value()] if self._wc.ndim == 4 else self._wc
                arr2d = np.asarray(w[min(layer, w.shape[0] - 1)], dtype=float)
            elif var == "Porosity" and self._por is not None:
                arr2d = np.asarray(self._por[min(layer, self._por.shape[0] - 1)], dtype=float)
            elif var == "Top" and self._top is not None:
                arr2d = np.asarray(self._top, dtype=float)
            elif var == "Bottom" and self._bot is not None:
                arr2d = np.asarray(self._bot[min(layer, self._bot.shape[0] - 1)], dtype=float)
        except Exception as exc:  # noqa: BLE001
            self.log(f"Display update failed: {exc}", "error")
            return
        if arr2d is None:
            return
        self._map.set_array(arr2d)
        if self._point1 and self._point2:
            self._map.set_profile_points(self._point1, self._point2)

    # -- profile -------------------------------------------------------------
    def _on_profile_selected(self, p1: list, p2: list) -> None:
        self._point1 = [float(p1[0]), float(p1[1])]
        self._point2 = [float(p2[0]), float(p2[1])]
        self._sync_manual_spins()
        self._after_points_changed()

    def _apply_manual_points(self) -> None:
        self._point1 = [float(self._p1x.value()), float(self._p1y.value())]
        self._point2 = [float(self._p2x.value()), float(self._p2y.value())]
        self._map.set_profile_points(self._point1, self._point2)
        self._after_points_changed()

    def _sync_manual_spins(self) -> None:
        if self._point1 and self._point2:
            self._p1x.setValue(int(self._point1[0])); self._p1y.setValue(int(self._point1[1]))
            self._p2x.setValue(int(self._point2[0])); self._p2y.setValue(int(self._point2[1]))

    def _after_points_changed(self) -> None:
        self.state.selected_points = [self._point1, self._point2]
        self._profile_label.setText(
            f"Profile: ({self._point1[0]:.0f}, {self._point1[1]:.0f}) → "
            f"({self._point2[0]:.0f}, {self._point2[1]:.0f})")
        self._preview_debounced.trigger()
        self._update_strip()

    def _update_preview(self) -> None:
        if not (self._point1 and self._point2):
            return
        # Capture params on the UI thread; extract the profile off-thread.
        params = self._collect_params()
        worker = TaskWorker(
            hydro_pipeline.extract_profile, self.state.context, params, self._point1, self._point2)
        worker.succeeded.connect(self._on_preview_ready)
        worker.failed.connect(lambda msg: self.log(f"Profile preview failed: {msg}", "warn"))
        self._preview_worker = self.register_worker(worker)
        worker.start()

    def _on_preview_ready(self, profile: dict) -> None:
        self._profile = profile
        try:
            wc = np.asarray(profile["water_content_profile"], dtype=float)
            self._preview.set_array(wc)
        except Exception as exc:  # noqa: BLE001
            self.log(f"Profile preview failed: {exc}", "warn")

    def _clear_profile(self) -> None:
        self._point1 = self._point2 = self._profile = None
        self.state.selected_points = []
        self._map.clear_profile()
        self._profile_label.setText("No profile selected.")
        self._update_strip()

    # -- params / forward ----------------------------------------------------
    def _collect_params(self) -> Dict[str, Any]:
        out_dir = self.state.hydro_output_dir or self.state.output_dir
        return {
            "snapshot_index": self._snapshot.value(),
            "num_samples": self._num_samples.value(),
            "hydro_data_dir": str(self._data_dir()) if self._data_dir() else "",
            "output_dir": str(out_dir) if out_dir else "",
            "seed": self._seed.value(),
            "mesh": {
                "quality": self._mesh_quality.value(),
                "area": self._mesh_area.value(),
            },
            "em_stations": self._em_stations.value(),
            "ert": {
                "num_electrodes": self._ert_count.value(),
                "electrode_spacing": self._ert_spacing.value(),
                "electrode_start": self._ert_start.value(),
                "scheme_name": self._ert_scheme.currentText(),
                "noise_level": self._ert_noise.value(),
                "rel_error": self._ert_rel.value(),
                "abs_error": self._ert_abs.value(),
            },
            "srt": {
                "num_sensors": self._srt_count.value(),
                "sensor_spacing": self._srt_spacing.value(),
                "shot_distance": self._srt_shot.value(),
                "sensor_start": self._ert_start.value(),
                "noise_level": self._srt_noise.value(),
                "noise_abs": self._srt_noise_abs.value(),
            },
            "petro": {
                "rho_sat": [s.value() for s in self._rho],
                "archie_n": [s.value() for s in self._archie],
                "sigma_s": [s.value() for s in self._sigma_s],
            },
            "srt_vel": {
                "top": {
                    "bulk_modulus": self._vel_bulk[0].value(),
                    "shear_modulus": self._vel_shear[0].value(),
                    "mineral_density": self._vel_rho[0].value(),
                    "depth": self._vel_top_depth.value(),
                },
                "mid": {
                    "bulk_modulus": self._vel_bulk[1].value(),
                    "shear_modulus": self._vel_shear[1].value(),
                    "mineral_density": self._vel_rho[1].value(),
                    "aspect_ratio": self._vel_mid_aspect.value(),
                },
                "bot": {
                    "bulk_modulus": self._vel_bulk[2].value(),
                    "shear_modulus": self._vel_shear[2].value(),
                    "mineral_density": self._vel_rho[2].value(),
                    "aspect_ratio": self._vel_bot_aspect.value(),
                },
            },
            "tdem": {
                "t_min_s": self._tdem_tmin.value(),
                "t_max_s": self._tdem_tmax.value(),
                "n_gates": self._tdem_ngates.value(),
                "sigma_w": self._tdem_sigma_w.value(),
                "m": self._tdem_m.value(),
                "n": self._tdem_n.value(),
                "sigma_s": self._tdem_sigma_s.value(),
                "source_radius": self._tdem_radius.value(),
                "noise_level": self._tdem_noise.value(),
            },
            "fdem": {
                "f_min_hz": self._fdem_fmin.value(),
                "f_max_hz": self._fdem_fmax.value(),
                "n_freqs": self._fdem_nfreq.value(),
                "sigma_w": self._fdem_sigma_w.value(),
                "m": self._fdem_m.value(),
                "n": self._fdem_n.value(),
                "sigma_s": self._fdem_sigma_s.value(),
                "source_x": self._fdem_src_x.value(),
                "receiver_x": self._fdem_rec_x.value(),
                "source_radius": self._fdem_radius.value(),
                "receiver_orientation": self._fdem_orient.currentText(),
                "receiver_component": self._fdem_comp.currentText(),
                "waveform_type": self._fdem_wave.currentText(),
                "noise_level": self._fdem_noise.value(),
            },
            "gravity": {
                "rho_matrix": self._grav_rho_matrix.value(),
                "rho_water": self._grav_rho_water.value(),
                "rho_air": self._grav_rho_air.value(),
                "sensor_height": self._grav_sensor_h.value(),
                "noise_level": self._grav_noise.value(),
                "mesh_nx": self._grav_mesh_nx.value(),
                "mesh_nz": self._grav_mesh_nz.value(),
                "model_width_y": self._grav_width_y.value(),
            },
        }

    def _collect_methods(self) -> List[str]:
        if not hasattr(self, "_method_boxes"):
            return []
        return [name for name, box in self._method_boxes.items() if box.isChecked()]

    def _export_config(self) -> str:
        params = self._collect_params()
        methods = self._collect_methods()
        config = hydro_pipeline.build_survey_config(
            self.state.context, params, methods, self._point1 or [0, 0], self._point2 or [0, 0],
            profile=self._profile)
        active = self.state.active_run(self.module_key)
        out_dir = active.outputs_dir if active else self.state.ensure_results_store().scratch_dir(self.module_key)
        config_path = out_dir / "survey_config.json"
        io_utils.write_json(config_path, config)
        self.log(f"Exported survey config to {config_path}", "success")
        self.report_result({
            "selected_points": [self._point1, self._point2], "methods": methods,
            "config_path": str(config_path), "status": "config_exported"})
        return str(config_path)

    def _run_forward(self) -> None:
        status = self._step_status()
        if not status["all"]:
            self.log("Complete Data, Profile and Methods first.", "warn")
            return
        params = self._collect_params()
        methods = self._collect_methods()
        seed = int(params.pop("seed"))
        data_dir = self._data_dir()
        params.pop("hydro_data_dir", None)
        params.pop("output_dir", None)
        try:
            run = self.begin_persisted_run(
                "hydro_geophysics.forward",
                workflow_id="hydro_geophysics.forward",
            )
        except Exception as exc:  # noqa: BLE001
            self.log(f"Could not prepare Project run: {exc}", "error")
            return
        hydro_files: Dict[str, ArtifactRef] = {}
        if data_dir is not None:
            for filename in ("Watercontent.npy", "Porosity.npy", "top.npy", "bot.npy"):
                path = data_dir / filename
                if path.is_file():
                    stored = run.inputs_dir / filename
                    try:
                        shutil.copy2(path, stored)
                    except OSError as exc:
                        self.fail_persisted_run(str(exc))
                        self.log(f"Could not persist {filename}: {exc}", "error")
                        return
                    hydro_files[filename] = ArtifactRef.from_path(
                        stored,
                        artifact_id=f"hydro-input:{filename}",
                        kind="hydrology_array",
                        base_dir=run.run_dir,
                    )
        spec = WorkflowSpec(
            workflow_id="hydro_geophysics.forward",
            inputs={
                "context": {},
                "hydro_files": hydro_files,
            },
            parameters={
                **params,
                "methods": methods,
                "point1": list(self._point1 or []),
                "point2": list(self._point2 or []),
            },
            seed=seed,
            metadata={"profile_source": "qt_map_picker"},
        )
        recipe_path, script_path = export_workflow_bundle(
            spec, run.run_dir, stem="hydro_forward"
        )
        self._reproduce.set_bundle(recipe_path, script_path)
        self._workflow_recipe_path = str(recipe_path)
        self._run_busy = BusyStateController([self._run_btn])
        self._run_busy.start()
        self._run_btn.setText("Running…")
        self._progress.setVisible(True)
        self._progress.setRange(0, 0)  # busy
        self._run_status.setText("Starting forward modeling…")
        self.log("Starting hydro → geophysics forward modeling…", "info")
        self._worker = self.register_worker(
            WorkflowWorker(
                spec,
                RunContext(
                    project_root=run.run_dir,
                    output_dir=run.outputs_dir,
                ),
            )
        )
        self._worker.logged.connect(lambda message: self._on_worker_log(message, "info"))
        self._worker.succeeded.connect(self._on_workflow_ok)
        self._worker.failed.connect(lambda message: self._on_forward_failed(message, False))
        self._worker.finished.connect(self._reset_run_button)
        self._worker.start()

    def _on_workflow_ok(self, result: WorkflowRunResult) -> None:
        if hasattr(self.state, "update_workflow_result"):
            self.state.update_workflow_result(
                self.module_key,
                "hydro_geophysics.forward",
                result.to_dict(),
                recipe_path=self._workflow_recipe_path,
            )
        self._on_forward_ok(result.legacy_payload())

    def _on_worker_log(self, message: str, level: str) -> None:
        self._run_status.setText(message)
        self.log(message, level)

    def _on_forward_ok(self, result: dict) -> None:
        self._progress.setRange(0, 1); self._progress.setValue(1)
        self.log(f"Forward modeling complete: {result.get('methods')}.", "success")
        result["selected_points"] = [self._point1, self._point2]
        self.report_result(result)
        self._populate_results(result)
        self._go_to(5)

    def _on_forward_failed(self, message: str, backend_unavailable: bool) -> None:
        self.fail_persisted_run(message)
        self._progress.setRange(0, 1); self._progress.setValue(0)
        level = "warn" if backend_unavailable else "error"
        self.log(f"Forward run problem: {message}", level)
        config_path = self._export_config()
        note = ("Forward-modeling backend was not found or failed. The survey "
                f"configuration has been exported to {config_path}.")
        self._run_status.setText(note)
        self.log(note, "warn")
        self._populate_results({"status": "config_exported", "methods": self._collect_methods(),
                                "config_path": config_path, "figure_paths": []})
        self._go_to(5)

    def _reset_run_button(self) -> None:
        if self._run_busy is not None:
            self._run_busy.finish()
            self._run_busy = None
        self._run_btn.setText("Run forward modeling")
        self._progress.setVisible(False)
        self._update_strip()

    # -- AQUAH agent interface ----------------------------------------------
    def agent_describe(self) -> Dict[str, Any]:
        """Advertise the actions and current state to the AQUAH assistant."""
        return {
            "module": self.module_key,
            "title": self.module_title,
            "current_step": _STEPS[self._current],
            "state": self._agent_status(),
            "actions": [
                {"name": "use_example_data", "args": {},
                 "desc": "Load bundled example hydro data after the user explicitly chooses it."},
                {"name": "set_data_dir", "args": {"path": "str"},
                 "desc": "Load the user's hydro data after the user chooses it and supplies a folder."},
                {"name": "start_profile_pick", "args": {},
                 "desc": "Open the Profile step and pause while the user clicks two map points."},
                {"name": "pick_profile", "args": {"p1": "[col, row]", "p2": "[col, row]"},
                 "desc": "Define the 2D cross-section by two grid points."},
                {"name": "select_methods", "args": {"methods": list(_METHODS)},
                 "desc": "Choose which forward methods to simulate."},
                {"name": "set_params", "args": {"params": {"<key>": "value"}},
                  "desc": ("Set parameters. Keys: snapshot_index, num_samples, seed, "
                           "ert.num_electrodes, ert.electrode_spacing, ert.electrode_start, "
                           "ert.scheme_name (one of wa/dd/slm/dp), ert.noise_level, "
                           "ert.rel_error, ert.abs_error, srt.num_sensors, srt.sensor_spacing, "
                           "srt.shot_distance, srt.noise_level, srt.noise_abs, plus the "
                            "TDEM/FDEM/Gravity keys returned in parameter_defaults.")},
                {"name": "confirm_parameters", "args": {"mode": ["defaults", "custom"]},
                 "desc": "Confirm the user's explicit parameter choice before running; defaults keeps displayed values."},
                {"name": "goto_step", "args": {"step": list(_STEPS)},
                 "desc": "Jump to a wizard step by name or index."},
                {"name": "run", "args": {},
                 "desc": "Run hydro -> geophysics forward modeling (needs data, profile, methods)."},
                {"name": "get_status", "args": {},
                 "desc": "Report readiness and the last result status."},
            ],
        }

    def agent_apply(self, action: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute one AQUAH action against this module (called on the UI thread)."""
        args = args or {}
        handlers = {
            "use_example_data": lambda: self._agent_use_example_data(),
            "set_data_dir": lambda: self._agent_set_data_dir(args.get("path")),
            "start_profile_pick": lambda: self._agent_start_profile_pick(),
            "pick_profile": lambda: self._agent_pick_profile(args),
            "select_methods": lambda: self._agent_select_methods(args.get("methods")),
            "set_params": lambda: self._agent_set_params(args.get("params", args)),
            "confirm_parameters": lambda: self._agent_confirm_parameters(args.get("mode")),
            "goto_step": lambda: self._agent_goto_step(args.get("step")),
            "run": lambda: self._agent_run(),
            "get_status": lambda: self._agent_status(),
        }
        handler = handlers.get(action)
        if handler is None:
            return {"status": "failed", "error": f"Unknown action '{action}'.",
                    "valid_actions": list(handlers.keys())}
        return handler()

    def _agent_status(self) -> Dict[str, Any]:
        status = self._step_status()
        last = self.state.module_results.get(self.module_key, {})
        return {
            "status": "ok",
            "steps": status,
            "current_step": _STEPS[self._current],
            "data_dir": str(self._data_dir() or ""),
            "data_source_confirmation_required": not self._agent_data_source_confirmed,
            "data_source": self._agent_data_source,
            "data_source_options": ["example", "user_data"],
            "methods": self._collect_methods(),
            "parameter_defaults": self._agent_parameter_defaults(),
            "parameters_confirmed": self._agent_parameters_confirmed,
            "parameter_mode": self._agent_parameter_mode,
            "profile": {"p1": self._point1, "p2": self._point2},
            "last_result_status": last.get("status"),
        }

    def _agent_use_example_data(self) -> Dict[str, Any]:
        self._agent_data_source_confirmed = False
        self._agent_data_source = ""
        example_dir = self._example_data_dir()
        if example_dir is None:
            return {"status": "failed", "error": "Bundled example hydro data were not found."}
        self._manual_dir = example_dir
        self._reload()
        loaded = self._top is not None or self._wc is not None
        if loaded:
            self._agent_data_source_confirmed = True
            self._agent_data_source = "example"
        return {"status": "ok" if loaded else "failed",
                "data_dir": str(self._data_dir() or ""), "loaded": loaded,
                "data_source": self._agent_data_source}

    def _agent_set_data_dir(self, path: Any) -> Dict[str, Any]:
        self._agent_data_source_confirmed = False
        self._agent_data_source = ""
        if not path:
            return {"status": "failed", "error": "Provide 'path' to a data folder."}
        data_dir = Path(str(path))
        if not data_dir.is_dir():
            return {"status": "failed", "error": f"Hydro data folder does not exist: {data_dir}"}
        self._manual_dir = data_dir
        self._reload()
        loaded = self._top is not None or self._wc is not None
        if loaded:
            self._agent_data_source_confirmed = True
            self._agent_data_source = "user_data"
        return {"status": "ok" if loaded else "failed",
                "data_dir": str(self._data_dir() or ""), "loaded": loaded,
                "data_source": self._agent_data_source,
                **({} if loaded else {"error": "No usable hydro arrays were found in the folder."})}

    def _agent_require_data_source(self) -> Optional[Dict[str, Any]]:
        """Return an agent-facing checkpoint until the user chooses the data source."""
        if self._agent_data_source_confirmed:
            return None
        return {
            "status": "failed",
            "error": "Ask the user to choose bundled example data or their own hydro data first.",
            "required_choice": ["example", "user_data"],
            "available_context_data_dir": str(self.state.hydro_data_dir or ""),
        }

    def _agent_parameter_defaults(self) -> Dict[str, Dict[str, Any]]:
        """Return the visible default acquisition settings for AQUAH to present."""
        params = self._collect_params()
        selected = set(self._collect_methods())
        defaults: Dict[str, Dict[str, Any]] = {}
        if "ERT" in selected:
            defaults["ERT"] = {
                "electrodes": params["ert"]["num_electrodes"],
                "spacing_m": params["ert"]["electrode_spacing"],
                "array": params["ert"]["scheme_name"],
                "noise_level": params["ert"]["noise_level"],
                "relative_error": params["ert"]["rel_error"],
                "absolute_error": params["ert"]["abs_error"],
            }
        if "SRT" in selected:
            defaults["SRT"] = {
                "sensors": params["srt"]["num_sensors"],
                "spacing_m": params["srt"]["sensor_spacing"],
                "shot_distance_sensors": params["srt"]["shot_distance"],
                "noise_level": params["srt"]["noise_level"],
                "absolute_noise_s": params["srt"]["noise_abs"],
            }
        if "TDEM" in selected:
            defaults["TDEM"] = {
                "time_range_s": [params["tdem"]["t_min_s"], params["tdem"]["t_max_s"]],
                "time_gates": params["tdem"]["n_gates"], "noise_level": params["tdem"]["noise_level"],
            }
        if "FDEM" in selected:
            defaults["FDEM"] = {
                "frequency_range_hz": [params["fdem"]["f_min_hz"], params["fdem"]["f_max_hz"]],
                "frequencies": params["fdem"]["n_freqs"], "noise_level": params["fdem"]["noise_level"],
            }
        if "Gravity" in selected:
            defaults["Gravity"] = {
                "sensor_height_m": params["gravity"]["sensor_height"],
                "noise_level": params["gravity"]["noise_level"],
                "stations": params["em_stations"],
            }
        return defaults

    def _agent_start_profile_pick(self) -> Dict[str, Any]:
        """Open the interactive Profile step and pause AQUAH for two map clicks."""
        source_required = self._agent_require_data_source()
        if source_required:
            return source_required
        if self._top is None and self._wc is None:
            return {"status": "failed", "error": "Load hydrologic data before picking a profile.",
                    "hint": "Use example data or select a hydrologic model-output folder first."}
        self._go_to(1)
        self._profile_mode.setChecked(True)
        return {
            "status": "awaiting_user",
            "step": _STEPS[self._current],
            "message": "Profile map is open. Click two endpoints, then say ‘continue’.",
        }

    def _agent_pick_profile(self, args: Dict[str, Any]) -> Dict[str, Any]:
        source_required = self._agent_require_data_source()
        if source_required:
            return source_required
        p1, p2 = args.get("p1"), args.get("p2")
        if not (isinstance(p1, (list, tuple)) and isinstance(p2, (list, tuple))
                and len(p1) == 2 and len(p2) == 2):
            return {"status": "failed", "error": "Provide p1 and p2 as [col, row] pairs."}
        self._p1x.setValue(int(p1[0])); self._p1y.setValue(int(p1[1]))
        self._p2x.setValue(int(p2[0])); self._p2y.setValue(int(p2[1]))
        self._apply_manual_points()
        return {"status": "ok", "profile": {"p1": self._point1, "p2": self._point2}}

    def _agent_select_methods(self, methods: Any) -> Dict[str, Any]:
        source_required = self._agent_require_data_source()
        if source_required:
            return source_required
        if not isinstance(methods, list) or not methods:
            return {"status": "failed", "error": "Provide 'methods' as a non-empty list.",
                    "valid": list(_METHODS)}
        invalid = [m for m in methods if m not in _METHODS]
        if invalid:
            return {"status": "failed", "error": f"Invalid methods {invalid}.",
                    "valid": list(_METHODS)}
        for name, box in self._method_boxes.items():
            box.setChecked(name in methods)
        self._agent_parameters_confirmed = False
        self._agent_parameter_mode = ""
        self._go_to(3)
        self._update_strip()
        return {"status": "ok", "selected": self._collect_methods(),
                "parameter_defaults": self._agent_parameter_defaults(),
                "parameter_confirmation_required": True}

    def _agent_set_params(self, params: Any) -> Dict[str, Any]:
        if not isinstance(params, dict):
            return {"status": "failed", "error": "Provide 'params' as a JSON object."}

        def set_combo(combo, value):
            items = [combo.itemText(i) for i in range(combo.count())]
            if str(value) not in items:
                raise ValueError(f"must be one of {items}")
            combo.setCurrentText(str(value))

        handlers = {
            "snapshot_index": lambda v: self._snapshot.setValue(int(v)),
            "num_samples": lambda v: self._num_samples.setValue(int(v)),
            "seed": lambda v: self._seed.setValue(int(v)),
            "ert.num_electrodes": lambda v: self._ert_count.setValue(int(v)),
            "ert.electrode_spacing": lambda v: self._ert_spacing.setValue(float(v)),
            "ert.electrode_start": lambda v: self._ert_start.setValue(float(v)),
            "ert.scheme_name": lambda v: set_combo(self._ert_scheme, v),
            "ert.noise_level": lambda v: self._ert_noise.setValue(float(v)),
            "ert.rel_error": lambda v: self._ert_rel.setValue(float(v)),
            "ert.abs_error": lambda v: self._ert_abs.setValue(float(v)),
            "srt.num_sensors": lambda v: self._srt_count.setValue(int(v)),
            "srt.sensor_spacing": lambda v: self._srt_spacing.setValue(float(v)),
            "srt.shot_distance": lambda v: self._srt_shot.setValue(int(v)),
            "srt.noise_level": lambda v: self._srt_noise.setValue(float(v)),
            "srt.noise_abs": lambda v: self._srt_noise_abs.setValue(float(v)),
            "tdem.t_min_s": lambda v: self._tdem_tmin.setValue(float(v)),
            "tdem.t_max_s": lambda v: self._tdem_tmax.setValue(float(v)),
            "tdem.n_gates": lambda v: self._tdem_ngates.setValue(int(v)),
            "tdem.noise_level": lambda v: self._tdem_noise.setValue(float(v)),
            "fdem.f_min_hz": lambda v: self._fdem_fmin.setValue(float(v)),
            "fdem.f_max_hz": lambda v: self._fdem_fmax.setValue(float(v)),
            "fdem.n_freqs": lambda v: self._fdem_nfreq.setValue(int(v)),
            "fdem.noise_level": lambda v: self._fdem_noise.setValue(float(v)),
            "gravity.sensor_height": lambda v: self._grav_sensor_h.setValue(float(v)),
            "gravity.noise_level": lambda v: self._grav_noise.setValue(float(v)),
            "em_stations": lambda v: self._em_stations.setValue(int(v)),
        }
        applied: Dict[str, Any] = {}
        ignored: Dict[str, str] = {}
        for key, value in params.items():
            handler = handlers.get(key)
            if handler is None:
                ignored[key] = "unknown parameter"
                continue
            try:
                handler(value)
                applied[key] = value
            except Exception as exc:  # noqa: BLE001
                ignored[key] = str(exc)
        if applied:
            self._agent_parameters_confirmed = False
            self._agent_parameter_mode = ""
        return {"status": "ok" if applied else "failed", "applied": applied, "ignored": ignored}

    def _agent_confirm_parameters(self, mode: Any) -> Dict[str, Any]:
        """Record the user's explicit default/custom parameter confirmation."""
        if mode not in ("defaults", "custom"):
            return {"status": "failed", "error": "mode must be 'defaults' or 'custom'."}
        self._agent_parameters_confirmed = True
        self._agent_parameter_mode = str(mode)
        return {"status": "ok", "mode": self._agent_parameter_mode,
                "methods": self._collect_methods(), "parameter_defaults": self._agent_parameter_defaults()}

    def _agent_goto_step(self, step: Any) -> Dict[str, Any]:
        names = {name.lower(): i for i, name in enumerate(_STEPS)}
        idx = step if isinstance(step, int) else names.get(str(step).lower())
        if idx is None or not (0 <= idx < len(_STEPS)):
            return {"status": "failed", "error": f"Unknown step '{step}'.", "valid": list(_STEPS)}
        self._go_to(idx)
        return {"status": "ok", "step": _STEPS[self._current]}

    def _agent_run(self) -> Dict[str, Any]:
        source_required = self._agent_require_data_source()
        if source_required:
            return source_required
        status = self._step_status()
        if not status["all"]:
            missing = [k for k in ("data", "profile", "methods") if not status[k]]
            return {"status": "failed", "error": "Not ready to run.", "missing": missing,
                    "hint": "Load data, pick a profile, and select methods first."}
        if not self._agent_parameters_confirmed:
            return {"status": "failed",
                    "error": "Confirm the user's parameter choice before running.",
                    "required_action": "confirm_parameters",
                    "parameter_defaults": self._agent_parameter_defaults()}
        self._run_forward()
        return {"status": "started",
                "message": "Forward modeling started. Ask for status shortly.",
                "methods": self._collect_methods()}
