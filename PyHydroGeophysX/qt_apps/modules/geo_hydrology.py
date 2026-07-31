"""Geophysics -> Hydro guided wizard.

The inverse counterpart to the Hydro -> Geophysics module. A six-step workflow
(Data -> Layers -> Target -> Parameters -> Run -> Results) that turns an
already-inverted (time-lapse) ERT resistivity model into probabilistic water
content (and, optionally, saturated-zone porosity) via Monte Carlo sampling of
per-layer petrophysics. The inverted model is loaded first, its geological layers
are detected and named, the hydrologic target is chosen, then per-layer parameter
distributions are set before the run. The heavy lifting is reused from
``PyHydroGeophysX.qt_apps.geo_pipeline`` (a numpy-only model summary plus the real
pygimli-backed Monte Carlo run with a config-export fallback).
"""

from __future__ import annotations

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
    QLineEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QStackedWidget,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from PyHydroGeophysX.Geophy_modular import ERT_to_WC as geo_pipeline
from PyHydroGeophysX.qt_apps import io_utils, theme
from PyHydroGeophysX.qt_apps.modules.base import BaseModule, LogFn
from PyHydroGeophysX.qt_apps.qt_utils import (
    BusyStateController,
    ReproduceBar,
    WizardNavigator,
    make_double_spinbox,
    select_directory,
)
from PyHydroGeophysX.qt_apps.widgets.image_view import ZoomableImageView
from PyHydroGeophysX.qt_apps.workers import WorkflowWorker
from PyHydroGeophysX.workflows import (
    ArtifactRef,
    RunContext,
    WorkflowRunResult,
    WorkflowSpec,
    export_workflow_bundle,
)

_PRODUCTS = [("water_content", "Water content"), ("porosity", "Porosity (saturated zone)")]
_STEPS = ["Data", "Layers", "Target", "Parameters", "Run", "Results"]
_STEP_ICONS = [
    "fa5s.database",
    "fa5s.layer-group",
    "fa5s.bullseye",
    "fa5s.sliders-h",
    "fa5s.dice",
    "fa5s.chart-area",
]
_P = theme.PALETTE

# Per-parameter spin configuration: (lo, hi, step, decimals) for mean and std.
_PARAM_SPECS = {
    "m": ((1.0, 3.0, 0.1, 2), (0.0, 1.0, 0.05, 3)),
    "rho_fluid": ((0.1, 1000.0, 1.0, 2), (0.0, 500.0, 1.0, 2)),
    "n": ((1.0, 3.0, 0.1, 2), (0.0, 1.0, 0.05, 3)),
    "sigma_sur": ((0.0, 1.0, 0.001, 4), (0.0, 1.0, 0.001, 4)),
    "porosity": ((0.01, 0.9, 0.01, 3), (0.0, 0.5, 0.01, 3)),
}
_PARAM_LABELS = {
    "m": "Cementation m",
    "rho_fluid": "Fluid resistivity (Ω·m)",
    "n": "Saturation exponent n",
    "sigma_sur": "Surface cond. (S/m)",
    "porosity": "Porosity φ",
}

_GENERIC_LAYER = {
    "m": {"mean": 1.8, "std": 0.2},
    "rho_fluid": {"mean": 20.0, "std": 0.0},
    "n": {"mean": 2.0, "std": 0.1},
    "sigma_sur": {"mean": 0.0, "std": 0.0},
    "porosity": {"mean": 0.30, "std": 0.05},
}

_INPUT_FORMAT_FALLBACK = (
    "# ERT → Water Content input format\n\n"
    "Place an inverted-model bundle in one folder:\n\n"
    "- `mesh_res.bms` - pyGIMLi inversion mesh\n"
    "- `resmodel.npy` - inverted resistivity (n_cells, n_time)\n"
    "- `index_marker.npy` - layer id per cell (n_cells)\n"
    "- `all_coverage.npy` - coverage mask (n_time, n_cells), optional\n\n"
    "See geo_input_format.md for details."
)


class GeoHydrologyModule(BaseModule):
    module_key = "geo_hydrology"
    module_title = "ERT → Water Content"

    def __init__(self, state: Any, log: LogFn, parent=None) -> None:
        super().__init__(state, log, parent)
        self._manual_dir: Optional[Path] = None
        self._summary: Optional[Dict[str, Any]] = None
        self._layer_rows: List[Dict[str, Any]] = []
        self._param_widgets: Dict[int, Dict[str, Tuple[QDoubleSpinBox, QDoubleSpinBox]]] = {}
        self._last_result: Optional[Dict[str, Any]] = None
        self._markers_override: Optional[List[int]] = None
        self._worker: Optional[WorkflowWorker] = None
        self._run_busy: Optional[BusyStateController] = None
        self._workflow_recipe_path = ""
        self._current = 0

        root = QVBoxLayout(self)
        root.addWidget(self._build_strip())
        self._stack = QStackedWidget()
        self._stack.addWidget(self._build_data_step())        # 0 Data
        self._stack.addWidget(self._build_layers_step())      # 1 Layers
        self._stack.addWidget(self._build_target_step())      # 2 Target
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
        done = [
            status["data"],
            status["layers"],
            status["target"],
            status["target"],
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
        if self._current == 1:  # Layers
            self._refresh_layer_rows()
        if self._current == 3:  # Parameters
            self._refresh_parameter_panels()
        if self._current == 4:  # Run
            self._refresh_run_checklist()
        self._update_strip()

    def _step_status(self) -> Dict[str, bool]:
        data_ok = self._summary is not None
        layers_ok = len(self._enabled_layers()) > 0
        target_ok = len(self._collect_products()) > 0
        ran = self.state.module_results.get(self.module_key, {}).get("status") == "ok"
        return {"data": data_ok, "layers": layers_ok, "target": target_ok,
                "all": data_ok and layers_ok and target_ok, "results": ran}

    # -- Step 1: Data --------------------------------------------------------
    def _build_data_step(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(QLabel("<h3>Step 1 · Load an inverted ERT model</h3>"))
        layout.addWidget(QLabel(
            "Choose a folder containing <code>mesh_res.bms</code>, "
            "<code>resmodel.npy</code>, <code>index_marker.npy</code> "
            "(and optionally <code>all_coverage.npy</code>)."))
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
        for b in (ex_btn, sel_btn, reload_btn, help_btn):
            row.addWidget(b)
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
            self._checklist_layout.addWidget(QLabel("Select a model folder to validate."))
            return
        files = geo_pipeline.find_model_files(Path(data_dir))
        optional = {"coverage"}
        for key, name in geo_pipeline.MODEL_FILES.items():
            present = files.get(key) is not None
            opt = key in optional
            icon = "fa5s.check-circle" if present else ("fa5s.minus-circle" if opt else "fa5s.times-circle")
            color = _P["green"] if present else (_P["muted"] if opt else _P["red"])
            line = QHBoxLayout()
            badge = QLabel(); badge.setPixmap(theme.icon(icon, color=color).pixmap(16, 16))
            label = name + ("  (optional)" if opt else "")
            text = QLabel(label)
            line.addWidget(badge); line.addWidget(text); line.addStretch(1)
            wrap = QWidget(); wrap.setLayout(line)
            self._checklist_layout.addWidget(wrap)

    # -- Step 2: Layers ------------------------------------------------------
    def _build_layers_step(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(QLabel("<h3>Step 2 · Detect and name geological layers</h3>"))
        layout.addWidget(QLabel(
            "Layer ids come from <code>index_marker.npy</code>. Enable the layers "
            "to estimate and give each a name; these drive the petrophysics in "
            "Step 4. You can also derive the layers from a seismic bedrock "
            "interface handed over from the Seismic structure module."))
        detect_btn = QPushButton("Re-detect from data")
        detect_btn.setIcon(theme.icon("fa5s.sync"))
        detect_btn.clicked.connect(self._refresh_layer_rows)
        struct_btn = QPushButton("Derive markers from structure…")
        struct_btn.setIcon(theme.icon("fa5s.mountain"))
        struct_btn.setToolTip("Classify the model mesh into regolith / bedrock using a "
                              "seismic bedrock interface (from the Seismic module handoff "
                              "or an interface file).")
        struct_btn.clicked.connect(self._derive_markers_from_structure)
        top = QHBoxLayout(); top.addWidget(detect_btn); top.addWidget(struct_btn)
        top.addStretch(1)
        layout.addLayout(top)

        self._layers_box = QGroupBox("Detected layers")
        self._layers_layout = QVBoxLayout(self._layers_box)
        layout.addWidget(self._layers_box)
        self._layers_status = QLabel("")
        self._layers_status.setWordWrap(True)
        layout.addWidget(self._layers_status)
        layout.addStretch(1)
        return page

    def _refresh_layer_rows(self) -> None:
        while self._layers_layout.count():
            item = self._layers_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._layer_rows = []
        if self._summary is None:
            self._layers_layout.addWidget(QLabel("Load a model in Step 1 first."))
            self._layers_status.setText("")
            return
        header = QHBoxLayout()
        for txt, w in (("Use", 40), ("Marker", 70), ("Cells", 80), ("Name", 200)):
            lbl = QLabel(f"<b>{txt}</b>"); lbl.setMinimumWidth(w); header.addWidget(lbl)
        header.addStretch(1)
        hwrap = QWidget(); hwrap.setLayout(header)
        self._layers_layout.addWidget(hwrap)
        for layer in self._summary["layers"]:
            marker = int(layer["marker"])
            row = QHBoxLayout()
            enabled = QCheckBox()
            # Enable by default the layers that have known petrophysics defaults.
            enabled.setChecked(marker in geo_pipeline.DEFAULT_LAYER_DISTRIBUTIONS)
            enabled.toggled.connect(self._update_strip)
            mlbl = QLabel(str(marker)); mlbl.setMinimumWidth(70)
            clbl = QLabel(str(int(layer["n_cells"]))); clbl.setMinimumWidth(80)
            name = QLineEdit(str(layer.get("name", f"Layer {marker}")))
            name.setMinimumWidth(200)
            for w in (enabled, mlbl, clbl, name):
                row.addWidget(w)
            row.addStretch(1)
            wrap = QWidget(); wrap.setLayout(row)
            self._layers_layout.addWidget(wrap)
            self._layer_rows.append({"marker": marker, "n_cells": int(layer["n_cells"]),
                                     "enabled": enabled, "name": name})
        self._layers_status.setText(
            f"{len(self._summary['layers'])} marker(s) from {self._summary['markers_source']}; "
            f"{self._summary['n_cells']} cells × {self._summary['n_timesteps']} timesteps.")
        self._update_strip()

    def _enabled_layers(self) -> List[Dict[str, Any]]:
        return [r for r in self._layer_rows if r["enabled"].isChecked()]

    def _derive_markers_from_structure(self) -> None:
        if self._summary is None or self._data_dir() is None:
            self.log("Load an inverted-model bundle in Step 1 first.", "warn")
            return
        interface = None
        source = ""
        ss = self.state.shared_structure if self.state else None
        if ss and ss.get("interface_xz"):
            interface = np.asarray(ss["interface_xz"], dtype=float)
            source = str(ss.get("source", "seismic handoff"))
        else:
            path, _ = QFileDialog.getOpenFileName(
                self, "Load bedrock interface (x, z)", str(self._data_dir() or Path.cwd()),
                "Interface (*.csv *.txt *.npy);;All files (*)")
            if not path:
                return
            try:
                arr = np.atleast_2d(io_utils.load_2d_array(path))
                interface = arr[:, :2].astype(float)
            except Exception as exc:  # noqa: BLE001
                self.log(f"Could not read interface file: {exc}", "error")
                return
            source = Path(path).name
        if interface is None or interface.shape[0] < 2:
            self.log("Interface needs at least two (x, z) points.", "warn")
            return
        try:
            ctx = {"geo_data_dir": str(self._data_dir())}
            params = {"model_data_dir": str(self._data_dir()),
                      "output_dir": str(self.state.output_dir or "")}
            info = geo_pipeline.derive_markers_from_interface(
                ctx, params, interface, log=lambda m: self.log(m, "info"))
        except geo_pipeline.BackendUnavailable as exc:
            self.log(f"pygimli is needed to classify the mesh: {exc}", "warn")
            return
        except Exception as exc:  # noqa: BLE001
            self.log(f"Could not derive markers: {exc}", "error")
            return
        self._markers_override = info["markers"]
        self.log(f"Layers derived from {source}: counts={info['counts']}. "
                 f"Sidecar saved to {info['markers_path']}.", "success")
        self._reload()
        self._go_to(1)

    # -- Step 3: Target ------------------------------------------------------
    def _build_target_step(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(QLabel("<h3>Step 3 · Choose the hydrologic target</h3>"))
        layout.addWidget(QLabel(
            "Pick the product(s) to estimate. Water content samples all five "
            "petrophysics parameters; porosity assumes a known saturation."))

        prod_box = QGroupBox("Products")
        pform = QVBoxLayout(prod_box)
        self._product_boxes: Dict[str, QCheckBox] = {}
        for key, label in _PRODUCTS:
            box = QCheckBox(label)
            box.setChecked(key == "water_content")
            box.toggled.connect(self._on_products_changed)
            self._product_boxes[key] = box
            pform.addWidget(box)
        layout.addWidget(prod_box)

        opts = QGroupBox("Options")
        form = QFormLayout(opts)
        self._saturation = self._dspin(1.0, 0.01, 1.0, 0.05, 2)
        self._saturation.setToolTip("Assumed saturation for porosity estimation "
                                    "(1.0 = saturated zone).")
        self._preview_t = self._ispin(0, 0, 0)
        self._coverage_thr = self._dspin(-1.0, -100.0, 100.0, 0.1, 2)
        form.addRow("Saturation (porosity mode)", self._saturation)
        form.addRow("Preview timestep", self._preview_t)
        form.addRow("Coverage threshold", self._coverage_thr)
        layout.addWidget(opts)
        layout.addStretch(1)
        self._on_products_changed()
        return page

    def _on_products_changed(self) -> None:
        self._saturation.setEnabled(self._product_boxes["porosity"].isChecked())
        self._update_strip()

    def _collect_products(self) -> List[str]:
        if not hasattr(self, "_product_boxes"):
            return []
        return [k for k, b in self._product_boxes.items() if b.isChecked()]

    # -- Step 4: Parameters --------------------------------------------------
    def _build_parameters_step(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.addWidget(QLabel("<h3>Step 4 · Set per-layer petrophysics</h3>"))
        outer.addWidget(QLabel(
            "Each enabled layer gets mean and standard deviation for the "
            "Waxman-Smits / Archie parameters. Monte Carlo samples these."))

        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        inner = QWidget(); ilayout = QVBoxLayout(inner)
        ilayout.addWidget(self._build_shared_group())
        self._param_host = QWidget()
        self._param_host_layout = QVBoxLayout(self._param_host)
        self._param_host_layout.setContentsMargins(0, 0, 0, 0)
        ilayout.addWidget(self._param_host)
        ilayout.addStretch(1)
        scroll.setWidget(inner)
        outer.addWidget(scroll, stretch=1)
        return page

    def _build_shared_group(self) -> QGroupBox:
        box = QGroupBox("Monte Carlo settings")
        form = QFormLayout(box)
        self._n_real = self._ispin(100, 2, 5000)
        self._seed = self._ispin(7, 0, 10_000_000)
        form.addRow("Realizations", self._n_real)
        form.addRow("Random seed", self._seed)
        adv, aform = self._advanced_box()
        self._tortuosity = self._dspin(1.0, 0.1, 5.0, 0.1, 2)
        self._wc_cmin = self._dspin(0.0, 0.0, 1.0, 0.01, 2)
        self._wc_cmax = self._dspin(0.32, 0.01, 1.0, 0.01, 2)
        aform.addRow("Tortuosity a", self._tortuosity)
        aform.addRow("WC color min", self._wc_cmin)
        aform.addRow("WC color max", self._wc_cmax)
        form.addRow(adv)
        return box

    def _layer_default(self, marker: int) -> Dict[str, Dict[str, float]]:
        src = geo_pipeline.DEFAULT_LAYER_DISTRIBUTIONS.get(marker, _GENERIC_LAYER)
        return {k: dict(src[k]) for k in _PARAM_SPECS}

    def _refresh_parameter_panels(self) -> None:
        while self._param_host_layout.count():
            item = self._param_host_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._param_widgets = {}
        enabled = self._enabled_layers()
        if not enabled:
            self._param_host_layout.addWidget(
                QLabel("Enable at least one layer in Step 2."))
            return
        for row in enabled:
            marker = int(row["marker"])
            name = row["name"].text().strip() or f"Layer {marker}"
            panel = QGroupBox(f"{name}  (marker {marker})")
            grid = QGridLayout(panel)
            grid.addWidget(QLabel("<b>Parameter</b>"), 0, 0)
            grid.addWidget(QLabel("<b>Mean</b>"), 0, 1)
            grid.addWidget(QLabel("<b>Std</b>"), 0, 2)
            defaults = self._layer_default(marker)
            self._param_widgets[marker] = {}
            for r, key in enumerate(_PARAM_SPECS, start=1):
                (mlo, mhi, mstep, mdec), (slo, shi, sstep, sdec) = _PARAM_SPECS[key]
                mean_spin = self._dspin(float(defaults[key]["mean"]), mlo, mhi, mstep, mdec)
                std_spin = self._dspin(float(defaults[key]["std"]), slo, shi, sstep, sdec)
                grid.addWidget(QLabel(_PARAM_LABELS[key]), r, 0)
                grid.addWidget(mean_spin, r, 1)
                grid.addWidget(std_spin, r, 2)
                self._param_widgets[marker][key] = (mean_spin, std_spin)
            self._param_host_layout.addWidget(panel)

    # -- Step 5: Run ---------------------------------------------------------
    def _build_run_step(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(QLabel("<h3>Step 5 · Review and run the estimation</h3>"))
        self._run_checklist = QGroupBox("Readiness")
        self._run_checklist_layout = QVBoxLayout(self._run_checklist)
        layout.addWidget(self._run_checklist)

        self._run_btn = QPushButton("Run water-content estimation")
        self._run_btn.setProperty("primary", True)
        self._run_btn.setIcon(theme.icon("fa5s.play", color="#ffffff"))
        self._run_btn.clicked.connect(self._run_inverse)
        layout.addWidget(self._run_btn)
        cfg_btn = QPushButton("Export petrophysics config JSON (no backend needed)")
        cfg_btn.setIcon(theme.icon("fa5s.file-export"))
        cfg_btn.clicked.connect(self._export_config)
        layout.addWidget(cfg_btn)

        self._progress = QProgressBar(); self._progress.setVisible(False)
        layout.addWidget(self._progress)
        self._run_status = QLabel(""); self._run_status.setWordWrap(True)
        layout.addWidget(self._run_status)
        layout.addStretch(1)
        return page

    def _refresh_run_checklist(self) -> None:
        while self._run_checklist_layout.count():
            item = self._run_checklist_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        status = self._step_status()
        enabled = self._enabled_layers()
        rows = [
            ("Model loaded", status["data"], str(self._data_dir() or "—")),
            ("Layers enabled", status["layers"],
             ", ".join(f"{r['name'].text().strip() or r['marker']}" for r in enabled) or "none"),
            ("Target chosen", status["target"], ", ".join(self._collect_products()) or "none"),
        ]
        for label, ok, detail in rows:
            icon = "fa5s.check-circle" if ok else "fa5s.times-circle"
            color = _P["green"] if ok else _P["red"]
            line = QHBoxLayout()
            badge = QLabel(); badge.setPixmap(theme.icon(icon, color=color).pixmap(16, 16))
            text = QLabel(f"<b>{label}</b> — {detail}"); text.setWordWrap(True)
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

        self._result_summary = QLabel(
            "No results yet. Complete Steps 1–5 and run the estimation.")
        self._result_summary.setWordWrap(True)
        layout.addWidget(self._result_summary)

        self._result_tabs = QTabWidget()

        maps_page = QWidget()
        maps_layout = QVBoxLayout(maps_page)
        picker = QHBoxLayout()
        picker.addWidget(QLabel("Result map:"))
        self._result_figure = QComboBox()
        self._result_figure.currentIndexChanged.connect(self._show_selected_result_map)
        picker.addWidget(self._result_figure, stretch=1)
        maps_layout.addLayout(picker)
        self._result_map_caption = QLabel(
            "Run the estimation to view spatial products and uncertainty.")
        self._result_map_caption.setWordWrap(True)
        maps_layout.addWidget(self._result_map_caption)
        self._result_map_view = ZoomableImageView()
        self._result_map_view.setMinimumHeight(480)
        maps_layout.addWidget(self._result_map_view, stretch=1)
        self._result_tabs.addTab(maps_page, "Maps")

        self._monitor_page = QWidget()
        monitor_layout = QVBoxLayout(self._monitor_page)
        mon = QGroupBox("Monitoring points (water content time series)")
        mform = QFormLayout(mon)
        self._mx1 = self._dspin(30.0, -1e6, 1e6, 1.0, 1)
        self._my1 = self._dspin(1610.0, -1e6, 1e6, 1.0, 1)
        self._mx2 = self._dspin(65.0, -1e6, 1e6, 1.0, 1)
        self._my2 = self._dspin(1613.0, -1e6, 1e6, 1.0, 1)
        r1 = QHBoxLayout(); r1.addWidget(QLabel("P1 x")); r1.addWidget(self._mx1)
        r1.addWidget(QLabel("P1 y")); r1.addWidget(self._my1)
        r2 = QHBoxLayout(); r2.addWidget(QLabel("P2 x")); r2.addWidget(self._mx2)
        r2.addWidget(QLabel("P2 y")); r2.addWidget(self._my2)
        w1 = QWidget(); w1.setLayout(r1); w2 = QWidget(); w2.setLayout(r2)
        mform.addRow(w1); mform.addRow(w2)
        plot_btn = QPushButton("Plot time series at points")
        plot_btn.setIcon(theme.icon("fa5s.chart-line"))
        plot_btn.clicked.connect(self._plot_points)
        mform.addRow(plot_btn)
        monitor_layout.addWidget(mon)
        self._monitor_caption = QLabel(
            "Choose two locations and plot their water-content time series.")
        self._monitor_caption.setWordWrap(True)
        monitor_layout.addWidget(self._monitor_caption)
        self._monitor_view = ZoomableImageView()
        self._monitor_view.setMinimumHeight(420)
        monitor_layout.addWidget(self._monitor_view, stretch=1)
        self._result_tabs.addTab(self._monitor_page, "Monitoring")

        self._result_files = QTextBrowser()
        self._result_files.setReadOnly(True)
        self._result_files.setPlainText("Output files will be listed after a run.")
        self._result_tabs.addTab(self._result_files, "Files")

        layout.addWidget(self._result_tabs, stretch=1)
        return page

    def _populate_results(self, result: Dict[str, Any]) -> None:
        product_labels = {
            "water_content": "Water content",
            "porosity": "Porosity",
        }
        products = ", ".join(
            product_labels.get(str(product), str(product).replace("_", " ").title())
            for product in result.get("products", [])
        )
        self._result_summary.setText(
            f"<b>Status:</b> {result.get('status')} &nbsp; "
            f"<b>Products:</b> {products or '—'} &nbsp; "
            f"<b>Realizations:</b> {result.get('n_realizations', '—')} &nbsp; "
            f"<b>Mesh cells:</b> {result.get('mesh_cells', '—')} &nbsp; "
            f"<b>Timesteps:</b> {result.get('n_timesteps', '—')} &nbsp; "
            f"<b>Displayed timestep:</b> {result.get('preview_timestep', '—')}")

        entries: List[Tuple[str, str]] = []
        display_paths = result.get("display_paths", {})
        for product, statistics in display_paths.items():
            if not isinstance(statistics, dict):
                continue
            for statistic, path in statistics.items():
                if path and Path(path).exists():
                    entries.append((
                        self._result_map_label(str(product), str(statistic)), str(path)))

        # Backward-compatible fallback for results produced before display_paths.
        known_paths = {path for _, path in entries}
        for path in result.get("figure_paths", []):
            if path and Path(path).exists() and str(path) not in known_paths:
                entries.append((self._legacy_result_map_label(Path(path)), str(path)))

        self._result_figure.blockSignals(True)
        self._result_figure.clear()
        for label, path in entries:
            self._result_figure.addItem(label, path)
        self._result_figure.blockSignals(False)
        if self._result_figure.count():
            self._result_figure.setCurrentIndex(0)
        self._show_selected_result_map()

        file_lines = []
        figures = [str(path) for path in result.get("figure_paths", [])]
        stats = [str(path) for path in result.get("stats_paths", [])]
        if figures:
            file_lines.append("Map figures:\n" + "\n".join(f"  - {path}" for path in figures))
        if stats:
            file_lines.append("Derived arrays:\n" + "\n".join(f"  - {path}" for path in stats))
        cfg = result.get("config_path")
        if cfg:
            file_lines.append(f"Petrophysics config:\n  - {cfg}")
        output_dir = result.get("output_dir")
        if output_dir:
            file_lines.append(f"Output directory:\n  - {output_dir}")
        self._result_files.setPlainText(
            "\n\n".join(file_lines) if file_lines else "No output files were produced.")

    @staticmethod
    def _result_map_label(product: str, statistic: str) -> str:
        products = {
            "water_content": "Water content",
            "porosity": "Porosity",
        }
        statistics = {
            "mean": "Mean",
            "std": "Standard deviation (1σ)",
            "p10": "P10",
            "p50": "P50 (median)",
            "p90": "P90",
        }
        return f"{products.get(product, product.replace('_', ' ').title())} — " \
               f"{statistics.get(statistic, statistic.upper())}"

    @classmethod
    def _legacy_result_map_label(cls, path: Path) -> str:
        stem = path.stem.lower()
        product = "porosity" if "porosity" in stem else "water_content"
        statistic = next(
            (key for key in ("mean", "std", "p10", "p50", "p90") if key in stem),
            stem,
        )
        return cls._result_map_label(product, statistic)

    def _show_selected_result_map(self) -> None:
        path = self._result_figure.currentData()
        if path and Path(path).exists() and self._result_map_view.set_image_file(path):
            self._result_map_caption.setText(
                f"<b>{self._result_figure.currentText()}</b> &nbsp; "
                f"<code>{Path(path).name}</code> &nbsp; · drag to pan, wheel to zoom")
        else:
            self._result_map_view.clear()
            self._result_map_caption.setText(
                "No map figures were produced. Check the Files tab for exported configuration or arrays.")

    def _plot_points(self) -> None:
        res = self._last_result
        if not res or res.get("_wc_mean") is None:
            self.log("Run a water-content estimation first.", "warn")
            return
        cc = res.get("_cell_centers")
        wc_mean = res.get("_wc_mean")
        wc_std = res.get("_wc_std")
        positions = [(self._mx1.value(), self._my1.value()),
                     (self._mx2.value(), self._my2.value())]
        try:
            series, idx = geo_pipeline.extract_point_series(cc, wc_mean, positions)
            stds = None
            if wc_std is not None:
                stds, _ = geo_pipeline.extract_point_series(cc, wc_std, positions)
            t = res.get("_timestep_indices") or list(range(series.shape[1]))
            import matplotlib
            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, len(positions), figsize=(5.0 * len(positions), 3.4),
                                     squeeze=False)
            for i in range(len(positions)):
                ax = axes[0][i]
                ax.plot(t, series[i], "o-", color="tab:blue", label="mean")
                if stds is not None:
                    ax.fill_between(t, series[i] - stds[i], series[i] + stds[i],
                                    color="tab:blue", alpha=0.2, label="±1σ")
                ax.set_xlabel("Timestep"); ax.set_ylabel("Water content (-)")
                ax.set_title(f"P{i + 1} (cell {idx[i]})"); ax.grid(True, alpha=0.3)
                if i == 0:
                    ax.legend(frameon=False)
            fig.tight_layout()
            out = Path(res.get("output_dir", ".")) / "monitoring_series.png"
            fig.savefig(out, dpi=170, bbox_inches="tight")
            plt.close(fig)
        except Exception as exc:  # noqa: BLE001
            self.log(f"Could not plot monitoring series: {exc}", "error")
            return
        if self._monitor_view.set_image_file(str(out)):
            self._monitor_caption.setText(
                f"<b>Water-content time series</b> &nbsp; <code>{out.name}</code> &nbsp; "
                "±1σ shading shows propagated petrophysical uncertainty.")
            self._result_tabs.setCurrentWidget(self._monitor_page)
        self.log(f"Saved monitoring series to {out}", "success")

    def _open_output_folder(self) -> None:
        out = self.state.module_results.get(self.module_key, {}).get("output_dir")
        out = out or str(self.state.output_dir or "")
        if out and Path(out).exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(out))
        else:
            self.log("Output folder not available yet.", "warn")

    # -- data loading --------------------------------------------------------
    def _data_dir(self) -> Optional[Path]:
        if self._manual_dir is not None:
            return self._manual_dir
        geo_dir = self.state.context.get("geo_data_dir") if self.state.context else None
        if geo_dir:
            return Path(geo_dir)
        return None

    def _use_context_data(self) -> None:
        self._manual_dir = None
        self._markers_override = None
        if self._data_dir() is None:
            self._manual_dir = self._example_data_dir()
        self._reload()

    def _example_data_dir(self) -> Optional[Path]:
        """Locate the bundled synthetic inverted-model demo for standalone launches.

        Prefer a full-resolution dataset produced by
        ``examples/generate_synthetic_examples.py`` and fall back to the compact
        copy shipped with the package. The real Treeline demo in
        ``examples/results/Structure_WC`` can still be opened via "Select folder...".
        """
        candidates: List[Path] = []
        if self.state.project_root:
            candidates.append(Path(self.state.project_root) / "examples" / "results" / "synthetic_Structure_WC")
        here = Path(__file__).resolve()
        candidates.append(here.parents[2] / "data" / "qt_examples" / "geo_hydrology")
        candidates.extend(p / "examples" / "results" / "synthetic_Structure_WC" for p in here.parents)
        for cand in candidates:
            files = geo_pipeline.find_model_files(cand)
            if files.get("resistivity") is not None and files.get("mesh") is not None:
                return cand
        return None

    def _show_format_help(self) -> None:
        doc_path = Path(__file__).with_name("geo_input_format.md")
        try:
            text = doc_path.read_text(encoding="utf-8")
        except Exception:  # noqa: BLE001
            text = _INPUT_FORMAT_FALLBACK
        dlg = QDialog(self)
        dlg.setWindowTitle("ERT → Water Content input format")
        dlg.resize(760, 640)
        lay = QVBoxLayout(dlg)
        browser = QTextBrowser(); browser.setOpenExternalLinks(True)
        try:
            browser.setMarkdown(text)
        except Exception:  # noqa: BLE001
            browser.setPlainText(text)
        lay.addWidget(browser)
        close_btn = QPushButton("Close"); close_btn.clicked.connect(dlg.accept)
        lay.addWidget(close_btn)
        dlg.exec()

    def _select_folder(self) -> None:
        start = str(self._data_dir() or Path.cwd())
        path = select_directory(self, "Select inverted-model folder", start)
        if path:
            self._manual_dir = path
            self._markers_override = None
            self._reload()

    def _reload(self) -> None:
        data_dir = self._data_dir()
        if hasattr(self, "_checklist_layout"):
            self._refresh_checklist()
        self._summary = None
        if data_dir is None or not Path(data_dir).exists():
            if hasattr(self, "_data_status"):
                self._data_status.setText("No model folder set. Use 'Select folder…'.")
            self._update_strip()
            return
        context = {"geo_data_dir": str(data_dir)}
        summary_params: Dict[str, Any] = {"model_data_dir": str(data_dir)}
        if self._markers_override is not None:
            summary_params["markers_override"] = self._markers_override
        try:
            self._summary = geo_pipeline.extract_model_summary(
                context, summary_params, log=lambda m: self.log(m, "debug"))
        except Exception as exc:  # noqa: BLE001
            self._summary = None
            if hasattr(self, "_data_status"):
                self._data_status.setText(f"Could not read model: {exc}")
            self.log(f"Failed to read inverted model: {exc}", "error")
            self._update_strip()
            return
        s = self._summary
        if hasattr(self, "_data_status"):
            self._data_status.setText(
                f"Folder: {data_dir}<br>Cells: {s['n_cells']} · Timesteps: "
                f"{s['n_timesteps']} · Markers: "
                f"{', '.join(str(l['marker']) for l in s['layers'])}")
        if hasattr(self, "_preview_t"):
            self._preview_t.setRange(0, max(0, s["n_timesteps"] - 1))
        self.log(f"Loaded inverted model from {data_dir}", "success")
        self._refresh_layer_rows()
        self._update_strip()

    # -- params / run --------------------------------------------------------
    def _collect_layers(self) -> List[Dict[str, Any]]:
        layers: List[Dict[str, Any]] = []
        for row in self._enabled_layers():
            marker = int(row["marker"])
            entry: Dict[str, Any] = {"marker": marker,
                                     "name": row["name"].text().strip() or f"Layer {marker}"}
            widgets = self._param_widgets.get(marker)
            defaults = self._layer_default(marker)
            for key in _PARAM_SPECS:
                if widgets and key in widgets:
                    mean_spin, std_spin = widgets[key]
                    entry[key] = {"mean": mean_spin.value(), "std": std_spin.value()}
                else:
                    entry[key] = dict(defaults[key])
            layers.append(entry)
        return layers

    def _collect_params(self) -> Dict[str, Any]:
        out_dir = self.state.output_dir
        return {
            "model_data_dir": str(self._data_dir()) if self._data_dir() else "",
            "output_dir": str(out_dir) if out_dir else "",
            "seed": self._seed.value(),
            "n_realizations": self._n_real.value(),
            "products": self._collect_products(),
            "saturation_value": self._saturation.value(),
            "tortuosity_a": self._tortuosity.value(),
            "coverage_threshold": self._coverage_thr.value(),
            "preview_timestep": self._preview_t.value(),
            "wc_cmin": self._wc_cmin.value(),
            "wc_cmax": self._wc_cmax.value(),
            "layers": self._collect_layers(),
            "markers_override": self._markers_override,
        }

    def _context(self) -> Dict[str, Any]:
        ctx = dict(self.state.context or {})
        if self._data_dir():
            ctx["geo_data_dir"] = str(self._data_dir())
        if self.state.output_dir:
            ctx.setdefault("output_dir", str(self.state.output_dir))
        return ctx

    def _export_config(self) -> str:
        params = self._collect_params()
        config = geo_pipeline.build_petro_config(self._context(), params)
        out_dir = io_utils.ensure_dir(Path(params["output_dir"] or ".") / "qt_geo_inverse")
        config_path = out_dir / "petro_config.json"
        io_utils.write_json(config_path, config)
        self.log(f"Exported petrophysics config to {config_path}", "success")
        self.report_result({
            "products": params["products"], "config_path": str(config_path),
            "status": "config_exported", "output_dir": str(out_dir)})
        return str(config_path)

    def _run_inverse(self) -> None:
        status = self._step_status()
        if not status["all"]:
            self.log("Complete Data, Layers and Target first.", "warn")
            return
        params = self._collect_params()
        seed = int(params.pop("seed"))
        data_dir = self._data_dir()
        params.pop("model_data_dir", None)
        output_base = Path(params.pop("output_dir", "") or ".")
        model_files: Dict[str, ArtifactRef] = {}
        if data_dir is not None:
            for filename in (
                "mesh_res.bms",
                "resmodel.npy",
                "index_marker.npy",
                "all_coverage.npy",
            ):
                path = data_dir / filename
                if path.is_file():
                    model_files[filename] = ArtifactRef.from_path(
                        path,
                        artifact_id=f"ert-model-input:{filename}",
                        kind="ert_model_bundle",
                    )
        spec = WorkflowSpec(
            workflow_id="geo_hydrology.ert_to_wc",
            inputs={
                "context": {},
                "model_files": model_files,
            },
            parameters=params,
            seed=seed,
            metadata={"source": "qt", "rng": "numpy.default_rng"},
        )
        bundle_dir = io_utils.ensure_dir(output_base / "qt_geo_inverse")
        recipe_path, script_path = export_workflow_bundle(
            spec, bundle_dir, stem="ert_to_water_content"
        )
        self._reproduce.set_bundle(recipe_path, script_path)
        self._workflow_recipe_path = str(recipe_path)
        self._run_busy = BusyStateController([self._run_btn])
        self._run_busy.start()
        self._run_btn.setText("Running…")
        self._progress.setVisible(True)
        self._progress.setRange(0, 0)
        self._run_status.setText("Starting Monte Carlo estimation…")
        self.log("Starting geophysics → hydrology estimation…", "info")
        self._worker = self.register_worker(
            WorkflowWorker(
                spec,
                RunContext(project_root=bundle_dir, output_dir=output_base),
            )
        )
        self._worker.logged.connect(lambda message: self._on_worker_log(message, "info"))
        self._worker.succeeded.connect(self._on_workflow_ok)
        self._worker.failed.connect(lambda message: self._on_run_failed(message, False))
        self._worker.finished.connect(self._reset_run_button)
        self._worker.start()

    def _on_workflow_ok(self, result: WorkflowRunResult) -> None:
        if hasattr(self.state, "update_workflow_result"):
            self.state.update_workflow_result(
                self.module_key,
                "geo_hydrology.ert_to_wc",
                result.to_dict(),
                recipe_path=self._workflow_recipe_path,
            )
        self._on_run_ok(result.legacy_payload())

    def _on_worker_log(self, message: str, level: str) -> None:
        self._run_status.setText(message)
        self.log(message, level)

    def _on_run_ok(self, result: dict) -> None:
        self._progress.setRange(0, 1); self._progress.setValue(1)
        self.log(f"Estimation complete: {result.get('products')}.", "success")
        self._last_result = result
        # Strip the in-process numpy payload before publishing to the bridge JSON.
        published = {k: v for k, v in result.items() if not k.startswith("_")}
        self.report_result(published)
        self._populate_results(result)
        self._go_to(5)

    def _on_run_failed(self, message: str, backend_unavailable: bool) -> None:
        self._progress.setRange(0, 1); self._progress.setValue(0)
        level = "warn" if backend_unavailable else "error"
        self.log(f"Estimation problem: {message}", level)
        config_path = self._export_config()
        note = ("The Monte Carlo backend (pygimli) was not found or failed. The "
                f"petrophysics configuration has been exported to {config_path}.")
        self._run_status.setText(note)
        self.log(note, "warn")
        self._last_result = None
        self._populate_results({"status": "config_exported",
                                "products": self._collect_products(),
                                "config_path": config_path, "figure_paths": []})
        self._go_to(5)

    def _reset_run_button(self) -> None:
        if self._run_busy is not None:
            self._run_busy.finish()
            self._run_busy = None
        self._run_btn.setText("Run water-content estimation")
        self._progress.setVisible(False)
        self._update_strip()

    # -- AQUAH agent interface ----------------------------------------------
    def agent_describe(self) -> Dict[str, Any]:
        return {
            "module": self.module_key,
            "title": self.module_title,
            "current_step": _STEPS[self._current],
            "state": self._agent_status(),
            "actions": [
                {"name": "use_example_data", "args": {},
                 "desc": "Load the bundled / context inverted-ERT model bundle."},
                {"name": "set_data_dir", "args": {"path": "str"},
                 "desc": "Load an inverted model folder (mesh_res.bms, resmodel.npy, index_marker.npy)."},
                {"name": "detect_layers", "args": {},
                 "desc": "Re-detect geological layers from the loaded model markers."},
                {"name": "select_products", "args": {"products": [k for k, _ in _PRODUCTS]},
                 "desc": "Choose hydrologic products to estimate."},
                {"name": "set_params", "args": {"params": {"<key>": "value"}},
                 "desc": ("Set parameters. Keys: n_realizations, seed, saturation, "
                          "coverage_threshold, preview_timestep, tortuosity, wc_cmin, wc_cmax.")},
                {"name": "goto_step", "args": {"step": list(_STEPS)},
                 "desc": "Jump to a wizard step."},
                {"name": "run", "args": {},
                 "desc": "Run the Monte Carlo water-content / porosity estimation."},
                {"name": "get_status", "args": {},
                 "desc": "Report readiness and the last result status."},
            ],
        }

    def agent_apply(self, action: str, args: Dict[str, Any]) -> Dict[str, Any]:
        args = args or {}
        handlers = {
            "use_example_data": lambda: self._agent_use_example_data(),
            "set_data_dir": lambda: self._agent_set_data_dir(args.get("path")),
            "detect_layers": lambda: self._agent_detect_layers(),
            "select_products": lambda: self._agent_select_products(args.get("products")),
            "set_params": lambda: self._agent_set_params(args.get("params", args)),
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
            "products": self._collect_products(),
            "enabled_layers": [r["marker"] for r in self._enabled_layers()],
            "last_result_status": last.get("status"),
        }

    def _agent_use_example_data(self) -> Dict[str, Any]:
        self._use_context_data()
        return {"status": "ok", "data_dir": str(self._data_dir() or ""),
                "loaded": self._summary is not None}

    def _agent_set_data_dir(self, path: Any) -> Dict[str, Any]:
        if not path:
            return {"status": "failed", "error": "Provide 'path' to a model folder."}
        self._manual_dir = Path(str(path))
        self._markers_override = None
        self._reload()
        return {"status": "ok", "data_dir": str(self._data_dir() or ""),
                "loaded": self._summary is not None}

    def _agent_detect_layers(self) -> Dict[str, Any]:
        if self._summary is None:
            return {"status": "failed", "error": "Load a model first (use_example_data or set_data_dir)."}
        self._refresh_layer_rows()
        return {"status": "ok", "enabled_layers": [r["marker"] for r in self._enabled_layers()]}

    def _agent_select_products(self, products: Any) -> Dict[str, Any]:
        valid = {k for k, _ in _PRODUCTS}
        if not isinstance(products, list) or not products:
            return {"status": "failed", "error": "Provide 'products' as a non-empty list.",
                    "valid": sorted(valid)}
        invalid = [p for p in products if p not in valid]
        if invalid:
            return {"status": "failed", "error": f"Invalid products {invalid}.", "valid": sorted(valid)}
        for key, box in self._product_boxes.items():
            box.setChecked(key in products)
        self._on_products_changed()
        return {"status": "ok", "selected": self._collect_products()}

    def _agent_set_params(self, params: Any) -> Dict[str, Any]:
        if not isinstance(params, dict):
            return {"status": "failed", "error": "Provide 'params' as a JSON object."}
        handlers = {
            "n_realizations": lambda v: self._n_real.setValue(int(v)),
            "seed": lambda v: self._seed.setValue(int(v)),
            "saturation": lambda v: self._saturation.setValue(float(v)),
            "coverage_threshold": lambda v: self._coverage_thr.setValue(float(v)),
            "preview_timestep": lambda v: self._preview_t.setValue(int(v)),
            "tortuosity": lambda v: self._tortuosity.setValue(float(v)),
            "wc_cmin": lambda v: self._wc_cmin.setValue(float(v)),
            "wc_cmax": lambda v: self._wc_cmax.setValue(float(v)),
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
        return {"status": "ok" if applied else "failed", "applied": applied, "ignored": ignored}

    def _agent_goto_step(self, step: Any) -> Dict[str, Any]:
        names = {name.lower(): i for i, name in enumerate(_STEPS)}
        idx = step if isinstance(step, int) else names.get(str(step).lower())
        if idx is None or not (0 <= idx < len(_STEPS)):
            return {"status": "failed", "error": f"Unknown step '{step}'.", "valid": list(_STEPS)}
        self._go_to(idx)
        return {"status": "ok", "step": _STEPS[self._current]}

    def _agent_run(self) -> Dict[str, Any]:
        status = self._step_status()
        if not status["all"]:
            missing = [k for k in ("data", "layers", "target") if not status[k]]
            return {"status": "failed", "error": "Not ready to run.", "missing": missing,
                    "hint": "Load a model, ensure at least one layer is enabled, and select a product."}
        self._run_inverse()
        return {"status": "started",
                "message": "Water-content estimation started. Ask for status shortly.",
                "products": self._collect_products()}
