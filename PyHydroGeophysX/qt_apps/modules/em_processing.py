"""EM module: 1D FDEM / TDEM inversion.

Load a sounding (FDEM: frequency, real, imag; TDEM: time, response) and invert it
for a layered resistivity model. A file that stacks several soundings side by side
(a survey line) is inverted sounding-by-sounding and stitched into a position x
depth resistivity section. The numerics live in the Qt-free
``PyHydroGeophysX.qt_apps.em_pipeline`` (a thin wrapper over the package's SimPEG
forward operators). Results export to npy / csv.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
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

from PyHydroGeophysX.qt_apps import em_pipeline, io_utils, theme
from PyHydroGeophysX.qt_apps.modules.base import BaseModule, LogFn
from PyHydroGeophysX.qt_apps.widgets.curve_viewer import CurveViewer
from PyHydroGeophysX.qt_apps.widgets.image_view import ZoomableImageView
from PyHydroGeophysX.qt_apps.widgets.model3d_view import Model3DView
from PyHydroGeophysX.qt_apps.widgets.plan_slice_view import PlanSliceView
from PyHydroGeophysX.qt_apps.widgets.quality_view import InversionQualityView
from PyHydroGeophysX.qt_apps.workers import TaskWorker

_FILE_FILTER = "Sounding (*.csv *.txt *.dat);;All files (*)"


class InversionWorker(QThread):
    """Runs a single-sounding 1D EM inversion off the UI thread."""

    logged = Signal(str)
    succeeded = Signal(dict)
    failed = Signal(str, bool)

    def __init__(self, method, data, geom, inv) -> None:
        super().__init__()
        self._method = method
        self._data = data
        self._geom = geom
        self._inv = inv

    def run(self) -> None:  # noqa: D401
        try:
            fn = em_pipeline.fdem_invert if self._method == "FDEM" else em_pipeline.tdem_invert
            result = fn(self._data, self._geom, self._inv, log=lambda m: self.logged.emit(str(m)))
            self.succeeded.emit(result)
        except em_pipeline.BackendUnavailable as exc:
            self.failed.emit(str(exc), True)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc), False)


class EMProcessingModule(BaseModule):
    module_key = "em_processing"
    module_title = "EM Processing"

    def __init__(self, state: Any, log: LogFn, parent=None) -> None:
        super().__init__(state, log, parent)
        self._data: Optional[Dict[str, np.ndarray]] = None
        self._source_path: Optional[Path] = None
        self._last_result: Optional[Dict[str, Any]] = None
        self._last_section: Optional[Dict[str, Any]] = None
        self._inv_worker: Optional[InversionWorker] = None
        self._line_worker: Optional[TaskWorker] = None
        self._geom_positions: Optional[np.ndarray] = None
        self._geom_heights: Optional[np.ndarray] = None
        self._geom_x: Optional[np.ndarray] = None
        self._geom_y: Optional[np.ndarray] = None
        self._example_id: Optional[str] = None

        root = QHBoxLayout(self)
        self._tabs = QTabWidget()
        self._curve = CurveViewer()
        # The "Resistivity model" tab adapts to the result: a 1D depth profile
        # (single sounding), or — for a line — a plan-view depth slice (a map you
        # slice by depth) or the position x depth section, chosen with "View".
        self._inv_view = ZoomableImageView()       # page 0: single-sounding profile
        self._plan_view = PlanSliceView()          # page 1: plan-view depth slice
        self._section_view = Model3DView()         # page 2: position x depth section
        self._model_stack = QStackedWidget()
        self._model_stack.addWidget(self._inv_view)
        self._model_stack.addWidget(self._plan_view)
        self._model_stack.addWidget(self._section_view)
        self._model_tab = QWidget()
        mlay = QVBoxLayout(self._model_tab); mlay.setContentsMargins(0, 0, 0, 0)
        self._view_row = QWidget()
        vr = QHBoxLayout(self._view_row); vr.setContentsMargins(6, 2, 6, 2)
        vr.addWidget(QLabel("View:"))
        self._view_mode = QComboBox(); self._view_mode.addItems(["Plan slice (map)", "Section"])
        self._view_mode.currentIndexChanged.connect(self._on_view_mode)
        vr.addWidget(self._view_mode); vr.addStretch(1)
        self._view_row.setVisible(False)
        mlay.addWidget(self._view_row)
        mlay.addWidget(self._model_stack, stretch=1)
        self._quality_view = InversionQualityView()
        self._tabs.addTab(self._curve, "Sounding")
        self._tabs.addTab(self._model_tab, "Resistivity model")
        self._tabs.addTab(self._quality_view, "Inversion quality")
        root.addWidget(self._tabs, stretch=1)
        root.addWidget(self._build_controls())
        self._on_method_changed()

    def _on_view_mode(self, idx: int) -> None:
        self._model_stack.setCurrentWidget(self._plan_view if idx == 0 else self._section_view)

    # -- helpers -------------------------------------------------------------
    @staticmethod
    def _dspin(value, lo, hi, step, dec) -> QDoubleSpinBox:
        s = QDoubleSpinBox(); s.setRange(lo, hi); s.setSingleStep(step)
        s.setDecimals(dec); s.setValue(value)
        return s

    @staticmethod
    def _ispin(value, lo, hi) -> QSpinBox:
        s = QSpinBox(); s.setRange(lo, hi); s.setValue(value)
        return s

    def _build_controls(self) -> QScrollArea:
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        # Wide enough to fit the controls without a horizontal scrollbar.
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(450); scroll.setMaximumWidth(500)
        panel = QWidget(); scroll.setWidget(panel)
        layout = QVBoxLayout(panel)

        layout.addWidget(self._build_loader_group())
        layout.addWidget(self._build_geometry_group())
        layout.addWidget(self._build_inversion_group())
        layout.addStretch(1)
        return scroll

    def _build_loader_group(self) -> QGroupBox:
        box = QGroupBox("Load sounding data"); v = QVBoxLayout(box)
        self._method = QComboBox(); self._method.addItems(list(em_pipeline.METHODS))
        self._method.currentTextChanged.connect(self._on_method_changed)
        mrow = QFormLayout(); mrow.addRow("Method", self._method)
        v.addLayout(mrow)

        row = QHBoxLayout()
        load_btn = QPushButton("Load sounding(s)…")
        load_btn.setProperty("primary", True)
        load_btn.setIcon(theme.icon("fa5s.folder-open", color="#ffffff"))
        load_btn.clicked.connect(self._load)
        fmt_btn = QPushButton("Data format")
        fmt_btn.setIcon(theme.icon("fa5s.file-alt"))
        fmt_btn.clicked.connect(self._show_format_help)
        row.addWidget(load_btn); row.addWidget(fmt_btn)
        v.addLayout(row)

        # Optional per-sounding geometry (positions / height) for a survey line;
        # sits next to the loader, shown once a multi-sounding file is loaded.
        self._geom_row = QWidget()
        gv = QVBoxLayout(self._geom_row); gv.setContentsMargins(0, 0, 0, 0)
        grow = QHBoxLayout()
        self._geom_btn = QPushButton("Load geometry…")
        self._geom_btn.setIcon(theme.icon("fa5s.map-marker-alt"))
        self._geom_btn.setToolTip("Load a file of along-line positions (and optional sensor "
                                  "height), one row per sounding, so the section uses real "
                                  "distances instead of uniform spacing. See Data format.")
        self._geom_btn.clicked.connect(self._load_geometry)
        grow.addWidget(self._geom_btn); grow.addStretch(1)
        gv.addLayout(grow)
        self._geom_info = QLabel("Geometry: uniform spacing.")
        self._geom_info.setStyleSheet("color:#5a6a7a; font-size:8pt;"); self._geom_info.setWordWrap(True)
        gv.addWidget(self._geom_info)
        self._geom_row.setVisible(False)
        v.addWidget(self._geom_row)

        hint = QLabel("Upload one file. FDEM: columns freq, real, imag. TDEM: columns "
                      "time, response. Stack several soundings side by side for a survey "
                      "line (it is inverted into a resistivity section).")
        hint.setWordWrap(True); hint.setStyleSheet("color:#5a6a7a; font-size:8pt;")
        v.addWidget(hint)

        # Picker for files that hold several soundings (drives the preview curve).
        self._sounding_row = QWidget()
        srow = QFormLayout(self._sounding_row); srow.setContentsMargins(0, 0, 0, 0)
        self._sounding = QSpinBox(); self._sounding.setRange(1, 1); self._sounding.setValue(1)
        self._sounding.setToolTip("This file holds several soundings; pick which one to preview.")
        self._sounding.valueChanged.connect(self._on_sounding_changed)
        srow.addRow("Preview sounding #", self._sounding)
        self._sounding_row.setVisible(False)
        v.addWidget(self._sounding_row)

        self._info = QLabel("No sounding loaded.")
        self._info.setWordWrap(True)
        v.addWidget(self._info)
        self._example_note = QLabel("")
        self._example_note.setWordWrap(True)
        self._example_note.setStyleSheet("color:#5a6a7a; font-size:8pt;")
        self._example_note.setVisible(False)
        v.addWidget(self._example_note)
        return box

    def _build_geometry_group(self) -> QGroupBox:
        box = QGroupBox("Survey geometry (system)"); form = QFormLayout(box)
        f = em_pipeline.DEFAULT_FDEM
        self._src_radius = self._dspin(f["source_radius"], 0.5, 200.0, 1.0, 1)
        self._tx_rx = self._dspin(f["tx_rx_sep"], 0.0, 500.0, 1.0, 1)
        self._height = self._dspin(f["height"], 0.0, 500.0, 1.0, 1)
        self._orient = QComboBox(); self._orient.addItems(["z", "x", "y"])
        self._component = QComboBox(); self._component.addItems(["secondary", "total", "both"])
        self._waveform = QComboBox()
        form.addRow("Source radius (m)", self._src_radius)
        self._tx_rx_label = QLabel("Tx-Rx sep (m)")
        form.addRow(self._tx_rx_label, self._tx_rx)
        form.addRow("Height (m)", self._height)
        form.addRow("Orientation", self._orient)
        self._component_label = QLabel("Component")
        form.addRow(self._component_label, self._component)
        form.addRow("Waveform", self._waveform)
        return box

    def _build_inversion_group(self) -> QGroupBox:
        box = QGroupBox("Inversion (1D Occam)"); form = QFormLayout(box)
        d = em_pipeline.DEFAULT_INVERSION
        self._n_layers = self._ispin(d["n_layers"], 3, 60)
        self._min_thick = self._dspin(d["min_thickness"], 0.2, 50.0, 0.5, 2)
        self._max_thick = self._dspin(d["max_thickness"], 1.0, 500.0, 1.0, 1)
        self._smooth = self._dspin(d["smoothness"], 0.0, 10.0, 0.1, 2)
        self._rel_err = self._dspin(d["rel_error"], 0.0, 1.0, 0.01, 3)
        self._data_scale = self._dspin(1.0, 1e-4, 1e6, 0.1, 4)
        self._data_scale.setToolTip(
            "Multiply the observed data before inversion. Use for data in normalized units "
            "(e.g. moment-normalized airborne dB/dt) that need a system calibration constant. "
            "1.0 = no scaling.")
        self._auto_scale = QCheckBox("Auto-calibrate")
        self._auto_scale.setChecked(True)
        self._auto_scale.setToolTip("Rough amplitude guess from the data shape (de-rails the "
                                    "inversion, but the absolute resistivity is only approximate). "
                                    "For reliable absolute values, use Reference resistivity instead.")
        self._ref_res = self._dspin(0.0, 0.0, 1e5, 10.0, 1)
        self._ref_res.setToolTip(
            "Pin the ABSOLUTE resistivity to a known/expected value (ohm-m) from a borehole or "
            "regional geology — it ties the data amplitude to a half-space at this value, the "
            "same way for every dataset, so results are consistent (the data alone cannot fix the "
            "level). Recovered resistivity lands near this value. 0 = off; overrides Auto-calibrate.")
        self._max_iter = self._ispin(d["max_iterations"], 3, 200)
        form.addRow("Layers", self._n_layers)
        form.addRow("Min thickness (m)", self._min_thick)
        form.addRow("Max thickness (m)", self._max_thick)
        form.addRow("Smoothness", self._smooth)
        form.addRow("Relative error", self._rel_err)
        form.addRow("Data scale / calib.", self._data_scale)
        form.addRow("", self._auto_scale)
        form.addRow("Reference ρ (Ω·m)", self._ref_res)
        form.addRow("Max iterations", self._max_iter)

        # Line-only parameters, shown when the loaded file holds several soundings.
        self._line_rows = QWidget()
        lrow = QFormLayout(self._line_rows); lrow.setContentsMargins(0, 0, 0, 0)
        self._line_spacing = self._dspin(50.0, 0.1, 100000.0, 10.0, 2)
        self._line_spacing.setToolTip("Uniform sounding spacing used for the section's x-axis "
                                      "when no geometry file is loaded.")
        self._line_max = self._ispin(12, 1, 500)
        self._line_max.setToolTip("Cap on how many soundings to invert (keeps a long line fast).")
        lrow.addRow("Sounding spacing (m)", self._line_spacing)
        lrow.addRow("Max soundings", self._line_max)
        self._line_rows.setVisible(False)
        form.addRow(self._line_rows)

        self._inv_btn = QPushButton("Run inversion")
        self._inv_btn.setProperty("primary", True)
        self._inv_btn.setIcon(theme.icon("fa5s.bullseye", color="#ffffff"))
        self._inv_btn.clicked.connect(self._run_inversion)
        form.addRow(self._inv_btn)
        self._backend_label = QLabel()
        self._backend_label.setWordWrap(True)
        self._backend_label.setStyleSheet("font-size:8pt;")
        form.addRow("Backend", self._backend_label)
        self._inv_progress = QProgressBar(); self._inv_progress.setVisible(False)
        form.addRow(self._inv_progress)
        self._inv_export = QPushButton("Export recovered model (npy + csv)…")
        self._inv_export.setIcon(theme.icon("fa5s.file-export"))
        self._inv_export.setEnabled(False)
        self._inv_export.clicked.connect(self._export_inversion)
        form.addRow(self._inv_export)
        return box

    # -- method switch -------------------------------------------------------
    def _on_method_changed(self) -> None:
        fdem = self._method.currentText() == "FDEM"
        self._waveform.clear()
        self._waveform.addItems(["dipole", "loop"] if fdem else ["step_off", "ramp_off"])
        self._component_label.setVisible(fdem)
        self._component.setVisible(fdem)
        self._tx_rx_label.setVisible(fdem)
        self._tx_rx.setVisible(fdem)
        self._refresh_backend_state()

    def _refresh_backend_state(self) -> bool:
        """Reflect the method-specific SimPEG availability in the run controls."""
        status = em_pipeline.backend_status(self._method.currentText())
        available = bool(status["available"])
        if available:
            self._backend_label.setText("Ready: SimPEG backend available.")
            self._backend_label.setStyleSheet("color:#27734b; font-size:8pt;")
        else:
            self._backend_label.setText(
                "Unavailable: install the geophysics extra (SimPEG + discretize). "
                f"{status['error']}")
            self._backend_label.setStyleSheet("color:#9b5a00; font-size:8pt;")
        if not self._inv_progress.isVisible():
            self._inv_btn.setEnabled(available)
        return available

    # -- geometry / params ---------------------------------------------------
    def _collect_geom(self) -> Dict[str, Any]:
        geom: Dict[str, Any] = {
            "source_radius": self._src_radius.value(),
            "height": self._height.value(),
            "orientation": self._orient.currentText(),
            "waveform": self._waveform.currentText(),
        }
        if self._method.currentText() == "FDEM":
            geom.update(tx_rx_sep=self._tx_rx.value(), component=self._component.currentText())
        return geom

    def _collect_inv(self) -> Dict[str, Any]:
        return {"n_layers": self._n_layers.value(), "min_thickness": self._min_thick.value(),
                "max_thickness": self._max_thick.value(), "smoothness": self._smooth.value(),
                "rel_error": self._rel_err.value(), "max_iterations": self._max_iter.value(),
                "data_scale": self._data_scale.value(), "starting_resistivity": 100.0}

    # -- data ----------------------------------------------------------------
    def _load(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load sounding", "", _FILE_FILTER)
        if not path:
            return
        self._example_id = None
        self._example_note.setVisible(False)
        self._source_path = Path(path)
        self._sounding.blockSignals(True)
        self._sounding.setValue(1)
        self._sounding.blockSignals(False)
        self._load_sounding(0)

    def _load_example(self, example_id: str) -> Dict[str, Any]:
        """Load a documented demo dataset and apply its compatible settings."""
        catalog = em_pipeline.example_catalog()
        spec = catalog.get(example_id)
        if spec is None:
            return {"status": "failed", "error": f"Unknown EM example '{example_id}'.",
                    "valid_examples": sorted(catalog)}
        path = Path(spec["path"])
        if not path.exists():
            return {"status": "failed", "error": f"Example file not found: {path}"}
        self._method.setCurrentText(str(spec["method"]))
        self._agent_set_params(dict(spec.get("params", {})))
        self._example_id = example_id
        self._source_path = path
        self._sounding.blockSignals(True)
        self._sounding.setValue(1)
        self._sounding.blockSignals(False)
        self._load_sounding(0)
        if self._data is None:
            return {"status": "failed", "error": f"Could not load example '{example_id}'."}
        self._example_note.setText(f"Example: {spec['note']}")
        self._example_note.setVisible(True)
        channels = self._data.get("frequencies", self._data.get("times"))
        self.log(f"Loaded EM example: {spec['label']}", "success")
        return {
            "status": "ok", "example": example_id, "method": self._method.currentText(),
            "channels": int(np.asarray(channels).size),
            "n_soundings": int(self._data.get("n_soundings", 1)),
            "note": spec["note"],
        }

    def _load_sounding(self, index: int) -> None:
        if self._source_path is None:
            return
        try:
            self._data = em_pipeline.load_sounding(
                str(self._source_path), self._method.currentText(), sounding=int(index))
        except Exception as exc:  # noqa: BLE001
            self.log(f"Could not load sounding: {exc}", "error")
            self._info.setText(f"Load failed: {exc}")
            return
        n = self._data["frequencies"].size if "frequencies" in self._data else self._data["times"].size
        n_snd = int(self._data.get("n_soundings", 1))
        self._sounding.blockSignals(True)
        self._sounding.setRange(1, max(1, n_snd))
        self._sounding.setValue(int(self._data.get("sounding", 0)) + 1)
        self._sounding.blockSignals(False)
        self._sounding_row.setVisible(n_snd > 1)
        self._line_rows.setVisible(n_snd > 1)
        self._geom_row.setVisible(n_snd > 1)
        # A new file invalidates any previously loaded per-sounding geometry.
        self._geom_positions = None; self._geom_heights = None
        self._geom_x = None; self._geom_y = None
        self._geom_info.setText("Geometry: uniform spacing.")
        if n_snd > 1:
            self._maybe_auto_geometry()
        snd_txt = f" · sounding {self._data.get('sounding', 0) + 1}/{n_snd}" if n_snd > 1 else ""
        kind = "line of soundings" if n_snd > 1 else "single sounding"
        self._info.setText(f"{self._source_path.name}<br>{n} channels "
                           f"({self._method.currentText()}, {kind}){snd_txt}")
        self.log(f"Loaded sounding {self._source_path.name}{snd_txt}", "success")
        self._plot_data()

    def _on_sounding_changed(self, value: int) -> None:
        self._load_sounding(int(value) - 1)

    def _maybe_auto_geometry(self) -> None:
        """Auto-load a companion geometry file (so UTM coordinates are used without
        a manual step). Prefers ``<stem>_geometry.csv`` / ``<stem>_geom.csv``; else a
        single ``*geom*.csv`` beside the data file. Silent if none/ambiguous."""
        if self._source_path is None:
            return
        d, stem = self._source_path.parent, self._source_path.stem
        found = next((c for c in (d / f"{stem}_geometry.csv", d / f"{stem}_geom.csv")
                      if c.exists()), None)
        if found is None:
            others = [p for p in d.glob("*geom*.csv")
                      if p.resolve() != self._source_path.resolve()]
            found = others[0] if len(others) == 1 else None
        if found is None:
            return
        try:
            g = em_pipeline.load_line_geometry(str(found))
        except Exception:  # noqa: BLE001 - a bad companion file just means manual load
            return
        self._geom_positions = g["positions"]
        self._geom_heights = g["heights"] if g["has_heights"] else None
        self._geom_x = g.get("x"); self._geom_y = g.get("y")
        extra = (" + height" if g["has_heights"] else "") + (" + map x,y" if g.get("has_xy") else "")
        self._geom_info.setText(f"Geometry (auto): {found.name}, {g['n']} soundings{extra}.")
        self.log(f"Auto-loaded companion geometry {found.name}{extra}.", "info")

    def _load_geometry(self) -> None:
        if self._source_path is None:
            self.log("Load the sounding file first.", "warn")
            return
        path, _ = QFileDialog.getOpenFileName(self, "Load line geometry", "", _FILE_FILTER)
        if not path:
            return
        try:
            g = em_pipeline.load_line_geometry(path)
        except Exception as exc:  # noqa: BLE001
            self.log(f"Could not load geometry: {exc}", "error")
            return
        self._geom_positions = g["positions"]
        self._geom_heights = g["heights"] if g["has_heights"] else None
        self._geom_x = g.get("x"); self._geom_y = g.get("y")
        span = float(g["positions"][-1] - g["positions"][0]) if g["n"] else 0.0
        extra = ""
        if g["has_heights"]:
            extra += " + height"
        if g.get("has_xy"):
            extra += " + map x,y"
        self._geom_info.setText(f"Geometry: {g['n']} soundings, {span:.0f} m span{extra}.")
        self.log(f"Loaded line geometry: {g['n']} positions{extra}.", "success")

    def _plot_data(self) -> None:
        if self._data is None:
            return
        self._curve.clear_curves()
        if self._method.currentText() == "FDEM" and "frequencies" in self._data:
            x = self._data["frequencies"]
            self._curve.add_curve(x, self._data["real"], name="obs real")
            self._curve.add_curve(x, self._data["imag"], name="obs imag")
        elif "times" in self._data:
            self._curve.add_curve(self._data["times"], self._data["response"], name="obs")
        self._curve.set_log_x(True)
        self._tabs.setCurrentWidget(self._curve)

    # -- inversion (one button; single sounding -> profile, line -> section) --
    def _run_inversion(self) -> None:
        if self._data is None:
            self.log("Load a sounding first.", "warn")
            return
        method = self._method.currentText()
        status = em_pipeline.backend_status(method)
        if not status["available"]:
            self.log(f"{method} inversion unavailable: {status['error']}", "warn")
            self._refresh_backend_state()
            return
        n_snd = int(self._data.get("n_soundings", 1))
        ref = float(self._ref_res.value())
        # Reference-resistivity calibration for a single sounding, or the rough
        # auto-calibration, happen up front. For a line with a reference, invert_line
        # calibrates inside the worker (off-thread) and reports the value it used.
        if self._source_path is not None:
            try:
                if ref > 0 and n_snd == 1:
                    self.log(f"Calibrating to reference {ref:.0f} Ω·m…", "info")
                    k = em_pipeline.calibrate_to_reference(
                        str(self._source_path), method, self._collect_geom(),
                        self._collect_inv(), ref, log=lambda m: self.log(m, "info"))
                    self._data_scale.setValue(float(k))
                elif ref <= 0 and self._auto_scale.isChecked():
                    k = em_pipeline.estimate_data_scale(str(self._source_path), method,
                                                        self._collect_geom(), log=lambda m: self.log(m, "info"))
                    self._data_scale.setValue(float(k))
            except Exception as exc:  # noqa: BLE001
                self.log(f"Calibration failed ({exc}); using data_scale="
                         f"{self._data_scale.value():g}", "warn")
        self._inv_btn.setEnabled(False); self._inv_btn.setText("Inverting…")
        self._inv_progress.setVisible(True); self._inv_progress.setRange(0, 0)
        if n_snd > 1:
            self._start_line(method)
        else:
            self._start_single(method)

    def _start_single(self, method: str) -> None:
        self.log(f"Starting {method} 1D inversion…", "info")
        self._inv_worker = InversionWorker(method, self._data, self._collect_geom(), self._collect_inv())
        self._inv_worker.logged.connect(lambda m: self.log(m, "info"))
        self._inv_worker.succeeded.connect(self._on_inversion_ok)
        self._inv_worker.failed.connect(self._on_inversion_failed)
        self._inv_worker.finished.connect(self._reset_inv_button)
        self.register_worker(self._inv_worker)
        self._inv_worker.start()

    def _start_line(self, method: str) -> None:
        out_dir = str(io_utils.ensure_dir(Path(self.state.output_dir or ".") / "em_results"))
        self.log(f"Starting {method} line inversion (up to {self._line_max.value()} soundings)…", "info")
        worker = TaskWorker(
            em_pipeline.invert_line, str(self._source_path), method,
            self._collect_geom(), self._collect_inv(), with_log=True,
            spacing=float(self._line_spacing.value()), positions=self._geom_positions,
            heights=self._geom_heights, max_soundings=int(self._line_max.value()),
            ref_resistivity=float(self._ref_res.value()), out_dir=Path(out_dir))
        worker.logged.connect(lambda m: self.log(m, "info"))
        worker.succeeded.connect(self._on_line_ok)
        worker.failed.connect(self._on_line_failed)
        worker.finished.connect(self._reset_inv_button)
        self._line_worker = self.register_worker(worker)
        worker.start()

    def _reset_inv_button(self) -> None:
        self._inv_btn.setText("Run inversion")
        self._inv_progress.setVisible(False)
        self._refresh_backend_state()

    def _on_inversion_ok(self, result: dict) -> None:
        self._last_result = result
        self._inv_export.setEnabled(True)
        png = self._render_inversion(result)
        if png:
            self._inv_view.set_image_file(png)
            self._view_row.setVisible(False)
            self._model_stack.setCurrentWidget(self._inv_view)
            self._tabs.setCurrentWidget(self._model_tab)
        self._quality_view.show_quality(
            {"chi2": float(result["chi2"]), "n_data": result.get("n_data"),
             "method": f"{result['method']} 1D Occam",
              "extra": {"layers": result.get("n_layers"), "forward evals": result.get("nfev"),
                        "iterations": len(result.get("convergence") or [])},
             "note": "1D least-squares inversion (χ² is the mean weighted squared residual)."},
            convergence=result.get("convergence"), title=f"{result['method']} inversion")
        self.log(f"{result['method']} inversion complete (chi2={result['chi2']:.3f}).", "success")
        self.report_result({"method": result["method"], "chi2": float(result["chi2"]),
                            "n_data": result.get("n_data"), "nfev": result.get("nfev"),
                            "n_layers": int(np.asarray(result["resistivity"]).size)})

    def _on_line_ok(self, result: dict) -> None:
        self._last_section = result
        if "data_scale" in result:  # show the value the worker actually used (calibrated)
            self._data_scale.setValue(float(result["data_scale"]))
        self._section_view.show_model(result["edges"], result["model3d"],
                                      label=result["label"], cmap=result["cmap"],
                                      log_scale=result.get("log_scale", True))
        self._populate_plan(result)
        self._view_row.setVisible(True)
        self._view_mode.blockSignals(True); self._view_mode.setCurrentIndex(0)
        self._view_mode.blockSignals(False)
        self._model_stack.setCurrentWidget(self._plan_view)  # default to the plan-view map
        self._tabs.setCurrentWidget(self._model_tab)
        rng = result.get("model_range", [float("nan"), float("nan")])
        chi2 = result.get("chi2")
        chi_txt = f", mean chi2={chi2:.2f}" if isinstance(chi2, float) and chi2 == chi2 else ""
        self.log(f"{result['method']} line inversion complete: {result['n_soundings']} soundings, "
                 f"resistivity {rng[0]:.3g}..{rng[1]:.3g} Ω·m{chi_txt}.", "success")
        self._quality_view.show_quality(
            {"chi2": float(chi2) if isinstance(chi2, float) else float("nan"),
             "n_data": result.get("n_data"),
             "method": f"{result['method']} line (stitched 1D Occam)",
             "extra": {"soundings": result.get("n_soundings"), "layers": result.get("n_layers")},
             "note": "Mean χ² over the soundings on the line (each inverted independently in 1D)."},
            convergence=None, title=f"{result['method']} line inversion")
        saved = result.get("saved") or []
        for path in saved:
            self.log(f"Saved section to {path}", "info")
        self.report_result({"method": result["method"], "n_soundings": result.get("n_soundings"),
                            "mean_chi2": chi2, "section_npz": saved[0] if saved else None})

    def _populate_plan(self, result: dict) -> None:
        """Feed the plan-view depth-slice map from a line-inversion result: each
        sounding's map coordinate + its resistivity per depth layer."""
        model = np.asarray(result["model3d"], dtype=float)[:, 0, :]  # (n_pos, n_layers), deepest-first
        n_pos = model.shape[0]
        depth_edges = np.asarray(result["depth_edges"], dtype=float)
        depth_ctr = 0.5 * (depth_edges[:-1] + depth_edges[1:])       # surface-ordered
        res_surface = model[:, ::-1]                                 # surface-ordered in depth
        if (self._geom_x is not None and self._geom_y is not None
                and self._geom_x.size >= n_pos and self._geom_y.size >= n_pos):
            xy = np.column_stack([self._geom_x[:n_pos], self._geom_y[:n_pos]])
            x_label, y_label = "Easting (m)", "Northing (m)"
        else:  # no map coordinates loaded: lay soundings along the distance axis
            pos = np.asarray(result["positions"], dtype=float)[:n_pos]
            xy = np.column_stack([pos, np.zeros_like(pos)])
            x_label, y_label = "Distance along line (m)", ""
        self._plan_view.show_slices(xy, res_surface, depth_ctr,
                                    label=result.get("label", "resistivity (Ω·m)"),
                                    log_scale=result.get("log_scale", True),
                                    x_label=x_label, y_label=y_label)

    def _on_inversion_failed(self, message: str, backend: bool) -> None:
        self.log(f"Inversion {'unavailable' if backend else 'failed'}: {message}",
                 "warn" if backend else "error")

    def _on_line_failed(self, message: str) -> None:
        if any(k in message.lower() for k in ("backend", "simpeg", "discretize")):
            self.log(f"Line inversion needs SimPEG: {message}", "warn")
        else:
            self.log(f"Line inversion failed: {message}", "error")

    def _render_inversion(self, result: dict) -> Optional[str]:
        try:
            import matplotlib
            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.4))
            ax1.plot(result["resistivity_step"], result["depth"], "-", color="tab:red")
            ax1.invert_yaxis(); ax1.set_xscale("log")
            ax1.set_xlabel("Resistivity (Ω·m)"); ax1.set_ylabel("Depth (m)")
            ax1.set_title("Recovered model"); ax1.grid(True, which="both", alpha=0.3)
            if result["method"] == "FDEM":
                f = result["frequencies"]
                ax2.plot(f, result["obs_real"], "o", ms=4, color="tab:blue", label="obs real")
                ax2.plot(f, result["pred_real"], "-", color="tab:blue", label="pred real")
                ax2.plot(f, result["obs_imag"], "s", ms=4, color="tab:green", label="obs imag")
                ax2.plot(f, result["pred_imag"], "-", color="tab:green", label="pred imag")
                ax2.set_xlabel("Frequency (Hz)")
            else:
                t = result["times"]
                ax2.plot(t, result["obs"], "o", ms=4, color="tab:blue", label="obs")
                ax2.plot(t, result["pred"], "-", color="tab:blue", label="pred")
                ax2.set_xlabel("Time (s)")
            ax2.set_xscale("log"); ax2.set_title(f"Data fit (chi2={result['chi2']:.2f})")
            ax2.grid(True, which="both", alpha=0.3); ax2.legend(fontsize=8, frameon=False)
            fig.tight_layout()
            out = io_utils.ensure_dir(Path(self.state.output_dir or ".") / "em_results")
            p = out / f"{result['method'].lower()}_inversion.png"
            fig.savefig(p, dpi=160, bbox_inches="tight"); plt.close(fig)
            return str(p)
        except Exception as exc:  # noqa: BLE001
            self.log(f"Could not render inversion: {exc}", "warn")
            return None

    def _export_inversion(self) -> None:
        if not self._last_result:
            self.log("Run a single-sounding inversion first (a line auto-saves its section).", "warn")
            return
        folder = QFileDialog.getExistingDirectory(self, "Export recovered model to folder",
                                                  str(self.state.output_dir or Path.cwd()))
        if not folder:
            return
        paths = em_pipeline.save_inversion(self._last_result, Path(folder))
        self.log(f"Exported recovered model ({len(paths)} files) to {folder}", "success")

    def _show_format_help(self) -> None:
        doc_path = Path(__file__).with_name("em_input_format.md")
        try:
            text = doc_path.read_text(encoding="utf-8")
        except Exception:  # noqa: BLE001
            text = "EM sounding — FDEM: frequency, real, imag · TDEM: time, response."
        dlg = QDialog(self); dlg.setWindowTitle("EM sounding input format")
        dlg.resize(720, 600); lay = QVBoxLayout(dlg)
        browser = QTextBrowser(); browser.setOpenExternalLinks(True)
        try:
            browser.setMarkdown(text)
        except Exception:  # noqa: BLE001
            browser.setPlainText(text)
        lay.addWidget(browser)
        close = QPushButton("Close"); close.clicked.connect(dlg.accept); lay.addWidget(close)
        dlg.exec()

    # -- AQUAH agent interface ----------------------------------------------
    def agent_describe(self) -> Dict[str, Any]:
        return {
            "module": self.module_key,
            "title": self.module_title,
            "state": self._agent_status(),
            "actions": [
                {"name": "use_example_data",
                 "args": {"example": ["east_river_vtem", "skytem_bhmar", "synthetic_fdem"]},
                 "desc": ("Load a documented EM example. Default is east_river_vtem; "
                          "synthetic_fdem is the reproducible FDEM demo.")},
                {"name": "set_method", "args": {"method": list(em_pipeline.METHODS)},
                 "desc": "Choose the EM method (FDEM or TDEM)."},
                {"name": "load_data", "args": {"path": "str", "sounding": "int (optional, 1-based)"},
                 "desc": ("Load a sounding file for the current method. If the file stacks several "
                          "soundings side by side (a survey line), 'sounding' picks which one to "
                          "preview (default 1); the inversion still uses the whole line.")},
                {"name": "set_params", "args": {"params": {"<key>": "value"}},
                 "desc": ("Set parameters. Geometry: source_radius, tx_rx_sep, height, orientation "
                          "(z/x/y), component (secondary/total/both), waveform. Inversion: n_layers, "
                          "min_thickness, max_thickness, smoothness, rel_error, data_scale "
                          "(calibration multiplier), auto_scale (bool, rough), ref_resistivity "
                          "(ohm-m; calibrate the absolute level to a known value, the reliable "
                          "option), max_iterations. Line: spacing, max_soundings.")},
                {"name": "auto_calibrate", "args": {},
                 "desc": ("Estimate the data_scale calibration from the loaded data and set it. Use "
                          "for normalized airborne data (e.g. moment-normalized dB/dt).")},
                {"name": "load_geometry", "args": {"path": "str"},
                 "desc": ("Load per-sounding line geometry (a file with along-line position and "
                          "optional sensor height) so the section uses real distances instead of "
                          "uniform spacing.")},
                {"name": "run_inversion", "args": {},
                 "desc": ("Run the inversion. A single-sounding file gives a 1D layered model; a "
                          "multi-sounding file is inverted sounding-by-sounding into a position x "
                          "depth resistivity section. If auto-calibrate is on, data_scale is "
                          "estimated first.")},
                {"name": "get_status", "args": {},
                 "desc": "Report the method, loaded data, and last result."},
            ],
        }

    def agent_apply(self, action: str, args: Dict[str, Any]) -> Dict[str, Any]:
        args = args or {}
        handlers = {
            "use_example_data": lambda: self._load_example(args.get("example", "east_river_vtem")),
            "set_method": lambda: self._agent_set_method(args.get("method")),
            "load_data": lambda: self._agent_load(args.get("path"), args.get("sounding", 1)),
            "set_params": lambda: self._agent_set_params(args.get("params", args)),
            "auto_calibrate": lambda: self._agent_auto_calibrate(),
            "load_geometry": lambda: self._agent_load_geometry(args.get("path")),
            "run_inversion": lambda: self._agent_run_inversion(),
            "get_status": lambda: self._agent_status(),
        }
        handler = handlers.get(action)
        if handler is None:
            return {"status": "failed", "error": f"Unknown action '{action}'.",
                    "valid_actions": list(handlers.keys())}
        return handler()

    def _agent_status(self) -> Dict[str, Any]:
        last = self.state.module_results.get(self.module_key, {})
        n_snd = int(self._data.get("n_soundings", 1)) if self._data else 0
        return {
            "status": "ok",
            "method": self._method.currentText(),
            "data_loaded": self._data is not None,
            "source": str(self._source_path or ""),
            "example": self._example_id,
            "n_soundings": n_snd,
            "backend": em_pipeline.backend_status(self._method.currentText()),
            "last_result_keys": sorted(last.keys()),
        }

    def _agent_set_method(self, method: Any) -> Dict[str, Any]:
        methods = list(em_pipeline.METHODS)
        if method not in methods:
            return {"status": "failed", "error": f"Unknown method '{method}'.", "valid": methods}
        self._method.setCurrentText(method)
        return {"status": "ok", "method": self._method.currentText()}

    def _agent_load(self, path: Any, sounding: Any = 1) -> Dict[str, Any]:
        if not path:
            return {"status": "failed", "error": "Provide 'path' to a sounding file."}
        p = Path(str(path))
        if not p.exists():
            return {"status": "failed", "error": f"File not found: {p}"}
        self._example_id = None
        self._example_note.setVisible(False)
        self._source_path = Path(p)
        try:
            self._load_sounding(max(0, int(sounding) - 1))
        except Exception as exc:  # noqa: BLE001
            return {"status": "failed", "error": f"Could not load sounding: {exc}"}
        if self._data is None:
            return {"status": "failed", "error": "Could not load sounding (wrong method/format?)."}
        arr = self._data.get("frequencies", self._data.get("times"))
        n = int(arr.size) if arr is not None else 0
        return {"status": "ok", "channels": n, "method": self._method.currentText(),
                "n_soundings": int(self._data.get("n_soundings", 1)),
                "sounding": int(self._data.get("sounding", 0)) + 1}

    def _agent_set_params(self, params: Any) -> Dict[str, Any]:
        if not isinstance(params, dict):
            return {"status": "failed", "error": "Provide 'params' as a JSON object."}

        def set_combo(combo, value):
            items = [combo.itemText(i) for i in range(combo.count())]
            if str(value) not in items:
                raise ValueError(f"must be one of {items}")
            combo.setCurrentText(str(value))

        handlers = {
            "source_radius": lambda v: self._src_radius.setValue(float(v)),
            "tx_rx_sep": lambda v: self._tx_rx.setValue(float(v)),
            "height": lambda v: self._height.setValue(float(v)),
            "orientation": lambda v: set_combo(self._orient, v),
            "component": lambda v: set_combo(self._component, v),
            "waveform": lambda v: set_combo(self._waveform, v),
            "n_layers": lambda v: self._n_layers.setValue(int(v)),
            "min_thickness": lambda v: self._min_thick.setValue(float(v)),
            "max_thickness": lambda v: self._max_thick.setValue(float(v)),
            "smoothness": lambda v: self._smooth.setValue(float(v)),
            "rel_error": lambda v: self._rel_err.setValue(float(v)),
            "data_scale": lambda v: self._data_scale.setValue(float(v)),
            "auto_scale": lambda v: self._auto_scale.setChecked(bool(v)),
            "ref_resistivity": lambda v: self._ref_res.setValue(float(v)),
            "max_iterations": lambda v: self._max_iter.setValue(int(v)),
            "spacing": lambda v: self._line_spacing.setValue(float(v)),
            "max_soundings": lambda v: self._line_max.setValue(int(v)),
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

    def _agent_run_inversion(self) -> Dict[str, Any]:
        if self._data is None:
            return {"status": "failed", "error": "Load a sounding first."}
        backend = em_pipeline.backend_status(self._method.currentText())
        if not backend["available"]:
            self._refresh_backend_state()
            return {"status": "failed", "error": f"EM backend unavailable: {backend['error']}",
                    "backend": backend}
        n_snd = int(self._data.get("n_soundings", 1))
        self._run_inversion()
        return {"status": "started",
                "message": ("EM %s inversion started. Ask for status shortly."
                            % ("line" if n_snd > 1 else "1D")),
                "method": self._method.currentText(), "n_soundings": n_snd}

    def _agent_auto_calibrate(self) -> Dict[str, Any]:
        if self._data is None or self._source_path is None:
            return {"status": "failed", "error": "Load a sounding first."}
        try:
            k = em_pipeline.estimate_data_scale(str(self._source_path), self._method.currentText(),
                                                self._collect_geom(), log=lambda m: self.log(m, "info"))
        except Exception as exc:  # noqa: BLE001
            return {"status": "failed", "error": f"Auto-calibration failed: {exc}"}
        self._data_scale.setValue(float(k))
        return {"status": "ok", "data_scale": float(k)}

    def _agent_load_geometry(self, path: Any) -> Dict[str, Any]:
        if not path:
            return {"status": "failed", "error": "Provide 'path' to a geometry file."}
        p = Path(str(path))
        if not p.exists():
            return {"status": "failed", "error": f"File not found: {p}"}
        try:
            g = em_pipeline.load_line_geometry(str(p))
        except Exception as exc:  # noqa: BLE001
            return {"status": "failed", "error": f"Could not load geometry: {exc}"}
        self._geom_positions = g["positions"]
        self._geom_heights = g["heights"] if g["has_heights"] else None
        self._geom_x = g.get("x"); self._geom_y = g.get("y")
        extra = (" + height" if g["has_heights"] else "") + (" + map x,y" if g.get("has_xy") else "")
        span = float(g["positions"][-1] - g["positions"][0]) if g["n"] else 0.0
        self._geom_info.setText(f"Geometry: {g['n']} soundings, {span:.0f} m span{extra}.")
        return {"status": "ok", "n": g["n"], "has_heights": g["has_heights"],
                "has_xy": bool(g.get("has_xy"))}
