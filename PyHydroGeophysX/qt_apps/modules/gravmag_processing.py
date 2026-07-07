"""Gravity / Magnetics module: 3D potential-field inversion.

Load station data (x, y, value) and run a SimPEG 3D inversion for a subsurface
model: gravity → density contrast (g/cc), magnetics → susceptibility (SI). A
polynomial regional trend is removed first (the "Detrend degree" control) so a raw
field is inverted on its residual anomaly. The result is shown as a 3D model you
can slice at different depths and positions. The numerics live in the Qt-free
``PyHydroGeophysX.qt_apps.gravmag_pipeline``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import (
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
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from PySide6.QtCore import Qt

from PyHydroGeophysX.qt_apps import gravmag_pipeline as gmp
from PyHydroGeophysX.qt_apps import io_utils, theme
from PyHydroGeophysX.qt_apps.modules.base import BaseModule, LogFn
from PyHydroGeophysX.qt_apps.widgets.model3d_view import Model3DView
from PyHydroGeophysX.qt_apps.widgets.quality_view import InversionQualityView
from PyHydroGeophysX.qt_apps.workers import TaskWorker

_FILE_FILTER = "Station data (*.csv *.txt *.dat);;All files (*)"


class GravMagProcessingModule(BaseModule):
    module_key = "gravmag_processing"
    module_title = "Gravity / Magnetics"

    def __init__(self, state: Any, log: LogFn, parent=None) -> None:
        super().__init__(state, log, parent)
        self._x: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None
        self._fields: Dict[str, np.ndarray] = {}
        self._source_path: Optional[Path] = None
        self._inv_result: Optional[Dict[str, Any]] = None
        self._inv_worker: Optional[TaskWorker] = None
        self._cmap = pg.colormap.get("viridis")

        root = QHBoxLayout(self)
        self._tabs = QTabWidget()
        self._plot_widget = pg.PlotWidget(); self._plot_widget.setBackground("w")
        self._plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self._plot_widget.setLabel("bottom", "x"); self._plot_widget.setLabel("left", "y")
        self._scatter = pg.ScatterPlotItem(size=13)
        self._plot_widget.getPlotItem().addItem(self._scatter)
        self._model_view = Model3DView()
        self._quality_view = InversionQualityView()
        self._tabs.addTab(self._plot_widget, "Station map")
        self._tabs.addTab(self._model_view, "Inversion model")
        self._tabs.addTab(self._quality_view, "Inversion quality")
        root.addWidget(self._tabs, stretch=1)
        root.addWidget(self._build_controls())
        self._on_kind_changed()

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
        layout.addWidget(self._build_field_group())
        layout.addWidget(self._build_inversion_group())
        layout.addStretch(1)
        return scroll

    def _build_loader_group(self) -> QGroupBox:
        box = QGroupBox("Load station data"); v = QVBoxLayout(box)
        row = QHBoxLayout()
        load_btn = QPushButton("Load stations…")
        load_btn.setProperty("primary", True)
        load_btn.setIcon(theme.icon("fa5s.folder-open", color="#ffffff"))
        load_btn.clicked.connect(self._load)
        fmt_btn = QPushButton("Data format")
        fmt_btn.setIcon(theme.icon("fa5s.file-alt"))
        fmt_btn.clicked.connect(self._show_format_help)
        row.addWidget(load_btn); row.addWidget(fmt_btn)
        v.addLayout(row)

        krow = QFormLayout()
        self._kind = QComboBox(); self._kind.addItems(["gravity", "magnetics"])
        self._kind.currentTextChanged.connect(self._on_kind_changed)
        krow.addRow("Field type", self._kind)
        v.addLayout(krow)

        hint = QLabel("Upload a table with three columns: x, y, value. Gravity in mGal, "
                      "magnetics in nT. The map colours the uploaded value at each station.")
        hint.setWordWrap(True); hint.setStyleSheet("color:#5a6a7a; font-size:8pt;")
        v.addWidget(hint)
        self._info = QLabel("No data loaded."); self._info.setWordWrap(True)
        v.addWidget(self._info)
        return box

    def _build_field_group(self) -> QGroupBox:
        # Ambient field for magnetics; hidden in gravity mode by _on_kind_changed.
        self._field_box = QGroupBox("Ambient field (magnetics)")
        ff = QFormLayout(self._field_box)
        self._B0 = self._dspin(50000.0, 100.0, 100000.0, 100.0, 0)
        self._inc = self._dspin(60.0, -90.0, 90.0, 1.0, 1)
        self._dec = self._dspin(0.0, -180.0, 180.0, 1.0, 1)
        ff.addRow("Strength (nT)", self._B0)
        ff.addRow("Inclination (°)", self._inc)
        ff.addRow("Declination (°)", self._dec)
        return self._field_box

    def _build_inversion_group(self) -> QGroupBox:
        box = QGroupBox("3D inversion (SimPEG)"); form = QFormLayout(box)
        hint = QLabel("Inverts the uploaded field for a subsurface model: gravity → density "
                      "contrast (g/cc), magnetics → susceptibility (SI). Slice the result at "
                      "different depths and positions in the Inversion model tab.")
        hint.setWordWrap(True); hint.setStyleSheet("color:#5a6a7a; font-size:8pt;")
        form.addRow(hint)
        self._detrend = self._ispin(1, 0, 3)
        self._detrend.setToolTip("Remove a polynomial regional trend before inverting "
                                 "(0 = none, 1 = linear, 2 = quadratic). Inversion works on "
                                 "the residual anomaly.")
        self._inv_nxy = self._ispin(22, 8, 40)
        self._inv_nz = self._ispin(12, 4, 30)
        self._inv_iter = self._ispin(20, 3, 60)
        form.addRow("Detrend degree", self._detrend)
        form.addRow("Lateral cells", self._inv_nxy)
        form.addRow("Depth cells", self._inv_nz)
        form.addRow("Max iterations", self._inv_iter)
        self._inv_btn = QPushButton("Run 3D inversion")
        self._inv_btn.setProperty("primary", True)
        self._inv_btn.setIcon(theme.icon("fa5s.cubes", color="#ffffff"))
        self._inv_btn.clicked.connect(self._run_inversion)
        form.addRow(self._inv_btn)
        self._inv_progress = QProgressBar(); self._inv_progress.setVisible(False)
        form.addRow(self._inv_progress)
        return box

    # -- kind ----------------------------------------------------------------
    def _on_kind_changed(self) -> None:
        self._field_box.setVisible(self._kind.currentText() == "magnetics")

    # -- station map ---------------------------------------------------------
    def _refresh_scatter(self) -> None:
        vals = self._fields.get("Observed")
        if vals is None or self._x is None:
            return
        vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
        rng = vmax - vmin if vmax > vmin else 1.0
        norm = (vals - vmin) / rng
        lut = self._cmap.map(norm, mode="byte")
        spots = [{"pos": (float(self._x[i]), float(self._y[i])),
                  "brush": pg.mkBrush(int(lut[i, 0]), int(lut[i, 1]), int(lut[i, 2])),
                  "size": 13, "pen": pg.mkPen("#333333", width=0.5)}
                 for i in range(self._x.size)]
        self._scatter.setData(spots)
        self._tabs.setCurrentWidget(self._plot_widget)

    # -- data ----------------------------------------------------------------
    def _load(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load station file", "", _FILE_FILTER)
        if not path:
            return
        try:
            table = io_utils.load_xyz_table(path, min_cols=3)
        except Exception as exc:  # noqa: BLE001
            self.log(f"Could not load data: {exc}", "error")
            self._info.setText(f"Load failed: {exc}")
            return
        self._x = table[:, 0]; self._y = table[:, 1]
        self._fields = {"Observed": table[:, 2].astype(float)}
        self._source_path = Path(path)
        self._info.setText(f"{self._source_path.name}<br>{self._x.size} stations")
        self.log(f"Loaded {self._x.size} stations from {self._source_path.name}", "success")
        self._refresh_scatter()
        self._publish()

    # -- 3D inversion --------------------------------------------------------
    def _run_inversion(self) -> None:
        vals = self._fields.get("Observed")
        if vals is None or self._x is None:
            self.log("Load station data first.", "warn")
            return
        kind = self._kind.currentText()
        field = None
        if kind != "gravity":
            field = {"strength_nT": self._B0.value(), "inclination": self._inc.value(),
                     "declination": self._dec.value()}
        out_dir = str(self.state.output_dir or Path.cwd())
        self._inv_btn.setEnabled(False); self._inv_btn.setText("Inverting…")
        self._inv_progress.setVisible(True); self._inv_progress.setRange(0, 0)
        self.log(f"Running SimPEG 3D {kind} inversion ({self._x.size} stations, "
                 f"detrend {self._detrend.value()})…", "info")
        worker = TaskWorker(gmp.invert_gravmag, self._x, self._y, vals, kind,
                            field=field, detrend=self._detrend.value(),
                            n_xy=self._inv_nxy.value(), n_z=self._inv_nz.value(),
                            max_iterations=self._inv_iter.value(), out_dir=out_dir,
                            log=lambda _m: None)
        worker.succeeded.connect(self._on_inversion_ok)
        worker.failed.connect(self._on_inversion_failed)
        worker.finished.connect(self._reset_inv_button)
        self._inv_worker = self.register_worker(worker)
        worker.start()

    def _on_inversion_ok(self, result: dict) -> None:
        self._inv_result = result
        self._model_view.show_model(result["edges"], result["model3d"],
                                    label=result["label"], cmap=result["cmap"],
                                    log_scale=result.get("log_scale", False))
        self._tabs.setCurrentWidget(self._model_view)
        rng = result.get("model_range", [0.0, 0.0]); chi2 = result.get("chi2")
        chi_txt = f", chi2={chi2:.1f}" if isinstance(chi2, float) and chi2 == chi2 else ""
        self.log(f"3D {result['kind']} inversion complete: {result['n_cells']} cells, "
                 f"model {rng[0]:.3g}..{rng[1]:.3g}{chi_txt}.", "success")
        self._quality_view.show_quality(
            {"chi2": float(chi2) if isinstance(chi2, float) else float("nan"),
             "n_data": result.get("n_data"),
             "method": f"{result['kind']} 3D SimPEG",
             "extra": {"cells": result.get("n_cells"), "detrend degree": self._detrend.value()},
             "note": "3D potential-field inversion (χ² is the normalized data misfit)."},
            convergence=None, title="Gravity/Magnetics inversion")
        vtk = result.get("vtk")
        if vtk:
            self.log(f"Saved model VTK to {vtk}", "info")
        self.report_result({"inversion_kind": result["kind"], "chi2": chi2,
                            "model_vtk": vtk, "n_cells": result["n_cells"]})

    def _on_inversion_failed(self, message: str) -> None:
        if "backend" in message.lower() or "simpeg" in message.lower() or "discretize" in message.lower():
            self.log(f"SimPEG 3D inversion needs SimPEG + discretize: {message}", "warn")
        else:
            self.log(f"3D inversion failed: {message}", "error")

    def _reset_inv_button(self) -> None:
        self._inv_btn.setEnabled(True); self._inv_btn.setText("Run 3D inversion")
        self._inv_progress.setVisible(False)

    # -- publish -------------------------------------------------------------
    def _publish(self) -> None:
        self.report_result({
            "kind": self._kind.currentText(),
            "source_file": str(self._source_path) if self._source_path else "",
            "num_stations": int(self._x.size) if self._x is not None else 0,
        })

    # -- AQUAH agent interface ----------------------------------------------
    def agent_describe(self) -> Dict[str, Any]:
        return {
            "module": self.module_key,
            "title": self.module_title,
            "state": self._agent_status(),
            "actions": [
                {"name": "load_data", "args": {"path": "str"},
                 "desc": "Load a station file (x, y, value) for gravity or magnetics."},
                {"name": "set_field_type", "args": {"kind": ["gravity", "magnetics"]},
                 "desc": "Set the field type (gravity or magnetics)."},
                {"name": "set_params", "args": {"params": {"<key>": "value"}},
                 "desc": ("Set parameters. Inversion: detrend (0..3 regional-trend degree), "
                          "lateral_cells, depth_cells, max_iterations. Magnetics ambient field: "
                          "field_strength, field_inclination, field_declination.")},
                {"name": "run_inversion", "args": {},
                 "desc": ("Run a SimPEG 3D inversion of the uploaded field (gravity → density, "
                          "magnetics → susceptibility). A regional trend of the chosen degree is "
                          "removed first. Result shows in the Inversion model tab.")},
                {"name": "get_status", "args": {},
                 "desc": "Report loaded data and the field type."},
            ],
        }

    def agent_apply(self, action: str, args: Dict[str, Any]) -> Dict[str, Any]:
        args = args or {}
        handlers = {
            "load_data": lambda: self._agent_load(args.get("path")),
            "set_field_type": lambda: self._agent_set_field_type(args.get("kind")),
            "set_params": lambda: self._agent_set_params(args.get("params", args)),
            "run_inversion": lambda: self._agent_run_inversion(),
            "get_status": lambda: self._agent_status(),
        }
        handler = handlers.get(action)
        if handler is None:
            return {"status": "failed", "error": f"Unknown action '{action}'.",
                    "valid_actions": list(handlers.keys())}
        return handler()

    def _agent_status(self) -> Dict[str, Any]:
        return {
            "status": "ok",
            "kind": self._kind.currentText(),
            "data_loaded": self._x is not None,
            "source": str(self._source_path or ""),
            "stations": int(self._x.size) if self._x is not None else 0,
        }

    def _agent_load(self, path: Any) -> Dict[str, Any]:
        if not path:
            return {"status": "failed", "error": "Provide 'path' to a station file."}
        p = Path(str(path))
        if not p.exists():
            return {"status": "failed", "error": f"File not found: {p}"}
        try:
            table = io_utils.load_xyz_table(str(p), min_cols=3)
        except Exception as exc:  # noqa: BLE001
            return {"status": "failed", "error": f"Could not load data: {exc}"}
        self._x = table[:, 0]; self._y = table[:, 1]
        self._fields = {"Observed": table[:, 2].astype(float)}
        self._source_path = Path(p)
        self._info.setText(f"{self._source_path.name}<br>{self._x.size} stations")
        self._refresh_scatter()
        self._publish()
        return {"status": "ok", "stations": int(self._x.size)}

    def _agent_set_field_type(self, kind: Any) -> Dict[str, Any]:
        if kind not in ("gravity", "magnetics"):
            return {"status": "failed", "error": "kind must be 'gravity' or 'magnetics'."}
        self._kind.setCurrentText(kind)
        return {"status": "ok", "kind": self._kind.currentText()}

    def _agent_run_inversion(self) -> Dict[str, Any]:
        if self._x is None or self._fields.get("Observed") is None:
            return {"status": "failed", "error": "Load station data first."}
        self._run_inversion()
        return {"status": "started", "message": "SimPEG 3D inversion started. Ask for status shortly.",
                "kind": self._kind.currentText(), "detrend": self._detrend.value()}

    def _agent_set_params(self, params: Any) -> Dict[str, Any]:
        if not isinstance(params, dict):
            return {"status": "failed", "error": "Provide 'params' as a JSON object."}
        handlers = {
            "detrend": lambda v: self._detrend.setValue(int(v)),
            "lateral_cells": lambda v: self._inv_nxy.setValue(int(v)),
            "depth_cells": lambda v: self._inv_nz.setValue(int(v)),
            "max_iterations": lambda v: self._inv_iter.setValue(int(v)),
            "field_strength": lambda v: self._B0.setValue(float(v)),
            "field_inclination": lambda v: self._inc.setValue(float(v)),
            "field_declination": lambda v: self._dec.setValue(float(v)),
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

    def _show_format_help(self) -> None:
        doc_path = Path(__file__).with_name("gravmag_input_format.md")
        try:
            text = doc_path.read_text(encoding="utf-8")
        except Exception:  # noqa: BLE001
            text = "Station file: three columns x, y, value (gravity mGal / magnetics nT)."
        dlg = QDialog(self); dlg.setWindowTitle("Gravity / Magnetics input format")
        dlg.resize(720, 600); lay = QVBoxLayout(dlg)
        browser = QTextBrowser(); browser.setOpenExternalLinks(True)
        try:
            browser.setMarkdown(text)
        except Exception:  # noqa: BLE001
            browser.setPlainText(text)
        lay.addWidget(browser)
        close = QPushButton("Close"); close.clicked.connect(dlg.accept); lay.addWidget(close)
        dlg.exec()
