"""EM module: 1D FDEM / TDEM forward modeling and inversion.

Load a sounding (FDEM: frequency, real, imag; TDEM: time, response), build a
layered resistivity model and compute its forward response, and invert the
sounding for a layered model (Occam-style 1D fit). The numerics live in the
Qt-free ``PyHydroGeophysX.qt_apps.em_pipeline`` (a thin wrapper over the package's
SimPEG forward operators). Results export to npy / csv / json.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PySide6.QtCore import QThread, Signal
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
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from PyHydroGeophysX.qt_apps import em_pipeline, io_utils, theme
from PyHydroGeophysX.qt_apps.modules.base import BaseModule, LogFn
from PyHydroGeophysX.qt_apps.widgets.curve_viewer import CurveViewer
from PyHydroGeophysX.qt_apps.widgets.image_view import ZoomableImageView

_FILE_FILTER = "Sounding (*.csv *.txt *.dat);;All files (*)"


class InversionWorker(QThread):
    """Runs a 1D EM inversion off the UI thread."""

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
        self._inv_worker: Optional[InversionWorker] = None

        root = QHBoxLayout(self)
        self._tabs = QTabWidget()
        self._curve = CurveViewer()
        self._inv_view = ZoomableImageView()
        self._tabs.addTab(self._curve, "Sounding")
        self._tabs.addTab(self._inv_view, "Inversion")
        root.addWidget(self._tabs, stretch=1)
        root.addWidget(self._build_controls())
        self._on_method_changed()

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
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setMaximumWidth(330)
        panel = QWidget(); scroll.setWidget(panel)
        layout = QVBoxLayout(panel)

        self._method = QComboBox(); self._method.addItems(list(em_pipeline.METHODS))
        self._method.currentTextChanged.connect(self._on_method_changed)
        mrow = QFormLayout(); mrow.addRow("Method", self._method)
        layout.addLayout(mrow)

        row = QHBoxLayout()
        load_btn = QPushButton("Load sounding…")
        load_btn.setIcon(theme.icon("fa5s.folder-open"))
        load_btn.clicked.connect(self._load)
        fmt_btn = QPushButton("Data format")
        fmt_btn.setIcon(theme.icon("fa5s.file-alt"))
        fmt_btn.clicked.connect(self._show_format_help)
        row.addWidget(load_btn); row.addWidget(fmt_btn)
        layout.addLayout(row)
        self._info = QLabel("No sounding loaded. FDEM: freq, real, imag · TDEM: time, response.")
        self._info.setWordWrap(True)
        layout.addWidget(self._info)

        layout.addWidget(self._build_geometry_group())
        layout.addWidget(self._build_model_group())
        layout.addWidget(self._build_inversion_group())

        exp = QPushButton("Export config JSON…")
        exp.setIcon(theme.icon("fa5s.file-export"))
        exp.clicked.connect(self._export_config)
        layout.addWidget(exp)
        layout.addStretch(1)
        return scroll

    def _build_geometry_group(self) -> QGroupBox:
        box = QGroupBox("Survey geometry"); form = QFormLayout(box)
        f = em_pipeline.DEFAULT_FDEM
        self._x_min = self._dspin(f["freq_min"], 1e-3, 1e7, 10.0, 3)
        self._x_max = self._dspin(f["freq_max"], 1.0, 1e8, 100.0, 1)
        self._n_x = self._ispin(f["n_freq"], 4, 200)
        self._src_radius = self._dspin(f["source_radius"], 0.5, 200.0, 1.0, 1)
        self._tx_rx = self._dspin(f["tx_rx_sep"], 0.0, 500.0, 1.0, 1)
        self._height = self._dspin(f["height"], 0.0, 500.0, 1.0, 1)
        self._orient = QComboBox(); self._orient.addItems(["z", "x", "y"])
        self._component = QComboBox(); self._component.addItems(["secondary", "total", "both"])
        self._waveform = QComboBox()
        self._xlabel = QLabel("Frequencies (min / max / n)")
        form.addRow(self._xlabel, self._x_min)
        form.addRow("", self._x_max)
        form.addRow("", self._n_x)
        form.addRow("Source radius (m)", self._src_radius)
        self._tx_rx_row = self._tx_rx
        form.addRow("Tx-Rx sep (m)", self._tx_rx)
        form.addRow("Height (m)", self._height)
        form.addRow("Orientation", self._orient)
        self._component_label = QLabel("Component")
        form.addRow(self._component_label, self._component)
        form.addRow("Waveform", self._waveform)
        return box

    def _build_model_group(self) -> QGroupBox:
        box = QGroupBox("Layered model (forward)"); v = QVBoxLayout(box)
        self._model_table = QTableWidget(0, 2)
        self._model_table.setHorizontalHeaderLabels(["Thickness (m)", "Resistivity (Ω·m)"])
        self._model_table.horizontalHeader().setStretchLastSection(True)
        self._model_table.setMaximumHeight(150)
        v.addWidget(self._model_table)
        self._set_model(em_pipeline.DEFAULT_MODEL)
        row = QHBoxLayout()
        add = QPushButton("Add layer"); add.clicked.connect(self._add_layer)
        rem = QPushButton("Remove layer"); rem.clicked.connect(self._remove_layer)
        row.addWidget(add); row.addWidget(rem)
        v.addLayout(row)
        fwd = QPushButton("Compute forward response")
        fwd.setIcon(theme.icon("fa5s.play"))
        fwd.clicked.connect(self._compute_forward)
        v.addWidget(fwd)
        note = QLabel("Last row is the half-space (thickness ignored).")
        note.setStyleSheet("color:#5a6a7a; font-size:8pt;"); note.setWordWrap(True)
        v.addWidget(note)
        return box

    def _build_inversion_group(self) -> QGroupBox:
        box = QGroupBox("Inversion (1D Occam)"); form = QFormLayout(box)
        d = em_pipeline.DEFAULT_INVERSION
        self._n_layers = self._ispin(d["n_layers"], 3, 60)
        self._min_thick = self._dspin(d["min_thickness"], 0.2, 50.0, 0.5, 2)
        self._max_thick = self._dspin(d["max_thickness"], 1.0, 500.0, 1.0, 1)
        self._smooth = self._dspin(d["smoothness"], 0.0, 10.0, 0.1, 2)
        self._rel_err = self._dspin(d["rel_error"], 0.0, 1.0, 0.01, 3)
        self._max_iter = self._ispin(d["max_iterations"], 3, 200)
        form.addRow("Layers", self._n_layers)
        form.addRow("Min thickness (m)", self._min_thick)
        form.addRow("Max thickness (m)", self._max_thick)
        form.addRow("Smoothness", self._smooth)
        form.addRow("Relative error", self._rel_err)
        form.addRow("Max iterations", self._max_iter)
        self._inv_btn = QPushButton("Run inversion")
        self._inv_btn.setProperty("primary", True)
        self._inv_btn.setIcon(theme.icon("fa5s.bullseye", color="#ffffff"))
        self._inv_btn.clicked.connect(self._run_inversion)
        form.addRow(self._inv_btn)
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
        method = self._method.currentText()
        fdem = method == "FDEM"
        if fdem:
            d = em_pipeline.DEFAULT_FDEM
            self._xlabel.setText("Frequencies (min / max / n)")
            self._waveform.clear(); self._waveform.addItems(["dipole", "loop"])
        else:
            d = em_pipeline.DEFAULT_TDEM
            self._xlabel.setText("Times (min / max / n)")
            self._waveform.clear(); self._waveform.addItems(["step_off", "ramp_off"])
        self._x_min.setValue(d.get("freq_min", d.get("t_min", 1e-5)))
        self._x_max.setValue(d.get("freq_max", d.get("t_max", 1e-2)))
        self._n_x.setValue(d.get("n_freq", d.get("n_times", 16)))
        self._component_label.setVisible(fdem)
        self._component.setVisible(fdem)
        self._tx_rx.setEnabled(fdem)

    # -- model table ---------------------------------------------------------
    def _set_model(self, model: Dict[str, Any]) -> None:
        thick = list(model.get("thickness", []))
        res = list(model.get("resistivity", []))
        self._model_table.setRowCount(len(res))
        for i in range(len(res)):
            t_item = QTableWidgetItem("—" if i == len(res) - 1 else f"{thick[i]:g}")
            self._model_table.setItem(i, 0, t_item)
            self._model_table.setItem(i, 1, QTableWidgetItem(f"{res[i]:g}"))

    def _read_model(self) -> Dict[str, List[float]]:
        n = self._model_table.rowCount()
        thick, res = [], []
        for i in range(n):
            r_item = self._model_table.item(i, 1)
            res.append(float(r_item.text()) if r_item else 100.0)
            if i < n - 1:
                t_item = self._model_table.item(i, 0)
                try:
                    thick.append(float(t_item.text()))
                except Exception:  # noqa: BLE001
                    thick.append(10.0)
        return {"thickness": thick, "resistivity": res}

    def _add_layer(self) -> None:
        n = self._model_table.rowCount()
        self._model_table.insertRow(n)
        # The previous half-space becomes a real layer; give it a thickness.
        if n >= 1:
            self._model_table.setItem(n - 1, 0, QTableWidgetItem("10"))
        self._model_table.setItem(n, 0, QTableWidgetItem("—"))
        self._model_table.setItem(n, 1, QTableWidgetItem("100"))

    def _remove_layer(self) -> None:
        n = self._model_table.rowCount()
        if n > 2:
            self._model_table.removeRow(n - 1)
            self._model_table.setItem(n - 2, 0, QTableWidgetItem("—"))

    # -- geometry ------------------------------------------------------------
    def _collect_geom(self) -> Dict[str, Any]:
        method = self._method.currentText()
        geom: Dict[str, Any] = {
            "source_radius": self._src_radius.value(),
            "height": self._height.value(),
            "orientation": self._orient.currentText(),
            "waveform": self._waveform.currentText(),
        }
        if method == "FDEM":
            geom.update(freq_min=self._x_min.value(), freq_max=self._x_max.value(),
                        n_freq=self._n_x.value(), tx_rx_sep=self._tx_rx.value(),
                        component=self._component.currentText())
        else:
            geom.update(t_min=self._x_min.value(), t_max=self._x_max.value(),
                        n_times=self._n_x.value())
        return geom

    def _collect_inv(self) -> Dict[str, Any]:
        return {"n_layers": self._n_layers.value(), "min_thickness": self._min_thick.value(),
                "max_thickness": self._max_thick.value(), "smoothness": self._smooth.value(),
                "rel_error": self._rel_err.value(), "max_iterations": self._max_iter.value(),
                "starting_resistivity": 100.0}

    # -- data ----------------------------------------------------------------
    def _load(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load sounding", "", _FILE_FILTER)
        if not path:
            return
        try:
            self._data = em_pipeline.load_sounding(path, self._method.currentText())
        except Exception as exc:  # noqa: BLE001
            self.log(f"Could not load sounding: {exc}", "error")
            self._info.setText(f"Load failed: {exc}")
            return
        self._source_path = Path(path)
        n = self._data["frequencies"].size if "frequencies" in self._data else self._data["times"].size
        self._info.setText(f"{self._source_path.name}<br>{n} channels ({self._method.currentText()})")
        self.log(f"Loaded sounding {self._source_path.name}", "success")
        self._plot_data()

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

    # -- forward -------------------------------------------------------------
    def _compute_forward(self) -> None:
        method = self._method.currentText()
        try:
            model = self._read_model()
            geom = self._collect_geom()
            fn = em_pipeline.fdem_forward if method == "FDEM" else em_pipeline.tdem_forward
            resp = fn(model, geom, log=lambda m: self.log(m, "info"))
        except em_pipeline.BackendUnavailable as exc:
            self.log(f"SimPEG is needed for EM forward modeling: {exc}", "warn")
            return
        except Exception as exc:  # noqa: BLE001
            self.log(f"Forward modeling failed: {exc}", "error")
            return
        if method == "FDEM":
            x = resp["frequencies"]
            if self._data is None:
                self._curve.clear_curves()
            self._curve.add_curve(x, resp["real"], name="model real")
            self._curve.add_curve(x, resp["imag"], name="model imag")
        else:
            x = resp["times"]
            if self._data is None:
                self._curve.clear_curves()
            self._curve.add_curve(x, resp["response"], name="model")
        self._curve.set_log_x(True)
        self._tabs.setCurrentWidget(self._curve)
        self.log(f"{method} forward computed ({x.size} channels).", "success")
        self.report_result({"method": method, "forward_channels": int(x.size)})

    # -- inversion -----------------------------------------------------------
    def _run_inversion(self) -> None:
        if self._data is None:
            self.log("Load a sounding first.", "warn")
            return
        method = self._method.currentText()
        self._inv_btn.setEnabled(False); self._inv_btn.setText("Inverting…")
        self._inv_progress.setVisible(True); self._inv_progress.setRange(0, 0)
        self.log(f"Starting {method} 1D inversion…", "info")
        self._inv_worker = InversionWorker(method, self._data, self._collect_geom(), self._collect_inv())
        self._inv_worker.logged.connect(lambda m: self.log(m, "info"))
        self._inv_worker.succeeded.connect(self._on_inversion_ok)
        self._inv_worker.failed.connect(self._on_inversion_failed)
        self._inv_worker.finished.connect(self._reset_inv_button)
        self.register_worker(self._inv_worker)
        self._inv_worker.start()

    def _on_inversion_ok(self, result: dict) -> None:
        self._last_result = result
        self._inv_export.setEnabled(True)
        png = self._render_inversion(result)
        if png:
            self._inv_view.set_image_file(png)
            self._tabs.setCurrentWidget(self._inv_view)
        self.log(f"{result['method']} inversion complete (chi2={result['chi2']:.3f}).", "success")
        self.report_result({"method": result["method"], "chi2": float(result["chi2"]),
                            "n_layers": int(np.asarray(result["resistivity"]).size)})

    def _on_inversion_failed(self, message: str, backend: bool) -> None:
        self.log(f"Inversion {'unavailable' if backend else 'failed'}: {message}",
                 "warn" if backend else "error")

    def _reset_inv_button(self) -> None:
        self._inv_btn.setEnabled(True); self._inv_btn.setText("Run inversion")
        self._inv_progress.setVisible(False)

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
            self.log("Run an inversion first.", "warn")
            return
        folder = QFileDialog.getExistingDirectory(self, "Export recovered model to folder",
                                                  str(self.state.output_dir or Path.cwd()))
        if not folder:
            return
        paths = em_pipeline.save_inversion(self._last_result, Path(folder))
        self.log(f"Exported recovered model ({len(paths)} files) to {folder}", "success")

    def _export_config(self) -> None:
        cfg = em_pipeline.build_em_config(self._method.currentText(), self._read_model(),
                                          self._collect_geom(), self._collect_inv())
        path, _ = QFileDialog.getSaveFileName(self, "Export EM config", "em_config.json", "JSON (*.json)")
        if not path:
            return
        io_utils.write_json(path, cfg)
        self.log(f"Exported EM config to {path}", "success")

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
                {"name": "set_method", "args": {"method": list(em_pipeline.METHODS)},
                 "desc": "Choose the EM method (FDEM or TDEM)."},
                {"name": "load_data", "args": {"path": "str"},
                 "desc": "Load a sounding file for the current method."},
                {"name": "set_params", "args": {"params": {"<key>": "value"}},
                 "desc": ("Set parameters. Geometry: x_min, x_max, n_channels, source_radius, "
                          "tx_rx_sep, height, orientation (z/x/y), component (secondary/total/both), "
                          "waveform. Inversion: n_layers, min_thickness, max_thickness, smoothness, "
                          "rel_error, max_iterations.")},
                {"name": "compute_forward", "args": {},
                 "desc": "Compute the forward EM response of the layered model."},
                {"name": "run_inversion", "args": {},
                 "desc": "Run the 1D Occam inversion on the loaded sounding."},
                {"name": "get_status", "args": {},
                 "desc": "Report the method, loaded data, and last result."},
            ],
        }

    def agent_apply(self, action: str, args: Dict[str, Any]) -> Dict[str, Any]:
        args = args or {}
        handlers = {
            "set_method": lambda: self._agent_set_method(args.get("method")),
            "load_data": lambda: self._agent_load(args.get("path")),
            "set_params": lambda: self._agent_set_params(args.get("params", args)),
            "compute_forward": lambda: self._agent_compute_forward(),
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
        return {
            "status": "ok",
            "method": self._method.currentText(),
            "data_loaded": self._data is not None,
            "source": str(self._source_path or ""),
            "model_layers": self._model_table.rowCount(),
            "last_result_keys": sorted(last.keys()),
        }

    def _agent_set_method(self, method: Any) -> Dict[str, Any]:
        methods = list(em_pipeline.METHODS)
        if method not in methods:
            return {"status": "failed", "error": f"Unknown method '{method}'.", "valid": methods}
        self._method.setCurrentText(method)
        return {"status": "ok", "method": self._method.currentText()}

    def _agent_load(self, path: Any) -> Dict[str, Any]:
        if not path:
            return {"status": "failed", "error": "Provide 'path' to a sounding file."}
        p = Path(str(path))
        if not p.exists():
            return {"status": "failed", "error": f"File not found: {p}"}
        try:
            self._data = em_pipeline.load_sounding(str(p), self._method.currentText())
        except Exception as exc:  # noqa: BLE001
            return {"status": "failed", "error": f"Could not load sounding: {exc}"}
        self._source_path = Path(p)
        arr = self._data.get("frequencies", self._data.get("times"))
        n = int(arr.size) if arr is not None else 0
        self._info.setText(f"{self._source_path.name}<br>{n} channels ({self._method.currentText()})")
        self._plot_data()
        return {"status": "ok", "channels": n, "method": self._method.currentText()}

    def _agent_set_params(self, params: Any) -> Dict[str, Any]:
        if not isinstance(params, dict):
            return {"status": "failed", "error": "Provide 'params' as a JSON object."}

        def set_combo(combo, value):
            items = [combo.itemText(i) for i in range(combo.count())]
            if str(value) not in items:
                raise ValueError(f"must be one of {items}")
            combo.setCurrentText(str(value))

        handlers = {
            "x_min": lambda v: self._x_min.setValue(float(v)),
            "x_max": lambda v: self._x_max.setValue(float(v)),
            "n_channels": lambda v: self._n_x.setValue(int(v)),
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
            "max_iterations": lambda v: self._max_iter.setValue(int(v)),
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

    def _agent_compute_forward(self) -> Dict[str, Any]:
        self._compute_forward()
        return {"status": "ok", "message": "Forward response computed.",
                "method": self._method.currentText()}

    def _agent_run_inversion(self) -> Dict[str, Any]:
        if self._data is None:
            return {"status": "failed", "error": "Load a sounding first."}
        self._run_inversion()
        return {"status": "started", "message": "EM inversion started. Ask for status shortly.",
                "method": self._method.currentText()}
