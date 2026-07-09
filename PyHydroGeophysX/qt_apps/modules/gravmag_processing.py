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
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from PySide6.QtCore import QRectF, Qt

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
        self._z: Optional[np.ndarray] = None
        self._fields: Dict[str, np.ndarray] = {}
        self._source_path: Optional[Path] = None
        self._inv_result: Optional[Dict[str, Any]] = None
        self._inv_worker: Optional[TaskWorker] = None
        self._cmap = pg.colormap.get("viridis")
        self._qc: Optional[Dict[str, Any]] = None
        self._qc_picks = []

        root = QHBoxLayout(self)
        self._tabs = QTabWidget()
        self._plot_widget = pg.PlotWidget(); self._plot_widget.setBackground("w")
        self._plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self._plot_widget.setLabel("bottom", "x"); self._plot_widget.setLabel("left", "y")
        self._scatter = pg.ScatterPlotItem(size=13)
        self._plot_widget.getPlotItem().addItem(self._scatter)
        self._qc_tab = self._build_qc_tab()
        self._model_view = Model3DView()
        self._quality_view = InversionQualityView()
        self._tabs.addTab(self._plot_widget, "Station map")
        self._tabs.addTab(self._qc_tab, "Data QC")
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

        hint = QLabel("Upload x, y, value and optionally z_m (positive-up station elevation). "
                       "Gravity is mGal; magnetics is nT. Use Data QC before inversion.")
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
        self._station_z = self._dspin(1.0, -10000.0, 100000.0, 1.0, 1)
        self._station_z.setToolTip("Used only when the loaded table has no fourth z_m column.")
        self._rel_error = self._dspin(0.03, 0.0, 1.0, 0.01, 3)
        self._noise_floor = self._dspin(0.5, 0.0, 1e6, 0.1, 3)
        self._max_stations = self._ispin(600, 20, 50000)
        self._max_stations.setToolTip(
            "Maximum stations passed to SimPEG. Larger files are reduced with deterministic "
            "spatially balanced sampling rather than file-row order.")
        self._elevation_source = QLabel("Using global station elevation.")
        self._elevation_source.setWordWrap(True)
        self._elevation_source.setStyleSheet("color:#5a6a7a; font-size:8pt;")
        form.addRow("Detrend degree", self._detrend)
        form.addRow("Station elevation (m)", self._station_z)
        form.addRow("Elevation source", self._elevation_source)
        form.addRow("Relative error", self._rel_error)
        form.addRow("Noise floor", self._noise_floor)
        form.addRow("Max stations", self._max_stations)
        form.addRow("Lateral cells", self._inv_nxy)
        form.addRow("Depth cells", self._inv_nz)
        form.addRow("Max iterations", self._inv_iter)
        self._inv_btn = QPushButton("Run 3D inversion")
        self._inv_btn.setProperty("primary", True)
        self._inv_btn.setIcon(theme.icon("fa5s.cubes", color="#ffffff"))
        self._inv_btn.clicked.connect(self._run_inversion)
        form.addRow(self._inv_btn)
        self._backend_label = QLabel()
        self._backend_label.setWordWrap(True)
        self._backend_label.setStyleSheet("font-size:8pt;")
        form.addRow("Backend", self._backend_label)
        self._inv_progress = QProgressBar(); self._inv_progress.setVisible(False)
        form.addRow(self._inv_progress)
        self._detrend.valueChanged.connect(lambda _value: self._refresh_qc())
        return box

    # -- kind ----------------------------------------------------------------
    def _on_kind_changed(self) -> None:
        self._field_box.setVisible(self._kind.currentText() == "magnetics")
        if hasattr(self, "_noise_floor"):
            self._noise_floor.setValue(2.0 if self._kind.currentText() == "magnetics" else 0.5)
        if hasattr(self, "_qc_field"):
            self._refresh_qc()
        if hasattr(self, "_backend_label"):
            self._refresh_backend_state()

    # -- data QC -------------------------------------------------------------
    def _build_qc_tab(self) -> QWidget:
        """Build the gridded observed/regional/residual map and profile picker."""
        page = QWidget(); layout = QVBoxLayout(page)
        row = QHBoxLayout()
        row.addWidget(QLabel("Map field:"))
        self._qc_field = QComboBox(); self._qc_field.addItems(["Observed", "Regional", "Residual"])
        self._qc_field.currentTextChanged.connect(lambda _name: self._draw_qc())
        row.addWidget(self._qc_field)
        self._profile_pick = QCheckBox("Pick profile endpoints")
        self._profile_pick.setToolTip("Click two points on the map to extract a profile.")
        self._profile_pick.toggled.connect(lambda checked: self._clear_qc_profile() if not checked else None)
        row.addWidget(self._profile_pick)
        reset = QPushButton("Reset profile")
        reset.clicked.connect(self._clear_qc_profile)
        row.addWidget(reset); row.addStretch(1)
        layout.addLayout(row)

        self._qc_graphics = pg.GraphicsLayoutWidget()
        self._qc_plot = self._qc_graphics.addPlot(row=0, col=0)
        self._qc_plot.setLabel("bottom", "x (m)")
        self._qc_plot.setLabel("left", "y (m)")
        self._qc_plot.showGrid(x=True, y=True, alpha=0.25)
        self._qc_image = pg.ImageItem()
        self._qc_image.setColorMap(self._cmap)
        self._qc_plot.addItem(self._qc_image)
        self._qc_stations = pg.ScatterPlotItem(size=4, pen=pg.mkPen("#222", width=0.5),
                                                brush=pg.mkBrush(255, 255, 255, 120))
        self._qc_picks_item = pg.ScatterPlotItem(size=11, symbol="o", pen=pg.mkPen("#b51f1f", width=2),
                                                  brush=pg.mkBrush(255, 255, 255, 180))
        self._qc_plot.addItem(self._qc_stations); self._qc_plot.addItem(self._qc_picks_item)
        self._qc_hist = pg.HistogramLUTItem()
        self._qc_hist.setImageItem(self._qc_image)
        self._qc_graphics.addItem(self._qc_hist, row=0, col=1)
        layout.addWidget(self._qc_graphics, stretch=3)

        self._qc_stats = QLabel("Load station data to calculate QC maps.")
        self._qc_stats.setWordWrap(True)
        layout.addWidget(self._qc_stats)
        self._profile_plot = pg.PlotWidget(); self._profile_plot.setBackground("w")
        self._profile_plot.showGrid(x=True, y=True, alpha=0.3)
        self._profile_plot.setLabel("bottom", "Distance (m)")
        self._profile_plot.setLabel("left", "Anomaly")
        self._profile_plot.setMinimumHeight(180)
        layout.addWidget(self._profile_plot, stretch=1)
        self._qc_plot.scene().sigMouseClicked.connect(self._on_qc_map_click)
        return page

    def _refresh_qc(self) -> None:
        vals = self._fields.get("Observed")
        if vals is None or self._x is None or self._y is None:
            return
        try:
            self._qc = gmp.qc_products(self._x, self._y, vals, detrend=self._detrend.value())
        except Exception as exc:  # noqa: BLE001
            self._qc = None
            self._qc_stats.setText(f"QC unavailable: {exc}")
            return
        self._clear_qc_profile()
        self._draw_qc(auto_range=True)

    def _draw_qc(self, auto_range: bool = False) -> None:
        if self._qc is None:
            return
        field_name = self._qc_field.currentText()
        grid = self._qc["grids"][field_name]
        xx, yy, zz = grid["xx"], grid["yy"], grid["zz"]
        self._qc_image.setImage(np.asarray(zz, dtype=float).T, autoLevels=True)
        self._qc_image.setRect(QRectF(float(xx.min()), float(yy.min()),
                                      max(float(np.ptp(xx)), 1.0), max(float(np.ptp(yy)), 1.0)))
        self._qc_stations.setData(self._qc["x"], self._qc["y"])
        stats = self._qc["stats"][field_name]
        unit = "mGal" if self._kind.currentText() == "gravity" else "nT"
        self._qc_plot.setTitle(f"{field_name} anomaly ({unit})")
        self._profile_plot.setLabel("left", f"Anomaly ({unit})")
        self._qc_stats.setText(
            f"Degree-{self._qc['detrend']} detrend · {field_name}: "
            f"min {stats['min']:.4g}, max {stats['max']:.4g}, mean {stats['mean']:.4g}, "
            f"std {stats['std']:.4g} {unit}.")
        if auto_range:
            self._qc_plot.autoRange()

    def _clear_qc_profile(self) -> None:
        self._qc_picks = []
        if hasattr(self, "_qc_picks_item"):
            self._qc_picks_item.setData([], [])
        if hasattr(self, "_profile_plot"):
            self._profile_plot.clear()

    def _on_qc_map_click(self, event) -> None:
        if not self._profile_pick.isChecked() or self._qc is None:
            return
        if event.button() != Qt.LeftButton or not self._qc_plot.sceneBoundingRect().contains(event.scenePos()):
            return
        point = self._qc_plot.vb.mapSceneToView(event.scenePos())
        if len(self._qc_picks) >= 2:
            self._clear_qc_profile()
        self._qc_picks.append((float(point.x()), float(point.y())))
        self._qc_picks_item.setData([p[0] for p in self._qc_picks], [p[1] for p in self._qc_picks])
        if len(self._qc_picks) == 2:
            self._update_qc_profile()

    def _update_qc_profile(self) -> Optional[Dict[str, np.ndarray]]:
        if self._qc is None or len(self._qc_picks) != 2:
            return None
        profile = gmp.extract_profile(self._qc["grids"][self._qc_field.currentText()],
                                      self._qc_picks[0], self._qc_picks[1])
        self._profile_plot.clear()
        self._profile_plot.plot(profile["distance"], profile["value"], pen=pg.mkPen("#1f77b4", width=2))
        return profile

    def _refresh_backend_state(self) -> bool:
        status = gmp.backend_status()
        available = bool(status["available"])
        if available:
            self._backend_label.setText("Ready: SimPEG potential-field backend available.")
            self._backend_label.setStyleSheet("color:#27734b; font-size:8pt;")
        else:
            self._backend_label.setText(
                "Unavailable: install the geophysics extra (SimPEG + pymatsolver). "
                f"{status['error']}")
            self._backend_label.setStyleSheet("color:#9b5a00; font-size:8pt;")
        if not self._inv_progress.isVisible():
            self._inv_btn.setEnabled(available)
        return available

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
        self._set_station_data(table, Path(path))

    def _load_example(self) -> Dict[str, Any]:
        """Load the gravity or magnetic demo selected by the current field type."""
        root = Path(__file__).resolve().parents[3] / "examples" / "data" / "Gravity_Magnetics"
        kind = self._kind.currentText()
        path = root / ("bushveld_gravity_disturbance.csv" if kind == "gravity"
                       else "britain_aeromagnetic_anomaly.csv")
        if not path.exists():
            return {"status": "failed", "error": f"Example file not found: {path}"}
        try:
            table = io_utils.load_xyz_table(path, min_cols=3)
        except Exception as exc:  # noqa: BLE001
            return {"status": "failed", "error": f"Could not load example: {exc}"}
        self._set_station_data(table, path)
        self.log(f"Loaded {kind} example: {path.name}", "success")
        return {"status": "ok", "example": kind, "stations": int(table.shape[0]),
                "path": str(path)}

    def _set_station_data(self, table: np.ndarray, path: Path) -> None:
        """Store a three- or four-column station table and update all previews."""
        self._x = np.asarray(table[:, 0], dtype=float)
        self._y = np.asarray(table[:, 1], dtype=float)
        self._z = np.asarray(table[:, 3], dtype=float) if table.shape[1] >= 4 else None
        self._fields = {"Observed": np.asarray(table[:, 2], dtype=float)}
        self._source_path = Path(path)
        elevation = "per-station z_m" if self._z is not None else "global elevation fallback"
        self._elevation_source.setText(f"Using {elevation}.")
        self._info.setText(f"{self._source_path.name}<br>{self._x.size} stations · {elevation}")
        self.log(f"Loaded {self._x.size} stations from {self._source_path.name}", "success")
        self._refresh_scatter()
        self._refresh_qc()
        self._publish()

    def _effective_z(self) -> np.ndarray:
        """Return per-station elevation, preferring the optional z_m column."""
        if self._x is None:
            return np.asarray([], dtype=float)
        if self._z is not None:
            return np.asarray(self._z, dtype=float)
        return np.full(self._x.size, self._station_z.value(), dtype=float)

    # -- 3D inversion --------------------------------------------------------
    def _run_inversion(self) -> None:
        vals = self._fields.get("Observed")
        if vals is None or self._x is None:
            self.log("Load station data first.", "warn")
            return
        status = gmp.backend_status()
        if not status["available"]:
            self.log(f"3D inversion unavailable: {status['error']}", "warn")
            self._refresh_backend_state()
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
                 f"cap {self._max_stations.value()}, detrend {self._detrend.value()})…", "info")
        worker = TaskWorker(gmp.invert_gravmag, self._x, self._y, vals, kind,
                             z=self._effective_z(), field=field, detrend=self._detrend.value(),
                             n_xy=self._inv_nxy.value(), n_z=self._inv_nz.value(),
                             max_iterations=self._inv_iter.value(), max_stations=self._max_stations.value(),
                             relative_error=self._rel_error.value(), noise_floor=self._noise_floor.value(),
                             out_dir=out_dir,
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
              "extra": {"cells": result.get("n_cells"), "stations used": result.get("n_data"),
                         "detrend degree": self._detrend.value(),
                         "iterations": len(result.get("convergence") or [])},
             "note": "3D potential-field inversion (χ² is the normalized data misfit)."},
            convergence=result.get("convergence"), title="Gravity/Magnetics inversion")
        vtk = result.get("vtk")
        if vtk:
            self.log(f"Saved model VTK to {vtk}", "info")
        self.report_result({"inversion_kind": result["kind"], "chi2": chi2,
                            "model_vtk": vtk, "n_cells": result["n_cells"]})

    def _on_inversion_failed(self, message: str) -> None:
        if ("backend" in message.lower() or "simpeg" in message.lower()
                or "discretize" in message.lower() or "pymatsolver" in message.lower()):
            self.log(f"SimPEG 3D inversion needs SimPEG + pymatsolver: {message}", "warn")
        else:
            self.log(f"3D inversion failed: {message}", "error")

    def _reset_inv_button(self) -> None:
        self._inv_btn.setText("Run 3D inversion")
        self._inv_progress.setVisible(False)
        self._refresh_backend_state()

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
                {"name": "use_example_data", "args": {"kind": ["gravity", "magnetics"]},
                 "desc": "Load the bundled gravity or magnetic example (default: current field type)."},
                {"name": "load_data", "args": {"path": "str"},
                 "desc": "Load station data (x, y, value, optional z_m) for gravity or magnetics."},
                {"name": "set_field_type", "args": {"kind": ["gravity", "magnetics"]},
                 "desc": "Set the field type (gravity or magnetics)."},
                {"name": "set_params", "args": {"params": {"<key>": "value"}},
                  "desc": ("Set parameters. Inversion: detrend (0..3), lateral_cells, depth_cells, "
                           "max_iterations, station_elevation, relative_error, noise_floor, max_stations. "
                           "Magnetics ambient field: field_strength, field_inclination, field_declination.")},
                {"name": "run_qc", "args": {},
                 "desc": "Calculate observed/regional/residual QC products and report their statistics."},
                {"name": "extract_profile", "args": {"point1": "[x, y]", "point2": "[x, y]"},
                 "desc": "Extract a profile through the current QC map between two map points."},
                {"name": "run_forward_bodies", "args": {"bodies": "list[object]"},
                 "desc": "Agent-only analytic forward model: gravity sphere/prism or magnetic sphere bodies."},
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
            "use_example_data": lambda: self._agent_use_example(args.get("kind")),
            "load_data": lambda: self._agent_load(args.get("path")),
            "set_field_type": lambda: self._agent_set_field_type(args.get("kind")),
            "set_params": lambda: self._agent_set_params(args.get("params", args)),
            "run_qc": lambda: self._agent_run_qc(),
            "extract_profile": lambda: self._agent_extract_profile(args.get("point1"), args.get("point2")),
            "run_forward_bodies": lambda: self._agent_forward_bodies(args.get("bodies")),
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
            "has_station_elevation": self._z is not None,
            "backend": gmp.backend_status(),
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
        self._set_station_data(table, Path(p))
        return {"status": "ok", "stations": int(self._x.size), "has_station_elevation": self._z is not None}

    def _agent_use_example(self, kind: Any) -> Dict[str, Any]:
        selected = self._kind.currentText() if kind in (None, "") else str(kind)
        if selected not in ("gravity", "magnetics"):
            return {"status": "failed", "error": "kind must be 'gravity' or 'magnetics'."}
        self._kind.setCurrentText(selected)
        return self._load_example()

    def _agent_set_field_type(self, kind: Any) -> Dict[str, Any]:
        if kind not in ("gravity", "magnetics"):
            return {"status": "failed", "error": "kind must be 'gravity' or 'magnetics'."}
        self._kind.setCurrentText(kind)
        return {"status": "ok", "kind": self._kind.currentText()}

    def _agent_run_inversion(self) -> Dict[str, Any]:
        if self._x is None or self._fields.get("Observed") is None:
            return {"status": "failed", "error": "Load station data first."}
        backend = gmp.backend_status()
        if not backend["available"]:
            self._refresh_backend_state()
            return {"status": "failed", "error": f"Gravity/magnetics backend unavailable: {backend['error']}",
                    "backend": backend}
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
            "station_elevation": lambda v: self._station_z.setValue(float(v)),
            "relative_error": lambda v: self._rel_error.setValue(float(v)),
            "noise_floor": lambda v: self._noise_floor.setValue(float(v)),
            "max_stations": lambda v: self._max_stations.setValue(int(v)),
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

    def _agent_run_qc(self) -> Dict[str, Any]:
        if self._x is None or self._fields.get("Observed") is None:
            return {"status": "failed", "error": "Load station data first."}
        self._refresh_qc()
        if self._qc is None:
            return {"status": "failed", "error": "Could not calculate QC products."}
        return {"status": "ok", "detrend": self._qc["detrend"], "stats": self._qc["stats"]}

    def _agent_extract_profile(self, point1: Any, point2: Any) -> Dict[str, Any]:
        if self._x is None or self._fields.get("Observed") is None:
            return {"status": "failed", "error": "Load station data first."}
        try:
            p1 = np.asarray(point1, dtype=float).ravel()
            p2 = np.asarray(point2, dtype=float).ravel()
            if p1.size != 2 or p2.size != 2:
                raise ValueError("point1 and point2 must each be [x, y].")
            self._refresh_qc()
            if self._qc is None:
                raise ValueError("QC products are unavailable.")
            profile = gmp.extract_profile(self._qc["grids"][self._qc_field.currentText()], p1, p2)
        except Exception as exc:  # noqa: BLE001
            return {"status": "failed", "error": str(exc)}
        finite = np.asarray(profile["value"], dtype=float)
        return {"status": "ok", "n_samples": int(finite.size),
                "min": float(np.nanmin(finite)), "max": float(np.nanmax(finite))}

    def _agent_forward_bodies(self, bodies: Any) -> Dict[str, Any]:
        if self._x is None or self._y is None:
            return {"status": "failed", "error": "Load station data first."}
        if not isinstance(bodies, list) or not all(isinstance(body, dict) for body in bodies):
            return {"status": "failed", "error": "bodies must be a list of body objects."}
        field = None
        if self._kind.currentText() == "magnetics":
            field = {"strength": self._B0.value(), "inclination": self._inc.value(),
                     "declination": self._dec.value()}
        try:
            anomaly = gmp.forward_bodies(self._x, self._y, self._kind.currentText(), bodies, field=field)
        except Exception as exc:  # noqa: BLE001
            return {"status": "failed", "error": str(exc)}
        return {"status": "ok", "kind": self._kind.currentText(), "bodies": len(bodies),
                "min": float(np.nanmin(anomaly)), "max": float(np.nanmax(anomaly))}

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
