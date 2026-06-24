"""Gravity / Magnetics module: anomaly tools + analytic forward modeling.

Load station data (x, y, value), separate a polynomial regional from the
residual, grid the field and extract profiles, and forward-model buried bodies
(gravity: sphere / prism; magnetics: dipole) overlaid on the data. Grids and
profiles export to npy / CSV / VTK. The numerics live in the Qt-free
``PyHydroGeophysX.qt_apps.gravmag_pipeline``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

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
    QPushButton,
    QScrollArea,
    QSpinBox,
    QStackedWidget,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from PyHydroGeophysX.qt_apps import gravmag_pipeline as gmp
from PyHydroGeophysX.qt_apps import io_utils, theme
from PyHydroGeophysX.qt_apps.modules.base import BaseModule, LogFn
from PyHydroGeophysX.qt_apps.widgets.array_viewer import ArrayViewer

_FILE_FILTER = "Station/grid (*.csv *.txt *.dat);;All files (*)"
_DISPLAY = ["Observed", "Regional", "Residual", "Forward", "Obs − Forward"]


class GravMagProcessingModule(BaseModule):
    module_key = "gravmag_processing"
    module_title = "Gravity / Magnetics"

    def __init__(self, state: Any, log: LogFn, parent=None) -> None:
        super().__init__(state, log, parent)
        self._x: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None
        self._fields: Dict[str, np.ndarray] = {}
        self._grid: Optional[Dict[str, np.ndarray]] = None
        self._profile: Optional[Dict[str, np.ndarray]] = None
        self._source_path: Optional[Path] = None
        self._cmap = pg.colormap.get("viridis")

        root = QHBoxLayout(self)
        self._tabs = QTabWidget()
        self._plot_widget = pg.PlotWidget(); self._plot_widget.setBackground("w")
        self._plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self._plot_widget.setLabel("bottom", "x"); self._plot_widget.setLabel("left", "y")
        self._scatter = pg.ScatterPlotItem(size=13)
        self._plot_widget.getPlotItem().addItem(self._scatter)
        self._grid_view = ArrayViewer()
        self._profile_plot = pg.PlotWidget(); self._profile_plot.setBackground("w")
        self._profile_plot.showGrid(x=True, y=True, alpha=0.3)
        self._profile_plot.setLabel("bottom", "distance"); self._profile_plot.setLabel("left", "value")
        self._tabs.addTab(self._plot_widget, "Station map")
        self._tabs.addTab(self._grid_view, "Grid")
        self._tabs.addTab(self._profile_plot, "Profile")
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
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setMaximumWidth(330)
        panel = QWidget(); scroll.setWidget(panel)
        layout = QVBoxLayout(panel)

        load_btn = QPushButton("Load station file (x, y, value)…")
        load_btn.setIcon(theme.icon("fa5s.folder-open"))
        load_btn.clicked.connect(self._load)
        layout.addWidget(load_btn)
        fmt_btn = QPushButton("Data format")
        fmt_btn.setIcon(theme.icon("fa5s.file-alt"))
        fmt_btn.clicked.connect(self._show_format_help)
        layout.addWidget(fmt_btn)
        self._info = QLabel("No data loaded."); self._info.setWordWrap(True)
        layout.addWidget(self._info)

        top = QFormLayout()
        self._kind = QComboBox(); self._kind.addItems(["gravity", "magnetics"])
        self._kind.currentTextChanged.connect(self._on_kind_changed)
        self._display = QComboBox(); self._display.addItems(_DISPLAY)
        self._display.currentTextChanged.connect(self._refresh_scatter)
        top.addRow("Field type", self._kind)
        top.addRow("Display", self._display)
        layout.addLayout(top)

        layout.addWidget(self._build_separation_group())
        layout.addWidget(self._build_grid_group())
        layout.addWidget(self._build_profile_group())
        layout.addWidget(self._build_forward_group())

        exp = QPushButton("Export displayed field CSV…")
        exp.setIcon(theme.icon("fa5s.file-export"))
        exp.clicked.connect(self._export_field_csv)
        layout.addWidget(exp)
        layout.addStretch(1)
        return scroll

    def _build_separation_group(self) -> QGroupBox:
        box = QGroupBox("Regional / residual"); form = QFormLayout(box)
        self._degree = self._ispin(1, 1, 3)
        form.addRow("Polynomial degree", self._degree)
        btn = QPushButton("Compute regional + residual")
        btn.clicked.connect(self._compute_separation)
        form.addRow(btn)
        return box

    def _build_grid_group(self) -> QGroupBox:
        box = QGroupBox("Gridding"); form = QFormLayout(box)
        self._nx = self._ispin(120, 10, 600); self._ny = self._ispin(120, 10, 600)
        self._grid_method = QComboBox(); self._grid_method.addItems(["linear", "cubic", "nearest"])
        form.addRow("nx", self._nx); form.addRow("ny", self._ny)
        form.addRow("Method", self._grid_method)
        grid_btn = QPushButton("Grid displayed field"); grid_btn.clicked.connect(self._grid_field)
        form.addRow(grid_btn)
        exp_btn = QPushButton("Export grid (npy + csv + VTK)…")
        exp_btn.setIcon(theme.icon("fa5s.file-export")); exp_btn.clicked.connect(self._export_grid)
        form.addRow(exp_btn)
        return box

    def _build_profile_group(self) -> QGroupBox:
        box = QGroupBox("Profile"); form = QFormLayout(box)
        self._p1x = self._dspin(0.0, -1e6, 1e6, 1.0, 2); self._p1y = self._dspin(0.0, -1e6, 1e6, 1.0, 2)
        self._p2x = self._dspin(0.0, -1e6, 1e6, 1.0, 2); self._p2y = self._dspin(0.0, -1e6, 1e6, 1.0, 2)
        self._p_n = self._ispin(200, 10, 2000)
        r1 = QHBoxLayout(); r1.addWidget(QLabel("P1")); r1.addWidget(self._p1x); r1.addWidget(self._p1y)
        r2 = QHBoxLayout(); r2.addWidget(QLabel("P2")); r2.addWidget(self._p2x); r2.addWidget(self._p2y)
        w1 = QWidget(); w1.setLayout(r1); w2 = QWidget(); w2.setLayout(r2)
        form.addRow(w1); form.addRow(w2); form.addRow("Samples", self._p_n)
        ext = QPushButton("Extract profile"); ext.clicked.connect(self._extract_profile)
        form.addRow(ext)
        exp = QPushButton("Export profile CSV…"); exp.setIcon(theme.icon("fa5s.file-export"))
        exp.clicked.connect(self._export_profile)
        form.addRow(exp)
        return box

    def _build_forward_group(self) -> QGroupBox:
        box = QGroupBox("Forward body"); v = QVBoxLayout(box)
        self._body_type = QComboBox()
        v.addWidget(self._body_type)
        self._body_stack = QStackedWidget()

        # sphere page
        sp = QWidget(); sf = QFormLayout(sp)
        self._s_x0 = self._dspin(0.0, -1e6, 1e6, 1.0, 1); self._s_y0 = self._dspin(0.0, -1e6, 1e6, 1.0, 1)
        self._s_z0 = self._dspin(20.0, 0.1, 1e5, 1.0, 1); self._s_R = self._dspin(8.0, 0.1, 1e4, 0.5, 1)
        self._s_drho = self._dspin(300.0, -5000.0, 5000.0, 10.0, 0)
        self._s_chi = self._dspin(0.05, 0.0, 5.0, 0.01, 3)
        for lbl, w in (("x0", self._s_x0), ("y0", self._s_y0), ("depth z0 (m)", self._s_z0),
                       ("radius (m)", self._s_R), ("Δρ (kg/m³)", self._s_drho), ("susceptibility", self._s_chi)):
            sf.addRow(lbl, w)
        self._body_stack.addWidget(sp)

        # prism page
        pp = QWidget(); pf = QFormLayout(pp)
        self._p_x1 = self._dspin(-10.0, -1e6, 1e6, 1.0, 1); self._p_x2 = self._dspin(10.0, -1e6, 1e6, 1.0, 1)
        self._p_y1 = self._dspin(-10.0, -1e6, 1e6, 1.0, 1); self._p_y2 = self._dspin(10.0, -1e6, 1e6, 1.0, 1)
        self._p_z1 = self._dspin(5.0, 0.0, 1e5, 1.0, 1); self._p_z2 = self._dspin(25.0, 0.1, 1e5, 1.0, 1)
        self._p_drho = self._dspin(300.0, -5000.0, 5000.0, 10.0, 0)
        for lbl, w in (("x1", self._p_x1), ("x2", self._p_x2), ("y1", self._p_y1), ("y2", self._p_y2),
                       ("z top (m)", self._p_z1), ("z bot (m)", self._p_z2), ("Δρ (kg/m³)", self._p_drho)):
            pf.addRow(lbl, w)
        self._body_stack.addWidget(pp)
        v.addWidget(self._body_stack)

        self._field_box = QGroupBox("Ambient field (magnetics)")
        ff = QFormLayout(self._field_box)
        self._B0 = self._dspin(50000.0, 100.0, 100000.0, 100.0, 0)
        self._inc = self._dspin(60.0, -90.0, 90.0, 1.0, 1)
        self._dec = self._dspin(0.0, -180.0, 180.0, 1.0, 1)
        ff.addRow("Strength (nT)", self._B0); ff.addRow("Inclination (°)", self._inc)
        ff.addRow("Declination (°)", self._dec)
        v.addWidget(self._field_box)

        fwd = QPushButton("Compute forward at stations")
        fwd.setIcon(theme.icon("fa5s.play")); fwd.clicked.connect(self._compute_forward)
        v.addWidget(fwd)
        return box

    # -- kind / display ------------------------------------------------------
    def _on_kind_changed(self) -> None:
        kind = self._kind.currentText()
        self._body_type.clear()
        if kind == "gravity":
            self._body_type.addItems(["sphere", "prism"])
            self._field_box.setVisible(False)
        else:
            self._body_type.addItems(["dipole"])
            self._field_box.setVisible(True)
        self._body_type.currentIndexChanged.connect(self._sync_body_page)
        self._sync_body_page()

    def _sync_body_page(self) -> None:
        # sphere + dipole share the sphere page (index 0); prism is index 1.
        self._body_stack.setCurrentIndex(1 if self._body_type.currentText() == "prism" else 0)

    def _current_values(self) -> Optional[np.ndarray]:
        return self._fields.get(self._display.currentText())

    def _refresh_scatter(self) -> None:
        vals = self._current_values()
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
        # Default profile across the survey.
        self._p1x.setValue(float(self._x.min())); self._p1y.setValue(float(np.median(self._y)))
        self._p2x.setValue(float(self._x.max())); self._p2y.setValue(float(np.median(self._y)))
        self._info.setText(f"{self._source_path.name}<br>{self._x.size} stations")
        self.log(f"Loaded {self._x.size} stations from {self._source_path.name}", "success")
        self._display.setCurrentText("Observed")
        self._refresh_scatter()
        self._publish()

    # -- separation ----------------------------------------------------------
    def _compute_separation(self) -> None:
        if "Observed" not in self._fields:
            self.log("Load station data first.", "warn")
            return
        regional, residual = gmp.regional_residual(self._x, self._y, self._fields["Observed"],
                                                   degree=self._degree.value())
        self._fields["Regional"] = regional
        self._fields["Residual"] = residual
        self._display.setCurrentText("Residual")
        self._refresh_scatter()
        self.log(f"Computed regional (degree {self._degree.value()}) + residual.", "success")
        self._publish()

    # -- gridding ------------------------------------------------------------
    def _grid_field(self) -> None:
        vals = self._current_values()
        if vals is None:
            self.log("Nothing to grid; load data / pick a field.", "warn")
            return
        try:
            self._grid = gmp.grid_data(self._x, self._y, vals, self._nx.value(),
                                       self._ny.value(), self._grid_method.currentText())
        except Exception as exc:  # noqa: BLE001
            self.log(f"Gridding failed: {exc}", "error")
            return
        self._grid_view.set_array(np.asarray(self._grid["zz"], dtype=float))
        self._tabs.setCurrentWidget(self._grid_view)
        self.log(f"Gridded '{self._display.currentText()}' to {self._nx.value()}×{self._ny.value()}.", "success")

    def _export_grid(self) -> None:
        if self._grid is None:
            self.log("Grid a field first.", "warn")
            return
        folder = QFileDialog.getExistingDirectory(self, "Export grid to folder",
                                                  str(self.state.output_dir or Path.cwd()))
        if not folder:
            return
        name = self._display.currentText().split()[0].lower()
        paths = gmp.save_grid(self._grid, Path(folder), name=name, log=lambda m: self.log(m, "info"))
        self.log(f"Exported grid ({len(paths)} files) to {folder}", "success")
        self.report_result({"kind": self._kind.currentText(), "grid_files": paths,
                            "field": self._display.currentText()})

    # -- profile -------------------------------------------------------------
    def _extract_profile(self) -> None:
        if self._grid is None:
            self._grid_field()
        if self._grid is None:
            return
        self._profile = gmp.extract_profile(
            self._grid, (self._p1x.value(), self._p1y.value()),
            (self._p2x.value(), self._p2y.value()), self._p_n.value())
        self._profile_plot.clear()
        self._profile_plot.plot(self._profile["distance"], self._profile["value"],
                                pen=pg.mkPen("#1f77b4", width=2))
        self._tabs.setCurrentWidget(self._profile_plot)
        self.log(f"Extracted profile ({self._p_n.value()} samples).", "success")

    def _export_profile(self) -> None:
        if self._profile is None:
            self.log("Extract a profile first.", "warn")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export profile", "profile.csv", "CSV (*.csv)")
        if not path:
            return
        rows = list(zip(self._profile["distance"].tolist(), self._profile["x"].tolist(),
                        self._profile["y"].tolist(), self._profile["value"].tolist()))
        io_utils.write_csv(path, rows, header=["distance", "x", "y", "value"])
        self.log(f"Exported profile to {path}", "success")

    # -- forward -------------------------------------------------------------
    def _collect_body(self) -> Dict[str, Any]:
        bt = self._body_type.currentText()
        if bt == "prism":
            return {"type": "prism", "x1": self._p_x1.value(), "x2": self._p_x2.value(),
                    "y1": self._p_y1.value(), "y2": self._p_y2.value(),
                    "z1": self._p_z1.value(), "z2": self._p_z2.value(),
                    "density_contrast": self._p_drho.value()}
        return {"type": bt, "x0": self._s_x0.value(), "y0": self._s_y0.value(),
                "z0": self._s_z0.value(), "radius": self._s_R.value(),
                "density_contrast": self._s_drho.value(), "susceptibility": self._s_chi.value()}

    def _compute_forward(self) -> None:
        if self._x is None:
            self.log("Load station coordinates first.", "warn")
            return
        kind = self._kind.currentText()
        field = {"strength": self._B0.value(), "inclination": self._inc.value(),
                 "declination": self._dec.value()}
        try:
            fwd = gmp.forward_bodies(self._x, self._y, kind, [self._collect_body()],
                                     field=field, log=lambda m: self.log(m, "info"))
        except Exception as exc:  # noqa: BLE001
            self.log(f"Forward modeling failed: {exc}", "error")
            return
        self._fields["Forward"] = np.asarray(fwd, dtype=float)
        if "Observed" in self._fields:
            self._fields["Obs − Forward"] = self._fields["Observed"] - self._fields["Forward"]
        self._display.setCurrentText("Forward")
        self._refresh_scatter()
        unit = "mGal" if kind == "gravity" else "nT"
        self.log(f"Computed {kind} forward (range {np.min(fwd):.3g} – {np.max(fwd):.3g} {unit}).", "success")
        self.report_result({"kind": kind, "forward_range": [float(np.min(fwd)), float(np.max(fwd))]})

    # -- export / publish ----------------------------------------------------
    def _export_field_csv(self) -> None:
        vals = self._current_values()
        if vals is None:
            self.log("Nothing to export.", "warn")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export field", "gravmag_field.csv", "CSV (*.csv)")
        if not path:
            return
        rows = [(float(self._x[i]), float(self._y[i]), float(vals[i])) for i in range(self._x.size)]
        io_utils.write_csv(path, rows, header=["x", "y", self._display.currentText()])
        self.log(f"Exported '{self._display.currentText()}' to {path}", "success")

    def _publish(self) -> None:
        self.report_result({
            "kind": self._kind.currentText(),
            "source_file": str(self._source_path) if self._source_path else "",
            "num_stations": int(self._x.size) if self._x is not None else 0,
            "fields": list(self._fields.keys()),
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
                 "desc": "Set the field type."},
                {"name": "set_display", "args": {"name": "str"},
                 "desc": "Choose which field to display (e.g. Observed, Regional, Residual, Forward)."},
                {"name": "compute_separation", "args": {},
                 "desc": "Compute regional + residual by polynomial fit."},
                {"name": "grid_field", "args": {},
                 "desc": "Grid the displayed field onto a regular grid."},
                {"name": "extract_profile", "args": {"p1": "[x, y]", "p2": "[x, y]", "samples": "int"},
                 "desc": "Extract a profile across the gridded field between two points."},
                {"name": "compute_forward", "args": {},
                 "desc": "Compute the forward response of the configured body at the stations."},
                {"name": "set_params", "args": {"params": {"<key>": "value"}},
                 "desc": ("Set parameters. Keys: degree, nx, ny, grid_method (linear/cubic/nearest), "
                          "profile_samples, body_type, sphere_x0/sphere_y0/sphere_depth/sphere_radius/"
                          "sphere_drho/sphere_susceptibility, prism_x1/prism_x2/prism_y1/prism_y2/"
                          "prism_z_top/prism_z_bot/prism_drho, field_strength/field_inclination/field_declination.")},
                {"name": "get_status", "args": {},
                 "desc": "Report loaded data, available fields, and grid state."},
            ],
        }

    def agent_apply(self, action: str, args: Dict[str, Any]) -> Dict[str, Any]:
        args = args or {}
        handlers = {
            "load_data": lambda: self._agent_load(args.get("path")),
            "set_field_type": lambda: self._agent_set_field_type(args.get("kind")),
            "set_display": lambda: self._agent_set_display(args.get("name")),
            "compute_separation": lambda: self._agent_compute_separation(),
            "grid_field": lambda: self._agent_grid_field(),
            "extract_profile": lambda: self._agent_extract_profile(args),
            "compute_forward": lambda: self._agent_compute_forward(),
            "set_params": lambda: self._agent_set_params(args.get("params", args)),
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
            "fields": list(self._fields.keys()),
            "display": self._display.currentText(),
            "gridded": self._grid is not None,
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
        self._p1x.setValue(float(self._x.min())); self._p1y.setValue(float(np.median(self._y)))
        self._p2x.setValue(float(self._x.max())); self._p2y.setValue(float(np.median(self._y)))
        self._info.setText(f"{self._source_path.name}<br>{self._x.size} stations")
        self._display.setCurrentText("Observed")
        self._refresh_scatter()
        self._publish()
        return {"status": "ok", "stations": int(self._x.size)}

    def _agent_set_field_type(self, kind: Any) -> Dict[str, Any]:
        if kind not in ("gravity", "magnetics"):
            return {"status": "failed", "error": "kind must be 'gravity' or 'magnetics'."}
        self._kind.setCurrentText(kind)
        return {"status": "ok", "kind": self._kind.currentText()}

    def _agent_set_display(self, name: Any) -> Dict[str, Any]:
        items = [self._display.itemText(i) for i in range(self._display.count())]
        if name not in items:
            return {"status": "failed", "error": f"Unknown display '{name}'.", "valid": items}
        self._display.setCurrentText(str(name))
        return {"status": "ok", "display": self._display.currentText()}

    def _agent_compute_separation(self) -> Dict[str, Any]:
        if "Observed" not in self._fields:
            return {"status": "failed", "error": "Load station data first."}
        self._compute_separation()
        return {"status": "ok", "fields": list(self._fields.keys())}

    def _agent_grid_field(self) -> Dict[str, Any]:
        if self._current_values() is None:
            return {"status": "failed", "error": "Load data / pick a field first."}
        self._grid_field()
        return {"status": "ok", "gridded": self._grid is not None}

    def _agent_extract_profile(self, args: Dict[str, Any]) -> Dict[str, Any]:
        p1, p2 = args.get("p1"), args.get("p2")
        try:
            if isinstance(p1, (list, tuple)) and len(p1) == 2:
                self._p1x.setValue(float(p1[0])); self._p1y.setValue(float(p1[1]))
            if isinstance(p2, (list, tuple)) and len(p2) == 2:
                self._p2x.setValue(float(p2[0])); self._p2y.setValue(float(p2[1]))
            if "samples" in args:
                self._p_n.setValue(int(args["samples"]))
        except Exception as exc:  # noqa: BLE001
            return {"status": "failed", "error": str(exc)}
        self._extract_profile()
        return {"status": "ok" if self._profile is not None else "failed",
                "samples": self._p_n.value()}

    def _agent_compute_forward(self) -> Dict[str, Any]:
        if self._x is None:
            return {"status": "failed", "error": "Load station coordinates first."}
        self._compute_forward()
        return {"status": "ok", "kind": self._kind.currentText()}

    def _agent_set_params(self, params: Any) -> Dict[str, Any]:
        if not isinstance(params, dict):
            return {"status": "failed", "error": "Provide 'params' as a JSON object."}

        def set_combo(combo, value):
            items = [combo.itemText(i) for i in range(combo.count())]
            if str(value) not in items:
                raise ValueError(f"must be one of {items}")
            combo.setCurrentText(str(value))

        handlers = {
            "degree": lambda v: self._degree.setValue(int(v)),
            "nx": lambda v: self._nx.setValue(int(v)),
            "ny": lambda v: self._ny.setValue(int(v)),
            "grid_method": lambda v: set_combo(self._grid_method, v),
            "profile_samples": lambda v: self._p_n.setValue(int(v)),
            "body_type": lambda v: set_combo(self._body_type, v),
            "sphere_x0": lambda v: self._s_x0.setValue(float(v)),
            "sphere_y0": lambda v: self._s_y0.setValue(float(v)),
            "sphere_depth": lambda v: self._s_z0.setValue(float(v)),
            "sphere_radius": lambda v: self._s_R.setValue(float(v)),
            "sphere_drho": lambda v: self._s_drho.setValue(float(v)),
            "sphere_susceptibility": lambda v: self._s_chi.setValue(float(v)),
            "prism_x1": lambda v: self._p_x1.setValue(float(v)),
            "prism_x2": lambda v: self._p_x2.setValue(float(v)),
            "prism_y1": lambda v: self._p_y1.setValue(float(v)),
            "prism_y2": lambda v: self._p_y2.setValue(float(v)),
            "prism_z_top": lambda v: self._p_z1.setValue(float(v)),
            "prism_z_bot": lambda v: self._p_z2.setValue(float(v)),
            "prism_drho": lambda v: self._p_drho.setValue(float(v)),
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
