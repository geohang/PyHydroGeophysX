"""Seismic -> 3D Model guided wizard.

Builds a 3D subsurface model from one or more 2D seismic velocity sections. Each
section is an inverted velocity model on a pyGIMLi mesh, positioned in map
coordinates by its line endpoints. The wizard collects the lines, sets the
bedrock-interface velocity threshold, configures the 3D grid, then interpolates a
velocity volume and a bedrock-interface surface across the survey area. The heavy
lifting is reused from ``PyHydroGeophysX.qt_apps.seismic3d_pipeline`` (which in
turn reuses ``Geophy_modular.seismic_processor`` and ``core.kriging_3d``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PySide6.QtCore import Qt, QThread, QUrl, Signal
from PySide6.QtGui import QDesktopServices
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
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from PyHydroGeophysX.qt_apps import io_utils, seismic3d_pipeline, theme
from PyHydroGeophysX.qt_apps.modules.base import BaseModule, LogFn
from PyHydroGeophysX.qt_apps.widgets.image_view import ZoomableImageView

_STEPS = ["Lines", "Structure", "Grid", "Build", "Results"]
_STEP_ICONS = ["fa5s.wave-square", "fa5s.mountain", "fa5s.th", "fa5s.cubes", "fa5s.chart-area"]
_P = theme.PALETTE
_COLUMNS = ["Name", "x0", "y0", "x1", "y1", "Mesh", "Velocity"]

_INPUT_FORMAT_FALLBACK = (
    "# Seismic → 3D Model input format\n\n"
    "Add one or more 2D seismic velocity lines. Each line needs:\n\n"
    "- a pyGIMLi mesh (`velmesh.bms`)\n"
    "- a per-cell velocity array (`Vinvmodel.npy`)\n"
    "- map coordinates of the line endpoints (x0, y0) → (x1, y1)\n\n"
    "See seismic3d_input_format.md for details."
)


class Seismic3DWorker(QThread):
    """Runs the seismic -> 3D model build off the UI thread."""

    logged = Signal(str, str)
    succeeded = Signal(dict)
    failed = Signal(str, bool)

    def __init__(self, context, params) -> None:
        super().__init__()
        self._context = context
        self._params = params

    def run(self) -> None:  # noqa: D401 - QThread entry point
        try:
            result = seismic3d_pipeline.build_3d_model(
                self._context, self._params,
                log=lambda m: self.logged.emit(str(m), "info"))
            self.succeeded.emit(result)
        except seismic3d_pipeline.BackendUnavailable as exc:
            self.failed.emit(str(exc), True)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc), False)


class Seismic3DModule(BaseModule):
    module_key = "seismic3d"
    module_title = "Seismic → 3D Model"

    def __init__(self, state: Any, log: LogFn, parent=None) -> None:
        super().__init__(state, log, parent)
        self._lines: List[Dict[str, Any]] = []
        self._worker: Optional[Seismic3DWorker] = None
        self._last_vtk: str = ""
        self._current = 0

        root = QVBoxLayout(self)
        root.addWidget(self._build_strip())
        self._stack = QStackedWidget()
        self._stack.addWidget(self._build_lines_step())      # 0 Lines
        self._stack.addWidget(self._build_structure_step())  # 1 Structure
        self._stack.addWidget(self._build_grid_step())       # 2 Grid
        self._stack.addWidget(self._build_build_step())      # 3 Build
        self._stack.addWidget(self._build_results_step())    # 4 Results
        root.addWidget(self._stack, stretch=1)
        root.addLayout(self._build_nav())
        self._go_to(0)

    # -- helpers -------------------------------------------------------------
    @staticmethod
    def _dspin(value: float, lo: float, hi: float, step: float, decimals: int) -> QDoubleSpinBox:
        s = QDoubleSpinBox(); s.setRange(lo, hi); s.setSingleStep(step)
        s.setDecimals(decimals); s.setValue(value)
        return s

    @staticmethod
    def _ispin(value: int, lo: int, hi: int) -> QSpinBox:
        s = QSpinBox(); s.setRange(lo, hi); s.setValue(value)
        return s

    def _advanced_box(self, title: str = "Advanced") -> Tuple[QGroupBox, QFormLayout]:
        box = QGroupBox(title); box.setCheckable(True); box.setChecked(False)
        inner = QWidget(); form = QFormLayout(inner); form.setContentsMargins(8, 4, 8, 4)
        outer = QVBoxLayout(box); outer.setContentsMargins(8, 4, 8, 8); outer.addWidget(inner)
        inner.setVisible(False); box.toggled.connect(inner.setVisible)
        return box, form

    # -- status strip --------------------------------------------------------
    def _build_strip(self) -> QWidget:
        bar = QWidget(); layout = QHBoxLayout(bar)
        layout.setContentsMargins(4, 2, 4, 6); layout.setSpacing(6)
        self._chips: List[QPushButton] = []
        for i, name in enumerate(_STEPS):
            chip = QPushButton(f"  {i + 1}. {name}")
            chip.setIcon(theme.icon(_STEP_ICONS[i], color="#ffffff"))
            chip.clicked.connect(lambda _=False, idx=i: self._go_to(idx))
            self._chips.append(chip); layout.addWidget(chip)
            if i < len(_STEPS) - 1:
                arrow = QLabel("›"); arrow.setStyleSheet(f"color:{_P['muted']}; font-size:16pt;")
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
        done = [status["lines"], status["lines"], status["lines"], status["all"], status["results"]]
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
        self._back_btn = QPushButton("Back"); self._back_btn.setIcon(theme.icon("fa5s.arrow-left"))
        self._back_btn.clicked.connect(lambda: self._go_to(self._current - 1))
        self._next_btn = QPushButton("Next"); self._next_btn.setIcon(theme.icon("fa5s.arrow-right"))
        self._next_btn.clicked.connect(lambda: self._go_to(self._current + 1))
        nav.addWidget(self._back_btn); nav.addStretch(1); nav.addWidget(self._next_btn)
        return nav

    def _go_to(self, step: int) -> None:
        self._current = max(0, min(step, len(_STEPS) - 1))
        self._stack.setCurrentIndex(self._current)
        self._back_btn.setEnabled(self._current > 0)
        self._next_btn.setEnabled(self._current < len(_STEPS) - 1)
        if self._current == 3:
            self._refresh_build_checklist()
        self._update_strip()

    def _step_status(self) -> Dict[str, bool]:
        self._sync_lines_from_table()
        lines_ok = len(self._lines) > 0
        ran = self.state.module_results.get(self.module_key, {}).get("status") == "ok"
        return {"lines": lines_ok, "all": lines_ok, "results": ran}

    # -- Step 1: Lines -------------------------------------------------------
    def _build_lines_step(self) -> QWidget:
        page = QWidget(); layout = QVBoxLayout(page)
        layout.addWidget(QLabel("<h3>Step 1 · Add 2D seismic velocity lines</h3>"))
        layout.addWidget(QLabel(
            "Each line is an inverted velocity section (mesh + per-cell velocity) "
            "positioned by its map endpoints (x0, y0) → (x1, y1)."))
        row = QHBoxLayout()
        for label, slot, icon in (
            ("Use example", self._use_example, "fa5s.box-open"),
            ("Add line (folder)…", self._add_line_folder, "fa5s.folder-plus"),
            ("Add line (files)…", self._add_line_files, "fa5s.file-medical"),
            ("Remove selected", self._remove_selected, "fa5s.trash"),
            ("Data format", self._show_format_help, "fa5s.file-alt"),
        ):
            b = QPushButton(label); b.setIcon(theme.icon(icon)); b.clicked.connect(slot)
            row.addWidget(b)
        row.addStretch(1)
        layout.addLayout(row)

        self._table = QTableWidget(0, len(_COLUMNS))
        self._table.setHorizontalHeaderLabels(_COLUMNS)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.itemChanged.connect(lambda *_: self._update_strip())
        layout.addWidget(self._table, stretch=1)
        self._lines_status = QLabel("No lines added."); self._lines_status.setWordWrap(True)
        layout.addWidget(self._lines_status)
        return page

    def _add_row(self, line: Dict[str, Any]) -> None:
        r = self._table.rowCount()
        self._table.insertRow(r)
        values = [line.get("name", f"line{r + 1}"), line.get("x0", 0.0), line.get("y0", 0.0),
                  line.get("x1", 100.0), line.get("y1", 0.0), str(line.get("mesh", "")),
                  str(line.get("velocity", ""))]
        for c, val in enumerate(values):
            item = QTableWidgetItem(str(val))
            if _COLUMNS[c] in ("Mesh", "Velocity"):
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                item.setToolTip(str(val))
            self._table.setItem(r, c, item)
        self._update_lines_status()

    def _sync_lines_from_table(self) -> None:
        lines: List[Dict[str, Any]] = []
        for r in range(self._table.rowCount()):
            def cell(c, default=""):
                it = self._table.item(r, c)
                return it.text() if it is not None else default
            try:
                line = {"name": cell(0, f"line{r + 1}"),
                        "x0": float(cell(1, 0)), "y0": float(cell(2, 0)),
                        "x1": float(cell(3, 0)), "y1": float(cell(4, 0)),
                        "mesh": cell(5), "velocity": cell(6)}
            except ValueError:
                continue
            if line["mesh"] and line["velocity"]:
                lines.append(line)
        self._lines = lines

    def _update_lines_status(self) -> None:
        self._sync_lines_from_table()
        self._lines_status.setText(f"{len(self._lines)} line(s) ready.")
        self._update_strip()

    def _example_dir(self) -> Optional[Path]:
        candidates: List[Path] = []
        if self.state.project_root:
            candidates.append(Path(self.state.project_root) / "examples" / "results" / "seismic_example")
        here = Path(__file__).resolve()
        candidates.extend(p / "examples" / "results" / "seismic_example" for p in here.parents)
        for cand in candidates:
            files = seismic3d_pipeline.find_velocity_files(cand)
            if files.get("mesh") is not None and files.get("velocity") is not None:
                return cand
        return None

    def _use_example(self) -> None:
        d = self._example_dir()
        if d is None:
            self.log("Bundled seismic_example not found.", "warn")
            return
        files = seismic3d_pipeline.find_velocity_files(d)
        mesh, vel = str(files["mesh"]), str(files["velocity"])
        for i, y in enumerate((0.0, 60.0, 120.0)):
            self._add_row({"name": f"L{i + 1}", "mesh": mesh, "velocity": vel,
                           "x0": 0.0, "y0": y, "x1": 110.0, "y1": y})
        self.log(f"Added 3 example lines from {d}", "success")

    def _add_line_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select velocity-model folder", str(Path.cwd()))
        if not path:
            return
        files = seismic3d_pipeline.find_velocity_files(Path(path))
        if files.get("mesh") is None or files.get("velocity") is None:
            self.log(f"Folder must contain {seismic3d_pipeline.VELOCITY_FILES['mesh']} and "
                     f"{seismic3d_pipeline.VELOCITY_FILES['velocity']}.", "warn")
            return
        r = self._table.rowCount()
        self._add_row({"name": f"L{r + 1}", "mesh": str(files["mesh"]),
                       "velocity": str(files["velocity"]), "x0": 0.0, "y0": float(r * 50),
                       "x1": 100.0, "y1": float(r * 50)})

    def _add_line_files(self) -> None:
        mesh, _ = QFileDialog.getOpenFileName(self, "Select mesh", "", "Mesh (*.bms);;All files (*)")
        if not mesh:
            return
        vel, _ = QFileDialog.getOpenFileName(self, "Select velocity array", "", "Array (*.npy);;All files (*)")
        if not vel:
            return
        r = self._table.rowCount()
        self._add_row({"name": f"L{r + 1}", "mesh": mesh, "velocity": vel,
                       "x0": 0.0, "y0": float(r * 50), "x1": 100.0, "y1": float(r * 50)})

    def _remove_selected(self) -> None:
        rows = sorted({i.row() for i in self._table.selectedIndexes()}, reverse=True)
        for r in rows:
            self._table.removeRow(r)
        self._update_lines_status()

    # -- Step 2: Structure ---------------------------------------------------
    def _build_structure_step(self) -> QWidget:
        page = QWidget(); layout = QVBoxLayout(page)
        layout.addWidget(QLabel("<h3>Step 2 · Bedrock interface from velocity</h3>"))
        layout.addWidget(QLabel(
            "The interface is the depth where velocity crosses the threshold "
            "(e.g. ~1200 m/s for the regolith / bedrock boundary)."))
        box = QGroupBox("Interface extraction"); form = QFormLayout(box)
        self._threshold = self._dspin(1200.0, 200.0, 8000.0, 50.0, 0)
        self._interval = self._dspin(4.0, 0.5, 50.0, 0.5, 1)
        form.addRow("Velocity threshold (m/s)", self._threshold)
        form.addRow("Horizontal sampling (m)", self._interval)
        layout.addWidget(box)
        btn_row = QHBoxLayout()
        prev = QPushButton("Preview interface of first line")
        prev.setIcon(theme.icon("fa5s.eye")); prev.clicked.connect(self._preview_interface)
        btn_row.addWidget(prev)
        handoff = QPushButton("Send bedrock structure → Hydro")
        handoff.setIcon(theme.icon("fa5s.share"))
        handoff.setToolTip("Pass the bedrock interface of the first line to the "
                           "Geophysics → Hydro module to define its layers.")
        handoff.clicked.connect(self._send_structure_to_hydrology)
        btn_row.addWidget(handoff)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)
        self._interface_view = ZoomableImageView(); self._interface_view.setMinimumHeight(320)
        layout.addWidget(self._interface_view, stretch=1)
        return page

    def _send_structure_to_hydrology(self) -> None:
        self._sync_lines_from_table()
        if not self._lines:
            self.log("Add at least one seismic line first.", "warn")
            return
        line = self._lines[0]
        try:
            res = seismic3d_pipeline.extract_line_structure(
                line, self._threshold.value(), self._interval.value(), log=lambda m: None)
            raw = res["raw_interface"]
        except Exception as exc:  # noqa: BLE001
            self.log(f"Could not extract bedrock interface: {exc}", "error")
            return
        self.state.shared_structure = {
            "interface_xz": [[float(x), float(z)] for x, z in raw],
            "source": f"seismic:{line.get('name', 'line')}",
            "threshold": float(self._threshold.value()),
        }
        self.log(f"Sent bedrock structure ({len(raw)} pts, {self._threshold.value():.0f} m/s) "
                 f"to Geophysics → Hydro. Use 'Derive markers from structure' in its "
                 f"Layers step.", "success")
        self.navigateRequested.emit("geo_hydrology")

    def _preview_interface(self) -> None:
        self._sync_lines_from_table()
        if not self._lines:
            self.log("Add at least one line first.", "warn")
            return
        line = self._lines[0]
        try:
            import matplotlib
            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt
            res = seismic3d_pipeline.extract_line_structure(
                line, self._threshold.value(), self._interval.value(), log=lambda m: None)
            raw = res["raw_interface"]
            fig, ax = plt.subplots(figsize=(7.0, 3.4))
            ax.plot(raw[:, 0], raw[:, 1], "-", color="saddlebrown", lw=2)
            ax.set_xlabel("Distance (m)"); ax.set_ylabel("Elevation (m)")
            ax.set_title(f"{line['name']}: bedrock interface @ {self._threshold.value():.0f} m/s")
            ax.grid(True, alpha=0.3); fig.tight_layout()
            out = Path(self.state.output_dir or ".") / "qt_seismic3d"
            io_utils.ensure_dir(out)
            p = out / "interface_preview.png"
            fig.savefig(p, dpi=160, bbox_inches="tight"); plt.close(fig)
            self._interface_view.set_image_file(str(p))
            self.log(f"Interface preview saved to {p}", "success")
        except Exception as exc:  # noqa: BLE001
            self.log(f"Interface preview failed: {exc}", "error")

    # -- Step 3: Grid --------------------------------------------------------
    def _build_grid_step(self) -> QWidget:
        page = QWidget(); layout = QVBoxLayout(page)
        layout.addWidget(QLabel("<h3>Step 3 · 3D grid and interpolation</h3>"))
        box = QGroupBox("Grid"); form = QFormLayout(box)
        self._grid_res = self._ispin(50, 10, 200)
        self._depth = self._dspin(50.0, 5.0, 500.0, 5.0, 1)
        self._n_layers = self._ispin(19, 2, 100)
        self._interp = QComboBox(); self._interp.addItems(["griddata", "kriging"])
        try:
            from PyHydroGeophysX.core.kriging_3d import GSTOOLS_AVAILABLE
        except Exception:  # noqa: BLE001
            GSTOOLS_AVAILABLE = False
        if not GSTOOLS_AVAILABLE:
            idx = self._interp.findText("kriging")
            self._interp.model().item(idx).setEnabled(False)
            self._interp.setToolTip("Kriging needs the optional 'gstools' package; "
                                    "griddata is used otherwise.")
        form.addRow("Grid resolution (nx = ny)", self._grid_res)
        form.addRow("Model depth (m)", self._depth)
        form.addRow("Vertical layers", self._n_layers)
        form.addRow("Interpolation", self._interp)
        layout.addWidget(box)
        adv, aform = self._advanced_box()
        self._z_scale = self._dspin(1.0, 0.1, 50.0, 0.5, 1)
        self._z_scale.setToolTip("Vertical exaggeration applied during 3D interpolation "
                                 "so thin vertical structure is not over-smoothed.")
        self._max_points = self._ispin(40000, 2000, 500000)
        self._max_points.setSingleStep(5000)
        self._max_points.setToolTip("Velocity sample points are thinned to this many "
                                    "before 3D interpolation, to keep the build responsive.")
        aform.addRow("Vertical scale (interp)", self._z_scale)
        aform.addRow("Max velocity points", self._max_points)

        # Kriging variogram (used only when Interpolation = kriging).
        krig_label = QLabel("<b>Kriging variogram</b> (used when interpolation = kriging)")
        krig_label.setWordWrap(True)
        aform.addRow(krig_label)
        d = seismic3d_pipeline.DEFAULT_KRIGING
        self._krig_model = QComboBox()
        self._krig_model.addItems(list(seismic3d_pipeline.VARIOGRAM_MODELS))
        self._krig_model.setCurrentText(d["model"])
        self._krig_model.setToolTip("Covariance model fitted to the velocity field.")
        self._len_x = self._dspin(d["len_scale_x"], 1.0, 100000.0, 10.0, 1)
        self._len_y = self._dspin(d["len_scale_y"], 1.0, 100000.0, 10.0, 1)
        self._len_z = self._dspin(d["len_scale_z"], 0.5, 10000.0, 1.0, 1)
        self._len_x.setToolTip("Horizontal (x) correlation length / range in metres.")
        self._len_y.setToolTip("Horizontal (y) correlation length / range in metres.")
        self._len_z.setToolTip("Vertical (z) correlation length / range in metres.")
        self._krig_var = self._dspin(d["var"], 0.001, 100.0, 0.1, 3)
        self._krig_var.setToolTip("Variance (sill) of the variogram model.")
        self._krig_nugget = self._dspin(d["nugget"], 0.0, 100.0, 0.05, 3)
        self._krig_nugget.setToolTip("Nugget (micro-scale variance / noise floor).")
        aform.addRow("Variogram model", self._krig_model)
        aform.addRow("Range x (m)", self._len_x)
        aform.addRow("Range y (m)", self._len_y)
        aform.addRow("Range z (m)", self._len_z)
        aform.addRow("Variance (sill)", self._krig_var)
        aform.addRow("Nugget", self._krig_nugget)
        layout.addWidget(adv)
        layout.addStretch(1)
        return page

    # -- Step 4: Build -------------------------------------------------------
    def _build_build_step(self) -> QWidget:
        page = QWidget(); layout = QVBoxLayout(page)
        layout.addWidget(QLabel("<h3>Step 4 · Build the 3D model</h3>"))
        self._build_checklist = QGroupBox("Readiness")
        self._build_checklist_layout = QVBoxLayout(self._build_checklist)
        layout.addWidget(self._build_checklist)
        self._run_btn = QPushButton("Build 3D model")
        self._run_btn.setProperty("primary", True)
        self._run_btn.setIcon(theme.icon("fa5s.cubes", color="#ffffff"))
        self._run_btn.clicked.connect(self._run_build)
        layout.addWidget(self._run_btn)
        cfg_btn = QPushButton("Export config JSON (no backend needed)")
        cfg_btn.setIcon(theme.icon("fa5s.file-export")); cfg_btn.clicked.connect(self._export_config)
        layout.addWidget(cfg_btn)
        self._progress = QProgressBar(); self._progress.setVisible(False)
        layout.addWidget(self._progress)
        self._run_status = QLabel(""); self._run_status.setWordWrap(True)
        layout.addWidget(self._run_status)
        layout.addStretch(1)
        return page

    def _refresh_build_checklist(self) -> None:
        while self._build_checklist_layout.count():
            item = self._build_checklist_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._sync_lines_from_table()
        rows = [("Seismic lines", len(self._lines) > 0, f"{len(self._lines)} line(s)")]
        for label, ok, detail in rows:
            icon = "fa5s.check-circle" if ok else "fa5s.times-circle"
            color = _P["green"] if ok else _P["red"]
            line = QHBoxLayout()
            badge = QLabel(); badge.setPixmap(theme.icon(icon, color=color).pixmap(16, 16))
            text = QLabel(f"<b>{label}</b> — {detail}"); text.setWordWrap(True)
            line.addWidget(badge); line.addWidget(text, stretch=1)
            wrap = QWidget(); wrap.setLayout(line)
            self._build_checklist_layout.addWidget(wrap)
        self._run_btn.setEnabled(len(self._lines) > 0)

    # -- Step 5: Results -----------------------------------------------------
    def _build_results_step(self) -> QWidget:
        page = QWidget(); layout = QVBoxLayout(page)
        header = QHBoxLayout()
        header.addWidget(QLabel("<h3>Step 5 · Results</h3>")); header.addStretch(1)
        self._view3d_btn = QPushButton("View 3D volume in Mesh 3D")
        self._view3d_btn.setIcon(theme.icon("fa5s.cube"))
        self._view3d_btn.setToolTip("Open the interactive 3D viewer (Mesh 3D module) "
                                    "with the velocity volume loaded.")
        self._view3d_btn.clicked.connect(self._view_in_mesh3d)
        self._view3d_btn.setEnabled(False)
        header.addWidget(self._view3d_btn)
        open_btn = QPushButton("Open output folder")
        open_btn.setIcon(theme.icon("fa5s.folder-open")); open_btn.clicked.connect(self._open_output_folder)
        header.addWidget(open_btn)
        layout.addLayout(header)
        self._gallery_scroll = QScrollArea(); self._gallery_scroll.setWidgetResizable(True)
        self._gallery_host = QWidget(); self._gallery_layout = QVBoxLayout(self._gallery_host)
        self._gallery_layout.addWidget(QLabel("No results yet. Complete Steps 1–4 and build the model."))
        self._gallery_scroll.setWidget(self._gallery_host)
        layout.addWidget(self._gallery_scroll, stretch=1)
        return page

    def _populate_results(self, result: Dict[str, Any]) -> None:
        while self._gallery_layout.count():
            item = self._gallery_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        summary = QLabel(
            f"<b>Status:</b> {result.get('status')} &nbsp; "
            f"<b>Lines:</b> {result.get('n_lines', '—')} &nbsp; "
            f"<b>Grid:</b> {result.get('grid_resolution', '—')} &nbsp; "
            f"<b>Velocity:</b> {self._fmt_range(result.get('velocity_range'))} m/s &nbsp; "
            f"<b>Bedrock depth:</b> {self._fmt_range(result.get('bedrock_depth_range'))} m")
        summary.setWordWrap(True)
        self._gallery_layout.addWidget(summary)
        for fig in result.get("figure_paths", []):
            if Path(fig).exists():
                self._gallery_layout.addWidget(QLabel(f"<b>{Path(fig).name}</b>"))
                view = ZoomableImageView(); view.setMinimumHeight(380)
                if view.set_image_file(fig):
                    self._gallery_layout.addWidget(view)
        vtk = result.get("vtk_path")
        self._last_vtk = vtk or ""
        if hasattr(self, "_view3d_btn"):
            self._view3d_btn.setEnabled(bool(vtk) and Path(vtk).exists())
        if vtk:
            self._gallery_layout.addWidget(QLabel(
                f"3D volume (open in ParaView or the Mesh 3D module): <code>{vtk}</code>"))
        cfg = result.get("config_path")
        if cfg:
            self._gallery_layout.addWidget(QLabel(f"Config: <code>{cfg}</code>"))
        self._gallery_layout.addStretch(1)

    @staticmethod
    def _fmt_range(rng) -> str:
        if not rng:
            return "—"
        return f"{rng[0]:.0f}–{rng[1]:.0f}"

    def _view_in_mesh3d(self) -> None:
        if self._last_vtk and Path(self._last_vtk).exists():
            self.viewMeshRequested.emit(self._last_vtk)
        else:
            self.log("No 3D volume available to view yet.", "warn")

    def _open_output_folder(self) -> None:
        out = self.state.module_results.get(self.module_key, {}).get("output_dir")
        out = out or str(self.state.output_dir or "")
        if out and Path(out).exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(out))
        else:
            self.log("Output folder not available yet.", "warn")

    # -- format help ---------------------------------------------------------
    def _show_format_help(self) -> None:
        doc_path = Path(__file__).with_name("seismic3d_input_format.md")
        try:
            text = doc_path.read_text(encoding="utf-8")
        except Exception:  # noqa: BLE001
            text = _INPUT_FORMAT_FALLBACK
        dlg = QDialog(self); dlg.setWindowTitle("Seismic → 3D Model input format")
        dlg.resize(760, 620); lay = QVBoxLayout(dlg)
        browser = QTextBrowser(); browser.setOpenExternalLinks(True)
        try:
            browser.setMarkdown(text)
        except Exception:  # noqa: BLE001
            browser.setPlainText(text)
        lay.addWidget(browser)
        close_btn = QPushButton("Close"); close_btn.clicked.connect(dlg.accept)
        lay.addWidget(close_btn); dlg.exec()

    # -- params / run --------------------------------------------------------
    def _context(self) -> Dict[str, Any]:
        ctx = dict(self.state.context or {})
        if self.state.output_dir:
            ctx.setdefault("output_dir", str(self.state.output_dir))
        return ctx

    def _collect_params(self) -> Dict[str, Any]:
        self._sync_lines_from_table()
        out_dir = self.state.output_dir
        return {
            "output_dir": str(out_dir) if out_dir else "",
            "lines": self._lines,
            "threshold": self._threshold.value(),
            "interval": self._interval.value(),
            "grid_resolution": self._grid_res.value(),
            "depth": self._depth.value(),
            "n_layers": self._n_layers.value(),
            "interp_method": self._interp.currentText(),
            "z_scale": self._z_scale.value(),
            "max_velocity_points": self._max_points.value(),
            "kriging": {
                "model": self._krig_model.currentText(),
                "len_scale_x": self._len_x.value(),
                "len_scale_y": self._len_y.value(),
                "len_scale_z": self._len_z.value(),
                "var": self._krig_var.value(),
                "nugget": self._krig_nugget.value(),
            },
        }

    def _export_config(self) -> str:
        params = self._collect_params()
        config = seismic3d_pipeline.build_seismic3d_config(self._context(), params)
        out_dir = io_utils.ensure_dir(Path(params["output_dir"] or ".") / "qt_seismic3d")
        config_path = out_dir / "seismic3d_config.json"
        io_utils.write_json(config_path, config)
        self.log(f"Exported config to {config_path}", "success")
        self.report_result({"n_lines": config["n_lines"], "config_path": str(config_path),
                            "status": "config_exported", "output_dir": str(out_dir)})
        return str(config_path)

    def _run_build(self) -> None:
        self._sync_lines_from_table()
        if not self._lines:
            self.log("Add at least one seismic line first.", "warn")
            return
        params = self._collect_params()
        self._run_btn.setEnabled(False); self._run_btn.setText("Building…")
        self._progress.setVisible(True); self._progress.setRange(0, 0)
        self._run_status.setText("Building 3D model…")
        self.log("Starting seismic → 3D model build…", "info")
        self._worker = Seismic3DWorker(self._context(), params)
        self._worker.logged.connect(self._on_worker_log)
        self._worker.succeeded.connect(self._on_build_ok)
        self._worker.failed.connect(self._on_build_failed)
        self._worker.finished.connect(self._reset_run_button)
        self._worker.start()

    def _on_worker_log(self, message: str, level: str) -> None:
        self._run_status.setText(message); self.log(message, level)

    def _on_build_ok(self, result: dict) -> None:
        self._progress.setRange(0, 1); self._progress.setValue(1)
        self.log(f"3D model built: {result.get('n_grid_points')} grid points.", "success")
        self.report_result(result)
        self._populate_results(result)
        self._go_to(4)

    def _on_build_failed(self, message: str, backend_unavailable: bool) -> None:
        self._progress.setRange(0, 1); self._progress.setValue(0)
        level = "warn" if backend_unavailable else "error"
        self.log(f"Build problem: {message}", level)
        config_path = self._export_config()
        note = ("The 3D backend (pygimli / pyvista) was not found or failed. The "
                f"configuration has been exported to {config_path}.")
        self._run_status.setText(note); self.log(note, "warn")
        self._populate_results({"status": "config_exported", "n_lines": len(self._lines),
                                "config_path": config_path, "figure_paths": []})
        self._go_to(4)

    def _reset_run_button(self) -> None:
        self._run_btn.setEnabled(True); self._run_btn.setText("Build 3D model")
        self._progress.setVisible(False); self._update_strip()

    # -- AQUAH agent interface ----------------------------------------------
    def agent_describe(self) -> Dict[str, Any]:
        self._sync_lines_from_table()
        return {
            "module": self.module_key,
            "title": self.module_title,
            "current_step": _STEPS[self._current],
            "state": self._agent_status(),
            "actions": [
                {"name": "use_example", "args": {},
                 "desc": "Add three bundled example velocity lines."},
                {"name": "add_line", "args": {"name": "str", "mesh": "str", "velocity": "str",
                                              "x0": "float", "y0": "float", "x1": "float", "y1": "float"},
                 "desc": "Add a 2D velocity line by its mesh + velocity files and map endpoints."},
                {"name": "set_params", "args": {"params": {"<key>": "value"}},
                 "desc": ("Set parameters. Keys: velocity_threshold, horizontal_sampling, "
                          "grid_resolution, depth, vertical_layers, interpolation (griddata/kriging), "
                          "z_scale, max_points, kriging_model, range_x, range_y, range_z, variance, nugget.")},
                {"name": "preview_interface", "args": {},
                 "desc": "Preview the bedrock interface of the first line."},
                {"name": "send_structure_to_hydro", "args": {},
                 "desc": "Send the first line's bedrock interface to the Geophysics -> Hydro module."},
                {"name": "goto_step", "args": {"step": list(_STEPS)},
                 "desc": "Jump to a wizard step."},
                {"name": "run", "args": {},
                 "desc": "Build the 3D velocity model from the added lines."},
                {"name": "get_status", "args": {},
                 "desc": "Report the added lines and last result."},
            ],
        }

    def agent_apply(self, action: str, args: Dict[str, Any]) -> Dict[str, Any]:
        args = args or {}
        handlers = {
            "use_example": lambda: self._agent_use_example(),
            "add_line": lambda: self._agent_add_line(args),
            "set_params": lambda: self._agent_set_params(args.get("params", args)),
            "preview_interface": lambda: self._agent_preview_interface(),
            "send_structure_to_hydro": lambda: self._agent_send_structure(),
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
        self._sync_lines_from_table()
        last = self.state.module_results.get(self.module_key, {})
        return {
            "status": "ok",
            "current_step": _STEPS[self._current],
            "lines": len(self._lines),
            "line_names": [ln.get("name") for ln in self._lines],
            "last_result_status": last.get("status"),
        }

    def _agent_use_example(self) -> Dict[str, Any]:
        self._use_example()
        self._sync_lines_from_table()
        return {"status": "ok", "lines": len(self._lines)}

    def _agent_add_line(self, args: Dict[str, Any]) -> Dict[str, Any]:
        mesh, velocity = args.get("mesh"), args.get("velocity")
        if not mesh or not velocity:
            return {"status": "failed", "error": "Provide 'mesh' and 'velocity' file paths."}
        for label, path in (("mesh", mesh), ("velocity", velocity)):
            if not Path(str(path)).exists():
                return {"status": "failed", "error": f"{label} file not found: {path}"}
        try:
            self._add_row({
                "name": args.get("name") or f"line{self._table.rowCount() + 1}",
                "mesh": str(mesh), "velocity": str(velocity),
                "x0": float(args.get("x0", 0.0)), "y0": float(args.get("y0", 0.0)),
                "x1": float(args.get("x1", 100.0)), "y1": float(args.get("y1", 0.0)),
            })
        except Exception as exc:  # noqa: BLE001
            return {"status": "failed", "error": str(exc)}
        self._sync_lines_from_table()
        return {"status": "ok", "lines": len(self._lines)}

    def _agent_preview_interface(self) -> Dict[str, Any]:
        self._sync_lines_from_table()
        if not self._lines:
            return {"status": "failed", "error": "Add at least one line first."}
        self._preview_interface()
        return {"status": "ok"}

    def _agent_send_structure(self) -> Dict[str, Any]:
        self._sync_lines_from_table()
        if not self._lines:
            return {"status": "failed", "error": "Add at least one line first."}
        self._send_structure_to_hydrology()
        return {"status": "ok", "message": "Bedrock structure sent to Geophysics -> Hydro."}

    def _agent_goto_step(self, step: Any) -> Dict[str, Any]:
        names = {name.lower(): i for i, name in enumerate(_STEPS)}
        idx = step if isinstance(step, int) else names.get(str(step).lower())
        if idx is None or not (0 <= idx < len(_STEPS)):
            return {"status": "failed", "error": f"Unknown step '{step}'.", "valid": list(_STEPS)}
        self._go_to(idx)
        return {"status": "ok", "step": _STEPS[self._current]}

    def _agent_run(self) -> Dict[str, Any]:
        self._sync_lines_from_table()
        if not self._lines:
            return {"status": "failed", "error": "Add at least one seismic line first."}
        self._run_build()
        return {"status": "started", "message": "3D model build started. Ask for status shortly.",
                "lines": len(self._lines)}

    def _agent_set_params(self, params: Any) -> Dict[str, Any]:
        if not isinstance(params, dict):
            return {"status": "failed", "error": "Provide 'params' as a JSON object."}

        def set_combo(combo, value):
            items = [combo.itemText(i) for i in range(combo.count())]
            if str(value) not in items:
                raise ValueError(f"must be one of {items}")
            combo.setCurrentText(str(value))

        handlers = {
            "velocity_threshold": lambda v: self._threshold.setValue(float(v)),
            "horizontal_sampling": lambda v: self._interval.setValue(float(v)),
            "grid_resolution": lambda v: self._grid_res.setValue(int(v)),
            "depth": lambda v: self._depth.setValue(float(v)),
            "vertical_layers": lambda v: self._n_layers.setValue(int(v)),
            "interpolation": lambda v: set_combo(self._interp, v),
            "z_scale": lambda v: self._z_scale.setValue(float(v)),
            "max_points": lambda v: self._max_points.setValue(int(v)),
            "kriging_model": lambda v: set_combo(self._krig_model, v),
            "range_x": lambda v: self._len_x.setValue(float(v)),
            "range_y": lambda v: self._len_y.setValue(float(v)),
            "range_z": lambda v: self._len_z.setValue(float(v)),
            "variance": lambda v: self._krig_var.setValue(float(v)),
            "nugget": lambda v: self._krig_nugget.setValue(float(v)),
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
