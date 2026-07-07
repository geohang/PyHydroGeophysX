"""ERT processing module.

Imports resistivity data from many instrument/file formats (the user picks the
device), shows the electrode layout and a QC apparent-resistivity pseudosection,
runs a pygimli ERT inversion, and shows the inverted resistivity section. The
electrode geometry can still be edited and exported.

Loading reuses ``data_processing.ert_data_agent.load_ert_resipy`` for the
device/format the user selects (there is no auto-detect: pygimli's native reader
silently mis-parses several common formats, e.g. E4D). ``pygimli.physics.ert.load``
remains an internal fallback when a device parser raises.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from PyHydroGeophysX.qt_apps import ert_load, io_utils, theme
from PyHydroGeophysX.qt_apps.modules.base import BaseModule, LogFn
from PyHydroGeophysX.qt_apps.widgets.mesh_view import MeshResultView
from PyHydroGeophysX.qt_apps.widgets.quality_view import InversionQualityView, metrics_from_manager
from PyHydroGeophysX.qt_apps.workers import TaskWorker

_ELEC_FILTER = "Electrodes (*.csv *.txt *.dat);;All files (*)"
_DATA_FILTER = "ERT data (*.dat *.ohm *.txt *.csv *.Data *.bin *.stg *.amp *.udf);;All files (*)"

_INSTRUMENTS: List[Tuple[str, Optional[str]]] = [
    ("BERT / Unified (.ohm/.dat)", "BERT"),
    ("E4D", "E4D"),
    ("DAS-1", "DAS-1"),
    ("Syscal", "Syscal"),
    ("ABEM-Lund", "ABEM-Lund"),
    ("Res2DInv", "ResInv"),
    ("Protocol DC", "Protocol DC"),
    ("Protocol IP", "Protocol IP"),
    ("Sting / SuperSting", "Sting"),
    ("ARES", "ARES"),
    ("Lippmann", "Lippmann"),
    ("Electra", "Electra"),
    ("Custom", "Custom"),
]


class ERTInversionWorker(QThread):
    """Run a pygimli ERT inversion off the UI thread and rasterize the section."""

    succeeded = Signal(dict)
    failed = Signal(str)
    logged = Signal(str)

    def __init__(self, data, lam: float, out_dir: str, max_iter: int = 15,
                 rel_err: float = 0.05, quality: float = 34.0) -> None:
        super().__init__()
        self._data = data
        self._lam = lam
        self._out_dir = out_dir
        self._max_iter = int(max_iter)
        self._rel_err = float(rel_err)
        self._quality = float(quality)

    def run(self) -> None:  # noqa: D401
        try:
            from pygimli.physics import ert as pg_ert

            data = self._data
            if not data.haveData("k"):
                self.logged.emit("Computing geometric factors…")
                data["k"] = pg_ert.createGeometricFactors(data, numerical=False)
            if not data.haveData("rhoa"):
                if data.haveData("r"):
                    data["rhoa"] = data["r"] * data["k"]
                elif data.haveData("u") and data.haveData("i"):
                    data["rhoa"] = data["u"] / data["i"] * data["k"]
            import numpy as _np
            try:
                err = _np.asarray(pg_ert.estimateError(data, relativeError=self._rel_err), dtype=float)
            except Exception:  # noqa: BLE001
                err = _np.asarray(data["err"], dtype=float) if data.haveData("err") else _np.full(data.size(), self._rel_err)
            err[~_np.isfinite(err)] = self._rel_err
            err[err <= 0] = self._rel_err
            data["err"] = err
            self.logged.emit("Inverting ERT data…")
            mgr = pg_ert.ERTManager(data)
            inv_mesh = mgr.createMesh(data=data, quality=self._quality)
            mgr.invert(data, mesh=inv_mesh, lam=self._lam, maxIter=self._max_iter, verbose=False)
            metrics, convergence = metrics_from_manager(
                mgr, n_data=int(data.size()), lam=float(self._lam))
            chi2 = metrics.get("chi2", float("nan"))
            mesh = mgr.paraDomain
            resistivity = _np.asarray(mgr.model, dtype=float)
            vtk_path = ""
            try:
                mesh["resistivity"] = resistivity
                vtk_path = str(Path(self._out_dir) / "resistivity_model.vtk")
                mesh.exportVTK(vtk_path)
            except Exception:  # noqa: BLE001
                vtk_path = ""
            self.succeeded.emit({"mgr": mgr, "chi2": chi2, "vtk": vtk_path,
                                 "metrics": metrics, "convergence": convergence})
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class TimeLapseWorker(QThread):
    """Run a temporal-regularized time-lapse ERT inversion off the UI thread."""

    succeeded = Signal(dict)
    failed = Signal(str, bool)
    logged = Signal(str)

    def __init__(self, files, times, params, out_dir) -> None:
        super().__init__()
        self._files = files
        self._times = times
        self._params = params
        self._out_dir = out_dir

    def run(self) -> None:  # noqa: D401
        from PyHydroGeophysX.qt_apps import ert_timelapse
        try:
            res = ert_timelapse.run_timelapse_ert(
                self._files, self._times, self._params, self._out_dir,
                log=lambda m: self.logged.emit(str(m)))
            self.succeeded.emit(res)
        except ert_timelapse.BackendUnavailable as exc:
            self.failed.emit(str(exc), True)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc), False)


class ERTProcessingModule(BaseModule):
    module_key = "ert_processing"
    module_title = "ERT Processing"

    def __init__(self, state: Any, log: LogFn, parent=None) -> None:
        super().__init__(state, log, parent)
        self._x: List[float] = []
        self._z: List[float] = []
        self._labels: List[str] = []
        self._selected: Optional[int] = None
        self._electrode_path: Optional[Path] = None
        self._data_path: Optional[Path] = None
        self._pseudo: List[Tuple[float, float, float]] = []
        self._n_meas = 0
        self._ert_data = None        # pygimli DataContainerERT for inversion (filtered)
        self._ert_data_full = None   # unfiltered original
        self._inv_worker: Optional[ERTInversionWorker] = None
        self._load_worker: Optional[TaskWorker] = None
        self._tl_files: List[str] = []
        self._tl_labels: List[str] = []
        self._tl_times: List[float] = []
        self._tl_worker: Optional[TimeLapseWorker] = None
        self._tl_out: Optional[str] = None
        self._tl_result: Optional[dict] = None
        self._tl_mesh = None              # in-memory time-lapse result for the
        self._tl_models = None            # interactive per-step viewer
        self._tl_coverage = None
        self._tl_step_titles: List[str] = []
        self._cmap = pg.colormap.get("viridis")

        root = QHBoxLayout(self)
        self._tabs = QTabWidget()
        self._plot_widget = pg.PlotWidget()
        self._plot_widget.setBackground("w")
        self._plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self._plot_widget.setLabel("bottom", "x (m)")
        self._plot_widget.setLabel("left", "z / elevation (m)")
        self._plot = self._plot_widget.getPlotItem()
        self._scatter = pg.ScatterPlotItem(size=12, pen=pg.mkPen("#1565ff", width=1), brush=pg.mkBrush(30, 120, 255, 180))
        self._sel_scatter = pg.ScatterPlotItem(size=18, pen=pg.mkPen("#ff8c00", width=2), brush=pg.mkBrush(255, 140, 0, 120))
        self._plot.addItem(self._scatter)
        self._plot.addItem(self._sel_scatter)
        self._plot.scene().sigMouseClicked.connect(self._on_click)

        self._pseudo_widget = pg.PlotWidget()
        self._pseudo_widget.setBackground("w")
        self._pseudo_widget.showGrid(x=True, y=True, alpha=0.25)
        self._pseudo_widget.setLabel("bottom", "x (m)")
        self._pseudo_widget.setLabel("left", "pseudo-depth (m)")
        self._pseudo_plot = self._pseudo_widget.getPlotItem()
        self._pseudo_scatter = pg.ScatterPlotItem(size=11)
        self._pseudo_plot.addItem(self._pseudo_scatter)

        self._model_view = MeshResultView()
        # The "Resistivity model" tab shows the single inversion OR any time step
        # of a time-lapse run, picked with the step selector (hidden until a
        # time-lapse result is available) — so there is no separate time-lapse tab.
        model_tab = QWidget()
        self._model_tab = model_tab
        model_layout = QVBoxLayout(model_tab)
        model_layout.setContentsMargins(0, 0, 0, 0)
        self._tl_step_row = QWidget()
        step_bar = QHBoxLayout(self._tl_step_row)
        step_bar.setContentsMargins(6, 2, 6, 2)
        step_bar.addWidget(QLabel("Time step:"))
        self._tl_step_combo = QComboBox()
        self._tl_step_combo.setToolTip("Choose which time-lapse inversion result to display.")
        self._tl_step_combo.currentIndexChanged.connect(self._show_tl_step)
        step_bar.addWidget(self._tl_step_combo, stretch=1)
        self._tl_prev_btn = QPushButton("◀"); self._tl_prev_btn.setMaximumWidth(34)
        self._tl_prev_btn.setToolTip("Previous time step")
        self._tl_prev_btn.clicked.connect(lambda: self._step_tl(-1))
        self._tl_next_btn = QPushButton("▶"); self._tl_next_btn.setMaximumWidth(34)
        self._tl_next_btn.setToolTip("Next time step")
        self._tl_next_btn.clicked.connect(lambda: self._step_tl(1))
        step_bar.addWidget(self._tl_prev_btn); step_bar.addWidget(self._tl_next_btn)
        self._tl_step_row.setVisible(False)
        model_layout.addWidget(self._tl_step_row)
        model_layout.addWidget(self._model_view, stretch=1)

        self._quality_view = InversionQualityView()

        self._tabs.addTab(self._plot_widget, "Electrodes")
        self._tabs.addTab(self._pseudo_widget, "Pseudosection")
        self._tabs.addTab(model_tab, "Resistivity model")
        self._tabs.addTab(self._quality_view, "Inversion quality")
        root.addWidget(self._tabs, stretch=1)
        root.addWidget(self._build_controls())

    # -- controls ------------------------------------------------------------
    def _build_controls(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        # Wide enough that the time-lapse list, button row, and long labels fit
        # without a horizontal scrollbar (vertical scrolling only). The minimum
        # covers the widest group + the vertical scrollbar gutter so nothing clips.
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(450)
        scroll.setMaximumWidth(500)
        panel = QWidget()
        scroll.setWidget(panel)
        layout = QVBoxLayout(panel)

        loader = QGroupBox("Load resistivity data")
        lform = QFormLayout(loader)
        self._instrument = QComboBox()
        for label, value in _INSTRUMENTS:
            self._instrument.addItem(label, value)
        # Don't let the longest item ("BERT / Unified (.ohm/.dat)") force the whole
        # control panel wide; elide in the closed box (full text in the dropdown).
        self._instrument.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self._instrument.setMinimumContentsLength(16)
        lform.addRow("Instrument / format", self._instrument)

        load_hint = QLabel("Add one or more ERT files (one per time step for time-lapse). "
                           "Click a row to preview it; order top→bottom is time order.")
        load_hint.setWordWrap(True)
        lform.addRow(load_hint)

        self._tl_list = QListWidget()
        self._tl_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._tl_list.setMaximumHeight(150)
        self._tl_list.setToolTip("Loaded ERT files. Click a row to preview it; two or more "
                                 "become a time sequence for time-lapse inversion.")
        self._tl_list.itemSelectionChanged.connect(self._on_tl_selection_changed)
        self._tl_list.itemClicked.connect(self._preview_tl_item)
        lform.addRow(self._tl_list)

        tl_btns = QHBoxLayout()
        add_btn = QPushButton("Add files…")
        add_btn.setProperty("primary", True)
        add_btn.setIcon(theme.icon("fa5s.file-import", color="#ffffff"))
        add_btn.clicked.connect(self._add_tl_files)
        rm_btn = QPushButton("Remove"); rm_btn.setIcon(theme.icon("fa5s.minus"))
        rm_btn.clicked.connect(self._remove_tl_files)
        up_btn = QPushButton("↑"); up_btn.setToolTip("Move selected up"); up_btn.setMaximumWidth(34)
        up_btn.clicked.connect(lambda: self._move_tl_files(-1))
        down_btn = QPushButton("↓"); down_btn.setToolTip("Move selected down"); down_btn.setMaximumWidth(34)
        down_btn.clicked.connect(lambda: self._move_tl_files(1))
        clr_btn = QPushButton("Clear"); clr_btn.setIcon(theme.icon("fa5s.trash"))
        clr_btn.clicked.connect(self._clear_tl_files)
        for b in (add_btn, rm_btn, up_btn, down_btn, clr_btn):
            tl_btns.addWidget(b)
        lform.addRow(tl_btns)

        self._tl_info = QLabel("No files added."); self._tl_info.setWordWrap(True)
        lform.addRow(self._tl_info)

        load_e = QPushButton("Electrode file (optional)…")
        load_e.setIcon(theme.icon("fa5s.folder-open"))
        load_e.clicked.connect(self._load_electrodes)
        lform.addRow(load_e)
        layout.addWidget(loader)

        self._info = QLabel("No data loaded.")
        self._info.setWordWrap(True)
        layout.addWidget(self._info)

        qc = QGroupBox("Data QC / filter")
        qform = QFormLayout(qc)
        self._rmin = QDoubleSpinBox(); self._rmin.setRange(0.0, 1e6); self._rmin.setValue(0.0); self._rmin.setSuffix(" Ω·m")
        self._rmax = QDoubleSpinBox(); self._rmax.setRange(1.0, 1e7); self._rmax.setValue(100000.0); self._rmax.setSuffix(" Ω·m")
        self._max_err = QDoubleSpinBox(); self._max_err.setRange(0.0, 100.0); self._max_err.setValue(0.0); self._max_err.setSuffix(" %")
        self._max_err.setToolTip("Drop measurements with relative error above this (0 = off).")
        qform.addRow("Min ρa", self._rmin)
        qform.addRow("Max ρa", self._rmax)
        qform.addRow("Max error", self._max_err)
        qrow = QHBoxLayout()
        apply_btn = QPushButton("Apply filter")
        apply_btn.setIcon(theme.icon("fa5s.filter"))
        apply_btn.clicked.connect(self._apply_filter)
        reset_btn = QPushButton("Reset")
        reset_btn.setIcon(theme.icon("fa5s.undo"))
        reset_btn.clicked.connect(self._reset_filter)
        qrow.addWidget(apply_btn); qrow.addWidget(reset_btn)
        qform.addRow(qrow)
        layout.addWidget(qc)

        # One inversion section. λ / iterations / relative error / mesh quality are
        # shared by single and time-lapse inversion; ticking "Time-lapse" reveals
        # the time-lapse-only options and swaps the Run button.
        inv = QGroupBox("Inversion")
        iform = QFormLayout(inv)
        self._lam = QDoubleSpinBox()
        self._lam.setRange(1.0, 1000.0); self._lam.setValue(20.0)
        self._lam.setToolTip("Spatial regularization strength (smoothness).")
        iform.addRow("Lambda", self._lam)
        self._iter = QSpinBox(); self._iter.setRange(2, 60); self._iter.setValue(15)
        self._iter.setToolTip("Maximum number of Gauss-Newton iterations.")
        iform.addRow("Max iterations", self._iter)
        self._relerr = QDoubleSpinBox(); self._relerr.setRange(0.005, 1.0)
        self._relerr.setDecimals(3); self._relerr.setSingleStep(0.01); self._relerr.setValue(0.05)
        self._relerr.setToolTip("Assumed relative data error, used to weight the fit.")
        iform.addRow("Relative error", self._relerr)
        self._quality = QDoubleSpinBox(); self._quality.setRange(20.0, 40.0); self._quality.setValue(34.0)
        self._quality.setToolTip("Inversion-mesh quality (higher = finer triangulation).")
        iform.addRow("Mesh quality", self._quality)

        self._tl_mode = QCheckBox("Time-lapse (multiple ERT files)")
        self._tl_mode.setToolTip("Off: invert the single loaded dataset.  On: jointly invert an "
                                 "ordered sequence of ERT files with temporal regularization "
                                 "(the time-lapse options appear below).")
        self._tl_mode.toggled.connect(self._on_tl_mode)
        iform.addRow(self._tl_mode)

        self._invert_btn = QPushButton("Run inversion")
        self._invert_btn.setProperty("primary", True)
        self._invert_btn.setIcon(theme.icon("fa5s.play", color="#ffffff"))
        self._invert_btn.clicked.connect(self._run_inversion)
        iform.addRow(self._invert_btn)
        self._inv_progress = QProgressBar()
        self._inv_progress.setVisible(False)
        iform.addRow(self._inv_progress)

        iform.addRow(self._build_timelapse_panel())
        layout.addWidget(inv)

        modes = QGroupBox("Electrode editing")
        mbox = QVBoxLayout(modes)
        self._add_mode = QCheckBox("Add electrode (click to place)")
        self._edit_mode = QCheckBox("Edit (click select, click move)")
        self._add_mode.toggled.connect(lambda on: on and self._edit_mode.setChecked(False))
        self._edit_mode.toggled.connect(lambda on: on and self._add_mode.setChecked(False))
        mbox.addWidget(self._add_mode)
        mbox.addWidget(self._edit_mode)
        mbox.addWidget(QLabel("Right-click an electrode: delete / set label"))
        clear_btn = QPushButton("Clear electrodes")
        clear_btn.clicked.connect(self._clear)
        mbox.addWidget(clear_btn)
        layout.addWidget(modes)

        exp = QGroupBox("Export")
        ebox = QVBoxLayout(exp)
        exp_e = QPushButton("Export electrode file…")
        exp_e.setIcon(theme.icon("fa5s.file-csv"))
        exp_e.clicked.connect(self._export_electrodes)
        exp_g = QPushButton("Export survey geometry JSON…")
        exp_g.setIcon(theme.icon("fa5s.file-export"))
        exp_g.clicked.connect(self._export_geometry)
        self._model_export_btn = QPushButton("Export resistivity model…")
        self._model_export_btn.setIcon(theme.icon("fa5s.cube"))
        self._model_export_btn.setToolTip("Export the inverted model as npy + pygimli mesh (.bms) + VTK.")
        self._model_export_btn.setEnabled(False)
        self._model_export_btn.clicked.connect(self._export_resistivity_model)
        ebox.addWidget(exp_e)
        ebox.addWidget(exp_g)
        ebox.addWidget(self._model_export_btn)
        layout.addWidget(exp)

        layout.addStretch(1)
        return scroll

    def _build_timelapse_panel(self) -> QWidget:
        """The time-lapse-only options, shown only when "Time-lapse" is ticked. The
        ERT file list (which doubles as the single-file loader) lives in the Load
        group at the top; shared λ / iterations / relative error / mesh quality live
        in the Inversion group above. Only the temporal controls are here."""
        panel = QWidget()
        self._tl_panel = panel
        tlform = QFormLayout(panel)
        tlform.setContentsMargins(0, 6, 0, 0)
        self._tl_alpha = QDoubleSpinBox(); self._tl_alpha.setRange(0.0, 1000.0); self._tl_alpha.setValue(10.0)
        self._tl_alpha.setToolTip("Temporal regularization strength (couples consecutive time steps).")
        tlform.addRow("Alpha (temporal)", self._tl_alpha)
        self._tl_type = QComboBox(); self._tl_type.addItems(["L2", "L1", "L1L2"])
        self._tl_type.setToolTip("Temporal norm: L2 smooth, L1 blocky, L1L2 hybrid.")
        tlform.addRow("Norm", self._tl_type)

        self._tl_windowed = QCheckBox("Windowed (sliding window)")
        self._tl_windowed.setToolTip("Process consecutive time steps in overlapping windows: "
                                     "cheaper and lower-memory for long monitoring sequences.")
        self._tl_window = QSpinBox(); self._tl_window.setRange(2, 50); self._tl_window.setValue(3)
        self._tl_window.setEnabled(False)
        self._tl_windowed.toggled.connect(self._tl_window.setEnabled)
        tlform.addRow(self._tl_windowed)
        tlform.addRow("Window size", self._tl_window)
        self._tl_lowmem = QCheckBox("Low memory (sparse)")
        self._tl_lowmem.setToolTip("Use single-precision sparse operators to cut RAM "
                                   "(for many files / large meshes). Auto-enabled for "
                                   "large problems; check to force it on.")
        tlform.addRow(self._tl_lowmem)

        self._tl_btn = QPushButton("Run time-lapse inversion")
        self._tl_btn.setProperty("primary", True)
        self._tl_btn.setIcon(theme.icon("fa5s.history", color="#ffffff"))
        self._tl_btn.clicked.connect(self._run_timelapse)
        tlform.addRow(self._tl_btn)
        self._tl_progress = QProgressBar(); self._tl_progress.setVisible(False)
        tlform.addRow(self._tl_progress)
        self._tl_export_btn = QPushButton("Export results (VTK + npy + mesh)…")
        self._tl_export_btn.setIcon(theme.icon("fa5s.cube"))
        self._tl_export_btn.setToolTip("Save the time-lapse models to a folder you pick: a combined VTK, "
                                       "per-step VTKs, final_models.npy, the mesh (.bms), times CSV, and the figure.")
        self._tl_export_btn.setEnabled(False)
        self._tl_export_btn.clicked.connect(self._export_tl_results)
        tlform.addRow(self._tl_export_btn)
        self._tl_open = QPushButton("Open output folder")
        self._tl_open.setIcon(theme.icon("fa5s.folder-open"))
        self._tl_open.setEnabled(False)
        self._tl_open.clicked.connect(self._open_tl_output)
        tlform.addRow(self._tl_open)

        panel.setVisible(False)
        return panel

    def _on_tl_mode(self, checked: bool) -> None:
        """Toggle between single-file and time-lapse inversion."""
        self._tl_panel.setVisible(bool(checked))
        self._invert_btn.setVisible(not checked)

    # -- loading -------------------------------------------------------------
    def _start_load(self, path: str) -> None:
        """Load one ERT file (off the UI thread) into the electrode + pseudosection
        view and make it the current single-inversion dataset. Used when a file is
        added and when a row in the file list is clicked to preview it."""
        instrument = self._instrument.currentData()
        # Capture widget/state values on the UI thread; the parse runs off-thread.
        out_dir = self.state.output_dir or Path.cwd()
        elec_file = str(self._electrode_path) if self._electrode_path and self._electrode_path.exists() else None
        spacing = None  # geometry comes from the file; instrument loaders handle layout
        self._info.setText(f"Loading {Path(path).name}…")
        worker = TaskWorker(self._parse_ert, path, instrument, out_dir, elec_file, spacing)
        worker.succeeded.connect(lambda res: self._on_ert_loaded(path, res))
        worker.failed.connect(self._on_ert_load_failed)
        self._load_worker = self.register_worker(worker)
        worker.start()

    def _parse_ert(self, path, instrument, out_dir, elec_file, spacing):
        """Parse ERT data off the UI thread. Returns a plain dict for the slot."""
        warning = ""
        if instrument is None:  # defensive: the dropdown has no auto/None option
            elec, pseudo, nmeas, data = self._load_pygimli(path)
        else:
            try:
                elec, pseudo, nmeas, data = self._load_resipy(path, instrument, out_dir, elec_file, spacing)
            except Exception as exc:  # noqa: BLE001
                warning = f"{instrument} loader failed ({exc}); fell back to pygimli's native reader."
                elec, pseudo, nmeas, data = self._load_pygimli(path)
        return {"elec": elec, "pseudo": pseudo, "nmeas": nmeas, "data": data, "warning": warning}

    def _on_ert_loaded(self, path: str, res: dict) -> None:
        if res.get("warning"):
            self.log(res["warning"], "warn")
        elec, pseudo, nmeas, data = res["elec"], res["pseudo"], res["nmeas"], res["data"]
        self._x = [float(e[0]) for e in elec]
        self._z = [float(e[1]) for e in elec]
        self._labels = [str(i + 1) for i in range(len(self._x))]
        self._selected = None
        self._data_path = Path(path)
        self._pseudo = pseudo
        self._n_meas = nmeas
        self._ert_data = data
        self._ert_data_full = data
        self._refresh()
        self._draw_pseudosection()
        if pseudo:
            self._tabs.setCurrentWidget(self._pseudo_widget)
        if nmeas == 0:
            self.log(f"{Path(path).name}: parsed {len(self._x)} electrodes but 0 measurements — "
                     f"the Instrument / format is probably wrong for this file.", "warn")
        else:
            self.log(f"Loaded {len(self._x)} electrodes, {nmeas} measurements from {Path(path).name}", "success")

    def _on_ert_load_failed(self, message: str) -> None:
        self.log(f"Could not load ERT data: {message}", "error")
        self._info.setText(f"Load failed: {message}")

    def _load_pygimli(self, path: str):
        import pygimli.physics.ert as ert

        data = ert.load(path, verbose=False)
        if not data.haveData("rhoa"):
            try:
                data["k"] = ert.createGeometricFactors(data, numerical=False)
                if data.haveData("r"):
                    data["rhoa"] = data["r"] * data["k"]
                elif data.haveData("u") and data.haveData("i"):
                    data["rhoa"] = data["u"] / data["i"] * data["k"]
            except Exception:  # noqa: BLE001
                pass
        pos = np.asarray(data.sensors(), dtype=float)
        x = pos[:, 0]
        if pos.shape[1] >= 3 and np.std(pos[:, 2]) > 1e-9:
            z = pos[:, 2]
        elif pos.shape[1] >= 2:
            z = pos[:, 1]
        else:
            z = np.zeros_like(x)
        a = np.asarray(data["a"], dtype=int)
        b = np.asarray(data["b"], dtype=int)
        m = np.asarray(data["m"], dtype=int)
        nn = np.asarray(data["n"], dtype=int)
        rhoa = np.asarray(data["rhoa"], dtype=float) if data.haveData("rhoa") else np.full(data.size(), np.nan)
        elec = [(float(x[i]), float(z[i])) for i in range(len(x))]
        pseudo = self._build_pseudo_from_indices(x, a, b, m, nn, rhoa)
        return elec, pseudo, int(data.size()), data

    def _load_resipy(self, path: str, instrument: str, out_dir, electrode_file, spacing):
        from PyHydroGeophysX.data_processing.ert_data_agent import load_ert_resipy

        project_dir = str(Path(out_dir) / "resipy_project")
        std = load_ert_resipy(
            project_dir=project_dir, data_file=path, instrument=instrument,
            spacing=spacing, electrode_file=electrode_file,
        )
        electrodes = std.electrodes or []
        elev = self._electrode_elevation(electrodes)
        elec = [(float(e.x), float(elev[i])) for i, e in enumerate(electrodes)]
        data = self._standard_to_pg(std)
        if data is not None:
            # Use the corrected apparent resistivity (rhoa = R * k) for QC display.
            pseudo = self._pseudo_from_data(data)
        else:
            # Fallback when pygimli is unavailable: plot raw observation values.
            x_by_id = {int(e.id): float(e.x) for e in electrodes}
            pseudo = []
            for obs in (std.observations or []):
                if obs.app_res is None:
                    continue
                ids = [obs.quad.A, obs.quad.B, obs.quad.M, obs.quad.N]
                xs = [x_by_id.get(int(i), np.nan) for i in ids]
                if np.isfinite(xs).all():
                    span = float(np.max(xs) - np.min(xs))
                    pseudo.append((float(np.mean(xs)), max(span * 0.19, 0.01), float(obs.app_res)))
        return elec, pseudo, len(std.observations or []), data

    # Geometry/topography + StandardERT->pygimli conversion live in the shared
    # ``ert_load`` module so the single-inversion loader and the time-lapse
    # pipeline behave identically. Kept as thin static wrappers for callers here.
    _electrode_elevation = staticmethod(ert_load.electrode_elevation)
    _standard_to_pg = staticmethod(ert_load.standard_to_pg)

    @staticmethod
    def _build_pseudo_from_indices(x, a, b, m, n, rhoa) -> List[Tuple[float, float, float]]:
        nx = len(x)
        if len(a):
            allidx = np.concatenate([a, b, m, n])
            if allidx.min() >= 1 and allidx.max() >= nx:  # 1-based indices
                a, b, m, n = a - 1, b - 1, m - 1, n - 1
        pseudo: List[Tuple[float, float, float]] = []
        for i in range(len(a)):
            ids = [a[i], b[i], m[i], n[i]]
            if min(ids) < 0 or max(ids) >= nx:
                continue
            xs = x[ids]
            span = float(np.max(xs) - np.min(xs))
            pseudo.append((float(np.mean(xs)), max(span * 0.19, 0.01), float(rhoa[i])))
        return pseudo

    def _pseudo_from_data(self, data) -> List[Tuple[float, float, float]]:
        pos = np.asarray(data.sensors(), dtype=float)
        x = pos[:, 0]
        a = np.asarray(data["a"], dtype=int)
        b = np.asarray(data["b"], dtype=int)
        m = np.asarray(data["m"], dtype=int)
        nn = np.asarray(data["n"], dtype=int)
        rhoa = np.asarray(data["rhoa"], dtype=float) if data.haveData("rhoa") else np.full(data.size(), np.nan)
        return self._build_pseudo_from_indices(x, a, b, m, nn, rhoa)

    def _apply_filter(self) -> None:
        if self._ert_data_full is None:
            self.log("Load ERT data first.", "warn")
            return
        try:
            import pygimli as pg
            data = pg.DataContainerERT(self._ert_data_full)
            rhoa = np.asarray(data["rhoa"], dtype=float)
            keep = np.isfinite(rhoa) & (rhoa >= self._rmin.value()) & (rhoa <= self._rmax.value())
            if self._max_err.value() > 0 and data.haveData("err"):
                keep &= np.asarray(data["err"], dtype=float) <= (self._max_err.value() / 100.0)
            removed = int((~keep).sum())
            data.set("valid", pg.Vector(keep.astype(float)))
            data.removeInvalid()
        except Exception as exc:  # noqa: BLE001
            self.log(f"Filter failed: {exc}", "error")
            return
        self._ert_data = data
        self._pseudo = self._pseudo_from_data(data)
        self._n_meas = int(data.size())
        self._draw_pseudosection()
        self._refresh()
        self._tabs.setCurrentWidget(self._pseudo_widget)
        self.log(f"Filter applied: kept {data.size()}, removed {removed}.", "success")

    def _reset_filter(self) -> None:
        if self._ert_data_full is None:
            return
        try:
            import pygimli as pg
            self._ert_data = pg.DataContainerERT(self._ert_data_full)
        except Exception:  # noqa: BLE001
            self._ert_data = self._ert_data_full
        self._pseudo = self._pseudo_from_data(self._ert_data)
        self._n_meas = int(self._ert_data.size())
        self._draw_pseudosection()
        self._refresh()
        self.log("Filter reset.", "info")

    def _load_electrodes(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load electrode file", "", _ELEC_FILTER)
        if not path:
            return
        try:
            table = io_utils.load_xyz_table(path, min_cols=2)
        except Exception as exc:  # noqa: BLE001
            self.log(f"Could not load electrodes: {exc}", "error")
            return
        self._x = [float(v) for v in table[:, 0]]
        self._z = [float(v) for v in table[:, -1]]
        self._labels = [str(i + 1) for i in range(len(self._x))]
        self._selected = None
        self._electrode_path = Path(path)
        self._refresh()
        self.log(f"Loaded {len(self._x)} electrodes from {Path(path).name}", "success")

    # -- inversion -----------------------------------------------------------
    def _run_inversion(self) -> None:
        if self._ert_data is None:
            self.log("Load ERT data with apparent resistivity first.", "warn")
            return
        out = self.state.output_dir or Path.cwd()
        out_dir = str(io_utils.ensure_dir(Path(out) / "ert_results"))
        self._invert_btn.setEnabled(False)
        self._invert_btn.setText("Inverting…")
        self._inv_progress.setVisible(True)
        self._inv_progress.setRange(0, 0)
        self._inv_worker = ERTInversionWorker(
            self._ert_data, self._lam.value(), out_dir,
            max_iter=self._iter.value(), rel_err=self._relerr.value(), quality=self._quality.value())
        self._inv_worker.logged.connect(lambda msg: self.log(msg, "info"))
        self._inv_worker.succeeded.connect(self._on_inversion_ok)
        self._inv_worker.failed.connect(self._on_inversion_failed)
        self._inv_worker.finished.connect(self._reset_invert_button)
        self.register_worker(self._inv_worker)
        self._inv_worker.start()

    def _on_inversion_ok(self, result: dict) -> None:
        mgr = result.get("mgr")
        self._inv_mgr = mgr
        if mgr is not None:
            self._tl_step_row.setVisible(False)  # single result: no step selector
            self._model_view.show_model(mgr, kind="ert")
            self._tabs.setCurrentWidget(self._model_tab)
            self._model_export_btn.setEnabled(True)
        metrics = dict(result.get("metrics") or {})
        metrics.setdefault("method", self._instrument.currentText())
        self._quality_view.show_quality(metrics, result.get("convergence"), title="ERT inversion")
        chi2 = result.get("chi2")
        self.log(f"ERT inversion complete (chi2={chi2:.2f})." if chi2 == chi2 else "ERT inversion complete.", "success")
        vtk = result.get("vtk")
        if vtk:
            self.log(f"Saved resistivity mesh to {vtk}", "info")
        self.report_result({"resistivity_vtk": vtk, "chi2": result.get("chi2"),
                            "rrms": metrics.get("rrms"), "iterations": metrics.get("iterations"),
                            "num_measurements": self._n_meas, "instrument": self._instrument.currentText()})

    def _on_inversion_failed(self, message: str) -> None:
        self.log(f"ERT inversion failed: {message}", "error")

    def _reset_invert_button(self) -> None:
        self._invert_btn.setEnabled(True)
        self._invert_btn.setText("Run inversion")
        self._inv_progress.setVisible(False)

    # -- time-lapse inversion ------------------------------------------------
    def _set_tl_files(self, paths: List[str]) -> None:
        """Replace the ordered time-lapse file set and refresh the list + times."""
        self._tl_files = [str(p) for p in paths]
        self._tl_times, self._tl_labels = ert_load.measurement_times_for(self._tl_files)
        self._refresh_tl_list()

    def _refresh_tl_list(self) -> None:
        self._tl_list.blockSignals(True)
        self._tl_list.clear()
        for i, path in enumerate(self._tl_files):
            label = self._tl_labels[i] if i < len(self._tl_labels) else str(i + 1)
            item = QListWidgetItem(f"{i + 1}.  {label}    ·    {Path(path).name}")
            item.setData(Qt.UserRole, path)
            item.setToolTip(path)
            self._tl_list.addItem(item)
        self._tl_list.blockSignals(False)
        n = len(self._tl_files)
        if n == 0:
            self._tl_info.setText("No files added.")
        elif n == 1:
            self._tl_info.setText(f"<b>1</b> file. Tick “Time-lapse” and add more for a "
                                  f"time sequence. Instrument: {self._instrument.currentText()}.")
        else:
            dated = any(lbl and not lbl.isdigit() for lbl in self._tl_labels)
            span = f"{self._tl_labels[0]} → {self._tl_labels[-1]}" if dated else f"{n} steps"
            self._tl_info.setText(f"<b>{n}</b> files ({span}). "
                                  f"Instrument: {self._instrument.currentText()}.")

    def _add_tl_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Add ERT data file(s)", "", _DATA_FILTER)
        if not paths:
            return
        # Append new files, preserving order and dropping duplicates.
        old_n = len(self._tl_files)
        merged = list(self._tl_files)
        added = 0
        for p in paths:
            if p not in merged:
                merged.append(p); added += 1
        self._set_tl_files(merged)
        self.log(f"Added {added} ERT file(s); {len(self._tl_files)} loaded.", "info")
        # Auto-preview the first newly added file so the user sees data immediately.
        if added and old_n < len(self._tl_files):
            self._tl_list.setCurrentRow(old_n)
            self._preview_tl_item(self._tl_list.item(old_n))

    def _selected_tl_rows(self) -> List[int]:
        return sorted(self._tl_list.row(it) for it in self._tl_list.selectedItems())

    def _remove_tl_files(self) -> None:
        rows = set(self._selected_tl_rows())
        if not rows:
            self.log("Select one or more files in the list to remove.", "warn")
            return
        self._set_tl_files([p for i, p in enumerate(self._tl_files) if i not in rows])
        self.log(f"Removed {len(rows)} file(s); {len(self._tl_files)} remain.", "info")

    def _move_tl_files(self, delta: int) -> None:
        rows = self._selected_tl_rows()
        if not rows:
            return
        order = list(range(len(self._tl_files)))
        seq = rows if delta < 0 else list(reversed(rows))
        for r in seq:
            j = r + delta
            if 0 <= j < len(order):
                order[r], order[j] = order[j], order[r]
        self._tl_files = [self._tl_files[i] for i in order]
        self._tl_times, self._tl_labels = ert_load.measurement_times_for(self._tl_files)
        moved = {order.index(i) for i in rows}
        self._refresh_tl_list()
        for r in moved:
            self._tl_list.item(r).setSelected(True)

    def _clear_tl_files(self) -> None:
        self._set_tl_files([])
        self.log("Cleared time-lapse file list.", "info")

    def _on_tl_selection_changed(self) -> None:
        rows = self._selected_tl_rows()
        if len(rows) > 1:
            self._tl_info.setText(f"{len(rows)} files selected — Remove / Move ↑ ↓, "
                                  f"or click one to preview.")

    def _preview_tl_item(self, item: QListWidgetItem) -> None:
        path = item.data(Qt.UserRole)
        if path and Path(str(path)).exists():
            self.log(f"Preview: loading {Path(str(path)).name} …", "info")
            self._start_load(str(path))

    def _preview_selected_tl(self) -> None:
        items = self._tl_list.selectedItems()
        if items:
            self._preview_tl_item(items[0])

    def _run_timelapse(self) -> None:
        if len(self._tl_files) < 2:
            self.log("Add at least two ordered ERT data files (a time sequence).", "warn")
            return
        out_dir = str(self.state.output_dir or Path.cwd())
        instrument = self._instrument.currentData()
        params = {
            "lambda_val": self._lam.value(), "alpha": self._tl_alpha.value(),
            "inversion_type": self._tl_type.currentText(), "max_iterations": self._iter.value(),
            "relativeError": self._relerr.value(), "mesh_quality": self._quality.value(),
            "windowed": self._tl_windowed.isChecked(), "window_size": self._tl_window.value(),
            "instrument": instrument,
        }
        if self._tl_lowmem.isChecked():
            params["save_memory"] = True
        times = self._tl_times if len(self._tl_times) == len(self._tl_files) else None
        self._tl_btn.setEnabled(False); self._tl_btn.setText("Inverting…")
        self._tl_progress.setVisible(True); self._tl_progress.setRange(0, 0)
        self.log(f"Starting {params['inversion_type']} time-lapse ERT inversion "
                 f"({len(self._tl_files)} steps)…", "info")
        self._tl_worker = TimeLapseWorker(self._tl_files, times, params, out_dir)
        self._tl_worker.logged.connect(lambda m: self.log(m, "info"))
        self._tl_worker.succeeded.connect(self._on_tl_ok)
        self._tl_worker.failed.connect(self._on_tl_failed)
        self._tl_worker.finished.connect(self._reset_tl_button)
        self.register_worker(self._tl_worker)
        self._tl_worker.start()

    def _on_tl_ok(self, result: dict) -> None:
        # Pull out the in-memory mesh + models for the interactive viewer, then
        # drop them so the published result stays JSON-serializable.
        self._tl_mesh = result.pop("mesh", None)
        self._tl_models = result.pop("final_models", None)
        self._tl_coverage = result.pop("coverage", None)
        self._tl_step_titles = list(result.pop("step_titles", []) or [])
        self._tl_result = result
        self._tl_out = result.get("output_dir")
        self._tl_open.setEnabled(bool(self._tl_out))
        self._tl_export_btn.setEnabled(True)

        self._populate_tl_steps()
        self._quality_view.show_quality(
            {"chi2": result.get("chi2"), "iterations": len(result.get("chi2_history") or []) or None,
             "n_data": result.get("n_data"), "lambda": self._lam.value(),
             "method": f"time-lapse {result.get('inversion_type', '')} ({result.get('n_times')} steps)",
             "note": "Joint χ² over all time steps."},
            result.get("chi2_history"), title="Time-lapse ERT inversion")
        lowmem = " · low-memory" if result.get("save_memory") else ""
        n_vtk = len(result.get("vtk_step_paths") or [])
        self.log(f"Time-lapse inversion complete ({result.get('mode')}{lowmem}): "
                 f"{result.get('n_times')} steps, {result.get('mesh_cells')} cells. "
                 f"Saved VTK (combined + {n_vtk} per-step), npy, mesh. "
                 f"Pick a step in the Resistivity model tab; “Export results…” saves them.", "success")
        self.report_result(result)

    def _populate_tl_steps(self) -> None:
        """Fill the step selector from the loaded time-lapse models and show step 0."""
        import numpy as np
        if self._tl_models is None or self._tl_mesh is None:
            self._tl_step_row.setVisible(False)
            return
        n = int(np.asarray(self._tl_models).shape[1])
        self._tl_step_combo.blockSignals(True)
        self._tl_step_combo.clear()
        for i in range(n):
            title = self._tl_step_titles[i] if i < len(self._tl_step_titles) else f"Time step {i + 1}"
            self._tl_step_combo.addItem(f"{i + 1}/{n}  ·  {title}", i)
        self._tl_step_combo.setCurrentIndex(0)
        self._tl_step_combo.blockSignals(False)
        self._tl_step_row.setVisible(n > 1)
        self._show_tl_step(0)
        self._tabs.setCurrentWidget(self._model_tab)

    def _step_tl(self, delta: int) -> None:
        n = self._tl_step_combo.count()
        if n:
            self._tl_step_combo.setCurrentIndex((self._tl_step_combo.currentIndex() + delta) % n)

    def _show_tl_step(self, idx: int) -> None:
        import numpy as np
        if self._tl_models is None or self._tl_mesh is None:
            return
        models = np.asarray(self._tl_models, dtype=float)
        if idx < 0 or idx >= models.shape[1]:
            return
        values = models[:, idx]
        cov = None
        if self._tl_coverage is not None:
            cov_all = np.asarray(self._tl_coverage, dtype=float)
            if cov_all.ndim == 2 and idx < cov_all.shape[0] and cov_all.shape[1] == values.size:
                cov = cov_all[idx] > -1.0  # boolean mask, matching the panel figure
        title = self._tl_step_titles[idx] if idx < len(self._tl_step_titles) else f"Time step {idx + 1}"
        self._model_view.show_field(self._tl_mesh, values, kind="ert", coverage=cov, title=title)

    def _on_tl_failed(self, message: str, backend: bool) -> None:
        self.log(f"Time-lapse inversion {'unavailable' if backend else 'failed'}: {message}",
                 "warn" if backend else "error")

    def _reset_tl_button(self) -> None:
        self._tl_btn.setEnabled(True); self._tl_btn.setText("Run time-lapse inversion")
        self._tl_progress.setVisible(False)

    def _tl_result_files(self) -> List[str]:
        """All result files worth exporting (figure + data + config), de-duplicated."""
        res = self._tl_result or {}
        files: List[str] = []
        for key in ("figure_paths", "data_paths"):
            files.extend(res.get(key) or [])
        if res.get("config_path"):
            files.append(res["config_path"])
        seen, unique = set(), []
        for f in files:
            if f and f not in seen and Path(f).exists():
                seen.add(f); unique.append(f)
        return unique

    def _export_tl_results(self, folder: Optional[str] = None) -> Optional[str]:
        """Copy the time-lapse result files (VTK, npy, mesh, CSV, figure) to a folder."""
        if not self._tl_result:
            self.log("Run the time-lapse inversion first.", "warn")
            return None
        files = self._tl_result_files()
        if not files:
            self.log("No time-lapse result files found to export.", "warn")
            return None
        if not folder:
            folder = QFileDialog.getExistingDirectory(
                self, "Export time-lapse results to folder", str(self.state.output_dir or Path.cwd()))
            if not folder:
                return None
        import shutil
        src_root = Path(self._tl_result.get("output_dir") or "")
        dest = io_utils.ensure_dir(Path(folder))
        copied = 0
        for f in files:
            try:
                src = Path(f)
                # Preserve the vtk_steps/ subfolder so per-step VTKs stay grouped.
                rel = src.relative_to(src_root) if src_root and src_root in src.parents else Path(src.name)
                target = dest / rel
                io_utils.ensure_dir(target.parent)
                shutil.copy2(str(src), str(target))
                copied += 1
            except Exception as exc:  # noqa: BLE001
                self.log(f"Could not copy {Path(f).name}: {exc}", "warn")
        self.log(f"Exported {copied} time-lapse result file(s) to {dest}", "success")
        return str(dest)

    def _open_tl_output(self) -> None:
        out = self._tl_out or str(self.state.output_dir or "")
        if out and Path(out).exists():
            from PySide6.QtCore import QUrl
            from PySide6.QtGui import QDesktopServices
            QDesktopServices.openUrl(QUrl.fromLocalFile(out))
        else:
            self.log("No time-lapse output yet.", "warn")

    # -- pseudosection -------------------------------------------------------
    def _draw_pseudosection(self) -> None:
        if not self._pseudo:
            self._pseudo_scatter.setData([])
            self._pseudo_plot.setTitle("")
            return
        arr = np.asarray(self._pseudo, dtype=float)
        mid, depth, rhoa = arr[:, 0], arr[:, 1], arr[:, 2]
        valid = np.isfinite(rhoa) & (rhoa > 0)
        mid, depth, rhoa = mid[valid], depth[valid], rhoa[valid]
        if rhoa.size == 0:
            self._pseudo_scatter.setData([])
            return
        log_rhoa = np.log10(rhoa)
        lo, hi = np.percentile(log_rhoa, [3, 97])
        rng = hi - lo if hi > lo else 1.0
        norm = np.clip((log_rhoa - lo) / rng, 0.0, 1.0)
        lut = self._cmap.map(norm, mode="byte")
        spots = [
            {"pos": (float(mid[i]), -float(depth[i])),
             "brush": pg.mkBrush(int(lut[i, 0]), int(lut[i, 1]), int(lut[i, 2])), "size": 11}
            for i in range(mid.size)
        ]
        self._pseudo_scatter.setData(spots)
        self._pseudo_plot.setTitle(f"Apparent resistivity (Ω·m): {rhoa.min():.0f} – {rhoa.max():.0f}  (n={rhoa.size})")

    # -- interaction ---------------------------------------------------------
    def _nearest(self, x: float, z: float) -> Optional[int]:
        if not self._x:
            return None
        dx = np.asarray(self._x) - x
        dz = np.asarray(self._z) - z
        return int(np.argmin(dx * dx + dz * dz))

    def _on_click(self, event) -> None:
        if not self._plot.sceneBoundingRect().contains(event.scenePos()):
            return
        vp = self._plot.vb.mapSceneToView(event.scenePos())
        x, z = float(vp.x()), float(vp.y())
        if event.button() == Qt.RightButton:
            idx = self._nearest(x, z)
            if idx is not None:
                self._context_menu(idx, event)
            return
        if event.button() != Qt.LeftButton:
            return
        if self._add_mode.isChecked():
            self._x.append(x); self._z.append(z); self._labels.append(str(len(self._x)))
            self._refresh()
        elif self._edit_mode.isChecked():
            if self._selected is None:
                self._selected = self._nearest(x, z)
            else:
                self._x[self._selected] = x; self._z[self._selected] = z
                self._selected = None
            self._refresh()
        else:
            self._selected = self._nearest(x, z)
            self._refresh()

    def _context_menu(self, idx: int, event) -> None:
        menu = QMenu(self)
        delete_action = menu.addAction("Delete electrode")
        label_action = menu.addAction("Set electrode label…")
        chosen = menu.exec(event.screenPos().toPoint())
        if chosen == delete_action:
            self._delete(idx)
        elif chosen == label_action:
            text, ok = QInputDialog.getText(self, "Electrode label", "Label:", text=self._labels[idx])
            if ok:
                self._labels[idx] = text
                self._refresh()

    def _delete(self, idx: int) -> None:
        for seq in (self._x, self._z, self._labels):
            del seq[idx]
        self._selected = None
        self._refresh()

    def _clear(self) -> None:
        self._x, self._z, self._labels = [], [], []
        self._selected = None
        self._refresh()

    # -- rendering / publish -------------------------------------------------
    def _refresh(self) -> None:
        self._scatter.setData(self._x, self._z)
        if self._selected is not None and 0 <= self._selected < len(self._x):
            self._sel_scatter.setData([self._x[self._selected]], [self._z[self._selected]])
        else:
            self._sel_scatter.setData([], [])
        rhoa_txt = ""
        if self._pseudo:
            vals = np.asarray([p[2] for p in self._pseudo], dtype=float)
            vals = vals[np.isfinite(vals) & (vals > 0)]
            if vals.size:
                rhoa_txt = f"<br>ρa: {vals.min():.0f}–{vals.max():.0f} Ω·m"
        self._info.setText(
            f"Electrodes: {len(self._x)} &nbsp; Measurements: {self._n_meas}"
            f"<br>Data: {self._data_path.name if self._data_path else '—'}{rhoa_txt}"
        )
        self._publish()

    def _coords(self) -> List[List[float]]:
        return [[self._x[i], self._z[i]] for i in range(len(self._x))]

    def _export_electrodes(self) -> None:
        if not self._x:
            self.log("No electrodes to export.", "warn")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export electrodes", "electrodes.csv", "CSV (*.csv)")
        if not path:
            return
        rows = [(self._labels[i], self._x[i], self._z[i]) for i in range(len(self._x))]
        io_utils.write_csv(path, rows, header=["label", "x", "z"])
        self._electrode_path = Path(path)
        self.log(f"Exported {len(rows)} electrodes to {path}", "success")
        self._publish()

    def _export_geometry(self) -> None:
        if not self._x:
            self.log("No electrodes to export.", "warn")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export geometry", "ert_geometry.json", "JSON (*.json)")
        if not path:
            return
        geometry = {
            "method": "ERT",
            "instrument": self._instrument.currentText(),
            "num_electrodes": len(self._x),
            "num_measurements": self._n_meas,
            "electrodes": self._coords(),
            "labels": self._labels,
            "data_file": str(self._data_path) if self._data_path else "",
        }
        io_utils.write_json(path, geometry)
        self.log(f"Exported survey geometry to {path}", "success")
        self._publish(geometry_path=path)

    def _export_resistivity_model(self) -> None:
        mgr = getattr(self, "_inv_mgr", None)
        if mgr is None:
            self.log("Run the ERT inversion first.", "warn")
            return
        folder = QFileDialog.getExistingDirectory(
            self, "Export resistivity model to folder", str(self.state.output_dir or Path.cwd()))
        if not folder:
            return
        try:
            import numpy as np

            out = io_utils.ensure_dir(Path(folder))
            mesh = mgr.paraDomain
            model = np.asarray(mgr.model, dtype=float)  # resistivity (ohm-m)
            np.save(out / "resistivity_model.npy", model)
            mesh.save(str(out / "resistivity_mesh.bms"))
            mesh["resistivity"] = model
            mesh.exportVTK(str(out / "resistivity_model.vtk"))
            try:
                cov = np.asarray(mgr.coverage(), dtype=float)
                np.save(out / "coverage.npy", cov)
            except Exception:  # noqa: BLE001 - coverage is optional
                pass
            self.log(f"Exported resistivity model (npy + bms + vtk) to {out}", "success")
        except Exception as exc:  # noqa: BLE001
            self.log(f"Resistivity model export failed: {exc}", "error")

    def _publish(self, geometry_path: Optional[str] = None) -> None:
        result = {
            "instrument": self._instrument.currentText(),
            "num_electrodes": len(self._x),
            "num_measurements": self._n_meas,
            "electrodes": self._coords(),
            "electrode_file": str(self._electrode_path) if self._electrode_path else "",
            "data_file": str(self._data_path) if self._data_path else "",
        }
        if geometry_path:
            result["geometry_path"] = geometry_path
        self.report_result(result)

    # -- AQUAH agent interface ----------------------------------------------
    def agent_describe(self) -> Dict[str, Any]:
        return {
            "module": self.module_key,
            "title": self.module_title,
            "state": self._agent_status(),
            "actions": [
                {"name": "load_data", "args": {"path": "str", "instrument": "str (optional)"},
                 "desc": ("Load ERT data; pick the instrument/format that matches the file "
                          "(no auto-detect). One of: " +
                          ", ".join(v for _, v in _INSTRUMENTS if v) + ".")},
                {"name": "load_electrodes", "args": {"path": "str"},
                 "desc": "Load an optional electrode geometry file (x, z table)."},
                {"name": "apply_filter", "args": {"min_rhoa": "float", "max_rhoa": "float", "max_error": "float (%)"},
                 "desc": "Filter measurements by apparent resistivity range and max relative error."},
                {"name": "set_params", "args": {"params": {"<key>": "value"}},
                 "desc": ("Set parameters. Shared by single + time-lapse inversion: lambda, "
                          "max_iterations, relative_error, mesh_quality, time_lapse (bool). "
                          "Time-lapse-only: tl_alpha, tl_norm (L2/L1/L1L2), tl_windowed, "
                          "tl_window_size, tl_low_memory.")},
                {"name": "run_inversion", "args": {},
                 "desc": "Run a single-time ERT inversion on the loaded data."},
                {"name": "add_timelapse_files", "args": {"paths": ["str", "str"], "append": "bool (optional)"},
                 "desc": ("Add ERT files for time-lapse inversion (one file per time step, like seismic "
                          "shots). append=true adds to the current list; otherwise replaces it. Files load "
                          "with the selected instrument; times are parsed from filenames when dated.")},
                {"name": "list_timelapse_files", "args": {},
                 "desc": "List the ordered time-lapse files with their parsed time labels."},
                {"name": "remove_timelapse_files", "args": {"indices": ["int"]},
                 "desc": "Remove time-lapse files by 0-based index (or omit to clear all)."},
                {"name": "preview_timelapse_file", "args": {"index": "int"},
                 "desc": "Load one time-lapse file (0-based index) into the electrode + pseudosection view."},
                {"name": "run_timelapse", "args": {},
                 "desc": "Run time-lapse ERT inversion (needs >=2 files)."},
                {"name": "export_timelapse", "args": {"folder": "str"},
                 "desc": ("Export the last time-lapse result to a folder: combined VTK, per-step VTKs, "
                          "final_models.npy, mesh (.bms), times CSV, and the figure.")},
                {"name": "get_status", "args": {},
                 "desc": "Report loaded data, electrode/measurement counts, and last result."},
            ],
        }

    def agent_apply(self, action: str, args: Dict[str, Any]) -> Dict[str, Any]:
        args = args or {}
        handlers = {
            "load_data": lambda: self._agent_load(args.get("path"), args.get("instrument")),
            "load_electrodes": lambda: self._agent_load_electrodes(args.get("path")),
            "apply_filter": lambda: self._agent_apply_filter(args),
            "set_params": lambda: self._agent_set_params(args.get("params", args)),
            "run_inversion": lambda: self._agent_run_inversion(),
            "add_timelapse_files": lambda: self._agent_add_timelapse_files(
                args.get("paths"), args.get("append", False)),
            "list_timelapse_files": lambda: self._agent_list_timelapse(),
            "remove_timelapse_files": lambda: self._agent_remove_timelapse(args.get("indices")),
            "preview_timelapse_file": lambda: self._agent_preview_timelapse(args.get("index")),
            "run_timelapse": lambda: self._agent_run_timelapse(),
            "export_timelapse": lambda: self._agent_export_timelapse(args.get("folder")),
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
            "data_loaded": self._ert_data is not None,
            "electrodes": len(self._x),
            "measurements": self._n_meas,
            "instrument": self._instrument.currentText(),
            "data_file": str(self._data_path or ""),
            "timelapse_files": len(self._tl_files),
            "timelapse_labels": list(self._tl_labels),
            "timelapse_low_memory": self._tl_lowmem.isChecked(),
            "lambda": self._lam.value(),
            "has_model": getattr(self, "_inv_mgr", None) is not None,
            "last_result_keys": sorted(last.keys()),
        }

    def _agent_load(self, path: Any, instrument: Any) -> Dict[str, Any]:
        if not path:
            return {"status": "failed", "error": "Provide 'path' to an ERT data file."}
        p = Path(str(path))
        if not p.exists():
            return {"status": "failed", "error": f"File not found: {p}"}
        # An explicit instrument is matched; "auto"/"none" keeps the current
        # selection (there is no auto-detect format — the user picks the device).
        if instrument is not None and str(instrument).strip() \
                and str(instrument).strip().lower() not in ("auto", "none", "auto-detect"):
            inst_key = str(instrument).strip().lower()
            idx = None
            for i, (label, value) in enumerate(_INSTRUMENTS):
                if value is not None and (inst_key == value.lower() or inst_key == label.lower()):
                    idx = i
                    break
            if idx is None:
                return {"status": "failed", "error": f"Unknown instrument '{instrument}'.",
                        "valid": [v for _, v in _INSTRUMENTS if v]}
            self._instrument.setCurrentIndex(idx)
        inst = self._instrument.currentData()
        out_dir = self.state.output_dir or Path.cwd()
        elec_file = str(self._electrode_path) if self._electrode_path and self._electrode_path.exists() else None
        spacing = None  # geometry comes from the file
        try:
            res = self._parse_ert(str(p), inst, out_dir, elec_file, spacing)
            self._on_ert_loaded(str(p), res)
        except Exception as exc:  # noqa: BLE001
            return {"status": "failed", "error": f"Could not load: {exc}"}
        # Reflect the loaded file in the file list (it doubles as the loader).
        if str(p) not in self._tl_files:
            self._set_tl_files(self._tl_files + [str(p)])
        self._tl_list.setCurrentRow(self._tl_files.index(str(p)))
        return {"status": "ok", "electrodes": len(self._x), "measurements": self._n_meas,
                "instrument": self._instrument.currentText()}

    def _agent_load_electrodes(self, path: Any) -> Dict[str, Any]:
        if not path:
            return {"status": "failed", "error": "Provide 'path' to an electrode file."}
        p = Path(str(path))
        if not p.exists():
            return {"status": "failed", "error": f"File not found: {p}"}
        try:
            table = io_utils.load_xyz_table(str(p), min_cols=2)
        except Exception as exc:  # noqa: BLE001
            return {"status": "failed", "error": f"Could not load electrodes: {exc}"}
        self._x = [float(v) for v in table[:, 0]]
        self._z = [float(v) for v in table[:, -1]]
        self._labels = [str(i + 1) for i in range(len(self._x))]
        self._selected = None
        self._electrode_path = Path(p)
        self._refresh()
        return {"status": "ok", "electrodes": len(self._x)}

    def _agent_apply_filter(self, args: Dict[str, Any]) -> Dict[str, Any]:
        if self._ert_data_full is None:
            return {"status": "failed", "error": "Load ERT data first."}
        try:
            if "min_rhoa" in args:
                self._rmin.setValue(float(args["min_rhoa"]))
            if "max_rhoa" in args:
                self._rmax.setValue(float(args["max_rhoa"]))
            if "max_error" in args:
                self._max_err.setValue(float(args["max_error"]))
        except Exception as exc:  # noqa: BLE001
            return {"status": "failed", "error": str(exc)}
        self._apply_filter()
        return {"status": "ok", "measurements": self._n_meas}

    def _agent_set_params(self, params: Any) -> Dict[str, Any]:
        if not isinstance(params, dict):
            return {"status": "failed", "error": "Provide 'params' as a JSON object."}

        def set_combo(combo, value):
            items = [combo.itemText(i) for i in range(combo.count())]
            if str(value) not in items:
                raise ValueError(f"must be one of {items}")
            combo.setCurrentText(str(value))

        handlers = {
            # shared by single + time-lapse inversion
            "lambda": lambda v: self._lam.setValue(float(v)),
            "max_iterations": lambda v: self._iter.setValue(int(v)),
            "relative_error": lambda v: self._relerr.setValue(float(v)),
            "mesh_quality": lambda v: self._quality.setValue(float(v)),
            "time_lapse": lambda v: self._tl_mode.setChecked(bool(v)),
            # time-lapse-only
            "tl_alpha": lambda v: self._tl_alpha.setValue(float(v)),
            "tl_norm": lambda v: set_combo(self._tl_type, v),
            "tl_windowed": lambda v: self._tl_windowed.setChecked(bool(v)),
            "tl_window_size": lambda v: self._tl_window.setValue(int(v)),
            "tl_low_memory": lambda v: self._tl_lowmem.setChecked(bool(v)),
            # backward-compatible aliases (these params are now shared)
            "tl_lambda": lambda v: self._lam.setValue(float(v)),
            "tl_iterations": lambda v: self._iter.setValue(int(v)),
            "tl_relative_error": lambda v: self._relerr.setValue(float(v)),
            "tl_mesh_quality": lambda v: self._quality.setValue(float(v)),
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
        if self._ert_data is None:
            return {"status": "failed", "error": "Load ERT data with apparent resistivity first."}
        self._run_inversion()
        return {"status": "started", "message": "ERT inversion started. Ask for status shortly.",
                "lambda": self._lam.value()}

    def _agent_add_timelapse_files(self, paths: Any, append: Any = False) -> Dict[str, Any]:
        if not isinstance(paths, list) or not paths:
            return {"status": "failed", "error": "Provide 'paths' as a non-empty list of files."}
        missing = [str(p) for p in paths if not Path(str(p)).exists()]
        if missing:
            return {"status": "failed", "error": f"Files not found: {missing}"}
        merged = list(self._tl_files) if append else []
        for p in paths:
            if str(p) not in merged:
                merged.append(str(p))
        self._set_tl_files(merged)
        self._tl_mode.setChecked(True)  # reveal the time-lapse options in the UI
        return {"status": "ok", "files": len(self._tl_files),
                "time_labels": list(self._tl_labels)}

    def _agent_list_timelapse(self) -> Dict[str, Any]:
        return {
            "status": "ok",
            "count": len(self._tl_files),
            "files": [{"index": i, "label": self._tl_labels[i] if i < len(self._tl_labels) else str(i + 1),
                       "time": self._tl_times[i] if i < len(self._tl_times) else float(i + 1),
                       "name": Path(p).name, "path": p}
                      for i, p in enumerate(self._tl_files)],
        }

    def _agent_remove_timelapse(self, indices: Any) -> Dict[str, Any]:
        if indices is None:
            self._set_tl_files([])
            return {"status": "ok", "files": 0, "message": "Cleared all time-lapse files."}
        try:
            drop = {int(i) for i in indices}
        except (TypeError, ValueError):
            return {"status": "failed", "error": "Provide 'indices' as a list of integers."}
        self._set_tl_files([p for i, p in enumerate(self._tl_files) if i not in drop])
        return {"status": "ok", "files": len(self._tl_files)}

    def _agent_preview_timelapse(self, index: Any) -> Dict[str, Any]:
        try:
            i = int(index)
        except (TypeError, ValueError):
            return {"status": "failed", "error": "Provide 'index' as an integer (0-based)."}
        if not (0 <= i < len(self._tl_files)):
            return {"status": "failed", "error": f"index out of range (0..{len(self._tl_files) - 1})."}
        self._start_load(self._tl_files[i])
        return {"status": "started", "message": f"Loading time step {i} for preview.",
                "name": Path(self._tl_files[i]).name}

    def _agent_run_timelapse(self) -> Dict[str, Any]:
        if len(self._tl_files) < 2:
            return {"status": "failed", "error": "Add at least two ordered ERT files first.",
                    "files": len(self._tl_files)}
        self._run_timelapse()
        return {"status": "started", "message": "Time-lapse inversion started. Ask for status shortly.",
                "steps": len(self._tl_files)}

    def _agent_export_timelapse(self, folder: Any) -> Dict[str, Any]:
        if not self._tl_result:
            return {"status": "failed", "error": "Run the time-lapse inversion first."}
        if not folder:
            return {"status": "failed", "error": "Provide 'folder' to export the results into."}
        dest = self._export_tl_results(str(folder))
        if not dest:
            return {"status": "failed", "error": "Export failed (no result files found)."}
        return {"status": "ok", "folder": dest, "files": len(self._tl_result_files()),
                "vtk_combined": (self._tl_result or {}).get("vtk_combined", ""),
                "vtk_steps": len((self._tl_result or {}).get("vtk_step_paths") or [])}
