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
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QAbstractSpinBox,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from PyHydroGeophysX.data_processing import ert_io as ert_load
from PyHydroGeophysX.qt_apps import io_utils, theme
from PyHydroGeophysX.qt_apps.modules.base import BaseModule, LogFn
from PyHydroGeophysX.qt_apps.qt_utils import (
    BusyStateController,
    ReproduceBar,
    merged_row,
    select_directory,
)
from PyHydroGeophysX.qt_apps.widgets.mesh_view import MeshResultView
from PyHydroGeophysX.qt_apps.widgets.quality_view import InversionQualityView
from PyHydroGeophysX.qt_apps.workers import TaskWorker, WorkflowWorker
from PyHydroGeophysX.data_processing.ert_io import save_edited_ert_container
from PyHydroGeophysX.workflows import (
    ArtifactRef,
    RunContext,
    WorkflowRunResult,
    WorkflowSpec,
    export_workflow_bundle,
)

# Imported rather than repeated: the bounds used to be copied here because
# ert_inversion pulls in pygimli and this module stays importable without it.
# lambda_search imports nothing but numpy, so the copy is no longer needed.
from PyHydroGeophysX.inversion.lambda_search import (  # noqa: E402
    LAMBDA_BOUNDS as _LAMBDA_BOUNDS,
)

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


class ERTProcessingModule(BaseModule):
    module_key = "ert_processing"
    module_title = "ERT Processing"

    def __init__(self, state: Any, log: LogFn, parent=None) -> None:
        super().__init__(state, log, parent)
        self._x: List[float] = []
        self._z: List[float] = []
        self._labels: List[str] = []
        self._electrode_origins: List[Optional[int]] = []
        self._selected: Optional[int] = None
        self._electrode_path: Optional[Path] = None
        self._data_path: Optional[Path] = None
        self._pseudo: List[Tuple[float, float, float]] = []
        self._n_meas = 0
        self._ert_data = None        # pygimli DataContainerERT for inversion (filtered)
        self._ert_data_full = None   # unfiltered original
        self._qc_mask: Optional[List[bool]] = None
        self._inv_worker: Optional[WorkflowWorker] = None
        self._inv_busy: Optional[BusyStateController] = None
        self._ert_recipe_path: str = ""
        # Geometric factors are validated and repaired on every run, with no UI
        # control: a k that disagrees with the geometry rescales the whole section
        # and leaves chi2 untouched, so there is no reading of the result that
        # would make skipping the check the right choice. Scripts and the agent
        # can still set "check" or "off" through set_params.
        self._geom_policy = "fix"
        # Single-inversion results. When the auto-λ search moves off the requested
        # λ, both runs are kept: _inv_choices holds one entry per selectable model
        # and _inv_mgr always points at the one on screen.
        self._inv_mgr = None
        self._inv_choices: List[Dict[str, Any]] = []
        self._load_worker: Optional[TaskWorker] = None
        self._tl_files: List[str] = []
        self._tl_labels: List[str] = []
        self._tl_times: List[float] = []
        self._tl_worker: Optional[WorkflowWorker] = None
        self._tl_busy: Optional[BusyStateController] = None
        self._tl_recipe_path = ""
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
        # When the auto-λ search settles on a different λ than the one typed, both
        # models are kept and this row switches which of the two is displayed.
        self._lam_pick_row = QWidget()
        pick_bar = QHBoxLayout(self._lam_pick_row)
        pick_bar.setContentsMargins(6, 2, 6, 2)
        pick_bar.addWidget(QLabel("Model:"))
        self._lam_pick = QComboBox()
        self._lam_pick.setToolTip(
            "The auto-λ search re-inverted at a different λ. Switch between that model "
            "and the one at the λ you set.")
        self._lam_pick.currentIndexChanged.connect(self._show_lambda_choice)
        pick_bar.addWidget(self._lam_pick, stretch=1)
        self._lam_pick_row.setVisible(False)
        model_layout.addWidget(self._lam_pick_row)
        model_layout.addWidget(self._model_view, stretch=1)

        self._quality_view = InversionQualityView()

        self._tabs.addTab(self._plot_widget, "Electrodes")
        self._tabs.addTab(self._pseudo_widget, "Pseudosection")
        self._tabs.addTab(model_tab, "Resistivity model")
        self._tabs.addTab(self._quality_view, "Inversion quality")
        self._reproduce = ReproduceBar()
        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.addWidget(self._tabs, stretch=1)
        center_layout.addWidget(self._reproduce)
        root.addWidget(center, stretch=1)
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

        # The inversion controls are split three ways: what defines the run, how
        # the data are weighted, and what the software is allowed to change on its
        # own. Everything configurable comes before the Run button at the bottom.
        # λ / iterations / errors / mesh quality are shared by single and
        # time-lapse inversion; ticking "Time-lapse" reveals the time-lapse-only
        # options and swaps the Run button.
        inv = QGroupBox("Inversion")
        iform = QFormLayout(inv)
        self._inv_form = iform

        self._engine = QComboBox()
        for label, value in (("In-house Gauss-Newton", "pyhydro"),
                             ("PyGIMLi ERTManager", "pygimli")):
            self._engine.addItem(label, value)
        self._engine.setToolTip(
            "Solver. The in-house Gauss-Newton inversion exposes its own stopping rule "
            "and line search, so the fit assistance below can drive it directly; the "
            "PyGIMLi manager is kept as a cross-check.")
        iform.addRow("Engine", self._engine)

        self._lam = QDoubleSpinBox()
        self._lam.setDecimals(3)
        self._lam.setRange(*_LAMBDA_BOUNDS); self._lam.setValue(50.0)
        self._lam.setStepType(QAbstractSpinBox.StepType.AdaptiveDecimalStepType)
        self._lam.setToolTip(
            "Spatial regularization strength (smoothness). Lower fits the data harder, "
            "higher gives a smoother model. Start on the smooth side: the auto-λ search "
            "relaxes downward, continuing each λ from the previous solution, and that "
            "direction is the stable one. Values below 1 are allowed; the range is "
            f"{_LAMBDA_BOUNDS[0]:g} to {_LAMBDA_BOUNDS[1]:g} and the arrows step by "
            "one significant digit.")
        iform.addRow("Lambda", self._lam)

        # Iteration budget: one number per attempt, one for the total. They belong
        # on one line because they are two ends of the same setting.
        self._iter = QSpinBox(); self._iter.setRange(2, 60); self._iter.setValue(15)
        self._iter.setToolTip(
            "Iterations per attempt. A run that uses all of them while still improving "
            "is continued from its own model rather than being judged there, up to the "
            "ceiling beside it, so λ is never blamed for an unfinished descent.")
        self._iter_ceiling = QSpinBox(); self._iter_ceiling.setRange(5, 400)
        self._iter_ceiling.setValue(60)
        self._iter_ceiling.setToolTip(
            "Total iterations allowed at one λ, counting continuations. Reaching it "
            "means the reported χ² is an upper bound, and the log says so.")
        iter_row = QHBoxLayout()
        iter_row.setContentsMargins(0, 0, 0, 0)
        iter_row.addWidget(self._iter)
        iter_row.addWidget(QLabel("per pass, up to"))
        iter_row.addWidget(self._iter_ceiling)
        iter_row.addStretch(1)  # pack left; stretched spin boxes leave odd gaps
        self._iter_row = QWidget(); self._iter_row.setLayout(iter_row)
        iform.addRow("Iterations", self._iter_row)

        self._plateau = QDoubleSpinBox()
        self._plateau.setRange(0.01, 10.0); self._plateau.setDecimals(2)
        self._plateau.setSingleStep(0.1); self._plateau.setValue(0.5)
        self._plateau.setSuffix(" %")
        self._plateau.setToolTip(
            "A λ is finished once χ² improves by less than this per iteration. Loosen it "
            "for speed, tighten it to be sure a λ has really run out of room.")
        iform.addRow("Stop below", self._plateau)

        # An imported mesh. Building one from the electrode line is fine for a
        # 2D profile and hopeless for a 3D domain with topography, boreholes or
        # known structure, which is meshed externally (usually in Gmsh).
        self._mesh_path = ""
        self._mesh_btn = QPushButton("Import mesh…")
        self._mesh_btn.setIcon(theme.icon("fa5s.project-diagram"))
        self._mesh_btn.setToolTip(
            "Invert on a mesh you built elsewhere instead of one generated from "
            "the electrode positions. PyGIMLi .bms, Gmsh .msh, VTK, or .poly. "
            "The region to invert must carry marker 2 or above; marker 0 and 1 "
            "are treated as background and stay fixed. The file is checked "
            "against the survey before the run starts.")
        self._mesh_btn.clicked.connect(self._import_mesh)
        self._mesh_clear = QPushButton("✕")
        self._mesh_clear.setMaximumWidth(32)
        self._mesh_clear.setToolTip("Go back to a mesh generated from the data.")
        self._mesh_clear.setEnabled(False)
        self._mesh_clear.clicked.connect(self._clear_mesh)
        self._mesh_row = merged_row(self._mesh_btn, self._mesh_clear)
        iform.addRow("Mesh", self._mesh_row)
        self._mesh_note = QLabel("Built from the electrode positions.")
        self._mesh_note.setWordWrap(True)
        self._mesh_note.setStyleSheet("color:#5a6a7a; font-size:8pt;")
        iform.addRow("", self._mesh_note)

        self._quality = QDoubleSpinBox(); self._quality.setRange(20.0, 40.0); self._quality.setValue(34.0)
        self._quality.setToolTip("Inversion-mesh quality (higher = finer triangulation).")
        iform.addRow("Mesh quality", self._quality)

        self._para_depth = QDoubleSpinBox()
        self._para_depth.setRange(0.0, 10000.0); self._para_depth.setDecimals(1)
        self._para_depth.setSingleStep(5.0); self._para_depth.setValue(0.0)
        self._para_depth.setSuffix(" m")
        self._para_depth.setSpecialValueText("auto")
        self._para_depth.setToolTip(
            "How deep to invert. PyGIMLi sizes the parameter domain from the array "
            "length, which for a long line reaches well below anything the data "
            "resolve; capping it removes unknowns the inversion cannot constrain and "
            "shortens every iteration. Leave at auto unless the sensitivity plot shows "
            "the bottom of the section is empty.")
        iform.addRow("Invert to depth", self._para_depth)
        layout.addWidget(inv)

        # -- data errors -----------------------------------------------------
        errs = QGroupBox("Data errors")
        eform = QFormLayout(errs)
        self._err_source = QComboBox()
        for label, value in (("File err column", "file"),
                             ("Estimate from the values below", "estimate"),
                             ("Larger of the two", "max")):
            self._err_source.addItem(label, value)
        self._err_source.setToolTip(
            "Where the per-measurement error comes from. Most instruments write an err "
            "column; replacing it with an assumed percentage makes χ² report on an error "
            "model the data never had.")
        eform.addRow("Taken from", self._err_source)

        self._relerr = QDoubleSpinBox(); self._relerr.setRange(0.005, 1.0)
        self._relerr.setDecimals(3); self._relerr.setSingleStep(0.01); self._relerr.setValue(0.05)
        self._relerr.setToolTip(
            "Assumed relative data error, used when estimating and as the fallback where "
            "the file has no usable value.")
        self._abserr = QDoubleSpinBox(); self._abserr.setRange(0.0, 100.0)
        self._abserr.setDecimals(4); self._abserr.setSingleStep(0.001); self._abserr.setValue(0.0)
        self._abserr.setSuffix(" Ω")
        self._abserr.setToolTip(
            "Absolute resistance error, added as absolute/|R|. It raises the error bar on "
            "weak readings, which is where a flat percentage is most optimistic. Leave at "
            "0 when the survey spans a narrow signal range.")
        # Laid out as the formula it is, so the two numbers read as one model
        # rather than as two unrelated settings.
        err_row = QHBoxLayout()
        err_row.setContentsMargins(0, 0, 0, 0)
        err_row.addWidget(self._relerr)
        err_row.addWidget(QLabel("+"))
        err_row.addWidget(self._abserr)
        err_row.addWidget(QLabel("/ |R|"))
        err_row.addStretch(1)
        self._errmodel_row = QWidget(); self._errmodel_row.setLayout(err_row)
        eform.addRow("Estimate", self._errmodel_row)
        layout.addWidget(errs)

        # -- fit assistance --------------------------------------------------
        assist = QGroupBox("Fit assistance")
        aform = QFormLayout(assist)

        # Kept short enough to fit the panel; the tooltip carries the detail.
        self._auto_lam = QCheckBox("Auto-λ: re-invert to reach target χ²")
        self._auto_lam.setChecked(True)
        self._auto_lam.setToolTip(
            "The inversion at the λ above always runs first and is always kept. If its χ² "
            "misses the target band, the same mesh is re-inverted at other λ values "
            "(bracket, then bisect in log λ) and the closest one becomes the displayed "
            "model. Each trial continues from the nearest λ already solved rather than "
            "restarting, so the later ones are cheap, but every trial is still a full "
            "inversion.")
        self._auto_lam.toggled.connect(self._on_auto_lambda)
        aform.addRow(self._auto_lam)

        self._target_chi2 = QDoubleSpinBox()
        self._target_chi2.setRange(0.1, 100.0); self._target_chi2.setDecimals(2)
        self._target_chi2.setSingleStep(0.1); self._target_chi2.setValue(1.0)
        self._target_chi2.setToolTip(
            "χ² = 1 means the model explains the data to within the assumed relative "
            "error. Raise it if the data are noisier than the error estimate admits.")
        self._chi2_tol = QDoubleSpinBox()
        self._chi2_tol.setRange(0.01, 10.0); self._chi2_tol.setDecimals(2)
        self._chi2_tol.setSingleStep(0.05); self._chi2_tol.setValue(0.2)
        self._chi2_tol.setToolTip(
            "Half-width of the accepted band. The search stops as soon as a trial lands "
            "inside target ± tolerance.")
        chi_row = QHBoxLayout()
        chi_row.addWidget(self._target_chi2)
        chi_row.addWidget(QLabel("±")); chi_row.addWidget(self._chi2_tol)
        chi_row.addStretch(1)
        self._chi2_row = QWidget(); self._chi2_row.setLayout(chi_row)
        chi_row.setContentsMargins(0, 0, 0, 0)
        aform.addRow("Target χ²", self._chi2_row)

        self._lam_trials = QSpinBox(); self._lam_trials.setRange(1, 20); self._lam_trials.setValue(6)
        self._lam_trials.setToolTip(
            "Upper bound on the extra inversions the λ search may run, on top of the "
            "one at your λ. Reached only when the target stays out of range.")
        aform.addRow("Max λ trials", self._lam_trials)

        self._reject = QCheckBox("Reject outliers: drop data the model cannot explain")
        self._reject.setChecked(False)
        self._reject.setToolTip(
            "After the inversion has converged, drop the measurements whose residual "
            "exceeds the threshold below and invert again. This is what brings χ² down "
            "when a handful of readings are simply bad; it also shrinks the dataset, so "
            "the floor below keeps it from gutting the survey.")
        self._reject.toggled.connect(self._on_reject_outliers)
        aform.addRow(self._reject)

        self._reject_sigma = QDoubleSpinBox()
        self._reject_sigma.setRange(1.5, 20.0); self._reject_sigma.setDecimals(1)
        self._reject_sigma.setSingleStep(0.5); self._reject_sigma.setValue(3.0)
        self._reject_sigma.setToolTip(
            "Rejection cut in units of the assumed error. A datum at 3 means the model "
            "misses it by three times its own error bar.")
        self._reject_passes = QSpinBox(); self._reject_passes.setRange(1, 5)
        self._reject_passes.setValue(2)
        self._reject_passes.setToolTip(
            "How many reject-and-re-invert cycles to run. Each pass is a full inversion.")
        rej_row = QHBoxLayout()
        rej_row.setContentsMargins(0, 0, 0, 0)
        rej_row.addWidget(self._reject_sigma)
        rej_row.addWidget(QLabel("σ, passes")); rej_row.addWidget(self._reject_passes)
        rej_row.addStretch(1)
        self._reject_row = QWidget(); self._reject_row.setLayout(rej_row)
        aform.addRow("Cut beyond", self._reject_row)

        self._min_keep = QDoubleSpinBox()
        self._min_keep.setRange(10.0, 100.0); self._min_keep.setDecimals(0)
        self._min_keep.setSingleStep(5.0); self._min_keep.setValue(50.0)
        self._min_keep.setSuffix(" %")
        self._min_keep.setToolTip(
            "Rejection stops before it would leave less than this share of the "
            "measurements. A χ² bought by deleting most of the survey is not a fit.")
        aform.addRow("Keep at least", self._min_keep)
        layout.addWidget(assist)

        # -- run -------------------------------------------------------------
        runbox = QGroupBox("Run")
        rform = QFormLayout(runbox)
        iform = rform  # the time-lapse panel and Run button live here

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
        # Reflect the initial checkbox states; setChecked() above emitted nothing.
        self._on_auto_lambda(self._auto_lam.isChecked())
        self._on_reject_outliers(self._reject.isChecked())
        layout.addWidget(runbox)

        # Electrode editing has no panel: placing and dragging electrodes by mouse
        # was fiddly and rarely the right way to fix a geometry. The operations
        # live on as agent actions (add/move/delete/label/clear/list electrodes),
        # which is both more precise and reproducible. Clicking the plot still
        # selects an electrode so its position can be read off.

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
        self._electrode_origins = list(range(len(self._x)))
        self._selected = None
        self._data_path = Path(path)
        self._pseudo = pseudo
        self._n_meas = nmeas
        self._ert_data = data
        self._ert_data_full = data
        self._qc_mask = [True] * int(data.size())
        if hasattr(self.state, "register_geophysical_resource"):
            self.state.register_geophysical_resource(
                "ERT", "observed_data", data,
                label=f"ERT observations · {Path(path).name}", path=str(path),
                metadata={"measurements": int(nmeas), "electrodes": len(self._x)},
                resource_id="ert:observed_data:active",
            )
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
            self._qc_mask = keep.astype(bool).tolist()
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
        self._qc_mask = [True] * int(self._ert_data_full.size())
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
        original_count = (
            int(self._ert_data.sensorCount()) if self._ert_data is not None else 0
        )
        self._electrode_origins = [
            index if index < original_count else None
            for index in range(len(self._x))
        ]
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
        out_path = io_utils.ensure_dir(Path(out) / "ert_results")
        input_path = out_path / "filtered_ert_data.dat"
        electrode_rows = [
            {
                "order": index,
                "label": self._labels[index],
                "x": float(self._x[index]),
                "z": float(self._z[index]),
                "original_index": self._electrode_origins[index],
            }
            for index in range(len(self._x))
        ]
        electrode_path = out_path / "edited_electrodes.csv"
        qc_path = out_path / "ert_qc_mask.json"
        io_utils.write_csv(
            electrode_path,
            [
                (
                    row["order"],
                    row["label"],
                    row["x"],
                    row["z"],
                    "" if row["original_index"] is None else row["original_index"],
                )
                for row in electrode_rows
            ],
            header=("order", "label", "x", "z", "original_index"),
        )
        io_utils.write_json(
            qc_path,
            {
                "keep": list(self._qc_mask or []),
                "source_measurements": int(self._ert_data_full.size())
                if self._ert_data_full is not None else int(self._ert_data.size()),
                "filtered_measurements": int(self._ert_data.size()),
            },
        )
        try:
            save_edited_ert_container(self._ert_data, input_path, electrode_rows)
        except Exception as exc:  # noqa: BLE001
            self.log(f"Could not serialize edited/QC-filtered ERT data: {exc}", "error")
            return
        project_root = out_path.resolve()
        spec = WorkflowSpec(
            workflow_id="ert.single_inversion",
            inputs={
                "data": ArtifactRef.from_path(
                    input_path,
                    artifact_id="ert:single:filtered_data",
                    kind="ert_data",
                    format="dat",
                    base_dir=project_root,
                    metadata={
                        "instrument": self._instrument.currentText(),
                        "electrodes": len(self._x),
                        "measurements": self._n_meas,
                        "qc_filtered": True,
                    },
                ),
                "electrodes": ArtifactRef.from_path(
                    electrode_path,
                    artifact_id="ert:single:electrodes",
                    kind="electrode_geometry",
                    format="csv",
                    base_dir=project_root,
                ),
                "qc_mask": ArtifactRef.from_path(
                    qc_path,
                    artifact_id="ert:single:qc_mask",
                    kind="qc_mask",
                    format="json",
                    base_dir=project_root,
                ),
            },
            parameters={
                "lambda": float(self._lam.value()),
                "max_iterations": int(self._iter.value()),
                "relative_error": float(self._relerr.value()),
                "mesh_quality": float(self._quality.value()),
                "para_depth": float(self._para_depth.value()),
                "mesh_file": str(self._mesh_path or ""),
                "instrument": "BERT",
                "engine": str(self._engine.currentData()),
                "geometric_factor_policy": str(self._geom_policy),
                "error_source": str(self._err_source.currentData()),
                "absolute_error": float(self._abserr.value()),
                "plateau_tolerance": float(self._plateau.value()) / 100.0,
                "max_total_iterations": int(self._iter_ceiling.value()),
                "reject_outliers": bool(self._reject.isChecked()),
                "outlier_threshold": float(self._reject_sigma.value()),
                "outlier_passes": int(self._reject_passes.value()),
                "min_data_fraction": float(self._min_keep.value()) / 100.0,
                "auto_lambda": bool(self._auto_lam.isChecked()),
                "target_chi2": float(self._target_chi2.value()),
                "chi2_tolerance": float(self._chi2_tol.value()),
                "max_lambda_trials": int(self._lam_trials.value()),
            },
            metadata={"source_instrument": self._instrument.currentText()},
        )
        recipe_path, script_path = export_workflow_bundle(spec, out_path, stem="ert")
        self._reproduce.set_bundle(recipe_path, script_path)
        self._ert_recipe_path = str(recipe_path)
        self._inv_busy = BusyStateController([self._invert_btn])
        self._inv_busy.start()
        self._invert_btn.setText("Inverting…")
        self._inv_progress.setVisible(True)
        self._inv_progress.setRange(0, 0)
        self._inv_worker = WorkflowWorker(
            spec,
            RunContext(project_root=project_root, output_dir=out_path),
        )
        self._inv_worker.logged.connect(lambda msg: self.log(msg, "info"))
        self._inv_worker.succeeded.connect(self._on_ert_workflow_ok)
        self._inv_worker.failed.connect(self._on_inversion_failed)
        self._inv_worker.finished.connect(self._reset_invert_button)
        self.register_worker(self._inv_worker)
        self._inv_worker.start()

    def _abs_path(self, value: Any) -> str:
        """Resolve a workflow-relative output path against the recipe directory."""
        if not value:
            return ""
        path = Path(str(value))
        if path.is_absolute() or not self._ert_recipe_path:
            return str(path)
        return str(Path(self._ert_recipe_path).resolve().parent / path)

    def _on_ert_workflow_ok(self, result: WorkflowRunResult) -> None:
        summary = dict(result.summary)
        vtk = self._abs_path(summary.get("vtk"))
        if not vtk:
            vtk_ref = next(
                (artifact for artifact in result.artifacts if "vtk" in artifact.format),
                None,
            )
            vtk = self._abs_path(vtk_ref.path) if vtk_ref is not None else ""
        payload = {
            "mgr": result.objects.get("manager"),
            "chi2": result.metrics.get("chi2", float("nan")),
            "vtk": vtk,
            "metrics": dict(result.metrics),
            "convergence": result.objects.get("convergence") or [],
            "fixed_mgr": result.objects.get("fixed_manager"),
            "fixed_convergence": result.objects.get("fixed_convergence") or [],
            "fixed_metrics": dict(summary.get("fixed_metrics") or {}),
            "fixed_lambda": dict(summary.get("fixed_lambda") or {}),
            "fixed_vtk": self._abs_path(summary.get("fixed_vtk_path")),
            "lambda_requested": summary.get("lambda_requested"),
            "lambda_used": summary.get("lambda_used"),
            "lambda_trials": list(summary.get("lambda_trials") or []),
            "auto_lambda_status": summary.get("auto_lambda_status", "off"),
            "auto_lambda_note": summary.get("auto_lambda_note", ""),
            "data_error": dict(summary.get("data_error") or {}),
            "geometric_factors": dict(summary.get("geometric_factors") or {}),
            "cold_retry": dict(summary.get("cold_retry") or {}),
            "convergence_track": list(summary.get("convergence_track") or []),
            "outliers": dict(summary.get("outliers") or {}),
            "convergence_stop": summary.get("convergence_stop", ""),
            "engine": summary.get("engine", ""),
        }
        self._on_inversion_ok(payload)
        if hasattr(self.state, "update_workflow_result"):
            self.state.update_workflow_result(
                self.module_key,
                "ert.single_inversion",
                result.to_dict(),
                recipe_path=self._ert_recipe_path,
            )

    def _on_inversion_ok(self, result: dict) -> None:
        metrics = dict(result.get("metrics") or {})
        metrics.setdefault("method", self._instrument.currentText())
        note = str(result.get("auto_lambda_note") or "")
        status = str(result.get("auto_lambda_status") or "off")
        lam_used = result.get("lambda_used")
        lam_req = result.get("lambda_requested")
        trials = list(result.get("lambda_trials") or [])
        errors = dict(result.get("data_error") or {})
        geometry = dict(result.get("geometric_factors") or {})
        outliers = dict(result.get("outliers") or {})
        dropped = int(outliers.get("dropped") or 0)
        switched = (
            lam_used is not None and lam_req is not None
            and float(lam_used) != float(lam_req)
        )
        # The kept run is the one at the λ you set, on the full dataset. It is worth
        # keeping whenever either of those differs from what is on screen.
        keep_fixed = result.get("fixed_mgr") is not None and (switched or dropped > 0)

        # This panel sits above the convergence plot, so it stays to a couple of
        # short entries. Anything omitted here is in the log.
        extra: Dict[str, Any] = dict(metrics.get("extra") or {})
        if dropped:
            extra["data"] = f"{outliers.get('kept')} of {outliers.get('n_start')} kept"
            if outliers.get("limited_by_floor"):
                extra["data"] += " (floor reached)"
        # How chi2 responded to lambda is the one thing with no other home.
        if len(trials) > 1:
            extra["λ"] = " → ".join(
                f"{float(t['lambda']):g} (χ²{float(t['chi2']):.2f})" for t in trials
            )
        if extra:
            metrics["extra"] = extra
        # The whole iteration history, so the convergence plot can show every
        # stage rather than only the run that happened to finish last.
        track = list(result.get("convergence_track") or [])
        if track:
            metrics["convergence_track"] = track
        # Only warn in the panel; the routine outcome is already in the numbers.
        if geometry.get("repaired"):
            metrics["note"] = "Geometric factors were recomputed; see the log."
        elif geometry.get("checked") and not geometry.get("ok", True):
            metrics["note"] = "⚠ Geometric factors look wrong; the scale is off."
        elif status not in ("converged", "already_on_target", "off") and note:
            metrics["note"] = note

        choices: List[Dict[str, Any]] = []
        primary = result.get("mgr")
        if primary is not None:
            parts = []
            if lam_used is not None:
                parts.append(f"{'Auto λ' if switched else 'λ'} = {float(lam_used):g}")
            if dropped:
                parts.append(f"{outliers.get('kept')} data")
            chi2 = result.get("chi2")
            if chi2 == chi2:
                parts.append(f"χ² = {float(chi2):.2f}")
            choices.append({"label": "  ·  ".join(parts) or "Model", "mgr": primary,
                            "metrics": metrics,
                            "convergence": result.get("convergence") or [],
                            "vtk": result.get("vtk") or "", "chi2": chi2,
                            "lambda": lam_used})
        if keep_fixed:
            fixed_metrics = dict(result.get("fixed_metrics") or {})
            fixed_metrics.setdefault("method", self._instrument.currentText())
            fixed_info = dict(result.get("fixed_lambda") or {})
            reasons = []
            if switched:
                reasons.append(f"before the λ search moved to {float(lam_used):g}")
            if dropped:
                reasons.append(f"before {dropped} measurement(s) were rejected")
            fixed_metrics["note"] = (
                "The run at the settings you entered, kept for comparison: "
                + " and ".join(reasons) + "."
            )
            fixed_chi2 = fixed_info.get("chi2", fixed_metrics.get("chi2"))
            parts = [f"Yours: λ = {float(lam_req):g}"]
            if dropped:
                parts.append(f"{fixed_info.get('n_data', outliers.get('n_start'))} data")
            if fixed_chi2 is not None and fixed_chi2 == fixed_chi2:
                parts.append(f"χ² = {float(fixed_chi2):.2f}")
            choices.append({"label": "  ·  ".join(parts), "mgr": result.get("fixed_mgr"),
                            "metrics": fixed_metrics,
                            "convergence": result.get("fixed_convergence") or [],
                            "vtk": result.get("fixed_vtk") or "",
                            "chi2": fixed_chi2, "lambda": lam_req})

        self._inv_choices = choices
        self._lam_pick.blockSignals(True)
        self._lam_pick.clear()
        for choice in choices:
            self._lam_pick.addItem(choice["label"])
        self._lam_pick.setCurrentIndex(0)
        self._lam_pick.blockSignals(False)
        self._lam_pick_row.setVisible(len(choices) > 1)

        if choices:
            self._tl_step_row.setVisible(False)  # single result: no step selector
            self._show_lambda_choice(0)
            self._tabs.setCurrentWidget(self._model_tab)
            self._model_export_btn.setEnabled(True)
        else:
            self._inv_mgr = None
            self._quality_view.show_quality(
                metrics, result.get("convergence"), title="ERT inversion")

        # The library already logged the stage-by-stage detail while it ran. Only
        # add what it could not know, and only once: a second copy of the same
        # sentence is what turned this panel into a wall of text.
        chi2 = result.get("chi2")
        summary = f"ERT inversion complete: χ² = {chi2:.2f}" if chi2 == chi2 \
            else "ERT inversion complete"
        if lam_used is not None:
            summary += f" at λ = {float(lam_used):g}"
        if dropped:
            summary += f", {outliers.get('kept')}/{outliers.get('n_start')} data"
        self.log(summary + ".", "success")
        # A bad geometric factor rescales the whole section without touching chi2,
        # so this outranks anything the fit statistics say.
        if geometry.get("repaired"):
            factor = geometry.get("averted_factor")
            self.log("Geometric factors were recomputed"
                     + (f"; the section would otherwise be {float(factor):.2f}× off."
                        if factor else "."), "warn")
        elif geometry.get("checked") and not geometry.get("ok", True):
            factor = geometry.get("suspected_factor")
            self.log("Geometric factors look wrong"
                     + (f": the scale is about {1.0 / float(factor):.2f}× off."
                        if factor else "."), "error")
        if result.get("convergence_stop") == "iteration_cap":
            self.log("Still improving at the iteration ceiling; χ² is an upper bound.",
                     "warn")
        if outliers.get("limited_by_floor"):
            self.log(f"“Keep at least” capped the cut at {outliers.get('kept')} data; "
                     "outliers remain. Lower it, or QC the data first.", "warn")
        retry = dict(result.get("cold_retry") or {})
        if retry:
            self.log(f"Warm sweep stalled at χ² {retry['warm_chi2']:.1f}; a cold sweep "
                     + (f"reached {retry['cold_chi2']:.1f}." if retry.get("helped")
                        else "did not do better."), "warn")
        if keep_fixed:
            self.log("Your own settings are kept as the second entry in Model.", "info")
        vtk = result.get("vtk")
        if vtk:
            self.log(f"Saved {Path(vtk).name} to {Path(vtk).parent}", "info")

        mgr = result.get("mgr")
        if mgr is not None and hasattr(self.state, "register_geophysical_resource"):
            self.state.register_geophysical_resource(
                "ERT", "model", np.asarray(mgr.model, dtype=float),
                label="Latest ERT resistivity model", path=str(vtk or ""),
                metadata={"chi2": chi2, "mesh": getattr(mgr, "paraDomain", None),
                          "lambda": lam_used, "lambda_requested": lam_req,
                          "auto_lambda_status": status},
                resource_id="ert:model:latest",
            )
        self.report_result({"resistivity_vtk": vtk, "chi2": chi2,
                            "rrms": metrics.get("rrms"), "iterations": metrics.get("iterations"),
                            "num_measurements": metrics.get("n_data", self._n_meas),
                            "instrument": self._instrument.currentText(),
                            "engine": result.get("engine", ""),
                            "lambda": lam_used, "lambda_requested": lam_req,
                            "auto_lambda_status": status,
                            "auto_lambda_note": note,
                            "lambda_trials": trials,
                            "data_error": errors,
                            "geometric_factors": geometry,
                            "outliers": outliers,
                            "convergence_stop": result.get("convergence_stop", ""),
                            "convergence_track": track,
                            "fixed_lambda": dict(result.get("fixed_lambda") or {})})

    def _show_lambda_choice(self, index: int) -> None:
        """Display one of the kept single-inversion models (auto-λ or fixed λ)."""
        if not (0 <= index < len(self._inv_choices)):
            return
        choice = self._inv_choices[index]
        self._inv_mgr = choice["mgr"]
        if self._inv_mgr is not None:
            self._model_view.show_model(self._inv_mgr, kind="ert")
        self._quality_view.show_quality(
            choice["metrics"], choice["convergence"],
            title=f"ERT inversion — {choice['label']}")

    @staticmethod
    def row_label(widget):
        """The QFormLayout label paired with ``widget``, or None.

        Resolved through the widget's own parent rather than a stored layout, so
        moving a row between group boxes cannot silently leave its label behind.
        """
        parent = widget.parentWidget()
        form = parent.layout() if parent is not None else None
        return form.labelForField(widget) if isinstance(form, QFormLayout) else None

    def _set_rows_visible(self, widgets, on: bool) -> None:
        """Hide a form row, label included, since QFormLayout keeps them separate."""
        for widget in widgets:
            widget.setVisible(bool(on))
            label = self.row_label(widget)
            if label is not None:
                label.setVisible(bool(on))

    # -- imported mesh --------------------------------------------------------
    def _import_mesh(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Import inversion mesh", "",
            "Meshes (*.bms *.msh *.vtk *.vtu *.poly);;All files (*)")
        if path:
            self._apply_mesh_file(path)

    def _apply_mesh_file(self, path: str) -> str:
        """Load and check the mesh now, not when the inversion starts.

        A mesh whose origin or units are wrong fails deep in the forward solver,
        minutes into a run. Checking on import turns that into an immediate
        message next to the button that caused it.
        """
        from PyHydroGeophysX.inversion.ert_inversion import load_inversion_mesh

        # Checked against the loaded survey when there is one; importing before
        # loading data is allowed, and the run re-checks it either way.
        mesh = load_inversion_mesh(path, data=self._ert_data, log=lambda m: None)
        invertible = sum(1 for cell in mesh.cells() if cell.marker() > 1)
        self._mesh_path = str(path)
        self._mesh_clear.setEnabled(True)
        summary = (f"<b>{Path(path).name}</b>: {mesh.cellCount()} cells "
                   f"({invertible} inverted), {mesh.dim()}D.")
        self._mesh_note.setText(summary)
        self._sync_mesh_source()
        self.log(f"Imported inversion mesh {Path(path).name}: "
                 f"{mesh.cellCount()} cells, {invertible} inverted.", "success")
        return summary

    def _clear_mesh(self) -> None:
        self._mesh_path = ""
        self._mesh_clear.setEnabled(False)
        self._mesh_note.setText("Built from the electrode positions.")
        self._sync_mesh_source()

    def _sync_mesh_source(self) -> None:
        """An imported mesh describes its own domain, so the sizing knobs are dead."""
        generated = not self._mesh_path
        for widget in (self._quality, self._para_depth):
            widget.setEnabled(generated)
            label = self.row_label(widget)
            if label is not None:
                label.setEnabled(generated)

    def _on_auto_lambda(self, on: bool) -> None:
        """Show the auto-λ target and trial budget only while auto-λ is on."""
        self._set_rows_visible((self._chi2_row, self._lam_trials), on)

    def _on_reject_outliers(self, on: bool) -> None:
        """Show the rejection cut and the data floor only while rejection is on."""
        self._set_rows_visible((self._reject_row, self._min_keep), on)

    def _on_inversion_failed(self, message: str) -> None:
        self.log(f"ERT inversion failed: {message}", "error")

    def _reset_invert_button(self) -> None:
        if self._inv_busy is not None:
            self._inv_busy.finish()
            self._inv_busy = None
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
            # Same auto-λ switch as the single inversion; the trial budget is
            # smaller because each trial is a joint inversion over every step.
            "auto_lambda": bool(self._auto_lam.isChecked()),
            "target_chi2": float(self._target_chi2.value()),
            "chi2_tolerance": float(self._chi2_tol.value()),
            "max_lambda_trials": min(int(self._lam_trials.value()), 4),
        }
        if self._tl_lowmem.isChecked():
            params["save_memory"] = True
        times = self._tl_times if len(self._tl_times) == len(self._tl_files) else None
        output_base = Path(out_dir)
        bundle_dir = io_utils.ensure_dir(output_base / "qt_ert_timelapse")
        spec = WorkflowSpec(
            workflow_id="ert.timelapse_inversion",
            inputs={
                "data_files": [
                    ArtifactRef.from_path(
                        Path(path),
                        artifact_id=f"ert-timestep:{index}",
                        kind="ert_observations",
                        metadata={
                            "sequence_index": index,
                            "measurement_time": (
                                float(times[index]) if times is not None else index
                            ),
                        },
                    )
                    for index, path in enumerate(self._tl_files)
                ],
                "measurement_times": list(times or range(len(self._tl_files))),
            },
            parameters=params,
            metadata={"source": "qt", "sequence_order_persisted": True},
        )
        recipe_path, script_path = export_workflow_bundle(
            spec, bundle_dir, stem="ert_timelapse"
        )
        self._reproduce.set_bundle(recipe_path, script_path)
        self._tl_recipe_path = str(recipe_path)
        self._tl_busy = BusyStateController([self._tl_btn])
        self._tl_busy.start()
        self._tl_btn.setText("Inverting…")
        self._tl_progress.setVisible(True); self._tl_progress.setRange(0, 0)
        self.log(f"Starting {params['inversion_type']} time-lapse ERT inversion "
                 f"({len(self._tl_files)} steps)…", "info")
        self._tl_worker = WorkflowWorker(
            spec,
            RunContext(project_root=bundle_dir, output_dir=output_base),
        )
        self._tl_worker.logged.connect(lambda m: self.log(m, "info"))
        self._tl_worker.succeeded.connect(self._on_tl_workflow_ok)
        self._tl_worker.failed.connect(lambda message: self._on_tl_failed(message, False))
        self._tl_worker.finished.connect(self._reset_tl_button)
        self.register_worker(self._tl_worker)
        self._tl_worker.start()

    def _on_tl_workflow_ok(self, result: WorkflowRunResult) -> None:
        if hasattr(self.state, "update_workflow_result"):
            self.state.update_workflow_result(
                self.module_key,
                "ert.timelapse_inversion",
                result.to_dict(),
                recipe_path=self._tl_recipe_path,
            )
        self._on_tl_ok(result.legacy_payload())

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
                cov = cov_all[idx]  # raw log-coverage, matching ERTManager.coverage()
        title = self._tl_step_titles[idx] if idx < len(self._tl_step_titles) else f"Time step {idx + 1}"
        self._model_view.show_field(self._tl_mesh, values, kind="ert", coverage=cov, title=title)

    def _on_tl_failed(self, message: str, backend: bool) -> None:
        self.log(f"Time-lapse inversion {'unavailable' if backend else 'failed'}: {message}",
                 "warn" if backend else "error")

    def _reset_tl_button(self) -> None:
        if self._tl_busy is not None:
            self._tl_busy.finish()
            self._tl_busy = None
        self._tl_btn.setText("Run time-lapse inversion")
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
            selected = select_directory(
                self, "Export time-lapse results to folder",
                self.state.output_dir or Path.cwd(),
            )
            folder = str(selected) if selected else ""
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
        """Left-click selects the nearest electrode so its position can be read.

        Editing is deliberately not bound to the mouse; see the agent actions.
        """
        if event.button() != Qt.LeftButton:
            return
        if not self._plot.sceneBoundingRect().contains(event.scenePos()):
            return
        vp = self._plot.vb.mapSceneToView(event.scenePos())
        self._selected = self._nearest(float(vp.x()), float(vp.y()))
        self._refresh()

    def _delete(self, idx: int) -> None:
        for seq in (self._x, self._z, self._labels, self._electrode_origins):
            del seq[idx]
        self._selected = None
        self._refresh()

    def _clear(self) -> None:
        self._x, self._z, self._labels, self._electrode_origins = [], [], [], []
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
        folder = select_directory(
            self, "Export resistivity model to folder",
            self.state.output_dir or Path.cwd(),
        )
        if not folder:
            return
        try:
            import numpy as np

            out = io_utils.ensure_dir(folder)
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
                {"name": "list_electrodes", "args": {},
                 "desc": ("List electrodes as index, label, x, z, and whether each came "
                          "from the loaded file or was added afterwards.")},
                {"name": "add_electrode", "args": {"x": "float", "z": "float",
                                                   "label": "str (optional)"},
                 "desc": "Append an electrode at (x, z). It carries no measurements."},
                {"name": "move_electrode", "args": {"index": "int", "x": "float (optional)",
                                                    "z": "float (optional)"},
                 "desc": ("Move electrode `index` (0-based). Give x, z, or both; the "
                          "omitted coordinate is left alone.")},
                {"name": "delete_electrode", "args": {"index": "int"},
                 "desc": ("Remove electrode `index` (0-based). Measurements that "
                          "reference it are dropped when the data are serialized.")},
                {"name": "set_electrode_label", "args": {"index": "int", "label": "str"},
                 "desc": "Rename electrode `index` (0-based)."},
                {"name": "clear_electrodes", "args": {},
                 "desc": "Remove every electrode. Load a geometry file to start over."},
                {"name": "apply_filter", "args": {"min_rhoa": "float", "max_rhoa": "float", "max_error": "float (%)"},
                 "desc": "Filter measurements by apparent resistivity range and max relative error."},
                {"name": "set_params", "args": {"params": {"<key>": "value"}},
                 "desc": ("Set parameters. Shared by single + time-lapse inversion: lambda, "
                          "max_iterations, relative_error, mesh_quality, time_lapse (bool). "
                          "Mesh: para_depth (m, 0 = auto), mesh_file (path to a "
                          ".bms/.msh/.vtk/.poly mesh to invert on instead of a "
                          "generated one; the region to invert needs marker 2 or "
                          "above, and '' goes back to generating it). "
                          "Single-inversion error model: error_source (file/estimate/max), "
                          "absolute_error (Ohm). Convergence: plateau_tolerance (fraction), "
                          "max_total_iterations, engine (pyhydro/pygimli). "
                          "Geometric factors: geometric_factor_policy "
                          "('fix' recomputes k numerically when a homogeneous forward run "
                          "does not return the model resistivity, 'check' only reports, "
                          "'off' skips), geometric_factor_tolerance. "
                          "Outlier rejection: reject_outliers (bool), outlier_threshold "
                          "(sigma), outlier_passes, min_data_fraction. "
                          "Auto-lambda: auto_lambda (bool), target_chi2, chi2_tolerance, "
                          "max_lambda_trials. "
                          "Time-lapse-only: tl_alpha, tl_norm (L2/L1/L1L2), tl_windowed, "
                          "tl_window_size, tl_low_memory.")},
                {"name": "run_inversion", "args": {},
                 "desc": ("Run a single-time ERT inversion. Stages run in the order that "
                          "lowers chi2: fix the error model, iterate at the set lambda "
                          "until the misfit flattens, optionally reject data the model "
                          "cannot explain, and only then search for a lambda whose chi2 "
                          "lands inside target_chi2 +/- chi2_tolerance. The run at the "
                          "settings you gave is always kept for comparison.")},
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
            "list_electrodes": lambda: self._agent_list_electrodes(),
            "add_electrode": lambda: self._agent_add_electrode(
                args.get("x"), args.get("z"), args.get("label")),
            "move_electrode": lambda: self._agent_move_electrode(
                args.get("index"), args.get("x"), args.get("z")),
            "delete_electrode": lambda: self._agent_delete_electrode(args.get("index")),
            "set_electrode_label": lambda: self._agent_set_electrode_label(
                args.get("index"), args.get("label")),
            "clear_electrodes": lambda: self._agent_clear_electrodes(),
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
            "lambda_bounds": list(_LAMBDA_BOUNDS),
            "mesh_file": str(self._mesh_path or ""),
            "engine": self._engine.currentData(),
            "geometric_factor_policy": self._geom_policy,
            "error_source": self._err_source.currentData(),
            "relative_error": self._relerr.value(),
            "absolute_error": self._abserr.value(),
            "plateau_tolerance": self._plateau.value() / 100.0,
            "max_total_iterations": self._iter_ceiling.value(),
            "reject_outliers": self._reject.isChecked(),
            "outlier_threshold": self._reject_sigma.value(),
            "outlier_passes": self._reject_passes.value(),
            "min_data_fraction": self._min_keep.value() / 100.0,
            "auto_lambda": self._auto_lam.isChecked(),
            "target_chi2": self._target_chi2.value(),
            "chi2_tolerance": self._chi2_tol.value(),
            "max_lambda_trials": self._lam_trials.value(),
            "has_model": getattr(self, "_inv_mgr", None) is not None,
            "models_available": [c["label"] for c in self._inv_choices],
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

    # -- electrode editing (no UI panel; these are the whole surface) --------
    def _electrode_index(self, index: Any) -> int:
        """Validate a 0-based electrode index, raising with the allowed range."""
        try:
            idx = int(index)
        except (TypeError, ValueError):
            raise ValueError(f"'index' must be an integer, got {index!r}")
        if not self._x:
            raise ValueError("No electrodes are loaded.")
        if not -len(self._x) <= idx < len(self._x):
            raise ValueError(f"index {idx} out of range for {len(self._x)} electrodes")
        return idx % len(self._x)

    def _agent_list_electrodes(self) -> Dict[str, Any]:
        return {
            "status": "ok",
            "count": len(self._x),
            "electrodes": [
                {"index": i, "label": self._labels[i], "x": float(self._x[i]),
                 "z": float(self._z[i]),
                 "from_file": self._electrode_origins[i] is not None}
                for i in range(len(self._x))
            ],
        }

    def _agent_add_electrode(self, x: Any, z: Any, label: Any = None) -> Dict[str, Any]:
        if x is None or z is None:
            return {"status": "failed", "error": "Provide both 'x' and 'z'."}
        try:
            xv, zv = float(x), float(z)
        except (TypeError, ValueError):
            return {"status": "failed", "error": "'x' and 'z' must be numbers."}
        self._x.append(xv)
        self._z.append(zv)
        self._labels.append(str(label) if label else f"new-{len(self._x)}")
        self._electrode_origins.append(None)  # carries no measurements
        self._refresh()
        self.log(f"Added electrode {len(self._x) - 1} at x={xv:g}, z={zv:g}.", "info")
        return {"status": "ok", "index": len(self._x) - 1, "electrodes": len(self._x)}

    def _agent_move_electrode(self, index: Any, x: Any = None, z: Any = None) -> Dict[str, Any]:
        if x is None and z is None:
            return {"status": "failed", "error": "Provide 'x', 'z', or both."}
        try:
            idx = self._electrode_index(index)
            if x is not None:
                self._x[idx] = float(x)
            if z is not None:
                self._z[idx] = float(z)
        except (ValueError, TypeError) as exc:
            return {"status": "failed", "error": str(exc)}
        self._refresh()
        self.log(f"Moved electrode {idx} to x={self._x[idx]:g}, z={self._z[idx]:g}.", "info")
        return {"status": "ok", "index": idx,
                "x": float(self._x[idx]), "z": float(self._z[idx])}

    def _agent_delete_electrode(self, index: Any) -> Dict[str, Any]:
        try:
            idx = self._electrode_index(index)
        except (ValueError, TypeError) as exc:
            return {"status": "failed", "error": str(exc)}
        label = self._labels[idx]
        self._delete(idx)
        self.log(f"Deleted electrode {idx} ({label}); {len(self._x)} remain. "
                 "Measurements referencing it are dropped at inversion time.", "warn")
        return {"status": "ok", "deleted": idx, "electrodes": len(self._x)}

    def _agent_set_electrode_label(self, index: Any, label: Any) -> Dict[str, Any]:
        if label is None or not str(label).strip():
            return {"status": "failed", "error": "Provide a non-empty 'label'."}
        try:
            idx = self._electrode_index(index)
        except (ValueError, TypeError) as exc:
            return {"status": "failed", "error": str(exc)}
        self._labels[idx] = str(label)
        self._refresh()
        return {"status": "ok", "index": idx, "label": self._labels[idx]}

    def _agent_clear_electrodes(self) -> Dict[str, Any]:
        removed = len(self._x)
        self._clear()
        self.log(f"Cleared {removed} electrode(s).", "warn")
        return {"status": "ok", "removed": removed}

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

        def set_geom_policy(value):
            allowed = ("off", "check", "fix")
            key = str(value).strip().lower()
            if key not in allowed:
                raise ValueError(f"must be one of {list(allowed)}")
            self._geom_policy = key

        def set_combo_data(combo, value):
            """Match on the stable itemData key, not on the display label."""
            keys = [combo.itemData(i) for i in range(combo.count())]
            key = str(value).strip().lower()
            for index, candidate in enumerate(keys):
                if str(candidate).lower() == key:
                    combo.setCurrentIndex(index)
                    return
            raise ValueError(f"must be one of {keys}")

        handlers = {
            # shared by single + time-lapse inversion
            "lambda": lambda v: self._lam.setValue(float(v)),
            "max_iterations": lambda v: self._iter.setValue(int(v)),
            "relative_error": lambda v: self._relerr.setValue(float(v)),
            "mesh_quality": lambda v: self._quality.setValue(float(v)),
            "para_depth": lambda v: self._para_depth.setValue(float(v)),
            "mesh_file": lambda v: (self._apply_mesh_file(str(v)) if str(v)
                                    else self._clear_mesh()),
            "time_lapse": lambda v: self._tl_mode.setChecked(bool(v)),
            # single-inversion fit assistance
            "engine": lambda v: set_combo_data(self._engine, v),
            "geometric_factor_policy": lambda v: set_geom_policy(v),
            "error_source": lambda v: set_combo_data(self._err_source, v),
            "absolute_error": lambda v: self._abserr.setValue(float(v)),
            "plateau_tolerance": lambda v: self._plateau.setValue(float(v) * 100.0),
            "max_total_iterations": lambda v: self._iter_ceiling.setValue(int(v)),
            "reject_outliers": lambda v: self._reject.setChecked(bool(v)),
            "outlier_threshold": lambda v: self._reject_sigma.setValue(float(v)),
            "outlier_passes": lambda v: self._reject_passes.setValue(int(v)),
            "min_data_fraction": lambda v: self._min_keep.setValue(float(v) * 100.0),
            "auto_lambda": lambda v: self._auto_lam.setChecked(bool(v)),
            "target_chi2": lambda v: self._target_chi2.setValue(float(v)),
            "chi2_tolerance": lambda v: self._chi2_tol.setValue(float(v)),
            "max_lambda_trials": lambda v: self._lam_trials.setValue(int(v)),
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
                "lambda": self._lam.value(),
                "auto_lambda": self._auto_lam.isChecked(),
                "target_chi2": self._target_chi2.value(),
                "chi2_tolerance": self._chi2_tol.value(),
                "max_lambda_trials": self._lam_trials.value()}

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
