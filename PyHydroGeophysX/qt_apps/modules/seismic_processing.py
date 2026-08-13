"""Seismic processing module.

Loads real field formats (SEG-Y, Geometrics DAT, SEG-2) by reusing
``PyHydroGeophysX.data_processing.seismic`` plus generic ``.npy/.csv`` arrays,
lets the user browse shot gathers, apply display/QC processing (gain, clip,
polarity, trace normalization, AGC), pick first arrivals (manual + assisted),
and export picks to CSV and a PyGIMLi travel-time ``.dat`` file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import Qt

from PyHydroGeophysX._internal.utils import velocity_of
from PyHydroGeophysX.inversion.lambda_search import LAMBDA_BOUNDS as _LAMBDA_BOUNDS
from PyHydroGeophysX.qt_apps import io_utils, theme
from PyHydroGeophysX.qt_apps.modules.base import BaseModule, LogFn
from PyHydroGeophysX.qt_apps.qt_utils import (
    BusyStateController,
    Debouncer,
    ReproduceBar,
    make_double_spinbox,
    make_spinbox,
    merged_row,
    select_directory,
    set_rows_enabled,
)
from PyHydroGeophysX.qt_apps.widgets.mesh_view import MeshResultView
from PyHydroGeophysX.qt_apps.widgets.quality_view import InversionQualityView
from PyHydroGeophysX.qt_apps.widgets.seismic_viewer import SeismicViewer, first_arrival_onsets
from PyHydroGeophysX.qt_apps.workers import TaskWorker, WorkflowWorker
from PyHydroGeophysX.workflows import (
    ArtifactRef,
    RunContext,
    WorkflowRunResult,
    WorkflowSpec,
    export_workflow_bundle,
)

try:
    from PyHydroGeophysX.data_processing.seismic import (
        FirstBreakPick,
        apply_agc,
        export_first_breaks,
        export_traveltime_container,
        first_breaks_to_traveltime,
        normalize_traces,
        read_geometrics_dat,
        read_segy,
    )
    from PyHydroGeophysX.data_processing.field_formats import read_seg2_seismic

    _SEISMIC_OK = True
    _SEISMIC_ERR = ""
except Exception as _exc:  # noqa: BLE001 - degrade to array-only mode
    _SEISMIC_OK = False
    _SEISMIC_ERR = str(_exc)

_FILE_FILTER = (
    "Seismic (*.sgy *.segy *.dat *.sg2 *.seg2 *.npy *.npz *.csv *.txt);;"
    "SEG-Y (*.sgy *.segy);;Geometrics DAT (*.dat);;SEG-2 (*.sg2 *.seg2);;"
    "Array (*.npy *.npz *.csv *.txt);;All files (*)"
)


class SeismicProcessingModule(BaseModule):
    module_key = "seismic_processing"
    module_title = "Seismic Processing"

    def __init__(self, state: Any, log: LogFn, parent=None) -> None:
        super().__init__(state, log, parent)
        self._raw: Optional[np.ndarray] = None
        self._dt: Optional[float] = None
        self._dataset = None
        self._current_gather = None
        self._headers = None
        self._source_path: Optional[Path] = None
        self._picks: Dict[int, Any] = {}
        self._order: List[int] = []
        self._pick_src: Dict[int, str] = {}
        self._current_record: Optional[int] = None
        self._shot_pos: Dict[int, float] = {}
        self._all_picks: Dict[int, Dict[int, Any]] = {}
        self._all_src: Dict[int, Dict[int, str]] = {}
        self._geo_positions: Optional[Dict[int, Tuple[float, float]]] = None  # trace idx -> (x, z)
        self._shot_spacing: Optional[float] = None  # regular shot interval (m); auto-fills shot_x per record
        self._shot0_x: float = 0.0  # x of the first record's shot
        self._srt_worker: Optional[WorkflowWorker] = None
        self._srt_busy: Optional[BusyStateController] = None
        self._srt_spec: Optional[WorkflowSpec] = None
        self._srt_recipe_path: str = ""
        self._tt_data = None                 # uploaded pre-picked travel-time DataContainer
        self._tt_path: Optional[Path] = None
        self._load_worker: Optional[TaskWorker] = None
        self._load_busy: Optional[BusyStateController] = None
        self._proc_cache: Optional[np.ndarray] = None
        self._proc_key: Optional[tuple] = None
        self._recompute_debounced = Debouncer(self._recompute, 80)

        root = QHBoxLayout(self)
        self._viewer = SeismicViewer()
        self._viewer.pointPicked.connect(self._on_point_picked)
        self._viewer.linePicked.connect(self._on_line_picked)
        self._center_tabs = QTabWidget()
        self._center_tabs.addTab(self._viewer, "Gather")
        self._tt_widget = pg.PlotWidget()
        self._tt_widget.setBackground("w")
        self._tt_widget.showGrid(x=True, y=True, alpha=0.3)
        self._tt_widget.setLabel("bottom", "geophone position x (m)")
        self._tt_widget.setLabel("left", "travel time (ms)")
        self._tt_plot = self._tt_widget.getPlotItem()
        self._tt_plot.addLegend()
        self._center_tabs.addTab(self._tt_widget, "Travel-time")
        self._vel_view = MeshResultView()
        self._center_tabs.addTab(self._vel_view, "Velocity model")
        self._quality_view = InversionQualityView()
        self._center_tabs.addTab(self._quality_view, "Inversion quality")
        self._reproduce = ReproduceBar()
        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.addWidget(self._center_tabs, stretch=1)
        center_layout.addWidget(self._reproduce)
        root.addWidget(center, stretch=1)
        self._processing_panel = self._build_controls()
        root.addWidget(self._processing_panel)
        self._inversion_panel = self._build_inversion_panel()
        root.addWidget(self._inversion_panel)
        self._center_tabs.currentChanged.connect(self._on_center_tab_changed)
        self._on_center_tab_changed()
        if not _SEISMIC_OK:
            self.log(f"Field-format readers unavailable ({_SEISMIC_ERR}); array files only.", "warn")

    # -- controls ------------------------------------------------------------
    def _build_controls(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        load_btn = QPushButton("Load seismic data…")
        load_btn.setIcon(theme.icon("fa5s.folder-open"))
        load_btn.clicked.connect(self._load_gather)
        layout.addWidget(load_btn)
        self._load_btn = load_btn
        formats = "SEG-Y, Geometrics DAT, SEG-2, NPY/CSV" if _SEISMIC_OK else "NPY/NPZ/CSV/TXT"
        hint = QLabel(f"Formats: {formats}")
        hint.setStyleSheet("color:#5a6a7a; font-size:8pt;")
        layout.addWidget(hint)
        self._info = QLabel("No data loaded.")
        self._info.setWordWrap(True)
        layout.addWidget(self._info)

        pos_btn = QPushButton("Load geophone positions / topography…")
        pos_btn.setIcon(theme.icon("fa5s.map-marker-alt"))
        pos_btn.setToolTip("Load a text file of per-geophone x distance and elevation "
                           "(columns: station distance_m elevation_m, or x z). Picks then carry real "
                           "positions + topography into the SRT inversion.")
        pos_btn.clicked.connect(self._load_geometry_dialog)
        layout.addWidget(pos_btn)
        self._geo_info = QLabel("Even spacing (no position file).")
        self._geo_info.setStyleSheet("color:#5a6a7a; font-size:8pt;")
        self._geo_info.setWordWrap(True)
        layout.addWidget(self._geo_info)

        self._shot_group = QGroupBox("Geometry")
        sform = QFormLayout(self._shot_group)
        self._shot_combo = QComboBox()
        self._shot_combo.currentIndexChanged.connect(self._on_shot_changed)
        sform.addRow("Record", self._shot_combo)
        self._spacing = QDoubleSpinBox()
        self._spacing.setRange(0.05, 1000.0); self._spacing.setValue(1.0); self._spacing.setSuffix(" m")
        sform.addRow("Geophone spacing", self._spacing)
        self._geo_start = QDoubleSpinBox()
        self._geo_start.setRange(-100000.0, 100000.0); self._geo_start.setValue(0.0); self._geo_start.setSuffix(" m")
        sform.addRow("Geophone 0 x", self._geo_start)
        self._shot_x = QDoubleSpinBox()
        self._shot_x.setRange(-100000.0, 100000.0); self._shot_x.setValue(0.0); self._shot_x.setSuffix(" m")
        self._shot_x.setToolTip("Shot position for this record; may be before geophone 0 (negative).")
        self._shot_x.valueChanged.connect(self._on_shot_x_changed)
        sform.addRow("Shot x (this record)", self._shot_x)
        self._shot_group.setVisible(False)
        layout.addWidget(self._shot_group)

        proc = QGroupBox("Display / processing")
        form = QFormLayout(proc)
        self._gain = QSlider(Qt.Horizontal)
        self._gain.setRange(1, 100)
        self._gain.setValue(10)
        self._gain.valueChanged.connect(self._recompute_debounced.trigger)
        form.addRow("Gain", self._gain)
        self._clip = QDoubleSpinBox()
        self._clip.setRange(80.0, 100.0)
        self._clip.setValue(99.0)
        self._clip.setSuffix(" %ile")
        self._clip.valueChanged.connect(self._recompute_debounced.trigger)
        form.addRow("Clip", self._clip)
        self._polarity = QCheckBox("Flip polarity")
        self._polarity.setChecked(True)
        self._polarity.toggled.connect(self._recompute)
        form.addRow(self._polarity)
        self._normalize = QCheckBox("Normalize trace")
        self._normalize.setChecked(True)
        self._normalize.toggled.connect(self._recompute)
        form.addRow(self._normalize)
        self._agc = QCheckBox("AGC")
        self._agc.setChecked(True)
        self._agc.toggled.connect(self._recompute)
        self._agc_window = QDoubleSpinBox()
        self._agc_window.setRange(5.0, 500.0)
        self._agc_window.setValue(50.0)
        self._agc_window.setSuffix(" ms")
        self._agc_window.valueChanged.connect(self._recompute_debounced.trigger)
        agc_row = QHBoxLayout()
        agc_row.addWidget(self._agc)
        agc_row.addWidget(self._agc_window)
        agc_wrap = QWidget(); agc_wrap.setLayout(agc_row)
        form.addRow(agc_wrap)
        layout.addWidget(proc)

        picks = QGroupBox("First-arrival picking")
        pbox = QVBoxLayout(picks)
        self._pick_mode = QCheckBox("Manual pick mode (click a trace)")
        self._pick_mode.toggled.connect(self._viewer.set_pick_mode)
        pbox.addWidget(self._pick_mode)
        tip = QLabel("Tip: Ctrl+drag draws a line and picks every trace it crosses.")
        tip.setWordWrap(True)
        tip.setStyleSheet("color:#5a6a7a; font-size:8pt;")
        pbox.addWidget(tip)
        tform = QFormLayout()
        self._threshold = QDoubleSpinBox()
        self._threshold.setRange(2.0, 15.0)
        self._threshold.setSingleStep(0.5)
        self._threshold.setValue(5.0)
        self._threshold.setToolTip("STA/LTA energy-ratio threshold for onset detection (lower = more sensitive / earlier).")
        tform.addRow("STA/LTA ratio", self._threshold)
        pbox.addLayout(tform)
        auto_btn = QPushButton("Auto-pick first breaks")
        auto_btn.setIcon(theme.icon("fa5s.magic"))
        auto_btn.clicked.connect(self._auto_pick)
        pbox.addWidget(auto_btn)
        row = QHBoxLayout()
        undo_btn = QPushButton("Undo")
        undo_btn.setIcon(theme.icon("fa5s.undo"))
        undo_btn.clicked.connect(self._undo_pick)
        clear_btn = QPushButton("Clear")
        clear_btn.setIcon(theme.icon("fa5s.eraser"))
        clear_btn.clicked.connect(self._clear_picks_and_publish)
        row.addWidget(undo_btn)
        row.addWidget(clear_btn)
        pbox.addLayout(row)
        export_btn = QPushButton("Export picks CSV…")
        export_btn.setProperty("primary", True)
        export_btn.setIcon(theme.icon("fa5s.file-export", color="#ffffff"))
        export_btn.clicked.connect(self._export_picks)
        pbox.addWidget(export_btn)
        tt_btn = QPushButton("Export travel-time .dat…")
        tt_btn.setIcon(theme.icon("fa5s.project-diagram"))
        tt_btn.clicked.connect(self._export_traveltime)
        pbox.addWidget(tt_btn)
        self._pick_info = QLabel("0 picks")
        pbox.addWidget(self._pick_info)
        layout.addWidget(picks)

        # Everything to do with travel times, including where they come from,
        # lives in the inversion column and appears with the Travel-time tab.
        # This strip is the work done on the Gather tab: load, display, pick.
        layout.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        # Wide enough to fit the controls without a horizontal scrollbar.
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(450)
        scroll.setMaximumWidth(500)
        scroll.setWidget(panel)
        return scroll

    def _build_inversion_panel(self) -> QScrollArea:
        """The travel-time column: where the times come from, and how to invert them.

        Shown with the Travel-time tab and hidden on the Gather tab, so the two
        halves of the work, getting picks and inverting them, do not compete for
        one scroll.
        """
        panel = QWidget()
        layout = QVBoxLayout(panel)

        src = QGroupBox("Travel times to invert")
        srcbox = QVBoxLayout(src)
        acc = QLabel("Pick first breaks across shots and set each shot's x on the "
                     "Gather tab, or skip picking and upload pre-picked times.")
        acc.setWordWrap(True)
        acc.setStyleSheet("color:#5a6a7a; font-size:8pt;")
        srcbox.addWidget(acc)

        up_row = QHBoxLayout()
        self._tt_upload_btn = QPushButton("Upload travel times…")
        self._tt_upload_btn.setIcon(theme.icon("fa5s.file-upload"))
        self._tt_upload_btn.setToolTip(
            "Load a pre-picked travel-time file and invert it directly (no picking). "
            "Formats: pyGIMLi/BERT .sgt/.dat (sensors + 's g t'), or a CSV/text with columns "
            "source_x, receiver_x, time  (or source_x, source_z, receiver_x, receiver_z, time). "
            "Times in seconds (milliseconds auto-detected).")
        self._tt_upload_btn.clicked.connect(self._upload_traveltime)
        self._tt_clear_btn = QPushButton("✕")
        self._tt_clear_btn.setToolTip("Clear the uploaded travel times (go back to using picks).")
        self._tt_clear_btn.setMaximumWidth(32)
        self._tt_clear_btn.setEnabled(False)
        self._tt_clear_btn.clicked.connect(self._clear_traveltime)
        up_row.addWidget(self._tt_upload_btn)
        up_row.addWidget(self._tt_clear_btn)
        srcbox.addLayout(up_row)
        # One line, written in one place. Two labels describing the same thing
        # drift the moment either path forgets to update the other.
        self._tt_status = QLabel()
        self._tt_status.setWordWrap(True)
        self._tt_status.setStyleSheet("color:#5a6a7a; font-size:8pt;")
        srcbox.addWidget(self._tt_status)
        layout.addWidget(src)

        layout.addWidget(self._build_srt_inversion_group())
        layout.addWidget(self._build_srt_assist_group())

        run = QGroupBox("Run")
        runbox = QVBoxLayout(run)
        self._srt_btn = QPushButton("Run SRT inversion")
        self._srt_btn.setProperty("primary", True)
        self._srt_btn.setIcon(theme.icon("fa5s.layer-group", color="#ffffff"))
        self._srt_btn.clicked.connect(self._run_srt)
        runbox.addWidget(self._srt_btn)
        self._srt_progress = QProgressBar()
        self._srt_progress.setVisible(False)
        runbox.addWidget(self._srt_progress)
        self._srt_export_btn = QPushButton("Export velocity model…")
        self._srt_export_btn.setIcon(theme.icon("fa5s.cube"))
        self._srt_export_btn.setToolTip(
            "Write the recovered velocity model as npy, the mesh, and a VTK file.")
        self._srt_export_btn.setEnabled(False)
        self._srt_export_btn.clicked.connect(self._export_velocity_model)
        runbox.addWidget(self._srt_export_btn)
        layout.addWidget(run)
        layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # Narrower than the processing strip: these rows are short, and the
        # gather still needs room between the two columns.
        scroll.setMinimumWidth(380)
        scroll.setMaximumWidth(420)
        scroll.setWidget(panel)
        scroll.setVisible(False)
        return scroll

    def _on_center_tab_changed(self, _index: int = 0) -> None:
        """One side panel per tab: the controls for what is on screen, nothing else.

        Gather shows loading, display and picking. Travel-time and the two
        result tabs show the inversion instead: gain, clip and STA/LTA decide
        nothing once the picks exist, and leaving them up costs the width the
        plot wants. The result tabs keep the inversion side so the settings that
        produced a model stay readable next to it and a re-run does not mean
        navigating back.
        """
        on_gather = self._center_tabs.currentWidget() is self._viewer
        self._processing_panel.setVisible(on_gather)
        self._inversion_panel.setVisible(not on_gather)

    def _has_traveltimes(self) -> bool:
        return self._tt_data is not None or any(self._all_picks.values()) \
            or bool(self._picks)

    def _describe_traveltime_source(self) -> str:
        """What the Run button would invert, said in one line."""
        if self._tt_data is not None:
            name = self._tt_path.name if self._tt_path else "an uploaded file"
            return (f"<b>Using {int(self._tt_data.size())} uploaded travel "
                    f"times</b> from {name}.")
        counted = sum(len(picks) for picks in self._all_picks.values()) or len(self._picks)
        records = sum(1 for picks in self._all_picks.values() if picks)
        if not counted:
            return "Inversion source: picks (none yet)."
        return (f"Inversion source: {counted} picks"
                + (f" across {records} records." if records > 1 else "."))

    # -- SRT inversion controls ----------------------------------------------
    # Grouped the same way as the ERT page: what defines the run, then what the
    # software is allowed to change on its own. The travel-time inversion now
    # shares ERT's stopping rule, plateau continuation, and lambda search, so it
    # earns the same controls rather than running entirely on library defaults.
    def _build_srt_inversion_group(self) -> QGroupBox:
        box = QGroupBox("Inversion")
        form = QFormLayout(box)

        self._srt_engine = QComboBox()
        for label, value in (("In-house Gauss-Newton", "pyhydro"),
                             ("PyGIMLi TravelTimeManager", "pygimli")):
            self._srt_engine.addItem(label, value)
        self._srt_engine.setToolTip(
            "Solver. The in-house Gauss-Newton inversion exposes its own stopping "
            "rule, so the fit assistance below can drive it; the PyGIMLi manager "
            "runs once and is the historical default.")
        self._srt_engine.currentIndexChanged.connect(self._sync_srt_engine)
        form.addRow("Engine", self._srt_engine)

        self._srt_lam = QDoubleSpinBox()
        self._srt_lam.setDecimals(3)
        self._srt_lam.setRange(*_LAMBDA_BOUNDS)
        self._srt_lam.setValue(50.0)
        self._srt_lam.setStepType(QAbstractSpinBox.StepType.AdaptiveDecimalStepType)
        self._srt_lam.setToolTip(
            "Smoothness of the velocity model. Lower fits the travel times harder, "
            "higher gives a smoother model. Start on the smooth side: the search "
            "relaxes downward, continuing each λ from the previous solution.")
        form.addRow("Lambda", self._srt_lam)

        self._srt_iter = make_spinbox(20, 2, 60, tooltip=(
            "Iterations per attempt. A run that uses all of them while still "
            "improving is continued from its own model, up to the ceiling beside "
            "it, so λ is never blamed for an unfinished descent."))
        self._srt_iter_ceiling = make_spinbox(60, 5, 400, tooltip=(
            "Total iterations allowed at one λ, counting continuations. Reaching "
            "it means the reported χ² is an upper bound, and the log says so."))
        self._srt_iter_row = merged_row(
            self._srt_iter, "per pass, up to", self._srt_iter_ceiling)
        form.addRow("Iterations", self._srt_iter_row)

        self._srt_plateau = QDoubleSpinBox()
        self._srt_plateau.setRange(0.01, 10.0)
        self._srt_plateau.setDecimals(2)
        self._srt_plateau.setSingleStep(0.1)
        self._srt_plateau.setValue(0.5)
        self._srt_plateau.setSuffix(" %")
        self._srt_plateau.setToolTip(
            "A λ is finished once χ² improves by less than this per iteration.")
        form.addRow("Stop below", self._srt_plateau)

        self._srt_quality = make_double_spinbox(32.0, 20.0, 40.0, 1.0, 1)
        self._srt_quality.setToolTip(
            "Inversion-mesh quality: the minimum triangle angle. Higher gives a "
            "finer, better-conditioned triangulation at more cost per iteration.")
        form.addRow("Mesh quality", self._srt_quality)

        self._srt_para_depth = make_double_spinbox(0.0, 0.0, 10000.0, 5.0, 1,
                                                   suffix=" m")
        self._srt_para_depth.setSpecialValueText("auto")
        self._srt_para_depth.setToolTip(
            "How deep to invert. PyGIMLi sizes the domain from the array length, "
            "which for refraction reaches well past where any ray turns, so the "
            "deep cells are unconstrained and only slow the run. Cap it when the "
            "ray-path plot shows the bottom of the section is empty.")
        form.addRow("Invert to depth", self._srt_para_depth)

        self._srt_cell_size = make_double_spinbox(0.0, 0.0, 1000.0, 0.5, 2,
                                                  suffix=" m²")
        self._srt_cell_size.setSpecialValueText("auto")
        self._srt_cell_size.setToolTip(
            "Largest cell in the inverted domain. Smaller resolves more detail "
            "and adds unknowns; leave at auto unless the model looks blocky "
            "against what the ray coverage supports.")
        form.addRow("Max cell size", self._srt_cell_size)

        self._srt_sec_nodes = make_spinbox(3, 1, 10, tooltip=(
            "Extra nodes placed along cell edges for the ray tracer. They sharpen "
            "the computed travel times without adding unknowns to the inversion, "
            "so raise this when the fit stalls on a coarse mesh."))
        form.addRow("Secondary nodes", self._srt_sec_nodes)
        return box

    def _build_srt_assist_group(self) -> QGroupBox:
        box = QGroupBox("Fit assistance")
        form = QFormLayout(box)

        self._srt_auto_lam = QCheckBox("Auto-λ: re-invert to reach target χ²")
        self._srt_auto_lam.setChecked(True)
        self._srt_auto_lam.setToolTip(
            "The inversion at the λ above always runs first and is always kept. If "
            "its χ² misses the target band, the same mesh is re-inverted at other "
            "λ values and the closest one becomes the displayed model. Each trial "
            "continues from the nearest λ already solved, so the later ones are "
            "cheap, but every trial is still a full inversion.")
        self._srt_auto_lam.toggled.connect(self._sync_srt_engine)
        form.addRow(self._srt_auto_lam)

        self._srt_target_chi2 = make_double_spinbox(1.0, 0.1, 100.0, 0.1, 2)
        self._srt_target_chi2.setToolTip(
            "χ² = 1 means the model explains the picks to within their assumed "
            "error. Raise it if the picks are noisier than that admits.")
        self._srt_chi2_tol = make_double_spinbox(0.2, 0.01, 10.0, 0.05, 2)
        self._srt_chi2_tol.setToolTip(
            "Half-width of the accepted band. The search stops as soon as a trial "
            "lands inside target ± tolerance.")
        self._srt_chi2_row = merged_row(
            self._srt_target_chi2, "±", self._srt_chi2_tol)
        form.addRow("Target χ²", self._srt_chi2_row)

        self._srt_lam_trials = make_spinbox(6, 1, 20, tooltip=(
            "Upper bound on the extra inversions the λ search may run, on top of "
            "the one at your λ. Reached only when the target stays out of range."))
        form.addRow("Max λ trials", self._srt_lam_trials)
        self._sync_srt_engine()  # establish the interlock before the page shows
        return box

    def _sync_srt_engine(self) -> None:
        """Only the in-house engine reports the per-iteration state the search needs."""
        in_house = str(self._srt_engine.currentData()) == "pyhydro"
        self._srt_auto_lam.setEnabled(in_house)
        if not in_house and self._srt_auto_lam.isChecked():
            self._srt_auto_lam.setChecked(False)
        set_rows_enabled(
            [self._srt_iter_row, self._srt_plateau], in_house)
        searching = in_house and self._srt_auto_lam.isChecked()
        set_rows_enabled([self._srt_chi2_row, self._srt_lam_trials], searching)

    def _set_srt_engine(self, value: str) -> None:
        index = self._srt_engine.findData(str(value).strip().lower())
        if index < 0:
            raise ValueError(
                f"engine must be pyhydro or pygimli; got {value!r}.")
        self._srt_engine.setCurrentIndex(index)

    def _collect_srt_params(self) -> Dict[str, Any]:
        return {
            "engine": str(self._srt_engine.currentData()),
            "lam": float(self._srt_lam.value()),
            "max_iterations": int(self._srt_iter.value()),
            "max_total_iterations": int(self._srt_iter_ceiling.value()),
            "plateau_tolerance": float(self._srt_plateau.value()) / 100.0,
            "mesh_quality": float(self._srt_quality.value()),
            "para_depth": float(self._srt_para_depth.value()),
            "para_max_cell_size": float(self._srt_cell_size.value()),
            "secondary_nodes": int(self._srt_sec_nodes.value()),
            "auto_lambda": bool(self._srt_auto_lam.isChecked()),
            "target_chi2": float(self._srt_target_chi2.value()),
            "chi2_tolerance": float(self._srt_chi2_tol.value()),
            "max_lambda_trials": int(self._srt_lam_trials.value()),
        }

    # -- loading -------------------------------------------------------------
    def _load_gather(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load seismic data", "", _FILE_FILTER)
        if not path:
            return
        self._load_busy = BusyStateController([self._load_btn])
        self._load_busy.start()
        self._load_btn.setText("Loading…")
        self._info.setText(f"Loading {Path(path).name}…")
        worker = TaskWorker(self._parse_seismic, path)
        worker.succeeded.connect(lambda res: self._on_seismic_loaded(path, res))
        worker.failed.connect(self._on_seismic_load_failed)
        worker.finished.connect(self._reset_load_btn)
        self._load_worker = self.register_worker(worker)
        worker.start()

    def _parse_seismic(self, path):
        """Parse a seismic file off the UI thread. Returns a plain dict for the slot."""
        p = Path(path)
        suffix = p.suffix.lower()
        if _SEISMIC_OK and suffix in (".sgy", ".segy"):
            return {"kind": "dataset", "dataset": read_segy(str(p), max_traces=4000), "warning": ""}
        if _SEISMIC_OK and suffix == ".dat":
            try:
                return {"kind": "dataset", "dataset": read_geometrics_dat(str(p)), "warning": ""}
            except Exception as exc:  # noqa: BLE001
                return {"kind": "raw", "arr": io_utils.load_2d_array(p), "dt": None,
                        "warning": f"Not a Geometrics DAT ({exc}); read as a text matrix."}
        if _SEISMIC_OK and suffix in (".sg2", ".seg2"):
            arr, dt = self._read_seg2(p)
            return {"kind": "raw", "arr": arr, "dt": dt, "warning": ""}
        return {"kind": "raw", "arr": io_utils.load_2d_array(p), "dt": None, "warning": ""}

    def _on_seismic_loaded(self, path: str, res: dict) -> None:
        if res.get("warning"):
            self.log(res["warning"], "warn")
        if res["kind"] == "dataset":
            self._set_dataset(res["dataset"])
        else:
            self._set_raw(res["arr"], dt=res["dt"])
        self._source_path = Path(path)
        self._clear_picks()
        self._recompute()
        self._update_info()
        self.log(f"Loaded {Path(path).name}", "success")

    def _on_seismic_load_failed(self, message: str) -> None:
        self.log(f"Could not load seismic data: {message}", "error")
        self._info.setText(f"Load failed: {message}")

    def _reset_load_btn(self) -> None:
        if self._load_busy is not None:
            self._load_busy.finish()
            self._load_busy = None
        self._load_btn.setText("Load seismic data…")

    def _set_dataset(self, dataset) -> None:
        self._dataset = dataset
        self._dt = float(dataset.metadata.sample_interval_s)
        records = list(dataset.field_records)
        self._shot_combo.blockSignals(True)
        self._shot_combo.clear()
        for record in records:
            self._shot_combo.addItem(f"Shot {record}", record)
        self._shot_combo.blockSignals(False)
        self._shot_group.setVisible(True)
        self._all_picks = {}
        self._all_src = {}
        self._shot_pos = {}
        self._current_record = None
        if records:
            self._select_gather(records[0])
        else:
            self._set_raw(np.asarray(dataset.traces, dtype=float), dt=self._dt)

    def _select_gather(self, field_record: int) -> None:
        self._save_current_picks()
        record = int(field_record)
        gather = self._dataset.get_gather(record)
        self._current_gather = gather
        self._current_record = record
        self._raw = np.asarray(gather.traces, dtype=float)
        self._proc_cache = None
        self._headers = gather.headers
        self._picks = dict(self._all_picks.get(record, {}))
        self._pick_src = dict(self._all_src.get(record, {}))
        self._order = list(self._picks.keys())
        self._shot_x.blockSignals(True)
        self._shot_x.setValue(self._shot_pos.get(record, self._default_shot_x(record)))
        self._shot_x.blockSignals(False)
        self._recompute()
        self._update_info()
        self._update_pick_info()

    def _on_shot_changed(self) -> None:
        record = self._shot_combo.currentData()
        if record is not None and self._dataset is not None:
            self._select_gather(int(record))

    def _on_shot_x_changed(self, value: float) -> None:
        if self._current_record is not None:
            self._shot_pos[self._current_record] = float(value)

    def _save_current_picks(self) -> None:
        if self._current_record is not None:
            self._all_picks[self._current_record] = dict(self._picks)
            self._all_src[self._current_record] = dict(self._pick_src)
            self._shot_pos[self._current_record] = self._shot_x.value()
        self._refresh_inversion_source()

    def _refresh_inversion_source(self) -> None:
        """Keep the column's line on what would actually be inverted current."""
        if not hasattr(self, "_tt_status"):
            return  # still building
        self._tt_status.setText(self._describe_traveltime_source())

    def _set_raw(self, arr: np.ndarray, dt: Optional[float] = None) -> None:
        arr = np.atleast_2d(np.asarray(arr, dtype=float))
        if arr.ndim != 2:
            raise ValueError(f"Expected a 2D matrix, got shape {arr.shape}.")
        self._raw = arr
        self._proc_cache = None
        self._dt = dt
        self._dataset = None
        self._current_gather = None
        self._headers = None
        self._shot_group.setVisible(False)

    def _read_seg2(self, path: Path):
        result = read_seg2_seismic(str(path))
        traces = result.get("traces")
        sr = np.ravel(result.get("sampling_rate", [np.nan]))
        dt = 1.0 / float(sr[0]) if sr.size and np.isfinite(sr[0]) and sr[0] > 0 else None
        if isinstance(traces, np.ndarray) and traces.dtype == object:
            seqs = [np.asarray(t, dtype=float) for t in traces]
            maxlen = max((s.size for s in seqs), default=0)
            mat = np.full((maxlen, len(seqs)), np.nan)
            for i, s in enumerate(seqs):
                mat[: s.size, i] = s
            return mat, dt
        arr = np.atleast_2d(np.asarray(traces, dtype=float))
        return arr.T, dt  # text fallback is (n_traces, n_samples) -> (samples, traces)

    def _update_info(self) -> None:
        if self._raw is None:
            self._info.setText("No data loaded.")
            return
        lines = [f"<b>{self._source_path.name}</b>" if self._source_path else "data"]
        lines.append(f"samples × traces: {self._raw.shape}")
        if self._dt:
            lines.append(f"dt: {self._dt * 1000:.3f} ms ({1.0 / self._dt:.0f} Hz)")
        else:
            lines.append("dt: unknown (axis = sample index)")
        if self._dataset is not None:
            lines.append(f"shots: {len(self._dataset.field_records)}  ·  format code: {self._dataset.metadata.format_code}")
        self._info.setText("<br>".join(lines))

    # -- processing ----------------------------------------------------------
    def _processed_base(self) -> Optional[np.ndarray]:
        """Polarity + AGC + normalization (everything except gain), cached so a
        gain or clip change reuses it instead of re-running AGC/normalization."""
        if self._raw is None:
            return None
        key = (self._polarity.isChecked(), self._agc.isChecked(),
               round(self._agc_window.value(), 4), self._normalize.isChecked())
        if self._proc_cache is not None and self._proc_key == key:
            return self._proc_cache
        disp = self._raw.astype(float, copy=True)
        if self._polarity.isChecked():
            disp = -disp
        if self._agc.isChecked():
            if _SEISMIC_OK and self._dt:
                try:
                    disp = apply_agc(disp, self._dt, window=self._agc_window.value() / 1000.0)
                except Exception as exc:  # noqa: BLE001
                    self.log(f"AGC failed: {exc}", "warn")
            # No dt (raw .npy/.csv): AGC has no time scale, so skip it silently.
        if self._normalize.isChecked():
            disp = self._normalize_traces(disp)
        self._proc_cache = disp
        self._proc_key = key
        return disp

    def _processed(self) -> Optional[np.ndarray]:
        base = self._processed_base()
        if base is None:
            return None
        return base * (self._gain.value() / 10.0)

    @staticmethod
    def _normalize_traces(disp: np.ndarray) -> np.ndarray:
        if _SEISMIC_OK:
            try:
                return np.asarray(normalize_traces(disp, trace_axis=1), dtype=float)
            except Exception:
                pass
        peak = np.nanmax(np.abs(disp), axis=0, keepdims=True)
        peak[peak == 0] = 1.0
        return disp / peak

    def _recompute(self) -> None:
        disp = self._processed()
        if disp is None:
            return
        self._viewer.show_gather(disp, self._dt, self._clip.value())
        self._viewer.set_pick_mode(self._pick_mode.isChecked())
        self._redraw_markers()

    # -- picking -------------------------------------------------------------
    def _make_pick(self, trace: int, sample: int, value: float):
        dt = self._dt or 1.0
        receiver_x, receiver_z = self._receiver_position(trace)
        shot_x = self._shot_x.value()
        shot_z = self._interp_topography(shot_x)
        if not _SEISMIC_OK:
            return {"trace": trace, "sample": sample, "time_s": sample * dt, "value": value,
                    "source_x": float(shot_x), "source_z": float(shot_z),
                    "receiver_x": float(receiver_x), "receiver_z": float(receiver_z)}
        record = self._current_record if self._current_record is not None else 1
        return FirstBreakPick(
            source_id=int(record), receiver_id=trace + 1, time_s=float(sample * dt),
            source_x=float(shot_x), source_z=float(shot_z),
            receiver_x=float(receiver_x), receiver_z=float(receiver_z),
            field_record=int(record), trace_number=trace + 1, trace_index=trace, amplitude=float(value),
        )

    def _interp_topography(self, x: float) -> float:
        """Surface elevation at x, interpolated from loaded geophone positions (0 if none)."""
        if not self._geo_positions:
            return 0.0
        pts = sorted(self._geo_positions.values())
        xs = [a for a, _ in pts]
        zs = [b for _, b in pts]
        if len(xs) < 2:
            return float(zs[0]) if zs else 0.0
        return float(np.interp(float(x), xs, zs))

    def _default_shot_x(self, record: int) -> float:
        """Shot x for a record that has no manual override: a regular shot pattern
        (first_shot_x + order * shot_spacing) if set, else the SEG-Y header source x,
        else geophone-0 x."""
        records = self._agent_records()
        if self._shot_spacing is not None and records and record in records:
            return float(self._shot0_x + records.index(record) * self._shot_spacing)
        hx = self._header_shot_x(record)
        if hx is not None:
            return hx
        return float(self._geo_start.value())

    def _header_shot_x(self, record: int) -> Optional[float]:
        """Source x for a record from the SEG-Y trace headers, if populated."""
        if self._dataset is None:
            return None
        try:
            headers = self._dataset.get_gather(int(record)).headers
            xs = [float(h.source_x) for h in headers if np.isfinite(h.source_x)]
        except Exception:  # noqa: BLE001
            return None
        if not xs:
            return None
        if max(xs) - min(xs) > 1e-6:
            return None
        x0 = float(xs[0])
        if abs(x0) <= 1e-9:
            has_coordinate_context = any(
                abs(float(getattr(h, "receiver_x", 0.0))) > 1e-9
                or abs(float(getattr(h, "receiver_y", 0.0))) > 1e-9
                or abs(float(getattr(h, "offset", 0.0))) > 1e-9
                for h in headers
            )
            if not has_coordinate_context:
                return None
        return x0

    def _receiver_position(self, trace: int) -> Tuple[float, float]:
        if self._geo_positions and int(trace) in self._geo_positions:
            return self._geo_positions[int(trace)]
        return (self._geo_start.value() + int(trace) * self._spacing.value(), 0.0)

    @staticmethod
    def _parse_geometry_file(path: str) -> Dict[int, Tuple[float, float]]:
        """Parse a geophone position/topography file into ``{trace_index: (x, z)}``.

        Whitespace/comma separated; a non-numeric header row is skipped. 3+ columns
        read as (station, x, elevation); 2 as (x, elevation); 1 as x with elevation
        0. Geophone order follows file row order.
        """
        rows: List[List[float]] = []
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                parts = line.replace(",", " ").split()
                if not parts:
                    continue
                try:
                    rows.append([float(p) for p in parts])
                except ValueError:
                    continue  # header / comment line
        positions: Dict[int, Tuple[float, float]] = {}
        for i, nums in enumerate(rows):
            if len(nums) >= 3:
                x, z = nums[1], nums[2]
            elif len(nums) == 2:
                x, z = nums[0], nums[1]
            else:
                x, z = nums[0], 0.0
            positions[i] = (float(x), float(z))
        return positions

    def _apply_geometry_file(self, path: str) -> int:
        positions = self._parse_geometry_file(path)
        if not positions:
            return 0
        self._geo_positions = positions
        xs = [positions[k][0] for k in sorted(positions)]
        if len(xs) >= 2:
            self._geo_start.blockSignals(True); self._geo_start.setValue(xs[0]); self._geo_start.blockSignals(False)
            step = abs(xs[1] - xs[0])
            if step > 0:
                self._spacing.blockSignals(True); self._spacing.setValue(step); self._spacing.blockSignals(False)
        self._restamp_all_picks()
        if hasattr(self, "_geo_info"):
            zs = [positions[k][1] for k in positions]
            self._geo_info.setText(
                f"{len(positions)} geophones from file · x {min(xs):.1f}–{max(xs):.1f} m · "
                f"elev {min(zs):.1f}–{max(zs):.1f} m")
        self._redraw_markers()
        self._update_pick_info()
        self._publish()
        return len(positions)

    def _restamp_all_picks(self) -> None:
        """Update receiver x/z (and per-shot source elevation) on existing picks after
        the geophone positions change. Only the geophones move, so each pick keeps its
        own source_x / source_id / field_record — never re-stamp them with another
        record's shot."""
        import dataclasses

        def restamp(picks: Dict[int, Any]) -> None:
            for tr in list(picks):
                p = picks[tr]
                rx, rz = self._receiver_position(int(tr))
                if _SEISMIC_OK and hasattr(p, "receiver_x"):
                    picks[tr] = dataclasses.replace(
                        p, receiver_x=float(rx), receiver_z=float(rz),
                        source_z=float(self._interp_topography(p.source_x)))
                elif isinstance(p, dict):
                    p["receiver_x"] = float(rx); p["receiver_z"] = float(rz)
                    p["source_z"] = float(self._interp_topography(p.get("source_x", 0.0)))

        restamp(self._picks)
        for rec in self._all_picks:
            restamp(self._all_picks[rec])

    def _load_geometry_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load geophone positions / topography", "",
            "Text/CSV (*.txt *.csv *.dat);;All files (*)")
        if not path:
            return
        try:
            n = self._apply_geometry_file(path)
        except Exception as exc:  # noqa: BLE001
            self.log(f"Could not load geophone positions: {exc}", "error")
            return
        if n:
            self.log(f"Loaded {n} geophone positions with elevation from {Path(path).name}.", "success")
        else:
            self.log("No numeric position rows found in the file.", "warn")

    def _viewer_picks(self) -> Dict[int, Any]:
        out: Dict[int, Any] = {}
        for trace, pick in self._picks.items():
            time_s = pick.time_s if (_SEISMIC_OK and hasattr(pick, "time_s")) else pick["time_s"]
            out[int(trace)] = (float(time_s), self._pick_src.get(int(trace), "auto"))
        return out

    def _on_point_picked(self, trace: int, time_s: float, amplitude: float) -> None:
        trace = int(trace)
        dt = self._dt or 1.0
        sample = int(round(time_s / dt))
        if trace not in self._picks:
            self._order.append(trace)
        self._picks[trace] = self._make_pick(trace, sample, amplitude)
        self._pick_src[trace] = "manual"
        self.log(f"Manual pick: trace {trace}, time {time_s:.5g}", "info")
        self._redraw_markers()
        self._update_pick_info()
        self._publish()

    def _on_line_picked(self, points: list) -> None:
        dt = self._dt or 1.0
        for trace, time_s, amp in points:
            trace = int(trace)
            if trace not in self._picks:
                self._order.append(trace)
            self._picks[trace] = self._make_pick(trace, int(round(time_s / dt)), amp)
            self._pick_src[trace] = "manual"
        self._redraw_markers()
        self._update_pick_info()
        self._publish()
        self.log(f"Line pick: {len(points)} traces", "info")

    def _auto_pick(self) -> None:
        if self._raw is None:
            self.log("Load seismic data first.", "warn")
            return
        try:
            onsets = first_arrival_onsets(self._raw, self._dt, ratio_thr=self._threshold.value())
        except Exception as exc:  # noqa: BLE001
            self.log(f"Auto-pick failed: {exc}", "error")
            return
        self._picks = {}
        self._order = []
        self._pick_src = {}
        for j in range(self._raw.shape[1]):
            sample = onsets[j] if j < onsets.size else np.nan
            if np.isfinite(sample):
                s = int(sample)
                self._picks[j] = self._make_pick(j, s, float(self._raw[s, j]))
                self._order.append(j)
                self._pick_src[j] = "auto"
        self.log(
            f"Auto-picked {len(self._picks)} first breaks (STA/LTA ratio {self._threshold.value():.1f}).",
            "success",
        )
        self._redraw_markers()
        self._update_pick_info()
        self._publish()

    def _redraw_markers(self) -> None:
        self._viewer.set_picks(self._viewer_picks())

    def _undo_pick(self) -> None:
        if not self._order:
            return
        trace = self._order.pop()
        self._picks.pop(trace, None)
        self._pick_src.pop(trace, None)
        self._redraw_markers()
        self._update_pick_info()
        self._publish()

    def _clear_picks(self) -> None:
        self._picks = {}
        self._order = []
        self._pick_src = {}
        self._viewer.set_picks({})
        self._update_pick_info()

    def _clear_picks_and_publish(self) -> None:
        self._clear_picks()
        self._publish()

    def _update_pick_info(self) -> None:
        self._pick_info.setText(f"{len(self._picks)} picks")
        self._update_tt_qc()
        self._refresh_inversion_source()

    def _update_tt_qc(self) -> None:
        if not hasattr(self, "_tt_plot"):
            return
        self._tt_plot.clear()
        picks = self._all_first_breaks()
        if not picks:
            return
        from collections import defaultdict

        # PyGIMLi-style first-pick plot: travel time vs ABSOLUTE geophone position,
        # one connected branch per shot, each shot marked with a star at t = 0.
        by_shot = defaultdict(list)
        shot_x: Dict[int, float] = {}
        for p in picks:
            by_shot[int(p.source_id)].append((float(p.receiver_x), float(p.time_s) * 1000.0))
            shot_x[int(p.source_id)] = float(p.source_x)
        colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b", "#e377c2"]
        for i, shot in enumerate(sorted(by_shot, key=lambda s: shot_x.get(s, 0.0))):
            pts = sorted(by_shot[shot])  # by geophone position
            xs = [a for a, _ in pts]
            ys = [b for _, b in pts]
            color = colors[i % len(colors)]
            self._tt_plot.plot(xs, ys, pen=pg.mkPen(color, width=1.5), symbol="o", symbolSize=5,
                               symbolBrush=color, symbolPen=None, name=f"shot @ {shot_x.get(shot, 0.0):.0f} m")
            # shot location on the t = 0 baseline (like pygimli drawFirstPicks)
            self._tt_plot.plot([shot_x.get(shot, 0.0)], [0.0], pen=None, symbol="star",
                               symbolSize=15, symbolBrush=color, symbolPen=pg.mkPen("#222", width=0.8))

    # -- export --------------------------------------------------------------
    def _ordered_picks(self) -> list:
        return [self._picks[t] for t in self._order if t in self._picks]

    def _export_picks(self) -> None:
        if not self._picks:
            self.log("No picks to export.", "warn")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export picks", "seismic_picks.csv", "CSV (*.csv)")
        if not path:
            return
        picks = self._ordered_picks()
        if _SEISMIC_OK:
            export_first_breaks(picks, path)
        else:
            rows = [(p["trace"], p["sample"], p["time_s"], p["value"]) for p in picks]
            io_utils.write_csv(path, rows, header=["trace", "sample", "time_s", "amplitude"])
        self.log(f"Exported {len(picks)} picks to {path}", "success")
        self._publish(picks_csv=path)

    def _export_traveltime(self) -> None:
        if not self._picks:
            self.log("No picks to export.", "warn")
            return
        if not _SEISMIC_OK:
            self.log("Travel-time export needs the seismic processing module.", "warn")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export travel-time", "traveltime.dat", "PyGIMLi data (*.dat)")
        if not path:
            return
        try:
            first_breaks_to_traveltime(self._ordered_picks(), path, receiver_spacing=self._spacing.value())
        except Exception as exc:  # noqa: BLE001
            self.log(f"Travel-time export failed: {exc}", "error")
            return
        self.log(f"Exported travel-time file to {path}", "success")
        self._publish(traveltime_dat=path)

    # -- SRT inversion -------------------------------------------------------
    def _all_first_breaks(self) -> list:
        self._save_current_picks()
        if not _SEISMIC_OK:
            return []
        out = []
        for record, picks in self._all_picks.items():
            for trace, pick in picks.items():
                time_s = pick.time_s if hasattr(pick, "time_s") else pick["time_s"]
                if not np.isfinite(time_s) or time_s <= 0:
                    continue
                default_shot_x = self._shot_pos.get(record, self._default_shot_x(record))
                source_x = getattr(pick, "source_x", None) if hasattr(pick, "source_x") else pick.get("source_x")
                if source_x is None or not np.isfinite(float(source_x)):
                    source_x = default_shot_x
                source_z = getattr(pick, "source_z", None) if hasattr(pick, "source_z") else pick.get("source_z")
                if source_z is None or not np.isfinite(float(source_z)):
                    source_z = self._interp_topography(float(source_x))
                receiver_x = getattr(pick, "receiver_x", None) if hasattr(pick, "receiver_x") else pick.get("receiver_x")
                receiver_z = getattr(pick, "receiver_z", None) if hasattr(pick, "receiver_z") else pick.get("receiver_z")
                receiver_ok = (
                    receiver_x is not None
                    and receiver_z is not None
                    and np.isfinite(float(receiver_x))
                    and np.isfinite(float(receiver_z))
                )
                if not receiver_ok:
                    receiver_x, receiver_z = self._receiver_position(int(trace))
                out.append(FirstBreakPick(
                    source_id=int(record), receiver_id=int(trace) + 1, time_s=float(time_s),
                    source_x=float(source_x), source_z=float(source_z),
                    receiver_x=float(receiver_x), receiver_z=float(receiver_z),
                    field_record=int(record), trace_number=int(trace) + 1,
                    trace_index=int(trace), amplitude=0.0))
        return out

    # -- upload pre-picked travel times --------------------------------------
    def _upload_traveltime(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Upload picked travel times", "",
            "Travel-time data (*.sgt *.tt *.gtt *.dat *.csv *.txt);;All files (*)")
        if not path:
            return
        try:
            data = self._load_traveltime_container(path)
        except Exception as exc:  # noqa: BLE001
            self.log(f"Could not load travel times: {exc}", "error")
            return
        self._tt_data = data
        self._tt_path = Path(path)
        n = int(data.size())
        if hasattr(self.state, "register_geophysical_resource"):
            self.state.register_geophysical_resource(
                "SRT", "observed_data", data,
                label=f"SRT travel times · {Path(path).name}", path=str(path),
                metadata={"traveltimes": n}, resource_id="srt:observed_data:active",
            )
        self._tt_clear_btn.setEnabled(True)
        self._refresh_inversion_source()
        self._plot_tt_container(data)
        self._center_tabs.setCurrentWidget(self._tt_widget)
        self.log(f"Loaded {n} travel times from {Path(path).name}; run SRT inversion to invert them.",
                 "success")

    def _load_traveltime_container(self, path: str):
        """Load a travel-time file into a pyGIMLi DataContainer. Tries the native
        pyGIMLi/BERT format first, then a generic table (source_x [source_z]
        receiver_x [receiver_z] time)."""
        import pygimli.physics.traveltime as tt
        p = str(path)
        # 1. Native pyGIMLi / BERT travel-time format (sensors + 's g t').
        try:
            data = tt.load(p)
            if data is not None and int(data.size()) > 0 and data.haveData("t"):
                return data
        except Exception:  # noqa: BLE001 - fall through to the table parser
            pass
        # 2. Generic columnar text / CSV.
        arr = np.atleast_2d(np.asarray(io_utils.load_2d_array(p), dtype=float))
        if arr.shape[1] == 3:
            sx, gx, t = arr[:, 0], arr[:, 1], arr[:, 2]
            sz = np.zeros(len(t)); gz = np.zeros(len(t))
        elif arr.shape[1] >= 5:
            sx, sz, gx, gz, t = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4]
        else:
            raise ValueError("Expected 3 columns (source_x, receiver_x, time) or 5 "
                             "(source_x, source_z, receiver_x, receiver_z, time).")
        t = np.asarray(t, dtype=float)
        pos_t = t[np.isfinite(t) & (t > 0)]
        if pos_t.size and np.median(pos_t) > 5.0:  # SRT times are <~1 s; >5 ⇒ milliseconds
            t = t / 1000.0
            self.log("Travel times look like milliseconds; interpreted as ms (÷1000).", "info")
        from PyHydroGeophysX.data_processing.seismic import FirstBreakPick, first_breaks_to_traveltime
        src_ids: Dict[float, int] = {}
        rec_ids: Dict[float, int] = {}
        picks = []
        for i in range(len(t)):
            if not np.isfinite(t[i]) or t[i] <= 0:
                continue
            skey = round(float(sx[i]), 4); gkey = round(float(gx[i]), 4)
            sid = src_ids.setdefault(skey, len(src_ids) + 1)
            gid = rec_ids.setdefault(gkey, len(rec_ids) + 1)
            picks.append(FirstBreakPick(
                source_id=sid, receiver_id=gid, time_s=float(t[i]),
                source_x=float(sx[i]), source_z=float(sz[i]),
                receiver_x=float(gx[i]), receiver_z=float(gz[i]),
                field_record=sid, trace_number=gid, trace_index=gid - 1, amplitude=0.0))
        if not picks:
            raise ValueError("No valid (finite, positive) travel times found.")
        out = self.state.ensure_results_store().scratch_dir(self.module_key)
        tmp = str(out / "uploaded_traveltime.dat")
        first_breaks_to_traveltime(picks, tmp)
        return tt.load(tmp)

    def _clear_traveltime(self) -> None:
        self._tt_data = None
        self._tt_path = None
        self._tt_clear_btn.setEnabled(False)
        self._refresh_inversion_source()
        self._update_tt_qc()  # revert the travel-time plot to the picks
        self.log("Cleared uploaded travel times; SRT inversion will use picks.", "info")

    def _plot_tt_container(self, data) -> None:
        """Plot an uploaded travel-time container in the Travel-time tab: time (ms)
        vs absolute receiver position, one branch per shot (pyGIMLi style)."""
        if not hasattr(self, "_tt_plot"):
            return
        self._tt_plot.clear()
        pos = np.asarray(data.sensors(), dtype=float)
        s = np.asarray(data["s"], dtype=int)
        g = np.asarray(data["g"], dtype=int)
        t = np.asarray(data["t"], dtype=float) * 1000.0
        from collections import defaultdict
        by_shot = defaultdict(list)
        shot_x: Dict[int, float] = {}
        for i in range(len(t)):
            si, gi = int(s[i]), int(g[i])
            if not (0 <= si < len(pos) and 0 <= gi < len(pos)):
                continue
            by_shot[si].append((float(pos[gi, 0]), float(t[i])))
            shot_x[si] = float(pos[si, 0])
        colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b", "#e377c2"]
        for i, shot in enumerate(sorted(by_shot, key=lambda ss: shot_x.get(ss, 0.0))):
            pts = sorted(by_shot[shot])
            xs = [a for a, _ in pts]; ys = [b for _, b in pts]
            color = colors[i % len(colors)]
            self._tt_plot.plot(xs, ys, pen=pg.mkPen(color, width=1.5), symbol="o", symbolSize=5,
                               symbolBrush=color, symbolPen=None, name=f"shot @ {shot_x.get(shot, 0.0):.0f} m")
            self._tt_plot.plot([shot_x.get(shot, 0.0)], [0.0], pen=None, symbol="star",
                               symbolSize=15, symbolBrush=color, symbolPen=pg.mkPen("#222", width=0.8))

    def _run_srt(self) -> None:
        picks = None
        if self._tt_data is not None:
            n = int(self._tt_data.size())
            if n < 4:
                self.log("Uploaded travel-time file has too few measurements to invert.", "warn")
                return
            start_msg = f"Running SRT inversion on {n} uploaded travel times."
        else:
            picks = self._all_first_breaks()
            n_shots = sum(1 for v in self._all_picks.values() if v)
            if len(picks) < 8:
                self.log("Pick first breaks on at least a couple of shots, or upload "
                         "travel times, before SRT inversion.", "warn")
                return
            start_msg = f"Running SRT inversion: {len(picks)} picks from {n_shots} shot(s)."
        try:
            run = self.begin_persisted_run(
                "seismic.srt_inversion", "seismic.srt_inversion"
            )
        except Exception as exc:  # noqa: BLE001
            self.log(f"Could not prepare Project run: {exc}", "error")
            return
        travel_path = run.inputs_dir / "traveltime.dat"
        picks_path: Optional[Path] = None
        sources_path: Optional[Path] = None
        pick_source = "uploaded"

        # Uploaded travel times take priority, but the live DataContainer is
        # materialized so a generated script never depends on this process.
        if self._tt_data is not None:
            try:
                export_traveltime_container(self._tt_data, str(travel_path))
            except Exception as exc:  # noqa: BLE001
                self.log(f"Could not serialize uploaded travel times: {exc}", "error")
                self.fail_persisted_run(str(exc), "seismic.srt_inversion")
                return
        else:
            assert picks is not None
            picks_path = run.inputs_dir / "first_break_picks.csv"
            sources_path = run.inputs_dir / "first_break_sources.json"
            export_first_breaks(picks, str(picks_path))
            first_breaks_to_traveltime(
                picks, str(travel_path), receiver_spacing=self._spacing.value()
            )
            sources = {
                str(record): {str(trace): source for trace, source in record_sources.items()}
                for record, record_sources in self._all_src.items()
            }
            io_utils.write_json(sources_path, sources)
            source_values = {
                str(source)
                for record_sources in self._all_src.values()
                for source in record_sources.values()
            }
            pick_source = (
                "mixed" if {"manual", "auto"}.issubset(source_values)
                else "manual" if "manual" in source_values
                else "automatic"
            )

        # Materialized interaction artifacts live beside the recipe/script, so
        # generated code is portable without knowing the original checkout.
        project_root = run.run_dir
        inputs: Dict[str, Any] = {
            "traveltime": ArtifactRef.from_path(
                travel_path,
                artifact_id="seismic:srt:traveltime",
                kind="travel_time",
                format="dat",
                base_dir=project_root,
            ),
        }
        if picks_path is not None:
            inputs["picks"] = ArtifactRef.from_path(
                picks_path,
                artifact_id="seismic:srt:first_break_picks",
                kind="first_break_picks",
                format="csv",
                base_dir=project_root,
                metadata={"source": pick_source},
            )
        if sources_path is not None:
            inputs["pick_sources"] = ArtifactRef.from_path(
                sources_path,
                artifact_id="seismic:srt:pick_sources",
                kind="pick_provenance",
                format="json",
                base_dir=project_root,
            )
        spec = WorkflowSpec(
            workflow_id="seismic.srt_inversion",
            inputs=inputs,
            parameters={"receiver_spacing": float(self._spacing.value()),
                        **self._collect_srt_params()},
            metadata={"pick_source": pick_source},
        )
        recipe_path, script_path = export_workflow_bundle(spec, run.run_dir, stem="srt")
        self._reproduce.set_bundle(recipe_path, script_path)
        context = RunContext(project_root=project_root, output_dir=run.outputs_dir)
        worker = WorkflowWorker(spec, context)
        self._srt_spec = spec
        self._srt_recipe_path = str(recipe_path)
        self._srt_busy = BusyStateController([self._srt_btn])
        self._srt_busy.start()
        self._srt_btn.setText("Inverting…")
        self._srt_progress.setVisible(True)
        self._srt_progress.setRange(0, 0)
        self.log(start_msg, "info")
        self._srt_worker = worker
        self._srt_worker.logged.connect(lambda m: self.log(m, "info"))
        self._srt_worker.succeeded.connect(self._on_srt_workflow_ok)
        self._srt_worker.failed.connect(self._on_srt_failed)
        self._srt_worker.finished.connect(self._reset_srt_button)
        self.register_worker(self._srt_worker)
        self._srt_worker.start()

    def _on_srt_workflow_ok(self, result: WorkflowRunResult) -> None:
        vtk_ref = next(
            (artifact for artifact in result.artifacts if artifact.kind == "velocity_model"),
            None,
        )
        vtk = ""
        if vtk_ref is not None:
            path = Path(vtk_ref.path)
            vtk = str(
                path if path.is_absolute()
                else Path(self._srt_recipe_path).resolve().parent / path
            )
        payload = {
            "mgr": result.objects.get("manager"),
            "n": result.summary.get("n"),
            "vtk": vtk,
            "metrics": dict(result.metrics),
            "convergence": result.objects.get("convergence") or [],
            "auto_lambda_note": str(result.summary.get("auto_lambda_note", "")),
            "auto_lambda_status": str(result.summary.get("auto_lambda_status", "off")),
        }
        try:
            self._on_srt_ok(payload)
        finally:
            if hasattr(self.state, "update_workflow_result"):
                self.state.update_workflow_result(
                    self.module_key,
                    "seismic.srt_inversion",
                    result.to_dict(),
                    recipe_path=self._srt_recipe_path,
                )

    def _on_srt_ok(self, result: dict) -> None:
        mgr = result.get("mgr")
        self._srt_mgr = mgr
        if mgr is not None:
            self._vel_view.show_model(mgr, kind="srt")
            self._center_tabs.setCurrentWidget(self._vel_view)
            self._srt_export_btn.setEnabled(True)
        metrics = dict(result.get("metrics") or {})
        self._quality_view.show_quality(metrics, result.get("convergence"), title="SRT inversion")
        vtk = result.get("vtk")
        if vtk:
            self.log(f"Saved velocity mesh to {vtk}", "info")
        chi2 = metrics.get("chi2")
        self.log(f"SRT inversion complete (chi2={chi2:.2f})." if isinstance(chi2, float) and chi2 == chi2
                 else "SRT inversion complete.", "success")
        # One line, only when the search actually moved lambda. The ERT page
        # learned that a wall of search commentary crowds out the result.
        note = str(result.get("auto_lambda_note") or "")
        if note:
            self.log(note, "warning"
                     if result.get("auto_lambda_status") == "no_improvement"
                     else "info")
        if mgr is not None and hasattr(self.state, "register_geophysical_resource"):
            observed = self._tt_data if self._tt_data is not None else getattr(mgr, "data", None)
            if observed is not None:
                self.state.register_geophysical_resource(
                    "SRT", "observed_data", observed,
                    label="Current SRT travel times", path=str(self._tt_path or ""),
                    metadata={"traveltimes": result.get("n")},
                    resource_id="srt:observed_data:active",
                )
            velocity = np.asarray(velocity_of(mgr), dtype=float)
            self.state.register_geophysical_resource(
                "SRT", "model", velocity,
                label="Latest SRT velocity model", path=str(vtk or ""),
                metadata={"chi2": metrics.get("chi2"), "mesh": getattr(mgr, "paraDomain", None)},
                resource_id="srt:model:latest",
            )
        self.report_result({"velocity_vtk": vtk, "num_traveltimes": result.get("n"),
                            "chi2": metrics.get("chi2"), "rrms": metrics.get("rrms"),
                            "iterations": metrics.get("iterations")})

    def _export_velocity_model(self) -> None:
        mgr = getattr(self, "_srt_mgr", None)
        if mgr is None:
            self.log("Run SRT inversion first.", "warn")
            return
        folder = select_directory(
            self, "Export velocity model to folder",
            self.state.output_dir or Path.cwd(),
        )
        if not folder:
            return
        try:
            out = io_utils.ensure_dir(folder)
            mesh = mgr.paraDomain
            velocity = np.asarray(velocity_of(mgr), dtype=float)
            np.save(out / "velocity_model.npy", velocity)
            mesh.save(str(out / "velocity_mesh.bms"))
            mesh["velocity"] = velocity
            mesh.exportVTK(str(out / "velocity_model.vtk"))
            self.log(f"Exported velocity model (npy + bms + vtk) to {out}", "success")
        except Exception as exc:  # noqa: BLE001
            self.log(f"Velocity model export failed: {exc}", "error")

    def _on_srt_failed(self, message: str) -> None:
        self.fail_persisted_run(message, "seismic.srt_inversion")
        self.log(f"SRT inversion failed: {message}", "error")

    def _reset_srt_button(self) -> None:
        if self._srt_busy is not None:
            self._srt_busy.finish()
            self._srt_busy = None
        self._srt_btn.setText("Run SRT inversion")
        self._srt_progress.setVisible(False)

    def _publish(self, picks_csv: Optional[str] = None, traveltime_dat: Optional[str] = None) -> None:
        result = {
            "source_file": str(self._source_path) if self._source_path else "",
            "format": self._source_path.suffix.lower() if self._source_path else "",
            "dt_s": self._dt,
            "num_traces": int(self._raw.shape[1]) if self._raw is not None else 0,
            "num_picks": len(self._picks),
            "settings": {
                "gain": self._gain.value() / 10.0,
                "clip_percentile": self._clip.value(),
                "polarity_flip": self._polarity.isChecked(),
                "normalize_trace": self._normalize.isChecked(),
                "agc": self._agc.isChecked(),
                "receiver_spacing": self._spacing.value(),
            },
        }
        if picks_csv:
            result["picks_csv"] = picks_csv
        if traveltime_dat:
            result["traveltime_dat"] = traveltime_dat
        self.report_result(result)

    # -- AQUAH agent interface ----------------------------------------------
    def agent_describe(self) -> Dict[str, Any]:
        return {
            "module": self.module_key,
            "title": self.module_title,
            "state": self._agent_status(),
            "actions": [
                {"name": "load_data", "args": {"path": "str"},
                 "desc": "Load a seismic file (SEG-Y .sgy/.segy, Geometrics .dat, SEG-2, or .npy/.csv matrix)."},
                {"name": "list_records", "args": {},
                 "desc": "List shot / field records in the loaded dataset."},
                {"name": "select_record", "args": {"record": "int"},
                 "desc": "Switch to a shot record by its field-record number."},
                {"name": "set_geometry",
                 "args": {"spacing": "float", "geophone_start": "float", "shot_x": "float",
                          "shot_spacing": "float", "first_shot_x": "float"},
                 "desc": ("Set geophone spacing (m) and geophone-0 x (m). For a REGULAR shot layout, pass "
                          "first_shot_x and shot_spacing ONCE — shot_x then auto-fills for every record on "
                          "select_record. Use shot_x only to set/override one record's shot (may be negative).")},
                {"name": "load_geometry", "args": {"path": "str"},
                 "desc": ("Load per-geophone positions + topography from a text file (columns: "
                          "'station distance_m elevation_m', or 'x z'). Applies real receiver x and "
                          "elevation so the SRT inversion honors topography; re-stamps existing picks.")},
                {"name": "set_params", "args": {"params": {"<key>": "value"}},
                 "desc": ("Set processing/pick params. Picking and display: sta_lta_ratio, "
                          "gain (slider 1-100), clip_percentile, agc_window_ms, "
                          "flip_polarity, normalize, agc. Inversion: engine "
                          "(pyhydro/pygimli), lam, max_iterations, "
                          "max_total_iterations, plateau_tolerance (fraction). "
                          "Mesh: mesh_quality, para_depth (m, 0 = auto), "
                          "para_max_cell_size (0 = auto), secondary_nodes. "
                          "Fit assistance (in-house engine only): auto_lambda, "
                          "target_chi2, chi2_tolerance, max_lambda_trials.")},
                {"name": "auto_pick", "args": {},
                 "desc": "Auto-pick first breaks on the current record (STA/LTA)."},
                {"name": "pick_next_shot", "args": {},
                 "desc": ("FAST per-shot step: advance to the next shot record that still needs picking, "
                          "auto-pick it, and pause for review (returns 'awaiting_user' with records_remaining "
                          "and next_record). ONE call replaces select_record + auto_pick + review_picks — "
                          "prefer it to step through shots; use the individual actions only for manual re-picking.")},
                {"name": "review_picks", "args": {},
                 "desc": ("Pause for the user to review/correct first-break picks: turns on Manual pick mode, "
                          "flags suspect traces, and returns status 'awaiting_user'. ALWAYS call this after "
                          "auto_pick and before run_srt; do not run_srt until the user says the picks are good.")},
                {"name": "set_pick", "args": {"trace": "int", "time_s": "float"},
                 "desc": "Set/override one trace's first-break pick to time_s (seconds) on the current record."},
                {"name": "delete_pick", "args": {"trace": "int"},
                 "desc": "Delete one trace's first-break pick on the current record."},
                {"name": "list_picks", "args": {},
                 "desc": "List current-record picks as {trace: {time_s, source}}, with suspect traces flagged."},
                {"name": "clear_picks", "args": {},
                 "desc": "Clear picks on the current record."},
                {"name": "load_traveltime", "args": {"path": "str"},
                 "desc": ("Upload a pre-picked travel-time file and invert it directly (no picking). "
                          "Formats: pyGIMLi/BERT .sgt/.dat, or a CSV/text with columns source_x, receiver_x, "
                          "time (or source_x, source_z, receiver_x, receiver_z, time). Then call run_srt.")},
                {"name": "clear_traveltime", "args": {},
                 "desc": "Clear uploaded travel times so run_srt uses the picks again."},
                {"name": "run_srt", "args": {},
                 "desc": ("Run SRT travel-time tomography. Inverts uploaded travel times if any were loaded, "
                          "otherwise the picked shots (needs >=8 picks total).")},
                {"name": "get_status", "args": {},
                 "desc": "Report loaded data, current record, pick counts, and last result."},
            ],
        }

    def agent_apply(self, action: str, args: Dict[str, Any]) -> Dict[str, Any]:
        args = args or {}
        handlers = {
            "load_data": lambda: self._agent_load(args.get("path")),
            "list_records": lambda: self._agent_list_records(),
            "select_record": lambda: self._agent_select_record(args.get("record")),
            "set_geometry": lambda: self._agent_set_geometry(args),
            "load_geometry": lambda: self._agent_load_geometry(args.get("path")),
            "set_params": lambda: self._agent_set_params(args.get("params", args)),
            "auto_pick": lambda: self._agent_auto_pick(),
            "pick_next_shot": lambda: self._agent_pick_next_shot(),
            "review_picks": lambda: self._agent_review_picks(),
            "set_pick": lambda: self._agent_set_pick(args),
            "delete_pick": lambda: self._agent_delete_pick(args),
            "list_picks": lambda: self._agent_list_picks(),
            "clear_picks": lambda: self._agent_clear_picks(),
            "load_traveltime": lambda: self._agent_load_traveltime(args.get("path")),
            "clear_traveltime": lambda: self._agent_clear_traveltime(),
            "run_srt": lambda: self._agent_run_srt(),
            "get_status": lambda: self._agent_status(),
        }
        handler = handlers.get(action)
        if handler is None:
            return {"status": "failed", "error": f"Unknown action '{action}'.",
                    "valid_actions": list(handlers.keys())}
        return handler()

    def _agent_records(self) -> List[int]:
        if self._dataset is None:
            return []
        return [self._shot_combo.itemData(i) for i in range(self._shot_combo.count())]

    def _agent_status(self) -> Dict[str, Any]:
        if self._current_record is not None:
            self._save_current_picks()
        total = sum(len(v) for v in self._all_picks.values())
        last = self.state.module_results.get(self.module_key, {})
        return {
            "status": "ok",
            "loaded": self._raw is not None,
            "source": str(self._source_path or ""),
            "records": self._agent_records(),
            "current_record": self._current_record,
            "current_picks": len(self._picks),
            "total_picks_all_shots": total,
            "geometry": {
                "spacing": self._spacing.value(),
                "geophone_start": self._geo_start.value(),
                "shot_x": self._shot_x.value(),
            },
            "has_velocity_model": getattr(self, "_srt_mgr", None) is not None,
            "last_result_keys": sorted(last.keys()),
        }

    def _agent_load(self, path: Any) -> Dict[str, Any]:
        if not path:
            return {"status": "failed", "error": "Provide 'path' to a seismic file."}
        p = Path(str(path))
        if not p.exists():
            return {"status": "failed", "error": f"File not found: {p}"}
        try:
            res = self._parse_seismic(str(p))
            self._on_seismic_loaded(str(p), res)
        except Exception as exc:  # noqa: BLE001
            return {"status": "failed", "error": f"Could not load: {exc}"}
        return {"status": "ok", "source": str(p),
                "shape": list(self._raw.shape) if self._raw is not None else None,
                "records": self._agent_records()}

    def _agent_list_records(self) -> Dict[str, Any]:
        records = self._agent_records()
        if not records:
            return {"status": "ok", "records": [], "note": "Single matrix loaded (no shot records)."}
        return {"status": "ok", "records": records, "current_record": self._current_record}

    def _agent_select_record(self, record: Any) -> Dict[str, Any]:
        records = self._agent_records()
        if not records:
            return {"status": "failed", "error": "No multi-record dataset loaded."}
        if record is None:
            return {"status": "failed", "error": "Provide 'record'.", "records": records}
        rec = int(record)
        if rec not in records:
            return {"status": "failed", "error": f"Record {rec} not found.", "records": records}
        self._shot_combo.setCurrentIndex(records.index(rec))
        return {"status": "ok", "current_record": self._current_record,
                "num_traces": int(self._raw.shape[1]) if self._raw is not None else 0}

    def _agent_set_geometry(self, args: Dict[str, Any]) -> Dict[str, Any]:
        applied: Dict[str, Any] = {}
        try:
            if "spacing" in args:
                self._spacing.setValue(float(args["spacing"])); applied["spacing"] = args["spacing"]
            if "geophone_start" in args:
                self._geo_start.setValue(float(args["geophone_start"])); applied["geophone_start"] = args["geophone_start"]
            if args.get("shot_spacing") is not None:
                self._shot_spacing = float(args["shot_spacing"]); applied["shot_spacing"] = args["shot_spacing"]
            if "first_shot_x" in args:
                self._shot0_x = float(args["first_shot_x"]); applied["first_shot_x"] = args["first_shot_x"]
            # A new/updated shot pattern re-fills the current record's shot from it.
            if ("shot_spacing" in args or "first_shot_x" in args) and self._current_record is not None:
                self._shot_pos.pop(self._current_record, None)
                self._shot_x.setValue(self._default_shot_x(self._current_record))
                applied["shot_x"] = self._shot_x.value()
            if "shot_x" in args:  # explicit per-record override wins
                self._shot_x.setValue(float(args["shot_x"])); applied["shot_x"] = args["shot_x"]
        except Exception as exc:  # noqa: BLE001
            return {"status": "failed", "error": str(exc)}
        if not applied:
            return {"status": "failed",
                    "error": "Provide spacing, geophone_start, shot_x, shot_spacing, and/or first_shot_x."}
        result: Dict[str, Any] = {"status": "ok", "applied": applied}
        if self._shot_spacing is not None:
            result["shot_pattern"] = {
                "first_shot_x": self._shot0_x, "shot_spacing": self._shot_spacing,
                "note": "shot_x now auto-fills per record on select_record; set shot_x only to "
                        "override one irregular shot."}
        return result

    def _agent_load_geometry(self, path: Any) -> Dict[str, Any]:
        if not path:
            return {"status": "failed", "error": "Provide 'path' to a geophone position/topography file."}
        p = Path(str(path))
        if not p.exists():
            return {"status": "failed", "error": f"File not found: {p}"}
        try:
            n = self._apply_geometry_file(str(p))
        except Exception as exc:  # noqa: BLE001
            return {"status": "failed", "error": f"Could not parse positions: {exc}"}
        if not n:
            return {"status": "failed", "error": "No numeric position rows found in the file."}
        xs = [v[0] for v in self._geo_positions.values()]
        zs = [v[1] for v in self._geo_positions.values()]
        return {"status": "ok", "geophones": n,
                "x_range": [min(xs), max(xs)], "elevation_range": [min(zs), max(zs)],
                "note": "Per-geophone x + elevation applied; picks re-stamped; the SRT inversion will honor topography."}

    def _agent_set_params(self, params: Any) -> Dict[str, Any]:
        if not isinstance(params, dict):
            return {"status": "failed", "error": "Provide 'params' as a JSON object."}
        handlers = {
            "sta_lta_ratio": lambda v: self._threshold.setValue(float(v)),
            "gain": lambda v: self._gain.setValue(int(v)),
            "clip_percentile": lambda v: self._clip.setValue(float(v)),
            "agc_window_ms": lambda v: self._agc_window.setValue(float(v)),
            "flip_polarity": lambda v: self._polarity.setChecked(bool(v)),
            "normalize": lambda v: self._normalize.setChecked(bool(v)),
            "agc": lambda v: self._agc.setChecked(bool(v)),
            "engine": lambda v: self._set_srt_engine(str(v)),
            "lam": lambda v: self._srt_lam.setValue(float(v)),
            "max_iterations": lambda v: self._srt_iter.setValue(int(v)),
            "max_total_iterations": lambda v: self._srt_iter_ceiling.setValue(int(v)),
            "plateau_tolerance": lambda v: self._srt_plateau.setValue(float(v) * 100.0),
            "mesh_quality": lambda v: self._srt_quality.setValue(float(v)),
            "para_depth": lambda v: self._srt_para_depth.setValue(float(v)),
            "para_max_cell_size": lambda v: self._srt_cell_size.setValue(float(v)),
            "secondary_nodes": lambda v: self._srt_sec_nodes.setValue(int(v)),
            "auto_lambda": lambda v: self._srt_auto_lam.setChecked(bool(v)),
            "target_chi2": lambda v: self._srt_target_chi2.setValue(float(v)),
            "chi2_tolerance": lambda v: self._srt_chi2_tol.setValue(float(v)),
            "max_lambda_trials": lambda v: self._srt_lam_trials.setValue(int(v)),
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

    def _agent_auto_pick(self) -> Dict[str, Any]:
        if self._raw is None:
            return {"status": "failed", "error": "Load data first."}
        self._auto_pick()
        return {"status": "ok", "picks": len(self._picks)}

    def _agent_clear_picks(self) -> Dict[str, Any]:
        self._clear_picks_and_publish()
        return {"status": "ok", "picks": 0}

    def _agent_pick_next_shot(self) -> Dict[str, Any]:
        """Fast loop step: select the next un-picked shot, auto-pick it, pause for review.
        Collapses select_record + auto_pick + review_picks into one tool call so the
        per-shot loop costs one LLM round-trip and one approval instead of three."""
        if self._raw is None:
            return {"status": "failed", "error": "Load data first."}
        all_records = self._agent_records()
        if not all_records:  # single matrix, no shot records
            self._auto_pick()
            return self._agent_review_picks()
        self._save_current_picks()
        picked = [r for r in all_records if self._all_picks.get(r)]
        remaining = [r for r in all_records if r not in picked]
        if not remaining:
            return {"status": "ok", "records_remaining": [],
                    "message": "All shots are already picked — call run_srt to invert."}
        target = remaining[0]
        self._shot_combo.setCurrentIndex(all_records.index(target))
        self._auto_pick()
        return self._agent_review_picks()

    def _agent_review_picks(self) -> Dict[str, Any]:
        """Human-in-the-loop checkpoint: hand control to the user to correct picks."""
        if self._raw is None:
            return {"status": "failed", "error": "Load data and auto-pick first."}
        if not self._picks:
            return {"status": "failed", "error": "No picks on this record yet — run auto_pick first."}
        self._save_current_picks()
        # Turn on manual editing so the user can click / Ctrl+drag to correct traces.
        self._pick_mode.setChecked(True)
        auto = sum(1 for s in self._pick_src.values() if s == "auto")
        manual = sum(1 for s in self._pick_src.values() if s == "manual")
        all_records = self._agent_records()
        picked = [r for r in all_records if self._all_picks.get(r)]
        remaining = [r for r in all_records if r not in picked]
        next_record = remaining[0] if remaining else None
        if remaining:
            tail = (f"This is shot {self._current_record}. {len(remaining)} more shot(s) still need "
                    f"picking (next: {next_record}). After you say 'continue' I will pick the next shot; "
                    "I run the SRT inversion only once every shot is reviewed.")
        else:
            tail = "Every shot is picked — say 'continue' to run the SRT inversion."
        return {
            "status": "awaiting_user",
            "current_record": self._current_record,
            "picks": len(self._picks),
            "auto": auto,
            "manual": manual,
            "suspect_traces": self._suspect_pick_traces(),
            "records_total": all_records,
            "records_picked": sorted(picked),
            "records_remaining": remaining,
            "next_record": next_record,
            "resume": {"action": "pick_next_shot" if remaining else "run_srt", "args": {}},
            "message": (
                "Auto-picks are in and Manual pick mode is ON. Correct any bad traces by clicking / "
                "Ctrl+dragging, or ask me to set_pick / delete_pick a trace. " + tail
            ),
        }

    def _suspect_pick_traces(self) -> List[int]:
        """Flag picks deviating strongly from a robust linear move-out fit (offset vs time)."""
        try:
            shot_x = self._shot_x.value()
            geo0 = self._geo_start.value()
            spacing = self._spacing.value()
            traces: List[int] = []
            offsets: List[float] = []
            times: List[float] = []
            for tr, pick in self._picks.items():
                t = pick.time_s if (_SEISMIC_OK and hasattr(pick, "time_s")) else pick["time_s"]
                traces.append(int(tr))
                offsets.append(abs((geo0 + int(tr) * spacing) - shot_x))
                times.append(float(t))
            if len(traces) < 4:
                return []
            x = np.asarray(offsets, dtype=float)
            y = np.asarray(times, dtype=float)
            a, b = np.polyfit(x, y, 1)
            resid = y - (a * x + b)
            med = float(np.median(resid))
            mad = float(np.median(np.abs(resid - med)))
            scale = 1.4826 * mad if mad > 0 else (float(np.std(resid)) or 1e-9)
            thr = 3.0 * scale
            return sorted(int(traces[i]) for i in range(len(traces)) if abs(resid[i] - med) > thr)
        except Exception:  # noqa: BLE001
            return []

    def _agent_set_pick(self, args: Dict[str, Any]) -> Dict[str, Any]:
        if self._raw is None:
            return {"status": "failed", "error": "Load data first."}
        if "trace" not in args or "time_s" not in args:
            return {"status": "failed", "error": "Provide 'trace' (int) and 'time_s' (seconds)."}
        try:
            trace = int(args["trace"])
            time_s = float(args["time_s"])
        except (TypeError, ValueError):
            return {"status": "failed", "error": "'trace' must be an int and 'time_s' a number."}
        n_traces = int(self._raw.shape[1])
        if not 0 <= trace < n_traces:
            return {"status": "failed", "error": f"trace out of range 0..{n_traces - 1}."}
        dt = self._dt or 1.0
        n_samples = int(self._raw.shape[0])
        sample = max(0, min(n_samples - 1, int(round(time_s / dt))))
        amp = float(self._raw[sample, trace])
        if trace not in self._picks:
            self._order.append(trace)
        self._picks[trace] = self._make_pick(trace, sample, amp)
        self._pick_src[trace] = "manual"
        self._redraw_markers()
        self._update_pick_info()
        self._publish()
        return {"status": "ok", "trace": trace, "time_s": sample * dt, "picks": len(self._picks)}

    def _agent_delete_pick(self, args: Dict[str, Any]) -> Dict[str, Any]:
        if "trace" not in args:
            return {"status": "failed", "error": "Provide 'trace' (int)."}
        try:
            trace = int(args["trace"])
        except (TypeError, ValueError):
            return {"status": "failed", "error": "'trace' must be an int."}
        if trace not in self._picks:
            return {"status": "ok", "note": f"No pick on trace {trace}.", "picks": len(self._picks)}
        self._picks.pop(trace, None)
        self._pick_src.pop(trace, None)
        if trace in self._order:
            self._order.remove(trace)
        self._redraw_markers()
        self._update_pick_info()
        self._publish()
        return {"status": "ok", "deleted": trace, "picks": len(self._picks)}

    def _agent_list_picks(self) -> Dict[str, Any]:
        picks = {int(tr): {"time_s": round(float(ts), 6), "source": src}
                 for tr, (ts, src) in self._viewer_picks().items()}
        return {"status": "ok", "current_record": self._current_record,
                "count": len(picks), "picks": picks,
                "suspect_traces": self._suspect_pick_traces()}

    def agent_view_context(self, view: str) -> Optional[Dict[str, Any]]:
        """Ship the pick table with a captured gather.

        Trace indices are the one thing a model reads unreliably off this panel:
        two dozen traces share a narrow axis and each marker sits far above its
        tick label. Sending the picks as numbers leaves the picture to do what it
        is actually good for, judging whether a pick sits on the first arrival.
        """
        if view != "gather" or self._raw is None:
            return None
        picks = {int(tr): round(float(ts) * 1000.0, 2)
                 for tr, (ts, _src) in self._viewer_picks().items()}
        if not picks:
            return None
        return {
            "current_record": self._current_record,
            "pick_times_ms": dict(sorted(picks.items())),
            "suspect_traces": self._suspect_pick_traces(),
            "note": ("Trace indices come from the workbench and are exact. Use the image to "
                     "judge whether each pick follows the first arrival, and these numbers "
                     "to say which trace you mean."),
        }

    def _agent_load_traveltime(self, path: Any) -> Dict[str, Any]:
        if not path:
            return {"status": "failed", "error": "Provide 'path' to a travel-time file."}
        p = Path(str(path))
        if not p.exists():
            return {"status": "failed", "error": f"File not found: {p}"}
        try:
            data = self._load_traveltime_container(str(p))
        except Exception as exc:  # noqa: BLE001
            return {"status": "failed", "error": f"Could not load travel times: {exc}"}
        self._tt_data = data
        self._tt_path = p
        self._tt_clear_btn.setEnabled(True)
        n = int(data.size())
        self._refresh_inversion_source()
        self._plot_tt_container(data)
        return {"status": "ok", "traveltimes": n,
                "note": "Run SRT inversion to invert these directly (no picking needed)."}

    def _agent_clear_traveltime(self) -> Dict[str, Any]:
        self._clear_traveltime()
        return {"status": "ok", "message": "Uploaded travel times cleared; run_srt will use picks."}

    def _agent_run_srt(self) -> Dict[str, Any]:
        if self._tt_data is not None:
            self._run_srt()
            return {"status": "started", "message": "SRT inversion started on uploaded travel times.",
                    "traveltimes": int(self._tt_data.size())}
        picks = self._all_first_breaks()
        if len(picks) < 8:
            return {"status": "failed", "error": "Need at least 8 first-break picks across shots.",
                    "picks": len(picks),
                    "hint": "Auto-pick first breaks on a couple of shots, or upload a travel-time file."}
        self._run_srt()
        return {"status": "started", "message": "SRT inversion started. Ask for status shortly.",
                "picks": len(picks)}
