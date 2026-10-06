"""Seismic processing module.

Loads real field formats (SEG-Y, Geometrics DAT, SEG-2) by reusing
``PyHydroGeophysX.data_processing.seismic`` plus generic ``.npy/.csv`` arrays,
lets the user browse shot gathers, apply display/QC processing (gain, clip,
polarity, trace normalization, AGC), pick first arrivals (manual + assisted),
and export picks to CSV and a PyGIMLi travel-time ``.dat`` file.

The Reflection tab stacks the same shots into a CMP section with
``PyHydroGeophysX.data_processing.seismic_shallow`` (trace QC, air-wave timing
and removal, causal filters, NMO and stack), on the geometry set for picking.
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
    QLineEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import QCoreApplication, QEventLoop, Qt, QTimer

from PyHydroGeophysX._internal.utils import velocity_of
from PyHydroGeophysX.inversion.lambda_search import LAMBDA_BOUNDS as _LAMBDA_BOUNDS
from PyHydroGeophysX.qt_apps import io_utils, theme
from PyHydroGeophysX.qt_apps.modules.base import BaseModule, LogFn
from PyHydroGeophysX.qt_apps.qt_utils import (
    BusyStateController,
    ContentWidthScrollArea,
    Debouncer,
    ReproduceBar,
    make_double_spinbox,
    make_spinbox,
    merged_row,
    select_directory,
    set_rows_enabled,
    set_rows_visible,
)
from PyHydroGeophysX.qt_apps.widgets import colormaps as cmaps
from PyHydroGeophysX.qt_apps.widgets import length_units
from PyHydroGeophysX.qt_apps.widgets.mesh_view import MeshResultView
from PyHydroGeophysX.qt_apps.widgets.quality_view import InversionQualityView
from PyHydroGeophysX.qt_apps.widgets.reflection_view import ReflectionView
from PyHydroGeophysX.qt_apps.widgets.run_controls import progress_with_stop
from PyHydroGeophysX.qt_apps.widgets.seismic_viewer import SeismicViewer
from PyHydroGeophysX.qt_apps.workers import ProcessWorkflowWorker, TaskWorker
from PyHydroGeophysX.visualization.axis_units import to_display_length
from PyHydroGeophysX.workflows import (
    ArtifactRef,
    WorkflowRunResult,
    WorkflowSpec,
    export_workflow_bundle,
)

try:
    from PyHydroGeophysX.data_processing.seismic import (
        FirstBreakPick,
        apply_agc,
        apply_record_interval,
        export_first_breaks,
        export_traveltime_container,
        first_breaks_to_traveltime,
        normalize_traces,
        pick_and_correct,
        read_geometrics_dat,
        read_segy,
        screen_picks,
    )
    from PyHydroGeophysX.data_processing.field_formats import read_seg2_seismic
    from PyHydroGeophysX.data_processing import seismic_shallow as shallow

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


def parse_shot_list(text: str) -> List[int]:
    """``"1, 5-7 12"`` -> ``[1, 5, 6, 7, 12]``; raises ValueError on anything else."""
    shots: List[int] = []
    for part in str(text).replace(";", ",").replace(",", " ").split():
        if "-" in part[1:]:
            first, last = part.split("-", 1)
            shots.extend(range(int(first), int(last) + 1))
        else:
            shots.append(int(part))
    return sorted(set(shots))


def _stack_reflections(traces: np.ndarray, dt: float, geometry: Any, params: Dict[str, Any],
                       prepared: Any = None) -> Dict[str, Any]:
    """The Reflection tab's run, off the window's thread: prepare the line, then stack it.

    ``prepared`` is the line from an earlier run with the same traces and
    clean-up, so a change of velocity or stack setting re-stacks without
    re-timing every shot.
    """
    if prepared is None:
        qc = shallow.trace_qc(traces, dt, geometry, exclude_records=params["skip_shots"])
        if not params["drop_clipped"]:
            qc.clipped = np.zeros_like(qc.clipped)
        if not params["drop_early"]:
            qc.early_energy = np.zeros_like(qc.early_energy)
        remove_air = params["remove_air"]
        prepared = shallow.prepare_line(
            traces, dt, geometry, qc=qc, statics=params["align"], suppress=remove_air,
            band=params["band"], mute=(0.75e-3, 5.0e-3) if remove_air else None)
    result = shallow.stack_line(prepared, params["velocity"], **params["stack"])
    return {"prepared": prepared, "result": result, "ray_limit": params.get("ray_limit")}


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
        self._srt_worker: Optional[ProcessWorkflowWorker] = None
        self._srt_busy: Optional[BusyStateController] = None
        self._srt_spec: Optional[WorkflowSpec] = None
        self._srt_recipe_path: str = ""
        self._tt_data = None                 # uploaded pre-picked travel-time DataContainer
        self._tt_path: Optional[Path] = None
        self._load_worker: Optional[TaskWorker] = None
        self._load_busy: Optional[BusyStateController] = None
        self._proc_cache: Optional[np.ndarray] = None
        self._proc_key: Optional[tuple] = None
        self._spacing_from_headers = False
        self._refl_worker: Optional[TaskWorker] = None
        self._refl_busy: Optional[BusyStateController] = None
        self._refl_prepared = None           # (key, PreparedLine) of the last run
        self._refl_result = None
        self._refl_error = ""
        self._recompute_debounced = Debouncer(self._recompute, 80)
        # Typing "0.5" passes through 0 and 0.: the picks move once, at the end.
        self._geometry_debounced = Debouncer(self._on_geometry_changed, 250)

        root = QHBoxLayout(self)
        # Both views keep their colour maps in the session's shared choices.
        self._viewer = SeismicViewer(colormaps=cmaps.colormap_settings(self.state))
        self._viewer.pointPicked.connect(self._on_point_picked)
        self._viewer.linePicked.connect(self._on_line_picked)
        self._center_tabs = QTabWidget()
        self._center_tabs.addTab(self._viewer, "Gather")
        self._tt_widget = pg.PlotWidget()
        self._tt_widget.setBackground("w")
        self._tt_widget.showGrid(x=True, y=True, alpha=0.3)
        self._tt_widget.setLabel("left", "travel time (ms)")
        self._tt_plot = self._tt_widget.getPlotItem()
        length_units.pyqtgraph_axis(self._tt_plot, "bottom", "geophone position x")
        self._tt_plot.addLegend()
        # Which of the two travel-time plots is up, so a change of length unit
        # redraws that one: an uploaded container, or the picks.
        self._tt_shows_container = False
        length_units.notifier().changed.connect(self._on_length_unit_changed)
        self._center_tabs.addTab(self._tt_widget, "Travel-time")
        self._vel_view = MeshResultView(colormaps=cmaps.colormap_settings(self.state))
        self._center_tabs.addTab(self._vel_view, "Velocity model")
        self._quality_view = InversionQualityView()
        self._center_tabs.addTab(self._quality_view, "Inversion quality")
        self._refl_view = ReflectionView(colormaps=cmaps.colormap_settings(self.state))
        self._center_tabs.addTab(self._refl_view, "Reflection")
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
        self._reflection_panel = self._build_reflection_panel()
        root.addWidget(self._reflection_panel)
        self._center_tabs.currentChanged.connect(self._on_center_tab_changed)
        self._on_center_tab_changed()
        self._refresh_reflection_source()
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
        theme.set_tone(hint, "hint")
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
        theme.set_tone(self._geo_info, "hint")
        self._geo_info.setWordWrap(True)
        layout.addWidget(self._geo_info)

        self._shot_group = QGroupBox("Geometry")
        sform = QFormLayout(self._shot_group)
        self._shot_combo = QComboBox()
        self._shot_combo.currentIndexChanged.connect(self._on_shot_changed)
        sform.addRow("Record", self._shot_combo)
        self._spacing = QDoubleSpinBox()
        self._spacing.setRange(0.05, 1000.0); self._spacing.setValue(1.0); self._spacing.setSuffix(" m")
        self._spacing.valueChanged.connect(self._geometry_debounced.trigger)
        sform.addRow("Geophone spacing", self._spacing)
        self._geo_start = QDoubleSpinBox()
        self._geo_start.setRange(-100000.0, 100000.0); self._geo_start.setValue(0.0); self._geo_start.setSuffix(" m")
        self._geo_start.valueChanged.connect(self._geometry_debounced.trigger)
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
        theme.set_tone(tip, "hint")
        pbox.addWidget(tip)
        # The seismic workflow's picker (data_processing.seismic.pick_and_correct
        # and screen_picks), so a line picked here and one picked by a run agree.
        tform = QFormLayout()
        self._threshold = QDoubleSpinBox()
        self._threshold.setRange(0.05, 0.95)
        self._threshold.setSingleStep(0.05)
        self._threshold.setValue(0.20)
        self._threshold.setToolTip(
            "A first arrival is picked where the trace, with AGC when AGC is on, first "
            "reaches this fraction of its peak in the search window and five times the "
            "noise before it (lower = earlier).")
        tform.addRow("Threshold (of trace peak)", self._threshold)
        self._pick_window = QDoubleSpinBox()
        self._pick_window.setRange(1.0, 100000.0)
        self._pick_window.setDecimals(0)
        self._pick_window.setValue(150.0)
        self._pick_window.setSuffix(" ms")
        self._pick_window.setToolTip("The picker looks for first arrivals up to this time.")
        tform.addRow("Latest first arrival", self._pick_window)
        pbox.addLayout(tform)
        self._screen_picks = QCheckBox("Leave out stray picks and shots")
        self._screen_picks.setChecked(True)
        self._screen_picks.setToolTip(
            "Leave out what still strays once the stray picks are picked again: a pick that "
            "breaks its shot's first-arrival curve (a first arrival cannot come earlier farther "
            "from the shot), and, with all shots picked, a shot whose times disagree with their "
            "reciprocals and a pick that still disagrees with the neighbouring shots at its "
            "geophone once picked again.")
        pbox.addWidget(self._screen_picks)
        auto_row = QHBoxLayout()
        auto_btn = QPushButton("Auto-pick this shot")
        auto_btn.setIcon(theme.icon("fa5s.magic"))
        auto_btn.setToolTip("Pick this record's first arrivals as the seismic workflow does; a "
                            "pick that strays from the shot's first-arrival curve is picked again "
                            "along it (AIC picker, Maeda 1985).")
        auto_btn.clicked.connect(self._auto_pick)
        all_btn = QPushButton("All shots")
        all_btn.setIcon(theme.icon("fa5s.layer-group"))
        all_btn.setToolTip("Auto-pick every shot record, then check each shot against its "
                           "reciprocals and each pick against the neighbouring shots at the same "
                           "geophone. Manual picks are kept.")
        all_btn.clicked.connect(self._auto_pick_all)
        auto_row.addWidget(auto_btn)
        auto_row.addWidget(all_btn)
        pbox.addLayout(auto_row)
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
        theme.set_tone(acc, "hint")
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
        theme.set_tone(self._tt_status, "hint")
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
        self._srt_stop = self.stop_button("The SRT inversion")
        runbox.addWidget(progress_with_stop(self._srt_progress, self._srt_stop))
        self._srt_export_btn = QPushButton("Export velocity model…")
        self._srt_export_btn.setIcon(theme.icon("fa5s.cube"))
        self._srt_export_btn.setToolTip(
            "Write the recovered velocity model as npy, the mesh, and a VTK file.")
        self._srt_export_btn.setEnabled(False)
        self._srt_export_btn.clicked.connect(self._export_velocity_model)
        runbox.addWidget(self._srt_export_btn)
        runbox.addWidget(self.map_export_button())
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

    # -- reflection ------------------------------------------------------------
    # One column in the order the work is done: which traces, how they are
    # cleaned, the velocity, the stack, then Run. Every control is named by what
    # it does to the data; the numbers behind each step stay in the library
    # (data_processing.seismic_shallow) with its defaults, which suit a hammer
    # line of a few tens of metres.
    def _build_reflection_panel(self) -> QScrollArea:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        self._refl_source = QLabel()
        self._refl_source.setWordWrap(True)
        theme.set_tone(self._refl_source, "hint")
        layout.addWidget(self._refl_source)

        traces = QGroupBox("Traces")
        tbox = QVBoxLayout(traces)
        tform = QFormLayout()
        tbox.addLayout(tform)
        self._refl_skip = QLineEdit()
        self._refl_skip.setPlaceholderText("none, or e.g. 1, 5, 12")
        self._refl_skip.setToolTip(
            "Shots to leave out of the stack, by record number: a mistimed trigger, a "
            "missed blow, wind or traffic. Look through the shots on the Gather tab first.")
        tform.addRow("Leave out shots", self._refl_skip)
        self._refl_drop_clipped = QCheckBox("Leave out clipped traces")
        self._refl_drop_clipped.setChecked(True)
        self._refl_drop_clipped.setToolTip(
            "A trace whose recorder saturated - usually the geophones next to the hammer - "
            "has its peaks cut flat, which distorts every wave on it.")
        tform.addRow(self._refl_drop_clipped)
        self._refl_drop_early = QCheckBox("Leave out traces with signal before the first arrival")
        self._refl_drop_early.setChecked(True)
        self._refl_drop_early.setToolTip(
            "Nothing can reach a geophone two metres from the hammer in the first "
            "1.5 ms, so a trace with signal there has a trigger or wiring fault.")
        tform.addRow(self._refl_drop_early)
        # Notes sit under the form, not in a row of it: a wrapped label spanning a
        # form row is given one line's height and cut off.
        self._refl_traces_note = self._reflection_note()
        tbox.addWidget(self._refl_traces_note)
        layout.addWidget(traces)

        clean = QGroupBox("Clean-up")
        cbox = QVBoxLayout(clean)
        cform = QFormLayout()
        cbox.addLayout(cform)
        self._refl_align = QCheckBox("Line up shot timing on the air wave")
        self._refl_align.setChecked(True)
        self._refl_align.setToolTip(
            "A hammer trigger fires a little early or late on each blow. The sound of the "
            "blow crosses the line at the speed of sound, so where it arrives tells each "
            "shot's timing error, which is then taken out.")
        cform.addRow(self._refl_align)
        self._refl_remove_air = QCheckBox("Remove the air wave")
        self._refl_remove_air.setChecked(True)
        self._refl_remove_air.setToolTip(
            "Subtract the sound of the blow and mute what is left of it, so it cannot "
            "stack into something that looks like a reflection.")
        cform.addRow(self._refl_remove_air)
        self._refl_low = make_double_spinbox(50.0, 1.0, 5000.0, 10.0, decimals=0, suffix=" Hz")
        self._refl_high = make_double_spinbox(250.0, 10.0, 20000.0, 10.0, decimals=0, suffix=" Hz")
        band = merged_row(self._refl_low, "to", self._refl_high)
        band.setToolTip(
            "The frequencies kept. The filter only looks back in time, so it cannot smear "
            "later energy such as ground roll into earlier times.")
        cform.addRow("Keep frequencies", band)
        self._refl_clean_note = self._reflection_note()
        cbox.addWidget(self._refl_clean_note)
        layout.addWidget(clean)

        vel = QGroupBox("Velocity")
        vbox = QVBoxLayout(vel)
        # The form on a widget of its own: set_rows_visible finds a row's label
        # through the field's parent widget's layout, which must be the form.
        form_holder = QWidget()
        vform = QFormLayout(form_holder)
        vform.setContentsMargins(0, 0, 0, 0)
        vbox.addWidget(form_holder)
        self._refl_vsource = QComboBox()
        self._refl_vsource.addItem("Values set here", "values")
        self._refl_vsource.addItem("The refraction model", "model")
        self._refl_vsource.setToolTip(
            "The refraction model is the Velocity model tab's result, averaged along the "
            "stacked part of the line. Only the part the rays reached is used.")
        self._refl_vsource.currentIndexChanged.connect(self._sync_reflection_velocity)
        vform.addRow("Velocity from", self._refl_vsource)
        self._refl_v1 = make_double_spinbox(300.0, 50.0, 6000.0, decimals=0, step=10.0,
                                            suffix=" m/s")
        self._refl_v1.setToolTip("How fast the waves travel in the ground below the line.")
        vform.addRow("Velocity", self._refl_v1)
        self._refl_layer = QCheckBox("Faster below a certain time")
        self._refl_layer.setToolTip(
            "Use a second, faster velocity below a two-way time - for example below the "
            "water table, where the refraction on the Travel-time tab shows a faster layer.")
        vform.addRow(self._refl_layer)
        self._refl_t2 = make_double_spinbox(12.0, 0.5, 1000.0, decimals=1, step=0.5, suffix=" ms")
        vform.addRow("From", self._refl_t2)
        self._refl_v2 = make_double_spinbox(900.0, 50.0, 6000.0, decimals=0, step=10.0,
                                            suffix=" m/s")
        vform.addRow("Velocity there", self._refl_v2)
        self._refl_layer.toggled.connect(self._sync_reflection_velocity)
        self._refl_model_note = self._reflection_note()
        vbox.addWidget(self._refl_model_note)
        check = self._reflection_note("Check it with Show ▸ Velocity check: the blue line "
                                      "should run through the dark patches.")
        vbox.addWidget(check)
        layout.addWidget(vel)
        self._sync_reflection_velocity()

        stack = QGroupBox("Stack")
        sform = QFormLayout(stack)
        self._refl_off_lo = make_double_spinbox(2.0, 0.0, 10000.0, decimals=1, step=0.5,
                                                suffix=" m")
        self._refl_off_hi = make_double_spinbox(10.0, 0.5, 10000.0, decimals=1, step=0.5,
                                                suffix=" m")
        offsets = merged_row(self._refl_off_lo, "to", self._refl_off_hi)
        offsets.setToolTip(
            "Shot-to-geophone distances that go into the stack. The nearest are clipped "
            "or swamped by the blow; the farthest mostly carry ground roll.")
        sform.addRow("Offsets", offsets)
        self._refl_bin = make_double_spinbox(0.5, 0.05, 100.0, decimals=2, step=0.25, suffix=" m")
        self._refl_bin.setToolTip("Spacing of the stacked section's traces. The geophone "
                                  "spacing is a good choice.")
        sform.addRow("Trace spacing", self._refl_bin)
        self._refl_tmax = make_double_spinbox(50.0, 5.0, 5000.0, decimals=0, step=5.0, suffix=" ms")
        sform.addRow("Down to", self._refl_tmax)

        self._refl_more = QCheckBox("More settings")
        sform.addRow(self._refl_more)
        mform = sform
        self._refl_stretch = make_double_spinbox(30.0, 5.0, 100.0, decimals=0, step=5.0, suffix=" %")
        self._refl_stretch.setToolTip(
            "The velocity correction stretches the far traces' wavelets; past this much "
            "stretch they are muted rather than stacked.")
        mform.addRow("Stretch limit", self._refl_stretch)
        self._refl_min_fold = make_spinbox(3, 1, 100)
        self._refl_min_fold.setToolTip("A point of the section with fewer traces than this "
                                       "is left blank.")
        mform.addRow("Traces per point, at least", self._refl_min_fold)
        self._refl_ground_roll = make_double_spinbox(220.0, 20.0, 3000.0, decimals=0, step=10.0,
                                                     suffix=" m/s")
        self._refl_ground_roll.setToolTip(
            "Everything that travels slower than this - ground roll - is muted before "
            "stacking.")
        mform.addRow("Ground roll slower than", self._refl_ground_roll)
        more = (self._refl_stretch, self._refl_min_fold, self._refl_ground_roll)
        self._refl_more.toggled.connect(lambda on: set_rows_visible(more, on))
        set_rows_visible(more, False)
        layout.addWidget(stack)

        run = QGroupBox("Run")
        runbox = QVBoxLayout(run)
        self._refl_btn = QPushButton("Stack the line")
        self._refl_btn.setProperty("primary", True)
        self._refl_btn.setIcon(theme.icon("fa5s.layer-group", color="#ffffff"))
        self._refl_btn.clicked.connect(self._run_reflection)
        runbox.addWidget(self._refl_btn)
        self._refl_progress = QProgressBar()
        self._refl_progress.setRange(0, 0)
        self._refl_progress.setVisible(False)
        runbox.addWidget(self._refl_progress)
        self._refl_summary = QLabel()
        self._refl_summary.setWordWrap(True)
        self._refl_summary.setTextFormat(Qt.RichText)
        runbox.addWidget(self._refl_summary)
        self._refl_export_btn = QPushButton("Export section…")
        self._refl_export_btn.setIcon(theme.icon("fa5s.file-export"))
        self._refl_export_btn.setToolTip("Save the section as a picture (PNG) or as numbers "
                                         "(NumPy .npz, with the velocity and fold).")
        self._refl_export_btn.setEnabled(False)
        self._refl_export_btn.clicked.connect(self._export_reflection)
        runbox.addWidget(self._refl_export_btn)
        self._refl_map_btn = self.map_export_button()
        self._refl_map_btn.setToolTip("Save the stacked section, in depth, as a survey in the "
                                      "Project Map, where it can be read beside the wells.")
        self._refl_map_btn.setEnabled(False)
        runbox.addWidget(self._refl_map_btn)
        layout.addWidget(run)
        layout.addStretch(1)

        scroll = ContentWidthScrollArea(minimum=450, maximum=500)
        scroll.setWidget(panel)
        scroll.setVisible(False)
        return scroll

    @staticmethod
    def _reflection_note(text: str = "") -> QLabel:
        """A hint line under a group's settings; hidden while it has nothing to say."""
        note = QLabel(text)
        note.setTextFormat(Qt.RichText)
        note.setWordWrap(True)
        theme.set_tone(note, "hint")
        note.setVisible(bool(text))
        return note

    @staticmethod
    def _set_note(note: QLabel, text: str) -> None:
        note.setText(text)
        note.setVisible(bool(text))

    def _on_center_tab_changed(self, _index: int = 0) -> None:
        """One side panel per tab: the controls for what is on screen, nothing else.

        Gather shows loading, display and picking. Travel-time and the two
        result tabs show the inversion instead: gain, clip and the picker decide
        nothing once the picks exist, and leaving them up costs the width the
        plot wants. The result tabs keep the inversion side so the settings that
        produced a model stay readable next to it and a re-run does not mean
        navigating back. Reflection has its own column: stacking shares the
        shots and their geometry with picking, and nothing else.
        """
        current = self._center_tabs.currentWidget()
        on_gather = current is self._viewer
        on_reflection = current is self._refl_view
        self._processing_panel.setVisible(on_gather)
        self._reflection_panel.setVisible(on_reflection)
        self._inversion_panel.setVisible(not on_gather and not on_reflection)
        if on_reflection:
            self._refresh_reflection_source()

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
            # Every trace: this is the working dataset - its shots are picked and
            # inverted - not a preview. A 4000-trace cap dropped the later shots
            # without a word, and the metadata then reported the cut count.
            dataset = read_segy(str(p))
            # The header's whole microseconds against the acquisition record's
            # interval, as the seismic workflow reads it.
            note = apply_record_interval(dataset, str(p))
            return {"kind": "dataset", "dataset": dataset, "warning": note or ""}
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
        self._reset_reflection()
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
        self._geometry_from_headers(dataset)
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
        self._spacing_from_headers = False
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

    def _geophone_xs(self) -> List[float]:
        """x of every geophone of the record on screen, where a pick would put it."""
        count = (int(self._raw.shape[1]) if self._raw is not None
                 else len(self._geo_positions or {}))
        return [float(self._receiver_position(trace)[0]) for trace in range(count)]

    def _on_geometry_changed(self) -> None:
        """Put the picks' geophones where the spacing and geophone-0 x now say.

        A pick stamps its geophone's position when it is made, and the
        travel-time plot, the exported travel times and the SRT inversion all
        read that stamp. Nothing followed these two settings, so picks taken at
        the default 1 m stayed 1 m apart after the spacing was set to 0.5 m -
        offsets twice too long, and velocities twice too high.
        """
        self._spacing_from_headers = False   # set by hand now, whatever the file said
        if self._geo_positions:
            # A position file places every geophone; these boxes only show its
            # first position and step, so they go back to saying that.
            xs = [self._geo_positions[k][0] for k in sorted(self._geo_positions)]
            for box, value in ((self._geo_start, xs[0]),
                               (self._spacing, abs(xs[1] - xs[0]) if len(xs) > 1 else None)):
                if value is not None:
                    box.blockSignals(True); box.setValue(float(value)); box.blockSignals(False)
            self.log("The geophones are placed from the loaded position file, so the "
                     "spacing and geophone-0 x are not used for them.", "warn")
            return
        self._restamp_all_picks()
        self._redraw_markers()
        self._update_pick_info()
        self._tt_plot.enableAutoRange()      # the line moved; fit the plot to it
        self._publish()

    @staticmethod
    def _parse_geometry_file(path: str) -> Dict[int, Tuple[float, float]]:
        """Parse a geophone position/topography file into ``{trace_index: (x, z)}``.

        Whitespace/comma separated; a non-numeric header row is skipped. 3+ columns
        read as (station, x, elevation); 2 as (x, elevation); 1 as x with elevation
        0. With a station column the geophones are taken in station order, so a
        file listing stations 3, 1, 2 still puts station 1 on the first trace;
        without one they follow the file's row order.
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
        if rows and all(len(nums) >= 3 for nums in rows):
            # Stable, so repeated station numbers keep their file order.
            rows.sort(key=lambda nums: nums[0])
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

    def _pick_record(self, record: int, traces: np.ndarray, headers, shot_x: float):
        """One record's automatic picks on this page's geometry: ``(picks, repicked)``.

        The seismic workflow's picker (``pick_and_correct``): threshold picks on
        the traces with AGC when AGC is on, placed at the geophone positions and
        shot x set here, and those off their shot's first-arrival curve picked
        again along it. ``trace_index`` is the trace's column in the record.
        """
        import dataclasses

        shot_z = self._interp_topography(shot_x)

        def place(picks):
            placed = []
            for pick in picks:
                trace = int(pick.trace_index)
                receiver_x, receiver_z = self._receiver_position(trace)
                placed.append(dataclasses.replace(
                    pick, source_id=int(record), receiver_id=trace + 1,
                    source_x=float(shot_x), source_z=float(shot_z),
                    receiver_x=float(receiver_x), receiver_z=float(receiver_z),
                    field_record=int(record), trace_number=trace + 1))
            return placed

        agc = self._agc_window.value() / 1000.0 if self._agc.isChecked() else 0.0
        picks, repicked = pick_and_correct(
            np.asarray(traces, dtype=float), dt=self._dt, headers=headers, agc_window=agc,
            threshold=self._threshold.value(), max_time=self._pick_window.value() / 1000.0,
            place=place)
        # A trace the picker and the re-pick left at time zero has no arrival to show.
        return [p for p in picks if np.isfinite(p.time_s) and p.time_s > 0], repicked

    def _picker_label(self) -> str:
        agc = f", AGC {self._agc_window.value():g} ms" if self._agc.isChecked() else ", no AGC"
        return f"threshold {self._threshold.value():.2f} of each trace's peak{agc}"

    def _auto_pick_ready(self) -> bool:
        if self._raw is None:
            self.log("Load seismic data first.", "warn")
            return False
        if not _SEISMIC_OK:
            self.log(f"Auto-pick needs the seismic processing module ({_SEISMIC_ERR}).", "warn")
            return False
        if not self._dt:
            self.log("Auto-pick needs the sample interval, and this file does not give one; "
                     "pick the first arrivals by hand instead.", "warn")
            return False
        return True

    def _auto_pick(self) -> None:
        """Pick the record on screen as the seismic workflow would.

        Without the other shots there are no reciprocals or neighbouring shots
        to check against, so only the curve check leaves picks out here; "All
        shots" adds the other two.
        """
        if not self._auto_pick_ready():
            return
        record = self._current_record if self._current_record is not None else 1
        try:
            picks, repicked = self._pick_record(record, self._raw, self._headers, self._shot_x.value())
            rejected = []
            if self._screen_picks.isChecked():
                screen = screen_picks(picks, reciprocity_check=False, neighbour_check=False)
                picks, rejected = screen.kept, screen.rejected
        except Exception as exc:  # noqa: BLE001
            self.log(f"Auto-pick failed: {exc}", "error")
            return
        again = {int(p.trace_index) for p in repicked}
        self._picks = {int(p.trace_index): p for p in picks}
        self._pick_src = {t: "repick" if t in again else "auto" for t in self._picks}
        self._order = sorted(self._picks)
        message = (f"Auto-picked {len(self._picks) + len(rejected)} first arrivals on shot "
                   f"{record} ({self._picker_label()})")
        if again:
            message += (f"; {len(again)} that strayed from the shot's first-arrival curve were "
                        "picked again along it (AIC picker, Maeda 1985)")
        if rejected:
            message += (f"; {len(rejected)} left out for breaking it (trace "
                        f"{', '.join(str(int(p.trace_index) + 1) for p in rejected)})")
        self.log(message + ".", "success")
        self._redraw_markers()
        self._update_pick_info()
        self._publish()

    def _auto_pick_all(self) -> Optional[Dict[str, Any]]:
        """Pick every shot record as the seismic workflow does, all three checks included.

        Each record is picked at its own shot x (set by hand, from a regular
        shot layout, or from the headers). Picks made by hand are kept, in
        place of the automatic ones on their traces; a shot left out for its
        reciprocals loses only its automatic picks. Returns what was done, for
        the page's agent tool.
        """
        records = self._agent_records()
        if self._dataset is None or len(records) < 2:
            self._auto_pick()
            return None
        if not self._auto_pick_ready():
            return None
        self._save_current_picks()
        picks, repicked = [], []
        gathers: Dict[int, np.ndarray] = {}
        try:
            for record in records:
                gather = self._dataset.get_gather(int(record))
                gathers[int(record)] = np.asarray(gather.traces, dtype=float)
                shot_x = self._shot_pos.get(record, self._default_shot_x(record))
                mine, again = self._pick_record(int(record), gather.traces, gather.headers, shot_x)
                picks += mine
                repicked += again
            check = self._screen_picks.isChecked()
            # trace_index is the trace's column in its own record.
            screen = screen_picks(
                picks, monotonic_check=check, reciprocity_check=check, neighbour_check=check,
                traces=lambda p: gathers[int(p.field_record)][:, int(p.trace_index)], dt=self._dt)
        except Exception as exc:  # noqa: BLE001
            self.log(f"Auto-pick failed: {exc}", "error")
            return None
        kept, dropped = screen.kept, screen.dropped
        rejected = list(screen.rejected) + list(screen.neighbour_rejected)
        manual = {record: {trace: self._all_picks[record][trace]
                           for trace, source in sources.items()
                           if source == "manual" and trace in self._all_picks.get(record, {})}
                  for record, sources in self._all_src.items()}
        again = {(int(p.field_record), int(p.trace_index))
                 for p in list(repicked) + list(screen.neighbour_repicked)}
        for record in records:
            self._all_picks[record] = {}
            self._all_src[record] = {}
        for pick in kept:
            record, trace = int(pick.field_record), int(pick.trace_index)
            self._all_picks.setdefault(record, {})[trace] = pick
            self._all_src.setdefault(record, {})[trace] = (
                "repick" if (record, trace) in again else "auto")
        n_manual = 0
        for record, by_trace in manual.items():
            for trace, pick in by_trace.items():
                self._all_picks.setdefault(record, {})[trace] = pick
                self._all_src.setdefault(record, {})[trace] = "manual"
                n_manual += 1
        current = self._current_record
        self._picks = dict(self._all_picks.get(current, {}))
        self._pick_src = dict(self._all_src.get(current, {}))
        self._order = sorted(self._picks)

        # Counted as the seismic workflow counts them, so the two read alike.
        message = (f"Auto-picked {len(picks)} first arrivals on {len(records)} shots "
                   f"({self._picker_label()})")
        if repicked:
            message += (f"; {len(repicked)} that strayed from their shot's first-arrival curve "
                        "were picked again along it (AIC picker, Maeda 1985)")
        if screen.rejected:
            message += (f"; {len(screen.rejected)} single picks left out for breaking their "
                        "shot's curve")
        if screen.neighbour_repicked:
            message += (f"; {len(screen.neighbour_repicked)} that disagreed with the "
                        "neighbouring shots at the same geophone were picked again where those "
                        "put them")
        if screen.neighbour_rejected:
            message += (f"; {len(screen.neighbour_rejected)} left out for still disagreeing "
                        "with them")
        if dropped:
            message += f"; {len(dropped)} shot{'s' if len(dropped) != 1 else ''} left out (below)"
        if n_manual:
            message += f"; {n_manual} picks made by hand kept"
        self.log(f"{message}. {len(kept)} automatic picks stand.", "success")
        records_at: Dict[float, List[int]] = {}
        for pick in picks:
            records_at.setdefault(round(float(pick.source_x), 3), [])
            if int(pick.field_record) not in records_at[round(float(pick.source_x), 3)]:
                records_at[round(float(pick.source_x), 3)].append(int(pick.field_record))
        left_out = []
        for shot in dropped:
            which = records_at.get(round(float(shot["source_x"]), 3), [])
            left_out += which
            self.log(f"Shot {', '.join(map(str, which)) or '?'} (x = {shot['source_x']:g} m) "
                     f"left out: its times differ from their reciprocals by a median of "
                     f"{shot['median_difference_ms']:g} ms over {shot['pairs']} pairs, a timing "
                     "error of its own. Pick it by hand to keep it.", "warn")
        self._redraw_markers()
        self._update_pick_info()
        self._publish()
        return {"picks": len(kept), "repicked": len(again), "rejected": len(rejected),
                "neighbour_repicked": len(screen.neighbour_repicked),
                "neighbour_rejected": len(screen.neighbour_rejected),
                "manual_kept": n_manual, "shots_left_out": dropped, "records_left_out": left_out}

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
        self._geometry_debounced.flush()     # a spacing just typed moves the picks first
        self._tt_plot.clear()
        self._tt_shows_container = False
        # The geophones, on the t = 0 line, so the plot shows the line the picks
        # are placed on before there are any. Empty, it kept the range of what it
        # last drew, which read as the spacing not having taken.
        geophones = self._geophone_xs()
        if geophones:
            self._tt_plot.plot(geophones, [0.0] * len(geophones), pen=None, symbol="t1",
                               symbolSize=7, symbolBrush="#7a8794", symbolPen=None)
            if not self._geo_positions:
                origin = ("from the file headers" if self._spacing_from_headers
                          else "(no position file)")
                self._geo_info.setText(
                    f"Even spacing {origin}: {len(geophones)} geophones, "
                    f"x {min(geophones):g} to {max(geophones):g} m.")
        picks = self._all_first_breaks()
        if not picks:
            self._tt_plot.enableAutoRange()
            return
        from collections import defaultdict

        # First-pick plot: travel time vs ABSOLUTE geophone position,
        # one connected branch per shot, each shot marked with a star at t = 0.
        by_shot = defaultdict(list)
        shot_x: Dict[int, float] = {}
        for p in picks:
            by_shot[int(p.source_id)].append((float(p.receiver_x), float(p.time_s) * 1000.0))
            shot_x[int(p.source_id)] = float(p.source_x)
        colors = ["#007aff", "#ff3b30", "#34c759", "#af52de", "#ff9500", "#30b0c7", "#a2845e", "#ff2d55"]
        for i, shot in enumerate(sorted(by_shot, key=lambda s: shot_x.get(s, 0.0))):
            pts = sorted(by_shot[shot])  # by geophone position
            xs = [a for a, _ in pts]
            ys = [b for _, b in pts]
            color = colors[i % len(colors)]
            self._tt_plot.plot(xs, ys, pen=pg.mkPen(color, width=1.5), symbol="o", symbolSize=5,
                               symbolBrush=color, symbolPen=None,
                               name=self._shot_legend(shot_x.get(shot, 0.0)))
            # shot location on the t = 0 baseline
            self._tt_plot.plot([shot_x.get(shot, 0.0)], [0.0], pen=None, symbol="star",
                               symbolSize=15, symbolBrush=color, symbolPen=pg.mkPen("#222", width=0.8))

    # -- export --------------------------------------------------------------
    def _ordered_picks(self) -> list:
        self._geometry_debounced.flush()     # an export reads the geophones stamped here
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
        self._geometry_debounced.flush()     # so does the inversion
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
        self._tt_shows_container = True
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
        colors = ["#007aff", "#ff3b30", "#34c759", "#af52de", "#ff9500", "#30b0c7", "#a2845e", "#ff2d55"]
        for i, shot in enumerate(sorted(by_shot, key=lambda ss: shot_x.get(ss, 0.0))):
            pts = sorted(by_shot[shot])
            xs = [a for a, _ in pts]; ys = [b for _, b in pts]
            color = colors[i % len(colors)]
            self._tt_plot.plot(xs, ys, pen=pg.mkPen(color, width=1.5), symbol="o", symbolSize=5,
                               symbolBrush=color, symbolPen=None,
                               name=self._shot_legend(shot_x.get(shot, 0.0)))
            self._tt_plot.plot([shot_x.get(shot, 0.0)], [0.0], pen=None, symbol="star",
                               symbolSize=15, symbolBrush=color, symbolPen=pg.mkPen("#222", width=0.8))

    @staticmethod
    def _shot_legend(shot_x: float) -> str:
        """A shot's legend entry, in the unit the position axis is ticked in."""
        return f"shot @ {to_display_length(shot_x):.0f} {length_units.current()}"

    def _on_length_unit_changed(self, _unit: str) -> None:
        """Retick the travel-time plot and redraw it, so the legend follows."""
        length_units.pyqtgraph_axis(self._tt_plot, "bottom", "geophone position x")
        self._sync_reflection_velocity()       # its note gives depths in the unit
        if self._tt_shows_container and self._tt_data is not None:
            self._plot_tt_container(self._tt_data)
        else:
            self._update_tt_qc()

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
                "mixed" if "manual" in source_values and source_values & {"auto", "repick"}
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
        # In a process of its own: the ray tracing and PyGIMLi's solves would
        # otherwise hold the window still. The fitted model comes back as what
        # the section draws - velocity, coverage, ray paths.
        worker = ProcessWorkflowWorker(recipe_path, project_root, run.outputs_dir,
                                       run.result_path, objects=("manager", "convergence"))
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
        self._srt_stop.attach(self._srt_worker, "seismic.srt_inversion")

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
        self._sync_reflection_velocity()      # the Reflection tab can stack with it now
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
        self.offer_map_export()

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
            from PyHydroGeophysX.core.mesh_serialization import via_ascii_path
            from PyHydroGeophysX.data_processing.model_csv import export_model_csv

            out = io_utils.ensure_dir(folder)
            mesh = mgr.paraDomain
            velocity = np.asarray(velocity_of(mgr), dtype=float)
            np.save(out / "velocity_model.npy", velocity)
            try:
                coverage = np.asarray(mgr.coverage(), dtype=float)
            except Exception:  # noqa: BLE001 - ray coverage is optional
                coverage = None
            export_model_csv(
                out, mesh, velocity,
                value_name="velocity", units="m/s", coverage=coverage,
            )
            # PyGIMLi's writers take a narrow path and cannot open one Windows'
            # ANSI codepage cannot represent; via_ascii_path stages those writes.
            # Without it an export to a localized folder leaves only the .npy.
            via_ascii_path(mesh.save, out / "velocity_mesh.bms", mode="write")
            mesh["velocity"] = velocity
            via_ascii_path(mesh.exportVTK, out / "velocity_model.vtk", mode="write")
            self.log(f"Exported velocity model (csv + npy + bms + vtk) to {out}", "success")
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

    # -- reflection: run -------------------------------------------------------
    def _refresh_reflection_source(self) -> None:
        """Say in one line what the Stack button would stack."""
        if not hasattr(self, "_refl_source"):
            return  # still building
        ready = self._dataset is not None and bool(self._dt) and _SEISMIC_OK
        self._refl_btn.setEnabled(ready and self._refl_busy is None)
        if not ready:
            self._refl_source.setText(
                "Load a shot file on the Gather tab first: stacking needs several shots "
                "recorded along one line (SEG-Y or Geometrics DAT).")
            return
        self._geometry_debounced.flush()
        records = self._agent_records()
        xs = self._geophone_xs()
        step = abs(xs[1] - xs[0]) if len(xs) > 1 else 0.0
        self._refl_source.setText(
            f"Stacks the {len(records)} shots loaded on the Gather tab, with the geophones "
            f"and shot positions set there (geophones every {step:g} m).")

    def _reflection_line(self) -> Tuple[np.ndarray, Any]:
        """Every shot's traces side by side, and where each was recorded.

        From the Gather tab's geometry, the same positions the picks are stamped
        with, so the stack and the travel times describe one line.
        """
        self._geometry_debounced.flush()
        blocks, record, channel, sx, gx, sz, gz = [], [], [], [], [], [], []
        for rec in self._agent_records():
            gather = self._dataset.get_gather(int(rec))
            arr = np.asarray(gather.traces, dtype=float)
            if rec == self._current_record:
                shot = float(self._shot_x.value())
            else:
                shot = float(self._shot_pos.get(rec, self._default_shot_x(rec)))
            shot_z = self._interp_topography(shot)
            for i in range(arr.shape[1]):
                x, z = self._receiver_position(i)
                header = gather.headers[i] if i < len(gather.headers) else None
                record.append(int(rec))
                channel.append(int(header.trace_number) if header is not None else i + 1)
                sx.append(shot); sz.append(shot_z); gx.append(x); gz.append(z)
            blocks.append(arr)
        traces = np.concatenate(blocks, axis=1)
        geometry = shallow.LineGeometry(record, channel, sx, gx, sz, gz, origin="Gather tab")
        return traces, geometry

    # -- reflection: velocity ----------------------------------------------------
    def _sync_reflection_velocity(self, *_args) -> None:
        """Show the rows of the velocity source chosen, and what the model would give."""
        if not hasattr(self, "_refl_model_note"):
            return  # still building
        from_model = self._refl_vsource.currentData() == "model"
        set_rows_visible((self._refl_v1, self._refl_layer), not from_model)
        set_rows_visible((self._refl_t2, self._refl_v2),
                         not from_model and self._refl_layer.isChecked())
        if not from_model:
            self._set_note(self._refl_model_note, "")
            return
        try:
            velocity, covered = self._model_velocity()
        except ValueError as exc:
            self._set_note(self._refl_model_note, str(exc))
            return
        self._set_note(self._refl_model_note, self._describe_model_velocity(velocity, covered))

    def _stack_x_range(self, geometry: Any = None) -> Tuple[float, float]:
        """Where along the line the stack's points fall: the midpoints of its offsets."""
        if geometry is None:
            geometry = self._reflection_line()[1]
        near, far = self._refl_off_lo.value(), self._refl_off_hi.value()
        off = geometry.abs_offset
        mid = geometry.midpoint[(off >= near) & (off <= far)]
        if mid.size == 0:
            raise ValueError("No trace has an offset in the range set under Stack.")
        return float(mid.min()), float(mid.max())

    def _model_velocity(self, geometry: Any = None) -> Tuple[Any, float]:
        """The refraction model as a stacking velocity, and the deepest depth the rays reach.

        Averaged along the stacked part of the line, over the cells the rays
        crossed: below them a travel-time model is the inversion's starting
        guess, not a measurement.
        """
        mgr = getattr(self, "_srt_mgr", None)
        if mgr is None:
            raise ValueError("No refraction model yet: pick the first arrivals and run the "
                             "inversion on the Travel-time tab, or set the values here.")
        if self._dataset is None:
            raise ValueError("Load the shots on the Gather tab first.")
        from PyHydroGeophysX.core.section_geometry import cell_centers, cell_depths

        mesh = mgr.paraDomain
        values = np.asarray(velocity_of(mgr), dtype=float)
        x = cell_centers(mesh)[:, 0]
        depth = cell_depths(mesh)
        keep = np.ones(values.size, dtype=bool)
        for method, threshold in (("standardizedCoverage", 0.5), ("coverage", 0.0)):
            try:
                cov = np.asarray(getattr(mgr, method)(), dtype=float)
            except Exception:  # noqa: BLE001 - the engine may not report coverage
                continue
            if cov.size == values.size and (cov > threshold).any():
                keep = cov > threshold
                break
        x_range = self._stack_x_range(geometry)
        inside = keep & (x >= x_range[0]) & (x <= x_range[1])
        if not inside.any():
            raise ValueError(f"The refraction model has no ray-covered cells between "
                             f"x = {x_range[0]:g} and {x_range[1]:g} m, where the stack lies.")
        velocity = shallow.velocity_from_section(
            x[keep], depth[keep], values[keep], x_range,
            label=f"the refraction model, x = {x_range[0]:g}-{x_range[1]:g} m")
        return velocity, float(depth[inside].max())

    @staticmethod
    def _describe_model_velocity(velocity: Any, covered: float) -> str:
        unit = length_units.current()
        depths = [0.0, covered / 2.0, covered]
        times = np.interp(depths, velocity.depth(velocity.times), velocity.times)
        values = velocity.interval(times)
        # One short line per depth: a long sentence wraps "m/s" after its slash.
        lines = [f"{values[0]:,.0f} m/s at the surface",
                 f"{values[1]:,.0f} m/s at {to_display_length(depths[1], unit):.1f} {unit}",
                 f"{values[2]:,.0f} m/s at {to_display_length(depths[2], unit):.1f} {unit}, "
                 "as deep as the rays reach"]
        return ("From the refraction model, averaged along the stacked part of the line:<br>"
                + "<br>".join(f"&nbsp;&nbsp;{line}" for line in lines)
                + "<br>Deeper, the last velocity is kept.")

    def _reflection_params(self, geometry: Any = None) -> Dict[str, Any]:
        try:
            skip = parse_shot_list(self._refl_skip.text())
        except ValueError:
            raise ValueError("Write the shots to leave out as numbers, e.g. 1, 5, 12 "
                             "or 3-6.") from None
        unknown = sorted(set(skip) - set(self._agent_records()))
        if unknown:
            raise ValueError(f"There is no shot {', '.join(map(str, unknown))} in this file.")
        low, high = self._refl_low.value(), self._refl_high.value()
        if high <= low:
            raise ValueError("The upper frequency has to be above the lower one.")
        if self._dt and high >= 0.5 / self._dt:
            raise ValueError(f"The upper frequency has to stay below {0.5 / self._dt:.0f} Hz, "
                             "half the sampling rate.")
        near, far = self._refl_off_lo.value(), self._refl_off_hi.value()
        if far <= near:
            raise ValueError("The farthest offset has to be beyond the nearest.")
        ray_limit = None
        if self._refl_vsource.currentData() == "model":
            velocity, ray_limit = self._model_velocity(geometry)
        elif self._refl_layer.isChecked():
            velocity = shallow.VelocityFunction.layered(
                self._refl_v1.value(), self._refl_t2.value() * 1e-3, self._refl_v2.value())
        else:
            velocity = shallow.VelocityFunction.layered(self._refl_v1.value())
        return {
            "skip_shots": skip,
            "drop_clipped": self._refl_drop_clipped.isChecked(),
            "drop_early": self._refl_drop_early.isChecked(),
            "align": self._refl_align.isChecked(),
            "remove_air": self._refl_remove_air.isChecked(),
            "band": (low, high),
            "velocity": velocity,
            "ray_limit": ray_limit,
            "stack": {"offset_range": (near, far), "bin_width": self._refl_bin.value(),
                      "tmax": self._refl_tmax.value() * 1e-3,
                      "stretch": self._refl_stretch.value() / 100.0,
                      "min_fold": int(self._refl_min_fold.value()),
                      "surface_wave_velocity": self._refl_ground_roll.value()},
        }

    def _say_reflection(self, html: str) -> None:
        self._refl_summary.setText(html)

    def _run_reflection(self) -> Optional[str]:
        """Start the stack; returns why it could not start, or None."""
        self._refl_error = ""
        if self._dataset is None or not self._dt or not _SEISMIC_OK:
            self._refresh_reflection_source()
            return self._refl_source.text()
        if self._refl_worker is not None:
            return "A stack is already running."
        try:
            traces, geometry = self._reflection_line()
            params = self._reflection_params(geometry)
        except ValueError as exc:
            self._say_reflection(f"<span style='color:{theme.color('red')}'>{exc}</span>")
            return str(exc)
        key = (id(self._dataset), geometry.source_x.tobytes(), geometry.receiver_x.tobytes(),
               tuple(params["skip_shots"]), params["drop_clipped"], params["drop_early"],
               params["align"], params["remove_air"], params["band"])
        prepared = (self._refl_prepared[1]
                    if self._refl_prepared is not None and self._refl_prepared[0] == key else None)
        self._refl_busy = BusyStateController([self._refl_btn])
        self._refl_busy.start()
        self._refl_btn.setText("Stacking…")
        self._refl_progress.setVisible(True)
        self._say_reflection("Stacking with the new settings…" if prepared is not None else
                             "Checking the traces, timing and cleaning every shot, then "
                             "stacking…")
        worker = TaskWorker(_stack_reflections, traces, float(self._dt), geometry, params, prepared)
        worker.succeeded.connect(lambda out: self._on_reflection_done(key, out))
        worker.failed.connect(self._on_reflection_failed)
        worker.finished.connect(self._reset_reflection_btn)
        self._refl_worker = self.register_worker(worker)
        worker.start()
        return None

    def _on_reflection_done(self, key: tuple, out: Dict[str, Any]) -> None:
        prepared, result = out["prepared"], out["result"]
        self._refl_prepared = (key, prepared)
        self._refl_result = result
        self._refl_ray_limit = out.get("ray_limit")
        self._refl_view.set_ray_limit(self._refl_ray_limit)
        self._refl_view.show_result(result)
        self._refl_export_btn.setEnabled(True)
        self._refl_map_btn.setEnabled(True)

        counts = prepared.qc.counts()
        left_out = [f"{counts[k]} {label}" for k, label in (
            ("excluded record", "in the shots left out"), ("clipped", "clipped"),
            ("early energy", "with early signal"), ("dead", "dead")) if counts[k]]
        self._set_note(self._refl_traces_note,
                       f"Using {counts['kept']} of {counts['total']} traces."
                       + (" Left out: " + ", ".join(left_out) + "." if left_out else ""))
        st = prepared.statics
        if st is not None:
            shifts = [abs(v) for r, v in st.statics.items() if r not in st.median_records]
            # The speed first: at the end of a line QLabel breaks "m/s" after its slash.
            clean = (f"Air wave at {st.velocity:.0f} m/s; shot timing corrected by up to "
                     f"{max(shifts) * 1e3:.1f} ms.")
        else:
            clean = "Shot timing left as recorded."
        self._set_note(self._refl_clean_note, clean)

        stack = result.stack
        lines = [f"Stacked {result.traces_used} traces into {stack.cmp_x.size} points along "
                 f"the line, up to {stack.max_fold} traces per point."]
        sg = result.supergather
        unit = length_units.current()
        if sg is None:
            lines.append("<b>Not checked.</b> Too few traces in the middle of the line to test "
                         "whether an event stays flat across offsets.")
        elif not sg.passing:
            lines.append(f"<b style='color:{theme.color('amber')}'>No reflection confirmed.</b> "
                         "No event stays flat across the offsets, so the bands in the section "
                         "are most likely left over from the air-wave and ground-roll mutes.")
        else:
            found = []
            for event in sg.passing:
                depth = float(result.velocity.depth(np.array([event["t0"]]))[0])
                found.append(f"{event['t0'] * 1e3:.1f} ms (about "
                             f"{to_display_length(depth, unit):.1f} {unit} deep)")
            plural = len(found) > 1
            lines.append(f"<b style='color:{theme.color('green')}'>Possible reflection"
                         f"{'s' if plural else ''}</b> at {' and '.join(found)}: "
                         f"{'they stay' if plural else 'it stays'} flat across the offsets. "
                         "Add it to the Map and compare it with a well there before "
                         "reading it as a layer.")
        if (result.peaks_linear and result.peaks_hyperbolic
                and result.peaks_linear[0][2] > result.peaks_hyperbolic[0][2]):
            lines.append("The strongest energy lines up straight rather than curved: ground "
                         "roll or refractions, not reflections.")
        self._say_reflection("<br><br>".join(lines))
        self._refl_view.set_verdict(self._reflection_verdict(result))
        # Once the new summary is laid out, or it scrolls to where the old one ended.
        QTimer.singleShot(0, lambda: self._reflection_panel.ensureWidgetVisible(
            self._refl_summary))
        for note in prepared.notes + result.notes:
            self.log(f"Reflection: {note}", "info")
        for warning in prepared.warnings + result.warnings:
            self.log(f"Reflection: {warning}", "warn")
        self.log("Reflection stack finished.", "success")

    @staticmethod
    def _reflection_verdict(result: Any) -> str:
        """The section's title: what the flat-event check found, in a few words."""
        sg = result.supergather
        if sg is None:
            return "Not checked: too few traces to test for flat events"
        if not sg.passing:
            return "No reflection confirmed: no event stays flat across the offsets"
        times = ", ".join(f"{e['t0'] * 1e3:.1f} ms" for e in sg.passing)
        return f"Possible reflection at {times}: flat across the offsets"

    def _on_reflection_failed(self, message: str) -> None:
        self._refl_error = message
        self._say_reflection(f"<span style='color:{theme.color('red')}'>Could not stack: "
                             f"{message}</span>")
        self.log(f"Reflection stack failed: {message}", "error")

    def _reset_reflection_btn(self) -> None:
        if self._refl_busy is not None:
            self._refl_busy.finish()
            self._refl_busy = None
        self._refl_btn.setText("Stack the line")
        self._refl_progress.setVisible(False)
        self._refl_worker = None

    def _reset_reflection(self) -> None:
        """A new file: nothing stacked, and settings that suit its geometry."""
        self._refl_prepared = None
        self._refl_result = None
        self._refl_view.clear()
        self._refl_export_btn.setEnabled(False)
        self._refl_map_btn.setEnabled(False)
        for note in (self._refl_traces_note, self._refl_clean_note):
            self._set_note(note, "")
        self._refl_summary.setText("")
        self._refl_skip.clear()
        if self._dataset is not None:
            xs = self._geophone_xs()
            shots = [self._default_shot_x(r) for r in self._agent_records()]
            if xs and shots:
                farthest = max(abs(g - s) for g in xs for s in shots)
                step = abs(xs[1] - xs[0]) if len(xs) > 1 else self._spacing.value()
                self._refl_bin.setValue(step)
                self._refl_off_lo.setValue(min(2.0, 0.25 * farthest))
                self._refl_off_hi.setValue(round(min(farthest, max(10.0, 0.5 * farthest)), 1))
        self._refresh_reflection_source()
        self._sync_reflection_velocity()

    def map_snapshot(self):
        """What Add to Map saves from this page: the tab on screen decides.

        On the Reflection tab, the stacked section in depth at its stacking
        velocity (``project_map.section_snapshot``); elsewhere the refraction
        velocity model, as before.
        """
        from PyHydroGeophysX.qt_apps.project_map import mesh_snapshot, section_snapshot

        if self._center_tabs.currentWidget() is self._refl_view:
            result = self._refl_result
            if result is None:
                raise ValueError("Stack the line on the Reflection tab first.")
            st = result.stack
            z_of_t = np.asarray(result.velocity.depth(st.t0), dtype=float)
            depth_edges = np.linspace(0.0, float(z_of_t[-1]), 151)
            centres = 0.5 * (depth_edges[:-1] + depth_edges[1:])
            section = np.where(st.fold >= st.min_fold, st.stack, np.nan)
            values = np.array([np.interp(centres, z_of_t, column, left=np.nan, right=np.nan)
                               for column in section.T]).T
            return section_snapshot(st.bin_edges, depth_edges, values, "Seismic",
                                    "stack amplitude", "Reflection stack")
        manager = getattr(self, "_srt_mgr", None)
        if manager is None:
            raise ValueError("Run a seismic inversion first.")
        if getattr(manager, "paraDomain", None) is None:
            # The inversion ran in its own process and its mesh did not come
            # back (the run's warnings say why - a path too long, say).
            raise ValueError("The velocity model's mesh did not come back from the inversion; "
                             "run it again, from a project folder with a shorter path.")
        return mesh_snapshot(manager.paraDomain, velocity_of(manager), "Seismic")

    def _export_reflection(self) -> None:
        result = self._refl_result
        if result is None:
            return
        path, chosen = QFileDialog.getSaveFileName(
            self, "Export reflection section", "reflection_section.png",
            "PNG image (*.png);;NumPy archive (*.npz)")
        if not path:
            return
        if path.lower().endswith(".npz") or ("npz" in chosen and not path.lower().endswith(".png")):
            st = result.stack
            np.savez(path, position_m=st.cmp_x, time_s=st.t0,
                     depth_m=result.velocity.depth(st.t0), section=st.stack, traces=st.fold,
                     velocity_times_s=result.velocity.times,
                     velocity_m_s=result.velocity.velocities)
        else:
            self._refl_view.export_png(path)
        self.log(f"Exported the reflection section to {path}.", "success")

    def _geometry_from_headers(self, dataset) -> None:
        """Take the geophone spacing and first position from the file, when it has them.

        A SEG-Y line laid out along x carries each geophone's position. Left at
        the 1 m default, a 0.5 m line was picked and stacked at twice its offsets.
        Map coordinates are left alone: projecting them is a choice for the user.
        """
        self._spacing_from_headers = False
        if not _SEISMIC_OK or not shallow.headers_have_positions(dataset):
            return
        headers = dataset.headers
        if any(abs(float(h.receiver_y) - float(headers[0].receiver_y)) > 1e-6
               or abs(float(h.source_y) - float(headers[0].receiver_y)) > 1e-6 for h in headers):
            return
        first = dataset.get_gather(int(list(dataset.field_records)[0])).headers
        xs = np.array([float(h.receiver_x) for h in first])
        steps = np.diff(xs)
        if xs.size < 2 or steps[0] <= 0 or not np.allclose(steps, steps[0], atol=1e-3):
            return
        for box, value in ((self._spacing, float(steps[0])), (self._geo_start, float(xs[0]))):
            box.blockSignals(True)
            box.setValue(value)
            box.blockSignals(False)
        self._spacing_from_headers = True

    def export_actions(self):
        actions = []
        if getattr(self, "_srt_mgr", None) is not None:
            actions.append(("Add to Project Map…", self.add_to_map))
            actions.append((
                "Velocity model (CSV + npy + mesh + VTK)",
                self._export_velocity_model,
            ))
        if self._picks:
            actions.append(("First-arrival picks (CSV)", self._export_picks))
            actions.append(("Travel-time container (PyGIMLi .dat)", self._export_traveltime))
        return actions

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
                 "desc": ("Set processing/pick params. Picking: pick_threshold (fraction of "
                          "each trace's peak, 0.05-0.95), pick_max_time_ms (latest first "
                          "arrival), screen_picks (leave out stray picks and shots). Display: "
                          "display (image / wiggle / both), "
                          "gain (slider 1-100), clip_percentile, agc_window_ms, "
                          "flip_polarity, normalize, agc. Inversion: engine "
                          "(pyhydro/pygimli), lam, max_iterations, "
                          "max_total_iterations, plateau_tolerance (fraction). "
                          "Mesh: mesh_quality, para_depth (m, 0 = auto), "
                          "para_max_cell_size (0 = auto), secondary_nodes. "
                          "Fit assistance (in-house engine only): auto_lambda, "
                          "target_chi2, chi2_tolerance, max_lambda_trials.")},
                {"name": "auto_pick", "args": {},
                 "desc": ("Auto-pick first arrivals on the current record with the seismic "
                          "workflow's picker: threshold picks on the AGC traces, picks off the "
                          "shot's first-arrival curve picked again along it, and what still "
                          "breaks the curve left out.")},
                {"name": "auto_pick_all", "args": {},
                 "desc": ("Auto-pick every shot record at once with the same picker, leave out "
                          "shots whose times disagree with their reciprocals, and pick again "
                          "picks that disagree with the neighbouring shots at the same geophone. "
                          "Set each record's shot x first (set_geometry); manual picks are kept. "
                          "Follow with review_picks.")},
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
                {"name": "set_reflection",
                 "args": {"skip_shots": "list[int] or '1, 5, 12'", "leave_out_clipped": "bool",
                          "leave_out_early_signal": "bool", "align_timing": "bool",
                          "remove_air_wave": "bool", "band_hz": "[low, high]",
                          "velocity_source": "values | refraction_model",
                          "velocity_m_s": "float", "deeper_from_ms": "float or null",
                          "deeper_velocity_m_s": "float", "offsets_m": "[near, far]",
                          "trace_spacing_m": "float", "down_to_ms": "float",
                          "stretch_percent": "float", "min_traces": "int",
                          "ground_roll_m_s": "float"},
                 "desc": ("Set the Reflection tab's settings (only the keys given change); returns "
                          "them all. skip_shots leaves whole shot records out (a bad trigger, "
                          "wind). velocity_source 'refraction_model' stacks with the SRT model "
                          "from run_srt, averaged along the line over the ray-covered cells; "
                          "'values' uses velocity_m_s, and below deeper_from_ms (two-way time) "
                          "deeper_velocity_m_s (null turns the deeper layer off).")},
                {"name": "stack_reflection", "args": {},
                 "desc": ("Stack every loaded shot into a CMP reflection section with the "
                          "Reflection tab's settings and the Gather tab's geometry, and wait for "
                          "it (seconds). Returns the traces used, the timing correction and the "
                          "flat-event check: an event is a reflection CANDIDATE only when "
                          "flat_across_offsets is true, and even then it must be compared with a "
                          "well (Add to Map, then the Project Map's compare_well); with none "
                          "flat, say no reflection was confirmed.")},
                {"name": "show_reflection",
                 "args": {"view": "section | velocity_check | flat_event_check",
                          "vertical": "time | depth"},
                 "desc": ("Open the Reflection tab on one view, e.g. before capture_view "
                          "'reflection'.")},
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
            "auto_pick_all": lambda: self._agent_auto_pick_all(),
            "pick_next_shot": lambda: self._agent_pick_next_shot(),
            "review_picks": lambda: self._agent_review_picks(),
            "set_pick": lambda: self._agent_set_pick(args),
            "delete_pick": lambda: self._agent_delete_pick(args),
            "list_picks": lambda: self._agent_list_picks(),
            "clear_picks": lambda: self._agent_clear_picks(),
            "load_traveltime": lambda: self._agent_load_traveltime(args.get("path")),
            "clear_traveltime": lambda: self._agent_clear_traveltime(),
            "run_srt": lambda: self._agent_run_srt(),
            "set_reflection": lambda: self._agent_set_reflection(args),
            "stack_reflection": lambda: self._agent_stack_reflection(),
            "show_reflection": lambda: self._agent_show_reflection(args),
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

    def show_run_inputs(self, inputs):
        """Open the automatic run's data here, so this panel is not blank.

        Part of the studio's contract: a module the run brings to the front has
        to show the user a result, not an empty tool with a banner over it.
        Display only - the workflow does its own work in its own process - and
        anything already loaded is left alone.
        """
        status = self._agent_status()
        if status.get("data_loaded") or status.get("traveltime_loaded"):
            return ""
        geometry = inputs.get("geophone_file") or inputs.get("topography_file")
        if geometry and Path(str(geometry)).is_file():
            self._agent_load_geometry(str(geometry))
        travel = inputs.get("seismic_file")
        if travel and Path(str(travel)).is_file():
            if self._agent_load_traveltime(str(travel)).get("status") != "failed":
                return Path(str(travel)).name
        raw = inputs.get("raw_seismic_file")
        if raw and Path(str(raw)).is_file():
            if self._agent_load(str(raw)).get("status") != "failed":
                return Path(str(raw)).name
        return ""

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
            "reflection": (self._reflection_report() if self._refl_result is not None
                           else {"stacked": False}),
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
            # The picks follow now, not after the pause a typed value waits for:
            # the assistant's next call may run the inversion.
            self._geometry_debounced.flush()
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
            "pick_threshold": lambda v: self._threshold.setValue(float(v)),
            "pick_max_time_ms": lambda v: self._pick_window.setValue(float(v)),
            "screen_picks": lambda v: self._screen_picks.setChecked(bool(v)),
            "display": lambda v: self._viewer.set_display_style(str(v)),
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
            if key == "sta_lta_ratio":
                ignored[key] = ("The picker no longer uses an STA/LTA ratio; set pick_threshold "
                                "(a fraction of each trace's peak, default 0.2) instead.")
                continue
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
        return {"status": "ok", "picks": len(self._picks),
                "repicked": sum(1 for s in self._pick_src.values() if s == "repick")}

    def _agent_auto_pick_all(self) -> Dict[str, Any]:
        if self._raw is None:
            return {"status": "failed", "error": "Load data first."}
        if not self._dt:
            return {"status": "failed", "error": "The file gives no sample interval to pick in."}
        if len(self._agent_records()) < 2:  # a single record, picked as auto_pick does
            return self._agent_auto_pick()
        done = self._auto_pick_all()
        if done is None:
            return {"status": "failed", "error": "Auto-pick failed; see the log."}
        return {"status": "ok", **done,
                "next": "Call review_picks so the user can check the picks before run_srt."}

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
        auto = sum(1 for s in self._pick_src.values() if s in ("auto", "repick"))
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

    # -- reflection: the assistant's actions ------------------------------------------
    def _reflection_settings(self) -> Dict[str, Any]:
        return {
            "skip_shots": self._refl_skip.text().strip(),
            "leave_out_clipped": self._refl_drop_clipped.isChecked(),
            "leave_out_early_signal": self._refl_drop_early.isChecked(),
            "align_timing": self._refl_align.isChecked(),
            "remove_air_wave": self._refl_remove_air.isChecked(),
            "band_hz": [self._refl_low.value(), self._refl_high.value()],
            "velocity_source": ("refraction_model" if self._refl_vsource.currentData() == "model"
                                else "values"),
            "velocity_m_s": self._refl_v1.value(),
            "deeper_from_ms": self._refl_t2.value() if self._refl_layer.isChecked() else None,
            "deeper_velocity_m_s": self._refl_v2.value(),
            "offsets_m": [self._refl_off_lo.value(), self._refl_off_hi.value()],
            "trace_spacing_m": self._refl_bin.value(),
            "down_to_ms": self._refl_tmax.value(),
            "stretch_percent": self._refl_stretch.value(),
            "min_traces": int(self._refl_min_fold.value()),
            "ground_roll_m_s": self._refl_ground_roll.value(),
        }

    def _agent_set_reflection(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Set the Reflection tab's controls by name, the way a user would."""
        args = dict(args or {})
        args.pop("params", None)
        try:
            if "skip_shots" in args:
                shots = args.pop("skip_shots")
                text = (", ".join(str(int(v)) for v in shots)
                        if isinstance(shots, (list, tuple)) else str(shots or ""))
                try:
                    parse_shot_list(text)
                except ValueError:
                    raise ValueError("skip_shots takes record numbers, e.g. [1, 5, 12] or "
                                     "'1, 5, 12' or '3-6'.") from None
                self._refl_skip.setText(text)
            for key, box in (("leave_out_clipped", self._refl_drop_clipped),
                             ("leave_out_early_signal", self._refl_drop_early),
                             ("align_timing", self._refl_align),
                             ("remove_air_wave", self._refl_remove_air)):
                if key in args:
                    box.setChecked(bool(args.pop(key)))
            for key, (low, high) in (("band_hz", (self._refl_low, self._refl_high)),
                                     ("offsets_m", (self._refl_off_lo, self._refl_off_hi))):
                if key in args:
                    a, b = (float(v) for v in args.pop(key))
                    low.setValue(a)
                    high.setValue(b)
            if "velocity_source" in args:
                source = str(args.pop("velocity_source")).strip().lower()
                if source not in ("values", "refraction_model", "model"):
                    raise ValueError("velocity_source is 'values' or 'refraction_model'.")
                self._refl_vsource.setCurrentIndex(self._refl_vsource.findData(
                    "values" if source == "values" else "model"))
            if "deeper_from_ms" in args:
                value = args.pop("deeper_from_ms")
                self._refl_layer.setChecked(value is not None)
                if value is not None:
                    self._refl_t2.setValue(float(value))
            for key, box in (("velocity_m_s", self._refl_v1),
                             ("deeper_velocity_m_s", self._refl_v2),
                             ("trace_spacing_m", self._refl_bin),
                             ("down_to_ms", self._refl_tmax),
                             ("stretch_percent", self._refl_stretch),
                             ("min_traces", self._refl_min_fold),
                             ("ground_roll_m_s", self._refl_ground_roll)):
                if key in args:
                    value = args.pop(key)
                    box.setValue(int(value) if box is self._refl_min_fold else float(value))
        except (TypeError, ValueError) as exc:
            return {"status": "failed", "error": str(exc)}
        out: Dict[str, Any] = {"status": "ok", "reflection_settings": self._reflection_settings()}
        if args:
            out["ignored"] = sorted(args)
        return out

    def _agent_stack_reflection(self) -> Dict[str, Any]:
        """Stack, and wait for it: the result is what the assistant needs to answer."""
        self._center_tabs.setCurrentWidget(self._refl_view)
        error = self._run_reflection()
        if error:
            return {"status": "failed", "error": error}
        worker = self._refl_worker
        if worker is not None:
            # The stack runs on its worker while a local loop keeps the window
            # painting; connected before the check, so a finish cannot slip by.
            loop = QEventLoop()
            worker.finished.connect(loop.quit)
            if not worker.isFinished():
                loop.exec()
            QCoreApplication.processEvents()   # a stack that ended first still delivers
        if self._refl_error:
            return {"status": "failed", "error": self._refl_error}
        if self._refl_result is None:
            return {"status": "failed", "error": "The stack produced no result."}
        return {"status": "ok", **self._reflection_report()}

    def _agent_show_reflection(self, args: Dict[str, Any]) -> Dict[str, Any]:
        from PyHydroGeophysX.qt_apps.widgets import reflection_view as rv

        views = {"section": rv.SECTION, "velocity_check": rv.VELOCITY,
                 "flat_event_check": rv.FLATNESS}
        view = str(args.get("view") or "section").strip().lower()
        if view not in views:
            return {"status": "failed", "error": f"view is one of {', '.join(views)}."}
        vertical = str(args.get("vertical") or "").strip().lower()
        if vertical and vertical not in ("time", "depth"):
            return {"status": "failed", "error": "vertical is 'time' or 'depth'."}
        self._center_tabs.setCurrentWidget(self._refl_view)
        self._refl_view.set_view(views[view])
        if vertical:
            self._refl_view.set_vertical(vertical)
        return {"status": "ok", "view": view, "stacked": self._refl_result is not None}

    def _reflection_report(self) -> Dict[str, Any]:
        """The last stack in numbers: what was used, and what the flat-event check found."""
        result = self._refl_result
        prepared = self._refl_prepared[1] if self._refl_prepared is not None else None
        stack = result.stack
        sg = result.supergather
        events = []
        for event in (sg.events if sg is not None else []):
            depth = float(result.velocity.depth(np.array([event["t0"]]))[0])
            events.append({"t0_ms": round(event["t0"] * 1e3, 2), "depth_m": round(depth, 2),
                           "flat_across_offsets": bool(event["passes"]),
                           "why": event["reason"]})
        report: Dict[str, Any] = {
            "stacked": True,
            "verdict": self._reflection_verdict(result),
            "traces_stacked": int(result.traces_used),
            "positions": int(stack.cmp_x.size),
            "position_range_m": [round(float(stack.cmp_x[0]), 2),
                                 round(float(stack.cmp_x[-1]), 2)],
            "max_traces_per_position": int(stack.max_fold),
            "velocity": result.velocity.describe(),
            "events": events,
            "linear_energy_dominates": bool(
                result.peaks_linear and result.peaks_hyperbolic
                and result.peaks_linear[0][2] > result.peaks_hyperbolic[0][2]),
        }
        if getattr(self, "_refl_ray_limit", None) is not None:
            # Depths below this come from a velocity carried down, not measured.
            report["velocity_measured_to_m"] = round(float(self._refl_ray_limit), 2)
        if prepared is not None:
            report["traces"] = prepared.qc.counts()
            st = prepared.statics
            if st is not None:
                shifts = [abs(v) for r, v in st.statics.items() if r not in st.median_records]
                report["timing_correction_max_ms"] = round(max(shifts) * 1e3, 2)
                report["air_wave_m_s"] = round(float(st.velocity), 1)
        return report

    def agent_view_context(self, view: str) -> Optional[Dict[str, Any]]:
        """Ship the pick table with a captured gather.

        Trace indices are the one thing a model reads unreliably off this panel:
        two dozen traces share a narrow axis and each marker sits far above its
        tick label. Sending the picks as numbers leaves the picture to do what it
        is actually good for, judging whether a pick sits on the first arrival.
        """
        if view == "reflection":
            if self._refl_result is None:
                return None
            report = self._reflection_report()
            return {
                "showing": self._refl_view.view(),
                "verdict": report["verdict"],
                "events": report["events"],
                "note": ("Event times and the flat-event outcome are exact. Only an event with "
                         "flat_across_offsets true is a reflection candidate; bands that are not "
                         "flat are left over from the mutes or ground roll."),
            }
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
            "note": ("Trace indices come from the studio and are exact. Use the image to "
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


# Names a 0.3.0 script could import from this page, which it no longer defines.
from PyHydroGeophysX._internal.deprecations import legacy_names as _legacy_names  # noqa: E402

__getattr__ = _legacy_names(__name__, {
    "metrics_from_manager": "PyHydroGeophysX.inversion.metrics.metrics_from_manager",
    "pick_first_breaks": "PyHydroGeophysX.data_processing.seismic.pick_first_breaks",
})
