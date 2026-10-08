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

import json
import os
from pathlib import Path
import re
import shutil
from typing import Any, Callable, Dict, List, Optional, Tuple
import uuid

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QEventLoop, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QAbstractSpinBox,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from PyHydroGeophysX.data_processing import run_inputs
from PyHydroGeophysX.data_processing import ert_io as ert_load
from PyHydroGeophysX.data_processing import survey_timing
from PyHydroGeophysX.data_processing import table_io
from PyHydroGeophysX.qt_apps import io_utils, theme
from PyHydroGeophysX.qt_apps.modules.base import BaseModule, LogFn
from PyHydroGeophysX.qt_apps.qt_utils import (
    BusyStateController,
    ContentWidthScrollArea,
    ReproduceBar,
    merged_row,
    select_directory,
)
from PyHydroGeophysX.qt_apps.widgets import colormaps as cmaps
from PyHydroGeophysX.qt_apps.widgets import length_units
from PyHydroGeophysX.qt_apps.widgets import temperature_panel
from PyHydroGeophysX.qt_apps.widgets.mesh_preview import MeshPreviewView
from PyHydroGeophysX.qt_apps.widgets.mesh_view import MeshResultView
from PyHydroGeophysX.qt_apps.widgets.quality_view import InversionQualityView
from PyHydroGeophysX.qt_apps.widgets.reciprocal_view import ReciprocalErrorView
from PyHydroGeophysX.qt_apps.widgets.run_controls import progress_with_stop
from PyHydroGeophysX.qt_apps.workers import (
    ProcessProbeWorker,
    ProcessWorkflowWorker,
    TaskWorker,
)
from PyHydroGeophysX.data_processing.ert_io import save_edited_ert_container
from PyHydroGeophysX.visualization.axis_units import set_length_axis
from PyHydroGeophysX.workflows import (
    ArtifactRef,
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

#: The highest mesh quality offered, ert_mesh.MAX_MESH_QUALITY: Triangle
#: never finishes much above it. Repeated for the reason the bounds above no
#: longer are - ert_mesh pulls in pygimli.
_MAX_MESH_QUALITY = 34.0

_TC_WAITING = ("Run an inversion - one survey or a time-lapse series; Apply then "
               "corrects the model shown here, without inverting again.")

#: The Reciprocal errors tab before data are loaded, and how to get pairs.
_RECIP_NO_DATA = ("Load ERT data to see how far each reading and its reciprocal disagree. "
                  "In time-lapse mode the tab shows every survey of the list.")
_RECIP_HOW = ("To get them, measure reciprocals in the field - each reading repeated with "
              "its current and potential electrodes exchanged; most instruments can add "
              "them to a protocol. For a survey written as two files, add the reciprocal "
              "file to the list as well and keep “Pair reciprocal files with their forward "
              "file” ticked.")

#: Where the Data QC checks start. None acts until Apply filter is pressed, and
#: those under "More checks" only while it is unfolded, so none changes a run
#: nobody filtered. Each was checked against the shipped BERT, DAS-1 and E4D
#: surveys and a Subsurface Insights line: it drops readings that are plainly
#: bad and leaves the bulk of a survey alone. At zero they all did nothing.
_QC_DEFAULTS: Dict[str, Any] = {
    "max_error": 20.0,           # %; a file with no error column of its own is all 5 %
    "drop_nonpositive": True,    # a polarity or geometry error, not a measurement
    "min_voltage": 1e-5,         # V; 10 uV, the noise floor the DAS-1 reader filters at
    "min_current": 1e-4,         # A; below 0.1 mA the injection failed
    "k_over_median": 20.0,       # |k| cap, as a multiple of the loaded file's median
    "max_contact_r": 3e4,        # ohm; 30 kOhm, as the DAS-1 reader filters
    "max_stack": 50.0,           # %; the potential scattered by half of itself
    "max_reciprocal": 5.0,       # %; the usual reciprocal-error limit
}


def _value_name(correction: Optional[Dict[str, Any]]) -> str:
    """Column name for exported resistivity: it says the temperature when corrected."""
    if not correction:
        return "resistivity"
    reference = float(correction.get("reference_temperature_C", 25.0))
    return f"resistivity_at_{reference:g}C".replace(".", "p")

_ELEC_FILTER = "Electrodes (*.csv *.txt *.dat);;All files (*)"
_DATA_FILTER = "ERT data (*.dat *.ohm *.txt *.csv *.Data *.bin *.stg *.amp *.udf);;All files (*)"

#: How the file list finds each survey's time, on hover over its summary line.
#: survey_timing reads every form listed; ert_input_format.md says the same.
_TIME_FORMATS_TIP = (
    "How survey times are read\n\n"
    "1. From the file name. Put the year first:\n"
    "      site_2026-01-12_05-50-38.dat   (recommended)\n"
    "      site_2026_01_12_05_50_38.dat\n"
    "      site_20260112_055038.dat\n"
    "      site_2026-01-12T05-50-38.dat\n"
    "2. Otherwise from a date line at the top of the file, e.g.\n"
    "      # date: 2026-01-12 05:50:38\n"
    "   (for BERT / pyGIMLi files: the very first line, before the electrode count).\n"
    "3. Otherwise, if “Use file times” is ticked, from each file's modified time.\n\n"
    "Avoid month-first or day-first names such as 01-12-2026: they can be read two\n"
    "ways. Hover over a file to see where its time came from. Data format has more.")

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
    ("Subsurface Insights (.csv)", "Subsurface Insights"),
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
        # The electrodes as chosen from that file, as plain x y z columns: what
        # the instrument readers are handed on the next load.
        self._electrode_table: Optional[Path] = None
        self._data_path: Optional[Path] = None
        # The file the single-inversion model on screen was inverted from. A
        # preview loaded since replaces _data_path while that model stays up.
        self._inv_source: Optional[Path] = None
        # The thresholds of the last "Apply filter", until Reset: what a
        # time-lapse run applies to every one of its surveys.
        self._qc_applied: Optional[Dict[str, Any]] = None
        # What that filter did to the data loaded now: the run's QC report.
        self._qc_report: Optional[Dict[str, Any]] = None
        self._pseudo: List[Tuple[float, float, float]] = []
        self._n_meas = 0
        self._ert_data = None        # pygimli DataContainerERT for inversion (filtered)
        self._ert_data_full = None   # unfiltered original
        self._data_note = ""         # why the loaded file has no usable container
        self._qc_mask: Optional[List[bool]] = None
        self._inv_worker: Optional[Any] = None
        self._inv_busy: Optional[BusyStateController] = None
        self._adtlert_probe_worker: Optional[ProcessProbeWorker] = None
        self._adtlert_probe_serial = 0
        self._adtlert_runtime_ready: Optional[bool] = None
        self._adtlert_single_ready: Optional[bool] = None
        self._adtlert_timelapse_ready: Optional[bool] = None
        self._adtlert_checks: Dict[str, str] = {}
        self._e4d_probe_worker: Optional[ProcessProbeWorker] = None
        self._e4d_probe_serial = 0
        self._e4d_found: Optional[Dict[str, Any]] = None   # the last E4D check
        self._r2_probe_worker: Optional[ProcessProbeWorker] = None
        self._r2_probe_serial = 0
        self._r2_found: Optional[Dict[str, Any]] = None    # the last R2/R3t check
        self._ert_recipe_path: str = ""
        # The running runs' inversion_settings.txt, which their outcome is added
        # to; the time-lapse one with the parameters it was launched with.
        self._ert_settings_path: Optional[Path] = None
        self._tl_settings: Optional[Tuple[Path, Dict[str, Any]]] = None
        # Set by the Mode selector, which trades the pre-inversion checks against
        # turnaround. Quick skips them; Full validates and repairs k. A k that
        # disagrees with the geometry rescales the whole section while leaving
        # chi2 untouched, so Quick cannot warn you that it went wrong: the result
        # panel and the log only report that the check did not run. Scripts and
        # the agent can still set "check" or "off" directly through set_params.
        self._geom_policy = "off"
        # Single-inversion results. When the auto-λ search moves off the requested
        # λ, both runs are kept: _inv_choices holds one entry per selectable model
        # and _inv_mgr always points at the one on screen.
        self._inv_mgr = None
        self._inv_choices: List[Dict[str, Any]] = []
        self._agent_run_state = "idle"
        self._agent_run_error = ""
        # A temperature correction of the single-inversion model on screen: what
        # was applied, and the corrected model. The manager keeps the model as
        # inverted, so the correction can be changed or taken off.
        self._inv_correction: Optional[Dict[str, Any]] = None
        self._inv_corrected: Optional[np.ndarray] = None
        self._load_worker: Optional[TaskWorker] = None
        # Every file added to the list, reciprocal files included, in the order
        # the list keeps them. _tl_files holds the surveys made of them: while
        # "Pair reciprocal files" is ticked, a forward file stands for itself and
        # its reciprocal (_tl_partner), and a reciprocal file no forward file
        # matches is left out (_tl_left_out).
        self._tl_all: List[str] = []
        self._tl_partner: Dict[str, str] = {}
        self._tl_left_out: List[str] = []
        # The reciprocal file merged with the survey on screen, and its counts.
        self._data_partner: Optional[Path] = None
        self._pair_info: Optional[Dict[str, Any]] = None
        self._tl_files: List[str] = []
        self._tl_labels: List[str] = []
        self._tl_times: List[float] = []
        self._tl_timing: Optional[Any] = None   # acquisition times of the sequence
        self._tl_worker: Optional[Any] = None
        self._tl_busy: Optional[BusyStateController] = None
        self._tl_recipe_path = ""
        self._tl_out: Optional[str] = None
        self._tl_result: Optional[dict] = None
        self._tl_mesh = None              # in-memory time-lapse result for the
        self._tl_models = None            # interactive per-step viewer
        # The models as inverted, kept beside the ones on screen so a
        # temperature correction can be applied, changed and taken off again
        # without inverting; and what that correction was, or None.
        self._tl_models_raw = None
        self._tl_correction: Optional[Dict[str, Any]] = None
        self._tl_coverage = None
        self._tl_step_titles: List[str] = []
        # The pseudosection's colour map: viridis unless the user picks another,
        # remembered for the session like every other view's.
        self._pseudo_colormap = cmaps.ColormapChooser(
            cmaps.APPARENT_RESISTIVITY, "viridis",
            shared=cmaps.colormap_settings(self.state))
        self._pseudo_colormap.colormapChanged.connect(self._on_pseudo_colormap_changed)
        self._cmap = cmaps.to_pyqtgraph(self._pseudo_colormap.colormap())

        root = QHBoxLayout(self)
        self._tabs = QTabWidget()
        self._plot_widget = pg.PlotWidget()
        self._plot_widget.setBackground("w")
        self._plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self._plot = self._plot_widget.getPlotItem()
        self._electrode_axes: Optional[Tuple[str, str]] = None
        self._label_electrode_axes()
        # View > Length Units: both plots here relabel and retick; the data stay
        # in metres.
        length_units.notifier().changed.connect(self._on_length_unit_changed)
        self._scatter = pg.ScatterPlotItem(size=12, pen=pg.mkPen("#007aff", width=1), brush=pg.mkBrush(0, 122, 255, 170))
        self._sel_scatter = pg.ScatterPlotItem(size=18, pen=pg.mkPen("#ff8c00", width=2), brush=pg.mkBrush(255, 140, 0, 120))
        self._plot.addItem(self._scatter)
        self._plot.addItem(self._sel_scatter)
        self._plot.scene().sigMouseClicked.connect(self._on_click)

        # Render the apparent-resistivity section with Matplotlib rather than a
        # pyqtgraph ScatterPlotItem.  On some Windows Qt/pyqtgraph combinations
        # the scatter loses the ViewBox transform after a resize: axes remain in
        # data coordinates while markers are painted as scene pixels above/left
        # of the plot.  This section is intentionally static; reliable placement
        # is more important than per-point hover/pick interaction here.
        self._pseudo_widget = QWidget()
        pseudo_layout = QVBoxLayout(self._pseudo_widget)
        pseudo_layout.setContentsMargins(0, 0, 0, 0)
        pseudo_layout.setSpacing(4)
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        self._pseudo_figure = Figure(facecolor="white", constrained_layout=True)
        self._pseudo_canvas = FigureCanvasQTAgg(self._pseudo_figure)
        self._pseudo_ax = self._pseudo_figure.add_subplot(111)
        pseudo_layout.addWidget(self._pseudo_canvas, stretch=1)

        self._pseudo_legend = QWidget()
        # Reserve the legend's full text height; the canvas takes the remaining
        # space, including in short windows and with larger display fonts.
        self._pseudo_legend.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        legend_layout = QVBoxLayout(self._pseudo_legend)
        legend_layout.setContentsMargins(8, 0, 8, 6)
        legend_layout.setSpacing(1)
        # The colour map beside the colour scale it changes. The title takes the
        # stretch itself: the page lets long labels elide, and an eliding label
        # beside a separate stretch is given no width at all.
        legend_title = QHBoxLayout()
        legend_title.setContentsMargins(0, 0, 0, 0)
        legend_title.addWidget(QLabel("Apparent resistivity (Ω·m)"), 1)
        legend_title.addWidget(self._pseudo_colormap)
        legend_layout.addLayout(legend_title)
        self._pseudo_scale_bar = QFrame()
        self._pseudo_scale_bar.setFixedHeight(12)
        self._paint_pseudo_scale()
        legend_layout.addWidget(self._pseudo_scale_bar)
        scale_row = QHBoxLayout()
        scale_row.setContentsMargins(0, 0, 0, 0)
        scale_row.setSpacing(2)
        self._pseudo_scale_labels: List[QLabel] = []
        for index in range(5):
            label = QLabel("—")
            label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
            label.setAlignment(
                Qt.AlignLeft if index == 0 else Qt.AlignRight if index == 4 else Qt.AlignCenter
            )
            scale_row.addWidget(label, stretch=1)
            self._pseudo_scale_labels.append(label)
        legend_layout.addLayout(scale_row)
        pseudo_layout.addWidget(self._pseudo_legend)
        # Until data arrive the tab says what to do. Left as built it showed an
        # empty 0-1 axis over a colour bar of "—" placeholders, which read as a
        # bar whose numbers had been covered.
        self._show_pseudosection_message(
            "Add ERT data files on the right to see their pseudosection.")

        # The studio state's colormap choices, shared with Saved Results: a
        # section recoloured on either page is recoloured on both.
        self._model_view = MeshResultView(colormaps=cmaps.colormap_settings(self.state))
        # Tools that act on the model sit beside it and serve whichever result is
        # on screen - a single inversion or a time-lapse step - rather than living
        # in one run mode's settings, where the other mode's results cannot reach
        # them. Saved Results puts the same panel in the same place.
        self._model_view.add_side_panel(self._build_temperature_group())
        self._model_view._side.setSizes([1000, 360])
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
        self._tl_step_combo.setToolTip("Which time-lapse inversion result is displayed.")
        self._tl_step_combo.currentIndexChanged.connect(self._show_tl_step)
        step_bar.addWidget(self._tl_step_combo, stretch=1)
        self._tl_prev_btn = QPushButton("◀"); self._tl_prev_btn.setMaximumWidth(34)
        self._tl_prev_btn.setToolTip("Previous time step")
        self._tl_prev_btn.clicked.connect(lambda: self._step_tl(-1))
        self._tl_next_btn = QPushButton("▶"); self._tl_next_btn.setMaximumWidth(34)
        self._tl_next_btn.setToolTip("Next time step")
        self._tl_next_btn.clicked.connect(lambda: self._step_tl(1))
        step_bar.addWidget(self._tl_prev_btn); step_bar.addWidget(self._tl_next_btn)
        # Absolute resistivity is dominated by the static structure, which is the
        # same in every step; the change against the first survey is what the
        # repeat measurement was made to see.
        step_bar.addWidget(QLabel("Show:"))
        self._tl_view_mode = QComboBox()
        self._tl_view_mode.addItem("Resistivity", "model")
        self._tl_view_mode.addItem("% change from baseline", "change")
        self._tl_view_mode.setToolTip(
            "Resistivity shows the inverted model. “% change from baseline” shows "
            "100 × (ρ − ρ₀) / ρ₀ against the first time step, on a diverging scale "
            "centred on zero.")
        self._tl_view_mode.currentIndexChanged.connect(self._on_tl_view_mode_changed)
        step_bar.addWidget(self._tl_view_mode)
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
            "The auto-λ search re-inverted at a different λ. Selects between that "
            "model and the one at the λ set above.")
        self._lam_pick.currentIndexChanged.connect(self._show_lambda_choice)
        pick_bar.addWidget(self._lam_pick, stretch=1)
        self._lam_pick_row.setVisible(False)
        model_layout.addWidget(self._lam_pick_row)
        model_layout.addWidget(self._model_view, stretch=1)

        self._quality_view = InversionQualityView()

        # The mesh the next run will invert on, built the way the run builds it,
        # with the a-priori zones drawn on it. Rebuilt when the tab is opened
        # after the data or the mesh settings changed, a moment after the last
        # change while it is open, and on request.
        self._mesh_tab = MeshPreviewView()
        self._mesh_tab.rebuildRequested.connect(
            lambda: self._refresh_mesh_preview(force=True))
        # A zone edit rebuilds the mesh while it follows the zone outlines; the
        # two zone options also change what the Inversion panel says.
        self._mesh_tab.zonesChanged.connect(self._mesh_inputs_changed)
        self._mesh_tab.conformChanged.connect(self._mesh_inputs_changed)
        self._mesh_tab.decoupleChanged.connect(self._mesh_inputs_changed)
        self._mesh_preview_key: Optional[tuple] = None      # inputs of the mesh shown
        self._mesh_preview_pending: Optional[tuple] = None  # inputs being built
        self._mesh_worker: Optional[TaskWorker] = None
        self._mesh_timer = QTimer(self)
        self._mesh_timer.setSingleShot(True)
        self._mesh_timer.setInterval(400)
        self._mesh_timer.timeout.connect(self._refresh_mesh_preview)

        # Each reciprocal pair's error against its resistance, with the error
        # model fitted through them: the survey on screen, or every survey of a
        # time-lapse list. Its "Fit to" chooser is the page's one setting for
        # which pairs the model is fitted to - the data errors, the QC report,
        # the settings file and the run's figure all follow it.
        self._recip_view = ReciprocalErrorView()
        self._recip_view.fitToChanged.connect(self._on_error_fit_changed)
        self._recip_view.useRequested.connect(self._use_reciprocal_errors)
        # The loaded survey's pairing, made with it off the UI thread.
        self._loaded_pairing: Optional[Dict[str, Any]] = None
        # Each survey's QC report - its pairing among it - as read for the
        # series view or by the time-lapse QC: {survey key: {filter key:
        # report}} (_recip_survey_key, _recip_qc_key). A series of 420 surveys
        # takes minutes to read, so it is read once per file and filter.
        self._recip_cache: Dict[tuple, Dict[str, Dict[str, Any]]] = {}
        self._recip_worker: Optional[TaskWorker] = None
        self._recip_reading: Optional[tuple] = None
        self._recip_timer = QTimer(self)
        self._recip_timer.setSingleShot(True)
        self._recip_timer.setInterval(500)
        self._recip_timer.timeout.connect(self._refresh_recip_view)

        self._tabs.addTab(self._plot_widget, "Electrodes")
        self._tabs.addTab(self._pseudo_widget, "Pseudosection")
        self._tabs.addTab(self._recip_view, "Reciprocal errors")
        self._tabs.addTab(self._mesh_tab, "Mesh")
        self._tabs.addTab(model_tab, "Resistivity model")
        self._tabs.addTab(self._quality_view, "Inversion quality")
        self._tabs.currentChanged.connect(self._on_tab_changed)
        self._reproduce = ReproduceBar()
        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.addWidget(self._tabs, stretch=1)
        center_layout.addWidget(self._reproduce)
        root.addWidget(center, stretch=1)
        self._controls = self._build_controls()
        self._sync_recip_use()
        root.addWidget(self._controls)
        self._sync_tab_controls()

    # -- controls ------------------------------------------------------------
    def _build_controls(self) -> QWidget:
        # Vertical scrolling only, so the column must never be narrower than its
        # widest row: 450-500 px is the preferred width, and the column widens
        # past it when the content needs more - a fixed 500 px cut the file-list
        # buttons and hint off at the right edge once the font was wider than
        # the one it was sized with.
        scroll = ContentWidthScrollArea(minimum=450, maximum=500)
        panel = QWidget()
        scroll.setWidget(panel)
        layout = QVBoxLayout(panel)

        loader = QGroupBox("Load resistivity data")
        self._load_group = loader
        lform = QFormLayout(loader)
        self._instrument = QComboBox()
        for label, value in _INSTRUMENTS:
            self._instrument.addItem(label, value)
        # Don't let the longest item ("BERT / Unified (.ohm/.dat)") force the whole
        # control panel wide; elide in the closed box (full text in the dropdown).
        self._instrument.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self._instrument.setMinimumContentsLength(12)
        # A time-lapse run reads its first survey with this reader to build the
        # mesh, and so does the Mesh tab's preview of it; the Reciprocal errors
        # tab reads a series' surveys with it.
        self._instrument.currentIndexChanged.connect(self._mesh_inputs_changed)
        self._instrument.currentIndexChanged.connect(self._recip_inputs_changed)
        self._resipy_ready = QLabel()
        self._resipy_ready.setWordWrap(False)
        instrument_row = QHBoxLayout()
        instrument_row.setSpacing(8)
        instrument_row.addWidget(self._instrument, stretch=1)
        instrument_row.addWidget(self._resipy_ready)
        lform.addRow("Instrument / format", instrument_row)

        # Which reader handled the file is not visible in the result, and the two
        # do not always agree on how many measurements a file holds, so it is
        # stated rather than left to the log.
        self._reader_status = QLabel()
        theme.set_tone(self._reader_status, "hint")
        self._reader_status.setWordWrap(True)
        lform.addRow("", self._reader_status)
        self._show_reader_status()

        self._tl_list = QListWidget()
        self._tl_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._tl_list.setMaximumHeight(150)
        self._tl_list.setToolTip("Click a row to preview it. Each file is one time step; "
                                 "the list runs from earliest at the top to latest at the bottom. "
                                 "Use Sort by time to order the sequence.")
        self._tl_list.itemSelectionChanged.connect(self._on_tl_selection_changed)
        self._tl_list.itemClicked.connect(self._preview_tl_item)
        lform.addRow(self._tl_list)

        tl_btns = QHBoxLayout()
        add_btn = QPushButton("Add files…")
        add_btn.setToolTip("Add one or more ERT files, one per survey for time-lapse. "
                           "Choose the instrument / format first.")
        add_btn.setProperty("primary", True)
        add_btn.setIcon(theme.icon("fa5s.file-import", color="#ffffff"))
        add_btn.clicked.connect(self._add_tl_files)
        rm_btn = QPushButton("Remove"); rm_btn.setIcon(theme.icon("fa5s.minus"))
        rm_btn.clicked.connect(self._remove_tl_files)
        up_btn = QPushButton("↑"); up_btn.setToolTip("Move selected up"); up_btn.setMaximumWidth(34)
        up_btn.clicked.connect(lambda: self._move_tl_files(-1))
        down_btn = QPushButton("↓"); down_btn.setToolTip("Move selected down"); down_btn.setMaximumWidth(34)
        down_btn.clicked.connect(lambda: self._move_tl_files(1))
        sort_btn = QPushButton("Sort by time"); sort_btn.setIcon(theme.icon("fa5s.clock"))
        sort_btn.setToolTip(
            "Put the files in acquisition order, using the times read from their "
            "names or headers. A time-lapse inversion compares each survey with "
            "the one before it, so the order is part of the result.")
        sort_btn.clicked.connect(self._sort_tl_files_by_time)
        clr_btn = QPushButton("Clear"); clr_btn.setIcon(theme.icon("fa5s.trash"))
        clr_btn.clicked.connect(self._clear_tl_files)
        # Two rows rather than one: all six in a row were the widest thing in the
        # column. Adding and removing files on top, ordering them below.
        for b in (add_btn, rm_btn, clr_btn):
            tl_btns.addWidget(b)
        lform.addRow(tl_btns)
        order_btns = QHBoxLayout()
        for b in (up_btn, down_btn, sort_btn):
            order_btns.addWidget(b)
        order_btns.addStretch(1)
        lform.addRow(order_btns)

        self._tl_info = QLabel("No files added."); self._tl_info.setWordWrap(True)
        # The accepted time formats on hover, so the line itself stays short.
        self._tl_info.setToolTip(_TIME_FORMATS_TIP)
        lform.addRow(self._tl_info)

        # Shown only when names in the list look like forward/reciprocal halves,
        # and on then: the detection goes by name and time, so it can be wrong,
        # and unticking it puts every file back as a survey of its own.
        self._tl_pair_box = QCheckBox("Pair reciprocal files with their forward file")
        self._tl_pair_box.setChecked(True)
        self._tl_pair_box.setToolTip(
            "Some instruments write each survey as two files: the forward readings, "
            "and a few minutes later the reciprocal readings (“recip” in the name). "
            "Ticked, each reciprocal file is joined to the forward file measured just "
            "before it with the same name, and the two are one survey: one row in "
            "the list, at the forward file's time, one time step, and the readings "
            "of both in one data set, so each reading can be compared with its "
            "reciprocal.\n\n"
            "A forward file with no reciprocal stays a survey on its own. A "
            "reciprocal file with no forward file is left out, and the line above "
            "names it.\n\n"
            "Untick it if a file was joined to the wrong partner: every file is "
            "then a survey of its own again.")
        self._tl_pair_box.toggled.connect(self._on_tl_pairing_changed)
        self._tl_pair_box.setVisible(False)
        lform.addRow(self._tl_pair_box)

        load_e = QPushButton("Electrode file (optional)…")
        load_e.setIcon(theme.icon("fa5s.folder-open"))
        load_e.clicked.connect(self._load_electrodes)
        fmt_btn = QPushButton("Data format")
        fmt_btn.setIcon(theme.icon("fa5s.file-alt"))
        fmt_btn.setToolTip("The file formats this page reads, the electrode file, and "
                           "how to give each survey its time.")
        fmt_btn.clicked.connect(self._show_format_help)
        elec_row = QHBoxLayout()
        elec_row.addWidget(load_e)
        elec_row.addWidget(fmt_btn)
        lform.addRow(elec_row)
        layout.addWidget(loader)

        self._info = QLabel("No data loaded.")
        self._info.setWordWrap(True)
        layout.addWidget(self._info)

        qc = QGroupBox("Data QC / filter")
        self._qc_group = qc
        qform = QFormLayout(qc)
        # Check 0, before every other check, as in Craig Ulrich's processing.
        # Off by default: kept apart, the two readings of a pair let the fit see
        # how far they disagree, and an averaged pair hides that.
        self._qc_average = QCheckBox("Average each reading with its reciprocal")
        self._qc_average.setChecked(False)
        self._qc_average.setEnabled(False)
        self._qc_average.setToolTip("Load ERT data first.")
        self._qc_average.toggled.connect(self._sync_error_rows)
        qform.addRow(self._qc_average)
        self._rmin = QDoubleSpinBox(); self._rmin.setRange(0.0, 1e6); self._rmin.setValue(0.0); self._rmin.setSuffix(" Ω·m")
        self._rmax = QDoubleSpinBox(); self._rmax.setRange(1.0, 1e7); self._rmax.setValue(100000.0); self._rmax.setSuffix(" Ω·m")
        self._max_err = QDoubleSpinBox(); self._max_err.setRange(0.0, 100.0)
        self._max_err.setValue(_QC_DEFAULTS["max_error"]); self._max_err.setSuffix(" %")
        self._max_err.setToolTip(
            "Drop measurements with relative error above this (0 = off). 20 % drops "
            "the readings a file's own error column calls noisy; a file with none of "
            "its own carries the 5 % assumed for every reading, and loses nothing.")
        qform.addRow("Min ρa", self._rmin)
        qform.addRow("Max ρa", self._rmax)
        qform.addRow("Max error", self._max_err)

        # The rest of the criteria are folded away, because which of them a file
        # can even support varies by instrument: voltage and current come with
        # some formats and not others, and a reciprocal error needs the survey to
        # contain reciprocal pairs at all. Each row is enabled against the loaded
        # data in _refresh_qc_availability, so an unusable one says why rather
        # than filtering on a field that is not there.
        self._qc_more = QGroupBox("More checks")
        self._qc_more.setCheckable(True)
        self._qc_more.setChecked(False)
        self._qc_more.setToolTip(
            "Extra QC criteria. They stay folded because most surveys need none of "
            "them, and the ones a given file cannot support are disabled with the "
            "reason in their tooltip.")
        more_outer = QVBoxLayout(self._qc_more)
        more_outer.setContentsMargins(0, 0, 0, 0)
        self._qc_more_body = QWidget()
        mform = QFormLayout(self._qc_more_body)
        mform.setContentsMargins(0, 6, 0, 0)

        self._qc_drop_neg = QCheckBox("Drop ρa ≤ 0")
        self._qc_drop_neg.setChecked(_QC_DEFAULTS["drop_nonpositive"])
        self._qc_drop_neg.setToolTip(
            "A non-positive apparent resistivity is a polarity or geometry error "
            "rather than a measurement, and the inversion cannot use it. On with "
            "More checks. Min ρa above already removes negatives once it is set "
            "above zero.")
        mform.addRow(self._qc_drop_neg)

        # Six decimals, because files that report volts and amperes need floors
        # of a tenth of a millivolt or milliampere; three decimals silently turned
        # 0.0001 into 0, which switches the check off.
        self._qc_min_v = QDoubleSpinBox(); self._qc_min_v.setRange(0.0, 1e6)
        self._qc_min_v.setDecimals(6); self._qc_min_v.setValue(_QC_DEFAULTS["min_voltage"])
        mform.addRow("Min |V|", self._qc_min_v)

        self._qc_min_i = QDoubleSpinBox(); self._qc_min_i.setRange(0.0, 1e6)
        self._qc_min_i.setDecimals(6); self._qc_min_i.setValue(_QC_DEFAULTS["min_current"])
        mform.addRow("Min |I|", self._qc_min_i)

        # |k| scales with the electrode spacing - the shipped E4D survey runs to
        # 54,000 m, a Subsurface Insights line stops at 244 - so no one number
        # suits every survey. Until it is set by hand the cap follows each file
        # loaded, at a multiple of its median (_refresh_qc_availability).
        self._qc_max_k_follows_file = True
        self._qc_max_k = QDoubleSpinBox(); self._qc_max_k.setRange(0.0, 1e9)
        self._qc_max_k.setDecimals(0); self._qc_max_k.setValue(0.0)
        self._qc_max_k.setToolTip(
            "Drop configurations whose geometric factor exceeds this (0 = off). A large "
            "|k| multiplies the measured resistance, and its noise with it, which is how "
            "a clean-looking rhoa outlier is produced by geometry rather than by ground. "
            "Set from the file once one is loaded.")
        self._qc_max_k.valueChanged.connect(self._max_k_set_by_hand)
        mform.addRow("Max |k|", self._qc_max_k)

        self._qc_max_rc = QDoubleSpinBox(); self._qc_max_rc.setRange(0.0, 1e9)
        self._qc_max_rc.setDecimals(0); self._qc_max_rc.setValue(_QC_DEFAULTS["max_contact_r"])
        self._qc_max_rc.setSuffix(" Ω")
        mform.addRow("Max contact R", self._qc_max_rc)

        # Percent, like the reciprocal error below it, though the spread can run
        # to thousands of percent on a failing electrode.
        self._qc_max_stack = QDoubleSpinBox(); self._qc_max_stack.setRange(0.0, 1e6)
        self._qc_max_stack.setDecimals(1); self._qc_max_stack.setValue(_QC_DEFAULTS["max_stack"])
        self._qc_max_stack.setSuffix(" %")
        self._qc_max_stack.setToolTip(
            "Drop readings whose potential scattered by more than this across its own "
            "samples, as a percentage of the potential (0 = off). Subsurface Insights "
            "records it. A few percent is normal; hundreds of percent is an electrode "
            "or a contact failing.")
        mform.addRow("Max stacking spread", self._qc_max_stack)

        self._qc_max_recip = QDoubleSpinBox(); self._qc_max_recip.setRange(0.0, 100.0)
        self._qc_max_recip.setDecimals(2); self._qc_max_recip.setValue(_QC_DEFAULTS["max_reciprocal"])
        self._qc_max_recip.setSuffix(" %")
        mform.addRow("Max recip. error", self._qc_max_recip)

        self._qc_support_note = QLabel("Load data to see which checks are available.")
        self._qc_support_note.setWordWrap(True)
        theme.set_tone(self._qc_support_note, "hint")
        mform.addRow(self._qc_support_note)

        more_outer.addWidget(self._qc_more_body)
        self._qc_more_body.setVisible(False)
        self._qc_more.toggled.connect(self._qc_more_body.setVisible)
        qform.addRow(self._qc_more)

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
        self._inversion_group = inv
        iform = QFormLayout(inv)
        self._inv_form = iform

        # Mode comes first because it overrides several controls below it. The
        # split is by cost: the geometric-factor check is one extra forward run,
        # and the auto-λ search is a full inversion per trial. On a 3647-point
        # field survey they were 13 s and 189 s of a 247 s run.
        self._inv_mode = QComboBox()
        for label, value in (("Quick (no pre-checks)", "quick"),
                             ("Full (validate k, search λ)", "full")):
            self._inv_mode.addItem(label, value)
        self._inv_mode.setToolTip(
            "Quick runs the inversion and nothing else, which is the shorter run "
            "while λ and the mesh are still being settled.\n\n"
            "Full adds the two stages Quick drops. It validates the geometric factors "
            "against the mesh, repairing them when they disagree, and it searches λ "
            "for the target χ².\n\n"
            "The k check is the one with the most evidence behind it: a wrong k scales "
            "the whole section by a constant and χ² never notices, so a Quick result "
            "that looks perfect can still be uniformly wrong. Re-run in Full whenever "
            "the resistivities themselves look off, not only when the fit looks bad.")
        self._inv_mode.currentIndexChanged.connect(self._on_inv_mode_changed)
        iform.addRow("Mode", self._inv_mode)

        self._engine = QComboBox()
        for label, value in (("In-house Gauss-Newton", "pyhydro"),
                             ("PyGIMLi ERTManager", "pygimli"),
                             ("ADTLERT 2.5D (CUDA)", "adtlert"),
                             ("E4D 3D (PNNL, external)", "e4d"),
                             ("R2 2D (Binley, external)", "r2"),
                             ("R3t 3D (Binley, external)", "r3t")):
            self._engine.addItem(label, value)
        self._engine.setToolTip(
            "Solver. The in-house Gauss-Newton inversion exposes its own stopping rule "
            "and line search, so the fit assistance below drives it directly. The "
            "PyGIMLi manager runs its own loop and is available as a cross-check.\n\n"
            "ADTLERT uses a CUDA-accelerated cuDSS forward solve and GPU CGLS. "
            "CUDA 12 plus cuDSS are required. Windows is supported; Linux is "
            "recommended for the best performance. The controls below remain "
            "available when this engine is selected.\n\n"
            "E4D is PNNL's parallel 3D code, run as an external program under MPI; "
            "it is not installed with PyHydroGeophysX. It runs on Linux, on Windows "
            "only inside WSL 2, and on macOS when built from source. A profile is "
            "inverted in 3D on a mesh built around the line and shown as the "
            "section along it. A time-lapse series runs as E4D's own time-lapse "
            "inversion: the first survey is the baseline, and each later one starts "
            "from the solution before it, with the change from it smoothed.\n\n"
            "R2 (2D profiles) and R3t (3D surveys, or a profile on a 3D mesh around "
            "the line) are Andrew Binley's codes, the ones ResIPy runs; they are found "
            "in an installed ResIPy unless named below. They are Windows programs: "
            "native on Windows, through Wine on Linux and macOS, free for "
            "non-commercial use. Both choose their own smoothing weight at every "
            "iteration, so λ and auto-λ do not apply; they hold fixed zones fixed. A "
            "time-lapse series is their difference inversion against the first survey.")
        self._engine.currentIndexChanged.connect(self._on_engine_changed)
        self._engine.currentIndexChanged.connect(self._sync_mesh_engine)
        iform.addRow("Engine", self._engine)
        self._adtlert_status = QLabel()
        self._adtlert_status.setWordWrap(True)
        self._adtlert_status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._adtlert_status.setVisible(False)
        iform.addRow("", self._adtlert_status)
        self._build_e4d_rows(iform)
        self._build_r2_rows(iform)

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

        # How long an inversion may run, and when it counts as done. Each mode and
        # engine shows only the limits it obeys (_sync_iteration_rows): the three
        # numbers on one line read as one setting, and two of them never applied
        # to a time-lapse run.
        self._iter_ceiling = QSpinBox(); self._iter_ceiling.setRange(5, 400)
        self._iter_ceiling.setValue(60)
        self._iter_ceiling.setToolTip(
            "The most iterations the inversion may take (for each λ when Auto-λ is "
            "on). It normally stops sooner, once χ² stops improving (next row). "
            "Reaching this limit means the fit could still improve, and the log "
            "says so.")
        iform.addRow("Max iterations", self._iter_ceiling)
        # A single run checks its progress after this many iterations and goes on
        # from its own model while still improving, so λ is never judged on an
        # unfinished descent. That checkpoint is an internal detail there; a
        # time-lapse run has no continuation, and this is its whole limit.
        self._iter = QSpinBox(); self._iter.setRange(2, 60); self._iter.setValue(15)
        self._iter.setToolTip(
            "The most iterations of the time-lapse inversion; with Windowed on, of "
            "each window.")
        iform.addRow("Max iterations", self._iter)

        self._plateau = QDoubleSpinBox()
        self._plateau.setRange(0.01, 10.0); self._plateau.setDecimals(2)
        self._plateau.setSingleStep(0.1); self._plateau.setValue(0.5)
        self._plateau.setPrefix("< ")
        self._plateau.setSuffix(" % per iteration")
        self._plateau.setToolTip(
            "The inversion is done when one more iteration lowers χ² by less than "
            "this. Larger stops sooner; smaller makes sure the fit has really run "
            "out of room.")
        iform.addRow("Stop when χ² improves", self._plateau)

        # The mesh settings live on the Mesh tab, next to the mesh they build;
        # this row says what is set and leads there.
        self._mesh_tab.set_settings_panel(self._build_mesh_settings())
        self._mesh_link = QLabel()
        self._mesh_link.setWordWrap(True)
        self._mesh_link.setTextFormat(Qt.RichText)
        theme.set_tone(self._mesh_link, "hint")
        self._mesh_link.setToolTip(
            "The mesh, its sizing and what the a-priori zones do to it are set on "
            "the Mesh tab, which draws the mesh the run will invert on.")
        self._mesh_link.linkActivated.connect(
            lambda _link: self._tabs.setCurrentWidget(self._mesh_tab))
        iform.addRow("Mesh", self._mesh_link)
        self._update_mesh_link()
        layout.addWidget(inv)

        # -- data errors -----------------------------------------------------
        errs = QGroupBox("Data errors")
        self._errors_group = errs
        eform = QFormLayout(errs)
        self._err_source = QComboBox()
        for label, value in (("File err column", "file"),
                             ("Estimate from the values below", "estimate"),
                             ("Larger of the two", "max"),
                             ("Stacking spread with the estimate", "stack"),
                             ("From the reciprocal error model", "reciprocal")):
            self._err_source.addItem(label, value)
        self._err_source.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self._err_source.setMinimumContentsLength(16)
        self._err_source.setToolTip(
            "Where the per-measurement error comes from. Most instruments write an err "
            "column; replacing it with an assumed percentage makes χ² report on an error "
            "model the data never had. The stacking spread is offered for data that "
            "record it, the reciprocal error model for data measured both ways.")
        self._err_source.currentIndexChanged.connect(self._sync_error_rows)
        # The Reciprocal errors tab says whether the next run uses its model.
        self._err_source.currentIndexChanged.connect(self._sync_recip_use)
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

        # The smallest error a reciprocal can give a reading. 1 %: what the
        # in-house and ADTLERT time-lapse engines take as their own minimum, so
        # the errors recorded are the errors used; and a pair that agrees to a
        # tenth of a percent still shares every error that cannot differ
        # between its two directions.
        self._recip_floor = QDoubleSpinBox()
        self._recip_floor.setRange(0.1, 20.0); self._recip_floor.setDecimals(1)
        self._recip_floor.setSingleStep(0.5); self._recip_floor.setValue(1.0)
        self._recip_floor.setSuffix(" %")
        self._recip_floor.setToolTip(
            "No reading is given a smaller error than this when its error comes from "
            "reciprocals: the error model, or an averaged pair's own difference. Two "
            "directions that agree closely still share the errors that cannot differ "
            "between them - the electrode positions, the 2-D assumption - so a "
            "near-perfect pair is not an error-free reading. 1 % is also the smallest "
            "error the in-house and ADTLERT time-lapse inversions accept. Averaged pairs "
            "take it when Apply filter averages them; the model takes it when the "
            "inversion runs.")
        eform.addRow("Minimum", self._recip_floor)
        self._refresh_error_sources(None)
        # Data errors are settled before the inversion is, so the group sits
        # between the QC and the inversion settings: load, QC, errors, invert.
        layout.insertWidget(layout.indexOf(inv), errs)

        # -- fit assistance --------------------------------------------------
        assist = QGroupBox("Fit assistance")
        self._fit_group = assist
        aform = QFormLayout(assist)

        # Kept short enough to fit the panel; the tooltip carries the detail.
        self._auto_lam = QCheckBox("Auto-λ: re-invert to reach target χ²")
        # Off in Quick, which is the default mode; Full turns it back on.
        self._auto_lam.setChecked(False)
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
            "error. A larger target accepts a looser fit, matching data noisier than "
            "the error estimate describes.")
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
            "first one at the λ set above. Reached only when the target stays out of "
            "range.")
        aform.addRow("Max λ trials", self._lam_trials)

        self._reject = QCheckBox("Reject outliers: drop data the model cannot explain")
        self._reject.setChecked(False)
        self._reject.setToolTip(
            "After the inversion converges, the measurements whose residual exceeds "
            "the threshold below are dropped and the inversion runs again. This "
            "addresses a high χ² caused by individual bad readings rather than by the "
            "model.\n\n"
            "It also shrinks the dataset; the floor below bounds how much can be "
            "removed.")
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
        self._run_group = runbox
        rform = QFormLayout(runbox)
        iform = rform  # the time-lapse panel and Run button live here

        self._tl_mode = QCheckBox("Time-lapse (multiple ERT files)")
        self._tl_mode.setToolTip("Off: invert the single loaded dataset.  On: jointly invert an "
                                 "ordered sequence of ERT files with temporal regularization "
                                 "(the time-lapse options appear below).")
        self._tl_mode.toggled.connect(self._on_tl_mode)
        iform.addRow(self._tl_mode)
        self._sync_iteration_rows()

        self._invert_btn = QPushButton("Run inversion")
        self._invert_btn.setProperty("primary", True)
        self._invert_btn.setIcon(theme.icon("fa5s.play", color="#ffffff"))
        self._invert_btn.clicked.connect(self._run_inversion)
        iform.addRow(self._invert_btn)
        self._inv_progress = QProgressBar()
        self._inv_progress.setVisible(False)
        self._inv_stop = self.stop_button()
        iform.addRow(progress_with_stop(self._inv_progress, self._inv_stop))

        iform.addRow(self._build_timelapse_panel())
        # Reflect the initial checkbox states; setChecked() above emitted nothing.
        self._on_inv_mode_changed()
        self._on_auto_lambda(self._auto_lam.isChecked())
        self._on_reject_outliers(self._reject.isChecked())
        layout.addWidget(runbox)

        # Electrode editing has no panel: placing and dragging electrodes by mouse
        # was fiddly and rarely the right way to fix a geometry. The operations
        # live on as agent actions (add/move/delete/label/clear/list electrodes),
        # which is both more precise and reproducible. Clicking the plot still
        # selects an electrode so its position can be read off.

        exp = QGroupBox("Export")
        self._geometry_export_group = exp
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
        layout.addWidget(exp)

        # Result exports belong beside the result, where model post-processing
        # remains available without retaining the entire preparation column.
        self._model_export_group = QGroupBox("Export model")
        model_exports = QVBoxLayout(self._model_export_group)
        model_exports.addWidget(self._model_export_btn)
        model_exports.addWidget(self.map_export_button())
        self._postprocess_layout.addWidget(self._model_export_group)
        self._postprocess_layout.addStretch(1)
        self._postprocess_scroll.fit_to_content()

        layout.addStretch(1)
        scroll.fit_to_content()
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

        self._tl_dt_weight = QCheckBox("Weight by the interval between surveys")
        self._tl_dt_weight.setChecked(True)
        self._tl_dt_weight.setToolTip(
            "Penalize the rate of change rather than the raw difference, so a "
            "month-long gap is allowed proportionally more change than an hour-long "
            "one. Weights are normalized by the median interval, so an evenly "
            "sampled series is unchanged and Alpha keeps its meaning. Untick to "
            "constrain every pair equally, as before.")
        tlform.addRow(self._tl_dt_weight)

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

        # Acquisition times. The file list above shows what was read and the gap
        # between surveys; this is the escape hatch for a set whose names carry no
        # time at all.
        self._tl_use_mtime = QCheckBox("Use file times when the names carry none")
        self._tl_use_mtime.setToolTip(
            "Fall back to each file's modification time when neither its name nor "
            "its header gives an acquisition time. Off by default: a file that was "
            "copied or re-exported carries the time of the copy, not of the survey.")
        self._tl_use_mtime.toggled.connect(self._on_tl_time_source_changed)
        tlform.addRow(self._tl_use_mtime)

        self._tl_btn = QPushButton("Run time-lapse inversion")
        self._tl_btn.setProperty("primary", True)
        self._tl_btn.setIcon(theme.icon("fa5s.history", color="#ffffff"))
        self._tl_btn.clicked.connect(self._run_timelapse)
        tlform.addRow(self._tl_btn)
        self._tl_progress = QProgressBar(); self._tl_progress.setVisible(False)
        self._tl_stop = self.stop_button("The time-lapse inversion")
        tlform.addRow(progress_with_stop(self._tl_progress, self._tl_stop))
        self._tl_export_btn = QPushButton("Export results (VTK + npy + mesh)…")
        self._tl_export_btn.setIcon(theme.icon("fa5s.cube"))
        self._tl_export_btn.setToolTip("Saves the time-lapse models to a chosen folder: a combined VTK, "
                                       "per-step VTKs, final_models.npy, the mesh (.bms), times CSV, and the figure.")
        self._tl_export_btn.setEnabled(False)
        self._tl_export_btn.clicked.connect(self._export_tl_results)
        tlform.addRow(self._tl_export_btn)
        tlform.addRow(self.map_export_button())
        self._tl_open = QPushButton("Open output folder")
        self._tl_open.setIcon(theme.icon("fa5s.folder-open"))
        self._tl_open.setEnabled(False)
        self._tl_open.clicked.connect(self._open_tl_output)
        tlform.addRow(self._tl_open)

        panel.setVisible(False)
        return panel

    def _build_temperature_group(self) -> QWidget:
        """The shared temperature-correction panel, beside the model it corrects.

        Nothing is decided before the run. The inversion does not depend on the
        ground temperature, only what its models are reported at, so the user
        inverts, then sets the options and presses Apply, and the model on screen
        is corrected in place - and can be corrected differently, or put back,
        without inverting again. That holds for one survey as much as for a
        series. A correction applied with a guessed temperature is its own error
        source, which is why it is a button and never a default.

        The panel lives in widgets/temperature_panel.py and keeps its settings in
        the studio state, so Saved Results offers the same correction, with the
        same choices, on a result reopened there.
        """
        self._tc_box = temperature_panel.TemperatureOptions(
            shared=getattr(self.state, "temperature_settings", None))
        self._tc_box.applyRequested.connect(self._apply_temperature)
        self._tc_box.removeRequested.connect(lambda: self._apply_temperature(None))
        self._tc_box.set_available(False, _TC_WAITING)
        panel = QWidget()
        self._postprocess_layout = QVBoxLayout(panel)
        self._postprocess_layout.setContentsMargins(0, 0, 4, 0)
        self._postprocess_layout.addWidget(self._tc_box)
        side = ContentWidthScrollArea(minimum=320, maximum=380)
        side.setWidget(panel)
        self._postprocess_scroll = side
        return side

    def _apply_temperature(self, spec: Optional[Dict[str, Any]]) -> Tuple[bool, str]:
        """Correct the model on screen, whichever kind of result it is."""
        kind = getattr(self, "_map_result_kind", "")
        if kind == "timelapse" and self._tl_models_raw is not None:
            return self._apply_tl_temperature(spec)
        if kind == "single" and self._inv_mgr is not None:
            return self._apply_single_temperature(spec)
        self._tc_box.set_available(False, _TC_WAITING)
        return False, "Run an inversion first."

    def _single_survey_times(self) -> Tuple[Optional[list], Optional[list]]:
        """``(days, dates)`` of the one survey behind the single inversion.

        Its acquisition date, from the file name or header, is what places it
        in a dated temperature record; undated, only a constant or a
        single profile can correct it, and the correction says so. The file is
        the one recorded with the model, not whichever was previewed since.
        """
        stamp, _where = self._survey_time(self._inv_source)
        return ([0.0], [stamp]) if stamp is not None else (None, None)

    def _survey_time(self, path: Optional[Path]) -> Tuple[Optional[Any], str]:
        """``(time, where)`` one survey file was acquired, or ``(None, "")``.

        Read the way the file list reads a sequence - the name, then a date line
        at the top of the file - and ``where`` says which ("file name" or "file
        header"). Kept per file and modification time, because the data panel
        asks on every redraw and a file given a date line since must be read again.
        """
        if path is None:
            return None, ""
        try:
            key = (str(path), Path(path).stat().st_mtime)
        except OSError:
            key = (str(path), None)
        cache = getattr(self, "_survey_times", None)
        if cache is None:
            cache = self._survey_times = {}
        if key not in cache:
            timing = survey_timing.survey_timing([str(path)])
            stamp = timing.timestamps[0] if timing.timestamps else None
            cache[key] = (stamp, timing.sources[0] if stamp is not None else "")
        return cache[key]

    def _single_title(self) -> str:
        """The single model's heading, with the survey's time when it has one.

        Saved results heads the same run the same way, from the time the run
        recorded (``survey_time`` in its summary).
        """
        when = survey_timing.format_time(self._survey_time(self._inv_source)[0])
        return f"Resistivity · {when}" if when else "Resistivity"

    def _survey_time_record(self, path: Optional[Path]) -> Dict[str, Any]:
        """What the run records about the survey it inverts, for Saved results.

        The run stages the data under a generic name, so the file it came from
        and the time it was acquired survive only if they are written down here.
        """
        if path is None:
            return {}
        stamp, where = self._survey_time(path)
        record: Dict[str, Any] = {"source_file": Path(path).name}
        if stamp is not None:
            record.update(acquired=stamp.isoformat(sep=" "), acquired_from=where)
        return record

    def _show_format_help(self) -> None:
        """The ERT input-format note: formats, electrode file, survey times."""
        from PySide6.QtWidgets import QDialog, QTextBrowser

        doc_path = Path(__file__).with_name("ert_input_format.md")
        try:
            text = doc_path.read_text(encoding="utf-8")
        except Exception:  # noqa: BLE001
            text = ("Pick the instrument / format, then add one file per survey. "
                    "Name each file with its time, year first: "
                    "site_2026-01-12_05-50-38.dat.")
        dlg = QDialog(self); dlg.setWindowTitle("ERT input formats")
        dlg.resize(760, 640); lay = QVBoxLayout(dlg)
        browser = QTextBrowser(); browser.setOpenExternalLinks(True)
        try:
            browser.setMarkdown(text)
        except Exception:  # noqa: BLE001
            browser.setPlainText(text)
        lay.addWidget(browser)
        close = QPushButton("Close"); close.clicked.connect(dlg.accept); lay.addWidget(close)
        dlg.exec()

    def _apply_single_temperature(self, spec: Optional[Dict[str, Any]]) -> Tuple[bool, str]:
        """Correct the single-inversion model on screen, or put the inverted one back."""
        mgr = self._inv_mgr
        if spec is None:
            self._inv_correction = None
            self._show_single_model(None)
            self.log("Temperature correction removed; the section shows the "
                     "resistivity as inverted.", "info")
            self._tc_box.show_applied(None)
            return True, ""
        days, dates = self._single_survey_times()
        try:
            corrected, report = temperature_panel.correct_series(
                mgr.paraDomain, np.asarray(mgr.model, dtype=float), spec,
                days=days, dates=dates)
        except Exception as exc:  # noqa: BLE001 - report it, never lose the result
            self.log(f"Temperature correction not applied: {exc}", "warn")
            self._tc_box.show_problem(f"Not applied: {exc}")
            return False, str(exc)
        self._inv_correction = {**report, "spec": dict(spec)}
        self._show_single_model(corrected)
        self.log(f"Temperature correction: {report['note']}", "success")
        self._tc_box.show_applied(report)
        return True, str(report["note"])

    def _show_single_model(self, corrected: Optional[np.ndarray]) -> None:
        """Draw the single model: as inverted, or corrected and titled as such."""
        mgr = self._inv_mgr
        self._inv_corrected = None if corrected is None else np.asarray(corrected, dtype=float)
        if self._inv_corrected is None:
            self._model_view.show_model(mgr, kind="ert", title=self._single_title())
        else:
            # The coverage the uncorrected view would fade by, accepted on the same
            # terms: a flat or empty coverage is no mask at all, and handed over as
            # one it would make every cell transparent.
            coverage = None
            try:
                cov = np.asarray(mgr.coverage(), dtype=float)
                if (cov.size == self._inv_corrected.size and np.isfinite(cov).any()
                        and float(np.nanmax(cov)) > float(np.nanmin(cov))):
                    coverage = cov
            except Exception:  # noqa: BLE001 - coverage is optional
                pass
            self._model_view.show_field(
                mgr.paraDomain, self._inv_corrected, kind="ert", coverage=coverage,
                title=self._single_title()
                + temperature_panel.title_suffix(self._inv_correction))
        self._tabs.setCurrentWidget(self._model_tab)

    def _tl_summary(self) -> Dict[str, Any]:
        """The time-lapse result as its run recorded it, step titles included."""
        return {**(self._tl_result or {}), "step_titles": list(self._tl_step_titles)}

    def _apply_tl_temperature(self, spec: Optional[Dict[str, Any]]) -> Tuple[bool, str]:
        """Correct the time-lapse sections on screen, or put the inverted ones back.

        Works on the models this page holds; the correction is the same code
        Saved Results uses, so a run corrected on either page reads the same.
        Returns ``(ok, message)``.
        """
        raw = self._tl_models_raw
        if raw is None or self._tl_mesh is None:
            self._tc_box.set_available(False, _TC_WAITING)
            return False, "Run the time-lapse inversion first."
        if spec is None:
            self._tl_models = raw
            self._tl_correction = None
            self._redraw_tl()
            self.log("Temperature correction removed; the sections show the "
                     "resistivity as inverted.", "info")
            self._tc_box.show_applied(None)
            return True, ""
        days, dates = temperature_panel.series_survey_times(
            self._tl_summary(), raw.shape[1])
        try:
            corrected, report = temperature_panel.correct_series(
                self._tl_mesh, raw, spec, days=days, dates=dates)
        except Exception as exc:  # noqa: BLE001 - report it, never lose the result
            self.log(f"Temperature correction not applied: {exc}", "warn")
            self._tc_box.show_problem(f"Not applied: {exc}")
            return False, str(exc)
        self._tl_models = corrected
        # The spec travels with the report: it is what an export records as
        # having been applied, and it embeds the temperature data it read.
        self._tl_correction = {**report, "spec": dict(spec)}
        self._redraw_tl()
        self.log(f"Temperature correction: {report['note']}", "success")
        self._tc_box.show_applied(report)
        return True, str(report["note"])

    def _redraw_tl(self) -> None:
        """Show the current step again, on a scale fitted to the models now held."""
        self._seed_tl_color_range()
        self._show_tl_step(max(0, self._tl_step_combo.currentIndex()))
        self._tabs.setCurrentWidget(self._model_tab)

    def _on_tl_time_source_changed(self, *_args: Any) -> None:
        """Re-read the acquisition times when the allowed sources change; the
        pairs of forward and reciprocal files are found by them too."""
        if self._tl_all:
            self._set_tl_files(self._tl_all)

    def _on_tl_mode(self, checked: bool) -> None:
        """Toggle between single-file and time-lapse inversion."""
        self._tl_panel.setVisible(bool(checked))
        self._invert_btn.setVisible(not checked)
        # A time-lapse run builds its mesh from the first survey of the series.
        self._sync_mesh_engine()
        self._mesh_inputs_changed()
        self._sync_iteration_rows()
        # The Reciprocal errors tab shows the whole series in time-lapse mode.
        self._recip_inputs_changed()

    def _sync_iteration_rows(self) -> None:
        """Show the iteration limits the next run obeys, and only those.

        A single run stops at "Max iterations" or once χ² stops improving, on
        every engine. A time-lapse run on the in-house or ADTLERT engine has
        one limit, its own iteration count; E4D runs each survey until χ²
        stops improving and takes no count; R2 and R3t decide both themselves.
        """
        from PyHydroGeophysX.qt_apps.qt_utils import set_rows_enabled

        timelapse = self._tl_mode.isChecked()
        engine = self._engine.currentData()
        # The two limits share a name, so only the mode's own is on screen; a
        # limit the engine ignores stays visible but greyed, as elsewhere.
        self._set_rows_visible([self._iter_ceiling], not timelapse)
        self._set_rows_visible([self._iter], timelapse)
        set_rows_enabled([self._iter], engine not in ("e4d", "r2", "r3t"))
        set_rows_enabled([self._plateau], not timelapse or engine == "e4d")

    def _on_engine_changed(self, _index: int = -1) -> None:
        """Probe the selected CUDA backend without changing user parameters."""
        e4d = self._engine.currentData() == "e4d"
        self._set_rows_visible(self._e4d_rows, e4d)
        self._sync_iteration_rows()
        if e4d:
            self._check_e4d()
        program = self._r2_program_name()
        self._set_rows_visible(self._r2_rows, program is not None)
        if program is not None:
            self._load_r2_settings(program)
            self._check_r2()
        previous = self._adtlert_probe_worker
        if previous is not None and previous.isRunning():
            previous.cancel()
        if self._engine.currentData() != "adtlert":
            self._adtlert_probe_serial += 1
            self._adtlert_status.setVisible(False)
            return
        self._adtlert_probe_serial += 1
        serial = self._adtlert_probe_serial
        self._adtlert_runtime_ready = None
        self._adtlert_single_ready = None
        self._adtlert_timelapse_ready = None
        self._adtlert_checks = {}
        self._adtlert_status.setText("Checking CUDA, cuDSS, and GPU CGLS…")
        theme.set_tone(self._adtlert_status, "hint")
        self._adtlert_status.setVisible(True)
        worker = ProcessProbeWorker(
            "PyHydroGeophysX.inversion.adtlert_diagnostics",
            arguments=["--stage", "runtime"],
            timeout_ms=60000,
        )
        worker.succeeded.connect(
            lambda result, token=serial: self._on_adtlert_probe_ok(token, result)
        )
        worker.failed.connect(
            lambda message, token=serial: self._on_adtlert_probe_failed(
                token, message
            )
        )
        self._adtlert_probe_worker = self.register_worker(worker, activity="Checking the GPU inversion engine")
        worker.start()

    def _on_adtlert_probe_ok(self, serial: int, result: Dict[str, Any]) -> None:
        if serial != self._adtlert_probe_serial:
            return
        if self._engine.currentData() != "adtlert":
            return
        self._adtlert_runtime_ready = True
        self._adtlert_checks["runtime"] = f"CUDA: ready ({result['device']})"
        self._start_adtlert_numerical_check(serial, "single")

    def _start_adtlert_numerical_check(self, serial: int, stage: str) -> None:
        self._adtlert_status.setText(
            " · ".join(self._adtlert_checks.values()) + f" · Checking {stage} inversion…"
        )
        worker = ProcessProbeWorker(
            "PyHydroGeophysX.inversion.adtlert_diagnostics",
            arguments=["--stage", stage], timeout_ms=120000,
        )
        worker.succeeded.connect(lambda result: self._finish_adtlert_numerical_check(serial, stage, True, ""))
        worker.failed.connect(lambda error: self._finish_adtlert_numerical_check(serial, stage, False, error))
        self._adtlert_probe_worker = self.register_worker(worker, activity="Checking the GPU inversion engine")
        worker.start()

    def _finish_adtlert_numerical_check(self, serial: int, stage: str, ok: bool, error: str) -> None:
        if serial != self._adtlert_probe_serial or self._engine.currentData() != "adtlert":
            return
        setattr(self, f"_adtlert_{stage}_ready", ok)
        label = "Single GPU" if stage == "single" else "Time-lapse GPU"
        self._adtlert_checks[stage] = f"{label}: {'passed' if ok else 'failed'}"
        if error:
            self.log(f"{label} self-test failed: {error}. Select In-house Gauss-Newton / PyHydro for this mode. "
                     "Run python -m PyHydroGeophysX.inversion.adtlert_diagnostics --report gpu-check.json for full diagnostics.", "warn")
        if stage == "single":
            self._start_adtlert_numerical_check(serial, "timelapse")
        else:
            self._adtlert_status.setText(" · ".join(self._adtlert_checks.values()))
            healthy = self._adtlert_single_ready and self._adtlert_timelapse_ready
            theme.set_tone(self._adtlert_status, "ok" if healthy else "error")

    def _on_adtlert_probe_failed(self, serial: int, message: str) -> None:
        if serial != self._adtlert_probe_serial:
            return
        if self._engine.currentData() != "adtlert":
            return
        self._adtlert_runtime_ready = False
        self._adtlert_single_ready = False
        self._adtlert_timelapse_ready = False
        self._adtlert_status.setText(
            f"CUDA check failed; GPU inversion is not verified. Select PyHydro CPU. {message}"
        )
        theme.set_tone(self._adtlert_status, "error")

    # -- E4D -----------------------------------------------------------------
    _E4D_KEYS = {"launcher": "ert/e4d/launcher", "executable": "ert/e4d/executable",
                 "processes": "ert/e4d/processes"}

    def _build_e4d_rows(self, form: QFormLayout) -> None:
        """How E4D is reached: shown only while E4D is the engine.

        E4D is an external program, so where it is - this computer, WSL, or
        nowhere, in which case the run folder is written for elsewhere - is the
        first thing the page says about it. The choices are kept between
        sessions, since they describe the machine rather than the survey.
        """
        from PySide6.QtCore import QSettings

        saved = QSettings("PyHydroGeophysX", "Studio")
        self._e4d_launcher = QComboBox()
        for label, value in (("Auto (this computer, then WSL)", "auto"),
                             ("This computer", "local"),
                             ("WSL (Windows)", "wsl"),
                             ("Write files only", "files")):
            self._e4d_launcher.addItem(label, value)
        index = self._e4d_launcher.findData(str(saved.value(self._E4D_KEYS["launcher"], "auto")))
        self._e4d_launcher.setCurrentIndex(max(index, 0))
        self._e4d_launcher.setToolTip(
            "Where E4D runs. E4D is built from source (github.com/pnnl/E4D) with "
            "gfortran, PETSc and MPI, for Linux.\n\n"
            "This computer: Linux, or macOS with a source build; e4d and mpirun are "
            "found on PATH, through PYHYDRO_E4D / PYHYDRO_MPIRUN, or from the program "
            "given below.\n"
            "WSL: on Windows, E4D built inside a WSL 2 Linux distribution; the "
            "program below is then a Linux path. E4D has no Windows build of its own.\n"
            "Write files only: the complete E4D run folder is written and the run "
            "stops, for a cluster or another machine: run mpirun -np N e4d in it, "
            "then read it back with inversion.e4d.read_e4d_run.")
        form.addRow("Run E4D", self._e4d_launcher)
        self._e4d_program = QLineEdit(str(saved.value(self._E4D_KEYS["executable"], "")))
        self._e4d_program.setPlaceholderText("e4d on PATH")
        self._e4d_program.setToolTip(
            "The e4d program, when it is not on PATH: a path on this computer, or a "
            "Linux path for WSL. Leave empty to use PATH or PYHYDRO_E4D.")
        browse = QPushButton("…")
        browse.setMaximumWidth(32)
        browse.setToolTip("Choose the e4d program on this computer.")
        browse.clicked.connect(self._browse_e4d)
        self._e4d_program_row = merged_row(self._e4d_program, browse)
        form.addRow("E4D program", self._e4d_program_row)
        self._e4d_processes = QSpinBox()
        self._e4d_processes.setRange(2, 1024)
        self._e4d_processes.setValue(int(saved.value(self._E4D_KEYS["processes"], 4) or 4))
        self._e4d_processes.setToolTip(
            "MPI processes for mpirun -np. E4D needs at least two - one master and "
            "one or more workers - and no more workers than electrodes.")
        form.addRow("MPI processes", self._e4d_processes)
        self._e4d_status = QLabel()
        self._e4d_status.setWordWrap(True)
        self._e4d_status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        form.addRow("", self._e4d_status)
        self._e4d_rows = [self._e4d_launcher, self._e4d_program_row,
                          self._e4d_processes, self._e4d_status]
        self._set_rows_visible(self._e4d_rows, False)
        # A typed path is checked once typing pauses, not on every key.
        self._e4d_timer = QTimer(self)
        self._e4d_timer.setSingleShot(True)
        self._e4d_timer.setInterval(600)
        self._e4d_timer.timeout.connect(self._check_e4d)
        self._e4d_launcher.currentIndexChanged.connect(self._e4d_settings_changed)
        self._e4d_program.textChanged.connect(self._e4d_settings_changed)
        self._e4d_processes.valueChanged.connect(self._e4d_settings_changed)

    def _e4d_settings(self) -> Dict[str, Any]:
        """The E4D settings a run is given (``inversion.e4d.E4DSettings``)."""
        return {"launcher": str(self._e4d_launcher.currentData() or "auto"),
                "executable": self._e4d_program.text().strip(),
                "processes": int(self._e4d_processes.value())}

    def _browse_e4d(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choose the e4d program", "", "All files (*)")
        if path:
            self._e4d_program.setText(path)

    def _e4d_settings_changed(self, *_args: Any) -> None:
        from PySide6.QtCore import QSettings

        saved = QSettings("PyHydroGeophysX", "Studio")
        for key, value in self._e4d_settings().items():
            saved.setValue(self._E4D_KEYS[key], value)
        if self._engine.currentData() == "e4d":
            self._e4d_timer.start()

    def _check_e4d(self) -> None:
        """Find E4D the way a run will, in a separate process: asking WSL can
        take a few seconds while its virtual machine starts."""
        previous = self._e4d_probe_worker
        if previous is not None and previous.isRunning():
            previous.cancel()
        self._e4d_probe_serial += 1
        serial = self._e4d_probe_serial
        settings = self._e4d_settings()
        self._e4d_found = None
        self._e4d_status.setText("Looking for E4D…")
        theme.set_tone(self._e4d_status, "hint")
        arguments = ["--json", "--launcher", settings["launcher"],
                     "--processes", str(settings["processes"])]
        if settings["executable"]:
            arguments += ["--executable", settings["executable"]]
        worker = ProcessProbeWorker("PyHydroGeophysX.inversion.e4d", arguments=arguments,
                                    timeout_ms=90000)
        worker.succeeded.connect(lambda result, token=serial: self._on_e4d_found(token, result))
        worker.failed.connect(lambda message, token=serial: self._on_e4d_found(
            token, {"runs": False, "kind": "missing",
                    "message": f"The E4D check did not finish: {message}"}))
        self._e4d_probe_worker = self.register_worker(worker, activity="Looking for E4D")
        worker.start()

    def _on_e4d_found(self, serial: int, result: Dict[str, Any]) -> None:
        if serial != self._e4d_probe_serial or self._engine.currentData() != "e4d":
            return
        self._e4d_found = dict(result)
        kind = str(result.get("kind", "missing"))
        tone = {"local": "ok", "wsl": "ok", "command": "ok",
                "files": "warn"}.get(kind, "error")
        text = str(result.get("message", ""))
        if kind == "missing":
            text += " Runs will write the E4D run folder and stop."
        self._e4d_status.setText(text)
        theme.set_tone(self._e4d_status, tone)

    # -- R2 / R3t ------------------------------------------------------------
    #: The launcher is one choice for both programs; each has its own path.
    _R2_KEYS = {"launcher": "ert/r2/launcher", "r2": "ert/r2/executable",
                "r3t": "ert/r3t/executable"}

    def _r2_program_name(self) -> Optional[str]:
        """``r2`` or ``r3t`` while one of them is the engine, else None."""
        engine = str(self._engine.currentData() or "")
        return engine if engine in ("r2", "r3t") else None

    def _build_r2_rows(self, form: QFormLayout) -> None:
        """How R2 or R3t is reached: shown only while one of them is the engine.

        Like E4D they are external programs, so the page first says whether
        and how this computer runs them. The program path is kept per program
        between sessions, since it describes the machine.
        """
        self._r2_launcher = QComboBox()
        for label, value in (("Auto (this computer)", "auto"), ("Write files only", "files")):
            self._r2_launcher.addItem(label, value)
        self._r2_launcher.setToolTip(
            "Whether R2/R3t is run here. They are Windows programs (R2.exe, R3t.exe) from "
            "Andrew Binley's web page or any ResIPy installation.\n\n"
            "Auto: run natively on Windows, through Wine on Linux and macOS. The program "
            "is the one given below, else PYHYDRO_R2 / PYHYDRO_R3T, else ResIPy's copy, "
            "else the first on PATH.\n"
            "Write files only: the complete run folder is written and the run stops; run "
            "R2.exe or R3t.exe in it elsewhere, then read it back with "
            "inversion.r2.read_r2_run.")
        form.addRow("Run R2/R3t", self._r2_launcher)
        self._r2_program = QLineEdit()
        self._r2_program.setPlaceholderText("ResIPy's copy, or PATH")
        self._r2_program.setToolTip(
            "The R2.exe or R3t.exe to run, when another than ResIPy's or the one on "
            "PATH. Leave empty to search for it.")
        browse = QPushButton("…")
        browse.setMaximumWidth(32)
        browse.setToolTip("Choose the R2 or R3t program.")
        browse.clicked.connect(self._browse_r2)
        self._r2_program_row = merged_row(self._r2_program, browse)
        form.addRow("Program", self._r2_program_row)
        self._r2_status = QLabel()
        self._r2_status.setWordWrap(True)
        self._r2_status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        form.addRow("", self._r2_status)
        self._r2_rows = [self._r2_launcher, self._r2_program_row, self._r2_status]
        self._set_rows_visible(self._r2_rows, False)
        self._r2_timer = QTimer(self)
        self._r2_timer.setSingleShot(True)
        self._r2_timer.setInterval(600)
        self._r2_timer.timeout.connect(self._check_r2)
        self._r2_launcher.currentIndexChanged.connect(self._r2_settings_changed)
        self._r2_program.textChanged.connect(self._r2_settings_changed)

    def _load_r2_settings(self, program: str) -> None:
        """Show the saved choices for ``program`` without re-saving them."""
        from PySide6.QtCore import QSettings

        saved = QSettings("PyHydroGeophysX", "Studio")
        for widget in (self._r2_launcher, self._r2_program):
            widget.blockSignals(True)
        index = self._r2_launcher.findData(str(saved.value(self._R2_KEYS["launcher"], "auto")))
        self._r2_launcher.setCurrentIndex(max(index, 0))
        self._r2_program.setText(str(saved.value(self._R2_KEYS[program], "") or ""))
        self._r2_program.setPlaceholderText(
            f"{'R2' if program == 'r2' else 'R3t'}.exe from ResIPy, or PATH")
        for widget in (self._r2_launcher, self._r2_program):
            widget.blockSignals(False)

    def _r2_settings(self) -> Dict[str, Any]:
        """The R2/R3t settings a run is given (``inversion.r2.R2Settings``)."""
        return {"launcher": str(self._r2_launcher.currentData() or "auto"),
                "executable": self._r2_program.text().strip()}

    def _browse_r2(self) -> None:
        name = "R2" if self._r2_program_name() == "r2" else "R3t"
        path, _ = QFileDialog.getOpenFileName(self, f"Choose the {name} program", "",
                                              "Programs (*.exe);;All files (*)")
        if path:
            self._r2_program.setText(path)

    def _r2_settings_changed(self, *_args: Any) -> None:
        from PySide6.QtCore import QSettings

        program = self._r2_program_name()
        if program is None:
            return
        saved = QSettings("PyHydroGeophysX", "Studio")
        settings = self._r2_settings()
        saved.setValue(self._R2_KEYS["launcher"], settings["launcher"])
        saved.setValue(self._R2_KEYS[program], settings["executable"])
        self._r2_timer.start()

    def _check_r2(self) -> None:
        """Find R2 or R3t the way a run will, in a separate process."""
        program = self._r2_program_name()
        if program is None:
            return
        previous = self._r2_probe_worker
        if previous is not None and previous.isRunning():
            previous.cancel()
        self._r2_probe_serial += 1
        serial = self._r2_probe_serial
        settings = self._r2_settings()
        self._r2_found = None
        self._r2_status.setText("Looking for the program…")
        theme.set_tone(self._r2_status, "hint")
        arguments = ["--json", "--program", program, "--launcher", settings["launcher"]]
        if settings["executable"]:
            arguments += ["--executable", settings["executable"]]
        worker = ProcessProbeWorker("PyHydroGeophysX.inversion.r2", arguments=arguments,
                                    timeout_ms=60000)
        worker.succeeded.connect(lambda result, token=serial: self._on_r2_found(token, result))
        worker.failed.connect(lambda message, token=serial: self._on_r2_found(
            token, {"runs": False, "kind": "missing",
                    "message": f"The check did not finish: {message}"}))
        self._r2_probe_worker = self.register_worker(worker, activity="Looking for R2 / R3t")
        worker.start()

    def _on_r2_found(self, serial: int, result: Dict[str, Any]) -> None:
        if serial != self._r2_probe_serial or self._r2_program_name() is None:
            return
        self._r2_found = dict(result)
        kind = str(result.get("kind", "missing"))
        tone = {"native": "ok", "wine": "ok",
                "files": "warn"}.get(kind, "error")
        text = str(result.get("message", ""))
        if kind == "missing":
            text += " Runs will write the run folder and stop."
        # λ stays on the page for the other engines; say here that it is unused.
        text += (" It chooses its own smoothing weight at every iteration: λ and "
                 "auto-λ are not used.")
        self._r2_status.setText(text)
        theme.set_tone(self._r2_status, tone)

    def _warn_external_not_running(self, what: str) -> None:
        """Before a run on E4D, R2 or R3t, say when the program will not run
        here: the run then writes its folder and stops."""
        engine = str(self._engine.currentData() or "")
        if engine == "e4d":
            found, name = self._e4d_found or {}, "E4D"
        elif engine in ("r2", "r3t"):
            found, name = self._r2_found or {}, "R2" if engine == "r2" else "R3t"
        else:
            return
        if not found.get("runs", False):
            self.log(f"{name} will not run here ("
                     + (str(found.get("message")) if found else "it has not been found yet")
                     + f"). The run writes the complete {name} {what} and stops; its path "
                     "is in the message.", "warn")

    # -- loading -------------------------------------------------------------
    def _start_load(self, path: str,
                    done: Optional[Callable[[str, Any], None]] = None) -> None:
        """Load one ERT file (off the UI thread) into the electrode + pseudosection
        view and make it the current single-inversion dataset. Used when a file is
        added and when a row in the file list is clicked to preview it.

        ``done(outcome, value)`` is told how this load ended, once the page has
        been updated: ``("loaded", result)``, ``("failed", message)``, or
        ``("superseded", None)`` when a newer load replaced it first. It is
        connected before the worker starts, so a fast parse cannot be missed.

        A forward file paired with its reciprocal file in the list is loaded
        with it, the two merged into one survey."""
        instrument = self._instrument.currentData()
        # Capture widget/state values on the UI thread; the parse runs off-thread.
        out_dir = self.state.ensure_results_store().scratch_dir(self.module_key)
        elec_file = self._electrode_file()
        spacing = None  # geometry comes from the file; instrument loaders handle layout
        reciprocal = self._tl_partner.get(str(path))
        self._info.setText(f"Loading {Path(path).name}"
                           + (" with its reciprocal file" if reciprocal else "") + "…")
        worker = TaskWorker(self._parse_ert, path, instrument, out_dir, elec_file, spacing,
                            reciprocal)
        # Only the latest request may land. Each click starts a load without
        # waiting for the one before, and a slow earlier file finishing last
        # replaced the data under a list highlighting the newer one - so the
        # next inversion ran on a survey nobody had selected.
        def landed(res, w=worker):
            if w is not self._load_worker:
                if done is not None:
                    done("superseded", None)
                return
            if done is None:
                self._on_ert_loaded(path, res)
                return
            try:
                self._on_ert_loaded(path, res)
            except Exception as exc:  # noqa: BLE001 - the caller reports it, as it did
                done("failed", str(exc))
            else:
                done("loaded", res)

        def refused(message, w=worker):
            if w is self._load_worker:
                self._on_ert_load_failed(message)
                if done is not None:
                    done("failed", message)
            elif done is not None:
                done("superseded", None)

        worker.succeeded.connect(landed)
        worker.failed.connect(refused)
        if done is not None:
            # A cancelled load emits no result at all, only ``finished``.
            worker.finished.connect(lambda: done("superseded", None))
        self._supersede_load()
        self._load_worker = self.register_worker(worker, activity="Reading the data file")
        worker.start()

    def _load_and_wait(self, path: str) -> Tuple[str, Any]:
        """Load ``path`` through :meth:`_start_load` and wait for it to land.

        For callers that need the outcome before they can answer - the
        assistant's ``load_data``, which reports the loaded counts. They run on
        the UI thread, and parsing there froze the studio for 0.65-0.9 s per
        survey; the parse now runs on the worker while a local event loop keeps
        the window painting and responding until this load has settled.

        Returns
        -------
        tuple
            ``(outcome, value)`` as :meth:`_start_load` reports them.
        """
        settled: Dict[str, Any] = {}
        loop = QEventLoop()

        def done(outcome: str, value: Any) -> None:
            if not settled:        # the first report wins; ``finished`` follows it
                settled.update(outcome=outcome, value=value)
                loop.quit()

        self._start_load(path, done)
        if not settled:
            loop.exec()
        return settled["outcome"], settled["value"]

    def _supersede_load(self) -> None:
        """Drop a load still in flight: a newer request replaces it."""
        previous, self._load_worker = self._load_worker, None
        if previous is not None and previous.isRunning():
            previous.cancel()

    def _electrode_file(self) -> Optional[str]:
        """The electrode table the instrument readers are handed, if any."""
        table = self._electrode_table
        return str(table) if table is not None and table.exists() else None

    def _run_label(self, files, unit: str = "files") -> str:
        """The name a new run is listed under in Saved Results.

        Drawn from its data - the file, or the name a sequence shares and how
        many surveys - with the engine added when it is not the in-house one,
        so runs of different surveys no longer all read ``Run`` and a code.
        """
        from PyHydroGeophysX.qt_apps.results_store import run_label_from_files

        label = run_label_from_files(files, unit=unit)
        if label and str(self._engine.currentData() or "pyhydro") != "pyhydro":
            label += " · " + self._engine.currentText().split(" ")[0]
        return label

    @staticmethod
    def _resipy_version() -> str:
        """Whether ResIPy is available, and its version where it reports one.

        Returns ``""`` when it is not installed. The installed distribution's
        metadata comes first: importing ResIPy checks the SHA1 of each of its
        executables, which held up the UI thread for most of a second as this
        page opened, and the loader imports it anyway, on its worker thread,
        when it reads a file. Without metadata the module is imported and asked:
        it carries its version as ``ResIPy_version`` rather than the usual
        ``__version__``, older builds carry neither, and a bare "yes" is the
        last resort.
        """
        import importlib.util
        import sys
        from importlib.metadata import version

        resipy = sys.modules.get("resipy")
        if resipy is None:
            try:
                if importlib.util.find_spec("resipy") is None:
                    return ""
                return str(version("resipy"))
            except Exception:  # noqa: BLE001 - no metadata; ask the module
                pass
            try:
                import resipy
            except Exception:  # noqa: BLE001 - an optional reader
                return ""
        for attribute in ("ResIPy_version", "__version__"):
            found = getattr(resipy, attribute, None)
            if found:
                return str(found)
        try:
            return str(version("resipy"))
        except Exception:  # noqa: BLE001
            return "yes"

    def _show_reader_status(self, reader: str = "", detail: str = "") -> None:
        """One line naming the reader available, or the one that read the file.

        ``reader`` is empty before a file is loaded, when the line reports only
        what is installed.
        """
        version = self._resipy_version()
        named = "ResIPy" if version == "yes" else f"ResIPy {version}"
        self._resipy_ready.setText("ResIPy ready" if version else "ResIPy unavailable")
        theme.set_tone(self._resipy_ready, "ok" if version else "warn")
        self._resipy_ready.setToolTip(
            f"{named} available for loading data." if version else
            "ResIPy is not installed; some formats require it. See Data format.")
        if not reader:
            if version:
                self._reader_status.setText(f"{named} available for loading data.")
                self._instrument.setToolTip(self._reader_status.text())
                self._reader_status.hide()
                return
            from PyHydroGeophysX.data_processing.ert_data_agent import (
                _EMBEDDED_PARSER_MAP, resipy_install_hint,
            )
            self._reader_status.setText("ResIPy unavailable; some formats need it.")
            details = ("Available readers: " + ", ".join(sorted(_EMBEDDED_PARSER_MAP))
                       + ".\n\n" + resipy_install_hint())
            self._reader_status.setToolTip(details)
            self._instrument.setToolTip(details)
            self._reader_status.show()
            return
        note = f", {detail}" if detail else ""
        self._reader_status.setText(f"Data loaded by {reader}{note}.")
        self._reader_status.setToolTip("")
        self._reader_status.show()

    def _parse_ert(self, path, instrument, out_dir, elec_file, spacing, reciprocal=None):
        """Parse ERT data off the UI thread. Returns a plain dict for the slot.

        With ``reciprocal``, that file is read the same way and its readings
        appended to ``path``'s (``ert_io.merge_reciprocal``): one survey, on
        the forward file's electrodes, whose readings pair across the two files.
        ``res["pair"]`` then holds the counts of each and of the pairs, and
        ``res["pairing"]`` is the survey's reciprocal pairing either way, for
        the Reciprocal errors tab and the error model.
        """
        from PyHydroGeophysX.qt_apps.ert_records import reciprocal_pairing

        res = self._parse_one(path, instrument, out_dir, elec_file, spacing)
        if not reciprocal:
            res["pairing"] = (reciprocal_pairing(self._reciprocal_scores(res["data"]))
                              if res["data"] is not None else None)
            return res
        if res["data"] is None:
            res["warning"] = "; ".join(filter(None, [res["warning"], (
                f"{Path(path).name} could not be converted for inversion, so its "
                f"reciprocal file {Path(reciprocal).name} was not merged with it.")]))
            return res
        other = self._parse_one(reciprocal, instrument, out_dir, elec_file, spacing)
        if other["data"] is None:
            raise ValueError(
                f"the reciprocal file {Path(reciprocal).name} could not be converted for "
                f"inversion ({other['data_note'] or 'no usable readings'}), so it cannot be "
                f"merged with {Path(path).name}. Untick “Pair reciprocal files” to load "
                "each file on its own.")
        try:
            merged, info = ert_load.merge_reciprocal(res["data"], other["data"])
        except ValueError as exc:
            raise ValueError(
                f"{Path(path).name} and its reciprocal file {Path(reciprocal).name} could "
                f"not be merged: {exc} Untick “Pair reciprocal files” to load each file on "
                "its own.") from exc
        pairing = reciprocal_pairing(self._reciprocal_scores(merged))
        info.update(file=str(reciprocal), pairs=int(pairing.get("pairs", 0) or 0),
                    unpaired=int(pairing.get("unpaired_readings", 0) or 0))
        res.update(data=merged, nmeas=int(merged.size()), pair=info, pairing=pairing,
                   pseudo=self._pseudo_from_data(merged),
                   warning="; ".join(filter(None, [res["warning"], other["warning"]])))
        return res

    def _parse_one(self, path, instrument, out_dir, elec_file, spacing):
        """Parse one ERT file; see :meth:`_parse_ert`."""
        warning = ""
        reader, reason = "ResIPy", ""
        if instrument is None:  # defensive: the dropdown has no auto/None option
            elec, pseudo, nmeas, data, note = self._load_pygimli(path, elec_file)
            reader = "PyGIMLi"
        else:
            try:
                elec, pseudo, nmeas, data, note, (reader, reason) = self._load_resipy(
                    path, instrument, out_dir, elec_file, spacing)
            except NotImplementedError:
                # No reader for this format. PyGIMLi's reader would accept the
                # file and return a quadrupole table built from the wrong
                # columns, so this has to stop here and say what is missing
                # rather than hand back a section nobody can trust.
                raise
            except Exception as exc:  # noqa: BLE001
                # PyGIMLi's reader answers for the unified family only; outside
                # it the chosen reader's refusal is the answer. Retried natively,
                # a DAS-1 file missing its data section loaded as one reading and
                # the reader's own explanation was lost. Same rule, and message,
                # as ert_io.load_ert_container, so a file fails the same way
                # here as in a time-lapse run.
                if ert_load._canonical_instrument(str(instrument)) not in ert_load._NATIVE_RETRY:
                    raise ValueError(
                        f"'{Path(path).name}' could not be read as {instrument}: {exc}"
                    ) from exc
                warning = f"{instrument} loader failed ({exc}); fell back to pygimli's native reader."
                elec, pseudo, nmeas, data, note = self._load_pygimli(path, elec_file)
                reader, reason = "PyGIMLi", "ResIPy could not parse this file"
        return {"elec": elec, "pseudo": pseudo, "nmeas": nmeas, "data": data,
                "warning": warning, "reader": reader, "reader_reason": reason,
                "data_note": note}

    def _on_ert_loaded(self, path: str, res: dict) -> None:
        if res.get("warning"):
            self.log(res["warning"], "warn")
        self._show_reader_status(str(res.get("reader", "")),
                                 str(res.get("reader_reason", "")))
        elec, pseudo, nmeas, data = res["elec"], res["pseudo"], res["nmeas"], res["data"]
        self._x = [float(e[0]) for e in elec]
        self._z = [float(e[1]) for e in elec]
        self._labels = [str(i + 1) for i in range(len(self._x))]
        self._electrode_origins = list(range(len(self._x)))
        self._selected = None
        self._data_path = Path(path)
        pair = res.get("pair")
        self._pair_info = dict(pair) if pair else None
        self._data_partner = Path(pair["file"]) if pair else None
        self._pseudo = pseudo
        self._n_meas = nmeas
        self._ert_data = data
        self._ert_data_full = data
        # A file can parse into electrodes and readings and still not convert
        # into a pygimli container — no pygimli, or quadrupoles that cite
        # electrodes the file never defines. The pseudosection is still drawn
        # from the raw values, so say here what has been lost rather than
        # leaving the module looking loaded until an inversion asks for it.
        self._qc_mask = [True] * int(data.size()) if data is not None else []
        self._qc_report = None       # these data are unfiltered until Apply filter
        self._loaded_pairing = res.get("pairing")
        self._data_note = ""
        if data is None:
            cause = str(res.get("data_note") or
                        "this file could not be converted for inversion.")
            self._data_note = cause  # _refresh puts it on the panel as well
            self.log(
                f"{Path(path).name}: {cause} The pseudosection below is drawn "
                "from the raw values, but QC filtering and inversion are "
                "unavailable for it.", "warn")
        # Which extra criteria this file can support is a property of the file,
        # so it is settled here rather than being rechecked on every Apply.
        self._refresh_qc_availability()
        if data is not None and hasattr(self.state, "register_geophysical_resource"):
            self.state.register_geophysical_resource(
                "ERT", "observed_data", data,
                label=f"ERT observations · {Path(path).name}", path=str(path),
                metadata={"measurements": int(nmeas), "electrodes": len(self._x),
                          **({"reciprocal_file": self._data_partner.name}
                             if self._data_partner else {})},
                resource_id="ert:observed_data:active",
            )
        self._refresh()
        self._draw_pseudosection()
        self._recip_inputs_changed()
        # The Reciprocal errors tab stays up when it is the one being looked at:
        # clicking through the list compares the surveys' pairs there.
        if pseudo and self._tabs.currentWidget() is not self._recip_view:
            self._tabs.setCurrentWidget(self._pseudo_widget)
        if data is None:
            pass  # already reported above, naming the specific cause
        elif nmeas == 0:
            self.log(f"{Path(path).name}: parsed {len(self._x)} electrodes but 0 measurements — "
                     f"the Instrument / format is probably wrong for this file.", "warn")
        elif pair:
            # As Craig Ulrich's pairing summary gives it: both files, the
            # readings of each, and how many quadrupoles were measured both ways.
            self.log(f"Loaded {len(self._x)} electrodes, {nmeas} measurements from "
                     f"{Path(path).name} with its reciprocal file {Path(pair['file']).name}: "
                     f"forward {pair['forward']} | reciprocal {pair['reciprocal']} | total "
                     f"{pair['total']}; {pair['pairs']} quadrupoles measured both ways, "
                     f"{pair['unpaired']} readings without a reciprocal.", "success")
            if pair.get("dropped_fields"):
                self.log("Only one of the two files carries "
                         + ", ".join(pair["dropped_fields"])
                         + ", so the merged survey leaves it out.", "warn")
        else:
            self.log(f"Loaded {len(self._x)} electrodes, {nmeas} measurements from {Path(path).name}", "success")

    def _name_electrode_file(self, message: str) -> str:
        """``message`` with the readers' copy of the electrode file named as the user's.

        The readers are handed a plain copy (:meth:`_electrode_table_for`), so
        their refusal named a scratch file nobody had heard of.
        """
        table, source = self._electrode_table, self._electrode_path
        if table is not None and source is not None:
            return str(message).replace(Path(table).name, source.name)
        return str(message)

    def _on_ert_load_failed(self, message: str) -> None:
        message = self._name_electrode_file(message)
        self.log(f"Could not load ERT data: {message}", "error")
        self._info.setText(f"Load failed: {message}")
        # Neither reader got there, so the line goes back to what is installed
        # rather than keeping the last file's answer.
        self._show_reader_status()

    def _load_pygimli(self, path: str, electrode_file: Optional[str] = None):
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
        # PyGIMLi's reader takes no electrode file, so it is applied here - or
        # refused, as the instrument readers refuse one that does not fit. This
        # fallback used to read the header's positions whatever had been loaded.
        if electrode_file:
            ert_load.place_electrodes(data, electrode_file)
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
        return elec, pseudo, int(data.size()), data, ""

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
        note = ""
        if data is not None:
            # Use the corrected apparent resistivity (rhoa = R * k) for QC display.
            pseudo = self._pseudo_from_data(data)
        else:
            note = self._container_failure_reason(std)
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
                    pseudo.append((float(np.mean(xs)),
                                   max(span * ert_load.PSEUDO_DEPTH_FACTOR, 0.01),
                                   float(obs.app_res)))
        return elec, pseudo, len(std.observations or []), data, note, self._reader_of(std)

    @staticmethod
    def _reader_of(std) -> Tuple[str, str]:
        """``(reader, detail)`` naming what actually read the file.

        The instrument loader falls back to this package's own readers when
        ResIPy is absent or has no reader for the format, so "ResIPy" is not a
        safe assumption. The detail carries what the reader worked out beyond the
        table - above all where the electrode positions came from, since a
        format with no coordinates has to get them from somewhere.
        """
        meta = dict(getattr(std, "metadata", None) or {})
        if meta.get("loader") != "local_parsers_resipy_fallback":
            return "ResIPy", ""
        reader = f"PyHydroGeophysX's {meta.get('parser_used') or meta.get('instrument')} reader"
        notes = dict(meta.get("reader_notes") or {})
        parts = []
        if notes.get("sequence_file"):
            parts.append(f"sequence {notes['sequence_file']}")
        cables = notes.get("cables") or {}
        if cables:
            count = sum(int(size) for size in cables.values())
            parts.append(f"{count} electrodes on {len(cables)} "
                         f"cable{'s' if len(cables) != 1 else ''}")
        if notes.get("geometry_note"):
            parts.append(str(notes["geometry_note"]))
        return reader, "; ".join(parts)

    @staticmethod
    def _container_failure_reason(std) -> str:
        """Why a parsed file did not become a pygimli container.

        Worth distinguishing: a missing pygimli is fixed by an install, whereas
        quadrupoles that cite electrodes the file never defines almost always
        mean the Instrument setting does not match the file — and the reader is
        happy to parse the wrong columns without complaint.
        """
        try:
            import pygimli  # noqa: F401
        except Exception:  # noqa: BLE001
            return ("pygimli is not installed, so the file can be plotted but "
                    "not inverted.")
        observations = list(std.observations or [])
        if not observations:
            return "no measurements were parsed from it."
        with_res = [obs for obs in observations if obs.app_res is not None]
        if not with_res:
            return ("no measurement carries an apparent resistivity, and none "
                    "could be rebuilt from the columns that were read.")
        known = {int(e.id) for e in (std.electrodes or [])}
        keys = ("A", "B", "M", "N")
        matched = [
            obs for obs in with_res
            if all(int(getattr(obs.quad, key)) in known for key in keys)
        ]
        if matched:
            return "the measurements could not be matched to the electrode table."
        if not known:
            return ("it defines no electrodes, so its quadrupoles cannot be "
                    "placed. The Instrument setting is probably wrong for this "
                    "file.")
        cited = sorted({int(getattr(obs.quad, key)) for obs in with_res for key in keys})
        return (f"its measurements cite electrodes {cited[0]}–{cited[-1]}, which "
                f"the electrode table ({min(known)}–{max(known)}) does not "
                "define, so no quadrupole can be placed. The Instrument setting "
                "is probably wrong for this file.")

    # Geometry/topography + StandardERT->pygimli conversion live in the shared
    # ``ert_load`` module so the single-inversion loader and the time-lapse
    # pipeline behave identically. Kept as thin static wrappers for callers here.
    _electrode_elevation = staticmethod(ert_load.electrode_elevation)
    _standard_to_pg = staticmethod(ert_load.standard_to_pg)

    @staticmethod
    def _build_pseudo_from_indices(x, a, b, m, n, rhoa) -> List[Tuple[float, float, float]]:
        # Placed as the workflow's raw-data figure places them (ert_io).
        return [tuple(row) for row in ert_load.pseudosection_points(x, a, b, m, n, rhoa).tolist()]

    def _pseudo_from_data(self, data) -> List[Tuple[float, float, float]]:
        return [tuple(row) for row in ert_load.container_pseudosection(data).tolist()]

    @classmethod
    def _reciprocal_error(cls, data) -> Optional[np.ndarray]:
        """Relative reciprocal error per measurement, NaN where there is no partner.

        See :meth:`_reciprocal_scores`, whose ``reciprocalErrRel`` this is.
        """
        scored = cls._reciprocal_scores(data)
        return None if scored is None else scored["reciprocalErrRel"].to_numpy(dtype=float)

    @staticmethod
    def _reciprocal_scores(data):
        """The survey paired with its reciprocals: a frame of a, b, m, n, resist
        with each reading's ``reciprocalErrRel`` and ``reciprocalMean``.

        A reciprocal swaps the current and potential pairs, (A,B,M,N) -> (M,N,A,B),
        and reciprocity requires the same transfer resistance from both. How far
        apart they land is the only error estimate that comes from the data rather
        than from an assumed percentage, which is why it is worth computing even
        though most files do not carry it as a column.

        The pairing and the score are the library's own
        (``ert_formats.reciprocal_errors``), so this page and a file's reciprocal
        analysis agree: a pair written in reverse order has its resistance
        negated before comparison - without that, (A,B,M,N) and (M,N,B,A) read
        as R against -R, a 200 % disagreement for a perfect pair - and repeats in
        one direction are stacked, not scored as reciprocals of each other.

        Returns None when the file has neither a resistance nor the rhoa and k
        needed to rebuild one. The pairing itself is ``ert_io.reciprocal_scores``,
        which also averages pairs and merges a survey's two files.
        """
        return ert_load.reciprocal_scores(data)

    def _refresh_qc_availability(self) -> None:
        """Enable each folded QC row only where the loaded file can support it.

        Voltage and current arrive with some instrument formats and not others,
        and a reciprocal error needs the survey to actually contain reciprocals.
        A row that cannot work is disabled with the reason in its tooltip, which
        is more useful than a control that silently filters on nothing.
        """
        def gate(widget, ok: bool, message: str) -> None:
            widget.setEnabled(bool(ok))
            label = self.row_label(widget)
            if label is not None:
                label.setEnabled(bool(ok))
            widget.setToolTip(message)

        data = self._ert_data_full
        self._refresh_error_sources(data)
        if data is None:
            for w in (self._qc_min_v, self._qc_min_i, self._qc_max_k, self._qc_max_rc,
                      self._qc_max_stack, self._qc_max_recip, self._qc_average):
                gate(w, False, "Load ERT data first.")
            self._qc_support_note.setText(
                "Load data to see which checks are available."
            )
            return

        # Volts and amperes: ert_io.standard_to_pg keeps these tokens only from a
        # reader that says it gives them in those units, and PyGIMLi's own reader
        # uses them. The defaults assume them; the observed range is quoted too.
        unavailable: List[str] = []
        for widget, token, name, unit, floor in (
                (self._qc_min_v, "u", "voltage", "V", "10 µV, a common noise floor"),
                (self._qc_min_i, "i", "current", "A", "0.1 mA, below which the "
                                                      "injection failed")):
            if data.haveData(token):
                v = np.abs(np.asarray(data[token], dtype=float))
                v = v[np.isfinite(v)]
                span = f"{v.min():.4g} to {v.max():.4g} {unit}" if v.size else "empty"
                gate(widget, True,
                     f"Drop readings whose |{token}| falls below this, in {unit} (0 = off; "
                     f"the default is {floor}). This file's {name} spans {span}.")
            else:
                gate(widget, False,
                     f"This file carries no {name} column, so there is nothing to test.")
                unavailable.append(f"Min |{token.upper()}| (no {name} column)")

        if data.haveData("k"):
            k = np.abs(np.asarray(data["k"], dtype=float))
            k = k[np.isfinite(k)]
            span = f"{k.min():.4g} to {k.max():.4g}" if k.size else "empty"
            multiple = _QC_DEFAULTS["k_over_median"]
            if self._qc_max_k_follows_file and k.size:
                cap = float(f"{multiple * float(np.median(k)):.2g}")
                self._qc_max_k.blockSignals(True)
                self._qc_max_k.setValue(cap)
                self._qc_max_k.blockSignals(False)
            gate(self._qc_max_k, True,
                 "Drop configurations whose geometric factor exceeds this (0 = off). A large "
                 "|k| multiplies the measured resistance and its noise with it, which is how "
                 f"geometry alone produces a rhoa outlier. This file spans {span}; until "
                 f"you set it, the cap is {multiple:g} × its median |k| "
                 f"({float(np.median(k)) if k.size else 0.0:.3g} m), since |k| grows with "
                 "the electrode spacing and no one value suits every survey.")
        else:
            gate(self._qc_max_k, False, "This file carries no geometric factors.")
            unavailable.append("Max |k| (no geometric factors)")

        if data.haveData("rc"):
            rc = np.asarray(data["rc"], dtype=float)
            rc = rc[np.isfinite(rc)]
            span = (f"{rc.min():.3g} to {rc.max():.3g} Ω, median {float(np.median(rc)):.3g} Ω"
                    if rc.size else "empty")
            gate(self._qc_max_rc, True,
                 "Drop readings whose transmitter contact resistance exceeds this (0 = off; "
                 "the default is 30 kΩ). A failed or dried-out electrode shows up here "
                 f"before it shows up in the section. This file's contacts run {span}.")
        else:
            gate(self._qc_max_rc, False,
                 "This file carries no contact resistance, so there is nothing to test.")
            unavailable.append("Max contact R (no contact resistance)")

        if data.haveData("stack"):
            spread = 100.0 * np.asarray(data["stack"], dtype=float)
            spread = spread[np.isfinite(spread)]
            span = (f"median {float(np.median(spread)):.3g}%, 90% of readings under "
                    f"{float(np.percentile(spread, 90)):.3g}%, largest {spread.max():.3g}%"
                    if spread.size else "empty")
            gate(self._qc_max_stack, True,
                 "Drop readings whose potential scattered by more than this across its own "
                 "samples, as a percentage of the potential (0 = off; the default, 50 %, is "
                 "a scatter of half the signal). A few percent is normal; hundreds of "
                 "percent is an electrode or a contact failing. This file's spread: "
                 f"{span}.")
        else:
            gate(self._qc_max_stack, False,
                 "This file records no spread of the potential's samples (Subsurface "
                 "Insights exports do), so there is nothing to test.")
            unavailable.append("Max stacking spread (no sample spread)")

        rec = self._reciprocal_error(data)
        paired = 0 if rec is None else int(np.isfinite(rec).sum())
        if paired:
            finite = rec[np.isfinite(rec)]
            gate(self._qc_max_recip, True,
                 f"Drop measurements whose normal and reciprocal disagree by more than this "
                 f"(0 = off; the default is 5 %). {paired} of {data.size()} measurements "
                 f"have a reciprocal here; "
                 f"their disagreement runs to {100.0 * finite.max():.1f}%, median "
                 f"{100.0 * float(np.median(finite)):.1f}%. Unpaired measurements are kept.")
            gate(self._qc_average, True,
                 "Check 0, before every other check: each reading and its reciprocal "
                 "become one reading, with the mean of their two resistances and their "
                 "difference as its error (never below the Minimum under Data "
                 f"errors). {paired} of {data.size()} readings here have a reciprocal. "
                 "Readings without one are kept as they are. The reciprocal-error check "
                 "still judges each averaged pair by its difference.\n\n"
                 "Off by default: kept apart, both readings go to the inversion and the "
                 "fit sees how far they disagree. Applied with Apply filter, like the "
                 "checks below, and to every survey of a time-lapse run.")
        else:
            gate(self._qc_max_recip, False,
                  "This survey contains no reciprocal pairs, so there is nothing to compare. "
                  "Reciprocals have to be measured in the field; they cannot be recovered here.")
            gate(self._qc_average, False,
                 "This survey contains no reading measured both ways, so there is nothing "
                 "to average. For a survey written as two files, add both and keep “Pair "
                 "reciprocal files” ticked.")
            unavailable.append("Max reciprocal error (no reciprocal pairs)")

        if unavailable:
            self._qc_support_note.setText(
                "Unavailable for this file: " + "; ".join(unavailable) + ". "
                "These controls are disabled because the required measurements "
                "were not recorded, not because the input box is broken."
            )
        else:
            self._qc_support_note.setText(
                "All extra checks are supported by this file. A value of 0 turns "
                "each numerical check off."
            )

    def _max_k_set_by_hand(self, _value: float) -> None:
        """Keep a |k| cap somebody typed: files loaded afterwards no longer set it."""
        self._qc_max_k_follows_file = False

    def _refresh_error_sources(self, data) -> None:
        """Offer the stacking-spread error model only for data that record a
        spread, and the reciprocal error model only for data measured both ways."""
        model = self._err_source.model()
        index = self._err_source.findData("reciprocal")
        item = model.item(index) if index >= 0 and hasattr(model, "item") else None
        if item is not None:
            # A series counts too: a time-lapse run fits the model over all of
            # its surveys, whichever one is on screen.
            paired = self._reciprocal_pair_count(data)
            available = bool(paired or self._tl_partner)
            item.setEnabled(available)
            item.setToolTip(
                "Each reading's error from a line fitted to the reciprocal pairs: "
                "log10(dR) = m·log10(R) + b, where R is a pair's mean resistance and dR "
                "the difference between its two directions, so dR = 10^b·R^m and the "
                "relative error is dR / R, never below the Minimum set below. A "
                "single inversion fits it to the survey's own pairs; a time-lapse "
                "inversion fits one line to the pairs of every survey and gives it to "
                "all of them. The Reciprocal errors tab draws the pairs and the line, "
                "and its “Fit to” chooser sets whether the line is fitted to all pairs "
                "or only to those the filter kept. The Data QC report and the settings "
                "file state the fit."
                if available else
                "Needs readings measured both ways (each reading and its reciprocal). "
                "The loaded data have none; for a survey written as two files, add both "
                "and keep “Pair reciprocal files” ticked.")
            if not available and self._err_source.currentData() == "reciprocal":
                self._err_source.setCurrentIndex(self._err_source.findData("file"))
                if data is not None:
                    self.log("These data have no reciprocal pairs, so the data errors are "
                             "taken from the file's err column instead.", "warn")
        index = self._err_source.findData("stack")
        item = model.item(index) if index >= 0 and hasattr(model, "item") else None
        if item is None:
            return
        available = data is not None and data.haveData("stack")
        item.setEnabled(available)
        item.setToolTip(
            "√(spread² + estimate²) for each reading: the scatter of its potential's "
            "samples, as a fraction of the potential, added in quadrature to the estimate "
            "below. The scatter within one reading is not its uncertainty and usually "
            "overstates it, so χ² tends to come out low; it is never the default."
            if available else
            "Needs the spread of each reading's potential samples, which the loaded data "
            "do not record (Subsurface Insights exports do).")
        if not available and self._err_source.currentData() == "stack":
            self._err_source.setCurrentIndex(self._err_source.findData("file"))
            if data is not None:
                self.log("These data record no stacking spread, so the data errors are "
                         "taken from the file's err column instead.", "warn")
        self._sync_error_rows()

    def _reciprocal_pair_count(self, data) -> int:
        """How many readings of ``data`` have a reciprocal (0 for no data)."""
        if data is None:
            return 0
        rec = self._reciprocal_error(data)
        return 0 if rec is None else int(np.isfinite(rec).sum())

    def _sync_error_rows(self, *_args: Any) -> None:
        """Show "Minimum" only while an error comes from reciprocals."""
        floor = getattr(self, "_recip_floor", None)
        if floor is None:                       # still being built
            return
        used = (self._err_source.currentData() == "reciprocal"
                or self._qc_average.isChecked())
        self._set_rows_visible([floor], used)

    def _reciprocal_floor(self) -> float:
        """The "Minimum" error as a fraction."""
        return float(self._recip_floor.value()) / 100.0

    def _qc_settings(self) -> Dict[str, Any]:
        """The QC thresholds as set in the panel, as plain values.

        Plain values so the same filter can run off the UI thread, over every
        survey of a time-lapse series, exactly as Apply filter ran it here.
        """
        return {
            "min_rhoa": float(self._rmin.value()), "max_rhoa": float(self._rmax.value()),
            "max_error": float(self._max_err.value()),
            # Folding the section away also switches its criteria off.
            "more_checks": bool(self._qc_more.isChecked()),
            "drop_nonpositive": bool(self._qc_drop_neg.isChecked()),
            "min_voltage": float(self._qc_min_v.value()),
            "min_current": float(self._qc_min_i.value()),
            "max_k": float(self._qc_max_k.value()),
            "max_contact_r": float(self._qc_max_rc.value()),
            "max_stack": float(self._qc_max_stack.value()),
            "max_reciprocal": float(self._qc_max_recip.value()),
            # Check 0 and the error each averaged pair takes, as a fraction.
            "average_reciprocals": bool(self._qc_average.isChecked()
                                        and self._qc_average.isEnabled()),
            "reciprocal_floor": self._reciprocal_floor(),
        }

    @classmethod
    def _qc_survey(cls, data, qc: Dict[str, Any],
                   report: Optional[Dict[str, Any]] = None):
        """``(survey, keep, reasons)``: ``data`` through check 0 and the checks.

        Check 0, when ``qc`` asks for it, averages each reading with its
        reciprocal (``ert_io.average_reciprocal_pairs``); the checks then run on
        what is left, the reciprocal-error check judging each averaged pair by
        its own difference. ``survey`` is the copy to invert, and ``keep`` says
        which rows of ``data`` it holds - an averaged pair as its first reading.
        The pairing in ``report`` is the survey's as read, and its ``readings``
        the count before check 0, so the logs total what the files held.
        """
        import pygimli as pg

        scores = cls._reciprocal_scores(data)
        work, reciprocal, averaged = data, None, None
        rows = np.arange(int(data.size()))
        if qc.get("average_reciprocals"):
            work, reciprocal, averaged = ert_load.average_reciprocal_pairs(
                data, scores, floor=float(qc.get("reciprocal_floor") or 0.0))
            rows = np.asarray(averaged.pop("kept_rows"), dtype=int)
            if report is not None:
                report["averaged"] = averaged
        passed, reasons = cls._qc_keep(work, qc, report, scores=scores,
                                       reciprocal=reciprocal)
        if averaged and averaged["removed"]:
            reasons.insert(0, f"averaging {averaged['pairs']} reciprocal pairs removed "
                              f"{averaged['removed']}")
        if report is not None:
            report["readings"] = int(data.size())
        survey = pg.DataContainerERT(work)
        survey.set("valid", pg.Vector(passed.astype(float)))
        survey.removeInvalid()
        keep = np.zeros(int(data.size()), dtype=bool)
        keep[rows[passed.astype(bool)]] = True
        pairing = (report or {}).get("pairing")
        if pairing and pairing.get("available"):
            # Which pairs the filter kept, for the error model fitted to them
            # and the Reciprocal errors tab that draws the rest in grey.
            from PyHydroGeophysX.qt_apps.ert_records import reciprocal_pair_kept

            kept = reciprocal_pair_kept(scores, keep, averaged=averaged is not None)
            if kept is not None:
                pairing["kept"] = kept
        return survey, keep, reasons

    @classmethod
    def _qc_keep(cls, data, qc: Dict[str, Any],
                 report: Optional[Dict[str, Any]] = None, *, scores=None,
                 reciprocal: Optional[np.ndarray] = None) -> Tuple[np.ndarray, List[str]]:
        """``(keep, reasons)``: which measurements pass ``qc``, and what cut the rest.

        Every criterion that drops anything says how many, the error limit too:
        it starts at 20 %, and a filter that quietly took readings for it would
        read as the ρa range having done so.

        ``report``, when given, is filled in as the filter runs - the readings,
        how they pair with their reciprocals, and each check with its threshold
        and the count before and after it - for the run's QC logs
        (``qt_apps/ert_records.py``). Recorded here rather than worked out again
        afterwards, so the logs say what this filter did.

        ``scores`` is the survey's reciprocal pairing when already made - of the
        survey as read, when ``data`` is its averaged copy - and ``reciprocal``
        the reciprocal error of each row of ``data``, which an averaged copy can
        no longer be paired to find.
        """
        rhoa = np.asarray(data["rhoa"], dtype=float)
        total = int(rhoa.size)
        checks: Optional[List[Dict[str, Any]]] = None
        if scores is None and (report is not None or (
                reciprocal is None and qc["more_checks"] and qc["max_reciprocal"] > 0)):
            scores = cls._reciprocal_scores(data)
        if reciprocal is None and scores is not None and len(scores) == total:
            reciprocal = scores["reciprocalErrRel"].to_numpy(dtype=float)
        if report is not None:
            from PyHydroGeophysX.qt_apps.ert_records import reciprocal_pairing

            checks = []
            report.update(readings=total, pairing=reciprocal_pairing(scores), checks=checks)
        keep = np.isfinite(rhoa) & (rhoa >= qc["min_rhoa"]) & (rhoa <= qc["max_rhoa"])
        reasons: List[str] = []
        if int((~keep).sum()):
            reasons.append(f"ρa outside {qc['min_rhoa']:g}–{qc['max_rhoa']:g} Ω·m dropped "
                           f"{int((~keep).sum())}")
        if checks is not None:
            checks.append({"name": "Apparent resistivity range",
                           "threshold": f"{qc['min_rhoa']:g} to {qc['max_rhoa']:g} ohm-m",
                           "before": total, "after": int(keep.sum())})
        if qc["max_error"] > 0 and data.haveData("err"):
            before = int(keep.sum())
            keep &= np.asarray(data["err"], dtype=float) <= (qc["max_error"] / 100.0)
            if before - int(keep.sum()):
                reasons.append(f"error above {qc['max_error']:g} % dropped "
                               f"{before - int(keep.sum())}")
            if checks is not None:
                checks.append({"name": "Relative error (err column)",
                               "threshold": f"<= {qc['max_error']:g} %",
                               "before": before, "after": int(keep.sum())})
        elif checks is not None:
            checks.append({"name": "Relative error (err column)", "before": None,
                           "threshold": "off" if qc["max_error"] <= 0 else
                           f"<= {qc['max_error']:g} %",
                           "note": "" if qc["max_error"] <= 0 else
                           "not applied: the file has no error column"})
        if qc["more_checks"]:
            reasons += cls._apply_extra_filters(data, keep, qc, checks=checks,
                                                reciprocal=reciprocal)
        elif checks is not None:
            checks.append({"name": "More checks", "threshold": "off", "before": None,
                           "note": "not applied"})
        if report is not None:
            report["kept"] = int(keep.sum())
        return keep, reasons

    @classmethod
    def _apply_extra_filters(cls, data, keep: np.ndarray, qc: Dict[str, Any], *,
                             checks: Optional[List[Dict[str, Any]]] = None,
                             reciprocal: Optional[np.ndarray] = None) -> List[str]:
        """Apply the folded QC criteria to ``keep`` in place; report what each cost.

        A criterion whose field is missing is skipped rather than failing the
        whole filter, matching the row being disabled in the panel. ``checks``
        collects each criterion as :meth:`_qc_keep` describes; ``reciprocal``
        is each row's reciprocal error already worked out for it, used rather
        than made again.
        """
        reasons: List[str] = []

        def cut(mask: np.ndarray, label: str, name: str = "", threshold: str = "") -> None:
            before = int(keep.sum())
            # In place on purpose: `keep &= mask` here would rebind the enclosing
            # name and raise UnboundLocalError instead of narrowing the caller's array.
            np.logical_and(keep, mask, out=keep)
            lost = before - int(keep.sum())
            if lost:
                reasons.append(f"{label} dropped {lost}")
            if checks is not None:
                checks.append({"name": name or label, "threshold": threshold,
                               "before": before, "after": int(keep.sum())})

        def skipped(name: str, threshold: str, note: str) -> None:
            if checks is not None:
                checks.append({"name": name, "threshold": threshold, "before": None,
                               "note": note})

        if qc["drop_nonpositive"]:
            cut(np.asarray(data["rhoa"], dtype=float) > 0.0, "ρa ≤ 0",
                "Apparent resistivity above 0", "> 0")
        else:
            skipped("Apparent resistivity above 0", "off", "")
        for key, token, label, name, unit, column in (
                ("min_voltage", "u", "|V| floor", "Potential |V|", "V", "voltage"),
                ("min_current", "i", "|I| floor", "Current |I|", "A", "current")):
            if qc[key] > 0 and data.haveData(token):
                cut(np.abs(np.asarray(data[token], dtype=float)) >= qc[key], label,
                    name, f">= {qc[key]:g} {unit}")
            else:
                skipped(name, "off" if qc[key] <= 0 else f">= {qc[key]:g} {unit}",
                        "" if qc[key] <= 0 else f"not applied: no {column} column")
        if qc["max_k"] > 0 and data.haveData("k"):
            cut(np.abs(np.asarray(data["k"], dtype=float)) <= qc["max_k"], "|k| ceiling",
                "Geometric factor |k|", f"<= {qc['max_k']:g} m")
        else:
            skipped("Geometric factor |k|", "off" if qc["max_k"] <= 0 else
                    f"<= {qc['max_k']:g} m",
                    "" if qc["max_k"] <= 0 else "not applied: no geometric factors")
        if qc["max_contact_r"] > 0 and data.haveData("rc"):
            cut(np.asarray(data["rc"], dtype=float) <= qc["max_contact_r"],
                "contact R ceiling", "Contact resistance", f"<= {qc['max_contact_r']:g} ohm")
        else:
            skipped("Contact resistance", "off" if qc["max_contact_r"] <= 0 else
                    f"<= {qc['max_contact_r']:g} ohm",
                    "" if qc["max_contact_r"] <= 0 else
                    "not applied: no contact resistance column")
        # .get: QC settings saved before this check existed do not name it.
        stack = qc.get("max_stack", 0.0)
        if stack > 0 and data.haveData("stack"):
            cut(np.asarray(data["stack"], dtype=float) <= qc["max_stack"] / 100.0,
                "stacking spread ceiling", "Stacking spread", f"<= {stack:g} %")
        else:
            skipped("Stacking spread", "off" if stack <= 0 else f"<= {stack:g} %",
                    "" if stack <= 0 else "not applied: no stacking spread column")
        if qc["max_reciprocal"] > 0:
            rec = cls._reciprocal_error(data) if reciprocal is None else reciprocal
            threshold = f"<= {qc['max_reciprocal']:g} %"
            if rec is not None:
                limit = qc["max_reciprocal"] / 100.0
                # An unpaired measurement has no reciprocal to disagree with, so it
                # is kept rather than judged against a test it cannot take.
                cut(~(np.isfinite(rec) & (rec > limit)), "reciprocal error",
                    "Reciprocal error", threshold + " (unpaired kept)")
            else:
                skipped("Reciprocal error", threshold,
                        "not applied: no resistances to pair")
        else:
            skipped("Reciprocal error", "off", "")
        return reasons

    def _apply_filter(self) -> None:
        if self._ert_data_full is None:
            self.log("Load ERT data first.", "warn")
            return
        try:
            import pygimli as pg
            # Folding "More checks" away also switches its criteria off, so what
            # the panel shows is what the filter did.
            qc = self._qc_settings()
            report: Dict[str, Any] = {}
            data, keep, reasons = self._qc_survey(
                pg.DataContainerERT(self._ert_data_full), qc, report)
            if self._pair_info is not None:
                report["files"] = self._pair_record()
            if reasons:
                self.log("QC: " + "; ".join(reasons), "info")
            removed = int((~keep).sum())
            self._qc_mask = keep.astype(bool).tolist()
        except Exception as exc:  # noqa: BLE001
            self.log(f"Filter failed: {exc}", "error")
            return
        self._ert_data = data
        self._qc_applied = qc
        # What this filter did to the data now loaded, for the next run's QC
        # report; a new load or Reset leaves the data unfiltered and drops it.
        self._qc_report = report
        self._pseudo = self._pseudo_from_data(data)
        self._n_meas = int(data.size())
        self._draw_pseudosection()
        self._refresh()
        self._recip_inputs_changed()
        # On the Reciprocal errors tab the filter's effect on the pairs is what
        # is being looked at; elsewhere the section shows what it left.
        if self._tabs.currentWidget() is not self._recip_view:
            self._tabs.setCurrentWidget(self._pseudo_widget)
        self.log(f"Filter applied: kept {data.size()}, removed {removed}.", "success")

    def _pair_record(self) -> Dict[str, Any]:
        """The loaded survey's two files and their readings, for its QC log."""
        info = dict(self._pair_info or {})
        return {"forward": self._data_path.name if self._data_path else "",
                "reciprocal": self._data_partner.name if self._data_partner else None,
                "forward_readings": int(info.get("forward", self._n_meas) or 0),
                "reciprocal_readings": int(info.get("reciprocal", 0) or 0),
                "dropped_fields": list(info.get("dropped_fields") or [])}

    def _reset_filter(self) -> None:
        self._qc_applied = None
        self._qc_report = None
        self._recip_inputs_changed()
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

    # -- the Reciprocal errors tab --------------------------------------------
    def _error_fit_to(self) -> str:
        """What the reciprocal error model is fitted to: the tab's "Fit to"."""
        return self._recip_view.fit_to()

    def _on_error_fit_changed(self, fit_to: str) -> None:
        from PyHydroGeophysX.qt_apps.ert_records import FIT_TO_TEXT

        self.log(f"Reciprocal error model: fitted to {FIT_TO_TEXT[fit_to]} - for data errors "
                 "taken from it, the QC report and the run's figure.", "info")

    def _recip_inputs_changed(self, *_args: Any) -> None:
        """Something the Reciprocal errors tab draws changed: the data, the file
        list or its pairing, the filter. Redrawn a moment later while the tab is
        on screen, so a burst of changes reads a series once; off screen when
        the tab is next opened."""
        timer = getattr(self, "_recip_timer", None)
        if timer is not None and self._tabs.currentWidget() is self._recip_view:
            timer.start()

    def _recip_series_mode(self) -> bool:
        """Whether the tab shows the time-lapse series rather than one survey."""
        box = getattr(self, "_tl_mode", None)
        return bool(box is not None and box.isChecked() and len(self._tl_files) >= 2)

    def _single_pairing(self) -> Optional[Dict[str, Any]]:
        """The loaded survey's reciprocal pairing: with each pair marked kept or
        not when Apply filter filtered these data, otherwise as read."""
        if self._qc_report is not None and self._qc_applied is not None:
            pairing = self._qc_report.get("pairing")
            if pairing:
                return pairing
        if self._loaded_pairing is None and self._ert_data_full is not None:
            from PyHydroGeophysX.qt_apps.ert_records import reciprocal_pairing

            self._loaded_pairing = reciprocal_pairing(
                self._reciprocal_scores(self._ert_data_full))
        return self._loaded_pairing

    def _sync_recip_use(self, *_args: Any) -> None:
        """Tell the Reciprocal errors tab whether the next run uses its model."""
        view = getattr(self, "_recip_view", None)
        if view is not None:
            view.set_use_state(self._err_source.currentData() == "reciprocal")

    def _use_reciprocal_errors(self) -> None:
        """The tab's "Use for the data errors": the source under Data errors."""
        index = self._err_source.findData("reciprocal")
        if index >= 0:
            self._err_source.setCurrentIndex(index)
            self.log("Data errors now come from the reciprocal error model.", "info")

    def _refresh_recip_view(self) -> None:
        """Draw the Reciprocal errors tab for the data on the page: the survey
        on screen, or, in time-lapse mode, every survey of the list."""
        if not hasattr(self, "_tl_mode"):
            return                                   # still being built
        self._recip_timer.stop()
        if self._recip_series_mode():
            self._show_series_pairs()
            return
        from PyHydroGeophysX.qt_apps.ert_records import error_model_pairs

        view = self._recip_view
        if self._ert_data_full is None:
            view.show_message(
                _RECIP_NO_DATA if self._data_path is None else
                f"{self._data_path.name} could not be converted for inversion, so its "
                "readings cannot be paired with their reciprocals.")
            return
        pairing = self._single_pairing() or {}
        pairs = error_model_pairs([pairing])
        if not pairs["R"].size:
            name = self._data_path.name if self._data_path is not None else "These data"
            view.show_message(
                f"Every reciprocal pair of {name} agrees exactly, so there is no error to "
                "plot." if pairing.get("scored_pairs") else
                f"{name} has no reciprocal pairs: no reading was measured a second time "
                "with its current and potential electrodes exchanged, so there is nothing "
                "to plot.\n\n" + _RECIP_HOW)
            return
        view.show_pairs(pairs, series=False)

    def _recip_survey_key(self, source: str, partner: Optional[str]) -> tuple:
        """What a survey's pairs depend on: its files as they are on disk, the
        reader and the electrode file."""
        def stamp(path: Optional[str]) -> Optional[tuple]:
            if not path:
                return None
            try:
                status = os.stat(path)
            except OSError:
                return (str(path), None, None)
            return (str(path), status.st_mtime_ns, status.st_size)

        return (stamp(source), stamp(partner), str(self._instrument.currentData()),
                stamp(self._electrode_file()))

    @staticmethod
    def _recip_qc_key(qc: Optional[Dict[str, Any]]) -> str:
        """The filter a survey was read through, as a cache key ("" for none)."""
        return json.dumps(qc, sort_keys=True, default=str) if qc else ""

    def _series_recip_jobs(self) -> Tuple[List[Tuple[str, Optional[str], tuple]],
                                          Optional[Dict[str, Any]], str]:
        """``(jobs, qc, qc_key)``: each survey of the list with its reciprocal
        file and cache key, and the filter a time-lapse run puts them through -
        Apply filter's, as the run applies it to every survey."""
        jobs = []
        for source in self._tl_files:
            partner = self._tl_partner.get(source)
            jobs.append((source, partner, self._recip_survey_key(source, partner)))
        qc = dict(self._qc_applied) if self._qc_applied is not None else None
        return jobs, qc, self._recip_qc_key(qc)

    def _show_series_pairs(self) -> None:
        """Draw every survey's pairs, reading the surveys not read yet first."""
        from PyHydroGeophysX.qt_apps.ert_records import error_model_pairs

        jobs, qc, qc_key = self._series_recip_jobs()
        reports = [self._recip_cache.get(key, {}).get(qc_key) for _s, _p, key in jobs]
        missing = [job for job, report in zip(jobs, reports) if report is None]
        if missing:
            self._start_recip_read(missing, qc, qc_key, len(jobs))
            return
        failed = [Path(source).name for (source, _p, _k), report in zip(jobs, reports)
                  if report.get("error")]
        unread = ""
        if failed:
            unread = (f"{len(failed)} survey(s) could not be read and are not drawn: "
                      + ", ".join(failed[:3]) + (", …" if len(failed) > 3 else "") + ".")
        pairs = error_model_pairs([report.get("pairing") or {} for report in reports])
        if not pairs["R"].size:
            self._recip_view.show_message(
                f"None of the {len(jobs)} surveys has reciprocal pairs: no reading was "
                "measured a second time with its current and potential electrodes "
                "exchanged, so there is nothing to plot.\n\n" + _RECIP_HOW
                + (f"\n\n{unread}" if unread else ""))
            return
        self._recip_view.show_pairs(pairs, series=True,
                                    note=unread)

    def _start_recip_read(self, missing: List[Tuple[str, Optional[str], tuple]],
                          qc: Optional[Dict[str, Any]], qc_key: str, total: int) -> None:
        """Read the surveys of ``missing`` off the UI thread, with progress."""
        reading = (tuple(key for _s, _p, key in missing), qc_key)
        worker = self._recip_worker
        if worker is not None and worker.isRunning():
            if self._recip_reading == reading:
                return                                # already reading these
            worker.cancel()                           # what it read is kept
        self._recip_view.show_message(self._recip_reading_text(0, len(missing), total))
        worker = TaskWorker(self._read_recip_surveys, list(missing),
                            self._instrument.currentData(), self._electrode_file(), qc,
                            qc_key, self._recip_cache, with_log=True)
        worker.logged.connect(
            lambda message, w=worker, n=total: self._on_recip_read_logged(w, message, n))
        worker.failed.connect(lambda message: self.log(
            f"Reading the surveys' reciprocal pairs stopped: {message}", "warn"))
        worker.finished.connect(lambda w=worker: self._on_recip_read_done(w))
        self._recip_worker = self.register_worker(
            worker, activity="Reading the surveys' reciprocal pairs")
        self._recip_reading = reading
        worker.start()

    @staticmethod
    def _recip_reading_text(done: int, count: int, total: int) -> str:
        before = f" ({total - count} read before)" if count < total else ""
        return (f"Reading the reciprocal pairs of {count} surveys{before}: {done} of {count} "
                "done.\n\nA long series takes a few minutes; the page can be used "
                "meanwhile.")

    def _on_recip_read_logged(self, worker: Any, message: str, total: int) -> None:
        match = re.match(r"^(\d+)/(\d+) ", str(message))
        if match is None:
            return
        done, count = int(match.group(1)), int(match.group(2))
        self._on_worker_progress(worker, done, count, "Reading the surveys' reciprocal pairs")
        if worker is self._recip_worker and not self._recip_view.has_plot():
            self._recip_view.show_message(self._recip_reading_text(done, count, total))

    def _on_recip_read_done(self, worker: Any) -> None:
        if worker is not self._recip_worker:
            return                                    # replaced by a newer read
        self._recip_worker = None
        self._recip_reading = None
        if not worker.is_cancelled() and self._tabs.currentWidget() is self._recip_view:
            self._refresh_recip_view()

    @classmethod
    def _read_recip_surveys(cls, jobs: List[Tuple[str, Optional[str], tuple]],
                            instrument: Optional[str], electrode_file: Optional[str],
                            qc: Optional[Dict[str, Any]], qc_key: str,
                            cache: Dict[tuple, Dict[str, Dict[str, Any]]], log=None) -> int:
        """Read each survey of ``jobs`` as the time-lapse run prepares it
        (:meth:`_prepare_survey`) and keep its QC report in ``cache``; runs off
        the UI thread. Each survey is kept as soon as it is read, so a read
        stopped half way keeps that half; one that cannot be read is kept as
        its error, so it is named rather than read again and again."""
        log = log or (lambda _message: None)
        for done, (source, partner, key) in enumerate(jobs, start=1):
            try:
                _data, report, _reasons = cls._prepare_survey(
                    source, partner, instrument, electrode_file, qc)
                entry = cls._recip_entry(report)
            except Exception as exc:  # noqa: BLE001 - one unreadable survey is named
                entry = {"error": str(exc), "readings": 0, "kept": 0, "pairing": {}}
            cls._cache_recip(cache, key, qc_key, entry)
            log(f"{done}/{len(jobs)} {Path(source).name}")
        return len(jobs)

    @staticmethod
    def _recip_entry(report: Dict[str, Any]) -> Dict[str, Any]:
        """The part of a survey's QC report the tab and the run's records use."""
        return {key: report[key] for key in ("readings", "kept", "checks", "pairing",
                                             "files", "averaged") if key in report}

    @staticmethod
    def _cache_recip(cache: Dict[tuple, Dict[str, Dict[str, Any]]], key: tuple,
                     qc_key: str, entry: Dict[str, Any]) -> None:
        """Keep a survey's report under its filter; two filters per survey at most."""
        slot = cache.setdefault(key, {})
        slot.pop(qc_key, None)
        slot[qc_key] = entry
        while len(slot) > 2:
            slot.pop(next(iter(slot)))

    @staticmethod
    def _read_electrodes(path: str) -> Tuple[List[float], List[float], str]:
        """``(x, z, how)`` from an electrode file, its columns chosen by meaning.

        ``table_io.read_electrode_table`` decides, the reader the data loaders
        use as well, and z is each electrode's elevation. Taking the first
        column as x and the last as z read an ID column as the elevation, the
        electrode number as the position, or - for a table written the PyGIMLi
        way, x, elevation, 0 - a flat line at zero.
        """
        coords, elevation, how = table_io.read_electrode_table(path)
        return coords[:, 0].tolist(), elevation.tolist(), how

    def _electrode_table_for(self, path: str, x: List[float], z: List[float]) -> Optional[Path]:
        """A plain x y z copy of ``path`` for the readers, or None if none can be written.

        The instrument readers take the electrode file as positional x y z
        columns, so the vendor table itself would be misread there the way it
        was here. They get the columns as chosen above instead. Each file gets a
        copy of its own, so the one in use survives a file that is refused.
        """
        try:
            table = (self.state.ensure_results_store().scratch_dir(self.module_key)
                     / f"electrodes_xyz_{uuid.uuid4().hex[:8]}.txt")
            np.savetxt(table, np.column_stack([x, np.zeros(len(x)), z]))
            return table
        except Exception as exc:  # noqa: BLE001
            self.log(f"Later loads cannot use {Path(path).name}: {exc}", "warn")
            return None

    def _take_electrode_file(self, path: str, *, wait: bool = False) -> Dict[str, Any]:
        """Make ``path`` the electrode positions, through the reader the data use.

        Before any data is loaded, the positions are shown and every later load
        reads with them. With data on screen, the data are read again with the
        file, so both orders end in one result: the reader matches rows to the
        data's electrodes in order, refuses a file listing a different number of
        them, and carries apparent resistivities formed on the header's positions
        over to the file's. Laid over loaded data instead, a file one row short
        dropped that electrode and every reading on it, and the apparent
        resistivities stayed those of the old positions.

        ``wait`` holds the call until the data have been read again (the
        assistant needs the outcome to answer); otherwise that happens in the
        background and is reported in the log.
        """
        name = Path(path).name
        try:
            x, z, how = self._read_electrodes(path)
        except Exception as exc:  # noqa: BLE001
            return {"status": "failed", "error": f"Could not load electrodes: {exc}"}
        table = self._electrode_table_for(path, x, z)
        if self._ert_data is None or self._data_path is None:
            self._x = [float(v) for v in x]
            self._z = [float(v) for v in z]
            self._labels = [str(i + 1) for i in range(len(self._x))]
            self._electrode_origins = [None] * len(self._x)
            self._selected = None
            self._electrode_table = table
            self._electrode_path = Path(path)
            self._refresh()
            self.log(f"Loaded {len(self._x)} electrodes from {name} ({how}); data "
                     "loaded from now on are placed on them.", "success")
            return {"status": "ok", "electrodes": len(self._x), "columns": how}
        if table is None:
            return {"status": "failed",
                    "error": f"{name} could not be staged for the reader."}
        data_name = self._data_path.name
        previous = (self._electrode_table, self._electrode_path)
        refilter = self._qc_applied is not None
        kept_before = self._n_meas
        self._electrode_table, self._electrode_path = table, Path(path)
        self.log(f"Reading {data_name} again with the electrode positions from "
                 f"{name} ({how})…", "info")

        def settle(outcome: str, value: Any) -> Dict[str, Any]:
            if outcome == "loaded":
                refiltered = ""
                if refilter:
                    # The thresholds already applied. Apparent resistivity moves
                    # with the electrodes, so the same ρa limits can keep a
                    # different share of the readings; both counts are given.
                    self._apply_filter()
                    total = int(self._ert_data_full.size()) if self._ert_data_full is not None else 0
                    refiltered = (f"; the QC filter applied again keeps {self._n_meas} of "
                                  f"{total} readings ({kept_before} on the previous positions)")
                self.log(f"{data_name}: electrodes placed from {name}{refiltered}.",
                         "success")
                return {"status": "ok", "electrodes": len(self._x), "columns": how,
                        "measurements": self._n_meas}
            if outcome == "superseded":
                # A newer load took over, and it read with this file.
                return {"status": "failed", "error": "superseded by another load"}
            error = self._name_electrode_file(str(value))
            self._electrode_table, self._electrode_path = previous
            self._refresh()                    # the data on screen are the earlier read
            self.log(f"{name} was not applied; {data_name} stays as it was read "
                     "before.", "warn")
            return {"status": "failed", "error": error}

        if wait:
            return settle(*self._load_and_wait(str(self._data_path)))
        self._start_load(str(self._data_path), settle)
        return {"status": "pending"}

    def _load_electrodes(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load electrode file", "", _ELEC_FILTER)
        if not path:
            return
        outcome = self._take_electrode_file(path)
        if outcome["status"] == "failed":
            self.log(outcome["error"], "error")

    # -- inversion -----------------------------------------------------------
    def _report_data_health(self) -> None:
        """Report what the data say about themselves, before Full mode inverts them.

        This reports and never drops. Rejecting data is the QC panel's job, where
        the thresholds are visible and the count changes in front of you; a run
        that quietly shrank its own dataset is the harder thing to review later.
        What it adds is the checks whose inputs the inversion never looks at:
        reciprocity, and whether the readings had any signal behind them.
        """
        data = self._ert_data
        if data is None:
            return
        n = int(data.size())
        notes: List[str] = []

        rec = self._reciprocal_error(data)
        if rec is not None and np.isfinite(rec).any():
            finite = rec[np.isfinite(rec)]
            median = 100.0 * float(np.median(finite))
            over5 = int((finite > 0.05).sum())
            notes.append(
                f"reciprocals: {finite.size}/{n} paired, median disagreement {median:.1f}%"
                + (f", {over5} above 5%" if over5 else ""))
            if median > 5.0:
                notes.append(
                    "the median reciprocal disagreement is above 5%, so the assumed error "
                    "model is probably optimistic and chi2 will read low")
        else:
            notes.append(f"reciprocals: none in this survey, so the {n} errors are assumed, not measured")

        for token, name in (("u", "voltage"), ("i", "current")):
            if not data.haveData(token):
                continue
            v = np.abs(np.asarray(data[token], dtype=float))
            weak = int((~np.isfinite(v) | (v <= 0)).sum())
            if weak:
                notes.append(f"{name}: {weak} reading(s) at or below zero, which carry no signal")
            else:
                notes.append(f"{name}: all finite and positive, spanning {v.min():.4g} to {v.max():.4g}")

        err = np.asarray(data["err"], dtype=float) if data.haveData("err") else None
        if err is not None and err.size:
            notes.append(f"stated error: median {100.0 * float(np.median(err)):.1f}%")

        for note in notes:
            self.log("Full-mode check · " + note, "info")

    def _run_inversion(self) -> None:
        if self._ert_data is None:
            self.log("Load ERT data with apparent resistivity first.", "warn")
            return
        if (
            self._engine.currentData() == "adtlert"
            and self._adtlert_single_ready is not True
        ):
            self.log(
                "ADTLERT single self-test has not passed. Wait for the check, "
                "or select In-house Gauss-Newton / PyHydro. "
                "To retry, switch engines and select ADTLERT again.",
                "warn",
            )
            return
        self._warn_external_not_running("run folder")
        if str(self._inv_mode.currentData() or "quick") == "full":
            self._report_data_health()
        error_source, error_model = self._single_error_model()
        try:
            run = self.begin_persisted_run(
                "ert.single_inversion", "ert.single_inversion",
                label=self._run_label([self._data_path]),
            )
        except Exception as exc:  # noqa: BLE001
            self.log(f"Could not prepare Project run: {exc}", "error")
            return
        out_path = run.outputs_dir
        input_path = run.inputs_dir / "filtered_ert_data.dat"
        electrode_rows = self._electrode_rows()
        electrode_path = run.inputs_dir / "edited_electrodes.csv"
        qc_path = run.inputs_dir / "ert_qc_mask.json"
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
                # "keep" is over the forward and reciprocal readings together,
                # and an averaged pair is kept as its first reading.
                "reciprocal_file": self._data_partner.name if self._data_partner else "",
                "averaged_pairs": int(((self._qc_report or {}).get("averaged") or {})
                                      .get("pairs", 0)) if self._averaged_loaded() else 0,
            },
        )
        try:
            save_edited_ert_container(self._ert_data, input_path, electrode_rows)
        except Exception as exc:  # noqa: BLE001
            self.log(f"Could not serialize edited/QC-filtered ERT data: {exc}", "error")
            self.fail_persisted_run(str(exc), "ert.single_inversion")
            return
        project_root = run.run_dir
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
                        **({"reciprocal_file": self._data_partner.name}
                           if self._data_partner is not None else {}),
                        **self._survey_time_record(self._data_path),
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
                "max_iterations": min(int(self._iter.value()),
                                      int(self._iter_ceiling.value())),
                "relative_error": float(self._relerr.value()),
                **self._mesh_options(),
                "instrument": "BERT",
                "engine": str(self._engine.currentData()),
                "geometric_factor_policy": str(self._geom_policy),
                "error_source": error_source,
                **({"error_model": error_model} if error_model else {}),
                # The lowest error a reciprocal gives, for every reading, when
                # the errors come from reciprocals: the model's, or the one the
                # pairs were averaged with.
                **({"error_floor": float(error_model["floor"])} if error_model else
                   {"error_floor": float(self._qc_applied["reciprocal_floor"])}
                   if self._averaged_loaded() else {}),
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
                **self._zone_parameters(),
                **({"e4d": self._e4d_settings()}
                   if self._engine.currentData() == "e4d" else {}),
                **({"r2": self._r2_settings()}
                   if self._r2_program_name() is not None else {}),
            },
            metadata={"source_instrument": self._instrument.currentText()},
        )
        recipe_path, script_path = export_workflow_bundle(spec, run.run_dir, stem="ert")
        self._reproduce.set_bundle(recipe_path, script_path)
        self._ert_recipe_path = str(recipe_path)
        self._write_single_records(run, spec)
        self._inv_busy = BusyStateController([self._invert_btn])
        self._inv_busy.start()
        self._invert_btn.setText("Inverting…")
        self._inv_progress.setVisible(True)
        self._inv_progress.setRange(0, 0)
        # Every ERT engine performs long native numerical work.  Run all of
        # them outside Qt's interpreter: PyGIMLi and the in-house Gauss-Newton
        # path can retain the GIL, while ADTLERT owns a CUDA context.  The
        # process-safe model bundle below restores mesh/model/response/coverage
        # into the interactive viewer when the child exits.
        self._agent_run_state = "running"
        self._agent_run_error = ""
        self._inv_worker = ProcessWorkflowWorker(
            recipe_path,
            project_root,
            out_path,
            run.result_path,
        )
        self._inv_worker.logged.connect(lambda msg: self.log(msg, "info"))
        # The survey goes with its result: a preview loaded while this runs
        # replaces _data_path, and would date the model by the wrong file.
        self._inv_worker.succeeded.connect(
            lambda result, source=self._data_path: self._on_ert_workflow_ok(result, source))
        self._inv_worker.failed.connect(self._on_inversion_failed)
        self._inv_worker.finished.connect(self._reset_invert_button)
        self.register_worker(self._inv_worker, activity="Running the ERT inversion")
        self._inv_worker.start()
        self._inv_stop.attach(self._inv_worker, "ert.single_inversion")

    def _averaged_loaded(self) -> bool:
        """Whether the data on screen had their reciprocal pairs averaged."""
        return bool(self._qc_report is not None and self._qc_applied is not None
                    and self._qc_applied.get("average_reciprocals")
                    and (self._qc_report.get("averaged") or {}).get("pairs"))

    def _single_error_model(self) -> Tuple[str, Optional[Dict[str, Any]]]:
        """``(error_source, error_model)`` for a single inversion.

        "From the reciprocal error model" fits the survey's own pairs - all of
        them as read, or those the filter kept, as the Reciprocal errors tab's
        "Fit to" says: the fit the tab draws and the QC report states - and
        hands the model to the inversion, which sets each reading's error from
        it. With too few pairs to fit, the estimate is used and the log says so.
        """
        source = str(self._err_source.currentData())
        if source != "reciprocal":
            return source, None
        from PyHydroGeophysX.qt_apps.ert_records import (
            fit_series_error_model,
            fit_to_sentence,
            power_law,
        )

        fit_to = self._error_fit_to()
        pairing = self._single_pairing() or {}
        fit = fit_series_error_model([pairing], fit_to)
        if fit is None:
            self.log(f"Data errors: the survey has {pairing.get('scored_pairs', 0)} scored "
                     "reciprocal pairs, too few to fit an error model (at least 15 are "
                     "needed), so the errors are estimated from the values under Data "
                     "errors instead.", "warn")
            return "estimate", None
        model = {"m": fit["m"], "b": fit["b"], "r2": fit["r2"], "r2_raw": fit["r2_raw"],
                 "pairs": fit["n"], "floor": self._reciprocal_floor(),
                 "fitted_over": self._fitted_over(fit, series=False)}
        self.log(f"Data errors from the reciprocal error model {power_law(model)}, fitted to "
                 f"{fit_to_sentence(fit_to, fit['filtered'])} (binned R² {fit['r2']:.3f}, "
                 f"{fit['n']} pairs), at least {100.0 * model['floor']:g} %.", "info")
        return source, model

    def _error_model_text(self, timelapse: bool = False,
                          model: Optional[Dict[str, Any]] = None) -> str:
        """The data errors a run is set to use, in a sentence for its records."""
        estimate = (f"{float(self._relerr.value()):g} + {float(self._abserr.value()):g} "
                    "ohm / |R|")
        if model:
            from PyHydroGeophysX.qt_apps.ert_records import power_law

            return (f"the reciprocal error model {power_law(model)}, never below "
                    f"{100.0 * float(model.get('floor') or 0.0):g} %")
        if timelapse:
            return (f"each survey's err column where every reading has one, otherwise "
                    f"relative error {float(self._relerr.value()):g}")
        return f"{self._err_source.currentText()} (estimate {estimate})"

    def _write_single_records(self, run, spec: WorkflowSpec) -> None:
        """Write the run's ``inversion_settings.txt`` and, for filtered data, its
        ``qc_report.txt`` (``qt_apps/ert_records.py``), from the spec the run is
        handed and the counts Apply filter recorded. A file that cannot be
        written is said in the log; the run goes ahead, its recipe being the
        record it needs to rerun."""
        from PyHydroGeophysX.qt_apps import ert_records

        self._ert_settings_path = None
        try:
            source = str(self._data_path or "")
            stamp = survey_timing.survey_timing([source]).timestamps[0] if source else None
            acquired = f"{stamp:%Y-%m-%d %H:%M:%S}" if stamp is not None else ""
            reader = self._reader_status.text().removeprefix("Data loaded by ").rstrip(".")
            inverted = int(self._ert_data.size())
            total = (int(self._ert_data_full.size()) if self._ert_data_full is not None
                     else inverted)
            # The thresholds count only when they filtered the data loaded now.
            filtered = self._qc_report is not None and self._qc_applied is not None
            pair = self._pair_record() if self._data_partner is not None else None
            model = dict(spec.parameters.get("error_model") or {}) or None
            # The pairs the error model is fitted to, said wherever there are pairs.
            fit_to = self._error_fit_to()
            paired = bool((self._single_pairing() or {}).get("scored_pairs"))
            path = ert_records.write_settings(run.run_dir, ert_records.single_settings_text(
                spec, run_id=run.run_id, source=source, acquired=acquired,
                instrument=self._instrument.currentText(), reader=reader,
                electrode_file=str(self._electrode_path or ""), electrodes=len(self._x),
                measurements=(total, inverted),
                qc=self._qc_applied if filtered else None,
                mode=self._inv_mode.currentText(),
                reciprocal=({"file": pair["reciprocal"],
                             "forward_readings": pair["forward_readings"],
                             "reciprocal_readings": pair["reciprocal_readings"]}
                            if pair else None),
                fit_to=fit_to if paired or model else None))
            self._ert_settings_path = path
            self.log(f"Inversion settings written to {path}", "info")
            # A survey of two files, or errors from its pairs, is worth its QC
            # report - the pairing and the model - even when nothing filtered it.
            if filtered or pair or model:
                if filtered:
                    qc_log = dict(self._qc_report)
                else:
                    qc_log = {"readings": total, "kept": inverted, "checks": [],
                              "pairing": self._single_pairing() or {}}
                if pair:
                    qc_log["files"] = pair
                report = ert_records.write_single_qc_report(
                    run.run_dir, source=Path(source).name, acquired=acquired,
                    reader=f"{self._instrument.currentText()}, read by {reader}",
                    qc=self._qc_applied if filtered else None, report=qc_log,
                    error_model=self._error_model_text(model=model),
                    applied_model=model, fit_to=fit_to)
                self.log(f"Data QC report written to {report}", "info")
        except Exception as exc:  # noqa: BLE001 - a record must not stop the run
            self.log(f"Could not write the run's settings file: {exc}", "warn")

    def _record_outcome(self, path: Optional[Path], kind: str, summary: Dict[str, Any],
                        extra: Dict[str, Any]) -> None:
        """Add what the finished run reported to its ``inversion_settings.txt``:
        the engine that ran, the fit, and for a time-lapse run any setting the
        engine replaced. Warnings in it are logged, so they stay with the run."""
        from PyHydroGeophysX.qt_apps import ert_records

        if path is None:
            return
        try:
            if kind == "single":
                rows, notes = ert_records.single_outcome(dict(summary), dict(extra))
            else:
                rows, notes = ert_records.timelapse_outcome(dict(summary), dict(extra))
                zones = list(summary.get("zones_not_applied") or [])
                if zones:
                    self.log("The engine did not take the a-priori zone values ("
                             + ", ".join(map(str, zones)) + "); the run used none.", "warn")
            ert_records.append_outcome(Path(path), rows, notes)
            for note in notes[1:]:
                self.log(f"The engine ran with {note.strip()} (its own default).", "info")
        except Exception as exc:  # noqa: BLE001 - a record must not lose the result
            self.log(f"Could not add the outcome to {Path(path).name}: {exc}", "warn")

    def _abs_path(self, value: Any) -> str:
        """Resolve a workflow-relative output path against the recipe directory."""
        if not value:
            return ""
        path = Path(str(value))
        if path.is_absolute() or not self._ert_recipe_path:
            return str(path)
        return str(Path(self._ert_recipe_path).resolve().parent / path)

    def _on_ert_workflow_ok(self, result: WorkflowRunResult,
                            source: Optional[Path] = None) -> None:
        self._record_outcome(self._ert_settings_path, "single", result.summary,
                             result.metrics)
        summary = dict(result.summary)
        manager = result.objects.get("manager")
        fixed_manager = result.objects.get("fixed_manager")
        if manager is None and summary.get("model_bundle"):
            manager = self._load_model_bundle(summary["model_bundle"])
        if fixed_manager is None and summary.get("fixed_model_bundle"):
            fixed_manager = self._load_model_bundle(
                summary["fixed_model_bundle"]
            )
        vtk = self._abs_path(summary.get("vtk"))
        if not vtk:
            vtk_ref = next(
                (artifact for artifact in result.artifacts if "vtk" in artifact.format),
                None,
            )
            vtk = self._abs_path(vtk_ref.path) if vtk_ref is not None else ""
        payload = {
            "mgr": manager,
            "chi2": result.metrics.get("chi2", float("nan")),
            "vtk": vtk,
            "metrics": dict(result.metrics),
            "convergence": (
                result.objects.get("convergence")
                or summary.get("convergence")
                or []
            ),
            "fixed_mgr": fixed_manager,
            "fixed_convergence": (
                result.objects.get("fixed_convergence")
                or summary.get("fixed_convergence")
                or []
            ),
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
            "engine_requested": summary.get("engine_requested", ""),
            "e4d": dict(summary.get("e4d") or {}),
            "r2": dict(summary.get("r2") or {}),
            "r3t": dict(summary.get("r3t") or {}),
            "data_path": source,
        }
        try:
            self._on_inversion_ok(payload)
        finally:
            if hasattr(self.state, "update_workflow_result"):
                self.state.update_workflow_result(
                    self.module_key,
                    "ert.single_inversion",
                    result.to_dict(),
                    recipe_path=self._ert_recipe_path,
                )

    def _load_model_bundle(self, bundle: Dict[str, Any]):
        """Hydrate a process-safe ERT result into the viewer's manager shape."""
        try:
            from PyHydroGeophysX.core.mesh_serialization import read_bms
            # numpy only; the inversion code need not load in the window's process.
            from PyHydroGeophysX.inversion.model_result import ModelResult

            paths = {key: Path(str(value)) for key, value in dict(bundle).items()}
            # Without the cell-neighbour table, most of a 3-D mesh's load, spent
            # in one call that would hold the window still (see read_bms).
            mesh = read_bms(paths["mesh"], neighbours=False)
            model = np.load(paths["model"], allow_pickle=False)
            response = (
                np.load(paths["response"], allow_pickle=False)
                if "response" in paths else None
            )
            coverage = (
                np.load(paths["coverage"], allow_pickle=False)
                if "coverage" in paths else None
            )
            return ModelResult(mesh, model, response=response, coverage=coverage)
        except Exception as exc:  # noqa: BLE001 - keep metrics usable on load failure
            self.log(f"Could not load inversion model files: {exc}", "error")
            return None

    def _on_inversion_ok(self, result: dict) -> None:
        self._agent_run_state = "completed"
        self._agent_run_error = ""
        metrics = dict(result.get("metrics") or {})
        engine = str(result.get("engine") or "")
        requested_engine = str(result.get("engine_requested") or engine)
        solver = str(metrics.get("linearized_solver") or "")
        if engine in ("r2", "r3t"):
            # They choose their own smoothing weight (shown as alpha); the λ the
            # page holds was never used, so it is not shown as if it were.
            metrics.pop("lambda", None)
        if engine:
            metrics.setdefault(
                "method", f"{engine} · {self._instrument.currentText()}"
            )
        else:
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
        if engine == "adtlert":
            extra["compute"] = "cuDSS forward · GPU CGLS"
        elif engine == "e4d":
            elements = int(dict(result.get("e4d") or {}).get("elements") or 0)
            extra["compute"] = "E4D 3D" + (f" · {elements} elements" if elements else "")
        elif engine in ("r2", "r3t"):
            own = dict(result.get(engine) or {})
            name = "R2" if engine == "r2" else "R3t"
            elements = int(own.get("elements") or 0)
            alpha = own.get("alpha")
            extra["compute"] = f"{name} {'2D' if engine == 'r2' else '3D'}" + (
                f" · {elements} elements" if elements else "")
            if alpha is not None and alpha == alpha:
                # Its own smoothing weight stands where λ would.
                extra["alpha"] = f"{float(alpha):.4g} (chosen by {name}; λ not used)"
        if dropped:
            extra["data"] = f"{outliers.get('kept')} of {outliers.get('n_start')} kept"
            if outliers.get("limited_by_floor"):
                extra["data"] += " (floor reached)"
        # A skipped k check has no other symptom. Wrong geometric factors scale the
        # section by a constant and leave chi2 untouched, so if the panel stays
        # silent here a Quick result is indistinguishable from a validated one.
        # A pass says nothing worth a line; the other three states each do.
        # ``ok`` stays False after a successful repair, so ``repaired`` is what
        # separates a fixed run from one whose scale is still unverified.
        if not geometry.get("checked", False):
            extra["k"] = "not checked"
        elif geometry.get("repaired"):
            extra["k"] = "repaired"
        elif not geometry.get("ok", True):
            extra["k"] = "scale unverified"
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
            # A new result arrives as inverted; any earlier correction was of a
            # different model.
            self._inv_correction = None
            # Recorded with the result, because the correction dates the model
            # by it. A caller that does not say which file it inverted means
            # the one loaded now.
            source = result.get("data_path", self._data_path)
            self._inv_source = Path(source) if source else None
            self._show_lambda_choice(0)
            self._tabs.setCurrentWidget(self._model_tab)
            self._model_export_btn.setEnabled(True)
            days, dates = self._single_survey_times()
            self._tc_box.set_context(1, dates is not None, days=days, dates=dates)
            self._tc_box.set_available(True)
            self._tc_box.show_applied(None)
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
        if requested_engine == "adtlert" and engine != "adtlert":
            self.log(
                "ADTLERT was requested but the run used the original PyHydro "
                "ERT engine; CUDA/cuDSS was unavailable or the survey is not "
                "supported by ADTLERT.",
                "warn",
            )
        elif engine == "adtlert":
            self.log(
                f"Compute backend: ADTLERT · cuDSS forward · {solver or 'gpu_cgls'}.",
                "info",
            )
        vtk = result.get("vtk")
        if vtk:
            self.log(f"Saved {Path(vtk).name} to {Path(vtk).parent}", "info")
        e4d = dict(result.get("e4d") or {})
        if e4d.get("run_dir"):
            self.log(f"E4D's own files (e4d.log, sigma.N) and the 3D model "
                     f"resistivity_3d.vtk are in {e4d['run_dir']}", "info")
        for engine_key, name in (("r2", "R2"), ("r3t", "R3t")):
            own = dict(result.get(engine_key) or {})
            if own.get("run_dir"):
                self.log(f"{name}'s own files ({name}.out, f001_res.dat, its VTK) are in "
                         f"{own['run_dir']}; {name} stopped with "
                         f"\"{own.get('stop_message') or 'no message'}\"", "info")

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
                            "engine": engine,
                            "engine_requested": requested_engine,
                            "linearized_solver": solver,
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
        self.offer_map_export()

    def _show_lambda_choice(self, index: int) -> None:
        """Display one of the kept single-inversion models (auto-λ or fixed λ)."""
        if not (0 <= index < len(self._inv_choices)):
            return
        choice = self._inv_choices[index]
        self._map_result_kind = 'single'
        self._inv_mgr = choice["mgr"]
        if self._inv_mgr is not None:
            if self._inv_correction:
                # The correction stays on when the other model is picked: it is a
                # statement about the ground, not about either inversion.
                self._apply_single_temperature(self._inv_correction["spec"])
            else:
                self._show_single_model(None)
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

    # -- mesh settings ----------------------------------------------------------
    def _build_mesh_settings(self) -> QGroupBox:
        """The mesh the inversion runs on, set up on the Mesh tab that draws it.

        Every setting is one of PyGIMLi's parameter-mesh options under a name
        that says what it does; each tooltip gives PyGIMLi's name and the E4D
        setting that plays the same part. The view rebuilds a moment after a
        change, so the effect of each is seen at once, and a setting left at
        its default builds the mesh PyGIMLi builds by itself.
        """
        box = QGroupBox("Mesh")
        form = QFormLayout(box)

        # An imported mesh. Building one from the electrode line is fine for a
        # 2D profile and hopeless for a 3D domain with topography, boreholes or
        # known structure, which is meshed externally (usually in Gmsh).
        self._mesh_path = ""
        self._mesh_btn = QPushButton("Import mesh…")
        self._mesh_btn.setIcon(theme.icon("fa5s.project-diagram"))
        self._mesh_btn.setToolTip(
            "Inverts on an externally built mesh instead of one generated from "
            "the electrode positions. PyGIMLi .bms, Gmsh .msh, VTK, .poly, or an "
            "E4D mesh (.1.node / .1.ele, with its .trn). An E4D mesh configuration "
            "(.cfg) is meshed the way E4D does it, once, on import. "
            "The region to invert must carry marker 2 or above; marker 0 and 1 "
            "are treated as background and stay fixed - for an E4D mesh, zone 1 "
            "is the background. The file is checked against the survey before the "
            "run starts.")
        self._mesh_btn.clicked.connect(self._import_mesh)
        self._mesh_clear = QPushButton("✕")
        self._mesh_clear.setMaximumWidth(32)
        self._mesh_clear.setToolTip("Go back to a mesh generated from the data.")
        self._mesh_clear.setEnabled(False)
        self._mesh_clear.clicked.connect(self._clear_mesh)
        self._mesh_row = merged_row(self._mesh_btn, self._mesh_clear)
        form.addRow("Source", self._mesh_row)
        self._mesh_note = QLabel("Built from the electrode positions.")
        self._mesh_note.setWordWrap(True)
        theme.set_tone(self._mesh_note, "hint")
        form.addRow("", self._mesh_note)

        def section(title: str, text: str) -> None:
            form.addRow(QLabel(f"<b>{title}</b>"))
            hint = QLabel(text)
            hint.setWordWrap(True)
            theme.set_tone(hint, "hint")
            form.addRow(hint)

        section("Inverted region",
                "The cells the inversion solves for, under the electrodes "
                "(PyGIMLi's parameter domain, E4D's fine zone).")
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
            "the bottom of the section is empty. Auto is 0.4 times the electrode "
            "spread (PyGIMLi's paraDepth).")
        form.addRow("Depth", self._para_depth)
        self._para_boundary = QDoubleSpinBox()
        self._para_boundary.setRange(0.5, 20.0); self._para_boundary.setDecimals(1)
        self._para_boundary.setSingleStep(0.5); self._para_boundary.setValue(2.0)
        self._para_boundary.setSuffix(" spacings")
        self._para_boundary.setToolTip(
            "How far the inverted region reaches past the first and the last "
            "electrode, in electrode spacings. A wider margin lets the inversion "
            "put structure beside the line rather than forcing it under the end "
            "electrodes, at the cost of cells the data barely see. PyGIMLi's "
            "paraBoundary, 2 by default; the padding of E4D's fine zone.")
        form.addRow("Side margin", self._para_boundary)
        self._surface_nodes = QSpinBox()
        self._surface_nodes.setRange(1, 8); self._surface_nodes.setValue(1)
        self._surface_nodes.setSuffix(" between electrodes")
        self._surface_nodes.setToolTip(
            "Mesh nodes placed on the surface between two neighbouring electrodes. "
            "More give smaller cells near the surface, where the data resolve the "
            "most, and more unknowns. PyGIMLi's addNodes; 1, a node halfway "
            "between electrodes, by default.")
        form.addRow("Surface nodes", self._surface_nodes)
        self._para_max_cell = QDoubleSpinBox()
        self._para_max_cell.setRange(0.0, 1e6); self._para_max_cell.setDecimals(2)
        self._para_max_cell.setStepType(QAbstractSpinBox.StepType.AdaptiveDecimalStepType)
        self._para_max_cell.setSuffix(" m²")
        self._para_max_cell.setSpecialValueText("no limit")
        self._para_max_cell.setToolTip(
            "Upper limit on the area of an inverted cell. Without one the cells "
            "grow with depth; a limit keeps them small and even, with many more "
            "unknowns. The line under the plot gives the cell sizes of the mesh "
            "shown. PyGIMLi's paraMaxCellSize; the maximum volume of E4D's fine "
            "zone.")
        form.addRow("Largest cell", self._para_max_cell)
        self._quality = QDoubleSpinBox()
        self._quality.setRange(20.0, _MAX_MESH_QUALITY); self._quality.setDecimals(1)
        self._quality.setValue(34.0)
        self._quality.setSuffix("°")
        self._quality.setToolTip(
            "The smallest angle a triangle may have. Higher gives better-shaped "
            "triangles and more of them; Triangle cannot finish much above 34°, "
            "so that is the limit. PyGIMLi's quality; E4D sets the same for TetGen.")
        form.addRow("Quality", self._quality)

        section("Outer region",
                "Coarse cells that carry the boundary condition far from the "
                "electrodes; never inverted. Tick Outer region above the plot to "
                "see them.")
        self._outer_width = QDoubleSpinBox()
        self._outer_width.setRange(0.0, 50.0); self._outer_width.setDecimals(1)
        self._outer_width.setSingleStep(0.5); self._outer_width.setValue(0.0)
        self._outer_width.setSuffix(" × spread")
        self._outer_width.setSpecialValueText("auto (4 × spread)")
        self._outer_width.setToolTip(
            "How far the outer region reaches beyond the inverted region, sideways "
            "and down, in lengths of the electrode spread. Too narrow and the "
            "boundary distorts the forward response; wider costs little, because "
            "its cells are large. PyGIMLi's boundary, 4 by default; E4D's outer "
            "boundary distance.")
        form.addRow("Width", self._outer_width)
        self._outer_max_cell = QDoubleSpinBox()
        self._outer_max_cell.setRange(0.0, 1e8); self._outer_max_cell.setDecimals(1)
        self._outer_max_cell.setStepType(QAbstractSpinBox.StepType.AdaptiveDecimalStepType)
        self._outer_max_cell.setSuffix(" m²")
        self._outer_max_cell.setSpecialValueText("no limit")
        self._outer_max_cell.setToolTip(
            "Upper limit on the cell area in the outer region. Leave it unlimited "
            "unless the forward solution needs a finer outer mesh: these cells are "
            "never inverted. PyGIMLi's boundaryMaxCellSize; the maximum volume of "
            "E4D's outer zone.")
        form.addRow("Largest cell", self._outer_max_cell)

        self._mesh_defaults = QPushButton("Restore defaults")
        self._mesh_defaults.setToolTip(
            "Put every setting above back to PyGIMLi's default mesh. The imported "
            "mesh and the zones are left alone.")
        self._mesh_defaults.clicked.connect(self._restore_mesh_defaults)
        form.addRow("", self._mesh_defaults)
        # The Mesh tab shows what these build, so it follows them.
        for spin in self._generated_mesh_widgets():
            spin.valueChanged.connect(self._mesh_inputs_changed)
        return box

    def _generated_mesh_widgets(self) -> tuple:
        """The settings of a mesh generated from the electrodes."""
        return (self._para_depth, self._para_boundary, self._surface_nodes,
                self._para_max_cell, self._quality, self._outer_width,
                self._outer_max_cell)

    def _restore_mesh_defaults(self) -> None:
        defaults = {self._para_depth: 0.0, self._para_boundary: 2.0,
                    self._surface_nodes: 1, self._para_max_cell: 0.0,
                    self._quality: 34.0, self._outer_width: 0.0,
                    self._outer_max_cell: 0.0}
        for widget, value in defaults.items():
            widget.setValue(value)

    def _mesh_options(self) -> Dict[str, Any]:
        """The mesh settings under the names ``build_inversion_mesh``, both
        workflows and the assistant use."""
        return {
            "mesh_quality": float(self._quality.value()),
            "para_depth": float(self._para_depth.value()),
            "para_max_cell_size": float(self._para_max_cell.value()),
            "para_boundary": float(self._para_boundary.value()),
            "surface_nodes": int(self._surface_nodes.value()),
            "outer_width": float(self._outer_width.value()),
            "outer_max_cell_size": float(self._outer_max_cell.value()),
            "mesh_file": str(self._mesh_path or ""),
        }

    def _update_mesh_link(self) -> None:
        """The Inversion panel's one line on the mesh, pointing to the Mesh tab."""
        link = getattr(self, "_mesh_link", None)
        if link is None:
            return  # still being built
        if self._mesh_path:
            parts = [f"imported <b>{Path(self._mesh_path).name}</b>"]
        else:
            depth = float(self._para_depth.value())
            parts = ["generated", "depth " + ("auto" if depth <= 0 else f"{depth:g} m"),
                     f"quality {float(self._quality.value()):g}°"]
            others = sum(1 for widget, default in (
                (self._para_boundary, 2.0), (self._surface_nodes, 1),
                (self._para_max_cell, 0.0), (self._outer_width, 0.0),
                (self._outer_max_cell, 0.0)) if widget.value() != default)
            if others:
                parts.append(f"{others} more changed")
        if self._mesh_tab.zones():
            if self._mesh_tab.conform_to_zones() and not self._mesh_path:
                parts.append("follows the zones")
            if self._mesh_tab.decouple_zones():
                parts.append("sharp zone edges")
        # The link first, so a narrow panel wraps the details, not the way there.
        link.setText("<a href='mesh'>Mesh tab</a>: " + " · ".join(parts))

    # -- imported mesh --------------------------------------------------------
    def _import_mesh(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Import inversion mesh", "",
            "Meshes (*.bms *.msh *.vtk *.vtu *.poly *.node *.ele *.cfg);;"
            "E4D mesh (*.node *.ele);;E4D mesh configuration (*.cfg);;All files (*)")
        if not path:
            return
        try:
            self._apply_mesh_file(path)
        except Exception as exc:  # noqa: BLE001 - said next to the button, not in a traceback
            self._mesh_note.setText(f"<span style='color:{theme.color('red')}'>{Path(path).name}: {exc}</span>")
            self.log(f"Could not import the mesh {Path(path).name}: {exc}", "error")

    def _mesh_from_e4d_config(self, path: str) -> str:
        """Build an E4D configuration's mesh once, and keep it as a .bms.

        The build runs off the UI thread - a borehole mesh takes seconds - while a
        local event loop keeps the window responsive until it is done. The run
        then inverts the saved mesh rather than rebuilding it every time.
        """
        from PyHydroGeophysX.core.e4d_mesh import build_e4d_mesh, read_e4d_config

        stem = Path(path).stem
        folder = self.state.ensure_results_store().scratch_dir(self.module_key) / "e4d" / stem

        def build():
            built = build_e4d_mesh(read_e4d_config(path), folder, name=stem)
            target = folder / f"{stem}.bms"
            built["mesh"].save(str(target))
            return str(target), built["mesher"]

        settled: Dict[str, Any] = {}
        loop = QEventLoop()
        worker = TaskWorker(build)
        worker.succeeded.connect(lambda result: (settled.update(result=result), loop.quit()))
        worker.failed.connect(lambda message: (settled.update(error=message), loop.quit()))
        self.register_worker(worker, activity="Building the mesh")
        self._mesh_note.setText(f"Building the mesh {Path(path).name} describes, as E4D would…")
        worker.start()
        if not settled:
            loop.exec()
        if "error" in settled:
            raise RuntimeError(settled["error"])
        target, mesher = settled["result"]
        self.log(f"Built the E4D mesh of {Path(path).name} with {mesher}; saved as "
                 f"{Path(target).name}.", "info")
        return target

    def _apply_mesh_file(self, path: str) -> str:
        """Load and check the mesh now, not when the inversion starts.

        A mesh whose origin or units are wrong fails deep in the forward solver,
        minutes into a run. Checking on import turns that into an immediate
        message next to the button that caused it. An E4D configuration is
        meshed here, once, and the run inverts on that mesh.
        """
        from PyHydroGeophysX.inversion.ert_mesh import load_inversion_mesh

        source = Path(path).name
        if Path(path).suffix.lower() == ".cfg":
            path = self._mesh_from_e4d_config(path)
        # Checked against the loaded survey when there is one; importing before
        # loading data is allowed, and the run re-checks it either way.
        mesh = load_inversion_mesh(path, data=self._ert_data, log=lambda m: None)
        invertible = sum(1 for cell in mesh.cells() if cell.marker() > 1)
        self._mesh_path = str(path)
        self._mesh_clear.setEnabled(True)
        built_from = f" (built from {source})" if source != Path(path).name else ""
        summary = (f"<b>{Path(path).name}</b>{built_from}: {mesh.cellCount()} cells "
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
        """An imported mesh describes its own domain, so the sizing knobs are
        dead, and it cannot be rebuilt along the zone outlines either."""
        generated = not self._mesh_path
        for widget in self._generated_mesh_widgets():
            widget.setEnabled(generated)
            label = self.row_label(widget)
            if label is not None:
                label.setEnabled(generated)
        self._mesh_defaults.setEnabled(generated)
        self._mesh_tab.set_conform_available(
            generated, "" if generated else
            "An imported mesh is inverted as it is, so it cannot be rebuilt along "
            "the zone outlines. Go back to a generated mesh to use this.")
        self._mesh_inputs_changed()

    # -- mesh preview and a-priori zones -------------------------------------
    def _electrode_rows(self) -> List[Dict[str, Any]]:
        """The electrodes as edited here, in the form the data are saved with."""
        return [
            {
                "order": index,
                "label": self._labels[index],
                "x": float(self._x[index]),
                "z": float(self._z[index]),
                "original_index": self._electrode_origins[index],
            }
            for index in range(len(self._x))
        ]

    def _zone_parameters(self) -> Dict[str, Any]:
        """The a-priori zones for a recipe; nothing when there are none, so a run
        without zones records exactly what it did before they existed. The two
        zone options are recorded only when on - and rebuilding along the
        outlines only for a generated mesh, the one it can rebuild."""
        zones = self._mesh_tab.zones()
        if not zones:
            return {}
        parameters: Dict[str, Any] = {"zones": zones}
        if self._mesh_tab.conform_to_zones() and not self._mesh_path:
            parameters["conform_to_zones"] = True
        if self._mesh_tab.decouple_zones():
            parameters["decouple_zones"] = True
        return parameters

    def _sync_mesh_engine(self, *_args: Any) -> None:
        """Tell the zone panel which engine, and which kind of run, will use it."""
        self._mesh_tab.set_engine(str(self._engine.currentData() or "pyhydro"),
                                  self._tl_mode.isChecked())

    def _on_tab_changed(self, index: int) -> None:
        self._sync_tab_controls()
        if self._tabs.widget(index) is self._mesh_tab:
            self._refresh_mesh_preview()
        elif self._tabs.widget(index) is self._recip_view:
            self._refresh_recip_view()

    def _sync_tab_controls(self) -> None:
        """Show only the controls belonging to the selected ERT stage."""
        if not hasattr(self, "_controls"):
            return
        current = self._tabs.currentWidget()
        electrodes = current is self._plot_widget
        data = current is self._pseudo_widget
        quality = current is self._quality_view
        # Mesh and model views already have their own tools beside the plot.
        # Hiding this whole column gives their plot and tools the full width.
        self._controls.setVisible(electrodes or data or quality)
        for widget, visible in (
            (self._load_group, electrodes),
            (self._info, electrodes or data),
            (self._qc_group, data),
            (self._errors_group, data),
            (self._inversion_group, data or quality),
            (self._fit_group, data or quality),
            (self._run_group, data or quality),
            (self._geometry_export_group, electrodes),
        ):
            widget.setVisible(visible)
        self._controls.fit_to_content()

    def _mesh_inputs_changed(self, *_args: Any) -> None:
        """Something the mesh depends on changed; rebuild it if it is on screen.

        After a short pause, so stepping a spin box builds one mesh, not one per
        step. Off screen nothing happens until the tab is opened again, when the
        inputs are compared with those of the mesh shown. Zone edits come here
        too; they change the mesh only while it follows the zone outlines, and
        the comparison of inputs sees to the rest.
        """
        self._update_mesh_link()
        if self._tabs.currentWidget() is self._mesh_tab:
            self._mesh_timer.start()

    def _mesh_preview_request(self):
        """What the next run would build its mesh from.

        Returns ``((key, task arguments, description), "")``, or ``(None,
        reason)`` when there is nothing to build from. The key holds every input
        of the mesh, so a mesh is rebuilt exactly when one of them changed: the
        settings, and the zone outlines while the mesh follows them - not the
        zones' values, which leave the mesh as it is.
        """
        options = self._mesh_options()
        mesh_file = options["mesh_file"]
        zones = self._mesh_tab.zones()
        outlines: tuple = ()
        if zones and self._mesh_tab.conform_to_zones() and not mesh_file:
            options["conform_zones"] = zones
            outlines = tuple(tuple(tuple(vertex) for vertex in zone["polygon"])
                             for zone in zones)
        settings = tuple(sorted((name, value) for name, value in options.items()
                                if name != "conform_zones")) + (outlines,)
        if self._tl_mode.isChecked() and self._tl_files:
            # The time-lapse pipeline meshes around its first survey, read with
            # the chosen instrument and placed from the electrode file, as the
            # run places it; this page's hand edits are not applied to a series.
            first, instrument = self._tl_files[0], self._instrument.currentData()
            electrodes = self._electrode_file()
            key = ("timelapse", first, instrument, electrodes, settings)
            args = ("timelapse", first, instrument, electrodes)
            where = f"first survey of the series, {Path(first).name}"
        elif self._ert_data is not None:
            rows = self._electrode_rows()
            key = ("single", id(self._ert_data),
                   tuple((r["x"], r["z"], r["original_index"]) for r in rows), settings)
            args = ("single", self._ert_data, None, rows)
            where = self._data_path.name if self._data_path else "the loaded survey"
        elif mesh_file:
            key = ("mesh", mesh_file)
            args = ("mesh", None, None, None)
            where = ""
        else:
            return None, "Load ERT data, or import a mesh, to see the inversion mesh."
        if mesh_file:
            where = f"imported {Path(mesh_file).name}" + (f", with {where}" if where else "")
        return (key, args + (options,), where), ""

    def _refresh_mesh_preview(self, force: bool = False) -> None:
        """Build the mesh preview if its inputs changed, or when ``force``d."""
        request, message = self._mesh_preview_request()
        if request is None:
            self._supersede_mesh_preview()
            self._mesh_preview_key = None
            self._mesh_tab.set_message(message)
            return
        key, args, where = request
        if not force and key in (self._mesh_preview_key, self._mesh_preview_pending):
            return
        scratch = None
        if args[0] == "single":
            scratch = self.state.ensure_results_store().scratch_dir(self.module_key)
        self._supersede_mesh_preview()
        self._mesh_preview_pending = key
        self._mesh_tab.set_status("Building the inversion mesh…")
        worker = TaskWorker(self._mesh_preview_task, *args, scratch)
        worker.succeeded.connect(
            lambda preview, w=worker: self._on_mesh_preview_ready(w, key, where, preview))
        worker.failed.connect(
            lambda error, w=worker: self._on_mesh_preview_failed(w, key, error))
        self._mesh_worker = self.register_worker(worker, activity="Previewing the mesh")
        worker.start()

    def _mesh_summary(self) -> Dict[str, Any]:
        """The counts of the mesh on the Mesh tab, and whether it is current."""
        preview = self._mesh_tab.preview()
        if not preview:
            return {"built": False}
        request, _ = self._mesh_preview_request()
        summary = {key: preview[key] for key in
                   ("dim", "cells", "para_cells", "outer_cells", "nodes")}
        summary.update(built=True, current=bool(request) and request[0] == self._mesh_preview_key,
                       electrodes=int(len(preview.get("sensors", ()))),
                       zone_outline_edges=int((preview.get("build") or {}).get(
                           "zone_outline_edges", 0)))
        if preview.get("para_extent") is not None:
            summary["para_extent"] = [round(float(v), 3) for v in preview["para_extent"]]
            summary["full_extent"] = [round(float(v), 3) for v in preview["full_extent"]]
        return summary

    def _supersede_mesh_preview(self) -> None:
        previous, self._mesh_worker = self._mesh_worker, None
        self._mesh_preview_pending = None
        if previous is not None and previous.isRunning():
            previous.cancel()

    @staticmethod
    def _mesh_preview_task(kind, source, instrument, electrodes, options, scratch):
        """Build the mesh the next run would build, off the UI thread.

        Through the run's own steps: a single survey is saved with the electrode
        edits applied and read back, as the run inverts it; a series is meshed
        around its first survey, read with its instrument and placed from the
        electrode file (``electrodes``, a path there); the mesh itself comes from
        the builder both inversions use, with ``options`` as its keywords.
        """
        from PyHydroGeophysX.inversion.ert_mesh import build_inversion_mesh, mesh_preview

        data = None
        if kind == "single":
            path = save_edited_ert_container(
                source, Path(scratch) / "mesh_preview_data.dat", electrodes)
            data = ert_load.load_ert_container(path, instrument=None)
        elif kind == "timelapse":
            data = ert_load.load_ert_container(source, instrument=instrument,
                                               electrode_file=electrodes)
        report: Dict[str, Any] = {}
        mesh = build_inversion_mesh(data, **options, report=report)
        preview = mesh_preview(mesh, data)
        preview["build"] = report
        if data is not None and data.haveData("rhoa"):
            rhoa = np.asarray(data["rhoa"], dtype=float)
            rhoa = rhoa[np.isfinite(rhoa) & (rhoa > 0)]
            preview["median_rhoa"] = float(np.median(rhoa)) if rhoa.size else None
        return preview

    def _on_mesh_preview_ready(self, worker, key, where: str, preview: Dict[str, Any]) -> None:
        if worker is not self._mesh_worker:
            return  # superseded by a newer build
        self._mesh_preview_pending = None
        self._mesh_preview_key = key
        # A zone drawn next starts at the survey's typical value, not at 100.
        if preview.get("median_rhoa"):
            self._mesh_tab.set_default_resistivity(float(preview["median_rhoa"]))
        self._mesh_tab.set_preview(preview, where)

    def _on_mesh_preview_failed(self, worker, key, message: str) -> None:
        if worker is not self._mesh_worker:
            return
        self._mesh_preview_pending = None
        # Kept, so the same inputs are not retried on every change of tab.
        self._mesh_preview_key = key
        self._mesh_tab.set_message(f"The inversion mesh could not be built: {message}")
        self.log(f"Mesh preview failed: {message}", "warn")

    def _on_inv_mode_changed(self, *_args: Any) -> None:
        """Apply the Quick/Full split to the stages each mode owns.

        Quick drops the two stages that cost the most and change no model on a
        clean dataset: the geometric-factor check and the auto-λ search. Full
        restores both.

        This is a preset, not a lock. The auto-λ checkbox stays enabled in both
        modes, so ticking it under Quick runs the search and the panel shows that
        it will. The k check has no such control by design, because a wrong k is
        the one problem the result cannot reveal.
        """
        quick = str(self._inv_mode.currentData() or "quick") == "quick"
        self._geom_policy = "off" if quick else "fix"
        self._auto_lam.setChecked(not quick)

    def _on_auto_lambda(self, on: bool) -> None:
        """Show the auto-λ target and trial budget only while auto-λ is on."""
        self._set_rows_visible((self._chi2_row, self._lam_trials), on)

    def _on_reject_outliers(self, on: bool) -> None:
        """Show the rejection cut and the data floor only while rejection is on."""
        self._set_rows_visible((self._reject_row, self._min_keep), on)

    def _on_inversion_failed(self, message: str) -> None:
        self._agent_run_state = "failed"
        self._agent_run_error = message
        self.fail_persisted_run(message, "ert.single_inversion")
        self.log(f"ERT inversion failed: {message}", "error")

    def _reset_invert_button(self) -> None:
        if self._inv_busy is not None:
            self._inv_busy.finish()
            self._inv_busy = None
        self._invert_btn.setText("Run inversion")
        self._inv_progress.setVisible(False)

    # -- time-lapse inversion ------------------------------------------------
    def _set_tl_files(self, paths: List[str]) -> None:
        """Replace every file of the list, reciprocal files included, regroup
        them into surveys, and refresh the list and its times."""
        self._tl_all = [str(p) for p in paths]
        self._group_tl_surveys()
        self._read_tl_times()
        self._refresh_tl_list()

    def _group_tl_surveys(self) -> None:
        """Make each forward file and its reciprocal file one survey.

        ``survey_timing.reciprocal_pairs`` finds them by name and time; the
        pairs hold while "Pair reciprocal files" is ticked, and the box is shown
        only when the list holds a file named as a reciprocal. Unticked, or with
        none, every file is a survey of its own, as before pairing existed.
        """
        paths = list(self._tl_all)
        self._tl_partner, self._tl_left_out = {}, []
        found: Dict[str, Any] = {"pairs": [], "orphans": [], "gap_seconds": None}
        if paths:
            mtime_box = getattr(self, "_tl_use_mtime", None)
            timing = ert_load.survey_timing_for(
                paths, allow_mtime=bool(mtime_box is not None and mtime_box.isChecked()))
            found = survey_timing.reciprocal_pairs(paths, timing.found or None)
            # Each file's own time, the reciprocal files' included, for the rows.
            self._tl_stamp_of = dict(zip(paths, timing.found or []))
        self._tl_pairs_found = found
        detected = bool(found["pairs"] or found["orphans"])
        self._tl_pair_box.setVisible(detected)
        if detected and self._tl_pair_box.isChecked():
            self._tl_partner = {paths[f]: paths[r] for f, r in found["pairs"]}
            self._tl_left_out = [paths[r] for r in found["orphans"]]
            hidden = set(self._tl_partner.values()) | set(self._tl_left_out)
            self._tl_files = [p for p in paths if p not in hidden]
        else:
            self._tl_files = paths
        # The reciprocal error model is offered for a paired series.
        self._refresh_error_sources(self._ert_data_full)

    def _tl_flat(self, surveys: List[str]) -> List[str]:
        """Every file of ``surveys``, each forward file followed by its
        reciprocal file, and then the reciprocal files left out."""
        out: List[str] = []
        for path in surveys:
            out.append(path)
            if path in self._tl_partner:
                out.append(self._tl_partner[path])
        return out + [p for p in self._tl_left_out if p not in out]

    def _on_tl_pairing_changed(self, checked: bool) -> None:
        """Pair or unpair the list's reciprocal files, and reload the survey on
        screen when it was, or now is, one half of a pair."""
        before = dict(self._tl_partner)
        self._group_tl_surveys()
        self._read_tl_times()
        self._refresh_tl_list()
        if checked:
            self.log(f"Reciprocal files paired: {len(self._tl_partner)} survey(s) now "
                     "hold a forward and a reciprocal file each.", "info")
        else:
            self.log("Reciprocal files unpaired: every file is a time step of its own.",
                     "info")
        shown = str(self._data_path) if self._data_path is not None else ""
        if shown and (shown in before or shown in self._tl_partner) \
                and before.get(shown) != self._tl_partner.get(shown):
            self._start_load(shown)

    def _read_tl_times(self) -> None:
        """Read the acquisition time of every survey, from the names or the headers.

        Kept in one place because three operations reorder the sequence, and a set
        of times that no longer matches the list is worse than none: the inversion
        would pair each survey with someone else's date. A paired survey is timed
        by its forward file.
        """
        mtime_box = getattr(self, "_tl_use_mtime", None)   # absent until built
        self._tl_timing = ert_load.survey_timing_for(
            self._tl_files,
            allow_mtime=bool(mtime_box is not None and mtime_box.isChecked()))
        self._tl_times = list(self._tl_timing.times)
        self._tl_labels = list(self._tl_timing.labels)

    def _reciprocal_time(self, path: str) -> str:
        """When a reciprocal file was measured, as hh:mm, or "" when unknown."""
        stamp = getattr(self, "_tl_stamp_of", {}).get(path)
        return f"{stamp:%H:%M}" if stamp is not None else ""

    def _refresh_tl_list(self) -> None:
        self._tl_list.blockSignals(True)
        self._tl_list.clear()
        timing = getattr(self, "_tl_timing", None)
        gaps = timing.intervals if timing is not None else []
        # A forward file without its reciprocal is marked only in a list that
        # pairs some: in one without reciprocal files every file is unpaired.
        mark_unpaired = bool(self._tl_partner)
        for i, path in enumerate(self._tl_files):
            label = self._tl_labels[i] if i < len(self._tl_labels) else str(i + 1)
            # The gap to the previous survey, on the row it belongs to: an hourly
            # sequence and a monthly one look identical without it.
            gap = ""
            if i > 0 and i - 1 < len(gaps):
                step = gaps[i - 1]
                gap = (f"   (+{survey_timing.format_duration(step)})" if step >= 0
                       else f"   ({survey_timing.format_duration(step)}, out of order)")
            text = f"{i + 1}.  {label}    ·    {Path(path).name}{gap}"
            partner = self._tl_partner.get(path)
            # The reciprocal file on a line of its own under its forward file,
            # so a wrong pairing can be seen in the list itself.
            if partner:
                # Its time first: a long file name runs off the row's end.
                when = self._reciprocal_time(partner)
                text += (f"\n        + reciprocal{f' {when}' if when else ''}    ·    "
                         f"{Path(partner).name}")
            elif mark_unpaired:
                text += "   (unpaired: no reciprocal file)"
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, path)
            # Where this file's time came from: the one file whose time was not
            # read from its name, or was read two ways, is the one to look at.
            origin = timing.describe(i) if timing is not None else ""
            tip = f"{path}\n{origin}" if origin else path
            if partner:
                tip += (f"\n\nReciprocal file, merged with it into one survey:\n{partner}")
            item.setToolTip(tip)
            self._tl_list.addItem(item)
        self._tl_list.blockSignals(False)
        n = len(self._tl_files)
        if n == 0:
            self._tl_info.setText("No files added.")
        elif n == 1:
            self._tl_info.setText(f"<b>1</b> survey. Tick “Time-lapse” and add more for a "
                                  f"time sequence. Instrument: {self._instrument.currentText()}."
                                  + self._tl_pairs_text(timing))
        else:
            self._tl_info.setText(f"{self._tl_timing_text(timing)} "
                                  f"Instrument: {self._instrument.currentText()}.")
        # A time-lapse mesh is built from whichever survey is now first, and
        # the Reciprocal errors tab draws the series as the list now holds it.
        self._mesh_inputs_changed()
        self._recip_inputs_changed()

    def _tl_timing_text(self, timing: Optional[Any]) -> str:
        """The file list's summary line, and what to do when its times fall short.

        The accepted forms are on the line's tooltip; what is said here is only
        what this set of files needs: a time missing from some or all of them,
        files out of time order, or forward and reciprocal halves.
        """
        n = len(self._tl_files)
        if timing is None:
            return f"{n} files."
        how = ("  Put the time in each file name, year first (e.g. "
               "<code>site_2026-01-12_05-50-38.dat</code>), add a line "
               "<code># date: 2026-01-12 05:50:38</code> at the top of each file, "
               "or tick “Use file times” below.")
        # The summary names the files that keep an undated list from being dated
        # (no time, or a time another file has too); hovering a row says why.
        text = timing.summary()
        if timing.dated and any(gap < 0 for gap in timing.intervals):
            text += " The files are not in time order: press Sort by time."
        elif not timing.dated and timing.undated_files():
            text += how
        return text + self._tl_pairs_text(timing)

    def _tl_pairs_text(self, timing: Any) -> str:
        """What the list made of forward and reciprocal files.

        Paired, each survey is one forward file with its reciprocal: said once,
        with the forward files that have none and - in amber - the reciprocal
        files left out for want of one. Unpaired while the names say there are
        pairs, each survey is inverted twice and its reciprocals are never
        compared, which is warned about.
        """
        found = getattr(self, "_tl_pairs_found", None) or {"pairs": [], "orphans": []}
        if not (found["pairs"] or found["orphans"]):
            self._tl_pairs_logged = None
            return ""
        gap = survey_timing.format_duration(found["gap_seconds"]) \
            if found.get("gap_seconds") is not None else ""
        if self._tl_pair_box.isChecked():
            paired = len(self._tl_partner)
            alone = len(self._tl_files) - paired
            text = (f" {paired} of {len(self._tl_files)} surveys are a forward file with its "
                    f"reciprocal file{f' (about {gap} later)' if gap else ''}, merged into "
                    "one time step so each reading is compared with its reciprocal.")
            if alone:
                text += f" {alone} forward file(s) have no reciprocal file and are inverted alone."
            warning = ""
            if self._tl_left_out:
                warning = (f"{len(self._tl_left_out)} reciprocal file(s) have no forward file "
                           "and are left out: "
                           + ", ".join(Path(p).name for p in self._tl_left_out) + ".")
            key = ("paired", paired, tuple(self._tl_left_out))
            message = (text.strip() + (" " + warning if warning else ""), "warn" if warning
                       else "info")
            shown = text + (f" <span style='color:{theme.color('amber')}'>{warning}</span>"
                            if warning else "")
        else:
            warning = (
                f"{len(found['pairs'])} file(s) look like the reciprocal half of a survey "
                f"(“recip” in the name){f', each about {gap} after its forward file' if gap else ''}. "
                f"With “Pair reciprocal files” unticked every file is a time step of its "
                f"own, so each survey appears twice and no reciprocal errors are formed "
                f"between the two files.")
            key = ("unpaired", tuple(sorted(found["reciprocal"])))
            message = (warning, "warn")
            shown = f" <span style='color:{theme.color('amber')}'>{warning}</span>"
        # Logged once per state, so reordering the list does not repeat it.
        if getattr(self, "_tl_pairs_logged", None) != key:
            self._tl_pairs_logged = key
            self.log(*message)
        return shown

    def _add_tl_files(self) -> None:
        from PyHydroGeophysX.qt_apps.widgets.project_dialogs import confirm_project_for_data
        if not confirm_project_for_data(self):   # name a Project before the first data
            return
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Add ERT data file(s)", "", _DATA_FILTER)
        if not paths:
            return
        # Append new files, preserving order and dropping duplicates.
        old_n = len(self._tl_files)
        merged = list(self._tl_all)
        added = 0
        for p in paths:
            if p not in merged:
                merged.append(p); added += 1
        self._set_tl_files(merged)
        self.log(f"Added {added} ERT file(s); {len(self._tl_files)} survey(s) in the list.",
                 "info")
        # Auto-preview the first newly added survey so the user sees data immediately.
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
        # A row is a survey: its reciprocal file goes with it.
        self._set_tl_files(self._tl_flat(
            [p for i, p in enumerate(self._tl_files) if i not in rows]))
        self.log(f"Removed {len(rows)} survey(s); {len(self._tl_files)} remain.", "info")

    def _move_tl_files(self, delta: int) -> None:
        rows = self._selected_tl_rows()
        if not rows:
            return
        order = list(range(len(self._tl_files)))
        chosen = set(rows)
        # The selection moves as a block: a row steps past an unselected
        # neighbour, and one against the edge - or behind a selected row that
        # is - stays put. Swapped one at a time, a block already at the top was
        # reversed, and the top row is the baseline every survey is compared to.
        steps = range(1, len(order)) if delta < 0 else range(len(order) - 2, -1, -1)
        for i in steps:
            j = i - 1 if delta < 0 else i + 1
            if order[i] in chosen and order[j] not in chosen:
                order[i], order[j] = order[j], order[i]
        if order == list(range(len(order))):
            return
        # A pair moves as one row; the full list follows the surveys' order.
        self._tl_files = [self._tl_files[i] for i in order]
        self._tl_all = self._tl_flat(self._tl_files)
        self._read_tl_times()
        moved = {order.index(i) for i in rows}
        self._refresh_tl_list()
        for r in moved:
            self._tl_list.item(r).setSelected(True)

    def _sort_tl_files_by_time(self) -> None:
        """Reorder the list into acquisition order.

        The sequence order is part of the result - each survey is compared with
        the one before it - and a file dialog returns names in whatever order the
        filesystem gives, which for ``survey_9`` and ``survey_10`` is not the
        order they were recorded in. A paired survey sorts by its forward file.
        """
        timing = getattr(self, "_tl_timing", None)
        if timing is None or not timing.dated:
            self.log("Cannot sort: the files carry no readable acquisition time. "
                     "Use ↑ ↓ to put them in order by hand.", "warn")
            return
        order = sorted(range(len(self._tl_files)), key=lambda i: timing.timestamps[i])
        if order == list(range(len(self._tl_files))):
            self.log("Files are already in acquisition order.", "info")
            return
        self._set_tl_files(self._tl_flat([self._tl_files[i] for i in order]))
        self.log(f"Sorted {len(self._tl_files)} surveys into acquisition order.", "success")

    def _clear_tl_files(self) -> None:
        self._set_tl_files([])
        self.log("Cleared time-lapse file list.", "info")

    def _on_tl_selection_changed(self) -> None:
        rows = self._selected_tl_rows()
        if len(rows) > 1:
            self._tl_info.setText(f"{len(rows)} surveys selected — Remove / Move ↑ ↓, "
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
        self._warn_external_not_running("time-lapse folder")
        if (
            self._engine.currentData() == "adtlert"
            and self._adtlert_timelapse_ready is not True
        ):
            self.log(
                "ADTLERT timelapse self-test has not passed. Wait for the check, "
                "or select In-house Gauss-Newton / PyHydro. "
                "To retry, switch engines and select ADTLERT again.",
                "warn",
            )
            return
        if (
            self._engine.currentData() == "adtlert"
            and not self._tl_windowed.isChecked()
        ):
            self.log(
                "ADTLERT time-lapse inversion requires Windowed (sliding window).",
                "warn",
            )
            return
        instrument = self._instrument.currentData()
        params = {
            "lambda_val": self._lam.value(), "alpha": self._tl_alpha.value(),
            "inversion_type": self._tl_type.currentText(), "max_iterations": self._iter.value(),
            "relativeError": self._relerr.value(),
            # The mesh settings are shared with the single inversion, the
            # imported mesh among them, which switches the others off; the
            # series is inverted on it.
            **self._mesh_options(),
            **self._zone_parameters(),
            "windowed": self._tl_windowed.isChecked(), "window_size": self._tl_window.value(),
            "engine": str(self._engine.currentData()),
            "instrument": instrument,
            # Same auto-λ switch as the single inversion; the trial budget is
            # smaller because each trial is a joint inversion over every step.
            "auto_lambda": bool(self._auto_lam.isChecked()),
            "target_chi2": float(self._target_chi2.value()),
            "chi2_tolerance": float(self._chi2_tol.value()),
            "max_lambda_trials": min(int(self._lam_trials.value()), 4),
            "temporal_weighting": (
                "interval" if self._tl_dt_weight.isChecked() else "uniform"),
        }
        if self._tl_lowmem.isChecked():
            params["save_memory"] = True
        if self._engine.currentData() == "e4d":
            # E4D runs each survey to a plateau, judged by the same stopping rule.
            params["e4d"] = self._e4d_settings()
            params["plateau_tolerance"] = float(self._plateau.value()) / 100.0
        if self._r2_program_name() is not None:
            params["r2"] = self._r2_settings()
        # The exported panels are trimmed the way the Resistivity model view trims
        # the section ("Hide below" + "Clean cut"), the one place this is chosen.
        cut = self._model_view.clean_cut()
        if cut is not None:
            params["figure_clip"] = "envelope"
            params["figure_clip_threshold"] = cut
        times = self._tl_times if len(self._tl_times) == len(self._tl_files) else None
        timing = getattr(self, "_tl_timing", None)
        # The bundle below renames the files, so the acquisition times have to
        # travel with the run or the intervals cannot be reported from inside it.
        stamps = ([t.isoformat(sep=" ") for t in timing.timestamps]
                  if timing is not None and timing.dated else [])
        try:
            run = self.begin_persisted_run(
                "ert.timelapse_inversion", "ert.timelapse_inversion",
                label=self._run_label(self._tl_files, unit="surveys"),
            )
        except Exception as exc:  # noqa: BLE001
            self.log(f"Could not prepare Project run: {exc}", "error")
            return
        # Drop this module's references to the previous run's arrays before the
        # new one writes. They are ordinary in-memory arrays now, but releasing
        # them keeps a stale model out of the viewer if the run fails.
        self._tl_models = None
        self._tl_models_raw = None
        self._tl_correction = None
        self._tl_coverage = None
        if getattr(self, "_map_result_kind", "") == "timelapse":
            # The series the panel was correcting is gone; a single inversion on
            # screen instead stays correctable while this one runs.
            self._tc_box.set_available(False, _TC_WAITING)
        # The list can be edited while the surveys are filtered below, so the run
        # keeps the sequence as it stood when Run was pressed.
        sources, labels = list(self._tl_files), list(self._tl_labels)
        # The electrode file places every survey's electrodes, as it placed the
        # survey on screen. The run used to read each file's own header instead,
        # while the preview showed the electrode file's positions.
        electrodes = self._electrode_file()
        if electrodes is not None:
            self.log("Time-lapse: every survey's electrodes are placed from "
                     f"{(self._electrode_path or Path(electrodes)).name}.", "info")
        self._tl_busy = BusyStateController([self._tl_btn])
        self._tl_busy.start()
        self._tl_btn.setText("Inverting…")
        self._tl_progress.setVisible(True); self._tl_progress.setRange(0, 0)
        qc = self._qc_applied
        # Each survey's reciprocal file, as the list paired them when Run was
        # pressed, and the reciprocal files it left out.
        partners = [self._tl_partner.get(path) for path in sources]
        left_out = list(self._tl_left_out)
        fit_model = self._err_source.currentData() == "reciprocal"
        # What the error model is fitted to, as the Reciprocal errors tab said
        # when Run was pressed; the surveys' cache keys, so the pairs the QC
        # reads are kept for the tab, and the ones the tab read can serve here.
        fit_to = self._error_fit_to()
        survey_keys = [self._recip_survey_key(path, partner)
                       for path, partner in zip(sources, partners)]
        qc_key = self._recip_qc_key(qc)
        reader = f"{instrument or 'auto-detected format'}" + (
            # The user's electrode file, not the readers' copy of it.
            f", electrodes from {(self._electrode_path or Path(electrodes)).name}"
            if electrodes is not None else "")
        if qc is None and not any(partners) and not fit_model:
            self.log("Time-lapse QC: no filter applied, so every survey is inverted as "
                     "loaded. The Data QC thresholds take effect with Apply filter, and "
                     "then filter every survey of the series.", "info")
            self._launch_timelapse(run, sources, sources, labels, params, times, stamps, None,
                                   electrodes, fit_to=fit_to)
            return
        if qc is None and not any(partners):
            # Read only to fit the model: when the Reciprocal errors tab has read
            # every survey already, its pairs are fitted and the surveys go to
            # the run as they are, as they do with no model.
            cached = [self._recip_cache.get(key, {}).get(qc_key) for key in survey_keys]
            if all(entry is not None and not entry.get("error") for entry in cached):
                self._fit_timelapse_from_cache(run, sources, labels, params, times, stamps,
                                               electrodes, [dict(e) for e in cached],
                                               reader, left_out, fit_to)
                return
        # The QC applied on this page holds for the whole series, as it did for
        # the survey on screen. Each survey is filtered on its own - the
        # pipeline accepts surveys with different measurement sets, and ADTLERT
        # aligns them - and handed over in PyGIMLi's own format, which it reads
        # back as written, as the single inversion's filtered data is. A survey
        # of two files is merged first, so its pairs are formed across them; and
        # the reciprocal error model is fitted over every survey's pairs.
        doing = [f"filtering as Apply filter did ({self._describe_qc(qc)})"
                 if qc is not None else "no QC filter applied"]
        if any(partners):
            doing.insert(0, f"merging {sum(1 for p in partners if p)} with their "
                            "reciprocal file")
        if fit_model:
            doing.append("fitting the reciprocal error model over the series, to "
                         + self._fit_to_text(fit_to, qc))
        self.log(f"Time-lapse QC: preparing each of the {len(sources)} surveys - "
                 + "; ".join(doing) + "…", "info")
        if left_out:
            self.log(f"Time-lapse: {len(left_out)} reciprocal file(s) have no forward file "
                     "and are left out of the run: "
                     + ", ".join(Path(p).name for p in left_out) + ".", "warn")
        # Its QC logs and report go in the run folder (qt_apps/ert_records.py).
        worker = TaskWorker(self._qc_series, sources, instrument, qc,
                            run.inputs_dir / "ert_timesteps_qc", with_log=True,
                            electrode_file=electrodes, report_dir=run.run_dir,
                            acquired=stamps, error_model=self._error_model_text(True),
                            reader=reader, partners=partners, left_out=left_out,
                            fit_model=fit_model, floor=self._reciprocal_floor(),
                            fit_to=fit_to)

        def prepared(out: Dict[str, Any]) -> None:
            # Every survey's pairs as the run read them, for the Reciprocal
            # errors tab, which then need not read the series again.
            for key, report in zip(survey_keys, out.get("reports") or []):
                self._cache_recip(self._recip_cache, key, qc_key, self._recip_entry(report))
            self._recip_inputs_changed()
            run_params = {**params, "instrument": None}
            model = out.get("model")
            if fit_model and model:
                run_params["error_model"] = model
            elif fit_model:
                self.log("Data errors: the series has too few reciprocal pairs to fit an "
                         "error model (at least 15 are needed), so each survey keeps its "
                         "own errors.", "warn")
            self._launch_timelapse(
                run, sources, out["files"], labels, run_params, times, stamps,
                {"thresholds": qc, "source_instrument": instrument, "steps": out["steps"],
                 "partners": partners, "left_out": left_out}, electrodes, fit_to=fit_to)

        worker.logged.connect(self._on_tl_qc_logged)
        worker.succeeded.connect(prepared)
        worker.failed.connect(self._on_tl_qc_failed)
        # Stopped, it ends at the next survey without a word; the button comes back.
        worker.finished.connect(
            lambda w=worker: self._reset_tl_button() if w.is_cancelled() else None)
        self._tl_qc_worker = self.register_worker(worker, activity="Checking data quality")
        self._tl_progress.setRange(0, len(sources))
        self._tl_progress.setValue(0)
        self._tl_progress.setFormat(f"Data QC 0/{len(sources)}")
        worker.start()
        # Reading every survey takes most of a second each, minutes for a long
        # series, so the QC can be stopped like the inversion that follows it.
        self._tl_stop.attach(worker, "ert.timelapse_inversion", "The time-lapse data QC")

    @staticmethod
    def _fit_to_text(fit_to: str, qc: Optional[Dict[str, Any]]) -> str:
        from PyHydroGeophysX.qt_apps.ert_records import fit_to_sentence

        return fit_to_sentence(fit_to, qc is not None)

    def _fit_timelapse_from_cache(self, run, sources: List[str], labels: List[str],
                                  params: Dict[str, Any], times, stamps: List[str],
                                  electrodes: Optional[str], reports: List[Dict[str, Any]],
                                  reader: str, left_out: List[str], fit_to: str) -> None:
        """Fit the series' error model to the pairs the Reciprocal errors tab
        read, write the QC records from them, and start the run on the surveys
        as they are - with no filter and no reciprocal files to merge, the
        surveys were read only for these pairs, and are not read twice."""
        self.log(f"Time-lapse: fitting the reciprocal error model to the pairs of the "
                 f"{len(sources)} surveys the Reciprocal errors tab has read, to "
                 f"{self._fit_to_text(fit_to, None)}; the surveys are not read again.",
                 "info")
        worker = TaskWorker(self._finish_series_qc, list(sources), reports, with_log=True,
                            report_dir=run.run_dir, stamps=list(stamps), reader=reader,
                            qc=None, error_model=self._error_model_text(True), fit_model=True,
                            floor=self._reciprocal_floor(), left_out=list(left_out),
                            fit_to=fit_to, logs_written=False)

        def fitted(model: Optional[Dict[str, Any]]) -> None:
            run_params = dict(params)
            if model:
                run_params["error_model"] = model
            else:
                self.log("Data errors: the series has too few reciprocal pairs to fit an "
                         "error model (at least 15 are needed), so each survey keeps its "
                         "own errors.", "warn")
            self._launch_timelapse(run, sources, sources, labels, run_params, times, stamps,
                                   None, electrodes, fit_to=fit_to)

        worker.logged.connect(lambda message: self.log(message, "info"))
        worker.succeeded.connect(fitted)
        worker.failed.connect(self._on_tl_qc_failed)
        worker.finished.connect(
            lambda w=worker: self._reset_tl_button() if w.is_cancelled() else None)
        self._tl_qc_worker = self.register_worker(worker, activity="Fitting the error model")
        worker.start()
        self._tl_stop.attach(worker, "ert.timelapse_inversion", "The time-lapse data QC")

    def _on_tl_qc_logged(self, message: str) -> None:
        """Log a line of the time-lapse QC, and move its progress bar per survey."""
        self.log(message, "info")
        match = re.match(r"^QC (\d+)/(\d+) ", str(message))
        if match is not None:
            done, total = int(match.group(1)), int(match.group(2))
            self._on_tl_progress(done, total, f"Data QC {done}/{total}")
            # The status bar too: reading 420 surveys takes minutes, and a bare
            # "Checking data quality" for that long reads as a hang.
            worker = getattr(self, "_tl_qc_worker", None)
            if worker is not None:
                self._on_worker_progress(worker, done, total, "Checking data quality")

    @staticmethod
    def _describe_qc(qc: Dict[str, Any]) -> str:
        """The thresholds in ``qc`` that cut anything, in the panel's terms."""
        parts = [f"ρa {qc['min_rhoa']:g}–{qc['max_rhoa']:g} Ω·m"]
        if qc.get("average_reciprocals"):
            parts.insert(0, "each reading averaged with its reciprocal")
        if qc["max_error"] > 0:
            parts.append(f"error ≤ {qc['max_error']:g} %")
        if qc["more_checks"]:
            extra = [(qc["drop_nonpositive"], "ρa > 0"),
                     (qc["min_voltage"] > 0, f"|V| ≥ {qc['min_voltage']:g}"),
                     (qc["min_current"] > 0, f"|I| ≥ {qc['min_current']:g}"),
                     (qc["max_k"] > 0, f"|k| ≤ {qc['max_k']:g}"),
                     (qc["max_contact_r"] > 0, f"contact R ≤ {qc['max_contact_r']:g} Ω"),
                     (qc.get("max_stack", 0.0) > 0,
                      f"stacking spread ≤ {qc.get('max_stack', 0.0):g} %"),
                     (qc["max_reciprocal"] > 0, f"reciprocal error ≤ {qc['max_reciprocal']:g} %")]
            parts.extend(text for on, text in extra if on)
        return ", ".join(parts)

    @classmethod
    def _prepare_survey(cls, source: str, partner: Optional[str], instrument: Optional[str],
                        electrode_file: Optional[str], qc: Optional[Dict[str, Any]], *,
                        paired_series: bool = False, log=None):
        """``(data, report, reasons)``: one survey of a series, as the run prepares it.

        The file is read the way the time-lapse pipeline reads it, electrodes
        placed from ``electrode_file`` when there is one; with ``partner``,
        that reciprocal file is read too and merged with it
        (``ert_io.merge_reciprocal``), so its pairs are formed across the two.
        ``qc`` - None when no filter was applied - then filters it, check 0
        included (:meth:`_qc_survey`), marking which pairs it kept. ``report``
        is the survey's QC log as ``ert_records.survey_qc_log`` reads it. The
        run's QC (:meth:`_qc_series`) and the Reciprocal errors tab both read
        their surveys here, so the two hold the same pairs.
        """
        log = log or (lambda _message: None)
        data = ert_load.load_ert_container(source, instrument=instrument, log=log,
                                           electrode_file=electrode_file)
        report: Dict[str, Any] = {}
        if partner:
            other = ert_load.load_ert_container(partner, instrument=instrument, log=log,
                                                electrode_file=electrode_file)
            try:
                data, merged = ert_load.merge_reciprocal(data, other)
            except ValueError as exc:
                raise ValueError(
                    f"{Path(source).name} and its reciprocal file {Path(partner).name} "
                    f"could not be merged: {exc}") from exc
            report["files"] = {
                "forward": Path(source).name, "reciprocal": Path(partner).name,
                "forward_readings": merged["forward"],
                "reciprocal_readings": merged["reciprocal"],
                "dropped_fields": merged["dropped_fields"]}
        elif paired_series:
            report["files"] = {"forward": Path(source).name, "reciprocal": None,
                               "forward_readings": int(data.size())}
        if qc is not None:
            data, _keep, reasons = cls._qc_survey(data, qc, report)
        else:
            from PyHydroGeophysX.qt_apps.ert_records import reciprocal_pairing

            reasons = []
            report.update(readings=int(data.size()), kept=int(data.size()), checks=[],
                          pairing=reciprocal_pairing(cls._reciprocal_scores(data)))
        return data, report, reasons

    @classmethod
    def _qc_series(cls, files: List[str], instrument: Optional[str],
                   qc: Optional[Dict[str, Any]],
                   staging: Path, log=None,
                   electrode_file: Optional[str] = None,
                   report_dir: Optional[Path] = None,
                   acquired: Optional[List[str]] = None,
                   error_model: str = "", reader: str = "",
                   partners: Optional[List[Optional[str]]] = None,
                   left_out: Optional[List[str]] = None, fit_model: bool = False,
                   floor: float = 0.0, fit_to: str = "all") -> Dict[str, Any]:
        """Prepare every survey of a series for the run; runs off the UI thread.

        Each survey is read, merged with its reciprocal file in ``partners``
        and filtered by ``qc`` as :meth:`_prepare_survey` does it, and written
        back. Returns the prepared files, in order, per survey what was kept,
        ``reports`` (each survey's QC report, its pairing among it) and
        ``model``: with ``fit_model``, the reciprocal error model fitted over
        the pairs ``fit_to`` names - all of them as read, or those the filter
        kept - (``ert_records.fit_series_error_model``, the fit the QC report
        states), its errors never below ``floor``; None when there are too few
        pairs.

        With ``report_dir`` (the run folder) each survey's QC log goes in its
        ``qc/`` folder as it is prepared - a survey that stops the series
        included - and ``qc_report.txt`` sums them up at the end, naming the
        reciprocal files ``left_out`` for want of a forward file. Once the
        series' model is known each log is written again with it at the top,
        as Craig Ulrich's logs carry it (:meth:`_finish_series_qc`).
        """
        from PyHydroGeophysX.qt_apps import ert_records
        from PyHydroGeophysX.qt_apps.run_records import QC_FOLDER

        log = log or (lambda _message: None)
        staging = Path(staging)
        staging.mkdir(parents=True, exist_ok=True)
        out_files: List[str] = []
        steps: List[Dict[str, Any]] = []
        reports: List[Dict[str, Any]] = []
        stamps = list(acquired or [])
        partners = list(partners or [])
        paired_series = any(partners)
        reader = reader or f"{instrument or 'auto-detected format'}" + (
            f", electrodes from {Path(electrode_file).name}" if electrode_file else "")

        for index, source in enumerate(files):
            partner = partners[index] if index < len(partners) else None
            data, report, reasons = cls._prepare_survey(
                source, partner, instrument, electrode_file, qc,
                paired_series=paired_series, log=log)
            kept, total = int(report["kept"]), int(report["readings"])
            if report_dir is not None:
                ert_records.write_series_qc_logs(
                    Path(report_dir) / QC_FOLDER, index, len(files), source, report,
                    acquired=stamps[index] if index < len(stamps) else "", reader=reader,
                    fit_to=fit_to)
            reports.append(report)
            if kept < 4:
                raise ValueError(
                    f"the QC filter leaves {kept} of {total} measurements in "
                    f"{Path(source).name}, and a time-lapse step needs at least four. "
                    "Loosen the Data QC thresholds, or Reset them.")
            target = staging / f"step_{index:04d}.dat"
            data.save(str(target))
            out_files.append(str(target))
            steps.append({"file": Path(source).name, "kept": kept, "total": total,
                          "cut": list(reasons), "qc": ert_records.step_summary(report)})
            pair_note = (f" + {Path(partner).name} (forward {report['files']['forward_readings']}"
                         f" | reciprocal {report['files']['reciprocal_readings']})"
                         if partner else "")
            log(f"QC {index + 1}/{len(files)} {Path(source).name}{pair_note}: kept {kept} "
                f"of {total}" + (f" ({'; '.join(reasons)})" if reasons else ""))
        model = cls._finish_series_qc(
            files, reports, report_dir=report_dir, stamps=stamps, reader=reader, qc=qc,
            error_model=error_model, fit_model=fit_model, floor=floor,
            left_out=left_out, fit_to=fit_to, log=log)
        return {"files": out_files, "steps": steps, "model": model, "reports": reports}

    @staticmethod
    def _fitted_over(fit: Dict[str, Any], series: bool) -> str:
        """The pairs a fitted model is over, for the settings file."""
        from PyHydroGeophysX.qt_apps.ert_records import FIT_KEPT

        where = f" of {fit['surveys']} surveys" if series else ""
        if fit.get("fit_to") == FIT_KEPT and fit.get("filtered"):
            return (f"the {fit['n']} reciprocal pairs{where} the filter kept; the "
                    f"{fit['left_out']} it removed are left out")
        if fit.get("fit_to") == FIT_KEPT:
            return f"{fit['n']} reciprocal pairs{where} (no filter applied, so every pair)"
        return f"{fit['n']} reciprocal pairs{where}, as read, before any filter"

    @classmethod
    def _finish_series_qc(cls, files: List[str], reports: List[Dict[str, Any]], *,
                          report_dir: Optional[Path], stamps: List[str], reader: str,
                          qc: Optional[Dict[str, Any]], error_model: str, fit_model: bool,
                          floor: float, left_out: Optional[List[str]], fit_to: str,
                          logs_written: bool = True, log=None) -> Optional[Dict[str, Any]]:
        """Fit the series' reciprocal error model and finish its QC records.

        ``reports`` are each survey's QC report (:meth:`_prepare_survey`). One
        model over the whole series, over the pairs ``fit_to`` names - the
        fit ``error_model_lines`` reports and the figure draws - so the
        records and the run agree; returned when ``fit_model``, else None.
        With ``report_dir`` each survey's QC log is written again with the
        model at its top (written for the first time, ``logs_written`` False,
        when the surveys were read for the Reciprocal errors tab instead), and
        ``qc_report.txt`` with the figure.
        """
        from PyHydroGeophysX.qt_apps import ert_records
        from PyHydroGeophysX.qt_apps.run_records import QC_FOLDER

        log = log or (lambda _message: None)
        fitted = ert_records.fit_series_error_model(
            [r.get("pairing") or {} for r in reports], fit_to)
        model = None
        if fit_model and fitted is not None:
            model = {"m": fitted["m"], "b": fitted["b"], "r2": fitted["r2"],
                     "r2_raw": fitted["r2_raw"], "pairs": fitted["n"],
                     "surveys": fitted["surveys"], "floor": float(floor),
                     "fitted_over": cls._fitted_over(fitted, series=True)}
            log(f"Reciprocal error model over the series, fitted to "
                f"{ert_records.fit_to_sentence(fit_to, fitted['filtered'])}: "
                f"{ert_records.power_law(model)} (binned R2 {fitted['r2']:.3f}, "
                f"{fitted['n']} pairs from {fitted['surveys']} surveys); every reading's "
                f"error is dR / |R|, at least {100.0 * float(floor):g} %.")
        if report_dir is not None:
            if fitted is not None or not logs_written:
                line = "" if fitted is None else (
                    f"{ert_records.power_law(fitted)} (global fit over the series, to "
                    f"{ert_records.FIT_TO_TEXT[fitted['fit_to']]}"
                    + ("; used as the data errors)" if model else
                       "; for reference, not used as the data errors)"))
                for index, (source, report) in enumerate(zip(files, reports)):
                    ert_records.write_series_qc_logs(
                        Path(report_dir) / QC_FOLDER, index, len(files), source, report,
                        acquired=stamps[index] if index < len(stamps) else "",
                        reader=reader, model=line, fit_to=fit_to)
            path = ert_records.write_series_qc_report(
                Path(report_dir), sources=files, acquired=stamps, reader=reader, qc=qc,
                reports=reports, error_model=error_model, applied_model=model,
                left_out=list(left_out or []), fit_to=fit_to)
            log(f"Data QC report written to {path}; each survey's QC log is in "
                f"{Path(report_dir) / QC_FOLDER}")
        return model

    def _on_tl_qc_failed(self, message: str) -> None:
        """A survey could not be read or filtered, so the inversion never started."""
        self._on_tl_failed(message, False)
        self._reset_tl_button()

    def _launch_timelapse(self, run, sources: List[str], files: List[str],
                          labels: List[str], params: Dict[str, Any], times, stamps,
                          qc: Optional[Dict[str, Any]],
                          electrodes: Optional[str] = None, *,
                          fit_to: str = "all") -> None:
        """Persist the series as it will be inverted, and start the workflow.

        ``sources`` are the files as the user listed them, ``files`` what is
        inverted: the same files, or their QC-filtered copies when ``qc`` says
        what filtered them. ``electrodes`` is the electrode table every survey
        is placed on; the run keeps its own copy, so it reruns as it ran.
        ``fit_to`` is what the reciprocal error model was fitted to, for the
        settings file.
        """
        electrodes_ref = None
        # One compressed bundle rather than a copy of every step. A time-lapse
        # run is as many raw files as it has time steps, and BERT data files are
        # ASCII, so this is where the run directory grew fastest.
        try:
            stored = run_inputs.save_file_bundle(
                run.inputs_dir / "ert_timesteps",
                {
                    f"step_{index:04d}{Path(source).suffix}": Path(source)
                    for index, source in enumerate(files)
                },
                kind="ert_timelapse_observations",
                meta={
                    "measurement_times": [
                        float(times[index]) if times is not None else float(index)
                        for index in range(len(files))
                    ]
                },
            )
            if qc is not None:
                io_utils.write_json(run.inputs_dir / "ert_qc.json", {
                    **{key: value for key, value in qc.items()
                       if key not in ("partners", "left_out")},
                    "source_files": [Path(source).name for source in sources],
                    # Each survey's reciprocal file, merged into it, by name.
                    "reciprocal_files": [Path(p).name if p else None
                                         for p in (qc.get("partners") or [])],
                    "left_out": [Path(p).name for p in (qc.get("left_out") or [])]})
                # The bundle holds the filtered surveys; the loose copies go.
                shutil.rmtree(run.inputs_dir / "ert_timesteps_qc", ignore_errors=True)
            if electrodes is not None:
                table = run.inputs_dir / "electrodes_xyz.txt"
                shutil.copyfile(electrodes, table)
                electrodes_ref = ArtifactRef.from_path(
                    table,
                    artifact_id="ert-electrodes",
                    kind="electrode_geometry",
                    format="txt",
                    base_dir=run.run_dir,
                    metadata={"source_file": self._electrode_path.name
                              if self._electrode_path else ""},
                )
        except Exception as exc:  # noqa: BLE001
            self.fail_persisted_run(str(exc), "ert.timelapse_inversion")
            self.log(f"Could not persist time-lapse inputs: {exc}", "error")
            self._reset_tl_button()
            return
        metadata: Dict[str, Any] = {"source": "qt", "sequence_order_persisted": True}
        if qc is not None:
            metadata.update(qc_filtered=True, source_instrument=qc.get("source_instrument"))
        spec = WorkflowSpec(
            workflow_id="ert.timelapse_inversion",
            inputs={
                "data_bundle": ArtifactRef.from_path(
                    stored,
                    artifact_id="ert-timesteps",
                    kind="ert_timelapse_observations",
                    base_dir=run.run_dir,
                    metadata={"step_count": len(files)},
                ),
                "measurement_times": list(times or range(len(files))),
                # The bundle renames the files, so the acquisition dates parsed
                # from the originals have to travel with the times or the panels
                # end up headed by a bare elapsed-day number.
                "time_labels": list(labels),
                "timestamps": stamps,
                "time_unit": "d" if times is not None else "",
                **({"electrodes": electrodes_ref} if electrodes_ref is not None else {}),
            },
            parameters=params,
            metadata=metadata,
        )
        recipe_path, script_path = export_workflow_bundle(
            spec, run.run_dir, stem="ert_timelapse"
        )
        self._reproduce.set_bundle(recipe_path, script_path)
        self._tl_recipe_path = str(recipe_path)
        self._write_timelapse_settings(run, spec, sources, labels, stamps, times,
                                       electrodes, qc, fit_to)
        if not any(stamps):
            # A warning, so it is kept with the run: every interval reads as one
            # unit, which changes what the temporal smoothing does.
            self.log("Time-lapse: the surveys carry no acquisition times, so every "
                     "step is one unit apart.", "warn")
        self._tl_progress.setRange(0, 0)
        self.log(f"Starting {params['inversion_type']} time-lapse ERT inversion "
                 f"({len(files)} steps)…", "info")
        # Both supported time-lapse engines execute long native/GPU kernels.
        # Keep every time-lapse run outside Qt's interpreter so PyGIMLi cannot
        # retain the GIL and ADTLERT cannot monopolize the GUI CUDA context.
        # This also covers an unavailable ADTLERT request that falls back to
        # the PyHydro engine inside the workflow process.
        self._agent_run_state = "running"
        self._agent_run_error = ""
        self._tl_worker = ProcessWorkflowWorker(
            recipe_path,
            run.run_dir,
            run.outputs_dir,
            run.result_path,
        )
        self._tl_worker.logged.connect(lambda m: self.log(m, "info"))
        self._tl_worker.progressed.connect(self._on_tl_progress)
        self._tl_worker.succeeded.connect(self._on_tl_workflow_ok)
        self._tl_worker.failed.connect(lambda message: self._on_tl_failed(message, False))
        self._tl_worker.finished.connect(self._reset_tl_button)
        self.register_worker(self._tl_worker, activity="Running the time-lapse inversion")
        self._tl_worker.start()
        self._tl_stop.attach(self._tl_worker, "ert.timelapse_inversion")

    def _write_timelapse_settings(self, run, spec: WorkflowSpec, sources: List[str],
                                  labels: List[str], stamps: List[str], times,
                                  electrodes: Optional[str],
                                  qc: Optional[Dict[str, Any]],
                                  fit_to: str = "all") -> None:
        """Write the run's ``inversion_settings.txt`` from the spec it is handed
        (``qt_apps/ert_records.py``); the QC report was written as the surveys
        were filtered. A file that cannot be written is said in the log only.
        ``fit_to`` - what the error model was fitted to - is stated wherever
        the run fitted one: its QC report has the model, or it used it."""
        from PyHydroGeophysX.qt_apps import ert_records
        from PyHydroGeophysX.qt_apps.run_records import QC_REPORT_NAME

        self._tl_settings = None
        try:
            instrument = str((qc or {}).get("source_instrument")
                             or spec.parameters.get("instrument") or "auto-detected")
            model = dict(spec.parameters.get("error_model") or {}) or None
            path = ert_records.write_settings(run.run_dir, ert_records.timelapse_settings_text(
                spec, run_id=run.run_id, sources=sources, labels=labels, stamps=stamps,
                times=times, instrument=instrument,
                electrode_file=str(self._electrode_path or electrodes or ""), qc=qc,
                partners=(qc or {}).get("partners"), left_out=(qc or {}).get("left_out") or (),
                data_errors=self._error_model_text(True, model)
                + ("; pairs averaged by the QC carry their own difference"
                   if ((qc or {}).get("thresholds") or {}).get("average_reciprocals")
                   and not model else ""),
                fit_to=fit_to if model or (Path(run.run_dir) / QC_REPORT_NAME).is_file()
                else None))
            # With what was asked, to say afterwards what the engine changed.
            self._tl_settings = (path, dict(spec.parameters))
            self.log(f"Inversion settings written to {path}", "info")
        except Exception as exc:  # noqa: BLE001 - a record must not stop the run
            self.log(f"Could not write the run's settings file: {exc}", "warn")

    def _on_tl_progress(self, current: int, total: int, label: str) -> None:
        """Show completed ADTLERT windows while retaining the text log."""
        if total <= 0:
            return
        self._tl_progress.setRange(0, total)
        self._tl_progress.setValue(max(0, min(current, total)))
        self._tl_progress.setFormat(label or f"Progress {current}/{total}")

    def _on_tl_workflow_ok(self, result: WorkflowRunResult) -> None:
        if hasattr(self.state, "update_workflow_result"):
            self.state.update_workflow_result(
                self.module_key,
                "ert.timelapse_inversion",
                result.to_dict(),
                recipe_path=self._tl_recipe_path,
            )
        payload = result.legacy_payload()
        settings = self._tl_settings
        if settings is not None:
            self._record_outcome(settings[0], "timelapse", payload, settings[1])
        if payload.get("mesh") is None and payload.get("model_bundle"):
            payload.update(self._load_timelapse_bundle(payload["model_bundle"]))
        self._on_tl_ok(payload)

    def _load_timelapse_bundle(self, bundle: Dict[str, Any]) -> Dict[str, Any]:
        """Load process-safe time-lapse arrays for the interactive Qt viewer."""
        try:
            from PyHydroGeophysX.core.mesh_serialization import read_bms

            base = (
                Path(self._tl_recipe_path).resolve().parent
                if self._tl_recipe_path else Path.cwd()
            )

            def resolve(value: Any) -> Path:
                path = Path(str(value))
                return path if path.is_absolute() else base / path

            paths = dict(bundle)
            mesh_path = resolve(paths.get("mesh", ""))
            models_path = resolve(paths.get("models", ""))
            if not mesh_path.is_file() or not models_path.is_file():
                raise FileNotFoundError(
                    f"missing mesh or model bundle ({mesh_path}, {models_path})"
                )
            coverage_path = resolve(paths.get("coverage", ""))
            # Read into memory rather than memory-mapping. A mapping stays open for
            # as long as the viewer holds the array, and Windows refuses to let the
            # next run overwrite a mapped file: np.save then fails with
            # "OSError: [Errno 22] Invalid argument" after the inversion has already
            # finished, leaving the previous run's arrays on disk beside this run's
            # figures. These are per-cell result arrays, small enough that mapping
            # bought nothing.
            return {
                # Without the cell-neighbour table, as in _load_model_bundle.
                "mesh": read_bms(mesh_path, neighbours=False),
                "final_models": np.load(models_path, allow_pickle=False),
                "coverage": (
                    np.load(coverage_path, allow_pickle=False)
                    if coverage_path.is_file() else None
                ),
            }
        except Exception as exc:  # noqa: BLE001 - retain scalar results on failure
            self.log(f"Could not load time-lapse model files: {exc}", "error")
            return {"mesh": None, "final_models": None, "coverage": None}

    def _on_tl_ok(self, result: dict) -> None:
        self._agent_run_state = "completed"
        self._agent_run_error = ""
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
        self._tl_correction = None
        self._tl_models_raw = None
        if self._tl_models is not None and self._tl_mesh is not None:
            self._tl_models_raw = np.asarray(self._tl_models, dtype=float)
            n_steps = int(self._tl_models_raw.shape[1])
            days, dates = temperature_panel.series_survey_times(
                self._tl_summary(), n_steps)
            self._tc_box.set_context(n_steps, dates is not None, days=days, dates=dates)
            self._tc_box.set_available(True)
            self._tc_box.show_applied(None)

        self._populate_tl_steps()
        engine = str(result.get("engine") or "pyhydro")
        requested_engine = str(result.get("engine_requested") or engine)
        solver = str(result.get("linearized_solver") or "")
        compute_note = "Joint χ² over all time steps."
        if engine == "e4d":
            compute_note = ("E4D time-lapse: χ² of each survey in turn, the last one "
                            "shown; each started from the solution before it.")
        elif engine in ("r2", "r3t"):
            compute_note = (f"{'R2' if engine == 'r2' else 'R3t'} difference inversion: χ² "
                            "of each survey in turn, the last one shown; each later survey "
                            "inverted against the first. λ is not used.")
        elif engine == "adtlert":
            compute_note += f" cuDSS forward · {solver or 'gpu_cgls'}."
        elif requested_engine == "adtlert":
            compute_note += " ADTLERT was unavailable; original PyHydro ERT used."
        self._quality_view.show_quality(
            {"chi2": result.get("chi2"), "iterations": len(result.get("chi2_history") or []) or None,
             "n_data": result.get("n_data"),
             # R2 and R3t choose their own smoothing; a λ shown would be one never used.
             "lambda": None if engine in ("r2", "r3t") else self._lam.value(),
             "method": (f"{engine} time-lapse "
                        f"{result.get('inversion_type', '')} "
                        f"({result.get('n_times')} steps)"),
             "note": compute_note},
            result.get("chi2_history"), title="Time-lapse ERT inversion")
        lowmem = " · low-memory" if result.get("save_memory") else ""
        n_vtk = len(result.get("vtk_step_paths") or [])
        self.log(f"Time-lapse inversion complete "
                 f"({engine}, {result.get('mode')}{lowmem}): "
                 f"{result.get('n_times')} steps, {result.get('mesh_cells')} cells. "
                 f"Saved VTK (combined + {n_vtk} per-step), npy, mesh. "
                 f"Pick a step in the Resistivity model tab; “Export results…” saves them.", "success")
        if requested_engine == "adtlert" and engine != "adtlert":
            self.log(
                "ADTLERT was requested but the time-lapse run used the original "
                "PyHydro ERT engine.",
                "warn",
            )
        elif engine == "adtlert":
            self.log(
                f"Compute backend: ADTLERT · cuDSS forward · {solver or 'gpu_cgls'}.",
                "info",
            )
        e4d = dict(result.get("e4d") or {})
        if e4d.get("run_dir"):
            self.log(f"E4D's own files (e4d.log, tl_sig*) and the 3D models "
                     f"resistivity_3d_timelapse.vtk are in {e4d['run_dir']}", "info")
            if e4d.get("missing_fit"):
                self.log("E4D overwrote the simulated data of survey(s) "
                         + ", ".join(str(int(s) + 1) for s in e4d["missing_fit"])
                         + " before they could be kept: their models are shown, their "
                         "fit is not known.", "warn")
        for engine_key, name in (("r2", "R2"), ("r3t", "R3t")):
            own = dict(result.get(engine_key) or {})
            if own.get("run_dir"):
                self.log(f"{name}'s own files: the baseline in {own.get('baseline_run_dir')}, "
                         f"the later surveys (f002_res.dat, ...) in {own['run_dir']}", "info")
                stops = list(own.get("stop") or [])
                if "iteration_cap" in stops:
                    self.log(f"{name} reached the iteration limit on survey(s) "
                             + ", ".join(str(i + 1) for i, s in enumerate(stops)
                                         if s == "iteration_cap")
                             + "; raise Max iterations to let them converge.", "warn")
        self.report_result(result)
        self.offer_map_export()

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
        self._tl_view_mode.blockSignals(True)
        self._tl_view_mode.setCurrentIndex(0)
        self._tl_view_mode.blockSignals(False)
        self._tl_step_row.setVisible(n > 1)
        self._seed_tl_color_range()
        self._show_tl_step(0)
        self._tabs.setCurrentWidget(self._model_tab)

    def _step_tl(self, delta: int) -> None:
        n = self._tl_step_combo.count()
        if n:
            self._tl_step_combo.setCurrentIndex((self._tl_step_combo.currentIndex() + delta) % n)

    def _on_tl_view_mode_changed(self, _index: int) -> None:
        """Switch between absolute resistivity and change from the baseline."""
        self._seed_tl_color_range()
        self._show_tl_step(self._tl_step_combo.currentIndex())

    @staticmethod
    def _percent_change(models, idx: int):
        """Percentage change of step ``idx`` against the first survey.

        The definition lives in the shared series widget, so this panel and the
        saved-results browser cannot drift into two different percentages.
        """
        from PyHydroGeophysX.qt_apps.widgets.series_view import percent_change

        return percent_change(models, idx)

    def _seed_tl_color_range(self) -> None:
        """Put every time step on one colour scale, for the mode on screen.

        The viewer sees a single step at a time, so left alone it autoscales each
        one to its own extremes and the change between steps is rescaled away.
        Limits come from the whole series at the 2–98 percentile, matching the
        exported summary figure, so a handful of poorly covered cells cannot
        flatten everything else.
        """
        from PyHydroGeophysX.qt_apps.widgets.series_view import series_color_limits

        if self._tl_models is None:
            return
        limits = series_color_limits(
            self._tl_models, str(self._tl_view_mode.currentData() or "model"))
        if limits is not None:
            self._model_view.set_color_range(*limits)

    def _show_tl_step(self, idx: int) -> None:
        import numpy as np
        if self._tl_models is None or self._tl_mesh is None:
            return
        models = np.asarray(self._tl_models, dtype=float)
        if idx < 0 or idx >= models.shape[1]:
            return
        self._map_result_kind = 'timelapse'
        cov = None
        if self._tl_coverage is not None:
            cov_all = np.asarray(self._tl_coverage, dtype=float)
            if cov_all.ndim == 2 and idx < cov_all.shape[0] and cov_all.shape[1] == models.shape[0]:
                cov = cov_all[idx]  # raw log-coverage, matching ERTManager.coverage()
        title = self._tl_step_titles[idx] if idx < len(self._tl_step_titles) else f"Time step {idx + 1}"
        if self._tl_view_mode.currentData() == "change":
            baseline = (self._tl_step_titles[0] if self._tl_step_titles
                        else "Time step 1")
            values, kind = self._percent_change(models, idx), "change"
            title = f"{title} − {baseline}" if idx else f"{title} (baseline)"
        else:
            values, kind = models[:, idx], "ert"
        # A corrected section looks exactly like an uncorrected one, so its
        # title says which it is.
        title += temperature_panel.title_suffix(self._tl_correction)
        self._model_view.show_field(self._tl_mesh, values, kind=kind, coverage=cov, title=title)

    def _on_tl_failed(self, message: str, backend: bool) -> None:
        self._agent_run_state = "failed"
        self._agent_run_error = message
        self.fail_persisted_run(message, "ert.timelapse_inversion")
        text = f"Time-lapse inversion {'unavailable' if backend else 'failed'}: {message}"
        self.log(text, "warn" if backend else "error")
        # On the panel too: a survey its reader refused is the usual cause, and
        # the reason belongs where the files are listed, not only in the log.
        self._tl_info.setText(text)

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

    class _ExportWorker(TaskWorker):
        """A TaskWorker that reports ``progressed(current, total, label)``.

        The page's status bar shows that line while the export runs.
        """

        progressed = Signal(int, int, str)

        def __init__(self, fn: Callable[..., Any], **kwargs: Any) -> None:
            super().__init__(fn, **kwargs)
            self._kwargs["progress"] = self.progressed.emit

    def _export_tl_results(self, folder: Optional[str] = None) -> Optional[str]:
        """Copy the time-lapse result files (VTK, npy, mesh, CSV, figure) to a folder.

        Chosen here, on the page, the export runs on a worker thread with its
        progress in the status bar: for a long series it is hundreds of VTK
        files and a table of millions of numbers, which held the window still
        for half a minute. Given a ``folder`` - the assistant's export - it runs
        to the end before returning, as that caller expects.
        """
        if not self._tl_result:
            self.log("Run the time-lapse inversion first.", "warn")
            return None
        files = self._tl_result_files()
        if not files:
            self.log("No time-lapse result files found to export.", "warn")
            return None
        interactive = not folder
        if not folder:
            selected = select_directory(
                self, "Export time-lapse results to folder",
                self.state.output_dir or Path.cwd(),
            )
            folder = str(selected) if selected else ""
            if not folder:
                return None
        dest = io_utils.ensure_dir(Path(folder))
        # What is on screen now, so a step or a correction changed while the
        # export runs does not reach half of it.
        job = dict(
            files=files, src_root=Path(self._tl_result.get("output_dir") or ""), dest=dest,
            mesh=self._tl_mesh,
            models=None if self._tl_models is None else np.array(self._tl_models, dtype=float),
            coverage=self._tl_coverage, step_labels=list(self._tl_step_titles or []) or None,
            correction=dict(self._tl_correction) if self._tl_correction else None,
        )
        if not interactive:
            self._on_tl_exported(self._export_timelapse(**job))
            return str(dest)
        worker = self._ExportWorker(self._export_timelapse, **job)
        worker.succeeded.connect(self._on_tl_exported)
        worker.failed.connect(
            lambda message: self.log(f"Time-lapse export failed: {message}", "error"))
        worker.finished.connect(lambda: self._tl_export_btn.setEnabled(bool(self._tl_result)))
        self._tl_export_btn.setEnabled(False)
        self.log(f"Exporting the time-lapse results to {dest}…", "info")
        self.register_worker(worker, activity="Exporting the time-lapse results")
        worker.start()
        return str(dest)

    def _on_tl_exported(self, summary: Dict[str, Any]) -> None:
        """Report a finished time-lapse export."""
        for warning in summary.get("warnings") or []:
            self.log(warning, "warn")
        self.log(f"Exported {summary.get('files', 0)} time-lapse result file(s) to "
                 f"{summary.get('dest', '')}", "success")
        correction = summary.get("correction")
        if correction:
            reference = float(correction.get("reference_temperature_C", 25.0))
            self.log(
                f"The series on screen, corrected to {reference:g} °C, is in "
                f"final_models_temperature_corrected.npy, "
                f"timelapse_resistivity_temperature_corrected.vtk and the cell CSV; "
                f"temperature_correction.json records what was applied. "
                f"final_models.npy, the per-step VTKs and the figure are the "
                f"models as inverted.", "info")

    @staticmethod
    def _export_timelapse(*, files: List[str], src_root: Path, dest: Path, mesh: Any,
                          models: Any, coverage: Any, step_labels: Optional[List[str]],
                          correction: Optional[Dict[str, Any]],
                          progress: Optional[Callable[[int, int, str], None]] = None,
                          ) -> Dict[str, Any]:
        """Write a time-lapse export into ``dest``; safe to run off the GUI thread.

        Copies the run's files, keeping the ``vtk_steps`` folder, then writes
        the per-cell table and, with a temperature correction on screen, the
        corrected series. Touches no widget: what went wrong comes back in the
        summary's ``warnings`` for the page to log.
        """
        report = progress or (lambda current, total, label: None)
        warnings: List[str] = []
        total = len(files)
        every = max(1, -(-total // 20))          # about twenty updates, however many files
        copied = 0
        report(0, total, f"Exporting: copying files 0/{total}")
        for index, f in enumerate(files, start=1):
            try:
                src = Path(f)
                # Preserve the vtk_steps/ subfolder so per-step VTKs stay grouped.
                rel = src.relative_to(src_root) if src_root and src_root in src.parents else Path(src.name)
                target = dest / rel
                io_utils.ensure_dir(target.parent)
                shutil.copy2(str(src), str(target))
                copied += 1
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"Could not copy {Path(f).name}: {exc}")
            if index % every == 0 or index == total:
                report(index, total, f"Exporting: copying files {index}/{total}")
        written = 0
        # The per-cell table: the copied files are all PyGIMLi meshes and NumPy
        # arrays, and this is the one export a collaborator can open without
        # either, with one row per cell and one column per time step.
        if mesh is None or models is None:
            warnings.append("Time-lapse models are not in memory, so no CSV was written; "
                            "the mesh and .npy files were still copied.")
        else:
            report(0, 1, "Exporting: writing the per-cell table (CSV)")
            try:
                from PyHydroGeophysX.data_processing.model_csv import export_model_csv

                # The table holds the series on screen; with a correction applied
                # its columns say the temperature they are reported at.
                written += len(export_model_csv(
                    dest, mesh, models, value_name=_value_name(correction), units="ohm.m",
                    coverage=coverage, step_labels=step_labels))
            except Exception as exc:  # noqa: BLE001 - the copies already succeeded
                warnings.append(f"Could not write the time-lapse CSV: {exc}")
        # The temperature-corrected series beside the inverted one. The files
        # copied from the run are the models as inverted; with a correction on
        # screen, an export of only those would hand over a different series
        # from the one being looked at, and nothing in it would say so.
        if correction and models is not None:
            report(0, 1, "Exporting: writing the temperature-corrected series")
            series = np.asarray(models, dtype=float)
            try:
                np.save(dest / "final_models_temperature_corrected.npy", series)
                written += 1
                io_utils.write_json(dest / "temperature_correction.json", correction)
                written += 1
                try:
                    import pygimli as pygimli

                    from PyHydroGeophysX.core.mesh_serialization import via_ascii_path

                    combined = pygimli.Mesh(mesh)
                    for index in range(series.shape[1]):
                        combined[f"resistivity_t{index}"] = series[:, index]
                    via_ascii_path(combined.exportVTK,
                                   dest / "timelapse_resistivity_temperature_corrected.vtk",
                                   mode="write")
                    written += 1
                except Exception as exc:  # noqa: BLE001 - the arrays are already written
                    warnings.append(f"Temperature-corrected VTK skipped: {exc}")
            except Exception as exc:  # noqa: BLE001 - the copies already succeeded
                warnings.append(f"Could not write the temperature-corrected models: {exc}")
        return {"dest": str(dest), "files": copied + written, "warnings": warnings,
                "correction": correction}

    def _open_tl_output(self) -> None:
        out = self._tl_out or str(self.state.output_dir or "")
        if out and Path(out).exists():
            from PySide6.QtCore import QUrl
            from PySide6.QtGui import QDesktopServices
            QDesktopServices.openUrl(QUrl.fromLocalFile(out))
        else:
            self.log("No time-lapse output yet.", "warn")

    # -- pseudosection -------------------------------------------------------
    def _paint_pseudo_scale(self) -> None:
        """Draw the legend's colour bar in the pseudosection's colour map."""
        stops = []
        fractions = np.linspace(0.0, 1.0, 9)
        colours = self._cmap.map(fractions, mode="byte")
        for fraction, colour in zip(fractions, colours):
            stops.append(
                f"stop:{fraction:.3f} rgb({int(colour[0])},"
                f"{int(colour[1])},{int(colour[2])})"
            )
        self._pseudo_scale_bar.setStyleSheet(
            "QFrame { border: 1px solid #8b949e; border-radius: 2px; "
            "background: qlineargradient(x1:0, y1:0, x2:1, y2:0, "
            + ", ".join(stops)
            + "); }"
        )

    def _on_pseudo_colormap_changed(self, name: str) -> None:
        """Recolour the pseudosection from the measurements already loaded."""
        self._cmap = cmaps.to_pyqtgraph(name)
        self._paint_pseudo_scale()
        if self._pseudo:
            self._draw_pseudosection()

    def _label_electrode_axes(self) -> None:
        """Label the electrode plot's axes in the studio's length unit.

        A line read without elevations puts every electrode at z = 0, and
        calling that axis an elevation would present a made-up datum as a
        measured one.
        """
        z = np.asarray(self._z, dtype=float)
        name = "Elevation" if z.size and np.any(np.abs(z) > 1.0e-6) else "z"
        key = (length_units.current(), name)
        if key == self._electrode_axes:
            return
        length_units.pyqtgraph_axis(self._plot, "bottom", "x")
        length_units.pyqtgraph_axis(self._plot, "left", name)
        self._electrode_axes = key

    def _on_length_unit_changed(self, _unit: str) -> None:
        """Relabel the electrode plot and redraw the pseudosection in the new unit."""
        self._label_electrode_axes()
        if self._pseudo:
            self._draw_pseudosection()

    def _show_pseudosection_message(self, message: str) -> None:
        """Show a stable empty-state message on the static section canvas."""
        self._pseudo_ax.clear()
        self._pseudo_ax.set_axis_off()
        if message:
            self._pseudo_ax.text(
                0.5, 0.5, message,
                ha="center", va="center", transform=self._pseudo_ax.transAxes,
            )
        self._pseudo_legend.setVisible(False)
        self._pseudo_canvas.draw_idle()

    def _draw_pseudosection(self) -> None:
        if not self._pseudo:
            self._show_pseudosection_message(
                "No apparent-resistivity measurements loaded."
            )
            return
        arr = np.asarray(self._pseudo, dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 3:
            self._show_pseudosection_message(
                "The loaded pseudosection does not contain x, depth, and rhoa columns."
            )
            return
        mid, depth, rhoa = arr[:, 0], arr[:, 1], arr[:, 2]
        valid = (
            np.isfinite(mid)
            & np.isfinite(depth)
            & (depth >= 0.0)
            & np.isfinite(rhoa)
            & (rhoa > 0.0)
        )
        mid, depth, rhoa = mid[valid], depth[valid], rhoa[valid]
        if rhoa.size == 0:
            self._show_pseudosection_message(
                "All x/depth values are invalid, or apparent resistivity is missing, "
                "non-finite, or non-positive."
            )
            return
        log_rhoa = np.log10(rhoa)
        lo, hi = np.percentile(log_rhoa, [3, 97])
        if hi <= lo:
            lo, hi = float(lo) - 0.5, float(hi) + 0.5
        rng = hi - lo
        norm = np.clip((log_rhoa - lo) / rng, 0.0, 1.0)
        lut = self._cmap.map(norm, mode="byte")
        x_min, x_max = float(np.min(mid)), float(np.max(mid))
        x_span = x_max - x_min
        if x_span <= 1e-9:
            x_center = 0.5 * (x_min + x_max)
            x_min, x_max = x_center - 0.5, x_center + 0.5
        else:
            x_pad = 0.03 * x_span
            x_min, x_max = x_min - x_pad, x_max + x_pad
        depth_max = max(float(np.max(depth)) * 1.08, 0.02)

        self._pseudo_ax.clear()
        self._pseudo_ax.set_axis_on()
        rgba = np.ones((len(lut), 4), dtype=float)
        rgba[:, :3] = np.asarray(lut[:, :3], dtype=float) / 255.0
        self._pseudo_ax.scatter(
            mid,
            depth,
            s=34,
            c=rgba,
            marker="o",
            edgecolors="#394b59",
            linewidths=0.35,
        )
        self._pseudo_ax.set_xlim(x_min, x_max)
        self._pseudo_ax.set_ylim(depth_max, 0.0)
        # The y values are already depths below the surface, so only the unit
        # changes; the inverted axis keeps them positive down.
        set_length_axis(self._pseudo_ax, "x", "x")
        set_length_axis(self._pseudo_ax, "y", "Pseudo-depth")
        self._pseudo_ax.grid(True, which="major", alpha=0.28)
        self._pseudo_ax.minorticks_on()
        self._pseudo_ax.grid(True, which="minor", alpha=0.10)
        self._pseudo_ax.set_title(
            f"Apparent resistivity: {rhoa.min():.3g} – {rhoa.max():.3g} Ω·m "
            f"(n={rhoa.size})"
        )
        self._pseudo_canvas.draw_idle()
        legend_values = np.power(10.0, np.linspace(float(lo), float(hi), 5))
        for label, value in zip(self._pseudo_scale_labels, legend_values):
            label.setText(f"{float(value):.4g}")
        self._pseudo_legend.setVisible(True)

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
        self._label_electrode_axes()
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
        # A file read under the wrong format still fills this panel with a
        # plausible electrode and measurement count, so the reason it cannot be
        # inverted belongs next to those numbers and not only in the log.
        note_txt = ""
        if self._data_note:
            note_txt = (f"<br><span style='color:{theme.color('red')}'>Not usable for "
                        f"inversion: {self._data_note} Check the Instrument / "
                        f"format setting.</span>")
        # When the survey was measured, and from what: the same reading the file
        # list and the model heading use, so the three never disagree.
        stamp, where = self._survey_time(self._data_path)
        when_txt = (f"<br>Acquired: {survey_timing.format_time(stamp, seconds=True)} "
                    f"(from the {where})" if stamp is not None else "")
        partner_txt = (f"<br>+ reciprocal file: {self._data_partner.name}"
                       if self._data_partner is not None else "")
        self._info.setText(
            f"Electrodes: {len(self._x)} &nbsp; Measurements: {self._n_meas}"
            f"<br>Data: {self._data_path.name if self._data_path else '—'}{partner_txt}"
            f"{when_txt}{rhoa_txt}{note_txt}"
        )
        self._publish()
        # Every change of data or electrodes passes through here, and the mesh
        # is built around the electrodes.
        self._mesh_inputs_changed()

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

            from PyHydroGeophysX.core.mesh_serialization import via_ascii_path
            from PyHydroGeophysX.data_processing.model_csv import export_model_csv

            out = io_utils.ensure_dir(folder)
            mesh = mgr.paraDomain
            model = np.asarray(mgr.model, dtype=float)  # resistivity (ohm-m)
            # The table holds what is on screen, so a temperature-corrected model
            # is exported as one, with the temperature in its column name.
            shown = self._inv_corrected if self._inv_corrected is not None else model
            np.save(out / "resistivity_model.npy", model)
            try:
                cov = np.asarray(mgr.coverage(), dtype=float)
                np.save(out / "coverage.npy", cov)
            except Exception:  # noqa: BLE001 - coverage is optional
                cov = None
            # The CSV is written before the PyGIMLi files because it is the one
            # a collaborator without PyGIMLi can actually open, and a .bms write
            # that fails should not cost them the table as well.
            export_model_csv(
                out, mesh, shown,
                value_name=_value_name(self._inv_correction if shown is not model else None),
                units="ohm.m", coverage=cov,
            )
            # numpy takes wide paths, PyGIMLi's writers do not; a folder Windows'
            # ANSI codepage cannot represent needs the write staged through a
            # temporary ASCII one. Export to a localized "文档" folder otherwise
            # writes the .npy and then fails on the .bms.
            via_ascii_path(mesh.save, out / "resistivity_mesh.bms", mode="write")
            mesh["resistivity"] = model
            via_ascii_path(mesh.exportVTK, out / "resistivity_model.vtk", mode="write")
            if shown is not model:
                # Beside the inverted model, never in place of it.
                np.save(out / "resistivity_model_temperature_corrected.npy", shown)
                io_utils.write_json(out / "temperature_correction.json", self._inv_correction)
                mesh["resistivity"] = shown
                via_ascii_path(mesh.exportVTK,
                               out / "resistivity_model_temperature_corrected.vtk",
                               mode="write")
                mesh["resistivity"] = model
                self.log(
                    f"The model on screen, corrected to "
                    f"{float(self._inv_correction['reference_temperature_C']):g} °C, is "
                    f"in resistivity_model_temperature_corrected.npy/.vtk and the cell "
                    f"CSV; temperature_correction.json records what was applied. "
                    f"resistivity_model.npy/.vtk are the model as inverted.", "info")
            self.log(f"Exported resistivity model (csv + npy + bms + vtk) to {out}", "success")
        except Exception as exc:  # noqa: BLE001
            self.log(f"Resistivity model export failed: {exc}", "error")

    def export_actions(self):
        actions = []
        if getattr(self, "_inv_mgr", None) is not None or self._tl_result:
            actions.append(("Add displayed result to Project Map…", self.add_to_map))
        if getattr(self, "_inv_mgr", None) is not None:
            actions.append((
                "Resistivity model (CSV + npy + mesh + VTK)",
                self._export_resistivity_model,
            ))
        if self._tl_result:
            actions.append((
                "Time-lapse results (CSV + npy + mesh + VTK + figures)",
                self._export_tl_results,
            ))
        if self._x:
            actions.append(("Electrode positions (CSV)", self._export_electrodes))
            actions.append(("Survey geometry (JSON)", self._export_geometry))
        return actions

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
    def show_run_inputs(self, inputs: Dict[str, Any]) -> str:
        """Open the automatic run's surveys here so this panel is not empty.

        The workflow inverts its own copy in its own process; this loads the
        same files for display, through the same handlers the assistant uses,
        so the pseudosection and electrode layout are on screen while the run
        works. Nothing is inverted here.

        Anything already loaded is left alone: a run must not throw away work
        somebody was in the middle of.
        """
        if self._ert_data is not None or self._tl_files:
            return ""
        surveys = [str(p) for p in (inputs.get("time_lapse_files") or [])]
        single = inputs.get("data_file")
        electrodes = inputs.get("electrode_file")
        # The run's instrument, not this panel's. There is no format auto-detect
        # anywhere here - the device is chosen, not guessed - so leaving the
        # dropdown on its own default read the run's files with the wrong
        # reader: an E4D .ohm parsed as a unified file has its leading index
        # column taken for electrode A, and every quadrupole then lands out of
        # bounds. The panel listed the files and drew nothing.
        instrument = str(inputs.get("instrument") or "").strip()
        if not instrument and (surveys or single):
            # Nobody chose one, so the header decides - as it does in the run.
            instrument = self._instrument_from_header((surveys or [str(single)])[0])
        if instrument:
            self._agent_set_instrument(instrument)
        # Electrodes first: a survey file carrying no geometry of its own picks
        # them up as it loads, and loading them afterwards would not be applied.
        if electrodes:
            self._agent_load_electrodes(str(electrodes))
        if surveys:
            if self._agent_add_timelapse_files(surveys).get("status") != "ok":
                return ""
            self._agent_preview_timelapse(0)
            return (f"{len(surveys)} surveys, showing "
                    f"{Path(surveys[0]).name}")
        if single:
            if self._agent_load(str(single), None).get("status") == "failed":
                return ""
            return Path(str(single)).name
        return ""

    def show_run_stage(self, tool: str) -> str:
        """Follow the run through this page's own tabs.

        Loading belongs on the pseudosection, a finished inversion on the model,
        and the quality check on the quality view. Without this the page stayed
        on whichever tab it opened with while the run moved on underneath it.
        """
        views = {
            "load_ert_surveys": (self._pseudo_widget, "Pseudosection"),
            "invert_ert": (self._model_tab, "Resistivity model"),
            "invert_time_lapse": (self._model_tab, "Resistivity model"),
            "evaluate_inversion": (self._quality_view, "Inversion quality"),
        }
        widget, name = views.get(str(tool), (None, ""))
        if widget is None:
            return ""
        self._tabs.setCurrentWidget(widget)
        return name

    @staticmethod
    def _instrument_from_header(path: str) -> str:
        """The instrument a file's header names, as the run reads it; "" if none."""
        try:
            from PyHydroGeophysX.agents.ert_loader_agent import ERTLoaderAgent

            return str(ERTLoaderAgent()._detect_instrument_from_header(str(path)) or "")
        except Exception:  # noqa: BLE001 - nothing detected is no choice made
            return ""

    def _agent_set_instrument(self, instrument: str) -> bool:
        """Point the format dropdown at ``instrument``. True when it matched."""
        key = str(instrument).strip().lower()
        for index, (label, value) in enumerate(_INSTRUMENTS):
            if value is not None and key in (value.lower(), label.lower()):
                self._instrument.setCurrentIndex(index)
                return True
        return False

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
                {"name": "apply_filter",
                 "args": {"min_rhoa": "float", "max_rhoa": "float", "max_error": "float (%)",
                          "drop_nonpositive_rhoa": "bool",
                          "min_voltage": "float (V)",
                          "min_current": "float (A)",
                          "max_geometric_factor": "float",
                          "max_contact_resistance": "float (ohm)",
                          "max_stacking_spread": "float (fraction)",
                          "max_reciprocal_error": "float (fraction)",
                          "average_reciprocal_pairs": "bool (optional)"},
                 "desc": ("Filter measurements by apparent resistivity range and max relative "
                          "error (20 % unless set; 0 is off). average_reciprocal_pairs=true "
                          "first averages each reading with its reciprocal into one reading "
                          "(mean resistance, the pair's difference as its error, never below "
                          "reciprocal_error_floor) - check 0, off by default, for data "
                          "measured both ways. The optional criteria turn on "
                          "'More checks'; each is skipped where the loaded file does not "
                          "carry the field it tests. With More checks on, a criterion the "
                          "call does not name keeps the panel's value, which starts at: "
                          "drop rhoa <= 0, |V| >= 1e-5 V, |I| >= 1e-4 A, |k| <= 20 x the "
                          "file's median, contact R <= 30000 ohm, stacking spread <= 0.5, "
                          "reciprocal error <= 0.05. Pass 0 (false for the rhoa check) to "
                          "switch one off.")},
                {"name": "set_params", "args": {"params": {"<key>": "value"}},
                 "desc": ("Set parameters. Shared by single + time-lapse inversion: lambda, "
                          "max_iterations, relative_error, time_lapse (bool). "
                          "Mesh (the Mesh tab; each maps to a PyGIMLi parameter-mesh "
                          "option): mesh_quality (smallest triangle angle, 20-34), "
                          "para_depth (m, 0 = auto), para_max_cell_size (m^2, 0 = no "
                          "limit), para_boundary (inverted region past the end "
                          "electrodes, in electrode spacings, 0.5-20, default 2), "
                          "surface_nodes (nodes between neighbouring electrodes, 1-8), "
                          "outer_width (outer region in electrode spreads, 0 = PyGIMLi's "
                          "4), outer_max_cell_size (m^2, 0 = no limit); a value out of "
                          "range is refused, not clipped. mesh_file (path to a "
                          ".bms/.msh/.vtk/.poly mesh, an E4D mesh's .1.node/.1.ele, or "
                          "an E4D mesh configuration .cfg - meshed the way E4D does it - "
                          "to invert on instead of a generated one; the region to invert "
                          "needs marker 2 or above, and '' goes back to generating it). "
                          "A-priori zones: zones, a list of {name, polygon: [[x, z], ...] "
                          "in mesh coordinates (x along the line, z the elevation), "
                          "resistivity (ohm-m), fixed (bool)}; it replaces the zones on "
                          "the Mesh tab, and [] removes them. Every engine starts from "
                          "them; the in-house pyhydro engine also regularizes toward "
                          "them and does not invert a fixed zone, and the ADTLERT "
                          "time-lapse backend takes no zone values. conform_to_zones "
                          "(bool) rebuilds a generated mesh along the zone outlines, so "
                          "no cell straddles one; decouple_zones (bool) drops the "
                          "smoothness across the outlines so the model may jump there - "
                          "every engine honours it. "
                          "Error model: error_source (file/estimate/max, "
                          "or stack - each reading's stacking spread with the estimate in "
                          "quadrature, for data that record it - or reciprocal - each "
                          "reading's error from the power law dR = 10^b * R^m fitted to the "
                          "reciprocal pairs: the survey's own for a single inversion, one fit "
                          "over every survey for time-lapse; for data measured both ways), "
                          "reciprocal_error_floor (fraction, default 0.01: the smallest error "
                          "the model or an averaged pair gives), "
                          "error_model_fit ('all', the default: the model is fitted to every "
                          "reciprocal pair as read; 'kept': only to the pairs the applied QC "
                          "filter kept - its reciprocal-error limit removes outlier pairs; "
                          "the Reciprocal errors tab's Fit to, which every fit of the model "
                          "follows), "
                          "absolute_error (Ohm). pair_reciprocal_files (bool, default true): "
                          "a forward file and its reciprocal file in the list ('recip' in the "
                          "name, measured just after) are one survey; false makes every file "
                          "a survey of its own. Convergence: plateau_tolerance (fraction), "
                          "max_total_iterations, engine (pyhydro/pygimli/adtlert/e4d/r2/"
                          "r3t; e4d is PNNL's external 3D code, set up in the E4D rows the "
                          "page shows for it, and runs on Linux, or on Windows only "
                          "inside WSL 2 - elsewhere it writes the E4D run folder and "
                          "stops; r2 (2D profiles) and r3t (3D) are Binley's programs "
                          "that ResIPy runs, found in ResIPy, native on Windows and "
                          "through Wine elsewhere; they choose their own smoothing, so "
                          "lambda and auto_lambda do not apply to them). "
                          "Mode: inversion_mode ('quick', the default, skips the k check "
                          "and the lambda search; 'full' runs both). It is a preset, so "
                          "geometric_factor_policy or auto_lambda sent after it in the "
                          "same object override it. "
                          "Geometric factors: geometric_factor_policy "
                          "('fix' recomputes k numerically when a homogeneous forward run "
                          "does not return the model resistivity, 'check' only reports, "
                          "'off' skips), geometric_factor_tolerance. "
                          "Outlier rejection: reject_outliers (bool), outlier_threshold "
                          "(sigma), outlier_passes, min_data_fraction. "
                          "Auto-lambda: auto_lambda (bool), target_chi2, chi2_tolerance, "
                          "max_lambda_trials. "
                          "Time-lapse-only: tl_alpha, tl_norm (L2/L1/L1L2), tl_windowed, "
                          "tl_window_size, tl_low_memory. ADTLERT time-lapse currently "
                          "requires windowed mode and a common survey geometry.")},
                {"name": "preview_mesh", "args": {},
                 "desc": ("Show the Mesh tab: the mesh the next inversion will run on, "
                          "built from the loaded data (the first file in time-lapse "
                          "mode) and the mesh settings, with the zones on it. Reports "
                          "its cell counts, and how many cells each zone covers, once "
                          "it is built.")},
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
                          "with the selected instrument; times are parsed from filenames when dated. "
                          "A survey written as two files - a forward file and its reciprocal "
                          "('recip' in the name) - becomes one time step at the forward "
                          "file's time, both files merged, while pair_reciprocal_files is on; "
                          "a reciprocal file without a forward file is left out and named.")},
                {"name": "list_timelapse_files", "args": {},
                 "desc": ("List the ordered time-lapse surveys with their parsed time labels; a "
                          "paired survey names its reciprocal file.")},
                {"name": "remove_timelapse_files", "args": {"indices": ["int"]},
                 "desc": ("Remove time-lapse surveys by 0-based index (a paired survey's "
                          "reciprocal file goes with it), or omit to clear all.")},
                {"name": "preview_timelapse_file", "args": {"index": "int"},
                 "desc": "Load one time-lapse file (0-based index) into the electrode + pseudosection view."},
                {"name": "run_timelapse", "args": {},
                 "desc": "Run time-lapse ERT inversion (needs >=2 files)."},
                {"name": "export_timelapse", "args": {"folder": "str"},
                 "desc": ("Export the last time-lapse result to a folder: combined VTK, per-step VTKs, "
                          "final_models.npy, mesh (.bms), times CSV, and the figure.")},
                {"name": "get_status", "args": {},
                 "desc": ("Report inversion_status, inversion_running, has_model, and result_summary "
                          "(fit metrics, model range, source and output paths), plus data/settings. "
                          "Completed results are available before saving or adding them to Map.")},
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
            "preview_mesh": lambda: self._agent_preview_mesh(),
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
        result = self._agent_result_summary(last)
        return {
            "status": "ok",
            **result,
            "data_loaded": self._ert_data is not None,
            "electrodes": len(self._x),
            "measurements": self._n_meas,
            "instrument": self._instrument.currentText(),
            "data_file": str(self._data_path or ""),
            "reciprocal_file": str(self._data_partner or ""),
            "timelapse_files": len(self._tl_files),
            **self._agent_pairing(),
            "average_reciprocal_pairs": self._qc_average.isChecked(),
            "reciprocal_error_floor": self._reciprocal_floor(),
            "error_model_fit": self._error_fit_to(),
            "timelapse_labels": list(self._tl_labels),
            "timelapse_low_memory": self._tl_lowmem.isChecked(),
            "lambda": self._lam.value(),
            "lambda_bounds": list(_LAMBDA_BOUNDS),
            "mesh_file": str(self._mesh_path or ""),
            "mesh_settings": self._mesh_options(),
            "mesh_preview": self._mesh_summary(),
            "zones": self._mesh_tab.zone_report(),
            "conform_to_zones": self._mesh_tab.conform_to_zones(),
            "decouple_zones": self._mesh_tab.decouple_zones(),
            "engine": self._engine.currentData(),
            "inversion_mode": self._inv_mode.currentData(),
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
            "models_available": [c["label"] for c in self._inv_choices],
            "last_result_keys": sorted(last.keys()),
        }

    def _agent_result_summary(self, last: Dict[str, Any]) -> Dict[str, Any]:
        """Expose the displayed model and its fit independently of Map exports."""
        running = False
        for worker in (self._inv_worker, self._tl_worker):
            try:
                running |= worker is not None and worker.isRunning()
            except RuntimeError:  # Qt may already have deleted the finished worker.
                pass
        single = self._inv_mgr is not None
        series = self._tl_models is not None and self._tl_mesh is not None
        kind = getattr(self, "_map_result_kind", "single")
        use_series = series and (kind == "timelapse" or not single)
        summary = {}
        if single or series:
            if use_series:
                record = dict(self._tl_result or {})
                values = np.asarray(self._tl_models, dtype=float)
                summary = {"kind": "timelapse", "time_steps": int(values.shape[1]),
                           "output_dir": str(self._tl_out or "")}
            else:
                index = self._lam_pick.currentIndex()
                choice = self._inv_choices[index] if 0 <= index < len(self._inv_choices) else {}
                record = dict(choice.get("metrics") or {}) if choice else dict(last)
                for key in ("chi2", "lambda", "vtk"):
                    if key in choice:
                        record[key] = choice[key]
                values = np.asarray(self._inv_mgr.model, dtype=float)
                summary = {"kind": "single", "label": choice.get("label", "Model"),
                           "data_file": str(getattr(self, "_inv_source", None) or "")}
            for key in ("chi2", "rrms", "iterations", "lambda", "engine", "convergence_stop"):
                value = record.get(key)
                # Non-finite fit statistics mean unavailable, not a successful fit.
                if isinstance(value, (float, np.floating)) and not np.isfinite(value):
                    value = None
                summary[key] = value
            summary["resistivity_vtk"] = str(record.get("vtk") or record.get("resistivity_vtk") or "")
            finite = values[np.isfinite(values)]
            summary["resistivity_range_ohm_m"] = ([float(finite.min()), float(finite.max())]
                                                  if finite.size else None)
            summary["model_cells"] = int(values.shape[0])
        state = self._agent_run_state
        if running:
            state = "running"
        elif state != "failed" and (single or series or last):
            state = "completed"
        return {"inversion_status": state, "inversion_running": bool(running),
                "has_model": bool(single or series), "result_summary": summary,
                "inversion_error": self._agent_run_error}

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
        # The parse runs on the page's load worker - the path a click in the
        # file list takes - and this waits for it with the window live; it used
        # to parse right here and freeze the studio for the whole read. Starting
        # it supersedes a preview still in flight, as this always did.
        outcome, value = self._load_and_wait(str(p))
        if outcome == "failed":
            return {"status": "failed", "error": f"Could not load: {value}"}
        if outcome != "loaded":
            return {"status": "failed",
                    "error": f"A newer load replaced {p.name} before it finished."}
        # Reflect the loaded file in the file list (it doubles as the loader).
        if str(p) not in self._tl_all:
            self._set_tl_files(self._tl_all + [str(p)])
        if str(p) in self._tl_files:
            self._tl_list.setCurrentRow(self._tl_files.index(str(p)))
        out = {"status": "ok", "electrodes": len(self._x), "measurements": self._n_meas,
               "instrument": self._instrument.currentText()}
        if self._pair_info:
            out["reciprocal_file"] = str(self._data_partner or "")
            out["pairing"] = {key: self._pair_info.get(key) for key in (
                "forward", "reciprocal", "total", "pairs", "unpaired")}
        return out

    def _agent_load_electrodes(self, path: Any) -> Dict[str, Any]:
        if not path:
            return {"status": "failed", "error": "Provide 'path' to an electrode file."}
        p = Path(str(path))
        if not p.exists():
            return {"status": "failed", "error": f"File not found: {p}"}
        return self._take_electrode_file(str(p), wait=True)

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
            # The folded criteria are opt-in from the agent too: naming any one of
            # them unfolds the section, so the panel keeps matching what ran.
            if "average_reciprocal_pairs" in args:
                if args["average_reciprocal_pairs"] and not self._qc_average.isEnabled():
                    return {"status": "failed",
                            "error": "The loaded data have no reading measured both ways, "
                                     "so there are no reciprocal pairs to average."}
                self._qc_average.setChecked(bool(args["average_reciprocal_pairs"]))
            extra = {"drop_nonpositive_rhoa": lambda v: self._qc_drop_neg.setChecked(bool(v)),
                     "min_voltage": lambda v: self._qc_min_v.setValue(float(v)),
                     "min_current": lambda v: self._qc_min_i.setValue(float(v)),
                     "max_geometric_factor": lambda v: self._qc_max_k.setValue(float(v)),
                     "max_contact_resistance": lambda v: self._qc_max_rc.setValue(float(v)),
                     "max_stacking_spread": lambda v: self._qc_max_stack.setValue(float(v) * 100.0),
                     "max_reciprocal_error": lambda v: self._qc_max_recip.setValue(float(v) * 100.0)}
            used = [key for key in extra if key in args]
            for key in used:
                extra[key](args[key])
            if used:
                self._qc_more.setChecked(True)
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

        def set_inv_mode(value):
            """Select the Quick/Full preset.

            ``_agent_set_params`` walks the caller's object in its own order, so a
            ``geometric_factor_policy`` or ``auto_lambda`` listed after
            ``inversion_mode`` overrides what the preset just set, and one listed
            before it does not. Send the mode first and the exceptions after it.
            """
            allowed = ("quick", "full")
            key = str(value).strip().lower()
            if key not in allowed:
                raise ValueError(f"must be one of {list(allowed)}")
            self._inv_mode.setCurrentIndex(
                [self._inv_mode.itemData(i) for i in range(self._inv_mode.count())].index(key)
            )
            self._on_inv_mode_changed()

        def set_combo_data(combo, value):
            """Match on the stable itemData key, not on the display label."""
            keys = [combo.itemData(i) for i in range(combo.count())]
            key = str(value).strip().lower()
            for index, candidate in enumerate(keys):
                if str(candidate).lower() == key:
                    combo.setCurrentIndex(index)
                    return
            raise ValueError(f"must be one of {keys}")

        def set_error_source(value):
            """``stack`` needs data that record a spread, ``reciprocal`` data
            measured both ways; refused, not left greyed."""
            if str(value).strip().lower() == "stack" and not (
                    self._ert_data_full is not None and self._ert_data_full.haveData("stack")):
                raise ValueError("needs each reading's stacking spread, which the loaded "
                                 "data do not record (Subsurface Insights exports do)")
            if str(value).strip().lower() == "reciprocal" and not (
                    self._reciprocal_pair_count(self._ert_data_full) or self._tl_partner):
                raise ValueError("needs readings measured both ways; the loaded data and "
                                 "the file list have no reciprocal pairs")
            set_combo_data(self._err_source, value)

        def set_error_fit(value):
            """What the reciprocal error model is fitted to: 'all' or 'kept'."""
            key = str(value).strip().lower()
            if key not in ("all", "kept"):
                raise ValueError("must be one of ['all', 'kept']")
            self._recip_view.set_fit_to(key)

        def set_floor(value):
            """The "Minimum" error, given as a fraction."""
            set_in_range(self._recip_floor, float(value) * 100.0)

        def set_in_range(spin, value, cast=float):
            """A value the box would clip is refused, so the reply says so
            instead of reporting a setting that did not take."""
            number = cast(value)
            if not spin.minimum() <= number <= spin.maximum():
                raise ValueError(f"must be between {spin.minimum():g} and "
                                 f"{spin.maximum():g}")
            spin.setValue(number)

        handlers = {
            # shared by single + time-lapse inversion
            "lambda": lambda v: self._lam.setValue(float(v)),
            "max_iterations": lambda v: self._iter.setValue(int(v)),
            "relative_error": lambda v: self._relerr.setValue(float(v)),
            "mesh_quality": lambda v: set_in_range(self._quality, v),
            "para_depth": lambda v: set_in_range(self._para_depth, v),
            "para_max_cell_size": lambda v: set_in_range(self._para_max_cell, v),
            "para_boundary": lambda v: set_in_range(self._para_boundary, v),
            "surface_nodes": lambda v: set_in_range(self._surface_nodes, v, int),
            "outer_width": lambda v: set_in_range(self._outer_width, v),
            "outer_max_cell_size": lambda v: set_in_range(self._outer_max_cell, v),
            "mesh_file": lambda v: (self._apply_mesh_file(str(v)) if str(v)
                                    else self._clear_mesh()),
            # set_zones is silent, as a change from outside should be; the page
            # still has to follow it, and rebuild a mesh that follows the zones.
            "zones": lambda v: (self._mesh_tab.set_zones(v), self._mesh_inputs_changed()),
            "conform_to_zones": lambda v: self._mesh_tab.set_conform_to_zones(bool(v)),
            "decouple_zones": lambda v: self._mesh_tab.set_decouple_zones(bool(v)),
            "time_lapse": lambda v: self._tl_mode.setChecked(bool(v)),
            # single-inversion fit assistance
            "engine": lambda v: set_combo_data(self._engine, v),
            "inversion_mode": lambda v: set_inv_mode(v),
            "geometric_factor_policy": lambda v: set_geom_policy(v),
            "error_source": lambda v: set_error_source(v),
            "reciprocal_error_floor": lambda v: set_floor(v),
            "error_model_fit": lambda v: set_error_fit(v),
            # Ticked by default; the list regroups as it changes.
            "pair_reciprocal_files": lambda v: self._tl_pair_box.setChecked(bool(v)),
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
            "tl_mesh_quality": lambda v: set_in_range(self._quality, v),
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

    def _agent_preview_mesh(self) -> Dict[str, Any]:
        request, message = self._mesh_preview_request()
        if request is None:
            return {"status": "failed", "error": message}
        # Opening the tab builds the mesh when it is not current.
        self._tabs.setCurrentWidget(self._mesh_tab)
        self._refresh_mesh_preview()
        summary = self._mesh_summary()
        if not summary.get("current"):
            return {"status": "started",
                    "message": "Building the inversion mesh on the Mesh tab; ask for "
                               "status shortly for its counts."}
        return {"status": "ok", "mesh": summary, "zones": self._mesh_tab.zone_report()}

    def _agent_add_timelapse_files(self, paths: Any, append: Any = False) -> Dict[str, Any]:
        if not isinstance(paths, list) or not paths:
            return {"status": "failed", "error": "Provide 'paths' as a non-empty list of files."}
        missing = [str(p) for p in paths if not Path(str(p)).exists()]
        if missing:
            return {"status": "failed", "error": f"Files not found: {missing}"}
        merged = list(self._tl_all) if append else []
        for p in paths:
            if str(p) not in merged:
                merged.append(str(p))
        self._set_tl_files(merged)
        self._tl_mode.setChecked(True)  # reveal the time-lapse options in the UI
        return {"status": "ok", "files": len(self._tl_files),
                "time_labels": list(self._tl_labels),
                "timing": self._tl_timing.summary() if self._tl_timing else "",
                **self._agent_pairing()}

    def _agent_pairing(self) -> Dict[str, Any]:
        """How the list paired forward and reciprocal files, for the agent."""
        found = getattr(self, "_tl_pairs_found", None) or {}
        if not (found.get("pairs") or found.get("orphans")):
            return {}
        return {"pair_reciprocal_files": self._tl_pair_box.isChecked(),
                "paired_surveys": len(self._tl_partner),
                "reciprocal_files_left_out": [Path(p).name for p in self._tl_left_out]}

    def _agent_list_timelapse(self) -> Dict[str, Any]:
        timing = self._tl_timing
        gaps = timing.intervals if timing is not None else []
        return {
            "status": "ok",
            "count": len(self._tl_files),
            # The interval between surveys, in words: the sequence reads the same
            # whether it is hourly or monthly until something says which.
            "timing": timing.summary() if timing is not None else "",
            "time_source": timing.source if timing is not None else "index",
            "intervals": [survey_timing.format_duration(g) for g in gaps],
            # One entry per survey; a paired survey names its reciprocal file.
            "files": [{"index": i, "label": self._tl_labels[i] if i < len(self._tl_labels) else str(i + 1),
                       "time": self._tl_times[i] if i < len(self._tl_times) else float(i + 1),
                       # Where this file's time came from, as its row's tooltip says.
                       "time_note": timing.describe(i) if timing is not None else "",
                       "name": Path(p).name, "path": p,
                       **({"reciprocal": Path(self._tl_partner[p]).name,
                           "reciprocal_path": self._tl_partner[p]}
                          if p in self._tl_partner else {})}
                      for i, p in enumerate(self._tl_files)],
            **self._agent_pairing(),
        }

    def _agent_remove_timelapse(self, indices: Any) -> Dict[str, Any]:
        if indices is None:
            self._set_tl_files([])
            return {"status": "ok", "files": 0, "message": "Cleared all time-lapse files."}
        try:
            drop = {int(i) for i in indices}
        except (TypeError, ValueError):
            return {"status": "failed", "error": "Provide 'indices' as a list of integers."}
        # An index is a survey: a paired survey's reciprocal file goes with it.
        self._set_tl_files(self._tl_flat(
            [p for i, p in enumerate(self._tl_files) if i not in drop]))
        return {"status": "ok", "files": len(self._tl_files)}

    def _agent_preview_timelapse(self, index: Any) -> Dict[str, Any]:
        try:
            i = int(index)
        except (TypeError, ValueError):
            return {"status": "failed", "error": "Provide 'index' as an integer (0-based)."}
        if not (0 <= i < len(self._tl_files)):
            return {"status": "failed", "error": f"index out of range (0..{len(self._tl_files) - 1})."}
        self._start_load(self._tl_files[i])
        partner = self._tl_partner.get(self._tl_files[i])
        return {"status": "started", "message": f"Loading time step {i} for preview.",
                "name": Path(self._tl_files[i]).name,
                **({"reciprocal": Path(partner).name} if partner else {})}

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


# Names a 0.3.0 script could import from this page, which it no longer defines.
from PyHydroGeophysX._internal.deprecations import legacy_names as _legacy_names  # noqa: E402

__getattr__ = _legacy_names(__name__, {
    "metrics_from_manager": "PyHydroGeophysX.inversion.metrics.metrics_from_manager",
})
