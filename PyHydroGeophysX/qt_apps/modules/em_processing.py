"""EM module: 1D FDEM / TDEM inversion.

Load a sounding (FDEM: frequency, real, imag; TDEM: time, response) and invert it
for a layered resistivity model. A survey line can use joint LM+HM observations
and same-line lateral constraints to produce a position x depth resistivity
section. The numerics live in the Qt-free
``PyHydroGeophysX.qt_apps.em_pipeline`` (a thin wrapper over the package's SimPEG
forward operators). Results export to npy / csv.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from PySide6.QtCore import Qt
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
    QLineEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QStackedWidget,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from PyHydroGeophysX.inversion.em1d_lci import DOI_SENSITIVITY_THRESHOLD
from PyHydroGeophysX.workflows import em1d as em_pipeline
from PyHydroGeophysX.qt_apps import theme
from PyHydroGeophysX.qt_apps.modules.base import BaseModule, LogFn
from PyHydroGeophysX.qt_apps.qt_utils import (
    BusyStateController,
    ReproduceBar,
    make_double_spinbox,
    merged_row,
    set_rows_enabled,
    select_directory,
)
from PyHydroGeophysX.qt_apps.widgets.curve_viewer import CurveViewer
from PyHydroGeophysX.qt_apps.widgets.em_overview_view import EMOverviewView
from PyHydroGeophysX.qt_apps.widgets.image_view import ZoomableImageView
from PyHydroGeophysX.qt_apps.widgets.model3d_view import Model3DView
from PyHydroGeophysX.qt_apps.widgets.plan_slice_view import PlanSliceView
from PyHydroGeophysX.qt_apps.widgets.quality_view import InversionQualityView
from PyHydroGeophysX.qt_apps.workers import TaskWorker, WorkflowWorker
from PyHydroGeophysX.workflows import (
    ArtifactRef,
    RunContext,
    WorkflowRunResult,
    WorkflowSpec,
    export_workflow_bundle,
)

_FILE_FILTER = (
    "EM data (*.csv *.txt *.dat *.xyz);;"
    "TEMcompany XYZ (*.xyz);;"
    "All files (*)"
)
_TEM2GO_FORMAT = "TEMcompany / TEM2Go"
_TTEM_FORMAT = "TEMcompany tTEM raw"
_TEM_FORMATS = {_TEM2GO_FORMAT, _TTEM_FORMAT}


def _fit_scroll_width(scroll: QScrollArea, panel: QWidget, *, cap: int) -> None:
    """Size a side panel to the row that actually needs the most width.

    The panel sits in a row that gives all the stretch to the plots, and a scroll
    area's own size hint is a fixed default with nothing to do with what it
    holds. Left alone it therefore settles at exactly its minimum width, and any
    row wider than that is cut off however much room the window has: a hand-set
    minimum silently becomes the panel width, and stays wrong as soon as a row
    is added. Measuring the finished panel keeps the two in step.

    ``cap`` bounds what the panel may demand from a small screen; past it the
    horizontal scrollbar takes over.
    """
    panel.ensurePolished()
    panel.adjustSize()
    bar = scroll.verticalScrollBar().sizeHint().width()
    frame = 2 * scroll.frameWidth()
    needed = panel.sizeHint().width() + bar + frame
    scroll.setMinimumWidth(min(needed, int(cap)))
    scroll.setMaximumWidth(max(needed, int(cap)))


class EMProcessingModule(BaseModule):
    module_key = "em_processing"
    module_title = "EM Processing"

    #: Width the control panel may not exceed even if its widest row wants more.
    #: Past this the row scrolls rather than eating the space the section needs.
    _CONTROLS_MAX_WIDTH = 580

    def __init__(self, state: Any, log: LogFn, parent=None) -> None:
        super().__init__(state, log, parent)
        self._data: Optional[Dict[str, np.ndarray]] = None
        self._source_path: Optional[Path] = None
        self._last_result: Optional[Dict[str, Any]] = None
        self._last_section: Optional[Dict[str, Any]] = None
        self._inv_worker: Optional[WorkflowWorker] = None
        self._inv_busy: Optional[BusyStateController] = None
        self._project_start_res = 100.0   # fallback when auto is off
        self._line_worker: Optional[TaskWorker] = None
        self._workflow_recipe_path = ""
        self._geom_positions: Optional[np.ndarray] = None
        self._geom_heights: Optional[np.ndarray] = None
        self._geom_x: Optional[np.ndarray] = None
        self._geom_y: Optional[np.ndarray] = None
        self._example_id: Optional[str] = None
        self._project_layer_thicknesses: Optional[np.ndarray] = None
        self._ttem_gex_path: Optional[Path] = None
        self._ttem_tfi_path: Optional[Path] = None

        root = QHBoxLayout(self)
        self._tabs = QTabWidget()
        self._curve = CurveViewer()
        # The "Resistivity model" tab adapts to the result: a 1D depth profile
        # (single sounding), or — for a line — the map + section overview, a
        # plan-view depth slice (a map you slice by depth), or the position x
        # depth section on its own, chosen with "View".
        self._inv_view = ZoomableImageView()       # page 0: single-sounding profile
        self._overview_view = EMOverviewView()     # page 1: map + section overview
        self._plan_view = PlanSliceView()          # page 2: plan-view depth slice
        self._section_view = Model3DView()         # page 3: position x depth section
        self._model_stack = QStackedWidget()
        self._model_stack.addWidget(self._inv_view)
        self._model_stack.addWidget(self._overview_view)
        self._model_stack.addWidget(self._plan_view)
        self._model_stack.addWidget(self._section_view)
        self._model_tab = QWidget()
        mlay = QVBoxLayout(self._model_tab); mlay.setContentsMargins(0, 0, 0, 0)
        self._view_row = QWidget()
        vr = QHBoxLayout(self._view_row); vr.setContentsMargins(6, 2, 6, 2)
        vr.addWidget(QLabel("View:"))
        self._view_mode = QComboBox()
        self._view_mode.addItems(
            ["Overview (map + section)", "Plan slice (map)", "Section"])
        self._view_mode.setToolTip(
            "Overview pairs the survey map with the selected line's resistivity "
            "section. Plan slice maps one depth layer across the survey. Section "
            "shows the position x depth model on its own.")
        self._view_mode.currentIndexChanged.connect(self._on_view_mode)
        vr.addWidget(self._view_mode); vr.addStretch(1)
        self._view_row.setVisible(False)
        mlay.addWidget(self._view_row)
        mlay.addWidget(self._model_stack, stretch=1)
        self._quality_view = InversionQualityView()
        self._tabs.addTab(self._curve, "Sounding")
        self._tabs.addTab(self._model_tab, "Resistivity model")
        self._tabs.addTab(self._quality_view, "Inversion quality")
        self._reproduce = ReproduceBar()
        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.addWidget(self._tabs, stretch=1)
        center_layout.addWidget(self._reproduce)
        root.addWidget(center, stretch=1)
        root.addWidget(self._build_controls())
        self._on_method_changed()

    def _on_view_mode(self, idx: int) -> None:
        pages = {0: self._overview_view, 1: self._plan_view}
        self._model_stack.setCurrentWidget(pages.get(idx, self._section_view))

    # -- helpers -------------------------------------------------------------
    @staticmethod
    def _dspin(value, lo, hi, step, dec) -> QDoubleSpinBox:
        return make_double_spinbox(value, lo, hi, step, dec)

    @staticmethod
    def _ispin(value, lo, hi) -> QSpinBox:
        s = QSpinBox(); s.setRange(lo, hi); s.setValue(value)
        return s

    def _build_controls(self) -> QScrollArea:
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        # A horizontal scrollbar for the window that cannot spare the width.
        # Suppressing the bar does not make the content fit, it only makes what
        # does not fit unreachable, which is worse than a bar.
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        panel = QWidget(); scroll.setWidget(panel)
        layout = QVBoxLayout(panel)

        layout.addWidget(self._build_loader_group())
        layout.addWidget(self._build_geometry_group())
        layout.addWidget(self._build_inversion_group())
        layout.addWidget(self._build_errors_group())
        layout.addWidget(self._build_line_group())
        layout.addWidget(self._build_assist_group())
        layout.addWidget(self._build_run_group())
        layout.addStretch(1)
        _fit_scroll_width(scroll, panel, cap=self._CONTROLS_MAX_WIDTH)
        return scroll

    def _build_loader_group(self) -> QGroupBox:
        box = QGroupBox("Load sounding data"); v = QVBoxLayout(box)
        self._method = QComboBox(); self._method.addItems(list(em_pipeline.METHODS))
        self._method.currentTextChanged.connect(self._on_method_changed)
        self._data_format = QComboBox()
        self._data_format.addItems(["Generic table", _TEM2GO_FORMAT, _TTEM_FORMAT])
        self._data_format.currentTextChanged.connect(self._on_data_format_changed)
        mrow = QFormLayout()
        mrow.addRow("Method", self._method)
        mrow.addRow("Data format", self._data_format)
        v.addLayout(mrow)

        row = QHBoxLayout()
        load_btn = QPushButton("Load data…")
        load_btn.setProperty("primary", True)
        load_btn.setIcon(theme.icon("fa5s.folder-open", color="#ffffff"))
        load_btn.clicked.connect(self._load)
        fmt_btn = QPushButton("Format help")
        fmt_btn.setIcon(theme.icon("fa5s.file-alt"))
        fmt_btn.clicked.connect(self._show_format_help)
        row.addWidget(load_btn); row.addWidget(fmt_btn)
        v.addLayout(row)

        self._tem_moment_row = QWidget()
        moment_form = QFormLayout(self._tem_moment_row)
        moment_form.setContentsMargins(0, 0, 0, 0)
        self._tem_moment = QComboBox()
        self._tem_moment.addItems(list(em_pipeline.TEMCOMPANY_MOMENTS))
        self._tem_moment.setToolTip(
            "LM+HM fits all available gates to one shared model. HM uses later-time "
            "gates and LM uses early-time gates. In-use flags are applied automatically.")
        self._tem_moment.currentTextChanged.connect(self._on_tem_moment_changed)
        moment_form.addRow("Moment(s)", self._tem_moment)
        self._use_flags = QCheckBox("Use the project's in-use flags")
        self._use_flags.setChecked(True)
        self._use_flags.setToolTip(
            "On: only the gates the project file marks as in use are imported.\n\n"
            "Off: every gate holding a finite, non-dummy value is imported, "
            "including those the project marks unused. On a noisy survey that can "
            "be several times as many gates. Each one keeps its recorded stack "
            "error, so its weight in the fit is unchanged.")
        self._use_flags.toggled.connect(self._on_use_flags_changed)
        moment_form.addRow("", self._use_flags)
        self._tail_cut = self._dspin(0.30, 0.0, 5.0, 0.05, 2)
        self._tail_cut.setSpecialValueText("off")
        self._tail_cut.setToolTip(
            "Truncates each decay at the first gate that is negative or whose stack "
            "error exceeds this value, dropping that gate and every later one.\n\n"
            "0 turns the cut off. A response can cross zero for physical reasons "
            "over a very conductive near-surface and on offset-loop systems, where "
            "truncating would remove real signal.")
        self._tail_cut.valueChanged.connect(self._on_use_flags_changed)
        moment_form.addRow("Tail cut (σ)", self._tail_cut)
        self._tem_moment_row.setVisible(False)
        v.addWidget(self._tem_moment_row)

        self._ttem_calibration_row = QWidget()
        calibration_form = QFormLayout(self._ttem_calibration_row)
        calibration_form.setContentsMargins(0, 0, 0, 0)
        self._gex_path_edit = QLineEdit(); self._gex_path_edit.setReadOnly(True)
        self._gex_path_edit.setPlaceholderText("Auto-detect or select .gex")
        gex_button = QPushButton("Browse…")
        gex_button.clicked.connect(self._select_ttem_gex)
        gex_row = QWidget(); gex_layout = QHBoxLayout(gex_row)
        gex_layout.setContentsMargins(0, 0, 0, 0)
        gex_layout.addWidget(self._gex_path_edit, stretch=1); gex_layout.addWidget(gex_button)
        calibration_form.addRow("System GEX", gex_row)
        self._tfi_path_edit = QLineEdit(); self._tfi_path_edit.setReadOnly(True)
        self._tfi_path_edit.setPlaceholderText("Auto-detect or select .tfi")
        tfi_button = QPushButton("Browse…")
        tfi_button.clicked.connect(self._select_ttem_tfi)
        tfi_row = QWidget(); tfi_layout = QHBoxLayout(tfi_row)
        tfi_layout.setContentsMargins(0, 0, 0, 0)
        tfi_layout.addWidget(self._tfi_path_edit, stretch=1); tfi_layout.addWidget(tfi_button)
        calibration_form.addRow("Import filter TFI", tfi_row)
        self._ttem_calibration_row.setVisible(False)
        v.addWidget(self._ttem_calibration_row)

        # Optional per-sounding geometry (positions / height) for a survey line;
        # sits next to the loader, shown once a multi-sounding file is loaded.
        self._geom_row = QWidget()
        gv = QVBoxLayout(self._geom_row); gv.setContentsMargins(0, 0, 0, 0)
        grow = QHBoxLayout()
        self._geom_btn = QPushButton("Load geometry…")
        self._geom_btn.setIcon(theme.icon("fa5s.map-marker-alt"))
        self._geom_btn.setToolTip("Load a file of along-line positions (and optional sensor "
                                  "height), one row per sounding, so the section uses real "
                                  "distances instead of uniform spacing. See Data format.")
        self._geom_btn.clicked.connect(self._load_geometry)
        grow.addWidget(self._geom_btn); grow.addStretch(1)
        gv.addLayout(grow)
        self._geom_info = QLabel("Geometry: uniform spacing.")
        self._geom_info.setStyleSheet("color:#5a6a7a; font-size:8pt;"); self._geom_info.setWordWrap(True)
        gv.addWidget(self._geom_info)
        self._geom_row.setVisible(False)
        v.addWidget(self._geom_row)

        hint = QLabel("Choose a data format, then load a file or project folder. "
                      "FDEM: columns freq, real, imag. TDEM: columns time, response. "
                      "Several soundings are inverted into a resistivity section/map.")
        hint.setWordWrap(True); hint.setStyleSheet("color:#5a6a7a; font-size:8pt;")
        v.addWidget(hint)

        # Picker for files that hold several soundings (drives the preview curve).
        self._sounding_row = QWidget()
        srow = QFormLayout(self._sounding_row); srow.setContentsMargins(0, 0, 0, 0)
        self._sounding = QSpinBox(); self._sounding.setRange(1, 1); self._sounding.setValue(1)
        self._sounding.setToolTip(
            "Selects which of the file's soundings the preview shows. A line "
            "inversion uses the whole survey regardless of this setting.")
        self._sounding.valueChanged.connect(self._on_sounding_changed)
        srow.addRow("Preview sounding #", self._sounding)
        self._sounding_row.setVisible(False)
        v.addWidget(self._sounding_row)

        self._info = QLabel("No sounding loaded.")
        self._info.setWordWrap(True)
        v.addWidget(self._info)
        self._example_note = QLabel("")
        self._example_note.setWordWrap(True)
        self._example_note.setStyleSheet("color:#5a6a7a; font-size:8pt;")
        self._example_note.setVisible(False)
        v.addWidget(self._example_note)
        return box

    def _build_geometry_group(self) -> QGroupBox:
        box = QGroupBox("Survey geometry (system)"); form = QFormLayout(box)
        f = em_pipeline.DEFAULT_FDEM
        self._src_radius = self._dspin(f["source_radius"], 0.01, 200.0, 0.1, 3)
        self._loop_area = self._dspin(8.0, 0.01, 10000.0, 0.25, 2)
        self._loop_area.setToolTip(
            "Physical transmitter-loop area. For raw tTEM this also re-normalizes "
            "dB/dt by the chosen area, so changing it affects both data and forward model."
        )
        self._loop_area.editingFinished.connect(self._on_ttem_loop_area_changed)
        self._tx_rx = self._dspin(f["tx_rx_sep"], 0.0, 500.0, 0.05, 2)
        self._height = self._dspin(f["height"], 0.0, 500.0, 0.05, 2)
        self._orient = QComboBox(); self._orient.addItems(["z", "x", "y"])
        self._component = QComboBox(); self._component.addItems(["secondary", "total", "both"])
        self._waveform = QComboBox()
        self._src_radius_label = QLabel("Source radius (m)")
        form.addRow(self._src_radius_label, self._src_radius)
        self._loop_area_label = QLabel("Tx loop area (m²)")
        form.addRow(self._loop_area_label, self._loop_area)
        self._tx_rx_label = QLabel("Tx-Rx sep (m)")
        form.addRow(self._tx_rx_label, self._tx_rx)
        form.addRow("Height (m)", self._height)
        form.addRow("Orientation", self._orient)
        self._component_label = QLabel("Component")
        form.addRow(self._component_label, self._component)
        form.addRow("Waveform", self._waveform)
        self._system_note = QLabel("")
        self._system_note.setWordWrap(True)
        self._system_note.setStyleSheet("color:#9b5a00; font-size:8pt;")
        self._system_note.setVisible(False)
        form.addRow(self._system_note)
        return box

    # The controls are split the same way as the ERT page: what defines the
    # model, how the data are weighted, what couples the line, and what the
    # software may change on its own. Everything configurable comes before Run.
    def _build_inversion_group(self) -> QGroupBox:
        box = QGroupBox("Inversion"); form = QFormLayout(box)
        d = em_pipeline.DEFAULT_INVERSION
        self._n_layers = self._ispin(d["n_layers"], 3, 60)
        self._n_layers.setToolTip(
            "Fixed layers in the model. The grid is shared by every sounding on a "
            "line, which is what lets them be coupled layer by layer.")
        self._min_thick = self._dspin(d["min_thickness"], 0.2, 50.0, 0.5, 2)
        # Two decimals, like the minimum: a project's saved layer grid is only
        # reused when both ends still match it (see _collect_inv), and rounding
        # the deepest layer to 0.1 m was enough to lose that match.
        self._max_thick = self._dspin(d["max_thickness"], 1.0, 500.0, 1.0, 2)
        # Two ends of one setting: the layer grid is geometric between them.
        self._thick_row = merged_row(self._min_thick, "to", self._max_thick, "m")
        # 0 is "auto": the search below picks the half-space. One control rather
        # than a number plus a checkbox, because the two can never both apply.
        self._start_res = self._dspin(0.0, 0.0, 1e5, 10.0, 1)
        self._start_res.setSpecialValueText("auto")
        self._start_res.setToolTip(
            "Uniform resistivity assigned to every layer at the start of the "
            "inversion. It sets the optimizer's initial model and does not "
            "constrain the final resistivity.\n\n"
            "auto tries a set of uniform half-spaces on a sample of the "
            "soundings and starts from the one whose forward response comes "
            "closest to the data. Where a run starts decides which minimum it "
            "settles in: the Gauss-Newton step is built from a linearization "
            "about the current model, so a start a decade away from the ground "
            "describes a different problem. On one ground survey the project's "
            "own 40 Ω·m default reached χ² 387 where auto reached 8.7, and the "
            "40 Ω·m run drove a tenth of the section onto the 1 Ω·m bound.\n\n"
            "A value uses that resistivity, which is what a project file "
            "carries. For a line inversion this seeds the first solve; later "
            "refinement passes warm-start from the preceding result.")
        self._smooth = self._dspin(d["smoothness"], 0.0, 10.0, 0.1, 2)
        self._smooth.setToolTip(
            "Vertical smoothness down each sounding's layer stack. Higher gives a "
            "smoother profile and fits the data less well.")
        self._max_iter = self._ispin(d["max_iterations"], 3, 200)
        self._max_iter.setToolTip(
            "Iteration budget. The run stops earlier when it reaches the target χ² "
            "or when the misfit stops improving.")
        form.addRow("Layers", self._n_layers)
        form.addRow("Thickness", self._thick_row)
        form.addRow("Initial model ρ (Ω·m)", self._start_res)
        form.addRow("Smoothness", self._smooth)
        form.addRow("Max iterations", self._max_iter)
        return box

    def _build_errors_group(self) -> QGroupBox:
        box = QGroupBox("Data errors and calibration"); form = QFormLayout(box)
        self._rel_err = self._dspin(0.03, 0.0, 1.0, 0.01, 3)
        self._rel_err.setToolTip(
            "Uncertainty applied to every gate alike: system calibration, and the "
            "error in representing the ground as 1D layers. Where the file carries a "
            "per-gate stack error, this is added in quadrature rather than replacing "
            "it, so a noisy gate stays noisy and a clean one takes this as its "
            "floor.\n\n"
            "χ² is measured against the combined uncertainty, so this value is what "
            "\"fitting the data\" is measured against. A larger value makes the same "
            "model fit better; a smaller one makes it fit worse.")
        self._data_scale = self._dspin(1.0, 1e-4, 1e6, 0.1, 4)
        self._data_scale.setToolTip(
            "Multiplies the observed data before inversion, for data in normalized "
            "units such as moment-normalized airborne dB/dt that carry a system "
            "calibration constant. 1.0 leaves the data unscaled.")
        self._auto_scale = QCheckBox("Auto-calibrate")
        self._auto_scale.setChecked(True)
        self._auto_scale.setToolTip(
            "Estimates the amplitude scale from the shape of the decay. This keeps "
            "the inversion off the bounds; the absolute resistivity level it "
            "produces is approximate, since the shape alone does not fix it.")
        self._ref_res = self._dspin(0.0, 0.0, 1e5, 10.0, 1)
        self._ref_res.setToolTip(
            "Ties the amplitude scale to a half-space of this resistivity, in Ω·m, "
            "so the recovered model lands near this level. The data alone do not "
            "determine the absolute level, and the same value is applied the same "
            "way to every dataset.\n\n"
            "0 turns this off. A non-zero value takes precedence over "
            "Auto-calibrate.")
        form.addRow("Relative error", self._rel_err)
        form.addRow("Data scale / calib.", self._data_scale)
        form.addRow("", self._auto_scale)
        form.addRow("Reference ρ (Ω·m)", self._ref_res)
        return box

    def _build_line_group(self) -> QGroupBox:
        """Line-only controls, shown when the file holds several soundings."""
        box = QGroupBox("Line"); form = QFormLayout(box)
        d = em_pipeline.DEFAULT_INVERSION
        self._line_spacing = self._dspin(50.0, 0.1, 100000.0, 10.0, 2)
        self._line_spacing.setToolTip("Uniform sounding spacing used for the section's x-axis "
                                      "when no geometry file is loaded.")
        # The ceiling is raised to the station count when a survey is loaded. A
        # fixed one clamps the value silently: a QSpinBox told to hold 887 with a
        # maximum of 500 simply reads 500 afterwards, and the 387 stations past
        # it never reach the inversion.
        self._line_max = self._ispin(12, 1, 100000)
        self._line_max.setToolTip("Cap on how many soundings to invert (keeps a long line fast).")
        self._lateral_smooth = self._dspin(
            float(d.get("lateral_smoothness", 0.0)), 0.0, 20.0, 0.1, 2)
        self._lateral_smooth.setToolTip(
            "How tightly neighbouring soundings on the same survey line are tied "
            "together. The tie weakens with distance and never crosses a line. "
            "0 leaves the soundings independent, whatever the coupling below says.")
        self._lci_mode = QComboBox()
        for label, key in (("Simultaneous", "simultaneous"),
                           ("Block-coordinate", "sequential"),
                           ("Off", "off")):
            self._lci_mode.addItem(label, key)
        self._lci_mode.setCurrentIndex(0)
        self._lci_mode.setToolTip(
            "Simultaneous solves the whole line as one system, so the lateral "
            "constraint is enforced while each sounding is being fitted; it uses the "
            "analytic sensitivity, which makes it faster than inverting the "
            "soundings one at a time. Block-coordinate re-inverts one sounding at a "
            "time against its neighbours' models from the previous pass. Off "
            "inverts each sounding alone.")
        self._lci_passes = self._ispin(int(d.get("lci_passes", 1)), 0, 10)
        self._lci_passes.setToolTip(
            "Number of block-coordinate passes. The simultaneous solver does not "
            "use this setting.")
        self._lci_passes_label = QLabel("passes")
        self._lci_mode_row = merged_row(
            self._lci_mode, self._lci_passes_label, self._lci_passes)
        self._lci_mode.currentIndexChanged.connect(self._sync_lci_mode)
        import os as _os

        cores = getattr(_os, "process_cpu_count", None)
        available = int((cores() if cores is not None else _os.cpu_count()) or 1)
        # BaseModule._workers is the live QThread registry. Keep this widget on
        # a distinct name or line inversion replaces that list with a QSpinBox.
        self._parallel_workers_spin = self._ispin(0, 0, max(1, available))
        self._parallel_workers_spin.setSpecialValueText("auto")
        self._parallel_workers_spin.setToolTip(
            f"Threads used for the per-sounding forward and Jacobian. This "
            f"machine reports {available} usable cores; auto takes one thread "
            f"per sounding up to that number.\n\n"
            "Each sounding owns its forward operator and the threads only read "
            "the shared model, so the models come back identical whatever this "
            "is set to. The work is NumPy underneath and releases the GIL for "
            "most of its duration, so scaling is real but short of linear.")
        form.addRow("Sounding spacing (m)", self._line_spacing)
        form.addRow("Max soundings", self._line_max)
        form.addRow("Lateral smoothness", self._lateral_smooth)
        form.addRow("Coupling", self._lci_mode_row)
        form.addRow("CPU threads", self._parallel_workers_spin)
        self._sync_lci_mode()
        self._line_rows = box
        box.setVisible(False)
        return box

    def _build_assist_group(self) -> QGroupBox:
        box = QGroupBox("Fit assistance"); form = QFormLayout(box)
        self._auto_lam = QCheckBox("Auto-λ (re-solve for target χ²)")
        self._auto_lam.setChecked(True)
        self._auto_lam.setToolTip(
            "The solve at the smoothness above always runs first and is always kept. "
            "If its χ² misses the target band, the line is re-solved with the "
            "vertical and lateral smoothness scaled together by a single factor (λ "
            "here), and the closest result is displayed. Each trial warm-starts from "
            "the nearest factor already solved. Applies to the simultaneous line "
            "solver.")
        self._auto_lam.toggled.connect(self._sync_lci_mode)
        form.addRow(self._auto_lam)

        self._target_chi2 = self._dspin(1.0, 0.1, 100.0, 0.1, 2)
        self._target_chi2.setToolTip(
            "χ² = 1 means the model explains the data to within the assumed relative "
            "error. The solve stops on reaching the band rather than fitting past it.")
        self._chi2_tol = self._dspin(0.2, 0.01, 10.0, 0.05, 2)
        self._chi2_tol.setToolTip(
            "Half-width of the accepted band. A step that would overshoot below it is "
            "shortened instead, so the reported χ² is not an accident of step length.")
        self._chi2_row = merged_row(self._target_chi2, "±", self._chi2_tol)
        form.addRow("Target χ²", self._chi2_row)

        self._lam_trials = self._ispin(5, 1, 20)
        self._lam_trials.setToolTip(
            "Upper bound on the extra line solves the search may run, on top of the "
            "first one at the smoothness set above.")
        form.addRow("Max trials", self._lam_trials)

        # The other answer to a high χ²: some gates are wrong rather than the
        # model being too stiff. Same controls, wording and defaults as ERT.
        self._reject = QCheckBox("Reject outliers")
        self._reject.setChecked(False)
        self._reject.setToolTip(
            "After the line converges, the time gates whose residual exceeds the cut "
            "below are dropped and the line is solved again at the same smoothness. "
            "This addresses a high χ² caused by individual bad gates rather than by "
            "a model that is too stiff.\n\n"
            "Cutting is per gate, not per sounding, and it shrinks the data set; the "
            "floor below bounds how much can be removed. A TDEM station may carry "
            "only a handful of gates.")
        self._reject.toggled.connect(self._sync_reject)
        form.addRow(self._reject)

        self._reject_sigma = self._dspin(3.0, 1.5, 20.0, 0.5, 1)
        self._reject_sigma.setToolTip(
            "Rejection cut in units of the gate's own error. A gate at 3 means the "
            "model misses it by three times its stack error.")
        self._reject_passes = self._ispin(2, 1, 5)
        self._reject_passes.setToolTip(
            "How many reject-and-re-solve cycles to run. Each re-solve warm-starts "
            "from the model just found, so it costs far less than the first solve.")
        self._reject_row = merged_row(
            self._reject_sigma, "σ, passes", self._reject_passes)
        form.addRow("Cut beyond", self._reject_row)

        self._min_keep = self._dspin(50.0, 10.0, 100.0, 5.0, 0)
        self._min_keep.setSuffix(" %")
        self._min_keep.setToolTip(
            "Rejection stops before it would leave less than this share of the gates. "
            "A χ² bought by deleting most of the survey is not a fit.")
        self._min_gates = self._ispin(3, 1, 20)
        self._min_gates.setToolTip(
            "A sounding never drops below this many gates, keeping its best-fitting "
            "ones. Stations that arrive with fewer keep everything they have. Without "
            "this floor a station holding one or two gates loses them both and its "
            "column becomes a hole in the section, held up by the lateral constraint "
            "alone.")
        self._min_keep_row = merged_row(
            self._min_keep, "of the gates,", self._min_gates, "per sounding")
        form.addRow("Keep at least", self._min_keep_row)
        self._sync_reject()
        return box

    def _sync_reject(self) -> None:
        """Only enable the rejection knobs the checkbox actually uses."""
        set_rows_enabled([self._reject_row, self._min_keep_row],
                         self._reject.isEnabled() and self._reject.isChecked())

    def _build_run_group(self) -> QGroupBox:
        box = QGroupBox("Run"); form = QFormLayout(box)
        self._inv_btn = QPushButton("Run inversion")
        self._inv_btn.setProperty("primary", True)
        self._inv_btn.setIcon(theme.icon("fa5s.bullseye", color="#ffffff"))
        self._inv_btn.clicked.connect(self._run_inversion)
        form.addRow(self._inv_btn)
        self._backend_label = QLabel()
        self._backend_label.setWordWrap(True)
        self._backend_label.setStyleSheet("font-size:8pt;")
        form.addRow("Backend", self._backend_label)
        self._inv_progress = QProgressBar(); self._inv_progress.setVisible(False)
        form.addRow(self._inv_progress)
        self._inv_export = QPushButton("Export recovered model (csv)…")
        self._inv_export.setIcon(theme.icon("fa5s.file-export"))
        self._inv_export.setEnabled(False)
        self._inv_export.setToolTip(
            "After a line inversion: model_cells.csv, one row per layer per "
            "sounding with its map coordinate, elevation, resistivity, "
            "sensitivity and χ², plus soundings.csv, one row per station. "
            "After a single sounding: the recovered model as npy and csv.\n\n"
            "Both are written next to the section automatically. This writes a "
            "second copy to a chosen folder.")
        self._inv_export.clicked.connect(self._export_inversion)
        form.addRow(self._inv_export)
        return box

    # -- method switch -------------------------------------------------------
    def _on_method_changed(self) -> None:
        fdem = self._method.currentText() == "FDEM"
        self._waveform.clear()
        self._waveform.addItems(["dipole", "loop"] if fdem else ["step_off", "ramp_off"])
        self._component_label.setVisible(fdem)
        self._component.setVisible(fdem)
        self._tx_rx_label.setVisible(True)
        self._tx_rx.setVisible(True)
        self._src_radius_label.setVisible(fdem)
        self._src_radius.setVisible(fdem)
        self._loop_area_label.setVisible(not fdem)
        self._loop_area.setVisible(not fdem)
        if not fdem and hasattr(self, "_tem_moment_row"):
            self._tem_moment_row.setVisible(
                self._data_format.currentText() in _TEM_FORMATS)
        self._refresh_backend_state()

    def _on_data_format_changed(self, selected: str) -> None:
        temcompany = selected in _TEM_FORMATS
        if temcompany:
            self._method.setCurrentText("TDEM")
        self._use_flags.setEnabled(selected == _TEM2GO_FORMAT)
        self._use_flags.setToolTip(
            "Raw tTEM SKB files do not contain TEM2Go inversion flags."
            if selected == _TTEM_FORMAT else
            "Use the enabled-gate flags stored by the TEM2Go project."
        )
        self._system_note.setVisible(selected == _TTEM_FORMAT)
        self._ttem_calibration_row.setVisible(selected == _TTEM_FORMAT)
        if selected == _TTEM_FORMAT:
            self._system_note.setText(
                "Raw tTEM geometry is editable below. Manual defaults are only a "
                "fallback until the instrument-specific GEX/TFI files are supplied."
            )
        self._tem_moment_row.setVisible(
            temcompany and self._method.currentText() == "TDEM")

    def _on_ttem_loop_area_changed(self) -> None:
        """Re-normalize the preview when the raw tTEM loop area is edited."""
        if (self._source_path is not None and self._data is not None
                and self._data.get("ttem")):
            self._load_sounding(self._sounding.value() - 1, reset_geometry=False)

    def _set_ttem_calibration_path(self, kind: str, path: Optional[Path]) -> None:
        resolved = path.resolve() if path else None
        if kind == "gex":
            self._ttem_gex_path = resolved
            self._gex_path_edit.setText(str(resolved or ""))
        else:
            self._ttem_tfi_path = resolved
            self._tfi_path_edit.setText(str(resolved or ""))

    def _auto_detect_ttem_calibration(self, source: Path) -> None:
        root = source if source.is_dir() else source.parent
        for kind in ("gex", "tfi"):
            matches = sorted(root.rglob(f"*.{kind}"))
            self._set_ttem_calibration_path(kind, matches[0] if len(matches) == 1 else None)

    def _select_ttem_calibration(self, kind: str) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self, f"Select tTEM {kind.upper()}",
            str(self._source_path or Path.cwd()),
            f"tTEM {kind.upper()} (*.{kind});;All files (*)",
        )
        if not selected:
            return
        if Path(selected).suffix.lower() != f".{kind}":
            self.log(f"Selected calibration file must end in .{kind}.", "error")
            return
        self._set_ttem_calibration_path(kind, Path(selected))
        if self._source_path is not None:
            index = self._sounding.value() - 1
            if kind == "gex":
                self._data = None  # let the newly selected GEX seed geometry fields
            self._load_sounding(index, reset_geometry=True)

    def _select_ttem_gex(self) -> None:
        self._select_ttem_calibration("gex")

    def _select_ttem_tfi(self) -> None:
        self._select_ttem_calibration("tfi")

    def _refresh_backend_state(self) -> bool:
        """Reflect the method-specific SimPEG availability in the run controls."""
        status = em_pipeline.backend_status(self._method.currentText())
        available = bool(status["available"])
        if available:
            self._backend_label.setText("Ready: SimPEG backend available.")
            self._backend_label.setStyleSheet("color:#27734b; font-size:8pt;")
        else:
            self._backend_label.setText(
                "Unavailable: install the geophysics extra (SimPEG + discretize). "
                f"{status['error']}")
            self._backend_label.setStyleSheet("color:#9b5a00; font-size:8pt;")
        if not self._inv_progress.isVisible():
            self._inv_btn.setEnabled(available)
        return available

    # -- geometry / params ---------------------------------------------------
    def _collect_geom(self) -> Dict[str, Any]:
        geom: Dict[str, Any] = {
            # Which gates exist travels with the data, so it belongs beside the
            # moment rather than with the inversion settings.
            "use_project_flags": bool(self._use_flags.isChecked()),
            "tail_max_relative_std": (float(self._tail_cut.value())
                                      if self._tail_cut.value() > 0 else None),
            "source_radius": (
                float(np.sqrt(self._loop_area.value() / np.pi))
                if self._method.currentText() == "TDEM"
                else self._src_radius.value()
            ),
            "loop_area": self._loop_area.value(),
            "ttem_gex_path": str(self._ttem_gex_path or ""),
            "ttem_tfi_path": str(self._ttem_tfi_path or ""),
            "tx_rx_sep": self._tx_rx.value(),
            "height": self._height.value(),
            "orientation": self._orient.currentText(),
            "waveform": self._waveform.currentText(),
        }
        if self._method.currentText() == "TDEM":
            geom["tem_moment"] = self._tem_moment.currentText()
            if self._data and self._data.get("temcompany"):
                system = dict(self._data.get("system", {}))
                geom["loop_turns"] = int(system.get("loop_turns", 1))
                geom["receiver_type"] = str(system.get("receiver_type", "dbdt"))
                geom["response_sign"] = float(system.get("response_sign", -1.0))
                # The export is already divided by the transmitter moment, so
                # the forward models a unit moment rather than the real one.
                if system.get("source_moment") is not None:
                    geom["source_moment"] = float(system["source_moment"])
        if self._method.currentText() == "FDEM":
            geom["component"] = self._component.currentText()
        return geom

    def _sync_lci_mode(self) -> None:
        """Show which knobs the chosen coupling actually reads.

        The passes count belongs to the block-coordinate solver, and the χ²
        search drives the simultaneous one; neither applies to the other.
        """
        mode = str(self._lci_mode.currentData())
        sequential = mode == "sequential"
        self._lci_passes.setEnabled(sequential)
        self._lci_passes_label.setEnabled(sequential)
        if not hasattr(self, "_auto_lam"):
            return  # the assistance group is built after this one
        simultaneous = mode == "simultaneous"
        self._auto_lam.setEnabled(simultaneous)
        set_rows_enabled([self._chi2_row, self._lam_trials],
                         simultaneous and self._auto_lam.isChecked())
        if hasattr(self, "_reject"):
            self._reject.setEnabled(simultaneous)
            self._sync_reject()

    def _set_lci_mode(self, value: str) -> None:
        key = str(value).strip().lower()
        index = self._lci_mode.findData(key)
        if index < 0:
            raise ValueError(
                "lci_mode must be one of simultaneous, sequential, off; "
                f"got {value!r}.")
        self._lci_mode.setCurrentIndex(index)

    def _collect_inv(self) -> Dict[str, Any]:
        result = {
            "n_layers": self._n_layers.value(),
            "min_thickness": self._min_thick.value(),
            "max_thickness": self._max_thick.value(),
            "smoothness": self._smooth.value(),
            "lateral_smoothness": self._lateral_smooth.value(),
            "lci_mode": str(self._lci_mode.currentData()),
            "lci_passes": self._lci_passes.value(),
            "auto_lambda": bool(self._auto_lam.isChecked()),
            "target_chi2": float(self._target_chi2.value()),
            "chi2_tolerance": float(self._chi2_tol.value()),
            "max_lambda_trials": int(self._lam_trials.value()),
            "reject_outliers": bool(self._reject.isChecked()),
            "outlier_threshold": float(self._reject_sigma.value()),
            "outlier_passes": int(self._reject_passes.value()),
            "min_data_fraction": float(self._min_keep.value()) / 100.0,
            "min_gates_per_sounding": int(self._min_gates.value()),
            # 0 lets the solver size its own thread pool from the machine.
            "parallel_workers": int(self._parallel_workers_spin.value()),
            "rel_error": self._rel_err.value(),
            "max_iterations": self._max_iter.value(),
            "data_scale": self._data_scale.value(),
            # 0 in the field is "auto": the workflow picks the half-space from
            # the data. The fallback below is only what a disabled auto would
            # have started from, and is ignored while auto is on.
            "auto_starting_model": float(self._start_res.value()) <= 0.0,
            "starting_resistivity": (float(self._start_res.value())
                                     if self._start_res.value() > 0.0
                                     else float(self._project_start_res)),
        }
        project_layers = self._project_layer_thicknesses
        if (
            project_layers is not None
            and project_layers.size == self._n_layers.value() - 1
            and np.isclose(project_layers[0], self._min_thick.value(), rtol=1e-3)
            and np.isclose(project_layers[-1], self._max_thick.value(), rtol=1e-3)
        ):
            result["layer_thicknesses"] = project_layers.copy()
        defaults = dict(self._data.get("inversion_defaults", {})) if self._data else {}
        if defaults.get("reference_distance") is not None:
            result["reference_distance"] = float(defaults["reference_distance"])
        if defaults.get("lateral_weight_scale") is not None:
            result["lateral_weight_scale"] = float(defaults["lateral_weight_scale"])
        return result

    # -- data ----------------------------------------------------------------
    def _load(self) -> None:
        if self._data_format.currentText() in _TEM_FORMATS:
            selected = select_directory(self, "Load EM project folder", Path.cwd())
            path = str(selected) if selected else ""
        else:
            path, _ = QFileDialog.getOpenFileName(self, "Load EM data", "", _FILE_FILTER)
        if not path:
            return
        if em_pipeline.is_ttem_source(path):
            self._method.setCurrentText("TDEM")
            self._data_format.setCurrentText(_TTEM_FORMAT)
            self._auto_detect_ttem_calibration(Path(path))
        elif em_pipeline.is_temcompany_source(path):
            self._method.setCurrentText("TDEM")
            self._data_format.setCurrentText(_TEM2GO_FORMAT)
        self._example_id = None
        self._example_note.setVisible(False)
        self._source_path = Path(path)
        self._data = None
        self._sounding.blockSignals(True)
        self._sounding.setValue(1)
        self._sounding.blockSignals(False)
        self._load_sounding(0)

    def _on_tem_moment_changed(self, _moment: str) -> None:
        if (self._source_path is not None and self._data is not None
                and self._data.get("temcompany")):
            self._load_sounding(self._sounding.value() - 1, reset_geometry=False)

    def _on_use_flags_changed(self, _checked: bool) -> None:
        """Re-read the file: the flags decide which gates, and which stations, exist.

        A station whose every gate was switched off does not appear at all while
        the flags are honoured, so the coordinates, the station count and the
        sounding cap all have to be re-read with them. Unlike the moment picker,
        this cannot keep the geometry it already had.
        """
        if (self._source_path is not None and self._data is not None
                and self._data.get("temcompany")):
            self._load_sounding(self._sounding.value() - 1, reset_geometry=True)

    def _load_example(self, example_id: str) -> Dict[str, Any]:
        """Load a documented demo dataset and apply its compatible settings."""
        catalog = em_pipeline.example_catalog()
        spec = catalog.get(example_id)
        if spec is None:
            return {"status": "failed", "error": f"Unknown EM example '{example_id}'.",
                    "valid_examples": sorted(catalog)}
        path = Path(spec["path"])
        if not path.exists():
            return {"status": "failed", "error": f"Example file not found: {path}"}
        self._method.setCurrentText(str(spec["method"]))
        self._data_format.setCurrentText(
            _TEM2GO_FORMAT
            if em_pipeline.is_temcompany_source(str(path))
            else "Generic table"
        )
        self._agent_set_params(dict(spec.get("params", {})))
        self._example_id = example_id
        self._source_path = path
        self._sounding.blockSignals(True)
        self._sounding.setValue(1)
        self._sounding.blockSignals(False)
        self._load_sounding(0)
        if self._data is None:
            return {"status": "failed", "error": f"Could not load example '{example_id}'."}
        self._example_note.setText(f"Example: {spec['note']}")
        self._example_note.setVisible(True)
        channels = (
            sum(np.asarray(item["times"]).size
                for item in self._data.get("moments", {}).values())
            if self._data.get("moments")
            else np.asarray(
                self._data.get("frequencies", self._data.get("times", []))
            ).size
        )
        self.log(f"Loaded EM example: {spec['label']}", "success")
        return {
            "status": "ok", "example": example_id, "method": self._method.currentText(),
            "channels": int(channels),
            "n_soundings": int(self._data.get("n_soundings", 1)),
            "note": spec["note"],
        }

    def _load_sounding(self, index: int, *, reset_geometry: bool = True) -> None:
        if self._source_path is None:
            return
        area_override = (
            None if reset_geometry and self._data is None
            else float(self._loop_area.value())
        )
        try:
            self._data = em_pipeline.load_sounding(
                str(self._source_path), self._method.currentText(), sounding=int(index),
                moment=self._tem_moment.currentText(),
                use_flags=bool(self._use_flags.isChecked()),
                max_relative_std=(float(self._tail_cut.value())
                                  if self._tail_cut.value() > 0 else None),
                ttem_loop_area=area_override,
                ttem_gex_path=str(self._ttem_gex_path or ""),
                ttem_tfi_path=str(self._ttem_tfi_path or ""))
        except Exception as exc:  # noqa: BLE001
            self._data = None
            self.log(f"Could not load sounding: {exc}", "error")
            self._info.setText(f"Load failed: {exc}")
            return
        if self._data.get("moments"):
            n = sum(
                np.asarray(item["times"]).size
                for item in self._data["moments"].values()
            )
        else:
            n = (
                self._data["frequencies"].size
                if "frequencies" in self._data else self._data["times"].size
            )
        n_snd = int(self._data.get("n_soundings", 1))
        self._sounding.blockSignals(True)
        self._sounding.setRange(1, max(1, n_snd))
        self._sounding.setValue(int(self._data.get("sounding", 0)) + 1)
        self._sounding.blockSignals(False)
        self._sounding_row.setVisible(n_snd > 1)
        self._line_rows.setVisible(n_snd > 1)
        self._geom_row.setVisible(n_snd > 1)
        is_temcompany = bool(self._data.get("temcompany"))
        self._tem_moment_row.setVisible(
            is_temcompany and self._method.currentText() == "TDEM")
        if reset_geometry:
            # A new file invalidates any previously loaded per-sounding geometry.
            self._geom_positions = None; self._geom_heights = None
            self._geom_x = None; self._geom_y = None
            self._geom_info.setText("Geometry: uniform spacing.")
            if is_temcompany:
                self._apply_temcompany_metadata()
            elif n_snd > 1:
                self._project_layer_thicknesses = None
                self._maybe_auto_geometry()
            else:
                self._project_layer_thicknesses = None
        snd_txt = f" · sounding {self._data.get('sounding', 0) + 1}/{n_snd}" if n_snd > 1 else ""
        kind = "line of soundings" if n_snd > 1 else "single sounding"
        format_txt = str(self._data.get("source_format", self._method.currentText()))
        moment_txt = f", {self._data.get('tem_moment')}" if is_temcompany else ""
        self._info.setText(
            f"{self._source_path.name}<br>{n} enabled channels "
            f"({format_txt}{moment_txt}, {kind}){snd_txt}")
        self.log(f"Loaded sounding {self._source_path.name}{snd_txt}", "success")
        self._register_observed_resource()
        self._plot_data()

    def _register_observed_resource(self) -> None:
        """Publish the active sounding or line, including map coordinates when loaded."""
        if (self._data is None or self._source_path is None
                or not hasattr(self.state, "register_geophysical_resource")):
            return
        method = self._method.currentText()
        n_soundings = int(self._data.get("n_soundings", 1))
        n_channels = int(
            sum(np.asarray(item["times"]).size
                for item in self._data.get("moments", {}).values())
            if self._data.get("moments")
            else self._data["frequencies"].size
            if "frequencies" in self._data
            else self._data["times"].size
        )
        resource_payload: Any = dict(self._data)
        if n_soundings > 1:
            if not self._data.get("temcompany"):
                resource_payload = {
                    "soundings": [
                        em_pipeline.load_sounding(
                            str(self._source_path), method, sounding=index,
                            moment=self._tem_moment.currentText()
                        )
                        for index in range(n_soundings)
                    ]
                }
                if (self._geom_x is not None and self._geom_y is not None
                        and self._geom_x.size >= n_soundings
                        and self._geom_y.size >= n_soundings):
                    resource_payload["coordinates"] = np.column_stack(
                        (self._geom_x[:n_soundings], self._geom_y[:n_soundings])
                    )
            # TEMcompany projects are kept lazy here: loading all SQLite stacks
            # during every preview change makes the UI pause. Joint inversion
            # can reload the full survey from the registered source path on demand.
        self.state.register_geophysical_resource(
            method,
            "observed_data",
            resource_payload,
            label=f"{method} observations · {self._source_path.name}",
            path=str(self._source_path),
            metadata={
                "channels": n_channels,
                "soundings": n_soundings,
                "sounding": int(self._data.get("sounding", 0)),
            },
            resource_id=f"{method.lower()}:observed_data:active",
        )

    def _on_sounding_changed(self, value: int) -> None:
        self._load_sounding(int(value) - 1, reset_geometry=False)

    def _apply_temcompany_metadata(self) -> None:
        """Apply geometry and system settings embedded in a TEMcompany export."""
        if self._data is None:
            return
        positions = self._data.get("positions")
        if positions is not None:
            self._geom_positions = np.asarray(positions, dtype=float).ravel()
        heights = self._data.get("heights")
        if heights is not None and not self._data.get("ttem"):
            self._geom_heights = np.asarray(heights, dtype=float).ravel()
        x = self._data.get("x")
        y = self._data.get("y")
        if x is not None and y is not None:
            self._geom_x = np.asarray(x, dtype=float).ravel()
            self._geom_y = np.asarray(y, dtype=float).ravel()
        system = dict(self._data.get("system", {}))
        if system:
            self._loop_area.setValue(float(system.get("loop_area", self._loop_area.value())))
            self._src_radius.setValue(float(system.get("source_radius", self._src_radius.value())))
            self._tx_rx.setValue(float(system.get("tx_rx_sep", self._tx_rx.value())))
            self._height.setValue(float(system.get("height", self._height.value())))
            orientation = str(system.get("orientation", "z"))
            if self._orient.findText(orientation) >= 0:
                self._orient.setCurrentText(orientation)
            waveform = str(system.get("waveform", "step_off"))
            if self._waveform.findText(waveform) >= 0:
                self._waveform.setCurrentText(waveform)
            self._data_scale.setValue(float(system.get("data_scale", 1.0)))
            self._auto_scale.setChecked(bool(system.get("auto_scale", False)))
        protocol = dict(self._data.get("protocol", {}))
        if protocol.get("uniform_std") is not None:
            # The uniform uncertainty the instrument protocol actually ran
            # with, rather than the default that happens to match it.
            self._rel_err.setValue(float(protocol["uniform_std"]))
        inversion = dict(self._data.get("inversion_defaults", {}))
        if inversion:
            self._n_layers.setValue(int(inversion.get("n_layers", self._n_layers.value())))
            self._min_thick.setValue(float(
                inversion.get("min_thickness", self._min_thick.value())))
            self._max_thick.setValue(float(
                inversion.get("max_thickness", self._max_thick.value())))
            self._smooth.setValue(float(
                inversion.get("smoothness", self._smooth.value())))
            # Kept as the fallback for when auto is switched off, rather than
            # written into the field: auto reads the ground from the data, and a
            # project's stored value is what auto exists to improve on.
            self._project_start_res = float(
                inversion.get("starting_resistivity", self._project_start_res))
            self._lateral_smooth.setValue(float(
                inversion.get("lateral_smoothness", self._lateral_smooth.value())))
            self._project_layer_thicknesses = np.asarray(
                inversion.get("layer_thicknesses", []), dtype=float).ravel()
        n = int(self._data.get("n_soundings", 1))
        # Track the station count exactly, and outside the block above, because a
        # project without inversion defaults still has stations. Only raising the
        # value would leave the cap behind when a reload brings more stations in,
        # and inverting 71 of 94 without saying so is the kind of quiet
        # truncation that is very hard to notice on the section. The ceiling goes
        # up first: stations arrive ordered by line, so a cap below the count
        # does not thin the survey, it cuts whole lines off the end of it.
        self._line_max.setMaximum(max(1, n))
        self._line_max.setValue(min(max(1, n), 200) if self._data.get("ttem") else max(1, n))
        span = (float(self._geom_positions[-1] - self._geom_positions[0])
                if self._geom_positions is not None and self._geom_positions.size else 0.0)
        crs = str(self._data.get("coordinate_system", "embedded coordinates"))
        protocol_note = (f", protocol {protocol['protocol_file']}"
                         if protocol.get("protocol_file") else "")
        self._geom_info.setText(
            f"Geometry (TEMcompany): {n} soundings, {span:.1f} m path, "
            f"{crs}{protocol_note}.")
        self.log(
            "Applied TEMcompany geometry and system settings "
            f"(loop radius {self._src_radius.value():.3f} m, "
            f"Tx-Rx {self._tx_rx.value():.2f} m).",
            "info",
        )
        if self._data.get("ttem"):
            calibration = dict(self._data.get("calibration", {}))
            self._system_note.setVisible(True)
            if calibration.get("gex_applied") and calibration.get("tfi_applied"):
                filters = dict(calibration.get("analog_lowpass", {}))
                first_filter = next(iter(filters.values()), {})
                if calibration.get("analog_lowpass_modelled"):
                    receiver = float(first_filter.get("receiver_cutoff_hz", 0.0)) / 1e3
                    tib = float(first_filter.get("tib_cutoff_hz", 0.0)) / 1e3
                    order = int(first_filter.get("tib_order", 0))
                    self._system_note.setText(
                        f"Applied {Path(calibration['gex_path']).name} and "
                        f"{Path(calibration['tfi_path']).name}. Geometry, waveform, gates, "
                        "import FIR, and the GEX analog response are active in both the "
                        f"SimPEG forward response and Jacobian (receiver 2-pole {receiver:g} "
                        f"kHz; TiB {order}-pole {tib:g} kHz)."
                    )
                    self._system_note.setStyleSheet("color:#2f6f3e; font-size:8pt;")
                else:
                    self._system_note.setText(
                        f"Applied {Path(calibration['gex_path']).name} and "
                        f"{Path(calibration['tfi_path']).name}, but the GEX has no supported "
                        "RxCoilLPFilter/TiBLowPassFilter definition."
                    )
                    self._system_note.setStyleSheet("color:#9b5a00; font-size:8pt;")
            else:
                missing = "/".join(
                    name for name, key in (("GEX", "gex_applied"), ("TFI", "tfi_applied"))
                    if not calibration.get(key)
                )
                self._system_note.setText(
                    f"Missing {missing}. Area, Tx–Rx separation and height below are "
                    "active, but full instrument calibration is incomplete."
                )
                self._system_note.setStyleSheet("color:#9b5a00; font-size:8pt;")
            complete = (
                calibration.get("gex_applied")
                and calibration.get("tfi_applied")
                and calibration.get("analog_lowpass_modelled")
            )
            level = "info" if complete else "warn"
            self.log(self._system_note.text(), level)

    def _maybe_auto_geometry(self) -> None:
        """Auto-load a companion geometry file (so UTM coordinates are used without
        a manual step). Prefers ``<stem>_geometry.csv`` / ``<stem>_geom.csv``; else a
        single ``*geom*.csv`` beside the data file. Silent if none/ambiguous."""
        if self._source_path is None:
            return
        d, stem = self._source_path.parent, self._source_path.stem
        found = next((c for c in (d / f"{stem}_geometry.csv", d / f"{stem}_geom.csv")
                      if c.exists()), None)
        if found is None:
            others = [p for p in d.glob("*geom*.csv")
                      if p.resolve() != self._source_path.resolve()]
            found = others[0] if len(others) == 1 else None
        if found is None:
            return
        try:
            g = em_pipeline.load_line_geometry(str(found))
        except Exception:  # noqa: BLE001 - a bad companion file just means manual load
            return
        self._geom_positions = g["positions"]
        self._geom_heights = g["heights"] if g["has_heights"] else None
        self._geom_x = g.get("x"); self._geom_y = g.get("y")
        extra = (" + height" if g["has_heights"] else "") + (" + map x,y" if g.get("has_xy") else "")
        self._geom_info.setText(f"Geometry (auto): {found.name}, {g['n']} soundings{extra}.")
        self.log(f"Auto-loaded companion geometry {found.name}{extra}.", "info")

    def _load_geometry(self) -> None:
        if self._source_path is None:
            self.log("Load the sounding file first.", "warn")
            return
        path, _ = QFileDialog.getOpenFileName(self, "Load line geometry", "", _FILE_FILTER)
        if not path:
            return
        try:
            g = em_pipeline.load_line_geometry(path)
        except Exception as exc:  # noqa: BLE001
            self.log(f"Could not load geometry: {exc}", "error")
            return
        self._geom_positions = g["positions"]
        self._geom_heights = g["heights"] if g["has_heights"] else None
        self._geom_x = g.get("x"); self._geom_y = g.get("y")
        span = float(g["positions"][-1] - g["positions"][0]) if g["n"] else 0.0
        extra = ""
        if g["has_heights"]:
            extra += " + height"
        if g.get("has_xy"):
            extra += " + map x,y"
        self._geom_info.setText(f"Geometry: {g['n']} soundings, {span:.0f} m span{extra}.")
        self._register_observed_resource()
        self.log(f"Loaded line geometry: {g['n']} positions{extra}.", "success")

    def _plot_data(self) -> None:
        if self._data is None:
            return
        self._curve.clear_curves()
        if self._method.currentText() == "FDEM" and "frequencies" in self._data:
            x = self._data["frequencies"]
            self._curve.add_curve(x, self._data["real"], name="obs real")
            self._curve.add_curve(x, self._data["imag"], name="obs imag")
        elif "times" in self._data:
            if self._data.get("moments"):
                for name in ("LM", "HM"):
                    item = self._data["moments"].get(name)
                    if item:
                        self._curve.add_curve(
                            item["times"], item["response"], name=f"obs {name}")
            else:
                self._curve.add_curve(
                    self._data["times"], self._data["response"], name="obs")
        self._curve.set_log_x(True)
        self._tabs.setCurrentWidget(self._curve)

    # -- inversion (one button; single sounding -> profile, line -> section) --
    def _run_inversion(self) -> None:
        if self._data is None:
            self.log("Load a sounding first.", "warn")
            return
        method = self._method.currentText()
        status = em_pipeline.backend_status(method)
        if not status["available"]:
            self.log(f"{method} inversion unavailable: {status['error']}", "warn")
            self._refresh_backend_state()
            return
        n_snd = int(self._data.get("n_soundings", 1))
        ref = float(self._ref_res.value())
        # Reference-resistivity calibration for a single sounding, or the rough
        # auto-calibration, happen up front. For a line with a reference, invert_line
        # calibrates inside the worker (off-thread) and reports the value it used.
        if self._source_path is not None:
            try:
                if ref > 0 and n_snd == 1:
                    self.log(f"Calibrating to reference {ref:.0f} Ω·m…", "info")
                    k = em_pipeline.calibrate_to_reference(
                        str(self._source_path), method, self._collect_geom(),
                        self._collect_inv(), ref, log=lambda m: self.log(m, "info"))
                    self._data_scale.setValue(float(k))
                elif ref <= 0 and self._auto_scale.isChecked():
                    k = em_pipeline.estimate_data_scale(str(self._source_path), method,
                                                        self._collect_geom(), log=lambda m: self.log(m, "info"))
                    self._data_scale.setValue(float(k))
            except Exception as exc:  # noqa: BLE001
                self.log(f"Calibration failed ({exc}); using data_scale="
                         f"{self._data_scale.value():g}", "warn")
        self._inv_busy = BusyStateController([self._inv_btn])
        self._inv_busy.start()
        self._inv_btn.setText("Inverting…")
        self._inv_progress.setVisible(True); self._inv_progress.setRange(0, 0)
        if n_snd > 1:
            self._start_line(method)
        else:
            self._start_single(method)

    def _start_single(self, method: str) -> None:
        self.log(f"Starting {method} 1D inversion…", "info")
        if self._source_path is None:
            self._on_inversion_failed("No persisted EM source is available.", False)
            self._reset_inv_button()
            return
        try:
            run = self.begin_persisted_run("em.inversion", "em.inversion")
        except Exception as exc:  # noqa: BLE001
            self._on_inversion_failed(f"Could not prepare Project run: {exc}", False)
            self._reset_inv_button()
            return
        try:
            source = self._persist_source(run.inputs_dir, method)
        except OSError as exc:
            self.fail_persisted_run(str(exc), "em.inversion")
            self._on_inversion_failed(f"Could not persist EM input: {exc}", False)
            self._reset_inv_button()
            return
        spec = WorkflowSpec(
            workflow_id="em.inversion",
            inputs={
                "data": ArtifactRef.from_path(
                    source,
                    artifact_id="em-sounding",
                    kind="em_sounding",
                    base_dir=run.run_dir,
                )
            },
            parameters={
                **self._collect_inv(),
                "method": method,
                "moment": self._tem_moment.currentText(),
                "sounding": int(self._data.get("sounding", 0)),
                "geometry": self._collect_geom(),
            },
            metadata={"source": "qt"},
        )
        recipe_path, script_path = export_workflow_bundle(
            spec, run.run_dir, stem="em_inversion"
        )
        self._reproduce.set_bundle(recipe_path, script_path)
        self._workflow_recipe_path = str(recipe_path)
        self._inv_worker = WorkflowWorker(
            spec,
            RunContext(project_root=run.run_dir, output_dir=run.outputs_dir),
        )
        self._inv_worker.logged.connect(lambda m: self.log(m, "info"))
        self._inv_worker.succeeded.connect(self._on_workflow_ok)
        self._inv_worker.failed.connect(lambda message: self._on_inversion_failed(message, False))
        self._inv_worker.finished.connect(self._reset_inv_button)
        self.register_worker(self._inv_worker)
        self._inv_worker.start()

    def _on_workflow_ok(self, result: WorkflowRunResult) -> None:
        try:
            self._on_inversion_ok(result.legacy_payload())
        finally:
            if hasattr(self.state, "update_workflow_result"):
                self.state.update_workflow_result(
                    self.module_key,
                    "em.inversion",
                    result.to_dict(),
                    recipe_path=self._workflow_recipe_path,
                )

    def _start_line(self, method: str) -> None:
        try:
            run = self.begin_persisted_run("em.line_inversion")
        except Exception as exc:  # noqa: BLE001
            self._on_line_failed(f"Could not prepare Project run: {exc}")
            self._reset_inv_button()
            return
        try:
            source = self._persist_source(run.inputs_dir, method)
        except OSError as exc:
            self.fail_persisted_run(str(exc), "em.line_inversion")
            self._on_line_failed(f"Could not persist EM input: {exc}")
            self._reset_inv_button()
            return
        out_dir = str(run.outputs_dir)
        self.log(f"Starting {method} line inversion (up to {self._line_max.value()} soundings)…", "info")
        cap = int(self._line_max.value())
        loaded = int(self._data.get("n_soundings", 1))
        if cap < loaded:
            # Stations arrive ordered by line, so the cap does not thin the
            # survey, it stops partway through and drops whatever follows.
            lines = np.asarray(self._data.get("line_numbers", []), dtype=int)
            dropped = ""
            if lines.size >= loaded:
                gone = sorted(set(lines[cap:loaded].tolist())
                              - set(lines[:cap].tolist()))
                if gone:
                    dropped = (", and survey line(s) "
                               + ", ".join(str(v) for v in gone)
                               + " are left out of the section entirely")
            self.log(f"Max soundings is {cap} of {loaded} loaded: the last "
                     f"{loaded - cap} station(s) are not inverted{dropped}.", "warn")
        worker = TaskWorker(
            em_pipeline.invert_line, str(source), method,
            self._collect_geom(), self._collect_inv(), with_log=True,
            spacing=float(self._line_spacing.value()), positions=self._geom_positions,
            heights=self._geom_heights, max_soundings=int(self._line_max.value()),
            ref_resistivity=float(self._ref_res.value()), out_dir=Path(out_dir),
            # The whole model comes back with its sensitivity; the Resistivity
            # model tab applies the depth cut, so the threshold can be moved
            # without inverting again (the ERT view works the same way).
            doi_blank=False)
        worker.logged.connect(lambda m: self.log(m, "info"))
        worker.succeeded.connect(self._on_line_ok)
        worker.failed.connect(self._on_line_failed)
        worker.finished.connect(self._reset_inv_button)
        self._line_worker = self.register_worker(worker)
        worker.start()

    def _persist_source(self, inputs_dir: Path, method: str) -> Path:
        """Store the imported soundings so a recorded EM run is self-contained.

        This used to copy whatever the user selected, which made every inversion
        cost another copy of the acquisition: a TEMcompany project folder is
        hundreds of megabytes, and a Project opened on the survey folder itself
        made ``copytree`` walk into the duplicate it was still writing. The
        soundings are the only part the inversion reads and they compress to a
        fraction of that, so the run keeps those instead.
        """
        if self._source_path is None:
            raise FileNotFoundError("No persisted EM source is available.")
        source = self._source_path
        geom = self._collect_geom()
        try:
            return em_pipeline.save_sounding_container(
                inputs_dir / "em_soundings",
                str(source),
                method,
                moment=str(self._tem_moment.currentText()),
                use_flags=bool(geom.get("use_project_flags", True)),
                max_relative_std=geom.get("tail_max_relative_std"),
                ttem_loop_area=geom.get("loop_area"),
                ttem_gex_path=geom.get("ttem_gex_path"),
                ttem_tfi_path=geom.get("ttem_tfi_path"),
                progress=lambda message: self.log(message, "info"),
            )
        except Exception as exc:  # noqa: BLE001 - a run must not die on bookkeeping
            # Falling back to a reference keeps the run going and keeps the
            # recipe resolvable; it only costs the self-contained property,
            # which is what an unreadable survey would have cost anyway.
            self.log(
                f"Could not store the imported soundings ({exc}); the run "
                "references the original data where it is.",
                "warn",
            )
            return source

    def _reset_inv_button(self) -> None:
        if self._inv_busy is not None:
            self._inv_busy.finish()
            self._inv_busy = None
        self._inv_btn.setText("Run inversion")
        self._inv_progress.setVisible(False)
        self._refresh_backend_state()

    def _on_inversion_ok(self, result: dict) -> None:
        self._last_result = result
        self._last_section = None   # the export button now writes this profile
        self._inv_export.setEnabled(True)
        png = self._render_inversion(result)
        if png:
            self._inv_view.set_image_file(png)
            self._view_row.setVisible(False)
            self._model_stack.setCurrentWidget(self._inv_view)
            self._tabs.setCurrentWidget(self._model_tab)
        self._quality_view.show_quality(
            {"chi2": float(result["chi2"]), "n_data": result.get("n_data"),
             "method": (
                 f"{result['method']} joint LM+HM 1D"
                 if result.get("joint_moments")
                 else f"{result['method']} 1D Occam"
             ),
              "extra": {"layers": result.get("n_layers"), "forward evals": result.get("nfev"),
                        "iterations": len(result.get("convergence") or [])},
             "note": "1D least-squares inversion (χ² is the mean weighted squared residual)."},
            convergence=result.get("convergence"), title=f"{result['method']} inversion")
        self.log(f"{result['method']} inversion complete (chi2={result['chi2']:.3f}).", "success")
        if hasattr(self.state, "register_geophysical_resource"):
            method = str(result["method"])
            self.state.register_geophysical_resource(
                method, "model", np.asarray(result["resistivity"], dtype=float),
                label=f"Latest {method} resistivity model",
                metadata={"chi2": float(result["chi2"]),
                          "thickness": np.asarray(result.get("thickness", []), dtype=float)},
                resource_id=f"{method.lower()}:model:latest",
            )
        self.report_result({"method": result["method"], "chi2": float(result["chi2"]),
                            "n_data": result.get("n_data"), "nfev": result.get("nfev"),
                            "n_layers": int(np.asarray(result["resistivity"]).size)})

    def _on_line_ok(self, result: dict) -> None:
        self._last_section = result
        self._last_result = None    # the export button now writes the section
        self._inv_export.setEnabled(True)
        # The overview is the only UI-success step that also writes an
        # artifact.  Finalize even if rendering it fails so a successful,
        # expensive inversion is never left permanently marked as running.
        try:
            self._populate_overview(result)
        finally:
            self._finish_line_run(result)
        if "data_scale" in result:  # show the value the worker actually used (calibrated)
            self._data_scale.setValue(float(result["data_scale"]))
        self._section_view.show_model(result["edges"], result["model3d"],
                                      label=result["label"], cmap=result["cmap"],
                                      log_scale=result.get("log_scale", True))
        self._populate_plan(result)
        self._view_row.setVisible(True)
        self._view_mode.blockSignals(True); self._view_mode.setCurrentIndex(0)
        self._view_mode.blockSignals(False)
        # The map + section overview is what a reader needs first, so open there.
        self._model_stack.setCurrentWidget(self._overview_view)
        self._tabs.setCurrentWidget(self._model_tab)
        rng = result.get("model_range", [float("nan"), float("nan")])
        chi2 = result.get("chi2_global", result.get("chi2"))
        sounding_mean = result.get("chi2_sounding_mean")
        sounding_median = result.get("chi2_sounding_median")
        report = result.get("lci_report") or {}
        coupled = report.get("mode") == "simultaneous"
        chi_txt = ""
        if isinstance(chi2, float) and chi2 == chi2:
            chi_txt = f", global χ²={chi2:.2f}"
            if isinstance(sounding_median, float) and sounding_median == sounding_median:
                chi_txt += f", sounding median χ²={sounding_median:.2f}"
        self.log(f"{result['method']} line inversion complete: {result['n_soundings']} soundings, "
                 f"resistivity {rng[0]:.3g}..{rng[1]:.3g} Ω·m{chi_txt}.", "success")
        if coupled:
            # Terse: one line for how it stopped, one more only if the search moved.
            self.log(f"Coupled line solve: {report.get('iterations', 0)} iterations "
                     f"({report.get('stop_reason', '')}), "
                     f"{report.get('n_lateral_ties', 0)} lateral ties.", "info")
            search = report.get("lambda_search") or {}
            scale = float(report.get("smoothness_scale", 1.0))
            if len(search.get("trials", [])) > 1 and abs(scale - 1.0) > 1e-9:
                self.log(f"Smoothness scaled to {scale:.3g} to reach the χ² target "
                         f"(as set: χ²={float(search.get('fixed_chi2', float('nan'))):.2f}).",
                         "warning")

        method_txt = f"{result['method']} line ("
        method_txt += ("simultaneous LCI" if coupled
                       else "block-coordinate LCI" if report
                       else "independent 1D") + ")"
        extra = {"soundings": result.get("n_soundings"), "layers": result.get("n_layers")}
        if (isinstance(sounding_mean, float) and sounding_mean == sounding_mean
                and isinstance(sounding_median, float) and sounding_median == sounding_median):
            extra["sounding χ²"] = f"mean {sounding_mean:.2f}, median {sounding_median:.2f}"
        residual_median = result.get("data_residual_sounding_median")
        if isinstance(residual_median, float) and residual_median == residual_median:
            extra["normalized residual (√χ²)"] = f"median {residual_median:.2f}"
        if coupled:
            extra["stop"] = report.get("stop_reason", "")
        outliers = dict(result.get("outliers") or {})
        if outliers.get("enabled"):
            extra["data"] = f"{outliers.get('kept')} of {outliers.get('n_start')} gates kept"
            if outliers.get("limited_by_floor"):
                extra["data"] += " (floor reached)"
        doi = np.asarray(result.get("doi", []), dtype=float)
        if doi.size and np.isfinite(doi).any():
            extra["DOI"] = (f"median {np.nanmedian(doi):.0f} m "
                            f"(S ≥ {float(result.get('doi_threshold', DOI_SENSITIVITY_THRESHOLD)):g})")
        self._quality_view.show_quality(
            {"chi2": float(chi2) if isinstance(chi2, float) else float("nan"),
             "n_data": result.get("n_data"),
             "iterations": report.get("iterations") if coupled else None,
             "method": method_txt,
             "extra": extra,
             "convergence_track": report.get("convergence_track"),
             "note": (
                 "Global gate-weighted χ² from one coupled solve; every "
                 "sounding was fitted with its neighbours' models constrained "
                 "at the same time."
                 if coupled else
                 "Global gate-weighted χ² from the final sounding fits; adjacent "
                 "models are coupled by the selected lateral smoothness."
                 if report else
                 "Global gate-weighted χ² over independently inverted soundings."
             )},
            convergence=(report.get("chi2_history") if coupled else None),
            title=f"{result['method']} line inversion",
            per_item={
                "values": result.get("chi2_list") or [],
                "x": result.get("positions"),
                "groups": result.get("line_numbers"),
                "counts": result.get("data_count_list"),
                "x_label": "Distance along survey (m)",
                "item_label": "sounding",
            })
        saved = result.get("saved") or []
        for path in saved:
            self.log(f"Saved {Path(path).name} to {path}", "info")
        self.report_result({"method": result["method"], "n_soundings": result.get("n_soundings"),
                            "global_chi2": chi2,
                            "sounding_median_chi2": sounding_median,
                            "section_npz": saved[0] if saved else None})

    def _finish_line_run(self, result: dict) -> None:
        self.finish_persisted_run({
            "status": "success",
            "summary": {
                "method": result.get("method"),
                "n_soundings": result.get("n_soundings"),
                "model_range": result.get("model_range"),
            },
            "metrics": {
                "global_chi2": result.get("chi2_global", result.get("chi2")),
                "sounding_mean_chi2": result.get("chi2_sounding_mean"),
                "sounding_median_chi2": result.get("chi2_sounding_median"),
            },
            "artifacts": [],
            "warnings": [],
            "provenance": {"operation_id": "em.line_inversion"},
        }, "em.line_inversion")

    def _populate_overview(self, result: dict) -> None:
        """Feed the map + section overview and save it beside the section data."""
        n_pos = int(np.asarray(result["model3d"]).shape[0])
        x = y = None
        if (self._geom_x is not None and self._geom_y is not None
                and self._geom_x.size >= n_pos and self._geom_y.size >= n_pos):
            x, y = self._geom_x[:n_pos], self._geom_y[:n_pos]
        # Geographic coordinates ride along only so the map can place tiles; the
        # section and the axes stay in the projected metres of the survey.
        lon = np.asarray((self._data or {}).get("longitude", []), dtype=float).ravel()
        lat = np.asarray((self._data or {}).get("latitude", []), dtype=float).ravel()
        self._overview_view.show_result(
            result, x=x, y=y,
            lon=lon[:n_pos] if lon.size >= n_pos else None,
            lat=lat[:n_pos] if lat.size >= n_pos else None)
        active = self.state.active_run(self.module_key, "em.line_inversion")
        out = active.outputs_dir if active is not None else self.state.ensure_results_store().scratch_dir(self.module_key)
        saved = self._overview_view.save_figure(out / "em_line_overview.png")
        if saved:
            result.setdefault("saved", []).append(saved)

    def _populate_plan(self, result: dict) -> None:
        """Feed the plan-view depth-slice map from a line-inversion result: each
        sounding's map coordinate + its resistivity per depth layer."""
        model = np.asarray(result["model3d"], dtype=float)[:, 0, :]  # (n_pos, n_layers), deepest-first
        n_pos = model.shape[0]
        depth_edges = np.asarray(result["depth_edges"], dtype=float)
        depth_ctr = 0.5 * (depth_edges[:-1] + depth_edges[1:])       # surface-ordered
        res_surface = model[:, ::-1].copy()                          # surface-ordered in depth
        # The line result is no longer blanked at the source, so apply the same
        # depth-of-investigation cut here; a plan slice below it would map
        # regularization across the survey.
        sensitivity = np.asarray(result.get("sensitivity", []), dtype=float)
        if sensitivity.shape == res_surface.shape:
            res_surface[sensitivity < float(
                result.get("doi_threshold", DOI_SENSITIVITY_THRESHOLD))] = np.nan
        if (self._geom_x is not None and self._geom_y is not None
                and self._geom_x.size >= n_pos and self._geom_y.size >= n_pos):
            xy = np.column_stack([self._geom_x[:n_pos], self._geom_y[:n_pos]])
            x_label, y_label = "Easting (m)", "Northing (m)"
        else:  # no map coordinates loaded: lay soundings along the distance axis
            pos = np.asarray(result["positions"], dtype=float)[:n_pos]
            xy = np.column_stack([pos, np.zeros_like(pos)])
            x_label, y_label = "Distance along line (m)", ""
        self._plan_view.show_slices(xy, res_surface, depth_ctr,
                                    label=result.get("label", "resistivity (Ω·m)"),
                                    log_scale=result.get("log_scale", True),
                                    x_label=x_label, y_label=y_label)

    def _on_inversion_failed(self, message: str, backend: bool) -> None:
        self.fail_persisted_run(message, "em.inversion")
        self.log(f"Inversion {'unavailable' if backend else 'failed'}: {message}",
                 "warn" if backend else "error")

    def _on_line_failed(self, message: str) -> None:
        self.fail_persisted_run(message, "em.line_inversion")
        if any(k in message.lower() for k in ("backend", "simpeg", "discretize")):
            self.log(f"Line inversion needs SimPEG: {message}", "warn")
        else:
            self.log(f"Line inversion failed: {message}", "error")

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
                if result.get("joint_moments"):
                    for index, name in enumerate(("LM", "HM")):
                        item = result["moments"].get(name)
                        if not item:
                            continue
                        color = f"C{index}"
                        ax2.plot(item["times"], item["obs"], "o", ms=4,
                                 color=color, label=f"obs {name}")
                        ax2.plot(item["times"], item["pred"], "-",
                                 color=color, label=f"pred {name}")
                else:
                    t = result["times"]
                    ax2.plot(t, result["obs"], "o", ms=4,
                             color="tab:blue", label="obs")
                    ax2.plot(t, result["pred"], "-",
                             color="tab:blue", label="pred")
                ax2.set_xlabel("Time (s)")
            ax2.set_xscale("log"); ax2.set_title(f"Data fit (chi2={result['chi2']:.2f})")
            ax2.grid(True, which="both", alpha=0.3); ax2.legend(fontsize=8, frameon=False)
            fig.tight_layout()
            active = self.state.active_run(self.module_key, "em.inversion")
            out = active.outputs_dir if active is not None else self.state.ensure_results_store().scratch_dir(self.module_key)
            p = out / f"{result['method'].lower()}_inversion.png"
            fig.savefig(p, dpi=160, bbox_inches="tight"); plt.close(fig)
            return str(p)
        except Exception as exc:  # noqa: BLE001
            self.log(f"Could not render inversion: {exc}", "warn")
            return None

    def _export_inversion(self) -> None:
        # A line section and a single sounding are different tables, and the
        # button is the same one, so whichever ran last is what gets written.
        section = getattr(self, "_last_section", None)
        if not section and not self._last_result:
            self.log("Run an inversion first.", "warn")
            return
        folder = select_directory(
            self, "Export recovered model to folder",
            self.state.output_dir or Path.cwd(),
        )
        if not folder:
            return
        # An export that fails silently looks the same as one that wrote nothing,
        # so the failure has to reach the log rather than only stderr.
        try:
            if section:
                paths = em_pipeline.save_line_csv(section, Path(folder))
                self.log(f"Exported the section as {len(paths)} CSV file(s) to {folder}: "
                         f"{int(section.get('n_soundings', 0))} soundings x "
                         f"{int(section.get('n_layers', 0))} layers.", "success")
                return
            paths = em_pipeline.save_inversion(self._last_result, folder)
            self.log(f"Exported recovered model ({len(paths)} files) to {folder}", "success")
        except Exception as exc:  # noqa: BLE001
            self.log(f"EM model export failed: {exc}", "error")

    def export_actions(self):
        if getattr(self, "_last_section", None):
            return [("Line section model (CSV)", self._export_inversion)]
        if self._last_result:
            return [("Recovered sounding model (CSV + npy)", self._export_inversion)]
        return []

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
                {"name": "use_example_data",
                 "args": {"example": [
                     "east_river_vtem", "skytem_bhmar",
                     "synthetic_fdem", "synthetic_tem_lci",
                 ]},
                 "desc": ("Load a documented EM example. Default is east_river_vtem; "
                          "synthetic_fdem is the reproducible FDEM demo and "
                          "synthetic_tem_lci is the LM+HM line/LCI test.")},
                {"name": "set_method", "args": {"method": list(em_pipeline.METHODS)},
                 "desc": "Choose the EM method (FDEM or TDEM)."},
                {"name": "load_data",
                 "args": {"path": "str", "sounding": "int (optional, 1-based)",
                          "moment": list(em_pipeline.TEMCOMPANY_MOMENTS)},
                 "desc": ("Load a sounding file or TEMcompany/TEM2Go project directory. "
                          "For TEMcompany data, moment selects LM+HM, HM, or LM and "
                          "defaults to the joint LM+HM workflow. "
                          "If the source has several soundings, 'sounding' picks the preview; "
                          "the inversion still uses the survey.")},
                {"name": "set_params", "args": {"params": {"<key>": "value"}},
                 "desc": ("Set parameters. Geometry: source_radius, loop_area, tx_rx_sep, height, orientation "
                          "(z/x/y), component (secondary/total/both), waveform. Inversion: n_layers, "
                          "min_thickness, max_thickness, starting_resistivity, smoothness, "
                          "rel_error, data_scale "
                          "(calibration multiplier), auto_scale (bool, rough), ref_resistivity "
                          "(ohm-m; calibrate the absolute level to a known value, the reliable "
                          "option), max_iterations. TEMcompany: tem_moment "
                          "(LM+HM/HM/LM), use_project_flags (bool; false also "
                          "imports the gates the project's own QC switched off). "
                          "Line: spacing, max_soundings, "
                          "lateral_smoothness, lci_mode "
                          "(simultaneous/sequential/off), lci_passes "
                          "(block-coordinate only). Fit assistance (simultaneous "
                          "only): auto_lambda, target_chi2, chi2_tolerance, "
                          "max_lambda_trials, reject_outliers (bool; drop gates "
                          "the model cannot explain and re-solve), "
                          "outlier_threshold (sigma), outlier_passes, "
                          "min_data_fraction (0-1 floor on the gates kept). "
                          "Start: auto_starting_model (bool; pick the starting half-space "
                          "from the data before inverting). "
                          "Speed: parallel_workers (threads for the per-sounding "
                          "forward and Jacobian; 0 sizes it from the machine, and "
                          "the result is identical either way).")},
                {"name": "auto_calibrate", "args": {},
                 "desc": ("Estimate the data_scale calibration from the loaded data and set it. Use "
                          "for normalized airborne data (e.g. moment-normalized dB/dt).")},
                {"name": "load_geometry", "args": {"path": "str"},
                 "desc": ("Load per-sounding line geometry (a file with along-line position and "
                          "optional sensor height) so the section uses real distances instead of "
                          "uniform spacing.")},
                {"name": "run_inversion", "args": {},
                 "desc": ("Run the inversion. A single-sounding file gives a 1D layered model; a "
                          "multi-sounding LM+HM project uses same-line lateral constraints to "
                          "produce a position x depth resistivity section. If auto-calibrate is "
                          "on, data_scale is estimated first.")},
                {"name": "get_status", "args": {},
                 "desc": "Report the method, loaded data, and last result."},
            ],
        }

    def agent_apply(self, action: str, args: Dict[str, Any]) -> Dict[str, Any]:
        args = args or {}
        handlers = {
            "use_example_data": lambda: self._load_example(args.get("example", "east_river_vtem")),
            "set_method": lambda: self._agent_set_method(args.get("method")),
            "load_data": lambda: self._agent_load(
                args.get("path"), args.get("sounding", 1), args.get("moment")),
            "set_params": lambda: self._agent_set_params(args.get("params", args)),
            "auto_calibrate": lambda: self._agent_auto_calibrate(),
            "load_geometry": lambda: self._agent_load_geometry(args.get("path")),
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
        n_snd = int(self._data.get("n_soundings", 1)) if self._data else 0
        return {
            "status": "ok",
            "method": self._method.currentText(),
            "data_loaded": self._data is not None,
            "source": str(self._source_path or ""),
            "example": self._example_id,
            "n_soundings": n_snd,
            "tem_moment": (
                self._tem_moment.currentText()
                if self._data and self._data.get("temcompany") else None
            ),
            "backend": em_pipeline.backend_status(self._method.currentText()),
            "last_result_keys": sorted(last.keys()),
        }

    def _agent_set_method(self, method: Any) -> Dict[str, Any]:
        methods = list(em_pipeline.METHODS)
        if method not in methods:
            return {"status": "failed", "error": f"Unknown method '{method}'.", "valid": methods}
        self._method.setCurrentText(method)
        return {"status": "ok", "method": self._method.currentText()}

    def _agent_load(
        self, path: Any, sounding: Any = 1, moment: Any = None
    ) -> Dict[str, Any]:
        if not path:
            return {
                "status": "failed",
                "error": "Provide 'path' to a sounding file or TEMcompany project directory.",
            }
        p = Path(str(path))
        if not p.exists():
            return {"status": "failed", "error": f"File not found: {p}"}
        if em_pipeline.is_ttem_source(str(p)):
            self._method.setCurrentText("TDEM")
            self._data_format.setCurrentText(_TTEM_FORMAT)
            self._auto_detect_ttem_calibration(p)
            if moment is not None:
                selected = str(moment).upper()
                if selected not in em_pipeline.TEMCOMPANY_MOMENTS:
                    return {
                        "status": "failed",
                        "error": f"TEMcompany moment must be one of "
                                 f"{em_pipeline.TEMCOMPANY_MOMENTS}.",
                    }
                self._tem_moment.setCurrentText(selected)
        elif em_pipeline.is_temcompany_source(str(p)):
            self._method.setCurrentText("TDEM")
            self._data_format.setCurrentText(_TEM2GO_FORMAT)
            if moment is not None:
                selected = str(moment).upper()
                if selected not in em_pipeline.TEMCOMPANY_MOMENTS:
                    return {
                        "status": "failed",
                        "error": f"TEMcompany moment must be one of "
                                 f"{em_pipeline.TEMCOMPANY_MOMENTS}.",
                    }
                self._tem_moment.setCurrentText(selected)
        self._example_id = None
        self._example_note.setVisible(False)
        self._source_path = Path(p)
        self._data = None
        try:
            self._load_sounding(max(0, int(sounding) - 1))
        except Exception as exc:  # noqa: BLE001
            return {"status": "failed", "error": f"Could not load sounding: {exc}"}
        if self._data is None:
            return {"status": "failed", "error": "Could not load sounding (wrong method/format?)."}
        arr = self._data.get("frequencies", self._data.get("times"))
        n = int(arr.size) if arr is not None else 0
        return {"status": "ok", "channels": n, "method": self._method.currentText(),
                "n_soundings": int(self._data.get("n_soundings", 1)),
                "sounding": int(self._data.get("sounding", 0)) + 1,
                "source_format": self._data.get("source_format"),
                "tem_moment": self._data.get("tem_moment")}

    def _agent_set_params(self, params: Any) -> Dict[str, Any]:
        if not isinstance(params, dict):
            return {"status": "failed", "error": "Provide 'params' as a JSON object."}

        def set_combo(combo, value):
            items = [combo.itemText(i) for i in range(combo.count())]
            if str(value) not in items:
                raise ValueError(f"must be one of {items}")
            combo.setCurrentText(str(value))

        handlers = {
            "source_radius": lambda v: self._src_radius.setValue(float(v)),
            "loop_area": lambda v: self._loop_area.setValue(float(v)),
            "tx_rx_sep": lambda v: self._tx_rx.setValue(float(v)),
            "height": lambda v: self._height.setValue(float(v)),
            "orientation": lambda v: set_combo(self._orient, v),
            "component": lambda v: set_combo(self._component, v),
            "waveform": lambda v: set_combo(self._waveform, v),
            "tem_moment": lambda v: set_combo(self._tem_moment, str(v).upper()),
            "use_project_flags": lambda v: self._use_flags.setChecked(bool(v)),
            "tail_max_relative_std": lambda v: self._tail_cut.setValue(
                0.0 if v is None else float(v)),
            "n_layers": lambda v: self._n_layers.setValue(int(v)),
            "min_thickness": lambda v: self._min_thick.setValue(float(v)),
            "max_thickness": lambda v: self._max_thick.setValue(float(v)),
            "starting_resistivity": lambda v: self._start_res.setValue(float(v)),
            "smoothness": lambda v: self._smooth.setValue(float(v)),
            "lateral_smoothness": lambda v: self._lateral_smooth.setValue(float(v)),
            "lci_mode": lambda v: self._set_lci_mode(str(v)),
            "lci_passes": lambda v: self._lci_passes.setValue(int(v)),
            "auto_lambda": lambda v: self._auto_lam.setChecked(bool(v)),
            "target_chi2": lambda v: self._target_chi2.setValue(float(v)),
            "chi2_tolerance": lambda v: self._chi2_tol.setValue(float(v)),
            "max_lambda_trials": lambda v: self._lam_trials.setValue(int(v)),
            "parallel_workers": lambda v: self._parallel_workers_spin.setValue(
                max(0, min(int(v), self._parallel_workers_spin.maximum()))),
            "auto_starting_model": lambda v: (self._start_res.setValue(0.0)
                                              if bool(v) else None),
            "reject_outliers": lambda v: self._reject.setChecked(bool(v)),
            "outlier_threshold": lambda v: self._reject_sigma.setValue(float(v)),
            "outlier_passes": lambda v: self._reject_passes.setValue(int(v)),
            "min_data_fraction": lambda v: self._min_keep.setValue(float(v) * 100.0),
            "min_gates_per_sounding": lambda v: self._min_gates.setValue(int(v)),
            "rel_error": lambda v: self._rel_err.setValue(float(v)),
            "data_scale": lambda v: self._data_scale.setValue(float(v)),
            "auto_scale": lambda v: self._auto_scale.setChecked(bool(v)),
            "ref_resistivity": lambda v: self._ref_res.setValue(float(v)),
            "max_iterations": lambda v: self._max_iter.setValue(int(v)),
            "spacing": lambda v: self._line_spacing.setValue(float(v)),
            "max_soundings": lambda v: self._line_max.setValue(int(v)),
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
        if "loop_area" in applied:
            self._on_ttem_loop_area_changed()
        return {"status": "ok" if applied else "failed", "applied": applied, "ignored": ignored}

    def _agent_run_inversion(self) -> Dict[str, Any]:
        if self._data is None:
            return {"status": "failed", "error": "Load a sounding first."}
        backend = em_pipeline.backend_status(self._method.currentText())
        if not backend["available"]:
            self._refresh_backend_state()
            return {"status": "failed", "error": f"EM backend unavailable: {backend['error']}",
                    "backend": backend}
        n_snd = int(self._data.get("n_soundings", 1))
        self._run_inversion()
        return {"status": "started",
                "message": ("EM %s inversion started. Ask for status shortly."
                            % ("line" if n_snd > 1 else "1D")),
                "method": self._method.currentText(), "n_soundings": n_snd}

    def _agent_auto_calibrate(self) -> Dict[str, Any]:
        if self._data is None or self._source_path is None:
            return {"status": "failed", "error": "Load a sounding first."}
        try:
            k = em_pipeline.estimate_data_scale(str(self._source_path), self._method.currentText(),
                                                self._collect_geom(), log=lambda m: self.log(m, "info"))
        except Exception as exc:  # noqa: BLE001
            return {"status": "failed", "error": f"Auto-calibration failed: {exc}"}
        self._data_scale.setValue(float(k))
        return {"status": "ok", "data_scale": float(k)}

    def _agent_load_geometry(self, path: Any) -> Dict[str, Any]:
        if not path:
            return {"status": "failed", "error": "Provide 'path' to a geometry file."}
        p = Path(str(path))
        if not p.exists():
            return {"status": "failed", "error": f"File not found: {p}"}
        try:
            g = em_pipeline.load_line_geometry(str(p))
        except Exception as exc:  # noqa: BLE001
            return {"status": "failed", "error": f"Could not load geometry: {exc}"}
        self._geom_positions = g["positions"]
        self._geom_heights = g["heights"] if g["has_heights"] else None
        self._geom_x = g.get("x"); self._geom_y = g.get("y")
        extra = (" + height" if g["has_heights"] else "") + (" + map x,y" if g.get("has_xy") else "")
        span = float(g["positions"][-1] - g["positions"][0]) if g["n"] else 0.0
        self._geom_info.setText(f"Geometry: {g['n']} soundings, {span:.0f} m span{extra}.")
        self._register_observed_resource()
        return {"status": "ok", "n": g["n"], "has_heights": g["has_heights"],
                "has_xy": bool(g.get("has_xy"))}
