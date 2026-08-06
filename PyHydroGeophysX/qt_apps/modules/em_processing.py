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

from PyHydroGeophysX.workflows import em1d as em_pipeline
from PyHydroGeophysX.qt_apps import io_utils, theme
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


class EMProcessingModule(BaseModule):
    module_key = "em_processing"
    module_title = "EM Processing"

    def __init__(self, state: Any, log: LogFn, parent=None) -> None:
        super().__init__(state, log, parent)
        self._data: Optional[Dict[str, np.ndarray]] = None
        self._source_path: Optional[Path] = None
        self._last_result: Optional[Dict[str, Any]] = None
        self._last_section: Optional[Dict[str, Any]] = None
        self._inv_worker: Optional[WorkflowWorker] = None
        self._inv_busy: Optional[BusyStateController] = None
        self._line_worker: Optional[TaskWorker] = None
        self._workflow_recipe_path = ""
        self._geom_positions: Optional[np.ndarray] = None
        self._geom_heights: Optional[np.ndarray] = None
        self._geom_x: Optional[np.ndarray] = None
        self._geom_y: Optional[np.ndarray] = None
        self._example_id: Optional[str] = None
        self._project_layer_thicknesses: Optional[np.ndarray] = None

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
        # Wide enough to fit the controls without a horizontal scrollbar.
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(450); scroll.setMaximumWidth(500)
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
        return scroll

    def _build_loader_group(self) -> QGroupBox:
        box = QGroupBox("Load sounding data"); v = QVBoxLayout(box)
        self._method = QComboBox(); self._method.addItems(list(em_pipeline.METHODS))
        self._method.currentTextChanged.connect(self._on_method_changed)
        self._data_format = QComboBox()
        self._data_format.addItems(["Generic table", "TEMcompany / TEM2Go"])
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
            "On: import only the gates the acquisition software left switched on, "
            "which is what its own inversion used.\n\n"
            "Off: import every gate that holds a finite, non-dummy value, including "
            "the ones its QC switched off, and let this module's own outlier "
            "rejection decide. A noisy ground survey may leave only a quarter of "
            "the gates in use, so this can be several times more data; each gate "
            "still carries its recorded stack error, so a noisy one is weighted as "
            "noisy. Turn on 'Reject outliers' with it.")
        self._use_flags.toggled.connect(self._on_use_flags_changed)
        moment_form.addRow("", self._use_flags)
        self._tem_moment_row.setVisible(False)
        v.addWidget(self._tem_moment_row)

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
        self._sounding.setToolTip("This file holds several soundings; pick which one to preview.")
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
        self._tx_rx = self._dspin(f["tx_rx_sep"], 0.0, 500.0, 1.0, 1)
        self._height = self._dspin(f["height"], 0.0, 500.0, 1.0, 1)
        self._orient = QComboBox(); self._orient.addItems(["z", "x", "y"])
        self._component = QComboBox(); self._component.addItems(["secondary", "total", "both"])
        self._waveform = QComboBox()
        form.addRow("Source radius (m)", self._src_radius)
        self._tx_rx_label = QLabel("Tx-Rx sep (m)")
        form.addRow(self._tx_rx_label, self._tx_rx)
        form.addRow("Height (m)", self._height)
        form.addRow("Orientation", self._orient)
        self._component_label = QLabel("Component")
        form.addRow(self._component_label, self._component)
        form.addRow("Waveform", self._waveform)
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
        form.addRow("Smoothness", self._smooth)
        form.addRow("Max iterations", self._max_iter)
        return box

    def _build_errors_group(self) -> QGroupBox:
        box = QGroupBox("Data errors and calibration"); form = QFormLayout(box)
        d = em_pipeline.DEFAULT_INVERSION
        self._rel_err = self._dspin(d["rel_error"], 0.0, 1.0, 0.01, 3)
        self._rel_err.setToolTip(
            "Assumed relative error per gate. Where the file carries a stack error of "
            "its own (TEMcompany exports do), this is a FLOOR on it, not a "
            "replacement. χ² is measured against the result, so it decides what "
            "\"fitting the data\" means. A stack error only measures repeatability; "
            "raise this to make room for the error in representing the ground as 1D "
            "layers, which is what keeps χ² out of the hundreds on ground TDEM.")
        self._data_scale = self._dspin(1.0, 1e-4, 1e6, 0.1, 4)
        self._data_scale.setToolTip(
            "Multiply the observed data before inversion. Use for data in normalized units "
            "(e.g. moment-normalized airborne dB/dt) that need a system calibration constant. "
            "1.0 = no scaling.")
        self._auto_scale = QCheckBox("Auto-calibrate")
        self._auto_scale.setChecked(True)
        self._auto_scale.setToolTip("Rough amplitude guess from the data shape (de-rails the "
                                    "inversion, but the absolute resistivity is only approximate). "
                                    "For reliable absolute values, use Reference resistivity instead.")
        self._ref_res = self._dspin(0.0, 0.0, 1e5, 10.0, 1)
        self._ref_res.setToolTip(
            "Pin the ABSOLUTE resistivity to a known/expected value (ohm-m) from a borehole or "
            "regional geology — it ties the data amplitude to a half-space at this value, the "
            "same way for every dataset, so results are consistent (the data alone cannot fix the "
            "level). Recovered resistivity lands near this value. 0 = off; overrides Auto-calibrate.")
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
        self._line_max = self._ispin(12, 1, 500)
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
            "Block-coordinate passes. The simultaneous solver does not use this.")
        self._lci_passes_label = QLabel("passes")
        self._lci_mode_row = merged_row(
            self._lci_mode, self._lci_passes_label, self._lci_passes)
        self._lci_mode.currentIndexChanged.connect(self._sync_lci_mode)
        form.addRow("Sounding spacing (m)", self._line_spacing)
        form.addRow("Max soundings", self._line_max)
        form.addRow("Lateral smoothness", self._lateral_smooth)
        form.addRow("Coupling", self._lci_mode_row)
        self._sync_lci_mode()
        self._line_rows = box
        box.setVisible(False)
        return box

    def _build_assist_group(self) -> QGroupBox:
        box = QGroupBox("Fit assistance"); form = QFormLayout(box)
        self._auto_lam = QCheckBox("Auto-λ: re-solve to reach target χ²")
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
            "one at the smoothness you set.")
        form.addRow("Max trials", self._lam_trials)

        # The other answer to a high χ²: some gates are wrong rather than the
        # model being too stiff. Same controls, wording and defaults as ERT.
        self._reject = QCheckBox("Reject outliers: drop gates the model cannot explain")
        self._reject.setChecked(False)
        self._reject.setToolTip(
            "After the line has converged, drop the time gates whose residual exceeds "
            "the cut below and solve again at the same smoothness. This is what brings "
            "χ² down when a minority of gates are simply bad; it also shrinks the data "
            "set, so the floor below keeps it from gutting the survey. Cutting is per "
            "gate, not per sounding: a TDEM station may carry only a handful of gates.")
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
            self._min_keep, "of the survey, and", self._min_gates, "gates per sounding")
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
        self._inv_export = QPushButton("Export recovered model (npy + csv)…")
        self._inv_export.setIcon(theme.icon("fa5s.file-export"))
        self._inv_export.setEnabled(False)
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
        if not fdem and hasattr(self, "_tem_moment_row"):
            self._tem_moment_row.setVisible(
                self._data_format.currentText() == "TEMcompany / TEM2Go")
        self._refresh_backend_state()

    def _on_data_format_changed(self, selected: str) -> None:
        temcompany = selected == "TEMcompany / TEM2Go"
        if temcompany:
            self._method.setCurrentText("TDEM")
        self._tem_moment_row.setVisible(
            temcompany and self._method.currentText() == "TDEM")

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
            "source_radius": self._src_radius.value(),
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
                geom["loop_area"] = float(
                    system.get("loop_area", np.pi * self._src_radius.value() ** 2)
                )
                geom["receiver_type"] = str(system.get("receiver_type", "dbdt"))
                geom["response_sign"] = float(system.get("response_sign", -1.0))
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
            "rel_error": self._rel_err.value(),
            "max_iterations": self._max_iter.value(),
            "data_scale": self._data_scale.value(),
            "starting_resistivity": float(
                (self._data or {}).get("inversion_defaults", {}).get(
                    "starting_resistivity", 100.0
                )
            ),
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
        if self._data_format.currentText() == "TEMcompany / TEM2Go":
            selected = select_directory(self, "Load EM project folder", Path.cwd())
            path = str(selected) if selected else ""
        else:
            path, _ = QFileDialog.getOpenFileName(self, "Load EM data", "", _FILE_FILTER)
        if not path:
            return
        if em_pipeline.is_temcompany_source(path):
            self._method.setCurrentText("TDEM")
            self._data_format.setCurrentText("TEMcompany / TEM2Go")
        self._example_id = None
        self._example_note.setVisible(False)
        self._source_path = Path(path)
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
            "TEMcompany / TEM2Go"
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
        try:
            self._data = em_pipeline.load_sounding(
                str(self._source_path), self._method.currentText(), sounding=int(index),
                moment=self._tem_moment.currentText(),
                use_flags=bool(self._use_flags.isChecked()))
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
        if heights is not None:
            self._geom_heights = np.asarray(heights, dtype=float).ravel()
        x = self._data.get("x")
        y = self._data.get("y")
        if x is not None and y is not None:
            self._geom_x = np.asarray(x, dtype=float).ravel()
            self._geom_y = np.asarray(y, dtype=float).ravel()
        system = dict(self._data.get("system", {}))
        if system:
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
        inversion = dict(self._data.get("inversion_defaults", {}))
        if inversion:
            self._n_layers.setValue(int(inversion.get("n_layers", self._n_layers.value())))
            self._min_thick.setValue(float(
                inversion.get("min_thickness", self._min_thick.value())))
            self._max_thick.setValue(float(
                inversion.get("max_thickness", self._max_thick.value())))
            self._smooth.setValue(float(
                inversion.get("smoothness", self._smooth.value())))
            self._lateral_smooth.setValue(float(
                inversion.get("lateral_smoothness", self._lateral_smooth.value())))
            self._project_layer_thicknesses = np.asarray(
                inversion.get("layer_thicknesses", []), dtype=float).ravel()
            # Track the station count exactly. Only raising it would leave the
            # cap behind when a reload brings more stations in, and inverting 71
            # of 94 without saying so is the kind of quiet truncation that is
            # very hard to notice on the section.
            self._line_max.setValue(int(self._data.get("n_soundings", 1)))
        n = int(self._data.get("n_soundings", 1))
        span = (float(self._geom_positions[-1] - self._geom_positions[0])
                if self._geom_positions is not None and self._geom_positions.size else 0.0)
        crs = str(self._data.get("coordinate_system", "embedded coordinates"))
        self._geom_info.setText(
            f"Geometry (TEMcompany): {n} soundings, {span:.1f} m path, {crs}.")
        self.log(
            "Applied TEMcompany geometry and system settings "
            f"(loop radius {self._src_radius.value():.3f} m, "
            f"Tx-Rx {self._tx_rx.value():.2f} m).",
            "info",
        )

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
        output_dir = io_utils.ensure_dir(Path(self.state.output_dir or ".") / "em_results")
        spec = WorkflowSpec(
            workflow_id="em.inversion",
            inputs={
                "data": ArtifactRef.from_path(
                    self._source_path,
                    artifact_id="em-sounding",
                    kind="em_sounding",
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
            spec, output_dir, stem="em_inversion"
        )
        self._reproduce.set_bundle(recipe_path, script_path)
        self._workflow_recipe_path = str(recipe_path)
        self._inv_worker = WorkflowWorker(
            spec,
            RunContext(project_root=output_dir, output_dir=output_dir),
        )
        self._inv_worker.logged.connect(lambda m: self.log(m, "info"))
        self._inv_worker.succeeded.connect(self._on_workflow_ok)
        self._inv_worker.failed.connect(lambda message: self._on_inversion_failed(message, False))
        self._inv_worker.finished.connect(self._reset_inv_button)
        self.register_worker(self._inv_worker)
        self._inv_worker.start()

    def _on_workflow_ok(self, result: WorkflowRunResult) -> None:
        if hasattr(self.state, "update_workflow_result"):
            self.state.update_workflow_result(
                self.module_key,
                "em.inversion",
                result.to_dict(),
                recipe_path=self._workflow_recipe_path,
            )
        self._on_inversion_ok(result.legacy_payload())

    def _start_line(self, method: str) -> None:
        out_dir = str(io_utils.ensure_dir(Path(self.state.output_dir or ".") / "em_results"))
        self.log(f"Starting {method} line inversion (up to {self._line_max.value()} soundings)…", "info")
        worker = TaskWorker(
            em_pipeline.invert_line, str(self._source_path), method,
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

    def _reset_inv_button(self) -> None:
        if self._inv_busy is not None:
            self._inv_busy.finish()
            self._inv_busy = None
        self._inv_btn.setText("Run inversion")
        self._inv_progress.setVisible(False)
        self._refresh_backend_state()

    def _on_inversion_ok(self, result: dict) -> None:
        self._last_result = result
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
        if "data_scale" in result:  # show the value the worker actually used (calibrated)
            self._data_scale.setValue(float(result["data_scale"]))
        self._section_view.show_model(result["edges"], result["model3d"],
                                      label=result["label"], cmap=result["cmap"],
                                      log_scale=result.get("log_scale", True))
        self._populate_plan(result)
        self._populate_overview(result)
        self._view_row.setVisible(True)
        self._view_mode.blockSignals(True); self._view_mode.setCurrentIndex(0)
        self._view_mode.blockSignals(False)
        # The map + section overview is what a reader needs first, so open there.
        self._model_stack.setCurrentWidget(self._overview_view)
        self._tabs.setCurrentWidget(self._model_tab)
        rng = result.get("model_range", [float("nan"), float("nan")])
        chi2 = result.get("chi2")
        report = result.get("lci_report") or {}
        coupled = report.get("mode") == "simultaneous"
        chi_txt = ""
        if isinstance(chi2, float) and chi2 == chi2:
            chi_txt = f", chi2={chi2:.2f}" if coupled else f", mean chi2={chi2:.2f}"
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
                            f"(σ ≥ {float(result.get('doi_threshold', 20)):g})")
        self._quality_view.show_quality(
            {"chi2": float(chi2) if isinstance(chi2, float) else float("nan"),
             "n_data": result.get("n_data"),
             "iterations": report.get("iterations") if coupled else None,
             "method": method_txt,
             "extra": extra,
             "convergence_track": report.get("convergence_track"),
             "note": (
                 "Whole-line weighted residual from one coupled solve; every "
                 "sounding was fitted with its neighbours' models constrained "
                 "at the same time."
                 if coupled else
                 "Mean weighted residual over the line; adjacent models on each "
                 "survey line are coupled by the selected lateral smoothness."
                 if report else
                 "Mean weighted residual over independently inverted soundings."
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
                            "mean_chi2": chi2, "section_npz": saved[0] if saved else None})

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
        out = io_utils.ensure_dir(Path(self.state.output_dir or ".") / "em_results")
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
            res_surface[sensitivity < float(result.get("doi_threshold", 20.0))] = np.nan
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
        self.log(f"Inversion {'unavailable' if backend else 'failed'}: {message}",
                 "warn" if backend else "error")

    def _on_line_failed(self, message: str) -> None:
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
            out = io_utils.ensure_dir(Path(self.state.output_dir or ".") / "em_results")
            p = out / f"{result['method'].lower()}_inversion.png"
            fig.savefig(p, dpi=160, bbox_inches="tight"); plt.close(fig)
            return str(p)
        except Exception as exc:  # noqa: BLE001
            self.log(f"Could not render inversion: {exc}", "warn")
            return None

    def _export_inversion(self) -> None:
        if not self._last_result:
            self.log("Run a single-sounding inversion first (a line auto-saves its section).", "warn")
            return
        folder = select_directory(
            self, "Export recovered model to folder",
            self.state.output_dir or Path.cwd(),
        )
        if not folder:
            return
        paths = em_pipeline.save_inversion(self._last_result, folder)
        self.log(f"Exported recovered model ({len(paths)} files) to {folder}", "success")

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
                 "desc": ("Set parameters. Geometry: source_radius, tx_rx_sep, height, orientation "
                          "(z/x/y), component (secondary/total/both), waveform. Inversion: n_layers, "
                          "min_thickness, max_thickness, smoothness, rel_error, data_scale "
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
                          "min_data_fraction (0-1 floor on the gates kept).")},
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
        if em_pipeline.is_temcompany_source(str(p)):
            self._method.setCurrentText("TDEM")
            self._data_format.setCurrentText("TEMcompany / TEM2Go")
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
            "tx_rx_sep": lambda v: self._tx_rx.setValue(float(v)),
            "height": lambda v: self._height.setValue(float(v)),
            "orientation": lambda v: set_combo(self._orient, v),
            "component": lambda v: set_combo(self._component, v),
            "waveform": lambda v: set_combo(self._waveform, v),
            "tem_moment": lambda v: set_combo(self._tem_moment, str(v).upper()),
            "use_project_flags": lambda v: self._use_flags.setChecked(bool(v)),
            "n_layers": lambda v: self._n_layers.setValue(int(v)),
            "min_thickness": lambda v: self._min_thick.setValue(float(v)),
            "max_thickness": lambda v: self._max_thick.setValue(float(v)),
            "smoothness": lambda v: self._smooth.setValue(float(v)),
            "lateral_smoothness": lambda v: self._lateral_smooth.setValue(float(v)),
            "lci_mode": lambda v: self._set_lci_mode(str(v)),
            "lci_passes": lambda v: self._lci_passes.setValue(int(v)),
            "auto_lambda": lambda v: self._auto_lam.setChecked(bool(v)),
            "target_chi2": lambda v: self._target_chi2.setValue(float(v)),
            "chi2_tolerance": lambda v: self._chi2_tol.setValue(float(v)),
            "max_lambda_trials": lambda v: self._lam_trials.setValue(int(v)),
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
