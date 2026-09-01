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
from typing import Any, Dict, List, Optional

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
from PyHydroGeophysX.qt_apps.widgets.em_gate_view import EMGateView
from PyHydroGeophysX.qt_apps.widgets.em_survey_view import (
    EMMetadataView,
    EMSignalNoiseView,
    EMSurveyView,
)
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


def _plain(value: Any) -> Any:
    """Whatever the reader returned, as something that survives being sent.

    The reader hands back numpy arrays and numpy scalars, which no JSON encoder
    takes. Non-finite floats become None rather than the bare NaN some encoders
    emit and no parser accepts.
    """
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_plain(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return float(value) if np.isfinite(value) else None
    return str(value)


def _modelled_gates(result: Dict[str, Any]) -> Optional[Dict[str, Dict[str, Any]]]:
    """A single-sounding result as the gate view wants it: moment -> times, pred.

    A joint run already separates the moments. A single-moment run does not, so
    the moment is taken from the result and defaults to the one a TEMcompany
    project is read as when it holds only one. A frequency-domain result has no
    gates and gets no overlay.

    ``fit_mask`` travels with the response. Outlier rejection removes gates the
    converged model cannot explain, after the loader's own selection has run, so
    without it the view would draw a gate as used by a fit that discarded it.
    """
    if not result or "times" not in result and not result.get("moments"):
        return None
    if result.get("joint_moments") and result.get("moments"):
        return {
            str(name): {"times": item["times"], "pred": item["pred"],
                        "fit_mask": item.get("fit_mask")}
            for name, item in result["moments"].items()
            if item and "times" in item and "pred" in item
        }
    if "times" in result and "pred" in result:
        name = str(result.get("tem_moment") or result.get("moment") or "HM")
        return {name: {"times": result["times"], "pred": result["pred"],
                       "fit_mask": result.get("fit_mask")}}
    return None


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
        # Two views of the data as it stands, before anything is inverted: where
        # the stations are and what the gate selection keeps of each, and the
        # acquisition description the forward will model.
        self._survey_view = EMSurveyView()
        self._survey_view.stationPicked.connect(self._on_station_picked)
        self._metadata_view = EMMetadataView()
        # Whether a thinning line is resistive ground or a risen noise floor.
        # The stored relative error is one divided by the other, so it cannot
        # separate them; these are the two drawn apart.
        self._signal_view = EMSignalNoiseView()
        # And one view of a single station in full: every gate the file holds,
        # not only the ones that survive, so a thin sounding explains itself.
        self._gate_view = EMGateView()
        # One tab, two views. A project records the gates a station dropped and
        # gets the gate view; a text sounding or a frequency sweep does not, and
        # gets the curve viewer, which is the only view its data supports.
        self._sounding_stack = QStackedWidget()
        self._sounding_stack.addWidget(self._gate_view)   # page 0: a project
        self._sounding_stack.addWidget(self._curve)       # page 1: everything else
        # Nothing is loaded yet, and the gate view has nothing to draw until a
        # project is; the curve viewer at least shows its axes.
        self._sounding_stack.setCurrentIndex(1)
        self._tabs.addTab(self._sounding_stack, "Sounding")
        self._tabs.addTab(self._survey_view, "Survey and QC")
        self._tabs.addTab(self._signal_view, "Signal and noise")
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
        # Adding the first combo entry selects it without emitting a change, so
        # the opening preset has to be written into the controls explicitly.
        self._apply_preset(str(self._preset.currentData()))
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
        # Keep noisy, project-enabled gates; their errors control their influence.
        # Users can still explicitly opt into an import-time hard cut.
        self._tail_cut = self._dspin(0.0, 0.0, 5.0, 0.05, 2)
        self._tail_cut.setSpecialValueText("off")
        self._tail_cut.setToolTip(
            "Condemns a gate whose relative stack error exceeds this value, and "
            "(unless sign reversals are kept below) one whose value is negative.\n\n"
            "The stack error is the scatter of the repeat transients divided by "
            "their mean, so a gate near or above 0.3 disagrees with itself by as "
            "much as the quantity being measured.\n\n"
            "0 turns the cut off entirely.")
        self._tail_cut.valueChanged.connect(self._on_use_flags_changed)
        moment_form.addRow("Tail cut (σ)", self._tail_cut)
        self._gate_rejection = QComboBox()
        for label, key in (("Truncate the tail", "truncate"),
                           ("Reject gates individually", "individual")):
            self._gate_rejection.addItem(label, key)
        self._gate_rejection.setToolTip(
            "What a failed stack-error test above removes. It governs that test "
            "only: a sign reversal always costs its own gate and no other.\n\n"
            "Truncate the tail: the first failing gate ends the sounding and every "
            "later gate goes with it. Safe, because past the first bad gate the "
            "decay has usually fallen into the noise.\n\n"
            "Reject gates individually: only the failing gates are dropped and the "
            "later ones are kept. Diffusion depth grows with time, so the latest "
            "usable gate sets how deep the sounding can see; this keeps it when a "
            "neighbour is spoiled by a local interference spike, at the risk of "
            "keeping a gate that looks clean by chance.")
        self._gate_rejection.currentIndexChanged.connect(self._on_use_flags_changed)
        moment_form.addRow("Cut removes", self._gate_rejection)
        self._keep_negative = QCheckBox("Keep sign reversals")
        self._keep_negative.setChecked(True)
        self._keep_negative.setToolTip(
            "On (the default): a negative gate is judged on its stack error "
            "alone. This is what the TEMcompany inversion does. Measured over "
            "1,503 station-moment datasets of one project, its gate selection "
            "is exactly the stored in-use flags, and it keeps a non-positive "
            "gate in 87 low-moment and 251 high-moment datasets.\n\n"
            "Off: a negative gate is condemned by its sign.\n\n"
            "An offset-loop system genuinely reverses sign at early time, while "
            "the diffusing current system is still inside the transmitter-receiver "
            "offset. Whether that reversal reaches the gates being inverted has a "
            "number attached: the crossing sits near an induction number of one, "
            "so a gate at time t sees it only once the ground is more conductive "
            "than mu0*r^2/(4t). On a 15 m offset that is about 6 ohm-m at 12 us "
            "and 1 ohm-m at 61 us.\n\n"
            "Turn this on where that expression says the reversal is real. Where "
            "it says the crossing falls far earlier than the first gate, a "
            "negative gate is something a layered earth cannot produce, and "
            "keeping it lets the fit trade the rest of the sounding against a "
            "value it can never reach.")
        self._keep_negative.toggled.connect(self._on_use_flags_changed)
        moment_form.addRow("", self._keep_negative)
        self._min_hm_gates = QSpinBox()
        self._min_hm_gates.setRange(0, 20)
        self._min_hm_gates.setValue(0)
        self._min_hm_gates.setSpecialValueText("off")
        self._min_hm_gates.setToolTip(
            "Drops the deep (HM) moment from a station that survives gate "
            "selection with fewer than this many gates. LM is never dropped.\n\n"
            "A moment reduced to one or two gates still costs a forward call and "
            "still enters the misfit, and two points cannot separate the level of "
            "a decay from its slope, so what it adds is a degree of freedom the "
            "fit absorbs rather than a constraint on the model.\n\n"
            "Against that, the deep moment is the only thing that sees deep, so a "
            "station stripped of its HM keeps its shallow model and loses its "
            "depth. Which argument wins depends on the survey, so this is off by "
            "default. On one 929-station survey a floor of 3 removed HM from 91 "
            "stations, 163 gates, and left 7 with no data at all.")
        self._min_hm_gates.valueChanged.connect(self._on_use_flags_changed)
        moment_form.addRow("Min HM gates", self._min_hm_gates)
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
        self._tx_rx.setToolTip(
            "Distance from the transmitter loop to the receiver coil.\n\n"
            "A walking ground survey records its own for every station, and "
            "those are used in preference to this while the box below is "
            "ticked. One line of 882 stations held 322 distinct values between "
            "11.58 and 17.63 m against a nominal 15, so a single number here "
            "cannot describe the survey; it is the value for a format that "
            "records none, and the override when the box is cleared.")
        # The loop and the coil have their own heights and the forward reads
        # them separately, so one field for both could only set them equal.
        self._tx_height = self._dspin(f["height"], 0.0, 500.0, 0.05, 2)
        self._rx_height = self._dspin(f["height"], 0.0, 500.0, 0.05, 2)
        for widget, which in ((self._tx_height, "transmitter loop"),
                              (self._rx_height, "receiver coil")):
            widget.setToolTip(
                f"Height of the {which} above the ground. Every TEMcompany "
                "project seen so far records the two equal, but the forward "
                "places them separately, so a frame that does not can say so.")
        self._height_row = merged_row(self._tx_height, "Rx", self._rx_height)
        self._per_station_geometry = QCheckBox(
            "Use each station's own recorded geometry")
        self._per_station_geometry.setChecked(True)
        self._per_station_geometry.setToolTip(
            "A project records the separation and the two heights per station, "
            "and with this ticked those are what the run uses; the fields above "
            "then show the loaded station rather than the survey, and editing "
            "them changes nothing.\n\n"
            "Clear it to impose the values above on every station. That is the "
            "honest way to test a suspect geometry column, and it is what a "
            "format recording no per-station geometry does anyway.\n\n"
            "The separation is rounded to a quarter of a metre so that stations "
            "sharing a geometry share one warmed-up forward operator, which is "
            "well inside what the response can resolve.")
        self._orient = QComboBox(); self._orient.addItems(["z", "x", "y"])
        self._component = QComboBox(); self._component.addItems(["secondary", "total", "both"])
        self._waveform = QComboBox()
        self._src_radius_label = QLabel("Source radius (m)")
        form.addRow(self._src_radius_label, self._src_radius)
        self._loop_area_label = QLabel("Tx loop area (m²)")
        form.addRow(self._loop_area_label, self._loop_area)
        self._tx_rx_label = QLabel("Tx-Rx sep (m)")
        form.addRow(self._tx_rx_label, self._tx_rx)
        self._height_label = QLabel("Height Tx / Rx (m)")
        form.addRow(self._height_label, self._height_row)
        form.addRow("", self._per_station_geometry)
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
        self._preset = QComboBox()
        self._preset.addItem("Ground TEM (TEM2Go, tTEM)", "ground_tem")
        self._preset.addItem("Generic / airborne", "generic")
        self._preset.setToolTip(
            "A starting point for the settings below, which stay editable.\n\n"
            "Ground TEM carries what a walking ground survey was found to want: "
            "twenty layers over a hundred metres, fixed regularisation rather "
            "than a chi-square chase, robust errors keeping all imported gates, and a resistivity "
            "ceiling low enough that an unresolved deep layer cannot rail to "
            "1e5 and be read as a measurement. It is applied automatically when "
            "a ground project is loaded.\n\n"
            "Generic is the framework default, which has to serve an airborne "
            "line that sees several hundred metres as well as a ground sounding "
            "that sees thirty, so it is deliberately unopinionated.")
        self._preset.currentIndexChanged.connect(self._on_preset_changed)
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
            "Iteration budget for Gauss-Newton and independent 1D solves. "
            "Simultaneous TRF uses its separate forward-evaluation budget in "
            "the Line group.")
        self._doi_threshold = self._dspin(
            float(DOI_SENSITIVITY_THRESHOLD), 0.1, 50.0, 0.5, 2)
        self._doi_threshold.setToolTip(
            "Cumulative sensitivity below which a cell is reported as "
            "unresolved.\n\n"
            "The quantity is summed from the base of the model upward, and the "
            "deepest layer is the thickest, so a low threshold saturates: on "
            "one ground survey a threshold of 0.8 put the reported depth of "
            "investigation at the very bottom of the model for more than half "
            "the stations, which is the metric running out rather than the data "
            "reaching that deep. Six to eight keeps the reported depth inside "
            "the model.")
        form.addRow("Preset", self._preset)
        form.addRow("Layers", self._n_layers)
        form.addRow("Thickness", self._thick_row)
        form.addRow("Initial model ρ (Ω·m)", self._start_res)
        self._lateral_smooth = self._dspin(
            float(d.get("lateral_smoothness", 0.0)), 0.0, 20.0, 0.1, 2)
        self._lateral_smooth.setToolTip(
            "The other half of the regularisation: how tightly neighbouring "
            "soundings on the same survey line are tied to each other, where "
            "Smoothness above ties the layers within one sounding. It applies "
            "to a line inversion only, never crosses a survey line, and 0 "
            "leaves the soundings independent whatever the coupling in the "
            "Line group says.\n\n"
            "The tie does not weaken with distance at the spacings a ground "
            "survey walks: the weight falls off only once a pair is further "
            "apart than the reference distance, which defaults to 50 m, and is "
            "exactly 1 for anything closer.\n\n"
            "A settings file may also carry lateral_weight_scale. That is a "
            "multiplier on this value rather than a second control, so the "
            "effective weight is the product of the two; it stays at 1.0 and "
            "this is the number to change.")
        form.addRow("Smoothness", self._smooth)
        form.addRow("Lateral smoothness", self._lateral_smooth)
        form.addRow("Max iterations", self._max_iter)
        form.addRow("DOI sensitivity", self._doi_threshold)
        return box

    #: Preset key to the control that carries it. Only settings the panel
    #: exposes appear; anything else in a preset reaches the run through
    #: :meth:`_collect_inv` reading the same defaults.
    def _preset_targets(self) -> Dict[str, Any]:
        return {
            "n_layers": self._n_layers, "min_thickness": self._min_thick,
            "max_thickness": self._max_thick, "smoothness": self._smooth,
            "lateral_smoothness": self._lateral_smooth,
            "max_iterations": self._max_iter, "rel_error": self._rel_err,
            "min_rel_error": self._err_floor, "rho_min": self._rho_min,
            "rho_max": self._rho_max, "doi_threshold": self._doi_threshold,
            "robust_threshold": self._robust_sigma,
            "robust_passes": self._robust_passes,
            "robust_max_error_factor": self._robust_max_factor,
            "robust_min_unchanged_fraction": self._robust_fixed_fraction,
            "robust_target_chi2": self._robust_target,
            "robust_target_tolerance": self._robust_target_tol,
            "shallow_prior_depth_m": self._prior_depth,
            "shallow_prior_min_resistivity": self._prior_rho,
            "shallow_prior_resistivity_factor": self._prior_factor,
            "shallow_prior_weight": self._prior_weight,
            "shallow_prior_window": self._prior_window,
            "shallow_prior_snr_ratio": self._prior_ratio,
            "shallow_prior_signal_threshold": self._prior_signal_limit,
            "lci_max_nfev": self._trf_nfev,
            "lci_ftol": self._trf_ftol,
        }

    def _apply_preset(self, name: str) -> None:
        """Write a preset into the controls, leaving the panel editable.

        Signals are blocked while the values land so that a preset is one
        change rather than a dozen, each of which would otherwise re-read the
        file or redraw a plot on its way past.
        """
        try:
            settings = em_pipeline.preset_inversion(name)
        except (AttributeError, ValueError):
            return
        widgets = self._preset_targets()
        for key, widget in widgets.items():
            if key not in settings:
                continue
            widget.blockSignals(True)
            try:
                widget.setValue(type(widget.value())(settings[key]))
            finally:
                widget.blockSignals(False)
        for widget, value in (
            (self._robust, bool(settings.get("robust_errors", False))),
            (self._prior_enabled, bool(settings.get("shallow_prior_enabled", False))),
            (self._auto_lam, bool(settings.get("auto_lambda", True))),
        ):
            widget.blockSignals(True)
            widget.setChecked(value)
            widget.blockSignals(False)
        self._set_lci_solver(str(settings.get("lci_solver", "trf")),
                             block_signals=True)
        self._prior_mode.setCurrentIndex(self._prior_mode.findData(settings.get("shallow_prior_mode", "quality_trend")))
        low, high = settings.get("scale_bounds", (1e-4, 1e4))
        for widget, value in ((self._lam_min, low), (self._lam_max, high)):
            widget.blockSignals(True)
            widget.setValue(float(value))
            widget.blockSignals(False)
        # The initial-model control uses 0 for "auto"; a preset that asks for an
        # automatic start therefore writes 0 rather than its fallback value.
        self._project_start_res = float(settings.get("starting_resistivity", 100.0))
        self._start_res.blockSignals(True)
        self._start_res.setValue(
            0.0 if settings.get("auto_starting_model", True)
            else self._project_start_res)
        self._start_res.blockSignals(False)
        self._sync_robust()
        self._sync_prior()
        self._sync_lci_mode()

    def _on_station_picked(self, index: int) -> None:
        """Load the station the survey view was clicked on."""
        total = int(self._sounding.maximum())
        if not (0 <= index < total):
            return
        self._sounding.blockSignals(True)
        self._sounding.setValue(index + 1)
        self._sounding.blockSignals(False)
        self._load_sounding(index, reset_geometry=False)
        self._tabs.setCurrentWidget(self._sounding_stack)

    def _gate_qc(self) -> Dict[str, Any]:
        """The gate selection the controls currently describe.

        One reading, shared by the survey table and the gate view, so the two
        cannot disagree about what the run would keep.
        """
        cut = float(self._tail_cut.value())
        return {
            "use_flags": bool(self._use_flags.isChecked()),
            "max_relative_std": cut if cut > 0 else None,
            "gate_rejection": str(self._gate_rejection.currentData() or "truncate"),
            "reject_negative": not bool(self._keep_negative.isChecked()),
        }

    def _refresh_gate_view(self) -> None:
        """Show the loaded station gate by gate, where the format has gates.

        Only a project database records the gates a station dropped; a plain
        text sounding holds the survivors and nothing else, so there is nothing
        to report for one.
        """
        source = self._source_path
        if (source is None or self._data is None
                or not self._data.get("temcompany")):
            self._gate_view.set_report(None)
            self._sounding_stack.setCurrentIndex(1)
            return
        try:
            report = em_pipeline.gate_report(
                str(source), int(self._data.get("sounding", 0)),
                moment=str(self._tem_moment.currentText()), **self._gate_qc())
        except Exception as exc:  # noqa: BLE001 - a view must not stop a load
            self.log(f"Gate report unavailable: {exc}")
            report = None
        self._gate_view.set_report(report)
        # Page 0 only where there is something on it to see.
        self._sounding_stack.setCurrentIndex(0 if report else 1)

    def _refresh_survey_views(self) -> None:
        """Re-read the survey table and the acquisition description.

        The table is a whole-file pass, so it is only worth doing for a project
        that has one; a single-sounding text file has nothing to summarise. It
        is cheap where it applies, about a tenth of a second for 929 stations.
        """
        self._metadata_view.set_metadata(self._data)
        summary = None
        source = self._source_path
        if source is not None and self._data is not None and self._data.get("temcompany"):
            try:
                summary = em_pipeline.survey_summary(
                    str(source), moment=str(self._tem_moment.currentText()),
                    **self._gate_qc())
            except Exception as exc:  # noqa: BLE001 - a view must not stop a load
                self.log(f"Survey summary unavailable: {exc}")
                summary = None
        self._survey_view.set_summary(summary)
        self._signal_view.set_summary(summary)
        # A response modelled for the station before this one says nothing about
        # this one, so the overlay goes when the station changes.
        self._gate_view.set_model(None)
        self._refresh_gate_view()

    def _select_preset(self, name: str) -> None:
        """Move the preset selector, which applies it through the signal."""
        index = self._preset.findData(name)
        if index >= 0 and index != self._preset.currentIndex():
            self._preset.setCurrentIndex(index)

    def _on_preset_changed(self, _index: int) -> None:
        self._apply_preset(str(self._preset.currentData()))
        # The preset moves the two switches the assist rows hang off, so their
        # enabled state has to follow rather than wait for a user click.
        self._sync_lci_mode()

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
        # Bounds on the per-gate stack error the file supplies, applied before it
        # joins the value above in quadrature. Both off by default.
        self._err_floor = self._dspin(0.0, 0.0, 1.0, 0.01, 3)
        self._err_floor.setSpecialValueText("off")
        self._err_floor.setToolTip(
            "Smallest stack error a gate is allowed to claim. A gate that happens "
            "to stack quietly arrives at a few tenths of a percent, and on a "
            "station carrying four or five gates it then outweighs its neighbours "
            "by two orders of magnitude and drags the model onto itself.\n\n"
            "A stack error is itself estimated from a finite number of repeats, "
            "so it scatters; floor it at the repeatability the instrument can "
            "actually resolve, typically a few percent. 0 turns the floor off and "
            "uses the recorded value.")
        self._err_ceiling = self._dspin(0.0, 0.0, 2.0, 0.05, 3)
        self._err_ceiling.setSpecialValueText("off")
        self._err_ceiling.setToolTip(
            "Largest stack error used as an uncertainty. Capping a larger recorded "
            "error makes that noisy gate carry more weight than its file requests; "
            "it limits down-weighting and does not reject the gate.\n\n"
            "0 turns the cap off. To remove noisy gates, use the stack-error cut "
            "on the loader panel instead.")
        self._err_bounds_row = merged_row(
            self._err_floor, "to", self._err_ceiling)
        # Recovered resistivity is bounded too. The old fixed pair was 1 to 1e5,
        # which on sparse ground TDEM lets a deep layer with no sensitivity rail
        # into tens of thousands of ohm-m and dominate the colour scale.
        self._rho_min = self._dspin(1.0, 1e-3, 1e4, 1.0, 3)
        self._rho_min.setToolTip(
            "Lower bound on recovered resistivity, in Ω·m. The solver works in "
            "log10 resistivity and this is its box constraint.")
        self._rho_max = self._dspin(1e5, 10.0, 1e7, 1000.0, 0)
        self._rho_max.setToolTip(
            "Upper bound on recovered resistivity, in Ω·m. A layer the data cannot "
            "resolve is driven only by the regularisation, so it walks until this "
            "bound stops it. Set it to the highest value the target geology can "
            "plausibly reach; a section whose deepest cells sit exactly on it is "
            "reporting the bound rather than the ground.")
        self._rho_row = merged_row(self._rho_min, "to", self._rho_max)
        form.addRow("Relative error", self._rel_err)
        form.addRow("Stack error bounds", self._err_bounds_row)
        form.addRow("Resistivity bounds", self._rho_row)
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
        # Which survey lines to run. A survey whose lines differ in data quality
        # cannot be served by one set of settings, and the lateral constraint
        # never crosses a line anyway, so a line inverted on its own is tied
        # exactly as it would be inside a whole-survey run.
        self._line_pick = QComboBox()
        self._line_pick.addItem("All lines", None)
        self._line_pick.setToolTip(
            "Invert one survey line rather than the whole file. The lateral "
            "constraint never crosses a line, so a line run on its own is tied "
            "exactly as it would be in a full run; what changes is that it can "
            "carry its own gate selection and rejection settings, and that the "
            "other lines are not re-solved.\n\n"
            "Use it where one line is noisier than the rest: run the good lines "
            "at the usual settings, then that line with a lower retention "
            "floor, and export each section separately.")
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
        self._lci_solver = QComboBox()
        for label, key in (("TRF (formal, bound-aware)", "trf"),
                           ("Gauss-Newton (fast / legacy)", "gauss_newton")):
            self._lci_solver.addItem(label, key)
        self._set_lci_solver(str(d.get("lci_solver", "trf")),
                             block_signals=True)
        self._lci_solver.setToolTip(
            "TRF is the formal default: its trust-region step respects the "
            "resistivity bounds while solving. Gauss-Newton is retained for "
            "fast previews and exact reproduction of older project results.")
        self._lci_solver.currentIndexChanged.connect(self._sync_lci_mode)
        self._trf_nfev = self._ispin(int(d.get("lci_max_nfev", 90)), 1, 1000)
        self._trf_nfev.setToolTip(
            "Maximum forward evaluations in each fixed-error TRF stage. This "
            "is not the Gauss-Newton iteration count. Budget exhaustion is "
            "reported as incomplete and prevents error inflation.")
        self._trf_ftol = self._dspin(
            float(d.get("lci_ftol", 1e-4)), 1e-8, 1e-2, 1e-4, 8)
        self._trf_ftol.setToolTip(
            "TRF relative full-objective stopping tolerance. The 1e-4 default "
            "completed the full trailcreek robust run; 1e-6 exhausted 90 "
            "evaluations before robust reweighting.")
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
        form.addRow("Survey line", self._line_pick)
        form.addRow("Coupling", self._lci_mode_row)
        form.addRow("Solver", self._lci_solver)
        form.addRow("TRF max evaluations", self._trf_nfev)
        form.addRow("TRF ftol", self._trf_ftol)
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

        # How far the search may move the smoothness. The default span is four
        # decades either way, and on a station carrying four or five gates a large
        # relaxation buys a low χ² with a model that swings to match noise.
        self._lam_min = self._dspin(1e-4, 1e-6, 1.0, 0.1, 4)
        self._lam_min.setToolTip(
            "Smallest factor the search may apply to the smoothness. Values well "
            "below 1 let it relax the model a long way to chase the target χ².")
        self._lam_max = self._dspin(1e4, 1.0, 1e6, 10.0, 1)
        self._lam_max.setToolTip(
            "Largest factor the search may apply to the smoothness, used when the "
            "first solve fits better than the target and a stiffer model is wanted.")
        self._lam_range_row = merged_row(self._lam_min, "to", self._lam_max)
        form.addRow("λ scale range", self._lam_range_row)

        self._robust = QCheckBox("Robust errors (keep all gates)")
        self._robust.setToolTip(
            "Keep every imported gate. After an initial fit, Huber-style weights "
            "increase the effective error of large residuals without changing the "
            "recorded measurement error. A gate can regain weight on later passes. "
            "A large residual can also mean an inadequate model, not bad data. "
            "The quality report keeps original-error χ² separate from effective χ².")
        form.addRow(self._robust)
        self._robust_sigma = self._dspin(3.0, 0.5, 20.0, 0.5, 1)
        self._robust_sigma.setToolTip(
            "Start downweighting when |prediction - data| / original error exceeds "
            "this threshold. With the target off, effective error = original error × "
            "sqrt(|residual| / cut). With a target, bounded inflation is calibrated on "
            "eligible gates only. Below the cut, errors are unchanged.")
        self._robust_passes = self._ispin(3, 1, 10)
        self._robust_passes.setToolTip(
            "Maximum reweight-and-solve passes after the initial fit. Reuses the "
            "previous model and stops early if error factors are stable within 1%.")
        self._robust_row = merged_row(
            self._robust_sigma, "σ, passes", self._robust_passes)
        form.addRow("Downweight beyond", self._robust_row)
        self._robust_max_factor = self._dspin(10.0, 1.0, 100.0, 1.0, 1)
        self._robust_max_factor.setToolTip(
            "Maximum effective/original error ratio. 10 limits the inverse-variance "
            "weight to at least 1/100 of its original value; it is never zero. "
            "Set to 1 to disable inflation without changing other settings.")
        form.addRow("Max error factor", self._robust_max_factor)
        self._robust_fixed_fraction = self._dspin(0.70, 0.0, 1.0, 0.05, 2)
        self._robust_fixed_fraction.setToolTip(
            "Minimum fraction of gates whose effective error stays EXACTLY original "
            "on each pass. 0.70 protects at least 70%; only the worst remaining "
            "residuals above the sigma cut can be inflated. The protected identities "
            "may change. Applies to the whole simultaneous run, or each independent sounding.")
        form.addRow("Keep original errors (fraction)", self._robust_fixed_fraction)
        self._robust_target = self._dspin(1.75, 0.0, 100.0, 0.25, 2)
        self._robust_target.setSpecialValueText("off (Huber only)")
        self._robust_target.setToolTip(
            "Optional EFFECTIVE chi2 target, reached by bounded error calibration on "
            "eligible gates and refitting. This is NOT independent evidence of fit quality. "
            "Original-error chi2 is still reported. 0 uses Huber-only weights.")
        self._robust_target_tol = self._dspin(0.25, 0.0, 10.0, 0.05, 2)
        self._robust_target_row = merged_row(self._robust_target, "±", self._robust_target_tol)
        form.addRow("Effective χ² target", self._robust_target_row)
        self._robust.toggled.connect(self._sync_robust)
        self._sync_robust()
        # This is deliberately nested in a collapsed advanced group. It is an
        # empirical, niche prior rather than a routine inversion control, and a
        # weak LM response is not a shallow-depth observation.
        self._prior_advanced = QGroupBox(
            "Advanced: empirical resistive-background prior")
        self._prior_advanced.setCheckable(True)
        self._prior_advanced.setChecked(False)
        self._prior_inner = QWidget()
        prior_form = QFormLayout(self._prior_inner)
        prior_layout = QVBoxLayout(self._prior_advanced)
        prior_layout.setContentsMargins(8, 4, 8, 8)
        prior_layout.addWidget(self._prior_inner)
        self._prior_inner.setVisible(False)
        self._prior_advanced.toggled.connect(self._prior_inner.setVisible)
        form.addRow(self._prior_advanced)

        self._prior_enabled = QCheckBox("Enable empirical background tendency")
        self._prior_enabled.setToolTip(
            "A sustained weak absolute LM response can be consistent with a resistive "
            "background, but it cannot locate that resistivity in a shallow layer. "
            "This adds a weak one-sided tendency across the whole 1-D model. It is an "
            "empirical assumption, not independent geology, and is excluded from data "
            "chi2 and sensitivity/DOI evidence.")
        prior_form.addRow(self._prior_enabled)
        self._prior_mode = QComboBox()
        self._prior_mode.addItem("Weak absolute LM signal", "signal_threshold")
        self._prior_mode.addItem("Declining LM quality", "quality_trend")
        self._prior_mode.addItem("Manual (all selected stations)", "manual")
        self._prior_signal_limit = self._dspin(0., 0., 1., 1e-9, 14)
        self._prior_signal_limit.setSpecialValueText("Auto (instrument forward)")
        self._prior_signal_limit.setToolTip(
            "Raw project-response units, BEFORE inversion data_scale. 0 computes the "
            "LM response of a fixed 1000 Ω·m homogeneous reference half-space, "
            "using each station's actual waveform, filters, gate and geometry. A positive "
            "value overrides that limit; e.g. 3e-9 for trailcreek's ~8.95 µs raw LM gate. "
            "The fixed trigger reference is deliberately separate from the soft model "
            "target. 80% of a full rolling window must lie below the limit; missing "
            "gates do not count.")
        prior_form.addRow("Prior trigger", self._prior_mode)
        prior_form.addRow("LM signal threshold", self._prior_signal_limit)
        # Retained as hidden compatibility carriers for old saved configurations.
        # Depth is no longer used; zero explicit rho selects the factor below.
        self._prior_depth = self._dspin(0., 0., 100., 1., 1)
        self._prior_rho = self._dspin(0., 0., 1e6, 100., 1)
        self._prior_factor = self._dspin(2., 1., 20., .25, 2)
        self._prior_factor.setToolTip(
            "Automatic one-sided target divided by the effective starting half-space. "
            "2 is deliberately modest: it expresses a resistive tendency without "
            "forcing an order-of-magnitude jump. The target is capped at rho_max. "
            "When auto-start is active, this uses the half-space selected from the data.")
        self._prior_weight = self._dspin(1., 0., 100., .25, 2)
        self._prior_window = self._ispin(11, 3, 201)
        self._prior_ratio = self._dspin(.6, .01, .99, .05, 2)
        self._prior_weight.setToolTip("Residual multiplier; squared in the penalty. 0 disables its influence.")
        self._prior_ratio.setToolTip("Activate below this fraction of the first window's median early LM SNR.")
        prior_form.addRow("Target / initial model", self._prior_factor)
        prior_form.addRow("Prior strength", self._prior_weight)
        prior_form.addRow("Quality window (stations)", self._prior_window)
        prior_form.addRow("SNR / initial SNR trigger", self._prior_ratio)
        self._prior_enabled.toggled.connect(self._sync_prior)
        self._prior_mode.currentIndexChanged.connect(self._sync_prior)
        self._sync_prior()
        return box

    def _set_prior_mode(self, value: str) -> None:
        index = self._prior_mode.findData(value)
        if index < 0:
            raise ValueError("shallow_prior_mode must be signal_threshold, quality_trend or manual.")
        self._prior_mode.setCurrentIndex(index)

    def _sync_prior(self) -> None:
        enabled = self._prior_enabled.isChecked()
        mode = self._prior_mode.currentData()
        set_rows_enabled([self._prior_mode, self._prior_factor, self._prior_weight], enabled)
        self._prior_window.setEnabled(enabled and mode != "manual")
        self._prior_ratio.setEnabled(enabled and mode == "quality_trend")
        self._prior_signal_limit.setEnabled(enabled and mode == "signal_threshold")

    def _sync_robust(self) -> None:
        set_rows_enabled([self._robust_row, self._robust_max_factor,
                          self._robust_fixed_fraction, self._robust_target_row],
                         self._robust.isChecked())

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
        self._sync_gate_rejection()
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
            "gate_rejection": str(self._gate_rejection.currentData()),
            "reject_negative": not bool(self._keep_negative.isChecked()),
            "min_gates_per_moment": self._min_gates_per_moment(),
            "source_radius": (
                float(np.sqrt(self._loop_area.value() / np.pi))
                if self._method.currentText() == "TDEM"
                else self._src_radius.value()
            ),
            "loop_area": self._loop_area.value(),
            "ttem_gex_path": str(self._ttem_gex_path or ""),
            "ttem_tfi_path": str(self._ttem_tfi_path or ""),
            "tx_rx_sep": self._tx_rx.value(),
            # All three, or an override does nothing: the forward prefers the
            # two specific keys and falls back to the general one, so sending
            # only "height" leaves a caller's heights silently ignored while
            # looking as though they were applied.
            "height": self._rx_height.value(),
            "tx_height": self._tx_height.value(),
            "rx_height": self._rx_height.value(),
            "per_station_geometry": bool(self._per_station_geometry.isChecked()),
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
        trf = simultaneous and str(self._lci_solver.currentData()) == "trf"
        self._lci_solver.setEnabled(simultaneous)
        self._trf_nfev.setEnabled(trf)
        self._trf_ftol.setEnabled(trf)
        self._auto_lam.setEnabled(simultaneous)
        set_rows_enabled([self._chi2_row, self._lam_trials, self._lam_range_row],
                         simultaneous and self._auto_lam.isChecked())

    def _set_lci_mode(self, value: str) -> None:
        key = str(value).strip().lower()
        index = self._lci_mode.findData(key)
        if index < 0:
            raise ValueError(
                "lci_mode must be one of simultaneous, sequential, off; "
                f"got {value!r}.")
        self._lci_mode.setCurrentIndex(index)

    def _set_lci_solver(self, value: str, *, block_signals: bool = False) -> None:
        key = str(value).strip().lower()
        index = self._lci_solver.findData(key)
        if index < 0:
            raise ValueError(
                "lci_solver must be one of trf, gauss_newton; "
                f"got {value!r}.")
        previous = self._lci_solver.blockSignals(block_signals)
        try:
            self._lci_solver.setCurrentIndex(index)
        finally:
            self._lci_solver.blockSignals(previous)

    def _collect_inv(self) -> Dict[str, Any]:
        result = {
            "n_layers": self._n_layers.value(),
            "min_thickness": self._min_thick.value(),
            "max_thickness": self._max_thick.value(),
            "smoothness": self._smooth.value(),
            "lateral_smoothness": self._lateral_smooth.value(),
            "lci_mode": str(self._lci_mode.currentData()),
            "lci_passes": self._lci_passes.value(),
            "lci_solver": str(self._lci_solver.currentData()),
            "lci_max_nfev": int(self._trf_nfev.value()),
            "lci_ftol": float(self._trf_ftol.value()),
            "lci_xtol": float(em_pipeline.DEFAULT_INVERSION["lci_xtol"]),
            "lci_gtol": float(em_pipeline.DEFAULT_INVERSION["lci_gtol"]),
            "auto_lambda": bool(self._auto_lam.isChecked()),
            "target_chi2": float(self._target_chi2.value()),
            "chi2_tolerance": float(self._chi2_tol.value()),
            "max_lambda_trials": int(self._lam_trials.value()),
            "robust_errors": bool(self._robust.isChecked()),
            "robust_threshold": float(self._robust_sigma.value()),
            "robust_passes": int(self._robust_passes.value()),
            "robust_max_error_factor": float(self._robust_max_factor.value()),
            "robust_min_unchanged_fraction": float(self._robust_fixed_fraction.value()),
            "robust_target_chi2": float(self._robust_target.value()),
            "robust_target_tolerance": float(self._robust_target_tol.value()),
            "shallow_prior_enabled": bool(self._prior_enabled.isChecked()),
            "shallow_prior_mode": str(self._prior_mode.currentData()),
            "shallow_prior_signal_threshold": float(self._prior_signal_limit.value()),
            "shallow_prior_depth_m": float(self._prior_depth.value()),
            "shallow_prior_min_resistivity": float(self._prior_rho.value()),
            "shallow_prior_resistivity_factor": float(self._prior_factor.value()),
            "shallow_prior_weight": float(self._prior_weight.value()),
            "shallow_prior_window": int(self._prior_window.value()),
            "shallow_prior_snr_ratio": float(self._prior_ratio.value()),
            # Robust errors replaced deleting gates, so the panel no longer
            # offers the old path and states that it is off. Saying so beats
            # leaving the key out: a configuration loaded from elsewhere would
            # otherwise decide it, and nothing on screen would show that it had.
            "reject_outliers": False,
            # 0 lets the solver size its own thread pool from the machine.
            "parallel_workers": int(self._parallel_workers_spin.value()),
            "rel_error": self._rel_err.value(),
            # 0 in either field means "leave the recorded stack error alone".
            "min_rel_error": float(self._err_floor.value()),
            "max_rel_error": (float(self._err_ceiling.value())
                              if self._err_ceiling.value() > 0 else None),
            "rho_min": float(self._rho_min.value()),
            "rho_max": float(self._rho_max.value()),
            "doi_threshold": float(self._doi_threshold.value()),
            "scale_bounds": (float(self._lam_min.value()),
                             float(self._lam_max.value())),
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
            self._select_preset("ground_tem")
        elif em_pipeline.is_temcompany_source(path):
            self._method.setCurrentText("TDEM")
            self._data_format.setCurrentText(_TEM2GO_FORMAT)
            self._select_preset("ground_tem")
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

        The tail cut and the rejection mode share this handler because they act on
        the same step: the cut decides which gates fail, the mode decides what
        goes with them, and either can empty a station.
        """
        self._sync_gate_rejection()
        if (self._source_path is not None and self._data is not None
                and self._data.get("temcompany")):
            self._load_sounding(self._sounding.value() - 1, reset_geometry=True)

    def _sync_gate_rejection(self) -> None:
        """Grey the cut's options out while the tail cut is off.

        Guarded because the tail cut is built first and its ``valueChanged``
        can fire while the rest of the panel is still being assembled.
        """
        live = self._tail_cut.value() > 0
        if hasattr(self, "_gate_rejection"):
            self._gate_rejection.setEnabled(live)
        if hasattr(self, "_keep_negative"):
            self._keep_negative.setEnabled(live)

    def _min_gates_per_moment(self) -> Optional[Dict[str, int]]:
        """The per-moment gate floor, or ``None`` when the spin box is at "off"."""
        if not hasattr(self, "_min_hm_gates"):
            return None
        floor = int(self._min_hm_gates.value())
        # LM is left at 1 rather than mirrored: it is the shallow moment and a
        # single gate still pins the near surface, where one or two HM gates pin
        # nothing and still enter the fit.
        return {"LM": 1, "HM": floor} if floor > 0 else None

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
                gate_rejection=str(self._gate_rejection.currentData()),
                reject_negative=not bool(self._keep_negative.isChecked()),
                min_gates_per_moment=self._min_gates_per_moment(),
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
        self._refresh_survey_views()

    def _populate_line_pick(self) -> None:
        """Offer the survey's own line numbers, keeping the current choice.

        Rebuilt on every load because a different project has different lines.
        The previous selection is restored when the new survey still has that
        line, so re-reading the same project with another gate setting does not
        silently widen the run back to the whole survey.
        """
        previous = self._line_pick.currentData()
        numbers = np.asarray(
            (self._data or {}).get("line_numbers", []), dtype=int).ravel()
        self._line_pick.blockSignals(True)
        self._line_pick.clear()
        self._line_pick.addItem("All lines", None)
        for value in sorted(set(numbers.tolist())):
            count = int(np.count_nonzero(numbers == value))
            self._line_pick.addItem("Line %d  (%d soundings)" % (value, count),
                                    int(value))
        index = self._line_pick.findData(previous)
        self._line_pick.setCurrentIndex(max(0, index))
        self._line_pick.blockSignals(False)

    def _selected_lines(self) -> Optional[List[int]]:
        """The line filter for a run, or ``None`` for the whole survey."""
        value = self._line_pick.currentData()
        return None if value is None else [int(value)]

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
        # Recorded heights are already applied separately by _station_geometry.
        # ``heights=`` is an explicit COMMON Tx/Rx override (e.g. imported CSV),
        # not a place to put the project's receiver-only height vector.
        # Also clear a stale override when a new project is loaded.
        self._geom_heights = None
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
            fallback = float(system.get("height", self._rx_height.value()))
            self._tx_height.setValue(float(system.get("tx_height", fallback)))
            self._rx_height.setValue(float(system.get("rx_height", fallback)))
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
        self._populate_line_pick()
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
        self._tabs.setCurrentWidget(self._sounding_stack)

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
            lines=self._selected_lines(),
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
                gate_rejection=str(geom.get("gate_rejection", "truncate")),
                reject_negative=bool(geom.get("reject_negative", False)),
                min_gates_per_moment=geom.get("min_gates_per_moment"),
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
        self._gate_view.set_model(_modelled_gates(result))
        self._last_section = None   # the export button now writes this profile
        self._inv_export.setEnabled(True)
        png = self._render_inversion(result)
        if png:
            self._inv_view.set_image_file(png)
            self._view_row.setVisible(False)
            self._model_stack.setCurrentWidget(self._inv_view)
            self._tabs.setCurrentWidget(self._model_tab)
        robust = result.get("robust") or {}
        extra = {"layers": result.get("n_layers"), "forward evals": result.get("nfev"),
                 "iterations": len(result.get("convergence") or [])}
        if robust.get("enabled"):
            extra["downweighted gates"] = f"{robust['downweighted']}/{robust['kept']} (none removed)"
            if "unchanged_fraction" in robust:
                extra["errors unchanged"] = f"{robust['unchanged_fraction']:.1%}"
        self._quality_view.show_quality(
            {"chi2": float(robust["chi2_effective"] if robust.get("enabled") else result["chi2"]),
             "robust": robust, "n_data": result.get("n_data"),
             "method": (
                 f"{result['method']} joint LM+HM 1D"
                 if result.get("joint_moments")
                 else f"{result['method']} 1D Occam"
             ),
             "extra": extra,
             "note": (
                 "Solid: weighted χ². Dashed: original-error χ² at recorded solve endpoints. "
                 "Effective errors are fitting weights; their calibration is not improved raw fit."
                 if robust.get("enabled") else
                 "1D least-squares inversion (χ² is the mean weighted squared residual).")},
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
        robust = result.get("robust") or {}
        quality_chi2 = robust["chi2_effective"] if robust.get("enabled") else chi2
        quality_items = (result.get("chi2_effective_list", []) if robust.get("enabled")
                         else result.get("chi2_list", []))
        residual_median = result.get("data_residual_sounding_median")
        if robust.get("enabled"):
            finite_items = np.asarray(quality_items, float)
            finite_items = finite_items[np.isfinite(finite_items)]
            sounding_mean = float(np.mean(finite_items)) if finite_items.size else None
            sounding_median = float(np.median(finite_items)) if finite_items.size else None
            residual_median = float(np.median(np.sqrt(finite_items))) if finite_items.size else None
        extra = {"soundings": result.get("n_soundings"), "layers": result.get("n_layers")}
        show_median = bool(report.get("chi2_median_history")) and sounding_median is not None
        if show_median:
            extra["weighted global χ²" if robust.get("enabled") else "global χ²"] = f"{quality_chi2:.2f}"
            quality_chi2 = sounding_median
        if (isinstance(sounding_mean, float) and sounding_mean == sounding_mean
                and isinstance(sounding_median, float) and sounding_median == sounding_median):
            extra["sounding χ²"] = f"mean {sounding_mean:.2f}, median {sounding_median:.2f}"
        if isinstance(residual_median, float) and residual_median == residual_median:
            extra["normalized residual (√χ²)"] = f"median {residual_median:.2f}"
        if coupled:
            extra["stop"] = report.get("stop_reason", "")
        if robust.get("enabled"):
            extra["downweighted gates"] = f"{robust['downweighted']}/{robust['kept']} (none removed)"
            if "unchanged_fraction" in robust:
                extra["errors unchanged"] = f"{robust['unchanged_fraction']:.1%}"
            if robust.get("target_chi2", 0) > 0:
                extra["effective χ² target"] = (
                    f"{robust['target_chi2']:g} ± {robust.get('target_tolerance', 0):g} "
                    f"({'reached' if robust.get('target_reached') else 'not reached'})")
        prior = result.get("shallow_prior") or {}
        if prior.get("enabled"):
            extra["empirical resistive-background prior"] = (
                f"{prior.get('active_soundings', 0)}/{result.get('n_soundings')} stations, "
                f"whole model toward {prior.get('target_resistivity', 0):g} Ω·m "
                "(not a shallow-depth estimate or independent geology)")
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
            {"chi2": float(quality_chi2) if quality_chi2 is not None else float("nan"),
             "chi2_label": (("Weighted median χ²" if robust.get("enabled") else "Median χ²")
                            if show_median else None),
             "robust": robust,
             "n_data": result.get("n_data"),
             "iterations": robust.get("total_iterations", report.get("iterations")) if coupled else None,
             "method": method_txt,
             "extra": extra,
             "convergence_track": report.get("convergence_track"),
             "note": ("Median is an equal-sounding display statistic; optimization still uses all gates. "
                      if show_median else "") + (
                 "Solid: weighted χ². Dashed: original-error reference (solve endpoints / soundings). "
                 "DOI uses effective errors. Error calibration is not independent evidence of improved raw fit."
                 if robust.get("enabled") else
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
                "values": quality_items,
                "value_label": "Weighted χ²" if robust.get("enabled") else "χ²",
                "reference_values": result.get("chi2_list", []) if robust.get("enabled") else [],
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
                            "sounding_median_chi2": result.get("chi2_sounding_median"),
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
                "effective_chi2": result.get("chi2_effective"),
                "downweighted_gates": (result.get("robust") or {}).get("downweighted", 0),
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
                          "rel_error, min_rel_error (floor on the recorded stack "
                          "error; 0 is off), max_rel_error (cap that limits how "
                          "far noisy gates are down-weighted; null is off), "
                          "rho_min / rho_max (ohm-m bounds on the recovered "
                          "model), data_scale "
                          "(calibration multiplier), auto_scale (bool, rough), ref_resistivity "
                          "(ohm-m; calibrate the absolute level to a known value, the reliable "
                          "option), max_iterations. TEMcompany: tem_moment "
                          "(LM+HM/HM/LM), use_project_flags (bool; false also "
                          "imports the gates the project's own QC switched off), "
                          "gate_rejection (truncate/individual; what a failed "
                          "stack-error test removes, see Format help), "
                          "reject_negative (bool; false judges a negative gate on "
                          "its stack error alone, for a site conductive enough to "
                          "reverse sign within the gate range), "
                          "min_gates_per_moment (e.g. {\"LM\": 1, \"HM\": 3}; "
                          "drops a moment left with fewer gates than that). "
                          "Line: spacing, max_soundings, "
                          "lateral_smoothness, lci_mode "
                          "(simultaneous/sequential/off), lci_passes "
                          "(block-coordinate only). Fit assistance (simultaneous "
                          "only): auto_lambda, target_chi2, chi2_tolerance, "
                          "max_lambda_trials, scale_bounds ([low, high] on the "
                          "smoothness factor the search may apply), "
                          "reject_outliers (bool; drop gates "
                          "the model cannot explain and re-solve), "
                          "outlier_threshold (sigma), outlier_passes, "
                          "min_data_fraction (0-1 floor on the gates kept). "
                          "Robust fitting (all modes): robust_errors (bool; keep gates, "
                          "overrides hard rejection), robust_threshold (sigma), "
                          "robust_passes, robust_max_error_factor (>=1). "
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
                {"name": "get_acquisition",
                 "args": {"section": ["all", "instrument", "system", "protocol",
                                      "inversion_defaults"]},
                 "desc": ("Describe what the forward will actually model for the "
                          "loaded file: transmitter and receiver geometry, "
                          "waveform nodes, gate open/centre/close times, analog "
                          "filter cutoffs, coil areas and currents, the "
                          "acquisition protocol, and the gate selection that was "
                          "applied. Answers questions like what the Tx-Rx "
                          "separation is, which gate windows are modelled, or "
                          "whether DataFactor was applied.")},
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
            "get_acquisition": lambda: self._agent_acquisition(
                args.get("section", "all")),
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

    #: What ``get_acquisition`` can be asked for, and where each lives in what
    #: the reader returns.
    _ACQUISITION_SECTIONS = {
        "instrument": "forward_metadata",
        "system": "system",
        "protocol": "protocol",
        "inversion_defaults": "inversion_defaults",
    }

    def _agent_acquisition(self, section: Any = "all") -> Dict[str, Any]:
        """The acquisition description, as data rather than as a tree.

        Whole arrays rather than the summaries the tree showed: a reader
        skimming a column wants to know a waveform has twenty-two nodes, and
        something answering a question about it needs the nodes.
        """
        # The argument is checked before the data, so a misspelled section is
        # reported on the first ask rather than after a load that was never the
        # problem.
        wanted = str(section or "all").strip().lower()
        if wanted not in self._ACQUISITION_SECTIONS and wanted != "all":
            return {"status": "failed",
                    "error": f"Unknown section '{section}'.",
                    "valid": ["all"] + sorted(self._ACQUISITION_SECTIONS)}
        if self._data is None:
            return {"status": "failed",
                    "error": "No data is loaded, so nothing has been read yet."}
        names = (sorted(self._ACQUISITION_SECTIONS) if wanted == "all"
                 else [wanted])
        out: Dict[str, Any] = {"status": "ok", "source": str(self._source_path or "")}
        for name in names:
            payload = self._data.get(self._ACQUISITION_SECTIONS[name]) or {}
            out[name] = _plain(payload)
        if not any(out.get(name) for name in names):
            out["note"] = (
                "This format records no acquisition description; only a "
                "TEMcompany project carries one.")
        return out

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

        def set_combo_data(combo, value):
            """Select by the item's userData, for combos whose label reads as prose."""
            index = combo.findData(str(value).strip().lower())
            if index < 0:
                keys = [combo.itemData(i) for i in range(combo.count())]
                raise ValueError(f"must be one of {keys}")
            combo.setCurrentIndex(index)

        handlers = {
            "source_radius": lambda v: self._src_radius.setValue(float(v)),
            "loop_area": lambda v: self._loop_area.setValue(float(v)),
            "tx_rx_sep": lambda v: self._tx_rx.setValue(float(v)),
            "height": lambda v: (self._tx_height.setValue(float(v)),
                                 self._rx_height.setValue(float(v))),
            "tx_height": lambda v: self._tx_height.setValue(float(v)),
            "rx_height": lambda v: self._rx_height.setValue(float(v)),
            "per_station_geometry":
                lambda v: self._per_station_geometry.setChecked(bool(v)),
            "orientation": lambda v: set_combo(self._orient, v),
            "component": lambda v: set_combo(self._component, v),
            "waveform": lambda v: set_combo(self._waveform, v),
            "tem_moment": lambda v: set_combo(self._tem_moment, str(v).upper()),
            "use_project_flags": lambda v: self._use_flags.setChecked(bool(v)),
            "tail_max_relative_std": lambda v: self._tail_cut.setValue(
                0.0 if v is None else float(v)),
            "gate_rejection": lambda v: set_combo_data(self._gate_rejection, v),
            "reject_negative": lambda v: self._keep_negative.setChecked(not bool(v)),
            "min_gates_per_moment": lambda v: self._min_hm_gates.setValue(
                0 if not v else int(dict(v).get("HM", 0))),
            "n_layers": lambda v: self._n_layers.setValue(int(v)),
            "min_thickness": lambda v: self._min_thick.setValue(float(v)),
            "max_thickness": lambda v: self._max_thick.setValue(float(v)),
            "starting_resistivity": lambda v: self._start_res.setValue(float(v)),
            "smoothness": lambda v: self._smooth.setValue(float(v)),
            "lateral_smoothness": lambda v: self._lateral_smooth.setValue(float(v)),
            "lci_mode": lambda v: self._set_lci_mode(str(v)),
            "lci_solver": lambda v: self._set_lci_solver(str(v)),
            "lci_max_nfev": lambda v: self._trf_nfev.setValue(int(v)),
            "lci_ftol": lambda v: self._trf_ftol.setValue(float(v)),
            "lci_passes": lambda v: self._lci_passes.setValue(int(v)),
            "auto_lambda": lambda v: self._auto_lam.setChecked(bool(v)),
            "target_chi2": lambda v: self._target_chi2.setValue(float(v)),
            "chi2_tolerance": lambda v: self._chi2_tol.setValue(float(v)),
            "max_lambda_trials": lambda v: self._lam_trials.setValue(int(v)),
            "parallel_workers": lambda v: self._parallel_workers_spin.setValue(
                max(0, min(int(v), self._parallel_workers_spin.maximum()))),
            "auto_starting_model": lambda v: (self._start_res.setValue(0.0)
                                              if bool(v) else None),
            "robust_errors": lambda v: self._robust.setChecked(bool(v)),
            "robust_threshold": lambda v: self._robust_sigma.setValue(float(v)),
            "robust_passes": lambda v: self._robust_passes.setValue(int(v)),
            "robust_max_error_factor": lambda v: self._robust_max_factor.setValue(float(v)),
            "robust_min_unchanged_fraction": lambda v: self._robust_fixed_fraction.setValue(float(v)),
            "robust_target_chi2": lambda v: self._robust_target.setValue(float(v)),
            "robust_target_tolerance": lambda v: self._robust_target_tol.setValue(float(v)),
            "shallow_prior_enabled": lambda v: self._prior_enabled.setChecked(bool(v)),
            "shallow_prior_mode": lambda v: self._set_prior_mode(str(v)),
            "shallow_prior_signal_threshold": lambda v: self._prior_signal_limit.setValue(float(v)),
            "shallow_prior_depth_m": lambda v: self._prior_depth.setValue(float(v)),
            "shallow_prior_min_resistivity": lambda v: self._prior_rho.setValue(float(v)),
            "shallow_prior_resistivity_factor": lambda v: self._prior_factor.setValue(float(v)),
            "shallow_prior_weight": lambda v: self._prior_weight.setValue(float(v)),
            "shallow_prior_window": lambda v: self._prior_window.setValue(int(v)),
            "shallow_prior_snr_ratio": lambda v: self._prior_ratio.setValue(float(v)),
            "rel_error": lambda v: self._rel_err.setValue(float(v)),
            "min_rel_error": lambda v: self._err_floor.setValue(
                0.0 if v is None else float(v)),
            "max_rel_error": lambda v: self._err_ceiling.setValue(
                0.0 if v is None else float(v)),
            "rho_min": lambda v: self._rho_min.setValue(float(v)),
            "rho_max": lambda v: self._rho_max.setValue(float(v)),
            "scale_bounds": lambda v: (
                self._lam_min.setValue(float(tuple(v)[0])),
                self._lam_max.setValue(float(tuple(v)[1]))),
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
