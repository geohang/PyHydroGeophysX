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
import shutil
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
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
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
from PyHydroGeophysX.qt_apps.workers import (
    ProcessWorkflowWorker,
    TaskWorker,
)
from PyHydroGeophysX.data_processing.ert_io import save_edited_ert_container
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
        self._inv_worker: Optional[Any] = None
        self._inv_busy: Optional[BusyStateController] = None
        self._adtlert_probe_worker: Optional[TaskWorker] = None
        self._adtlert_probe_serial = 0
        self._adtlert_runtime_ready: Optional[bool] = None
        self._ert_recipe_path: str = ""
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
        self._load_worker: Optional[TaskWorker] = None
        self._tl_files: List[str] = []
        self._tl_labels: List[str] = []
        self._tl_times: List[float] = []
        self._tl_worker: Optional[Any] = None
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

        # Keep the readout and colour legend outside pyqtgraph's GraphicsLayout.
        # A changing LabelItem beside a hoverable scatter can feed its new size
        # back into the plot geometry: the point moves under the stationary mouse,
        # hover toggles again, and Windows Qt eventually dies with c00000fd (stack
        # overflow) rather than a Python traceback.  Fixed-height Qt widgets plus
        # click-to-read have no geometry/hover feedback path.
        self._pseudo_widget = QWidget()
        pseudo_layout = QVBoxLayout(self._pseudo_widget)
        pseudo_layout.setContentsMargins(0, 0, 0, 0)
        pseudo_layout.setSpacing(4)
        self._pseudo_plot_widget = pg.PlotWidget()
        self._pseudo_plot_widget.setBackground("w")
        self._pseudo_plot = self._pseudo_plot_widget.getPlotItem()
        self._pseudo_plot.showGrid(x=True, y=True, alpha=0.25)
        self._pseudo_plot.setLabel("bottom", "x (m)")
        self._pseudo_plot.setLabel("left", "pseudo-depth (m, positive down)")
        self._pseudo_plot.invertY(True)
        self._pseudo_scatter = pg.ScatterPlotItem(size=11)
        self._pseudo_plot.addItem(self._pseudo_scatter)
        self._pseudo_scatter.sigClicked.connect(self._on_pseudo_clicked)
        pseudo_layout.addWidget(self._pseudo_plot_widget, stretch=1)

        self._pseudo_legend = QWidget()
        legend_layout = QVBoxLayout(self._pseudo_legend)
        legend_layout.setContentsMargins(8, 0, 8, 0)
        legend_layout.setSpacing(1)
        legend_layout.addWidget(QLabel("Apparent resistivity (Ω·m)"))
        self._pseudo_scale_bar = QFrame()
        self._pseudo_scale_bar.setFixedHeight(12)
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
        legend_layout.addWidget(self._pseudo_scale_bar)
        scale_row = QHBoxLayout()
        scale_row.setContentsMargins(0, 0, 0, 0)
        scale_row.setSpacing(2)
        self._pseudo_scale_labels: List[QLabel] = []
        for index in range(5):
            label = QLabel("—")
            label.setAlignment(
                Qt.AlignLeft if index == 0 else Qt.AlignRight if index == 4 else Qt.AlignCenter
            )
            scale_row.addWidget(label, stretch=1)
            self._pseudo_scale_labels.append(label)
        legend_layout.addLayout(scale_row)
        pseudo_layout.addWidget(self._pseudo_legend)

        self._pseudo_readout = QLabel(
            "Click a measurement to read x, pseudo-depth, and apparent resistivity."
        )
        self._pseudo_readout.setContentsMargins(8, 0, 8, 0)
        self._pseudo_readout.setFixedHeight(24)
        self._pseudo_readout.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self._pseudo_readout.setTextInteractionFlags(Qt.TextSelectableByMouse)
        pseudo_layout.addWidget(self._pseudo_readout)

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
        self._qc_drop_neg.setToolTip(
            "A non-positive apparent resistivity is a polarity or geometry error, not "
            "a measurement. Off by default so the count does not change under you; "
            "Min ρa above already removes negatives once it is above zero.")
        mform.addRow(self._qc_drop_neg)

        self._qc_min_v = QDoubleSpinBox(); self._qc_min_v.setRange(0.0, 1e6)
        self._qc_min_v.setDecimals(3); self._qc_min_v.setValue(0.0)
        mform.addRow("Min |V|", self._qc_min_v)

        self._qc_min_i = QDoubleSpinBox(); self._qc_min_i.setRange(0.0, 1e6)
        self._qc_min_i.setDecimals(3); self._qc_min_i.setValue(0.0)
        mform.addRow("Min |I|", self._qc_min_i)

        self._qc_max_k = QDoubleSpinBox(); self._qc_max_k.setRange(0.0, 1e9)
        self._qc_max_k.setDecimals(0); self._qc_max_k.setValue(0.0)
        self._qc_max_k.setToolTip(
            "Drop configurations whose geometric factor exceeds this (0 = off). A large "
            "|k| multiplies the measured resistance, and its noise with it, which is how "
            "a clean-looking rhoa outlier is produced by geometry rather than by ground.")
        mform.addRow("Max |k|", self._qc_max_k)

        self._qc_max_recip = QDoubleSpinBox(); self._qc_max_recip.setRange(0.0, 100.0)
        self._qc_max_recip.setDecimals(2); self._qc_max_recip.setValue(0.0); self._qc_max_recip.setSuffix(" %")
        mform.addRow("Max recip. error", self._qc_max_recip)

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
            "Quick runs the inversion and nothing else, which is what you want while "
            "you are still choosing λ and a mesh.\n\n"
            "Full adds the two stages Quick drops. It validates the geometric factors "
            "against the mesh, repairing them when they disagree, and it searches λ "
            "for your target χ². Use it before you trust a section.\n\n"
            "The k check is the one you cannot postpone on evidence: a wrong k scales "
            "the whole section by a constant and χ² never notices, so a Quick result "
            "that looks perfect can still be uniformly wrong. Re-run in Full whenever "
            "the resistivities themselves look off, not only when the fit looks bad.")
        self._inv_mode.currentIndexChanged.connect(self._on_inv_mode_changed)
        iform.addRow("Mode", self._inv_mode)

        self._engine = QComboBox()
        for label, value in (("In-house Gauss-Newton", "pyhydro"),
                             ("PyGIMLi ERTManager", "pygimli"),
                             ("ADTLERT 2.5D (CUDA)", "adtlert")):
            self._engine.addItem(label, value)
        self._engine.setToolTip(
            "Solver. The in-house Gauss-Newton inversion exposes its own stopping rule "
            "and line search, so the fit assistance below can drive it directly; the "
            "PyGIMLi manager is kept as a cross-check.\n\n"
            "ADTLERT uses a CUDA-accelerated cuDSS forward solve and GPU CGLS. "
            "CUDA 12 plus cuDSS are required. Windows is supported; Linux is "
            "recommended for the best performance. The controls below remain "
            "under your control when this engine is selected.")
        self._engine.currentIndexChanged.connect(self._on_engine_changed)
        iform.addRow("Engine", self._engine)
        self._adtlert_status = QLabel()
        self._adtlert_status.setWordWrap(True)
        self._adtlert_status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._adtlert_status.setVisible(False)
        iform.addRow("", self._adtlert_status)

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

    def _on_engine_changed(self, _index: int = -1) -> None:
        """Probe the selected CUDA backend without changing user parameters."""
        if self._engine.currentData() != "adtlert":
            self._adtlert_probe_serial += 1
            self._adtlert_status.setVisible(False)
            return
        self._adtlert_probe_serial += 1
        serial = self._adtlert_probe_serial
        self._adtlert_runtime_ready = None
        self._adtlert_status.setText("Checking CUDA, cuDSS, and GPU CGLS…")
        self._adtlert_status.setStyleSheet("color:#5a6a7a; font-size:8pt;")
        self._adtlert_status.setVisible(True)
        worker = TaskWorker(self._probe_adtlert_runtime)
        worker.succeeded.connect(
            lambda result, token=serial: self._on_adtlert_probe_ok(token, result)
        )
        worker.failed.connect(
            lambda message, token=serial: self._on_adtlert_probe_failed(
                token, message
            )
        )
        self._adtlert_probe_worker = self.register_worker(worker)
        worker.start()

    @staticmethod
    def _probe_adtlert_runtime() -> Dict[str, Any]:
        """Return the actual ADTLERT CUDA path selected by this environment."""
        import cupy as cp
        import torch
        from nvmath.sparse.advanced import DirectSolver  # noqa: F401

        from PyHydroGeophysX.inversion.ert_inversion import (
            _adtlert_cudss_available,
            _adtlert_solver_name,
        )

        if not torch.cuda.is_available():
            raise RuntimeError("Torch cannot see a CUDA-capable GPU")
        if int(cp.cuda.runtime.getDeviceCount()) < 1:
            raise RuntimeError("CuPy cannot see a CUDA-capable GPU")
        if not _adtlert_cudss_available():
            raise RuntimeError("ADTLERT CUDA 12/cuDSS probe failed")
        device = cp.cuda.runtime.getDeviceProperties(0)["name"]
        if isinstance(device, bytes):
            device = device.decode(errors="replace")
        return {
            "device": str(device),
            "forward_solver": "cudss",
            "linearized_solver": _adtlert_solver_name("cgls", prefer_gpu=True),
        }

    def _on_adtlert_probe_ok(self, serial: int, result: Dict[str, Any]) -> None:
        if serial != self._adtlert_probe_serial:
            return
        if self._engine.currentData() != "adtlert":
            return
        self._adtlert_runtime_ready = True
        self._adtlert_status.setText(
            f"GPU ready: {result['device']} · cuDSS forward · GPU CGLS"
        )
        self._adtlert_status.setStyleSheet("color:#238636; font-size:8pt;")

    def _on_adtlert_probe_failed(self, serial: int, message: str) -> None:
        if serial != self._adtlert_probe_serial:
            return
        if self._engine.currentData() != "adtlert":
            return
        self._adtlert_runtime_ready = False
        self._adtlert_status.setText(
            "GPU unavailable: this selection will run the original PyHydro ERT "
            f"engine. {message}"
        )
        self._adtlert_status.setStyleSheet("color:#b42318; font-size:8pt;")

    # -- loading -------------------------------------------------------------
    def _start_load(self, path: str) -> None:
        """Load one ERT file (off the UI thread) into the electrode + pseudosection
        view and make it the current single-inversion dataset. Used when a file is
        added and when a row in the file list is clicked to preview it."""
        instrument = self._instrument.currentData()
        # Capture widget/state values on the UI thread; the parse runs off-thread.
        out_dir = self.state.ensure_results_store().scratch_dir(self.module_key)
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
        # Which extra criteria this file can support is a property of the file,
        # so it is settled here rather than being rechecked on every Apply.
        self._refresh_qc_availability()
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

    @staticmethod
    def _reciprocal_error(data) -> Optional[np.ndarray]:
        """Relative reciprocal error per measurement, NaN where there is no partner.

        A reciprocal swaps the current and potential pairs, (A,B,M,N) -> (M,N,A,B),
        and reciprocity requires the same transfer resistance from both. How far
        apart they land is the only error estimate that comes from the data rather
        than from an assumed percentage, which is why it is worth computing even
        though most files do not carry it as a column.

        Returns None when the file has neither a resistance nor the rhoa and k
        needed to rebuild one.
        """
        try:
            a, b, m, n = (np.asarray(data[t], dtype=int) for t in ("a", "b", "m", "n"))
        except Exception:  # noqa: BLE001 - a container without ABMN cannot be paired
            return None
        if data.haveData("r"):
            res = np.asarray(data["r"], dtype=float)
        elif data.haveData("rhoa") and data.haveData("k"):
            k = np.asarray(data["k"], dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                res = np.asarray(data["rhoa"], dtype=float) / np.where(np.abs(k) > 1e-12, k, np.nan)
        else:
            return None
        # Each pair is unordered, so canonicalise before matching or (A,B) and
        # (B,A) read as different configurations and nothing ever pairs up.
        cur = [tuple(sorted(p)) for p in zip(a, b)]
        pot = [tuple(sorted(p)) for p in zip(m, n)]
        first: Dict[Any, int] = {}
        for i, key in enumerate(zip(cur, pot)):
            first.setdefault(key, i)
        err = np.full(res.size, np.nan)
        for i, (c, p) in enumerate(zip(cur, pot)):
            j = first.get((p, c))
            if j is None or j == i:
                continue
            mean = 0.5 * (abs(res[i]) + abs(res[j]))
            if np.isfinite(mean) and mean > 1e-12:
                err[i] = abs(res[i] - res[j]) / mean
        return err

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
        if data is None:
            for w in (self._qc_min_v, self._qc_min_i, self._qc_max_k, self._qc_max_recip):
                gate(w, False, "Load ERT data first.")
            return

        # Units are whatever the file used, so the observed range is quoted rather
        # than a unit guessed from the magnitudes. 90 could be mA or A.
        for widget, token, name in ((self._qc_min_v, "u", "voltage"),
                                    (self._qc_min_i, "i", "current")):
            if data.haveData(token):
                v = np.abs(np.asarray(data[token], dtype=float))
                v = v[np.isfinite(v)]
                span = f"{v.min():.4g} to {v.max():.4g}" if v.size else "empty"
                gate(widget, True,
                     f"Drop readings whose |{token}| falls below this (0 = off). This file's "
                     f"{name} spans {span}, in the file's own units.")
            else:
                gate(widget, False,
                     f"This file carries no {name} column, so there is nothing to test.")

        if data.haveData("k"):
            k = np.abs(np.asarray(data["k"], dtype=float))
            k = k[np.isfinite(k)]
            span = f"{k.min():.4g} to {k.max():.4g}" if k.size else "empty"
            gate(self._qc_max_k, True,
                 "Drop configurations whose geometric factor exceeds this (0 = off). A large "
                 "|k| multiplies the measured resistance and its noise with it, which is how "
                 f"geometry alone produces a rhoa outlier. This file spans {span}.")
        else:
            gate(self._qc_max_k, False, "This file carries no geometric factors.")

        rec = self._reciprocal_error(data)
        paired = 0 if rec is None else int(np.isfinite(rec).sum())
        if paired:
            finite = rec[np.isfinite(rec)]
            gate(self._qc_max_recip, True,
                 f"Drop measurements whose normal and reciprocal disagree by more than this "
                 f"(0 = off). {paired} of {data.size()} measurements have a reciprocal here; "
                 f"their disagreement runs to {100.0 * finite.max():.1f}%, median "
                 f"{100.0 * float(np.median(finite)):.1f}%. Unpaired measurements are kept.")
        else:
            gate(self._qc_max_recip, False,
                 "This survey contains no reciprocal pairs, so there is nothing to compare. "
                 "Reciprocals have to be measured in the field; they cannot be recovered here.")

    def _apply_extra_filters(self, data, keep: np.ndarray) -> List[str]:
        """Apply the folded QC criteria to ``keep`` in place; report what each cost.

        A criterion whose field is missing is skipped rather than failing the
        whole filter, matching the row being disabled in the panel.
        """
        reasons: List[str] = []

        def cut(mask: np.ndarray, label: str) -> None:
            before = int(keep.sum())
            # In place on purpose: `keep &= mask` here would rebind the enclosing
            # name and raise UnboundLocalError instead of narrowing the caller's array.
            np.logical_and(keep, mask, out=keep)
            lost = before - int(keep.sum())
            if lost:
                reasons.append(f"{label} dropped {lost}")

        if self._qc_drop_neg.isChecked():
            cut(np.asarray(data["rhoa"], dtype=float) > 0.0, "ρa ≤ 0")
        for widget, token, label in ((self._qc_min_v, "u", "|V| floor"),
                                     (self._qc_min_i, "i", "|I| floor")):
            if widget.value() > 0 and data.haveData(token):
                cut(np.abs(np.asarray(data[token], dtype=float)) >= widget.value(), label)
        if self._qc_max_k.value() > 0 and data.haveData("k"):
            cut(np.abs(np.asarray(data["k"], dtype=float)) <= self._qc_max_k.value(), "|k| ceiling")
        if self._qc_max_recip.value() > 0:
            rec = self._reciprocal_error(data)
            if rec is not None:
                limit = self._qc_max_recip.value() / 100.0
                # An unpaired measurement has no reciprocal to disagree with, so it
                # is kept rather than judged against a test it cannot take.
                cut(~(np.isfinite(rec) & (rec > limit)), "reciprocal error")
        return reasons

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
            # Folding the section away also switches its criteria off, so what the
            # panel shows is what the filter did.
            if self._qc_more.isChecked():
                reasons = self._apply_extra_filters(data, keep)
                if reasons:
                    self.log("Extra QC: " + "; ".join(reasons), "info")
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
            and self._adtlert_runtime_ready is False
        ):
            self.log(
                "ADTLERT GPU is unavailable; this run will use the original "
                "PyHydro ERT engine.",
                "warn",
            )
        if str(self._inv_mode.currentData() or "quick") == "full":
            self._report_data_health()
        try:
            run = self.begin_persisted_run(
                "ert.single_inversion", "ert.single_inversion"
            )
        except Exception as exc:  # noqa: BLE001
            self.log(f"Could not prepare Project run: {exc}", "error")
            return
        out_path = run.outputs_dir
        input_path = run.inputs_dir / "filtered_ert_data.dat"
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
        recipe_path, script_path = export_workflow_bundle(spec, run.run_dir, stem="ert")
        self._reproduce.set_bundle(recipe_path, script_path)
        self._ert_recipe_path = str(recipe_path)
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
        self._inv_worker = ProcessWorkflowWorker(
            recipe_path,
            project_root,
            out_path,
            run.result_path,
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
            import pygimli as pygimli

            from PyHydroGeophysX.inversion.ert_inversion import ModelResult

            paths = {key: Path(str(value)) for key, value in dict(bundle).items()}
            mesh = pygimli.load(str(paths["mesh"]))
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
        metrics = dict(result.get("metrics") or {})
        engine = str(result.get("engine") or "")
        requested_engine = str(result.get("engine_requested") or engine)
        solver = str(metrics.get("linearized_solver") or "")
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
        if (
            self._engine.currentData() == "adtlert"
            and self._adtlert_runtime_ready is False
        ):
            self.log(
                "ADTLERT GPU is unavailable; this run will use the original "
                "PyHydro ERT engine.",
                "warn",
            )
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
            "relativeError": self._relerr.value(), "mesh_quality": self._quality.value(),
            "para_depth": self._para_depth.value(),
            "windowed": self._tl_windowed.isChecked(), "window_size": self._tl_window.value(),
            "engine": str(self._engine.currentData()),
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
        try:
            run = self.begin_persisted_run(
                "ert.timelapse_inversion", "ert.timelapse_inversion"
            )
        except Exception as exc:  # noqa: BLE001
            self.log(f"Could not prepare Project run: {exc}", "error")
            return
        # Drop this module's references to the previous run's arrays before the
        # new one writes. They are ordinary in-memory arrays now, but releasing
        # them keeps a stale model out of the viewer if the run fails.
        self._tl_models = None
        self._tl_coverage = None
        persisted_files = []
        try:
            for index, source in enumerate(self._tl_files):
                target = run.inputs_dir / f"step_{index:04d}{Path(source).suffix}"
                shutil.copy2(source, target)
                persisted_files.append(target)
        except Exception as exc:  # noqa: BLE001
            self.fail_persisted_run(str(exc), "ert.timelapse_inversion")
            self.log(f"Could not persist time-lapse inputs: {exc}", "error")
            return
        spec = WorkflowSpec(
            workflow_id="ert.timelapse_inversion",
            inputs={
                "data_files": [
                    ArtifactRef.from_path(
                        Path(path),
                        artifact_id=f"ert-timestep:{index}",
                        kind="ert_observations",
                        base_dir=run.run_dir,
                        metadata={
                            "sequence_index": index,
                            "measurement_time": (
                                float(times[index]) if times is not None else index
                            ),
                        },
                    )
                    for index, path in enumerate(persisted_files)
                ],
                "measurement_times": list(times or range(len(self._tl_files))),
            },
            parameters=params,
            metadata={"source": "qt", "sequence_order_persisted": True},
        )
        recipe_path, script_path = export_workflow_bundle(
            spec, run.run_dir, stem="ert_timelapse"
        )
        self._reproduce.set_bundle(recipe_path, script_path)
        self._tl_recipe_path = str(recipe_path)
        self._tl_busy = BusyStateController([self._tl_btn])
        self._tl_busy.start()
        self._tl_btn.setText("Inverting…")
        self._tl_progress.setVisible(True); self._tl_progress.setRange(0, 0)
        self.log(f"Starting {params['inversion_type']} time-lapse ERT inversion "
                 f"({len(self._tl_files)} steps)…", "info")
        # Both supported time-lapse engines execute long native/GPU kernels.
        # Keep every time-lapse run outside Qt's interpreter so PyGIMLi cannot
        # retain the GIL and ADTLERT cannot monopolize the GUI CUDA context.
        # This also covers an unavailable ADTLERT request that falls back to
        # the PyHydro engine inside the workflow process.
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
        self.register_worker(self._tl_worker)
        self._tl_worker.start()

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
        if payload.get("mesh") is None and payload.get("model_bundle"):
            payload.update(self._load_timelapse_bundle(payload["model_bundle"]))
        self._on_tl_ok(payload)

    def _load_timelapse_bundle(self, bundle: Dict[str, Any]) -> Dict[str, Any]:
        """Load process-safe time-lapse arrays for the interactive Qt viewer."""
        try:
            import pygimli as pygimli

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
                "mesh": pygimli.load(str(mesh_path)),
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
        engine = str(result.get("engine") or "pyhydro")
        requested_engine = str(result.get("engine_requested") or engine)
        solver = str(result.get("linearized_solver") or "")
        compute_note = "Joint χ² over all time steps."
        if engine == "adtlert":
            compute_note += f" cuDSS forward · {solver or 'gpu_cgls'}."
        elif requested_engine == "adtlert":
            compute_note += " ADTLERT was unavailable; original PyHydro ERT used."
        self._quality_view.show_quality(
            {"chi2": result.get("chi2"), "iterations": len(result.get("chi2_history") or []) or None,
             "n_data": result.get("n_data"), "lambda": self._lam.value(),
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
        self.fail_persisted_run(message, "ert.timelapse_inversion")
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
            self._pseudo_legend.setVisible(False)
            self._pseudo_readout.setText("No apparent-resistivity measurements loaded.")
            return
        arr = np.asarray(self._pseudo, dtype=float)
        mid, depth, rhoa = arr[:, 0], arr[:, 1], arr[:, 2]
        valid = np.isfinite(rhoa) & (rhoa > 0)
        mid, depth, rhoa = mid[valid], depth[valid], rhoa[valid]
        if rhoa.size == 0:
            self._pseudo_scatter.setData([])
            self._pseudo_plot.setTitle("No positive finite apparent resistivity to display")
            self._pseudo_legend.setVisible(False)
            self._pseudo_readout.setText(
                "All apparent-resistivity values are missing, non-finite, or non-positive."
            )
            return
        log_rhoa = np.log10(rhoa)
        lo, hi = np.percentile(log_rhoa, [3, 97])
        if hi <= lo:
            lo, hi = float(lo) - 0.5, float(hi) + 0.5
        rng = hi - lo
        norm = np.clip((log_rhoa - lo) / rng, 0.0, 1.0)
        lut = self._cmap.map(norm, mode="byte")
        spots = [
            {"pos": (float(mid[i]), float(depth[i])),
             "data": (float(mid[i]), float(depth[i]), float(rhoa[i])),
             "brush": pg.mkBrush(int(lut[i, 0]), int(lut[i, 1]), int(lut[i, 2])),
             "size": 11}
            for i in range(mid.size)
        ]
        self._pseudo_scatter.setData(spots)
        legend_values = np.power(10.0, np.linspace(float(lo), float(hi), 5))
        for label, value in zip(self._pseudo_scale_labels, legend_values):
            label.setText(f"{float(value):.4g}")
        self._pseudo_legend.setVisible(True)
        self._pseudo_readout.setText(
            "Click a measurement to read x, pseudo-depth, and apparent resistivity."
        )
        self._pseudo_plot.setTitle(
            f"Apparent resistivity: {rhoa.min():.3g} – {rhoa.max():.3g} Ω·m  "
            f"(n={rhoa.size})"
        )

    def _on_pseudo_clicked(self, _item, points, _event) -> None:
        """Report the physical value behind a pseudosection colour."""
        if not points:
            self._pseudo_readout.setText(
                "Click a measurement to read x, pseudo-depth, and apparent resistivity."
            )
            return
        data = points[0].data()
        if not data or len(data) != 3:
            return
        x, depth, rhoa = data
        self._pseudo_readout.setText(
            f"x = {float(x):.3g} m    ·    pseudo-depth = {float(depth):.3g} m"
            f"    ·    ρa = {float(rhoa):.4g} Ω·m"
        )

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

            from PyHydroGeophysX.core.mesh_serialization import via_ascii_path

            out = io_utils.ensure_dir(folder)
            mesh = mgr.paraDomain
            model = np.asarray(mgr.model, dtype=float)  # resistivity (ohm-m)
            np.save(out / "resistivity_model.npy", model)
            # numpy takes wide paths, PyGIMLi's writers do not; a folder Windows'
            # ANSI codepage cannot represent needs the write staged through a
            # temporary ASCII one. Export to a localized "文档" folder otherwise
            # writes the .npy and then fails on the .bms.
            via_ascii_path(mesh.save, out / "resistivity_mesh.bms", mode="write")
            mesh["resistivity"] = model
            via_ascii_path(mesh.exportVTK, out / "resistivity_model.vtk", mode="write")
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
                          "max_total_iterations, engine (pyhydro/pygimli/adtlert). "
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
        out_dir = self.state.ensure_results_store().scratch_dir(self.module_key)
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
            # The folded criteria are opt-in from the agent too: naming any one of
            # them unfolds the section, so the panel keeps matching what ran.
            extra = {"drop_nonpositive_rhoa": lambda v: self._qc_drop_neg.setChecked(bool(v)),
                     "min_voltage": lambda v: self._qc_min_v.setValue(float(v)),
                     "min_current": lambda v: self._qc_min_i.setValue(float(v)),
                     "max_geometric_factor": lambda v: self._qc_max_k.setValue(float(v)),
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
            "inversion_mode": lambda v: set_inv_mode(v),
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
