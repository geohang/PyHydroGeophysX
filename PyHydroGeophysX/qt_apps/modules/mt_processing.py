"""Magnetotellurics module: time series to transfer functions, then 1D and 2D models.

The page reads an instrument's recording (Phoenix MTU-5C and legacy MTU,
Metronix ATS, Zonge Z3D, LEMI-424) or sites processed elsewhere (EDI, EMTF
XML, Z- and J-files). It estimates each site's impedance and tipper with
robust, optionally remote-referenced processing, and inverts the sites for
resistivity: Occam 1D per site - with the static shift as a free parameter,
an optional TEM sounding that fixes it, and a water-content profile - and a
2D TE/TM section along a profile on SimPEG.

Laid out the way the seismic page is: the view tabs follow the work - time
series, the sites, the 1D model, the profile - and each brings up the side
panel for its own step, so the options on screen are the ones that act on
what is drawn.

Every computation is one of the ``mt.*`` workflows, run in a process of its
own; the numerics are in :mod:`PyHydroGeophysX.data_processing.mt`.
"""

from __future__ import annotations

import shutil
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
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
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from PyHydroGeophysX.data_processing import mt
from PyHydroGeophysX.qt_apps import io_utils, theme
from PyHydroGeophysX.qt_apps.modules.base import BaseModule, LogFn
from PyHydroGeophysX.qt_apps.qt_utils import (
    BusyStateController,
    ContentWidthScrollArea,
    ReproduceBar,
    make_double_spinbox,
    merged_row,
    select_directory,
    set_rows_enabled,
)
from PyHydroGeophysX.qt_apps.widgets import length_units
from PyHydroGeophysX.qt_apps.widgets.readout import navigation_toolbar
from PyHydroGeophysX.qt_apps.widgets.run_controls import StopButton, progress_with_stop
from PyHydroGeophysX.qt_apps.workers import ProcessWorkflowWorker, TaskWorker
from PyHydroGeophysX.workflows import (
    ArtifactRef,
    WorkflowRunResult,
    WorkflowSpec,
    export_workflow_bundle,
)

_RECORDING_FILTER = ("MT time series (*.bin *.td_* *.ats *.z3d *.Z3D *.tbl *.TBL *.txt *.TXT);;"
                     "All files (*)")
_TF_FILTER = "MT transfer functions (*.edi *.EDI *.xml *.zss *.zrr *.zmm *.j);;All files (*)"
_TEM_FILTER = "TEM sounding (*.csv *.txt *.dat *.npz);;All files (*)"
_EXAMPLE_SITE = "data/MT/NMX20.xml"


class _FigurePane(QWidget):
    """A matplotlib figure with the studio's navigation toolbar above it."""

    def __init__(self, figsize=(8, 6), parent=None) -> None:
        super().__init__(parent)
        self.figure = Figure(figsize=figsize)
        self.canvas = FigureCanvas(self.figure)
        toolbar, readout = navigation_toolbar(self.canvas, self)
        bar = QHBoxLayout()
        bar.addWidget(toolbar)
        bar.addWidget(readout, 1)
        self.controls = QHBoxLayout()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(self.controls)
        layout.addLayout(bar)
        layout.addWidget(self.canvas, 1)

    def reset(self) -> Figure:
        """An empty figure to draw on.

        Clearing axes that share a log x axis makes matplotlib warn about the
        limits it is discarding; nothing is drawn with them, so it is quiet here.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Attempt to set non-positive")
            self.figure.clear()
        return self.figure

    def message(self, text: str) -> None:
        """Clear the figure and say why there is nothing to draw."""
        self.reset()
        self.figure.text(0.5, 0.5, text, ha="center", va="center", color="#6e6e73", wrap=True)
        self.canvas.draw_idle()

    def finish(self) -> None:
        try:
            self.figure.tight_layout()
        except Exception:  # noqa: BLE001 - a crowded figure still draws
            pass
        self.canvas.draw_idle()


def _site_label(tf: Any) -> str:
    station = tf.station or "site"
    return (f"{station} · {tf.n_frequencies} periods, "
            f"{tf.period.min():.3g}–{tf.period.max():.3g} s")


def _finite(*values: Any) -> bool:
    """Whether every value is a finite number; a missing coordinate is None or NaN."""
    try:
        return bool(np.isfinite(np.asarray(values, dtype=float)).all())
    except (TypeError, ValueError):
        return False


def _safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in (text or "site")) or "site"


class MTProcessingModule(BaseModule):
    module_key = "mt_processing"
    module_title = "Magnetotellurics"

    def __init__(self, state: Any, log: LogFn, parent=None) -> None:
        super().__init__(state, log, parent)
        self._runs: List[Any] = []
        self._remote_runs: List[Any] = []
        self._recording_path: Optional[Path] = None
        self._remote_path: Optional[Path] = None
        #: Sites in list order: ``{"tf": TransferFunction, "source": Path | None}``.
        #: The Sites list, the 1D panel's site choice and the profile list all
        #: hold one row per entry, in this order.
        self._sites: List[Dict[str, Any]] = []
        self._result_1d: Optional[Dict[str, Any]] = None
        self._result_2d: Optional[Any] = None
        self._profile_sites: List[Dict[str, Any]] = []
        self._outputs: Dict[str, Path] = {}
        self._worker: Optional[ProcessWorkflowWorker] = None
        self._reader: Optional[TaskWorker] = None
        self._busy: Optional[BusyStateController] = None
        self._active_progress: Optional[QProgressBar] = None
        self._recipe_path = ""

        root = QHBoxLayout(self)
        self._tabs = QTabWidget()
        self._series_pane = _FigurePane((9, 6))
        self._series_run = QComboBox()
        self._series_run.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self._series_run.setMinimumContentsLength(24)
        self._series_run.currentIndexChanged.connect(lambda _i: self._draw_series())
        self._series_pane.controls.addWidget(QLabel("Run:"))
        self._series_pane.controls.addWidget(self._series_run, 1)
        self._sounding_pane = _FigurePane((7, 7))
        self._dimension_pane = _FigurePane((7, 7))
        self._model_pane = _FigurePane((8, 6))
        self._profile_pane = _FigurePane((9, 5))
        self._section_pane = _FigurePane((10, 5))
        # In the order the work is done: data, the sites, a 1D model per site,
        # then the profile - its phase tensors (to choose the strike) and its section.
        for pane, title in ((self._series_pane, "Time series"), (self._sounding_pane, "Sounding"),
                            (self._dimension_pane, "Dimensionality"), (self._model_pane, "1D model"),
                            (self._profile_pane, "Phase tensors"), (self._section_pane, "2D section")):
            self._tabs.addTab(pane, title)
        self._series_pane.message("Open a recording to see its channels.")
        self._sounding_pane.message("Process a recording, or add EDI / XML sites in the panel on the right.")
        self._dimension_pane.message("Process a recording, or add EDI / XML sites in the panel on the right.")
        self._model_pane.message("Run the 1D inversion from the panel on the right.")
        self._profile_pane.message("Tick two or more sites in the panel on the right.")
        self._section_pane.message("Run the 2D inversion from the panel on the right.")
        self._reproduce = ReproduceBar()
        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.addWidget(self._tabs, stretch=1)
        center_layout.addWidget(self._reproduce)
        root.addWidget(center, stretch=1)

        self._panels = {"data": self._build_data_panel(), "sites": self._build_sites_panel(),
                        "1d": self._build_1d_panel(), "2d": self._build_2d_panel()}
        for panel in self._panels.values():
            root.addWidget(panel)
        self._panel_of = {self._series_pane: "data", self._sounding_pane: "sites",
                          self._dimension_pane: "sites", self._model_pane: "1d",
                          self._profile_pane: "2d", self._section_pane: "2d"}
        self._tabs.currentChanged.connect(self._on_tab_changed)
        self._on_tab_changed()
        self._sync_options()
        self._update_site_info()
        # View > Length Units: depths and distances retick; the data stay in metres.
        length_units.notifier().changed.connect(self._on_length_unit_changed)

    # -- widget helpers -------------------------------------------------------
    @staticmethod
    def _dspin(value, lo, hi, step, dec) -> QDoubleSpinBox:
        return make_double_spinbox(value, lo, hi, step, dec)

    @staticmethod
    def _ispin(value, lo, hi) -> QSpinBox:
        spin = QSpinBox(); spin.setRange(lo, hi); spin.setValue(value)
        return spin

    @staticmethod
    def _hint(text: str) -> QLabel:
        label = QLabel(text); label.setWordWrap(True); theme.set_tone(label, "hint")
        return label

    @staticmethod
    def _primary(text: str, icon: str, slot: Callable[[], Any]) -> QPushButton:
        button = QPushButton(text); button.setProperty("primary", True)
        button.setIcon(theme.icon(icon, color="#ffffff"))
        button.clicked.connect(slot)
        return button

    @staticmethod
    def _button(text: str, icon: str, slot: Callable[[], Any], tip: str = "") -> QPushButton:
        button = QPushButton(text); button.setIcon(theme.icon(icon))
        button.clicked.connect(slot)
        if tip:
            button.setToolTip(tip)
        return button

    @staticmethod
    def _progress_bar() -> QProgressBar:
        bar = QProgressBar(); bar.setVisible(False)
        return bar

    def _path_row(self, edit: QLineEdit, browse: Callable[[], None]) -> QWidget:
        button = QPushButton("…"); button.setFixedWidth(32); button.clicked.connect(browse)
        clear = QPushButton("×"); clear.setFixedWidth(28); clear.clicked.connect(edit.clear)
        return merged_row(edit, button, clear)

    @staticmethod
    def _column(*groups: QWidget) -> ContentWidthScrollArea:
        scroll = ContentWidthScrollArea(minimum=430, maximum=500)
        panel = QWidget(); scroll.setWidget(panel)
        layout = QVBoxLayout(panel)
        for group in groups:
            layout.addWidget(group)
        layout.addStretch(1)
        scroll.setVisible(False)
        return scroll

    def _go_to(self, pane: QWidget) -> None:
        self._tabs.setCurrentWidget(pane)

    # -- panel: time series and processing (Time series tab) --------------------
    def _build_data_panel(self) -> ContentWidthScrollArea:
        recording = QGroupBox("Recording"); form = QFormLayout(recording)
        row = QHBoxLayout()
        row.addWidget(self._primary("Open folder…", "fa5s.folder-open",
                                    lambda: self._browse_recording(folder=True)))
        row.addWidget(self._button("Open file…", "fa5s.file", lambda: self._browse_recording(folder=False)))
        row.addWidget(self._button("Data format", "fa5s.file-alt", self._show_format_help))
        form.addRow(row)
        form.addRow(self._hint(
            "Phoenix MTU-5C and legacy MTU, Metronix ATS, Zonge Z3D and LEMI-424 are "
            "recognised from the files. Open a recording folder, or one file to read only "
            "that stream."))
        self._calibration = QLineEdit(); self._calibration.setPlaceholderText("found by the reader")
        self._calibration.setToolTip(
            "Phoenix and Metronix: a calibration file or a folder of them, used instead of "
            "what the reader finds beside the recording.")
        form.addRow("Calibration", self._path_row(self._calibration, self._browse_calibration))
        self._dipole_ex = self._dspin(100.0, 0.1, 10000.0, 10.0, 1)
        self._dipole_ey = self._dspin(100.0, 0.1, 10000.0, 10.0, 1)
        self._dipole_row = merged_row(self._dipole_ex, "Ex ×", self._dipole_ey, "Ey (m)")
        self._dipole_row.setToolTip("LEMI-424 files hold the electric field in mV, not mV/km; "
                                    "the dipole lengths convert it. Other formats store them.")
        form.addRow("LEMI dipoles", self._dipole_row)
        self._recording_info = QLabel("No recording loaded."); self._recording_info.setWordWrap(True)
        form.addRow(self._recording_info)

        remote = QGroupBox("Remote reference"); rv = QVBoxLayout(remote)
        rv.addWidget(self._hint(
            "A recording made at the same time at a quiet site. Its magnetic fields are the "
            "instruments of the regression, which keeps noise local to this site from biasing "
            "the impedance downward. Optional."))
        row = QHBoxLayout()
        row.addWidget(self._button("Open remote…", "fa5s.broadcast-tower", self._browse_remote))
        row.addWidget(self._button("Clear", "fa5s.times", self._clear_remote))
        rv.addLayout(row)
        self._remote_info = QLabel("No remote reference."); self._remote_info.setWordWrap(True)
        rv.addWidget(self._remote_info)

        processing = QGroupBox("Processing"); form = QFormLayout(processing)
        form.addRow(self._hint(
            "Windows are tapered, decimated level by level and Fourier transformed; in each "
            "band a robust regression (Huber, then a redescending cut) solves E = Z·H."))
        self._rates = QComboBox()
        self._rates.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self._rates.setMinimumContentsLength(16)
        self._rates.setToolTip("Which of the recording's sample rates to process. All of them "
                               "are processed separately and merged band by band.")
        form.addRow("Sample rates", self._rates)
        defaults = mt.ProcessingConfig()
        self._window = self._ispin(defaults.window_length, 16, 65536)
        self._overlap = self._ispin(defaults.overlap, 0, 32768)
        form.addRow("Window", merged_row(self._window, "samples, overlap", self._overlap))
        self._decimation = self._ispin(defaults.decimation_factor, 2, 16)
        self._levels = self._ispin(0, 0, 20)
        self._levels.setSpecialValueText("auto")
        self._levels.setToolTip("Decimation levels; auto adds levels while a band still holds "
                                "enough windows.")
        form.addRow("Decimate by", merged_row(self._decimation, "levels", self._levels))
        self._huber = self._dspin(defaults.huber, 0.5, 5.0, 0.1, 2)
        self._redescend = self._dspin(defaults.redescend, 1.0, 10.0, 0.1, 2)
        self._huber.setToolTip("Huber cut, in residual standard deviations.")
        self._redescend.setToolTip("Redescending cut: residuals well past it get no weight.")
        form.addRow("Robust cuts", merged_row(self._huber, "Huber,", self._redescend, "redescend"))
        self._min_points = self._ispin(defaults.min_points, 3, 1000)
        self._min_points.setToolTip("Bands with fewer independent estimates than this are dropped.")
        form.addRow("Min estimates", self._min_points)
        self._prewhiten = QCheckBox("Prewhiten (first difference)"); self._prewhiten.setChecked(defaults.prewhiten)
        self._tipper = QCheckBox("Tipper from Hz"); self._tipper.setChecked(defaults.tipper)
        self._uncalibrated = QCheckBox("Allow uncalibrated channels")
        self._uncalibrated.setToolTip("Process channels still in recorded units. The impedance "
                                      "is then wrong in scale and phase; for checks only.")
        form.addRow(self._prewhiten)
        form.addRow(self._tipper)
        form.addRow(self._uncalibrated)
        self._band_setup = QLineEdit(); self._band_setup.setPlaceholderText("EMTF default bands")
        self._band_setup.setToolTip("An EMTF band-setup file (bs_*.cfg) to use its bands.")
        form.addRow("Band setup", self._path_row(self._band_setup, self._browse_band_setup))

        run = QGroupBox("Run"); rv = QVBoxLayout(run)
        self._process_btn = self._primary("Process time series", "fa5s.wave-square", self._run_processing)
        rv.addWidget(self._process_btn)
        self._process_progress = self._progress_bar()
        self._process_stop = self.stop_button("The processing")
        rv.addWidget(progress_with_stop(self._process_progress, self._process_stop))
        rv.addWidget(self._hint("The processed site is added to the sites and opens on the "
                                "Sounding tab."))

        elsewhere = QGroupBox("Sites processed elsewhere"); ev = QVBoxLayout(elsewhere)
        ev.addWidget(self._hint("EDI, EMTF XML, Z- and J-files skip this step."))
        ev.addWidget(self._button("Add EDI / XML sites…", "fa5s.plus", self._add_sites_and_show))
        return self._column(recording, remote, processing, run, elsewhere)

    # -- panel: sites (Sounding and Dimensionality tabs) ------------------------
    def _build_sites_panel(self) -> ContentWidthScrollArea:
        sites = QGroupBox("Sites"); v = QVBoxLayout(sites)
        row = QHBoxLayout()
        row.addWidget(self._button("Add EDI / XML…", "fa5s.plus", self._browse_sites))
        row.addWidget(self._button("Example site", "fa5s.flask", self._load_example,
                                   "NMX20, a USMTArray long-period site in New Mexico (EMTF XML)."))
        row.addWidget(self._button("Remove", "fa5s.trash", self._remove_site))
        v.addLayout(row)
        self._site_list = QListWidget(); self._site_list.setMinimumHeight(130)
        self._site_list.currentRowChanged.connect(self._select_site)
        v.addWidget(self._site_list)
        v.addWidget(self._hint("The selected site is drawn on the Sounding and Dimensionality "
                               "tabs, and is the one the 1D model tab inverts."))

        selected = QGroupBox("Selected site"); sv = QVBoxLayout(selected)
        self._site_info = QLabel(); self._site_info.setWordWrap(True)
        self._site_info.setTextInteractionFlags(Qt.TextSelectableByMouse)
        sv.addWidget(self._site_info)
        sv.addWidget(self._button("Export EDI / XML…", "fa5s.file-export", self._export_site,
                                  "Write the selected site as EDI and EMTF XML to a folder."))

        display = QGroupBox("Sounding display"); dform = QFormLayout(display)
        self._show_xy = QCheckBox("Zxy"); self._show_xy.setChecked(True)
        self._show_yx = QCheckBox("Zyx"); self._show_yx.setChecked(True)
        self._show_det = QCheckBox("det(Z)")
        dform.addRow("Components", merged_row(self._show_xy, self._show_yx, self._show_det))
        self._show_errors = QCheckBox("Error bars"); self._show_errors.setChecked(True)
        self._show_fit = QCheckBox("1D model's fit")
        self._show_fit.setChecked(True)
        self._show_fit.setToolTip("Draw the 1D inversion's predicted curve when the selected site "
                                  "has been inverted.")
        dform.addRow(merged_row(self._show_errors, self._show_fit))
        for box in (self._show_xy, self._show_yx, self._show_det, self._show_errors, self._show_fit):
            box.toggled.connect(lambda _c: self._draw_sounding())

        next_step = QGroupBox("Next"); nv = QVBoxLayout(next_step)
        row = QHBoxLayout()
        row.addWidget(self._button("1D model →", "fa5s.layer-group", lambda: self._go_to(self._model_pane),
                                   "Invert the selected site for a layered model."))
        row.addWidget(self._button("Profile →", "fa5s.th", lambda: self._go_to(self._profile_pane),
                                   "Tick the sites of a line, check their phase tensors and "
                                   "invert them in 2D."))
        nv.addLayout(row)
        return self._column(sites, selected, display, next_step)

    # -- panel: 1D inversion (1D model tab) ---------------------------------------
    def _build_1d_panel(self) -> ContentWidthScrollArea:
        site = QGroupBox("Site"); sform = QFormLayout(site)
        self._site_combo = QComboBox()
        self._site_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self._site_combo.setMinimumContentsLength(20)
        self._site_combo.currentIndexChanged.connect(self._select_site)
        sform.addRow("Invert", self._site_combo)
        sform.addRow(self._hint("Occam's smoothest layered model that fits the site to the "
                                "target RMS."))

        data = QGroupBox("Data and layers"); form = QFormLayout(data)
        self._mode = QComboBox()
        for label, value in (("det(Z) - rotation invariant", "det"), ("Zxy", "xy"), ("Zyx", "yx"),
                             ("Zxy and Zyx together", "both")):
            self._mode.addItem(label, value)
        form.addRow("Data", self._mode)
        self._layers = self._ispin(50, 5, 200)
        self._target_rms = self._dspin(1.0, 0.3, 10.0, 0.1, 2)
        form.addRow("Layers", merged_row(self._layers, "target RMS", self._target_rms))
        self._error_floor_1d = self._dspin(0.05, 0.0, 0.5, 0.01, 3)
        self._error_floor_1d.setToolTip("Floor on the relative impedance error: 0.05 = 5 % of |Z|, "
                                        "about 10 % in apparent resistivity and 1.4° in phase.")
        form.addRow("Error floor", self._error_floor_1d)

        shift = QGroupBox("Static shift"); form = QFormLayout(shift)
        self._static_shift = QCheckBox("Solve for a shift per mode")
        self._static_shift.setToolTip(
            "One multiplier of each mode's apparent resistivity. From MT alone it is only "
            "relative between the modes; a TEM sounding fixes its absolute level.")
        form.addRow(self._static_shift)
        self._tem = QLineEdit(); self._tem.setPlaceholderText("none")
        self._tem.setToolTip("A central-loop TDEM sounding at the site: gate time (s) and "
                             "response columns, or a .npz with times and response.")
        self._tem.textChanged.connect(lambda _t: self._sync_options())
        form.addRow("TEM sounding", self._path_row(self._tem, self._browse_tem))
        self._tem_radius = self._dspin(20.0, 0.5, 1000.0, 1.0, 1)
        self._tem_error = self._dspin(0.03, 0.005, 0.5, 0.01, 3)
        self._tem_row = merged_row(self._tem_radius, "m loop, error", self._tem_error)
        form.addRow("TEM loop", self._tem_row)
        form.addRow(self._hint("TEM measures no electric field, so it carries no static shift: "
                               "fitted jointly, it fixes the shift's absolute level."))

        water = QGroupBox("Water content"); form = QFormLayout(water)
        self._water = QCheckBox("Convert the model (Waxman-Smits)")
        self._water.toggled.connect(lambda _c: self._sync_options())
        form.addRow(self._water)
        self._rhos = self._dspin(20.0, 0.01, 10000.0, 1.0, 2)
        self._sat_n = self._dspin(2.0, 1.0, 5.0, 0.1, 2)
        self._porosity = self._dspin(0.3, 0.01, 0.9, 0.01, 3)
        self._sigma_sur = self._dspin(0.0, 0.0, 1.0, 0.001, 4)
        self._water_row = merged_row(self._rhos, "Ω·m water, n", self._sat_n)
        self._porosity_row = merged_row(self._porosity, "porosity, σsur", self._sigma_sur, "S/m")
        form.addRow("Pore water", self._water_row)
        form.addRow("Rock", self._porosity_row)

        run = QGroupBox("Run"); rv = QVBoxLayout(run)
        self._run_1d_btn = self._primary("Run 1D inversion", "fa5s.layer-group", self._run_1d)
        rv.addWidget(self._run_1d_btn)
        self._progress_1d = self._progress_bar()
        self._stop_1d = self.stop_button("The 1D inversion")
        rv.addWidget(progress_with_stop(self._progress_1d, self._stop_1d))
        self._result_1d_info = QLabel("No 1D model yet."); self._result_1d_info.setWordWrap(True)
        rv.addWidget(self._result_1d_info)
        row = QHBoxLayout()
        row.addWidget(self._button("Export model…", "fa5s.file-csv",
                                   lambda: self._export_outputs("mt.invert_1d"),
                                   "The layered model and its fit as CSV, with the site's EDI."))
        row.addWidget(self.map_export_button())
        rv.addLayout(row)
        return self._column(site, data, shift, water, run)

    # -- panel: profile (Phase tensors and 2D section tabs) --------------------------
    def _build_2d_panel(self) -> ContentWidthScrollArea:
        profile = QGroupBox("Profile sites"); v = QVBoxLayout(profile)
        self._profile_list = QListWidget(); self._profile_list.setMinimumHeight(130)
        self._profile_list.setSelectionMode(QAbstractItemView.NoSelection)
        self._profile_list.itemChanged.connect(lambda _item: self._on_profile_changed())
        v.addWidget(self._profile_list)
        v.addWidget(self._hint("The ticked sites, in this order, form the profile. Their "
                               "distances come from their coordinates, or from the spacing below."))
        form = QFormLayout()
        self._spacing = self._dspin(500.0, 1.0, 1e6, 50.0, 1)
        self._spacing.setToolTip("Used only when the sites carry no latitude and longitude.")
        self._spacing.valueChanged.connect(lambda _v: self._on_profile_changed())
        form.addRow("Site spacing (m)", self._spacing)
        self._ellipse_colour = QComboBox()
        for label, value in (("Φmin (°)", "phi_min"), ("Skew β (°)", "beta"), ("Φmax (°)", "phi_max")):
            self._ellipse_colour.addItem(label, value)
        self._ellipse_colour.currentIndexChanged.connect(lambda _i: self._draw_phase_tensors())
        form.addRow("Ellipse colour", self._ellipse_colour)
        v.addLayout(form)
        self._profile_info = QLabel(); self._profile_info.setWordWrap(True)
        v.addWidget(self._profile_info)

        strike = QGroupBox("Strike and modes"); form = QFormLayout(strike)
        self._modes_2d = QComboBox()
        for label, value in (("TE + TM", ["te", "tm"]), ("TE only", ["te"]), ("TM only", ["tm"])):
            self._modes_2d.addItem(label, value)
        form.addRow("Modes", self._modes_2d)
        self._strike_auto = QCheckBox("normal to the profile")
        self._strike_auto.setChecked(True)
        self._strike_auto.toggled.connect(lambda _c: self._sync_options())
        self._strike = self._dspin(0.0, 0.0, 180.0, 5.0, 1)
        self._strike.setToolTip("Geoelectric strike, degrees clockwise from north. Check it against "
                                "the phase tensors before inverting.")
        form.addRow("Strike (°)", merged_row(self._strike, self._strike_auto))
        form.addRow(self._hint("The impedances are rotated to the strike: TE is the electric "
                               "field along it, TM across it. Read the strike off the Phase "
                               "tensors tab, where Φmax and Φmin part."))

        inversion = QGroupBox("Inversion"); form = QFormLayout(inversion)
        self._error_floor_2d = self._dspin(0.05, 0.01, 0.5, 0.01, 3)
        self._iterations_2d = self._ispin(20, 1, 100)
        form.addRow("Error floor", merged_row(self._error_floor_2d, "iterations", self._iterations_2d))
        self._n_freq_2d = self._ispin(16, 3, 200)
        self._n_freq_2d.setToolTip("Frequencies inverted, spread evenly in log over the first "
                                   "site's. Each costs one 2D forward solve per mode.")
        form.addRow("Frequencies", self._n_freq_2d)
        form.addRow(self._hint("SimPEG solves the 2D problem on a mesh laid out from the sites "
                               "and the frequencies, with air above and padding around."))

        run = QGroupBox("Run"); rv = QVBoxLayout(run)
        self._run_2d_btn = self._primary("Run 2D inversion", "fa5s.th", self._run_2d)
        rv.addWidget(self._run_2d_btn)
        self._progress_2d = self._progress_bar()
        self._stop_2d = self.stop_button("The 2D inversion")
        rv.addWidget(progress_with_stop(self._progress_2d, self._stop_2d))
        self._result_2d_info = QLabel("No 2D section yet."); self._result_2d_info.setWordWrap(True)
        rv.addWidget(self._result_2d_info)
        row = QHBoxLayout()
        row.addWidget(self._button("Export section…", "fa5s.file-csv",
                                   lambda: self._export_outputs("mt.invert_profile"),
                                   "The section as NPZ and CSV."))
        row.addWidget(self.map_export_button())
        rv.addLayout(row)
        return self._column(profile, strike, inversion, run)

    def _on_tab_changed(self, _index: int = 0) -> None:
        """One side panel per step: the options for what is on screen, nothing else.

        The Time series tab has reading and processing; Sounding and
        Dimensionality the sites; the 1D model tab the 1D inversion; Phase
        tensors and 2D section the profile. A result tab keeps the settings
        that made it beside it, so a re-run does not mean navigating back.
        """
        wanted = self._panel_of.get(self._tabs.currentWidget(), "data")
        for key, panel in self._panels.items():
            panel.setVisible(key == wanted)

    def _sync_options(self) -> None:
        set_rows_enabled([self._tem_row], bool(self._tem.text().strip()))
        set_rows_enabled([self._water_row, self._porosity_row], self._water.isChecked())
        self._strike.setEnabled(not self._strike_auto.isChecked())

    # -- browsing ------------------------------------------------------------
    def _browse_recording(self, folder: bool) -> None:
        from PyHydroGeophysX.qt_apps.widgets.project_dialogs import confirm_project_for_data
        if not confirm_project_for_data(self):   # name a Project before the first data
            return
        if folder:
            path = QFileDialog.getExistingDirectory(self, "Open MT recording folder")
        else:
            path, _ = QFileDialog.getOpenFileName(self, "Open MT time-series file", "", _RECORDING_FILTER)
        if path:
            self._read_recording(Path(path), remote=False)

    def _browse_remote(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Open remote-reference recording folder")
        if path:
            self._read_recording(Path(path), remote=True)

    def _clear_remote(self) -> None:
        self._remote_runs, self._remote_path = [], None
        self._update_recording_info()

    def _browse_calibration(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Calibration folder")
        if path:
            self._calibration.setText(path)

    def _browse_band_setup(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "EMTF band setup", "", "Band setup (*.cfg *.txt);;All files (*)")
        if path:
            self._band_setup.setText(path)

    def _browse_tem(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "TEM sounding", "", _TEM_FILTER)
        if path:
            self._tem.setText(path)

    def _browse_sites(self) -> None:
        from PyHydroGeophysX.qt_apps.widgets.project_dialogs import confirm_project_for_data
        if not confirm_project_for_data(self):   # name a Project before the first data
            return
        paths, _ = QFileDialog.getOpenFileNames(self, "Add MT sites", "", _TF_FILTER)
        for path in paths:
            self._add_site_file(Path(path))

    def _add_sites_and_show(self) -> None:
        """Add sites from the Time series tab, and go to where they are drawn."""
        before = len(self._sites)
        self._browse_sites()
        if len(self._sites) > before:
            self._go_to(self._sounding_pane)

    # -- time series ---------------------------------------------------------
    def _reader_options(self, path: Path) -> Dict[str, Any]:
        options: Dict[str, Any] = {}
        kind = mt.timeseries_format(path)
        calibration = self._calibration.text().strip()
        if calibration and kind in ("phoenix", "metronix"):
            options["calibration"] = calibration
        if kind == "lemi424":
            options["dipole_lengths"] = {"ex": float(self._dipole_ex.value()),
                                         "ey": float(self._dipole_ey.value())}
        return options

    def _read_recording(self, path: Path, *, remote: bool) -> None:
        if self._reader is not None and self._reader.isRunning():
            self.log("A recording is still being read.", "warn")
            return
        try:
            options = self._reader_options(path)
        except Exception as exc:  # noqa: BLE001
            self.log(f"Could not inspect {path.name}: {exc}", "error")
            return
        self.log(f"Reading {'remote ' if remote else ''}recording {path} …", "info")
        (self._remote_info if remote else self._recording_info).setText(f"Reading {path.name} …")
        worker = TaskWorker(mt.read_timeseries, path, **options)
        worker.succeeded.connect(lambda runs: self._on_recording_read(path, runs, remote))
        worker.failed.connect(lambda message: self._on_recording_failed(path, message))
        self._reader = self.register_worker(worker)
        worker.start()

    def _on_recording_failed(self, path: Path, message: str) -> None:
        self.log(f"Could not read {path.name}: {message}", "error")
        self._update_recording_info()

    def _on_recording_read(self, path: Path, runs: List[Any], remote: bool) -> None:
        if not runs:
            self._on_recording_failed(path, "no runs in it")
            return
        if remote:
            self._remote_runs, self._remote_path = list(runs), path
        else:
            self._runs, self._recording_path = list(runs), path
            self._rates.clear()
            self._rates.addItem("All sample rates", None)
            for rate in sorted({float(r.sample_rate) for r in runs}, reverse=True):
                count = sum(1 for r in runs if float(r.sample_rate) == rate)
                self._rates.addItem(f"{rate:g} Hz ({count} run{'s' if count > 1 else ''})", rate)
            self._series_run.blockSignals(True)
            self._series_run.clear()
            for k, run in enumerate(runs):
                self._series_run.addItem(f"{k + 1}: {run.sample_rate:g} Hz, {run.duration:g} s from "
                                         f"{str(run.start)[:19]}")
            self._series_run.blockSignals(False)
            self._draw_series()
            self._go_to(self._series_pane)
        for run in runs:
            self.log(run.summary(), "info")
        self._update_recording_info()
        self.log(f"Read {len(runs)} run(s) from {path.name}.", "success")

    def _update_recording_info(self) -> None:
        if self._runs:
            first = self._runs[0]
            uncalibrated = sorted({name for run in self._runs for name, ch in run.channels.items()
                                   if not ch.is_calibrated})
            lines = [f"<b>{first.station or self._recording_path.name}</b> · "
                     f"{first.instrument or 'recording'} · {len(self._runs)} run(s), "
                     f"{', '.join(first.components)}"]
            if uncalibrated:
                lines.append(f"<span style='color:#9b5a00'>Uncalibrated: {', '.join(uncalibrated)}</span>")
            self._recording_info.setText("<br>".join(lines))
        else:
            self._recording_info.setText("No recording loaded.")
        if self._remote_runs:
            self._remote_info.setText(f"<b>{self._remote_runs[0].station or self._remote_path.name}</b> · "
                                      f"{len(self._remote_runs)} run(s), "
                                      f"{', '.join(self._remote_runs[0].components)}")
        else:
            self._remote_info.setText("No remote reference.")

    def _selected_runs(self) -> List[Any]:
        rate = self._rates.currentData()
        return [r for r in self._runs if rate is None or float(r.sample_rate) == float(rate)]

    def _draw_series(self) -> None:
        index = self._series_run.currentIndex()
        if not (0 <= index < len(self._runs)):
            return
        run = self._runs[index]
        names = list(run.components)
        figure = self._series_pane.reset()
        axes = figure.subplots(len(names), 1, sharex=True, squeeze=False)[:, 0]
        # A long run is drawn as the range of each block of samples: every
        # n-th sample alone aliases a signal above the drawing's resolution.
        block = max(1, run.n_samples // 5000)
        n_blocks = run.n_samples // block
        t = (np.arange(n_blocks) * block + 0.5 * block) / run.sample_rate
        for ax, name in zip(axes, names):
            channel = run.channels[name]
            data = np.asarray(channel.data, dtype=float)
            if block > 1:
                blocks = data[: n_blocks * block].reshape(n_blocks, block)
                ax.fill_between(t, blocks.min(axis=1), blocks.max(axis=1), lw=0, color="#007aff")
            else:
                ax.plot(t, data, lw=0.6, color="#007aff")
            ax.set_ylabel(f"{name}\n({channel.units})", fontsize=8)
            ax.grid(True, alpha=0.25)
        axes[0].set_title(f"{run.station or 'run'} · {run.sample_rate:g} Hz from {str(run.start)[:19]}"
                          + (f" (range of every {block} samples)" if block > 1 else ""), fontsize=9)
        axes[-1].set_xlabel("Time (s)")
        self._series_pane.finish()

    # -- sites ---------------------------------------------------------------
    def _add_site(self, tf: Any, source: Optional[Path] = None, *, select: bool = True) -> None:
        self._sites.append({"tf": tf, "source": source})
        label = _site_label(tf)
        tip = str(source) if source else "processed in this session"
        self._site_list.blockSignals(True)
        self._site_list.addItem(label)
        self._site_list.item(self._site_list.count() - 1).setToolTip(tip)
        self._site_list.blockSignals(False)
        self._site_combo.blockSignals(True)
        self._site_combo.addItem(label)
        self._site_combo.blockSignals(False)
        item = QListWidgetItem(label)
        item.setFlags((item.flags() | Qt.ItemIsUserCheckable) & ~Qt.ItemIsSelectable)
        item.setCheckState(Qt.Checked)
        item.setToolTip(tip)
        self._profile_list.blockSignals(True)
        self._profile_list.addItem(item)
        self._profile_list.blockSignals(False)
        if select or self._site_list.currentRow() < 0:
            self._select_site(len(self._sites) - 1)
        self._on_profile_changed()
        self._publish()

    def _add_site_file(self, path: Path) -> bool:
        try:
            tf = mt.read_transfer_function(path)
        except Exception as exc:  # noqa: BLE001
            self.log(f"Could not read {path.name}: {exc}", "error")
            return False
        self._add_site(tf, path)
        self.log(f"Added site {tf.station or path.stem}: {tf.n_frequencies} periods "
                 f"{tf.period.min():.3g}–{tf.period.max():.3g} s.", "success")
        return True

    def _load_example(self) -> Dict[str, Any]:
        path = io_utils.find_example(_EXAMPLE_SITE, self.state.project_root)
        if path is None:
            message = io_utils.missing_example_message("The MT example site", _EXAMPLE_SITE)
            self.log(message, "warn")
            return {"status": "failed", "error": message}
        if not self._add_site_file(path):
            return {"status": "failed", "error": f"Could not read {path}"}
        return {"status": "ok", "site": self._sites[-1]["tf"].station, "path": str(path)}

    def _remove_site(self) -> None:
        row = self._site_list.currentRow()
        if not (0 <= row < len(self._sites)):
            return
        self._sites.pop(row)
        for widget in (self._site_list, self._profile_list):
            widget.blockSignals(True)
            widget.takeItem(row)
            widget.blockSignals(False)
        self._site_combo.blockSignals(True)
        self._site_combo.removeItem(row)
        self._site_combo.blockSignals(False)
        self._select_site(min(row, len(self._sites) - 1))
        self._on_profile_changed()

    def _current_site(self) -> Optional[Dict[str, Any]]:
        row = self._site_list.currentRow()
        return self._sites[row] if 0 <= row < len(self._sites) else None

    def _checked_sites(self) -> List[Dict[str, Any]]:
        if self._profile_list.count() != len(self._sites):
            return []
        return [site for k, site in enumerate(self._sites)
                if self._profile_list.item(k).checkState() == Qt.Checked]

    def _select_site(self, row: int) -> None:
        """Make ``row`` the selected site in the Sites list and the 1D panel alike."""
        for widget, current, setter in ((self._site_list, self._site_list.currentRow(),
                                         self._site_list.setCurrentRow),
                                        (self._site_combo, self._site_combo.currentIndex(),
                                         self._site_combo.setCurrentIndex)):
            if current != row:
                widget.blockSignals(True)
                setter(row)
                widget.blockSignals(False)
        self._update_site_info()
        if self._current_site() is None:
            self._sounding_pane.message("Process a recording, or add EDI / XML sites in the panel on the right.")
            self._dimension_pane.message("Process a recording, or add EDI / XML sites in the panel on the right.")
            return
        self._draw_sounding()
        self._draw_dimensionality()

    def _update_site_info(self) -> None:
        site = self._current_site()
        if site is None:
            self._site_info.setText("No site selected.")
            return
        tf = site["tf"]
        lines = [f"<b>{tf.station or 'site'}</b>: {tf.n_frequencies} periods, "
                 f"{tf.period.min():.3g}–{tf.period.max():.3g} s"]
        if _finite(tf.latitude, tf.longitude):
            elevation = getattr(tf, "elevation", None)
            lines.append(f"{tf.latitude:.5f}°, {tf.longitude:.5f}°"
                         + (f", {float(elevation):g} m" if _finite(elevation) else ""))
        else:
            lines.append("No coordinates")
        lines.append("Impedance" + (" and tipper" if tf.has_tipper else "")
                     + (", with errors" if tf.z_err is not None else ", no errors"))
        lines.append(f"From {Path(site['source']).name}" if site.get("source") else "Processed in this session")
        self._site_info.setText("<br>".join(lines))

    def _site_positions(self, sites: Sequence[Dict[str, Any]]):
        """Distances along the profile (m) and its azimuth, from coordinates or the spacing."""
        tfs = [site["tf"] for site in sites]
        try:
            return mt.station_distances(tfs)
        except ValueError:
            return np.arange(len(tfs), dtype=float) * float(self._spacing.value()), None

    def _on_profile_changed(self) -> None:
        sites = self._checked_sites()
        if len(sites) < 2:
            self._profile_info.setText(f"{len(sites)} site ticked; the profile needs two or more.")
        else:
            positions, azimuth = self._site_positions(sites)
            where = (f"azimuth {azimuth:.0f}°, so the strike normal to it is "
                     f"{(azimuth + 90.0) % 180.0:.0f}°" if azimuth is not None
                     else f"no coordinates, so spaced {self._spacing.value():g} m apart")
            self._profile_info.setText(f"<b>{len(sites)} sites over {np.ptp(positions):.0f} m</b>; {where}.")
        self._draw_phase_tensors()

    # -- drawing -------------------------------------------------------------
    def _draw_sounding(self) -> None:
        site = self._current_site()
        if site is None:
            return
        from PyHydroGeophysX.visualization import plot_mt_sounding

        tf = site["tf"]
        components = tuple(name for name, box in (("xy", self._show_xy), ("yx", self._show_yx),
                                                  ("det", self._show_det)) if box.isChecked())
        if not components:
            self._sounding_pane.message("Tick a component to draw in the panel on the right.")
            return
        figure = self._sounding_pane.reset()
        axes = figure.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 2]})
        predicted = None
        if self._show_fit.isChecked() and self._result_1d is not None and self._result_1d["site"] is site:
            result = self._result_1d["result"]
            sounding = result.sounding
            mode = sounding.modes[0] if len(sounding.modes) == 1 else "xy"
            shift = result.static_shift.get(mode, 1.0)
            predicted = {mode: (1 / sounding.frequency, 10 ** result.predicted["log_rho_a"] * shift,
                                result.predicted["phase"])}
        try:
            plot_mt_sounding(tf, components=components, axes=axes, predicted=predicted,
                             title=tf.station or "MT sounding", errors=self._show_errors.isChecked())
        except Exception as exc:  # noqa: BLE001
            self._sounding_pane.message(f"Cannot draw this sounding: {exc}")
            return
        self._sounding_pane.finish()

    def _draw_dimensionality(self) -> None:
        site = self._current_site()
        if site is None:
            return
        from PyHydroGeophysX.visualization import plot_mt_dimensionality

        tf = site["tf"]
        figure = self._dimension_pane.reset()
        try:
            plot_mt_dimensionality(tf, axes=figure.subplots(3, 1, sharex=True),
                                   title=f"{tf.station or 'site'}: |β| under 3° (grey) reads as 1D "
                                         "or 2D; strike is ambiguous by 90°")
        except Exception as exc:  # noqa: BLE001
            self._dimension_pane.message(f"No phase tensor: {exc}")
            return
        self._dimension_pane.finish()

    def _draw_phase_tensors(self) -> None:
        sites = self._checked_sites()
        if len(sites) < 2:
            self._profile_pane.message("Tick two or more sites in the panel on the right to see their "
                                       "phase tensors along the profile.")
            return
        from PyHydroGeophysX.visualization import plot_phase_tensor_pseudosection

        positions, _azimuth = self._site_positions(sites)
        figure = self._profile_pane.reset()
        ax = figure.add_subplot(111)
        try:
            plot_phase_tensor_pseudosection([s["tf"] for s in sites], positions, ax=ax,
                                            color_by=str(self._ellipse_colour.currentData()),
                                            length_unit=length_units.current())
        except Exception as exc:  # noqa: BLE001
            self._profile_pane.message(f"Cannot draw the phase tensors: {exc}")
            return
        self._profile_pane.finish()

    def _draw_model(self) -> None:
        if self._result_1d is None:
            return
        from PyHydroGeophysX.visualization import plot_mt_model_1d
        from PyHydroGeophysX.visualization.axis_units import set_length_axis

        result, water = self._result_1d["result"], self._result_1d.get("water")
        station = self._result_1d["site"]["tf"].station or "site"
        unit = length_units.current()
        figure = self._model_pane.reset()
        axes = figure.subplots(1, 2 if water else 1, sharey=True, squeeze=False)[0]
        plot_mt_model_1d({f"Occam 1D, {station} (RMS {result.rms:.2f})": result}, ax=axes[0],
                         length_unit=unit)
        if water:
            depth, _rho = result.depth_profile()
            values = np.repeat(np.asarray(water["water_content"], dtype=float), 2)
            axes[1].plot(values, depth, lw=1.8, color="#007aff")
            axes[1].set_xlabel("Water content (-)")
            set_length_axis(axes[1], "y", "Depth", unit=unit)
            axes[1].grid(True, alpha=0.25)
        self._model_pane.finish()

    def _draw_section(self) -> None:
        if self._result_2d is None:
            return
        from PyHydroGeophysX.visualization import plot_mt_section

        figure = self._section_pane.reset()
        ax = figure.add_subplot(111)
        result = self._result_2d
        plot_mt_section(result, ax=ax, length_unit=length_units.current(),
                        title=f"2D MT, {' + '.join(m.upper() for m in result.modes)} "
                              f"(RMS {result.rms:.2f})")
        self._section_pane.finish()

    def _on_length_unit_changed(self, _unit: str) -> None:
        self._draw_phase_tensors()
        self._draw_model()
        self._draw_section()

    # -- runs ----------------------------------------------------------------
    def _running(self) -> bool:
        return self._worker is not None and self._worker.isRunning()

    def _site_file(self, site: Dict[str, Any], folder: Path, index: int) -> Path:
        """The site as a file in the run's inputs: its own file, or an EDI of it."""
        folder.mkdir(parents=True, exist_ok=True)
        stem = f"{index:02d}_{_safe_name(site['tf'].station)}"
        source = site.get("source")
        if source is not None and Path(source).is_file():
            return Path(shutil.copy2(source, folder / f"{stem}{Path(source).suffix.lower()}"))
        return Path(mt.write_edi(site["tf"], folder / f"{stem}.edi"))

    def _start(self, spec: WorkflowSpec, run: Any, stem: str, objects: Sequence[str],
               on_ok: Callable[[WorkflowRunResult, Any], None], message: str,
               progress: QProgressBar, stop: StopButton) -> None:
        recipe_path, script_path = export_workflow_bundle(spec, run.run_dir, stem=stem)
        self._reproduce.set_bundle(recipe_path, script_path)
        self._recipe_path = str(recipe_path)
        # One run at a time: every panel's Run button waits for it.
        self._busy = BusyStateController([self._process_btn, self._run_1d_btn, self._run_2d_btn])
        self._busy.start()
        self._active_progress = progress
        progress.setVisible(True); progress.setRange(0, 0)
        self.log(message, "info")
        workflow_id = spec.workflow_id
        worker = ProcessWorkflowWorker(recipe_path, run.run_dir, run.outputs_dir, run.result_path,
                                       objects=tuple(objects))
        worker.logged.connect(lambda text: self.log(text, "info"))

        def succeeded(result: WorkflowRunResult) -> None:
            self._outputs[workflow_id] = run.outputs_dir
            try:
                on_ok(result, run)
            finally:
                if hasattr(self.state, "update_workflow_result"):
                    self.state.update_workflow_result(self.module_key, workflow_id, result.to_dict(),
                                                      recipe_path=self._recipe_path)

        def failed(text: str) -> None:
            self.fail_persisted_run(text)
            self.log(f"{workflow_id} failed: {text}", "error")

        worker.succeeded.connect(succeeded)
        worker.failed.connect(failed)
        worker.finished.connect(self._run_finished)
        self._worker = self.register_worker(worker)
        worker.start()
        stop.attach(worker, workflow_id)

    def _run_finished(self) -> None:
        if self._busy is not None:
            self._busy.finish()
            self._busy = None
        if self._active_progress is not None:
            self._active_progress.setVisible(False)
            self._active_progress = None

    def _processing_parameters(self) -> Dict[str, Any]:
        parameters: Dict[str, Any] = {
            "window_length": int(self._window.value()),
            "overlap": int(self._overlap.value()),
            "decimation_factor": int(self._decimation.value()),
            "huber": float(self._huber.value()),
            "redescend": float(self._redescend.value()),
            "min_points": int(self._min_points.value()),
            "prewhiten": bool(self._prewhiten.isChecked()),
            "tipper": bool(self._tipper.isChecked()),
            "allow_uncalibrated": bool(self._uncalibrated.isChecked()),
        }
        if self._levels.value() > 0:
            parameters["n_levels"] = int(self._levels.value())
        return parameters

    def _run_processing(self) -> Dict[str, Any]:
        runs = self._selected_runs()
        if not runs:
            self.log("Open a recording first.", "warn")
            return {"status": "failed", "error": "No recording is loaded."}
        if self._running():
            return {"status": "failed", "error": "A run is in progress."}
        try:
            run = self.begin_persisted_run("mt.process", "mt.process")
            inputs = {"recording": ArtifactRef.from_path(
                Path(mt.save_runs(run.inputs_dir / "recording_runs.npz", runs)),
                artifact_id="mt:recording", kind="mt_timeseries", format="npz", base_dir=run.run_dir)}
            if self._remote_runs:
                inputs["remote"] = ArtifactRef.from_path(
                    Path(mt.save_runs(run.inputs_dir / "remote_runs.npz", self._remote_runs)),
                    artifact_id="mt:remote", kind="mt_timeseries", format="npz", base_dir=run.run_dir)
            band_setup = self._band_setup.text().strip()
            if band_setup:
                copied = Path(shutil.copy2(band_setup, run.inputs_dir / Path(band_setup).name))
                inputs["band_setup"] = ArtifactRef.from_path(copied, artifact_id="mt:band_setup",
                                                             kind="mt_band_setup", base_dir=run.run_dir)
            spec = WorkflowSpec(
                workflow_id="mt.process", inputs=inputs, parameters=self._processing_parameters(),
                metadata={"source": str(self._recording_path or ""),
                          "remote": str(self._remote_path or "")})
            rates = ", ".join(sorted({f"{r.sample_rate:g} Hz" for r in runs}))
            self._start(spec, run, "mt_process", ("transfer_function",), self._on_processed,
                        f"Processing {len(runs)} run(s) at {rates}"
                        + (" with a remote reference" if self._remote_runs else "") + " …",
                        self._process_progress, self._process_stop)
        except Exception as exc:  # noqa: BLE001
            self.log(f"Could not start the processing: {exc}", "error")
            return {"status": "failed", "error": str(exc)}
        return {"status": "started", "runs": len(runs)}

    def _on_processed(self, result: WorkflowRunResult, run: Any) -> None:
        tf = result.objects.get("transfer_function")
        if tf is None:
            self.log("The processing returned no transfer function.", "error")
            return
        edi = next((a for a in result.artifacts if a.format == "edi"), None)
        source = None
        if edi is not None:
            path = Path(edi.path)
            source = path if path.is_absolute() else run.run_dir / path
        self._add_site(tf, source)
        self._go_to(self._sounding_pane)
        metrics = result.metrics
        self.log(f"{tf.station or 'Site'}: {tf.n_frequencies} frequencies, periods "
                 f"{tf.period.min():.3g}–{tf.period.max():.3g} s; median coherence "
                 f"{metrics.get('median_coherence', float('nan')):.2f}, median Zxy error "
                 f"{100 * metrics.get('median_relative_error_zxy', float('nan')):.1f} %.", "success")

    def _run_1d(self) -> Dict[str, Any]:
        site = self._current_site()
        if site is None:
            self.log("Add or process a site first.", "warn")
            return {"status": "failed", "error": "No site is selected."}
        if self._running():
            return {"status": "failed", "error": "A run is in progress."}
        try:
            run = self.begin_persisted_run("mt.invert_1d", "mt.invert_1d")
            path = self._site_file(site, run.inputs_dir, 0)
            inputs = {"transfer_function": ArtifactRef.from_path(
                path, artifact_id="mt:site", kind="mt_transfer_function", base_dir=run.run_dir)}
            parameters: Dict[str, Any] = {
                "mode": str(self._mode.currentData()),
                "n_layers": int(self._layers.value()),
                "target_rms": float(self._target_rms.value()),
                "error_floor": float(self._error_floor_1d.value()),
                "static_shift": bool(self._static_shift.isChecked()),
            }
            tem = self._tem.text().strip()
            if tem:
                copied = Path(shutil.copy2(tem, run.inputs_dir / f"tem{Path(tem).suffix.lower()}"))
                inputs["tem"] = ArtifactRef.from_path(copied, artifact_id="mt:tem", kind="tem_sounding",
                                                      base_dir=run.run_dir)
                parameters["tem_geometry"] = {"height": 0.0, "source_radius": float(self._tem_radius.value())}
                parameters["tem_inversion"] = {"rel_error": float(self._tem_error.value())}
            if self._water.isChecked():
                parameters["petrophysics"] = {"rhos": float(self._rhos.value()), "n": float(self._sat_n.value()),
                                              "porosity": float(self._porosity.value()),
                                              "sigma_sur": float(self._sigma_sur.value())}
            spec = WorkflowSpec(workflow_id="mt.invert_1d", inputs=inputs, parameters=parameters,
                                metadata={"site": site["tf"].station or ""})
            self._start(spec, run, "mt_invert_1d", ("result", "water_content"),
                        lambda result, _run: self._on_1d(result, site),
                        f"Occam 1D of {site['tf'].station or 'the site'}"
                        + (" jointly with a TEM sounding" if tem else "") + " …",
                        self._progress_1d, self._stop_1d)
        except Exception as exc:  # noqa: BLE001
            self.log(f"Could not start the 1D inversion: {exc}", "error")
            return {"status": "failed", "error": str(exc)}
        return {"status": "started", "site": site["tf"].station}

    def _on_1d(self, result: WorkflowRunResult, site: Dict[str, Any]) -> None:
        model = result.objects.get("result")
        if model is None:
            self.log("The 1D inversion returned no model.", "error")
            return
        self._result_1d = {"result": model, "water": result.objects.get("water_content"), "site": site}
        self._draw_model()
        if self._current_site() is site:
            self._draw_sounding()
        self._go_to(self._model_pane)
        shifts = ", ".join(f"{mode} {factor:.2f}" for mode, factor in model.static_shift.items())
        shown = shifts if shifts and self._static_shift.isChecked() else ""
        self._result_1d_info.setText(
            f"<b>{site['tf'].station or 'Site'}</b>: RMS {model.rms:.2f} after {model.iterations} "
            f"iterations, {model.resistivity.size} layers"
            + (f"<br>Static shift: {shown}" if shown else "")
            + ("<br>Water content converted" if self._result_1d["water"] else ""))
        self.log(f"Occam 1D: RMS {model.rms:.2f} after {model.iterations} iterations"
                 + (f"; static shift {shown}" if shown else "") + ".", "success")
        self.report_result({"mt_1d_rms": float(model.rms), "mt_1d_site": site["tf"].station,
                            "mt_static_shift": dict(model.static_shift)})
        self.offer_map_export()

    def _profile_frequencies(self, tfs: Sequence[Any]) -> List[float]:
        frequency = np.sort(np.asarray(tfs[0].frequency, dtype=float))[::-1]
        wanted = int(self._n_freq_2d.value())
        if frequency.size <= wanted:
            return [float(f) for f in frequency]
        index = np.unique(np.round(np.linspace(0, frequency.size - 1, wanted)).astype(int))
        return [float(f) for f in frequency[index]]

    def _run_2d(self) -> Dict[str, Any]:
        sites = self._checked_sites()
        if len(sites) < 2:
            self.log("Tick at least two profile sites.", "warn")
            return {"status": "failed", "error": "The profile needs two or more ticked sites."}
        if self._running():
            return {"status": "failed", "error": "A run is in progress."}
        try:
            positions, azimuth = self._site_positions(sites)
            run = self.begin_persisted_run("mt.invert_profile", "mt.invert_profile")
            refs = [ArtifactRef.from_path(self._site_file(site, run.inputs_dir, k), artifact_id=f"mt:site:{k}",
                                          kind="mt_transfer_function", base_dir=run.run_dir)
                    for k, site in enumerate(sites)]
            parameters: Dict[str, Any] = {
                "positions": [float(p) for p in positions],
                "modes": list(self._modes_2d.currentData()),
                "frequencies": self._profile_frequencies([s["tf"] for s in sites]),
                "error_floor": float(self._error_floor_2d.value()),
                "max_iterations": int(self._iterations_2d.value()),
            }
            if self._strike_auto.isChecked():
                parameters["strike"] = 0.0 if azimuth is None else (float(azimuth) + 90.0) % 180.0
            else:
                parameters["strike"] = float(self._strike.value())
            spec = WorkflowSpec(workflow_id="mt.invert_profile", inputs={"transfer_functions": refs},
                                parameters=parameters, seed=0,
                                metadata={"sites": [s["tf"].station or "" for s in sites]})
            self._start(spec, run, "mt_invert_profile", ("result",),
                        lambda result, _run: self._on_2d(result, sites),
                        f"2D inversion of {len(sites)} sites over {np.ptp(positions):.0f} m, strike "
                        f"{parameters['strike']:.0f}° …", self._progress_2d, self._stop_2d)
        except Exception as exc:  # noqa: BLE001
            self.log(f"Could not start the 2D inversion: {exc}", "error")
            return {"status": "failed", "error": str(exc)}
        return {"status": "started", "sites": len(sites)}

    def _on_2d(self, result: WorkflowRunResult, sites: List[Dict[str, Any]]) -> None:
        section = result.objects.get("result")
        if section is None:
            self.log("The 2D inversion returned no section.", "error")
            return
        self._result_2d, self._profile_sites = section, list(sites)
        self._draw_section()
        self._go_to(self._section_pane)
        strike = (result.summary or {}).get("strike_deg")
        self._result_2d_info.setText(
            f"<b>{len(sites)} sites</b>, {' + '.join(m.upper() for m in section.modes)}"
            + (f", strike {float(strike):.0f}°" if strike is not None else "")
            + f": RMS {section.rms:.2f} after {len(section.history)} iterations, "
              f"{section.resistivity.size} cells")
        self.log(f"2D inversion: RMS {section.rms:.2f} after {len(section.history)} iterations, "
                 f"{section.resistivity.size} cells.", "success")
        # Sites that gave the profile fewer values than it asks for, by name.
        for warning in result.warnings:
            self.log(str(warning), "warn")
        self.report_result({"mt_2d_rms": float(section.rms), "mt_2d_sites": len(sites)})
        self.offer_map_export()

    # -- export and map ------------------------------------------------------
    def _export_folder(self) -> Optional[Path]:
        folder = select_directory(self, "Export MT results to folder", self.state.output_dir or Path.cwd())
        return io_utils.ensure_dir(folder) if folder else None

    def _write_site(self, site: Dict[str, Any], out: Path) -> List[str]:
        name = _safe_name(site["tf"].station)
        return [Path(mt.write_edi(site["tf"], out / f"{name}.edi")).name,
                Path(mt.write_emtf_xml(site["tf"], out / f"{name}.xml")).name]

    def _copy_outputs(self, workflow_id: str, out: Path) -> List[str]:
        outputs = self._outputs.get(workflow_id)
        if outputs is None or not Path(outputs).is_dir():
            return []
        return [Path(shutil.copy2(path, out / path.name)).name for path in Path(outputs).iterdir()
                if path.is_file() and path.suffix.lower() in (".csv", ".npz")]

    def _export_site(self) -> None:
        site = self._current_site()
        if site is None:
            self.log("Add or process a site first.", "warn")
            return
        out = self._export_folder()
        if out is None:
            return
        try:
            self.log(f"Exported {', '.join(self._write_site(site, out))} to {out}", "success")
        except Exception as exc:  # noqa: BLE001
            self.log(f"MT export failed: {exc}", "error")

    def _export_outputs(self, workflow_id: str) -> None:
        """A 1D or 2D run's tables, beside the site files they came from."""
        if workflow_id not in self._outputs:
            self.log("Run the inversion first.", "warn")
            return
        out = self._export_folder()
        if out is None:
            return
        try:
            written = self._copy_outputs(workflow_id, out)
            sites = ([self._result_1d["site"]] if workflow_id == "mt.invert_1d" and self._result_1d
                     else self._profile_sites if workflow_id == "mt.invert_profile" else [])
            for site in sites:
                written += self._write_site(site, out)
            self.log(f"Exported {', '.join(written)} to {out}", "success")
        except Exception as exc:  # noqa: BLE001
            self.log(f"MT export failed: {exc}", "error")

    def _export(self) -> None:
        """Everything: the selected site and every inversion's tables."""
        site = self._current_site()
        if site is None and not self._outputs:
            self.log("Process a recording or add a site first.", "warn")
            return
        out = self._export_folder()
        if out is None:
            return
        try:
            written = self._write_site(site, out) if site is not None else []
            for workflow_id in ("mt.invert_1d", "mt.invert_profile"):
                written += self._copy_outputs(workflow_id, out)
            self.log(f"Exported {', '.join(written)} to {out}", "success")
        except Exception as exc:  # noqa: BLE001
            self.log(f"MT export failed: {exc}", "error")

    def export_actions(self):
        actions = []
        if self._sites or self._outputs:
            actions.append(("Sites and models (EDI, XML, CSV)", self._export))
        if self._result_1d is not None or self._result_2d is not None:
            actions.append(("Add model to Project Map…", self.add_to_map))
        return actions

    def map_snapshot(self):
        """The model on screen: the 2D section on the profile tabs, the 1D model on its own.

        Elsewhere, whichever exists, the section first.
        """
        from PyHydroGeophysX.qt_apps.project_map import em_snapshot

        panel = self._panel_of.get(self._tabs.currentWidget(), "")
        use_2d = self._result_2d is not None and (panel != "1d" or self._result_1d is None)
        if use_2d:
            section = self._result_2d
            x_nodes, z_nodes, grid = section.section()
            stations = np.asarray(section.profile.station_x, dtype=float)
            centres = 0.5 * (x_nodes[:-1] + x_nodes[1:])
            keep = (centres >= stations.min()) & (centres <= stations.max())
            payload = {"model3d": grid[:, keep].T[:, None, :], "positions": centres[keep],
                       "depth_edges": -np.asarray(z_nodes, dtype=float)[::-1], "method": "MT 2D"}
            coords = [(s["tf"].latitude, s["tf"].longitude) for s in self._profile_sites]
            if len(coords) == stations.size and all(_finite(*c) for c in coords):
                lat, lon = np.asarray(coords, dtype=float).T
                order = np.argsort(stations)
                payload["latitude"] = np.interp(centres[keep], stations[order], lat[order])
                payload["longitude"] = np.interp(centres[keep], stations[order], lon[order])
            return em_snapshot(payload)
        if self._result_1d is not None:
            model = self._result_1d["result"]
            tops = np.asarray(model.depth, dtype=float)
            bottom = tops[-1] + max(tops[-1] - tops[-2], 1.0) if tops.size > 1 else 10.0
            tf = self._result_1d["site"]["tf"]
            payload = {"model3d": np.asarray(model.resistivity, dtype=float)[::-1][None, None, :],
                       "positions": [0.0], "depth_edges": np.r_[tops, bottom], "method": "MT 1D"}
            if _finite(tf.latitude, tf.longitude):
                payload["latitude"], payload["longitude"] = [tf.latitude], [tf.longitude]
            return em_snapshot(payload)
        raise ValueError("Run a 1D or 2D MT inversion first.")

    # -- publish -------------------------------------------------------------
    def _publish(self) -> None:
        self.report_result({
            "recording": str(self._recording_path or ""),
            "runs": len(self._runs),
            "sites": [site["tf"].station for site in self._sites],
        })

    # -- automatic workflow display -----------------------------------------
    def show_run_inputs(self, inputs: Dict[str, Any]) -> str:
        """Add the run's MT sites to the list, for display; nothing is computed here."""
        values = inputs.get("mt_files") or inputs.get("mt_file") or []
        paths = [Path(str(p)) for p in (values if isinstance(values, (list, tuple)) else [values]) if p]
        added = [p.name for p in paths if p.is_file() and mt.is_transfer_function_file(p)
                 and self._add_site_file(p)]
        return f"MT sites {', '.join(added)}" if added else ""

    def show_run_stage(self, tool: str) -> str:
        if tool == "load_mt_sites":
            self._go_to(self._sounding_pane)
            return "Sounding"
        if tool in ("invert_mt", "evaluate_mt_inversion"):
            self._go_to(self._model_pane)
            return "1D model"
        if tool == "convert_mt_water_content":
            self._go_to(self._model_pane)
            return "1D model"
        return ""

    # -- AQUAH agent interface ----------------------------------------------
    def agent_describe(self) -> Dict[str, Any]:
        return {
            "module": self.module_key,
            "title": self.module_title,
            "state": self._agent_status(),
            "actions": [
                {"name": "use_example_data", "args": {},
                 "desc": "Add the bundled example site NMX20 (USMTArray, EMTF XML)."},
                {"name": "load_recording", "args": {"path": "str", "remote": "str (optional)"},
                 "desc": ("Read an instrument recording folder or file (Phoenix, legacy Phoenix, "
                          "Metronix ATS, Zonge Z3D, LEMI-424), and optionally a remote reference.")},
                {"name": "add_sites", "args": {"paths": "list[str]"},
                 "desc": "Add transfer-function files (EDI, EMTF XML, Z- or J-files) as sites."},
                {"name": "select_site", "args": {"site": "index or station name"},
                 "desc": "Select the site that is drawn and inverted in 1D."},
                {"name": "set_profile_sites", "args": {"sites": "list of indices or station names"},
                 "desc": "Tick exactly these sites for the 2D profile (the others are unticked)."},
                {"name": "show_view", "args": {"view": ["time_series", "sounding", "dimensionality",
                                                        "1d_model", "phase_tensors", "2d_section"]},
                 "desc": "Bring up a view, with the side panel for its step."},
                {"name": "set_params", "args": {"params": {"<key>": "value"}},
                 "desc": ("Processing: window_length, overlap, decimation_factor, n_levels, huber, "
                          "redescend, min_points, prewhiten, tipper, allow_uncalibrated. 1D: mode "
                          "(det/xy/yx/both), n_layers, target_rms, error_floor, static_shift, "
                          "tem_file, tem_loop_radius, tem_error, water_content, rhos, n, porosity, "
                          "sigma_sur. 2D: modes (te/tm/both), strike (number or 'auto'), "
                          "site_spacing, error_floor_2d, max_iterations, n_frequencies.")},
                {"name": "process", "args": {},
                 "desc": "Estimate the impedance and tipper of the loaded recording."},
                {"name": "run_1d", "args": {},
                 "desc": "Occam 1D inversion of the selected site (with TEM and water content when set)."},
                {"name": "run_2d", "args": {},
                 "desc": "2D TE/TM inversion of the ticked profile sites on SimPEG."},
                {"name": "get_status", "args": {}, "desc": "Report the recording, sites and results."},
            ],
        }

    def agent_apply(self, action: str, args: Dict[str, Any]) -> Dict[str, Any]:
        args = args or {}
        handlers = {
            "use_example_data": self._load_example,
            "load_recording": lambda: self._agent_load_recording(args.get("path"), args.get("remote")),
            "add_sites": lambda: self._agent_add_sites(args.get("paths") or args.get("path")),
            "select_site": lambda: self._agent_select_site(args.get("site")),
            "set_profile_sites": lambda: self._agent_set_profile_sites(args.get("sites")),
            "show_view": lambda: self._agent_show_view(args.get("view")),
            "set_params": lambda: self._agent_set_params(args.get("params", args)),
            "process": self._run_processing,
            "run_1d": self._run_1d,
            "run_2d": self._run_2d,
            "get_status": self._agent_status,
        }
        handler = handlers.get(action)
        if handler is None:
            return {"status": "failed", "error": f"Unknown action '{action}'.",
                    "valid_actions": list(handlers.keys())}
        return handler()

    def _agent_status(self) -> Dict[str, Any]:
        ticked = {id(site) for site in self._checked_sites()}
        status: Dict[str, Any] = {
            "status": "ok",
            "view": self._tabs.tabText(self._tabs.currentIndex()),
            "recording": str(self._recording_path or ""),
            "runs": len(self._runs),
            "sample_rates": sorted({float(r.sample_rate) for r in self._runs}, reverse=True),
            "remote_runs": len(self._remote_runs),
            "sites": [{"station": s["tf"].station, "periods": int(s["tf"].n_frequencies),
                       "in_profile": id(s) in ticked} for s in self._sites],
            "selected_site": self._site_list.currentRow(),
            "running": self._running(),
        }
        if self._result_1d is not None:
            model = self._result_1d["result"]
            status["result_1d"] = {"rms": float(model.rms), "iterations": int(model.iterations),
                                   "static_shift": dict(model.static_shift),
                                   "site": self._result_1d["site"]["tf"].station}
        if self._result_2d is not None:
            status["result_2d"] = {"rms": float(self._result_2d.rms), "sites": len(self._profile_sites)}
        return status

    def _agent_load_recording(self, path: Any, remote: Any) -> Dict[str, Any]:
        if not path:
            return {"status": "failed", "error": "Provide 'path' to a recording folder or file."}
        result: Dict[str, Any] = {"status": "ok"}
        for source, is_remote in ((path, False), (remote, True)):
            if not source:
                continue
            source = Path(str(source))
            if not source.exists():
                return {"status": "failed", "error": f"Not found: {source}"}
            try:
                runs = mt.read_timeseries(source, **self._reader_options(source))
            except Exception as exc:  # noqa: BLE001
                return {"status": "failed", "error": f"Could not read {source.name}: {exc}"}
            self._on_recording_read(source, runs, is_remote)
            result["remote_runs" if is_remote else "runs"] = len(runs)
        result["sample_rates"] = sorted({float(r.sample_rate) for r in self._runs}, reverse=True)
        return result

    def _agent_add_sites(self, paths: Any) -> Dict[str, Any]:
        items = paths if isinstance(paths, (list, tuple)) else [paths]
        added, failed = [], []
        for item in items:
            if not item:
                continue
            path = Path(str(item))
            sources = [path]
            if path.is_dir():
                sources = sorted({p for pattern in mt.TRANSFER_FUNCTION_PATTERNS
                                  for p in path.glob(pattern) if mt.is_transfer_function_file(p)})
            for source in sources:
                (added if self._add_site_file(source) else failed).append(source.name)
        if not added:
            return {"status": "failed", "error": "No site could be read.", "failed": failed}
        return {"status": "ok", "added": added, "failed": failed, "sites": len(self._sites)}

    def _site_index(self, site: Any) -> Optional[int]:
        if isinstance(site, (int, float)) or (isinstance(site, str) and site.isdigit()):
            index = int(site)
        else:
            names = [s["tf"].station for s in self._sites]
            index = names.index(site) if site in names else None
        return index if index is not None and 0 <= index < len(self._sites) else None

    def _agent_select_site(self, site: Any) -> Dict[str, Any]:
        index = self._site_index(site)
        if index is None:
            return {"status": "failed", "error": f"No site {site!r}."}
        self._select_site(index)
        return {"status": "ok", "site": self._sites[index]["tf"].station}

    def _agent_set_profile_sites(self, sites: Any) -> Dict[str, Any]:
        wanted = [self._site_index(s) for s in (sites if isinstance(sites, (list, tuple)) else [sites])]
        if not wanted or any(index is None for index in wanted):
            return {"status": "failed", "error": f"Unknown sites in {sites!r}."}
        self._profile_list.blockSignals(True)
        for k in range(self._profile_list.count()):
            self._profile_list.item(k).setCheckState(Qt.Checked if k in wanted else Qt.Unchecked)
        self._profile_list.blockSignals(False)
        self._on_profile_changed()
        return {"status": "ok", "profile": [self._sites[k]["tf"].station for k in sorted(set(wanted))]}

    def _agent_show_view(self, view: Any) -> Dict[str, Any]:
        panes = {"time_series": self._series_pane, "sounding": self._sounding_pane,
                 "dimensionality": self._dimension_pane, "1d_model": self._model_pane,
                 "phase_tensors": self._profile_pane, "2d_section": self._section_pane}
        pane = panes.get(str(view or "").lower())
        if pane is None:
            return {"status": "failed", "error": f"view must be one of {sorted(panes)}."}
        self._go_to(pane)
        return {"status": "ok", "view": self._tabs.tabText(self._tabs.currentIndex())}

    def _agent_set_params(self, params: Any) -> Dict[str, Any]:
        if not isinstance(params, dict):
            return {"status": "failed", "error": "Provide 'params' as a JSON object."}

        def combo(widget: QComboBox, value: Any) -> None:
            wanted = str(value).lower()
            for k in range(widget.count()):
                data = widget.itemData(k)
                flat = "+".join(data) if isinstance(data, list) else str(data)
                if wanted in (flat.lower(), widget.itemText(k).lower()) or (wanted == "both" and flat == "te+tm"):
                    widget.setCurrentIndex(k)
                    return
            raise ValueError(f"unknown choice {value!r}")

        def strike(value: Any) -> None:
            if str(value).lower() == "auto":
                self._strike_auto.setChecked(True)
            else:
                self._strike_auto.setChecked(False)
                self._strike.setValue(float(value))

        handlers = {
            "window_length": lambda v: self._window.setValue(int(v)),
            "overlap": lambda v: self._overlap.setValue(int(v)),
            "decimation_factor": lambda v: self._decimation.setValue(int(v)),
            "n_levels": lambda v: self._levels.setValue(int(v or 0)),
            "huber": lambda v: self._huber.setValue(float(v)),
            "redescend": lambda v: self._redescend.setValue(float(v)),
            "min_points": lambda v: self._min_points.setValue(int(v)),
            "prewhiten": lambda v: self._prewhiten.setChecked(bool(v)),
            "tipper": lambda v: self._tipper.setChecked(bool(v)),
            "allow_uncalibrated": lambda v: self._uncalibrated.setChecked(bool(v)),
            "band_setup": lambda v: self._band_setup.setText(str(v or "")),
            "calibration": lambda v: self._calibration.setText(str(v or "")),
            "mode": lambda v: combo(self._mode, v),
            "n_layers": lambda v: self._layers.setValue(int(v)),
            "target_rms": lambda v: self._target_rms.setValue(float(v)),
            "error_floor": lambda v: self._error_floor_1d.setValue(float(v)),
            "static_shift": lambda v: self._static_shift.setChecked(bool(v)),
            "tem_file": lambda v: self._tem.setText(str(v or "")),
            "tem_loop_radius": lambda v: self._tem_radius.setValue(float(v)),
            "tem_error": lambda v: self._tem_error.setValue(float(v)),
            "water_content": lambda v: self._water.setChecked(bool(v)),
            "rhos": lambda v: self._rhos.setValue(float(v)),
            "n": lambda v: self._sat_n.setValue(float(v)),
            "porosity": lambda v: self._porosity.setValue(float(v)),
            "sigma_sur": lambda v: self._sigma_sur.setValue(float(v)),
            "modes": lambda v: combo(self._modes_2d, "+".join(v) if isinstance(v, list) else v),
            "strike": strike,
            "site_spacing": lambda v: self._spacing.setValue(float(v)),
            "error_floor_2d": lambda v: self._error_floor_2d.setValue(float(v)),
            "max_iterations": lambda v: self._iterations_2d.setValue(int(v)),
            "n_frequencies": lambda v: self._n_freq_2d.setValue(int(v)),
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

    def _show_format_help(self) -> None:
        doc_path = Path(__file__).with_name("mt_input_format.md")
        try:
            text = doc_path.read_text(encoding="utf-8")
        except Exception:  # noqa: BLE001
            text = "Open a recording folder (Phoenix, Metronix, Zonge, LEMI) or add EDI / EMTF XML sites."
        dlg = QDialog(self); dlg.setWindowTitle("Magnetotelluric input formats")
        dlg.resize(760, 620); lay = QVBoxLayout(dlg)
        browser = QTextBrowser(); browser.setOpenExternalLinks(True)
        try:
            browser.setMarkdown(text)
        except Exception:  # noqa: BLE001
            browser.setPlainText(text)
        lay.addWidget(browser)
        close = QPushButton("Close"); close.clicked.connect(dlg.accept); lay.addWidget(close)
        dlg.exec()
