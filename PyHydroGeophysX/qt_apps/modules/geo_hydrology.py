"""ERT → Water Content: water content and porosity from an inverted ERT model.

The inverse counterpart of Hydro → Geophysics, laid out the same way and like
the ERT page. The views are in the middle: the inverted **Model** on its mesh
(one survey of a time-lapse series at a time, poorly covered cells blanked),
the **Layers** the rock physics is set for, and, after a run, the **Results**
maps and the water-content **Time series** at two points. The settings are on
the right in the order the work is done - the inverted model, the layers to
estimate in, each layer's rock physics, what to estimate - with the Estimate
button pinned under them and what still stops it said above it. Before a model
is loaded the middle says what the page does and how to start, and afterwards
a line above the views says what to do next.

The estimate is a Monte Carlo over per-layer petrophysics (Waxman-Smits /
Archie), so every map comes with its spread. The numerics are reused from
``PyHydroGeophysX.Geophy_modular.ERT_to_WC``, run in a worker process.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from PyHydroGeophysX.data_processing import run_inputs
from PyHydroGeophysX.Geophy_modular import ERT_to_WC as geo_pipeline
from PyHydroGeophysX.qt_apps import io_utils, theme
from PyHydroGeophysX.qt_apps.modules.base import BaseModule, LogFn
from PyHydroGeophysX.qt_apps.qt_utils import (
    BusyStateController,
    ContentWidthScrollArea,
    ReproduceBar,
    make_spinbox,
    merged_row,
    select_directory,
    set_rows_visible,
)
from PyHydroGeophysX.qt_apps.widgets import colormaps as cmaps
from PyHydroGeophysX.qt_apps.widgets import length_units
from PyHydroGeophysX.qt_apps.widgets.guided import (
    NextStepStrip,
    fold,
    hint as _hint,
    number_box as _number,
    start_card,
)
from PyHydroGeophysX.qt_apps.widgets.image_view import ZoomableImageView
from PyHydroGeophysX.qt_apps.widgets.mesh_view import MeshResultView
from PyHydroGeophysX.qt_apps.widgets.readout import toolbar_row
from PyHydroGeophysX.qt_apps.widgets.run_controls import progress_with_stop
from PyHydroGeophysX.qt_apps.workers import ProcessWorkflowWorker
from PyHydroGeophysX.visualization.axis_units import set_section_axes
from PyHydroGeophysX.workflows import (
    ArtifactRef,
    WorkflowRunResult,
    WorkflowSpec,
    export_workflow_bundle,
)

_PRODUCTS = [("water_content", "Water content"), ("porosity", "Porosity (where saturated)")]
#: The steps the assistant names. Each opens the view and the panel the step is
#: done in (see _go_to); the page itself no longer pages through them.
_STEPS = ["Data", "Layers", "Target", "Parameters", "Run", "Results"]
_MODEL_TAB, _LAYERS_TAB, _RESULTS_TAB, _SERIES_TAB = range(4)
#: The "Use example" model bundle, under the source repository's examples/: the
#: Treeline time-lapse ERT model of Ex_MC_Hydro, whose two layers the default
#: rock physics (ERT_to_WC.DEFAULT_LAYER_DISTRIBUTIONS) was set for.
_EXAMPLE_MODEL = "results/Structure_WC"
#: The coverage (log10) under which cells are blanked, on the view and in the maps.
_DEFAULT_COVERAGE_CUT = -1.0

# Per-parameter box configuration: (lo, hi, step, decimals) for mean and spread.
_PARAM_SPECS = {
    "m": ((1.0, 3.0, 0.1, 2), (0.0, 1.0, 0.05, 3)),
    "rho_fluid": ((0.1, 1000.0, 1.0, 2), (0.0, 500.0, 1.0, 2)),
    "n": ((1.0, 3.0, 0.1, 2), (0.0, 1.0, 0.05, 3)),
    "sigma_sur": ((0.0, 1.0, 0.001, 4), (0.0, 1.0, 0.001, 4)),
    "porosity": ((0.01, 0.9, 0.01, 3), (0.0, 0.5, 0.01, 3)),
}
_PARAM_LABELS = {
    "m": "Cementation exponent m",
    "rho_fluid": "Pore-water resistivity (Ω·m)",
    "n": "Saturation exponent n",
    "sigma_sur": "Surface conductivity (S/m)",
    "porosity": "Porosity",
}

_GENERIC_LAYER = {
    "m": {"mean": 1.8, "std": 0.2},
    "rho_fluid": {"mean": 20.0, "std": 0.0},
    "n": {"mean": 2.0, "std": 0.1},
    "sigma_sur": {"mean": 0.0, "std": 0.0},
    "porosity": {"mean": 0.30, "std": 0.05},
}
#: Fill of the layers on the Layers view, in marker order; a layer that is not
#: estimated is drawn in the last, neutral one.
_LAYER_FILLS = ["#d4a373", "#8d99ae", "#90be6d", "#577590", "#f4a261", "#b5838d"]
_NOT_ESTIMATED = "#e5e5ea"

_INPUT_FORMAT_FALLBACK = (
    "# ERT → Water Content input format\n\n"
    "Place an inverted-model bundle in one folder:\n\n"
    "- `mesh_res.bms` - pyGIMLi inversion mesh\n"
    "- `resmodel.npy` - inverted resistivity (n_cells, n_time)\n"
    "- `index_marker.npy` - layer id per cell (n_cells)\n"
    "- `all_coverage.npy` - coverage mask (n_time, n_cells), optional\n\n"
    "See geo_input_format.md for details."
)


def _clear(layout) -> None:
    """Empty ``layout``, hiding each widget at once: deleteLater alone left the
    old rows drawn under the new ones until the event loop came round."""
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        if widget is not None:
            widget.hide()
            widget.deleteLater()


class _LayerMapView(QWidget):
    """The model's cells coloured by layer, the estimated ones named in the legend."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._figure = Figure(figsize=(8, 4), layout="constrained")
        self._canvas = FigureCanvasQTAgg(self._figure)
        bar, self._toolbar = toolbar_row(self._canvas, self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(bar)
        layout.addWidget(self._canvas, stretch=1)
        self._polygons: Optional[List[np.ndarray]] = None
        self._markers: Optional[np.ndarray] = None
        self._names: Dict[int, str] = {}
        self._estimated: set = set()
        self._order: List[int] = []
        self._message = ""
        length_units.notifier().changed.connect(lambda _unit: self.draw())

    def set_mesh(self, polygons: Optional[List[np.ndarray]], message: str = "") -> None:
        self._polygons, self._message = polygons, message
        self.draw()

    def set_layers(self, markers: Optional[np.ndarray], names: Dict[int, str],
                   estimated: set, order: List[int]) -> None:
        """``order`` lists the markers from the top down; they are filled in that order."""
        self._markers, self._names, self._estimated = markers, dict(names), set(estimated)
        self._order = list(order)
        self.draw()

    def draw(self) -> None:
        self._figure.clear()
        ax = self._figure.add_subplot(111)
        if (self._polygons is None or self._markers is None
                or len(self._polygons) != len(self._markers)):
            ax.set_axis_off()
            ax.text(0.5, 0.5, self._message or "Load an inverted ERT model to see its layers.",
                    ha="center", va="center", transform=ax.transAxes, color="#6e6e73")
        else:
            present = np.unique(self._markers).tolist()
            order = [m for m in self._order if m in present] + [
                m for m in sorted(present) if m not in self._order]
            fills = {m: _LAYER_FILLS[i % len(_LAYER_FILLS)] for i, m in enumerate(order)}
            colours = [fills[int(m)] if int(m) in self._estimated else _NOT_ESTIMATED
                       for m in self._markers]
            ax.add_collection(PolyCollection(self._polygons, facecolors=colours,
                                             edgecolors="none"))
            ax.autoscale_view()
            ax.set_aspect("equal", adjustable="datalim")
            handles = [Patch(facecolor=fills[m], label=f"{self._names.get(m, m)} (marker {m})")
                       for m in order if m in self._estimated]
            if len(handles) < len(order):
                handles.append(Patch(facecolor=_NOT_ESTIMATED, label="Not estimated"))
            ax.legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.9)
            z = np.concatenate([p[:, 1] for p in self._polygons])
            set_section_axes(ax, z=z, unit=length_units.current())
            ax.set_title("Layers the rock physics is set for")
        self._canvas.draw_idle()
        self._toolbar.update()


class GeoHydrologyModule(BaseModule):
    module_key = "geo_hydrology"
    module_title = "ERT → Water Content"

    def __init__(self, state: Any, log: LogFn, parent=None) -> None:
        super().__init__(state, log, parent)
        self._manual_dir: Optional[Path] = None
        self._summary: Optional[Dict[str, Any]] = None
        #: The model drawn on the Model and Layers views: its pyGIMLi mesh, the
        #: resistivity (n_cells, n_time), markers, coverage and cell outlines.
        self._mesh = None
        self._resistivity: Optional[np.ndarray] = None
        self._markers: Optional[np.ndarray] = None
        self._coverage: Optional[np.ndarray] = None
        self._layer_rows: List[Dict[str, Any]] = []
        self._param_widgets: Dict[int, Dict[str, Tuple[QDoubleSpinBox, QDoubleSpinBox]]] = {}
        #: Per marker, the (mean, std) last set in a panel; kept for a layer
        #: switched off, so switching it back on restores them.
        self._param_values: Dict[int, Dict[str, Tuple[float, float]]] = {}
        self._param_layout: Optional[List[int]] = None
        self._param_summaries: Dict[int, QLabel] = {}
        self._param_titles: Dict[int, QLabel] = {}
        self._last_result: Optional[Dict[str, Any]] = None
        self._has_results = False
        self._markers_override: Optional[List[int]] = None
        self._worker: Optional[ProcessWorkflowWorker] = None
        self._run_busy: Optional[BusyStateController] = None
        self._workflow_recipe_path = ""
        #: Why the last run did not finish - its error, or that it was
        #: stopped - for the assistant, whose status otherwise still showed
        #: the result before it; "" while a run is under way or succeeded.
        self._run_problem = ""
        self._current = 0

        root = QHBoxLayout(self)
        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_model_tab(), "Model")
        self._layer_view = _LayerMapView()
        self._tabs.addTab(self._layer_view, "Layers")
        self._tabs.addTab(self._build_results_tab(), "Results")
        self._tabs.addTab(self._build_series_tab(), "Time series")
        self._tabs.currentChanged.connect(self._on_tab_changed)
        self._guide = NextStepStrip()
        self._reproduce = ReproduceBar()
        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.addWidget(self._guide)
        center_layout.addWidget(self._tabs, stretch=1)
        center_layout.addWidget(self._reproduce)
        root.addWidget(center, stretch=1)
        root.addWidget(self._build_controls())
        self._reload()

    # -- centre: the model, its layers, the results ---------------------------
    def _example_button(self) -> QPushButton:
        """"Use example", or "Use project data" when the studio was opened on a project's."""
        context = self._context_dir() is not None
        button = QPushButton("Use project data" if context else "Use example")
        button.setIcon(theme.icon("fa5s.box-open"))
        button.setToolTip(
            "Load this project's inverted ERT model." if context else
            "Load the Treeline time-lapse ERT model of the Ex_MC_Hydro example: "
            "12 surveys over regolith and bedrock.")
        button.clicked.connect(self._use_context_data)
        return button

    def _build_start_card(self) -> QWidget:
        example_btn = self._example_button()
        example_btn.setProperty("primary", True)
        example_btn.setIcon(theme.icon("fa5s.box-open", color="#ffffff"))
        open_btn = QPushButton("Open folder…")
        open_btn.setIcon(theme.icon("fa5s.folder-open"))
        open_btn.clicked.connect(self._select_folder)
        format_btn = QPushButton("Data format")
        format_btn.setIcon(theme.icon("fa5s.question-circle"))
        format_btn.clicked.connect(self._show_format_help)
        return start_card(
            "Estimate water content from an ERT model",
            "Turn an inverted, time-lapse ERT resistivity model into water content, "
            "with its uncertainty, through rock physics sampled many times over.",
            ["Load an inverted ERT model: its mesh, resistivity and layers.",
             "Tick the layers to estimate in, and check each one's rock physics.",
             "Press Estimate water content."],
            [example_btn, open_btn, format_btn],
            note=("The example is the Treeline time-lapse ERT model of the Ex_MC_Hydro "
                  "example." if self._context_dir() is None else ""))

    def _build_model_tab(self) -> QWidget:
        # The start card until a model is loaded, then the model.
        self._model_stack = QStackedWidget()
        self._model_stack.addWidget(self._build_start_card())
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        self._survey = QSpinBox()
        self._survey.setRange(0, 0)
        self._survey.setToolTip("The survey of the time-lapse series shown here and in "
                                "the result maps. Every survey is estimated.")
        self._survey.valueChanged.connect(self._show_model)
        self._survey_range = QLabel("")
        row.addWidget(QLabel("Survey"))
        row.addWidget(self._survey)
        row.addWidget(self._survey_range)
        row.addStretch(1)
        layout.addLayout(row)
        self._model_view = MeshResultView(colormaps=cmaps.colormap_settings(self.state))
        self._model_view.set_hide_below(_DEFAULT_COVERAGE_CUT)
        layout.addWidget(self._model_view, stretch=1)
        self._model_note = _hint()
        layout.addWidget(self._model_note)
        self._model_stack.addWidget(page)
        return self._model_stack

    def _build_results_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        header = QHBoxLayout()
        header.addWidget(QLabel("Show"))
        self._result_figure = QComboBox()
        self._result_figure.currentIndexChanged.connect(self._show_selected_result_map)
        header.addWidget(self._result_figure, stretch=1)
        open_btn = QPushButton("Open output folder")
        open_btn.setIcon(theme.icon("fa5s.folder-open"))
        open_btn.clicked.connect(self._open_output_folder)
        header.addWidget(open_btn)
        layout.addLayout(header)
        self._result_summary = QLabel("")
        self._result_summary.setWordWrap(True)
        layout.addWidget(self._result_summary)
        self._result_map_view = ZoomableImageView()
        self._result_map_view.setMinimumHeight(420)
        layout.addWidget(self._result_map_view, stretch=1)
        self._result_files = _hint()
        layout.addWidget(self._result_files)
        return page

    def _build_series_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(_hint("Water content through the surveys at two points, with "
                               "the ±1σ band the rock-physics uncertainty gives it."))
        self._mx1 = _number(30.0, -1e6, 1e6, 1.0, 1, " m")
        self._my1 = _number(1610.0, -1e6, 1e6, 1.0, 1, " m")
        self._mx2 = _number(65.0, -1e6, 1e6, 1.0, 1, " m")
        self._my2 = _number(1613.0, -1e6, 1e6, 1.0, 1, " m")
        plot_btn = QPushButton("Plot")
        plot_btn.setIcon(theme.icon("fa5s.chart-line"))
        plot_btn.clicked.connect(self._plot_points)
        row = QHBoxLayout()
        row.addWidget(merged_row("Point 1: x", self._mx1, "z", self._my1))
        row.addWidget(merged_row("Point 2: x", self._mx2, "z", self._my2))
        row.addWidget(plot_btn)
        row.addStretch(1)
        layout.addLayout(row)
        self._monitor_view = ZoomableImageView()
        self._monitor_view.setMinimumHeight(380)
        layout.addWidget(self._monitor_view, stretch=1)
        return page

    # -- right: the settings, in the order the work is done ------------------
    def _build_controls(self) -> QWidget:
        scroll = ContentWidthScrollArea(minimum=430, maximum=480)
        panel = QWidget()
        scroll.setWidget(panel)
        layout = QVBoxLayout(panel)
        self._data_group = self._build_data_group()
        self._layers_group = self._build_layers_group()
        self._physics_group = self._build_physics_group()
        self._estimate_group = self._build_estimate_group()
        #: Each numbered step's title; a finished step gets a tick after it.
        self._step_titles = {
            self._data_group: "1  Inverted ERT model",
            self._layers_group: "2  Layers",
            self._physics_group: "3  Rock physics",
            self._estimate_group: "4  Estimate",
        }
        for group in self._step_titles:
            layout.addWidget(group)
        layout.addStretch(1)
        self._controls_scroll = scroll

        # -- run: pinned under the column, as on the ERT page ---------------
        footer = QWidget()
        flayout = QVBoxLayout(footer)
        flayout.setContentsMargins(6, 4, 6, 6)
        flayout.setSpacing(6)
        rule = QFrame()
        rule.setFrameShape(QFrame.HLine)
        rule.setFrameShadow(QFrame.Plain)
        flayout.addWidget(rule)
        self._run_reason = _hint()
        flayout.addWidget(self._run_reason)
        self._run_btn = QPushButton("Estimate water content")
        self._run_btn.setProperty("primary", True)
        self._run_btn.setIcon(theme.icon("fa5s.play", color="#ffffff"))
        self._run_btn.setToolTip(
            "Sample each layer's rock physics many times, turn every survey's "
            "resistivity into water content with each sample, and map the mean and "
            "the spread.")
        self._run_btn.clicked.connect(self._run_inverse)
        flayout.addWidget(self._run_btn)
        self._progress = QProgressBar()
        self._progress.setVisible(False)
        self._stop = self.stop_button("The water-content estimation")
        flayout.addWidget(progress_with_stop(self._progress, self._stop))
        self._run_status = QLabel("")
        self._run_status.setWordWrap(True)
        flayout.addWidget(self._run_status)

        column = QWidget()
        column_layout = QVBoxLayout(column)
        column_layout.setContentsMargins(0, 0, 0, 0)
        column_layout.setSpacing(0)
        column_layout.addWidget(scroll, 1)
        column_layout.addWidget(footer)
        return column

    def _build_data_group(self) -> QGroupBox:
        box = QGroupBox("1  Inverted ERT model")
        form = QFormLayout(box)
        open_btn = QPushButton("Open folder…")
        open_btn.setProperty("primary", True)
        open_btn.setIcon(theme.icon("fa5s.folder-open", color="#ffffff"))
        open_btn.setToolTip("Open an ERT inversion's output folder, or a folder with "
                            "mesh_res.bms, resmodel.npy and index_marker.npy "
                            "(all_coverage.npy is optional).")
        open_btn.clicked.connect(self._select_folder)
        reload_btn = QPushButton()
        reload_btn.setIcon(theme.icon("fa5s.sync"))
        reload_btn.setToolTip("Read the folder again, after its files changed.")
        reload_btn.clicked.connect(self._reload)
        help_btn = QPushButton()
        help_btn.setIcon(theme.icon("fa5s.question-circle"))
        help_btn.setToolTip("What the folder must contain.")
        help_btn.clicked.connect(self._show_format_help)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        for button in (open_btn, self._example_button(), reload_btn, help_btn):
            row.addWidget(button)
        row.addStretch(1)
        form.addRow(row)
        self._data_status = QLabel("")
        self._data_status.setWordWrap(True)
        form.addRow(self._data_status)
        return box

    def _build_layers_group(self) -> QGroupBox:
        box = QGroupBox("2  Layers")
        layout = QVBoxLayout(box)
        layout.addWidget(_hint("Tick the layers to estimate water content in, and name "
                               "them as you like. The Layers tab shows where they are."))
        self._layers_host = QWidget()
        self._layers_form = QGridLayout(self._layers_host)
        self._layers_form.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._layers_host)
        adv, aform = fold()
        struct_btn = QPushButton("Layers from a seismic bedrock interface…")
        struct_btn.setIcon(theme.icon("fa5s.mountain"))
        struct_btn.setToolTip(
            "Split the model into regolith (above) and bedrock (below) along a bedrock "
            "interface: the one sent from Seismic → Structure, or one read from a file.")
        struct_btn.clicked.connect(self._derive_markers_from_structure)
        reset_btn = QPushButton("Back to the layers in the file")
        reset_btn.setIcon(theme.icon("fa5s.undo"))
        reset_btn.setToolTip("Use the model's own index_marker.npy again, with its "
                             "layers ticked and named as they start.")
        reset_btn.clicked.connect(self._reset_layers)
        aform.addRow(merged_row(struct_btn))
        aform.addRow(merged_row(reset_btn))
        layout.addWidget(adv)
        return box

    def _build_physics_group(self) -> QGroupBox:
        box = QGroupBox("3  Rock physics")
        layout = QVBoxLayout(box)
        layout.addWidget(_hint(
            "How resistivity becomes water content in each ticked layer. Each value "
            "is a mean ± spread, sampled many times so the result carries its "
            "uncertainty. Unfold a layer to change them."))
        self._physics_host = QWidget()
        self._physics_layout = QVBoxLayout(self._physics_host)
        self._physics_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._physics_host)
        return box

    def _build_estimate_group(self) -> QGroupBox:
        box = QGroupBox("4  Estimate")
        form = QFormLayout(box)
        self._product_boxes: Dict[str, QCheckBox] = {}
        products = QWidget()
        prow = QVBoxLayout(products)
        prow.setContentsMargins(0, 0, 0, 0)
        for key, label in _PRODUCTS:
            check = QCheckBox(label)
            check.setChecked(key == "water_content")
            check.toggled.connect(self._on_products_changed)
            self._product_boxes[key] = check
            prow.addWidget(check)
        self._product_boxes["porosity"].setToolTip(
            "Porosity instead of water content, for ground of known saturation "
            "(1 = saturated).")
        form.addRow(products)
        self._saturation = _number(1.0, 0.01, 1.0, 0.05, 2,
                                   tip="The saturation assumed for the porosity estimate.")
        form.addRow("Assumed saturation", self._saturation)
        self._n_real = make_spinbox(
            100, 2, 5000, tooltip="How many times the rock physics is sampled. More "
                                  "gives a steadier mean and spread, and a longer run.")
        form.addRow("Samples", self._n_real)
        adv, aform = fold()
        self._seed = make_spinbox(7, 0, 10_000_000,
                                  tooltip="The same seed gives the same samples.")
        self._tortuosity = _number(1.0, 0.1, 5.0, 0.1, 2, tip="Archie's tortuosity factor a.")
        self._wc_cmin = _number(0.0, 0.0, 1.0, 0.01, 2)
        self._wc_cmax = _number(0.32, 0.01, 1.0, 0.01, 2)
        aform.addRow("Random seed", self._seed)
        aform.addRow("Tortuosity a", self._tortuosity)
        aform.addRow("Map colours", merged_row(self._wc_cmin, "to", self._wc_cmax))
        export_btn = QPushButton("Save rock-physics settings")
        export_btn.setIcon(theme.icon("fa5s.file-export"))
        export_btn.setToolTip("Write the layers and their rock physics to "
                              "petro_config.json, without estimating anything.")
        export_btn.clicked.connect(self._export_config)
        aform.addRow(merged_row(export_btn))
        form.addRow(adv)
        self._on_products_changed()
        return box

    # -- navigation (the assistant's step names) -----------------------------
    def _go_to(self, step: int) -> None:
        """Show the view and the panel the step is done in."""
        step = max(0, min(int(step), len(_STEPS) - 1))
        loaded = self._summary is not None
        tab = {0: _MODEL_TAB, 1: _LAYERS_TAB if loaded else _MODEL_TAB,
               2: _MODEL_TAB, 3: _LAYERS_TAB if loaded else _MODEL_TAB, 4: _MODEL_TAB,
               5: _RESULTS_TAB if self._has_results else _MODEL_TAB}[step]
        self._tabs.setCurrentIndex(tab)
        self._current = step
        group = {0: self._data_group, 1: self._layers_group, 2: self._estimate_group,
                 3: self._physics_group}.get(step)
        if group is not None:
            self._controls_scroll.ensureWidgetVisible(group)

    def _on_tab_changed(self, index: int) -> None:
        self._current = {_MODEL_TAB: 0, _LAYERS_TAB: 1}.get(index, 5)

    def _step_status(self) -> Dict[str, bool]:
        data_ok = self._summary is not None
        layers_ok = len(self._enabled_layers()) > 0
        target_ok = len(self._collect_products()) > 0
        ran = self.state.module_results.get(self.module_key, {}).get("status") == "ok"
        return {"data": data_ok, "layers": layers_ok, "target": target_ok,
                "all": data_ok and layers_ok and target_ok, "results": ran}

    # -- readiness and guidance ----------------------------------------------
    def _run_blocker(self) -> str:
        """What still stops a run, in a sentence; "" when it can run."""
        if self._summary is None:
            return "Load an inverted ERT model (1) to estimate water content."
        if not self._enabled_layers():
            return "Tick at least one layer (2)."
        if not self._collect_products():
            return "Tick what to estimate (4)."
        return ""

    def _next_step(self) -> str:
        """The one thing to do now, for the line above the views."""
        if self._run_busy is not None:
            return (f"Estimating: {self._n_real.value()} samples of the rock physics "
                    "for every survey. This can take a few minutes.")
        if self._summary is None:
            return "Start by loading an inverted ERT model: use the example or open a folder."
        if not self._enabled_layers():
            return "Next: tick at least one layer to estimate water content in (2, on the right)."
        if not self._collect_products():
            return "Next: tick what to estimate (4, on the right)."
        if self._has_results:
            return ("Done: the maps are on the Results tab, and Time series follows two "
                    "points through the surveys. Change the rock physics and estimate "
                    "again to compare.")
        return ("Ready: check each layer's rock physics (3), then press Estimate water "
                "content.")

    def _sync_page(self) -> None:
        """Bring every cue on the page in line with where the work stands."""
        if not hasattr(self, "_run_reason"):
            return
        reason = self._run_blocker()
        self._run_reason.setText(reason)
        self._run_reason.setVisible(bool(reason))
        if self._run_busy is None:
            self._run_btn.setEnabled(not reason)
        loaded = self._summary is not None
        self._model_stack.setCurrentIndex(1 if loaded else 0)
        self._guide.setVisible(loaded or self._run_busy is not None)
        self._guide.setText(self._next_step())
        for group in (self._layers_group, self._physics_group, self._estimate_group):
            group.setEnabled(loaded)
        layers = bool(self._enabled_layers())
        done = {self._data_group: loaded, self._layers_group: layers,
                self._physics_group: layers,
                self._estimate_group: layers and bool(self._collect_products())}
        for group, title in self._step_titles.items():
            group.setTitle(f"{title}  ✓" if done[group] else title)
        self._tabs.setTabEnabled(_LAYERS_TAB, loaded)
        self._tabs.setTabEnabled(_RESULTS_TAB, self._has_results)
        self._tabs.setTabEnabled(_SERIES_TAB, self._last_result is not None
                                 and self._last_result.get("_wc_mean") is not None)

    # -- data loading --------------------------------------------------------
    def _context_dir(self) -> Optional[Path]:
        geo_dir = self.state.context.get("geo_data_dir") if self.state.context else None
        return Path(geo_dir) if geo_dir else None

    def _data_dir(self) -> Optional[Path]:
        if self._manual_dir is not None:
            return self._manual_dir
        return self._context_dir()

    def _use_context_data(self) -> None:
        self._manual_dir = None
        self._markers_override = None
        if self._data_dir() is None:
            self._manual_dir = self._example_data_dir()
            if self._manual_dir is None:
                self.log(io_utils.missing_example_message(
                    "The Treeline ERT model example", _EXAMPLE_MODEL), "warn")
        self._reload()

    def _example_data_dir(self) -> Optional[Path]:
        """Locate the bundled inverted-model example for standalone launches.

        The Treeline time-lapse model ``examples/Ex_MC_Hydro`` estimates water
        content from, in a clone of the source repository; the pip package
        carries no example data. The synthetic bundle
        ``examples/generate_synthetic_examples.py`` writes to
        ``examples/results/synthetic_Structure_WC`` can still be opened via
        "Open folder...".
        """
        def usable(folder: Path) -> bool:
            files = geo_pipeline.find_model_files(folder)
            return files.get("resistivity") is not None and files.get("mesh") is not None

        return io_utils.find_example(_EXAMPLE_MODEL, self.state.project_root, usable)

    def _show_format_help(self) -> None:
        doc_path = Path(__file__).with_name("geo_input_format.md")
        try:
            text = doc_path.read_text(encoding="utf-8")
        except Exception:  # noqa: BLE001
            text = _INPUT_FORMAT_FALLBACK
        dlg = QDialog(self)
        dlg.setWindowTitle("ERT → Water Content input format")
        dlg.resize(760, 640)
        lay = QVBoxLayout(dlg)
        browser = QTextBrowser(); browser.setOpenExternalLinks(True)
        try:
            browser.setMarkdown(text)
        except Exception:  # noqa: BLE001
            browser.setPlainText(text)
        lay.addWidget(browser)
        close_btn = QPushButton("Close"); close_btn.clicked.connect(dlg.accept)
        lay.addWidget(close_btn)
        dlg.exec()

    def _select_folder(self) -> None:
        start = str(self._data_dir() or Path.cwd())
        path = select_directory(self, "Select inverted-model folder", start)
        if path:
            self._manual_dir = path
            self._markers_override = None
            self._reload()

    def _reload(self) -> None:
        data_dir = self._data_dir()
        self._summary = None
        self._load_problem = ""
        self._mesh = self._resistivity = self._markers = self._coverage = None
        if data_dir is None or not Path(data_dir).exists():
            self._load_problem = (f"Folder not found: {data_dir}" if data_dir is not None
                                  else "No model folder set.")
            self._data_status.setText(
                "No model loaded. Open an ERT inversion's output folder, or a folder "
                "with mesh_res.bms, resmodel.npy and index_marker.npy, or use the example.")
            theme.set_tone(self._data_status, None)
            self._refresh_layer_rows()
            return
        context = {"geo_data_dir": str(data_dir)}
        summary_params: Dict[str, Any] = {"model_data_dir": str(data_dir)}
        if self._markers_override is not None:
            summary_params["markers_override"] = self._markers_override
        try:
            self._summary = geo_pipeline.extract_model_summary(
                context, summary_params, log=lambda m: self.log(m, "debug"))
        except Exception as exc:  # noqa: BLE001
            self._summary = None
            self._load_problem = str(exc)
            self._data_status.setText(f"Could not read the model: {exc}")
            theme.set_tone(self._data_status, "warn")
            self.log(f"Failed to read inverted model: {exc}", "error")
            self._refresh_layer_rows()
            return
        self._load_model_arrays(Path(data_dir))
        s = self._summary
        example = self._example_data_dir()
        same = example is not None and Path(example).resolve() == Path(data_dir).resolve()
        name = "Example: Treeline time-lapse ERT" if same else f"Folder: {Path(data_dir).name}"
        n_time = int(s["n_timesteps"])
        self._data_status.setText(
            f"<b>{name}</b><br>{n_time} survey{'s' if n_time != 1 else ''} · "
            f"{s['n_cells']:,} cells · {len(s['layers'])} layer"
            f"{'s' if len(s['layers']) != 1 else ''}")
        self._data_status.setToolTip(str(data_dir) + (f"\n{s['layering_note']}"
                                                      if s.get("layering_note") else ""))
        if s.get("layering_note"):
            self.log(s["layering_note"], "info")
        theme.set_tone(self._data_status, None)
        self._survey.blockSignals(True)
        self._survey.setRange(0, max(0, n_time - 1))
        self._survey.blockSignals(False)
        self._survey.setEnabled(n_time > 1)
        self._survey_range.setText(f"of 0–{n_time - 1}" if n_time > 1 else "(one survey)")
        self.log(f"Loaded inverted model from {data_dir}", "success")
        self._refresh_layer_rows()
        self._show_model()

    def _load_model_arrays(self, data_dir: Path) -> None:
        """The arrays the Model and Layers views draw, and the mesh they draw on."""
        files = geo_pipeline.find_model_files(data_dir)
        try:
            self._resistivity = geo_pipeline._load_resistivity(files["resistivity"])
            # As the summary and the estimate read them: markers that number
            # cells rather than group them are one layer.
            if self._markers_override is not None or files["markers"] is not None:
                self._markers = geo_pipeline.layer_markers(
                    files, self._resistivity.shape[0], self._markers_override)[0]
            if files["coverage"] is not None:
                self._coverage = np.asarray(np.load(files["coverage"]), dtype=float)
        except Exception as exc:  # noqa: BLE001 - the views say what is missing
            self.log(f"Could not read the model arrays: {exc}", "warn")
        message = ""
        polygons = None
        try:
            import pygimli as pg

            self._mesh = pg.load(str(files["mesh"]))
            polygons = [np.array([[n.pos()[0], n.pos()[1]] for n in cell.nodes()])
                        for cell in self._mesh.cells()]
        except Exception as exc:  # noqa: BLE001 - pyGIMLi missing or the mesh unreadable
            self._mesh = None
            message = f"The model's mesh could not be read ({exc}); the estimate needs it too."
        self._model_note.setText(message)
        self._layer_view.set_mesh(polygons, message)

    def _show_model(self, *_args) -> None:
        """Draw the shown survey's resistivity, blanked where coverage is poor."""
        if self._mesh is None or self._resistivity is None:
            return
        t = min(self._survey.value(), self._resistivity.shape[1] - 1)
        coverage = None
        if self._coverage is not None:
            coverage = (self._coverage[min(t, self._coverage.shape[0] - 1)]
                        if self._coverage.ndim == 2 else self._coverage)
        try:
            self._model_view.show_field(
                self._mesh, self._resistivity[:, t], kind="ert", coverage=coverage,
                title=f"Inverted resistivity, survey {t}")
        except Exception as exc:  # noqa: BLE001
            self._model_note.setText(f"The model could not be drawn: {exc}")

    # -- layers --------------------------------------------------------------
    def _refresh_layer_rows(self) -> None:
        """One row per layer in the model: a tick, a name, and its size."""
        _clear(self._layers_form)
        self._layer_rows = []
        if self._summary is not None:
            for i, layer in enumerate(self._layers_top_down()):
                marker = int(layer["marker"])
                enabled = QCheckBox()
                # Ticked when the layer has known petrophysics defaults.
                enabled.setChecked(marker in geo_pipeline.DEFAULT_LAYER_DISTRIBUTIONS)
                enabled.toggled.connect(self._on_layers_changed)
                name = QLineEdit(str(layer.get("name", f"Layer {marker}")))
                name.editingFinished.connect(self._on_layers_changed)
                size = _hint(f"marker {marker} · {int(layer['n_cells']):,} cells")
                self._layers_form.addWidget(enabled, i, 0)
                self._layers_form.addWidget(name, i, 1)
                self._layers_form.addWidget(size, i, 2)
                self._layer_rows.append({"marker": marker, "n_cells": int(layer["n_cells"]),
                                         "enabled": enabled, "name": name})
        # A new (or re-read) model's layers start from the defaults.
        self._param_values = {}
        self._param_widgets = {}
        self._param_layout = None
        self._on_layers_changed()

    def _layers_top_down(self) -> List[Dict[str, Any]]:
        """The model's layers from the surface down, by the mean height of their cells.

        In marker order the bedrock (2) came before the regolith (3) above it.
        """
        layers = list(self._summary["layers"]) if self._summary else []
        if self._mesh is None or self._markers is None:
            return layers
        try:
            z = np.asarray(self._mesh.cellCenters(), dtype=float)[:, 1]
            if z.size != self._markers.size:
                return layers
            height = {int(m): float(np.mean(z[self._markers == m]))
                      for m in np.unique(self._markers)}
        except Exception:  # noqa: BLE001 - marker order is still an order
            return layers
        return sorted(layers, key=lambda layer: -height.get(int(layer["marker"]), 0.0))

    def _enabled_layers(self) -> List[Dict[str, Any]]:
        return [r for r in self._layer_rows if r["enabled"].isChecked()]

    def _layer_names(self) -> Dict[int, str]:
        return {int(r["marker"]): r["name"].text().strip() or f"Layer {r['marker']}"
                for r in self._layer_rows}

    def _on_layers_changed(self, *_args) -> None:
        self._refresh_parameter_panels()
        self._layer_view.set_layers(self._markers, self._layer_names(),
                                    {int(r["marker"]) for r in self._enabled_layers()},
                                    [int(r["marker"]) for r in self._layer_rows])
        self._sync_page()

    def _reset_layers(self) -> None:
        self._markers_override = None
        self._reload()

    def _derive_markers_from_structure(self) -> None:
        if self._summary is None or self._data_dir() is None:
            self.log("Load an inverted ERT model first.", "warn")
            return
        interface = None
        source = ""
        ss = self.state.shared_structure if self.state else None
        if ss and ss.get("interface_xz"):
            interface = np.asarray(ss["interface_xz"], dtype=float)
            source = str(ss.get("source", "seismic handoff"))
        else:
            path, _ = QFileDialog.getOpenFileName(
                self, "Load bedrock interface (x, z)", str(self._data_dir() or Path.cwd()),
                "Interface (*.csv *.txt *.npy);;All files (*)")
            if not path:
                return
            try:
                arr = np.atleast_2d(io_utils.load_2d_array(path))
                interface = arr[:, :2].astype(float)
            except Exception as exc:  # noqa: BLE001
                self.log(f"Could not read interface file: {exc}", "error")
                return
            source = Path(path).name
        if interface is None or interface.shape[0] < 2:
            self.log("Interface needs at least two (x, z) points.", "warn")
            return
        try:
            ctx = {"geo_data_dir": str(self._data_dir())}
            params = {"model_data_dir": str(self._data_dir()),
                      "output_dir": str(self.state.output_dir or "")}
            info = geo_pipeline.derive_markers_from_interface(
                ctx, params, interface, log=lambda m: self.log(m, "info"))
        except geo_pipeline.BackendUnavailable as exc:
            self.log(f"pygimli is needed to classify the mesh: {exc}", "warn")
            return
        except Exception as exc:  # noqa: BLE001
            self.log(f"Could not derive markers: {exc}", "error")
            return
        self._markers_override = info["markers"]
        self.log(f"Layers derived from {source}: counts={info['counts']}. "
                 f"Sidecar saved to {info['markers_path']}.", "success")
        self._reload()
        self._go_to(1)

    # -- rock physics ---------------------------------------------------------
    def _layer_default(self, marker: int) -> Dict[str, Dict[str, float]]:
        src = geo_pipeline.DEFAULT_LAYER_DISTRIBUTIONS.get(marker, _GENERIC_LAYER)
        return {k: dict(src[k]) for k in _PARAM_SPECS}

    def _refresh_parameter_panels(self) -> None:
        """One panel per ticked layer, keeping the values already set in them.

        Rebuilt only when the ticked layers change; a marker's values carry
        across the rebuild, and its name is updated in place.
        """
        enabled = self._enabled_layers()
        markers = [int(row["marker"]) for row in enabled]
        names = self._layer_names()
        if markers == self._param_layout:
            for marker, label in self._param_titles.items():
                label.setText(f"<b>{names.get(marker, marker)}</b>")
            return
        for marker, widgets in self._param_widgets.items():
            self._param_values[marker] = {
                key: (mean.value(), std.value()) for key, (mean, std) in widgets.items()}
        self._param_layout = markers
        _clear(self._physics_layout)
        self._param_widgets = {}
        self._param_summaries = {}
        self._param_titles = {}
        if not enabled:
            self._physics_layout.addWidget(_hint("Tick a layer in 2 to set its rock physics."))
            return
        for row in enabled:
            marker = int(row["marker"])
            title = QLabel(f"<b>{names.get(marker, marker)}</b>")
            summary = _hint()
            adv, aform = fold("Change")
            grid_host = QWidget()
            grid = QGridLayout(grid_host)
            grid.setContentsMargins(0, 0, 0, 0)
            grid.addWidget(QLabel("<b>Mean</b>"), 0, 1)
            grid.addWidget(QLabel("<b>± spread</b>"), 0, 2)
            defaults = self._layer_default(marker)
            kept = self._param_values.get(marker, {})
            self._param_widgets[marker] = {}
            for r, key in enumerate(_PARAM_SPECS, start=1):
                (mlo, mhi, mstep, mdec), (slo, shi, sstep, sdec) = _PARAM_SPECS[key]
                mean, std = kept.get(key, (float(defaults[key]["mean"]),
                                           float(defaults[key]["std"])))
                mean_box = _number(float(mean), mlo, mhi, mstep, mdec)
                std_box = _number(float(std), slo, shi, sstep, sdec)
                for box in (mean_box, std_box):
                    box.valueChanged.connect(lambda _v, m=marker: self._update_param_summary(m))
                grid.addWidget(QLabel(_PARAM_LABELS[key]), r, 0)
                grid.addWidget(mean_box, r, 1)
                grid.addWidget(std_box, r, 2)
                self._param_widgets[marker][key] = (mean_box, std_box)
            aform.addRow(grid_host)
            for widget in (title, summary, adv):
                self._physics_layout.addWidget(widget)
            self._param_titles[marker] = title
            self._param_summaries[marker] = summary
            self._update_param_summary(marker)

    def _update_param_summary(self, marker: int) -> None:
        """A layer's rock physics in one line."""
        widgets = self._param_widgets.get(marker)
        label = self._param_summaries.get(marker)
        if not widgets or label is None:
            return

        def value(key: str) -> str:
            mean, std = (box.value() for box in widgets[key])
            return f"{mean:g} ± {std:g}" if std else f"{mean:g}"

        label.setText(f"porosity {value('porosity')}, m {value('m')}, n {value('n')}, "
                      f"pore water {value('rho_fluid')} Ω·m")

    # -- what to estimate ----------------------------------------------------
    def _on_products_changed(self, *_args) -> None:
        set_rows_visible([self._saturation], self._product_boxes["porosity"].isChecked())
        self._sync_page()

    def _collect_products(self) -> List[str]:
        if not hasattr(self, "_product_boxes"):
            return []
        return [k for k, b in self._product_boxes.items() if b.isChecked()]

    # -- results -------------------------------------------------------------
    def _populate_results(self, result: Dict[str, Any]) -> None:
        self._has_results = True
        products = [dict(_PRODUCTS).get(str(p), str(p)) for p in result.get("products", [])]
        if result.get("status") == "config_exported":
            self._result_summary.setText(
                "Nothing was estimated; the rock-physics settings were saved instead.")
        else:
            self._result_summary.setText(
                f"Estimated {', '.join(products).lower() or 'nothing'} from "
                f"{result.get('n_realizations', '—')} samples of the rock physics, for "
                f"{result.get('n_timesteps', '—')} surveys. The maps show survey "
                f"{result.get('preview_timestep', '—')}.")

        entries: List[Tuple[str, str]] = []
        # The estimate and its uncertainty first, then the percentiles.
        rank = {"mean": 0, "std": 1, "p50": 2, "p10": 3, "p90": 4}
        for product, statistics in (result.get("display_paths") or {}).items():
            if not isinstance(statistics, dict):
                continue
            for statistic, path in sorted(statistics.items(),
                                          key=lambda item: rank.get(item[0], 9)):
                if path and Path(path).exists():
                    entries.append((self._result_map_label(str(product), str(statistic)),
                                    str(path)))
        # Backward-compatible fallback for results produced before display_paths.
        known_paths = {path for _, path in entries}
        for path in result.get("figure_paths", []):
            if path and Path(path).exists() and str(path) not in known_paths:
                entries.append((self._legacy_result_map_label(Path(path)), str(path)))
        self._result_figure.blockSignals(True)
        self._result_figure.clear()
        for label, path in entries:
            self._result_figure.addItem(label, path)
        self._result_figure.blockSignals(False)
        if self._result_figure.count():
            self._result_figure.setCurrentIndex(0)
        self._show_selected_result_map()

        files = [Path(p).name for p in result.get("stats_paths", [])]
        bits = [f"Arrays: {', '.join(files)}" if files else ""]
        if result.get("config_path"):
            bits.append(f"Rock-physics settings: {result['config_path']}")
        if result.get("output_dir"):
            bits.append(f"Folder: {result['output_dir']}")
        self._result_files.setText("<br>".join(bit for bit in bits if bit))
        self._sync_page()

    @staticmethod
    def _result_map_label(product: str, statistic: str) -> str:
        products = {"water_content": "Water content", "porosity": "Porosity"}
        statistics = {
            "mean": "mean",
            "std": "uncertainty (1σ)",
            "p10": "P10",
            "p50": "median (P50)",
            "p90": "P90",
        }
        return (f"{products.get(product, product.replace('_', ' ').title())} — "
                f"{statistics.get(statistic, statistic.upper())}")

    @classmethod
    def _legacy_result_map_label(cls, path: Path) -> str:
        stem = path.stem.lower()
        product = "porosity" if "porosity" in stem else "water_content"
        statistic = next(
            (key for key in ("mean", "std", "p10", "p50", "p90") if key in stem),
            stem,
        )
        return cls._result_map_label(product, statistic)

    def _show_selected_result_map(self) -> None:
        path = self._result_figure.currentData()
        if not (path and Path(path).exists() and self._result_map_view.set_image_file(path)):
            self._result_map_view.clear()

    def _plot_points(self) -> None:
        res = self._last_result
        if not res or res.get("_wc_mean") is None:
            self.log("Estimate water content first.", "warn")
            return
        cc = res.get("_cell_centers")
        wc_mean = res.get("_wc_mean")
        wc_std = res.get("_wc_std")
        positions = [(self._mx1.value(), self._my1.value()),
                     (self._mx2.value(), self._my2.value())]
        try:
            series, idx = geo_pipeline.extract_point_series(cc, wc_mean, positions)
            stds = None
            if wc_std is not None:
                stds, _ = geo_pipeline.extract_point_series(cc, wc_std, positions)
            t = res.get("_timestep_indices") or list(range(series.shape[1]))
            figure = Figure(figsize=(5.0 * len(positions), 3.4), layout="constrained")
            axes = figure.subplots(1, len(positions), squeeze=False)
            for i in range(len(positions)):
                ax = axes[0][i]
                ax.plot(t, series[i], "o-", color="tab:blue", label="mean")
                if stds is not None:
                    ax.fill_between(t, series[i] - stds[i], series[i] + stds[i],
                                    color="tab:blue", alpha=0.2, label="±1σ")
                ax.set_xlabel("Survey"); ax.set_ylabel("Water content (-)")
                ax.set_title(f"Point {i + 1} (cell {idx[i]})"); ax.grid(True, alpha=0.3)
                if i == 0:
                    ax.legend(frameon=False)
            out = Path(res.get("output_dir", ".")) / "monitoring_series.png"
            figure.savefig(out, dpi=170, bbox_inches="tight")
        except Exception as exc:  # noqa: BLE001
            self.log(f"Could not plot monitoring series: {exc}", "error")
            return
        self._monitor_view.set_image_file(str(out))
        self.log(f"Saved monitoring series to {out}", "success")

    def _open_output_folder(self) -> None:
        out = self.state.module_results.get(self.module_key, {}).get("output_dir")
        out = out or str(self.state.output_dir or "")
        if out and Path(out).exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(out))
        else:
            self.log("Output folder not available yet.", "warn")

    # -- params / run --------------------------------------------------------
    def _collect_layers(self) -> List[Dict[str, Any]]:
        layers: List[Dict[str, Any]] = []
        names = self._layer_names()
        for row in self._enabled_layers():
            marker = int(row["marker"])
            entry: Dict[str, Any] = {"marker": marker, "name": names[marker]}
            widgets = self._param_widgets.get(marker)
            kept = self._param_values.get(marker, {})
            defaults = self._layer_default(marker)
            for key in _PARAM_SPECS:
                if widgets and key in widgets:
                    mean_box, std_box = widgets[key]
                    entry[key] = {"mean": mean_box.value(), "std": std_box.value()}
                elif key in kept:
                    entry[key] = {"mean": kept[key][0], "std": kept[key][1]}
                else:
                    entry[key] = dict(defaults[key])
            layers.append(entry)
        return layers

    def _coverage_threshold(self) -> float:
        """The view's "Hide below" cut, which the maps follow; off shows every cell."""
        cut = self._model_view.hide_below()
        return float(cut) if cut is not None else -1.0e9

    def _collect_params(self) -> Dict[str, Any]:
        out_dir = self.state.output_dir
        return {
            "model_data_dir": str(self._data_dir()) if self._data_dir() else "",
            "output_dir": str(out_dir) if out_dir else "",
            "seed": self._seed.value(),
            "n_realizations": self._n_real.value(),
            "products": self._collect_products(),
            "saturation_value": self._saturation.value(),
            "tortuosity_a": self._tortuosity.value(),
            "coverage_threshold": self._coverage_threshold(),
            "preview_timestep": self._survey.value(),
            "wc_cmin": self._wc_cmin.value(),
            "wc_cmax": self._wc_cmax.value(),
            "layers": self._collect_layers(),
            "markers_override": self._markers_override,
        }

    def _context(self) -> Dict[str, Any]:
        ctx = dict(self.state.context or {})
        if self._data_dir():
            ctx["geo_data_dir"] = str(self._data_dir())
        if self.state.output_dir:
            ctx.setdefault("output_dir", str(self.state.output_dir))
        return ctx

    def _export_config(self) -> str:
        params = self._collect_params()
        config = geo_pipeline.build_petro_config(self._context(), params)
        active = self.state.active_run(self.module_key)
        out_dir = active.outputs_dir if active else self.state.ensure_results_store().scratch_dir(self.module_key)
        config_path = out_dir / "petro_config.json"
        io_utils.write_json(config_path, config)
        self.log(f"Exported petrophysics config to {config_path}", "success")
        self.report_result({
            "products": params["products"], "config_path": str(config_path),
            "status": "config_exported", "output_dir": str(out_dir)})
        return str(config_path)

    def _run_inverse(self) -> None:
        reason = self._run_blocker()
        if reason:
            self.log(reason, "warn")
            return
        params = self._collect_params()
        seed = int(params.pop("seed"))
        data_dir = self._data_dir()
        params.pop("model_data_dir", None)
        params.pop("output_dir", None)
        try:
            run = self.begin_persisted_run(
                "geo_hydrology.ert_to_wc",
                workflow_id="geo_hydrology.ert_to_wc",
            )
        except Exception as exc:  # noqa: BLE001
            self.log(f"Could not prepare Project run: {exc}", "error")
            return
        # One compressed bundle rather than a copy of each file. The mesh is
        # read back by PyGIMLi and so keeps its bytes, but it travels with the
        # arrays instead of adding four more files to the run.
        inputs: Dict[str, Any] = {"context": {}}
        if data_dir is not None:
            # Under the bundle's own names, whichever layout the folder holds
            # (an ERT page's output folder names them its own way).
            found = geo_pipeline.find_model_files(data_dir)
            present = {
                geo_pipeline.MODEL_FILES[key]: path
                for key, path in found.items()
                if path is not None and path.is_file()
            }
            if present:
                try:
                    stored = run_inputs.save_file_bundle(
                        run.inputs_dir / "ert_model_inputs",
                        present,
                        kind="ert_model_bundle",
                        meta={"source_dir": str(data_dir)},
                    )
                except OSError as exc:
                    self.fail_persisted_run(str(exc))
                    self.log(f"Could not persist the ERT model bundle: {exc}", "error")
                    return
                inputs["model_bundle"] = ArtifactRef.from_path(
                    stored,
                    artifact_id="ert-model-input:bundle",
                    kind="ert_model_bundle",
                    base_dir=run.run_dir,
                    metadata={"files": sorted(present)},
                )
        # The run's figures are drawn in another process, which cannot see the
        # studio's View > Length Units choice.
        params["length_unit"] = length_units.current()
        spec = WorkflowSpec(
            workflow_id="geo_hydrology.ert_to_wc",
            inputs=inputs,
            parameters=params,
            seed=seed,
            metadata={"source": "qt", "rng": "numpy.default_rng"},
        )
        recipe_path, script_path = export_workflow_bundle(
            spec, run.run_dir, stem="ert_to_water_content"
        )
        self._reproduce.set_bundle(recipe_path, script_path)
        self._workflow_recipe_path = str(recipe_path)
        self._run_busy = BusyStateController([self._run_btn])
        self._run_busy.start()
        self._run_btn.setText("Estimating…")
        self._progress.setVisible(True)
        self._progress.setRange(0, 0)
        self._run_status.setText("Starting the estimate…")
        theme.set_tone(self._run_status, None)
        self._sync_page()
        self.log("Starting geophysics → hydrology estimation…", "info")
        # In a process of its own, so the window keeps painting through the
        # Monte Carlo; the point series plot needs the cell centres and the
        # water-content statistics back.
        worker = ProcessWorkflowWorker(recipe_path, run.run_dir, run.outputs_dir,
                                       run.result_path,
                                       objects=("_cell_centers", "_wc_mean", "_wc_std"))
        self._worker = self.register_worker(worker, activity="Estimating water content")
        self._run_problem = ""
        worker.logged.connect(lambda message: self._on_worker_log(message, "info"))
        worker.succeeded.connect(self._on_workflow_ok)
        worker.failed.connect(
            lambda message, w=worker: self._on_run_failed(message, w.missing_backend))
        worker.finished.connect(self._reset_run_button)
        worker.start()
        self._stop.attach(worker, "geo_hydrology.ert_to_wc")

    def _on_workflow_ok(self, result: WorkflowRunResult) -> None:
        if hasattr(self.state, "update_workflow_result"):
            self.state.update_workflow_result(
                self.module_key,
                "geo_hydrology.ert_to_wc",
                result.to_dict(),
                recipe_path=self._workflow_recipe_path,
            )
        self._on_run_ok(result.legacy_payload())

    def _on_worker_log(self, message: str, level: str) -> None:
        self._run_status.setText(message)
        self.log(message, level)

    def _on_run_ok(self, result: dict) -> None:
        self._progress.setRange(0, 1); self._progress.setValue(1)
        self.log(f"Estimation complete: {result.get('products')}.", "success")
        self._run_status.setText("Finished. The maps are on the Results tab.")
        theme.set_tone(self._run_status, "ok")
        self._last_result = result
        # Strip the in-process numpy payload before publishing to the bridge JSON.
        published = {k: v for k, v in result.items() if not k.startswith("_")}
        self.report_result(published)
        self._populate_results(result)
        self._go_to(5)

    def _on_run_failed(self, message: str, backend_unavailable: bool) -> None:
        self._run_problem = message
        self.fail_persisted_run(message, "geo_hydrology.ert_to_wc")
        self._progress.setRange(0, 1); self._progress.setValue(0)
        self._last_result = None
        if not backend_unavailable:
            # The run itself failed - out of memory, arrays of the wrong shape -
            # and its own reason is what helps, beside the settings that can be
            # changed before trying again. Exporting the configuration in its
            # place blamed a backend that was there.
            self.log(f"Estimation failed: {message}", "error")
            self._run_status.setText(f"The estimate failed: {message}")
            theme.set_tone(self._run_status, "error")
            return
        self.log(f"Estimation could not run: {message}", "warn")
        config_path = self._export_config()
        note = ("The Monte Carlo engine (pygimli) is not installed or could not be loaded. "
                f"The rock-physics settings were saved to {config_path}.")
        self._run_status.setText(note)
        theme.set_tone(self._run_status, "warn")
        self.log(note, "warn")
        self._populate_results({"status": "config_exported",
                                "products": self._collect_products(),
                                "config_path": config_path, "figure_paths": []})
        self._go_to(5)

    def _reset_run_button(self) -> None:
        if self._run_busy is not None:
            self._run_busy.finish()
            self._run_busy = None
        self._run_btn.setText("Estimate water content")
        self._progress.setVisible(False)
        if self._worker is not None and self._worker.is_cancelled():
            self._run_problem = "Stopped by user"
            self._run_status.setText("Stopped before it finished. Press Estimate water "
                                     "content to start again.")
            theme.set_tone(self._run_status, "warn")
        self._sync_page()

    # -- AQUAH agent interface ----------------------------------------------
    def agent_describe(self) -> Dict[str, Any]:
        return {
            "module": self.module_key,
            "title": self.module_title,
            "current_step": _STEPS[self._current],
            "state": self._agent_status(),
            "actions": [
                {"name": "use_ert_model", "args": {},
                 "desc": ("Load the model shown on the ERT Processing page (single or "
                          "time-lapse), as it is displayed there. Use this after an "
                          "inversion in the studio.")},
                {"name": "use_example_data", "args": {},
                 "desc": "Load the bundled / context inverted-ERT model bundle."},
                {"name": "set_data_dir", "args": {"path": "str"},
                 "desc": ("Load an inverted model folder: an ERT inversion's output folder, "
                          "or a bundle (mesh_res.bms, resmodel.npy, index_marker.npy).")},
                {"name": "detect_layers", "args": {},
                 "desc": "Re-detect geological layers from the loaded model markers."},
                {"name": "select_products", "args": {"products": [k for k, _ in _PRODUCTS]},
                 "desc": "Choose hydrologic products to estimate."},
                {"name": "set_params", "args": {"params": {"<key>": "value"}},
                 "desc": ("Set parameters. Keys: n_realizations, seed, saturation, "
                          "coverage_threshold, preview_timestep, tortuosity, wc_cmin, wc_cmax.")},
                {"name": "goto_step", "args": {"step": list(_STEPS)},
                 "desc": "Show the view and panel of a step, by name or index."},
                {"name": "run", "args": {},
                 "desc": "Run the Monte Carlo water-content / porosity estimation."},
                {"name": "get_status", "args": {},
                 "desc": "Report readiness and the last result status."},
            ],
        }

    def agent_apply(self, action: str, args: Dict[str, Any]) -> Dict[str, Any]:
        args = args or {}
        handlers = {
            "use_ert_model": lambda: self._agent_use_ert_model(),
            "use_example_data": lambda: self._agent_use_example_data(),
            "set_data_dir": lambda: self._agent_set_data_dir(args.get("path")),
            "detect_layers": lambda: self._agent_detect_layers(),
            "select_products": lambda: self._agent_select_products(args.get("products")),
            "set_params": lambda: self._agent_set_params(args.get("params", args)),
            "goto_step": lambda: self._agent_goto_step(args.get("step")),
            "run": lambda: self._agent_run(),
            "get_status": lambda: self._agent_status(),
        }
        handler = handlers.get(action)
        if handler is None:
            return {"status": "failed", "error": f"Unknown action '{action}'.",
                    "valid_actions": list(handlers.keys())}
        return handler()

    def show_run_inputs(self, inputs: Dict[str, Any]) -> str:
        """Open the model the run has just inverted, so this page is not blank.

        The automatic run reaches this page having already recovered the
        resistivity models, and the page was still showing "No model folder set.
        Use 'Select folder…'" - asking the user to go and find something the run
        was holding. The inversion step now writes the bundle these controls
        read, so the folder is known and the layers it found are on screen.

        Anything already loaded is left alone; a run must not discard work
        somebody was in the middle of.
        """
        if self._summary is not None:
            return ""
        folder = inputs.get("model_directory") or inputs.get("ert_model_dir")
        if not folder or not Path(str(folder)).is_dir():
            return ""
        result = self._agent_set_data_dir(str(folder))
        if not result.get("loaded"):
            return ""
        self._go_to(1)
        layers = len(self._enabled_layers())
        return f"the inverted model from this run ({layers} layer(s))"

    def show_run_stage(self, tool: str) -> str:
        """Follow the run through this page's steps."""
        views = {"fetch_climate": "Data", "convert_water_content": "Run"}
        step = views.get(str(tool), "")
        if not step:
            return ""
        self._agent_goto_step(step)
        return step

    def _agent_status(self) -> Dict[str, Any]:
        status = self._step_status()
        last = self.state.module_results.get(self.module_key, {})
        return {
            "status": "ok",
            "steps": status,
            "current_step": _STEPS[self._current],
            "data_dir": str(self._data_dir() or ""),
            "products": self._collect_products(),
            "enabled_layers": [r["marker"] for r in self._enabled_layers()],
            "run_blocked_because": self._run_blocker(),
            "last_result_status": last.get("status"),
            "last_run_problem": self._run_problem,
        }

    def _agent_use_example_data(self) -> Dict[str, Any]:
        self._use_context_data()
        return {"status": "ok", "data_dir": str(self._data_dir() or ""),
                "loaded": self._summary is not None}

    def _agent_set_data_dir(self, path: Any) -> Dict[str, Any]:
        if not path:
            return {"status": "failed", "error": "Provide 'path' to a model folder."}
        self._manual_dir = Path(str(path))
        self._markers_override = None
        self._reload()
        if self._summary is None:
            # Said, not just "loaded": false - the assistant has to know why.
            folder = Path(str(path))
            holds = sorted(p.name for p in folder.iterdir())[:20] if folder.is_dir() else []
            return {"status": "failed", "data_dir": str(folder), "loaded": False,
                    "error": getattr(self, "_load_problem", "") or "The model could not be read.",
                    "folder_holds": holds,
                    "hint": ("After an inversion on the ERT page, use_ert_model loads the "
                             "model it shows.")}
        return {"status": "ok", "data_dir": str(self._data_dir() or ""), "loaded": True,
                "n_cells": self._summary["n_cells"],
                "n_timesteps": self._summary["n_timesteps"],
                "layers": [layer["marker"] for layer in self._summary["layers"]],
                "layers_from": self._summary.get("markers_source"),
                "layering_note": self._summary.get("layering_note") or ""}

    def _agent_use_ert_model(self) -> Dict[str, Any]:
        """Load the model the ERT page shows, written out as this page's bundle."""
        pages = getattr(self.window(), "_pages", None) or {}
        ert = pages.get("ert")
        handover = getattr(ert, "agent_handover", None)
        if not callable(handover):
            return {"status": "failed", "error": (
                "The ERT Processing page has not been opened, so it has no model. Run an "
                "inversion there first, or load a model folder with set_data_dir.")}
        try:
            folder = (self.state.ensure_results_store().scratch_dir(self.module_key)
                      / f"ert_model_{time.strftime('%Y%m%d-%H%M%S')}")
            record = handover(folder)
        except Exception as exc:  # noqa: BLE001 - say why, the user can open a folder
            return {"status": "failed", "error": f"Could not take the ERT page's model: {exc}"}
        if not record:
            return {"status": "failed", "error": (
                "The ERT page shows no finished model (none run yet, or an inversion "
                "is still running). Check its status, then try again.")}
        result = self._agent_set_data_dir(record["model_dir"])
        if result.get("loaded"):
            self.log(f"Loaded the ERT page's {record.get('kind') or 'single'} model "
                     f"(chi2 {record.get('chi2')}).", "success")
            result.update(source="ERT Processing page", kind=record.get("kind"),
                          chi2=record.get("chi2"), source_files=record.get("source_files"))
        return result

    def _agent_detect_layers(self) -> Dict[str, Any]:
        if self._summary is None:
            return {"status": "failed", "error": "Load a model first (use_example_data or set_data_dir)."}
        self._refresh_layer_rows()
        return {"status": "ok", "enabled_layers": [r["marker"] for r in self._enabled_layers()]}

    def _agent_select_products(self, products: Any) -> Dict[str, Any]:
        valid = {k for k, _ in _PRODUCTS}
        if not isinstance(products, list) or not products:
            return {"status": "failed", "error": "Provide 'products' as a non-empty list.",
                    "valid": sorted(valid)}
        invalid = [p for p in products if p not in valid]
        if invalid:
            return {"status": "failed", "error": f"Invalid products {invalid}.", "valid": sorted(valid)}
        for key, box in self._product_boxes.items():
            box.setChecked(key in products)
        self._on_products_changed()
        return {"status": "ok", "selected": self._collect_products()}

    def _agent_set_params(self, params: Any) -> Dict[str, Any]:
        if not isinstance(params, dict):
            return {"status": "failed", "error": "Provide 'params' as a JSON object."}
        handlers = {
            "n_realizations": lambda v: self._n_real.setValue(int(v)),
            "seed": lambda v: self._seed.setValue(int(v)),
            "saturation": lambda v: self._saturation.setValue(float(v)),
            # The cut lives on the Model view ("Hide below"), which the maps follow.
            "coverage_threshold": lambda v: self._model_view.set_hide_below(float(v)),
            "preview_timestep": lambda v: self._survey.setValue(int(v)),
            "tortuosity": lambda v: self._tortuosity.setValue(float(v)),
            "wc_cmin": lambda v: self._wc_cmin.setValue(float(v)),
            "wc_cmax": lambda v: self._wc_cmax.setValue(float(v)),
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

    def _agent_goto_step(self, step: Any) -> Dict[str, Any]:
        names = {name.lower(): i for i, name in enumerate(_STEPS)}
        idx = step if isinstance(step, int) else names.get(str(step).lower())
        if idx is None or not (0 <= idx < len(_STEPS)):
            return {"status": "failed", "error": f"Unknown step '{step}'.", "valid": list(_STEPS)}
        self._go_to(idx)
        return {"status": "ok", "step": _STEPS[self._current]}

    def _agent_run(self) -> Dict[str, Any]:
        status = self._step_status()
        if not status["all"]:
            missing = [k for k in ("data", "layers", "target") if not status[k]]
            return {"status": "failed", "error": "Not ready to run.", "missing": missing,
                    "hint": "Load a model, ensure at least one layer is enabled, and select a product."}
        self._run_inverse()
        return {"status": "started",
                "message": "Water-content estimation started. Ask for status shortly.",
                "products": self._collect_products()}
