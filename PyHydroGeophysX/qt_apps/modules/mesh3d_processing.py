"""Mesh 3D module: an interactive 3D mesh builder.

Build sensor arrays (surface grid, single borehole, crosshole, surface-to-
borehole) for electrodes, geophones or other sensors, choose topography
(including loading x, y, z points from a file), generate a PyGIMLi 3D mesh
(surface-topography prism or Gmsh-free structured grid), and view the sensors +
the generated mesh in an interactive PyVistaQt 3D viewer. Mesh generation runs on
a worker thread; the mesh can be exported to BMS / VTK / sensor CSV. The compute lives in
``qt_apps.mesh3d_builder`` (Qt-free); this module is the UI.

If PyVistaQt is unavailable the builder still generates and exports meshes; only
the interactive 3D preview is replaced by a short note.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PySide6.QtCore import QTimer, Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
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
    QSpinBox,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from PyHydroGeophysX.core import mesh_3d as mesh3d_builder
from PyHydroGeophysX.qt_apps import io_utils, theme
from PyHydroGeophysX.qt_apps.modules.base import BaseModule, LogFn
from PyHydroGeophysX.qt_apps.qt_utils import (
    BusyStateController,
    ReproduceBar,
    make_double_spinbox,
    select_directory,
)
from PyHydroGeophysX.qt_apps.workers import TaskWorker, WorkflowWorker
from PyHydroGeophysX.workflows import (
    ArtifactRef,
    RunContext,
    WorkflowRunResult,
    WorkflowSpec,
    export_workflow_bundle,
)

_INSTALL_MESSAGE = (
    "Interactive 3D preview needs <code>pyvista</code>, <code>pyvistaqt</code> and a "
    "working <code>vtk</code> build:<br>"
    "<code>pip install \"pyhydrogeophysx[desktop-3d]\"</code><br><br>"
    "Install with whichever tool already manages the packages here, which is not "
    "always the one that created the environment. Run <code>conda list numpy</code>: "
    "if it reports <code>pypi</code>, use the pip command above; if it names a conda "
    "channel, use <code>conda install -c conda-forge pyvista pyvistaqt</code> instead. "
    "Pulling VTK in through the other one leaves two builds in the same "
    "environment.<br><br>"
    "Mesh generation and export still work without it."
)
_MESH_FILTER = "Mesh (*.vtk *.vtu *.vtp *.stl *.ply *.bms);;All files (*)"
_ARRAY_TYPES = ["Surface grid", "Single borehole", "Crosshole", "Surface-to-borehole"]
_MESH_TYPES = ["Surface with topography", "Box mesh"]
_TOPO_TYPES = ["Flat", "Linear tilt", "Gaussian hill", "Custom expression", "From file (x, y, z)"]
_MESH_ENGINES = ["Auto", "Gmsh (tetrahedral)", "PyGIMLi prism", "Structured grid"]
_ERT_FORWARD_SCHEMES = ("dd", "wa", "slm", "wb")


def _ensure_vtk_matplotlib_shim() -> None:
    """Stub the optional ``vtkmodules.vtkRenderingMatplotlib`` module when absent.

    Some VTK builds (notably conda-forge ``vtk-base``) omit it, yet pyvista imports
    it unconditionally for a side effect it does not need for mesh display.
    """
    import sys
    import types

    if "vtkmodules.vtkRenderingMatplotlib" in sys.modules:
        return
    try:
        import vtkmodules.vtkRenderingMatplotlib  # noqa: F401 - real module present
    except Exception:  # noqa: BLE001 - missing optional submodule; provide a stub
        sys.modules["vtkmodules.vtkRenderingMatplotlib"] = types.ModuleType(
            "vtkmodules.vtkRenderingMatplotlib"
        )


def _try_import_pyvista() -> Tuple[bool, Optional[Any], Optional[Any], str]:
    try:
        import os

        # Under the offscreen Qt platform (headless / CI / --self-test) there is no
        # usable OpenGL context, and constructing the live QtInteractor aborts the
        # process at the VTK level (not a catchable Python exception). Disable the
        # interactive viewer there; a real desktop session uses a normal platform.
        if os.environ.get("QT_QPA_PLATFORM", "").lower() == "offscreen":
            return False, None, None, "offscreen platform (interactive 3D viewer disabled)"

        # pyvistaqt uses qtpy; force PySide6 so it does not bind to a PyQt5 install.
        os.environ.setdefault("QT_API", "pyside6")
        _ensure_vtk_matplotlib_shim()
        import pyvista as pv
        from pyvistaqt import QtInteractor

        return True, pv, QtInteractor, ""
    except Exception as exc:  # noqa: BLE001
        return False, None, None, str(exc)


class Mesh3DModule(BaseModule):
    module_key = "mesh3d"
    module_title = "3D Mesh Builder"

    def __init__(self, state: Any, log: LogFn, parent=None) -> None:
        super().__init__(state, log, parent)
        self._pv = None
        self._plotter = None
        self._sensors = None         # last previewed/generated sensor DataFrame
        self._mesh = None            # last generated pygimli mesh
        self._topo_points = None     # loaded (x, y, z) topography points
        self._gen_worker: Optional[TaskWorker] = None
        self._gen_busy: Optional[BusyStateController] = None
        self._ert_fwd_worker: Optional[WorkflowWorker] = None
        self._ert_fwd_result = None  # last agent-triggered 3D ERT forward result
        self._ert_forward_params = {"scheme": "dd", "background_res": 100.0, "noise": 0.03}

        ok, pv, qt_interactor, err = _try_import_pyvista()
        self._pv = pv if ok else None

        root = QHBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.addWidget(self._build_controls())

        right = QVBoxLayout()
        right.addLayout(self._build_toolbar())
        if ok:
            try:
                self._plotter = qt_interactor(self)
                right.addWidget(self._plotter.interactor, stretch=1)
                self._plotter.set_background("white")
                self._plotter.add_axes()
                self.log("PyVistaQt 3D viewer ready.", "success")
            except Exception as exc:  # noqa: BLE001
                self._plotter = None
                self.log(f"3D viewer unavailable: {exc}", "warn")
                right.addWidget(self._viewer_placeholder(str(exc)), stretch=1)
        else:
            self.log(f"3D viewer unavailable: {err}", "warn")
            right.addWidget(self._viewer_placeholder(err), stretch=1)
        self._reproduce = ReproduceBar()
        right.addWidget(self._reproduce)
        wrap = QWidget()
        wrap.setLayout(right)
        root.addWidget(wrap, stretch=1)

        self._update_visibility()

    def _viewer_placeholder(self, err: str) -> QLabel:
        msg = QLabel(f"3D preview unavailable:<br><code>{err}</code><br><br>{_INSTALL_MESSAGE}")
        msg.setWordWrap(True)
        msg.setAlignment(Qt.AlignCenter)
        msg.setTextInteractionFlags(Qt.TextSelectableByMouse)
        return msg

    # -- small widget helpers ------------------------------------------------
    @staticmethod
    def _dspin(lo, hi, val, step=1.0, dec=2, suffix="") -> QDoubleSpinBox:
        return make_double_spinbox(val, lo, hi, step, dec, suffix=suffix)

    @staticmethod
    def _ispin(lo, hi, val) -> QSpinBox:
        s = QSpinBox()
        s.setRange(lo, hi)
        s.setValue(val)
        return s

    # -- toolbar (load/view tools) -------------------------------------------
    def _build_toolbar(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        for label, slot, icon in (
            ("Load mesh…", self._load_mesh, "fa5s.folder-open"),
            ("Clip", self._toggle_clip, "fa5s.cut"),
            ("Axes", self._toggle_axes, "fa5s.location-arrow"),
            ("Reset view", self._reset_view, "fa5s.expand"),
            ("Open output folder", self._open_output_folder, "fa5s.folder"),
        ):
            btn = QPushButton(label)
            btn.setIcon(theme.icon(icon))
            btn.clicked.connect(slot)
            bar.addWidget(btn)
        bar.addStretch(1)
        return bar

    # -- controls panel ------------------------------------------------------
    def _build_controls(self) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        # Wide enough to fit the controls without a horizontal scrollbar.
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(420)
        scroll.setMaximumWidth(480)
        panel = QWidget()
        scroll.setWidget(panel)
        layout = QVBoxLayout(panel)

        # Geometry
        geom = QGroupBox("Survey geometry")
        gform = QFormLayout(geom)
        self._mesh_type = QComboBox(); self._mesh_type.addItems(_MESH_TYPES)
        self._array_type = QComboBox(); self._array_type.addItems(_ARRAY_TYPES)
        self._mesh_type.currentTextChanged.connect(self._update_visibility)
        self._array_type.currentTextChanged.connect(self._update_visibility)
        self._mesh_engine = QComboBox(); self._mesh_engine.addItems(_MESH_ENGINES)
        self._mesh_engine.setToolTip(
            "Auto: prism for surface+topography, structured otherwise.\n"
            "Gmsh (tetrahedral): high-quality refined tet mesh (flat top; best for box / borehole / mild terrain).\n"
            "PyGIMLi prism: topography-conforming.\n"
            "Structured grid: fast regular grid.")
        gform.addRow("Mesh type", self._mesh_type)
        gform.addRow("Sensor array", self._array_type)
        gform.addRow("Mesh engine", self._mesh_engine)
        layout.addWidget(geom)

        layout.addWidget(self._build_array_pages())
        layout.addWidget(self._build_topo_group())
        layout.addWidget(self._build_box_group())
        layout.addWidget(self._build_meshparam_group())
        layout.addWidget(self._build_borehole_group())
        layout.addWidget(self._build_output_group())
        layout.addWidget(self._build_build_group())

        layout.addStretch(1)
        return scroll

    def _build_array_pages(self) -> QWidget:
        self._array_stack = QStackedWidget()

        # Surface grid
        p = QWidget(); f = QFormLayout(p)
        self._nx = self._ispin(2, 200, 10); self._ny = self._ispin(2, 200, 6)
        self._dx = self._dspin(0.1, 1000.0, 5.0, 0.5, 2, " m")
        self._dy = self._dspin(0.1, 1000.0, 5.0, 0.5, 2, " m")
        self._x_offset = self._dspin(-1e5, 1e5, 0.0, 1.0, 2, " m")
        self._y_offset = self._dspin(-1e5, 1e5, 0.0, 1.0, 2, " m")
        f.addRow("nx", self._nx); f.addRow("ny", self._ny)
        f.addRow("dx", self._dx); f.addRow("dy", self._dy)
        f.addRow("x offset", self._x_offset); f.addRow("y offset", self._y_offset)
        self._array_stack.addWidget(p)

        # Single borehole
        p = QWidget(); f = QFormLayout(p)
        self._bh_x = self._dspin(-1e5, 1e5, 0.0, 1.0, 2, " m")
        self._bh_y = self._dspin(-1e5, 1e5, 0.0, 1.0, 2, " m")
        self._bh_z_start = self._dspin(-1e5, 1e5, 0.0, 1.0, 2, " m")
        self._bh_z_end = self._dspin(-1e5, 1e5, -20.0, 1.0, 2, " m")
        self._bh_n = self._ispin(2, 300, 12)
        f.addRow("Borehole x", self._bh_x); f.addRow("Borehole y", self._bh_y)
        f.addRow("Top z", self._bh_z_start); f.addRow("Bottom z", self._bh_z_end)
        f.addRow("Sensors", self._bh_n)
        self._array_stack.addWidget(p)

        # Crosshole
        p = QWidget(); v = QVBoxLayout(p)
        cf = QFormLayout()
        self._cross_n = self._ispin(2, 8, 2)
        self._cross_n.valueChanged.connect(self._sync_borehole_table)
        self._cross_z_start = self._dspin(-1e5, 1e5, 0.0, 1.0, 2, " m")
        self._cross_z_end = self._dspin(-1e5, 1e5, -20.0, 1.0, 2, " m")
        self._cross_nelec = self._ispin(2, 300, 12)
        cf.addRow("Boreholes", self._cross_n)
        cf.addRow("Top z", self._cross_z_start); cf.addRow("Bottom z", self._cross_z_end)
        cf.addRow("Sensors / borehole", self._cross_nelec)
        v.addLayout(cf)
        v.addWidget(QLabel("Borehole positions:"))
        self._bh_table = QTableWidget(0, 2)
        self._bh_table.setHorizontalHeaderLabels(["x (m)", "y (m)"])
        self._bh_table.horizontalHeader().setStretchLastSection(True)
        self._bh_table.setMaximumHeight(160)
        v.addWidget(self._bh_table)
        self._sync_borehole_table()
        self._array_stack.addWidget(p)

        # Surface-to-borehole
        p = QWidget(); f = QFormLayout(p)
        self._s2b_n = self._ispin(2, 300, 24)
        self._s2b_dx = self._dspin(0.1, 1000.0, 2.0, 0.5, 2, " m")
        self._s2b_x0 = self._dspin(-1e5, 1e5, 0.0, 1.0, 2, " m")
        self._s2b_sy = self._dspin(-1e5, 1e5, 0.0, 1.0, 2, " m")
        self._s2b_sz = self._dspin(-1e5, 1e5, 0.0, 1.0, 2, " m")
        self._s2b_bhx = self._dspin(-1e5, 1e5, 20.0, 1.0, 2, " m")
        self._s2b_bhy = self._dspin(-1e5, 1e5, 0.0, 1.0, 2, " m")
        self._s2b_bhn = self._ispin(2, 300, 16)
        self._s2b_zs = self._dspin(-1e5, 1e5, 0.0, 1.0, 2, " m")
        self._s2b_ze = self._dspin(-1e5, 1e5, -30.0, 1.0, 2, " m")
        f.addRow("Surface sensors", self._s2b_n)
        f.addRow("Surface dx", self._s2b_dx); f.addRow("Surface start x", self._s2b_x0)
        f.addRow("Surface y", self._s2b_sy); f.addRow("Surface z", self._s2b_sz)
        f.addRow("Borehole x", self._s2b_bhx); f.addRow("Borehole y", self._s2b_bhy)
        f.addRow("Borehole sensors", self._s2b_bhn)
        f.addRow("Borehole top z", self._s2b_zs); f.addRow("Borehole bottom z", self._s2b_ze)
        self._array_stack.addWidget(p)

        return self._array_stack

    def _build_topo_group(self) -> QGroupBox:
        self._topo_group = QGroupBox("Topography")
        v = QVBoxLayout(self._topo_group)
        self._topo_type = QComboBox(); self._topo_type.addItems(_TOPO_TYPES)
        self._topo_type.currentTextChanged.connect(self._update_visibility)
        v.addWidget(self._topo_type)
        self._topo_stack = QStackedWidget()

        p = QWidget(); f = QFormLayout(p)
        self._z_flat = self._dspin(-1e5, 1e5, 0.0, 1.0, 2, " m")
        f.addRow("Surface elevation", self._z_flat)
        self._topo_stack.addWidget(p)

        p = QWidget(); f = QFormLayout(p)
        self._z_base = self._dspin(-1e5, 1e5, 100.0, 1.0, 2, " m")
        self._tilt_x = self._dspin(-1.0, 1.0, 0.05, 0.01, 3)
        self._tilt_y = self._dspin(-1.0, 1.0, 0.0, 0.01, 3)
        f.addRow("Base elevation", self._z_base)
        f.addRow("x slope", self._tilt_x); f.addRow("y slope", self._tilt_y)
        self._topo_stack.addWidget(p)

        p = QWidget(); f = QFormLayout(p)
        self._hill_base = self._dspin(-1e5, 1e5, 0.0, 1.0, 2, " m")
        self._hill_amp = self._dspin(-1e4, 1e4, 5.0, 0.5, 2, " m")
        self._hill_sigma = self._dspin(0.1, 1e4, 10.0, 1.0, 2, " m")
        self._hill_cx = self._dspin(-1e5, 1e5, 25.0, 1.0, 2, " m")
        self._hill_cy = self._dspin(-1e5, 1e5, 15.0, 1.0, 2, " m")
        f.addRow("Base z", self._hill_base); f.addRow("Amplitude", self._hill_amp)
        f.addRow("Width sigma", self._hill_sigma)
        f.addRow("Center x", self._hill_cx); f.addRow("Center y", self._hill_cy)
        self._topo_stack.addWidget(p)

        p = QWidget(); f = QFormLayout(p)
        self._topo_expr = QLineEdit("0.1*x - 0.05*y + 100")
        self._topo_expr.setToolTip("z = f(x, y). Allowed: x, y, np, sin, cos, exp, sqrt, abs, pi.")
        f.addRow("z = f(x, y)", self._topo_expr)
        self._topo_stack.addWidget(p)

        # From file (x, y, z): interpolate the surface from loaded points
        p = QWidget(); fv = QVBoxLayout(p)
        load_btn = QPushButton("Load topography file (x, y, z)…")
        load_btn.setIcon(theme.icon("fa5s.file-upload"))
        load_btn.clicked.connect(self._load_topo_file)
        fv.addWidget(load_btn)
        self._topo_file_label = QLabel("No file loaded. The surface is interpolated from the points.")
        self._topo_file_label.setWordWrap(True)
        fv.addWidget(self._topo_file_label)
        self._topo_stack.addWidget(p)

        v.addWidget(self._topo_stack)
        return self._topo_group

    def _load_topo_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load topography points (x, y, z)", "",
            "Points (*.csv *.txt *.dat *.xyz);;All files (*)")
        if not path:
            return
        try:
            table = io_utils.load_xyz_table(path, min_cols=3)
            self._topo_points = table[:, :3]
            self._topo_file_label.setText(
                f"{Path(path).name}: {len(self._topo_points)} points "
                f"(z {self._topo_points[:, 2].min():.1f} to {self._topo_points[:, 2].max():.1f} m)")
            self.log(f"Loaded topography: {len(self._topo_points)} points from {Path(path).name}", "success")
        except Exception as exc:  # noqa: BLE001
            self._topo_points = None
            self._topo_file_label.setText(f"Load failed: {exc}")
            self.log(f"Could not load topography file: {exc}", "error")

    def _build_box_group(self) -> QGroupBox:
        self._box_group = QGroupBox("Box dimensions")
        f = QFormLayout(self._box_group)
        self._box_length = self._dspin(1.0, 5000.0, 50.0, 1.0, 2, " m")
        self._box_width = self._dspin(1.0, 5000.0, 30.0, 1.0, 2, " m")
        self._box_height = self._dspin(1.0, 1000.0, 25.0, 1.0, 2, " m")
        f.addRow("Length x", self._box_length)
        f.addRow("Width y", self._box_width)
        f.addRow("Depth z", self._box_height)
        return self._box_group

    def _build_meshparam_group(self) -> QGroupBox:
        box = QGroupBox("Mesh parameters")
        f = QFormLayout(box)
        self._elec_refine = self._dspin(0.01, 50.0, 0.5, 0.1, 3, " m")
        self._attractor = self._dspin(0.1, 200.0, 5.0, 0.5, 2, " m")
        self._bound_refine = self._dspin(0.1, 100.0, 2.0, 0.5, 2, " m")
        self._para_depth = self._dspin(1.0, 500.0, 20.0, 1.0, 2, " m")
        self._dz_fine = self._dspin(0.05, 10.0, 0.5, 0.1, 2, " m")
        self._dz_coarse = self._dspin(0.5, 50.0, 2.0, 0.5, 2, " m")
        self._bound_ext = self._dspin(1.0, 3.0, 1.4, 0.1, 2)
        f.addRow("Sensor refinement", self._elec_refine)
        f.addRow("Attractor distance", self._attractor)
        f.addRow("Boundary refinement", self._bound_refine)
        f.addRow("Investigation depth", self._para_depth)
        f.addRow("Fine layer dz", self._dz_fine)
        f.addRow("Coarse layer dz", self._dz_coarse)
        f.addRow("Boundary extension", self._bound_ext)
        self._single_region = QCheckBox("Single region (one marker)")
        self._single_region.setToolTip(
            "Collapse the mesh to one marker instead of preserving the parameter-domain (2) / "
            "boundary (1) split.")
        f.addRow(self._single_region)
        return box

    def _build_borehole_group(self) -> QGroupBox:
        self._borehole_group = QGroupBox("Borehole survey domain")
        f = QFormLayout(self._borehole_group)
        self._bh_lat_pad = self._dspin(1.0, 500.0, 10.0, 1.0, 2, " m")
        self._bh_bot_pad = self._dspin(0.0, 500.0, 5.0, 1.0, 2, " m")
        self._bh_top_pad = self._dspin(0.0, 100.0, 2.0, 0.5, 2, " m")
        self._bh_hcell = self._dspin(0.1, 100.0, 2.0, 0.5, 2, " m")
        self._bh_vcell = self._dspin(0.1, 100.0, 1.0, 0.5, 2, " m")
        f.addRow("Lateral padding", self._bh_lat_pad)
        f.addRow("Bottom padding", self._bh_bot_pad)
        f.addRow("Top padding", self._bh_top_pad)
        f.addRow("Horizontal cell", self._bh_hcell)
        f.addRow("Vertical cell", self._bh_vcell)
        return self._borehole_group

    def _build_output_group(self) -> QGroupBox:
        box = QGroupBox("Mesh export")
        v = QVBoxLayout(box)
        f = QFormLayout()
        self._mesh_name = QLineEdit("my_3d_mesh")
        f.addRow("Mesh name", self._mesh_name)
        v.addLayout(f)
        row = QHBoxLayout()
        self._out_dir = QLineEdit("Managed by Project (one outputs/ folder per run)")
        self._out_dir.setReadOnly(True)
        self._out_dir.setToolTip(
            "Mesh files are saved under the active Project's unique run directory."
        )
        row.addWidget(self._out_dir, stretch=1)
        v.addLayout(row)
        fr = QHBoxLayout()
        self._fmt_bms = QCheckBox("BMS"); self._fmt_bms.setChecked(True)
        self._fmt_vtk = QCheckBox("VTK"); self._fmt_vtk.setChecked(True)
        self._fmt_csv = QCheckBox("Sensor CSV"); self._fmt_csv.setChecked(True)
        fr.addWidget(self._fmt_bms); fr.addWidget(self._fmt_vtk); fr.addWidget(self._fmt_csv)
        v.addLayout(fr)
        return box

    def _build_build_group(self) -> QGroupBox:
        """Keep the two build actions and their status together at the workflow end."""
        box = QGroupBox("Build & export")
        layout = QVBoxLayout(box)
        hint = QLabel(
            "1. Preview the sensor geometry.  2. Generate the mesh and save the selected exports."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#5a6a7a; font-size:8pt;")
        layout.addWidget(hint)

        buttons = QHBoxLayout()
        self._preview_btn = QPushButton("1. Preview sensors")
        self._preview_btn.setIcon(theme.icon("fa5s.eye"))
        self._preview_btn.clicked.connect(self._preview_sensors)
        buttons.addWidget(self._preview_btn)

        self._gen_btn = QPushButton("2. Generate mesh")
        self._gen_btn.setProperty("primary", True)
        self._gen_btn.setIcon(theme.icon("fa5s.cubes", color="#ffffff"))
        self._gen_btn.clicked.connect(self._generate_mesh)
        buttons.addWidget(self._gen_btn)
        layout.addLayout(buttons)

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        layout.addWidget(self._progress)

        self._info = QLabel("Configure the survey geometry, then preview the sensors.")
        self._info.setWordWrap(True)
        layout.addWidget(self._info)
        return box

    # -- visibility / config -------------------------------------------------
    @staticmethod
    def _fit_stack_to_current_page(stack: QStackedWidget) -> None:
        """Size a stacked panel to its visible page, not its tallest sibling."""
        page = stack.currentWidget()
        if page is not None:
            stack.setFixedHeight(page.sizeHint().height())

    def _update_visibility(self) -> None:
        mesh_type = self._mesh_type.currentText()
        array_type = self._array_type.currentText()
        self._array_stack.setCurrentIndex(_ARRAY_TYPES.index(array_type))
        self._topo_stack.setCurrentIndex(_TOPO_TYPES.index(self._topo_type.currentText()))
        self._fit_stack_to_current_page(self._array_stack)
        self._fit_stack_to_current_page(self._topo_stack)
        surface_topo = array_type == "Surface grid" and mesh_type == "Surface with topography"
        self._topo_group.setVisible(surface_topo)
        self._box_group.setVisible(mesh_type == "Box mesh")
        self._borehole_group.setVisible(array_type != "Surface grid")

    def _sync_borehole_table(self) -> None:
        n = self._cross_n.value()
        self._bh_table.setRowCount(n)
        for i in range(n):
            for j, default in ((0, i * 10.0), (1, 0.0)):
                if self._bh_table.item(i, j) is None:
                    self._bh_table.setItem(i, j, QTableWidgetItem(f"{default:g}"))

    def _read_boreholes(self):
        out = []
        for i in range(self._bh_table.rowCount()):
            try:
                x = float(self._bh_table.item(i, 0).text())
                y = float(self._bh_table.item(i, 1).text())
            except Exception:  # noqa: BLE001
                x, y = i * 10.0, 0.0
            out.append((x, y))
        return out

    def _default_output_dir(self) -> Path:
        base = self.state.output_dir or Path.cwd()
        return Path(base) / "mesh3d"

    def _browse_output_dir(self) -> None:
        path = select_directory(self, "Output directory", self._out_dir.text())
        if path:
            self._out_dir.setText(str(path))

    def _collect_config(self) -> dict:
        cfg = {
            "mesh_type": self._mesh_type.currentText(),
            "array_type": self._array_type.currentText(),
            "mesh_engine": self._mesh_engine.currentText(),
            "single_region": self._single_region.isChecked(),
            "electrode_refinement": self._elec_refine.value(),
            "attractor_distance": self._attractor.value(),
            "boundary_refinement": self._bound_refine.value(),
            "para_depth": self._para_depth.value(),
            "dz_fine": self._dz_fine.value(),
            "dz_coarse": self._dz_coarse.value(),
            "boundary_extension": self._bound_ext.value(),
            "borehole_lateral_padding": self._bh_lat_pad.value(),
            "borehole_bottom_padding": self._bh_bot_pad.value(),
            "borehole_top_padding": self._bh_top_pad.value(),
            "borehole_horizontal_cell": self._bh_hcell.value(),
            "borehole_vertical_cell": self._bh_vcell.value(),
            "output_dir": str(self._default_output_dir()),
        }
        array = cfg["array_type"]
        if array == "Surface grid":
            cfg.update(nx=self._nx.value(), ny=self._ny.value(), dx=self._dx.value(), dy=self._dy.value(),
                       x_offset=self._x_offset.value(), y_offset=self._y_offset.value())
        elif array == "Single borehole":
            cfg.update(bh_x=self._bh_x.value(), bh_y=self._bh_y.value(),
                       z_start=self._bh_z_start.value(), z_end=self._bh_z_end.value(), n_bh_elec=self._bh_n.value())
        elif array == "Crosshole":
            cfg.update(boreholes=self._read_boreholes(), z_start=self._cross_z_start.value(),
                       z_end=self._cross_z_end.value(), n_bh_elec=self._cross_nelec.value())
        else:
            cfg.update(n_surface_elec=self._s2b_n.value(), surface_dx=self._s2b_dx.value(),
                       surface_x0=self._s2b_x0.value(), surface_y=self._s2b_sy.value(), surface_z=self._s2b_sz.value(),
                       bh_x=self._s2b_bhx.value(), bh_y=self._s2b_bhy.value(), n_bh_elec=self._s2b_bhn.value(),
                       z_start=self._s2b_zs.value(), z_end=self._s2b_ze.value())
        # topography
        topo = self._topo_type.currentText()
        cfg["topography_type"] = topo if cfg["mesh_type"] == "Surface with topography" else "Flat"
        cfg.update(z_flat=self._z_flat.value(), z_base=self._z_base.value(),
                   tilt_x=self._tilt_x.value(), tilt_y=self._tilt_y.value(),
                   hill_base=self._hill_base.value(), hill_amp=self._hill_amp.value(),
                   hill_sigma=self._hill_sigma.value(), hill_cx=self._hill_cx.value(), hill_cy=self._hill_cy.value(),
                   topography_expr=self._topo_expr.text(), topography_points=self._topo_points)
        if cfg["mesh_type"] == "Box mesh":
            cfg.update(box_length=self._box_length.value(), box_width=self._box_width.value(),
                       box_height=self._box_height.value())
        return cfg

    def _selected_formats(self):
        fmts = []
        if self._fmt_bms.isChecked():
            fmts.append("BMS mesh (.bms)")
        if self._fmt_vtk.isChecked():
            fmts.append("VTK mesh (.vtk)")
        if self._fmt_csv.isChecked():
            fmts.append("Sensor CSV")
        return fmts

    # -- preview / generate --------------------------------------------------
    def _preview_sensors(self) -> None:
        try:
            cfg = self._collect_config()
            _, sensors = mesh3d_builder.build_electrodes(cfg)
        except Exception as exc:  # noqa: BLE001
            self.log(f"Sensor preview failed: {exc}", "error")
            self._info.setText(f"Preview failed: {exc}")
            return
        self._sensors = sensors
        self._render_sensors(sensors, cfg)
        x, y, z = sensors["x"], sensors["y"], sensors["z"]
        self._info.setText(
            f"{len(sensors)} sensors  ·  x [{x.min():.1f}, {x.max():.1f}]  "
            f"y [{y.min():.1f}, {y.max():.1f}]  z [{z.min():.2f}, {z.max():.2f}] m"
        )
        self.log(f"Previewed {len(sensors)} sensors.", "info")

    def _generate_mesh(self) -> None:
        cfg = self._collect_config()
        try:
            run = self.begin_persisted_run("mesh3d.build", "mesh3d.build")
        except Exception as exc:  # noqa: BLE001
            self.log(f"Could not prepare Project run: {exc}", "error")
            return
        output_dir = run.outputs_dir
        cfg["output_dir"] = str(output_dir)
        topography_points = cfg.pop("topography_points", None)
        inputs = {}
        if topography_points is not None:
            topo_path = run.inputs_dir / "mesh3d_topography_points.npy"
            np.save(topo_path, np.asarray(topography_points, dtype=float))
            inputs["topography_points"] = ArtifactRef.from_path(
                topo_path,
                artifact_id="mesh3d:topography_points",
                kind="topography_points",
                base_dir=run.run_dir,
            )
        cfg["output_formats"] = self._selected_formats()
        cfg["output_name"] = self._safe_name()
        spec = WorkflowSpec(
            workflow_id="mesh3d.build",
            inputs=inputs,
            parameters=cfg,
            metadata={"source": "qt"},
        )
        recipe_path, script_path = export_workflow_bundle(
            spec, run.run_dir, stem="mesh3d"
        )
        self._reproduce.set_bundle(recipe_path, script_path)
        self._mesh_recipe_path = str(recipe_path)
        self._gen_busy = BusyStateController([self._gen_btn])
        self._gen_busy.start()
        self._gen_btn.setText("Generating…")
        self._progress.setVisible(True)
        self._progress.setRange(0, 0)
        self.log("Generating 3D mesh…", "info")
        worker = WorkflowWorker(
            spec,
            RunContext(project_root=run.run_dir, output_dir=run.outputs_dir),
        )
        worker.logged.connect(lambda m: self.log(m, "info"))
        worker.succeeded.connect(lambda res: self._on_mesh_workflow_ok(cfg, res))
        worker.failed.connect(self._on_mesh_failed)
        worker.finished.connect(self._reset_gen_button)
        self._gen_worker = self.register_worker(worker)
        worker.start()

    def _on_mesh_workflow_ok(self, cfg: dict, result: WorkflowRunResult) -> None:
        try:
            self._on_mesh_generated(cfg, result.legacy_payload())
        finally:
            if hasattr(self.state, "update_workflow_result"):
                self.state.update_workflow_result(
                    self.module_key,
                    "mesh3d.build",
                    result.to_dict(),
                    recipe_path=self._mesh_recipe_path,
                )

    def _on_mesh_generated(self, cfg: dict, res: dict) -> None:
        mesh = res.get("mesh")
        sensors = res.get("electrodes")
        self._mesh = mesh
        self._sensors = sensors
        summary = mesh3d_builder.mesh_summary(mesh)
        self._render_mesh(mesh, sensors)
        # save selected outputs
        outputs = dict(res.get("outputs") or {})
        if not outputs:
            try:
                outputs = mesh3d_builder.save_outputs(
                    mesh, sensors, Path(cfg["output_dir"]), self._safe_name(), self._selected_formats())
            except Exception as exc:  # noqa: BLE001
                self.log(f"Saving outputs failed: {exc}", "warn")
        cells = summary.get("Cells", "?")
        nodes = summary.get("Nodes", "?")
        self._info.setText(
            f"<b>{res.get('generator')}</b><br>cells: {cells}  ·  nodes: {nodes}  ·  "
            f"sensors: {len(sensors)}"
        )
        self.log(f"Mesh generated: {cells} cells, {nodes} nodes ({res.get('generator')}).", "success")
        for key, path in outputs.items():
            self.log(f"Saved {key}: {path}", "info")
        # The mesh is in memory and on screen either way; only the file is missing.
        # Saying so beats a success line that quietly wrote nothing.
        if res.get("output_error"):
            self.log(
                f"The mesh is ready and displayed, but writing it to "
                f"{cfg.get('output_dir')} failed: {res['output_error']}. "
                "Use Export to save it elsewhere.", "warn")
        self.report_result({
            "generator": res.get("generator"), "n_sensors": int(len(sensors)),
            "cells": summary.get("Cells"), "nodes": summary.get("Nodes"),
            "outputs": outputs, "output_dir": cfg["output_dir"],
        })

    def _on_mesh_failed(self, message: str) -> None:
        self.fail_persisted_run(message, "mesh3d.build")
        self.log(f"Mesh generation failed: {message}", "error")
        self._info.setText(f"Generation failed: {message}")

    def _reset_gen_button(self) -> None:
        if self._gen_busy is not None:
            self._gen_busy.finish()
            self._gen_busy = None
        self._gen_btn.setText("Generate 3D mesh")
        self._progress.setVisible(False)

    # -- hidden 3D ERT forward action (AQUAH only) -------------------------
    def _set_ert_forward_param(self, key: str, value: Any) -> None:
        """Validate and store an agent-only 3D ERT forward parameter."""
        if key == "scheme":
            scheme = str(value)
            if scheme not in _ERT_FORWARD_SCHEMES:
                raise ValueError(f"scheme must be one of {list(_ERT_FORWARD_SCHEMES)}")
            self._ert_forward_params[key] = scheme
            return
        numeric = float(value)
        limits = {"background_res": (1.0, 100000.0), "noise": (0.0, 0.5)}
        low, high = limits[key]
        if not low <= numeric <= high:
            raise ValueError(f"{key} must be between {low:g} and {high:g}")
        self._ert_forward_params[key] = numeric

    def _run_ert_forward(self, marker_res=None) -> None:
        """Run the hidden 3D ERT forward action requested through AQUAH."""
        if self._mesh is None or self._sensors is None:
            self.log("Generate a 3D mesh first (Generate 3D mesh).", "warn")
            return
        try:
            run = self.begin_persisted_run("ert3d.forward", "ert3d.forward")
        except Exception as exc:  # noqa: BLE001
            self.log(f"Could not prepare Project run: {exc}", "error")
            return
        bundle_dir = run.run_dir
        input_dir = run.inputs_dir
        mesh_path = input_dir / "forward_mesh.bms"
        mesh_structure_path = input_dir / "forward_mesh.bms.structure.json"
        sensors_path = input_dir / "forward_sensors.csv"
        from PyHydroGeophysX.core.mesh_serialization import save_mesh_artifact

        save_mesh_artifact(self._mesh, mesh_path, mesh_structure_path)
        self._sensors.to_csv(sensors_path, index=False)
        spec = WorkflowSpec(
            workflow_id="ert3d.forward",
            inputs={
                "mesh": ArtifactRef.from_path(
                    mesh_path,
                    artifact_id="ert3d:mesh",
                    kind="pygimli_mesh",
                    base_dir=bundle_dir,
                ),
                "mesh_structure": ArtifactRef.from_path(
                    mesh_structure_path,
                    artifact_id="ert3d:mesh_structure",
                    kind="mesh_structure",
                    base_dir=bundle_dir,
                ),
                "sensors": ArtifactRef.from_path(
                    sensors_path,
                    artifact_id="ert3d:sensors",
                    kind="electrode_geometry",
                    base_dir=bundle_dir,
                ),
            },
            parameters={
                "scheme": self._ert_forward_params["scheme"],
                "background_res": self._ert_forward_params["background_res"],
                "marker_res": {
                    str(key): float(value) for key, value in (marker_res or {}).items()
                },
                "noise": self._ert_forward_params["noise"],
            },
            seed=42,
            metadata={"source": "qt", "mesh_roundtrip": "bms"},
        )
        recipe_path, script_path = export_workflow_bundle(
            spec, bundle_dir, stem="ert3d_forward"
        )
        self._reproduce.set_bundle(recipe_path, script_path)
        self._ert3d_recipe_path = str(recipe_path)
        self._progress.setVisible(True)
        self._progress.setRange(0, 0)
        self.log("Running agent-requested 3D ERT forward modeling…", "info")
        worker = WorkflowWorker(
            spec,
            RunContext(project_root=run.run_dir, output_dir=run.outputs_dir),
        )
        worker.logged.connect(lambda m: self.log(m, "info"))
        worker.succeeded.connect(self._on_ert3d_workflow_ok)
        worker.failed.connect(self._on_ert_forward_failed)
        worker.finished.connect(self._reset_ert_forward_progress)
        self._ert_fwd_worker = self.register_worker(worker)
        worker.start()

    def _on_ert3d_workflow_ok(self, result: WorkflowRunResult) -> None:
        if hasattr(self.state, "update_workflow_result"):
            self.state.update_workflow_result(
                self.module_key,
                "ert3d.forward",
                result.to_dict(),
                recipe_path=self._ert3d_recipe_path,
            )
        self._on_ert_forward_ok(result.legacy_payload())

    def _on_ert_forward_ok(self, result: dict) -> None:
        self._ert_fwd_result = result
        rmin, rmax = result.get("rhoa_min"), result.get("rhoa_max")
        rng = f"{rmin:.1f}-{rmax:.1f} ohm-m" if rmin is not None and rmax is not None else "n/a"
        self.log(f"3D ERT forward complete: {result.get('n_measurements')} measurements, "
                 f"rhoa {rng}. Saved {result.get('data_file')}.", "success")
        self.report_result({"ert3d_forward": result})
        vtk = result.get("vtk")
        if vtk:
            self.load_view_file(vtk)

    def _on_ert_forward_failed(self, message: str) -> None:
        self.fail_persisted_run(message, "ert3d.forward")
        self.log(f"3D ERT forward failed: {message}", "error")
        self._info.setText(f"3D ERT forward failed: {message}")

    def _reset_ert_forward_progress(self) -> None:
        self._progress.setVisible(False)

    def _safe_name(self) -> str:
        import re

        name = re.sub(r"[^A-Za-z0-9_.-]+", "_", self._mesh_name.text().strip())
        return name or "mesh3d"

    # -- rendering -----------------------------------------------------------
    def _overlay_sensors(self, sensors_df, labels: bool = False) -> None:
        import numpy as np

        pts = np.column_stack([
            np.asarray(sensors_df["x"], dtype=float),
            np.asarray(sensors_df["y"], dtype=float),
            np.asarray(sensors_df["z"], dtype=float),
        ])
        cloud = self._pv.PolyData(pts)
        self._plotter.add_mesh(cloud, color="#d7191c", render_points_as_spheres=True, point_size=12)
        if labels and len(pts) <= 60 and "n" in sensors_df:
            try:
                self._plotter.add_point_labels(
                    pts, [str(int(n)) for n in sensors_df["n"]],
                    font_size=11, point_size=1, text_color="#222222", always_visible=True)
            except Exception:  # noqa: BLE001
                pass

    def _render_sensors(self, sensors_df, cfg: dict) -> None:
        if self._plotter is None:
            return
        try:
            import numpy as np

            self._plotter.clear()
            self._overlay_sensors(sensors_df, labels=True)
            if cfg["array_type"] == "Surface grid" and cfg["mesh_type"] == "Surface with topography":
                topo = mesh3d_builder.topography_function(cfg)
                xs = np.asarray(sensors_df["x"], dtype=float)
                ys = np.asarray(sensors_df["y"], dtype=float)
                margin = max(cfg["dx"], cfg["dy"], 1.0) * 2.0
                gx = np.linspace(xs.min() - margin, xs.max() + margin, 40)
                gy = np.linspace(ys.min() - margin, ys.max() + margin, 40)
                gxx, gyy = np.meshgrid(gx, gy)
                gzz = np.vectorize(topo)(gxx, gyy)
                surf = self._pv.StructuredGrid(gxx, gyy, gzz)
                self._plotter.add_mesh(surf, cmap="gist_earth", opacity=0.35, show_scalar_bar=False)
            self._plotter.add_axes()
            self._plotter.reset_camera()
        except Exception as exc:  # noqa: BLE001
            self.log(f"Electrode render failed: {exc}", "warn")

    def _render_mesh(self, mesh, sensors_df) -> None:
        if self._plotter is None:
            return
        try:
            import tempfile

            tmp = Path(tempfile.mkdtemp()) / "mesh3d_view.vtk"
            mesh.exportVTK(str(tmp))
            pv_mesh = self._pv.read(str(tmp))
            self._plotter.clear()
            scalars = None
            for key in ("Marker", "marker", "Attribute", "region"):
                if key in pv_mesh.cell_data:
                    scalars = key
                    break
            self._plotter.add_mesh(
                pv_mesh, scalars=scalars, show_edges=True, edge_color="#555555",
                line_width=0.5, cmap="coolwarm", show_scalar_bar=bool(scalars))
            if sensors_df is not None:
                self._overlay_sensors(sensors_df, labels=False)
            self._plotter.add_axes()
            self._plotter.reset_camera()
        except Exception as exc:  # noqa: BLE001
            self.log(f"Mesh render failed: {exc}", "warn")

    # -- view tools ----------------------------------------------------------
    def _refresh_plotter(self) -> None:
        """Repaint the embedded VTK widget after a stacked-page transition."""
        if self._plotter is None:
            return
        try:
            interactor = getattr(self._plotter, "interactor", None)
            if interactor is not None:
                interactor.raise_()
                interactor.update()
            self._plotter.render()
        except Exception as exc:  # noqa: BLE001 - rendering backend specific
            self.log(f"3D viewer refresh failed: {exc}", "warn")

    def _load_mesh(self) -> None:
        if self._plotter is None:
            self.log("3D viewer unavailable.", "warn")
            return
        path, _ = QFileDialog.getOpenFileName(self, "Load mesh", "", _MESH_FILTER)
        if not path:
            return
        try:
            mesh = self._pv.read(path)
            self._plotter.clear()
            self._plotter.add_mesh(mesh, cmap="viridis", show_edges=False)
            self._plotter.add_axes()
            self._plotter.reset_camera()
            self.log(f"Loaded mesh {Path(path).name} ({getattr(mesh, 'n_points', 0)} points)", "success")
        except Exception as exc:  # noqa: BLE001
            self.log(f"Failed to load mesh '{Path(path).name}': {exc}", "error")

    def load_view_file(self, path: str) -> None:
        """Load and display a mesh / 3D volume file (e.g. a seismic 3D model).

        Called by the main window when another module asks to view its output in
        the 3D viewer. Picks a sensible cell/point scalar (Velocity, Marker, …).
        """
        p = Path(path)
        if not p.exists():
            self.log(f"File to view not found: {p}", "warn")
            return
        if self._plotter is None or self._pv is None:
            self.log(f"3D viewer unavailable; '{p.name}' was produced but cannot be "
                     f"shown here ({_INSTALL_MESSAGE}).", "warn")
            self._info.setText(f"3D viewer unavailable. File ready at:<br><code>{p}</code>")
            return
        try:
            mesh = self._pv.read(str(p))
            scalars = None
            for key in ("Velocity", "velocity", "Marker", "marker", "Elevation", "Attribute"):
                if key in getattr(mesh, "cell_data", {}) or key in getattr(mesh, "point_data", {}):
                    scalars = key
                    break
            self._plotter.clear()
            self._plotter.add_mesh(mesh, scalars=scalars, cmap="turbo",
                                   opacity=0.6 if scalars == "Velocity" else 1.0,
                                   show_edges=False, show_scalar_bar=bool(scalars))
            try:
                self._plotter.add_mesh(mesh.outline(), color="grey")
            except Exception:  # noqa: BLE001 - outline is cosmetic
                pass
            self._plotter.add_axes(); self._plotter.reset_camera()
            self._refresh_plotter()
            # A second repaint after pending resize/expose events prevents the
            # previous module's framebuffer from showing through QtInteractor.
            QTimer.singleShot(0, self._refresh_plotter)
            self._mesh = None  # this is a pyvista object, not a pygimli mesh
            self._info.setText(
                f"Viewing <b>{p.name}</b>  ·  {getattr(mesh, 'n_points', 0)} points"
                + (f"  ·  scalar: {scalars}" if scalars else ""))
            self.log(f"Loaded {p.name} into the 3D viewer (scalar: {scalars}).", "success")
        except Exception as exc:  # noqa: BLE001
            self.log(f"Failed to display '{p.name}': {exc}", "error")

    def _toggle_clip(self) -> None:
        if self._plotter is None or self._mesh is None:
            self.log("Generate or load a mesh first.", "debug")
            return
        try:
            import tempfile

            tmp = Path(tempfile.mkdtemp()) / "mesh3d_clip.vtk"
            self._mesh.exportVTK(str(tmp))
            pv_mesh = self._pv.read(str(tmp))
            self._plotter.clear()
            self._plotter.add_mesh_clip_plane(pv_mesh, cmap="coolwarm")
            self._plotter.add_axes()
            self._plotter.reset_camera()
            self.log("Added interactive clipping plane.", "success")
        except Exception as exc:  # noqa: BLE001
            self.log(f"Clipping plane not available: {exc}", "warn")

    def _toggle_axes(self) -> None:
        if self._plotter is None:
            return
        try:
            self._plotter.add_axes()
        except Exception as exc:  # noqa: BLE001
            self.log(f"Axes toggle failed: {exc}", "error")

    def _reset_view(self) -> None:
        if self._plotter is None:
            return
        try:
            self._plotter.reset_camera()
        except Exception as exc:  # noqa: BLE001
            self.log(f"Reset view failed: {exc}", "error")

    def _open_output_folder(self) -> None:
        out = self.state.module_results.get(self.module_key, {}).get("output_dir")
        out = out or str(self.state.results_store_root or self._default_output_dir())
        path = Path(out)
        if path.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
        else:
            self.log(f"Output folder does not exist yet: {path}", "warn")

    # -- AQUAH agent interface ----------------------------------------------
    def agent_describe(self) -> Dict[str, Any]:
        return {
            "module": self.module_key,
            "title": self.module_title,
            "state": self._agent_status(),
            "actions": [
                {"name": "set_params", "args": {"params": {"<key>": "value"}},
                 "desc": ("Set parameters. Geometry: mesh_type, array_type, mesh_engine, topo_type. "
                          "Surface grid: nx, ny, dx, dy, x_offset, y_offset. Box: box_length, box_width, "
                          "box_height. Topography: z_flat, z_base, tilt_x, tilt_y, hill_base, hill_amp, "
                          "hill_sigma, hill_cx, hill_cy, topo_expr. Mesh: elec_refine, attractor, "
                          "bound_refine, para_depth, dz_fine, dz_coarse, bound_ext, single_region. "
                          "Borehole: bh_x, bh_y, bh_z_start, bh_z_end, bh_n. Output: mesh_name, "
                          "fmt_bms, fmt_vtk, fmt_csv.")},
                {"name": "preview_sensors", "args": {},
                 "desc": "Build and preview the sensor layout for the current config."},
                {"name": "generate", "args": {},
                 "desc": "Generate the 3D mesh and save the selected output formats."},
                {"name": "run_ert_forward",
                 "args": {"scheme": "dd/wa/slm/wb", "background_res": "float",
                          "noise": "float", "marker_res": "{marker: resistivity} (optional)"},
                 "desc": "Hidden UI action: run 3D ERT forward modeling on the generated mesh."},
                {"name": "load_mesh", "args": {"path": "str"},
                 "desc": "Load a mesh / 3D volume file into the viewer."},
                {"name": "get_status", "args": {},
                 "desc": "Report the geometry settings and whether a sensor layout / mesh exists."},
            ],
        }

    def agent_apply(self, action: str, args: Dict[str, Any]) -> Dict[str, Any]:
        args = args or {}
        handlers = {
            "set_params": lambda: self._agent_set_params(args.get("params", args)),
            "preview_sensors": lambda: self._agent_preview_sensors(),
            "generate": lambda: self._agent_generate(),
            "run_ert_forward": lambda: self._agent_run_ert_forward(args),
            "load_mesh": lambda: self._agent_load_mesh(args.get("path")),
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
            "mesh_type": self._mesh_type.currentText(),
            "array_type": self._array_type.currentText(),
            "mesh_engine": self._mesh_engine.currentText(),
            "topo_type": self._topo_type.currentText(),
            "has_sensors": self._sensors is not None,
            "has_mesh": self._mesh is not None,
            "has_ert_forward": self._ert_fwd_result is not None,
            "output_dir": "Managed by active Project",
            "last_result_keys": sorted(last.keys()),
        }

    def _agent_preview_sensors(self) -> Dict[str, Any]:
        self._preview_sensors()
        return {"status": "ok",
                "sensors": int(len(self._sensors)) if self._sensors is not None else 0}

    def _agent_generate(self) -> Dict[str, Any]:
        self._generate_mesh()
        return {"status": "started", "message": "3D mesh generation started. Ask for status shortly."}

    def _agent_run_ert_forward(self, args: Dict[str, Any]) -> Dict[str, Any]:
        if self._mesh is None or self._sensors is None:
            return {"status": "failed", "error": "Generate a 3D mesh first (action 'generate')."}
        try:
            for key in ("scheme", "background_res", "noise"):
                if key in args:
                    self._set_ert_forward_param(key, args[key])
        except Exception as exc:  # noqa: BLE001
            return {"status": "failed", "error": str(exc)}
        marker_res = args.get("marker_res")
        if marker_res is not None and not isinstance(marker_res, dict):
            marker_res = None
        self._run_ert_forward(marker_res=marker_res)
        return {"status": "started",
                "message": "3D ERT forward modeling started. Ask for status shortly.",
                "scheme": self._ert_forward_params["scheme"]}

    def _agent_load_mesh(self, path: Any) -> Dict[str, Any]:
        if not path:
            return {"status": "failed", "error": "Provide 'path' to a mesh / volume file."}
        p = Path(str(path))
        if not p.exists():
            return {"status": "failed", "error": f"File not found: {p}"}
        self.load_view_file(str(p))
        return {"status": "ok", "viewing": str(p)}

    def _agent_set_params(self, params: Any) -> Dict[str, Any]:
        if not isinstance(params, dict):
            return {"status": "failed", "error": "Provide 'params' as a JSON object."}

        def set_combo(combo, value):
            items = [combo.itemText(i) for i in range(combo.count())]
            if str(value) not in items:
                raise ValueError(f"must be one of {items}")
            combo.setCurrentText(str(value))

        handlers = {
            "mesh_type": lambda v: set_combo(self._mesh_type, v),
            "array_type": lambda v: set_combo(self._array_type, v),
            "mesh_engine": lambda v: set_combo(self._mesh_engine, v),
            "topo_type": lambda v: set_combo(self._topo_type, v),
            "nx": lambda v: self._nx.setValue(int(v)),
            "ny": lambda v: self._ny.setValue(int(v)),
            "dx": lambda v: self._dx.setValue(float(v)),
            "dy": lambda v: self._dy.setValue(float(v)),
            "x_offset": lambda v: self._x_offset.setValue(float(v)),
            "y_offset": lambda v: self._y_offset.setValue(float(v)),
            "box_length": lambda v: self._box_length.setValue(float(v)),
            "box_width": lambda v: self._box_width.setValue(float(v)),
            "box_height": lambda v: self._box_height.setValue(float(v)),
            "z_flat": lambda v: self._z_flat.setValue(float(v)),
            "z_base": lambda v: self._z_base.setValue(float(v)),
            "tilt_x": lambda v: self._tilt_x.setValue(float(v)),
            "tilt_y": lambda v: self._tilt_y.setValue(float(v)),
            "hill_base": lambda v: self._hill_base.setValue(float(v)),
            "hill_amp": lambda v: self._hill_amp.setValue(float(v)),
            "hill_sigma": lambda v: self._hill_sigma.setValue(float(v)),
            "hill_cx": lambda v: self._hill_cx.setValue(float(v)),
            "hill_cy": lambda v: self._hill_cy.setValue(float(v)),
            "topo_expr": lambda v: self._topo_expr.setText(str(v)),
            "elec_refine": lambda v: self._elec_refine.setValue(float(v)),
            "attractor": lambda v: self._attractor.setValue(float(v)),
            "bound_refine": lambda v: self._bound_refine.setValue(float(v)),
            "para_depth": lambda v: self._para_depth.setValue(float(v)),
            "dz_fine": lambda v: self._dz_fine.setValue(float(v)),
            "dz_coarse": lambda v: self._dz_coarse.setValue(float(v)),
            "bound_ext": lambda v: self._bound_ext.setValue(float(v)),
            "single_region": lambda v: self._single_region.setChecked(bool(v)),
            "bh_x": lambda v: self._bh_x.setValue(float(v)),
            "bh_y": lambda v: self._bh_y.setValue(float(v)),
            "bh_z_start": lambda v: self._bh_z_start.setValue(float(v)),
            "bh_z_end": lambda v: self._bh_z_end.setValue(float(v)),
            "bh_n": lambda v: self._bh_n.setValue(int(v)),
            "mesh_name": lambda v: self._mesh_name.setText(str(v)),
            "fmt_bms": lambda v: self._fmt_bms.setChecked(bool(v)),
            "fmt_vtk": lambda v: self._fmt_vtk.setChecked(bool(v)),
            "fmt_csv": lambda v: self._fmt_csv.setChecked(bool(v)),
            "ert_scheme": lambda v: self._set_ert_forward_param("scheme", v),
            "ert_background_res": lambda v: self._set_ert_forward_param("background_res", v),
            "ert_noise": lambda v: self._set_ert_forward_param("noise", v),
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
