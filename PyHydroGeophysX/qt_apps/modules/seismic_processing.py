"""Seismic processing module.

Loads real field formats (SEG-Y, Geometrics DAT, SEG-2) by reusing
``PyHydroGeophysX.data_processing.seismic`` plus generic ``.npy/.csv`` arrays,
lets the user browse shot gathers, apply display/QC processing (gain, clip,
polarity, trace normalization, AGC), pick first arrivals (manual + assisted),
and export picks to CSV and a PyGIMLi travel-time ``.dat`` file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import (
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
from PySide6.QtCore import Qt, QThread, Signal

from PyHydroGeophysX.qt_apps import io_utils, theme
from PyHydroGeophysX.qt_apps.modules.base import BaseModule, LogFn
from PyHydroGeophysX.qt_apps.qt_utils import Debouncer
from PyHydroGeophysX.qt_apps.widgets.mesh_view import MeshResultView
from PyHydroGeophysX.qt_apps.widgets.seismic_viewer import SeismicViewer, first_arrival_onsets
from PyHydroGeophysX.qt_apps.workers import TaskWorker

try:
    from PyHydroGeophysX.data_processing.seismic import (
        FirstBreakPick,
        apply_agc,
        export_first_breaks,
        first_breaks_to_traveltime,
        normalize_traces,
        pick_first_breaks,
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


class SRTInversionWorker(QThread):
    """Build a travel-time dataset from picks and run a pygimli SRT inversion."""

    succeeded = Signal(dict)
    failed = Signal(str)
    logged = Signal(str)

    def __init__(self, picks, spacing: float, out_dir: str) -> None:
        super().__init__()
        self._picks = picks
        self._spacing = spacing
        self._out_dir = out_dir

    def run(self) -> None:  # noqa: D401
        try:
            import pygimli as pg
            import pygimli.physics.traveltime as tt
            from PyHydroGeophysX.data_processing.seismic import first_breaks_to_traveltime

            dat = str(Path(self._out_dir) / "traveltime.dat")
            self.logged.emit("Building travel-time data from picks…")
            first_breaks_to_traveltime(self._picks, dat, receiver_spacing=self._spacing)
            try:
                data = tt.load(dat)
            except Exception:  # noqa: BLE001
                data = pg.DataContainer(dat, "s g")
            self.logged.emit(f"Inverting {int(data.size())} travel times…")
            mgr = tt.TravelTimeManager(data)
            mgr.invert(data, verbose=False)
            mesh = mgr.paraDomain
            # pygimli's TravelTimeManager inverts for velocity (m/s) and exposes it
            # as ``mgr.velocity``; ``mgr.model`` is the same velocity, NOT slowness.
            try:
                velocity = np.asarray(mgr.velocity, dtype=float)
            except Exception:  # noqa: BLE001 - older pygimli may return slowness
                m = np.asarray(mgr.model, dtype=float)
                velocity = m if np.nanmedian(m) > 1.0 else 1.0 / m
            vtk_path = ""
            try:
                mesh["velocity"] = velocity
                vtk_path = str(Path(self._out_dir) / "velocity_model.vtk")
                mesh.exportVTK(vtk_path)
            except Exception:  # noqa: BLE001
                vtk_path = ""
            self.succeeded.emit({"mgr": mgr, "n": int(data.size()), "vtk": vtk_path})
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


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
        self._srt_worker: Optional[SRTInversionWorker] = None
        self._load_worker: Optional[TaskWorker] = None
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
        self._tt_widget.setLabel("bottom", "offset = geophone − shot (m)")
        self._tt_widget.setLabel("left", "travel time (ms)")
        self._tt_plot = self._tt_widget.getPlotItem()
        self._tt_plot.addLegend()
        self._center_tabs.addTab(self._tt_widget, "Travel-time")
        self._vel_view = MeshResultView()
        self._center_tabs.addTab(self._vel_view, "Velocity model")
        root.addWidget(self._center_tabs, stretch=1)
        root.addWidget(self._build_controls())
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

        srt = QGroupBox("SRT inversion (all shots)")
        srtbox = QVBoxLayout(srt)
        acc = QLabel("Picks are kept per shot record; pick across shots, set each shot's x, then invert.")
        acc.setWordWrap(True)
        acc.setStyleSheet("color:#5a6a7a; font-size:8pt;")
        srtbox.addWidget(acc)
        self._srt_btn = QPushButton("Run SRT inversion")
        self._srt_btn.setProperty("primary", True)
        self._srt_btn.setIcon(theme.icon("fa5s.layer-group", color="#ffffff"))
        self._srt_btn.clicked.connect(self._run_srt)
        srtbox.addWidget(self._srt_btn)
        self._srt_progress = QProgressBar()
        self._srt_progress.setVisible(False)
        srtbox.addWidget(self._srt_progress)
        self._srt_export_btn = QPushButton("Export velocity model (npy + mesh + VTK)…")
        self._srt_export_btn.setIcon(theme.icon("fa5s.cube"))
        self._srt_export_btn.setEnabled(False)
        self._srt_export_btn.clicked.connect(self._export_velocity_model)
        srtbox.addWidget(self._srt_export_btn)
        layout.addWidget(srt)

        layout.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumWidth(326)
        scroll.setWidget(panel)
        return scroll

    # -- loading -------------------------------------------------------------
    def _load_gather(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load seismic data", "", _FILE_FILTER)
        if not path:
            return
        self._load_btn.setEnabled(False)
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
        self._load_btn.setEnabled(True)
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
        self._shot_x.setValue(self._shot_pos.get(record, self._geo_start.value()))
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
        receiver_x = self._geo_start.value() + trace * self._spacing.value()
        shot_x = self._shot_x.value()
        if not _SEISMIC_OK:
            return {"trace": trace, "sample": sample, "time_s": sample * dt, "value": value}
        record = self._current_record if self._current_record is not None else 1
        return FirstBreakPick(
            source_id=int(record), receiver_id=trace + 1, time_s=float(sample * dt),
            source_x=float(shot_x), source_z=0.0, receiver_x=float(receiver_x), receiver_z=0.0,
            field_record=int(record), trace_number=trace + 1, trace_index=trace, amplitude=float(value),
        )

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

    def _update_tt_qc(self) -> None:
        if not hasattr(self, "_tt_plot"):
            return
        self._tt_plot.clear()
        picks = self._all_first_breaks()
        if not picks:
            return
        from collections import defaultdict

        by_shot = defaultdict(list)
        for p in picks:
            by_shot[int(p.source_id)].append((float(p.receiver_x - p.source_x), float(p.time_s) * 1000.0))
        colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b", "#e377c2"]
        for i, shot in enumerate(sorted(by_shot)):
            pts = sorted(by_shot[shot])
            xs = [a for a, _ in pts]
            ys = [b for _, b in pts]
            color = colors[i % len(colors)]
            self._tt_plot.plot(xs, ys, pen=pg.mkPen(color, width=1.5), symbol="o", symbolSize=5,
                               symbolBrush=color, symbolPen=None, name=f"shot {shot}")

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
        geo_start = self._geo_start.value()
        spacing = self._spacing.value()
        out = []
        for record, picks in self._all_picks.items():
            shot_x = self._shot_pos.get(record, geo_start)
            for trace, pick in picks.items():
                time_s = pick.time_s if hasattr(pick, "time_s") else pick["time_s"]
                if not np.isfinite(time_s) or time_s <= 0:
                    continue
                out.append(FirstBreakPick(
                    source_id=int(record), receiver_id=int(trace) + 1, time_s=float(time_s),
                    source_x=float(shot_x), source_z=0.0,
                    receiver_x=float(geo_start + int(trace) * spacing), receiver_z=0.0,
                    field_record=int(record), trace_number=int(trace) + 1,
                    trace_index=int(trace), amplitude=0.0))
        return out

    def _run_srt(self) -> None:
        picks = self._all_first_breaks()
        n_shots = sum(1 for v in self._all_picks.values() if v)
        if len(picks) < 8:
            self.log("Pick first breaks on at least a couple of shots before SRT inversion.", "warn")
            return
        out = io_utils.ensure_dir(Path(str(self.state.output_dir or Path.cwd())) / "srt_results")
        self._srt_btn.setEnabled(False)
        self._srt_btn.setText("Inverting…")
        self._srt_progress.setVisible(True)
        self._srt_progress.setRange(0, 0)
        self.log(f"Running SRT inversion: {len(picks)} picks from {n_shots} shot(s).", "info")
        self._srt_worker = SRTInversionWorker(picks, self._spacing.value(), str(out))
        self._srt_worker.logged.connect(lambda m: self.log(m, "info"))
        self._srt_worker.succeeded.connect(self._on_srt_ok)
        self._srt_worker.failed.connect(self._on_srt_failed)
        self._srt_worker.finished.connect(self._reset_srt_button)
        self.register_worker(self._srt_worker)
        self._srt_worker.start()

    def _on_srt_ok(self, result: dict) -> None:
        mgr = result.get("mgr")
        self._srt_mgr = mgr
        if mgr is not None:
            self._vel_view.show_model(mgr, kind="srt")
            self._center_tabs.setCurrentWidget(self._vel_view)
            self._srt_export_btn.setEnabled(True)
        vtk = result.get("vtk")
        if vtk:
            self.log(f"Saved velocity mesh to {vtk}", "info")
        self.log("SRT inversion complete.", "success")
        self.report_result({"velocity_vtk": vtk, "num_traveltimes": result.get("n")})

    def _export_velocity_model(self) -> None:
        mgr = getattr(self, "_srt_mgr", None)
        if mgr is None:
            self.log("Run SRT inversion first.", "warn")
            return
        folder = QFileDialog.getExistingDirectory(
            self, "Export velocity model to folder", str(self.state.output_dir or Path.cwd()))
        if not folder:
            return
        try:
            out = io_utils.ensure_dir(Path(folder))
            mesh = mgr.paraDomain
            try:
                velocity = np.asarray(mgr.velocity, dtype=float)
            except Exception:  # noqa: BLE001 - older pygimli may expose model only
                velocity = np.asarray(mgr.model, dtype=float)
            np.save(out / "velocity_model.npy", velocity)
            mesh.save(str(out / "velocity_mesh.bms"))
            mesh["velocity"] = velocity
            mesh.exportVTK(str(out / "velocity_model.vtk"))
            self.log(f"Exported velocity model (npy + bms + vtk) to {out}", "success")
        except Exception as exc:  # noqa: BLE001
            self.log(f"Velocity model export failed: {exc}", "error")

    def _on_srt_failed(self, message: str) -> None:
        self.log(f"SRT inversion failed: {message}", "error")

    def _reset_srt_button(self) -> None:
        self._srt_btn.setEnabled(True)
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
                {"name": "set_geometry", "args": {"spacing": "float", "geophone_start": "float", "shot_x": "float"},
                 "desc": "Set geophone spacing (m), geophone-0 x (m), and this record's shot x (m, may be negative)."},
                {"name": "set_params", "args": {"params": {"<key>": "value"}},
                 "desc": ("Set processing/pick params. Keys: sta_lta_ratio, gain (slider 1-100), "
                          "clip_percentile, agc_window_ms, flip_polarity, normalize, agc.")},
                {"name": "auto_pick", "args": {},
                 "desc": "Auto-pick first breaks on the current record (STA/LTA)."},
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
                {"name": "run_srt", "args": {},
                 "desc": "Run SRT travel-time tomography from all picked shots (needs >=8 picks total)."},
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
            "set_params": lambda: self._agent_set_params(args.get("params", args)),
            "auto_pick": lambda: self._agent_auto_pick(),
            "review_picks": lambda: self._agent_review_picks(),
            "set_pick": lambda: self._agent_set_pick(args),
            "delete_pick": lambda: self._agent_delete_pick(args),
            "list_picks": lambda: self._agent_list_picks(),
            "clear_picks": lambda: self._agent_clear_picks(),
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
            if "shot_x" in args:
                self._shot_x.setValue(float(args["shot_x"])); applied["shot_x"] = args["shot_x"]
        except Exception as exc:  # noqa: BLE001
            return {"status": "failed", "error": str(exc)}
        if not applied:
            return {"status": "failed", "error": "Provide spacing, geophone_start, and/or shot_x."}
        return {"status": "ok", "applied": applied}

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
        return {
            "status": "awaiting_user",
            "current_record": self._current_record,
            "picks": len(self._picks),
            "auto": auto,
            "manual": manual,
            "suspect_traces": self._suspect_pick_traces(),
            "message": (
                "Auto-picks are in and Manual pick mode is ON. Correct any bad traces by clicking / "
                "Ctrl+dragging in the plot, or ask me to set_pick / delete_pick a trace. Say 'continue' "
                "when the picks look good and I will run the SRT inversion."
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

    def _agent_run_srt(self) -> Dict[str, Any]:
        picks = self._all_first_breaks()
        if len(picks) < 8:
            return {"status": "failed", "error": "Need at least 8 first-break picks across shots.",
                    "picks": len(picks), "hint": "Auto-pick first breaks on a couple of shots first."}
        self._run_srt()
        return {"status": "started", "message": "SRT inversion started. Ask for status shortly.",
                "picks": len(picks)}
