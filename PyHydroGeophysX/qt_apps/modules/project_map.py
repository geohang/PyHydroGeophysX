"""Project-wide map membership and linked scientific result inspection.

A wells layer (added from the Boreholes page) is drawn as labelled wells; clicking
one sets the well's lithology, logs and water level beside every survey on the map
that reaches it, each read down the vertical there (``project_map.profile_at``).
"""
from __future__ import annotations

from collections import Counter

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog,
    QHBoxLayout, QInputDialog, QLabel, QMessageBox, QPushButton, QProgressBar, QSpinBox, QSplitter,
    QStackedWidget, QTreeWidget, QHeaderView, QTreeWidgetItem, QVBoxLayout, QWidget)

from .base import BaseModule
from PyHydroGeophysX.data_processing import boreholes as bh
from PyHydroGeophysX.qt_apps.project_map import (
    ProjectMapStore, em_result, profile_at, wells_data)
from PyHydroGeophysX.qt_apps.widgets import colormaps as cmaps
from PyHydroGeophysX.qt_apps.widgets.color_range import ColorRange
from PyHydroGeophysX.qt_apps.widgets.flow_layout import FlowLayout, group as control_group
from PyHydroGeophysX.qt_apps.widgets import length_units
from PyHydroGeophysX.qt_apps.widgets.readout import toolbar_row
from PyHydroGeophysX.qt_apps.workers import TaskWorker
from PyHydroGeophysX.visualization.axis_units import (
    set_length_axis, set_section_axes, to_display_length)
from PyHydroGeophysX.visualization.basemap import TILE_SOURCES, basemap_image
from PyHydroGeophysX.qt_apps import theme


COLORS = {'ERT': '#ff9500', 'EM': '#30b0c7', 'Seismic': '#af52de',
          'Gravity': '#ff2d55', 'Magnetics': '#34c759', 'Boreholes': '#3a3a3c'}

# Plan interpolation offered for any slice that carries one value per map
# position: TEM/AEM depth slices, recovered grid layers and imported point
# products alike. 'Points only' keeps the map showing measurements and nothing
# else, which is why it stays the default.
SURFACES = [('Points only', None), ('Kriging (ordinary)', 'kriging'),
            ('Inverse distance', 'idw'), ('Linear (triangulation)', 'linear'),
            ('Cubic (triangulation)', 'cubic'), ('Nearest neighbour', 'nearest'),
            ('Thin-plate spline', 'rbf')]

SURFACE_HELP = ('Choose a slice and method, then click Interpolate to draw a smooth map.\n'
                'Kriging fits a semivariogram and reports its own variance; the other\n'
                'methods are deterministic. Resistivity is interpolated in log space.\n'
                'Cells outside the convex hull of the stations are always blanked.')


class SurfaceWorker(TaskWorker):
    """Compute only numerical arrays in the worker; plotting stays on the UI thread."""

    progressed = Signal(int, str)

    def __init__(self, xy, values, options):
        from PyHydroGeophysX.core.plan_interpolation import plan_grid
        super().__init__(plan_grid, xy, values, **options)
        self._kwargs.update(progress=self.progressed.emit, cancelled=self.is_cancelled)


class SteadySpinBox:
    """Mixin: ignore the wheel unless focused.

    The map zooms on scroll, so a wheel that drifts onto this row would
    otherwise re-grid the slice with settings nobody chose -- a few notches of
    blanking is enough to cut a survey down to ribbons along its lines.
    """

    def wheelEvent(self, event):
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()


class GridSpinBox(SteadySpinBox, QSpinBox):
    pass


class DistanceSpinBox(SteadySpinBox, QDoubleSpinBox):
    pass


class VariogramDialog(QDialog):
    """The experimental semivariogram and the model kriging actually used."""

    def __init__(self, result, entry, layer, parent=None):
        super().__init__(parent)
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure
        from PyHydroGeophysX.core.plan_interpolation import variogram_function
        fit = result['variogram']
        self.setWindowTitle('Kriging variogram')
        self.resize(560, 420)
        layout = QVBoxLayout(self)
        figure = Figure(figsize=(5.5, 3.4), layout='constrained')
        canvas = FigureCanvasQTAgg(figure)
        bar, _toolbar = toolbar_row(canvas, self)
        layout.addWidget(bar)
        layout.addWidget(canvas, 1)
        axes = figure.add_subplot(111)
        lags, gamma = np.asarray(fit['lags']), np.asarray(fit['gamma'])
        axes.plot(lags, gamma, 'o', color='#30b0c7', label='Experimental')
        fine = np.linspace(0, lags.max() * 1.05, 200)
        model = variogram_function(fit['model'], fit['nugget'], fit['sill'], fit['range'])
        axes.plot(fine, model(fine), '-', color='#ff9500', lw=2, label=f"{fit['model']} model")
        axes.axhline(fit['sill'], ls=':', color='#8e8e93')
        axes.axvline(fit['range'], ls=':', color='#8e8e93')
        # The variogram describes the space that was interpolated, which is log10
        # for resistivity and the product's own units for everything else.
        units = 'log10 Ω·m' if result['log_values'] else entry.get('units', '')
        axes.set(xlabel='Separation (map metres)', ylabel=f'Semivariance ({units})$^2$',
                 title=f"{entry['name']} · {layer}")
        axes.legend(fontsize=8)
        axes.grid(alpha=.25, linestyle=':')
        summary = QLabel(
            f"model {fit['model']} · nugget {fit['nugget']:.4g} · sill {fit['sill']:.4g} · "
            f"range {fit['range']:.4g} m · fit RMSE {fit['rmse']:.3g}\n"
            'Separations are measured in the map frame. Web Mercator metres are '
            'stretched by 1/cos(latitude), so a geographic range reads larger than the '
            'ground distance it represents.')
        summary.setWordWrap(True)
        theme.set_tone(summary, "hint")
        layout.addWidget(summary)


class ProjectMapModule(BaseModule):
    module_key = 'project_map'
    module_title = 'Project Map'

    def __init__(self, state, log, parent=None):
        super().__init__(state, log, parent)
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure
        from PyHydroGeophysX.qt_apps.widgets.em_overview_view import EMOverviewView
        self._entries, self._arrays, self._artists = [], {}, {}
        self._store = None
        self._selected = None
        self._slices_for = None      # the survey whose slices the Slice list holds
        self._tile = None
        self._tile_worker = None
        self._root = None
        self._last_frame = None
        # Gridding a slice costs real time, and the map redraws on every layer,
        # line and visibility change, so each surface is kept under the settings
        # that produced it.
        self._surfaces = {}
        self._surface = None
        self._surface_note = ''
        self._surface_request = None
        self._surface_job = None
        self._surface_token = 0
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)
        self._note = QLabel('Add results from any survey method using Add to Map… or import a point product.')
        self._note.setWordWrap(True)
        theme.set_tone(self._note, "hint")
        split = QSplitter(Qt.Horizontal)
        root.addWidget(split, 1)
        root.addWidget(self._note)
        left = QWidget()
        left.setObjectName('MapLayersPanel')
        ll = QVBoxLayout(left)
        ll.setContentsMargins(6, 6, 6, 6)
        ll.setSpacing(4)
        layers_title = QLabel('SURVEY LAYERS')
        layers_title.setObjectName('MapSectionTitle')
        ll.addWidget(layers_title)
        self._frame = QComboBox()
        self._frame.addItem('Geographic surveys', 'geographic')
        self._frame.addItem('Local project coordinates', 'local')
        self._method = QComboBox()
        self._method.addItems(['All methods', 'ERT', 'EM', 'Seismic'])
        ll.addWidget(self._frame)
        ll.addWidget(self._method)
        self._list = QTreeWidget()
        self._list.setHeaderLabels(['Survey', 'Method'])
        self._list.setRootIsDecorated(False)
        self._list.setAlternatingRowColors(True)
        self._list.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self._list.header().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        ll.addWidget(self._list, 1)
        self._details = QLabel('')
        self._details.setWordWrap(True)
        self._details.setTextInteractionFlags(Qt.TextSelectableByMouse)
        ll.addWidget(self._details)
        actions = QHBoxLayout()
        for text, callback in [('Rename', self._rename), ('Remove', self._remove), ('Refresh', self.refresh)]:
            button = QPushButton(text)
            button.clicked.connect(callback)
            actions.addWidget(button)
        ll.addLayout(actions)
        for text, callback in [('Add available result…', self._add_resource),
                               ('Import point product CSV…', self._import_points)]:
            button = QPushButton(text)
            button.clicked.connect(callback)
            ll.addWidget(button)
        split.addWidget(left)
        right = QSplitter(Qt.Vertical)
        split.addWidget(right)
        map_panel = QWidget()
        map_panel.setObjectName('ProjectMapPanel')
        ml = QVBoxLayout(map_panel)
        ml.setContentsMargins(4, 4, 4, 4)
        ml.setSpacing(2)
        toolbar = QHBoxLayout()
        self._source = QComboBox()
        self._source.addItems(list(TILE_SOURCES))
        self._load_tiles = QPushButton('Load basemap')
        self._load_tiles.setToolTip('Fetch imagery for the visible extent. Optional; surveys remain available offline.')
        self._load_tiles.clicked.connect(self._fetch_tiles)
        self._depth = QComboBox()
        self._depth.addItem('Survey locations', None)
        self._depth.currentIndexChanged.connect(self._draw_map)
        toolbar.addWidget(self._source)
        toolbar.addWidget(self._load_tiles)
        fit = QPushButton('Fit surveys')
        fit.clicked.connect(lambda: self._draw_map(fit=True))
        toolbar.addWidget(fit)
        save = QPushButton('Export map PNG…')
        save.clicked.connect(self._export_png)
        toolbar.addWidget(save)
        ml.addLayout(toolbar)
        # Wraps in a narrow panel instead of widening the page (widgets.flow_layout).
        slices = FlowLayout(spacing=6)
        slices.addWidget(control_group('Slice:', self._depth))
        # The colour map of the slice on the map, beside the slice it colours:
        # turbo for a resistivity slice and coolwarm for other layers until one
        # is chosen for the map. Off while no slice is drawn.
        shared = cmaps.colormap_settings(self.state)
        self._map_colormap = cmaps.ColormapChooser(cmaps.MAP_LAYER, 'coolwarm', shared=shared)
        self._map_colormap.colormapChanged.connect(self._draw_map)
        self._map_colormap.setEnabled(False)
        slices.addWidget(self._map_colormap)
        # And its colour limits. Locked, they hold through the slices and across
        # surveys of the same quantity, so two depths read on one scale.
        self._map_range = ColorRange(decimals=6, what="the slice's colours")
        self._map_range.changed.connect(self._draw_map)
        self._map_range.setEnabled(False)
        self._map_quantity = None       # what the map's range was set for
        slices.addWidget(self._map_range)
        # Wraps in a narrow panel instead of widening the page (widgets.flow_layout).
        surface = FlowLayout(spacing=6)
        self._interp = QComboBox()
        for text, value in SURFACES:
            self._interp.addItem(text, value)
        self._interp.setToolTip(SURFACE_HELP)
        self._interp.currentIndexChanged.connect(self._surface_changed)
        surface.addWidget(control_group('Surface:', self._interp))
        self._interp_res = GridSpinBox()
        self._interp_res.setRange(40, 400)
        self._interp_res.setValue(140)
        self._interp_res.setSuffix(' cells')
        self._interp_res.setKeyboardTracking(False)
        self._interp_res.setToolTip('Cells across the longer map axis. More cells give a finer map\n'
                                   'but take longer to calculate. Click Interpolate after changing settings.')
        self._interp_res.valueChanged.connect(self._draw_map)
        surface.addWidget(self._interp_res)
        self._interp_blank = DistanceSpinBox()
        self._interp_blank.setRange(0, 1e7)
        self._interp_blank.setDecimals(0)
        self._interp_blank.setSuffix(' m blanking')
        # Typing "60" must not re-grid at 6 on the way, and 0 has to read as the
        # off switch it is rather than as a zero-metre radius.
        self._interp_blank.setKeyboardTracking(False)
        self._interp_blank.setSpecialValueText('no blanking')
        self._interp_blank.setToolTip(
            'Also blank cells farther than this from any station, so a wide line\n'
            'spacing does not read as coverage. Set it near the line spacing:\n'
            'well below that it cuts into the survey and leaves ribbons along the\n'
            'lines. The caption under the map reports what the outline needs.')
        self._interp_blank.valueChanged.connect(self._draw_map)
        surface.addWidget(self._interp_blank)
        self._stations = QCheckBox('Stations')
        self._stations.setToolTip(
            'Draw this survey\'s stations and line traces over its interpolated\n'
            'surface. Off by default: the surface already carries those values,\n'
            'and the markers cover the image. Other surveys keep their traces.')
        self._stations.toggled.connect(self._draw_map)
        surface.addWidget(self._stations)
        self._interpolate_button = QPushButton('Interpolate')
        self._interpolate_button.setToolTip(
            'Create a surface for the selected slice using the settings above.\n'
            'You can keep using the map while it calculates. Higher resolution takes longer.')
        self._interpolate_button.setEnabled(False)
        self._interpolate_button.clicked.connect(self._start_surface)
        surface.addWidget(self._interpolate_button)
        self._variogram_button = QPushButton('Variogram…')
        self._variogram_button.setEnabled(False)
        self._variogram_button.clicked.connect(self._show_variogram)
        surface.addWidget(self._variogram_button)
        self._export_grid = QPushButton('Export grid…')
        self._export_grid.setEnabled(False)
        self._export_grid.clicked.connect(self._export_surface)
        surface.addWidget(self._export_grid)
        self._fig = Figure(figsize=(9, 4), layout='constrained')
        self._canvas = FigureCanvasQTAgg(self._fig)
        ml.addLayout(slices)
        ml.addLayout(surface)
        progress_row = QHBoxLayout()
        self._interpolation_status = QLabel('Choose a slice and interpolation method to get started.')
        self._interpolation_status.setWordWrap(True)
        progress_row.addWidget(self._interpolation_status, 1)
        self._interpolation_progress = QProgressBar()
        self._interpolation_progress.setRange(0, 100)
        self._interpolation_progress.setFixedWidth(160)
        self._interpolation_progress.setVisible(False)
        progress_row.addWidget(self._interpolation_progress)
        self._cancel_interpolation = QPushButton('Cancel')
        self._cancel_interpolation.setToolTip('Stop interpolation after the current calculation batch.')
        self._cancel_interpolation.clicked.connect(lambda: self._cancel_surface())
        self._cancel_interpolation.setVisible(False)
        progress_row.addWidget(self._cancel_interpolation)
        ml.addLayout(progress_row)
        # Zoom, pan, Home and Save directly above the map, with the cursor
        # position in a readout of its own (see widgets.readout).
        map_bar, self._map_toolbar = toolbar_row(self._canvas, self)
        ml.addWidget(map_bar)
        ml.addWidget(self._canvas, 1)
        self._canvas.mpl_connect('pick_event', self._pick)
        self._canvas.mpl_connect('scroll_event', self._zoom_map)
        self._ax = None
        right.addWidget(map_panel)
        self._result_stack = QStackedWidget()
        self._empty = QLabel('Select a survey to inspect its saved result.')
        self._empty.setAlignment(Qt.AlignCenter)
        self._result_stack.addWidget(self._empty)
        self._em = EMOverviewView(section_only=True, colormaps=shared)
        self._em._line.currentIndexChanged.connect(self._draw_map)
        self._result_stack.addWidget(self._em)
        self._section_fig = Figure(figsize=(9, 3), layout='constrained')
        self._section_canvas = FigureCanvasQTAgg(self._section_fig)
        section = QWidget()
        sl = QVBoxLayout(section)
        sl.setContentsMargins(0, 0, 0, 0)
        # The section's colour map beside its toolbar. A resistivity or velocity
        # section shares the choice made for those sections on their own pages;
        # a grid or point product shares the map layer's.
        self._section_colormap = cmaps.ColormapChooser(cmaps.MAP_LAYER, 'coolwarm', shared=shared)
        self._section_colormap.colormapChanged.connect(self._redraw_section)
        # And its colour limits, held while locked through the depths and the
        # surveys of one quantity. A row of their own, so the toolbar below
        # keeps its width in a narrow panel.
        self._section_range = ColorRange(decimals=6, what="the section's colours")
        self._section_range.changed.connect(self._redraw_section)
        self._section_quantity = None   # what the section's range was set for
        colours = FlowLayout(spacing=6)
        colours.addWidget(self._section_colormap)
        colours.addWidget(self._section_range)
        sl.addLayout(colours)
        # Its cursor position in a readout of its own: the toolbar's label would
        # re-lay out the page on every mouse move (see widgets.readout).
        section_bar, self._section_toolbar = toolbar_row(self._section_canvas, self)
        sl.addWidget(section_bar)
        sl.addWidget(self._section_canvas)
        self._mesh_view = section
        self._result_stack.addWidget(section)
        self._result_stack.addWidget(self._build_compare())
        right.addWidget(self._result_stack)
        split.setSizes([240, 950])
        right.setSizes([430, 340])
        self._frame.currentIndexChanged.connect(self._filter_changed)
        self._method.currentIndexChanged.connect(self._filter_changed)
        self._list.currentItemChanged.connect(self._choose)
        self._list.itemChanged.connect(self._visibility)
        # View > Length Units: the map and the section are redrawn; the EM
        # section view listens for itself.
        length_units.notifier().changed.connect(self._on_length_unit_changed)

    def _on_length_unit_changed(self, _unit):
        """Redraw the map and the selected survey's section in the new unit."""
        self._draw_map()
        self._redraw_section()

    def refresh(self):
        # The window calls this every time the page is shown, not only when the
        # map changes. Gridding a slice costs real time, so leaving the page and
        # coming back must neither throw a surface away nor stop one still being
        # computed: only what came from a survey that has since changed goes.
        running, shown = self._surface_job, self._surface
        try:
            store = self.state.ensure_results_store()
            changed_project = self._root != store.root
            self._root = store.root
            self._store = ProjectMapStore(store.root, store.read_only)
            entries = self._store.entries()
            before = {} if changed_project else {e['id']: self._inputs(e) for e in self._entries}
            kept = {e['id'] for e in entries if before.get(e['id']) == self._inputs(e)}
            reason = 'a different project is open' if changed_project else 'its survey changed or left the map'
            self._entries = entries
            self._keep_cached(kept)
            method = self._method.currentText()
            self._method.blockSignals(True)
            self._method.clear()
            self._method.addItems(['All methods'] + sorted({e['method'] for e in self._entries}))
            if self._method.findText(method) >= 0:
                self._method.setCurrentText(method)
            self._method.blockSignals(False)
            requested = getattr(self.state, 'map_selected_id', None)
            if changed_project:
                self._selected = None
                self._tile = None
            chosen = next((e for e in self._entries if e['id'] == requested), None)
            if chosen is None and changed_project and self._entries:
                # A survey the map shows: opening on a hidden one, as the first
                # saved often is, left nothing drawn to interpolate.
                chosen = next((e for e in self._entries if e.get('visible', True)),
                              self._entries[0])
            if chosen:
                self._selected = chosen['id']
                self._frame.blockSignals(True)
                self._frame.setCurrentIndex(self._frame.findData(chosen['frame']))
                self._frame.blockSignals(False)
                self._method.blockSignals(True)
                self._method.setCurrentIndex(0)
                self._method.blockSignals(False)
                self.state.map_selected_id = None
            self._note.setText(f'{len(self._entries)} saved surveys · {store.root.name} · '
                               'Map snapshots are saved immediately. Removing a layer keeps source results.')
            self._filter_changed()
        except Exception as exc:
            kept, reason = set(), 'the project map could not be read'
            self._entries = []
            self._keep_cached(kept)
            self._note.setText(f'Could not read project map: {exc}')
            self._filter_changed()
        # The redraw leaves the status line on its idle text; when work is gone,
        # the reason goes ahead of it so nothing vanishes unexplained. A job can
        # also stop in the redraw itself when the map now shows another survey,
        # such as one that Add to Map has just saved from another page.
        notice = ''
        if running is not None and self._surface_job is None:
            notice = 'Interpolation stopped: ' + (
                reason if running[1][0] not in kept else 'the slice it was gridding is no longer shown')
        elif shown is not None and shown[0]['id'] not in kept:
            notice = f'Surface cleared: {reason}'
        if notice:
            self._interpolation_status.setText(f'{notice}. {self._interpolation_status.text()}')

    @staticmethod
    def _inputs(entry):
        """What a survey's surfaces are gridded from: its record less name and visibility.

        Those two are all the store edits in place, so any other difference means
        the snapshot behind the id is not the one a cached surface came from.
        """
        return {k: v for k, v in entry.items() if k not in ('name', 'visible')}

    def _keep_cached(self, kept):
        """Forget what was loaded or gridded for every survey not in ``kept``.

        A job still gridding one of them is stopped as well: what it returns
        would be a surface of a snapshot the map no longer holds.
        """
        self._arrays = {k: v for k, v in self._arrays.items() if k in kept}
        self._surfaces = {k: v for k, v in self._surfaces.items() if k[0] in kept}
        if self._slices_for not in kept:
            self._slices_for = None
        if self._surface_job is not None and self._surface_job[1][0] not in kept:
            self._cancel_surface()

    def _filtered(self):
        return [e for e in self._entries if e['frame'] == self._frame.currentData()
                and (self._method.currentIndex() == 0 or e['method'] == self._method.currentText())]

    def _filter_changed(self, *_):
        self._list.blockSignals(True)
        self._list.clear()
        entries = self._filtered()
        names = Counter(entry['name'] for entry in entries)
        for entry in entries:
            label = entry['name']
            if names[label] > 1:
                # Two surveys of one name ("TDEM", "TDEM") cannot be told apart
                # in the list, and the hidden one could be selected unnoticed;
                # the file each came from, or the day it was added, can.
                added = str(entry.get('created_at', ''))[:16].replace('T', ' ')
                label += f" · {entry.get('source_file') or added}"
            item = QTreeWidgetItem([label, entry['method']])
            item.setData(0, Qt.UserRole, entry['id'])
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(0, Qt.Checked if entry.get('visible', True) else Qt.Unchecked)
            item.setToolTip(0, f"{entry['crs']} · {entry['created_at']}")
            from PySide6.QtGui import QColor, QBrush
            item.setForeground(1, QBrush(QColor(COLORS.get(entry['method'], '#8e8e93'))))
            self._list.addTopLevelItem(item)
            if entry['id'] == self._selected:
                self._list.setCurrentItem(item)
        if self._list.currentItem() is None and self._list.topLevelItemCount():
            # Open on a survey the map shows. A hidden one has no slice drawn to
            # interpolate, so starting on it left Interpolate disabled beside a
            # slice and a method that looked chosen.
            items = [self._list.topLevelItem(i) for i in range(self._list.topLevelItemCount())]
            shown = [item for item in items if item.checkState(0) == Qt.Checked]
            self._list.setCurrentItem((shown or items)[0])
        self._list.blockSignals(False)
        self._ax = None
        self._choose(self._list.currentItem(), None)

    def _entry(self):
        return next((e for e in self._entries if e['id'] == self._selected), None)

    def _data(self, entry):
        if entry['id'] not in self._arrays:
            self._arrays[entry['id']] = self._store.load(entry)
        return self._arrays[entry['id']]

    def _choose(self, item, _previous):
        self._selected = item.data(0, Qt.UserRole) if item else None
        entry = self._entry()
        # Rebuilding the list for the survey it already holds -- the page shown
        # again, another layer added or renamed -- keeps the chosen slice. Back on
        # 'Survey locations', a surface kept for that slice would drop out of
        # view and a job still gridding it would be cancelled as a new selection.
        keep = self._depth.currentIndex() if entry and entry['id'] == self._slices_for else 0
        self._details.setText('')
        if entry:
            kind = {'em': 'Soundings / section', 'mesh': 'Section model',
                    'grid': 'Model grid footprint', 'points': 'Point product',
                    'wells': 'Wells, logs and water levels'}.get(entry['kind'], '')
            self._details.setText(f"{entry['name']}\n{kind} · {entry['units']}\n"
                                  f"CRS: {entry['crs']}\nAdded: {entry['created_at']}")
        self._depth.blockSignals(True)
        self._depth.clear()
        self._depth.addItem('Survey locations', None)
        self._depth.blockSignals(False)
        self._result_stack.setCurrentWidget(self._empty)
        self._empty.setText('Select a survey to inspect its saved result.')
        if entry:
            try:
                arrays = self._data(entry)
                if entry['kind'] == 'em':
                    self._em.show_result(em_result(entry, arrays))
                    self._result_stack.setCurrentWidget(self._em)
                    edges = arrays['depth_edges']
                    self._depth.blockSignals(True)
                    for index, depth in enumerate((edges[:-1] + edges[1:]) / 2):
                        self._depth.addItem(f'{depth:g} m', index)
                    self._depth.blockSignals(False)
                elif entry['kind'] == 'grid':
                    edges = arrays['z_edges']
                    self._depth.blockSignals(True)
                    for index, z in enumerate((edges[:-1] + edges[1:])/2):
                        self._depth.addItem(f'Model Z = {z:g} m', index)
                    self._depth.blockSignals(False)
                    self._draw_grid(entry, arrays)
                    self._result_stack.setCurrentWidget(self._mesh_view)
                elif entry['kind'] == 'points':
                    self._draw_points(entry, arrays)
                    self._result_stack.setCurrentWidget(self._mesh_view)
                elif entry['kind'] == 'wells':
                    self._fill_wells(entry, arrays)
                    self._result_stack.setCurrentWidget(self._compare)
                    self._draw_compare()
                else:
                    self._draw_section(entry, arrays)
                    self._result_stack.setCurrentWidget(self._mesh_view)
            except Exception as exc:
                self._empty.setText(f"Cannot load {entry['name']}: {exc}")
        if 0 < keep < self._depth.count():
            self._depth.blockSignals(True)
            self._depth.setCurrentIndex(keep)
            self._depth.blockSignals(False)
        self._slices_for = entry['id'] if entry else None
        self._depth.setEnabled(bool(entry and entry['kind'] in ('em', 'grid')))
        griddable = bool(entry and entry['kind'] in ('em', 'grid', 'points'))
        for widget in (self._interp, self._interp_res, self._interp_blank, self._stations):
            widget.setEnabled(griddable)
        self._interp.setToolTip(SURFACE_HELP if griddable else
                                'A section model has no plan slice to interpolate.')
        self._draw_map()

    def _surface_changed(self, *_):
        self._draw_map()

    def _layer_values(self, entry, arrays, layer):
        """One value per map position for the chosen slice, or None.

        This is what makes the plan view method-agnostic: an EM depth slice, a
        recovered grid layer and an imported point product all reduce to the
        same (value, colour-bar label, log scale) triple here, and everything
        downstream -- scatter, interpolation, export -- is shared.
        """
        if entry['kind'] == 'em':
            if layer is None:
                return None
            values = arrays['model3d'][:, 0, ::-1][:, int(layer)].copy()
            sensitivity = arrays.get('sensitivity')
            if sensitivity is not None and sensitivity.shape == arrays['model3d'][:, 0, :].shape:
                values[sensitivity[:, int(layer)] < entry.get('doi_threshold', .5)] = np.nan
            return values, 'Resistivity (Ω·m); below DOI hidden', True
        if entry['kind'] == 'grid':
            if layer is None:
                return None
            return arrays['model3d'][:, :, int(layer)].ravel(), entry['units'], False
        if entry['kind'] == 'points':
            return np.asarray(arrays['values'], dtype=float).ravel(), entry['units'], False
        return None

    def _draw_slice(self, entry, arrays, xy, groups, layer, chosen, errors):
        """Draw the selected slice as a filled surface, or as coloured stations.

        Returns the surface artist when one was drawn, which is what tells the
        caller to leave this survey's own line traces and station markers off.
        """
        from matplotlib.colors import LogNorm, Normalize
        slice_data = self._layer_values(entry, arrays, layer) if chosen else None
        if slice_data is None:
            return None
        values, units, log_scale = slice_data
        valid = np.isfinite(values)
        if log_scale:
            valid &= values > 0
        if not valid.any():
            if entry['kind'] == 'em':
                errors.append('Selected depth has no values above the DOI threshold.')
            return None
        low, high = float(values[valid].min()), float(values[valid].max())
        if low == high:
            low, high = (low * .99, high * 1.01) if log_scale else (low - 1, high + 1)
        # Locked limits hold through the slices and across surveys of the same
        # quantity; a slice in other units starts on its own scale.
        if (units, log_scale) != self._map_quantity:
            self._map_quantity = (units, log_scale)
            self._map_range.blockSignals(True)      # this drawing uses the new range
            self._map_range.unlock()
            self._map_range.blockSignals(False)
        locked = self._map_range.limits(low, high)
        if not (log_scale and locked[0] <= 0):      # no colour for it on a log scale
            low, high = locked
        norm = LogNorm(low, high) if log_scale else Normalize(low, high)
        # The map layer's chosen colour map, or what this kind of slice has
        # always been drawn with: turbo for resistivity, coolwarm otherwise.
        cmap = cmaps.to_matplotlib(self._map_colormap.set_target(
            cmaps.MAP_LAYER, 'turbo' if log_scale else 'coolwarm'))
        self._map_coloured = True
        surface = self._plan_surface(entry, xy, values, log_scale, norm, cmap, errors)
        mappable = surface
        if surface is None or self._stations.isChecked():
            # Station markers share the surface's norm, so one colour bar reads
            # the same for interpolated cells and measured points.
            square = entry['kind'] == 'grid'
            colored = self._ax.scatter(xy[valid, 0], xy[valid, 1], c=values[valid],
                s=34 if entry['kind'] == 'em' else 30, cmap=cmap, norm=norm,
                zorder=4, picker=6, marker='s' if square else 'o',
                edgecolors='white', linewidths=0 if entry['kind'] == 'em' else .25)
            self._artists[colored] = (entry['id'], groups[valid] if entry['kind'] == 'em' else None)
            mappable = surface if surface is not None else colored
        self._fig.colorbar(mappable, ax=self._ax, label=units)
        return surface

    def _plan_surface(self, entry, xy, values, log_scale, norm, cmap, errors):
        """Draw the selected interpolation of this slice beneath the stations."""
        method = self._interp.currentData()
        if method is None:
            return None
        blank = float(self._interp_blank.value()) or None
        key = (entry['id'], self._depth.currentIndex(), method,
               int(self._interp_res.value()), blank)
        self._surface_request = (key, xy, values, dict(method=method, log_values=log_scale,
                                resolution=int(self._interp_res.value()), max_distance=blank))
        result = self._surfaces.get(key, None)
        if result is None:
            return None
        if result is False:
            return None
        self._surface = (entry, result, self._depth.currentText())
        # Web Mercator metres are stretched by 1/cos(latitude), so calling them
        # plain metres would misstate every distance a geographic survey reports.
        unit = 'display m' if self._frame.currentData() == 'geographic' else 'm'
        self._surface_note = (
            f"{self._interp.currentText()} · {result['n_samples']} stations · "
            f"{result['cell_size']:.3g} {unit} cells · {result['coverage']:.0%} of the frame filled"
            + (f" · {result['variogram']['model']} variogram, range "
               f"{result['variogram']['range']:.4g} {unit}" if result['variogram'] else ''))
        if blank and blank < result['gap']:
            # A blanking radius under the station reach does not trim a wide line
            # spacing, it eats the survey; say so with the number that would not.
            self._surface_note = (
                f"Blanking at {blank:g} {unit} is cutting inside the survey: the outline "
                f"needs {result['gap']:.0f} {unit} to fill, so this leaves ribbons along the "
                f"lines. · {self._surface_note}")
        return self._ax.pcolormesh(result['x_edges'], result['y_edges'], result['grid'],
                                   cmap=cmap, norm=norm, zorder=1, alpha=.92,
                                   shading='flat', rasterized=True)

    def _start_surface(self):
        if self._surface_request is None or self._surface_job is not None:
            return
        key, xy, values, options = self._surface_request
        self._surface_token += 1
        token = self._surface_token
        worker = SurfaceWorker(np.array(xy, copy=True), np.array(values, copy=True), options)
        self._surface_job = (worker, key, token)
        worker.progressed.connect(lambda percent, message: self._surface_progress(token, percent, message))
        worker.succeeded.connect(lambda result: self._surface_ready(token, key, result))
        worker.failed.connect(lambda message: self._surface_failed(token, message))
        self._interpolate_button.setEnabled(False)
        self._interpolate_button.setText('Interpolating…')
        self._interpolation_progress.setValue(0)
        self._interpolation_progress.setVisible(True)
        self._cancel_interpolation.setVisible(True)
        self._interpolation_status.setText('Preparing interpolation… You can keep using the map.')
        self.register_worker(worker)
        worker.start()

    def _surface_progress(self, token, percent, message):
        if self._surface_job is not None and token == self._surface_token:
            self._interpolation_progress.setValue(percent)
            self._interpolation_status.setText(message)

    def _surface_ready(self, token, key, result):
        if self._surface_job is None or token != self._surface_token:
            return
        self._surface_job = None
        if self._surface_request is None or self._surface_request[0] != key:
            return
        if len(self._surfaces) > 24:
            self._surfaces.clear()
        self._surfaces[key] = result
        self._draw_map()
        self._interpolation_status.setText('Interpolation complete — the surface is ready to view or export.')
        self._interpolation_progress.setValue(100)

    def _surface_failed(self, token, message):
        if self._surface_job is None or token != self._surface_token:
            return
        self._surface_job = None
        self._sync_surface_controls()
        self._interpolation_status.setText(f'Could not interpolate: {message} Adjust the settings and try again.')

    def _cancel_surface(self, message='Interpolation cancelled. Click Interpolate to try again.'):
        if self._surface_job is None:
            return
        worker, _, _ = self._surface_job
        self._surface_token += 1
        self._surface_job = None
        worker.cancel()
        self._sync_surface_controls()
        self._interpolation_status.setText(message)

    def _sync_surface_controls(self):
        busy = self._surface_job is not None
        self._interpolate_button.setEnabled(self._surface_request is not None and not busy)
        self._interpolate_button.setText('Interpolating…' if busy else 'Interpolate')
        self._cancel_interpolation.setVisible(busy)
        self._interpolation_progress.setVisible(busy)
        if not busy:
            self._interpolation_status.setText(
                'Surface ready. Change settings and click Interpolate to update.' if self._surface is not None else
                'Ready — click Interpolate to create the map surface.' if self._surface_request is not None else
                self._surface_hint())

    def _surface_hint(self):
        """What stands between the selection and a surface to interpolate.

        The one line used to be "Choose a data slice and an interpolation method"
        whatever the cause, which reads as a fault when both are chosen and the
        selected survey is merely hidden on the map.
        """
        entry = self._entry()
        if entry is None:
            return 'Select a survey in the list to map its values.'
        if not entry.get('visible', True):
            return (f"“{entry['name']}” is hidden on the map. Tick it in the survey list "
                    "to show its slices and interpolate them.")
        if entry['kind'] == 'wells':
            return 'Wells have no plan slice; click one to compare it with the surveys around it.'
        if entry['kind'] not in ('em', 'grid', 'points'):
            return 'A section model has no plan slice to interpolate.'
        if entry['kind'] in ('em', 'grid') and self._depth.currentData() is None:
            return 'Choose a data slice to map, then an interpolation method under Surface.'
        if self._interp.currentData() is None:
            return 'Choose an interpolation method under Surface to grid this slice.'
        return 'This slice has no values to interpolate; the note below the map says why.'

    def _show_variogram(self):
        if self._surface is None or not self._surface[1].get('variogram'):
            return
        entry, result, layer = self._surface
        VariogramDialog(result, entry, layer, self).exec()

    def _export_surface(self):
        if self._surface is None:
            QMessageBox.information(self, 'Export grid', 'Select a survey slice and a Surface '
                                    'interpolation method first; the export writes that grid.')
            return
        from PyHydroGeophysX.core.plan_interpolation import write_plan_grid
        entry, result, layer = self._surface
        name = f"{entry['name']}_{layer}".replace(' ', '_').replace('·', '-')
        path, _ = QFileDialog.getSaveFileName(
            self, 'Export interpolated plan grid', f'{name}.asc',
            'ESRI ASCII grid (*.asc);;Point table (*.csv)')
        if not path:
            return
        try:
            write_plan_grid(result, path)
            self._note.setText(f"Wrote {result['method']} grid of {entry['name']} "
                               f"({layer}) in {entry['crs']} to {path}.")
        except Exception as exc:
            QMessageBox.warning(self, 'Export grid', str(exc))

    def _section_key(self, entry):
        """What a section figure shows, as the key its colour map is kept under."""
        if entry['kind'] != 'mesh':
            return cmaps.MAP_LAYER          # the same product the map layer shows
        if entry.get('units') == '%':
            return cmaps.RESISTIVITY_CHANGE
        return {'ERT': cmaps.RESISTIVITY, 'Seismic': cmaps.VELOCITY}.get(
            entry['method'], cmaps.MAP_LAYER)

    def _section_cmap(self, entry, default):
        """The colour map for ``entry``'s section: the one chosen, or ``default``."""
        return cmaps.to_matplotlib(
            self._section_colormap.set_target(self._section_key(entry), default))

    def _section_limits(self, entry, low, high, log_scale=False):
        """The section's colour limits: the ones typed in while locked, or ``(low, high)``.

        A lock holds for another survey of the same quantity and is let go for
        one in other units. On a log scale a lower limit of zero or below has
        no colour, so typed limits like that leave the section on its own.
        """
        quantity = (self._section_key(entry), entry.get('units'))
        if quantity != self._section_quantity:
            self._section_quantity = quantity
            self._section_range.blockSignals(True)      # the caller is drawing already
            self._section_range.unlock()
            self._section_range.blockSignals(False)
        locked = self._section_range.limits(low, high)
        return (low, high) if log_scale and locked[0] <= 0 else locked

    def _section_drawn(self):
        """Show the section just drawn from scratch; Home and Back go to it."""
        self._section_canvas.draw_idle()
        self._section_toolbar.update()

    def _redraw_section(self, *_):
        """Redraw the selected survey's section in the chosen colours."""
        entry = self._entry()
        if entry is None or entry['kind'] == 'em':
            return
        if entry['kind'] == 'wells':
            self._draw_compare()
            return
        try:
            arrays = self._data(entry)
            if entry['kind'] == 'grid':
                self._draw_grid(entry, arrays)
            elif entry['kind'] == 'points':
                self._draw_points(entry, arrays)
            else:
                self._draw_section(entry, arrays)
        except Exception as exc:
            self._note.setText(f"Cannot redraw {entry['name']}: {exc}")

    def _draw_grid(self, entry, arrays):
        self._section_fig.clear()
        plan, section = self._section_fig.subplots(1, 2)
        model = arrays['model3d']
        index = self._depth.currentData()
        index = model.shape[2] - 1 if index is None else int(index)
        x, y, z = (arrays[k] for k in ('x_edges', 'y_edges', 'z_edges'))
        finite = model[np.isfinite(model)]
        low, high = (float(finite.min()), float(finite.max())) if finite.size else (0., 1.)
        if low == high:
            low, high = low - 1, high + 1
        # One range for both panels: they share the colour bar.
        low, high = self._section_limits(entry, low, high)
        cmap = self._section_cmap(entry, 'coolwarm')
        image = plan.pcolormesh(x, y, model[:, :, index].T, cmap=cmap, vmin=low, vmax=high)
        set_length_axis(plan, 'x', 'Source X')
        set_length_axis(plan, 'y', 'Source Y')
        level = to_display_length((z[index] + z[index + 1]) / 2)
        plan.set_title(f"{entry['name']} · Z={level:.4g} {length_units.current()}")
        section.pcolormesh(x, z, model[:, model.shape[1]//2, :].T, cmap=cmap, vmin=low, vmax=high)
        set_length_axis(section, 'x', 'Source X')
        set_length_axis(section, 'y', 'Model Z')
        section.set_title('Central Y section')
        self._section_fig.colorbar(image, ax=[plan, section], label=entry['units'])
        self._section_drawn()

    def _draw_points(self, entry, arrays):
        self._section_fig.clear()
        ax = self._section_fig.add_subplot(111)
        xy = arrays['source_xy']
        values = np.ma.masked_invalid(np.asarray(arrays['values'], dtype=float))
        # The points' own range, as the scatter would scale itself, unless locked.
        low, high = ((float(values.min()), float(values.max())) if values.count()
                     else (None, None))
        if low is not None:
            low, high = self._section_limits(entry, low, high)
        points = ax.scatter(xy[:, 0], xy[:, 1], c=values, vmin=low, vmax=high,
                            cmap=self._section_cmap(entry, 'coolwarm'), s=20)
        ax.set(xlabel=f"Source X ({entry['crs']})", ylabel='Source Y', title=entry['name'])
        ax.set_aspect('equal', adjustable='box')
        self._section_fig.colorbar(points, ax=ax, label=entry['units'])
        self._section_drawn()

    def _draw_section(self, entry, arrays):
        from matplotlib.collections import PolyCollection
        from matplotlib.colors import LogNorm, Normalize
        self._section_fig.clear()
        ax = self._section_fig.add_subplot(111)
        vertices, offsets, values = arrays['vertices'], arrays['offsets'].astype(int), arrays['values']
        polygons = [vertices[a:b] for a, b in zip(offsets[:-1], offsets[1:])]
        valid = np.ma.masked_invalid(values)
        if entry['method'] == 'ERT':
            valid = np.ma.masked_less_equal(valid, 0)
        if not valid.count():
            raise ValueError('No finite model values to display.')
        low, high = float(valid.min()), float(valid.max())
        if low == high:
            low, high = (low * 0.99, high * 1.01) if low > 0 else (low - 1, high + 1)
        low, high = self._section_limits(entry, low, high, log_scale=entry['method'] == 'ERT')
        norm = LogNorm(low, high) if entry['method'] == 'ERT' else Normalize(low, high)
        collection = PolyCollection(polygons, array=valid, cmap=self._section_cmap(entry, 'turbo'),
                                    norm=norm, edgecolors='none')
        ax.add_collection(collection)
        ax.autoscale_view()
        # A section saved from a survey read without elevations has its top at
        # z = 0, and reads as depth.
        set_section_axes(ax, z=vertices[:, 1], xlabel='Profile distance',
                         elevation_name='Section elevation')
        ax.set_title(entry['name'])
        self._section_fig.colorbar(collection, ax=ax, label=entry['units'])
        self._section_drawn()

    def _draw_map(self, *_, fit=False):
        old = (self._ax.get_xlim(), self._ax.get_ylim()) if self._ax and self._last_frame == self._frame.currentData() and not fit else None
        self._last_frame = self._frame.currentData()
        self._fig.clear()
        self._ax = self._fig.add_subplot(111)
        self._artists = {}
        self._surface, self._surface_note = None, ''
        self._surface_request = None
        self._map_coloured = False     # set by _draw_slice when a slice is coloured
        all_xy = []
        errors = []
        for entry in self._filtered():
            if not entry.get('visible', True):
                continue
            try:
                arrays = self._data(entry)
                xy = arrays['map_xy']
                all_xy.append(xy)
                chosen = entry['id'] == self._selected
                if entry['kind'] == 'wells':
                    self._draw_wells(entry, arrays, chosen)
                    continue
                color = COLORS.get(entry['method'], '#586774')
                groups = arrays.get('line_numbers', np.zeros(len(xy)))
                is_em = entry['kind'] == 'em'
                line_groups = np.unique(groups)
                from matplotlib import colormaps
                palette = colormaps['tab20']
                layer = self._depth.currentData()
                if chosen and entry['kind'] == 'grid':
                    self._draw_grid(entry, arrays)
                # The slice is drawn first so the line traces know whether a filled
                # surface already carries this survey's values.
                surface = self._draw_slice(entry, arrays, xy, groups, layer, chosen, errors)
                # A survey that reads as an image does not also need its own stations
                # and line traces stamped over it; tick Stations to bring them back.
                bare = surface is not None and not self._stations.isChecked()
                traced = entry['kind'] in ('em', 'mesh') and not bare
                for group in (np.unique(groups) if traced else []):
                    part = xy[groups == group]
                    active = chosen and is_em and self._em._line.currentData() == group
                    # Index against all survey lines so colours survive selection changes.
                    line_color = palette(int(np.searchsorted(line_groups, group)) % 20) if is_em else color
                    artist, = self._ax.plot(part[:, 0], part[:, 1], '-', color=line_color,
                                           lw=3 if active else 1.2, alpha=1 if chosen else .55, picker=6)
                    self._artists[artist] = (entry['id'], group)
                    if is_em:
                        points = self._ax.scatter(part[:, 0], part[:, 1], color=line_color,
                            s=32 if active else 12, alpha=1 if chosen else .65,
                            edgecolors='white', linewidths=.4, picker=6, zorder=3,
                            label=f"{entry['name']} · Line {int(group)}" + (' (selected)' if active else ''))
                        self._artists[points] = (entry['id'], group)
                        if active:
                            # An outline stays visible when a physical depth colour scale is overlaid.
                            self._ax.scatter(part[:, 0], part[:, 1], s=58, facecolors='none',
                                edgecolors=line_color, linewidths=1.5, zorder=5)
                            self._ax.annotate(f'Line {int(group)}', part[-1], xytext=(8, 8),
                                textcoords='offset points', fontsize=8, fontweight='bold',
                                bbox=dict(facecolor='white', alpha=.9, edgecolor=line_color), zorder=6)
                if not is_em and not bare:
                    points = self._ax.scatter(xy[:, 0], xy[:, 1], c=color, s=24 if chosen else 10,
                                          label=f"{entry['name']} · {entry['method']}", picker=6, zorder=3,
                                          edgecolors='white', linewidths=.35, alpha=1 if chosen else .65)
                    self._artists[points] = (entry['id'], groups)
            except Exception as exc:
                errors.append(f"{entry['name']}: {exc}")
        geographic = self._frame.currentData() == 'geographic'
        if self._tile is not None and geographic:
            tile = self._tile
            self._ax.imshow(tile['image'], extent=tile['extent'], zorder=0, origin='upper')
            self._ax.text(.01, .01, tile.get('attribution', ''), transform=self._ax.transAxes,
                          fontsize=7, bbox={'facecolor': 'white', 'alpha': .8, 'edgecolor': 'none'})
        if all_xy:
            xy = np.vstack(all_xy)
            pad = max(float(np.ptp(xy, axis=0).max()) * .08, 10.)
            if old is not None:
                self._ax.set_xlim(old[0]); self._ax.set_ylim(old[1])
            else:
                self._ax.set_xlim(xy[:, 0].min()-pad, xy[:, 0].max()+pad)
                self._ax.set_ylim(xy[:, 1].min()-pad, xy[:, 1].max()+pad)
            handles, labels = self._ax.get_legend_handles_labels()
            selected = [(handle, label) for handle, label in zip(handles, labels)
                        if '(selected)' in label]
            if selected:
                handles, labels = zip(*selected)
                legend = self._ax.legend(handles, labels, loc='upper left', fontsize=7,
                                         borderpad=.3, labelspacing=.2, handletextpad=.4)
                # An overlay must never shrink the geographic axes to fit its text.
                legend.set_in_layout(False)
        else:
            self._ax.text(.5, .5, 'No visible surveys in this coordinate group.\nAdd a result or change the filters.',
                          transform=self._ax.transAxes, ha='center', va='center')
        # Keep map distances isotropic while filling wide and resized windows.
        self._ax.set_aspect('equal', adjustable='datalim')
        self._ax.set_facecolor(theme.color('canvas'))
        # Before the length axes: the unit's own formatter has no offset to turn
        # off, and ticklabel_format refuses any formatter but matplotlib's own.
        self._ax.ticklabel_format(useOffset=False, style='plain')
        if geographic:
            # Web Mercator metres stretch with latitude and are no ground
            # distance, so they are not converted to another unit.
            self._ax.set(xlabel='Web Mercator X (display metres)',
                         ylabel='Web Mercator Y (display metres)')
        else:
            set_length_axis(self._ax, 'x', 'Project X')
            set_length_axis(self._ax, 'y', 'Project Y')
        self._ax.grid(alpha=.2, linestyle=':', color='#8e8e93')
        for spine in self._ax.spines.values():
            spine.set_color('#bacbd7')
        self._load_tiles.setEnabled(geographic and bool(all_xy) and self._tile_worker is None)
        self._map_colormap.setEnabled(self._map_coloured)
        self._map_range.setEnabled(self._map_coloured)
        self._export_grid.setEnabled(self._surface is not None)
        self._variogram_button.setEnabled(
            self._surface is not None and bool(self._surface[1].get('variogram')))
        if self._surface_job is not None and (
                self._surface_request is None or self._surface_request[0] != self._surface_job[1]):
            self._cancel_surface('Selection changed. Click Interpolate to calculate the new surface.')
        self._sync_surface_controls()
        if errors:
            self._note.setText(' · '.join(errors))
        elif self._surface_note:
            self._note.setText(self._surface_note)
        self._canvas.draw_idle()
        # New axes: Home and Back start from this view, not from axes cleared.
        self._map_toolbar.update()

    def _zoom_map(self, event):
        """Zoom around the cursor without rebuilding survey artists or results."""
        if event.inaxes is not self._ax or event.xdata is None or event.ydata is None:
            return
        step = getattr(event, 'step', 0)
        if not step:
            return
        factor = 1.25 ** -float(np.clip(step, -10, 10))
        for centre, limits, setter in (
            (event.xdata, self._ax.get_xlim(), self._ax.set_xlim),
            (event.ydata, self._ax.get_ylim(), self._ax.set_ylim),
        ):
            setter(*(centre + (np.asarray(limits) - centre) * factor))
        self._canvas.draw_idle()

    def _pick(self, event):
        if self._map_toolbar.mode:      # the click pans or zooms; picking would redraw
            return
        selected = self._artists.get(event.artist)
        if selected is None:
            return
        survey_id, group = selected
        if isinstance(group, np.ndarray):
            indices = getattr(event, 'ind', [])
            group = group[int(indices[0])] if len(indices) else None
        for index in range(self._list.topLevelItemCount()):
            item = self._list.topLevelItem(index)
            if item.data(0, Qt.UserRole) == survey_id:
                self._list.setCurrentItem(item)
                if group is not None and self._entry()['kind'] == 'em':
                    index = self._em._line.findData(int(group))
                    if index >= 0:
                        self._em._line.setCurrentIndex(index)
                elif group is not None and self._entry()['kind'] == 'wells':
                    self._well_pick.setCurrentIndex(int(group))
                break

    # -- wells: a well beside the surveys around it ------------------------------
    def _build_compare(self):
        """The wells layer's view: one well beside every survey that reaches it."""
        view = QWidget()
        layout = QVBoxLayout(view)
        layout.setContentsMargins(0, 0, 0, 0)
        bar = QHBoxLayout()
        self._well_pick = QComboBox()
        self._well_pick.setToolTip('The well to compare; click a well on the map to choose it.')
        self._well_pick.currentIndexChanged.connect(self._on_well_changed)
        bar.addWidget(QLabel('Well'))
        bar.addWidget(self._well_pick)
        self._compare_within = DistanceSpinBox()
        self._compare_within.setRange(1, 1e5)
        self._compare_within.setDecimals(0)
        self._compare_within.setValue(50)
        self._compare_within.setSuffix(' m')
        self._compare_within.setKeyboardTracking(False)
        self._compare_within.setToolTip('Surveys read farther than this from the well are '
                                        'listed below the plot, not drawn.')
        self._compare_within.valueChanged.connect(self._draw_compare)
        bar.addWidget(QLabel('Surveys within'))
        bar.addWidget(self._compare_within)
        self._compare_depth = DistanceSpinBox()
        self._compare_depth.setRange(0, 1e4)
        self._compare_depth.setDecimals(1)
        self._compare_depth.setSuffix(' m')
        self._compare_depth.setSpecialValueText('auto')
        self._compare_depth.setKeyboardTracking(False)
        self._compare_depth.valueChanged.connect(self._draw_compare)
        bar.addWidget(QLabel('Down to'))
        bar.addWidget(self._compare_depth)
        bar.addStretch(1)
        layout.addLayout(bar)
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure
        self._compare_fig = Figure(figsize=(9, 3.4))
        self._compare_canvas = FigureCanvasQTAgg(self._compare_fig)
        compare_bar, self._compare_toolbar = toolbar_row(self._compare_canvas, view)
        layout.addWidget(compare_bar)
        layout.addWidget(self._compare_canvas, 1)
        self._compare_note = QLabel('')
        self._compare_note.setWordWrap(True)
        theme.set_tone(self._compare_note, 'hint')
        layout.addWidget(self._compare_note)
        self._compare = view
        return view

    def _fill_wells(self, entry, arrays):
        _data, order = wells_data(arrays)
        keep = self._well_pick.currentText()
        self._well_pick.blockSignals(True)
        self._well_pick.clear()
        self._well_pick.addItems(order)
        index = order.index(keep) if keep in order else 0
        self._well_pick.setCurrentIndex(index)
        self._well_pick.blockSignals(False)

    def _on_well_changed(self, *_):
        self._draw_compare()
        self._draw_map()

    def _draw_wells(self, entry, arrays, chosen):
        """A wells layer on the map: hollow circles, one label per nest of wells."""
        xy = np.asarray(arrays['map_xy'], dtype=float)
        try:
            _data, order = wells_data(arrays)
        except Exception:  # noqa: BLE001 - an unreadable payload still shows the wells
            order = [str(k + 1) for k in range(len(xy))]
        ink = COLORS['Boreholes']
        points = self._ax.scatter(xy[:, 0], xy[:, 1], s=44 if chosen else 30, marker='o',
                                  facecolors='white', edgecolors=ink, linewidths=1.6, zorder=6,
                                  picker=6, label=f"{entry['name']} · wells")
        self._artists[points] = (entry['id'], np.arange(len(xy)))
        current = self._well_pick.currentIndex() if chosen else -1
        if 0 <= current < len(xy):
            self._ax.scatter([xy[current, 0]], [xy[current, 1]], s=70, color=ink, zorder=7)
        for group in bh.nearby_groups(xy):
            names = sorted(order[k] for k in group)
            at = xy[group].max(axis=0)
            self._ax.annotate('\n'.join(names), (at[0], at[1]), xytext=(6, 2),
                              textcoords='offset points', fontsize=7, va='top', color=ink,
                              fontweight='bold' if current in group else 'normal', zorder=7)

    def _compare_surveys(self, entry, point):
        """Every other visible survey of the frame read at ``point``, sorted into what
        is drawn, what is too far, and point values."""
        drawn, far, values = [], [], []
        within = float(self._compare_within.value())
        unit = length_units.current()

        def away(metres):
            return f'{to_display_length(metres, unit):.1f} {unit}'

        for other in self._entries:
            if (other['id'] == entry['id'] or other['frame'] != entry['frame']
                    or other['kind'] == 'wells' or not other.get('visible', True)):
                continue
            try:
                profile = profile_at(other, self._data(other), point, entry['frame'])
            except Exception as exc:  # noqa: BLE001 - one bad survey never hides the rest
                far.append(f"{other['name']} (could not be read: {exc})")
                continue
            if profile is None:
                far.append(f"{other['name']} (does not reach the well)")
            elif profile['distance'] > within:
                far.append(f"{other['name']} ({away(profile['distance'])} away)")
            elif 'value' in profile:
                values.append(f"{other['name']}: {profile['value']:.4g} {other.get('units', '')} "
                              f"at the {profile['where']}, {away(profile['distance'])} away")
            else:
                drawn.append(dict(
                    name=other['name'], units=other.get('units', ''),
                    log_scale=other.get('units') == 'Ω·m',
                    trace='amplitude' in str(other.get('units', '')), top=profile['top'],
                    bottom=profile['bottom'], values=profile['values'], faded=profile['faded'],
                    distance=profile['distance'], where=profile['where'],
                    color=COLORS.get(other['method'], '#586774')))
        return drawn, far, values

    def _draw_compare(self, *_):
        entry = self._entry()
        if entry is None or entry['kind'] != 'wells' or self._well_pick.currentIndex() < 0:
            return
        try:
            arrays = self._data(entry)
            data, order = wells_data(arrays)
        except Exception as exc:  # noqa: BLE001
            self._compare_note.setText(f"Cannot read the wells of {entry['name']}: {exc}")
            return
        index = self._well_pick.currentIndex()
        well_id = order[index]
        point = np.asarray(arrays['map_xy'], dtype=float)[index]
        drawn, far, values = self._compare_surveys(entry, point)
        unit = length_units.current()
        bh.draw_well_comparison(self._compare_fig, data, well_id, drawn, length_unit=unit,
                                max_depth=float(self._compare_depth.value()) or None,
                                water_color=theme.DATA_COLOR, edge_color='#1d1d1f')
        self._compare_fig.subplots_adjust(left=0.07, right=0.98, top=0.82, bottom=0.2)
        if not drawn:
            self._compare_fig.text(0.6, 0.5, 'No survey on the map reaches this well within '
                                   f'{to_display_length(self._compare_within.value(), unit):.0f} '
                                   f'{unit}.', ha='center', va='center', color='#8e8e93')
        self._compare_canvas.draw_idle()
        self._compare_toolbar.update()      # Home and Back go to the new axes
        lines = []
        if drawn:
            lines.append('Read at: ' + ' · '.join(
                f"{s['name']} ({s['where']}, {to_display_length(s['distance'], unit):.1f} {unit} "
                'from the well)' for s in drawn) + '. Depths are below each survey\'s own '
                'surface there, and below ground in the well.')
        if values:
            lines.append('Point values: ' + ' · '.join(values) + '.')
        if far:
            lines.append('Not drawn: ' + ' · '.join(far) + '.')
        self._compare_note.setText('\n'.join(lines))

    def _visibility(self, item, column):
        if column != 0 or self._store is None:
            return
        try:
            survey_id = item.data(0, Qt.UserRole)
            visible = item.checkState(0) == Qt.Checked
            self._store.update(survey_id, visible=visible)
            next(e for e in self._entries if e['id'] == survey_id)['visible'] = visible
            self._draw_map()
        except Exception as exc:
            QMessageBox.warning(self, 'Map visibility', str(exc))
            self.refresh()

    def _rename(self):
        entry = self._entry()
        if entry:
            name, ok = QInputDialog.getText(self, 'Rename survey', 'Survey name', text=entry['name'])
            if ok:
                self._change(lambda: self._store.update(entry['id'], name=name))

    def _remove(self):
        entry = self._entry()
        if entry:
            self._change(lambda: self._store.remove(entry['id']))

    def _change(self, action):
        try:
            action()
            self.refresh()
        except Exception as exc:
            QMessageBox.warning(self, 'Project Map', str(exc))

    def _fetch_tiles(self):
        if self._tile_worker is not None or self._frame.currentData() != 'geographic':
            return
        xlim, ylim, root = self._ax.get_xlim(), self._ax.get_ylim(), self._root
        worker = TaskWorker(basemap_image, xlim, ylim, transform=(1+0j, 0j),
                            source=self._source.currentText(), max_tiles=6, timeout=2.)
        self._tile_worker = self.register_worker(worker)
        self._load_tiles.setEnabled(False)
        self._note.setText('Loading optional basemap… Survey results remain available.')
        def loaded(tile):
            if root != self._root or self._frame.currentData() != 'geographic':
                return
            self._tile = tile
            self._note.setText('Basemap loaded.' if tile else 'Basemap unavailable; showing surveys without imagery.')
            self._draw_map()
        def finished():
            self._tile_worker = None
            self._load_tiles.setEnabled(self._frame.currentData() == 'geographic')
        worker.succeeded.connect(loaded)
        worker.failed.connect(lambda message: self._note.setText(f'Basemap unavailable: {message}'))
        worker.finished.connect(finished)
        worker.start()

    def _export_png(self):
        path, _ = QFileDialog.getSaveFileName(self, 'Export current project map', 'project_map.png', 'PNG (*.png)')
        if path:
            try:
                self._fig.savefig(path, dpi=180)
            except Exception as exc:
                QMessageBox.warning(self, 'Map export', str(exc))

    def _import_points(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Point product: x,y,value columns', '', 'CSV (*.csv)')
        if not path:
            return
        method, ok = QInputDialog.getItem(self, 'Survey method', 'Method',
            ['Gravity', 'Magnetics', 'ERT', 'EM', 'Seismic', 'GPR', 'IP', 'SP'], 0, True)
        if not ok:
            return
        units, ok = QInputDialog.getText(self, 'Product units', 'Units (e.g. mGal, nT, Ω·m, m/s)')
        if not ok:
            return
        try:
            from pathlib import Path
            from PyHydroGeophysX.qt_apps.project_map import point_snapshot
            from PyHydroGeophysX.qt_apps.widgets.map_export import save_snapshot_to_map
            data = np.genfromtxt(path, delimiter=',', names=True, encoding='utf8')
            xy = np.column_stack([np.atleast_1d(data[k]) for k in ('x', 'y')])
            snapshot = point_snapshot(xy, data['value'], method, units, Path(path).stem)
            save_snapshot_to_map(self, *snapshot)
            self.refresh()
        except Exception as exc:
            QMessageBox.warning(self, 'Point product', str(exc))

    def _add_resource(self):
        resources = [r for r in self.state.geophysical_resources.values() if r.get('role') == 'model']
        if not resources:
            QMessageBox.information(self, 'Available results', 'No model resources in this session. '
                                    'Run a module or import a point product CSV.')
            return
        labels = [f"{i+1}. {r.get('label', r.get('method', 'Model'))}" for i, r in enumerate(resources)]
        label, ok = QInputDialog.getItem(self, 'Add available result', 'Model', labels, 0, False)
        if not ok:
            return
        try:
            from PyHydroGeophysX.qt_apps.project_map import grid_snapshot, mesh_snapshot
            from PyHydroGeophysX.qt_apps.widgets.map_export import save_snapshot_to_map
            resource = resources[labels.index(label)]
            meta = resource.get('metadata') or {}
            method = resource.get('method', 'Model')
            method = {'SRT': 'Seismic', 'GRAVITY': 'Gravity', 'MAGNETICS': 'Magnetics'}.get(method, method)
            units = {'ERT': 'Ω·m', 'Seismic': 'm/s', 'Gravity': 'g/cc', 'Magnetics': 'SI'}.get(method)
            if units is None:
                units, ok = QInputDialog.getText(self, 'Model units', 'Units')
                if not ok:
                    return
            if meta.get('mesh') is not None:
                snapshot = mesh_snapshot(meta['mesh'], resource['payload'], method, resource['label'])
                snapshot[0]['units'] = units
            elif meta.get('edges') is not None:
                snapshot = grid_snapshot(meta['edges'], resource['payload'], method, units, resource['label'])
            else:
                raise ValueError('This resource has no mesh/grid geometry. Use its module Add to Map '
                                 'or import an x,y,value point product.')
            save_snapshot_to_map(self, *snapshot)
            self.refresh()
        except Exception as exc:
            QMessageBox.warning(self, 'Available result', str(exc))

    def export_actions(self):
        return [('Current project map (PNG)', self._export_png),
                ('Interpolated plan grid (ASCII / CSV)', self._export_surface)]

    def stop_workers(self, wait_ms=30000):
        self._cancel_surface()
        super().stop_workers(wait_ms)

    def closeEvent(self, event):
        self.stop_workers()
        super().closeEvent(event)

    # -- agent command interface ----------------------------------------------
    def agent_describe(self):
        methods = [value or 'points' for _label, value in SURFACES]
        return {
            'module': self.module_key,
            'title': self.module_title,
            'state': self._agent_status(),
            'actions': [
                {'name': 'get_status', 'args': {},
                 'desc': ('The surveys on the map (id, name, method, kind, units, visible), '
                          'the one selected, its slices and the surface settings.')},
                {'name': 'select_survey', 'args': {'survey': 'id or name'},
                 'desc': ('Select a survey: its saved section is shown below the map and its '
                          'slices can be chosen.')},
                {'name': 'set_visible', 'args': {'survey': 'id or name', 'visible': 'bool'},
                 'desc': 'Show or hide a survey on the map; saved with the project map.'},
                {'name': 'set_frame', 'args': {'frame': ['geographic', 'local']},
                 'desc': 'Show the geographic surveys or those in local project coordinates.'},
                {'name': 'set_slice', 'args': {'slice': 'index (0 = survey locations) or its label'},
                 'desc': "Choose the selected survey's depth or model slice to map in plan view."},
                {'name': 'set_surface',
                 'args': {'method': methods, 'cells': 'int 40-400',
                          'blanking_m': 'number, 0 = no blanking', 'stations': 'bool'},
                 'desc': ('How the slice is gridded: kriging (with its own variance), '
                          'inverse distance, triangulation, nearest neighbour or thin-plate '
                          'spline; points draws the stations only.')},
                {'name': 'interpolate', 'args': {},
                 'desc': 'Grid the chosen slice with the surface settings; runs in the background.'},
                {'name': 'cancel_interpolation', 'args': {}, 'desc': 'Stop a running interpolation.'},
                {'name': 'fit_surveys', 'args': {}, 'desc': 'Zoom the map to the surveys shown.'},
                {'name': 'rename_survey', 'args': {'survey': 'id or name', 'name': 'str'},
                 'desc': 'Rename a survey on the map; its source result is unchanged.'},
                {'name': 'export_map_png', 'args': {'path': 'str (.png)'},
                 'desc': 'Save the map as shown to a PNG.'},
                {'name': 'export_grid', 'args': {'path': 'str (.asc or .csv)'},
                 'desc': 'Write the interpolated surface as an ESRI ASCII grid or a point table.'},
                {'name': 'import_points',
                 'args': {'path': 'CSV with x,y,value columns', 'method': 'str', 'units': 'str',
                          'crs': "'LOCAL' (default) or a CRS such as EPSG:32613",
                          'name': 'str (optional)'},
                 'desc': ('Add a point product (gravity, magnetics...) to the map at the x,y '
                          'the file gives, in the CRS named.')},
                {'name': 'compare_well',
                 'args': {'well': 'well id', 'within_m': 'number (optional, default 50)'},
                 'desc': ("Select the wells layer and one of its wells: its lithology, logs and "
                          "water level are set beside every visible survey within within_m, "
                          "each read down the vertical there. Returns what each survey reads "
                          "and how far from the well; capture_view 'page' to see it.")},
                {'name': 'refresh', 'args': {}, 'desc': 'Re-read the project map from disk.'},
            ],
            'note': 'Removing a survey from the map is left to the user (the Remove button).',
        }

    def agent_apply(self, action, args):
        args = args or {}
        handlers = {
            'get_status': self._agent_status,
            'select_survey': lambda: self._agent_select(args.get('survey')),
            'set_visible': lambda: self._agent_set_visible(args.get('survey'),
                                                           args.get('visible', True)),
            'set_frame': lambda: self._agent_set_frame(args.get('frame')),
            'set_slice': lambda: self._agent_set_slice(args.get('slice')),
            'set_surface': lambda: self._agent_set_surface(args),
            'interpolate': self._agent_interpolate,
            'cancel_interpolation': self._agent_cancel,
            'fit_surveys': self._agent_fit,
            'rename_survey': lambda: self._agent_rename(args.get('survey'), args.get('name')),
            'export_map_png': lambda: self._agent_export_png(args.get('path')),
            'export_grid': lambda: self._agent_export_grid(args.get('path')),
            'import_points': lambda: self._agent_import_points(
                args.get('path'), args.get('method'), args.get('units'),
                args.get('crs') or 'LOCAL', args.get('name')),
            'compare_well': lambda: self._agent_compare_well(args.get('well'),
                                                             args.get('within_m')),
            'refresh': lambda: (self.refresh(), self._agent_status())[1],
        }
        handler = handlers.get(action)
        if handler is None:
            return {'status': 'failed', 'error': f"Unknown action '{action}'.",
                    'valid_actions': list(handlers)}
        return handler()

    def _agent_status(self):
        entry = self._entry()
        return {
            'status': 'ok',
            'project': str(self._root or ''),
            'frame': self._frame.currentData(),
            'surveys': [{'id': e['id'], 'name': e.get('name'), 'method': e.get('method'),
                         'kind': e.get('kind'), 'units': e.get('units'),
                         'visible': bool(e.get('visible', True)), 'frame': e.get('frame'),
                         'crs': e.get('crs')} for e in self._entries],
            'selected': entry['id'] if entry else None,
            'slices': [self._depth.itemText(i) for i in range(self._depth.count())],
            'slice': self._depth.currentIndex(),
            'surface': {'method': self._interp.currentData() or 'points',
                        'cells': self._interp_res.value(),
                        'blanking_m': self._interp_blank.value(),
                        'stations': self._stations.isChecked()},
            'can_interpolate': self._surface_request is not None,
            'interpolating': self._surface_job is not None,
            'surface_ready': self._surface is not None,
            'message': self._interpolation_status.text(),
            'note': self._note.text(),
        }

    def _agent_compare_well(self, well, within=None):
        """Select the wells layer holding ``well`` and compare it; report in numbers."""
        for entry in self._entries:
            if entry['kind'] != 'wells':
                continue
            try:
                data, order = wells_data(self._data(entry))
            except Exception:  # noqa: BLE001
                continue
            if str(well) not in order:
                continue
            if within is not None:
                self._compare_within.setValue(float(within))
            self._frame.setCurrentIndex(self._frame.findData(entry['frame']))
            self._agent_select(entry['id'])
            self._well_pick.setCurrentIndex(order.index(str(well)))
            index = order.index(str(well))
            point = np.asarray(self._data(entry)['map_xy'], dtype=float)[index]
            drawn, far, values = self._compare_surveys(entry, point)
            water = data.water_range(str(well))
            return {
                'status': 'ok', 'well': str(well),
                'lithology': [(i.top, i.bottom, i.unit) for i in data.intervals(str(well))],
                'water_level_range_m': list(water[:2]) if water else None,
                'surveys': [{'name': d['name'], 'units': d['units'],
                             'distance_m': round(float(d['distance']), 2), 'read_at': d['where'],
                             'profile': [(round(float(t), 2), round(float(b), 2), float(v))
                                         for t, b, v in zip(d['top'], d['bottom'], d['values'])
                                         if np.isfinite(v)][:40]} for d in drawn],
                'point_values': values, 'not_drawn': far,
                'note': ('Depths are metres below ground in the well and below each survey\'s '
                         'surface where it was read.'),
            }
        wells = []
        for entry in self._entries:
            if entry['kind'] == 'wells':
                try:
                    wells += wells_data(self._data(entry))[1]
                except Exception:  # noqa: BLE001
                    pass
        return {'status': 'failed', 'error': f'No well {well!r} on the map.', 'wells': wells,
                'hint': 'Add wells from the Boreholes page (Add wells to the Map).'}

    def _agent_find(self, survey):
        """The map entry ``survey`` names - by id, by exact name, or by a unique part of one."""
        key = str(survey or '').strip()
        if not key:
            return None, "Provide 'survey' (an id or a name)."
        for entry in self._entries:
            if entry['id'] == key:
                return entry, ''
        exact = [e for e in self._entries if str(e.get('name', '')).lower() == key.lower()]
        partial = [e for e in self._entries if key.lower() in str(e.get('name', '')).lower()]
        found = exact or partial
        if len(found) == 1:
            return found[0], ''
        if not found:
            return None, f"No survey named '{key}'."
        return None, (f"'{key}' matches {len(found)} surveys; use an id: "
                      + ', '.join(f"{e['id']} ({e.get('name')})" for e in found))

    def _agent_select(self, survey):
        entry, problem = self._agent_find(survey)
        if entry is None:
            return {'status': 'failed', 'error': problem}
        self._selected = entry['id']
        for combo, index in ((self._frame, self._frame.findData(entry.get('frame'))),
                             (self._method, 0)):
            if index >= 0:
                combo.blockSignals(True)
                combo.setCurrentIndex(index)
                combo.blockSignals(False)
        self._filter_changed()
        return self._agent_status()

    def _agent_set_visible(self, survey, visible):
        entry, problem = self._agent_find(survey)
        if entry is None:
            return {'status': 'failed', 'error': problem}
        if self._store is None:
            return {'status': 'failed', 'error': 'The project map is not open.'}
        try:
            self._store.update(entry['id'], visible=bool(visible))
        except Exception as exc:  # noqa: BLE001
            return {'status': 'failed', 'error': str(exc)}
        entry['visible'] = bool(visible)
        self._filter_changed()
        return {'status': 'ok', 'survey': entry['id'], 'visible': bool(visible)}

    def _agent_set_frame(self, frame):
        index = self._frame.findData(str(frame or ''))
        if index < 0:
            return {'status': 'failed', 'error': "frame is 'geographic' or 'local'."}
        self._frame.setCurrentIndex(index)
        return self._agent_status()

    def _agent_set_slice(self, which):
        if self._depth.count() <= 1:
            return {'status': 'failed',
                    'error': 'The selected survey has no slices (select an EM or grid survey).'}
        index = -1
        if isinstance(which, (int, float)) or str(which).strip().lstrip('-').isdigit():
            index = int(which)
        else:
            index = self._depth.findText(str(which))
        if not 0 <= index < self._depth.count():
            return {'status': 'failed', 'error': f"No slice '{which}'.",
                    'slices': [self._depth.itemText(i) for i in range(self._depth.count())]}
        self._depth.setCurrentIndex(index)
        return {'status': 'ok', 'slice': index, 'label': self._depth.itemText(index),
                'can_interpolate': self._surface_request is not None,
                'message': self._interpolation_status.text()}

    def _agent_set_surface(self, args):
        if 'method' in args:
            method = None if str(args['method']).lower() in ('points', 'none', '') else args['method']
            index = self._interp.findData(method)
            if index < 0:
                return {'status': 'failed', 'error': f"Unknown method '{args['method']}'.",
                        'methods': [value or 'points' for _label, value in SURFACES]}
            self._interp.setCurrentIndex(index)
        if 'cells' in args:
            self._interp_res.setValue(int(args['cells']))
        if 'blanking_m' in args:
            self._interp_blank.setValue(float(args['blanking_m'] or 0))
        if 'stations' in args:
            self._stations.setChecked(bool(args['stations']))
        return self._agent_status()

    def _agent_interpolate(self):
        if self._surface_job is not None:
            return {'status': 'failed', 'error': 'An interpolation is already running.'}
        if self._surface_request is None:
            return {'status': 'failed', 'error': self._surface_hint()}
        self._start_surface()
        return {'status': 'ok',
                'detail': 'Interpolating in the background; get_status reports when the surface is ready.'}

    def _agent_cancel(self):
        if self._surface_job is None:
            return {'status': 'failed', 'error': 'No interpolation is running.'}
        self._cancel_surface()
        return {'status': 'ok'}

    def _agent_fit(self):
        self._draw_map(fit=True)
        return {'status': 'ok'}

    def _agent_rename(self, survey, name):
        entry, problem = self._agent_find(survey)
        if entry is None:
            return {'status': 'failed', 'error': problem}
        name = str(name or '').strip()
        if not name or self._store is None:
            return {'status': 'failed', 'error': "Provide 'name'."}
        try:
            self._store.update(entry['id'], name=name)
        except Exception as exc:  # noqa: BLE001
            return {'status': 'failed', 'error': str(exc)}
        self.refresh()
        return {'status': 'ok', 'survey': entry['id'], 'name': name}

    def _agent_export_png(self, path):
        if not path:
            return {'status': 'failed', 'error': "Provide 'path' (.png)."}
        try:
            self._fig.savefig(str(path), dpi=180)
        except Exception as exc:  # noqa: BLE001
            return {'status': 'failed', 'error': str(exc)}
        return {'status': 'ok', 'path': str(path)}

    def _agent_export_grid(self, path):
        if self._surface is None:
            return {'status': 'failed',
                    'error': 'No interpolated surface yet: set a slice and a method, then interpolate.'}
        if not path:
            return {'status': 'failed', 'error': "Provide 'path' (.asc or .csv)."}
        from PyHydroGeophysX.core.plan_interpolation import write_plan_grid
        entry, result, layer = self._surface
        try:
            write_plan_grid(result, str(path))
        except Exception as exc:  # noqa: BLE001
            return {'status': 'failed', 'error': str(exc)}
        self._note.setText(f"Wrote {result['method']} grid of {entry['name']} "
                           f"({layer}) in {entry['crs']} to {path}.")
        return {'status': 'ok', 'path': str(path), 'method': result.get('method')}

    def _agent_import_points(self, path, method, units, crs='LOCAL', name=None):
        """Add a point product without the placement dialog: a point product
        carries its own x,y, so the only choice the dialog asks for is the CRS."""
        from pathlib import Path

        from PyHydroGeophysX.qt_apps.project_map import ProjectMapStore, point_snapshot

        if not path or not Path(str(path)).is_file():
            return {'status': 'failed', 'error': f'Not found: {path}'}
        if not method or units is None:
            return {'status': 'failed', 'error': "Provide 'method' and 'units'."}
        crs = str(crs or 'LOCAL').strip()
        try:
            store = self.state.ensure_results_store()
            if store.read_only:
                return {'status': 'failed', 'error': 'This project is read-only.'}
            if crs.upper() != 'LOCAL':
                from pyproj import CRS
                CRS.from_user_input(crs)
            data = np.genfromtxt(str(path), delimiter=',', names=True, encoding='utf8')
            xy = np.column_stack([np.atleast_1d(data[k]) for k in ('x', 'y')])
            meta, arrays = point_snapshot(xy, data['value'], str(method), str(units),
                                          Path(str(path)).stem)
            meta.update(source_module=self.module_key, source_file=Path(str(path)).name)
            entry = ProjectMapStore(store.root).add(meta, arrays, arrays['survey_xy'], crs,
                                                    str(name or meta['label']))
        except Exception as exc:  # noqa: BLE001
            return {'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'}
        self.state.map_selected_id = entry['id']
        self.log(f"Saved '{entry['name']}' to Project Map.", 'success')
        self.refresh()
        return {'status': 'ok', 'survey': entry['id'], 'name': entry['name'], 'crs': crs}
