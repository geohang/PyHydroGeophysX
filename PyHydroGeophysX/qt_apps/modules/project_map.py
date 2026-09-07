"""Project-wide map membership and linked scientific result inspection."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QComboBox, QFileDialog, QHBoxLayout, QInputDialog,
    QLabel, QMessageBox, QPushButton, QSplitter, QStackedWidget, QTreeWidget, QHeaderView,
    QTreeWidgetItem, QVBoxLayout, QWidget)

from .base import BaseModule
from PyHydroGeophysX.qt_apps.project_map import ProjectMapStore, em_result
from PyHydroGeophysX.qt_apps.workers import TaskWorker
from PyHydroGeophysX.visualization.basemap import TILE_SOURCES, basemap_image


COLORS = {'ERT': '#cb6c24', 'EM': '#197a87', 'Seismic': '#8259b2',
          'Gravity': '#ad4965', 'Magnetics': '#527931'}


class ProjectMapModule(BaseModule):
    module_key = 'project_map'
    module_title = 'Project Map'

    def __init__(self, state, log, parent=None):
        super().__init__(state, log, parent)
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
        from matplotlib.figure import Figure
        from PyHydroGeophysX.qt_apps.widgets.em_overview_view import EMOverviewView
        self._entries, self._arrays, self._artists = [], {}, {}
        self._store = None
        self._selected = None
        self._tile = None
        self._tile_worker = None
        self._root = None
        self._last_frame = None
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)
        self._note = QLabel('Add results from any survey method using Add to Map… or import a point product.')
        self._note.setWordWrap(True)
        self._note.setStyleSheet('color: #597185; font-size: 11px;')
        split = QSplitter(Qt.Horizontal)
        root.addWidget(split, 1)
        root.addWidget(self._note)
        left = QWidget()
        left.setObjectName('MapLayersPanel')
        left.setStyleSheet('QWidget#MapLayersPanel {background: #ffffff; border: 1px solid #dce5ed; border-radius: 8px;}')
        left.setMinimumWidth(200)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(6, 6, 6, 6)
        ll.setSpacing(4)
        layers_title = QLabel('SURVEY LAYERS')
        layers_title.setStyleSheet('font-size: 10px; font-weight: 600; color: #597185; letter-spacing: 1px;')
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
        map_panel.setStyleSheet('QWidget#ProjectMapPanel {background: #ffffff; border: 1px solid #dce5ed; border-radius: 8px;}')
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
        slices = QHBoxLayout()
        slices.addWidget(QLabel('Slice:'))
        slices.addWidget(self._depth)
        slices.addStretch(1)
        self._fig = Figure(figsize=(9, 4), layout='constrained')
        self._canvas = FigureCanvasQTAgg(self._fig)
        navigation = NavigationToolbar2QT(self._canvas, self, coordinates=False)
        navigation.setContentsMargins(0, 0, 0, 0)
        slices.insertWidget(0, navigation)
        ml.addLayout(slices)
        ml.addWidget(self._canvas, 1)
        self._canvas.mpl_connect('pick_event', self._pick)
        self._canvas.mpl_connect('scroll_event', self._zoom_map)
        self._ax = None
        right.addWidget(map_panel)
        self._result_stack = QStackedWidget()
        self._empty = QLabel('Select a survey to inspect its saved result.')
        self._empty.setAlignment(Qt.AlignCenter)
        self._result_stack.addWidget(self._empty)
        self._em = EMOverviewView(section_only=True)
        self._em._line.currentIndexChanged.connect(self._draw_map)
        self._result_stack.addWidget(self._em)
        self._section_fig = Figure(figsize=(9, 3), layout='constrained')
        self._section_canvas = FigureCanvasQTAgg(self._section_fig)
        section = QWidget()
        sl = QVBoxLayout(section)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.addWidget(NavigationToolbar2QT(self._section_canvas, self))
        sl.addWidget(self._section_canvas)
        self._mesh_view = section
        self._result_stack.addWidget(section)
        right.addWidget(self._result_stack)
        split.setSizes([240, 950])
        right.setSizes([430, 340])
        self._frame.currentIndexChanged.connect(self._filter_changed)
        self._method.currentIndexChanged.connect(self._filter_changed)
        self._list.currentItemChanged.connect(self._choose)
        self._list.itemChanged.connect(self._visibility)

    def refresh(self):
        try:
            store = self.state.ensure_results_store()
            changed_project = self._root != store.root
            self._root = store.root
            self._store = ProjectMapStore(store.root, store.read_only)
            self._entries = self._store.entries()
            method = self._method.currentText()
            self._method.blockSignals(True)
            self._method.clear()
            self._method.addItems(['All methods'] + sorted({e['method'] for e in self._entries}))
            if self._method.findText(method) >= 0:
                self._method.setCurrentText(method)
            self._method.blockSignals(False)
            self._arrays = {}
            requested = getattr(self.state, 'map_selected_id', None)
            if changed_project:
                self._selected = None
                self._tile = None
            chosen = next((e for e in self._entries if e['id'] == requested), None)
            if chosen is None and changed_project and self._entries:
                chosen = self._entries[0]
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
            self._entries, self._arrays = [], {}
            self._note.setText(f'Could not read project map: {exc}')
            self._filter_changed()

    def _filtered(self):
        return [e for e in self._entries if e['frame'] == self._frame.currentData()
                and (self._method.currentIndex() == 0 or e['method'] == self._method.currentText())]

    def _filter_changed(self, *_):
        self._list.blockSignals(True)
        self._list.clear()
        for entry in self._filtered():
            item = QTreeWidgetItem([entry['name'], entry['method']])
            item.setData(0, Qt.UserRole, entry['id'])
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(0, Qt.Checked if entry.get('visible', True) else Qt.Unchecked)
            item.setToolTip(0, f"{entry['crs']} · {entry['created_at']}")
            from PySide6.QtGui import QColor, QBrush
            item.setForeground(1, QBrush(QColor(COLORS.get(entry['method'], '#597185'))))
            self._list.addTopLevelItem(item)
            if entry['id'] == self._selected:
                self._list.setCurrentItem(item)
        if self._list.currentItem() is None and self._list.topLevelItemCount():
            self._list.setCurrentItem(self._list.topLevelItem(0))
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
        self._details.setText('')
        if entry:
            kind = {'em': 'Soundings / section', 'mesh': 'Section model',
                    'grid': 'Model grid footprint', 'points': 'Point product'}.get(entry['kind'], '')
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
                else:
                    self._draw_section(entry, arrays)
                    self._result_stack.setCurrentWidget(self._mesh_view)
            except Exception as exc:
                self._empty.setText(f"Cannot load {entry['name']}: {exc}")
        self._depth.setEnabled(bool(entry and entry['kind'] in ('em', 'grid')))
        self._draw_map()

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
        image = plan.pcolormesh(x, y, model[:, :, index].T, cmap='coolwarm', vmin=low, vmax=high)
        plan.set(xlabel='Source X (m)', ylabel='Source Y (m)', title=f"{entry['name']} · Z={(z[index]+z[index+1])/2:g} m")
        section.pcolormesh(x, z, model[:, model.shape[1]//2, :].T, cmap='coolwarm', vmin=low, vmax=high)
        section.set(xlabel='Source X (m)', ylabel='Model Z (m)', title='Central Y section')
        self._section_fig.colorbar(image, ax=[plan, section], label=entry['units'])
        self._section_canvas.draw_idle()

    def _draw_points(self, entry, arrays):
        self._section_fig.clear()
        ax = self._section_fig.add_subplot(111)
        xy = arrays['source_xy']
        points = ax.scatter(xy[:, 0], xy[:, 1], c=np.ma.masked_invalid(arrays['values']), cmap='coolwarm', s=20)
        ax.set(xlabel=f"Source X ({entry['crs']})", ylabel='Source Y', title=entry['name'])
        ax.set_aspect('equal', adjustable='box')
        self._section_fig.colorbar(points, ax=ax, label=entry['units'])
        self._section_canvas.draw_idle()

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
        norm = LogNorm(low, high) if entry['method'] == 'ERT' else Normalize(low, high)
        collection = PolyCollection(polygons, array=valid, cmap='turbo', norm=norm, edgecolors='none')
        ax.add_collection(collection)
        ax.autoscale_view()
        ax.set(xlabel='Profile distance (m)', ylabel='Section elevation (m)', title=entry['name'])
        self._section_fig.colorbar(collection, ax=ax, label=entry['units'])
        self._section_canvas.draw_idle()

    def _draw_map(self, *_, fit=False):
        from matplotlib.colors import LogNorm, Normalize
        old = (self._ax.get_xlim(), self._ax.get_ylim()) if self._ax and self._last_frame == self._frame.currentData() and not fit else None
        self._last_frame = self._frame.currentData()
        self._fig.clear()
        self._ax = self._fig.add_subplot(111)
        self._artists = {}
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
                color = COLORS.get(entry['method'], '#586774')
                groups = arrays.get('line_numbers', np.zeros(len(xy)))
                is_em = entry['kind'] == 'em'
                line_groups = np.unique(groups)
                from matplotlib import colormaps
                palette = colormaps['tab20']
                for group in (np.unique(groups) if entry['kind'] in ('em', 'mesh') else []):
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
                if not is_em:
                    points = self._ax.scatter(xy[:, 0], xy[:, 1], c=color, s=24 if chosen else 10,
                                          label=f"{entry['name']} · {entry['method']}", picker=6, zorder=3,
                                          edgecolors='white', linewidths=.35, alpha=1 if chosen else .65)
                    self._artists[points] = (entry['id'], groups)
                layer = self._depth.currentData()
                if chosen and entry['kind'] == 'em' and layer is not None:
                    values = arrays['model3d'][:, 0, ::-1][:, int(layer)].copy()
                    sensitivity = arrays.get('sensitivity')
                    if sensitivity is not None and sensitivity.shape == arrays['model3d'][:, 0, :].shape:
                        values[sensitivity[:, int(layer)] < entry.get('doi_threshold', .5)] = np.nan
                    valid = np.isfinite(values) & (values > 0)
                    if valid.any():
                        low, high = values[valid].min(), values[valid].max()
                        if low == high:
                            low, high = low * .99, high * 1.01
                        colored = self._ax.scatter(xy[valid, 0], xy[valid, 1], c=values[valid],
                            s=34, cmap='turbo', norm=LogNorm(low, high), zorder=4, picker=6)
                        self._artists[colored] = (entry['id'], groups[valid])
                        self._fig.colorbar(colored, ax=self._ax, label='Resistivity (Ω·m); below DOI hidden')
                    else:
                        errors.append('Selected depth has no values above the DOI threshold.')
                if chosen and entry['kind'] in ('grid', 'points'):
                    if entry['kind'] == 'grid':
                        self._draw_grid(entry, arrays)
                        values = arrays['model3d'][:, :, int(layer)].ravel() if layer is not None else None
                    else:
                        values = arrays['values']
                    if values is not None:
                        valid = np.isfinite(values)
                        if valid.any():
                            low, high = values[valid].min(), values[valid].max()
                            if low == high:
                                low, high = low - 1, high + 1
                            colored = self._ax.scatter(xy[valid, 0], xy[valid, 1], c=values[valid],
                                s=30, cmap='coolwarm', norm=Normalize(low, high), zorder=4, picker=6,
                                marker='s' if entry['kind'] == 'grid' else 'o', edgecolors='white', linewidths=.25)
                            self._artists[colored] = (entry['id'], None)
                            self._fig.colorbar(colored, ax=self._ax, label=entry['units'])
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
        self._ax.set_facecolor('#f8fbfd')
        self._ax.set(xlabel='Web Mercator X (display metres)' if geographic else 'Project X (m)',
                     ylabel='Web Mercator Y (display metres)' if geographic else 'Project Y (m)')
        self._ax.ticklabel_format(useOffset=False, style='plain')
        self._ax.grid(alpha=.2, linestyle=':', color='#7890a2')
        for spine in self._ax.spines.values():
            spine.set_color('#bacbd7')
        self._load_tiles.setEnabled(geographic and bool(all_xy) and self._tile_worker is None)
        if errors:
            self._note.setText(' · '.join(errors))
        self._canvas.draw_idle()

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
                break

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
        return [('Current project map (PNG)', self._export_png)]

    def stop_workers(self, wait_ms=30000):
        super().stop_workers(wait_ms)

    def closeEvent(self, event):
        self.stop_workers()
        super().closeEvent(event)
