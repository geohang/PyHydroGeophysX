"""Explicit survey placement and Add to Map handoff shared by result pages."""
from pathlib import Path

import numpy as np
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
    QFileDialog, QFormLayout, QLabel, QLineEdit, QMessageBox, QPushButton, QVBoxLayout)

from PyHydroGeophysX.qt_apps.project_map import (
    ProjectMapStore, em_snapshot, mesh_snapshot, place_profile, grid_snapshot, point_snapshot,
)


def result_snapshot(page):
    """Capture the currently displayed recovered model before opening placement."""
    key = page.module_key
    if hasattr(page, 'map_snapshot'):
        return page.map_snapshot()
    if key == 'gravmag_processing':
        result = getattr(page, '_inv_result', None)
        choices = (['Recovered 3D model'] if result else []) + (['Observed'] if page._fields else [])
        if page._qc:
            choices += ['Regional', 'Residual']
        if not choices:
            raise ValueError('Load or process a gravity/magnetic product first.')
        from PySide6.QtWidgets import QInputDialog
        field, ok = QInputDialog.getItem(page, 'Gravity / Magnetics', 'Product to add', choices, 0, False)
        if not ok:
            return None
        if field != 'Recovered 3D model':
            result = None
        kind = str(result.get('kind', '') if result else page._kind.currentText()).lower()
        method, units = ('Gravity', 'g/cc') if kind.startswith('grav') else ('Magnetics', 'SI')
        if result:
            return grid_snapshot(result['edges'], result['model3d'], method, units,
                                 f'{method} recovered model')
        if field in ('Regional', 'Residual'):
            grid = page._qc['grids'][field]
            return point_snapshot(np.column_stack([grid['xx'].ravel(), grid['yy'].ravel()]),
                                  grid['zz'].ravel(), method,
                                  'mGal' if method == 'Gravity' else 'nT', f'{method} {field}')
        values = page._fields.get(field)
        if values is None:
            values = page._fields.get('Observed')
        if values is None or page._x is None:
            raise ValueError('Load or process a gravity/magnetic product first.')
        return point_snapshot(np.column_stack([page._x, page._y]), values, method,
                              'mGal' if method == 'Gravity' else 'nT', f'{method} {field}')
    if key == 'joint_inversion':
        result = page._result
        if result is None:
            raise ValueError('Run the joint inversion first.')
        from PySide6.QtWidgets import QInputDialog
        choices = [k for k, v in result.models.items() if np.size(v)]
        method, ok = QInputDialog.getItem(page, 'Joint result', 'Model to add', choices, 0, False)
        if not ok:
            return None
        values = result.models[method]
        if result.methods == ('ERT', 'SRT'):
            return mesh_snapshot(result.meta['mesh'], values,
                                 'Seismic' if method == 'SRT' else 'ERT', f'Joint {method}')
        if result.methods == ('Gravity', 'Magnetics'):
            return grid_snapshot(result.meta['edges'], values, method,
                                 'g/cc' if method == 'Gravity' else 'SI', f'Joint {method}')
        thickness = np.asarray(result.meta.get('thicknesses', result.meta.get('thickness', [])))
        values = np.asarray(values).ravel()
        if len(thickness) != len(values) - 1:
            raise ValueError('Joint layered result has no matching thicknesses.')
        edges = np.r_[0, np.cumsum(thickness), np.sum(thickness) + 20]
        return em_snapshot({'model3d': values[::-1][None, None, :], 'positions': [0],
                            'depth_edges': edges, 'method': 'Joint EM'})
    if key in ('em', 'em_processing'):
        result = getattr(page, '_last_section', None)
        if result is None:
            single = getattr(page, '_last_result', None)
            if single is None:
                raise ValueError('Run an EM inversion first.')
            # Preserve the recovered step profile as one EM column.
            depth = np.asarray(single['depth'], dtype=float)
            rho = np.asarray(single['resistivity_step'], dtype=float)
            unique = np.unique(depth)
            centers = (unique[:-1] + unique[1:]) / 2
            values = np.interp(centers, depth, rho)
            result = {'model3d': values[::-1][None, None, :], 'positions': [0.],
                      'depth_edges': unique, 'method': single.get('method', 'EM')}
        return em_snapshot(result)
    if key in ('ert', 'ert_processing'):
        tl = getattr(page, '_tl_models', None)
        if tl is not None and getattr(page, '_map_result_kind', '') == 'timelapse':
            index = max(0, page._tl_step_combo.currentIndex())
            return mesh_snapshot(page._tl_mesh, np.asarray(tl)[:, index], 'ERT',
                                 f'ERT {page._tl_step_combo.currentText()}')
        manager = getattr(page, '_inv_mgr', None)
        if manager is None:
            raise ValueError('Run an ERT inversion first.')
        try:
            coverage = manager.coverage()
        except Exception:
            coverage = None
        return mesh_snapshot(manager.paraDomain, manager.model, 'ERT', coverage=coverage)
    if key in ('seismic', 'seismic_processing'):
        manager = getattr(page, '_srt_mgr', None)
        if manager is None:
            raise ValueError('Run a seismic inversion first.')
        from PyHydroGeophysX.qt_apps.modules.seismic_processing import velocity_of
        return mesh_snapshot(manager.paraDomain, velocity_of(manager), 'Seismic')
    raise ValueError('This page does not have a supported survey result.')


class MapExportDialog(QDialog):
    def __init__(self, meta, arrays, parent=None):
        super().__init__(parent)
        self.meta, self.arrays = meta, arrays
        self.setWindowTitle('Add result to Project Map')
        self.resize(560, 380)
        layout = QVBoxLayout(self)
        note = QLabel('Save a map snapshot in this project. Source results stay unchanged.\n'
                      'Profile elevation is not a map coordinate. Confirm the survey location below.')
        note.setWordWrap(True)
        layout.addWidget(note)
        form = QFormLayout()
        layout.addLayout(form)
        self.name = QLineEdit(meta.get('label', meta['method']))
        form.addRow('Survey name', self.name)
        self.mode = QComboBox()
        if 'survey_xy' in arrays:
            self.mode.addItem('Coordinates carried by this result', 'embedded')
        self.mode.addItem('Straight line: start + bearing', 'straight')
        self.mode.addItem('CSV control points: distance,x,y', 'controls')
        self.mode.addItem('CSV positions: x,y (one row per sample)', 'rows')
        if meta['kind'] in ('grid', 'points'):
            self.mode.removeItem(self.mode.findData('straight'))
            self.mode.removeItem(self.mode.findData('controls'))
        form.addRow('Location', self.mode)
        self.crs = QComboBox()
        self.crs.setEditable(True)
        self.crs.addItems(['LOCAL', 'EPSG:4326', 'EPSG:32615', 'EPSG:26915'])
        self.crs.setCurrentText(meta.get('suggested_crs') or 'LOCAL')
        form.addRow('Coordinate system', self.crs)
        hint = QLabel('LOCAL = shared project coordinates in metres (no basemap).\n'
                      'EPSG:4326 = longitude, latitude. Or enter the survey EPSG code.\n'
                      'Start + bearing needs a projected metre CRS or LOCAL.')
        hint.setWordWrap(True)
        form.addRow(hint)
        self.east, self.north, self.bearing = (QDoubleSpinBox() for _ in range(3))
        for box in (self.east, self.north):
            box.setRange(-1e9, 1e9)
            box.setDecimals(4)
        self.bearing.setRange(0, 359.9999)
        self.bearing.setDecimals(4)
        self.bearing.setSuffix('° clockwise from north')
        distance = np.asarray(arrays['distance'])
        form.addRow(f'Start X at distance {distance.min():g} m', self.east)
        form.addRow('Start Y', self.north)
        form.addRow('Bearing', self.bearing)
        self.csv = QLineEdit()
        browse = QPushButton('Choose location CSV…')
        browse.clicked.connect(self._browse)
        form.addRow(browse, self.csv)
        sample_text = (f'{distance.size} horizontal model cells / stations. CSV rows follow stored result order.'
                       if meta['kind'] in ('grid', 'points') else
                       f'{distance.size} samples; distance {distance.min():g}–{distance.max():g} m. '
                       'CSV rows must follow result order. Control points must span this distance range.')
        self.sample_note = QLabel(sample_text)
        self.sample_note.setWordWrap(True)
        form.addRow(self.sample_note)
        self.mode.currentIndexChanged.connect(self._sync)
        self._sync()
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Save).setText('Add to Map')
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _sync(self):
        mode = self.mode.currentData()
        for widget in (self.east, self.north, self.bearing):
            widget.setEnabled(mode == 'straight')
        self.csv.setEnabled(mode in ('controls', 'rows'))

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Survey location CSV', '', 'CSV (*.csv)')
        if path:
            self.csv.setText(path)

    def placement(self):
        mode, crs = self.mode.currentData(), self.crs.currentText().strip()
        if mode == 'embedded':
            return self.arrays['survey_xy'], crs
        if mode in ('controls', 'rows'):
            data = np.genfromtxt(self.csv.text(), delimiter=',', names=True, encoding='utf8')
            names = data.dtype.names or ()
            required = ('distance', 'x', 'y') if mode == 'controls' else ('x', 'y')
            if not all(k in names for k in required):
                raise ValueError('CSV must have columns: ' + ','.join(required))
            table = np.column_stack([np.atleast_1d(data[k]) for k in required])
            if mode == 'rows':
                return table, crs
            return place_profile(self.arrays['distance'], None, 0, table), crs
        if crs.upper() != 'LOCAL':
            from pyproj import CRS
            source = CRS.from_user_input(crs)
            if not source.is_projected or any(abs(a.unit_conversion_factor - 1) > 1e-9 for a in source.axis_info[:2]):
                raise ValueError('Straight-line placement requires LOCAL or a projected CRS in metres.')
        if self.meta['kind'] == 'em' and np.unique(self.arrays['line_numbers']).size > 1:
            raise ValueError('Multiple EM lines need result coordinates or a per-sample CSV, not one straight line.')
        return place_profile(self.arrays['distance'], [self.east.value(), self.north.value()], self.bearing.value()), crs


def add_result_to_map(page):
    try:
        snapshot = result_snapshot(page)
        if snapshot is None:
            return
        save_snapshot_to_map(page, *snapshot)
    except Exception as exc:
        QMessageBox.warning(page, 'Add to Map', str(exc))


def save_snapshot_to_map(page, meta, arrays):
    try:
        store = page.state.ensure_results_store()
        if store.read_only:
            raise PermissionError('This project is read-only.')
        meta = dict(meta)
        meta['source_module'] = page.module_key
        source = getattr(page, '_source_path', None)
        if source:
            meta['source_file'] = Path(source).name
        workflow = page.state.workflow_results.get(page.module_key, {})
        recipe = workflow.get('recipe_path')
        if recipe:
            try:
                meta['source_recipe'] = Path(recipe).resolve().relative_to(store.root).as_posix()
            except ValueError:
                meta['source_recipe'] = Path(recipe).name
        dialog = MapExportDialog(meta, arrays, page)
        # Keep the completed form available if validation fails.
        while dialog.exec() == QDialog.Accepted:
            try:
                xy, crs = dialog.placement()
                entry = ProjectMapStore(store.root).add(meta, arrays, xy, crs, dialog.name.text())
            except Exception as exc:
                QMessageBox.warning(page, 'Survey location', str(exc))
                continue
            page.state.map_selected_id = entry['id']
            page.log(f"Saved '{entry['name']}' to Project Map.", 'success')
            page.navigateRequested.emit('project_map')
            break
    except Exception as exc:
        QMessageBox.warning(page, 'Add to Map', str(exc))
