"""Physical-coordinate sections and aligned comparisons of rectilinear models."""

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout, QGridLayout, QLabel, QDoubleSpinBox, QComboBox
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.ticker import MaxNLocator


def require_aligned(a, b):
    if not all(hasattr(m, axis) for m in (a, b) for axis in ('x', 'y', 'z')):
        raise ValueError('Comparison requires two rectilinear grids.')
    if not all(np.array_equal(getattr(a, axis), getattr(b, axis)) for axis in ('x', 'y', 'z')):
        raise ValueError('Model grids differ. Align them explicitly before comparison; this viewer does not resample results.')


class _CellCoordinate(QDoubleSpinBox):
    """Arrow keys visit adjacent cells, even when cell widths vary."""

    def __init__(self, centers):
        super().__init__()
        self.centers = centers

    def stepBy(self, steps):  # noqa: N802 - Qt override
        index = int(np.argmin(abs(self.centers - self.value())))
        self.setValue(float(self.centers[int(np.clip(index + steps, 0, len(self.centers) - 1))]))


class ScientificSections(QWidget):
    pointChanged = Signal(object)

    def __init__(self, mesh, field, metadata=None, cmap='viridis', parent=None, comparison=None):
        super().__init__(parent)
        self.mesh, self.comparison = mesh, comparison
        if comparison is not None:
            require_aligned(mesh, comparison)
        self.edges = tuple(np.asarray(getattr(mesh, a)) for a in ('x', 'y', 'z'))
        self.centers = tuple((a[:-1] + a[1:]) / 2 for a in self.edges)
        self.axis_scales = tuple(1000 if np.ptp(a) >= 2000 else 1 for a in self.edges)
        self.indices = [len(a) // 2 for a in self.centers]
        self.field, self.metadata, self.cmap = field, metadata or {}, cmap
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        controls = QGridLayout()
        self.coordinates = []
        for column, (axis, values) in enumerate(zip(('Easting', 'Northing', 'Z elevation'), self.centers)):
            controls.addWidget(QLabel(axis + ' (m)'), 0, column)
            spin = _CellCoordinate(values)
            spin.setDecimals(3)
            spin.setRange(float(values[0]), float(values[-1]))
            spin.setValue(float(values[len(values) // 2]))
            spin.setSingleStep(float(np.min(np.diff(values))) if len(values) > 1 else 1)
            spin.setAccessibleName(axis + ' in metres')
            spin.setKeyboardTracking(False)
            spin.valueChanged.connect(self._moved)
            controls.addWidget(spin, 1, column)
            self.coordinates.append(spin)
        layout.addLayout(controls)
        self.readout = QLabel()
        self.readout.setWordWrap(True)
        layout.addWidget(self.readout)
        self.figure = Figure(figsize=(10, 3.2), layout='constrained')
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumHeight(220)
        layout.addWidget(self.canvas, 1)
        self.canvas.mpl_connect('button_press_event', self._clicked)
        self.draw()

    def set_field(self, field, metadata=None, cmap='viridis'):
        self.field, self.metadata, self.cmap = field, metadata or {}, cmap
        self.draw()

    def select_point(self, point):
        for spin, value in zip(self.coordinates, point):
            spin.blockSignals(True)
            spin.setValue(float(value))
            spin.blockSignals(False)
        self._moved()

    def _moved(self):
        self.indices = [int(np.argmin(abs(c - s.value()))) for c, s in zip(self.centers, self.coordinates)]
        # Always show the sampled cell centre, including on nonuniform meshes.
        for centers, index, spin in zip(self.centers, self.indices, self.coordinates):
            spin.blockSignals(True)
            spin.setValue(float(centers[index]))
            spin.blockSignals(False)
        self.draw()
        self.pointChanged.emit([float(c[i]) for c, i in zip(self.centers, self.indices)])

    def _clicked(self, event):
        if event.inaxes not in self.axes or event.xdata is None or event.ydata is None:
            return
        panel = self.axes.index(event.inaxes)
        coords = [c[i] for c, i in zip(self.centers, self.indices)]
        x, y = ((0, 1), (0, 2), (1, 2))[panel] if self.comparison is None else (0, 1)
        coords[x], coords[y] = event.xdata * self.axis_scales[x], event.ydata * self.axis_scales[y]
        self.select_point(coords)

    def draw(self):
        shape = tuple(len(c) for c in self.centers)
        values = np.asarray(self.mesh.cell_data[self.field]).reshape(shape, order='F')
        if not np.isfinite(values).all():
            raise ValueError(f'{self.field} contains nonfinite model values.')
        x, y, z = self.indices
        xyz = [float(c[i]) for c, i in zip(self.centers, self.indices)]
        value = values[x, y, z]
        label = self.metadata.get('names', {}).get(str(int(value)), '') if self.metadata.get('colors') else ''
        self.readout.setText(f'{self.field}: {value:.6g} {label} · E {xyz[0]:,.3f}, N {xyz[1]:,.3f}, Z {xyz[2]:,.3f} m · Z is elevation, not depth')
        self.figure.clear()
        if self.comparison is None:
            grid = self.figure.add_gridspec(2, 2, width_ratios=(1, 1.25))
            self.axes = [self.figure.add_subplot(grid[:, 0]), self.figure.add_subplot(grid[0, 1]),
                         self.figure.add_subplot(grid[1, 1])]
        else:
            self.axes = list(self.figure.subplots(1, 3))
        limits = self.metadata.get('limits') or [float(values.min()), float(values.max())]
        categorical = self.metadata.get('colors')
        common = dict(cmap=self.cmap, vmin=limits[0], vmax=limits[1])
        panels = [(values[:, :, z].T, 0, 1, f'Plan · Z {xyz[2]:,.1f} m'),
                  (values[:, y, :].T, 0, 2, f'E–Z · N {xyz[1]:,.0f} m'),
                  (values[x, :, :].T, 1, 2, f'N–Z · E {xyz[0]:,.0f} m')]
        if self.comparison is not None:
            other = np.asarray(self.comparison.cell_data[self.field]).reshape(shape, order='F')
            if not np.isfinite(other).all():
                raise ValueError(f'{self.field} contains nonfinite comparison values.')
            # Unsigned/integer storage must not wrap a negative physical change.
            difference = other.astype(np.float64) - values.astype(np.float64)
            bound = max(float(np.abs(difference).max()), 1e-12)
            common.update(vmin=float(min(values.min(), other.min())), vmax=float(max(values.max(), other.max())))
            if 'Density' in self.field:
                total = max(abs(common['vmin']), abs(common['vmax']), 1e-12)
                common.update(vmin=-total, vmax=total)
            panels = [(values[:, :, z].T, 0, 1, 'A'), (other[:, :, z].T, 0, 1, 'B'), (difference[:, :, z].T, 0, 1, 'B − A')]
            self.readout.setText(self.readout.text() + f' · B: {other[x,y,z]:.6g} · Δ: {difference[x,y,z]:.6g}')
        artists = []
        for n, (ax, (plane, i, j, title)) in enumerate(zip(self.axes, panels)):
            style = dict(common)
            if categorical:
                labels = np.array(sorted(int(k) for k in categorical))
                plane = np.searchsorted(labels, plane)
                style = dict(cmap=ListedColormap([categorical[str(v)] for v in labels]), norm=BoundaryNorm(np.arange(len(labels)+1)-.5, len(labels)))
            if self.comparison is not None and n == 2:
                style = dict(cmap='RdBu_r', vmin=-bound, vmax=bound)
            artist = ax.pcolormesh(self.edges[i] / self.axis_scales[i], self.edges[j] / self.axis_scales[j], plane, shading='flat', **style)
            artists.append(artist)
            ax.axvline(xyz[i] / self.axis_scales[i], color='black', lw=.6)
            ax.axhline(xyz[j] / self.axis_scales[j], color='black', lw=.6)
            axis_labels = [f'{name} ({"km" if scale == 1000 else "m"})' for name, scale in zip(('E', 'N', 'Z'), self.axis_scales)]
            ax.set(xlabel=axis_labels[i], ylabel=axis_labels[j])
            ax.set_title(title, fontsize=10)
            ax.set_aspect(self.axis_scales[j] / self.axis_scales[i])
            ax.ticklabel_format(style='plain', useOffset=False)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
            ax.tick_params(labelsize=8)
        groups = [(artists[0], self.axes)] if self.comparison is None else [(artists[0], self.axes[:2]), (artists[2], self.axes[2:])]
        for artist, axes in groups:
            bar = self.figure.colorbar(artist, ax=axes, shrink=.7, fraction=.04, pad=.04)
            bar.ax.tick_params(labelsize=8)
            if categorical:
                bar.set_ticks(range(len(labels)), labels=[str(v) for v in labels])
        self.canvas.draw_idle()


class ModelComparison(QWidget):
    """Two saved models, identical coordinates and shared continuous colour limits."""

    def __init__(self, a, b, names=('A', 'B'), parent=None):
        super().__init__(parent)
        require_aligned(a, b)
        fields = [k for k in a.cell_data if k in b.cell_data and not k.lower().endswith(' id')
                  and np.asarray(a[k]).ndim == 1 and np.asarray(b[k]).ndim == 1
                  and np.asarray(a[k]).dtype.kind in 'iuf' and np.asarray(b[k]).dtype.kind in 'iuf']
        if not fields:
            raise ValueError('No shared continuous cell field is available for comparison.')
        if any(not np.isfinite(mesh.cell_data[field]).all() for mesh in (a, b) for field in fields):
            raise ValueError('Comparison fields contain nonfinite model values.')
        layout = QVBoxLayout(self)
        note = QLabel(f'A: {names[0]}\nB: {names[1]}\nSame grid · shared scale · difference B − A · no resampling')
        note.setWordWrap(True)
        layout.addWidget(note)
        select = QComboBox()
        select.addItems(fields)
        layout.addWidget(select)
        self.sections = ScientificSections(a, fields[0], cmap='RdBu_r' if 'Density' in fields[0] else 'viridis', comparison=b)
        select.currentTextChanged.connect(lambda name: self.sections.set_field(name, cmap='RdBu_r' if 'Density' in name else 'viridis'))
        layout.addWidget(self.sections)
