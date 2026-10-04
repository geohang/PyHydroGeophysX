"""Scientific viewer interactions preserve physical coordinates and source arrays."""

import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
import pytest

pv = pytest.importorskip('pyvista')
pytest.importorskip('PySide6')
from PySide6.QtWidgets import QApplication
from PyHydroGeophysX.qt_apps.widgets.scientific_sections import ScientificSections, ModelComparison, require_aligned


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def grid():
    mesh = pv.RectilinearGrid([100, 102, 109], [200, 203, 208], [-10, -8, 0])
    mesh.cell_data['Density contrast (g/cm3)'] = np.arange(8, dtype=float)
    mesh.cell_data['Susceptibility (SI)'] = np.arange(8, dtype=float) / 10
    return mesh


def test_linked_coordinates_and_field_change_preserve_selection(app):
    mesh = grid()
    original = mesh['Density contrast (g/cm3)'].copy()
    view = ScientificSections(mesh, 'Density contrast (g/cm3)', {'limits': [-7, 7]})
    view.select_point([106, 201, -4])
    assert view.indices == [1, 0, 1]
    assert '5' in view.readout.text()
    assert '105.500' in view.readout.text()
    assert view.coordinates[0].value() == 105.5
    view.set_field('Susceptibility (SI)')
    assert view.indices == [1, 0, 1]
    assert '0.5' in view.readout.text()
    np.testing.assert_array_equal(mesh['Density contrast (g/cm3)'], original)
    view.close()


def test_coordinate_arrows_visit_cells_with_unequal_spacing(app):
    mesh = pv.RectilinearGrid([0, 1, 2, 100], [0, 1], [0, 1])
    mesh.cell_data['Value'] = [1., 2., 3.]
    view = ScientificSections(mesh, 'Value')
    view.coordinates[0].stepBy(1)
    assert view.indices[0] == 2
    assert view.coordinates[0].value() == 51
    view.coordinates[0].stepBy(-1)
    assert view.indices[0] == 1
    view.close()


def test_kilometre_axis_click_preserves_metre_coordinates_and_aspect(app):
    from types import SimpleNamespace
    mesh = pv.RectilinearGrid([580000, 590000, 600000], [4780000, 4790000, 4800000], [-10, -8, 0])
    mesh.cell_data['Value'] = np.arange(8, dtype=float)
    view = ScientificSections(mesh, 'Value')
    view._clicked(SimpleNamespace(inaxes=view.axes[0], xdata=584, ydata=4784))
    assert view.indices[:2] == [0, 0]
    assert view.coordinates[0].value() == 585000
    assert view.axes[0].get_xlabel() == 'E (km)'
    assert view.axes[1].get_ylabel() == 'Z (m)'
    assert view.axes[1].get_aspect() == .001
    view.close()


def test_comparison_rejects_nonfinite_values_in_any_shared_field(app):
    a, b = grid(), grid()
    b.cell_data['Susceptibility (SI)'][0] = np.nan
    with pytest.raises(ValueError, match='nonfinite'):
        ModelComparison(a, b)


def test_linked_sections_remain_available_without_opengl(app, tmp_path, monkeypatch):
    from PyHydroGeophysX.qt_apps.widgets import model3d_view as views
    monkeypatch.setattr(views, 'try_import_pyvista', lambda: (False, None, None, 'No GL context'))
    mesh = grid()
    mesh.point_data['Point value'] = np.arange(mesh.n_points)
    path = tmp_path / 'model.vtk'
    mesh.save(path)
    view = views.VTKVolumeView()
    assert view.show_file(str(path), linked_sections=True)
    assert not view.interactive_available
    assert view._sections is not None
    assert view._field.findText('Point value') == -1
    view._field.setCurrentText('Susceptibility (SI)')
    assert view._sections.field == 'Susceptibility (SI)'
    view.close()


def test_comparison_uses_shared_limits_and_exact_difference(app):
    a, b = grid(), grid()
    b.cell_data['Density contrast (g/cm3)'] += 2
    view = ModelComparison(a, b)
    axes = view.sections.axes
    assert axes[0].collections[0].get_clim() == axes[1].collections[0].get_clim() == (-9, 9)
    np.testing.assert_array_equal(axes[2].collections[0].get_array(), np.full((2, 2), 2))
    assert axes[2].collections[0].get_clim() == (-2, 2)
    view.close()


def test_comparison_rejects_different_coordinates_even_with_same_shape():
    a, b = grid(), grid()
    b.x = b.x + 1
    with pytest.raises(ValueError, match='grids differ'):
        require_aligned(a, b)


def test_background_and_group_colours_come_from_exported_metadata(app):
    mesh = grid()
    mesh.cell_data['Geo ID'] = [0, 7] * 4
    meta = {'colors': {'0': '#efeff5', '7': '#d62728'}, 'names': {'7': 'Recorded group'}}
    view = ScientificSections(mesh, 'Geo ID', meta)
    view.select_point([106, 201, -4])
    assert 'Recorded group' in view.readout.text()
    assert view.axes[0].collections[0].cmap.colors == ['#efeff5', '#d62728']
    view.close()


def test_clipping_plane_survives_property_switch(app, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QWidget
    from PyHydroGeophysX.qt_apps.widgets import model3d_view as views

    class Plane:
        def GetNormal(self): return (0, 1, 0)
        def GetOrigin(self): return (104, 203, -7)

    class Plotter:
        def __init__(self, *args, **kwargs):
            self.interactor = QWidget()
            self.camera_position = 'camera'
            self.plane_widgets = []
            self.clip = None
        def clear_plane_widgets(self): self.plane_widgets = []
        def add_mesh_clip_plane(self, mesh, **kwargs): self.clip = kwargs
        def __getattr__(self, name): return lambda *a, **kw: None

    monkeypatch.setattr(views, 'try_import_pyvista', lambda: (True, pv, Plotter, ''))
    path = tmp_path / 'model.vtk'
    grid().save(path)
    view = views.VTKVolumeView()
    view.show_file(str(path))
    view._clip_cb.setChecked(True)
    view._plotter.plane_widgets = [Plane()]
    view._field.setCurrentText('Susceptibility (SI)')
    assert view._plotter.clip['normal'] == (0, 1, 0)
    assert view._plotter.clip['origin'] == (104, 203, -7)
    assert view._plotter.camera_position == 'camera'
    view.close()
