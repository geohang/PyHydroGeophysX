"""Native result adapters preserve model selection, geometry and physical units."""
import os
from types import SimpleNamespace
import numpy as np
import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
pytest.importorskip('PySide6')
from PySide6.QtWidgets import QApplication, QInputDialog
from PyHydroGeophysX.qt_apps.widgets.map_export import result_snapshot


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


def test_mesh_adapters_preserve_current_ert_step_and_srt_values(app):
    pg = pytest.importorskip('pygimli')
    mesh = pg.createGrid(x=[0., 5., 10.], y=[-5., 0.])
    values = np.array([100., 200.])
    manager = SimpleNamespace(paraDomain=mesh, model=values, coverage=lambda: [-1., 0.])
    page = SimpleNamespace(module_key='ert_processing', _inv_mgr=manager, _tl_models=None)
    meta, data = result_snapshot(page)
    np.testing.assert_array_equal(data['values'], values)
    assert meta['units'] == 'Ω·m'
    page._tl_models = np.column_stack([values, values*2])
    page._tl_mesh = mesh
    page._map_result_kind = 'timelapse'
    page._tl_step_combo = SimpleNamespace(currentIndex=lambda: 1, currentText=lambda: 'Second time')
    meta, data = result_snapshot(page)
    np.testing.assert_array_equal(data['values'], values*2)
    assert 'Second time' in meta['label']
    page._map_result_kind = 'single'
    np.testing.assert_array_equal(result_snapshot(page)[1]['values'], values)


@pytest.mark.parametrize('kind,method,unit', [('gravity','Gravity','g/cc'), ('magnetics','Magnetics','SI')])
def test_potential_field_result_and_observed_product(app, monkeypatch, kind, method, unit):
    model = np.arange(8.).reshape(2,2,2)-4
    page = SimpleNamespace(module_key='gravmag_processing', _fields={'Observed':np.array([-1.,2.])},
        _qc=None, _x=np.array([0.,1.]), _y=np.array([5.,6.]),
        _kind=SimpleNamespace(currentText=lambda:kind),
        _inv_result={'kind':kind, 'model3d':model, 'edges':[np.arange(3.)]*3})
    monkeypatch.setattr(QInputDialog, 'getItem', lambda *args: ('Recovered 3D model', True))
    meta, data = result_snapshot(page)
    assert meta['method'] == method and meta['units'] == unit
    np.testing.assert_array_equal(data['model3d'], model)
    monkeypatch.setattr(QInputDialog, 'getItem', lambda *args: ('Observed', True))
    meta, data = result_snapshot(page)
    assert meta['units'] == ('mGal' if method == 'Gravity' else 'nT')
    np.testing.assert_array_equal(data['values'], [-1,2])


def test_em_single_model_keeps_layer_values(app):
    from PyHydroGeophysX.forward.em1d import model_depth_profile
    depth, step = model_depth_profile(np.array([3.,7.]),np.array([50.,100.,250.]))
    page = SimpleNamespace(module_key='em_processing', _last_section=None,
                           _last_result={'method':'TDEM','depth':depth,'resistivity_step':step})
    meta, arrays = result_snapshot(page)
    np.testing.assert_array_equal(arrays['model3d'][0,0,::-1], [50,100,250])
    np.testing.assert_array_equal(arrays['depth_edges'], [0,3,10,30])


def test_joint_gravity_and_magnetics_select_the_requested_model(app, monkeypatch):
    gravity = np.ones((2,2,2))*.1
    magnetic = np.ones((2,2,2))*.005
    page = SimpleNamespace(module_key='joint_inversion', _result=SimpleNamespace(
        methods=('Gravity','Magnetics'), models={'Gravity':gravity, 'Magnetics':magnetic},
        meta={'edges':[np.arange(3.)]*3}))
    monkeypatch.setattr(QInputDialog, 'getItem', lambda *args: ('Magnetics', True))
    meta, data = result_snapshot(page)
    assert meta['method'] == 'Magnetics' and meta['units'] == 'SI'
    np.testing.assert_array_equal(data['model3d'], magnetic)
