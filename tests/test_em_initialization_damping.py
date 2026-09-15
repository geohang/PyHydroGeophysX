"""Single-station initialization and fixed-reference regularization."""
import numpy as np
import pytest

from PyHydroGeophysX.inversion.em1d import _occam_1d, _occam_with_optional_rejection


def forward(sigma):
    return np.log10(1.0 / np.asarray(sigma))[:2]


def jacobian(sigma):
    return np.diag(-1.0 / (np.log(10) * np.asarray(sigma)))[:2]


@pytest.mark.parametrize('analytic', [False, True])
def test_damping_holds_unobserved_layer_at_reference(analytic):
    settings = dict(smoothness=1., starting_resistivity=100., max_iterations=100,
                    model_damping=10., starting_model=[30., 30., 30.])
    got = _occam_1d(forward, np.log10([30., 30.]), np.full(2, .001),
                    3, settings, lambda _: None, jacobian if analytic else None)[0]
    assert got[-1] > 95.
    np.testing.assert_allclose(got[:2], 30., rtol=.002)


def test_auto_search_runs_once_and_preserves_settings():
    settings = dict(smoothness=.3, max_iterations=50, rho_min=1., rho_max=1000.,
                    robust_errors=True, robust_passes=2)
    logs=[]
    got = _occam_with_optional_rejection(
        forward, np.log10([30.,30.]), np.full(2,.01), 3, settings, logs.append, jacobian)
    assert len([s for s in logs if s.startswith('Automatic starting')]) == 1
    assert 'reference_model' not in settings
    np.testing.assert_allclose(got[0][:2],30.,rtol=.01)


def test_manual_start_skips_scan():
    logs=[]
    _occam_with_optional_rejection(forward,np.log10([30.,30.]),np.full(2,.01),3,
        dict(auto_starting_model=False,starting_resistivity=30.,model_damping=0.),logs.append,jacobian)
    assert not any(s.startswith('Automatic starting') for s in logs)


def test_qt_collects_damping_and_auto_start(monkeypatch):
    monkeypatch.setenv('QT_QPA_PLATFORM', 'offscreen')
    widgets = pytest.importorskip('PySide6.QtWidgets')
    from PyHydroGeophysX.qt_apps.modules.em_processing import EMProcessingModule
    app = widgets.QApplication.instance() or widgets.QApplication([])
    panel = EMProcessingModule(None, lambda *a, **k: None)
    try:
        settings = panel._collect_inv()
        assert settings['model_damping'] == .4
        assert settings['auto_starting_model'] is True
        panel._model_damping.setValue(.8)
        assert panel._collect_inv()['model_damping'] == .8
    finally:
        panel.close()
        panel.deleteLater()
        app.processEvents()


def test_neighbor_starts_reject_isolated_spike_and_respect_lines_and_gaps():
    from PyHydroGeophysX.workflows.em1d import _neighboring_starts
    raw = np.array([20.,22.,10000.,24.,26.,500.,520.,540.])
    positions = np.array([0.,10.,20.,30.,40.,0.,10.,20.])
    lines = np.array([1,1,1,1,1,2,2,2])
    got = _neighboring_starts(raw,positions,lines,(1.,10000.))
    assert 20 <= got[2] <= 26
    assert np.all(got[5:] >= 500)
    separated = _neighboring_starts([20.,22.,500.,520.], [0.,10.,1000.,1010.],
                                   [1,1,1,1], (1.,10000.))
    assert separated[1] < 25 and separated[2] > 490


def test_neighbor_start_is_used_as_fixed_reference():
    logs=[]
    fit = _occam_with_optional_rejection(forward,np.log10([30.,30.]),np.full(2,.01),3,
        dict(neighbor_starting_resistivity=25.),logs.append,jacobian)
    initialization=fit[-1]['initialization']
    assert initialization['starting_resistivity']==25.
    assert initialization['reference_model']==[25.]*3
    assert not any(s.startswith('Automatic starting half-space') for s in logs)


def test_bound_search_compares_layered_starts_with_fixed_reference(monkeypatch):
    from PyHydroGeophysX.inversion import em1d
    seen=[]
    def layered(f, obs, unc, n, settings, log, jac):
        seen.append(settings.copy())
        start=settings['starting_resistivity']
        return np.full(n,20.), abs(np.log10(start)-1.), 1, []
    monkeypatch.setattr(em1d,'_occam_1d',layered)
    result=em1d._occam_with_optional_rejection(
        lambda sigma: sigma[:2],np.zeros(2),np.ones(2),3,
        dict(rho_min=1.,rho_max=10000.,reference_model=np.full(3,50.)),lambda _:None)
    assert len(seen)>2
    for settings in seen:
        np.testing.assert_array_equal(settings['reference_model'],[50.]*3)
    assert result[-1]['initialization']['starting_resistivity'] < 100.
