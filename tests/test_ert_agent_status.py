"""The assistant can review an ERT inversion before exporting it to Map."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
import pytest


@pytest.fixture
def page(tmp_path, monkeypatch):
    pytest.importorskip('PySide6')
    from PySide6.QtWidgets import QApplication
    from PyHydroGeophysX.qt_apps.modules.ert_processing import ERTProcessingModule
    from PyHydroGeophysX.qt_apps.state import StudioState
    app = QApplication.instance() or QApplication([])
    widget = ERTProcessingModule(StudioState(output_dir=tmp_path), lambda *args: None)
    # Exercise the real success/status handlers without native plotting or Map.
    monkeypatch.setattr(widget, '_show_single_model', lambda *args: None)
    monkeypatch.setattr(widget, 'offer_map_export', lambda: None)
    yield widget
    widget.close()
    widget.deleteLater()
    app.processEvents()


def complete_single(page):
    page._data_path = Path('original.ohm')
    page._on_inversion_ok({
        'mgr': SimpleNamespace(model=np.array([54., 120., 3694.]), paraDomain=None),
        'chi2': 1.2, 'lambda_used': 50., 'lambda_requested': 50.,
        'vtk': 'resistivity.vtk', 'engine': 'pyhydro',
        'metrics': {'rrms': 3.5, 'iterations': 7, 'n_data': 3647},
    })


def test_completed_model_and_fit_reach_chat_before_map_export(page):
    from PyHydroGeophysX.qt_apps.agent.chat_panel import AssistantChatPanel
    complete_single(page)
    # Loading a different survey must not change the existing model's source.
    page._data_path = Path('preview-only.ohm')
    status = page.agent_apply('get_status', {})
    assert status['inversion_status'] == 'completed'
    assert status['has_model'] is True and status['inversion_running'] is False
    result = status['result_summary']
    assert result['data_file'] == 'original.ohm'
    assert (result['chi2'], result['rrms'], result['iterations'], result['lambda']) == (1.2, 3.5, 7, 50.)
    assert result['resistivity_range_ohm_m'] == [54., 3694.]
    assert result['model_cells'] == 3
    assert result['resistivity_vtk'] == 'resistivity.vtk'
    # The neutral model context contains the real status, not its 600-char UI preview.
    context = AssistantChatPanel._compact_for_context(status)
    assert context['result_summary'] == result
    assert len(json.dumps(context)) > 600
    json.dumps(context, allow_nan=False)


def test_running_and_failed_new_run_keep_previous_result_distinct(page):
    complete_single(page)
    page._inv_worker = SimpleNamespace(isRunning=lambda: True)
    page._agent_run_state = 'running'
    status = page.agent_apply('get_status', {})
    assert status['inversion_status'] == 'running'
    assert status['inversion_running'] is True and status['has_model'] is True
    page._inv_worker = None
    page._on_inversion_failed('Synthetic solver failure')
    status = page.agent_apply('get_status', {})
    assert status['inversion_status'] == 'failed'
    assert status['inversion_error'] == 'Synthetic solver failure'
    assert status['has_model'] is True
    assert status['result_summary']['chi2'] == 1.2


def test_selected_comparison_model_uses_its_own_fit(page):
    complete_single(page)
    page._inv_choices.append({
        'label': 'Comparison', 'mgr': SimpleNamespace(model=np.array([80., 200.])),
        'metrics': {'iterations': 5}, 'chi2': 2.3, 'lambda': 30., 'vtk': 'fixed.vtk',
    })
    page._lam_pick.blockSignals(True)
    page._lam_pick.addItem('Comparison')
    page._lam_pick.setCurrentIndex(1)
    page._lam_pick.blockSignals(False)
    page._inv_mgr = page._inv_choices[1]['mgr']
    result = page.agent_apply('get_status', {})['result_summary']
    assert (result['chi2'], result['lambda'], result['iterations']) == (2.3, 30., 5)
    assert result['rrms'] is None  # Do not borrow a different model's fit.
    assert result['resistivity_range_ohm_m'] == [80., 200.]
    assert result['resistivity_vtk'] == 'fixed.vtk'


def test_timelapse_model_is_available_without_single_manager(page):
    page._map_result_kind = 'timelapse'
    page._tl_models = np.array([[100., 110.], [200., 220.]])
    page._tl_mesh = object()
    page._tl_out = 'timelapse-output'
    page._tl_result = {'chi2': 1.1, 'rrms': float('nan'), 'engine': 'pyhydro'}
    status = page.agent_apply('get_status', {})
    assert status['has_model'] is True
    assert status['inversion_status'] == 'completed'
    assert status['result_summary']['kind'] == 'timelapse'
    assert status['result_summary']['time_steps'] == 2
    assert status['result_summary']['chi2'] == 1.1
    assert status['result_summary']['rrms'] is None
    assert status['result_summary']['resistivity_range_ohm_m'] == [100., 220.]
    json.dumps(status, allow_nan=False)


def test_empty_module_does_not_claim_completed_inversion(page):
    status = page.agent_apply('get_status', {})
    assert status['inversion_status'] == 'idle'
    assert status['has_model'] is False
    assert status['result_summary'] == {}


def test_tabs_show_stage_controls_and_preserve_settings(page):
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance()
    page.resize(1240, 860)
    page.show()
    page._lam.setValue(30.)
    app.processEvents()
    assert page._load_group.isVisible()
    assert page._geometry_export_group.isVisible()
    assert not page._qc_group.isVisible()
    assert not page._inversion_group.isVisible()
    width_with_preparation = page._tabs.width()

    page._tabs.setCurrentWidget(page._pseudo_widget)
    app.processEvents()
    assert page._qc_group.isVisible() and page._errors_group.isVisible()
    assert page._inversion_group.isVisible() and page._run_group.isVisible()
    assert not page._load_group.isVisible()

    for tab in (page._mesh_tab, page._model_tab):
        page._tabs.setCurrentWidget(tab)
        app.processEvents()
        assert not page._controls.isVisible()
        assert page._tabs.width() > width_with_preparation + 300
    assert page._tc_box.isVisible()
    assert page._model_export_group.isVisible()
    assert page._model_export_btn.parent() is page._model_export_group

    page._tabs.setCurrentWidget(page._quality_view)
    app.processEvents()
    assert page._inversion_group.isVisible() and page._fit_group.isVisible()
    assert page._run_group.isVisible()
    assert not page._qc_group.isVisible() and not page._load_group.isVisible()
    page._tabs.setCurrentWidget(page._plot_widget)
    app.processEvents()
    assert page._load_group.isVisible() and page._geometry_export_group.isVisible()
    assert page._lam.value() == 30.
