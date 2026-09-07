"""Behavioral regressions for defects found during the comment review."""
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.mark.parametrize('mode', ['truncate', 'individual'])
@pytest.mark.parametrize('reject', [True, False])
def test_sign_filter_without_noise_threshold(mode, reject):
    from PyHydroGeophysX.data_processing.em1d import _gate_disposition, _temcompany_valid_channels
    times = np.arange(1., 6.)
    response = np.array([-1., -2., -3., 4., 5.])
    status, _, _ = _gate_disposition(times, response, flags=[0, 1, 1, 1, 1],
                                    gate_rejection=mode, reject_negative=reject)
    assert status.tolist() == ['flagged out'] + (['reversed sign'] * 2 if reject else ['kept'] * 2) + ['kept'] * 2
    kept, _, _ = _temcompany_valid_channels(times, response, flags=[0, 1, 1, 1, 1],
                                           gate_rejection=mode, reject_negative=reject)
    np.testing.assert_array_equal(kept, times[3:] if reject else times[1:])


def mesh_utils():
    pytest.importorskip('pygimli')
    from PyHydroGeophysX.core import mesh_utils
    return mesh_utils


def test_layer_markers_and_legacy_elevation_alias():
    utils = mesh_utils()
    surface = np.array([[0., 0.], [10., 0.]])
    layers = [surface - [0, 2], surface - [0, 5]]
    creator = utils.MeshCreator()
    mesh, geom = creator.create_from_layers(surface, layers, bottom_elevation=-10, markers=[11, 22, 33])
    centers, marks = np.asarray(mesh.cellCenters()), np.asarray(mesh.cellMarkers())
    assert set(marks) == {11, 22, 33}
    np.testing.assert_array_equal(marks, np.where(centers[:, 1] > -2, 11, np.where(centers[:, 1] > -5, 22, 33)))
    assert min(p.y() for p in mesh.positions()) == pytest.approx(-10)
    for edge in mesh.boundaries():
        if edge.marker() == -1:
            assert all(n.pos().y() == pytest.approx(0) for n in edge.nodes())
    old, _ = creator.create_from_layers(surface, layers, bottom_depth=-10, markers=[11, 22, 33])
    np.testing.assert_allclose(np.asarray(old.cellCenters()), centers)
    np.testing.assert_array_equal(old.cellMarkers(), marks)
    default, _ = creator.create_from_layers(surface, layers, bottom_elevation=-10)
    assert set(default.cellMarkers()) == {2, 3}
    with pytest.raises(ValueError, match='only one'):
        creator.create_from_layers(surface, layers, bottom_depth=-10, bottom_elevation=-20)
    with pytest.raises(ValueError, match='three integers'):
        creator.create_from_layers(surface, layers, bottom_elevation=-10, markers=[1, 2])


def test_ert_depth_changes_parameter_domain():
    utils = mesh_utils()
    from pygimli.physics import ert
    data = ert.createData(elecs=np.linspace(0., 20., 11), schemeName='dd')
    meshes = [utils.MeshCreator().create_from_ert_data(data, max_depth=depth) for depth in (5., 12.)]
    for mesh, depth in zip(meshes, (5., 12.)):
        cells = [c for c in mesh.cells() if c.marker() == 2]
        bottom = min(n.pos().y() for c in cells for n in c.nodes())
        assert bottom == pytest.approx(-depth)
    with pytest.raises(ValueError, match='positive'):
        utils.MeshCreator().create_from_ert_data(data, max_depth=0)


@pytest.mark.parametrize('entry', ['core', 'seismic'])
@pytest.mark.parametrize('crossings', [0, 1, 2])
def test_velocity_interface_endpoints(entry, crossings):
    utils = mesh_utils()
    from PyHydroGeophysX.Geophy_modular.seismic_processor import extract_velocity_structure
    # Two interior bins follow the analytic interface z = 2*x - 20.
    points = [[0., -20.], [10., -20.]]
    velocities = [500., 500.]
    for x in [3., 7.][:crossings]:
        z = 2 * x - 20
        points.extend([[x, z - 1], [x, z + 1]])
        velocities.extend([1000., 1400.])
    mesh = SimpleNamespace(cellCenters=lambda: np.asarray(points))
    fn = utils.extract_velocity_interface if entry == 'core' else extract_velocity_structure
    if not crossings:
        with pytest.raises(ValueError, match='No velocity threshold crossing'):
            fn(mesh, np.asarray(velocities), interval=2)
        return
    x, z, *_ = fn(mesh, np.asarray(velocities), interval=2)
    expected = 2 * x - 20 if crossings == 2 else np.full_like(x, -14.)
    np.testing.assert_allclose(z, expected, atol=1e-10)


def test_task_cancellation_skips_work_and_stops_at_log_checkpoint():
    pytest.importorskip('PySide6')
    from PyHydroGeophysX.qt_apps.workers import TaskWorker
    events = []
    cancelled = TaskWorker(lambda: events.append('should not start'))
    cancelled.cancel()
    cancelled.run()
    assert not events

    def work(log):
        log('before')
        worker.cancel()
        log('after')
        events.append('should not continue')

    worker = TaskWorker(work, with_log=True)
    worker.logged.connect(events.append)
    worker.succeeded.connect(lambda _: events.append('success'))
    worker.failed.connect(lambda _: events.append('failure'))
    worker.run()
    assert events == ['before']


def test_task_cooperative_cancellation_callback():
    pytest.importorskip('PySide6')
    from PyHydroGeophysX.qt_apps.workers import TaskWorker
    seen = []
    def work(cancelled):
        seen.append(cancelled())
        worker.cancel()
        seen.append(cancelled())
    worker = TaskWorker(work, with_cancel=True)
    worker.run()
    assert seen == [False, True]


def test_running_task_stops_on_next_log_after_cancellation():
    pytest.importorskip('PySide6')
    from threading import Event
    from PyHydroGeophysX.qt_apps.workers import TaskWorker
    entered, release, continued = Event(), Event(), Event()
    def work(log):
        entered.set()
        assert release.wait(5)
        log('cancel checkpoint')
        continued.set()
    worker = TaskWorker(work, with_log=True)
    worker.start()
    try:
        assert entered.wait(5)
        worker.cancel()
    finally:
        release.set()
        assert worker.wait(5000)
    assert not continued.is_set()


def test_workflow_cancellation_and_progress_checkpoint(monkeypatch):
    pytest.importorskip('PySide6')
    from PyHydroGeophysX.qt_apps import workers
    events = []
    context = SimpleNamespace()
    def run(spec, runtime):
        runtime.progress('before')
        worker.cancel()
        runtime.progress('after')
        events.append('continued')
    monkeypatch.setattr(workers, 'run_workflow', run)
    worker = workers.WorkflowWorker(None, context)
    worker.logged.connect(events.append)
    worker.failed.connect(events.append)
    worker.run()
    assert events == ['before']
    events.clear()
    worker.run()
    assert not events


def test_cancelled_process_discards_new_progress(tmp_path):
    pytest.importorskip('PySide6')
    from PyHydroGeophysX.qt_apps.workers import ProcessWorkflowWorker
    worker = ProcessWorkflowWorker(tmp_path / 'recipe.json', tmp_path,
                                   tmp_path / 'output', tmp_path / 'result.json')
    logs, progress = [], []
    worker.logged.connect(logs.append)
    worker.progressed.connect(lambda *args: progress.append(args))
    worker.cancel()
    worker._emit_output(b'[progress 1/2] late output\n')
    assert not logs and not progress
