import json
import os
from pathlib import Path

import numpy as np
import pytest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from PySide6.QtWidgets import QApplication, QMenu, QMessageBox, QToolBar

from PyHydroGeophysX.qt_apps.main_window import PyHydroGeophysXWorkbench
from PyHydroGeophysX.qt_apps.modules.model_viewer import ModelViewerModule
from PyHydroGeophysX.qt_apps.state import WorkbenchState


def _find_leaf(item):
    if item.data(0, 256):
        return item
    for index in range(item.childCount()):
        found = _find_leaf(item.child(index))
        if found is not None:
            return found
    return None


def test_tools_model_viewer_opens_and_selects_missing_artifact(
    monkeypatch, tmp_path: Path
) -> None:
    app = QApplication.instance() or QApplication([])
    context = tmp_path / "context.json"
    context.write_text(json.dumps({"output_dir": str(tmp_path)}), encoding="utf-8")
    window = PyHydroGeophysXWorkbench(context_path=str(context))
    store = window.state.ensure_results_store()
    assert store.root == tmp_path.resolve()
    handle = store.begin_run("ert", "ert.single_inversion", "ert.single_inversion")
    store.finish_run(handle, {
        "status": "success",
        "metrics": {"chi2": 1.2},
        "artifacts": [{"path": "outputs/missing.png", "kind": "image", "format": "png"}],
    })

    tools = None
    for menu_action in window.menuBar().actions():
        menu = menu_action.menu()
        if isinstance(menu, QMenu) and menu.title().replace("&", "") == "Tools":
            tools = menu
            break
    assert tools is not None
    action = next(action for action in tools.actions() if action.text() == "Model Viewer")
    action.trigger()
    app.processEvents()

    viewer = window._pages["model_viewer"]
    assert isinstance(viewer, ModelViewerModule)
    leaf = _find_leaf(viewer._tree.topLevelItem(0))
    assert leaf is not None
    leaf.setSelected(True)
    app.processEvents()
    assert handle.run_id in viewer._overview.toPlainText()
    assert viewer._files.rowCount() >= 3

    toolbar = window.findChild(QToolBar, "main_toolbar")
    save = next(action for action in toolbar.actions() if action.text() == "Save")
    monkeypatch.setattr(
        "PyHydroGeophysX.qt_apps.main_window.QMessageBox.information",
        lambda *_args, **_kwargs: QMessageBox.Ok,
    )
    save.trigger()
    assert window.state.result_path.is_file()

    window.close()
    app.processEvents()


def test_import_keeps_active_project(monkeypatch, tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])
    project = tmp_path / "project"
    context = tmp_path / "context.json"
    context.write_text(json.dumps({"output_dir": str(project)}), encoding="utf-8")
    legacy = tmp_path / "legacy"; legacy.mkdir()
    (legacy / "ert_recipe.json").write_text(
        json.dumps({"workflow_id": "ert.single_inversion"}), encoding="utf-8"
    )
    (legacy / "ert_process_result.json").write_text(
        json.dumps({"status": "ok"}), encoding="utf-8"
    )
    window = PyHydroGeophysXWorkbench(context_path=str(context))
    monkeypatch.setattr(
        "PyHydroGeophysX.qt_apps.main_window.QFileDialog.getExistingDirectory",
        lambda *_args, **_kwargs: str(legacy),
    )
    monkeypatch.setattr(
        "PyHydroGeophysX.qt_apps.main_window.QMessageBox.question",
        lambda *_args, **_kwargs: QMessageBox.Yes,
    )
    window._import_existing_results()
    app.processEvents()
    assert window.state.results_store_root == project.resolve()
    assert any(record.imported for record in window.state.results_store.list_runs())
    window.close(); app.processEvents()


def test_rendered_memmap_is_released_before_run_deletion(
    monkeypatch, tmp_path: Path
) -> None:
    app = QApplication.instance() or QApplication([])
    state = WorkbenchState(output_dir=tmp_path)
    store = state.set_results_store(tmp_path)
    handle = store.begin_run("ert", "ert.timelapse_inversion")
    model_path = handle.outputs_dir / "models.npy"
    np.save(model_path, np.arange(48.0).reshape(3, 4, 4))
    record = store.finish_run(handle, {
        "status": "success",
        "artifacts": [{
            "artifact_id": "models",
            "kind": "array_stack",
            "format": "npy",
            "path": "outputs/models.npy",
        }],
    })
    viewer = ModelViewerModule(state, lambda *_args: None)
    viewer._current = record
    viewer._current_artifacts = [record.artifacts[0]]
    viewer._artifact.clear()
    viewer._artifact.addItem("models", 0)
    viewer._render_selected_artifact(0)
    assert viewer._visual_resources
    monkeypatch.setattr(
        "PyHydroGeophysX.qt_apps.modules.model_viewer.QMessageBox.question",
        lambda *_args, **_kwargs: QMessageBox.Yes,
    )
    viewer._delete_run()
    app.processEvents()
    assert not handle.run_dir.exists()
    assert not viewer._visual_resources
    viewer.close(); app.processEvents()
