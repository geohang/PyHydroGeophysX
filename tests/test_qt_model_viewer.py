import json
import os
from pathlib import Path

import numpy as np
import pytest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
try:
    # QtGui, not the PySide6 package: the offscreen platform still loads
    # libEGL/libGL, so a machine without them raises here rather than at the
    # top-level import. That is a plain ImportError, and pytest.importorskip
    # skips only on ModuleNotFoundError, so it would break collection instead
    # of skipping the module.
    import PySide6.QtGui  # noqa: F401
    import pyqtgraph  # noqa: F401
except ImportError as exc:  # pragma: no cover - environment dependent
    pytest.skip(f"Qt stack unavailable: {exc}", allow_module_level=True)

from PySide6.QtWidgets import QApplication, QMenu, QMessageBox

from PyHydroGeophysX.qt_apps.main_window import PyHydroGeophysXStudio
from PyHydroGeophysX.qt_apps.modules.model_viewer import ModelViewerModule
from PyHydroGeophysX.qt_apps.state import StudioState


@pytest.fixture(autouse=True)
def _answer_modal_prompts(monkeypatch):
    """Answer the window's modal prompts so a headless close cannot hang.

    ``closeEvent`` asks what to do with runs that were never saved to the
    Project. Under the offscreen platform a modal dialog has nobody to answer
    it, so any test that closes a window holding a staged run would block until
    the suite times out. Discard is the right default for a throwaway Project; a
    test that cares about the answer patches over this one.
    """
    monkeypatch.setattr(
        "PyHydroGeophysX.qt_apps.main_window.QMessageBox.question",
        lambda *_args, **_kwargs: QMessageBox.Discard,
    )


def _find_leaf(item):
    if item.data(0, 256):
        return item
    for index in range(item.childCount()):
        found = _find_leaf(item.child(index))
        if found is not None:
            return found
    return None


def _menu_action(window, menu_title: str, action_text: str):
    """Find an action by label under a top-level menu, descending into submenus.

    The whole traversal happens here rather than returning a ``QMenu`` to the
    caller: PySide6 destroys the C++ menu as soon as the last Python wrapper
    goes out of scope, so a helper that hands one back leaves the caller holding
    a deleted object. ``alive`` keeps every wrapper referenced until the search
    finishes.
    """
    alive: list = []

    def search(container):
        for action in container.actions():
            if action.text() == action_text:
                return action
            submenu = action.menu()
            if isinstance(submenu, QMenu):
                alive.append(submenu)
                found = search(submenu)
                if found is not None:
                    return found
        return None

    for menu_action in window.menuBar().actions():
        menu = menu_action.menu()
        if not isinstance(menu, QMenu) or menu.title().replace("&", "") != menu_title:
            continue
        alive.append(menu)
        found = search(menu)
        if found is not None:
            return found
    raise AssertionError(f"No {action_text!r} action under the {menu_title!r} menu.")


def test_tools_model_viewer_opens_and_selects_missing_artifact(
    monkeypatch, tmp_path: Path
) -> None:
    app = QApplication.instance() or QApplication([])
    context = tmp_path / "context.json"
    context.write_text(json.dumps({"output_dir": str(tmp_path)}), encoding="utf-8")
    window = PyHydroGeophysXStudio(context_path=str(context))
    store = window.state.ensure_results_store()
    assert store.root == tmp_path.resolve()
    handle = store.begin_run("ert", "ert.single_inversion", "ert.single_inversion")
    store.finish_run(handle, {
        "status": "success",
        "metrics": {"chi2": 1.2},
        "artifacts": [{"path": "outputs/missing.png", "kind": "image", "format": "png"}],
    })

    _menu_action(window, "Tools", "Model Viewer").trigger()
    app.processEvents()

    viewer = window._pages["model_viewer"]
    assert isinstance(viewer, ModelViewerModule)
    leaf = _find_leaf(viewer._tree.topLevelItem(0))
    assert leaf is not None
    leaf.setSelected(True)
    app.processEvents()
    assert handle.run_id in viewer._overview.toPlainText()
    assert viewer._files.rowCount() >= 3

    monkeypatch.setattr(
        "PyHydroGeophysX.qt_apps.main_window.QMessageBox.information",
        lambda *_args, **_kwargs: QMessageBox.Ok,
    )
    # The bridge manifest moved off the toolbar into File > Streamlit Bridge,
    # because it wrote JSON for the Streamlit app while reading as the button
    # that saved your science.
    _menu_action(window, "File", "Save Studio Result").trigger()
    assert window.state.result_path.is_file()

    window.close()
    app.processEvents()


def test_export_results_reports_when_a_module_has_nothing_to_write(
    monkeypatch, tmp_path: Path
) -> None:
    """``Export Results…`` explains an empty module instead of doing nothing."""
    app = QApplication.instance() or QApplication([])
    context = tmp_path / "context.json"
    context.write_text(json.dumps({"output_dir": str(tmp_path)}), encoding="utf-8")
    window = PyHydroGeophysXStudio(context_path=str(context))

    shown: list = []
    monkeypatch.setattr(
        "PyHydroGeophysX.qt_apps.main_window.QMessageBox.information",
        lambda _parent, _title, text, *_a, **_k: shown.append(text) or QMessageBox.Ok,
    )
    export = _menu_action(window, "File", "Export Results…")
    export.trigger()
    app.processEvents()
    assert shown and "no results to export yet" in shown[0]
    assert str(tmp_path.resolve()) in shown[0]

    window.close()
    app.processEvents()


def test_export_results_runs_a_modules_only_offer(monkeypatch, tmp_path: Path) -> None:
    """A single offer runs directly; several would raise a chooser instead."""
    app = QApplication.instance() or QApplication([])
    context = tmp_path / "context.json"
    context.write_text(json.dumps({"output_dir": str(tmp_path)}), encoding="utf-8")
    window = PyHydroGeophysXStudio(context_path=str(context))

    called: list = []
    page = window.current_module()
    monkeypatch.setattr(
        type(page), "export_actions",
        lambda _self: [("Only offer", lambda: called.append("ran"))],
        raising=False,
    )
    _menu_action(window, "File", "Export Results…").trigger()
    app.processEvents()
    assert called == ["ran"]

    chosen: list = []
    monkeypatch.setattr(
        type(page), "export_actions",
        lambda _self: [("First", lambda: chosen.append("first")),
                       ("Second", lambda: chosen.append("second"))],
        raising=False,
    )
    monkeypatch.setattr(
        "PyHydroGeophysX.qt_apps.main_window.QInputDialog.getItem",
        lambda *_args, **_kwargs: ("Second", True),
    )
    _menu_action(window, "File", "Export Results…").trigger()
    app.processEvents()
    assert chosen == ["second"]

    window.close()
    app.processEvents()


def test_module_result_export_uses_the_pages_own_key(monkeypatch, tmp_path: Path) -> None:
    """The navigator key and the page's ``module_key`` differ on four modules.

    ``ert`` in the navigator builds a page whose ``module_key`` is
    ``ert_processing``, and a module publishes its result under the page's key.
    Looking it up by the navigator's key found nothing on exactly the modules
    that had something, and reported that as "no result to export".
    """
    app = QApplication.instance() or QApplication([])
    context = tmp_path / "context.json"
    context.write_text(json.dumps({"output_dir": str(tmp_path)}), encoding="utf-8")
    window = PyHydroGeophysXStudio(context_path=str(context))

    window.show_module("ert")
    app.processEvents()
    page = window.current_module()
    assert page.module_key != window.state.selected_module, (
        "This test is meaningless unless the two keys actually differ."
    )
    page.report_result({"num_electrodes": 24})

    target = tmp_path / "ert_result.json"
    monkeypatch.setattr(
        "PyHydroGeophysX.qt_apps.main_window.QFileDialog.getSaveFileName",
        lambda *_args, **_kwargs: (str(target), "JSON (*.json)"),
    )
    _menu_action(window, "File", "Export Module Result (JSON)…").trigger()
    app.processEvents()
    assert target.is_file(), "the page's published result was not found"
    assert json.loads(target.read_text(encoding="utf-8"))["num_electrodes"] == 24

    window.close()
    app.processEvents()


def _window_with_finished_run(tmp_path: Path):
    """A window whose Project holds one finished, unsaved run."""
    app = QApplication.instance() or QApplication([])
    context = tmp_path / "context.json"
    context.write_text(json.dumps({"output_dir": str(tmp_path)}), encoding="utf-8")
    window = PyHydroGeophysXStudio(context_path=str(context))
    handle = window.state.begin_run("ert", "ert.single_inversion", "ert.single_inversion")
    (handle.outputs_dir / "model.npy").write_bytes(b"model")
    window.state.finish_run("ert", {"status": "ok", "metrics": {"chi2": 1.1}},
                            "ert.single_inversion")
    return app, window, handle


def test_a_finished_run_is_advertised_as_unsaved(tmp_path: Path) -> None:
    """The count has to be on screen; a run nobody saves is a run nobody keeps."""
    app, window, handle = _window_with_finished_run(tmp_path)
    assert window.state.has_unsaved_runs()
    assert "1 unsaved run" in window._unsaved_label.text()
    assert window._save_action.isEnabled()
    assert window.state.results_store.list_runs() == []
    assert not handle.record.metadata_path.exists()

    window.close()
    app.processEvents()


def test_saving_runs_puts_them_in_the_project(tmp_path: Path) -> None:
    app, window, handle = _window_with_finished_run(tmp_path)
    _menu_action(window, "File", "Save Runs to Project").trigger()
    app.processEvents()

    assert handle.record.metadata_path.is_file()
    assert [item.run_id for item in window.state.results_store.list_runs()] == [
        handle.run_id]
    assert window._unsaved_label.text() == ""
    assert not window._save_action.isEnabled()

    window.close()
    app.processEvents()


def test_closing_with_unsaved_runs_offers_save_discard_or_cancel(
    monkeypatch, tmp_path: Path
) -> None:
    """Cancel must abort the close, and Save must keep the run."""
    app, window, handle = _window_with_finished_run(tmp_path)

    monkeypatch.setattr(
        "PyHydroGeophysX.qt_apps.main_window.QMessageBox.question",
        lambda *_args, **_kwargs: QMessageBox.Cancel,
    )
    window.close()
    app.processEvents()
    assert window.isVisible() or window.state.has_unsaved_runs(), (
        "Cancel must not close the window nor resolve the runs"
    )
    assert window.state.has_unsaved_runs()
    assert handle.run_dir.is_dir()

    monkeypatch.setattr(
        "PyHydroGeophysX.qt_apps.main_window.QMessageBox.question",
        lambda *_args, **_kwargs: QMessageBox.Save,
    )
    window.close()
    app.processEvents()
    assert handle.record.metadata_path.is_file()
    assert not window.state.has_unsaved_runs()


def test_closing_and_discarding_removes_the_run_folder(monkeypatch, tmp_path: Path) -> None:
    app, window, handle = _window_with_finished_run(tmp_path)
    monkeypatch.setattr(
        "PyHydroGeophysX.qt_apps.main_window.QMessageBox.question",
        lambda *_args, **_kwargs: QMessageBox.Discard,
    )
    window.close()
    app.processEvents()
    assert not handle.run_dir.exists()
    assert not window.state.has_unsaved_runs()


def test_model_viewer_lists_unsaved_runs_and_can_save_one(tmp_path: Path) -> None:
    app, window, handle = _window_with_finished_run(tmp_path)
    window.show_module("model_viewer")
    app.processEvents()
    viewer = window._pages["model_viewer"]

    assert handle.run_id in viewer._unsaved_ids
    top = viewer._tree.topLevelItem(0)
    assert "Unsaved" in top.text(0), "unsaved runs must lead the tree"

    leaf = _find_leaf(top)
    assert leaf is not None
    leaf.setSelected(True)
    app.processEvents()
    # isVisible() is False while the window itself is never shown, so ask what
    # the button would be if the page were on screen.
    assert viewer._save_run.isVisibleTo(viewer)
    assert viewer._save_run.isEnabled()
    assert viewer._delete.text() == "Discard…"

    viewer._save_current_run()
    app.processEvents()
    assert handle.record.metadata_path.is_file()
    assert handle.run_id not in viewer._unsaved_ids

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
    window = PyHydroGeophysXStudio(context_path=str(context))
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
    state = StudioState(output_dir=tmp_path)
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
