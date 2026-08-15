import json
from pathlib import Path

import numpy as np
import pytest

from PyHydroGeophysX.qt_apps import results_store as results_store_module
from PyHydroGeophysX.qt_apps.results_store import ResultsStore, normalize_status
from PyHydroGeophysX.qt_apps.state import WorkbenchState
from PyHydroGeophysX.workflows import ArtifactRef, WorkflowRunResult


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("ok", "success"), ("completed", "success"), ("error", "failed"),
     ("canceled", "cancelled"), ("something-new", "unknown")],
)
def test_status_mapping(raw, expected):
    assert normalize_status(raw) == expected


def test_multiple_runs_do_not_overwrite_and_index_rebuilds(tmp_path: Path) -> None:
    store = ResultsStore.open_or_create(tmp_path)
    first = store.begin_run("ert", "ert.single_inversion", "ert.single_inversion")
    second = store.begin_run("ert", "ert.single_inversion", "ert.single_inversion")
    assert first.run_dir != second.run_dir

    output = first.outputs_dir / "model.npy"
    output.write_bytes(b"model")
    result = WorkflowRunResult(
        status="ok",
        metrics={"chi2": 1.1},
        artifacts=[ArtifactRef.from_path(
            output, artifact_id="ert:model", kind="resistivity_model",
            base_dir=first.run_dir,
        )],
    )
    store.finish_run(first, result)
    store.cancel_run(second)
    store.save_all_unsaved()

    store.index_path.write_text("not json", encoding="utf-8")
    reopened = ResultsStore.open_or_create(tmp_path)
    records = reopened.list_runs()
    assert {record.status for record in records} == {"success", "cancelled"}
    assert len(records) == 2
    assert json.loads(reopened.index_path.read_text(encoding="utf-8"))["runs"]


def test_a_finished_run_stays_out_of_the_history_until_saved(tmp_path: Path) -> None:
    """A computation is the user's to keep, so nothing records it on its own."""
    store = ResultsStore.open_or_create(tmp_path)
    handle = store.begin_run("ert", "ert.single_inversion", "ert.single_inversion")
    (handle.outputs_dir / "model.npy").write_bytes(b"model")
    store.finish_run(handle, {"status": "ok", "metrics": {"chi2": 1.1}})

    assert store.list_runs() == []
    assert store.has_unsaved()
    assert [record.run_id for record in store.list_unsaved_runs()] == [handle.run_id]
    assert not handle.record.metadata_path.exists()
    assert (handle.run_dir / results_store_module.UNSAVED_MARKER).is_file()
    # A second session opening the same Project must not find it either.
    assert ResultsStore.open_or_create(tmp_path).list_runs() == []


def test_saving_records_the_run_and_clears_the_marker(tmp_path: Path) -> None:
    store = ResultsStore.open_or_create(tmp_path)
    handle = store.begin_run("ert", "ert.single_inversion", "ert.single_inversion")
    (handle.outputs_dir / "model.npy").write_bytes(b"model")
    store.finish_run(handle, {"status": "ok", "metrics": {"chi2": 1.1}})

    saved = store.save_run(handle.run_id)
    assert saved.status == "success"
    assert not store.has_unsaved()
    assert handle.record.metadata_path.is_file()
    assert handle.result_path.is_file()
    assert not (handle.run_dir / results_store_module.UNSAVED_MARKER).exists()
    # The run's own outputs never moved, so a path captured earlier still works.
    assert (handle.outputs_dir / "model.npy").is_file()

    reopened = ResultsStore.open_or_create(tmp_path)
    assert [record.run_id for record in reopened.list_runs()] == [handle.run_id]
    assert reopened.get_run(handle.run_id).metrics["chi2"] == 1.1
    # Saving twice is a no-op rather than an error.
    assert store.save_run(handle.run_id).run_id == handle.run_id


def test_discarding_removes_the_folder(tmp_path: Path) -> None:
    store = ResultsStore.open_or_create(tmp_path)
    handle = store.begin_run("em", "em.inversion")
    store.finish_run(handle, {"status": "ok"})
    store.discard_run(handle.run_id)
    assert not handle.run_dir.exists()
    assert not store.has_unsaved()
    assert store.get_run(handle.run_id) is None


def test_a_labelled_run_is_still_not_saved_by_labelling_it(tmp_path: Path) -> None:
    """Naming a run is not a decision to keep it."""
    store = ResultsStore.open_or_create(tmp_path)
    handle = store.begin_run("em", "em.inversion")
    store.finish_run(handle, {"status": "ok"})
    store.update_run(handle.run_id, label="East line", notes="dry season")
    assert not handle.record.metadata_path.exists()
    assert store.has_unsaved()
    # …and the label survives into the record once it is saved.
    assert store.save_run(handle.run_id).label == "East line"
    assert ResultsStore.open_or_create(tmp_path).get_run(
        handle.run_id).notes == "dry season"


def test_an_earlier_sessions_unsaved_runs_are_reported_as_abandoned(tmp_path: Path) -> None:
    """A crash leaves a marked folder with no record; nothing else would list it."""
    store = ResultsStore.open_or_create(tmp_path)
    handle = store.begin_run("em", "em.inversion", "em.inversion")
    (handle.outputs_dir / "values.bin").write_bytes(b"12345")
    store.finish_run(handle, {"status": "ok"})
    assert store.abandoned_run_dirs() == []      # this session still owns it

    reopened = ResultsStore.open_or_create(tmp_path)
    assert reopened.list_runs() == []
    assert reopened.abandoned_run_dirs() == [handle.run_dir]
    assert reopened.clear_abandoned_runs() == 1
    assert not handle.run_dir.exists()


def test_a_saved_run_is_never_reported_as_abandoned(tmp_path: Path) -> None:
    store = ResultsStore.open_or_create(tmp_path)
    handle = store.begin_run("em", "em.inversion")
    store.finish_run(handle, {"status": "ok"})
    store.save_run(handle.run_id)
    assert ResultsStore.open_or_create(tmp_path).abandoned_run_dirs() == []


def test_artifact_path_must_stay_inside_run(tmp_path: Path) -> None:
    store = ResultsStore.open_or_create(tmp_path)
    handle = store.begin_run("ert", "ert.single_inversion")
    assert store.locate_run_artifact(handle.record, "outputs/model.npy") == (
        handle.run_dir / "outputs" / "model.npy"
    ).resolve()
    with pytest.raises(ValueError, match="escapes"):
        store.locate_run_artifact(handle.record, "../outside.npy")


def test_label_notes_size_and_delete(tmp_path: Path) -> None:
    store = ResultsStore.open_or_create(tmp_path)
    handle = store.begin_run("em", "em.inversion")
    (handle.outputs_dir / "values.bin").write_bytes(b"12345")
    store.finish_run(handle, {"status": "success"})
    record = store.update_run(handle.run_id, label="East line", notes="dry season")
    assert record.label == "East line"
    assert record.notes == "dry season"
    assert store.run_size(record) >= 5
    store.delete_run(handle.run_id)
    assert store.get_run(handle.run_id) is None


def test_read_only_store_does_not_recover_or_mutate(tmp_path: Path) -> None:
    writable = ResultsStore.open_or_create(tmp_path)
    handle = writable.begin_run("ert", "ert.single_inversion")
    writable.finish_run(handle, {"status": "ok"})
    writable.save_run(handle.run_id)
    readonly = ResultsStore.open_or_create(tmp_path, read_only=True)
    assert readonly.get_run(handle.run_id).status == "success"
    for call in (lambda: readonly.begin_run("ert", "ert.single_inversion"),
                 lambda: readonly.save_run(handle.run_id),
                 lambda: readonly.discard_run(handle.run_id),
                 lambda: readonly.clear_abandoned_runs()):
        with pytest.raises(PermissionError):
            call()


def test_atomic_replace_retries_transient_windows_lock(monkeypatch, tmp_path: Path) -> None:
    real_replace = results_store_module.os.replace
    attempts = []

    def flaky_replace(source, target):
        attempts.append(Path(target))
        if len(attempts) < 5:
            raise PermissionError("OneDrive has the file open")
        return real_replace(source, target)

    monkeypatch.setattr(results_store_module.os, "replace", flaky_replace)
    monkeypatch.setattr(results_store_module.time, "sleep", lambda _delay: None)
    target = tmp_path / "metadata.json"
    results_store_module._atomic_write_json(target, {"status": "success"})
    assert len(attempts) == 5
    assert json.loads(target.read_text(encoding="utf-8"))["status"] == "success"


def test_save_metadata_failure_leaves_a_recovery_record(monkeypatch, tmp_path: Path) -> None:
    """A locked run.json must not cost the user the run they asked to keep."""
    store = ResultsStore.open_or_create(tmp_path)
    handle = store.begin_run("ert", "ert.single_inversion")
    store.finish_run(handle, {"status": "ok", "metrics": {"chi2": float("nan")}})
    original = results_store_module._atomic_write_json

    def locked_run_json(path, payload, **kwargs):
        if Path(path).name == "run.json":
            raise PermissionError("run.json is synchronized")
        return original(path, payload, **kwargs)

    monkeypatch.setattr(results_store_module, "_atomic_write_json", locked_run_json)
    with pytest.raises(PermissionError):
        store.save_run(handle.run_id)
    assert list(handle.run_dir.glob("run.recovery.*.json"))
    # Still staged, so the user can retry rather than losing the run.
    assert store.has_unsaved()

    # Even if the session ends here, the run the user asked to keep comes back:
    # the next scan finds the recovery record and repairs run.json from it.
    monkeypatch.setattr(results_store_module, "_atomic_write_json", original)
    crashed = ResultsStore.open_or_create(tmp_path)
    assert crashed.abandoned_run_dirs() == [], (
        "a partly-written save is recovered, not treated as abandoned"
    )
    recovered = crashed.get_run(handle.run_id)
    assert recovered is not None
    assert recovered.status == "success"
    assert recovered.metrics["chi2"] is None
    assert handle.record.metadata_path.is_file()


def test_running_and_imported_runs_cannot_be_deleted(tmp_path: Path) -> None:
    store = ResultsStore.open_or_create(tmp_path)
    running = store.begin_run("ert", "ert.single_inversion")
    with pytest.raises(RuntimeError, match="running"):
        store.delete_run(running.run_id)

    legacy = tmp_path / "old"; legacy.mkdir()
    (legacy / "ert_recipe.json").write_text(
        json.dumps({"workflow_id": "ert.single_inversion"}), encoding="utf-8"
    )
    (legacy / "ert_process_result.json").write_text(
        json.dumps({"status": "ok"}), encoding="utf-8"
    )
    imported = store.import_legacy(tmp_path)
    assert len(imported) == 1
    with pytest.raises(PermissionError, match="Imported"):
        store.delete_run(imported[0].run_id)


def test_legacy_import_is_idempotent_and_keeps_unknown_attachments(tmp_path: Path) -> None:
    old = tmp_path / "legacy"; old.mkdir()
    (old / "em_recipe.json").write_text(
        json.dumps({"workflow_id": "em.inversion"}), encoding="utf-8"
    )
    (old / "em_process_result.json").write_text(
        json.dumps({
            "status": "completed",
            "artifacts": [{"path": "model.npy", "kind": "em_model", "format": "npy"}],
        }),
        encoding="utf-8",
    )
    (old / "model.npy").write_bytes(b"array")
    (old / "notes.csv").write_text("x,y\n1,2\n", encoding="utf-8")
    store = ResultsStore.open_or_create(tmp_path)

    imported = store.import_legacy(tmp_path)
    assert len(imported) == 1
    assert imported[0].status == "success"
    assert imported[0].managed is False and imported[0].imported is True
    assert any(item["path"] == "notes.csv" for item in imported[0].artifacts)
    assert store.import_legacy(tmp_path) == []


def test_legacy_preview_rejects_symlink_outside_selected_root(tmp_path: Path) -> None:
    selected = tmp_path / "selected"; selected.mkdir()
    outside = tmp_path / "outside"; outside.mkdir()
    (outside / "ert_recipe.json").write_text("{}", encoding="utf-8")
    (outside / "ert_process_result.json").write_text("{}", encoding="utf-8")
    link = selected / "linked"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Directory symlinks are not available for this account")
    assert ResultsStore.preview_legacy(selected) == []


def test_ert_two_runs_keep_independent_reloadable_model_bundles(tmp_path: Path) -> None:
    pg = pytest.importorskip("pygimli")
    mt = pytest.importorskip("pygimli.meshtools")
    store = ResultsStore.open_or_create(tmp_path)

    for index, value in enumerate((100.0, 250.0), start=1):
        handle = store.begin_run("ert", "ert.single_inversion", "ert.single_inversion")
        recipe = handle.run_dir / "ert_recipe.json"
        recipe.write_text(
            json.dumps({"workflow_id": "ert.single_inversion", "run": index}),
            encoding="utf-8",
        )
        handle.record.recipe_path = recipe.name
        mesh = mt.createMesh(
            mt.createRectangle(start=[0.0, 0.0], end=[2.0, -1.0]),
            quality=30,
            area=0.25,
        )
        mesh_path = handle.outputs_dir / "mesh.bms"
        mesh.save(str(mesh_path))
        model_path = handle.outputs_dir / "model.npy"
        np.save(model_path, np.full(mesh.cellCount(), value))
        store.finish_run(handle, WorkflowRunResult(
            status="ok",
            summary={
                "model_bundle": {
                    "mesh": "outputs/mesh.bms",
                    "model": "outputs/model.npy",
                }
            },
            artifacts=[
                ArtifactRef.from_path(
                    mesh_path, artifact_id="ert:mesh", kind="mesh", base_dir=handle.run_dir
                ),
                ArtifactRef.from_path(
                    model_path, artifact_id="ert:model", kind="resistivity_model",
                    base_dir=handle.run_dir,
                ),
            ],
        ))

    store.save_all_unsaved()
    records = ResultsStore.open_or_create(tmp_path).list_runs()
    assert len(records) == 2
    assert len({record.run_dir for record in records}) == 2
    recovered_values = set()
    for record in records:
        assert (record.run_dir / record.recipe_path).is_file()
        bundle = record.summary["model_bundle"]
        mesh = pg.load(str(store.locate_run_artifact(record, bundle["mesh"])))
        model = np.load(store.locate_run_artifact(record, bundle["model"]), allow_pickle=False)
        assert mesh.cellCount() == model.size
        recovered_values.add(float(model[0]))
    assert recovered_values == {100.0, 250.0}


def test_workbench_cannot_switch_project_during_active_run(tmp_path: Path) -> None:
    state = WorkbenchState(output_dir=tmp_path / "first")
    state.set_results_store(tmp_path / "first")
    state.begin_run("ert", "ert.single_inversion", "ert.single_inversion")
    with pytest.raises(RuntimeError, match="computation is running"):
        state.set_results_store(tmp_path / "second")
    state.cancel_run("ert", "test complete")
    assert state.set_results_store(tmp_path / "second").root == (tmp_path / "second").resolve()


def test_operations_in_same_module_can_run_concurrently(tmp_path: Path) -> None:
    state = WorkbenchState(output_dir=tmp_path)
    state.set_results_store(tmp_path)
    single = state.begin_run("ert", "ert.single_inversion", "ert.single_inversion")
    timelapse = state.begin_run(
        "ert", "ert.timelapse_inversion", "ert.timelapse_inversion"
    )
    assert state.active_run("ert", "ert.single_inversion") is single
    assert state.active_run("ert", "ert.timelapse_inversion") is timelapse
    assert state.active_run("ert") is None
    state.finish_run("ert", {"status": "ok"}, "ert.single_inversion")
    state.cancel_run("ert", "test complete", "ert.timelapse_inversion")
    assert state.results_store.list_runs() == []          # nothing saved yet
    assert {item.status for item in state.unsaved_runs()} == {"success", "cancelled"}
    state.save_all_runs()
    assert {item.status for item in state.results_store.list_runs()} == {
        "success", "cancelled"
    }


def test_list_and_get_use_memory_index_after_open(monkeypatch, tmp_path: Path) -> None:
    store = ResultsStore.open_or_create(tmp_path)
    handle = store.begin_run("ert", "ert.single_inversion")
    store.finish_run(handle, {"status": "ok"})
    store.save_run(handle.run_id)
    monkeypatch.setattr(
        store, "_scan_records", lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("routine lookup should not rescan the tree")
        )
    )
    assert store.get_run(handle.run_id) is handle.record
    assert store.list_runs() == [handle.record]


def test_external_legacy_import_does_not_change_project_and_reopens(tmp_path: Path) -> None:
    project = tmp_path / "project"
    legacy = tmp_path / "legacy"; legacy.mkdir()
    (legacy / "ert_recipe.json").write_text(
        json.dumps({"workflow_id": "ert.single_inversion"}), encoding="utf-8"
    )
    (legacy / "ert_process_result.json").write_text(
        json.dumps({"status": "ok"}), encoding="utf-8"
    )
    store = ResultsStore.open_or_create(project)
    imported = store.import_legacy(legacy)
    assert store.root == project.resolve()
    assert imported[0].run_dir == legacy.resolve()
    reopened = ResultsStore.open_or_create(project)
    assert reopened.get_run(imported[0].run_id).run_dir == legacy.resolve()


def test_auto_discovery_is_bounded_and_large_arrays_are_described(tmp_path: Path) -> None:
    store = ResultsStore.open_or_create(tmp_path)
    handle = store.begin_run("ert", "ert.timelapse_inversion")
    for index in range(225):
        (handle.outputs_dir / f"step_{index:04d}.csv").write_text(
            "x,y\n0,1\n", encoding="utf-8"
        )
    store.finish_run(handle, {
        "status": "ok",
        "summary": {"large": np.arange(5000, dtype=float)},
    })
    record = store.get_run(handle.run_id)
    assert len(record.artifacts) == 200
    assert any("additional output files" in warning for warning in record.warnings)
    store.save_run(handle.run_id)
    result = json.loads(handle.result_path.read_text(encoding="utf-8"))
    assert result["summary"]["large"]["$array"]["shape"] == [5000]
