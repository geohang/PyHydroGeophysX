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

    store.index_path.write_text("not json", encoding="utf-8")
    reopened = ResultsStore.open_or_create(tmp_path)
    records = reopened.list_runs()
    assert {record.status for record in records} == {"success", "cancelled"}
    assert len(records) == 2
    assert json.loads(reopened.index_path.read_text(encoding="utf-8"))["runs"]


def test_running_run_is_recovered_as_interrupted(tmp_path: Path) -> None:
    store = ResultsStore.open_or_create(tmp_path)
    handle = store.begin_run("em", "em.inversion", "em.inversion")
    reopened = ResultsStore.open_or_create(tmp_path)
    record = reopened.get_run(handle.run_id)
    assert record is not None
    assert record.status == "interrupted"


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
    readonly = ResultsStore.open_or_create(tmp_path, read_only=True)
    assert readonly.get_run(handle.run_id).status == "running"
    with pytest.raises(PermissionError):
        readonly.begin_run("ert", "ert.single_inversion")


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


def test_finish_metadata_failure_recovers_as_success(monkeypatch, tmp_path: Path) -> None:
    store = ResultsStore.open_or_create(tmp_path)
    handle = store.begin_run("ert", "ert.single_inversion")
    original = results_store_module._atomic_write_json

    def locked_run_json(path, payload, **kwargs):
        if Path(path).name == "run.json":
            raise PermissionError("run.json is synchronized")
        return original(path, payload, **kwargs)

    monkeypatch.setattr(results_store_module, "_atomic_write_json", locked_run_json)
    record = store.finish_run(handle, {"status": "ok", "metrics": {"chi2": float("nan")}})
    assert record.status == "success"
    assert list(handle.run_dir.glob("run.recovery.*.json"))

    monkeypatch.setattr(results_store_module, "_atomic_write_json", original)
    reopened = ResultsStore.open_or_create(tmp_path)
    recovered = reopened.get_run(handle.run_id)
    assert recovered is not None
    assert recovered.status == "success"
    assert recovered.metrics["chi2"] is None


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
    assert {item.status for item in state.results_store.list_runs()} == {
        "success", "cancelled"
    }


def test_list_and_get_use_memory_index_after_open(monkeypatch, tmp_path: Path) -> None:
    store = ResultsStore.open_or_create(tmp_path)
    handle = store.begin_run("ert", "ert.single_inversion")
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
    result = json.loads(handle.result_path.read_text(encoding="utf-8"))
    assert result["summary"]["large"]["$array"]["shape"] == [5000]
