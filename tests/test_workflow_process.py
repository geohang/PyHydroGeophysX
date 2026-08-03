import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PySide6.QtCore import QProcess

from PyHydroGeophysX.inversion.ert_inversion import _export_model_bundle
from PyHydroGeophysX.qt_apps.workers import ProcessWorkflowWorker
from PyHydroGeophysX.workflows import (
    ArtifactRef,
    RunContext,
    WorkflowRunResult,
    WorkflowSpec,
)
from PyHydroGeophysX.workflows import cli
from PyHydroGeophysX.workflows import domain


def test_cli_writes_process_safe_result_file(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    recipe = tmp_path / "recipe.json"
    recipe.write_text("{}", encoding="utf-8")
    result_path = tmp_path / "result.json"
    expected = WorkflowRunResult(
        status="success",
        summary={"engine": "pygimli"},
        metrics={"chi2": 1.25},
    )

    monkeypatch.setattr(cli, "load_recipe", lambda _path: object())
    def fake_run(_spec, context):
        context.progress("Building inversion mesh")
        context.progress("Lambda trial 1")
        return expected

    monkeypatch.setattr(cli, "run_workflow", fake_run)

    exit_code = cli.main([
        "run",
        str(recipe),
        "--output-dir",
        str(tmp_path / "output"),
        "--result-file",
        str(result_path),
    ])

    assert exit_code == 0
    assert json.loads(result_path.read_text(encoding="utf-8")) == expected.to_dict()
    assert not result_path.with_name(result_path.name + ".tmp").exists()
    output = capsys.readouterr().out
    assert "Building inversion mesh" in output
    assert "Lambda trial 1" in output


def test_process_worker_restores_result_from_json(tmp_path: Path) -> None:
    result_path = tmp_path / "result.json"
    expected = WorkflowRunResult(
        status="success",
        summary={"engine": "pygimli"},
        metrics={"chi2": 0.9},
    )
    result_path.write_text(json.dumps(expected.to_dict()), encoding="utf-8")
    worker = ProcessWorkflowWorker(
        tmp_path / "recipe.json",
        tmp_path,
        tmp_path / "output",
        result_path,
    )
    environment = worker.process.processEnvironment()
    assert environment.value("PYTHONUTF8") == "1"
    assert environment.value("PYTHONIOENCODING") == "utf-8"
    received = []
    worker.succeeded.connect(received.append)

    prefix_python = Path(sys.prefix) / "Scripts" / "python.exe"
    if prefix_python.is_file():
        assert Path(worker.process.program()) == prefix_python

    worker._on_finished(0, QProcess.ExitStatus.NormalExit)

    assert len(received) == 1
    assert received[0].to_dict() == expected.to_dict()


def test_process_worker_forwards_structured_progress(tmp_path: Path) -> None:
    worker = ProcessWorkflowWorker(
        tmp_path / "recipe.json",
        tmp_path,
        tmp_path / "output",
        tmp_path / "result.json",
    )
    progressed = []
    logged = []
    worker.progressed.connect(lambda *values: progressed.append(values))
    worker.logged.connect(logged.append)

    worker._emit_output(
        b"[progress 2/8] ADTLERT window 2/8 complete, chi2 1.100\n"
    )

    assert progressed == [(2, 8, "ADTLERT window 2/8 complete, chi2 1.100")]
    assert logged == ["ADTLERT window 2/8 complete, chi2 1.100"]


def test_isolated_ert_cli_does_not_load_unused_pyarrow(
    monkeypatch, tmp_path: Path
) -> None:
    recipe = tmp_path / "recipe.json"
    recipe.write_text("{}", encoding="utf-8")
    result_path = tmp_path / "result.json"
    seen = []
    monkeypatch.delitem(sys.modules, "pyarrow", raising=False)
    monkeypatch.setattr(
        cli,
        "load_recipe",
        lambda _path: SimpleNamespace(workflow_id="ert.timelapse_inversion"),
    )

    def fake_run(_spec, _context):
        seen.append(sys.modules.get("pyarrow", "missing"))
        return WorkflowRunResult(status="success")

    monkeypatch.setattr(cli, "run_workflow", fake_run)

    assert cli.main([
        "run",
        str(recipe),
        "--result-file",
        str(result_path),
    ]) == 0
    assert seen == [None]
    assert "pyarrow" not in sys.modules


def test_export_model_bundle_persists_process_safe_arrays(tmp_path: Path) -> None:
    class Mesh:
        def save(self, path: str) -> None:
            Path(path).write_text("mesh", encoding="utf-8")

    class Manager:
        paraDomain = Mesh()
        model = np.array([10.0, 20.0])
        response = np.array([12.0, 18.0, 21.0])

        @staticmethod
        def coverage():
            return np.array([-2.0, 0.5])

    bundle = _export_model_bundle(Manager(), tmp_path, "resistivity")

    assert Path(bundle["mesh"]).is_file()
    np.testing.assert_allclose(np.load(bundle["model"]), Manager.model)
    np.testing.assert_allclose(np.load(bundle["response"]), Manager.response)
    np.testing.assert_allclose(np.load(bundle["coverage"]), Manager.coverage())


def test_model_bundle_round_trips_a_pygimli_mesh(tmp_path: Path) -> None:
    import pygimli as pygimli
    import pygimli.meshtools as mt

    from PyHydroGeophysX.inversion.ert_inversion import ModelResult

    mesh = mt.createMesh(
        mt.createRectangle(start=[0.0, 0.0], end=[2.0, -1.0]),
        quality=30,
        area=0.2,
    )
    model = np.linspace(10.0, 20.0, mesh.cellCount())
    coverage = np.linspace(-2.0, 1.0, mesh.cellCount())
    bundle = _export_model_bundle(
        ModelResult(mesh, model, coverage=coverage),
        tmp_path,
        "roundtrip",
    )

    loaded_mesh = pygimli.load(bundle["mesh"])
    assert loaded_mesh.cellCount() == mesh.cellCount()
    np.testing.assert_allclose(np.load(bundle["model"]), model)
    np.testing.assert_allclose(np.load(bundle["coverage"]), coverage)


def test_normalized_ert_workflow_skips_the_instrument_parser(
    monkeypatch, tmp_path: Path
) -> None:
    from PyHydroGeophysX.inversion import ert_inversion

    data_path = tmp_path / "filtered.dat"
    data_path.write_text("normalized", encoding="utf-8")
    source = ArtifactRef.from_path(
        data_path,
        artifact_id="ert:filtered",
        kind="ert_data",
        base_dir=tmp_path,
        metadata={"qc_filtered": True},
    )
    captured = {}

    def fake_run(_data_path, _output_dir, **kwargs):
        captured.update(kwargs)
        return {
            "status": "success",
            "engine": "pygimli",
            "metrics": {"chi2": 1.0},
            "convergence": [2.0, 1.0],
        }

    monkeypatch.setattr(ert_inversion, "run_ert_manager_inversion", fake_run)
    spec = WorkflowSpec(
        workflow_id="ert.single_inversion",
        inputs={"data": source},
        parameters={"instrument": "BERT", "engine": "pygimli"},
    )

    result = domain.run_ert_single(
        spec,
        RunContext(project_root=tmp_path, output_dir=tmp_path / "output"),
    )

    assert result.status == "success"
    assert captured["instrument"] is None
