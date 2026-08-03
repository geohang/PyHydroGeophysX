"""Integration coverage for the optional ADTLERT ERT backend."""

from __future__ import annotations

import builtins
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pygimli")
pytest.importorskip("adtlert")

import pygimli.meshtools as mt  # noqa: E402
from pygimli.physics import ert  # noqa: E402
import PyHydroGeophysX.inversion.ert_inversion as ert_inversion  # noqa: E402
from PyHydroGeophysX.data_processing.ert_io import (  # noqa: E402
    normalize_for_timelapse,
)

from PyHydroGeophysX._internal.optional_dependencies import (  # noqa: E402
    BackendUnavailable,
)
from PyHydroGeophysX.inversion.ert_inversion import (  # noqa: E402
    _ADTLertEngine,
    _adtlert_cuda_available,
    _adtlert_forward_solver_backend,
    _adtlert_solver_name,
    run_ert_manager_inversion,
)
from PyHydroGeophysX.inversion.windowed import (  # noqa: E402
    WindowedTimeLapseERTInversion,
    _ADTLERTWindowProgress,
)
from PyHydroGeophysX.inversion.time_lapse import (  # noqa: E402
    run_timelapse_ert,
)


def test_adtlert_window_events_are_forwarded_as_readable_progress() -> None:
    messages = []
    progress = _ADTLERTWindowProgress(messages.append)

    progress({
        "event": "windowed_start",
        "n_windows": 8,
        "n_times": 10,
        "window_size": 3,
    })
    progress({
        "event": "window_start",
        "window_index": 1,
        "n_windows": 8,
        "start_idx": 0,
        "end_idx": 2,
    })
    progress({
        "event": "timelapse_iteration_done",
        "iteration": 2,
        "max_iterations": 5,
        "chi2": 1.2345,
    })
    progress({
        "event": "window_done",
        "window_index": 1,
        "n_windows": 8,
        "final_chi2": 1.2,
    })
    progress({"event": "windowed_prediction_start", "n_times": 10})
    progress({"event": "windowed_done", "n_windows": 8, "final_chi2": 0.9})

    assert messages[0].startswith("[progress 0/8]")
    assert "window 1/8 started (time steps 1-3)" in messages[1]
    assert "window 1/8, iteration 2/5: chi2 1.234" in messages[2]
    assert messages[3].startswith("[progress 1/8]")
    assert "assembling predictions for 10 time steps" in messages[4]
    assert "complete: 8/8 windows, final chi2 0.900" in messages[5]
    assert progress.iteration_chi2 == [1.2345]


requires_adtlert_cuda = pytest.mark.skipif(
    not _adtlert_cuda_available(),
    reason="ADTLERT inversion requires Torch and CuPy CUDA 12",
)


@pytest.fixture(scope="module")
def adtlert_case(tmp_path_factory):
    scheme = ert.createData(elecs=np.linspace(0.0, 23.0, 12), schemeName="dd")
    world = mt.createWorld(
        start=[-10.0, 0.0], end=[33.0, -15.0], worldMarker=True
    )
    block = mt.createRectangle(start=[9.0, -3.0], end=[15.0, -7.0], marker=2)
    simulation_mesh = mt.createMesh(
        mt.mergePLC([world, block]), quality=32, area=1.5
    )
    data = ert.simulate(
        simulation_mesh,
        res=[[1, 150.0], [2, 35.0]],
        scheme=scheme,
        noiseLevel=0.5,
        noiseAbs=1.0e-5,
        seed=7,
        verbose=False,
    )
    data.remove(data["rhoa"] <= 0.0)
    path = tmp_path_factory.mktemp("adtlert") / "line.dat"
    data.save(str(path))
    return path, data


@pytest.fixture(scope="module")
def adtlert_timelapse_case(tmp_path_factory):
    root = tmp_path_factory.mktemp("adtlert_timelapse")
    scheme = ert.createData(elecs=np.linspace(0.0, 23.0, 12), schemeName="dd")
    world = mt.createWorld(
        start=[-10.0, 0.0], end=[33.0, -15.0], worldMarker=True
    )
    block = mt.createRectangle(start=[9.0, -3.0], end=[15.0, -7.0], marker=2)
    simulation_mesh = mt.createMesh(
        mt.mergePLC([world, block]), quality=32, area=1.5
    )
    files = []
    datasets = []
    for index, block_resistivity in enumerate((45.0, 55.0, 70.0, 85.0)):
        data = ert.simulate(
            simulation_mesh,
            res=[[1, 150.0], [2, block_resistivity]],
            scheme=scheme,
            noiseLevel=0.0,
            verbose=False,
        )
        data["err"] = np.full(data.size(), 0.03)
        path = root / f"line_{index}.dat"
        data.save(str(path))
        files.append(path.name)
        datasets.append(data)
    inversion_mesh = mt.createMesh(
        mt.createParaMeshPLC(
            datasets[0].sensorPositions(),
            paraDepth=10.0,
            paraDX=0.6,
            boundary=1.0,
        ),
        quality=32,
    )
    return root, files, datasets, inversion_mesh


@requires_adtlert_cuda
def test_adtlert_backend_runs_through_the_public_ert_pipeline(
    adtlert_case, tmp_path: Path
) -> None:
    path, data = adtlert_case
    result = run_ert_manager_inversion(
        path,
        tmp_path,
        engine="adtlert",
        lam=10.0,
        max_iterations=1,
        max_total_iterations=1,
        auto_lambda=False,
        geometric_factor_policy="off",
    )

    manager = result["mgr"]
    model = np.asarray(manager.model, dtype=float)
    response = np.asarray(manager.response, dtype=float)
    coverage = np.asarray(manager.coverage(), dtype=float)
    assert result["engine"] == "adtlert"
    assert result["metrics"]["backend"] == "adtlert"
    assert result["metrics"]["sensitivity_profile"] == "paper"
    assert result["metrics"]["normal_sensitivity"] is True
    assert result["metrics"]["include_robin_boundary_derivative"] is False
    assert result["metrics"]["linearized_solver"] == "gpu_cgls"
    assert manager.paraDomain.cellCount() == model.size == coverage.size
    assert response.shape == (data.size(),)
    assert np.all(np.isfinite(model))
    assert np.all(np.isfinite(response))
    assert np.isfinite(result["chi2"])


@requires_adtlert_cuda
def test_adtlert_backend_preserves_an_imported_parameter_domain(
    adtlert_case, tmp_path: Path
) -> None:
    path, data = adtlert_case
    mesh = mt.createMesh(
        mt.createParaMeshPLC(
            data.sensorPositions(), paraDepth=10.0, paraDX=0.6, boundary=1.0
        ),
        quality=32,
    )
    mesh_path = tmp_path / "imported.bms"
    mesh.save(str(mesh_path))

    result = run_ert_manager_inversion(
        path,
        tmp_path / "result",
        engine="adtlert",
        mesh_file=str(mesh_path),
        lam=10.0,
        max_iterations=1,
        max_total_iterations=1,
        auto_lambda=False,
        geometric_factor_policy="off",
    )

    expected = sum(1 for cell in mesh.cells() if cell.marker() > 1)
    manager = result["mgr"]
    assert manager.paraDomain.cellCount() == expected
    assert np.asarray(manager.model).shape == (expected,)
    assert np.asarray(manager.coverage()).shape == (expected,)


@pytest.mark.parametrize("inversion_type", ["L2", "L1", "L1L2"])
@requires_adtlert_cuda
def test_adtlert_windowed_timelapse_reuses_one_forward_context(
    adtlert_timelapse_case, inversion_type: str
) -> None:
    root, files, datasets, mesh = adtlert_timelapse_case
    inversion = WindowedTimeLapseERTInversion(
        data_dir=str(root),
        ert_files=files,
        measurement_times=[0.0, 1.0, 2.0, 3.0],
        window_size=3,
        mesh=mesh,
        engine="adtlert",
        lambda_val=10.0,
        alpha=2.0,
        max_iterations=1,
        inversion_type=inversion_type,
    )

    result = inversion.run()

    expected_cells = sum(1 for cell in mesh.cells() if cell.marker() > 1)
    assert result.meta["backend"] == "adtlert"
    assert result.meta["sensitivity_profile"] == "paper"
    assert result.meta["normal_sensitivity"] is True
    assert result.meta["include_robin_boundary_derivative"] is False
    assert result.meta["linearized_solver"] == _adtlert_solver_name(
        "cgls", prefer_gpu=True
    )
    assert result.final_models.shape == (expected_cells, 4)
    assert result.predicted_data.shape == (4, datasets[0].size())
    assert result.coverage.shape == (expected_cells,)
    assert len(result.all_coverage) == 4
    assert len(result.meta["window_reports"]) == 2
    assert np.all(np.isfinite(result.final_models))
    assert np.all(np.isfinite(result.predicted_data))


def test_timelapse_normalization_applies_one_common_quality_mask(
    adtlert_timelapse_case, tmp_path: Path
) -> None:
    root, files, datasets, _ = adtlert_timelapse_case
    source_paths = []
    for index, name in enumerate(files[:3]):
        data = ert.load(str(root / name))
        errors = np.full(data.size(), 0.03, dtype=float)
        if index == 1:
            errors[0] = 0.5
        data["err"] = errors
        path = tmp_path / f"source_{index}.dat"
        data.save(str(path))
        source_paths.append(str(path))

    clean_dir, names, normalized = normalize_for_timelapse(
        source_paths, None, str(tmp_path / "normalized"), max_error=0.1
    )

    assert Path(clean_dir).is_dir()
    assert len(names) == len(normalized) == 3
    assert {int(data.size()) for data in normalized} == {
        int(datasets[0].size()) - 1
    }
    layouts = [
        np.column_stack(
            [np.asarray(data[key]) for key in ("a", "b", "m", "n")]
        )
        for data in normalized
    ]
    assert all(np.array_equal(layouts[0], layout) for layout in layouts[1:])


@requires_adtlert_cuda
def test_adtlert_windowed_rejects_multiprocess_gpu_contexts(
    adtlert_timelapse_case,
) -> None:
    root, files, _, mesh = adtlert_timelapse_case
    inversion = WindowedTimeLapseERTInversion(
        data_dir=str(root),
        ert_files=files,
        measurement_times=[0.0, 1.0, 2.0, 3.0],
        window_size=3,
        mesh=mesh,
        engine="adtlert",
    )
    with pytest.raises(ValueError, match="shared GPU context"):
        inversion.run(window_parallel=True)


@requires_adtlert_cuda
def test_adtlert_windowed_requires_common_abmn_order(
    adtlert_timelapse_case, tmp_path: Path
) -> None:
    root, files, _, mesh = adtlert_timelapse_case
    changed = ert.load(str(root / files[1]))
    changed.remove(np.arange(changed.size()) == 0)
    changed_path = tmp_path / "changed.dat"
    changed.save(str(changed_path))
    inversion = WindowedTimeLapseERTInversion(
        data_dir=str(root),
        ert_files=[files[0], str(changed_path)],
        measurement_times=[0.0, 1.0],
        window_size=2,
        mesh=mesh,
        engine="adtlert",
    )
    with pytest.raises(ValueError, match="identical ABMN ordering"):
        inversion.run()


@requires_adtlert_cuda
def test_adtlert_windowed_runs_through_the_public_timelapse_workflow(
    adtlert_timelapse_case, tmp_path: Path
) -> None:
    root, files, datasets, _ = adtlert_timelapse_case
    messages = []
    result = run_timelapse_ert(
        [str(root / name) for name in files[:3]],
        [0.0, 1.0, 2.0],
        {
            "engine": "adtlert",
            "windowed": True,
            "window_size": 3,
            "lambda_val": 10.0,
            "alpha": 2.0,
            "max_iterations": 1,
            "mesh_quality": 32.0,
        },
        str(tmp_path),
        log=messages.append,
    )

    assert result["status"] == "ok"
    assert result["mode"] == "windowed"
    assert result["engine"] == "adtlert"
    assert result["linearized_solver"] == _adtlert_solver_name(
        "cgls", prefer_gpu=True
    )
    assert result["n_times"] == 3
    assert result["final_models"].shape[1] == 3
    assert result["coverage"].shape == result["final_models"].T.shape
    assert result["n_data"] == 3 * datasets[0].size()
    bundle = result["model_bundle"]
    restored_mesh = ert_inversion.pg.load(bundle["mesh"])
    restored_models = np.load(bundle["models"], allow_pickle=False)
    restored_coverage = np.load(bundle["coverage"], allow_pickle=False)
    assert restored_mesh.cellCount() == result["mesh"].cellCount()
    np.testing.assert_allclose(restored_models, result["final_models"])
    np.testing.assert_allclose(restored_coverage, result["coverage"])
    assert any(message.startswith("[progress 0/1]") for message in messages)
    assert any(message.startswith("[progress 1/1]") for message in messages)
    assert any("window 1/1 started" in message for message in messages)


def test_missing_adtlert_reports_the_install_extra(
    adtlert_case, monkeypatch
) -> None:
    _, data = adtlert_case
    mesh = mt.createMesh(
        mt.createParaMeshPLC(
            data.sensorPositions(), paraDepth=12.0, boundary=1.0
        ),
        quality=32,
    )
    original_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "adtlert" or name.startswith("adtlert."):
            raise ModuleNotFoundError(
                "No module named 'adtlert'", name="adtlert"
            )
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    pattern = r"pyhydrogeophysx\[adtlert\]"
    with pytest.raises(BackendUnavailable, match=pattern):
        _ADTLertEngine(data, mesh)


def test_windowed_cgls_falls_back_without_cupy(monkeypatch) -> None:
    original_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "cupy" or name.startswith("cupy."):
            raise ModuleNotFoundError("No module named 'cupy'", name="cupy")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    assert (
        _adtlert_solver_name("cgls", prefer_gpu=True) == "pyhydro_cgls"
    )


def test_adtlert_forward_solver_prefers_cudss(monkeypatch) -> None:
    monkeypatch.setattr(
        ert_inversion, "_adtlert_cudss_available", lambda: True
    )
    assert _adtlert_forward_solver_backend() == "cudss"


def test_adtlert_forward_solver_rejects_missing_cudss(monkeypatch) -> None:
    monkeypatch.setattr(
        ert_inversion, "_adtlert_cudss_available", lambda: False
    )
    with pytest.raises(BackendUnavailable, match="cuDSS"):
        _adtlert_forward_solver_backend()
