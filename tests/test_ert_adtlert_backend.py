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

from PyHydroGeophysX._internal.optional_dependencies import (  # noqa: E402
    BackendUnavailable,
)
from PyHydroGeophysX.inversion.ert_inversion import (  # noqa: E402
    _ADTLertEngine,
    run_ert_manager_inversion,
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
    assert manager.paraDomain.cellCount() == model.size == coverage.size
    assert response.shape == (data.size(),)
    assert np.all(np.isfinite(model))
    assert np.all(np.isfinite(response))
    assert np.isfinite(result["chi2"])


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
