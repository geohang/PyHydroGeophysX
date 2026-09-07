"""Regression checks for scientific inputs, bounded sampling and HTTP caching."""

import io
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest

from PyHydroGeophysX.data_access.accessors import HttpHydroAccessor
from PyHydroGeophysX.petrophysics.monte_carlo import run_petrophysics_monte_carlo
from PyHydroGeophysX.petrophysics.resistivity_models import resistivity_to_saturation2


def test_missing_and_duplicate_layers_are_not_silent_zero_water():
    for layers in ([{"marker": 1}], [{"marker": 1}, {"marker": 1}, {"marker": 2}]):
        with pytest.raises(ValueError, match="Missing|unique"):
            run_petrophysics_monte_carlo([100., 100.], [1, 2], layers)


@pytest.mark.parametrize("override", [
    {"markers": [1.5, 2]}, {"resistivity": [100., np.nan]},
    {"products": []}, {"products": ["typo"]}, {"n_realizations": 1.5},
    {"cell_chunk_size": 0}, {"timestep_indices": [0.5]},
    {"layers": [{"marker": 1, "n": {"mean": 2, "std": -1}}, {"marker": 2}]},
])
def test_invalid_mc_inputs_fail_before_computation(override):
    arguments = dict(resistivity=[100., 200.], markers=[1, 2],
                     layers=[{"marker": 1}, {"marker": 2}])
    arguments.update(override)
    with pytest.raises(ValueError):
        run_petrophysics_monte_carlo(**arguments)


def test_chunked_samples_preserve_draws_and_exact_statistics():
    rho = np.array([[200., 300., 400.], [600., 800., 900.], [300., 500., 600.]])
    layers = [{"marker": 1, "rho_sat": {"mean": 80, "std": 4}, "sigma_sur": .0001},
              {"marker": 2, "m": {"mean": 1.5, "std": .05}, "sigma_sur": .0001}]
    kwargs = dict(layers=layers, markers=[1, 2, 1], products=["water_content", "porosity"],
                  n_realizations=9, seed=42, timestep_indices=[2, 0, 2])
    full = run_petrophysics_monte_carlo(rho, **kwargs, cell_chunk_size=100, return_realizations=True)
    small = run_petrophysics_monte_carlo(rho, **kwargs, cell_chunk_size=1)
    assert "water_content_all" not in small
    for marker in full["params_used"]:
        for key, values in full["params_used"][marker].items():
            np.testing.assert_array_equal(values, small["params_used"][marker][key])
    for product in kwargs["products"]:
        samples = full[product + "_all"]
        expected = dict(mean=samples.mean(axis=0), std=samples.std(axis=0),
                        p10=np.percentile(samples, 10, axis=0),
                        p50=np.percentile(samples, 50, axis=0), p90=np.percentile(samples, 90, axis=0))
        for key in expected:
            np.testing.assert_allclose(small["statistics"][product][key], expected[key], rtol=1e-14)


def test_analytical_water_content_and_integer_resistivities():
    result = run_petrophysics_monte_carlo([400, 900], [1, 1],
        [{"marker": 1, "rho_sat": 100., "n": 2., "porosity": .3}], n_realizations=3)
    np.testing.assert_allclose(result["statistics"]["water_content"]["mean"].ravel(), [.15, .1])
    np.testing.assert_allclose(resistivity_to_saturation2([400, 900], 100, 2), [.5, 1 / 3])


def test_summary_mode_does_not_allocate_full_sample_cube(monkeypatch):
    original_empty = np.empty
    def bounded_empty(shape, *args, **kwargs):
        assert shape != (7, 30, 2), "Summary mode allocated all realizations"
        return original_empty(shape, *args, **kwargs)
    monkeypatch.setattr(np, "empty", bounded_empty)
    run_petrophysics_monte_carlo(np.full((30, 2), 400.), np.ones(30),
        [{"marker": 1, "rho_sat": 100.}], n_realizations=7, cell_chunk_size=4)


def test_monte_carlo_preserves_global_rng():
    # Local Monte Carlo sampling must preserve the surrounding experiment.
    before = np.random.get_state()
    run_petrophysics_monte_carlo([400.], [1], [{"marker": 1}], n_realizations=2)
    after = np.random.get_state()
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_time_lapse_srt_accepts_sparse_ray_jacobians(monkeypatch):
    pg = pytest.importorskip("pygimli")
    from PyHydroGeophysX.inversion.srt_time_lapse import TimeLapseSRTInversion
    from scipy.sparse import issparse
    mesh = pg.createGrid(x=[0., 5., 10.], y=[-10., -5., 0.])
    data = pg.DataContainer()
    data.registerSensorIndex("s")
    data.registerSensorIndex("g")
    for pos in [(0., 0.), (5., 0.), (10., 0.)]:
        data.createSensor(pos)
    data.resize(3)
    data["s"], data["g"], data["t"] = [0, 0, 1], [1, 2, 2], [.005, .01, .005]
    monkeypatch.setattr(pg, "load", lambda *a, **kw: data)
    inv = TimeLapseSRTInversion(["a", "b"], [0., 1.], mesh=mesh,
                                max_iterations=2, target_chi_squared=0.)
    inv.setup()
    m = np.arange(inv.n_cells * inv.n_times).reshape(inv.n_times, inv.n_cells)
    np.testing.assert_array_equal(inv.Wt @ m.ravel(), (m[:-1] - m[1:]).ravel())
    assert inv.Wt.nnz == 2 * inv.n_cells
    model = np.full(inv.n_cells * inv.n_times, np.log(1 / 800.))
    predicted, jacobian = inv._forward_and_jacobian(model)
    assert issparse(jacobian)
    epsilon = 1e-5
    shifted, _ = inv._forward_and_jacobian(model + epsilon)
    np.testing.assert_allclose((shifted - predicted).ravel() / epsilon,
                               jacobian @ np.ones(model.size), rtol=1e-4, atol=1e-8)
    result = inv.run(initial_model=np.full(inv.n_cells, 800.))
    assert np.all(np.isfinite(result.final_models))
    assert result.all_chi2[-1] < result.all_chi2[0]


def _accessor(tmp_path, name="one"):
    return HttpHydroAccessor({"id": name, "base_url": "https://example.test/" + name,
                              "files": ["data.bin"]}, str(tmp_path))


def _response(data, length=None):
    response = io.BytesIO(data)
    response.headers = {"Content-Length": str(len(data) if length is None else length)}
    return response


def test_incomplete_download_is_retryable_and_never_cached(tmp_path, monkeypatch):
    accessor = _accessor(tmp_path)
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: _response(b"partial", 99))
    with pytest.raises(RuntimeError, match="Content-Length"):
        accessor._ensure_cached("data.bin")
    assert not accessor._cached_path("data.bin").exists()
    assert not list(accessor.cache_dir.glob("*.part"))
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: _response(b"complete"))
    assert accessor._ensure_cached("data.bin").read_bytes() == b"complete"


def test_shared_cache_clear_preserves_other_datasets_and_unrelated_files(tmp_path):
    first, second = _accessor(tmp_path), _accessor(tmp_path, "two")
    first._cached_path("data.bin").write_bytes(b"one")
    second._cached_path("data.bin").write_bytes(b"two")
    unrelated = first.cache_dir / "notes.txt"
    unrelated.write_text("keep")
    assert first.clear_cache() == 1
    assert unrelated.read_text() == "keep"
    assert second._cached_path("data.bin").read_bytes() == b"two"


def test_concurrent_cache_reads_download_once(tmp_path, monkeypatch):
    accessor = _accessor(tmp_path)
    calls = []
    def download(*args, **kwargs):
        calls.append(1)
        return _response(b"complete")
    monkeypatch.setattr("urllib.request.urlopen", download)
    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(lambda _: accessor._ensure_cached("data.bin").read_bytes(), range(8)))
    assert results == [b"complete"] * 8
    assert len(calls) == 1


@pytest.mark.parametrize("filename", ["../escape", "C:/escape", "/escape", "..\\escape"])
def test_manifest_cannot_escape_target(tmp_path, filename):
    with pytest.raises(ValueError):
        HttpHydroAccessor({"base_url": "https://example.test", "files": [filename]}, str(tmp_path))


def test_kriging_uses_real_coordinates(monkeypatch):
    pytest.importorskip("pyvista")
    from PyHydroGeophysX.core import kriging_3d
    monkeypatch.setattr(kriging_3d, "krige_seismic_velocity_3d", lambda topo, points, **kw: points)
    arguments = dict(profile_velocities={"a": np.array([[100., 200.], [300., 400.]])},
                     topography_data=np.array([[0., 0., 100.]]),
                     profile_locations={"a": ((0., 0.), (10., 0.))})
    with pytest.raises(ValueError, match="elevations"):
        kriging_3d.krige_from_2d_profiles(**arguments)
    points = kriging_3d.krige_from_2d_profiles(**arguments,
        profile_elevations={"a": [[99., 89.], [98., 88.]]}, profile_distances={"a": [0., 7.]})
    np.testing.assert_allclose(points, [[0, 0, 99, 100], [0, 0, 89, 200],
                                      [7, 0, 98, 300], [7, 0, 88, 400]])


def test_tdem_seed_preserves_global_rng_and_legacy_noise():
    pytest.importorskip("simpeg")
    from PyHydroGeophysX.forward.tdem_forward import TDEMForwardModeling
    model = SimpleNamespace(forward=lambda *a, **kw: np.array([1., 2., 3.]))
    np.random.seed(781)
    before = np.random.get_state()
    noisy, clean, _ = TDEMForwardModeling.forward_with_noise(model, [1.], seed=17)
    after = np.random.get_state()
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    expected = clean + .05 * np.abs(clean) * np.random.RandomState(17).randn(3)
    np.testing.assert_array_equal(noisy, expected)
