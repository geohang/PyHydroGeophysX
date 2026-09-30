"""Regression tests for the computations whose failures do not announce themselves.

Each of these once returned numbers that looked plausible and were wrong, or
read a file into the wrong values: a saturation inverse that missed its root,
water mass placed below the model, a data-assimilation schedule that used the
data twice, a Jacobian of a different model than the forward call's, a profile
run backwards that swapped regolith and bedrock, SEG-2 traces decoded at the
wrong width, an E4D mesh left in local coordinates. Optional packages are
guarded per test, so the file runs with the package alone.
"""

import struct
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from PyHydroGeophysX.analysis import compute_cumulative_sensitivity, compute_resolution_matrix
from PyHydroGeophysX.assimilation import ESMDA, EnsembleKalmanFilter, HydroGeophysObsOperator
from PyHydroGeophysX.core.hydro_profile import interpolate_layer_samples
from PyHydroGeophysX.core.interpolation import setup_profile_coordinates
from PyHydroGeophysX.petrophysics.monte_carlo import run_petrophysics_monte_carlo
from PyHydroGeophysX.petrophysics.resistivity_models import (
    resistivity_to_saturation, resistivity_to_saturation2, resistivity_to_water_content,
)
from PyHydroGeophysX.petrophysics.velocity_models import (
    velocity_to_water_content, water_content_to_velocity,
)
from PyHydroGeophysX.uncertainty import linearized_posterior


# --------------------------------------------------------------------------
# Petrophysics
# --------------------------------------------------------------------------

@pytest.mark.parametrize("n", [1.01, 1.1, 1.5, 2., 4., 6.])
@pytest.mark.parametrize("surface", [0., 1e-5, .1])
def test_saturation_inverse_recovers_forward_saturation_and_conductivity(n, surface):
    expected = np.array([1e-20, 1e-10, 1e-5, .01, .2, .8, 1.])
    conductivity = expected**n / 100. + surface * expected**(n - 1)
    actual = resistivity_to_saturation2(1. / conductivity, 100., n, surface)
    np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=0.)
    recovered = actual**n / 100. + surface * actual**(n - 1)
    np.testing.assert_allclose(recovered, conductivity, rtol=2e-12, atol=0.)


@pytest.mark.parametrize("n,surface,rho,expected", [
    (1.5, .1, 1000., .000099998000069997),
    (1.1, .1, 1000., 1e-20),
    (1.5, 1., 1e5, 1e-10),
    (2., 0., 1e12, 1e-5),
])
def test_all_public_routes_solve_the_dry_end(n, surface, rho, expected):
    # phi=.25, m=2, rho_fluid=6.25 gives rhos=100.
    results = [
        resistivity_to_saturation2(rho, 100., n, surface),
        resistivity_to_saturation(rho, .25, 2., 6.25, n, surface),
        resistivity_to_water_content(rho, 100., n, .25, surface) / .25,
    ]
    np.testing.assert_allclose(results, expected, rtol=1e-10, atol=0.)


def test_scalar_and_singleton_return_conventions_are_preserved():
    assert isinstance(resistivity_to_saturation2(400., 100., 2.), float)
    for rho in ([400.], np.array([400.]), np.array(400.)):
        result = resistivity_to_saturation2(rho, 100., 2.)
        assert isinstance(result, np.ndarray) and result.shape == (1,)
        np.testing.assert_allclose(result, [.5])
        assert isinstance(resistivity_to_saturation(rho, .25, 2., 6.25, 2.), float)
        assert isinstance(resistivity_to_water_content(rho, 100., 2., .25), float)
    # One value in a 2-D or 3-D array is still one value.
    for rho in (np.array([[400.]]), np.full((1, 1, 1), 400.)):
        assert isinstance(resistivity_to_saturation(rho, .25, 2., 6.25, 2.), float)
        assert isinstance(resistivity_to_water_content(rho, 100., 2., .25, .001), float)
    assert resistivity_to_saturation2([], 100., 2., .1).shape == (0,)


def test_monte_carlo_parameter_routes_match_known_saturations():
    expected = np.array([[1e-10, .01], [.1, .2], [.5, .8]])
    rho = 1. / (expected**1.5 / 100. + .1 * expected**.5)
    common = dict(marker=3, n=1.5, sigma_sur=.1, porosity=.25)
    for layer in (dict(common, rho_sat=100.), dict(common, m=2., rho_fluid=6.25)):
        result = run_petrophysics_monte_carlo(
            rho, np.full(3, 3), [layer], n_realizations=3,
            return_realizations=True, cell_chunk_size=1)
        assert result["saturation_all"].shape == (3, 3, 2)
        np.testing.assert_allclose(result["saturation_all"], np.tile(expected, (3, 1, 1)), rtol=1e-10)
        np.testing.assert_allclose(result["water_content_all"], result["saturation_all"] * .25)
        for key in ("mean", "p10", "p50", "p90"):
            np.testing.assert_allclose(result["statistics"]["water_content"][key],
                                       expected * .25, rtol=1e-10)


def test_seeded_monte_carlo_matches_the_legacy_converter_in_any_chunking():
    from PyHydroGeophysX.Geophy_modular.ert_to_wc_model import ERTtoWC

    rho = np.array([[1000., 2000.], [4000., 1e5], [100., 1000.]])
    distributions = {"rhos": {"mean": 100., "std": 10.}, "n": {"mean": 1.5, "std": .1},
                     "sigma_sur": {"mean": .1, "std": .01}, "porosity": {"mean": .25, "std": .01}}
    converter = ERTtoWC(None, rho, np.full(3, 3))
    converter.setup_layer_distributions({3: distributions})
    water, saturation, parameters = converter.run_monte_carlo(5, progress_bar=False, seed=42)
    for chunk in (1, 1024):
        result = run_petrophysics_monte_carlo(
            rho, np.full(3, 3), [dict(distributions, marker=3)], n_realizations=5,
            seed=42, return_realizations=True, cell_chunk_size=chunk)
        np.testing.assert_allclose(result["water_content_all"], water, rtol=1e-12)
        np.testing.assert_allclose(result["saturation_all"], saturation, rtol=1e-12)
        for name, value in parameters[3].items():
            np.testing.assert_array_equal(
                result["params_used"][3]["rho_sat" if name == "rhos" else name], value)


@pytest.mark.parametrize("model", ["linear", "wyllie", "raymer"])
def test_velocity_inverse_round_trips_with_variable_porosity(model):
    phi = np.array([.25, .3, .4, .5])
    water = phi * np.array([0., 1. / 3., 2. / 3., 1.])
    velocity = water_content_to_velocity(water, porosity=phi, model=model)
    np.testing.assert_allclose(velocity_to_water_content(velocity, porosity=phi, model=model),
                               water, atol=1e-12)


@pytest.mark.parametrize("model", ["linear", "wyllie", "raymer"])
@pytest.mark.parametrize("alias", [True, False])
def test_coupling_inverts_the_velocity_model_it_forwards_with(model, alias):
    pytest.importorskip("pygimli")
    from PyHydroGeophysX.inversion.cross_constraints import PetrophysicalCoupling

    params = {"porosity": .3, "velocity_model": "empirical" if alias else model,
              "empirical_model": model}
    wc = np.array([.1, .2])
    forward = PetrophysicalCoupling.water_content_to_all_geophysics(wc, **params)
    result = PetrophysicalCoupling.compare_inversions_to_hydro(
        wc, None, srt_result=SimpleNamespace(final_model=forward["velocity"]), petro_params=params)
    np.testing.assert_allclose(result["srt"]["water_content"], wc, atol=1e-12)


@pytest.mark.parametrize("offset", [1.0, 30.0, -10.0, 1000.0])
@pytest.mark.parametrize("start_day", [100.0, 365.0])
def test_seasonal_temperature_correction_is_relative_to_the_first_survey(offset, start_day):
    from PyHydroGeophysX.petrophysics.temperature import (
        correct_time_lapse_models, temperature_field)

    spec = {"mode": "seasonal", "start_day": start_day}
    times = np.array([0.0, 0.5, 2.0, 30.0])
    depths = [0.0, 0.5, 2.0]
    expected, _ = temperature_field(spec, depths, len(times), days=times)
    actual, _ = temperature_field(spec, depths, len(times), days=times + offset)
    np.testing.assert_allclose(actual, expected)
    models = np.full((len(depths), len(times)), 100.0)
    expected_models, _ = correct_time_lapse_models(models, spec, depths, days=times)
    actual_models, _ = correct_time_lapse_models(models, spec, depths, days=times + offset)
    np.testing.assert_allclose(actual_models, expected_models)


# --------------------------------------------------------------------------
# Gravity from a hydrological model
# --------------------------------------------------------------------------

def test_collinear_layer_samples_are_interpolated_along_their_line():
    x = np.array([0., 10., 20.])
    flat = np.column_stack((x, np.full(3, -5.)))
    query = np.array([[5., -1.], [15., -8.], [-4., -5.], [30., -2.]])
    np.testing.assert_allclose(interpolate_layer_samples(flat, [.1, .3, .2], query),
                               [.2, .25, .1, .2])
    # A single station of several layers is a vertical line.
    column = np.array([[0., -1.], [0., -3.], [0., -5.]])
    np.testing.assert_allclose(interpolate_layer_samples(column, [.3, .2, .1], [[4., -2.]]), [.25])


def test_a_single_flat_layer_runs_the_gravity_forward_model():
    pytest.importorskip("simpeg")
    from PyHydroGeophysX.Hydro_modular.hydro_to_gravity import hydro_to_gravity

    _, clean, _, contrast = hydro_to_gravity(
        np.full((1, 3), .2), np.full((1, 3), .4), [0., -10.],
        mesh_nx=3, mesh_nz=3, noise_level=0)
    assert clean.shape == (3,) and np.all(np.isfinite(clean)) and np.all(clean != 0)
    np.testing.assert_allclose(clean[0], clean[2])   # a symmetric model, a symmetric response
    np.testing.assert_allclose(contrast, 200., rtol=.01)   # 0.2 of water in place of air


@pytest.mark.parametrize("bottom", [np.full(4, -10.), np.array([-10., -7., -5., -8.])],
                         ids=["flat_bottom", "variable_bottom"])
def test_gravity_puts_no_water_mass_below_each_column(monkeypatch, bottom):
    pytest.importorskip("simpeg")
    from simpeg.potential_fields.gravity.simulation import Simulation3DIntegral

    from PyHydroGeophysX.Hydro_modular.hydro_to_gravity import hydro_to_gravity

    x = np.array([0., 3., 7., 10.])
    captured = {}
    original = Simulation3DIntegral.dpred

    def capture_and_compute(self, model, *args, **kwargs):
        captured["centers"], captured["density"] = self.mesh.cell_centers.copy(), model.copy()
        return original(self, model, *args, **kwargs)

    monkeypatch.setattr(Simulation3DIntegral, "dpred", capture_and_compute)
    hydro_to_gravity(np.full((2, 4), .2), np.full((2, 4), .4),
                     np.vstack([np.zeros(4), .5 * bottom, bottom]),
                     station_positions=x, mesh_nx=8, mesh_nz=12, noise_level=0.)
    centers, density = captured["centers"], captured["density"]
    outside = (centers[:, 2] < np.interp(centers[:, 0], x, bottom)) | (centers[:, 2] > 0.)
    np.testing.assert_array_equal(density[outside], 0.)
    np.testing.assert_allclose(density[~outside], .2 * (1000. - 1.225) / 1000.)


# --------------------------------------------------------------------------
# Inversion numerics
# --------------------------------------------------------------------------

def _identity():
    return HydroGeophysObsOperator(lambda x: x, lambda x: x)


@pytest.mark.parametrize("n_steps,schedule", [
    (4, [1., 1., 1., 1.]),          # the data four times over
    (2, [4., 4.]),                  # half of it
    (2, [2., -2.]),
    (2, [2., np.nan]),
    (1, [0.]),
])
def test_esmda_refuses_a_schedule_that_does_not_use_the_data_once(n_steps, schedule):
    with pytest.raises(ValueError, match="inflation_factors"):
        ESMDA(_identity(), np.eye(1), n_steps=n_steps, inflation_factors=schedule)


def test_esmda_accepts_valid_schedules_as_they_are_quoted():
    assert ESMDA(_identity(), np.eye(1), n_steps=3, inflation_factors=[2., 4., 4.])
    assert ESMDA(_identity(), np.eye(1), n_steps=4).inflation_factors == [4.] * 4
    # The decreasing schedule as it is usually quoted, to a few figures.
    assert ESMDA(_identity(), np.eye(1), n_steps=4, inflation_factors=[9.333, 7., 4., 2.])
    assert ESMDA(_identity(), np.eye(1), n_steps=4, inflation_factors=[9.33, 7., 4., 2.])


@pytest.mark.parametrize("method", ["enkf", "esmda"])
def test_ensemble_update_matches_the_gaussian_posterior(method):
    # Standardized members give exactly zero prior mean and unit sample variance.
    prior = np.random.default_rng(7).normal(size=(1, 20000))
    prior -= prior.mean()
    prior /= prior.std(ddof=1)
    if method == "enkf":
        posterior = EnsembleKalmanFilter(_identity(), np.array([[.25]])).update(
            prior, [2.], rng=np.random.default_rng(42))
    else:
        posterior = ESMDA(_identity(), np.array([[.25]]), n_steps=3,
                          inflation_factors=[2., 4., 4.]).update(
            prior, [2.], rng=np.random.default_rng(42))["ensemble"]
    assert posterior.mean() == pytest.approx(1.6, abs=.015)
    assert posterior.var(ddof=1) == pytest.approx(.2, abs=.01)


def test_correlated_posterior_matches_gaussian_conditioning():
    J = np.array([[1., 2., 0.], [0., 1., 1.]])
    prior = np.array([[3., .5, .2], [.5, 2., .3], [.2, .3, 1.]])
    Cd = np.array([[.4, .1], [.1, .5]])
    expected = prior - prior @ J.T @ np.linalg.solve(Cd + J @ prior @ J.T, J @ prior)
    posterior = linearized_posterior(J, Cd, prior)
    np.testing.assert_allclose(posterior, expected, atol=1e-13)
    assert np.linalg.eigvalsh(prior - posterior).min() > -1e-12


def test_resolution_whitens_with_the_transpose_of_a_nonsymmetric_weight():
    J, Wd = np.eye(2), np.array([[1., 1.], [0., 1.]])
    normal = (Wd @ J).T @ (Wd @ J)
    np.testing.assert_allclose(compute_resolution_matrix(J, Wd, 1., 1.),
                               np.linalg.solve(normal + np.eye(2), normal))


def test_cumulative_sensitivity_does_not_cancel_signed_derivatives():
    np.testing.assert_array_equal(
        compute_cumulative_sensitivity([[1., -2., 0.], [-1., 3., 0.]]), [2., 5., 0.])


@pytest.mark.parametrize("sparse", [False, True])
def test_dense_and_sparse_smoothness_matrices_scale_alike(sparse):
    pytest.importorskip("pygimli")
    from scipy.sparse import csr_matrix, issparse

    from PyHydroGeophysX.inversion.cross_constraints import StructuralConstraint

    scale = StructuralConstraint.apply_structural_weights_to_Wm
    Wm = np.array([[1., -1., 0.], [0., 1., -1.]])
    given = csr_matrix(Wm) if sparse else Wm

    def dense(result):
        assert issparse(result) is sparse
        result = result.toarray() if sparse else result
        assert result.dtype == float
        return result

    np.testing.assert_allclose(dense(scale(given, [1., 2., 3.])), [[1., -2., 0.], [0., 2., -3.]])
    np.testing.assert_allclose(dense(scale(given, [2., 5.])), [[2., -2., 0.], [0., 5., -5.]])


class _LinearForward:
    """Positive, non-diagonal response with an exact physical Jacobian."""

    def __init__(self, size):
        self.matrix = np.eye(size) + 0.1
        self.calls = 0

    def response(self, rho):
        self.calls += 1
        return self.matrix @ np.asarray(rho).ravel()

    def createJacobian(self, rho):
        import pygimli as pg

        self._jacobian = pg.matrix.Matrix(self.matrix)

    def jacobian(self):
        return self._jacobian


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("logs", [
    np.log([0.0005, 0.002, 0.1]),          # below the clipping limit
    np.log([1e5, 2e6, 5e6]),               # above it
    np.array([-1000., 2., 1000.]),
    np.log([20., 100., 300.]),
])
def test_ert_log_jacobian_differentiates_the_forward_calls_model(logs, cached):
    pytest.importorskip("pygimli")
    from PyHydroGeophysX.forward.ert_forward import ertforandjac2, ertforward2

    operator = _LinearForward(len(logs))
    expected, response = ertforward2(operator, logs, None, with_response=True)
    operator.calls = 0
    actual, jacobian, returned = ertforandjac2(
        operator, logs, None, response=response if cached else None, with_response=True)
    assert operator.calls == (0 if cached else 1)
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_allclose(returned, response)
    step = 1e-5
    columns = [(ertforward2(operator, logs + step * np.eye(len(logs))[i], None)
                - ertforward2(operator, logs - step * np.eye(len(logs))[i], None)) / (2 * step)
               for i in range(len(logs))]
    np.testing.assert_allclose(jacobian, np.column_stack(columns), rtol=2e-6, atol=1e-9)


@pytest.fixture
def em_line(monkeypatch):
    """A two-sounding line with its file input and physics replaced; the LCI solver is real."""
    from PyHydroGeophysX.inversion import em1d, em1d_lci
    from PyHydroGeophysX.inversion import em1d_line as workflow

    # Only the first two layers are observed. The third must be determined by
    # the specified prior, not accidentally anchored to the starting model.
    observed = np.log10([30.0, 30.0])
    uncertainty = np.full(2, 0.001)

    def forward(sigma):
        return np.log10(1.0 / np.asarray(sigma))[:2]

    def jacobian(sigma):
        return np.diag(-1.0 / (np.log(10.0) * np.asarray(sigma)))[:2]

    data = dict(n_soundings=2, times=np.array([1e-4, 2e-4]), frequencies=np.array([100., 200.]),
                response=observed, uncertainty=uncertainty)
    monkeypatch.setattr(workflow, "load_sounding", lambda *a, **kw: data)
    monkeypatch.setattr(em1d, "build_sounding_block", lambda *a, **kw: em1d_lci.SoundingBlock(
        forward, jacobian, observed, uncertainty, position=kw["position"], line=kw["line"]))
    monkeypatch.setattr(workflow, "_best_starting_resistivity", lambda *a, **kw: 75.0)
    monkeypatch.setattr(em1d, "tdem_moment_blocks", lambda *a, **kw: [
        {"observed": observed, "uncertainty": uncertainty}])
    monkeypatch.setattr(em1d, "_moment_forward", lambda blocks: forward)
    monkeypatch.setattr(em1d, "_moment_jacobian", lambda blocks: jacobian)
    return workflow


@pytest.mark.parametrize("method", ["TDEM", "FDEM"])
@pytest.mark.parametrize("automatic", [False, True])
@pytest.mark.parametrize("warm", [False, True])
@pytest.mark.parametrize("reference", [25.0, 0.0])
def test_em_line_damping_honors_its_reference_with_any_start(
        em_line, method, automatic, warm, reference):
    options = dict(n_layers=3, parallel_workers=1, auto_starting_model=automatic,
                   starting_resistivity=100.0, reference_resistivity=reference,
                   model_damping=100.0, lateral_smoothness=1.0, smoothness=0.01,
                   auto_lambda=False, verbose=False)
    result = em_line.invert_line(
        "synthetic.xyz", method, {}, options, max_soundings=2, doi_blank=False,
        initial_models=np.full((2, 3), 70.0) if warm else None, log=lambda message: None)
    # The section stores the deepest layer first.
    expected = reference or (75.0 if automatic else 100.0)
    np.testing.assert_allclose(result["model3d"][:, 0, 0], expected, rtol=1e-3)
    np.testing.assert_allclose(result["model3d"][:, 0, 1:], 30.0, rtol=1e-3)


# --------------------------------------------------------------------------
# ERT zones, profiles and meshes
# --------------------------------------------------------------------------

BLOCK = {"name": "Block", "polygon": [[5, -4], [12, -4], [12, -7], [5, -7]],
         "resistivity": 400.0, "fixed": True}
TOP = {"name": "Top", "polygon": [[14, 0], [20, 0], [20, -2], [14, -2]], "resistivity": 50.0}


@pytest.fixture(scope="module")
def synthetic_series(tmp_path_factory):
    """Three surveys over a two-layer ground, and the mesh built for the first."""
    pg = pytest.importorskip("pygimli")
    from pygimli.physics import ert

    from PyHydroGeophysX.forward.ert_forward import ERTForwardModeling
    from PyHydroGeophysX.inversion.ert_mesh import build_inversion_mesh

    folder = tmp_path_factory.mktemp("zones")
    grid = pg.createGrid(x=np.linspace(-5, 28, 34), y=np.linspace(-10, 0, 11))
    depth = np.asarray(grid.cellCenters())[:, 1]
    files = []
    for step, top in enumerate((50.0, 60.0, 75.0)):
        path = folder / f"step{step}.dat"
        ERTForwardModeling.create_synthetic_data(
            xpos=np.linspace(0, 23, 24), mesh=grid, res_models=np.where(depth > -3, top, 400.0),
            seed=step + 1, noise_level=0.01, relative_error=0.02, save_path=str(path))
        files.append(str(path))
    return files, build_inversion_mesh(ert.load(files[0]), mesh_quality=33, para_depth=8)


@pytest.mark.parametrize("method", ["cgls", "spd_cholesky"])
def test_single_inversion_holds_a_fixed_zone(synthetic_series, method):
    from PyHydroGeophysX.inversion.ert_inversion import ERTInversion

    files, mesh = synthetic_series
    inversion = ERTInversion(files[0], mesh=mesh, max_iterations=3, lambda_val=10.0,
                             method=method, verbose=False, zones=[BLOCK, TOP])
    model = np.asarray(inversion.run().final_model, dtype=float).ravel()
    prior = inversion.zone_prior()
    assert prior.fixed.sum() > 0 and np.allclose(model[prior.fixed], 400.0)
    # An a-priori zone that is not fixed is inverted like the rest.
    assert not np.allclose(model[prior.in_zone & ~prior.fixed], 50.0)


def test_time_lapse_holds_a_fixed_zone_in_every_survey(synthetic_series):
    from PyHydroGeophysX.inversion.time_lapse import TimeLapseERTInversion

    files, mesh = synthetic_series
    inversion = TimeLapseERTInversion(
        files, [0.0, 1.0, 2.0], mesh=mesh, lambda_val=20.0, alpha=10.0, max_iterations=2,
        method="spd_cholesky", model_constraints=(1.0, 1e4), verbose=False, zones=[BLOCK])
    models = np.asarray(inversion.run().final_models, dtype=float)
    fixed = inversion.zone_prior().fixed
    assert models.shape[1] == 3 and fixed.sum() > 0
    assert np.allclose(models[fixed, :], 400.0, rtol=1e-5)


def test_smoothness_stops_at_zone_outlines_and_only_there(synthetic_series):
    from pygimli.physics import ert

    from PyHydroGeophysX.inversion.ert_inversion import ERTInversion
    from PyHydroGeophysX.inversion.ert_mesh import mark_zone_interfaces

    files, mesh = synthetic_series
    data = ert.load(files[0])
    marked, edges = mark_zone_interfaces(mesh, [BLOCK])
    assert edges > 0
    empty_rows, responses = [], []
    for candidate in (mesh, marked):
        inversion = ERTInversion(data, mesh=candidate, verbose=False)
        inversion.setup()
        empty_rows.append(int((abs(inversion.Wm_r).sum(axis=1) == 0).sum()))
        fop = inversion.fwd_operator
        responses.append(np.asarray(fop.response(np.full(fop.paraDomain.cellCount(), 100.0))))
    # Exactly the marked edges lose their smoothness rows...
    assert empty_rows == [0, edges]
    # ...while the forward problem does not notice the marks.
    np.testing.assert_allclose(responses[0], responses[1], rtol=1e-10)


@pytest.mark.parametrize("reference_weight", [0.0, 1.0])
def test_depth_of_investigation_sees_the_reference_only_through_smallness(
        synthetic_series, reference_weight):
    from pygimli.physics import ert

    from PyHydroGeophysX.analysis import compute_depth_of_investigation
    from PyHydroGeophysX.inversion.ert_inversion import ERTInversion

    files, mesh = synthetic_series
    data = ert.load(files[0])
    # Both runs go to convergence, so they differ by their reference and not
    # by where each stopped: a run halted at the target misfit keeps its start
    # model wherever the data are weak.
    inversion = ERTInversion(data, mesh=mesh, lambda_val=10.0, max_iterations=8, verbose=False,
                             target_chi_squared=0.0, reference_weight=reference_weight)
    inversion.setup()
    cells = inversion.fwd_operator.paraDomain
    doi, _ = compute_depth_of_investigation(inversion, data, cells, reference_resistivity=100.0)
    # A fully reference-controlled cell scores (1.2 - 0.8) / (1.2 + 0.8).
    index = doi / 0.2
    x, z = np.asarray(cells.cellCenters())[:, :2].T
    under_array = (x > 2) & (x < 21)
    if reference_weight == 0.0:
        # First-order smoothness cancels a homogeneous reference: both runs
        # converge to one model.
        assert index.max() < 0.05
    else:
        assert index.max() > 0.5
        assert (index[under_array & (z < -6)].mean()
                > index[under_array & (z > -1)].mean() + 0.1)


@pytest.mark.parametrize("start,end", [([0, 0], [4, 0]), ([4, 0], [0, 0]), ([0, 0], [4, 2])])
@pytest.mark.parametrize("count", [2, 5, 17])
def test_profile_runs_from_point2_toward_point1(start, end, count):
    # Running it the other way swapped the regolith and bedrock markers of the
    # meshes the examples build, and misplaced their bundled data.
    top = np.zeros((3, 5), dtype=int)
    x, y, distance, _, _ = setup_profile_coordinates(
        start, end, top, origin_x=100., origin_y=200.,
        pixel_width=2., pixel_height=-3., num_points=count)
    p1 = np.array([100. + 2. * start[0], 200. - 3. * start[1]])
    p2 = np.array([100. + 2. * end[0], 200. - 3. * end[1]])
    assert len(x) == len(y) == len(distance) == count - 1
    np.testing.assert_allclose([x[0], y[0]], p2)
    np.testing.assert_allclose([x[-1], y[-1]], p2 + (p1 - p2) * (count - 2) / (count - 1))
    np.testing.assert_allclose(distance, np.linalg.norm(p1 - p2) / (count - 1) * np.arange(count - 1))


def test_example_mesh_keeps_the_regolith_on_top():
    """The EX_SRT_forward mesh: regolith (0) over fractured (3) over fresh bedrock (2)."""
    pytest.importorskip("pygimli")
    from PyHydroGeophysX.core.interpolation import ProfileInterpolator, create_surface_lines
    from PyHydroGeophysX.core.mesh_utils import MeshCreator

    data = Path(__file__).resolve().parents[1] / "examples" / "data"
    top = np.loadtxt(data / "top.txt")
    bot = np.load(data / "bot.npy")
    profile = ProfileInterpolator(point1=[115, 70], point2=[95, 180], surface_data=top,
                                  origin_x=569156.2983333333, origin_y=4842444.17,
                                  pixel_width=1.0, pixel_height=-1.0, num_points=400)
    structure = profile.interpolate_layer_data([top] + bot.tolist())
    surface, line1, line2 = create_surface_lines(L_profile=profile.L_profile, structure=structure,
                                                 top_idx=0, mid_idx=4, bot_idx=13)
    mesh, _ = MeshCreator(quality=32).create_from_layers(
        surface=surface, layers=[line1, line2], bottom_depth=np.min(line2[:, 1]) - 10)
    centers = np.array(mesh.cellCenters())
    markers = np.array(mesh.cellMarkers())
    depth = np.interp(centers[:, 0], surface[:, 0], surface[:, 1]) - centers[:, 1]
    median = {marker: np.median(depth[markers == marker]) for marker in (0, 3, 2)}
    assert median[0] < median[3] < median[2]


@pytest.mark.parametrize("name", ["site", "site.v2"])
def test_an_e4d_mesh_reads_back_into_survey_coordinates(tmp_path, name):
    """One-based numbering, the line E4D appends to the .node, and the .trn and
    .sig found beside the mesh however many dots its name has."""
    pytest.importorskip("pygimli")
    from PyHydroGeophysX.core import e4d_mesh as e4d

    nodes = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, -1], [1, 1, -1]], float)
    (tmp_path / f"{name}.1.node").write_text(
        "5 3 1 1\n" + "".join(f"{i + 1} {x} {y} {z} 1 {1 if z == 0 else 0}\n"
                              for i, (x, y, z) in enumerate(nodes))
        + " THIS NODE FILE WAS MODIFIED BY E4D.\n")
    (tmp_path / f"{name}.1.ele").write_text("2 4 1\n1 1 2 3 4 1\n2 2 3 4 5 2\n# Generated by tetgen\n")
    # An outer face on the surface, and the internal boundary between the zones.
    (tmp_path / f"{name}.1.face").write_text("2 1\n1 1 2 3 1\n2 2 3 4 7\n")
    (tmp_path / f"{name}.trn").write_text("  100.0  200.0  -10.0\n")
    (tmp_path / f"{name}.sig").write_text("2 1\n0.1\n0.02\n")
    mesh = e4d.read_e4d_mesh(tmp_path / f"{name}.1.node")
    assert (mesh.nodeCount(), mesh.cellCount()) == (5, 2)
    assert np.allclose(np.asarray(mesh.positions()), nodes + [100.0, 200.0, -10.0])
    assert list(mesh.cellMarkers()) == [1, 2]
    assert np.allclose(np.asarray(mesh["resistivity"]), [10.0, 50.0])
    # The face between the cells is left unmarked, where PyGIMLi would
    # otherwise cut the smoothness.
    marks = {tuple(sorted(n.id() for n in face.nodes())): face.marker()
             for face in mesh.boundaries() if face.marker()}
    assert marks == {(0, 1, 2): 1}


# --------------------------------------------------------------------------
# Seismic records (Geometrics DAT / SEG-2)
# --------------------------------------------------------------------------

def _seg2(path, traces, *, formats, intervals, descaling=None):
    """Write a minimal SEG-2 file: one descriptor block and samples per trace."""
    blocks = []
    for values, code, interval in zip(traces, formats, intervals):
        text = "" if interval is None else f"SAMPLE_INTERVAL {interval}\0"
        if descaling is not None:
            text += f"DESCALING_FACTOR {descaling}\0"
        strings = b"".join(struct.pack("<H", len(s) + 3) + s.encode() + b"\0"
                           for s in text.split("\0") if s) + b"\0\0"
        header = bytearray(32)
        struct.pack_into("<HHII", header, 0, 0x4422, 32 + len(strings), values.nbytes, values.size)
        header[12] = code      # the data format code, not a size in bytes
        blocks.append(bytes(header) + strings + values.tobytes())
    head = bytearray(32)
    struct.pack_into("<HHHH", head, 0, 0x3A55, 1, 4 * len(blocks), len(blocks))
    offset = 32 + 4 * len(blocks)
    pointers = b""
    for block in blocks:
        pointers += struct.pack("<I", offset)
        offset += len(block)
    path.write_bytes(bytes(head) + pointers + b"".join(blocks))
    return str(path)


@pytest.mark.parametrize("code, dtype", [(1, "<i2"), (2, "<i4"), (4, "<f4"), (5, "<f8")])
def test_each_seg2_sample_format_decodes_to_its_values(tmp_path, code, dtype):
    from PyHydroGeophysX.data_processing.seismic import read_geometrics_dat

    values = np.array([1, 256, -2, 32767], dtype=dtype)
    record = read_geometrics_dat(_seg2(tmp_path / "shot.dat", [values], formats=[code],
                                       intervals=[0.00025]))
    assert np.array_equal(record.traces[:, 0], values.astype(np.float32))
    assert np.allclose(record.time, [0.0, 0.00025, 0.0005, 0.00075])


def test_integer_seg2_samples_are_descaled(tmp_path):
    from PyHydroGeophysX.data_processing.seismic import read_geometrics_dat

    values = np.array([10, -20, 30], dtype="<i4")
    record = read_geometrics_dat(_seg2(tmp_path / "shot.dat", [values], formats=[2],
                                       intervals=[0.001], descaling=0.5))
    assert np.allclose(record.traces[:, 0], [5.0, -10.0, 15.0])


def test_a_seg2_record_keeps_one_clock(tmp_path):
    from PyHydroGeophysX.data_processing.seismic import read_geometrics_dat

    values = np.arange(4, dtype="<f4")
    # A trace sampled differently from the others is refused...
    with pytest.raises(ValueError, match="sample interval"):
        read_geometrics_dat(_seg2(tmp_path / "two.dat", [values, values], formats=[4, 4],
                                  intervals=[0.001, 0.002]))
    # ...and one that does not say takes the record's.
    record = read_geometrics_dat(_seg2(tmp_path / "one.dat", [values, values], formats=[4, 4],
                                       intervals=[None, 0.000125]))
    assert np.allclose(record.time, np.arange(4) * 0.000125)


# --------------------------------------------------------------------------
# Inversion inputs: data errors, smoothness, electrodes, comparisons
# --------------------------------------------------------------------------

@pytest.mark.parametrize("params,velocity,expected", [
    ({}, [3600., 3900., 4200., 4400.], None),       # Hertz-Mindlin with no calibrated inverse
    # Linear, S = (v - 3500) / 1000: the 450 m/s cell has no water content.
    ({"velocity_model": "linear"}, [450., 3900., 4200., 4400.], [np.nan, .12, .21, .27]),
])
def test_an_srt_model_that_cannot_be_converted_keeps_the_other_comparisons(
        params, velocity, expected):
    pytest.importorskip("pygimli")
    from PyHydroGeophysX.inversion.cross_constraints import PetrophysicalCoupling

    water = np.full(4, .15)
    with pytest.warns(UserWarning, match="SRT"):
        out = PetrophysicalCoupling.compare_inversions_to_hydro(
            water, SimpleNamespace(final_model=np.full(4, 300.)),
            SimpleNamespace(final_model=np.array(velocity)),
            SimpleNamespace(recovered_conductivity=np.full(4, 1. / 300.)), petro_params=params)
    for method in ("ert", "em"):
        np.testing.assert_allclose(out[method]["water_content"],
                                   resistivity_to_water_content(300., 100., 2., .3))
    if expected is None:
        assert "error" in out["srt"] and out["summary"]["skipped"] == ["srt"]
        assert out["summary"]["n_methods"] == 2
    else:
        np.testing.assert_allclose(out["srt"]["water_content"], expected)
        assert out["srt"]["stats"]["n_masked"] == 1
        assert out["srt"]["stats"]["rmse"] == pytest.approx(
            np.sqrt(np.mean((np.array(expected[1:]) - .15) ** 2)))


@pytest.mark.parametrize("with_resistance", [True, False])
def test_time_lapse_absolute_u_error_is_a_voltage_on_both_engines(tmp_path, with_resistance):
    pytest.importorskip("pygimli")
    from pygimli.physics import ert

    from PyHydroGeophysX.inversion.time_lapse import TimeLapseERTInversion
    from PyHydroGeophysX.inversion.windowed import WindowedTimeLapseERTInversion

    data = ert.createData(np.arange(6.) * 2, schemeName="dd")
    data["k"], data["rhoa"] = np.full(data.size(), 10.), np.full(data.size(), 100.)
    data["err"] = np.zeros(data.size())
    tokens = "a b m n rhoa k err"
    if with_resistance:
        data["r"] = np.full(data.size(), 10.)
        tokens += " r"
    data.save(str(tmp_path / "t.dat"), tokens)
    # |R| = 10 ohm, and no current recorded: pyGIMLi's 0.1 A gives U = 1 V, so
    # 0.02 V adds 0.02. An ohm floor of 0.5 adds 0.05; without 'r' the file used
    # to give 5e11. The full inversion's own default floor is 1e-4 ohm.
    for kwargs, windowed_error, full_error in (({}, .05, .05001),
                                               ({"absoluteUError": .02}, .07, .07001),
                                               ({"absoluteError": .5}, .10, .10)):
        windowed = WindowedTimeLapseERTInversion(str(tmp_path), ["t.dat"] * 2, [0, 1],
                                                 window_size=2, **kwargs)
        np.testing.assert_allclose(windowed._load_adtlert_series()[2], windowed_error)
        full = TimeLapseERTInversion([str(tmp_path / "t.dat")] * 2, [0, 1],
                                     verbose=False, **kwargs)
        full.setup()
        np.testing.assert_allclose(np.expm1(1. / full.Wd.diagonal()), full_error)


def test_srt_inversions_penalize_the_smoothness_they_are_given(tmp_path):
    tt = pytest.importorskip("pygimli.physics.traveltime")
    from PyHydroGeophysX.inversion.srt_inversion import SRTInversion
    from PyHydroGeophysX.inversion.srt_time_lapse import TimeLapseSRTInversion

    data = tt.createRAData(np.arange(0., 24., 2.))
    offsets = np.abs(np.asarray(data["g"]) - np.asarray(data["s"])) * 2.
    data["t"] = offsets / 1000. + .001
    mesh = tt.TravelTimeManager().createMesh(data=data, paraMaxCellSize=2., paraDepth=10.,
                                             quality=32)
    # zWeight scales the vertical constraints, as pyGIMLi's constraint weights do.
    weights = {}
    for z_weight in (1., .05):
        single = SRTInversion(data, mesh=mesh, zWeight=z_weight, verbose=False)
        single.setup()
        weights[z_weight] = np.asarray(abs(single.Wm).max(axis=1).todense()).ravel()
    np.testing.assert_allclose(weights[1.], 1.)
    assert weights[.05].min() == pytest.approx(.05, rel=.01) and weights[.05].max() < 1.

    # The time-lapse line search scores the objective run() descends, which
    # penalizes the departure from the start model: zero at the start itself.
    files = []
    for k in range(2):
        files.append(str(tmp_path / f"t{k}.sgt"))
        data.save(files[-1], "s g t")
    series = TimeLapseSRTInversion(files, [0., 1.], mesh=mesh, max_iterations=1)
    series.setup()
    start = np.log(1. / np.tile(series._build_initial_velocity(), 2))
    score, references = series._objective_terms, []
    series._objective_terms = lambda m, reference=None: (
        references.append(reference) or score(m, reference))
    series.run()
    assert references and all(np.allclose(ref, start) for ref in references)
    assert score(start, start)[1] == 0.


def test_electrodes_at_one_position_share_a_sensor_and_keep_their_geometric_factors():
    pytest.importorskip("pygimli")
    from PyHydroGeophysX.data_processing.ert_data_agent import (
        Electrode, Observation, Quadruplet, StandardERT)
    from PyHydroGeophysX.data_processing.ert_io import standard_to_pg

    # Electrode 2 repeats electrode 1's station, as a roll-along overlap does.
    xs = {1: 0., 2: 0., 3: 1., 4: 2., 5: 3., 6: 4.}
    quads = [(3, 4, 5, 6), (1, 3, 4, 5), (2, 3, 4, 5)]
    std = StandardERT(electrodes=[Electrode(i, x) for i, x in xs.items()],
                      observations=[Observation(Quadruplet(*q), 100.) for q in quads],
                      metadata={"app_res_source": "rhoa"})
    data = standard_to_pg(std)
    for row, quad in enumerate(quads):
        assert [data.sensorPosition(int(data[key][row])).x() for key in "abmn"] == \
            [xs[q] for q in quad]
    np.testing.assert_allclose(np.abs(data["k"]), 6. * np.pi)   # dipole-dipole, a = n = 1 m
    # A reading on both of them has no geometric factor, so the table is refused.
    std.observations.append(Observation(Quadruplet(1, 2, 3, 4), 100.))
    with pytest.raises(ValueError, match="electrodes 1, 2"):
        standard_to_pg(std)


def test_the_stacking_spread_error_model_reads_the_spread_back_from_the_saved_input(tmp_path):
    """The optional "stack" error model: each reading's sample spread, in quadrature.

    The spread reaches the inversion in a "stack" token of the saved input, which
    the studio's run reads with PyGIMLi's own reader and its walkthrough script
    with the BERT reader; both must hand it back. It is written for every reading
    or not at all, because a NaN in a saved row ends the data block for ResIPy's
    BERT reader, which then returns only the rows before it.
    """
    pytest.importorskip("pygimli")
    from pygimli.physics import ert

    from PyHydroGeophysX.data_processing.ert_data_agent import (
        Electrode, Observation, Quadruplet, StandardERT)
    from PyHydroGeophysX.data_processing.ert_io import standard_to_pg
    from PyHydroGeophysX.inversion.ert_inversion import _prepare_ert_data

    scheme = ert.createData(elecs=np.arange(12.), schemeName="dd")
    quads = [tuple(int(scheme[key][row]) + 1 for key in "abmn") for row in range(scheme.size())]
    spread = np.linspace(0.01, 0.4, len(quads))
    std = StandardERT(electrodes=[Electrode(i + 1, float(i)) for i in range(12)],
                      observations=[Observation(Quadruplet(*q), 100., rel_err=.05, stack=s)
                                    for q, s in zip(quads, spread)],
                      metadata={"app_res_source": "rhoa"})
    path = tmp_path / "filtered_ert_data.dat"
    standard_to_pg(std).save(str(path))
    for instrument in (None, "BERT"):
        estimate, _ = _prepare_ert_data(path, relative_error=.05, instrument=instrument,
                                        log=lambda *_: None, error_source="estimate")
        data, info = _prepare_ert_data(path, relative_error=.05, instrument=instrument,
                                       log=lambda *_: None, error_source="stack")
        assert info["source"].startswith("stacking spread"), instrument
        np.testing.assert_allclose(np.asarray(data["err"]),
                                   np.hypot(np.asarray(estimate["err"]), spread), rtol=1e-6)

    std.observations[3].stack = None
    assert not standard_to_pg(std).haveData("stack")


def test_an_electrode_file_keeps_each_measured_resistance(tmp_path):
    """Electrodes placed from a file keep R = rhoa/k; k and rhoa follow them.

    A file that reports apparent resistivity formed it on its own header's
    positions. With an electrode file doubling the spacing, k doubled and rhoa
    did not, so the resistance the data implied halved - whether the file came
    before the data or was laid over them afterwards - and a time-lapse run
    ignored the file altogether. A file listing a different number of electrodes
    is refused, where PyGIMLi's reader used to keep the header's positions.
    """
    pg = pytest.importorskip("pygimli")
    from pygimli.physics import ert

    from PyHydroGeophysX.data_processing import ert_io

    data = ert.createData(elecs=np.arange(12.), schemeName="dd")
    data["k"] = ert.createGeometricFactors(data, numerical=False)
    data["rhoa"] = 100.
    data["err"] = .03
    path = tmp_path / "line.dat"
    data.save(str(path))
    measured = 100. / np.asarray(data["k"])
    doubled, short = tmp_path / "doubled.txt", tmp_path / "short.txt"
    np.savetxt(doubled, np.column_stack([2. * np.arange(12.), np.zeros(12)]),
               header="x z", comments="")
    np.savetxt(short, np.column_stack([np.arange(11.), np.zeros(11)]),
               header="x z", comments="")

    for instrument in (None, "BERT"):          # PyGIMLi's own reader, and a device reader
        placed = ert_io.load_ert_container(str(path), instrument=instrument,
                                           electrode_file=str(doubled))
        assert placed.sensorPosition(1).x() == pytest.approx(2.), instrument
        np.testing.assert_allclose(np.asarray(placed["rhoa"]) / np.asarray(placed["k"]),
                                   measured, rtol=1e-6)
        with pytest.raises(ValueError, match="lists 11 electrodes"):
            ert_io.load_ert_container(str(path), instrument=instrument,
                                      electrode_file=str(short))

    rows = [{"order": i, "label": str(i + 1), "x": 2. * i, "z": 0., "original_index": i}
            for i in range(12)]
    saved = pg.DataContainerERT(
        ert_io.save_edited_ert_container(data, tmp_path / "edited.dat", rows))
    np.testing.assert_allclose(np.asarray(saved["rhoa"]) / np.asarray(saved["k"]),
                               measured, rtol=1e-6)

    _, _, series = ert_io.normalize_for_timelapse(
        [str(path), str(path)], None, str(tmp_path / "tl"), electrode_file=str(doubled))
    assert [step.sensorPosition(1).x() for step in series] == pytest.approx([2., 2.])


@pytest.mark.parametrize("rows,expected,kept", [
    # One quadrupole twice the same way round is a stack, not a reciprocal pair.
    ([(1, 2, 3, 4, 1.00), (1, 2, 3, 4, 1.08)], [np.nan, np.nan], 2),
    # The stack is averaged, then compared with its reciprocal: 1.04 against 1.04.
    ([(1, 2, 3, 4, 1.00), (1, 2, 3, 4, 1.08), (3, 4, 1, 2, 1.04)], [0., 0., 0.], 3),
    # Exchanged dipoles still pair, and 7.7 % fails a 5 % cut.
    ([(1, 2, 3, 4, 1.00), (3, 4, 1, 2, 1.08)], [.08 / 1.04] * 2, 0),
])
def test_same_direction_repeats_are_stacks_not_reciprocals(rows, expected, kept):
    import pandas as pd

    from PyHydroGeophysX.data_processing.ert_formats import reciprocal_errors

    frame = pd.DataFrame(rows, columns=["a", "b", "m", "n", "resist"])
    np.testing.assert_allclose(
        reciprocal_errors(frame, drop_failed=False)["reciprocalErrRel"], expected)
    assert len(reciprocal_errors(frame, max_reciprocal_error=.05)) == kept


# --------------------------------------------------------------------------
# Forward geometry, profile-to-mesh transfer, petrophysics and imports
# --------------------------------------------------------------------------

_RX12, _RX20 = [12., 0., 0.], [20., 0., 0.]


@pytest.mark.parametrize("survey,references", [
    # The same survey spelled as earlier releases accepted it.
    (dict(receiver_location=[_RX12]), [dict(receiver_location=_RX12)]),
    (dict(source_location=[[0., 0., 0.]], receiver_location=_RX12), [dict(receiver_location=_RX12)]),
    (dict(receiver_location=_RX12, waveform_type="magdipole"), [dict(receiver_location=_RX12)]),
    (dict(receiver_location=_RX12, waveform_type="VMD"), [dict(receiver_location=_RX12)]),
    # An omitted receiver sits at the origin wherever that is finite...
    (dict(source_location=[-5., 0., 0.]),
     [dict(source_location=[-5., 0., 0.], receiver_location=[0., 0., 0.])]),
    (dict(source_location=[0., 0., 1.5], waveform_type="loop"),
     [dict(source_location=[0., 0., 1.5], receiver_location=[0., 0., 0.], waveform_type="loop")]),
    # ...and 10 m out when the origin is below the dipole (a NaN response).
    (dict(), [dict(receiver_location=[10., 0., 0.])]),
    # Several receivers each keep their own real and imaginary parts.
    (dict(receiver_location=[_RX12, _RX20]),
     [dict(receiver_location=_RX12), dict(receiver_location=_RX20)]),
    (dict(receiver_location=[_RX12, _RX20], receiver_component="both"),
     [dict(receiver_location=_RX12, receiver_component="both"),
      dict(receiver_location=_RX20, receiver_component="both")]),
])
def test_fdem_survey_spellings_give_the_same_numbers(survey, references):
    pytest.importorskip("simpeg")
    from PyHydroGeophysX.forward.fdem_forward import FDEMForwardModeling, FDEMSurveyConfig
    from PyHydroGeophysX.inversion.fdem_inversion import FDEMInversion

    sigma, freqs = np.array([.01, .05, .02]), np.logspace(2, 4, 3)

    def run(config):
        model = FDEMForwardModeling(np.array([5., 10.]),
                                    FDEMSurveyConfig(frequencies=freqs, **config))
        return model, model.forward(sigma)

    model, actual = run(survey)
    singles = [run(config)[1].reshape(freqs.size, -1) for config in references]
    # One value per frequency, field and receiver, in that order.
    np.testing.assert_allclose(actual, np.stack(singles, axis=2).ravel(), rtol=1e-12, atol=0.)
    # The inversion writes such data back in SimPEG's own layout.
    np.testing.assert_allclose(FDEMInversion._to_simpeg_vector(actual, len(references)),
                               model.simulation.dpred(sigma), rtol=1e-12, atol=0.)


@pytest.mark.parametrize("layers", [5, 14, 20])
def test_profile_to_mesh_interpolation_takes_any_layer_count(layers):
    from PyHydroGeophysX.core.interpolation import interpolate_to_mesh

    # Layer k has value k; its top surface is at -k (top plus one bottom per layer).
    values = np.repeat(np.arange(layers, dtype=float)[:, None], 2, axis=1)
    surfaces = -np.repeat(np.arange(layers + 1, dtype=float)[:, None], 2, axis=1)
    y = np.array([-.5, 1.5 - layers])
    args = (np.array([0., 1.]), np.full(2, .5), y, np.zeros(2, dtype=int), np.zeros_like(values), [0])
    np.testing.assert_allclose(interpolate_to_mesh(values, args[0], surfaces, *args[1:]), -y)
    with pytest.raises(ValueError, match="depth_values"):
        interpolate_to_mesh(values, args[0], surfaces[:layers - 1], *args[1:])


@pytest.mark.parametrize("route,args,expected", [
    # No bulk conduction (rhos = inf): S = (C / B)**(1 / (n - 1)), not 1.
    (resistivity_to_saturation2, (1000., np.inf, 2., .005), .2),
    (resistivity_to_water_content, (1000., np.inf, 2., 1., .005), .2),
    (resistivity_to_saturation2, (100., np.inf, 3., .005), 1.),     # C >= B: saturated
    # A 2-D resistivity keeps its shape; rhos = 100 gives S = sqrt(100 / rho).
    (resistivity_to_saturation, (np.array([[400., 900.], [1600., 2500.]]), .25, 2., 6.25, 2.),
     np.array([[.5, 1. / 3.], [.25, .2]])),
    # A negative surface conductivity was answered as if it were zero.
    (resistivity_to_saturation2, (1000., 100., 2., -.001), ValueError),
    (resistivity_to_saturation, (1000., .25, 2., 6.25, 2., -.001), ValueError),
])
def test_waxman_smits_inverse_edge_cases(route, args, expected):
    if expected is ValueError:
        with pytest.raises(ValueError, match="sigma_sur"):
            route(*args)
    else:
        assert route(*args) == pytest.approx(expected, rel=1e-12, abs=0.)


@pytest.mark.parametrize("rho,variance,transform", [
    ([100.], [100.], lambda r: .3 * (50. / r) ** .5),                        # one cell
    ([100., 200.], [100., 400.], lambda r: np.mean(.3 * (50. / r) ** .5)),   # a summary
])
def test_a_single_output_uncertainty_keeps_a_covariance_matrix(rho, variance, transform):
    from PyHydroGeophysX.uncertainty import propagate_petro_uncertainty

    out = propagate_petro_uncertainty(np.array(rho), np.array(variance), transform,
                                      n_samples=200, seed=0)
    assert out["cov"].shape == (1, 1)
    np.testing.assert_allclose(np.diag(out["cov"]), np.var(out["samples"], axis=0, ddof=1))


@pytest.mark.parametrize("pygimli", ["blocked", "installed"])
def test_hydro_modular_imports_lazily_and_saves_what_it_is_asked_to(pygimli, tmp_path):
    import os
    import subprocess
    import sys

    import PyHydroGeophysX
    from PyHydroGeophysX._internal.optional_dependencies import optional_import_error

    if pygimli == "blocked":
        # Without PyGIMLi the package and its profile helpers still import,
        # and hydro_to_ert says what is missing.
        code = ("import sys; sys.modules['pygimli'] = None\n"
                "from PyHydroGeophysX.Hydro_modular import hydro_to_geophysics\n"
                "try:\n    from PyHydroGeophysX.Hydro_modular import hydro_to_ert\n"
                "except ImportError as exc:\n    print(exc)\n")
        root = str(Path(PyHydroGeophysX.__file__).resolve().parents[1])
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(
            filter(None, [root, os.environ.get("PYTHONPATH")])))
        run = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env,
                             capture_output=True, text=True, timeout=300)
        assert run.returncode == 0, run.stderr
        assert "'pygimli' could not be imported" in run.stdout
        # A module of the package itself failing is a broken install, not a
        # dependency to pip install.
        broken = ModuleNotFoundError("No module named 'PyHydroGeophysX.forward.x'",
                                     name="PyHydroGeophysX.forward.x")
        message = str(optional_import_error("X", broken))
        assert "'PyHydroGeophysX.forward.x'" in message and "pip install" not in message
        return

    pg = pytest.importorskip("pygimli")
    import PyHydroGeophysX.Hydro_modular.hydro_to_ert  # noqa: F401  (the submodule first)
    from PyHydroGeophysX.Hydro_modular import hydro_to_ert

    mesh = pg.createGrid(x=np.linspace(0, 12, 13), y=np.linspace(-4, 0, 5))
    mesh.setCellMarkers(np.full(mesh.cellCount(), 2))
    kwargs = dict(water_content=np.full(mesh.cellCount(), .2),
                  porosity=np.full(mesh.cellCount(), .35), mesh=mesh,
                  profile_interpolator=SimpleNamespace(L_profile=np.array([0., 12.]),
                                                       surface_profile=np.zeros(2)),
                  layer_idx=[0, 1, 2], structure=None, marker_labels=[3, 0, 2],
                  rho_parameters={}, num_electrodes=12, noise_level=0., seed=0)
    target = tmp_path / "synthetic_ert_data.dat"
    data, _ = hydro_to_ert(save_path=str(target), **kwargs)
    saved = pg.DataContainerERT(str(target))    # kept: its vectors are views into it
    np.testing.assert_allclose(np.array(saved["rhoa"]), np.array(data["rhoa"]), rtol=1e-5)


def test_a_moved_project_still_opens_its_runs(tmp_path):
    """Runs named their files by absolute path, so a moved Project could not reopen.

    New records are run-relative; an older record's absolute paths are rebased
    onto the run's new folder, and a path outside the run is still refused.
    """
    import json
    import shutil

    from PyHydroGeophysX.qt_apps.results_store import ResultsStore

    store = ResultsStore(tmp_path / "项目")
    run = store.begin_run("ert_processing", "ert.single_inversion")
    model = run.outputs_dir / "model.npy"
    model.write_bytes(b"0")
    store.finish_run(run, {"status": "ok", "summary": {"model_bundle": {"model": str(model)}}})
    store.save_run(run.run_id)
    record = json.loads(run.record.metadata_path.read_text(encoding="utf-8"))
    assert record["summary"]["model_bundle"] == {"model": "outputs/model.npy"}

    record["summary"]["model_bundle"] = {"model": str(model)}     # as written before
    run.record.metadata_path.write_text(json.dumps(record), encoding="utf-8")
    shutil.move(str(tmp_path / "项目"), str(tmp_path / "moved"))
    reopened = ResultsStore(tmp_path / "moved")
    found = reopened.list_runs()[0]
    assert (reopened.locate_run_artifact(found, found.summary["model_bundle"]["model"])
            == (found.run_dir / "outputs" / "model.npy").resolve())
    with pytest.raises(ValueError, match="outside this run's folder"):
        reopened.locate_run_artifact(found, str(tmp_path / "elsewhere.npy"))


def test_a_discard_windows_refuses_leaves_no_run_to_save(tmp_path, monkeypatch):
    """Windows refused to delete a run folder the next run's process waited in.

    The delete had already emptied the folder and the run stayed unsaved, so the
    Save that followed put a run without files into the history. The run is now
    discarded either way, and the folder is marked for the next session to clear.
    """
    import shutil

    from PyHydroGeophysX.qt_apps.results_store import ResultsStore

    store = ResultsStore(tmp_path / "project")
    run = store.begin_run("ert_processing", "ert.single_inversion")
    (run.outputs_dir / "model.npy").write_bytes(b"0")
    store.finish_run(run, {"status": "ok"})
    rmtree = shutil.rmtree

    def refused(path, *args, **kwargs):          # everything but the folder itself
        for child in Path(path).iterdir():
            rmtree(child) if child.is_dir() else child.unlink()
        raise PermissionError(32, "being used by another process", str(path))

    monkeypatch.setattr(shutil, "rmtree", refused)
    with pytest.raises(OSError, match="run itself is discarded"):
        store.discard_run(run.run_id)
    monkeypatch.undo()
    assert not store.has_unsaved()
    assert [path.resolve() for path in store.abandoned_run_dirs()] == [run.run_dir]


# --------------------------------------------------------------------------
# Code written for 0.3.0, the last release before 0.5.0
# --------------------------------------------------------------------------

#: 0.3.0 modules that are shims now. They warn once, as they are imported, so
#: their cases import them afresh.
_V030_SHIMS = ("model_output.modflow_output", "qt_apps.ert_timelapse", "agents.fetch_climate_data",
               "qt_apps.em_pipeline")


@pytest.mark.parametrize("statement, needs, error, says", [
    # Names 0.3.0 modules held, some of them only imported from elsewhere.
    ("from PyHydroGeophysX.Geophy_modular.ERT_to_WC import resistivity_to_saturation", (), None,
     "resistivity_models.resistivity_to_saturation2"),     # not petrophysics.resistivity_to_saturation
    ("from PyHydroGeophysX.model_output.modflow_output import HydroModelOutput", (), None,
     "use model_output.water_content"),
    ("from PyHydroGeophysX.qt_apps.ert_timelapse import DEFAULT_TL, INVERSION_TYPES, ert_load",
     ("pygimli",), None, "use inversion.time_lapse"),
    ("from PyHydroGeophysX.solvers.solver import gpu_available", (), None, "use_gpu=True"),
    ("from PyHydroGeophysX.qt_apps.em_pipeline import io_utils", (), None, "use workflows.em1d"),
    # A page module's helper that is now a module elsewhere.
    ("from PyHydroGeophysX.qt_apps.modules.gravmag_processing import gmp", ("PySide6", "pyqtgraph"), None,
     "use PyHydroGeophysX.workflows.gravmag"),
    ("from PyHydroGeophysX.agents.fetch_climate_data import DEFAULT_VARIABLES", (), None,
     "ClimateDataAgent.execute()"),
    # Removed with no drop-in replacement: the error names it, even through
    # "from ... import", which reports a module __getattr__'s AttributeError
    # only as "cannot import name".
    ("from PyHydroGeophysX.Geophy_modular.structure_integration import create_ert_mesh_with_structure",
     (), ImportError, "core.mesh_utils.add_velocity_interface"),
    ("from PyHydroGeophysX.Geophy_modular import create_joint_inversion_mesh", (), ImportError,
     "process_seismic_tomography"),
])
def test_a_030_import_warns_once_or_names_the_replacement(statement, needs, error, says, monkeypatch):
    import sys
    import warnings

    from PyHydroGeophysX.solvers import linear_solvers

    for package in needs:
        pytest.importorskip(package)
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    # Asking for gpu_available imports CuPy; whether it is here is not the question.
    monkeypatch.setattr(linear_solvers, "_load_gpu_backend", lambda: False)
    module = statement.split()[1]
    if module.split(".", 1)[1] in _V030_SHIMS:
        monkeypatch.delitem(sys.modules, module, raising=False)
    if error is not None:
        with pytest.raises(error, match=f"was removed in PyHydroGeophysX 0.5.0: .*{says}"):
            exec(statement, {})
        return
    namespace = {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        exec(statement, namespace)
    ours = [str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)
            and "PyHydroGeophysX 0.5.0" in str(w.message)]
    # "will be removed in 0.6.0", or "was removed in PyHydroGeophysX 0.5.0".
    assert len(ours) == 1 and says in ours[0] and "removed in" in ours[0], ours
    assert all(namespace[name.strip(",")] is not None for name in statement.split(" import ")[1].split())


@pytest.mark.parametrize("case", ["sim_ws", "sim_ws_attribute", "nlay_uzf", "file_obj", "binaryread_record",
                                  "both_names", "tdem_positional", "conda_climate", "cgls"])
def test_a_030_call_still_runs_as_it_did(case, tmp_path, monkeypatch, request):
    """Renamed keywords and attributes warn once and work; a field order,
    binaryread's record and the cgls iteration cap are as 0.3.0 had them, with
    no warning at all."""
    import json
    import warnings

    from PyHydroGeophysX.model_output.water_content import MODFLOWWaterContent, binaryread

    # A UZF WaterContent file of one record: two cells, one layer.
    idomain = np.ones((1, 2))
    header = np.zeros(1, MODFLOWWaterContent._RECORD_HEADER)
    header["maxbound"], header["1"] = 2, 1
    (tmp_path / "WaterContent").write_bytes(header.tobytes() + np.array([.1, .2]).tobytes())
    says, check = None, lambda value: value is not None

    if case == "sim_ws":
        says = "use MODFLOWWaterContent(model_directory=...)"
        call = lambda: MODFLOWWaterContent(sim_ws=str(tmp_path), idomain=idomain)   # noqa: E731
        check = lambda reader: reader.model_directory == str(tmp_path)              # noqa: E731
    elif case == "sim_ws_attribute":
        says = "use MODFLOWWaterContent.model_directory"
        call = lambda: MODFLOWWaterContent(str(tmp_path), idomain).sim_ws          # noqa: E731
        check = lambda folder: folder == str(tmp_path)                              # noqa: E731
    elif case == "nlay_uzf":
        says = "use MODFLOWWaterContent.load_timestep(nlay=...)"
        call = lambda: MODFLOWWaterContent(str(tmp_path), idomain).load_timestep(0, nlay_uzf=1)  # noqa: E731
        check = lambda values: values.tolist() == [[[.1, .2]]]                      # noqa: E731
    elif case == "file_obj":
        handle = (tmp_path / "WaterContent").open("rb")
        request.addfinalizer(handle.close)
        says = "use binaryread(file=...)"
        call = lambda: binaryread(file_obj=handle, vartype=header.dtype)           # noqa: E731
        check = lambda record: int(record[0]["maxbound"]) == 2                      # noqa: E731
    elif case == "binaryread_record":
        handle = (tmp_path / "WaterContent").open("rb")
        request.addfinalizer(handle.close)
        fields = [(name, header.dtype[name].str) for name in header.dtype.names]

        def call():
            record = binaryread(handle, fields)
            with pytest.raises(EOFError):   # 16 bytes left, less than a header
                binaryread(handle, fields)
            return record

        # 0.3.0 gave the record itself, fields by name, not a one-record array.
        check = lambda record: isinstance(record, np.void) and int(record["maxbound"]) == 2  # noqa: E731
    elif case == "both_names":
        with pytest.raises(TypeError, match="got both 'model_directory' and 'sim_ws'"):
            MODFLOWWaterContent(str(tmp_path), idomain, sim_ws=str(tmp_path))
        return
    elif case == "tdem_positional":
        pytest.importorskip("simpeg")
        from PyHydroGeophysX.forward.tdem_forward import TDEMSurveyConfig

        times = np.logspace(-5, -3, 5)
        # source_location, source_radius, source_current, receiver_location,
        # receiver_orientation, times, waveform_type: 0.3.0's order.
        call = lambda: TDEMSurveyConfig(np.zeros(3), 12., 2., np.ones(3), "x", times, "step_off")  # noqa: E731
        check = lambda cfg: (cfg.receiver_orientation, cfg.times is times, cfg.source_turns) == (  # noqa: E731
            "x", True, 1) and cfg.waveform_type == "step_off" and cfg.receiver_location.tolist() == [1.] * 3
    elif case == "conda_climate":
        import pandas as pd

        from PyHydroGeophysX.agents.climate_data_agent import ClimateDataAgent

        def open_meteo(self, lat, lon, start, end):     # no network
            days = pd.date_range(start, end, name="time")
            return pd.DataFrame({"prcp": 1., "tmin": 2., "tmax": 9., "pet": .5}, index=days), {}

        monkeypatch.setattr(ClimateDataAgent, "_fetch", open_meteo)
        monkeypatch.chdir(tmp_path)          # 0.3.0 saved the series under ./data/climate
        (tmp_path / "climate.json").write_text(json.dumps(
            {"coords": [-105.3, 40.], "dates": ["2021-09-01", "2021-09-05"], "variables": ["prcp", "srad"]}))
        says = "use ClimateDataAgent.execute("
        call = lambda: ClimateDataAgent().fetch_climate_data_with_conda("climate.json")  # noqa: E731
        check = lambda result: result["success"] and Path(result["csv_path"]).resolve() == (  # noqa: E731
            tmp_path / "data" / "climate" / "climate_data.csv").resolve() and "srad" in result["message"]
    else:   # cgls
        pytest.importorskip("pygimli")
        from PyHydroGeophysX.inversion import time_lapse

        files, mesh = request.getfixturevalue("synthetic_series")
        caps = []
        monkeypatch.setattr(time_lapse, "generalized_solver", lambda A, b, **options: (
            caps.append(options.get("maxiter")) or np.zeros(len(b))))
        call = lambda: time_lapse.TimeLapseERTInversion(                            # noqa: E731
            files[:2], [0., 1.], mesh=mesh, max_iterations=1, method="cgls", verbose=False).run()
        check = lambda result: caps == [2000]       # 0.3.0's cap, not generalized_solver's 200  # noqa: E731

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = call()
    ours = [str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)
            and "PyHydroGeophysX 0.5.0" in str(w.message)]
    assert len(ours) == (0 if says is None else 1) and all(says in message for message in ours), ours
    assert check(value)


# --------------------------------------------------------------------------
# Plot length units: feet, and depth for a survey without elevations
# --------------------------------------------------------------------------

@pytest.mark.parametrize("unit, first, second", [("m", "10", "20"), ("ft", "20", "40")])
def test_feet_put_round_feet_on_metre_data_and_leave_the_data_alone(unit, first, second):
    pytest.importorskip("matplotlib")
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    from PyHydroGeophysX.visualization.axis_units import set_length_axis

    fig = Figure()
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    line, = ax.plot([0.0, 40.0], [0.0, 1.0])
    assert set_length_axis(ax, "x", "Distance", unit=unit) == f"Distance ({unit})"
    fig.canvas.draw()
    shown = [t.get_text() for t in ax.get_xticklabels()]
    assert first in shown and second in shown, shown
    assert line.get_xdata().tolist() == [0.0, 40.0]        # still metres
    if unit == "ft":
        assert ax.format_coord(10.0, 0.5).startswith("(x, y) = (32.81")


@pytest.mark.parametrize("top, label, lowest_tick", [
    (0.0, "Depth (ft)", "0"),            # read without elevations: depth, positive down
    (300.0, "Elevation (ft)", None),     # real topography keeps its elevations
])
def test_a_section_without_elevation_reads_depth(top, label, lowest_tick):
    pg = pytest.importorskip("pygimli")
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    from PyHydroGeophysX.visualization import plot_model_section

    grid = pg.createGrid(x=np.linspace(0, 40, 21), y=np.linspace(-10, 0, 6) + top)
    fig = Figure()
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    plot_model_section(grid, np.linspace(10, 100, grid.cellCount()), ax=ax, length_unit="ft")
    # pyGIMLi's colorbar opens an empty pyplot figure, which its exit handler
    # would otherwise try to show once the session ends.
    import matplotlib.pyplot as plt
    plt.close("all")
    fig.canvas.draw()
    shown = [float(t.get_text().replace("−", "-")) for t in ax.get_yticklabels()]
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("Distance (ft)", label)
    if lowest_tick is None:
        assert min(shown) > 900.0        # 300 m is 984 ft
    else:
        assert min(shown) == 0.0 and max(shown) >= 30.0    # 10 m of depth is 32.8 ft
