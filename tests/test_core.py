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


@pytest.mark.parametrize("mode", ["direct", "spatial"])
def test_sparse_cross_gradient_operators_equal_the_dense_assembly(mode):
    pg = pytest.importorskip("pygimli")
    from scipy.sparse import csr_matrix, diags, issparse

    from PyHydroGeophysX.inversion.cross_constraints import StructuralConstraint
    from PyHydroGeophysX.inversion.joint_ert_srt import JointERTSRTInversion

    # Graded cells; every neighbourhood spans three non-collinear centres.
    mesh = pg.createGrid(x=30. * np.linspace(0., 1., 31)**1.3, y=-6. * np.linspace(1., 0., 9)**1.5)
    n = mesh.cellCount()
    pairs = np.asarray(StructuralConstraint._cell_neighbors(mesh))
    Wm = csr_matrix((np.tile([1., -1.], len(pairs)),
                     (np.repeat(np.arange(len(pairs)), 2), pairs.ravel())), shape=(len(pairs), n))
    X = StructuralConstraint.build_local_design_matrix(mesh)
    ma = np.sin(X[:, 0] / 2.) + X[:, 1] / 3.
    mb = np.cos(X[:, 0] / 3.) * np.exp(X[:, 1] / 4.)

    # The dense assembly the sparse one replaced, with W = diag(RCM[i]) per row.
    if mode == "direct":
        R = (Wm.T @ Wm).toarray()
    else:
        R = np.asarray(pg.utils.covarianceMatrix(mesh, I=[4., 4.]), dtype=float)
    R[np.abs(R) < .01] = 0.
    if mode == "direct":
        R = (R != 0).astype(float)
    B1_ref, B2_ref = np.zeros((n, n)), np.zeros((n, n))
    for i in range(n):
        W = np.diag(R[i])
        Xbar = np.linalg.solve(X.T.dot(W.T).dot(X), X.T.dot(W.T).dot(W))[:2]
        g1, g2 = Xbar @ ma, Xbar @ mb
        g1[np.abs(g1) < 1e-8] = 0.
        g2[np.abs(g2) < 1e-8] = 0.
        B1_ref[i] = Xbar[0] * g2[1] - Xbar[1] * g2[0]
        B2_ref[i] = -(Xbar[0] * g1[1] - Xbar[1] * g1[0])

    RCM = StructuralConstraint.build_neighborhood_matrix(
        mesh, Wm=Wm, source="smoothness" if mode == "direct" else "geostat",
        correlation_lengths=(4., 4.), threshold=.01, binarize=mode == "direct")
    assert issparse(RCM)
    np.testing.assert_array_equal(RCM.toarray(), R)
    for given in (RCM, R):
        B1, B2 = StructuralConstraint.build_linearized_cross_gradient_blocks(given, X, ma, mb, mode=mode)
        assert issparse(B1) and issparse(B2)
        for B, ref in ((B1, B1_ref), (B2, B2_ref)):
            np.testing.assert_allclose(B.toarray(), ref, rtol=0., atol=1e-11 * np.abs(ref).max())

    # The stacked system keeps a dense Jacobian dense and the rest sparse.
    inv = object.__new__(JointERTSRTInversion)
    J = np.random.default_rng(0).standard_normal((9, n))
    Wd = diags(np.linspace(1., 2., 9), format="csr")
    d_obs, d_pred = np.ones((9, 1)), np.full((9, 1), 1.1)
    A = inv._stack_system(J, Wd, Wm, B1, ma, 0. * ma, d_pred, d_obs, 10., 80.)[0]
    assert [issparse(block) for block in A.blocks] == [False, True, True]
    np.testing.assert_allclose(
        A.toarray(), np.vstack([Wd @ J, np.sqrt(10.) * Wm.toarray(), np.sqrt(80.) * B1_ref]),
        rtol=0., atol=1e-11 * np.abs(B1_ref).max())


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


def test_a_tem_line_inverts_alike_with_or_without_its_recorded_system():
    # The data sign was read from the caller's geometry alone, while the rest
    # of the forward geometry was completed from the station's recorded system:
    # a TEM2Go line given only its moment was fitted with every gate's sign
    # flipped, at chi-squared ~750 with a blank section, and no error.
    pytest.importorskip("simpeg")
    from PyHydroGeophysX.workflows import em1d

    project = str(em1d.example_catalog()["synthetic_tem_lci"]["path"])
    head = em1d.load_sounding(project, "TDEM", moment="LM+HM")
    assert head["system"]["response_sign"] == -1.0
    inversion = {**em1d.preset_inversion("ground_tem"), **head["inversion_defaults"],
                 "parallel_workers": 1}
    bare, full = (em1d.invert_line(project, "TDEM", geometry, inversion, max_soundings=9)
                  for geometry in ({"tem_moment": "LM+HM"},
                                   {**head["system"], "tem_moment": "LM+HM"}))
    assert bare["chi2_global"] < 2.0
    np.testing.assert_allclose(bare["chi2_global"], full["chi2_global"], rtol=1e-6)
    np.testing.assert_allclose(bare["model3d"], full["model3d"], rtol=1e-6)


def test_a_tem_jacobian_and_its_operators_are_what_the_line_needs():
    pytest.importorskip("simpeg")
    pytest.importorskip("numba")
    from PyHydroGeophysX.inversion import em1d as inv1d
    from PyHydroGeophysX.inversion.em1d_lci import _LANE, _map_soundings, _worker_pool
    from PyHydroGeophysX.workflows import em1d

    # The compiled response and conductivity Jacobian are SimPEG's dpred and
    # getJ, without the thickness and permeability gradients getJ also
    # computes and a line inversion discards.
    project = str(em1d.example_catalog()["synthetic_tem_lci"]["path"])
    data = em1d.load_sounding(project, "TDEM", sounding=2, moment="LM+HM")
    inv = {**em1d.preset_inversion("ground_tem"), **data["inversion_defaults"]}
    thick = inv1d._layer_thicknesses(int(inv["n_layers"]), float(inv["min_thickness"]),
                                     float(inv["max_thickness"]))
    sigma = 1.0 / np.geomspace(5.0, 2000.0, thick.size + 1)
    for item in inv1d.tdem_moment_blocks(data, {"tem_moment": "LM+HM"}, inv, thick):
        modeler = inv1d._block_modeler(item)
        weights = modeler._gate_weights
        for fast, slow in ((modeler.forward, modeler.simulation.dpred),
                           (modeler.sensitivity, modeler.simulation.getJ)):
            reference = np.asarray(slow(sigma))
            reference = reference if weights is None else weights @ reference
            np.testing.assert_allclose(fast(sigma), reference,
                                       atol=1e-8 * np.abs(reference).max())

    # A chunk of soundings keeps its operator cache from one pass to the next;
    # a cache per thread was rebuilt by every thread for nearly every station.
    with _worker_pool(4) as pool:
        passes = [_map_soundings(pool, lambda s: id(_LANE.cache), 12) for _ in range(3)]
    assert passes[0] == passes[1] == passes[2] and len(set(passes[0])) == 4


def test_a_survey_is_mapped_between_its_lines_and_no_further(tmp_path):
    pytest.importorskip("PIL")
    from PIL import Image

    from PyHydroGeophysX.visualization.basemap import local_basemap_image
    from PyHydroGeophysX.visualization.em_maps import resistivity_depth_slices

    # Three lines 20 m apart, a sounding every 2 m, a conductor under the
    # middle line at 5-10 m, the third line not reaching 10-20 m, and nothing
    # resolved below 20 m.
    x, y = np.repeat([0.0, 20.0, 40.0], 26), np.tile(np.arange(0.0, 52.0, 2.0), 3)
    lines = np.repeat([1, 2, 3], 26)
    edges = np.array([0.0, 2.0, 5.0, 10.0, 20.0, 40.0])
    models = np.full((x.size, 5), 100.0)
    models[lines == 2, 2] = 10.0
    models[:, 3] = 100.0 * 10 ** (0.2 * np.sin(y / 6.0))
    models[lines == 3, 3] = np.nan
    models[:, 4] = np.nan
    grids = resistivity_depth_slices(x, y, models, edges, depths=[7.5, 15.0, 30.0], lines=lines)
    gx, gy = np.meshgrid(grids["x"], grids["y"])

    def at(px, py, k=0, key="slices", source=None):
        return (source or grids)[key][k][np.unravel_index(
            np.argmin((gx - px) ** 2 + (gy - py) ** 2), gx.shape)]

    assert at(20, 25) == pytest.approx(10.0, rel=0.05)
    assert 10.0 < at(10, 25) < 100.0                         # between lines: kriged
    assert np.isnan(at(60, 25))                              # 20 m past the outer line
    assert np.isnan(grids["slices"][2]).all() and grids["soundings_used"][2] == 0
    # Filled where the third line does not reach 15 m, and less sure there.
    assert np.isfinite(at(38, 25, k=1))
    assert at(38, 25, k=1, key="std") > 2 * at(2, 25, k=1, key="std")
    # The kriging error is small beside a sounding and larger between lines.
    assert at(20, 24, key="std") < 0.1 < at(10, 25, key="std")
    assert grids["variograms"][0]["fitted"] and grids["variograms"][2] is None
    # Asked to, the maps stop short of the gaps between lines.
    near = resistivity_depth_slices(x, y, models, edges, depths=[7.5], lines=lines,
                                    max_distance=5.0)
    assert np.isnan(at(10, 25, source=near)) and np.isfinite(at(20, 25, source=near))
    # What the agent and the EM page both draw from a line inversion.
    from PyHydroGeophysX.visualization.em_maps import draw_survey_plan_maps, survey_plan_grids
    section = {"model3d": models[:, ::-1][:, None, :], "x": x, "y": y, "depth_edges": edges,
               "line_numbers": lines}
    drawn = draw_survey_plan_maps(survey_plan_grids(section, basemap="none"), tmp_path,
                                  unit="ft")
    assert Path(drawn["map_figure"]).exists() and Path(drawn["map_uncertainty_figure"]).exists()
    assert any(p.endswith("_std.asc") for p in drawn["map_grids"])
    with pytest.raises(ValueError, match="one line"):
        resistivity_depth_slices(np.zeros(5), np.arange(5.0), models[:5], edges)

    # A world-file image in the axes' own coordinates lands where it says.
    picture = np.zeros((20, 40, 3), dtype=np.uint8)
    picture[:, :20, 0] = 255                                  # west half red
    picture[:, 20:, 2] = 255                                  # east half blue
    Image.fromarray(picture).save(tmp_path / "field.png")
    (tmp_path / "field.pgw").write_text("2\n0\n0\n-2\n-9\n51\n")   # x -10..70, y 12..52
    image = local_basemap_image(tmp_path / "field.png", (0.0, 60.0), (20.0, 40.0),
                                transform=(1.0 + 0j, 0j), target_pixels=60)
    assert image["image"][5, 5, 0] == 255 and image["image"][5, 55, 2] == 255


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


#: E4D's modes 3 and 4 as far as their files go: inputs read the way
#: READ_INP.F90 reads them, refusing what E4D refuses, and E4D's outputs written
#: as it writes them - a sigma.N per update, the simulated data rewritten at
#: every forward run, a time-lapse step starting from the last forward run.
_E4D_STAND_IN = r'''
import sys, time
from pathlib import Path
import numpy as np

def records(name):
    return [line.split() for line in Path(name).read_text().splitlines() if line.strip()]

updates, pause = int(sys.argv[1]), float(sys.argv[2])
inp = records("e4d.inp")
mode, mesh, srv, sig, out, inv = (row[0] for row in inp[:6])
cut = mesh.index(".")
assert mode == ("4" if len(inp) > 7 else "3") and mesh[cut:] == ".1.node"
trn = np.loadtxt(mesh[:cut + 1] + "trn")
node = records(mesh[:cut + 2] + ".node")
nodes = np.array([[float(v) for v in row[1:4]] for row in node[1:]])
assert {row[5] for row in node[1:]} == {"0", "1", "2"}
elements = records(mesh[:cut + 2] + ".ele")
ne = int(elements[0][0])
assert min(int(row[5]) for row in elements[1:]) >= 1
assert int(records(mesh[:cut + 2] + ".neigh")[0][0]) == ne

def survey(name, base=None):
    rows = records(name)
    n = int(rows[0][0])
    where = np.array([[float(v) for v in row[1:4]] for row in rows[1:n + 1]])
    gap = np.linalg.norm(nodes[None] - (where - trn)[:, None], axis=2).min(axis=1)
    assert gap.max() < 1e-3                      # every electrode on a node
    m = int(rows[n + 1][0])
    meas = rows[n + 2:n + 2 + m]
    abmn = np.array([[int(v) for v in row[1:5]] for row in meas])
    assert abmn.min() >= 0 and abmn.max() <= n and len(meas) == m
    if base is not None:                         # get_dobs_tl: same ABMN, in order
        assert np.array_equal(abmn, base)
    data = np.array([[float(v) for v in row[5:7]] for row in meas])
    assert np.all(data[:, 1] > 0)
    return abmn, data[:, 0], data[:, 1]

abmn, dobs, sd = survey(srv)
start = records(sig)
assert int(start[0][0]) == ne
sigma = np.array([float(row[0]) for row in start[1:ne + 1]])
dpd = records(out)[1][0]
options = records(inv)
blocks, line = int(options[0][0]), 1
for _ in range(blocks):
    expected = ("8", "pref") if mode == "4" else ("2", "0.0")
    assert (options[line + 1][0], options[line + 4][0]) == expected
    line += 6
assert float(options[line + 1][0]) > 0 and options[line + 4][0] == "3"
steps = []
if mode == "4":
    assert inp[7][1] == "2"
    listing = records(inp[7][0])
    steps = [(row[0], float(row[1])) for row in listing[1:int(listing[0][0]) + 1]]
log = open("e4d.log", "w")

def report(header, chi2):
    log.write(f"\n {header}\n Chi2 is currently {chi2:10.4g}   Target value is   1.000\n"
              " ****\n")
    log.flush()

def forward(observed, factor):
    predicted = observed * factor
    with open(dpd, "w") as f:
        f.write(f" {len(observed)}\n" + "".join(
            f"{i + 1:8d}{a:8d}{b:8d}{m:8d}{n:8d}{o:15.6g}{p:15.6g}\n"
            for i, ((a, b, m, n), o, p) in enumerate(zip(abmn, observed, predicted))))
    return float(np.mean(((observed - predicted) / sd) ** 2))

def invert(observed, sigma, baseline):
    # A time-lapse step starts from the previous forward run: no new data file.
    chi2 = forward(observed, 1.3) if baseline else 9.0
    report("*** CONVERGENCE STATISTICS AT STARTING MODEL ***", chi2)
    for it in range(1, updates + 1):
        time.sleep(pause)                        # the Jacobian takes a while
        sigma = sigma * 1.1
        chi2 = forward(observed, 1.0 + 0.3 / (it + 1))
        if baseline:
            Path(f"sigma.{it}").write_text(f" {ne} 1 {chi2}\n"
                                           + "".join(f" {v}\n" for v in sigma))
        report(f"*** CONVERGENCE STATISTICS AFTER INVERSE UPDATE # {it:03d} ***", chi2)
    return sigma, chi2

sigma, chi2 = invert(dobs, sigma, True)
for name, t in steps:
    _, observed, sd = survey(name, abmn)
    sigma, chi2 = invert(observed, sigma, False)
    Path(f"tl_sig{t:8.3f}".replace(" ", "")).write_text(
        f" {ne} 1 {chi2}\n" + "".join(f" {v}\n" for v in sigma))
'''


def _tetrahedral_box(nx, ny, nz):
    """A box of unit cubes, each cut into six tetrahedra along one diagonal."""
    import pygimli as pg

    grid = np.stack(np.meshgrid(np.arange(nx + 1.0), np.arange(ny + 1.0),
                                -np.arange(nz + 1.0), indexing="ij"), -1)
    index = np.arange(grid[..., 0].size).reshape(grid.shape[:3])
    mesh = pg.Mesh(3)
    for point in grid.reshape(-1, 3):
        mesh.createNode(pg.Pos(*point))
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                v = {(a, b, c): int(index[i + a, j + b, k + c])
                     for a in (0, 1) for b in (0, 1) for c in (0, 1)}
                inner = abs(i + 0.5 - nx / 2) < nx / 2 - 1 and k == 0
                for path in ((1, 0, 0), (1, 1, 0)), ((1, 1, 0), (0, 1, 0)), \
                        ((0, 1, 0), (0, 1, 1)), ((0, 1, 1), (0, 0, 1)), \
                        ((0, 0, 1), (1, 0, 1)), ((1, 0, 1), (1, 0, 0)):
                    mesh.createCell([v[0, 0, 0], v[path[0]], v[path[1]], v[1, 1, 1]],
                                    marker=2 if inner else 1)
    mesh.createNeighborInfos()
    return mesh


def test_the_e4d_engine_writes_what_e4d_reads_and_reads_back_what_it_writes(tmp_path):
    """E4D is an external program, so a stand-in that reads its inputs the way
    E4D does checks the files; the engine must stop E4D at the iteration limit
    (E4D has none of its own) and keep each time-lapse survey's simulated
    data, which E4D overwrites at the next survey's first forward run."""
    pg = pytest.importorskip("pygimli")
    import sys

    from PyHydroGeophysX.inversion import e4d

    mesh = _tetrahedral_box(6, 2, 2)
    data = pg.DataContainerERT()
    for x in range(1, 6):
        data.createSensor(pg.Pos(float(x), 1.0, 0.0))
    for row, (a, b, m, n) in enumerate([(0, 1, 2, 3), (1, 2, 3, 4), (0, -1, 2, 3)]):
        data.createFourPointData(row, a, b, m, n)
    data["k"] = np.array([10.0, 10.0, -20.0])
    data["rhoa"] = np.array([80.0, 100.0, 120.0])
    data["err"] = np.full(3, 0.05)
    stand_in = tmp_path / "e4d_stand_in.py"
    stand_in.write_text(_E4D_STAND_IN)

    def options(updates, pause):
        return {"command": f'"{sys.executable}" "{stand_in}" {updates} {pause}',
                "workdir": str(tmp_path / f"e4d_{updates}")}

    engine = e4d.E4DEngine(data, mesh, options=options(5, 1.0))
    run = engine.fit(lam=10.0, max_iterations=2, plateau_tolerance=0.01, target_chi2=1.0)
    folder = Path(run.metrics["e4d_run"])
    survey = e4d.read_e4d_survey(folder / e4d.SURVEY_FILE)
    assert survey["abmn"].tolist() == [[1, 2, 3, 4], [2, 3, 4, 5], [1, 0, 3, 4]]
    assert np.allclose(survey["resistance"], [8.0, 10.0, -6.0])
    assert np.allclose(survey["std"], [0.4, 0.5, 0.3])
    assert (run.stop, run.iterations) == ("iteration_cap", 2)
    assert np.allclose(run.model, 100.0 / 1.1 ** 2)          # sigma.2, on the marked cells
    assert len(run.model) == int(np.sum(np.asarray(mesh.cellMarkers()) > 1))
    assert np.allclose(run.response, [80.0, 100.0, 120.0] * np.array([1.1, 1.1, 1.1]), rtol=1e-5)
    assert np.isclose(run.chi2, np.mean((0.1 * np.array([8.0, 10.0, 6.0])
                                         / np.array([0.4, 0.5, 0.3])) ** 2), rtol=1e-4)
    assert len(run.convergence) == 3                          # start and two updates

    later = pg.DataContainerERT(data)
    later["rhoa"] = np.asarray(data["rhoa"]) * 1.2
    missing = pg.DataContainerERT(data)
    missing.markInvalid(pg.core.BVector(np.array([False, True, False])))
    missing.removeInvalid()                                   # one survey lost a reading
    series = e4d.invert_e4d_time_lapse([data, missing, later], mesh, lam=10.0,
                                       options=options(1, 1.5))
    assert series.final_models.shape == (len(run.model), 3)
    assert np.allclose(series.final_models[:, -1], 100.0 / 1.1 ** 3)
    assert series.meta["e4d"]["measurements"] == 2
    assert series.meta["e4d"]["missing_fit"] == []
    assert np.all(np.isfinite(series.iteration_chi2))
    assert np.allclose(series.responses[2], np.asarray(later["rhoa"])[[0, 2]] * 1.15, rtol=1e-5)


_R2_OUT = """
 Processing dataset   1
   Iteration   1
     Initial RMS Misfit:        12.10       Number of data ignored:     0
     Alpha:         968.573   RMS Misfit:        4.72  Roughness:       14.393
     Alpha:         449.572   RMS Misfit:        3.68  Roughness:       15.153
     Final RMS Misfit:        3.68
   Iteration   2
     Initial RMS Misfit:         3.68       Number of data ignored:     0
     Alpha:         208.673   RMS Misfit:        1.00  Roughness:       25.289
     Final RMS Misfit:        1.00
 Solution converged - Outputing results to file
 Processing dataset   2
   Iteration   1
     Initial RMS Misfit:         2.00       Number of data ignored:     0
     Alpha:          96.857   RMS Misfit:        1.50  Roughness:       37.220
     Final RMS Misfit:        1.50
 WARNING: Solution not converged in   1 iterations
"""


def test_r2_and_r3t_hold_zones_land_on_their_cells_and_invert_series_and_3d(
        synthetic_series, tmp_path):
    """R2 and R3t are Binley's external programs, which ResIPy carries. Their
    log is read for the misfit, the weight they chose and why they stopped.
    Where R2 can run, a fixed zone must stay at its value, the model must land
    on the cells R2 reports it for, the prediction handed back must be R2's
    own, lambda must not be reported as if it had been used, and a series -
    R2's difference inversion - must see the shallow layer grow more resistive
    (50, 60, 75 ohm-m over 400). Every apparent resistivity of that short array
    rises by a third or more, so the smoothest change R2 finds fades with depth
    rather than stopping at the layer. Where R3t can run, data it predicts for
    a conductive block on a 3-D mesh must invert back to that block."""
    from PyHydroGeophysX.inversion import r2

    (tmp_path / "R2.out").write_text(_R2_OUT)
    datasets = r2.read_r2_out(tmp_path / "R2.out")["datasets"]
    assert [d["rms"] for d in datasets] == [[12.1, 3.68, 1.0], [2.0, 1.5]]
    assert [d["stop"] for d in datasets] == ["target", "iteration_cap"]
    assert datasets[0]["alpha"] == [449.572, 208.673]

    pytest.importorskip("pygimli")
    launcher = r2.find_r2("r2")
    if not launcher.runs:
        pytest.skip(f"R2 cannot run here: {launcher.reason}")
    from pygimli.physics import ert

    from PyHydroGeophysX.inversion.ert_inversion import run_ert_manager_inversion
    from PyHydroGeophysX.inversion.ert_zones import zone_prior

    files, mesh = synthetic_series
    result = run_ert_manager_inversion(
        files[0], tmp_path / "single", engine="r2", zones=[BLOCK, TOP], auto_lambda=True,
        max_iterations=10, mesh_quality=33, para_depth=8)
    assert result["auto_lambda_status"] == "not_applicable" and result["zones_fixed_held"]
    assert "lambda" not in result["metrics"]
    assert np.isfinite(result["metrics"]["smoothing_alpha"])
    manager = result["mgr"]
    prior = zone_prior(manager.paraDomain, [BLOCK, TOP])
    assert prior.fixed.sum() > 0 and np.allclose(manager.model[prior.fixed], 400.0)
    assert not np.allclose(manager.model[prior.in_zone & ~prior.fixed], 50.0)

    folder = Path(result["r2"]["run_dir"])
    written = np.loadtxt(folder / "f001_res.dat")          # x, z, rho, log10 rho
    order = np.load(folder / "order.npy")
    row = np.empty(len(order), dtype=int)
    row[order] = np.arange(len(order))
    mine = written[row[np.load(folder / "cell_elements.npy")]]
    centres = np.asarray(manager.paraDomain.cellCenters())[:, :2]
    assert np.ptp(centres[:, 0] - mine[:, 0]) < 5e-3     # one shift, to the array's centre
    assert np.allclose(centres[:, 1], mine[:, 1], atol=5e-3)
    assert np.allclose(manager.model, mine[:, 2])
    data = np.load(folder / "data_001.npz")
    errors = r2.read_r2_errors(folder / "f001_err.dat")
    assert np.allclose(np.asarray(manager.response) / data["k"] / data["fitted"],
                       errors["calculated"] / errors["observed"], rtol=1e-4)

    series = r2.invert_r2_time_lapse([ert.load(f) for f in files], mesh, program="r2",
                                     max_iterations=10,
                                     options={"workdir": str(tmp_path / "series")})
    assert series.final_models.shape[1] == 3
    depth = np.asarray(series.mesh.cellCenters())[:, 1]
    change = series.final_models[:, 2] / series.final_models[:, 0]
    shallow, deep = np.median(change[depth > -2.0]), np.median(change[depth < -6.0])
    assert 1.3 < shallow < 1.7 and shallow > deep + 0.1

    # R3t on a 3-D mesh: data R3t itself predicts for a conductive block are
    # inverted back to it, on the marked cells of the mesh.
    three_d = r2.find_r2("r3t")
    if not three_d.runs:
        return
    pg = pytest.importorskip("pygimli")
    box = _tetrahedral_box(10, 4, 3)
    survey = pg.DataContainerERT()
    places = [(x, y) for y in (1, 2, 3) for x in range(1, 10)]
    for x, y in places:
        survey.createSensor(pg.Pos(float(x), float(y), 0.0))
    lines = [[places.index((x, y)) for x in range(1, 10)] for y in (1, 2, 3)]
    quads = [(line[a], line[a + 1], line[a + 1 + s], line[a + 2 + s])
             for line in lines for s in (1, 2, 3) for a in range(len(line) - 2 - s)]
    for row, quad in enumerate(quads):
        survey.createFourPointData(row, *quad)
    survey["k"] = np.asarray(pg.physics.ert.geometricFactors(survey), dtype=float)
    survey["rhoa"] = np.full(len(quads), 100.0)
    survey["err"] = np.full(len(quads), 0.03)
    truth = r2.R2Engine(survey, box, program="r3t", options={"workdir": str(tmp_path / "truth")})
    centres = np.asarray(box.cellCenters())
    block = (np.abs(centres[:, 0] - 5.0) < 1.5) & (centres[:, 2] > -1.0)
    folder = truth._prepare([{"abmn": truth._abmn, "data": truth._resistance, "std": truth._std,
                              "fitted": truth._resistance, "k": truth._k}],
                            start=np.where(block, 20.0, 100.0), tolerance=1.0, max_iterations=1)
    (folder / "f001_res.dat").write_bytes((folder / r2.START_FILE).read_bytes())
    survey["r"] = r2.forward_r2(folder, three_d)
    survey["rhoa"] = np.asarray(survey["r"]) * np.asarray(survey["k"])
    fit = r2.R2Engine(survey, box, program="r3t", options={"workdir": str(tmp_path / "r3t")}).fit(
        lam=0.0, max_iterations=6, plateau_tolerance=0.0, target_chi2=1.0)
    inside = (np.abs(np.asarray(fit.mesh.cellCenters())[:, 0] - 5.0) < 1.0)
    assert len(fit.model) == int(np.sum(np.asarray(box.cellMarkers()) > 1))
    assert np.median(fit.model[inside]) < 0.8 * np.median(fit.model[~inside])
    assert np.allclose(fit.response, np.asarray(survey["rhoa"]), rtol=0.1)


# --------------------------------------------------------------------------
# Magnetotellurics
# --------------------------------------------------------------------------

_MT_RHO, _MT_THICKNESS = np.array([100.0, 10.0, 1000.0]), np.array([500.0, 2000.0])


def _mt_run(rng, sample_rate=64.0, n=2 ** 16, noise=0.03):
    """Natural-field-like H and the E a layered earth makes of it, in field units."""
    from PyHydroGeophysX.data_processing import mt

    def field():
        walk = np.cumsum(rng.standard_normal(n)) * 0.02 + rng.standard_normal(n)
        return walk - walk.mean()

    hx, hy = field(), field()
    f = np.fft.rfftfreq(n, 1.0 / sample_rate)
    z = np.zeros(f.size, complex)
    z[1:] = mt.impedance_1d(_MT_RHO, _MT_THICKNESS, f[1:]) / mt.FIELD_TO_OHM
    ex = np.fft.irfft(z * np.fft.rfft(hy), n=n)
    ey = np.fft.irfft(-z * np.fft.rfft(hx), n=n)
    ex += noise * ex.std() * rng.standard_normal(n)
    ey += noise * ey.std() * rng.standard_normal(n)
    channels = [mt.Channel("ex", ex, "mV/km"), mt.Channel("ey", ey, "mV/km"),
                mt.Channel("hx", hx, "nT"), mt.Channel("hy", hy, "nT")]
    return mt.TimeSeriesRun(channels, sample_rate, "2024-01-01T00:00:00", station="T1",
                            latitude=41.66, longitude=-91.53)


def test_mt_processing_recovers_a_layered_impedance_with_honest_errors(tmp_path):
    """The impedance, its sign convention and its error bars; and EDI / EMTF XML keep them."""
    from PyHydroGeophysX.data_processing import mt

    tf = mt.process_mt(_mt_run(np.random.default_rng(3)))
    true = mt.impedance_1d(_MT_RHO, _MT_THICKNESS, tf.frequency)
    for (i, j), sign in (((0, 1), 1.0), ((1, 0), -1.0)):
        miss = np.abs(tf.z[:, i, j] - sign * true)
        assert np.median(miss / np.abs(true)) < 0.04
        # Errors neither hide the scatter nor swamp it.
        assert 0.5 < np.median(miss / tf.z_err[:, i, j]) < 1.6
    for name, write in (("site.edi", mt.write_edi), ("site.xml", mt.write_emtf_xml)):
        back = mt.read_transfer_function(write(tf, tmp_path / name))
        assert np.allclose(back.frequency, tf.frequency, rtol=1e-6)
        assert np.allclose(back.z, tf.z, rtol=1e-4, atol=1e-12 * np.abs(tf.z).max())
        assert np.allclose(back.z_err, tf.z_err, rtol=1e-3)


def test_mt_1d_sensitivity_differentiates_the_forward_model():
    from PyHydroGeophysX.data_processing.mt import forward1d

    f = np.logspace(-3, 3, 13)
    z, d_rho, d_phase = forward1d.sensitivity_1d(_MT_RHO, _MT_THICKNESS, f)
    for k in range(_MT_RHO.size):
        step = _MT_RHO.copy()
        step[k] *= np.exp(1e-6)
        z2 = forward1d.impedance_1d(step, _MT_THICKNESS, f)
        assert np.allclose(d_rho[:, k], 2 * np.log10(np.abs(z2 / z)) / 1e-6, atol=1e-5)
        assert np.allclose(d_phase[:, k], np.angle(z2 / z) / 1e-6, atol=1e-5)


def test_a_tem_sounding_fixes_the_mt_static_shift_that_mt_alone_cannot():
    pytest.importorskip("simpeg")
    from PyHydroGeophysX.data_processing import mt
    from PyHydroGeophysX.inversion.em1d import DEFAULT_INVERSION, build_sounding_block

    f = np.logspace(-1, 3, 25)
    z = mt.impedance_1d(_MT_RHO, _MT_THICKNESS, f)
    Z = np.zeros((f.size, 2, 2), complex)
    Z[:, 0, 1], Z[:, 1, 0] = z, -z
    tf = mt.TransferFunction(frequency=f, z=Z, z_err=0.02 * np.abs(Z) + 1e-30, station="T")
    shifted = mt.apply_static_shift(tf, 1 / 2.0, 1 / 2.0)    # rho_a up by a factor 2
    times = np.logspace(-5.5, -3.0, 20)
    geometry = {"height": 0.0, "source_radius": 20.0}
    block = build_sounding_block({"times": times, "response": np.ones_like(times)}, geometry,
                                 {**DEFAULT_INVERSION, "n_layers": 3, "layer_thicknesses": _MT_THICKNESS},
                                 "TDEM")
    tem = {"data": {"times": times, "response": block.forward(1.0 / _MT_RHO)},
           "geometry": geometry, "inversion": {"rel_error": 0.03}}
    alone = mt.occam1d(shifted, mode="both", static_shift=True, error_floor=0.02, n_layers=30)
    joint = mt.occam1d(shifted, mode="both", static_shift=True, error_floor=0.02, n_layers=30, tem=tem)
    assert all(abs(np.log(v / 2.0)) > abs(np.log(1.5)) for v in alone.static_shift.values())
    assert all(abs(np.log(v / 2.0)) < np.log(1.15) for v in joint.static_shift.values())


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


def test_the_shared_picker_keeps_the_origin_trace_and_times_traces_on_a_dc_level():
    """pick_and_correct, as the seismic workflow and the studio page use it.

    A shot over the geophone at x = 0 is a real position, not missing
    coordinates (it was moved to x = 3 and 1, a 1.3 ms time between them, and
    took one line's fit from chi-squared 9 to 14); and a trace that starts on a
    DC level, where the threshold picker stops at time zero, is picked again
    along its shot's curve instead of being lost.
    """
    from PyHydroGeophysX.data_processing.seismic import (
        SeismicTraceHeader, pick_and_correct, screen_picks)

    dt, n, xs = 0.0005, 400, np.arange(12.0)
    arrival = 0.004 + 0.003 * xs
    rng = np.random.default_rng(3)
    t = np.arange(n) * dt
    traces = rng.normal(0.0, 0.01, (n, xs.size))
    for r, onset in enumerate(arrival):
        k = int(onset / dt)
        traces[k:, r] += np.sin(np.arange(n - k) * 0.6)
    traces[:, [8, 9]] += 3.0 * np.exp(-t / 0.01)[:, None]
    headers = [SeismicTraceHeader(field_record=1, trace_number=r + 1, energy_source_point=1,
                                  source_x=0.0, source_y=0.0, source_z=0.0, receiver_x=x,
                                  receiver_y=0.0, receiver_z=0.0, offset=x)
               for r, x in enumerate(xs)]

    picks, repicked = pick_and_correct(traces, dt=dt, headers=headers)
    assert (picks[0].source_x, picks[0].receiver_x) == (0.0, 0.0)
    assert {p.receiver_x for p in repicked} >= {8.0, 9.0}
    screen = screen_picks(picks)
    assert not screen.rejected and not screen.dropped
    away = [p for p in screen.kept if p.receiver_x > 0]
    assert np.allclose([p.time_s for p in away], arrival[1:], atol=2 * dt)


def test_the_neighbour_check_corrects_the_stray_pick_and_not_its_neighbour():
    """An off-end shot's far pick 10.5 ms early, inside its own shot's tolerance.

    At the geophone it and the next shot's pick disagree by the same amount
    either way: the curve finds the conflict, not which pick is wrong, and
    taken by curve alone the next shot's right pick was the one changed (as
    on a field line). The shot gathers decide it.
    """
    from PyHydroGeophysX.data_processing.seismic import FirstBreakPick, neighbour_shot_check

    dt, n = 0.00025, 400
    rng = np.random.default_rng(5)
    picks, columns = [], []
    for s in range(-4, 13, 2):
        for r in range(12):
            if s == r:
                continue
            offset = abs(s - r)
            true = min(offset / 300, 0.008 + offset / 500)
            k = int(round(true / dt))
            trace = rng.normal(0.0, 0.01, n)
            trace[k:] += np.sin(np.arange(n - k) * 0.6)
            columns.append(trace)
            picks.append(FirstBreakPick(1, r + 1, true - (0.0105 if (s, r) == (-4, 9) else 0.0),
                                        float(s), 0.0, float(r), 0.0, s + 10, r + 1,
                                        len(columns) - 1, 1.0))
    kept, rejected, repicked = neighbour_shot_check(picks, np.column_stack(columns), dt)
    assert [(p.source_x, p.receiver_x) for p in repicked] == [(-4.0, 9.0)]
    assert not rejected
    time = {(p.source_x, p.receiver_x): p.time_s for p in kept}
    assert abs(time[(-4.0, 9.0)] - 0.034) <= 2 * dt
    assert time[(-2.0, 9.0)] == pytest.approx(0.030)


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


def test_adtlert_forward_releases_its_terrain_setup_without_changing_a_solve(monkeypatch):
    # The release reaches into ADTLERT's private state, so an ADTLERT that
    # changes it must fail here rather than in a user's run.
    pytest.importorskip("pygimli")
    from pygimli.physics import ert

    from PyHydroGeophysX.inversion import ert_inversion
    from PyHydroGeophysX.inversion.ert_mesh import build_inversion_mesh

    ert_inversion._enable_adtlert_float64()  # before adtlert is first imported
    pytest.importorskip("adtlert")
    x = np.linspace(0., 23., 24)
    data = ert.createData(elecs=np.column_stack([x, .6 * np.sin(x / 4.)]), schemeName="dd")
    data["k"], data["rhoa"] = ert.createGeometricFactors(data), np.full(data.size(), 100.)
    mesh = build_inversion_mesh(data, mesh_quality=33, para_depth=8)
    built = {}
    for release in (False, True):
        with monkeypatch.context() as patch:
            if not release:
                patch.setattr(ert_inversion, "_release_adtlert_terrain_setup", lambda forward: False)
            built[release], _, active, _ = ert_inversion._build_adtlert_forward(data, mesh)
    inner = built[True].forward_operator
    assert inner.primary_potential_discretization is None
    assert inner.geometric_auxiliary_discretization is None
    model = np.log(100.) + np.linspace(-.5, .5, active.size)
    for actual, expected in zip(built[True].forward_and_jacobian(model, log_transform=True),
                                built[False].forward_and_jacobian(model, log_transform=True)):
        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-12 * np.abs(expected).max())


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
# Hydrological model outputs: ATS and PFLOTRAN (HDF5)
# --------------------------------------------------------------------------

def test_ats_output_reads_cycles_volumetric_water_content_and_its_own_mesh(tmp_path):
    """ATS names the subsurface's variables "domain-<name>" and writes cycles
    in any order; water content is porosity times saturation (ATS's own
    water_content is extensive); the cell centres come from the mesh file ATS
    writes beside the data, in the data's cell order, and map the state onto a
    geophysical section."""
    h5py = pytest.importorskip("h5py")
    from PyHydroGeophysX import ATSPorosity, ATSSaturation, ATSWaterContent
    from PyHydroGeophysX.petrophysics.resistivity_models import water_content_to_resistivity

    # Four cells along x in two layers, saturation rising along x.
    saturation = np.tile(0.2 + 0.1 * np.arange(4), 2)[:, None]
    with h5py.File(tmp_path / "ats_vis_data.h5", "w") as handle:
        handle.attrs["time unit"] = "d"
        for cycle, time in ((10, 2.0), (2, 1.0)):
            for name, values in (("saturation_liquid", saturation),
                                 ("porosity", np.full((8, 1), 0.4))):
                handle.create_dataset(f"domain-{name}.cell.0/{cycle}",
                                      data=values).attrs["Time"] = time
        handle.create_dataset("domain-water_content.cell.0/2", data=np.full((8, 1), 1000.0))

    def node(x, y, level):
        return x + 5 * y + 10 * level

    nodes = [(x, y, -level) for level in range(3) for y in range(2) for x in range(5)]
    hexes = [[9] + [node(i + dx, dy, level + dz) for dz in (0, 1)
                    for dx, dy in ((0, 0), (1, 0), (1, 1), (0, 1))]
             for level in range(2) for i in range(4)]
    with h5py.File(tmp_path / "ats_vis_mesh.h5", "w") as handle:   # a fixed mesh, written once
        handle.create_dataset("2/Mesh/Nodes", data=np.asarray(nodes, dtype=float))
        handle.create_dataset("2/Mesh/MixedElements", data=np.asarray(hexes).reshape(-1, 1))

    reader = ATSSaturation(tmp_path)
    assert reader.get_timestep_info() == [(2, 1.0), (10, 2.0)] and reader.time_unit == "d"
    np.testing.assert_allclose(reader.load_timestep(0), saturation[:, 0])
    assert reader.load_time_range(1, 1).shape == (0, 8)
    np.testing.assert_allclose(reader.load_time_range(0, -1), reader.load_time_range()[:1])
    water = ATSWaterContent(tmp_path)
    np.testing.assert_allclose(water.load_timestep(1), 0.4 * saturation[:, 0])
    np.testing.assert_allclose(ATSPorosity(tmp_path).load_timestep(0), 0.4)
    with pytest.raises(ValueError, match="Porosity shape"):
        water.get_water_content(0, porosity=np.ones((8, 1)))
    with pytest.raises(ValueError, match="between 0 and 1"):
        water.get_water_content(0, porosity=1.1)
    np.testing.assert_allclose(reader.output_cell_centers(1),
                               [[i + 0.5, 0.5, -level - 0.5] for level in range(2) for i in range(4)])
    # On a section in x and z: half way between the second and third cells.
    theta = water.interpolate_timestep(0, np.array([[2.0, -1.0], [9.0, -1.0]]), axes="xz")
    np.testing.assert_allclose(theta[0], 0.4 * 0.35)
    assert np.isnan(theta[1])                                # outside the model
    assert np.isfinite(water_content_to_resistivity(theta[:1], rhos=100.0, n=2.0, porosity=0.4)).all()

    # A file under another name: no mesh beside it, names that must be chosen.
    with h5py.File(tmp_path / "custom_data.h5", "w") as handle:
        handle.create_dataset("saturation_liquid/0", data=np.ones((4, 1)))
        handle.create_dataset("domain-saturation_liquid/0", data=np.ones((4, 1)))
        handle.create_dataset("vector/0", data=np.ones((4, 3)))
    with pytest.raises(KeyError, match="uniquely"):
        ATSSaturation(tmp_path, "custom_data.h5")
    custom = ATSSaturation(tmp_path, "custom_data.h5",
                           variable_map={"saturation": "domain-saturation_liquid"})
    assert custom.time_unit is None and np.isnan(custom.times[0])
    with pytest.raises(ValueError, match="scalar cell"):
        custom.read_field("vector", 0)
    with pytest.raises(FileNotFoundError, match="custom_mesh.h5"):
        custom.output_cell_centers(0)


def test_pflotran_output_keeps_runs_apart_and_finds_its_cell_centres(tmp_path):
    """PFLOTRAN's numbered files make one run, in time order, and a run whose
    name merely starts with it is not joined in. A structured grid's fields are
    (nx, ny, nz) and its Coordinates hold the cell edges; an unstructured
    grid's Domain holds XDMF cells. Either way the centres line up with the
    values."""
    h5py = pytest.importorskip("h5py")
    from PyHydroGeophysX import PFLOTRANPorosity, PFLOTRANSaturation, PFLOTRANWaterContent

    edges = {"X": [0.0, 1.0, 3.0, 6.0], "Y": [0.0, 1.0], "Z": [-2.0, -1.0, 0.0]}
    x, _, z = np.meshgrid([0.5, 2.0, 4.5], [0.5], [-1.5, -0.5], indexing="ij")   # (nx, ny, nz)

    def write(path, times, unit="y"):
        with h5py.File(path, "w") as handle:
            for axis, values in edges.items():
                handle.create_dataset(f"Coordinates/{axis} [m]", data=values)
            for time in times:
                group = handle.create_group(f"Time:  {time:.5E} {unit}")
                group.create_dataset("Liquid_Saturation", data=x / 10 - z / 100 + time / 1000)
                group.create_dataset("Porosity", data=np.full(x.shape, 0.4))

    write(tmp_path / "run-001.h5", [10.0])
    write(tmp_path / "run-002.h5", [2.0, 0.0])
    write(tmp_path / "run-hires.h5", [0.5])
    reader = PFLOTRANSaturation(tmp_path, "run")
    assert reader.times.tolist() == [0.0, 2.0, 10.0] and reader.time_unit == "y"
    assert reader.load_time_range().shape == (3, 3, 1, 2)
    np.testing.assert_allclose(PFLOTRANWaterContent(tmp_path, "run").load_timestep(2),
                               0.4 * (x / 10 - z / 100 + 0.01))
    np.testing.assert_allclose(PFLOTRANPorosity(tmp_path, "run").load_timestep(0), 0.4)
    assert PFLOTRANSaturation(tmp_path, filename="run-hires.h5").times.tolist() == [0.5]
    centres = reader.output_cell_centers(0)
    np.testing.assert_allclose(reader.load_timestep(0).ravel(),
                               centres[:, 0] / 10 - centres[:, 2] / 100)
    np.testing.assert_allclose(reader.interpolate_timestep(0, np.array([[1.25, -1.0]]), axes="xz"),
                               [0.125 + 0.01])

    write(tmp_path / "dup-1.h5", [0.0])
    write(tmp_path / "dup-2.h5", [0.0])
    with pytest.raises(ValueError, match="Duplicate"):
        PFLOTRANSaturation(tmp_path, "dup")
    write(tmp_path / "dup-2.h5", [2.0], unit="d")
    with pytest.raises(ValueError, match="Mixed"):
        PFLOTRANSaturation(tmp_path, "dup")

    # An unstructured grid of a tetrahedron and a wedge.
    vertices = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1),
                (2, 0, 0), (3, 0, 0), (2, 1, 0), (2, 0, 1), (3, 0, 1), (2, 1, 1)]
    with h5py.File(tmp_path / "unstructured.h5", "w") as handle:
        handle.create_dataset("Domain/Vertices", data=np.asarray(vertices, dtype=float))
        handle.create_dataset("Domain/Cells", data=[6, 0, 1, 2, 3, 8, 4, 5, 6, 7, 8, 9])
        handle.create_group("Time:  0.00000E+00 d").create_dataset("Liquid_Saturation", data=[0.3, 0.6])
    np.testing.assert_allclose(
        PFLOTRANSaturation(tmp_path, filename="unstructured.h5").output_cell_centers(),
        [[0.25, 0.25, 0.25], [7 / 3, 1 / 3, 0.5]])


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
