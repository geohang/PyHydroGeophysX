"""Physical meaning and numerical stability of plan-view (map) gridding."""
import numpy as np
import pytest

from PyHydroGeophysX.core.plan_interpolation import (
    METHODS, VARIOGRAM_MODELS, auto_variogram, empirical_variogram, fit_variogram,
    inverse_distance, ordinary_kriging, plan_grid, prepare_samples, spread_ratio,
    variogram_function, write_plan_grid,
)


def tem_survey(lines=8, stations=40, line_spacing=120.0, station_spacing=25.0, seed=3):
    """A TEM-like layout: several lines, dense along-line, wide across-line."""
    generator = np.random.default_rng(seed)
    x, y = np.meshgrid(np.arange(lines) * line_spacing,
                       np.arange(stations) * station_spacing, indexing='ij')
    xy = np.column_stack([x.ravel(), y.ravel()]) + generator.normal(0, 3, (x.size, 2))
    truth = 50 * 10 ** (0.6 * np.sin(xy[:, 0] / 300.) * np.cos(xy[:, 1] / 260.))
    return xy, truth * 10 ** generator.normal(0, .03, len(truth))


def sample_grid(result, xy):
    """Read a plan grid back at arbitrary map positions."""
    from scipy.interpolate import RegularGridInterpolator
    reader = RegularGridInterpolator((result['y'], result['x']), result['grid'],
                                     bounds_error=False, fill_value=np.nan)
    return reader(np.column_stack([xy[:, 1], xy[:, 0]]))


@pytest.mark.parametrize('method', METHODS)
def test_every_method_stays_inside_the_measured_range_and_honours_log_space(method):
    xy, rho = tem_survey()
    result = plan_grid(xy, rho, method=method, resolution=110, log_values=True)
    grid = result['grid']
    assert grid.shape == (result['y'].size, result['x'].size)
    assert np.isfinite(grid).any()
    # Resistivity interpolated in log space cannot come back non-positive, and no
    # method may invent a value far outside what the soundings recovered. A
    # Gaussian variogram without a nugget used to overshoot by a factor of four.
    assert np.nanmin(grid) > 0
    assert np.nanmin(grid) > rho.min() / 1.3
    assert np.nanmax(grid) < rho.max() * 1.3


def test_log_interpolation_returns_the_geometric_mean_between_two_soundings():
    xy = np.array([[0., 0.], [100., 0.], [50., 90.], [50., -90.]])
    values = np.array([10., 1000., 100., 100.])
    midpoint = np.array([[50., 0.]])
    linear = plan_grid(xy, values, method='linear', resolution=120, log_values=False)
    logged = plan_grid(xy, values, method='linear', resolution=120, log_values=True)
    assert sample_grid(linear, midpoint)[0] == pytest.approx(np.mean(values[:2]), rel=.15)
    assert sample_grid(logged, midpoint)[0] == pytest.approx(np.sqrt(10. * 1000.), rel=.15)


def test_kriging_reproduces_its_own_samples_and_knows_where_it_does_not():
    xy, rho = tem_survey(lines=6, stations=12, seed=7)
    fit = auto_variogram(xy, np.log10(rho))
    far = np.array([[-900., -900.]])
    estimate, variance = ordinary_kriging(xy, np.log10(rho), np.vstack([xy[:5], far]), fit)
    # Ordinary kriging is an exact interpolator: at a sample it returns that
    # sample and reports zero variance.
    np.testing.assert_allclose(estimate[:5], np.log10(rho[:5]), atol=1e-6)
    np.testing.assert_allclose(variance[:5], 0., atol=1e-8)
    assert variance[-1] > 1e-3
    assert variance[-1] > variance[:5].max()


def test_local_and_global_kriging_agree_so_the_size_switch_is_invisible():
    xy, rho = tem_survey(lines=6, stations=15, seed=5)
    values = np.log10(rho)
    fit = auto_variogram(xy, values)
    # Targets inside the survey: outside it every kriging flavour extrapolates,
    # and the map blanks those cells anyway.
    generator = np.random.default_rng(1)
    targets = np.column_stack([generator.uniform(50, 550, 200), generator.uniform(50, 300, 200)])
    globally, global_variance = ordinary_kriging(xy, values, targets, fit)
    locally, local_variance = ordinary_kriging(xy, values, targets, fit, neighbors=40)
    assert np.abs(globally - locally).max() < .05 * np.ptp(values)
    assert np.abs(global_variance - local_variance).max() < .1 * fit['sill']


def test_gaussian_variogram_is_given_a_nugget_so_the_kriging_system_is_solvable():
    xy, rho = tem_survey()
    values = np.log10(rho)
    experimental = empirical_variogram(xy, values)
    fit = fit_variogram(experimental['lags'], experimental['gamma'],
                        experimental['counts'], model='gaussian')
    assert fit['nugget'] > 0
    gamma = variogram_function('gaussian', fit['nugget'], fit['sill'], fit['range'])
    count = len(values)
    system = np.ones((count + 1, count + 1))
    system[:count, :count] = gamma(np.hypot(xy[:, 0, None] - xy[None, :, 0],
                                            xy[:, 1, None] - xy[None, :, 1]))
    system[count, count] = 0.
    assert np.linalg.cond(system) < 1e12
    # A fitted range beyond the pairs that were binned is not identifiable.
    assert fit['range'] <= 2.0 * experimental['lags'].max() * (1 + 1e-9)


@pytest.mark.parametrize('model', VARIOGRAM_MODELS)
def test_variogram_models_run_from_zero_to_the_sill_at_the_stated_range(model):
    gamma = variogram_function(model, nugget=.2, sill=1.2, range_=50.)
    assert gamma(0.) == 0.
    assert gamma(1e-6) == pytest.approx(.2, abs=.01)
    assert gamma(50.) == pytest.approx(1.2, rel=.06)
    assert gamma(500.) == pytest.approx(1.2, rel=.01)
    assert np.all(np.diff(gamma(np.linspace(0, 200, 50))) >= -1e-12)


def test_blanking_keeps_the_map_from_claiming_coverage_it_does_not_have():
    xy, rho = tem_survey(lines=4, line_spacing=400.)
    wide = plan_grid(xy, rho, method='idw', resolution=120, log_values=True)
    tight = plan_grid(xy, rho, method='idw', resolution=120, log_values=True, max_distance=60.)
    assert tight['coverage'] < wide['coverage']
    filled = np.column_stack(np.meshgrid(tight['x'], tight['y']))
    from scipy.spatial import cKDTree
    grid_xy = np.column_stack([c.ravel() for c in np.meshgrid(tight['x'], tight['y'])])
    distance = cKDTree(xy).query(grid_xy)[0].reshape(tight['grid'].shape)
    assert not np.isfinite(tight['grid'][distance > 60.]).any()
    assert filled.size  # meshgrid stacking used above stays consistent


def test_convex_hull_clipping_refuses_to_extrapolate_beyond_the_survey():
    xy, rho = tem_survey(lines=5, stations=8)
    clipped = plan_grid(xy, rho, method='linear', resolution=100, log_values=True)
    open_ended = plan_grid(xy, rho, method='nearest', resolution=100, log_values=True,
                           clip_to_hull=False)
    assert clipped['coverage'] < 1.0
    assert open_ended['coverage'] == pytest.approx(1.0)
    corner = np.array([[xy[:, 0].min() - 500., xy[:, 1].min() - 500.]])
    assert not np.isfinite(sample_grid(clipped, corner)).any()


def test_a_single_line_of_soundings_is_refused_instead_of_smeared():
    line = np.column_stack([np.arange(40.) * 25., np.zeros(40)])
    with pytest.raises(ValueError, match='single line'):
        plan_grid(line, np.full(40, 50.), method='kriging')
    assert spread_ratio(line) < 1e-9
    assert spread_ratio(tem_survey()[0]) > .1


def test_duplicate_positions_are_averaged_rather_than_left_singular():
    repeated = np.array([[0., 0.], [0., 0.], [100., 0.], [50., 80.], [50., -80.]])
    cleaned_xy, cleaned = prepare_samples(repeated, np.array([10., 30., 100., 60., 60.]))
    assert len(cleaned) == 4
    assert cleaned[np.argmin(np.hypot(*cleaned_xy.T))] == pytest.approx(20.)
    # Re-occupied stations are common in a repeated TEM sounding; they must not
    # leave the kriging matrix with two identical rows.
    xy, rho = tem_survey(lines=5, stations=12)
    doubled = np.vstack([xy, xy[:10]])
    values = np.concatenate([rho, rho[:10] * 1.2])
    assert len(prepare_samples(doubled, values)[1]) == len(xy)
    result = plan_grid(doubled, values, method='kriging', resolution=60, log_values=True)
    assert np.isfinite(result['grid']).any()
    assert result['n_samples'] == len(xy)


def test_non_finite_and_non_positive_samples_are_dropped_with_a_clear_reason():
    xy, rho = tem_survey(lines=4, stations=6)
    rho = rho.copy()
    rho[:5] = np.nan
    rho[5:8] = -1.
    result = plan_grid(xy, rho, method='linear', resolution=60, log_values=True)
    assert result['n_samples'] == len(rho) - 8
    with pytest.raises(ValueError, match='No positive finite values'):
        plan_grid(xy, np.full(len(rho), -1.), log_values=True)
    with pytest.raises(ValueError, match='No finite values'):
        plan_grid(xy, np.full(len(rho), np.nan))


def test_inverse_distance_takes_a_sample_value_at_the_sample_itself():
    xy = np.array([[0., 0.], [100., 0.], [50., 80.]])
    values = np.array([10., 100., 55.])
    estimate = inverse_distance(xy, values, np.vstack([xy, [[50., 20.]]]))
    np.testing.assert_allclose(estimate[:3], values)
    assert values.min() <= estimate[3] <= values.max()


def test_grid_cells_are_square_so_the_ascii_export_is_valid(tmp_path):
    xy, rho = tem_survey(lines=5, stations=30)
    result = plan_grid(xy, rho, method='nearest', resolution=64, log_values=True)
    assert np.allclose(np.diff(result['x']), result['cell_size'])
    assert np.allclose(np.diff(result['y']), result['cell_size'])
    path = write_plan_grid(result, tmp_path / 'layer.asc')
    header = dict(line.split() for line in open(path).read().splitlines()[:6])
    assert int(header['ncols']) == result['x'].size
    assert int(header['nrows']) == result['y'].size
    assert float(header['cellsize']) == pytest.approx(result['cell_size'], rel=1e-5)
    assert float(header['xllcorner']) == pytest.approx(result['x'][0] - result['cell_size'] / 2, rel=1e-5)
    body = np.loadtxt(path, skiprows=6)
    assert body.shape == result['grid'].shape
    # ESRI ASCII rows run north to south, and blanked cells become NODATA.
    np.testing.assert_allclose(np.where(np.isfinite(result['grid'][-1]), result['grid'][-1], -9999.),
                               body[0], rtol=1e-4)
    table = write_plan_grid(result, tmp_path / 'layer.csv')
    rows = np.loadtxt(table, delimiter=',', skiprows=1)
    assert len(rows) == np.isfinite(result['grid']).sum()
    with pytest.raises(ValueError, match='ESRI ASCII'):
        write_plan_grid(result, tmp_path / 'layer.tif')


def test_a_reused_variogram_fit_reproduces_the_same_grid():
    xy, rho = tem_survey(lines=5, stations=12)
    first = plan_grid(xy, rho, method='kriging', resolution=70, log_values=True)
    again = plan_grid(xy, rho, method='kriging', resolution=70, log_values=True,
                      variogram=first['variogram'])
    np.testing.assert_allclose(first['grid'], again['grid'], rtol=1e-10, equal_nan=True)
    assert first['variogram']['model'] in VARIOGRAM_MODELS


def test_the_reported_gap_is_the_blanking_distance_that_costs_nothing():
    xy, rho = tem_survey(lines=4, line_spacing=400.)
    free = plan_grid(xy, rho, method='idw', resolution=120, log_values=True)
    gap = free['gap']
    assert gap > 0
    # Blanking at the reported gap leaves the outline whole; well under it the
    # map is reduced to ribbons along the lines, which is what the caption warns
    # about rather than leaving the reader to guess a number.
    at_gap = plan_grid(xy, rho, method='idw', resolution=120, log_values=True, max_distance=gap)
    ribbons = plan_grid(xy, rho, method='idw', resolution=120, log_values=True,
                        max_distance=gap / 10)
    assert at_gap['coverage'] == pytest.approx(free['coverage'])
    assert ribbons['coverage'] < free['coverage'] / 2
    assert ribbons['gap'] == pytest.approx(gap), 'the gap describes the survey, not the setting'


def test_gridding_refuses_an_unknown_method_and_says_what_over_blanking_needed():
    xy, rho = tem_survey(lines=4, stations=6)
    with pytest.raises(ValueError, match='Unknown interpolation method'):
        plan_grid(xy, rho, method='spline')
    with pytest.raises(ValueError, match=r'blanked.*needs') as refusal:
        plan_grid(xy, rho, method='idw', max_distance=1e-6)
    assert '1e-06' in str(refusal.value)
