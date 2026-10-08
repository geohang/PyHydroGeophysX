"""Reciprocal diagnostics must describe the report without changing errors."""
from copy import deepcopy

import numpy as np
import pytest

from PyHydroGeophysX.qt_apps import ert_records
from PyHydroGeophysX.qt_apps.run_records import ERROR_MODEL_FIGURE_NAME, run_documents


def _pairing(R, dR):
    return dict(available=True, readings=2 * len(R), normal=len(R), reverse=len(R),
                pairs=len(R), paired_readings=2 * len(R), unpaired_readings=0,
                scored_pairs=len(R), error_median=.01, error_p90=.02, error_max=.03,
                R=np.asarray(R), dR=np.asarray(dR))


def _qc():
    return dict(min_rhoa=.1, max_rhoa=10000., max_error=20., more_checks=False,
                drop_nonpositive=True, min_voltage=0., min_current=0., max_k=0.,
                max_contact_r=0., max_stack=0., max_reciprocal=5.)


@pytest.mark.parametrize('series', [False, True])
def test_qc_figure_uses_report_fit_and_preserves_inputs(tmp_path, monkeypatch, series):
    from matplotlib.figure import Figure
    from PyHydroGeophysX.qt_apps.artifact_renderers import select_renderer

    R = np.geomspace(.05, 2., 120)
    dR = .007 * R ** .58 * np.exp(np.random.default_rng(12).normal(0, .2, R.size))
    pairing = _pairing(R, dR)
    report = dict(pairing=pairing, readings=240, kept=200, checks=[])
    original = deepcopy(report)
    qc = _qc()
    qc_before = dict(qc)
    captured = {}
    savefig = Figure.savefig

    def inspect(figure, *args, **kwargs):
        ax = figure.axes[0]
        assert ax.get_xscale() == ax.get_yscale() == 'log'
        captured['points'] = np.array(ax.collections[0].get_offsets())
        captured['bins'] = np.array(ax.collections[1].get_offsets())
        captured['line'] = tuple(np.array(v) for v in ax.lines[0].get_data())
        assert 'binned R²' in ax.lines[0].get_label()
        assert any('not used in inversion weights' in item.get_text() for item in figure.texts)
        return savefig(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, 'savefig', inspect)
    if series:
        path = ert_records.write_series_qc_report(tmp_path, sources=['a.dat', 'b.dat'],
            acquired=['', ''], reader='BERT', qc=qc, reports=[report, report], error_model='file err')
    else:
        path = ert_records.write_single_qc_report(tmp_path, source='a.dat', acquired='',
            reader='BERT', qc=qc, report=report, error_model='file err')
    r, dr = (np.tile(R, 2), np.tile(dR, 2)) if series else (R, dR)
    fit = ert_records.fit_error_model_binned(r, dr)
    x, y = captured['line']
    np.testing.assert_allclose(y, 10 ** fit['b'] * x ** fit['m'])
    # Independently fit the means actually drawn as black squares.
    np.testing.assert_allclose(np.polyfit(np.log10(captured['bins'][:, 0]),
                                       np.log10(captured['bins'][:, 1]), 1), [fit['m'], fit['b']])
    assert len(captured['points']) == len(r)
    text = path.read_text()
    assert f"m = {fit['m']:.6f}, b = {fit['b']:.6f}" in text
    assert ERROR_MODEL_FIGURE_NAME in text
    assert (tmp_path / ERROR_MODEL_FIGURE_NAME).read_bytes().startswith(b'\x89PNG')
    artifact = next(item for item in run_documents(tmp_path) if item['format'] == 'png')
    assert select_renderer(artifact) == 'image'
    assert 'diagnostic only' in artifact['label']
    np.testing.assert_array_equal(pairing['R'], original['pairing']['R'])
    np.testing.assert_array_equal(pairing['dR'], original['pairing']['dR'])
    assert report['kept'] == original['kept'] and qc == qc_before


@pytest.mark.parametrize('R,dR,expected', [
    ([], [], False), ([1, 2], [0, np.nan], False),
    ([1, 2, 3], [.01, .02, .03], True),
    (np.ones(30), np.linspace(.01, .03, 30), True),
])
def test_sparse_or_invalid_pairs_do_not_invent_a_fit(tmp_path, monkeypatch, R, dR, expected):
    from matplotlib.figure import Figure
    savefig = Figure.savefig

    def inspect(figure, *args, **kwargs):
        assert not figure.axes[0].lines
        assert 'Fit unavailable' in figure.axes[0].texts[0].get_text()
        return savefig(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, 'savefig', inspect)
    result = ert_records.write_error_model_figure(tmp_path, [('a', _pairing(R, dR))])
    assert (result is not None) == expected


def test_fit_to_the_pairs_the_filter_kept_leaves_outliers_out(tmp_path):
    """"Pairs kept by the filter" fits the model without the pairs a check
    removed - a pair goes when either of its readings does - and the records
    state that choice and the very model the inversion applied."""
    import pandas as pd
    from PyHydroGeophysX.data_processing.ert_formats import reciprocal_errors
    from PyHydroGeophysX.qt_apps.artifact_renderers import select_renderer
    from PyHydroGeophysX.qt_apps.run_records import ERROR_PAIRS_NAME

    n = 200
    R = np.geomspace(.05, 50., n)
    dR = 10 ** -2.2 * R ** .6 * np.exp(np.random.default_rng(5).normal(0, .2, n))
    outlier = np.arange(n) % 25 == 3
    dR[outlier] = .3 * R[outlier]
    i = np.arange(n)
    frame = pd.DataFrame({'a': np.r_[np.ones(n), 3 + i], 'b': np.r_[2 * np.ones(n), 4 + i],
                          'm': np.r_[3 + i, np.ones(n)], 'n': np.r_[4 + i, 2 * np.ones(n)],
                          'resist': np.r_[R + dR / 2, R - dR / 2]}).astype({k: int for k in 'abmn'})
    scores = reciprocal_errors(frame, drop_failed=False)
    keep = scores['reciprocalErrRel'].to_numpy() <= .05     # the reciprocal-error check
    keep[7] = False                                          # another check drops one reading
    pairing = ert_records.reciprocal_pairing(scores)
    pairing['kept'] = ert_records.reciprocal_pair_kept(scores, keep)
    expected = ~outlier & (i != 7)
    np.testing.assert_array_equal(pairing['kept'], expected)

    kept = ert_records.fit_series_error_model([pairing], 'kept')
    every = ert_records.fit_series_error_model([pairing], 'all')
    reference = ert_records.fit_error_model_binned(R[expected], dR[expected])
    assert kept['m'] == pytest.approx(reference['m']) and kept['b'] == pytest.approx(reference['b'])
    # The outliers lift the all-pairs model; without them the law used to make the data returns.
    assert abs(kept['b'] + 2.2) < .05 < .2 < every['b'] - kept['b'] and abs(kept['m'] - .6) < .05
    assert kept['left_out'] == n - expected.sum() and every['left_out'] == 0

    model = dict(m=kept['m'], b=kept['b'], floor=.01)
    report = dict(pairing=pairing, readings=2 * n, kept=int(keep.sum()), checks=[])
    text = ert_records.write_single_qc_report(tmp_path, source='a.dat', acquired='', reader='BERT',
        qc=_qc(), report=report, error_model='', applied_model=model, fit_to='kept').read_text()
    assert 'Error model fitted to: pairs kept by the filter' in text
    assert f"m = {kept['m']:.6f}, b = {kept['b']:.6f}" in text and f"10^{kept['b']:.4f}" in text
    saved = ert_records.read_error_model_pairs(tmp_path / ERROR_PAIRS_NAME)
    assert (saved['fit_to'], saved['applied'], int(saved['kept'].sum())) == ('kept', True, expected.sum())
    assert ert_records.fit_pairs(saved, 'kept')['m'] == pytest.approx(kept['m'])
    documents = {item['format']: item for item in run_documents(tmp_path)}
    assert select_renderer(documents['npz']) == 'reciprocal_errors'
    assert 'used as the data errors' in documents['png']['label']


def test_plot_failure_preserves_the_qc_report(tmp_path, monkeypatch):
    def fail(*args):
        raise OSError('Figure locked')
    monkeypatch.setattr(ert_records, 'write_error_model_figure', fail)
    path = ert_records.write_series_qc_report(tmp_path, sources=[], acquired=[], reader='BERT',
        qc=_qc(), reports=[], error_model='file err')
    text = path.read_text()
    assert 'Data QC report' in text and 'Figure locked' in text
