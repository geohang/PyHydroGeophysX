from pathlib import Path

import numpy as np
import pytest


@pytest.mark.parametrize('pairing', ['none', 'complete', 'mixed'])
def test_default_export_accepts_optional_reciprocals(tmp_path, pairing):
    pg = pytest.importorskip('pygimli')
    pytest.importorskip('resipy')
    from PyHydroGeophysX.data_processing.ert_data_agent import load_ert_resipy, export_for_inversion
    lines = ['6', '# x y z'] + [f'{i} 0 0' for i in range(6)]
    rows = ['1 2 3 4 100 0.03']
    if pairing != 'none':
        rows.append('3 4 1 2 100 0.03')
    if pairing == 'mixed':
        rows.append('2 3 5 6 100 0.03')
    source = tmp_path / 'survey.ohm'
    source.write_text('\n'.join(lines + [str(len(rows)), '# a b m n rhoa err'] + rows) + '\n')
    electrodes = tmp_path / 'electrodes.dat'
    np.savetxt(electrodes, [[i, 0, 0] for i in range(6)])
    survey = load_ert_resipy(str(tmp_path / 'project'), str(source), 'BERT', electrode_file=str(electrodes))
    output = export_for_inversion(survey, tmp_path / 'export', use_source_error=True)
    result = pg.DataContainerERT(str(output))
    assert result.size() > 0
    if pairing == 'mixed':
        # Unpaired quadrupole must not disappear just because another has a reciprocal.
        quads = set(zip(np.asarray(result['a']), np.asarray(result['b']), np.asarray(result['m']), np.asarray(result['n'])))
        assert (1, 2, 4, 5) in quads
    assert np.all(np.isfinite(result['err'])) and np.all(np.asarray(result['err']) > 0)
    np.testing.assert_allclose(np.asarray(result['r']) * np.asarray(result['k']), result['rhoa'])


def test_das_example_retains_unpaired_and_exports_external_electrodes(tmp_path):
    pytest.importorskip('resipy')
    pg = pytest.importorskip('pygimli')
    from PyHydroGeophysX.data_processing.ert_data_agent import load_ert_resipy, export_for_inversion
    data = Path(__file__).resolve().parents[1] / 'examples/data/ERT/DAS'
    if not (data / '20171105_1418.Data').exists():
        pytest.skip('Example data not installed')
    survey = load_ert_resipy(str(tmp_path / 'project'), str(data / '20171105_1418.Data'),
                            'DAS-1', electrode_file=str(data / 'electrodes.dat'))
    assert len(survey.observations) == 945  # 15 unpaired readings are not failed QC.
    output = export_for_inversion(survey, tmp_path / 'export', use_source_error=True)
    assert pg.DataContainerERT(str(output)).size() > 0
