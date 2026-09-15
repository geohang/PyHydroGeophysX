"""Small real-data example: input integrity and optional actual solver check."""
import json
import os
from pathlib import Path

import numpy as np
import pytest


def example(output, monkeypatch, *, mf6=None, write_only=False):
    """Run the notebook cells with test settings, without a script-only __file__."""
    path = Path(__file__).parents[1]/'examples/Ex_MODFLOW_geophysics_feedback.ipynb'
    monkeypatch.chdir(path.parent)
    notebook = json.loads(path.read_text(encoding='utf-8'))
    namespace = {}
    for index, cell in enumerate(notebook['cells']):
        if cell['cell_type'] != 'code':
            continue
        source = ''.join(cell['source'])
        exec(compile(source, f'{path.name}:cell{index}', 'exec'), namespace)
        if source.lstrip().startswith('mf6 = None'):
            namespace.update(output=output, mf6=mf6, download=False, write_only=write_only)
    return namespace['summary']


def test_structure_example_writes_consistent_models(tmp_path, monkeypatch):
    flopy = pytest.importorskip('flopy')
    summary = example(tmp_path/'demo', monkeypatch, write_only=True)
    assert summary['status'] == 'inputs_written_not_run'
    models = [flopy.mf6.MFSimulation.load(sim_ws=str(tmp_path/'demo'/name),verbosity_level=0).get_model()
              for name in ['baseline','informed']]
    assert models[0].modelgrid.shape == (3,37,31)
    assert not np.array_equal(models[0].dis.botm.array,models[1].dis.botm.array)
    np.testing.assert_array_equal(models[0].npf.k.array,models[1].npf.k.array)
    np.testing.assert_array_equal(models[0].dis.botm.array[-1],models[1].dis.botm.array[-1])
    for model in models:
        assert np.all(np.diff(np.concatenate([model.dis.top.array[None],model.dis.botm.array]),axis=0)<0)


@pytest.mark.skipif(not os.environ.get('MF6_EXE'),reason='Set MF6_EXE to run the MODFLOW integration test')
def test_structure_example_solver(tmp_path, monkeypatch):
    pytest.importorskip('flopy')
    summary = example(tmp_path/'run', monkeypatch, mf6=os.environ['MF6_EXE'])
    assert summary['steps'] == 30
    assert summary['max_budget_discrepancy_percent'] <= .1
    assert summary['max_head_change_m'] > 0
