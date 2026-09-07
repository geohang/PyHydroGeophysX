"""Small real-data example: input integrity and optional actual solver check."""
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest


def example():
    path = Path(__file__).parents[1]/'examples/Ex_MODFLOW_geophysics_feedback.py'
    spec = importlib.util.spec_from_file_location('feedback_example',path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_structure_example_writes_consistent_models(tmp_path):
    flopy = pytest.importorskip('flopy')
    demo = example()
    summary = demo.run_example(tmp_path/'demo',write_only=True)
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
def test_structure_example_solver(tmp_path):
    pytest.importorskip('flopy')
    summary = example().run_example(tmp_path/'run',mf6=os.environ['MF6_EXE'])
    assert summary['steps'] == 30
    assert summary['max_budget_discrepancy_percent'] <= .1
    assert summary['max_head_change_m'] > 0
