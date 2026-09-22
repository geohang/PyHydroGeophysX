"""Which figures a report draws, and why one it was asked for went missing.

Two consecutive runs asked for water content, got the numbers, and got four
resistivity figures with no water-content figure. The cause was an indentation
mistake - the block that draws it sat inside an `except Exception` handler, so
it ran only when the other figures had failed - and the reason nobody noticed
is that nothing compared the figures produced against the figures the request
implied. These cover both halves.
"""

from pathlib import Path

import numpy as np
import pytest


SITE = {'survey_dates': ['2017-11-05', '2017-11-06', '2017-11-07'],
        'study_period': '2017-11-05 to 2017-11-07'}


def _agent():
    from PyHydroGeophysX.agents.report_agent import ReportAgent
    agent = ReportAgent.__new__(ReportAgent)
    agent.execution_log = []
    agent.name = 'ReportAgent'
    agent.api_key = None
    return agent


# ---------------------------------------------------------------------------
# The figure that went missing
# ---------------------------------------------------------------------------
def test_the_water_content_figure_is_drawn_on_a_successful_run(tmp_path):
    """The block that draws it ran only when the other figures had failed."""
    pg = pytest.importorskip('pygimli')

    mesh = pg.createGrid(x=np.linspace(0, 20, 11), y=np.linspace(-5, 0, 4))
    cells = mesh.cellCount()
    results = {
        'mesh': mesh,
        'n_timesteps': 3,
        'final_models': np.column_stack([np.full(cells, 500.0)] * 3),
        'chi2_values': [[1.4, 0.0, 0.0]],
        'time_lapse_water_content': [
            {'water_content_mean': np.full(cells, 0.07 + 0.005 * i),
             'water_content_std': np.full(cells, 0.01)} for i in range(3)],
    }
    files = _agent()._generate_timelapse_visualizations(
        results, None, None, str(tmp_path), SITE)
    assert 'timelapse_water_content' in files, 'the requested product has no figure'
    assert Path(files['timelapse_water_content']).stat().st_size > 0


def test_water_content_without_a_mesh_is_reported_rather_than_skipped(tmp_path, capsys):
    """No mesh is a reason the figure is absent, and the reason must be visible."""
    agent = _agent()
    vis = {}
    agent._timelapse_water_content_figure(
        {'time_lapse_water_content': [{'water_content_mean': np.zeros(4)}]},
        str(tmp_path), vis)
    assert vis == {}
    logged = capsys.readouterr().out
    assert 'no mesh' in logged and 'WARNING' in logged


# ---------------------------------------------------------------------------
# Figure selection reads the request
# ---------------------------------------------------------------------------
def test_the_model_decides_which_figures_the_request_wants():
    from PyHydroGeophysX.agents._figures import llm_figure_topics, plan_figures
    available = ['baseline_resistivity', 'timelapse_changes_percent',
                 'timelapse_water_content']
    asked = llm_figure_topics(
        'just show me the resistivity section',
        lambda prompt: '{"topics": ["resistivity"], "only_these": true}')
    plan = plan_figures(available, topics=asked)
    assert plan['draw'] == ['baseline_resistivity']
    assert set(plan['omitted']) == {'timelapse_changes_percent',
                                    'timelapse_water_content'}
    assert plan['missing'] == []


def test_the_request_is_read_in_any_language_and_through_a_typo():
    from PyHydroGeophysX.agents._figures import llm_figure_topics
    # The model does the reading; this checks the answer survives the round trip.
    reply = '{"topics": ["water_content", "change"], "only_these": false}'
    asked = llm_figure_topics('帮我画含水量和变化', lambda prompt: reply)
    assert asked['topics'] == {'water_content', 'change'}
    assert asked['only_these'] is False


def test_a_requested_topic_the_run_cannot_show_becomes_a_warning():
    """The failure the planner exists to catch, stated instead of silent."""
    from PyHydroGeophysX.agents._figures import missing_figure_warnings, plan_figures
    plan = plan_figures(['baseline_resistivity'],
                        topics={'topics': {'resistivity', 'water_content'},
                                'only_these': False})
    assert plan['missing'] == ['water_content']
    warning, = missing_figure_warnings(plan['missing'])
    assert 'water content' in warning and 'no data to plot' in warning


def test_a_model_that_cannot_answer_never_suppresses_a_figure():
    """Include on weak evidence; exclude only on explicit evidence."""
    from PyHydroGeophysX.agents._figures import plan_figures
    available = ['baseline_resistivity', 'timelapse_water_content']

    def refuses(prompt):
        raise RuntimeError('no API key')

    for query in (refuses, lambda p: 'I am not sure', lambda p: '{"topics": []}'):
        assert plan_figures(available, 'anything', query)['draw'] == available


def test_an_open_request_keeps_every_figure_but_leads_with_what_was_asked():
    from PyHydroGeophysX.agents._figures import plan_figures
    plan = plan_figures(['baseline_resistivity', 'timelapse_water_content'],
                        topics={'topics': {'water_content'}, 'only_these': False})
    assert plan['draw'] == ['timelapse_water_content', 'baseline_resistivity']
    assert plan['omitted'] == []


def test_omitted_figures_are_named_rather_than_dropped_silently():
    section = _agent()._figures_section(
        {'baseline_resistivity': '/out/baseline_resistivity.png',
         'timelapse_water_content': '/out/timelapse_water_content.png'},
        {'draw': ['timelapse_water_content'], 'omitted': ['baseline_resistivity'],
         'missing': []})
    assert '**Figure 1.** Volumetric water content' in section
    assert 'baseline_resistivity.png' in section and 'left out here' in section


def test_captions_have_one_home():
    """Captions were duplicated in the report agent and in the catalog."""
    from PyHydroGeophysX.agents import report_agent
    from PyHydroGeophysX.agents._figures import FIGURE_CATALOG
    assert not hasattr(report_agent.ReportAgent, 'FIGURE_CAPTIONS')
    assert FIGURE_CATALOG['timelapse_water_content']['topic'] == 'water_content'


# ---------------------------------------------------------------------------
# Other defects the same run exposed
# ---------------------------------------------------------------------------
def test_the_solver_row_names_the_linear_solver_not_the_time_lapse_scheme():
    """It printed "Linear solver: TEMPORAL_CONSTRAINT", which is not a solver."""
    section = _agent()._generate_timelapse_method_section(
        {'n_timesteps': 5, 'method': 'cgls', 'lambda': 15.0}, {})
    assert '| Linear solver | CGLS |' in section
    assert 'TEMPORAL_CONSTRAINT' not in section


def test_the_interpretation_does_not_carry_a_second_heading():
    """"6. Interpretation" was followed immediately by "7. Integrated Analysis"."""
    body = _agent()._compile_timelapse_report(
        '# Title\n', '## Executive Summary\n\nText.', '', '', '', '', '', '',
        '**AI-generated interpretation - verify before citing.**\n\nProse.', '')
    assert '## 2. Interpretation' in body
    assert 'Integrated Analysis' not in body


def test_one_file_used_under_several_roles_is_listed_once():
    """Seven files filled seventeen rows, which read as seventeen inputs."""
    from PyHydroGeophysX.agents.workflow_audit import _inventory_rows
    survey = r'C:\data\20171105.Data'
    rows = _inventory_rows([
        {'role': 'time_lapse_files', 'path': survey, 'bytes': 180456,
         'sha256': 'a' * 64},
        {'role': 'data_file', 'path': survey, 'bytes': 180456, 'sha256': 'a' * 64},
        {'role': 'baseline_file', 'path': survey, 'bytes': 180456, 'sha256': 'a' * 64},
        {'role': 'electrode_file', 'path': r'C:\data\electrodes.dat', 'bytes': 2072},
    ])
    assert len(rows) == 2
    assert rows[0][1] == 'time lapse files, data file, baseline file'
