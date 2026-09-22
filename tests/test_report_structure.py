"""The report as a document: what it claims, what it includes, how it is built.

These cover the three faults a reader found in a real run's report - it named
an inversion scheme the package does not implement, it ended with several
hundred words of retrieved documentation the run never used, and its structure
was a run of bullet lists and JSON dumps rather than a deliverable - plus the
numbering and labelling machinery introduced to fix the third.
"""

import json
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# The scheme the report names must be the scheme the solver runs
# ---------------------------------------------------------------------------
def test_the_scheme_reported_is_the_one_the_solver_implements():
    """``time_lapse_method`` never reaches the solver, so it cannot name it.

    ``TimeLapseERTInversion`` inverts every survey at once under a temporal
    constraint. The configuration's 'difference' was written into the report as
    "Difference Inversion", describing time steps solved independently against
    a baseline - the opposite of what produced the numbers beside it.
    """
    from PyHydroGeophysX.agents._method import (IMPLEMENTED_SCHEME, SCHEME_LABEL,
                                                resolve_scheme, scheme_note)
    scheme, note = resolve_scheme('difference')
    assert scheme == IMPLEMENTED_SCHEME
    assert 'difference inversion' in note
    assert 'does not implement it' in note
    assert resolve_scheme(IMPLEMENTED_SCHEME)[1] is None
    assert resolve_scheme(None)[1] is None
    assert scheme_note({'time_lapse_method': 'temporal_constraint'}) is None
    assert 'temporal' in SCHEME_LABEL.lower()


def test_an_unknown_scheme_is_named_rather_than_silently_replaced():
    from PyHydroGeophysX.agents._method import resolve_scheme
    note = resolve_scheme('bayesian-4d')[1]
    assert "'bayesian-4d'" in note and 'does not implement' in note


def test_the_report_states_the_scheme_and_the_mismatch():
    from PyHydroGeophysX.agents.report_agent import ReportAgent
    agent = ReportAgent.__new__(ReportAgent)
    agent.execution_log = []
    section = agent._generate_timelapse_method_section(
        {'n_timesteps': 3, 'lambda': 15.0, 'temporal_regularization': 10.0},
        {'time_lapse_method': 'difference'})
    assert 'simultaneously' in section
    assert 'difference inversion' in section and 'does not implement it' in section
    assert 'Difference Inversion' not in section


# ---------------------------------------------------------------------------
# Retrieval noise
# ---------------------------------------------------------------------------
def test_only_cited_references_reach_the_document():
    """A semantic search returns its best matches, relevant or not.

    A time-lapse ERT report was ending with documentation for the seismic agent
    and the gravity module, quoted at length under a heading that implied the
    run had used them.
    """
    from PyHydroGeophysX.agents.workflow_audit import _references_block
    sources = [
        {'source': r'docs\agents\agent_reference.rst', 'line': 361,
         'excerpt': 'SeismicAgent processes seismic refraction data'},
        {'source': r'docs\auto_examples\Ex_Time_lapse_measurement.rst', 'line': 697,
         'excerpt': 'Seasonal patterns and infiltration dynamics'},
    ]
    body = 'The report cites Ex_Time_lapse_measurement.rst, line 697, and nothing else.'
    block = _references_block(sources, body)
    assert 'Ex_Time_lapse_measurement.rst' in block
    assert 'agent_reference.rst' not in block
    assert 'SeismicAgent' not in block
    assert '1 of 2' in block


def test_uncited_retrieval_is_reported_as_a_count_not_as_content():
    from PyHydroGeophysX.agents.workflow_audit import _references_block
    sources = [{'source': 'docs/a.rst', 'line': 1, 'excerpt': 'unused text'}]
    block = _references_block(sources, 'A report that cites nothing.')
    assert 'unused text' not in block
    assert '1 local reference' in block and 'processing_audit.json' in block


# ---------------------------------------------------------------------------
# Document structure
# ---------------------------------------------------------------------------
def test_section_numbers_stay_continuous_when_a_section_is_omitted():
    """Climate and water-content sections appear only when they have content."""
    from PyHydroGeophysX.agents._document import renumber
    with_climate = renumber('## Summary\n## Results\n## Climate\n## Recommendations\n')
    without = renumber('## Summary\n## Results\n## Recommendations\n')
    assert '## 4. Recommendations' in with_climate
    assert '## 3. Recommendations' in without
    assert '## 4.' not in without


def test_renumbering_twice_does_not_double_number():
    """The audit wrapper renumbers a body the report agent already numbered."""
    from PyHydroGeophysX.agents._document import renumber
    once = renumber('## Scope\n### Detail\n## Results\n')
    assert renumber(once) == once


def test_a_pipe_in_a_value_cannot_break_the_row_it_is_in():
    from PyHydroGeophysX.agents._document import table
    rendered = table(['Path'], [[r'C:\a|b\c']])
    row = rendered.splitlines()[2]
    # Two delimiters and nothing else: an unescaped pipe would split the value
    # into a second column that the header has no room for.
    assert row.count('|') - row.count(r'\|') == 2
    assert row == r'| C:\a\|b\c |'


def test_tables_disappear_rather_than_leaving_an_empty_frame():
    from PyHydroGeophysX.agents._document import bullets, facts, numbered, table
    assert table(['A'], []) == '' and facts([]) == ''
    assert bullets([]) == '' and numbered([]) == ''


# ---------------------------------------------------------------------------
# Survey identity: the same survey must be numbered the same everywhere
# ---------------------------------------------------------------------------
def _agent():
    from PyHydroGeophysX.agents.report_agent import ReportAgent
    agent = ReportAgent.__new__(ReportAgent)
    agent.execution_log = []
    return agent


SITE = {'survey_dates': ['2017-11-05', '2017-11-06', '2017-11-07'],
        'study_period': '2017-11-05 to 2017-11-07'}


def test_surveys_are_numbered_and_dated_the_same_way_in_every_table():
    """"Time Step 4" meant survey 5 in one table and survey 4 in another.

    The resistivity table indexed repeats from the baseline while the water
    content table labelled the baseline separately and then restarted at one,
    so the two tables disagreed about which survey a row described.
    """
    agent = _agent()
    base = np.full(40, 2000.)
    models = np.column_stack([base, base + 5, base + 12])
    resistivity = agent._generate_timelapse_inversion_section(
        {'final_models': models, 'n_timesteps': 3, 'chi2_values': [[1.6, 0, 0]]},
        'temporal_constraint', SITE)
    water = agent._generate_timelapse_water_content_section(
        {'time_lapse_water_content': [
            {'water_content_mean': np.full(40, 0.1 + 0.01 * i),
             'water_content_std': np.full(40, 0.01)} for i in range(3)]},
        {}, SITE)
    # Survey 3 was acquired on 2017-11-07 in both, and neither invents a label.
    for section in (resistivity, water):
        assert '| 3 | 2017-11-07 |' in section
        assert 'Time Step' not in section


def test_missing_dates_leave_the_column_empty_rather_than_guessing():
    agent = _agent()
    base = np.full(20, 1000.)
    section = agent._generate_timelapse_inversion_section(
        {'final_models': np.column_stack([base, base + 1]), 'n_timesteps': 2,
         'chi2_values': [[1.2, 0, 0]]}, 'temporal_constraint', {})
    assert '| 2 | \u2014 |' in section


# ---------------------------------------------------------------------------
# The assembled document
# ---------------------------------------------------------------------------
def test_the_detailed_report_is_a_document_not_a_json_dump(tmp_path):
    """Input inventory, stages, plan and quality were pasted in as JSON.

    Each is now a table; the machine-readable copy stays in the audit JSON,
    which is what a reviewer reconstructing the run actually needs.
    """
    from PyHydroGeophysX.agents.workflow_audit import write_report
    data = tmp_path / 'survey_20171105.dat'
    data.write_text('data')
    reports = write_report(
        tmp_path, {'data_file': str(data), 'user_request': 'Invert this line'},
        {'warnings': ['Inversion quality needs review'],
         'evaluation_results': {'status': 'needs_review', 'quality_score': 54.757,
                                'quality_metrics': {'data_fit': {'final_chi2': 1.63}},
                                'recommendations': ['Reduce lambda']}},
        [{'step': 'Invert', 'agent': 'ERTInversionAgent', 'description': 'Run it'}],
        'Model-written interpretation.', {},
        [{'step': 'Solve', 'elapsed_seconds': 1.5, 'details': 'done'}],
        {'numpy.array': 2}, [], {'model': 'gpt-5.6-luna', 'provider': 'openai'})
    text = Path(reports['report_markdown']).read_text(encoding='utf-8')

    assert text.startswith('# ')                       # a titled document
    assert '| Field | Detail |' in text                # document control block
    assert '| File | Used as | Size | SHA-256 |' in text  # inventory as a table
    assert '| Elapsed | Stage | Detail |' in text      # execution record
    assert '54.8 / 100' in text                        # a score, not a float dump
    assert 'survey_20171105.dat' in text
    assert '"data_file"' in text                       # config still quoted in full
    # The warning is stated where it will be read, not appended as raw JSON.
    assert 'Warnings Raised by This Run' in text
    assert '["Inversion quality needs review"]' not in text
    # Full digests and excerpts stay in the machine-readable record.
    audit = json.loads((tmp_path / 'processing_audit.json').read_text(encoding='utf-8'))
    assert len(audit['inputs'][0]['sha256']) == 64


def test_limitations_come_before_the_recommendations_that_rest_on_them(tmp_path):
    from PyHydroGeophysX.agents.workflow_audit import write_report
    body = tmp_path / 'engine.md'
    body.write_text('# Report\n\n## 1. Results\n\nText.\n\n## 2. Recommendations\n\n'
                    '1. Do the thing.\n', encoding='utf-8')
    reports = write_report(tmp_path, {}, {'warnings': []}, [], '',
                           {'report_markdown': str(body)}, [], {}, [], {})
    text = Path(reports['report_markdown']).read_text(encoding='utf-8')
    assert text.index('Limitations and Uncertainty') < text.index('Recommendations\n\n1.')


def test_a_missing_input_file_is_not_reported_as_a_directory():
    from PyHydroGeophysX.agents.workflow_audit import _size_or_status
    assert _size_or_status({'bytes': 180456}) == '176.2 kB'
    assert _size_or_status({'exists': True, 'note': 'Directory input'}) == 'directory'
    assert _size_or_status({'exists': False}) == 'not found'


def test_a_run_without_an_engine_report_still_gets_a_titled_document(tmp_path):
    from PyHydroGeophysX.agents.workflow_audit import write_report
    reports = write_report(tmp_path, {'user_request': 'Do the work'},
                           {'warnings': []}, [], 'Interpretation only.', {}, [],
                           {}, [], {})
    text = Path(reports['report_markdown']).read_text(encoding='utf-8')
    assert text.startswith('# Geophysical Processing Report')
    assert 'Do the work' in text and 'Interpretation only.' in text


def test_figures_are_captioned_rather_than_labelled_with_their_file_key():
    agent = _agent()
    section = agent._figures_section({
        'timelapse_changes_percent': '/out/timelapse_resistivity_changes_percent.png',
        'unknown_product': '/out/other.png'})
    assert '**Figure 1.** Resistivity change relative to the baseline, as a percentage.'\
        in section
    assert '![Figure 1](timelapse_resistivity_changes_percent.png)' in section
    assert '**Figure 2.** Unknown product.' in section


def test_recommendations_follow_from_the_run_rather_than_being_a_standing_list():
    agent = _agent()
    without_climate = agent._timelapse_recommendations(
        None, {'n_timesteps': 5, 'time_lapse_water_content': [{}]}, {})
    with_everything = agent._timelapse_recommendations(
        {'precipitation': [1, 2]}, {'n_timesteps': 40},
        {'petrophysical_params': {'n': 2.0}})
    assert 'Supply meteorological data' in without_climate
    assert 'Calibrate the petrophysical relationship' in without_climate
    assert 'Supply meteorological data' not in with_everything
    assert 'Calibrate the petrophysical relationship' not in with_everything
    assert 'Continue monitoring' in with_everything


@pytest.mark.parametrize('chi2, fragment', [
    ([[1.05, 0, 0]], 'within the range normally accepted'),
    ([[4.80, 0, 0]], 'outside the range normally accepted'),
])
def test_the_confidence_statement_reads_the_run_it_describes(chi2, fragment):
    agent = _agent()
    statement = agent._timelapse_confidence_statement(
        {'chi2_values': chi2}, {'status': 'success', 'quality_score': 88.0}, {})
    assert fragment in statement
    assert 'does not make it unique' in statement
