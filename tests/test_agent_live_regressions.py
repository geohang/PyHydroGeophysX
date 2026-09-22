"""Contracts exposed by the real desktop synthetic ERT smoke run."""
from types import SimpleNamespace

import numpy as np
import pytest

from PyHydroGeophysX.agents.context_input_agent import ContextInputAgent
from PyHydroGeophysX.agents.inversion_evaluation_agent import InversionEvaluationAgent


def test_selected_files_and_nested_workflow_limits():
    agent = ContextInputAgent(api_key=None)
    context = agent._build_context({'data_file': 'survey.ohm', 'reference_file': 'notes.md'})
    assert 'survey.ohm' in context and 'reference_file' in context
    config = agent._validate_and_complete_config({'inversion_params': {
        'max_attempts': 1, 'auto_adjust': False, 'max_iter': 2}})
    assert config['max_attempts'] == 1 and config['auto_adjust'] is False
    assert config['inversion_params']['max_iterations'] == 2
    assert 'max_attempts' not in config['inversion_params']


@pytest.mark.parametrize('chi2,status,direction', [(5., 'underfit', -1), (.2, 'overfit', 1)])
def test_single_survey_metrics_and_adjustment_direction(chi2, status, direction):
    agent = InversionEvaluationAgent(api_key=None)
    results = dict(status='success', chi2=chi2, resistivity_model=np.array([100., 120., 90.]),
                   inversion_result=SimpleNamespace(iteration_chi2=[20., 10., chi2]))
    evaluation = agent._evaluate_quality(results, {'lambda': 20}, 70)
    metrics = evaluation['metrics']
    assert metrics['data_fit']['final_chi2'] == chi2
    assert metrics['data_fit']['status'] == status
    assert metrics['physical_plausibility']['min_resistivity'] == 90
    assert metrics['convergence']['final_chi2'] == chi2
    adjusted = agent._adjust_parameters({'lambda': 20}, metrics, [])
    assert np.sign(adjusted['lambda'] - 20) == direction


# One iteration of the time-lapse solver, verbatim from a five-dataset DAS run:
# [chi2, phi_m, phi_t] per iteration, not one history per dataset.
TIME_LAPSE_CHI2 = [[2461.2821443561206, 0.0, 0.0],
                   [14.212428665180862, 9638.770770140432, 1.73199812405151],
                   [5.520374871767444, 4645.02639073468, 1.695613649307923],
                   [2.0063390382480977, 3741.4359165559645, 1.5530748006727024],
                   [1.6305122138802701, 3476.039151762083, 1.4317184152926536]]


def test_a_time_lapse_chi2_table_reads_as_the_misfit_column():
    """Each row is [chi2, phi_m, phi_t]; only column 0 is on the chi-squared scale."""
    from PyHydroGeophysX.agents._chi2 import chi2_history, chi2_summary
    assert chi2_history(TIME_LAPSE_CHI2) == [row[0] for row in TIME_LAPSE_CHI2]
    assert chi2_history([5., 2., 1.6]) == [5., 2., 1.6]      # single-dataset shape
    assert chi2_history(None) == [] and chi2_history([]) == []
    summary = chi2_summary(TIME_LAPSE_CHI2)
    assert summary.startswith('1.631') and '2461.282' in summary and '5 iterations' in summary
    assert chi2_summary([]) == 'N/A'


def test_time_lapse_quality_scores_the_data_misfit_not_the_regularization():
    """Reading row[-1] scored phi_t - a plausible number from the wrong column.

    On the run this was found in, that averaged to 1.427 while the inversion's
    actual final chi-squared was 1.631. Nothing raised; the quality score was
    simply measuring the temporal regularization term.
    """
    agent = InversionEvaluationAgent(api_key=None)
    results = dict(status='success', chi2_values=TIME_LAPSE_CHI2,
                   resistivity_model=np.array([100., 120., 90.]))
    metrics = agent._evaluate_quality(results, {}, 70)['metrics']
    assert metrics['data_fit']['final_chi2'] == pytest.approx(1.6305122138802701)
    # Convergence must walk the misfit column, not iteration zero's three terms.
    assert metrics['convergence']['total_iterations'] == 5
    assert metrics['convergence']['initial_chi2'] == pytest.approx(2461.2821443561206)
    assert metrics['convergence']['final_chi2'] == pytest.approx(1.6305122138802701)


def test_time_lapse_interpretation_survives_a_nested_chi2_table():
    """The summary line used min()/max() over the rows, which returns a row."""
    from PyHydroGeophysX.agents._chi2 import chi2_summary
    with pytest.raises(TypeError):           # what the old expression did
        f"{min(TIME_LAPSE_CHI2):.3f}"
    assert '1.631' in chi2_summary(TIME_LAPSE_CHI2)


@pytest.mark.parametrize('request_text,wanted', [
    # The request from the run where water content was silently dropped.
    ('help me processs ERT data and estimate the water conent', True),
    ('estimate soil moisture from these surveys', True),
    ('convert resistivity to watercontent', True),
    ('run petrophysics with archie', True),
    ('帮我算含水量', True),
    ('处理ERT数据并估计含水率', True),
    ('just inversion, no conversion', False),
    ('resistivity only', False),
    ('只做反演', False),
    ('invert this line and make a report', False),
])
def test_water_content_intent_survives_a_typo_and_another_language(request_text, wanted):
    """Keyword matching read "water conent" as "no", and the product vanished."""
    from PyHydroGeophysX.agents._intent import wants_water_content
    assert wants_water_content({'user_request': request_text}) is wanted


def test_the_model_reads_intent_and_phrase_matching_is_only_the_fallback():
    """A phrase list only knows the wordings someone thought of in advance."""
    from PyHydroGeophysX.agents._intent import llm_deliverables, wants_water_content

    # Wording no keyword list covers, which the model understands.
    obscure = 'I want to know how wet the soil is down there'
    assert not wants_water_content({'user_request': obscure})          # matcher misses it
    answered = llm_deliverables(obscure, lambda prompt: '{"water_content": true}')
    assert answered == {'convert_to_water_content': True}
    assert wants_water_content({'user_request': obscure, **answered})  # and the flag wins

    # An explicit refusal from the model outranks hopeful wording.
    assert llm_deliverables('water content but only the section', 
                            lambda p: '{"water_content": true, "resistivity_only": true}') == {
        'convert_to_water_content': False}


def test_an_unusable_model_reply_falls_back_instead_of_guessing():
    from PyHydroGeophysX.agents._intent import llm_deliverables

    def refuses(prompt):
        raise RuntimeError('no API key')

    assert llm_deliverables('estimate water content', refuses) == {}
    assert llm_deliverables('estimate water content', lambda p: 'I am not sure') == {}
    assert llm_deliverables('estimate water content', lambda p: '{"water_content": null}') == {}
    # Prose or a code fence around the JSON must not throw the answer away.
    fenced = lambda p: 'Sure:\n```json\n{"water_content": true}\n```'
    assert llm_deliverables('x', fenced) == {'convert_to_water_content': True}


def test_an_explicit_flag_and_supplied_parameters_outrank_the_wording():
    from PyHydroGeophysX.agents._intent import wants_water_content
    assert wants_water_content({'user_request': 'resistivity only',
                                'convert_to_water_content': True})
    assert not wants_water_content({'user_request': 'water content please',
                                    'convert_to_water_content': False})
    assert wants_water_content({'user_request': 'invert this',
                                'petrophysical_params': {'rho_sat': 500}})


def _water_content_steps(n=5, sigma=0.077, drift=0.004):
    rng = np.random.default_rng(0)
    base = rng.uniform(0.028, 0.162, 200)
    return [{'water_content_mean': base + i * drift,
             'water_content_std': np.full(200, sigma),
             'layering': 'treated as a single unit with one petrophysical parameter set'}
            for i in range(n)]


def test_the_report_shows_the_water_content_it_computed():
    """It was converted for five time steps and reached the report as one sentence.

    The arrays were written under petrophysics/ and never appeared in the
    document that reports the run: no per-step numbers, no figure.
    """
    from PyHydroGeophysX.agents.report_agent import ReportAgent
    agent = ReportAgent.__new__(ReportAgent)
    section = agent._generate_timelapse_water_content_section(
        {'time_lapse_water_content': _water_content_steps()}, {})
    assert section.startswith('## Water Content')
    # Surveys are numbered once, from the baseline, and carry their role.
    assert '| 1 | — | Baseline |' in section and '| 5 | — | Repeat |' in section
    assert '10th-90th percentile' in section
    assert 'Change from baseline' in section
    assert 'single unit' in section              # how the section was divided
    assert '±0.077' in section                   # and what the error bar is


def test_a_change_below_the_error_bar_is_marked_as_such():
    """+0.004 against +/-0.077 is not a measured change and must not read as one."""
    from PyHydroGeophysX.agents.report_agent import ReportAgent
    agent = ReportAgent.__new__(ReportAgent)
    noisy = agent._generate_timelapse_water_content_section(
        {'time_lapse_water_content': _water_content_steps(sigma=0.077, drift=0.004)}, {})
    assert noisy.count('below the uncertainty on a single value') == 4

    # A drift well above the error bar carries no such caveat.
    clear = agent._generate_timelapse_water_content_section(
        {'time_lapse_water_content': _water_content_steps(sigma=0.002, drift=0.030)}, {})
    assert 'below the uncertainty on a single value' not in clear
    assert clear.count('exceeds the uncertainty on a single value') == 4


def test_no_water_content_section_when_none_was_computed():
    from PyHydroGeophysX.agents.report_agent import ReportAgent
    agent = ReportAgent.__new__(ReportAgent)
    assert agent._generate_timelapse_water_content_section({'status': 'success'}, {}) == ''
    assert agent._generate_timelapse_water_content_section({}, None) == ''


def test_the_resistivity_contrast_metric_is_named_for_what_it_holds():
    """94.5 was reported as 'resistivity_range' for a model spanning 105.8-10000."""
    from PyHydroGeophysX.agents.inversion_evaluation_agent import InversionEvaluationAgent

    agent = InversionEvaluationAgent.__new__(InversionEvaluationAgent)
    agent.quality_thresholds = {'min_resistivity': 1.0, 'max_resistivity': 100000.0}
    agent._log_execution = lambda *a, **k: None
    model = np.concatenate([np.full(50, 105.8), np.full(50, 10000.0)])
    _, metrics = agent._evaluate_physics({'resistivity_model': model})
    assert 'resistivity_range' not in metrics                 # the misleading name is gone
    assert metrics['resistivity_span'] == pytest.approx(9894.2, abs=0.1)
    assert metrics['resistivity_ratio'] == pytest.approx(94.5, abs=0.1)


@pytest.mark.parametrize('stop_reason,expected', [
    ('target', 'converged'),          # the solver hit its chi-squared target
    ('plateau', 'converged'),         # and this is the other "Convergence reached" branch
    ('iteration_cap', 'still_improving'),
    (None, 'still_improving'),        # no reason recorded: fall back to the slope
])
def test_convergence_defers_to_the_solver_that_ran(stop_reason, expected):
    """The log said "Convergence reached at iteration 8"; the report said still_improving."""
    from types import SimpleNamespace
    from PyHydroGeophysX.agents.inversion_evaluation_agent import InversionEvaluationAgent

    agent = InversionEvaluationAgent.__new__(InversionEvaluationAgent)
    agent.quality_thresholds = {'convergence_ratio': 0.01}
    agent._log_execution = lambda *a, **k: None
    rows = [[2461.28, 0, 0], [14.2, 0, 0], [5.5, 0, 0], [2.0, 0, 0],
            [1.73, 0, 0], [1.68, 0, 0], [1.64, 0, 0], [1.6304, 0, 0]]
    meta = {'stop_reason': stop_reason} if stop_reason else {}
    _, metrics = agent._evaluate_convergence(
        {'chi2_values': rows, 'time_lapse_result': SimpleNamespace(meta=meta)})
    assert metrics['status'] == expected
    assert metrics['solver_stop_reason'] == stop_reason


def test_the_study_period_comes_from_the_file_names():
    """The dates were on disk the whole time; the report said "N/A to N/A"."""
    from PyHydroGeophysX.agents.base_agent import _dates_from_filenames
    files = [f'C:/data/ERT/DAS/2017110{d}_141{d}.Data' for d in range(5, 10)]
    dates = _dates_from_filenames(files)
    assert dates[0] == '2017-11-05' and dates[-1] == '2017-11-09'
    assert _dates_from_filenames(['survey_line2.dat']) == []      # no guessing
    assert _dates_from_filenames(['run_99999999.dat']) == []      # not a real date
    assert _dates_from_filenames(None) == []


def test_empty_climate_sections_are_omitted_rather_than_rendered_empty():
    """A heading whose only content is "there is none" reads as a failure."""
    from PyHydroGeophysX.agents.report_agent import ReportAgent
    agent = ReportAgent.__new__(ReportAgent)
    assert agent._generate_timelapse_climate_section(None, {}) == ''
    assert agent._generate_timelapse_correlation_section(None, None) == ''


def test_resistivity_changes_carry_no_invented_cause():
    """Every decrease was labelled "moisture increase", every increase "drying/freezing".

    The report's own interpretation says those cannot be separated from ERT
    alone, so the table was contradicting the prose beside it.
    """
    from PyHydroGeophysX.agents.report_agent import ReportAgent
    agent = ReportAgent.__new__(ReportAgent)
    base = np.full(60, 2000.)
    models = np.column_stack([base, base - 190.83, base + 300.57])
    section = agent._generate_timelapse_inversion_section(
        {'final_models': models, 'n_timesteps': 3, 'chi2_values': [[1.6, 0, 0]]}, 'joint')
    assert 'moisture increase' not in section
    assert 'drying/freezing' not in section
    assert 'Largest decrease' in section and 'Largest increase' in section


def _parser_with(reply):
    """A ContextInputAgent whose LLM is a stub; returns (agent, calls list)."""
    from PyHydroGeophysX.agents.context_input_agent import ContextInputAgent
    agent = ContextInputAgent.__new__(ContextInputAgent)
    agent.api_key, agent.system_message = 'key', 'system'
    agent._log_execution = lambda *a, **k: None
    agent._build_context = lambda data: ''
    calls = []

    def query(prompt, *a, **k):
        calls.append(prompt)
        return reply if 'You decide which products' in prompt else '{}'

    agent.query_llm = query
    return agent, calls


ERT_ONLY = ('{"water_content": true, "aspects": {"climate": false, "fusion": false,'
            ' "tdem": false, "seismic": false, "hydro_model": false}}')


def test_a_plain_ert_request_stops_asking_about_every_other_topic():
    """Two of the four calls returned only nulls and false for a plain ERT request.

    Data fusion and climate were extracted unconditionally, at about seven
    seconds each, to answer questions the request had not raised.
    """
    request = 'help me processs ERT data and estimate the water conent'
    agent, calls = _parser_with(ERT_ONLY)
    agent.parse_request(request, {})
    assert len(calls) == 2                       # the router, then the ERT config
    # Compared against the prompts themselves: substrings are no good here,
    # because the router's prompt names every topic it can route and the ERT
    # prompt has a use_climate field of its own.
    assert agent._create_climate_prompt(request) not in calls
    assert agent._create_data_fusion_prompt(request) not in calls
    assert agent._create_inversion_prompt(request, '') in calls


def test_a_keyword_gate_still_runs_its_stage_when_the_router_disagrees():
    """"MODFLOW" in the request means MODFLOW, whatever the model decided."""
    request = 'invert the ERT line and compare with my MODFLOW outputs'
    agent, calls = _parser_with(ERT_ONLY)       # router says hydro_model: false
    agent.parse_request(request, {})
    assert agent._create_hydro_model_prompt(request) in calls


def test_an_unreachable_model_extracts_every_topic_as_before():
    """No router answer must not silently narrow what gets extracted."""
    request = 'invert this line'
    agent, calls = _parser_with('not json at all')
    agent.parse_request(request, {})
    assert agent._create_climate_prompt(request) in calls
    assert agent._create_data_fusion_prompt(request) in calls


@pytest.mark.parametrize('stated,keywords,expected', [
    (False, None, False),    # router skips a stage that has no keyword gate
    (True, None, True),
    (None, None, True),      # nothing known: run it
    (False, True, True),     # keywords win over a wrong router
    (True, False, True),
    (None, False, False),    # model silent: the keyword gate decides alone
    (None, True, True),
])
def test_the_two_routing_signals_compose_without_either_outvoting_the_other(
        stated, keywords, expected):
    from PyHydroGeophysX.agents._intent import stage_enabled
    assert stage_enabled({'tdem': stated}, 'tdem', keywords) is expected


def test_monte_carlo_never_draws_a_non_physical_archie_exponent():
    """A 100%-std prior on n=2 put 31% of the ensemble below n=1.

    Below one, the Waxman-Smits residual evaluates 0**(n-1) and the initial
    guess raises a ratio to 1/n: a divide-by-zero and an overflow per
    realization, and a third of the ensemble carrying exponents no rock has.
    """
    import warnings
    from PyHydroGeophysX.petrophysics.resistivity_models import resistivity_to_saturation

    rng = np.random.default_rng(1)
    resistivity = rng.uniform(100, 10000, 300)
    n = rng.normal(2.0, 2.0, 300)          # the prior the run actually used
    m = rng.normal(1.5, 1.5, 300)
    assert (n < 1).mean() > 0.2            # the draws really are out of range

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        saturation = resistivity_to_saturation(resistivity, 0.3, m, 20.0, n,
                                               sigma_sur=1 / 300.)
    assert [w for w in caught if issubclass(w.category, RuntimeWarning)] == []
    assert np.isfinite(saturation).all()
    assert ((saturation >= 0) & (saturation <= 1)).all()


def test_the_sampler_bounds_the_exponents_it_draws():
    from PyHydroGeophysX.petrophysics.monte_carlo import EXPONENT_BOUNDS, _sample_layer

    rng = np.random.default_rng(0)
    layer = {'m': {'mean': 1.5, 'std': 1.5}, 'n': {'mean': 2.0, 'std': 2.0},
             'porosity': {'mean': 0.3, 'std': 0.3}, 'rho_fluid': {'mean': 20.0, 'std': 5.0}}
    drawn = [_sample_layer(rng, layer) for _ in range(500)]
    assert all(EXPONENT_BOUNDS[0] <= s['n'] <= EXPONENT_BOUNDS[1] for s in drawn)
    assert all(EXPONENT_BOUNDS[0] <= s['m'] <= EXPONENT_BOUNDS[1] for s in drawn)


def test_time_lapse_interpretation_reads_the_attributes_that_exist():
    """It read baseline_model/time_lapse_models/changes, none of which are there.

    Every time-lapse run logged "object has no attribute 'baseline_model'" and
    returned no interpretation at all.
    """
    from types import SimpleNamespace
    from PyHydroGeophysX.agents.ert_inversion_agent import ERTInversionAgent

    agent = ERTInversionAgent.__new__(ERTInversionAgent)
    agent.api_key, agent.system_message = 'key', 'system'
    agent._log_execution = lambda *a, **k: None
    agent.query_llm = lambda prompt, *a, **k: prompt      # echo the prompt back

    base = np.full(100, 2000.)
    models = np.column_stack([base, base + 3.85, base + 14.02])
    prompt = agent._interpret_time_lapse_results(
        SimpleNamespace(final_models=models), 'difference')
    assert 'Number of time steps: 3' in prompt
    assert 'Baseline resistivity range: 2000.0 to 2000.0' in prompt
    assert 'Maximum absolute resistivity change: 14.0' in prompt
    assert '+3.85, +14.02' in prompt
    # Nothing to interpret must not raise either.
    assert agent._interpret_time_lapse_results(
        SimpleNamespace(final_models=None), 'difference') is None


def test_html_reports_are_no_longer_produced():
    from PyHydroGeophysX.agents.report_agent import ReportAgent
    agent = ReportAgent.__new__(ReportAgent)
    assert agent._save_html_report('# report', '/tmp', 'x') is None


@pytest.mark.parametrize('markers,n_cells,n_layers,fragment', [
    # The real case: an inversion parameter mesh numbers its cells, it does not
    # map geology. Reading that as 832 layers gives each cell its own parameter
    # set estimated from one sample.
    (list(range(832)), 832, 1, 'do not describe geological structure'),
    ([0] * 832, 832, 1, 'one region'),
    ([0] * 300 + [1] * 400 + [2] * 132, 832, 3, 'Using 3 geological layers'),
    ([0] * 829 + [1] * 3, 832, 1, 'holds 3 cell'),      # a layer too small to parameterise
    (list(range(10)), 832, 1, 'did not match the model'),
])
def test_layers_are_used_when_they_group_the_mesh_and_ignored_when_they_do_not(
        markers, n_cells, n_layers, fragment):
    """With no structural information the section is one unit - stated, not stumbled into."""
    from PyHydroGeophysX.agents.petrophysics_agent import resolve_layers
    resolved, unique, why = resolve_layers(np.asarray(markers), n_cells)
    assert len(unique) == n_layers
    assert len(resolved) == n_cells
    assert fragment in why
    if n_layers == 1:
        assert set(np.unique(resolved)) == {0}          # genuinely one unit


@pytest.mark.parametrize('sigma,usable,fragment', [
    (0.0806, False, 'not distinguishable'),   # the real run: sigma covers 61% of the range
    (0.040, True, 'much larger than that error bar'),
    (0.005, True, 'resolved well enough'),
])
def test_water_content_states_what_its_error_bar_allows(sigma, usable, fragment):
    """0.028-0.162 +/- 0.081 reads like a finding and is the absence of one."""
    from PyHydroGeophysX.agents._uncertainty import water_content_reliability
    result = {'water_content_mean': np.linspace(0.0284, 0.1617, 200),
              'water_content_std': np.full(200, sigma)}
    verdict = water_content_reliability(result, {})
    assert verdict['usable'] is usable
    assert fragment in verdict['sentence']


def test_uncalibrated_petrophysics_is_named_as_the_thing_to_fix():
    from PyHydroGeophysX.agents._uncertainty import water_content_reliability
    result = {'water_content_mean': [0.10, 0.30], 'water_content_std': [0.01, 0.01]}
    assert 'generated defaults' in water_content_reliability(result, {})['sentence']
    calibrated = water_content_reliability(result, {'petrophysical_params': {'n': 2.0}})
    assert 'generated defaults' not in calibrated['sentence']
    assert calibrated['calibrated'] is True


def test_an_unusable_water_content_becomes_a_run_warning():
    """It exists, it will be read, and it will be read at face value."""
    from PyHydroGeophysX.agents._uncertainty import result_caveats
    loose = {'water_content_mean': [0.03, 0.16], 'water_content_std': [0.08, 0.08]}
    tight = {'water_content_mean': [0.03, 0.16], 'water_content_std': [0.002, 0.002]}
    assert len(result_caveats({}, loose)) == 1
    assert 'too uncertain to interpret quantitatively' in result_caveats({}, loose)[0]
    assert result_caveats({}, tight) == []
    assert result_caveats({}, {'status': 'success'}) == []   # nothing produced, nothing to caveat


def test_the_model_names_the_site_and_a_gazetteer_places_it():
    """The split that matters: the model reads a name, a service returns a position.

    Asking the model for coordinates directly is the tempting shortcut and the
    wrong one - a confidently wrong latitude puts the climate series in another
    valley and nothing downstream can catch it.
    """
    from PyHydroGeophysX.agents._intent import llm_deliverables
    from PyHydroGeophysX.agents._geocode import geocode_place, coords_from_config

    reply = ('{"water_content": null, "resistivity_only": false,'
             ' "site_location": "Medicine Bow, Wyoming"}')
    extracted = llm_deliverables('这是怀俄明州 Medicine Bow 的数据，和降水对比一下',
                                 lambda prompt: reply)
    assert extracted == {'site_location': 'Medicine Bow, Wyoming'}

    service = lambda url: [{'lon': '-106.1978', 'lat': '41.8994',
                            'display_name': 'Medicine Bow, Carbon County, Wyoming'}]
    located = geocode_place(extracted['site_location'], fetch=service)
    assert located['coords'] == (-106.1978, 41.8994)
    assert 'Carbon County' in located['matched_name']   # shown so a wrong match is visible
    assert coords_from_config({'climate_config': {'coords': list(located['coords'])}})         == (-106.1978, 41.8994)


def test_coordinates_from_the_model_are_refused():
    """site_location must be a name. Numbers mean the model answered anyway."""
    from PyHydroGeophysX.agents._intent import llm_deliverables
    assert llm_deliverables('x', lambda p: '{"site_location": "41.4, -106.2"}') == {}
    assert llm_deliverables('x', lambda p: '{"site_location": "-106.2 41.4"}') == {}
    assert llm_deliverables('x', lambda p: '{"site_location": "Medicine Bow"}') == {
        'site_location': 'Medicine Bow'}


def test_a_geocoding_service_that_cannot_answer_never_invents_a_position():
    from PyHydroGeophysX.agents._geocode import geocode_place
    assert geocode_place('anywhere', fetch=lambda url: []) is None
    assert geocode_place('anywhere', fetch=lambda url: None) is None
    assert geocode_place('', fetch=lambda url: [{'lon': '1', 'lat': '2'}]) is None
    # Out-of-range values are a broken service, not a location.
    assert geocode_place('x', fetch=lambda url: [{'lon': '999', 'lat': '2'}]) is None


@pytest.mark.parametrize('config,expected_fragment', [
    ({'crs': 'local'}, 'local coordinate frame'),
    ({'crs': 'local', 'site_location': 'Nowhere Creek'}, 'could not be resolved'),
    ({'crs': 'EPSG:32613'}, 'no site longitude and latitude'),
])
def test_a_blocked_climate_step_says_which_piece_is_missing(config, expected_fragment):
    """"No climate data" is not actionable; naming the missing piece is."""
    from PyHydroGeophysX.agents._intent import climate_blocker, unmet_requests
    assert expected_fragment in climate_blocker(config)
    warning = unmet_requests({**config, 'user_request': 'compare against rainfall'},
                             {'status': 'success'})
    assert len(warning) == 1 and expected_fragment in warning[0]


def test_supplied_coordinates_need_no_lookup_and_block_nothing():
    from PyHydroGeophysX.agents._intent import climate_blocker, unmet_requests
    config = {'crs': 'local', 'climate_config': {'coords': [-106.2, 41.4]},
              'user_request': 'compare against rainfall'}
    assert climate_blocker(config) is None
    assert unmet_requests(config, {'status': 'success', 'climate_data': {'prcp': [1]}}) == []


def test_the_evaluation_on_the_results_stays_serialisable():
    """evaluation_results["final_results"] is the results dict itself.

    Storing the evaluation whole made the results self-referential, so the run
    finished every stage and then died writing the audit with "Circular
    reference detected". The verdict is what the audit needs; the copy of the
    models is what breaks it.
    """
    import json
    from PyHydroGeophysX.agents._intent import unmet_requests

    results = {'status': 'success', 'chi2_values': [[1.63, 0, 0]]}
    evaluation = {'status': 'needs_review', 'quality_score': 54.8,
                  'final_results': results}
    results['evaluation_results'] = evaluation
    with pytest.raises(ValueError, match='Circular reference'):
        json.dumps(results, default=str)

    results['evaluation_results'] = {k: v for k, v in evaluation.items()
                                     if k != 'final_results'}
    json.dumps(results, default=str)                       # no longer raises
    assert results['evaluation_results']['quality_score'] == 54.8
    assert results['evaluation_results']['status'] == 'needs_review'
    assert unmet_requests({'user_request': 'invert'}, results) == []


def _findings(inversion, climate=None, evaluation=None):
    from PyHydroGeophysX.agents.report_agent import ReportAgent
    return ReportAgent.__new__(ReportAgent)._timelapse_key_findings(
        inversion, climate, evaluation)


def test_key_findings_report_the_run_instead_of_a_template():
    """The fixed text asserted good convergence and climate insight either way.

    On the run that prompted this, no climate data existed and the quality score
    was 54.8/100, so two of the three canned findings were false - printed right
    under an AI narrative that had correctly hedged both.
    """
    base = np.full(50, 2000.)
    models = np.column_stack([base, base + 3.85, base + 14.02])
    text = _findings({'final_models': models, 'chi2_values': [[2461.28, 0, 0], [1.6303, 3477., 1.43]]},
                     climate=None, evaluation={'quality_score': 54.8, 'status': 'needs_review'})
    assert '54.8/100' in text and 'needs review' in text
    assert '1.630' in text                      # the real misfit, not a mean of three columns
    assert 'good convergence' not in text
    assert 'No meteorological data were supplied' in text
    assert 'Water Content' not in text          # nothing was produced, so nothing is claimed


def test_key_findings_change_when_the_run_goes_well():
    base = np.full(50, 2000.)
    text = _findings({'final_models': np.column_stack([base, base - 40.]),
                      'chi2_values': [[1.05, 0, 0]],
                      'time_lapse_water_content': [{}, {}]},
                     climate={'prcp': [1, 2]},
                     evaluation={'quality_score': 82.4, 'status': 'success'})
    assert 'a decrease of 40.00' in text
    assert 'meets the configured threshold' in text and '82.4/100' in text
    assert 'Meteorological data were integrated' in text
    assert 'Water Content' in text and '2 time step' in text


def test_key_findings_stay_honest_when_the_run_reports_nothing():
    text = _findings({}, climate=None, evaluation=None)
    assert 'did not return' in text
    assert 'no quality score' in text and 'no reported chi-squared' in text


def test_a_requested_product_that_never_appeared_is_reported_as_missing():
    """The safety net: no branch may finish silently short of the request."""
    from PyHydroGeophysX.agents._intent import unmet_requests
    config = {'user_request': 'estimate the water conent'}
    assert unmet_requests(config, {'status': 'success'}) == [
        'Water content was requested but this run produced none.']
    # Any of the shapes a branch can deliver it in counts as delivered.
    for delivered in ({'water_content_mean': [0.3]}, {'petrophysics_results': {'x': 1}},
                      {'time_lapse_water_content': [{'x': 1}]}):
        assert unmet_requests(config, {'status': 'success', **delivered}) == []
    assert unmet_requests({'user_request': 'invert the ERT data'}, {'status': 'success'}) == []


def test_one_attempt_no_retry_and_interpretation_without_coverage(monkeypatch):
    agent = InversionEvaluationAgent(api_key=None)
    agent.api_key = 'fake'
    monkeypatch.setattr(agent, 'query_llm', lambda *a, **k: 'review required')
    monkeypatch.setattr(agent, '_rerun_inversion', lambda *a: pytest.fail('Unexpected retry'))
    result = agent.execute(dict(max_attempts=1, inversion_params={'lambda': 20},
        inversion_results=dict(status='success', chi2=12., resistivity_model=[100., 110., 120.])))
    assert result['status'] == 'needs_review'
    assert result['attempts'] == 1
    assert result['interpretation'] == 'review required'


def test_best_result_metrics_and_parameters_stay_together(monkeypatch):
    agent = InversionEvaluationAgent(api_key=None)
    monkeypatch.setattr(agent, '_evaluate_quality', lambda result, params, threshold: dict(
        quality_score=result['score'], is_acceptable=False,
        metrics={'marker': result['marker']}, recommendations=[], parameters=params))
    monkeypatch.setattr(agent, '_adjust_parameters', lambda params, *a: {'lambda': params['lambda'] + 1})
    monkeypatch.setattr(agent, '_rerun_inversion', lambda *a: dict(status='success', score=40, marker='retry'))
    result = agent.execute(dict(max_attempts=2, inversion_params={'lambda': 20},
        inversion_results=dict(status='success', score=80, marker='initial')))
    assert result['status'] == 'needs_review'  # Overall score alone is insufficient.
    assert result['quality_metrics']['marker'] == 'initial'
    assert result['final_results']['marker'] == 'initial'
    assert result['adjusted_params']['lambda'] == 20


def test_report_does_not_claim_convergence_or_require_a_chi2():
    from PyHydroGeophysX.agents.report_agent import ReportAgent
    agent = ReportAgent(api_key=None)
    data = {'inversion_results': {'chi2': None, 'iterations': 2},
            'evaluation_results': {'status': 'needs_review', 'summary': 'Iteration limit reached'}}
    summary = agent._generate_executive_summary(data, {})
    assert 'needs_review' in summary and 'converged' not in summary
    assert 'Iteration limit reached' in agent._generate_inversion_summary(data)


@pytest.mark.parametrize('source', ['rhoa', 'resistance'])
def test_ert_export_preserves_rhoa_and_signed_resistance(tmp_path, source):
    pg = pytest.importorskip('pygimli')
    from PyHydroGeophysX.data_processing.ert_data_agent import (
        StandardERT, Electrode, Observation, Quadruplet, export_for_inversion)
    # Dipole-dipole uses negative R and K for this electrode ordering.
    k = -6 * np.pi
    survey = StandardERT(instrument='BERT',
        electrodes=[Electrode(i + 1, float(i), 0., 0.) for i in range(4)],
        observations=[Observation(quad=Quadruplet(1, 2, 3, 4),
            app_res=rho if source == 'rhoa' else rho/k,
            dV=rho/k, I=1., K=k if source == 'rhoa' else None) for rho in (100., 120., 140.)],
        metadata={'app_res_source': source})
    path = export_for_inversion(survey, outdir=tmp_path, export_strategy='legacy', use_source_error=True)
    data = pg.DataContainerERT(str(path))
    assert data.size() == 3
    assert np.all(np.asarray(data['r']) < 0)
    np.testing.assert_allclose(np.asarray(data['r']) * np.asarray(data['k']), data['rhoa'])
    np.testing.assert_allclose(data['rhoa'], [100., 120., 140.], rtol=.01)


def test_retry_keeps_inputs_and_writes_under_run_directory(tmp_path, monkeypatch):
    from PyHydroGeophysX.agents.ert_inversion_agent import ERTInversionAgent
    agent = InversionEvaluationAgent(api_key=None)
    agent.history = [{}]
    monkeypatch.setattr(ERTInversionAgent, 'execute', lambda self, inputs: inputs)
    result = agent._rerun_inversion({'output_dir': str(tmp_path), 'instrument': 'BERT',
                                    'inversion_results': {}}, {'lambda': 10, 'max_iterations': 2})
    assert result['output_dir'] == str(tmp_path / 'attempt_2')
    assert result['instrument'] == 'BERT'
    assert result['inversion_params']['max_iterations'] == 2
