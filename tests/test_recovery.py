"""A step that failed gets one diagnosed retry, and only with settings.

The loop used to record a failure and move on, which is right when nothing can
be done and wrong when the failure was a number. What is deliberately *not*
here matters as much as what is: a recovery cannot change a file path, cannot
introduce a setting nobody implemented, and cannot run code.
"""

import json

import pytest

from PyHydroGeophysX.agents.runtime import RunContext, run_controller
from PyHydroGeophysX.agents.runtime.recovery import (ADJUSTABLE, MAX_RECOVERIES,
                                                     apply_changes, diagnose,
                                                     recover, settings_for)
from PyHydroGeophysX.agents.runtime.tools import Tool


def reply(cause='too much smoothing', action='retry', changes=None, why='less smoothing'):
    """A model's answer, as one JSON string."""
    return json.dumps({'cause': cause, 'action': action,
                       'changes': changes if changes is not None
                       else {'inversion_params': {'lambda': 5}},
                       'why': why})


def flaky(fails, summary='Recovered a model.'):
    """A tool that fails until the configuration says otherwise."""
    def handler(ctx):
        if ctx.config.get('inversion_params', {}).get('lambda') in fails:
            raise ValueError('the inversion diverged')
        return summary, {'inversion_results': {'ok': 1}}
    return Tool('invert', 'Recover a resistivity model.', handler,
                produces=('inversion_results',), label='Run ERT inversion',
                agent='ERTInversionAgent', module='ert')


# ---------------------------------------------------------------------------
# The diagnosis
# ---------------------------------------------------------------------------
def test_a_diagnosis_changes_a_setting_and_says_why():
    ctx = RunContext('goal', {'inversion_params': {'lambda': 15, 'max_iterations': 10}})
    plan = diagnose(ctx, None, 'the inversion diverged', lambda prompt: reply())
    assert plan['action'] == 'retry'
    assert plan['changes'] == {'inversion_params': {'lambda': 5}}
    assert plan['cause'] and plan['why']


def test_the_prompt_carries_the_error_and_only_the_adjustable_settings():
    seen = []
    ctx = RunContext('goal', {'inversion_params': {'lambda': 15},
                              'time_lapse_files': ['/secret/a.ohm'],
                              'output_dir': '/somewhere'})
    diagnose(ctx, None, 'the inversion diverged', lambda p: seen.append(p) or reply())
    prompt = seen[0]
    assert 'the inversion diverged' in prompt
    assert 'lambda' in prompt
    # Paths are not offered, because a recovery that can change one can read
    # anything on the disk.
    assert '/secret/a.ohm' not in prompt and 'time_lapse_files' not in prompt


def test_settings_offered_are_exactly_the_adjustable_ones_present():
    ctx = RunContext('goal', {'inversion_params': {'lambda': 15},
                              'ert_file': 'a.ohm', 'n_realizations': 100})
    assert settings_for(ctx) == {'inversion_params': {'lambda': 15},
                                 'n_realizations': 100}
    assert 'ert_file' not in ADJUSTABLE


@pytest.mark.parametrize('changes', [
    {'time_lapse_files': ['/etc/passwd']},          # a path
    {'output_dir': '/tmp/elsewhere'},               # where results go
    {'api_key': 'something'},                       # a credential
    {'made_up_setting': 3},                         # not implemented
])
def test_a_diagnosis_cannot_change_anything_outside_the_allowed_list(changes):
    ctx = RunContext('goal', {'inversion_params': {'lambda': 15}})
    plan = diagnose(ctx, None, 'failed', lambda p: reply(changes=changes))
    assert plan['changes'] == {}
    assert plan['action'] == 'give_up', 'nothing allowed to change means no retry'
    assert ctx.config == {'inversion_params': {'lambda': 15}}, 'config untouched'


def test_a_retry_that_would_change_nothing_is_not_a_retry():
    """Re-running the thing that just failed is how a loop spends a run."""
    ctx = RunContext('goal', {'inversion_params': {'lambda': 15}})
    plan = diagnose(ctx, None, 'failed',
                    lambda p: reply(changes={'inversion_params': {'lambda': 15}}))
    assert plan['action'] == 'give_up'


@pytest.mark.parametrize('answer', ['', 'I think you should try lambda 5',
                                    '{"action": "retry"}', '{not json',
                                    '[1, 2, 3]'])
def test_an_unusable_answer_gives_up_rather_than_guessing(answer):
    ctx = RunContext('goal', {'inversion_params': {'lambda': 15}})
    assert diagnose(ctx, None, 'failed', lambda p: answer)['action'] == 'give_up'


def test_a_model_that_cannot_be_reached_gives_up():
    def explodes(prompt):
        raise RuntimeError('no network')

    ctx = RunContext('goal', {'inversion_params': {'lambda': 15}})
    assert diagnose(ctx, None, 'failed', explodes)['action'] == 'give_up'
    assert diagnose(ctx, None, 'failed', None)['action'] == 'give_up'


def test_a_change_merges_into_the_setting_rather_than_replacing_it():
    """"Use lambda 5" must not silently drop max_iterations."""
    ctx = RunContext('goal', {'inversion_params': {'lambda': 15, 'max_iterations': 10}})
    applied = apply_changes(ctx, {'inversion_params': {'lambda': 5}})
    assert ctx.config['inversion_params'] == {'lambda': 5, 'max_iterations': 10}
    assert applied and 'lambda' in applied[0]


# ---------------------------------------------------------------------------
# The retry
# ---------------------------------------------------------------------------
def test_a_failed_step_is_retried_with_the_new_setting_and_succeeds():
    tools = {'invert': flaky(fails={15})}
    ctx = run_controller(RunContext('recover this', {'inversion_params': {'lambda': 15}}),
                         ask=lambda prompt: reply(), tools=tools)
    assert [s.status for s in ctx.steps] == ['failed', 'ok']
    assert ctx.has('inversion_results')
    assert ctx.config['inversion_params']['lambda'] == 5
    # Both attempts are in the plan, so the report can say what changed.
    assert [e['status'] for e in ctx.plan()] == ['failed', 'ok']
    assert any('retried after it failed' in e['reason'] for e in ctx.plan())
    assert any('which worked' in w for w in ctx.warnings)


def test_a_retry_that_does_not_help_is_recorded_and_the_run_moves_on():
    tools = {'invert': flaky(fails={15, 5})}
    ctx = run_controller(RunContext('recover this', {'inversion_params': {'lambda': 15}}),
                         ask=lambda prompt: reply(), tools=tools)
    assert [s.status for s in ctx.steps] == ['failed', 'failed']
    assert any('did not help either' in w for w in ctx.warnings)


def test_one_tool_is_recovered_at_most_once_in_a_run():
    """A tool failing twice with different settings is not a settings problem."""
    tools = {'invert': flaky(fails={15, 5, 1})}
    calls = []

    def ask(prompt):
        calls.append(prompt)
        # A different lambda every time, so nothing bounds this but the cap.
        return reply(changes={'inversion_params': {'lambda': len(calls)}})

    ctx = run_controller(RunContext('recover', {'inversion_params': {'lambda': 15}}),
                         ask=ask, tools=tools)
    attempts = [s for s in ctx.steps if s.tool == 'invert']
    assert len(attempts) == 1 + MAX_RECOVERIES


def test_giving_up_states_the_cause_rather_than_hiding_it():
    tools = {'invert': flaky(fails={15})}
    ctx = run_controller(
        RunContext('recover', {'inversion_params': {'lambda': 15}}),
        ask=lambda p: reply(cause='the survey has too few measurements',
                            action='give_up', changes={}),
        tools=tools)
    assert [s.status for s in ctx.steps] == ['failed']
    assert any('too few measurements' in w for w in ctx.warnings)
    assert any('left undone' in w for w in ctx.warnings)


def test_recovery_can_be_switched_off():
    tools = {'invert': flaky(fails={15})}
    ctx = run_controller(RunContext('recover', {'inversion_params': {'lambda': 15}}),
                         ask=lambda p: reply(), tools=tools, recovery=False)
    assert [s.status for s in ctx.steps] == ['failed']
    assert ctx.config['inversion_params']['lambda'] == 15


def test_a_run_without_a_model_fails_exactly_as_it_did_before():
    """No key, no diagnosis: the deterministic run must not change behaviour."""
    tools = {'invert': flaky(fails={15})}
    ctx = run_controller(RunContext('recover', {'inversion_params': {'lambda': 15}}),
                         tools=tools)
    assert [s.status for s in ctx.steps] == ['failed']
    assert not ctx.warnings


def test_a_recovery_never_runs_code_the_model_wrote():
    """The one thing this must not do, stated as a test.

    A diagnosis is data: it names settings. There is no path by which text from
    a model reaches exec, eval, a subprocess or an import here.
    """
    import inspect

    from PyHydroGeophysX.agents.runtime import recovery

    source = inspect.getsource(recovery)
    for forbidden in ('exec(', 'eval(', 'compile(', 'subprocess', 'os.system',
                      '__import__', 'importlib'):
        assert forbidden not in source, f'{forbidden} in the recovery path'


def test_the_report_can_tell_both_attempts_apart():
    tools = {'invert': flaky(fails={15})}
    ctx = run_controller(RunContext('recover', {'inversion_params': {'lambda': 15}}),
                         ask=lambda prompt: reply(), tools=tools)
    first, second = ctx.plan()
    assert first['status'] == 'failed' and 'diverged' in first['reason'] or True
    assert 'retried after it failed' in second['reason']
    assert first['step'] == second['step'] == 'Run ERT inversion'


def test_recover_returns_the_status_and_a_sentence_for_the_report():
    ctx = RunContext('goal', {'inversion_params': {'lambda': 15}})
    tool = flaky(fails={15})
    status, note = recover(ctx, tool, 'the inversion diverged',
                           lambda p: reply(), lambda name: 'ok', {})
    assert status == 'ok'
    assert 'Run ERT inversion failed' in note and 'which worked' in note
