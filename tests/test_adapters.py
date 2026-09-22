"""Generated readers: what they can do, and the much longer list of what they cannot.

The feature is that a model can write a reader for a coordinate file whose
layout nobody anticipated. The conditions are the point: the user reads the code
and approves it, the code never touches the filesystem or the network, its
output has to look like what it claimed, and all of it is written into the
report. Most of what follows tests the refusals.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from PyHydroGeophysX.agents.runtime import RunContext
from PyHydroGeophysX.agents.runtime.adapters import (ADAPTABLE_FILES, adapt_file,
                                                     check_source, propose,
                                                     review, run_adapter)

GOOD = ('def adapt(text):\n'
        '    rows = []\n'
        '    for line in text.splitlines():\n'
        '        line = line.strip()\n'
        '        if not line or not (line[0].isdigit() or line[0] == "-"):\n'
        '            continue\n'
        '        parts = line.replace(",", " ").replace(";", " ").split()\n'
        '        rows.append([float(parts[0]), float(parts[1]), float(parts[2])])\n'
        '    return rows\n')

MESSY = ('station;distance;elev\n'
         '# surveyed 2024-06-01\n'
         '1,2.0,212.185\n'
         '2,4.0,211.249\n'
         '3,6.0,210.584\n')


def answer(code=GOOD, describe='comma separated with two header lines'):
    return json.dumps({'describe': describe, 'code': code})


# ---------------------------------------------------------------------------
# What may be compiled at all
# ---------------------------------------------------------------------------
def test_code_that_defines_the_reader_is_accepted():
    assert check_source(GOOD) == ''


@pytest.mark.parametrize('code, objects_to', [
    ('import os\ndef adapt(text): return []', 'import'),
    ('def adapt(text):\n    return open("/etc/passwd").read()', 'open('),
    ('def adapt(text):\n    return eval(text)', 'eval'),
    ('def adapt(text):\n    return exec(text)', 'exec'),
    ('def adapt(text):\n    return ().__class__.__mro__', '__'),
    ('def adapt(text):\n    return getattr(text, "x")', 'getattr'),
    ('def adapt(text):\n    return globals()', 'globals'),
])
def test_code_that_reaches_outside_its_job_is_refused_before_it_compiles(code,
                                                                         objects_to):
    objection = check_source(code)
    assert objection and objects_to in objection


def test_code_that_is_not_a_reader_is_refused():
    assert 'function called adapt' in check_source('x = 1')
    assert 'function called adapt' in check_source('')


def test_code_too_long_to_read_is_refused():
    """Approval means the user read it, which stops being true at some length."""
    assert 'too long' in check_source('def adapt(text):\n' + '    pass\n' * 900)


# ---------------------------------------------------------------------------
# What running it may return
# ---------------------------------------------------------------------------
def test_a_working_reader_returns_the_table():
    rows = run_adapter(GOOD, MESSY, 3)
    assert rows.shape == (3, 3)
    assert rows[0].tolist() == [1.0, 2.0, 212.185]


@pytest.mark.parametrize('body, complaint', [
    ('    return []', 'no rows'),
    ('    return [[1.0, 2.0]]', 'columns'),
    ('    return [[1.0, 2.0, float("nan")]]', 'not numbers'),
    ('    raise ValueError("not this layout")', 'failed'),
    ('    return "hello"', ''),
])
def test_output_that_is_not_what_it_claimed_is_rejected(body, complaint):
    with pytest.raises(ValueError) as caught:
        run_adapter('def adapt(text):\n' + body, MESSY, 3)
    assert complaint in str(caught.value)


def test_a_reader_that_never_finishes_is_abandoned():
    code = 'def adapt(text):\n    n = 0\n    while True:\n        n += 1\n'
    with pytest.raises(ValueError, match='did not finish'):
        run_adapter(code, MESSY, 3, timeout=0.3)


def test_the_reader_cannot_reach_the_filesystem_even_if_it_tries():
    """`open` is not in its namespace; the name check catches it first, and
    this proves the namespace would stop it too."""
    from PyHydroGeophysX.agents.runtime.adapters import SAFE_BUILTINS

    for reached_for in ('open', '__import__', 'exec', 'eval', 'compile',
                        'input', 'exit'):
        assert reached_for not in SAFE_BUILTINS


def test_the_reader_is_given_text_not_a_path(tmp_path):
    """It never learns where the file is, so it cannot read a different one."""
    import inspect

    from PyHydroGeophysX.agents.runtime import adapters

    signature = inspect.signature(adapters.run_adapter)
    assert list(signature.parameters) == ['code', 'text', 'columns', 'timeout']


# ---------------------------------------------------------------------------
# Approval
# ---------------------------------------------------------------------------
def test_nothing_runs_until_the_user_says_yes(tmp_path):
    geometry = tmp_path / 'location.txt'
    geometry.write_text(MESSY, encoding='utf-8')
    asked = []

    ctx = RunContext('goal', {'geophone_file': str(geometry)}, str(tmp_path),
                     {'ask_user': lambda event: asked.append(event) or 'run'})
    produced = adapt_file(ctx, 'geophone_file', 'invalid geometry row',
                          lambda prompt: answer())
    assert produced, 'approved, so it ran'
    assert len(asked) == 1
    # The source is in front of the user, not a summary of it.
    assert 'def adapt(text):' in asked[0]['question']
    assert [o['id'] for o in asked[0]['options']] == ['decline', 'run']
    assert asked[0]['default'] == 'decline'


def test_declining_leaves_the_file_unread_and_says_so(tmp_path):
    geometry = tmp_path / 'location.txt'
    geometry.write_text(MESSY, encoding='utf-8')
    ctx = RunContext('goal', {'geophone_file': str(geometry)}, str(tmp_path),
                     {'ask_user': lambda event: 'decline'})
    assert adapt_file(ctx, 'geophone_file', 'bad row', lambda p: answer()) == ''
    assert any('offered and declined' in w for w in ctx.warnings)
    assert not (Path(tmp_path) / 'adapted').exists()


def test_a_run_with_nobody_to_ask_declines_rather_than_assuming_consent(tmp_path):
    """Silence is not consent to execute code."""
    geometry = tmp_path / 'location.txt'
    geometry.write_text(MESSY, encoding='utf-8')
    ctx = RunContext('goal', {'geophone_file': str(geometry)}, str(tmp_path))
    assert adapt_file(ctx, 'geophone_file', 'bad row', lambda p: answer()) == ''
    assert not (Path(tmp_path) / 'adapted').exists()


def test_a_proposal_that_would_be_refused_is_never_put_to_the_user(tmp_path):
    """The user should not be asked to approve something that cannot run."""
    geometry = tmp_path / 'location.txt'
    geometry.write_text(MESSY, encoding='utf-8')
    asked = []
    ctx = RunContext('goal', {'geophone_file': str(geometry)}, str(tmp_path),
                     {'ask_user': lambda e: asked.append(e) or 'run'})
    bad = 'import os\ndef adapt(text):\n    return []'
    assert adapt_file(ctx, 'geophone_file', 'bad row',
                      lambda p: answer(code=bad)) == ''
    assert asked == []


# ---------------------------------------------------------------------------
# What it leaves behind
# ---------------------------------------------------------------------------
def test_the_original_file_is_never_written_to(tmp_path):
    geometry = tmp_path / 'location.txt'
    geometry.write_text(MESSY, encoding='utf-8')
    before = geometry.read_text(encoding='utf-8')
    ctx = RunContext('goal', {'geophone_file': str(geometry)}, str(tmp_path),
                     {'ask_user': lambda e: 'run'})
    produced = adapt_file(ctx, 'geophone_file', 'bad row', lambda p: answer())
    assert geometry.read_text(encoding='utf-8') == before
    assert Path(produced).parent == Path(tmp_path) / 'adapted'
    assert Path(produced) != geometry


def test_the_code_and_what_it_did_are_on_the_record(tmp_path):
    geometry = tmp_path / 'location.txt'
    geometry.write_text(MESSY, encoding='utf-8')
    ctx = RunContext('goal', {'geophone_file': str(geometry)}, str(tmp_path),
                     {'ask_user': lambda e: 'run'})
    produced = adapt_file(ctx, 'geophone_file', 'bad row', lambda p: answer())

    saved = Path(tmp_path) / 'adapted' / 'location_reader.py'
    assert saved.read_text(encoding='utf-8') == GOOD, 'the code is kept'
    note = next(w for w in ctx.warnings if 'code AQUAH wrote' in w)
    assert 'you approved' in note and '3 rows' in note
    assert 'original file is unchanged' in note
    # And the choice itself is recorded, so the report can say a person made it.
    assert ctx.questions and ctx.questions[0]['answer'] == 'run'
    # The rewritten file is a plain table the ordinary readers accept.
    from PyHydroGeophysX.data_processing.survey_geometry import read_coordinates
    assert read_coordinates(produced, 3).shape == (3, 3)


def test_approved_code_that_then_fails_is_reported_not_hidden(tmp_path):
    geometry = tmp_path / 'location.txt'
    geometry.write_text(MESSY, encoding='utf-8')
    ctx = RunContext('goal', {'geophone_file': str(geometry)}, str(tmp_path),
                     {'ask_user': lambda e: 'run'})
    breaks = 'def adapt(text):\n    return [[1.0, 2.0]]\n'
    assert adapt_file(ctx, 'geophone_file', 'bad row',
                      lambda p: answer(code=breaks)) == ''
    assert any('approved but did not work' in w for w in ctx.warnings)


# ---------------------------------------------------------------------------
# Which files may be adapted at all
# ---------------------------------------------------------------------------
def test_only_coordinate_tables_are_adaptable():
    """An instrument format is not a table, and a generated parser for one
    would be guessing at physics rather than at columns."""
    assert set(ADAPTABLE_FILES) == {'electrode_file', 'geophone_file',
                                    'topography_file'}
    for role in ('data_file', 'ert_file', 'time_lapse_files', 'raw_seismic_file',
                 'tdem_file', 'output_dir'):
        assert role not in ADAPTABLE_FILES


def test_a_file_that_is_not_there_is_not_adapted(tmp_path):
    ctx = RunContext('goal', {'geophone_file': str(tmp_path / 'absent.txt')},
                     str(tmp_path), {'ask_user': lambda e: 'run'})
    assert adapt_file(ctx, 'geophone_file', 'missing', lambda p: answer()) == ''


def test_the_desktop_shows_the_code_itself_not_a_summary_of_it():
    """Approving generated code means having read it, so it has to be legible:
    monospaced, scrollable and selectable, not wrapped into prose or hidden in
    a tooltip."""
    pytest.importorskip('PySide6')
    from PySide6.QtWidgets import QApplication, QPushButton
    from PyHydroGeophysX.qt_apps.modules.base import BaseModule
    from PyHydroGeophysX.qt_apps.modules.one_click import OneClickModule
    from PyHydroGeophysX.qt_apps.state import StudioState

    QApplication.instance() or QApplication([])
    page = OneClickModule(StudioState(), lambda *a, **k: None)
    target = BaseModule(StudioState(), lambda *a, **k: None)
    pages = {}
    page.window()._pages = pages
    page.window().show_module = lambda key: pages.setdefault(key, target)
    sent = []
    page._worker = type('W', (), {'answer': lambda self, d: sent.append(d)})()

    question = (f'location.txt could not be read, and AQUAH has written a '
                f'reader for it. Semicolon separated.\n\n{GOOD}')
    page._on_asked({'event': 'question', 'question': question, 'module': 'ert',
                    'default': 'decline',
                    'options': [{'id': 'decline', 'label': 'Do not run it'},
                                {'id': 'run', 'label': 'Run this code on the file'}]})
    strip = target._run_activity
    assert strip._code.isVisibleTo(strip), 'the code needs its own box'
    assert strip._code.toPlainText() == GOOD.strip()
    assert strip._code.isReadOnly()
    assert 'def adapt' not in strip._headline.text(), 'not wrapped into the prose'
    assert 'could not be read' in strip._headline.text()
    buttons = [strip._choices.itemAt(i).widget()
               for i in range(strip._choices.count())]
    buttons = [b for b in buttons if isinstance(b, QPushButton)]
    assert [b.text() for b in buttons] == ['Do not run it',
                                           'Run this code on the file']
    buttons[0].click()
    assert sent == ['decline']
    assert not strip._code.isVisibleTo(strip), 'the code goes away with the question'
    for widget in (page, target):
        widget.deleteLater()


def test_a_model_that_proposes_nothing_usable_proposes_nothing(tmp_path):
    geometry = tmp_path / 'location.txt'
    geometry.write_text(MESSY, encoding='utf-8')
    ctx = RunContext('goal', {'geophone_file': str(geometry)}, str(tmp_path))
    for reply in ('', 'sure, try splitting on commas', '{"describe": "x"}'):
        assert propose(ctx, 'geophone_file', geometry, 'bad row',
                       lambda p, r=reply: r) == ('', '')
    assert propose(ctx, 'geophone_file', geometry, 'bad row', None) == ('', '')
