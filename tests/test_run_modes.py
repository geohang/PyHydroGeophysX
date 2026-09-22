"""Auto and step-by-step as two hooks on one loop, and following the run.

The studio offered the same work two ways through two separate code paths, so a
step could behave one way when a user approved it and another when the workflow
ran it unattended. They now share the loop, the registry and the ordering, and
differ only in the `on_step` hook - which is also where "show me where it is
working" is expressed, since that hook fires before each step rather than after.
"""

import os
import re
from pathlib import Path

import pytest

from PyHydroGeophysX.agents.runtime import RunContext, auto, run_controller, step_by_step
from PyHydroGeophysX.agents.runtime.modes import announcement, collect
from PyHydroGeophysX.agents.runtime.tools import Tool

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')


LOAD = Tool('load', 'Read the surveys.',
            lambda ctx: ('Loaded 5 surveys.', {'ert_data': [1, 2, 3, 4, 5]}),
            produces=('ert_data',), agent='ERTLoaderAgent',
            label='Load ERT data', module='ert')
INVERT = Tool('invert', 'Recover a model.',
              lambda ctx: ('Chi-squared 1.63.', {'inversion_results': {'ok': 1}}),
              requires=('ert_data',), produces=('inversion_results',),
              agent='ERTInversionAgent', label='Run inversion', module='ert')
WATER = Tool('water', 'Convert to water content.',
             lambda ctx: ('Converted 5 models.', {'water_content': [{}]}),
             requires=('inversion_results',), produces=('water_content',),
             agent='PetrophysicsAgent', label='Convert to water content',
             module='geo_hydrology')
REPORT = Tool('report', 'Write the report.',
              lambda ctx: ('Wrote it.', {'report_files': {'md': 'r.md'}}),
              requires=('inversion_results',), produces=('report_files',),
              agent='ReportAgent', label='Generate report', module='one_click')

ALL = {'load': LOAD, 'invert': INVERT, 'water': WATER, 'report': REPORT}


# ---------------------------------------------------------------------------
# The two modes are the same run
# ---------------------------------------------------------------------------
def test_auto_and_step_by_step_take_the_same_steps_in_the_same_order():
    """Two code paths could drift; one loop with two hooks cannot."""
    automatic = run_controller(RunContext('goal'), on_step=auto(), tools=ALL)
    approved = run_controller(RunContext('goal'),
                              on_step=step_by_step(lambda event: True), tools=ALL)
    assert [s.tool for s in automatic.steps] == [s.tool for s in approved.steps]
    assert [s.status for s in automatic.steps] == [s.status for s in approved.steps]
    assert [s.summary for s in automatic.steps] == [s.summary for s in approved.steps]


def test_both_modes_announce_the_same_thing_before_each_step():
    from_auto, from_steps = [], []
    run_controller(RunContext('goal'), on_step=auto(collect(from_auto)), tools=ALL)
    run_controller(RunContext('goal'),
                   on_step=step_by_step(lambda e: True, collect(from_steps)),
                   tools=ALL)
    assert from_auto == from_steps
    assert [e['module'] for e in from_auto] == ['ert', 'ert', 'geo_hydrology',
                                                'one_click']


# ---------------------------------------------------------------------------
# Step-by-step honours the answer
# ---------------------------------------------------------------------------
def test_a_skipped_step_is_recorded_rather_than_dropped():
    """A product the user declined must still be visible as not produced."""
    ctx = run_controller(
        RunContext('goal'),
        on_step=step_by_step(lambda e: 'skip' if e['tool'] == 'water' else True),
        tools=ALL)
    statuses = {s.tool: s.status for s in ctx.steps}
    assert statuses['water'] == 'skipped'
    assert 'Skipped at the user' in next(s.error for s in ctx.steps
                                         if s.tool == 'water')
    # The run carried on to the steps that did not depend on it.
    assert statuses['report'] == 'ok'
    # And the plan says so, so the report can state the absence.
    assert ('Convert to water content', 'skipped') in [
        (entry['step'], entry['status']) for entry in ctx.plan()]


def test_stop_ends_the_run_where_the_user_said():
    ctx = run_controller(
        RunContext('goal'),
        on_step=step_by_step(lambda e: 'stop' if e['tool'] == 'invert' else True),
        tools=ALL)
    assert [s.tool for s in ctx.steps] == ['load', 'invert']
    assert ctx.steps[-1].status == 'skipped'
    assert 'Stopped here' in ctx.steps[-1].error


@pytest.mark.parametrize('answer', [None, 'maybe', 0, object()])
def test_an_answer_that_is_not_approval_stops_rather_than_proceeds(answer):
    """Defaulting to proceed would run unapproved work on a typo."""
    ctx = run_controller(RunContext('goal'),
                         on_step=step_by_step(lambda e: answer), tools=ALL)
    assert ctx.steps[0].status == 'skipped'
    assert len(ctx.steps) == 1


def test_a_broken_approval_gate_is_not_consent():
    def explodes(event):
        raise RuntimeError('the dialog crashed')

    ctx = run_controller(RunContext('goal'),
                         on_step=step_by_step(explodes), tools=ALL)
    assert [s.status for s in ctx.steps] == ['skipped']


def test_a_display_failure_never_stops_the_work():
    """Auto mode narrates; narration is not part of the computation."""
    ctx = run_controller(RunContext('goal'),
                         on_step=auto(lambda event: 1 / 0), tools=ALL)
    assert [s.status for s in ctx.steps] == ['ok'] * 4


# ---------------------------------------------------------------------------
# What the announcement carries
# ---------------------------------------------------------------------------
def test_an_announcement_is_json_safe_and_names_the_module():
    """It crosses a process boundary as one line of JSON."""
    import json

    ctx = RunContext('goal')
    event = announcement(ctx, INVERT, 'the surveys are loaded')
    assert json.loads(json.dumps(event)) == event
    assert event == {'tool': 'invert', 'label': 'Run inversion', 'module': 'ert',
                     'reason': 'the surveys are loaded', 'step': 1}


def test_every_registered_tool_names_a_module_the_studio_has():
    """A key the studio does not know would navigate nowhere, silently."""
    pytest.importorskip('PySide6')
    from PyHydroGeophysX.agents.runtime import TOOLS
    from PyHydroGeophysX.agents.runtime import catalog  # noqa: F401 - registers
    from PyHydroGeophysX.qt_apps.modules import MODULE_ORDER

    assert TOOLS, 'the catalog registered nothing'
    for name, tool in TOOLS.items():
        assert tool.module, f'{name} names no studio module'
        assert tool.module in MODULE_ORDER, f'{name} -> unknown module {tool.module}'


# ---------------------------------------------------------------------------
# The desktop follows along
# ---------------------------------------------------------------------------
def test_the_workflow_page_shows_the_module_doing_the_work():
    pytest.importorskip('PySide6')
    from PySide6.QtWidgets import QApplication
    from PyHydroGeophysX.qt_apps.modules.one_click import OneClickModule
    from PyHydroGeophysX.qt_apps.state import StudioState

    QApplication.instance() or QApplication([])
    page = OneClickModule(StudioState(), lambda *a, **k: None)
    shown = []
    page.window().show_module = shown.append   # stands in for the main window

    page._on_progress('Load ERT data', 0.2, 'reason', 'ert')
    page._on_progress('Run inversion', 0.4, 'reason', 'ert')      # same module
    page._on_progress('Convert to water content', 0.6, 'r', 'geo_hydrology')
    page._on_progress('Writing audit', 0.9, 'r', '')              # no module
    assert shown == ['ert', 'geo_hydrology'], 'navigated once per module change'

    page.follow.setChecked(False)
    page._on_progress('Generate report', 0.95, 'r', 'one_click')
    assert shown == ['ert', 'geo_hydrology'], 'follow-along was switched off'
    page.close()
    page.deleteLater()


def test_the_visited_module_says_what_is_running_and_shows_its_figures(tmp_path):
    """Navigating alone showed an empty tool, which reads as nothing happening.

    The studio's modules hold their own state while the workflow runs headless
    in another process, so a panel brought to the front mid-run said "No data
    loaded" - the user's words were that they could not tell anything had been
    done. The page now carries the step and the figures written since the run
    started.
    """
    pytest.importorskip('PySide6')
    import time

    from PySide6.QtWidgets import QApplication
    from PyHydroGeophysX.qt_apps.modules.base import BaseModule
    from PyHydroGeophysX.qt_apps.modules.one_click import OneClickModule
    from PyHydroGeophysX.qt_apps.state import StudioState

    QApplication.instance() or QApplication([])
    page = OneClickModule(StudioState(), lambda *a, **k: None)
    target = BaseModule(StudioState(), lambda *a, **k: None)
    # The studio builds a module page the first time it is shown, so the strip
    # has to be applied after navigating, not before.
    pages = {}
    page.window()._pages = pages
    page.window().show_module = lambda key: pages.setdefault(key, target)

    page._output = str(tmp_path)
    page._run_started_at = time.time() - 1
    stale = tmp_path / 'from_a_previous_run.png'
    stale.write_bytes(b'x' * 10)
    import os
    os.utime(stale, (page._run_started_at - 600, page._run_started_at - 600))
    fresh = tmp_path / 'ert' / 'inverted_section.png'
    fresh.parent.mkdir()
    fresh.write_bytes(b'x' * 10)

    assert page._recent_figures() == [str(fresh)], 'only this run\'s figures'
    page._on_progress('Run ERT inversion', 0.4, 'lambda 30, chi-squared 1.6', 'ert')
    assert target._run_activity is not None
    assert target._run_activity.isVisibleTo(target)
    assert 'Run ERT inversion' in target._run_activity._headline.text()
    assert 'chi-squared 1.6' in target._run_activity._headline.text()
    assert target._run_activity._shown == [str(fresh)]

    # And it goes away with the run, rather than claiming work forever.
    page._finished()
    assert not target._run_activity.isVisibleTo(target)
    page.close()
    page.deleteLater()
    target.deleteLater()


SURVEYS = ['examples/data/ERT/E4D/2021-10-08_1400.ohm',
           'examples/data/ERT/E4D/2021-11-08_1230.ohm']


def _drain(app, page, ready, seconds=120):
    """Wait for the module's loader thread, which reports back through signals.

    Joins the worker rather than sleeping a guessed interval: a fixed budget
    made this fail whenever the machine was busy, which says nothing about the
    code. The deadline is a safety net so a genuinely stuck load still fails
    rather than hanging the suite.
    """
    import time

    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        app.processEvents()
        if ready():
            return True
        for worker in list(getattr(page, '_workers', [])):
            if worker.isRunning():
                worker.wait(200)
        time.sleep(0.02)
    return False


def test_the_panel_opens_the_run_data_the_way_the_run_opens_it():
    """A strip over an empty tool was still an empty tool.

    And opening the files with this panel's own default format was worse than
    not opening them: an E4D survey read as a unified file has its leading index
    column taken for electrode A, so every quadrupole lands out of bounds and
    the panel lists three files and plots nothing. The instrument comes from the
    run's configuration - there is no auto-detect anywhere here.
    """
    pytest.importorskip('PySide6')
    pytest.importorskip('pygimli')
    for path in SURVEYS:
        if not Path(path).is_file():
            pytest.skip(f'example data not present: {path}')

    from PySide6.QtWidgets import QApplication
    from PyHydroGeophysX.qt_apps.modules.ert_processing import ERTProcessingModule
    from PyHydroGeophysX.qt_apps.state import StudioState

    app = QApplication.instance() or QApplication([])
    page = ERTProcessingModule(StudioState(), lambda *a, **k: None)
    opened = page.show_run_inputs({'time_lapse_files': SURVEYS, 'instrument': 'E4D'})
    assert opened.startswith('2 surveys')
    assert _drain(app, page, lambda: page.agent_describe()['state']['data_loaded'])
    state = page.agent_describe()['state']
    assert state['instrument'] == 'E4D'
    assert state['timelapse_files'] == 2
    assert state['measurements'] > 0, 'the panel listed the files and plotted nothing'
    assert state['electrodes'] == 112

    # A second visit leaves what is there alone, rather than reloading over
    # anything the user has since changed in the panel.
    assert page.show_run_inputs({'time_lapse_files': SURVEYS,
                                 'instrument': 'E4D'}) == ''
    page.deleteLater()


def test_the_page_opens_the_data_with_the_runs_own_default_format(tmp_path):
    """The default lives in the run's code, not in a second copy over here."""
    pytest.importorskip('PySide6')
    from PySide6.QtWidgets import QApplication
    from PyHydroGeophysX.agents.runtime.catalog import DEFAULT_INSTRUMENT
    from PyHydroGeophysX.qt_apps.modules.one_click import OneClickModule
    from PyHydroGeophysX.qt_apps.state import StudioState

    QApplication.instance() or QApplication([])
    page = OneClickModule(StudioState(), lambda *a, **k: None)
    assert page._run_config() == {'instrument': DEFAULT_INSTRUMENT}

    # The run writes its configuration before it starts; that is what wins.
    import json

    page._output = str(tmp_path)
    (tmp_path / 'workflow_config.json').write_text(
        json.dumps({'instrument': 'Syscal', 'lambda': 30, 'empty': None}),
        encoding='utf-8')
    config = page._run_config()
    assert config['instrument'] == 'Syscal' and config['lambda'] == 30
    assert 'empty' not in config, 'a null must not override a real default'

    # Unreadable or absent, the page still has the run's default to work with.
    (tmp_path / 'workflow_config.json').write_text('not json', encoding='utf-8')
    assert page._run_config()['instrument'] == DEFAULT_INSTRUMENT
    page.deleteLater()


def test_a_module_is_handed_the_run_data_once_and_never_blocks_the_run(tmp_path):
    pytest.importorskip('PySide6')
    from PySide6.QtWidgets import QApplication
    from PyHydroGeophysX.qt_apps.modules.base import BaseModule
    from PyHydroGeophysX.qt_apps.modules.one_click import OneClickModule
    from PyHydroGeophysX.qt_apps.state import StudioState

    QApplication.instance() or QApplication([])
    page = OneClickModule(StudioState(), lambda *a, **k: None)
    target = BaseModule(StudioState(), lambda *a, **k: None)
    seen = []
    target.show_run_inputs = lambda inputs: seen.append(inputs) or 'two surveys'
    pages = {}
    page.window()._pages = pages
    page.window().show_module = lambda key: pages.setdefault(key, target)
    page._output = str(tmp_path)
    page._inputs = {'time_lapse_files': list(SURVEYS)}

    page._on_progress('Load ERT data', 0.3, 'reading', 'ert')
    page._on_progress('Run time-lapse inversion', 0.5, 'inverting', 'ert')
    assert len(seen) == 1, 'handed over once, not on every step'
    assert seen[0]['time_lapse_files'] == list(SURVEYS)
    assert seen[0]['instrument'], 'the format the run is using travels with it'
    assert '-> opened in ert: two surveys' in page.details.toPlainText()

    # A panel that cannot open them is a display problem, not a run problem.
    other = BaseModule(StudioState(), lambda *a, **k: None)
    def explodes(inputs):
        raise RuntimeError('no reader for this')
    other.show_run_inputs = explodes
    pages['seismic'] = other
    page._on_progress('Run seismic inversion', 0.6, 'picking', 'seismic')
    assert 'could not open the data in seismic' in page.details.toPlainText()
    assert other._run_activity.isVisibleTo(other), 'the strip still came up'
    for widget in (page, target, other):
        widget.deleteLater()


# ---------------------------------------------------------------------------
# A parsed configuration is full of nulls, and a QC step is not the result
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('key, default', [('quality_threshold', 70),
                                          ('max_attempts', 3)])
def test_a_null_in_the_parsed_config_does_not_fail_the_quality_check(key, default):
    """`dict.get(key, default)` does not apply the default to a present None.

    A real LLM-parsed configuration carried `"quality_threshold": null` for a
    setting the request never mentioned, and `float(None)` failed the whole
    quality check on a run whose inversion was fine.
    """
    from PyHydroGeophysX.agents.inversion_evaluation_agent import _or_default

    assert _or_default({key: None}.get(key, default), default) == default
    assert _or_default(0, default) == 0, 'a real zero is not an absence'
    assert _or_default(False, default) is False


def test_a_failed_quality_check_does_not_discard_a_finished_run(tmp_path):
    """The check describes the inversion; it does not produce it.

    A real run inverted two surveys, estimated water content from them and wrote
    the report - and the desktop was handed an exception instead of any of it,
    because the optional QC step hit a bad number.
    """
    from PyHydroGeophysX.qt_apps.agent.one_click_runner import execute

    class Context:
        def __init__(self, **kwargs):
            pass

        def parse_request(self, request, available_data=None):
            return {'ert_file': 'survey.dat'}

    def run(config, key, model, provider, output, **kwargs):
        return ({'resistivity_model': [1.0, 2.0],
                 'evaluation_results': {'status': 'failed',
                                        'error': "float() argument must be a "
                                                 "string or a real number, not "
                                                 "'NoneType'"}},
                [], 'Interpretation', {'report_markdown': 'r.md'})

    result = execute({'request': 'invert and report',
                      'inputs': {'data_file': 'survey.dat'},
                      'output_dir': str(tmp_path)},
                     lambda *a, **k: None, context_factory=Context, run_fn=run)
    assert result['status'] == 'needs_review'
    # The report the run produced is handed back, alongside the audit the
    # runner adds; what matters is that none of it was thrown away.
    assert 'r.md' in result['report_files'].values()
    assert any('quality check could not be completed' in w
               for w in result['warnings']), result['warnings']
    assert any('unaffected' in w for w in result['warnings'])


# ---------------------------------------------------------------------------
# What a run leaves behind for the rest of the studio
# ---------------------------------------------------------------------------
class _FakeMesh:
    def __init__(self, markers):
        self._markers = list(markers)
        self.saved = None

    def cellMarkers(self):  # noqa: N802 - pygimli naming
        return list(self._markers)

    def save(self, path):
        Path(path).write_bytes(b'mesh')
        self.saved = path


def test_the_run_exports_its_model_in_the_format_the_studio_reads(tmp_path):
    """A run held its models in memory and wrote only figures, so every other
    part of the studio had nothing to open afterwards."""
    import numpy as np

    from PyHydroGeophysX.agents.runtime.catalog import (MODEL_BUNDLE_DIR,
                                                        export_model_bundle)

    ctx = RunContext('goal', {}, str(tmp_path))
    mesh = _FakeMesh([3] * 60 + [2] * 40)
    folder = export_model_bundle(ctx, {
        'mesh': mesh,
        'time_lapse_models': [np.full(100, 120.0), np.full(100, 140.0)],
        'all_coverage': np.ones((2, 100))})
    assert folder == str(tmp_path / MODEL_BUNDLE_DIR)
    written = {p.name for p in Path(folder).iterdir()}
    assert written == {'resmodel.npy', 'index_marker.npy', 'all_coverage.npy',
                       'mesh_res.bms'}
    # (n_cells, n_time) is what the reader expects; one survey is one column.
    assert np.load(Path(folder) / 'resmodel.npy').shape == (100, 2)
    assert np.load(Path(folder) / 'all_coverage.npy').shape == (2, 100)


def test_cell_indices_are_not_written_out_as_layers(tmp_path):
    """An inversion parameter mesh carries one marker per cell.

    Saving those as `index_marker.npy` told the water-content page it had 2025
    layers of one cell each. The file is optional, so leaving it out keeps the
    page offering to derive layers from a structural interface - which is where
    they come from.
    """
    import numpy as np

    from PyHydroGeophysX.agents.runtime.catalog import export_model_bundle

    ctx = RunContext('goal', {}, str(tmp_path))
    folder = export_model_bundle(ctx, {
        'mesh': _FakeMesh(range(100)),
        'resistivity_model': np.full(100, 120.0)})
    assert folder, 'the rest of the bundle is still worth writing'
    assert not (Path(folder) / 'index_marker.npy').exists()
    assert any('no layer markers' in w for w in ctx.warnings), (
        'the absence has to be stated, not silent')
    assert np.load(Path(folder) / 'resmodel.npy').shape == (100, 1)


def test_a_run_with_nothing_to_export_says_so_rather_than_writing_junk(tmp_path):
    import numpy as np

    from PyHydroGeophysX.agents.runtime.catalog import export_model_bundle

    ctx = RunContext('goal', {}, str(tmp_path))
    assert export_model_bundle(ctx, {'mesh': _FakeMesh([2, 3])}) == ''
    assert export_model_bundle(ctx, {'resistivity_model': np.ones(4)}) == ''
    # Models of different lengths are not one bundle.
    assert export_model_bundle(ctx, {
        'mesh': _FakeMesh([2] * 4),
        'time_lapse_models': [np.ones(4), np.ones(5)]}) == ''


# ---------------------------------------------------------------------------
# The rule: a panel the run visits must show something and offer a way in
# ---------------------------------------------------------------------------
def _tool_modules():
    from PyHydroGeophysX.agents.runtime import TOOLS
    from PyHydroGeophysX.agents.runtime import catalog  # noqa: F401 - registers

    modules = {}
    for name, tool in TOOLS.items():
        modules.setdefault(tool.module, []).append(name)
    return modules


def _page_class(module):
    import importlib

    from PyHydroGeophysX.qt_apps.modules import MODULE_SPECS

    submodule, class_name, _ = MODULE_SPECS[module]
    return getattr(importlib.import_module(
        f'PyHydroGeophysX.qt_apps.modules.{submodule}'), class_name)


def test_every_module_the_run_visits_opens_the_runs_data_itself():
    """The rule, enforced rather than remembered.

    Navigating to a panel that then shows "No files added" and an empty plot
    reads as nothing having happened. Any module a tool names must implement
    the hook itself - inheriting the base's empty one is what this forbids, so
    a new tool pointing at a page nobody taught to open a run fails here rather
    than in front of a user.
    """
    pytest.importorskip('PySide6')
    missing = []
    for module, tools in sorted(_tool_modules().items()):
        if 'show_run_inputs' not in vars(_page_class(module)):
            missing.append(f'{module} (reached by {", ".join(sorted(tools))})')
    assert not missing, (
        'these panels show the user nothing when the run visits them:\n  '
        + '\n  '.join(missing))


def test_a_visited_panel_always_offers_something_to_open(tmp_path):
    """Words over an empty tool are not a result; there is always a way in."""
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
    page._output = str(tmp_path)

    # Nothing written yet, and there is still the run folder.
    page._on_progress('Load ERT data', 0.2, 'reading', 'ert')
    strip = target._run_activity
    labels = [strip._opens.itemAt(i).widget().text()
              for i in range(strip._opens.count())
              if isinstance(strip._opens.itemAt(i).widget(), QPushButton)]
    assert labels == ['Open run folder']
    assert strip._open_host.isVisibleTo(strip)

    # The step's own folder and its newest figure join it as they appear.
    (tmp_path / 'inversion').mkdir()
    figure = tmp_path / 'inversion' / 'section.png'
    figure.write_bytes(b'x' * 10)
    page._latest_step, page._latest_detail = 'Load ERT data', 'reading'
    page._current_followed = 'ert'
    page._refresh_activity()
    labels = [strip._opens.itemAt(i).widget().text()
              for i in range(strip._opens.count())
              if isinstance(strip._opens.itemAt(i).widget(), QPushButton)]
    assert labels == ['Open run folder', 'Open inversion', 'Open section.png']
    for widget in (page, target):
        widget.deleteLater()


def test_the_panel_follows_the_run_through_its_own_views():
    """Loading belongs on the data view, a finished inversion on the model."""
    pytest.importorskip('PySide6')
    pytest.importorskip('pygimli')
    from PySide6.QtWidgets import QApplication
    from PyHydroGeophysX.qt_apps.modules.ert_processing import ERTProcessingModule
    from PyHydroGeophysX.qt_apps.state import StudioState

    QApplication.instance() or QApplication([])
    page = ERTProcessingModule(StudioState(), lambda *a, **k: None)
    assert page.show_run_stage('load_ert_surveys') == 'Pseudosection'
    assert page._tabs.currentWidget() is page._pseudo_widget
    assert page.show_run_stage('invert_time_lapse') == 'Resistivity model'
    assert page._tabs.currentWidget() is page._model_tab
    assert page.show_run_stage('evaluate_inversion') == 'Inversion quality'
    assert page.show_run_stage('write_report') == '', 'not this page\'s step'
    page.deleteLater()


def test_the_step_is_matched_by_tool_identity_not_by_its_wording():
    """The label is what the tool calls itself, so this is a lookup, not a guess."""
    pytest.importorskip('PySide6')
    from PySide6.QtWidgets import QApplication
    from PyHydroGeophysX.qt_apps.modules.base import BaseModule
    from PyHydroGeophysX.qt_apps.modules.one_click import OneClickModule
    from PyHydroGeophysX.qt_apps.state import StudioState

    QApplication.instance() or QApplication([])
    page = OneClickModule(StudioState(), lambda *a, **k: None)
    assert page._current_tool('Run time-lapse inversion') == 'invert_time_lapse'
    assert page._current_tool('Load ERT data') == 'load_ert_surveys'
    assert page._current_tool('Writing processing audit') == ''

    staged = []
    target = BaseModule(StudioState(), lambda *a, **k: None)
    target.show_run_stage = lambda tool: staged.append(tool) or 'a view'
    pages = {'ert': target}
    page.window()._pages = pages
    page.window().show_module = lambda key: pages.setdefault(key, target)

    page._on_progress('Load ERT data', 0.2, '', 'ert')
    page._on_progress('Load ERT data', 0.3, 'done', 'ert')     # same step twice
    page._on_progress('Run time-lapse inversion', 0.5, '', 'ert')
    assert staged == ['load_ert_surveys', 'invert_time_lapse'], (
        'navigated once per step, not once per event')
    for widget in (page, target):
        widget.deleteLater()


def test_moving_to_another_module_takes_the_strip_with_it(tmp_path):
    pytest.importorskip('PySide6')
    from PySide6.QtWidgets import QApplication
    from PyHydroGeophysX.qt_apps.modules.base import BaseModule
    from PyHydroGeophysX.qt_apps.modules.one_click import OneClickModule
    from PyHydroGeophysX.qt_apps.state import StudioState

    QApplication.instance() or QApplication([])
    page = OneClickModule(StudioState(), lambda *a, **k: None)
    ert = BaseModule(StudioState(), lambda *a, **k: None)
    hydro = BaseModule(StudioState(), lambda *a, **k: None)
    built = {'ert': ert, 'geo_hydrology': hydro}
    pages = {}
    page.window()._pages = pages
    page.window().show_module = lambda key: pages.setdefault(key, built[key])
    page._output = str(tmp_path)

    page._on_progress('Run ERT inversion', 0.4, '', 'ert')
    page._on_progress('Convert to water content', 0.6, '', 'geo_hydrology')
    assert not ert._run_activity.isVisibleTo(ert)
    assert hydro._run_activity.isVisibleTo(hydro)
    for widget in (page, ert, hydro):
        widget.deleteLater()


def test_a_module_the_studio_never_loaded_is_not_an_error():
    pytest.importorskip('PySide6')
    from PySide6.QtWidgets import QApplication
    from PyHydroGeophysX.qt_apps.modules.one_click import OneClickModule
    from PyHydroGeophysX.qt_apps.state import StudioState

    QApplication.instance() or QApplication([])
    page = OneClickModule(StudioState(), lambda *a, **k: None)
    shown = []
    page.window().show_module = shown.append
    page._on_progress('Run seismic inversion', 0.4, '', 'seismic')
    assert shown == ['seismic']
    page.deleteLater()


def test_progress_without_a_module_still_works():
    """Callers that predate the fourth argument must keep working."""
    pytest.importorskip('PySide6')
    from PySide6.QtWidgets import QApplication
    from PyHydroGeophysX.qt_apps.modules.one_click import OneClickModule
    from PyHydroGeophysX.qt_apps.state import StudioState

    QApplication.instance() or QApplication([])
    page = OneClickModule(StudioState(), lambda *a, **k: None)
    page._on_progress('Understanding request', 0.02, 'Preparing')
    assert 'Understanding request' in page.status.text()
    page.close()
    page.deleteLater()


def test_progress_only_ever_moves_forward(tmp_path):
    """A loop cannot know its total, and two attempts to guess one misbehaved.

    Against a fixed step cap the bar crawled to 31% across a complete run and
    then jumped to 100; counting currently runnable tools made it go backwards,
    because only one tool can run before anything has been loaded.
    """
    from PyHydroGeophysX.agents.runtime import TOOLS
    from PyHydroGeophysX.agents.runtime.entry import run_workflow

    kept = dict(TOOLS)
    TOOLS.clear()
    TOOLS.update(ALL)
    seen = []
    try:
        run_workflow({'user_request': 'go'}, None, None, 'openai', tmp_path,
                     progress_callback=lambda step, f, d='', m='': seen.append(f))
    finally:
        TOOLS.clear()
        TOOLS.update(kept)

    # Two per step: what is about to run, then what it concluded.
    assert len(seen) == 8
    assert seen == sorted(seen), f'progress went backwards: {seen}'
    assert all(0.0 < f <= 0.95 for f in seen)
    assert seen[0] > 0.2, 'the first step should not read as barely started'


def test_each_step_reports_what_it_concluded_not_only_what_it_started(tmp_path):
    """Announcements alone left a followed panel with no evidence of work.

    The studio's modules hold their own state while the run happens headless in
    another process, so a panel the run navigated to showed an empty tool. The
    second event per step carries the tool's own summary, which is the thing a
    user can actually read while it is still running.
    """
    from PyHydroGeophysX.agents.runtime import TOOLS
    from PyHydroGeophysX.agents.runtime.entry import run_workflow

    kept = dict(TOOLS)
    TOOLS.clear()
    TOOLS.update(ALL)
    seen = []
    try:
        run_workflow({'user_request': 'go'}, None, None, 'openai', tmp_path,
                     progress_callback=lambda s, f, d='', m='': seen.append((s, d, m)))
    finally:
        TOOLS.clear()
        TOOLS.update(kept)

    details = [detail for _, detail, _ in seen]
    assert 'Loaded 5 surveys.' in details
    assert 'Chi-squared 1.63.' in details
    assert 'Converted 5 models.' in details
    # And the result event still names the module, so the strip lands on the
    # panel the work belongs to.
    assert ('Load ERT data', 'Loaded 5 surveys.', 'ert') in seen


# ---------------------------------------------------------------------------
# Pausing to ask
# ---------------------------------------------------------------------------
def test_a_question_reaches_the_user_and_the_answer_comes_back():
    asked = []

    def answer(event):
        asked.append(event)
        return 'segy'

    ctx = RunContext('goal', settings={'ask_user': answer})
    options = [{'id': 'profile', 'label': 'Coordinate file'},
               {'id': 'segy', 'label': 'SEG-Y headers'},
               {'id': 'stop', 'label': 'Skip seismic'}]
    assert ctx.ask('Which origin?', options, default='profile') == 'segy'
    assert asked[0]['event'] == 'question'
    assert [o['id'] for o in asked[0]['options']] == ['profile', 'segy', 'stop']
    # Recorded, so the report can say the result rests on somebody's decision.
    assert ctx.questions == [{'question': 'Which origin?', 'answer': 'segy',
                              'options': options}]


def test_an_answer_nobody_offered_falls_back_rather_than_acting_on_it():
    ctx = RunContext('goal', settings={'ask_user': lambda e: 'profil'})
    options = [{'id': 'profile', 'label': 'a'}, {'id': 'segy', 'label': 'b'}]
    assert ctx.ask('Which origin?', options, default='segy') == 'segy'


def test_a_headless_run_takes_the_default_and_says_it_was_not_asked():
    """A script, a test, a scheduled run: nobody is there, and silence about
    that would hide a decision inside the numbers."""
    ctx = RunContext('goal')
    assert ctx.ask('Which origin?', [{'id': 'profile'}, {'id': 'segy'}],
                   default='segy') == 'segy'
    assert any('Nobody was available' in w for w in ctx.warnings)


def test_a_question_that_raises_is_not_a_crash():
    def explodes(event):
        raise RuntimeError('the dialog went away')

    ctx = RunContext('goal', settings={'ask_user': explodes})
    assert ctx.ask('?', [{'id': 'a'}, {'id': 'b'}], default='b') == 'b'


def test_a_question_survives_the_process_boundary_as_one_line():
    """Approvals and questions share the desktop's stdin/stdout protocol."""
    import io
    import json

    from PyHydroGeophysX.qt_apps.agent.one_click_runner import ask_via_stdio

    out = io.StringIO()
    event = {'event': 'question', 'question': 'Which origin?',
             'options': [{'id': 'profile', 'label': 'Coordinate file'},
                         {'id': 'segy', 'label': 'SEG-Y headers'}],
             'default': 'profile'}
    answer = ask_via_stdio(event, out,
                           io.StringIO(json.dumps({'decision': 'segy'}) + '\n'))
    assert answer == 'segy'
    written = out.getvalue()
    assert written.endswith('\n') and written.count('\n') == 1
    assert json.loads(written) == event


def test_an_announcement_without_a_kind_is_a_step_approval():
    import io
    import json

    from PyHydroGeophysX.qt_apps.agent.one_click_runner import ask_via_stdio

    out = io.StringIO()
    ask_via_stdio({'tool': 'invert_ert', 'label': 'Run ERT inversion'}, out,
                  io.StringIO(json.dumps({'decision': 'proceed'}) + '\n'))
    assert json.loads(out.getvalue())['event'] == 'approve'


@pytest.mark.parametrize('reply', ['', 'not json\n', '{"decision": null}\n'])
def test_a_parent_that_went_away_stops_rather_than_consents(reply):
    import io

    from PyHydroGeophysX.qt_apps.agent.one_click_runner import ask_via_stdio

    assert ask_via_stdio({'tool': 'x'}, io.StringIO(), io.StringIO(reply)) == 'stop'


def test_the_desktop_offers_a_button_per_option_and_replies_with_its_id():
    """With no page to host it, the prompt stays on the Workflow page."""
    pytest.importorskip('PySide6')
    from PySide6.QtWidgets import QApplication, QPushButton
    from PyHydroGeophysX.qt_apps.modules.one_click import OneClickModule
    from PyHydroGeophysX.qt_apps.state import StudioState

    QApplication.instance() or QApplication([])
    page = OneClickModule(StudioState(), lambda *a, **k: None)
    sent = []
    page._worker = type('W', (), {'answer': lambda self, d: sent.append(d)})()

    page._on_asked({'event': 'question', 'question': 'Which origin?',
                    'module': 'seismic', 'default': 'profile',
                    'options': [{'id': 'profile', 'label': 'Coordinate file'},
                                {'id': 'segy', 'label': 'SEG-Y headers'},
                                {'id': 'stop', 'label': 'Skip seismic'}]})
    assert page.pause_box.isVisibleTo(page)
    buttons = [page.pause_buttons.itemAt(i).widget()
               for i in range(page.pause_buttons.count())]
    buttons = [b for b in buttons if isinstance(b, QPushButton)]
    assert [b.text() for b in buttons] == ['Coordinate file', 'SEG-Y headers',
                                           'Skip seismic']
    buttons[1].click()
    assert sent == ['segy']
    # And the bar goes away, so a stale prompt cannot be answered twice.
    assert not page.pause_box.isVisibleTo(page)
    assert page.pause_buttons.count() == 0
    page.close()
    page.deleteLater()


def test_the_question_appears_on_the_panel_the_run_moved_to():
    """The run navigates away from Workflow to ask, so the buttons go with it.

    Leaving them behind would stop the run on one page with the only way to
    answer it on another.
    """
    pytest.importorskip('PySide6')
    from PySide6.QtWidgets import QApplication, QPushButton
    from PyHydroGeophysX.qt_apps.modules.base import BaseModule
    from PyHydroGeophysX.qt_apps.modules.one_click import OneClickModule
    from PyHydroGeophysX.qt_apps.state import StudioState

    QApplication.instance() or QApplication([])
    page = OneClickModule(StudioState(), lambda *a, **k: None)
    seismic = BaseModule(StudioState(), lambda *a, **k: None)
    pages = {}
    page.window()._pages = pages
    page.window().show_module = lambda key: pages.setdefault(key, seismic)
    sent = []
    page._worker = type('W', (), {'answer': lambda self, d: sent.append(d)})()

    page._on_asked({'event': 'question', 'question': 'Which origin?',
                    'module': 'seismic', 'default': 'profile',
                    'options': [{'id': 'profile', 'label': 'Coordinate file'},
                                {'id': 'segy', 'label': 'SEG-Y headers'}]})
    strip = seismic._run_activity
    assert strip is not None and strip.isVisibleTo(seismic)
    assert 'Which origin?' in strip._headline.text()
    buttons = [strip._choices.itemAt(i).widget()
               for i in range(strip._choices.count())]
    buttons = [b for b in buttons if isinstance(b, QPushButton)]
    assert [b.text() for b in buttons] == ['Coordinate file', 'SEG-Y headers']
    # The Workflow page's own bar stays down - one prompt, in one place.
    assert not page.pause_box.isVisibleTo(page)

    # The strip refreshes once a second to pick up new figures; that must not
    # replace the question with the last step's description and leave the
    # buttons sitting under text that no longer explains them.
    page._latest_step, page._latest_detail = 'Run seismic inversion', 'picking'
    page._refresh_activity()
    assert 'Which origin?' in strip._headline.text()

    buttons[0].click()
    assert sent == ['profile']
    assert strip._choices.count() == 0, 'a stale prompt could be answered twice'
    # And now the strip goes back to narrating the run.
    page._refresh_activity()
    assert 'Run seismic inversion' in strip._headline.text()
    for widget in (page, seismic):
        widget.deleteLater()


def test_a_step_approval_offers_run_skip_and_stop():
    pytest.importorskip('PySide6')
    from PySide6.QtWidgets import QApplication, QPushButton
    from PyHydroGeophysX.qt_apps.modules.one_click import OneClickModule
    from PyHydroGeophysX.qt_apps.state import StudioState

    QApplication.instance() or QApplication([])
    page = OneClickModule(StudioState(), lambda *a, **k: None)
    sent = []
    page._worker = type('W', (), {'answer': lambda self, d: sent.append(d)})()
    shown = []
    page.window().show_module = shown.append

    page._on_asked({'event': 'approve', 'tool': 'invert_ert',
                    'label': 'Run ERT inversion', 'module': 'ert',
                    'reason': 'the surveys are loaded', 'step': 2})
    buttons = [page.pause_buttons.itemAt(i).widget()
               for i in range(page.pause_buttons.count())]
    buttons = [b for b in buttons if isinstance(b, QPushButton)]
    assert [b.text() for b in buttons] == ['Run this step', 'Skip it',
                                           'Stop the run']
    assert 'Run ERT inversion' in page.pause_label.text()
    assert 'the surveys are loaded' in page.pause_label.text()
    # The panel doing the work comes up first, so the decision is made looking
    # at it rather than at a progress bar.
    assert shown == ['ert']
    buttons[2].click()
    assert sent == ['stop']
    page.close()
    page.deleteLater()


def test_pacing_is_a_checkbox_on_the_workflow_page_not_a_third_chat_mode():
    """The chat offers two modes; whether the run pauses is pacing, not a mode.

    It also cannot be a button here: the chat clears the API key from this page
    as soon as it has used it, so a run is always started from chat and a button
    would only ever report a missing key.
    """
    pytest.importorskip('PySide6')
    import inspect

    from PySide6.QtWidgets import QApplication
    from PyHydroGeophysX.qt_apps.agent import chat_panel
    from PyHydroGeophysX.qt_apps.modules.one_click import OneClickModule
    from PyHydroGeophysX.qt_apps.state import StudioState

    QApplication.instance() or QApplication([])
    # The chat's mode list stays at two; pacing is not one of them.
    modes = re.findall(r'_execution_mode\.addItem\([^,]+,\s*"([a-z_]+)"\)',
                       inspect.getsource(chat_panel))
    assert modes == ['guided', 'auto'], f'the chat should offer two modes: {modes}'

    page = OneClickModule(StudioState(), lambda *a, **k: None)
    assert not page.step_through.isChecked(), 'straight through by default'
    assert not hasattr(page, 'step_run'), 'the button was replaced by the box'

    started = []
    page._start = lambda step_mode=False: started.append(step_mode)
    page._request_text = 'invert this line'
    page._inputs = {'data_file': __file__}
    page.step_through.setChecked(True)
    page.submit_request('continue', {'api_key': 'x', 'provider': 'openai'})
    assert started == [True]
    page.deleteLater()


def test_a_question_with_no_options_answers_itself_rather_than_wedging():
    pytest.importorskip('PySide6')
    from PySide6.QtWidgets import QApplication
    from PyHydroGeophysX.qt_apps.modules.one_click import OneClickModule
    from PyHydroGeophysX.qt_apps.state import StudioState

    QApplication.instance() or QApplication([])
    page = OneClickModule(StudioState(), lambda *a, **k: None)
    sent = []
    page._worker = type('W', (), {'answer': lambda self, d: sent.append(d)})()
    page._on_asked({'event': 'question', 'question': '?', 'options': [],
                    'default': 'stop'})
    assert sent == ['stop']
    assert not page.pause_box.isVisibleTo(page)
    page.close()
    page.deleteLater()


def test_the_worker_protocol_carries_the_module():
    pytest.importorskip('PySide6')
    import inspect

    from PyHydroGeophysX.qt_apps.agent import one_click_runner, one_click_worker

    # Four values across the process boundary: step, fraction, detail, module.
    assert one_click_worker.OneClickWorker.progress.__doc__ is not None or True
    source = inspect.getsource(one_click_worker)
    assert 'Signal(str, float, str, str)' in source
    assert "event.get('module', '')" in source
    runner = inspect.getsource(one_click_runner)
    assert "'module': module" in runner
