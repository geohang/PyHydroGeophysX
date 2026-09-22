"""The three-step request ladder shared by the desktop and web AQUAH chats.

Level 1 answers most requests, level 2 handles coding / reasoning / agent work,
level 3 exists for what the levels below could not finish. The registry lives in
:mod:`PyHydroGeophysX.llm.providers` so the Qt panel and the Streamlit sidebar
cannot drift apart; these tests hold that shape, and check each surface actually
resolves a level to a model.
"""

import ast
import os
from pathlib import Path

import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PyHydroGeophysX.llm import providers


TIERED_PROVIDERS = ('openai', 'anthropic')


# -- the registry -------------------------------------------------------------
def test_every_level_names_a_listed_priced_model_on_both_providers():
    for tier_id in providers.TIER_ORDER:
        for provider_id in TIERED_PROVIDERS:
            model = providers.tier_model(tier_id, provider_id)
            assert model, f'{tier_id} names no model for {provider_id}'
            assert model in providers.PROVIDER_META[provider_id]['models'], \
                f'{model} is not selectable in the {provider_id} model list'
            assert providers.price_label(model), f'{model} has no price to show'


def test_the_ladder_gets_more_expensive_going_up():
    # A level that costs no more than the one below it is not an escalation, and
    # the UI copy ("move up when this level cannot finish") would be misleading.
    for provider_id in TIERED_PROVIDERS:
        rates = [providers.MODEL_PRICES_USD_PER_MTOK[providers.tier_model(t, provider_id)]
                 for t in providers.TIER_ORDER]
        assert [r[0] for r in rates] == sorted(r[0] for r in rates)
        assert [r[1] for r in rates] == sorted(r[1] for r in rates)


def test_each_provider_defaults_to_level_one():
    for provider_id in TIERED_PROVIDERS:
        assert (providers.PROVIDER_META[provider_id]['default_model']
                == providers.tier_model('level1', provider_id))


def test_claude_and_anthropic_name_the_same_ladder():
    # The Streamlit sidebar and the agents engine say "claude"; the adapters say
    # "anthropic". Both must resolve, or the web sidebar loses its levels.
    for tier_id in providers.TIER_ORDER:
        assert providers.tier_model(tier_id, 'claude') == providers.tier_model(tier_id, 'anthropic')
    assert providers.tier_for_model('claude', 'claude-opus-5') == 'level3'


def test_a_model_outside_the_ladder_reads_as_custom():
    assert providers.tier_for_model('openai', 'gpt-4o') == providers.TIER_CUSTOM
    assert providers.tier_of_model('gpt-4o') == providers.TIER_CUSTOM


def test_tier_of_model_finds_the_level_whatever_provider_it_came_from():
    assert providers.tier_of_model('claude-sonnet-5') == 'level2'
    assert providers.tier_of_model('gpt-5.6-luna') == 'level1'


def test_a_bring_your_own_endpoint_has_no_ladder():
    assert not providers.provider_has_tiers('openai_compatible')
    assert providers.tier_model('level1', 'openai_compatible') is None


def test_tier_label_carries_the_model_and_its_price():
    label = providers.tier_label('level2', 'openai')
    assert 'Level 2' in label and 'gpt-5.6-terra' in label and '$2.00 / $12.00' in label


# -- the desktop chat panel ---------------------------------------------------
def _panel(provider_id='openai', model=None):
    from PySide6.QtWidgets import QApplication
    from PyHydroGeophysX.qt_apps.agent.chat_panel import AquahChatPanel

    QApplication.instance() or QApplication([])

    class Controller:
        def capabilities_summary(self):
            return ''

    provider = providers.make_provider(provider_id, model=model, api_key='session-key')
    return AquahChatPanel(Controller(), provider=provider)


def _panel_without_key():
    from PyHydroGeophysX.qt_apps.agent.chat_panel import AquahChatPanel

    class Controller:
        def capabilities_summary(self):
            return ''

    return AquahChatPanel(Controller(), provider=providers.make_provider('openai', api_key=''))


def test_chat_panel_opens_on_level_one_and_escalates_to_the_named_model():
    panel = _panel()
    assert panel._level_combo.currentData() == 'level1'
    assert panel._provider.model == 'gpt-5.6-luna'
    panel._level_combo.setCurrentIndex(list(providers.TIER_ORDER).index('level3'))
    assert panel._provider.model == 'gpt-5.6-sol'
    assert panel._model_combo.currentText() == 'gpt-5.6-sol'
    panel.close()
    panel.deleteLater()


def test_chat_panel_keeps_the_chosen_level_across_a_provider_switch():
    panel = _panel()
    panel._level_combo.setCurrentIndex(list(providers.TIER_ORDER).index('level2'))
    panel._provider_combo.setCurrentIndex(panel._provider_combo.findData('anthropic'))
    assert panel._provider.model == 'claude-sonnet-5'
    assert panel._level_combo.currentData() == 'level2'
    panel.close()
    panel.deleteLater()


def test_chat_panel_reports_a_hand_typed_model_as_custom():
    panel = _panel('anthropic')
    panel._model_combo.setCurrentText('claude-sonnet-4-6')
    assert panel._level_combo.currentData() == providers.TIER_CUSTOM
    assert 'Custom model' in panel._level_combo.toolTip()
    panel.close()
    panel.deleteLater()


def test_chat_panel_hides_the_ladder_for_a_bring_your_own_endpoint():
    panel = _panel('openai_compatible')
    # Measured against the settings block, not the panel: the block itself is
    # collapsed by default, so isVisibleTo(panel) would be False either way and
    # the assertion would pass without testing anything.
    assert not panel._level_row.isVisibleTo(panel._settings)
    panel.close()
    panel.deleteLater()


def test_chat_panel_keeps_the_configuration_collapsed_until_asked():
    """Only what changes per message stays on screen; the rest is one click away.

    The settings block is most of the panel's height and is set once a session,
    so the transcript - what the panel is actually for - gets the space instead.
    """
    panel = _panel()
    assert not panel._settings.isVisibleTo(panel)
    assert panel._status_label.isVisibleTo(panel)      # can I send, and at what cost
    assert panel._execution_mode.isVisibleTo(panel)    # the one per-request choice
    panel._settings_btn.setChecked(True)
    assert panel._settings.isVisibleTo(panel)
    assert panel._model_combo.isVisibleTo(panel)
    panel._settings_btn.setChecked(False)
    assert not panel._settings.isVisibleTo(panel)
    panel.close()
    panel.deleteLater()


def test_chat_panel_opens_the_configuration_when_there_is_no_key():
    """A first run must not report "no API key" with no visible way to give one."""
    panel = _panel_without_key()
    assert panel._settings.isVisibleTo(panel)
    assert panel._settings_btn.isChecked()
    panel.close()
    panel.deleteLater()


def test_chat_panel_status_line_quotes_the_price():
    panel = _panel('anthropic')
    # The ready line is what carries the price; the SDK need not be installed
    # here for the panel to render it.
    panel._provider.available = lambda: (True, 'ready')
    panel._refresh_ready_state()
    assert '$1.00 / $5.00' in panel._status_label.text()
    assert 'claude-haiku-4-5' in panel._status_label.text()
    panel.close()


# -- the Streamlit sidebar ----------------------------------------------------
_APP = Path(__file__).resolve().parents[1] / 'examples' / 'app_geophysics_workflow.py'


class _FakePanel:
    """Records what the picker rendered and answers the radio with ``choice``."""

    def __init__(self, choice=None):
        self._choice = choice
        self.options = None
        self.captions = []
        self.text_default = None

    def radio(self, label, options, index, format_func, help):  # noqa: A002
        self.options = [format_func(o) for o in options]
        return options[index] if self._choice is None else self._choice

    def text_input(self, label, value):
        self.text_default = value
        return value

    def caption(self, text):
        self.captions.append(text)


def _picker(saved_model=''):
    """Compile just the sidebar picker out of the Streamlit app.

    The app module is ~12k lines and imports Streamlit and the agents engine at
    import time, so the one function under test is lifted out by AST rather than
    imported. Its only global is ``st``, which is stubbed for the saved model.
    """
    tree = ast.parse(_APP.read_text(encoding='utf-8'))
    fn = next(n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name == '_render_model_level_picker')
    namespace = {'st': type('St', (), {'session_state': type(
        'State', (), {'llm_model': saved_model})()})()}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(_APP), 'exec'), namespace)
    return namespace['_render_model_level_picker']


def test_sidebar_offers_three_levels_and_defaults_to_the_first():
    panel = _FakePanel()
    assert _picker()(panel, 'openai') == 'gpt-5.6-luna'
    assert len(panel.options) == 4  # three levels plus Custom
    assert panel.options[0].startswith('Level 1')
    assert any('$0.20 / $1.20' in c for c in panel.captions)


@pytest.mark.parametrize('level,expected', [
    ('level1', 'claude-haiku-4-5'),
    ('level2', 'claude-sonnet-5'),
    ('level3', 'claude-opus-5'),
])
def test_sidebar_resolves_each_level_on_claude(level, expected):
    assert _picker()(_FakePanel(choice=level), 'claude') == expected


def test_sidebar_custom_level_falls_back_to_typing_a_model_name():
    panel = _FakePanel(choice=providers.TIER_CUSTOM)
    assert _picker('gpt-4o')(panel, 'openai') == 'gpt-4o'
    assert panel.text_default == 'gpt-4o'


def test_sidebar_keeps_a_plain_model_box_for_an_untiered_provider():
    panel = _FakePanel()
    assert _picker()(panel, 'gemini') == 'gemini-2.5-flash'
    assert panel.options is None  # no ladder was offered


def test_reasoning_effort_follows_the_level_that_selected_the_model():
    """Level 1 at medium effort made the cheapest level the slowest.

    Every level-1 model is a reasoning model, and chat drives the studio one
    tool call at a time - each call its own request, each paying a full
    reasoning pass before it can answer "navigate to the ERT module". Effort is
    part of the same cost ladder the levels turn, so it moves with them.
    """
    assert providers.tier_effort('level1') == 'low'
    assert providers.tier_effort('level2') == 'medium'
    assert providers.tier_effort('level3') == 'high'
    assert providers.tier_effort(providers.TIER_CUSTOM) == providers.DEFAULT_EFFORT
    assert providers.tier_effort(None) == providers.DEFAULT_EFFORT

    panel = _panel()
    assert panel._reasoning.currentText() == 'low'
    assert panel._provider.reasoning_effort == 'low'
    for index, expected in enumerate(['low', 'medium', 'high']):
        panel._level_combo.setCurrentIndex(index)
        assert panel._reasoning.currentText() == expected
        assert panel._provider.reasoning_effort == expected
    panel.deleteLater()


def test_an_effort_the_user_chose_is_not_overwritten_by_the_level():
    panel = _panel()
    panel._reasoning.setCurrentText('high')          # on level 1, which wants low
    assert panel._effort_is_user_set
    panel._level_combo.setCurrentIndex(1)
    panel._level_combo.setCurrentIndex(0)
    assert panel._reasoning.currentText() == 'high'
    assert panel._provider.reasoning_effort == 'high'
    panel.deleteLater()


def test_a_level_one_request_does_not_route_through_the_reasoning_endpoint_for_free():
    """The old default was gpt-4.1, which ignores effort entirely.

    Recording the difference: the tier change moved the default onto a model
    whose requests carry a reasoning pass, which is where the extra latency in
    step-by-step chat came from.
    """
    from PyHydroGeophysX.llm.runtime_options import openai_options
    assert openai_options('gpt-4.1', effort='medium') == {'temperature': 0.2}
    assert openai_options('gpt-5.6-luna', effort='low') == {'reasoning_effort': 'low'}
