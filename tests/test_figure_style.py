"""One visual style across every figure, and the reader's control over it.

A single five-survey report shipped three figure sizes, six font sizes, four
ways of naming the same survey and three colormaps including `jet`. These pin
the style down and cover the two ways it can be changed: explicitly through the
configuration, and by asking for it in the request.
"""

import re
from pathlib import Path

import pytest

from PyHydroGeophysX.agents import _figstyle as figstyle


# ---------------------------------------------------------------------------
# One survey has one name
# ---------------------------------------------------------------------------
def test_a_survey_is_named_the_same_way_everywhere():
    """Baseline / Baseline (t=0) / Time Step 2 / Survey 2 all meant one thing."""
    dates = ['2017-11-05', '2017-11-06', '2017-11-07']
    assert figstyle.survey_title(0, dates) == 'Survey 1 (baseline)\n2017-11-05'
    assert figstyle.survey_title(2, dates) == 'Survey 3\n2017-11-07'
    # The number matches the report tables, which count from the baseline.
    assert figstyle.change_title(2, dates) == 'Survey 3 - Survey 1\n2017-11-07'


def test_a_missing_date_leaves_the_title_shorter_rather_than_wrong():
    assert figstyle.survey_title(1) == 'Survey 2'
    assert figstyle.survey_title(5, ['2017-11-05']) == 'Survey 6'
    assert figstyle.change_title(1, ['a', '']) == 'Survey 2 - Survey 1'


# ---------------------------------------------------------------------------
# One size, capped
# ---------------------------------------------------------------------------
def test_every_figure_is_sized_by_the_same_rule():
    style = figstyle.FigureStyle()
    one = style.figure_size(1)
    three = style.figure_size(3)                    # below the width cap
    assert three[0] == pytest.approx(one[0] * 3)    # width scales with panels
    assert three[1] == one[1]                       # height does not
    # And a five-panel row is the same shape as any other five-panel row,
    # which is what the three different figsizes used to prevent.
    assert style.figure_size(5) == style.figure_size(5)
    assert style.figure_size(5)[1] == one[1]


def test_a_wide_figure_is_capped_rather_than_left_to_grow():
    """Twelve panels at 'large' would be 8400 pixels across at 300 dpi."""
    style = figstyle.style_from_config({'figure_style': {'size': 'large'}})
    width, _ = style.figure_size(12)
    assert width == figstyle.MAX_FIGURE_WIDTH_IN


# ---------------------------------------------------------------------------
# The reader can change it, within limits
# ---------------------------------------------------------------------------
def test_the_configuration_can_change_the_style():
    style = figstyle.style_from_config({'figure_style': {
        'size': 'compact', 'dpi': 150, 'font': 'DejaVu Sans',
        'change_cmap': 'coolwarm', 'colorbar': 'horizontal'}})
    assert (style.panel_width, style.panel_height) == figstyle.SIZES['compact']
    assert style.dpi == 150 and style.font_family == 'DejaVu Sans'
    assert style.cmap_for('change') == 'coolwarm'
    assert style.colorbar_orientation == 'horizontal'


@pytest.mark.parametrize('bad', [
    {'size': 'enormous'},          # not a named size
    {'size': 99},                  # outside the sane range
    {'dpi': 5},                    # would be unreadable
    {'dpi': 5000},                 # would be a 100-megapixel figure
    {'resistivity_cmap': 'jet'},   # not perceptually uniform
    {'resistivity_cmap': 'not-a-colormap'},
    {'colorbar': 'diagonal'},
    'not a dict',
])
def test_an_unusable_preference_is_dropped_not_raised(bad):
    """A style is a preference; an invalid one must not fail a finished run.

    An invented colormap would otherwise raise inside matplotlib, several
    steps from anything that could explain it.
    """
    assert figstyle.style_from_config({'figure_style': bad}) == figstyle.FigureStyle()


def test_jet_is_not_offered_for_any_quantity():
    """It is not perceptually uniform: it invents banding readers see as structure."""
    assert all('jet' not in maps for maps in figstyle.COLORMAPS.values())


def test_a_change_keeps_a_diverging_colormap():
    """A change plot needs a map whose midpoint is zero; a sequential one hides sign."""
    for name in figstyle.COLORMAPS['change']:
        assert name not in figstyle.COLORMAPS['resistivity']


# ---------------------------------------------------------------------------
# The request can change it too
# ---------------------------------------------------------------------------
def test_the_request_can_ask_for_a_different_size_or_colormap():
    from PyHydroGeophysX.agents._figures import llm_figure_topics

    reply = ('{"topics": ["resistivity"], "only_these": false, '
             '"style": {"size": "large", "colormap": "cividis"}}')
    asked = llm_figure_topics('these plots are tiny, and use cividis',
                              lambda prompt: reply)
    assert asked['style'] == {'size': 'large', 'resistivity_cmap': 'cividis',
                              'water_content_cmap': 'cividis'}
    style = figstyle.style_from_config({'figure_style': asked['style']})
    assert style.panel_width == figstyle.SIZES['large'][0]
    assert style.cmap_for('resistivity') == 'cividis'
    # A change plot keeps its diverging map: a sequential one would hide the sign.
    assert style.cmap_for('change') == figstyle.COLORMAPS['change'][0]


def test_an_explicit_setting_outranks_a_sentence():
    from PyHydroGeophysX.agents.report_agent import ReportAgent

    agent = ReportAgent.__new__(ReportAgent)
    agent.execution_log = []
    agent.name = 'ReportAgent'
    agent.api_key = 'present'
    agent.query_llm = lambda prompt, **kw: (
        '{"topics": ["resistivity"], "style": {"size": "large"}}')
    config = {'user_request': 'make them bigger',
              'figure_style': {'size': 'compact'}}
    agent._read_figure_request(config)
    assert config['figure_style']['size'] == 'compact'


def test_a_request_the_model_cannot_read_leaves_the_style_alone():
    from PyHydroGeophysX.agents.report_agent import ReportAgent

    agent = ReportAgent.__new__(ReportAgent)
    agent.execution_log = []
    agent.name = 'ReportAgent'
    agent.api_key = 'present'

    def refuses(prompt, **kw):
        raise RuntimeError('rate limited')

    agent.query_llm = refuses
    config = {'user_request': 'anything'}
    assert agent._read_figure_request(config) is None
    assert 'figure_style' not in config


# ---------------------------------------------------------------------------
# No generator keeps its own decisions
# ---------------------------------------------------------------------------
def test_no_time_lapse_figure_hard_codes_its_own_style():
    """Each generator used to choose its own size, fonts, colormap and titles."""
    source = Path('PyHydroGeophysX/agents/report_agent.py').read_text(encoding='utf-8')
    start = source.index('def _generate_timelapse_visualizations')
    end = source.index('def _generate_timelapse_narrative')
    block = source[start:end]

    assert 'figsize=' not in block, 'a figure still sets its own size'
    assert "cMap='" not in block, 'a figure still names its own colormap'
    assert 'fontfamily=' not in block, 'a figure still sets its own font'
    assert 'dpi=' not in block, 'a figure still chooses its own resolution'
    # Every remaining font size is read from the style rather than written out.
    for match in re.finditer(r'fontsize=([^,)\n]+)', block):
        assert match.group(1).startswith('style.'), match.group(0)
