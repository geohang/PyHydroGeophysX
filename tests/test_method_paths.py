"""The non-ERT method paths, and the four bugs that blocked them.

I claimed these could not be verified here because the machine had no seismic,
EM or hydrological-model data. It does: `examples/data` carries a refraction
line, a SkyTEM sounding, MODFLOW and ParFlow outputs, and a SEG-Y volume. Running
them found four real defects, each covered below.

The two that matter most are not in the new runtime at all. They are in agents
that had presumably never been run against the data shipped beside them:
`TDEMAgent` could not read its own example file, and a BERT survey could not be
exported for inversion at all.
"""

from pathlib import Path

import numpy as np
import pytest


DATA = Path('examples/data')
requires_data = pytest.mark.skipif(not DATA.is_dir(),
                                   reason='examples/data is not present')


# ---------------------------------------------------------------------------
# 1. TDEM could not read its own example file
# ---------------------------------------------------------------------------
@requires_data
def test_a_comma_separated_tdem_file_loads():
    """`np.loadtxt` with the whitespace default failed on the shipped CSV."""
    from PyHydroGeophysX.agents.tdem_agent import TDEMAgent
    source = DATA / 'EM' / 'skytem_bhmar_tdem.csv'
    if not source.exists():
        pytest.skip('SkyTEM example not present')
    agent = TDEMAgent.__new__(TDEMAgent)
    agent.name = 'tdem'
    times, dobs, uncertainties = agent._load_tdem_data(str(source))
    assert times.size == dobs.size == uncertainties.size > 0
    assert np.all(times > 0) and np.all(np.isfinite(dobs))


@requires_data
def test_a_second_sounding_is_not_mistaken_for_an_uncertainty():
    """The file holds five soundings; column 2 is another site, not an error bar.

    Treating it as the uncertainty on column 1 weights the inversion by a
    measurement taken eleven kilometres away.
    """
    from PyHydroGeophysX.agents.tdem_agent import TDEMAgent
    source = DATA / 'EM' / 'skytem_bhmar_tdem.csv'
    if not source.exists():
        pytest.skip('SkyTEM example not present')
    agent = TDEMAgent.__new__(TDEMAgent)
    agent.name = 'tdem'
    _, first, uncertainty = agent._load_tdem_data(str(source), sounding=0)
    _, second, _ = agent._load_tdem_data(str(source), sounding=1)
    # dB/dt is of order 1e-9, so allclose's default atol of 1e-8 would call any
    # two of these equal. Compare relatively, with no absolute tolerance.
    close = dict(rtol=1e-6, atol=0.0)
    assert not np.allclose(first, second, **close)        # different soundings
    assert not np.allclose(uncertainty, second, **close)  # neither is the error bar
    assert np.allclose(uncertainty, np.abs(first) * 0.05 + 1e-15, **close)


def test_a_named_uncertainty_column_is_used_as_one(tmp_path):
    from PyHydroGeophysX.agents.tdem_agent import TDEMAgent
    source = tmp_path / 'sounding.csv'
    source.write_text('time_s,dBdt,uncertainty\n1e-5,2e-9,1e-11\n2e-5,1e-9,1e-11\n',
                      encoding='utf-8')
    agent = TDEMAgent.__new__(TDEMAgent)
    agent.name = 'tdem'
    _, dobs, uncertainties = agent._load_tdem_data(str(source))
    assert np.allclose(dobs, [2e-9, 1e-9])
    assert np.allclose(uncertainties, [1e-11, 1e-11])


def test_whitespace_files_without_a_header_still_load(tmp_path):
    """The format the loader used to assume must keep working."""
    from PyHydroGeophysX.agents.tdem_agent import TDEMAgent
    source = tmp_path / 'sounding.dat'
    source.write_text('1e-5 2e-9 1e-11\n2e-5 1e-9 1e-11\n', encoding='utf-8')
    agent = TDEMAgent.__new__(TDEMAgent)
    agent.name = 'tdem'
    times, dobs, uncertainties = agent._load_tdem_data(str(source))
    assert np.allclose(times, [1e-5, 2e-5])
    assert np.allclose(uncertainties, [1e-11, 1e-11])


def test_the_loader_is_defined_once():
    """Two `_load_tdem_data` methods existed; the second silently shadowed the first."""
    import inspect

    from PyHydroGeophysX.agents import tdem_agent
    assert inspect.getsource(tdem_agent).count('def _load_tdem_data') == 1


# ---------------------------------------------------------------------------
# 2. Electrode indices must be 1-based, not merely non-negative
# ---------------------------------------------------------------------------
@requires_data
@pytest.mark.parametrize('name,instrument', [
    ('ERT/DAS/20171106_1417.Data', 'DAS-1'),
    ('ERT/Bert/fielddataline2.dat', 'BERT'),
])
def test_electrode_indices_are_one_based_after_loading(name, instrument):
    """[0..N-1] is non-negative and still wrong: every consumer labels from 1.

    fielddataline2.dat loaded as [0..71] for 72 electrodes, so ResIPy looked up
    the label '0' among its own '1'..'72' and the export died on a bare
    KeyError('0').
    """
    pytest.importorskip('pygimli')
    source = DATA / name
    if not source.exists():
        pytest.skip(f'{name} is not present')
    from PyHydroGeophysX.agents.ert_loader_agent import ERTLoaderAgent

    loaded = ERTLoaderAgent(api_key=None).execute(
        {'data_file': str(source), 'instrument': instrument, 'crs': 'local'})
    assert loaded['status'] == 'success'
    idx = np.array([[o.quad.A, o.quad.B, o.quad.M, o.quad.N]
                    for o in loaded['ert_data'].observations])
    assert idx.min() == 1, 'electrode indices must start at 1'
    assert idx.max() <= len(loaded['ert_data'].electrodes)


# ---------------------------------------------------------------------------
# 3. A file ResIPy cannot pair must fall back, not lose the survey
# ---------------------------------------------------------------------------
@requires_data
def test_a_survey_resipy_cannot_pair_still_exports(tmp_path, capsys):
    """The fallback triggered on the message text, so KeyError('0') went unhandled."""
    pytest.importorskip('pygimli')
    source = DATA / 'ERT' / 'Bert' / 'fielddataline2.dat'
    if not source.exists():
        pytest.skip('BERT example not present')
    from PyHydroGeophysX.agents.ert_loader_agent import ERTLoaderAgent
    from PyHydroGeophysX.data_processing.ert_data_agent import export_for_inversion

    loaded = ERTLoaderAgent(api_key=None).execute(
        {'data_file': str(source), 'instrument': 'BERT', 'crs': 'local'})
    exported = export_for_inversion(loaded['ert_data'], outdir=str(tmp_path),
                                    fmt='pgimli', use_source_error=True)
    assert Path(exported).exists() and Path(exported).stat().st_size > 0
    # And it says what it did rather than quietly changing the error model.
    assert 'full-dataset export' in capsys.readouterr().out


# ---------------------------------------------------------------------------
# 4. Adapters between agents that disagree about shape
# ---------------------------------------------------------------------------
def test_one_seismic_interface_is_handed_over_as_a_pair():
    """SeismicAgent returns {threshold: {'x','z'}}; StructureConstraintAgent
    unpacks two values. Passing the dict failed with "expected 2, got 1"."""
    from PyHydroGeophysX.agents.runtime.catalog import interface_coords
    result = {'interfaces': {1200.0: {'x': [0, 1, 2], 'z': [-1, -2, -3]},
                             800.0: {'x': [0, 1], 'z': [-4, -5]}}}
    assert interface_coords(result, 1200.0) == ([0, 1, 2], [-1, -2, -3])
    assert interface_coords(result, '800') == ([0, 1], [-4, -5])   # keys vary in type
    assert interface_coords(result, 999.0) is None                 # two present, no match
    assert interface_coords({'interfaces': {}}, 1200.0) is None


def test_the_fusion_pattern_follows_what_the_run_produced():
    """Configured methods said what was hoped for; a fusion needs what exists."""
    from PyHydroGeophysX.agents.runtime.catalog import available_methods, fusion_pattern
    from PyHydroGeophysX.agents.runtime import RunContext

    ctx = RunContext('combine everything')
    ctx.put('seismic_results', {'velocity_model': [1]})
    ctx.put('inversion_results', {'final_models': [1]})
    assert available_methods(ctx) == ['seismic', 'ert']
    assert fusion_pattern(available_methods(ctx)) == 'structure_constraint'

    ctx.put('water_content', [{}])
    assert fusion_pattern(available_methods(ctx)) == 'full_integration'
    # One method satisfies no pattern, so the tool must not be offered at all.
    assert fusion_pattern(['ert']) is None


def test_fusion_is_not_offered_when_no_pattern_can_be_satisfied():
    from PyHydroGeophysX.agents.runtime import RunContext, TOOLS
    from PyHydroGeophysX.agents.runtime import catalog  # noqa: F401 - registers
    from PyHydroGeophysX.agents.runtime.tools import runnable_tools

    ctx = RunContext('invert this line', {'ert_file': 'a.dat'})
    ctx.put('ert_data', [object()])
    ctx.put('inversion_results', {'resistivity_model': [1]})
    assert 'fuse_methods' not in {t.name for t in runnable_tools(ctx, TOOLS)}
    ctx.put('seismic_results', {'velocity_model': [1]})
    assert 'fuse_methods' in {t.name for t in runnable_tools(ctx, TOOLS)}


def test_a_loaded_survey_is_converted_before_it_reaches_pygimli_consumers(tmp_path):
    """StandardERT is not subscriptable; several agents index the container."""
    from PyHydroGeophysX.agents.runtime.catalog import as_pygimli_data

    class AlreadyPyGimli:
        def sensorCount(self):
            return 24

    assert as_pygimli_data(AlreadyPyGimli(), tmp_path).sensorCount() == 24
    with pytest.raises(ValueError, match='No ERT survey'):
        as_pygimli_data(None, tmp_path)


# ---------------------------------------------------------------------------
# The paths themselves, against the shipped data
# ---------------------------------------------------------------------------
@requires_data
@pytest.mark.parametrize('name,paths,settings', [
    ('seismic', {'seismic_file': 'Seismic/srtfieldline2.dat'},
     {'velocity_threshold': 1200.0}),
    ('modflow', {'modflow_dir': 'modflow'}, {'hydro_model': 'modflow'}),
    ('parflow', {'parflow_dir': 'parflow/test2'}, {'hydro_model': 'parflow'}),
])
def test_each_method_path_runs_against_the_shipped_data(name, paths, settings,
                                                        tmp_path):
    """Each was registered as a tool; none had been run end to end."""
    pytest.importorskip('pygimli')
    from PyHydroGeophysX.agents.runtime import RunContext, run_controller
    from PyHydroGeophysX.agents.runtime import catalog  # noqa: F401 - registers

    # Paths and plain settings are kept apart: resolving every value against
    # examples/ turned hydro_model='modflow' into a directory path, and the
    # agent then reported no model at all.
    resolved = {key: str(DATA / value) for key, value in paths.items()}
    absent = [value for value in resolved.values() if not Path(value).exists()]
    if absent:
        pytest.skip(f'{absent} not present')

    config = {'user_request': f'run the {name} path', **resolved, **settings}
    ctx = RunContext(config['user_request'], config, str(tmp_path))
    run_controller(ctx)
    assert ctx.steps, 'no tool was offered for this configuration'
    failures = [f'{s.tool}: {s.error}' for s in ctx.steps if s.status == 'failed']
    assert not failures, failures


# ---------------------------------------------------------------------------
# 5. Survey geometry files with a header
# ---------------------------------------------------------------------------
@requires_data
def test_the_shipped_geometry_file_parses():
    """Its header is `station distance_m elevation_m`, which was not on the
    two-entry allowlist the reader matched against, so line 1 was an error."""
    from PyHydroGeophysX.data_processing.survey_geometry import read_coordinates
    source = DATA / 'Seismic' / 'location.txt'
    if not source.exists():
        pytest.skip('location.txt not present')
    rows = read_coordinates(str(source), 3)
    assert rows.shape == (24, 3)
    assert rows[0][0] == 1 and rows[-1][0] == 24


def test_a_header_is_recognised_by_shape_not_by_wording(tmp_path):
    from PyHydroGeophysX.data_processing.survey_geometry import read_coordinates
    for header in ('x z', 'distance_m elevation_m', 'Easting Height'):
        path = tmp_path / 'geom.txt'
        path.write_text(f'{header}\n0 100\n10 101\n', encoding='utf-8')
        assert read_coordinates(str(path), 2).shape == (2, 2)


def test_a_second_non_numeric_row_is_still_an_error(tmp_path):
    """One header is a convention; two means the file is malformed."""
    from PyHydroGeophysX.data_processing.survey_geometry import read_coordinates
    path = tmp_path / 'geom.txt'
    path.write_text('x z\n0 100\nbroken row\n20 102\n', encoding='utf-8')
    with pytest.raises(ValueError, match='invalid geometry row'):
        read_coordinates(str(path), 2)


def test_a_header_with_the_wrong_number_of_columns_is_an_error(tmp_path):
    from PyHydroGeophysX.data_processing.survey_geometry import read_coordinates
    path = tmp_path / 'geom.txt'
    path.write_text('x y z\n0 100\n10 101\n', encoding='utf-8')
    with pytest.raises(ValueError, match='invalid geometry row'):
        read_coordinates(str(path), 2)


def test_geometry_outside_the_profile_says_which_numbers_disagree(tmp_path):
    """The bare message sent a reader looking for a unit problem when the
    cause was an origin offset of one station spacing.

    Checked against the whole survey up front rather than through whichever
    pick happens to fall off the end: the offset is systematic, and a tolerance
    for off-end shots would otherwise swallow it.
    """
    from PyHydroGeophysX.data_processing.survey_geometry import (
        OriginMismatch, apply_pick_geometry)

    picks, geom = _offset_picks_and_file(tmp_path)
    with pytest.raises(OriginMismatch) as caught:
        apply_pick_geometry(picks, geophone_file=geom)
    message = str(caught.value)
    assert 'x=2 to 4' in message and '2 m away' in message
    assert 'source x=0 to 0' in message
    assert 'do not share an origin' in message
    # And it carries the offset, so a caller can put the choice to someone who
    # knows the survey instead of throwing the picking work away.
    assert caught.value.shift == pytest.approx(2.0)


def test_a_receiver_off_the_profile_is_still_an_error(tmp_path):
    """Distinct from an origin offset: the two files do not describe one line."""
    from PyHydroGeophysX.data_processing.survey_geometry import (
        OriginMismatch, apply_pick_geometry)

    profile = tmp_path / 'topo.csv'
    profile.write_text('x,z\n0,100\n10,110\n', encoding='utf-8')
    picks, _ = _offset_picks_and_file(tmp_path)
    picks[0].receiver_x = -5.0
    with pytest.raises(OriginMismatch, match='Receiver x lies outside'):
        apply_pick_geometry(picks, topography_file=str(profile))


def test_a_shot_off_the_end_of_the_spread_is_ordinary_and_says_so(tmp_path):
    """Off-end shots are how refraction surveys are acquired.

    Refusing them killed the run on the SEG-Y shipped in examples/data, where
    the last shot sits exactly one station beyond the last geophone.
    """
    from PyHydroGeophysX.data_processing.survey_geometry import apply_pick_geometry

    picks, geom = _offset_picks_and_file(tmp_path)
    picks[1].source_x = 4.0        # one spacing past the last station at x=4
    notes = []
    out = apply_pick_geometry(picks, geophone_file=geom, align_origin='profile',
                              warn=notes.append)
    assert [p.source_x for p in out] == pytest.approx([2.0, 6.0])
    # The elevation of the station it sits beyond, not an extrapolated one.
    assert out[1].source_z == pytest.approx(211.0)
    assert notes and 'off-end acquisition' in notes[0]


def test_a_shot_far_past_the_spread_is_not_ordinary(tmp_path):
    from PyHydroGeophysX.data_processing.survey_geometry import (
        OriginMismatch, apply_pick_geometry)

    picks, geom = _offset_picks_and_file(tmp_path)
    picks[1].source_x = 40.0
    with pytest.raises(OriginMismatch, match='station spacing'):
        apply_pick_geometry(picks, geophone_file=geom, align_origin='profile')


def _offset_picks_and_file(tmp):
    """One station of offset: the file starts at x=2, the SEG-Y at x=0."""
    from dataclasses import dataclass

    @dataclass
    class Pick:
        receiver_id: int = 1
        receiver_x: float = 0.0
        receiver_z: float = 0.0
        source_x: float = 0.0
        source_z: float = 0.0
        traveltime: float = 0.01

    geom = Path(tmp) / 'geom.txt'
    geom.write_text('station distance_m elevation_m\n1 2 212\n2 4 211\n',
                    encoding='utf-8')
    return [Pick(receiver_id=1, source_x=0.0), Pick(receiver_id=2, source_x=0.0)], str(geom)


@pytest.mark.parametrize('origin, expected_source_x, expected_receiver_x', [
    ('profile', 2.0, [2.0, 4.0]),   # shots move into the file's frame
    ('segy', 0.0, [0.0, 2.0]),      # the file moves into the SEG-Y frame
])
def test_aligning_the_origin_is_one_rigid_translation(tmp_path, origin,
                                                      expected_source_x,
                                                      expected_receiver_x):
    """Both answers give the same survey; only the x labels differ.

    Which matters exactly once - when the section is laid beside an ERT line -
    which is why it is put to the user rather than picked here.
    """
    from PyHydroGeophysX.data_processing.survey_geometry import apply_pick_geometry

    picks, geom = _offset_picks_and_file(tmp_path)
    out = apply_pick_geometry(picks, geophone_file=geom, align_origin=origin)
    assert [p.receiver_x for p in out] == pytest.approx(expected_receiver_x)
    assert [p.source_x for p in out] == pytest.approx([expected_source_x] * 2)
    # The spacing - the part that is physics rather than labelling - survives.
    assert out[1].receiver_x - out[0].receiver_x == pytest.approx(2.0)


def test_the_seismic_tool_asks_which_origin_and_uses_the_answer(tmp_path,
                                                                monkeypatch):
    """The run has already picked first breaks by the time it finds out;
    asking beats both failing and guessing."""
    from PyHydroGeophysX.agents.runtime import RunContext
    from PyHydroGeophysX.agents.runtime.catalog import _invert_seismic

    calls = []

    class FakeAgent:
        def __init__(self, **kwargs):
            pass

        def execute(self, inputs):
            calls.append(inputs.get('align_origin'))
            if not inputs.get('align_origin'):
                return {'status': 'failed', 'error': 'outside the profile',
                        'origin_mismatch': {'shift': 2.0}}
            return {'status': 'success', 'n_data': 120,
                    'velocity_range': (400.0, 3100.0)}

    import PyHydroGeophysX.agents.seismic_agent as seismic_agent
    monkeypatch.setattr(seismic_agent, 'SeismicAgent', FakeAgent)

    asked = []

    def answer(event):
        asked.append(event)
        return 'segy'

    ctx = RunContext('process the seismic line',
                     {'raw_seismic_file': 'line.sgy', 'geophone_file': 'g.txt'},
                     str(tmp_path), {'ask_user': answer})
    summary, outputs = _invert_seismic(ctx)

    # Asked once, then re-ran with the answer rather than failing.
    assert calls == [None, 'segy']
    assert outputs['seismic_results']['status'] == 'success'
    assert '2 m apart' in asked[0]['question']
    assert [o['id'] for o in asked[0]['options']] == ['profile', 'segy', 'stop']
    # The choice is on the record, because the coordinates now depend on it.
    assert any('SEG-Y header origin' in w for w in ctx.warnings)
    assert ctx.questions[0]['answer'] == 'segy'


def test_declining_to_choose_leaves_seismic_out_rather_than_guessing(tmp_path,
                                                                     monkeypatch):
    from PyHydroGeophysX.agents.runtime import RunContext
    from PyHydroGeophysX.agents.runtime.catalog import _invert_seismic

    class FakeAgent:
        def __init__(self, **kwargs):
            pass

        def execute(self, inputs):
            return {'status': 'failed', 'error': 'outside the profile',
                    'origin_mismatch': {'shift': 2.0}}

    import PyHydroGeophysX.agents.seismic_agent as seismic_agent
    monkeypatch.setattr(seismic_agent, 'SeismicAgent', FakeAgent)

    ctx = RunContext('process the seismic line', {'raw_seismic_file': 'line.sgy'},
                     str(tmp_path), {'ask_user': lambda event: 'stop'})
    with pytest.raises(ValueError, match='disagree about the origin'):
        _invert_seismic(ctx)
