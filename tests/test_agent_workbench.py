import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

from PyHydroGeophysX.agents.folder_catalog import scan_folder, classify_catalog, catalog_inputs
from PyHydroGeophysX.llm.runtime_options import openai_options


def test_folder_scan_bounds_previews_and_skips_secrets(tmp_path):
    (tmp_path / 'survey.dat').write_text('a b m n rhoa\n' + '1 2 3 4 10\n' * 1000)
    (tmp_path / 'api_key.txt').write_text('secret')
    (tmp_path / '.env').write_text('secret')
    (tmp_path / 'data.sgy').write_bytes(b'\0' * 8000)
    catalog = scan_folder(tmp_path)
    assert len(catalog['files']) == 2
    assert max(len(row['preview']) for row in catalog['files']) <= 1200
    assert scan_folder(tmp_path, limit=1)['warnings']


def test_classifier_does_not_accept_invented_paths_or_indices(tmp_path):
    (tmp_path / 'a.dat').write_text('1 2 3')
    catalog = scan_folder(tmp_path)
    provider = SimpleNamespace(complete=lambda *args: {'content': json.dumps({'files': [
        {'index': 0, 'role': 'electrode_file', 'path': 'invented', 'confidence': .8, 'reason': 'coordinates'}]})})
    classified = classify_catalog(catalog, 'ERT', provider)
    assert classified['files'][0]['path'] == str(tmp_path / 'a.dat')
    provider.complete = lambda *args: {'content': '{"files":[{"index":99,"role":"data_file"}]}'}
    with pytest.raises(ValueError, match='index'):
        classify_catalog(catalog, 'ERT', provider)


def test_catalog_requires_roles_and_does_not_guess_multiple_surveys():
    with pytest.raises(ValueError, match='Assign a role'):
        catalog_inputs([{'role': 'unknown', 'path': 'a.dat'}])
    with pytest.raises(ValueError, match='Multiple'):
        catalog_inputs([{'role': 'data_file', 'path': p} for p in ['a.dat', 'b.dat']])
    assert catalog_inputs([{'role': 'time_lapse_files', 'path': p} for p in ['b.dat', 'a.dat']]) == {'time_lapse_files': ['b.dat', 'a.dat']}


def test_reasoning_uses_compatible_wire_parameters():
    options = openai_options('gpt-5.6-luna', max_tokens=1000, effort='high')
    assert options['reasoning_effort'] == 'high'
    assert 'temperature' not in options and 'max_tokens' not in options
    assert openai_options('gpt-4.1', max_tokens=1000) == {'temperature': .2, 'max_tokens': 1000}


def test_provider_sends_reasoning_and_omits_empty_tool_array():
    from PyHydroGeophysX.llm.providers import make_provider
    sent = {}
    def create(**kwargs):
        sent.update(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='ok', tool_calls=[]))])
    provider = make_provider('openai', model='gpt-5.6-luna', api_key='test')
    provider._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    provider.reasoning_effort = 'high'
    assert provider.complete('system', [{'role': 'user', 'content': 'hello'}], [])['content'] == 'ok'
    assert sent['reasoning_effort'] == 'high'
    assert 'temperature' not in sent and 'tools' not in sent


def test_rag_returns_evidence_locations(tmp_path):
    from PyHydroGeophysX.agents.local_knowledge import retrieve
    source = tmp_path / 'notes.md'
    source.write_text('UniqueSurveyXYZ topography constraint and elevation datum')
    found = retrieve('UniqueSurveyXYZ', [source])
    assert found[0]['source'] == str(source)
    assert found[0]['line'] == 1
    assert 'elevation' in found[0]['excerpt']


def test_seismic_topography_changes_exported_elevation(tmp_path):
    from PyHydroGeophysX.data_processing.survey_geometry import apply_pick_geometry
    from PyHydroGeophysX.data_processing.seismic import FirstBreakPick, first_breaks_to_traveltime
    profile = tmp_path / 'topography.csv'
    profile.write_text('x,z\n0,100\n10,110\n')
    pick = FirstBreakPick(source_id=1, receiver_id=7, time_s=.01, source_x=0,
                         source_z=0, receiver_x=5, receiver_z=0,
                         field_record=1, trace_number=1, trace_index=0, amplitude=1.)
    adjusted = apply_pick_geometry([pick], topography_file=profile)
    assert adjusted[0].receiver_z == 105 and adjusted[0].source_z == 100
    exported = first_breaks_to_traveltime(adjusted, str(tmp_path / 'times.dat'))
    assert '105' in Path(exported).read_text()
    from dataclasses import replace
    with pytest.raises(ValueError, match='outside'):
        apply_pick_geometry([replace(pick, source_x=-1)], topography_file=profile)


def test_audit_contains_hashes_sources_uncertainty_and_only_observed_software(tmp_path):
    from PyHydroGeophysX.agents.workflow_audit import write_report
    data = tmp_path / 'survey.dat'
    data.write_text('data')
    reports = write_report(tmp_path, {'data_file': str(data)}, {'warnings': ['poor fit']},
                           [], 'interpretation', {}, [{'step': 'Solve', 'elapsed_seconds': 1., 'details': 'done'}],
                           {'numpy.array': 2}, [], {'model': 'gpt-5.6-luna'})
    audit = json.loads((tmp_path / 'processing_audit.json').read_text())
    assert len(audit['inputs'][0]['sha256']) == 64
    assert [p['name'] for p in audit['software']] == ['numpy']
    text = Path(reports['report_markdown']).read_text()
    assert 'Uncertainty' in text and 'poor fit' in text and 'survey.dat' in text


def test_mcp_root_and_readonly_tool_policy(tmp_path):
    pytest.importorskip('mcp')
    from PyHydroGeophysX.mcp_server import contained, create_server
    with pytest.raises(ValueError):
        contained(tmp_path, '../outside')
    names = {tool.name for tool in asyncio.run(create_server(tmp_path).list_tools())}
    assert {'list_workflows', 'scan_data_folder', 'search_docs', 'validate_recipe'} <= names
    assert 'run_recipe' not in names
    assert 'run_recipe' in {t.name for t in asyncio.run(create_server(tmp_path, True).list_tools())}


def test_real_mcp_stdio_roundtrip(tmp_path):
    pytest.importorskip('mcp')
    from PyHydroGeophysX.agents.mcp_client import catalog
    result = catalog(tmp_path)
    assert 'ert.single_inversion' in result
