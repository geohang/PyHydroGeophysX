"""Local attachments reach workflow inputs and cited RAG context without model calls."""
import os
import time
from dataclasses import replace
from types import SimpleNamespace
from zipfile import ZipFile

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import pytest


@pytest.fixture
def app():
    pytest.importorskip('PySide6')
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture
def controller(app, tmp_path, monkeypatch):
    from PySide6.QtCore import QObject
    from PyHydroGeophysX.agents import assistants
    from PyHydroGeophysX.qt_apps.agent.controller import StudioController
    from PyHydroGeophysX.qt_apps.modules.one_click import OneClickModule
    from PyHydroGeophysX.qt_apps.state import StudioState
    assistant = assistants.get_assistant('aquah')
    monkeypatch.setattr(assistants, 'active', lambda: assistant)
    window = QObject()
    window.state = StudioState(output_dir=tmp_path)
    page = OneClickModule(window.state, lambda *a: None)
    page.set_assistant(assistant)
    window._pages = {'one_click': page}
    ctl = StudioController(window)
    yield ctl
    page.close()
    page.deleteLater()
    app.processEvents()


def test_data_attachments_are_workflow_inputs_and_preserve_survey_order(controller, tmp_path):
    first, second = tmp_path/'first.dat', tmp_path/'second.dat'
    first.write_text('survey 1')
    second.write_text('survey 2')
    assert controller.add_chat_attachments('time_lapse_files', [first, second])['status'] == 'ok'
    assert controller.add_chat_attachments('time_lapse_files', [first])['status'] == 'ok'
    page = controller._attachment_page()
    assert page._inputs['time_lapse_files'] == [str(first), str(second)]
    assert controller.remove_chat_attachment('time_lapse_files', str(first))['status'] == 'ok'
    assert page._inputs['time_lapse_files'] == [str(second)]
    assert first.read_text() == 'survey 1'
    page._worker = object()
    assert controller.add_chat_attachments('data_file', [first])['status'] == 'failed'
    page._worker = None


def test_reference_scope_deduplication_and_atomic_validation(controller, tmp_path, monkeypatch):
    from PyHydroGeophysX.agents import assistants
    first = tmp_path/'geology.txt'
    first.write_text('zorplith mineral deposits')
    assert controller.add_chat_attachments('chat_reference', [first, first])['status'] == 'ok'
    assert len(controller.chat_attachments()) == 1
    assert controller.add_chat_attachments('chat_reference', [first, tmp_path/'missing.txt'])['status'] == 'failed'
    assert len(controller.chat_attachments()) == 1
    controller._window.state.output_dir = tmp_path/'other-project'
    assert controller.chat_attachments() == []
    controller._window.state.output_dir = tmp_path
    assistant = replace(assistants.get_assistant('aquah'), key='geosage', retrieval=())
    monkeypatch.setattr(assistants, 'active', lambda: assistant)
    assert controller.chat_attachments() == []
    assert controller.add_chat_attachments('chat_reference', [first])['status'] == 'ok'
    assert controller.remove_chat_attachment('chat_reference', str(first))['status'] == 'ok'
    assert first.exists()


def test_reference_validation_does_not_partially_add_workflow_data(controller, tmp_path):
    file = tmp_path/'survey.dat'
    file.write_text('survey')
    assert controller.add_chat_attachments('data_file', [file], use_rag=True)['status'] == 'failed'
    assert controller._attachment_page()._inputs == {}
    assert controller.add_chat_attachments('wrong_role', [file])['status'] == 'failed'


def test_optional_geosage_roles_are_discovered_and_used(controller, tmp_path, monkeypatch):
    from PyHydroGeophysX.agents import assistants
    assistant = replace(assistants.get_assistant('aquah'), key='geosage',
                        input_roles=(('Configuration', 'config_file'), ('Gravity', 'gravity_file')),
                        ordered_roles=(), retrieval=())
    monkeypatch.setattr(assistants, 'active', lambda: assistant)
    controller._attachment_page().set_assistant(assistant)
    file = tmp_path/'config.json'
    file.write_text('{}')
    assert controller.add_chat_attachments('config_file', [file])['status'] == 'ok'
    assert controller._attachment_page()._inputs['config_file'] == str(file)
    assert controller.add_chat_attachments('data_file', [file])['status'] == 'failed'


def test_text_word_and_chinese_retrieval_have_sources_and_invalidate_cache(tmp_path):
    from PyHydroGeophysX.agents.local_knowledge import retrieve, format_context
    file = tmp_path/'geology.docx'
    with ZipFile(file, 'w') as archive:
        archive.writestr('word/document.xml', '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>zorplith 蛇纹岩地质模型</w:t></w:r></w:p></w:body></w:document>')
    sources = retrieve('zorplith 蛇纹岩', [file])
    assert sources[0]['source'] == str(file)
    assert sources[0]['line'] == 1
    assert 'not instructions' in format_context(sources)
    text = tmp_path/'notes.txt'
    text.write_text('zorplith old unit')
    assert 'old unit' in retrieve('zorplith', [text])[0]['excerpt']
    text.write_text('zorplith revised geological unit')
    assert 'revised' in retrieve('zorplith', [text])[0]['excerpt']


def test_pdf_page_citations_and_unreadable_diagnostics(tmp_path):
    pypdf = pytest.importorskip('pypdf')
    from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject
    from PyHydroGeophysX.agents.local_knowledge import retrieve
    writer = pypdf.PdfWriter()
    page = writer.add_blank_page(width=500, height=500)
    font = DictionaryObject({NameObject('/Type'): NameObject('/Font'),
                             NameObject('/Subtype'): NameObject('/Type1'),
                             NameObject('/BaseFont'): NameObject('/Helvetica')})
    page[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'): DictionaryObject({NameObject('/F1'): font})})
    stream = DecodedStreamObject()
    stream.set_data(b'BT /F1 12 Tf 20 100 Td (zorplith mineral deposit) Tj ET')
    page[NameObject('/Contents')] = stream
    file = tmp_path/'paper.pdf'
    writer.write(file)
    sources = retrieve('zorplith', [file])
    assert sources[0]['source'] == str(file) and sources[0]['page'] == 1
    bad = tmp_path/'bad.pdf'
    bad.write_bytes(b'not a PDF')
    diagnostics = []
    retrieve('zorplith', [bad], diagnostics=diagnostics)
    assert any('bad.pdf' in warning for warning in diagnostics)
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=500, height=500)
    blank = tmp_path/'scan.pdf'
    writer.write(blank)
    diagnostics = []
    retrieve('zorplith', [blank], diagnostics=diagnostics)
    assert any('OCR' in warning for warning in diagnostics)


@pytest.mark.parametrize('mode', ['guided', 'auto'])
@pytest.mark.parametrize('key', ['aquah', 'geosage'])
def test_chat_reference_excerpts_reach_both_assistants_and_modes(app, controller, tmp_path, monkeypatch, mode, key):
    from PySide6.QtTest import QTest
    from PyHydroGeophysX.agents import assistants
    from PyHydroGeophysX.qt_apps.agent import chat_panel
    from PyHydroGeophysX.llm.providers import make_provider
    assistant = replace(assistants.get_assistant('aquah'), key=key,
                        retrieval=() if key == 'geosage' else ('rag', 'mcp'))
    monkeypatch.setattr(assistants, 'active', lambda: assistant)
    monkeypatch.setattr(chat_panel, 'prewarm', lambda provider: None)
    provider = make_provider('openai')
    monkeypatch.setattr(provider, 'available', lambda: (True, ''))
    file = tmp_path/'paper.txt'
    file.write_text('zorplith mineral geology')
    controller.add_chat_attachments('chat_reference', [file])
    panel = chat_panel.AssistantChatPanel(controller, provider=provider)
    panel._rag.setChecked(True)
    assert panel._rag.isEnabled()
    panel._execution_mode.setCurrentIndex(panel._execution_mode.findData(mode))
    sent = []
    monkeypatch.setattr(panel, '_start_request', lambda: sent.append(panel._messages[-1]))
    monkeypatch.setattr(controller, 'run_to_report', lambda text, settings, *a, **kw:
                        sent.append((text, settings)) or 'Finished')
    panel._input.setPlainText('Explain zorplith geology')
    panel._on_send()
    for _ in range(150):
        if sent and not panel._reference_workers:
            break
        time.sleep(0.02)
        app.processEvents()
    assert sent, (panel._transcript.toPlainText(), len(panel._reference_workers), panel._busy)
    if mode == 'auto':
        request, settings = sent[0]
        assert request == 'Explain zorplith geology'
        assert 'zorplith mineral geology' in settings['chat_context']
        assert settings['chat_reference_sources'][0]['source'] == str(file)
        assert settings['use_rag'] is (key == 'aquah')
    else:
        assert 'zorplith mineral geology' in sent[0]['content']
        assert str(file) in sent[0]['content']
    assert 'RAG sources' in panel._transcript.toPlainText()
    assert not hasattr(panel, '_attach_btn')
    panel._set_busy(False)
    sent.clear()
    panel._input.setPlainText('continue')
    panel._on_send()
    for _ in range(150):
        if sent and not panel._reference_workers:
            break
        time.sleep(0.02)
        app.processEvents()
    assert sent
    if mode == 'auto':
        assert sent[0][0] == 'continue'
        assert 'zorplith mineral geology' in sent[0][1]['chat_context']
    else:
        assert 'zorplith mineral geology' in sent[0]['content']
    panel._set_busy(False)
    panel.close()
    panel.deleteLater()
    app.processEvents()


def test_rag_off_sends_paths_without_reading_reference_contents(app, controller, tmp_path, monkeypatch):
    from PyHydroGeophysX.qt_apps.agent import chat_panel
    from PyHydroGeophysX.agents import local_knowledge
    from PyHydroGeophysX.llm.providers import make_provider
    file = tmp_path/'private.txt'
    file.write_text('confidential zorplith evidence')
    controller.add_chat_attachments('chat_reference', [file])
    provider = make_provider('openai')
    monkeypatch.setattr(provider, 'available', lambda: (True, ''))
    monkeypatch.setattr(chat_panel, 'prewarm', lambda provider: None)
    monkeypatch.setattr(local_knowledge, 'retrieve', lambda *a, **kw: pytest.fail('RAG must remain off'))
    panel = chat_panel.AssistantChatPanel(controller, provider=provider)
    panel._rag.setChecked(False)
    sent = []
    monkeypatch.setattr(panel, '_start_request', lambda: sent.append(panel._messages[-1]))
    panel._input.setPlainText('Explain zorplith')
    panel._on_send()
    assert str(file) in sent[0]['content']
    assert 'confidential' not in sent[0]['content']
    assert not panel._reference_workers
    panel._set_busy(False)
    panel.close()
    panel.deleteLater()
    app.processEvents()


def test_sources_are_saved_after_optional_workflow_accepts_clean_output(tmp_path, monkeypatch):
    import json
    from PyHydroGeophysX.agents import assistants
    from PyHydroGeophysX.qt_apps.agent.one_click_runner import execute
    source = {'source': 'paper.txt', 'line': 1, 'excerpt': 'zorplith', 'score': 1}
    def run(payload, progress, **kwargs):
        assert list(tmp_path.iterdir()) == []
        return {'status': 'complete'}
    assistant = SimpleNamespace(availability=lambda: (True, ''), load_workflow=lambda: run)
    monkeypatch.setattr(assistants, 'get_assistant', lambda key: assistant)
    result = execute({'assistant': 'geosage', 'output_dir': str(tmp_path),
                      'chat_reference_sources': [source]}, lambda *a: None)
    assert result['status'] == 'complete'
    assert json.loads((tmp_path/'chat_reference_sources.json').read_text()) == [source]


def test_clipboard_file_urls_and_paths_preserve_prompt_text(app, tmp_path):
    from PySide6.QtCore import QMimeData, QUrl
    from PyHydroGeophysX.qt_apps.agent.chat_panel import ChatInputEdit
    edit = ChatInputEdit()
    received = []
    edit.filesPasted.connect(received.append)
    files = [tmp_path/'survey one.dat', tmp_path/'notes.txt']
    for file in files:
        file.write_text('local data')
    edit.setPlainText('Run an ERT inversion')
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(file)) for file in files])
    edit.insertFromMimeData(mime)
    assert received == [[str(file) for file in files]]
    assert edit.toPlainText() == 'Run an ERT inversion'
    paths = QMimeData()
    paths.setText('\n'.join('"' + str(file) + '"' for file in files))
    edit.insertFromMimeData(paths)
    assert received[-1] == [str(file) for file in files]
    normal = QMimeData()
    normal.setText('ordinary prompt text')
    edit.clear()
    edit.insertFromMimeData(normal)
    assert edit.toPlainText() == 'ordinary prompt text'
    edit.close()


def test_unknown_classification_is_atomic_and_queue_is_scoped(controller, tmp_path):
    files = [tmp_path/'ert.dat', tmp_path/'unlabelled.csv']
    for file in files:
        file.write_text('numbers')
    assert controller.queue_chat_files(files)['status'] == 'ok'
    controller.queue_chat_files(files)
    assert len(controller.pending_chat_files()) == 2
    rows = [{'path': str(files[0]), 'role': 'data_file'},
            {'path': str(files[1]), 'role': 'unknown', 'reason': 'Missing units'}]
    assert controller.apply_chat_classification(rows)['status'] == 'needs_input'
    assert controller._attachment_page()._inputs == {}
    assert controller.pending_chat_files() == [str(file) for file in files]
    controller._window.state.output_dir = tmp_path/'other'
    assert controller.pending_chat_files() == []
    controller._window.state.output_dir = tmp_path
    controller.remove_chat_attachment('pending', str(files[1]))
    assert controller.pending_chat_files() == [str(files[0])]
    assert files[1].exists()


@pytest.mark.parametrize('key', ['aquah', 'geosage'])
@pytest.mark.parametrize('mode', ['guided', 'auto'])
def test_pasted_data_and_literature_are_classified_then_delivered(app, controller, tmp_path, monkeypatch, key, mode):
    import json
    from PySide6.QtCore import QMimeData, QUrl
    from PyHydroGeophysX.agents import assistants
    from PyHydroGeophysX.qt_apps.agent import chat_panel
    from PyHydroGeophysX.llm.providers import make_provider
    assistant = assistants.get_assistant('aquah')
    data_role = 'data_file'
    if key == 'geosage':
        data_role = 'gravity_file'
        assistant = replace(assistant, key=key, input_roles=(('Gravity data', data_role),),
                            ordered_roles=(), retrieval=())
    monkeypatch.setattr(assistants, 'active', lambda: assistant)
    controller._attachment_page().set_assistant(assistant)
    monkeypatch.setattr(chat_panel, 'prewarm', lambda provider: None)
    provider = make_provider('openai')
    monkeypatch.setattr(provider, 'available', lambda: (True, ''))
    calls = []
    def classify(system, messages, specs):
        payload = json.loads(messages[0]['content'])
        calls.append(payload)
        assert data_role in system
        return {'content': json.dumps({'files': [
            {'index': 0, 'role': data_role, 'confidence': 0.9, 'reason': 'Measured survey'},
            {'index': 1, 'role': 'chat_reference', 'confidence': 0.9, 'reason': 'Background geological literature'}]})}
    monkeypatch.setattr(provider, 'complete', classify)
    data, reference = tmp_path/'survey.csv', tmp_path/'paper.txt'
    data.write_text('x,y,gravity_mgal\n0,1,0.5')
    reference.write_text('zorplith mineral geological interpretation')
    panel = chat_panel.AssistantChatPanel(controller, provider=provider)
    panel._execution_mode.setCurrentIndex(panel._execution_mode.findData(mode))
    sent = []
    monkeypatch.setattr(panel, '_start_request', lambda: sent.append(panel._messages[-1]))
    monkeypatch.setattr(controller, 'run_to_report', lambda text, settings, *a, **kw:
                        sent.append((text, settings)) or 'Finished')
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(data)), QUrl.fromLocalFile(str(reference))])
    panel._input.insertFromMimeData(mime)
    assert not calls and not controller._attachment_page()._inputs
    assert panel._attachments.count() == 2
    panel._input.setPlainText('Interpret the survey using zorplith geology')
    panel._on_send()
    for _ in range(200):
        if sent and not panel._reference_workers:
            break
        time.sleep(0.02)
        app.processEvents()
    assert sent
    assert controller._attachment_page()._inputs[data_role] == str(data)
    assert not controller.pending_chat_files()
    assert panel._rag.isChecked()
    assert 'gravity_mgal' in calls[0]['files'][0]['preview']
    if mode == 'auto':
        assert 'zorplith mineral geological' in sent[0][1]['chat_context']
    else:
        assert 'zorplith mineral geological' in sent[0]['content']
    panel._set_busy(False)
    panel.close()
    panel.deleteLater()
    app.processEvents()
