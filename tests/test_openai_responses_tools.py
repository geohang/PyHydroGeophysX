from types import SimpleNamespace

from PyHydroGeophysX.llm.providers import make_provider, to_responses_input


def test_reasoning_tool_roundtrip_preserves_call_ids_and_reasoning():
    sent = []
    raw = [{'type': 'reasoning', 'id': 'rs_1', 'summary': [], 'encrypted_content': 'opaque'},
           {'type': 'function_call', 'id': 'fc_1', 'call_id': 'call_1',
            'name': 'inspect', 'arguments': '{"path":"survey.dat"}'}]
    class Item(SimpleNamespace):
        def model_dump(self, **kwargs):
            return vars(self)
    def create(**kwargs):
        sent.append(kwargs)
        return SimpleNamespace(status='completed', output_text='', output=[Item(**item) for item in raw])
    provider = make_provider('openai', model='gpt-5.6-luna', api_key='fake')
    provider.reasoning_effort = 'high'
    provider._client = SimpleNamespace(responses=SimpleNamespace(create=create))
    specs = [{'name': 'inspect', 'description': 'Inspect file', 'parameters': {
        'type': 'object', 'properties': {'path': {'type': 'string'}}}}]
    messages = [{'role': 'user', 'content': 'Inspect my data'}]
    reply = provider.complete('system', messages, specs)
    assert reply['tool_calls'] == [{'id': 'call_1', 'name': 'inspect', 'arguments': {'path': 'survey.dat'}}]
    messages += [{'role': 'assistant', **reply}, {'role': 'tool', 'id': 'call_1', 'content': 'File ready'}]
    provider.complete('system', messages, specs)
    assert sent[0]['reasoning'] == {'effort': 'high'}
    assert sent[0]['store'] is False
    assert sent[0]['tools'][0]['strict'] is False
    assert sent[1]['input'][1:3] == raw
    assert sent[1]['input'][3] == {'type': 'function_call_output', 'call_id': 'call_1', 'output': 'File ready'}


def test_response_input_keeps_screenshot_content():
    items = to_responses_input([{'role': 'user', 'content': [
        {'type': 'text', 'text': 'Inspect this'},
        {'type': 'image', 'data': 'BASE64', 'media_type': 'image/png'}]}])
    assert items[0]['content'][1]['image_url'] == 'data:image/png;base64,BASE64'


def test_qt_retains_opaque_response_state():
    from PySide6.QtWidgets import QApplication
    from PyHydroGeophysX.qt_apps.agent.chat_panel import AquahChatPanel
    app = QApplication.instance() or QApplication([])
    controller = SimpleNamespace(capabilities_summary=lambda: '')
    panel = AquahChatPanel(controller, provider=make_provider('openai', model='gpt-5.6-luna', api_key='fake'))
    opaque = [{'type': 'reasoning', 'id': 'rs_1', 'encrypted_content': 'opaque', 'summary': []}]
    panel._on_llm_ok({'content': 'Ready', 'tool_calls': [], '_openai_output': opaque})
    assert panel._messages[-1]['_openai_output'] == opaque
    assert 'opaque' not in panel._transcript.toPlainText()
    panel.close()
    panel.deleteLater()
