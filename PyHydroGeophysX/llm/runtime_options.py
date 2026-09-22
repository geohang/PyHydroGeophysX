"""Per-process agent settings; never persist credentials."""
from contextvars import ContextVar

reasoning_effort = ContextVar('reasoning_effort', default='medium')
retrieved_context = ContextVar('retrieved_context', default='')


def openai_options(model, temperature=0.2, max_tokens=None, effort=None):
    if str(model).startswith(('gpt-5', 'gpt-6', 'o1', 'o3', 'o4')):
        options = {'reasoning_effort': effort or reasoning_effort.get()}
        if max_tokens:
            options['max_completion_tokens'] = max(16384, max_tokens)
        return options
    options = {'temperature': temperature}
    if max_tokens:
        options['max_tokens'] = max_tokens
    return options
