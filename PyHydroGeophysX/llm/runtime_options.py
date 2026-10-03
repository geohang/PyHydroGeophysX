"""Per-process agent settings; never persist credentials."""
from contextvars import ContextVar
import threading

reasoning_effort = ContextVar('reasoning_effort', default='medium')
retrieved_context = ContextVar('retrieved_context', default='')

# Who hears about each model call's tokens, for a live count. Process-wide
# rather than a ContextVar: an agent may call its model from a worker thread,
# which a ContextVar set in the main thread would not reach.
_usage_listeners = []
_usage_lock = threading.Lock()


def add_usage_listener(listener):
    """Call ``listener(record)`` after every model call; returns a remover.

    ``record`` is the agent's usage ledger entry: provider, model, prompt and
    completion tokens, and the estimated cost in USD.

    >>> seen = []
    >>> remove = add_usage_listener(seen.append)
    >>> report_usage({'total_tokens': 3})
    >>> remove(); report_usage({'total_tokens': 4})
    >>> seen
    [{'total_tokens': 3}]
    """
    with _usage_lock:
        _usage_listeners.append(listener)

    def remove():
        with _usage_lock:
            if listener in _usage_listeners:
                _usage_listeners.remove(listener)
    return remove


def report_usage(record):
    """Tell every listener about one model call. A listener that raises is ignored."""
    with _usage_lock:
        listeners = list(_usage_listeners)
    for listener in listeners:
        try:
            listener(dict(record))
        except Exception:  # noqa: BLE001 - a counter must not stop a run
            pass


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
