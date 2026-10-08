"""Run the chosen assistant's workflow in its own process, for the desktop.

The studio starts this module as a subprocess and talks to it in JSON lines:
the payload on stdin, then progress, step events and questions on stdout, and
the user's answers back on stdin. Notes for the run, pause and resume go
through a file in the run's folder instead (the payload's ``control_file``;
see :class:`~PyHydroGeophysX.agents.runtime.steering.ControlFile`), read between
steps. Which workflow runs is the payload's ``assistant``
(AQUAH unless it says otherwise); see :mod:`PyHydroGeophysX.agents.assistants`
for how an assistant plugs in.
"""
from __future__ import annotations

import json
import sys
import threading
import time


def execute(payload, progress, approve=None, on_event=None, **kwargs):
    from PyHydroGeophysX.llm.runtime_options import reasoning_effort, retrieved_context
    token = reasoning_effort.set(payload.get('reasoning_effort') or 'medium')
    context_token = retrieved_context.set('')
    started = time.monotonic()
    events = []
    def stepped(event):
        # The step itself - why the controller chose it, then what it found and
        # whether it worked - kept as fields rather than flattened into a
        # progress line, for the desktop's live timeline.
        if on_event is not None:
            on_event(dict(event, elapsed_seconds=time.monotonic()-started))
    def observed(step, fraction, details='', module=''):
        # `module` names the studio panel this step belongs to, so the desktop
        # can bring it to the front while the work happens. It is optional:
        # stages that are not one tool's work (parsing the request, writing the
        # audit) pass nothing and the app stays where it is.
        events.append(dict(step=step, progress=fraction, details=details,
                           module=module,
                           elapsed_seconds=time.monotonic()-started))
        progress(step, fraction, f"[{time.monotonic()-started:.1f}s] {details}",
                 module)
    # Every model call's tokens and estimated cost, summed as the run goes, so
    # the desktop can show what the run is spending while it spends it.
    from PyHydroGeophysX.llm.runtime_options import add_usage_listener
    spent = {'calls': 0, 'tokens': 0, 'prompt': 0, 'completion': 0, 'cost': 0.0}
    spent_lock = threading.Lock()

    def counted(record):
        with spent_lock:
            spent['calls'] += 1
            spent['prompt'] += int(record.get('prompt_tokens') or 0)
            spent['completion'] += int(record.get('completion_tokens') or 0)
            spent['tokens'] = spent['prompt'] + spent['completion']
            spent['cost'] += float(record.get('cost_estimate_usd') or 0.0)
            total = dict(spent)
        stepped({'phase': 'usage', 'calls': total['calls'], 'tokens': total['tokens'],
                 'prompt_tokens': total['prompt'], 'completion_tokens': total['completion'],
                 'cost_usd': round(total['cost'], 6), 'model': record.get('model', ''),
                 'agent': record.get('agent', '')})
    stop_counting = add_usage_listener(counted)
    try:
        from PyHydroGeophysX.agents.assistants import get_assistant

        assistant = get_assistant(payload.get('assistant') or 'aquah')
        ready, why = assistant.availability()
        if not ready:
            raise RuntimeError(why)
        run = assistant.load_workflow()
        result = run(payload, observed, events=events, approve=approve,
                     on_event=stepped, **kwargs)
        sources = payload.get('chat_reference_sources') or []
        if sources:
            from pathlib import Path
            (Path(payload['output_dir']) / 'chat_reference_sources.json').write_text(
                json.dumps(sources, indent=2, ensure_ascii=False), encoding='utf-8')
        return result
    finally:
        stop_counting()
        reasoning_effort.reset(token)
        retrieved_context.reset(context_token)


def ask_via_stdio(event, out=None, inp=None):
    """Put a question to the desktop and wait for the answer.

    This is the whole of step-by-step, and of a tool's own question, across the
    process boundary: the child writes one line saying what it needs and blocks
    on the next line back. A step approval and a question travel the same way;
    ``event`` says which it is, and the desktop renders the right controls.

    Parameters
    ----------
    event : dict
        The announcement, or the question with its options.
    out, inp : file, optional
        Streams to use instead of the process's own, for tests.

    Returns
    -------
    str
        The decision. End of input means the parent is gone, which is a stop
        rather than silent consent - and so is an unreadable reply.
    """
    out = out if out is not None else sys.stdout
    inp = inp if inp is not None else sys.stdin
    out.write(json.dumps({'event': event.get('event', 'approve'),
                          **{k: v for k, v in event.items() if k != 'event'}},
                         default=str) + '\n')
    out.flush()
    line = inp.readline()
    if not line:
        return 'stop'
    try:
        return str(json.loads(line).get('decision') or 'stop')
    except ValueError:
        return 'stop'


def main():
    # This child exports figures; a GUI backend may block pg.show on Windows.
    import matplotlib
    matplotlib.use('Agg', force=True)

    def progress(step, fraction, details='', module=''):
        print(json.dumps({'event': 'progress', 'step': step, 'progress': fraction,
                          'details': details, 'module': module}), flush=True)

    printing = threading.Lock()

    def step_event(event):
        # Events come from the run and, for token counts, possibly from a
        # thread a model call ran on: one line at a time.
        line = json.dumps({**event, 'event': 'step'}, default=str)
        with printing:
            print(line, flush=True)

    from PyHydroGeophysX.agents.runtime import steering

    try:
        # One line, not the whole stream: stdin stays open so approvals can be
        # written back to this process while it runs. Nothing reads it in the
        # background (see steering.ControlFile for why).
        payload = json.loads(sys.stdin.readline())
        control = payload.get('control_file')
        steering.current.set(steering.Steering(
            notify=step_event, source=steering.ControlFile(control) if control else None))
        result = execute(payload, progress, approve=ask_via_stdio, on_event=step_event)
        print(json.dumps({'ok': True, 'result': result}, default=str), flush=True)
        return 0
    except Exception as exc:
        print(json.dumps({'ok': False, 'error': f'{type(exc).__name__}: {exc}'}), flush=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
