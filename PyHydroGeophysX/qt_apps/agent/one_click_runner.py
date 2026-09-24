"""Streamlit's unified workflow, exposed through a desktop subprocess protocol."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time


def execute(payload, progress, approve=None, **kwargs):
    from PyHydroGeophysX.llm.runtime_options import reasoning_effort, retrieved_context
    token = reasoning_effort.set(payload.get('reasoning_effort') or 'medium')
    context_token = retrieved_context.set('')
    started = time.monotonic()
    events = []
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
    try:
        return _execute(payload, observed, events=events, approve=approve, **kwargs)
    finally:
        reasoning_effort.reset(token)
        retrieved_context.reset(context_token)


def _execute(payload, progress, *, context_factory=None, run_fn=None, events=None,
             approve=None):
    if payload.get('mode') == 'classify':
        from PyHydroGeophysX.agents.folder_catalog import scan_folder, classify_catalog
        from PyHydroGeophysX.llm.providers import make_provider
        progress('Scanning selected folder', .02, 'Reading bounded file previews')
        catalog = scan_folder(payload['data_folder'])
        provider = make_provider('anthropic' if payload.get('provider') == 'claude' else 'openai',
                                 model=payload.get('model'), api_key=payload.get('api_key'))
        provider.reasoning_effort = payload.get('reasoning_effort') or 'medium'
        progress('Classifying files with AI', .1, f"{len(catalog['files'])} files; model {provider.model}")
        catalog = classify_catalog(catalog, payload['request'], provider, progress)
        output = Path(payload['output_dir'])
        output.mkdir(parents=True, exist_ok=True)
        (output / 'classification.json').write_text(json.dumps(catalog, indent=2), encoding='utf-8')
        return {'status': 'classified', 'catalog': catalog}
    if context_factory is None or run_fn is None:
        from PyHydroGeophysX.agents import BaseAgent, ContextInputAgent
        context_factory = context_factory or ContextInputAgent
        run_fn = run_fn or BaseAgent.run_unified_agent_workflow
    request = str(payload['request']).strip()
    if not request:
        raise ValueError('Describe the workflow you want to run.')
    provider = payload.get('provider', 'openai')
    key = payload.get('api_key') or None
    model = payload.get('model') or None
    output = Path(payload['output_dir']).resolve()
    output.mkdir(parents=True, exist_ok=True)
    progress('Understanding request', 0.02, 'Preparing the workflow configuration')
    inputs = dict(payload.get('inputs', {}))
    sources = []
    from PyHydroGeophysX.llm.runtime_options import retrieved_context
    if payload.get('use_rag'):
        from PyHydroGeophysX.agents.local_knowledge import retrieve
        progress('Retrieving local references', .03, 'Searching documentation and reference files')
        refs = inputs.get('reference_file', [])
        sources = retrieve(request, refs if isinstance(refs, list) else [refs])
        retrieved_context.set(json.dumps(sources))
        progress('References retrieved', .04, f'{len(sources)} source excerpts; citations will be retained')
    if payload.get('use_mcp'):
        from PyHydroGeophysX.agents.mcp_client import catalog
        progress('Connecting local MCP tools', .05, 'Initializing stdio MCP and requesting workflow catalog')
        tool_catalog = catalog(output)
        retrieved_context.set(retrieved_context.get() + '\nAvailable local workflow APIs:\n' + tool_catalog)
        (output / 'mcp_catalog.json').write_text(tool_catalog, encoding='utf-8')
    if 'topography_file' in inputs and 'raw_seismic_file' not in inputs:
        raise ValueError('Independent terrain grids are not yet consumed by this unified workflow. '
                         'Use an electrode coordinate file for ERT, or a geophone coordinate file for raw seismic. '
                         'For an XYZ terrain/DEM, use the 3D mesh workflow; do not relabel it as sensor coordinates.')
    if 'geophone_file' in inputs and 'raw_seismic_file' not in inputs:
        raise ValueError('Separate geophone coordinates currently require raw SEG-Y; travel-time files must already contain sensor geometry.')
    config = context_factory(api_key=key, model=model, llm_provider=provider).parse_request(request, available_data=inputs)
    if not isinstance(config, dict) or config.get('error'):
        raise ValueError(f'Could not interpret request: {config}')
    config.update(inputs)
    if 'time_lapse_files' in inputs:
        config['timelapse_files'] = inputs['time_lapse_files']
        config['inversion_mode'] = 'time-lapse'
    elif 'data_file' in inputs:
        config['ert_file'] = inputs['data_file']
        config.pop('time_lapse_files', None)
        config.pop('timelapse_files', None)
    if 'raw_seismic_file' in inputs:
        config['raw_seismic_processing'] = True
    config.update(user_request=request, output_dir=str(output))
    (output / 'workflow_config.json').write_text(json.dumps(config, indent=2, default=str), encoding='utf-8')
    from PyHydroGeophysX.agents.workflow_audit import record_calls, write_report
    # Step-by-step and auto are the same run; the only difference is whether
    # a hook pauses before each step to ask. A run_fn that predates the hook
    # (a stub in a test, an older caller) simply does not get one.
    on_step = None
    if payload.get('step_mode') and approve is not None:
        from PyHydroGeophysX.agents.runtime import step_by_step
        on_step = step_by_step(approve)
    # Which hooks the run takes is read off its signature before it starts. It
    # used to be tried and caught: any TypeError raised inside the workflow
    # then restarted it from scratch, without the hooks, so a step-by-step run
    # went on without asking and its calls were recorded twice.
    hooks = _accepted(run_fn, on_step=on_step, ask_user=approve)
    with record_calls() as calls:
        results, plan, interpretation, files = run_fn(
            config, key, model, provider, output, progress_callback=progress, **hooks)
    if isinstance(results, dict) and (results.get('success') is False or
            str(results.get('status', '')).lower() in {'failed', 'error'} or results.get('error')):
        raise RuntimeError(str(results.get('error') or results.get('message') or 'Workflow failed'))
    evaluation = results.get('evaluation_results') or {} if isinstance(results, dict) else {}
    warnings = list(results.get('warnings') or []) if isinstance(results, dict) else []
    if evaluation.get('status') == 'failed':
        # A failed quality check is a warning, not the end of the run. Raising
        # here threw away a complete result: a real run inverted two surveys,
        # estimated water content from them and wrote the report, and the
        # desktop was handed an exception instead of any of it, because the
        # optional QC step hit a bad number. The check describes the inversion;
        # it does not produce it, and what it could not judge still exists.
        warnings.append(
            'The automatic quality check could not be completed'
            + (f" ({evaluation.get('error')})" if evaluation.get('error') else '')
            + '. Judge the inversion from the fit and the figures yourself; the '
              'recovered models and everything derived from them are unaffected.')
    if evaluation.get('status') == 'needs_review':
        warnings.append(evaluation.get('summary') or 'Inversion quality needs review.')
    # Last line of defence, independent of which workflow branch ran: if the
    # request named a product the results do not contain, say so here rather
    # than handing back a report that looks complete without it.
    from PyHydroGeophysX.agents._intent import unmet_requests
    from PyHydroGeophysX.agents._uncertainty import result_caveats
    finished = results if isinstance(results, dict) else {}
    warnings.extend(unmet_requests(config, finished))
    # A product that exists but cannot bear the weight put on it is its own
    # kind of warning: it will be read, and read at face value.
    warnings.extend(result_caveats(config, finished))
    # Several sources now raise the same warning - the controller records what
    # a step found, and this function re-derives the same checks as a last line
    # of defence - so the quality summary arrived twice. dict.fromkeys keeps
    # the first occurrence of each and its order.
    warnings = list(dict.fromkeys(w for w in warnings if str(w).strip()))
    if isinstance(results, dict):
        results['warnings'] = warnings
    # Reports and numeric products remain in the run directory. The manifest is
    # deliberately small; no pickles or credentials are stored or sent to Qt.
    progress('Writing processing audit', .97, 'Recording input hashes, parameters, software references and uncertainty')
    files = write_report(output, config, results, plan, interpretation, files or {},
                         events or [], calls, sources,
                         {k: payload.get(k) for k in ('model', 'provider', 'reasoning_effort', 'use_rag', 'use_mcp')})
    if payload.get('classification'):
        (output / 'reviewed_classification.json').write_text(json.dumps(payload['classification'], indent=2), encoding='utf-8')
    result = dict(execution_plan=plan, interpretation=interpretation, warnings=warnings,
                  status='needs_review' if warnings else 'success',
                  report_files=files or {}, output_dir=str(output), workflow_config=config)
    def encode(value):
        if hasattr(value, 'tolist'):
            return value.tolist()
        if hasattr(value, 'item'):
            return value.item()
        return str(value)
    (output / 'numerical_results.json').write_text(json.dumps(results, default=encode), encoding='utf-8')
    (output / 'workflow_result.json').write_text(json.dumps(result, indent=2, default=str), encoding='utf-8')
    return result


def _accepted(fn, **hooks):
    """The ``hooks`` that ``fn`` takes, by name or through ``**kwargs``.

    A run function that predates them - an older caller, or a stub in a test -
    still runs; it simply cannot pause or ask. One whose signature cannot be
    read is given them all, as every workflow here takes them.
    """
    import inspect

    try:
        parameters = list(inspect.signature(fn).parameters.values())
    except (TypeError, ValueError):
        return dict(hooks)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters):
        return dict(hooks)
    names = {p.name for p in parameters
             if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                           inspect.Parameter.KEYWORD_ONLY)}
    return {name: value for name, value in hooks.items() if name in names}


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

    try:
        # One line, not the whole stream: stdin stays open so approvals can be
        # written back to this process while it runs.
        payload = json.loads(sys.stdin.readline())
        result = execute(payload, progress, approve=ask_via_stdio)
        print(json.dumps({'ok': True, 'result': result}, default=str), flush=True)
        return 0
    except Exception as exc:
        print(json.dumps({'ok': False, 'error': f'{type(exc).__name__}: {exc}'}), flush=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
