"""AQUAH's run: what "Auto to report" does when AQUAH is the assistant.

Moved here from the desktop's workflow runner, which now only starts the chosen
assistant's workflow in its own process (``qt_apps.agent.one_click_runner``).
"""
from __future__ import annotations

import json
from pathlib import Path

from PyHydroGeophysX.agents.assistants import accepted_hooks


def tools():
    """AQUAH's tool registry: the runtime catalog's tools."""
    from PyHydroGeophysX.agents.runtime import catalog  # noqa: F401 - registers them
    from PyHydroGeophysX.agents.runtime.tools import TOOLS

    return TOOLS


def run(payload, progress, *, context_factory=None, run_fn=None, events=None,
        approve=None, on_event=None):
    """Run AQUAH on ``payload``: the studio's "Auto to report", in the run's process.

    Follows :data:`PyHydroGeophysX.agents.assistants.WORKFLOW_CONTRACT`. With
    ``mode='classify'`` it sorts the files of ``data_folder`` into AQUAH's input
    roles with the model instead of running. Otherwise it reads the request with
    :class:`~PyHydroGeophysX.agents.ContextInputAgent`, runs the controller over
    AQUAH's tools, writes the processing audit and returns the result.

    ``context_factory`` and ``run_fn`` replace the request parser and the
    workflow, for tests.
    """
    if payload.get('mode') == 'classify':
        from PyHydroGeophysX.agents.folder_catalog import scan_folder, classify_catalog
        from PyHydroGeophysX.llm.providers import make_provider
        progress('Scanning selected folder', .02, 'Reading bounded file previews')
        catalog = scan_folder(payload['data_folder'])
        provider = make_provider(_classifier_provider(payload.get('provider')),
                                 model=payload.get('model'), api_key=payload.get('api_key'),
                                 base_url=payload.get('base_url'))
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
    # The agents call Claude 'claude'; the chat adapters call it 'anthropic'.
    provider = {'anthropic': 'claude'}.get(payload.get('provider') or 'openai',
                                           payload.get('provider') or 'openai')
    key = payload.get('api_key') or None
    model = payload.get('model') or None
    output = Path(payload['output_dir']).resolve()
    output.mkdir(parents=True, exist_ok=True)
    progress('Understanding request', 0.02, 'Preparing the workflow configuration')
    inputs = dict(payload.get('inputs', {}))
    sources = list(payload.get('chat_reference_sources') or [])
    from PyHydroGeophysX.llm.runtime_options import retrieved_context
    if payload.get('use_rag'):
        from PyHydroGeophysX.agents.local_knowledge import retrieve
        progress('Retrieving local references', .03, 'Searching documentation and reference files')
        refs = inputs.get('reference_file', [])
        if not sources:
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
    hooks = accepted_hooks(run_fn, on_step=on_step, ask_user=approve, on_event=on_event)
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
    # The run states a missing product with its reason; the same sentence
    # without one would only repeat it.
    for sentence in unmet_requests(config, finished):
        stem = sentence.rstrip('.').split(' (')[0]
        if not any(str(w).startswith(stem) for w in warnings):
            warnings.append(sentence)
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
    # A run that fell short says so in its status, not only in a warning: the
    # desktop reads the status to decide whether to call the run complete.
    if isinstance(results, dict) and results.get('status') == 'incomplete':
        status = 'incomplete'
    else:
        status = 'needs_review' if warnings else 'success'
    result = dict(execution_plan=plan, interpretation=interpretation, warnings=warnings,
                  status=status,
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


def _classifier_provider(name):
    """The chat adapter that classifies files, for a desktop provider id.

    Classification runs on :func:`PyHydroGeophysX.llm.providers.make_provider`,
    which covers OpenAI, Claude and OpenAI-compatible endpoints. Every id but
    'claude' used to be sent to OpenAI, so a Gemini or OpenAI-compatible key
    was presented to OpenAI and refused there with an authentication error.

    >>> _classifier_provider('claude'), _classifier_provider('openai_compatible')
    ('anthropic', 'openai_compatible')
    >>> _classifier_provider('gemini')
    Traceback (most recent call last):
    ...
    ValueError: File classification cannot use the provider 'gemini'. It runs on openai, anthropic, openai_compatible; choose one of those, or assign the file roles by hand.
    """
    from PyHydroGeophysX.llm.providers import PROVIDER_META, PROVIDER_ORDER, normalise_provider_id

    key = normalise_provider_id(name or 'openai')
    if key not in PROVIDER_META:
        raise ValueError(f"File classification cannot use the provider '{name}'. It runs on "
                         f"{', '.join(PROVIDER_ORDER)}; choose one of those, or assign the "
                         f"file roles by hand.")
    return key
