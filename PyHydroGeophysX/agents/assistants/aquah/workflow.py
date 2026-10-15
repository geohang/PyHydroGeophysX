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
        # A model that cannot be set up or reached leaves the table as the
        # scan read it - roles the files themselves proved, the rest unknown -
        # with the reason, for the user to complete or retry.
        try:
            provider = make_provider(_classifier_provider(payload.get('provider')),
                                     model=payload.get('model'), api_key=payload.get('api_key'),
                                     base_url=payload.get('base_url'))
            provider.reasoning_effort = payload.get('reasoning_effort') or 'medium'
            progress('Classifying files with AI', .1, f"{len(catalog['files'])} files; model {provider.model}")
            catalog = classify_catalog(catalog, payload['request'], provider, progress)
        except Exception as exc:  # noqa: BLE001 - the scan stands on its own
            catalog = {**catalog, 'warnings': [
                *(catalog.get('warnings') or []),
                f'The AI could not classify the files ({type(exc).__name__}: {str(exc)[:300]}); '
                'give each unknown file a role, or retry.']}
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
    # Nothing before the first step stops the run. A stage that fails falls
    # back to the plain way (no references, no MCP catalog, the request read
    # by rules) and leaves a note; the notes reach the controller as warnings,
    # so the model sees what went wrong at its first decision and can carry
    # on, ask, or propose a temporary script for the user to approve.
    input_notes = []

    def setback(stage, exc, fallback):
        input_notes.append(f'{stage} failed ({type(exc).__name__}: {str(exc)[:300]}); '
                           f'the run goes on {fallback}.')
        progress('Carrying on past a problem', .05, input_notes[-1])

    from PyHydroGeophysX.llm.runtime_options import retrieved_context
    if payload.get('use_rag'):
        progress('Retrieving local references', .03, 'Searching documentation and reference files')
        try:
            from PyHydroGeophysX.agents.local_knowledge import retrieve
            refs = inputs.get('reference_file', [])
            if not sources:
                sources = retrieve(request, refs if isinstance(refs, list) else [refs])
        except Exception as exc:  # noqa: BLE001 - see setback
            setback('Retrieving local references', exc, 'without reference excerpts')
        retrieved_context.set(json.dumps(sources))
        progress('References retrieved', .04, f'{len(sources)} source excerpts; citations will be retained')
    if payload.get('use_mcp'):
        progress('Connecting local MCP tools', .05, 'Initializing stdio MCP and requesting workflow catalog')
        try:
            from PyHydroGeophysX.agents.mcp_client import catalog
            tool_catalog = catalog(output)
            retrieved_context.set(retrieved_context.get() + '\nAvailable local workflow APIs:\n' + tool_catalog)
            (output / 'mcp_catalog.json').write_text(tool_catalog, encoding='utf-8')
        except Exception as exc:  # noqa: BLE001 - see setback
            setback('Connecting the local MCP tools', exc, 'without the MCP workflow catalog')
    # Inputs this workflow cannot take as given are settled or set aside here,
    # with a note - not a run that stops at 0 s.
    try:
        input_notes.extend(_settle_inputs(inputs))
    except Exception as exc:  # noqa: BLE001 - see setback
        setback('Checking the inputs', exc, 'with the files as they were given')
    config = _read_request(context_factory, request, inputs, key, model, provider, setback)
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
    studio = (payload.get('studio_work') or {}).get('ert')
    if studio:
        try:
            _carry_over_studio_inversion(config, studio, progress)
        except Exception as exc:  # noqa: BLE001 - see setback
            setback('Taking over the inversion finished in the studio', exc,
                    'and inverts the data itself')
    config.update(user_request=request, output_dir=str(output))
    if input_notes:
        config['input_notes'] = input_notes
        for note in input_notes:
            progress('Checking the inputs', .05, note)
    try:
        (output / 'workflow_config.json').write_text(json.dumps(config, indent=2, default=str),
                                                     encoding='utf-8')
    except Exception as exc:  # noqa: BLE001 - a record of the run, not the run
        input_notes.append(f'workflow_config.json could not be written ({exc}); the run '
                           'goes on, and its configuration is in the processing audit.')
        config['input_notes'] = input_notes
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
    # What follows checks and records a run that has already done its work. A
    # check or a record that fails is a warning; it used to throw the finished
    # results away with it.
    try:
        # The run states a missing product with its reason; the same sentence
        # without one would only repeat it.
        for sentence in unmet_requests(config, finished):
            stem = sentence.rstrip('.').split(' (')[0]
            if not any(str(w).startswith(stem) for w in warnings):
                warnings.append(sentence)
        # A product that exists but cannot bear the weight put on it is its own
        # kind of warning: it will be read, and read at face value.
        warnings.extend(result_caveats(config, finished))
    except Exception as exc:  # noqa: BLE001 - see above
        warnings.append(f'The final check of the requested products failed ({type(exc).__name__}: '
                        f'{exc}); compare the results with the request yourself.')
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
    try:
        files = write_report(output, config, results, plan, interpretation, files or {},
                             events or [], calls, sources,
                             {k: payload.get(k) for k in ('model', 'provider', 'reasoning_effort', 'use_rag', 'use_mcp')})
    except Exception as exc:  # noqa: BLE001 - the run's own report and products stand
        warnings.append(f'The processing audit could not be written ({type(exc).__name__}: {exc}); '
                        "the run's report and products are unaffected.")
    if payload.get('classification'):
        try:
            (output / 'reviewed_classification.json').write_text(
                json.dumps(payload['classification'], indent=2, default=str), encoding='utf-8')
        except Exception as exc:  # noqa: BLE001 - a record, not a result
            warnings.append(f'reviewed_classification.json could not be written ({exc}).')
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
    for name, write in (
            ('numerical_results.json', lambda: json.dumps(results, default=encode)),
            ('workflow_result.json', lambda: json.dumps(result, indent=2, default=str))):
        try:
            (output / name).write_text(write(), encoding='utf-8')
        except Exception as exc:  # noqa: BLE001 - a record of results already on disk
            warnings.append(f'{name} could not be written ({type(exc).__name__}: {exc}).')
            if result['status'] == 'success':
                result['status'] = 'needs_review'
    return result


def _read_request(context_factory, request, inputs, key, model, provider, setback):
    """The run's configuration from ``request``; never an exception.

    The parser falls back to its rules when the model call fails. When the
    parser itself fails, or answers with an error or no configuration at all,
    the rules read the request here instead, with a note; "Could not interpret
    request" used to end the run before its first step.

    >>> class Broken:
    ...     def __init__(self, **kw): pass
    ...     def parse_request(self, request, available_data=None): return {'error': 'bad JSON'}
    >>> notes = []
    >>> config = _read_request(Broken, 'Invert line1.ohm', {}, None, None, 'openai',
    ...                        lambda stage, exc, then: notes.append(f'{stage}: {exc}; {then}'))
    >>> config['data_file'], notes
    ('line1.ohm', ['Reading the request: bad JSON; with the request read by rules, not the model'])
    """
    from PyHydroGeophysX.agents import ContextInputAgent

    fallback = 'with the request read by rules, not the model'
    try:
        config = context_factory(api_key=key, model=model,
                                 llm_provider=provider).parse_request(request, available_data=inputs)
        if isinstance(config, dict) and not config.get('error'):
            return config
        problem = ValueError(str(config.get('error') if isinstance(config, dict) else
                                 f'the parser returned {type(config).__name__}, not a configuration'))
    except Exception as exc:  # noqa: BLE001 - the rules still read the request
        problem = exc
    setback('Reading the request', problem, fallback)
    try:
        preview = ContextInputAgent(api_key=None).preview_config(request, inputs)
        return dict(preview.get('workflow_config') or {})
    except Exception as exc:  # noqa: BLE001 - the run's files and words still say a lot
        setback('Reading the request by rules', exc, 'with only the files it was given')
        return {'user_request': request}


def _settle_inputs(inputs):
    """Settle the inputs this workflow cannot take as labelled; return a note on each.

    A terrain table beside ERT data used to stop the run before its first step
    ("Independent terrain grids are not yet consumed"), and a classifier calls
    an electrode file terrain often enough - ``electrodes.dat``, 56 rows for 56
    electrodes, was one. Now a terrain table with one evenly spaced row per
    electrode of the ERT surveys is used as their electrode file; any other is
    left out with a note, as are geophone coordinates without raw seismic.
    ``inputs`` is changed in place.

    >>> notes = _settle_inputs({'data_file': 'a.ohm', 'topography_file': 'dem.xyz'})
    >>> print(notes[0])  # doctest: +ELLIPSIS
    dem.xyz was given as terrain / topography, which this workflow does not read...
    """
    from PyHydroGeophysX.agents.folder_catalog import electrode_table_match, table_facts

    notes = []
    raw_seismic = bool(inputs.get('raw_seismic_file'))
    terrain = inputs.get('topography_file')
    terrain = (terrain[0] if isinstance(terrain, list) and terrain else terrain) or None
    if terrain and not raw_seismic:
        inputs.pop('topography_file', None)
        ert = list(inputs.get('time_lapse_files') or []) + (
            [inputs['data_file']] if inputs.get('data_file') else [])
        match = (electrode_table_match([{'path': str(terrain), 'facts': table_facts(terrain)}],
                                       [{'path': str(p)} for p in ert]).get(0)
                 if ert else None)
        name = Path(str(terrain)).name
        if match and not inputs.get('electrode_file'):
            inputs['electrode_file'] = terrain
            notes.append(f'{name} was given as terrain, but it holds one evenly spaced position '
                         f'for each of the {match[0]} electrodes of {match[1]}, so the run uses it '
                         'as the electrode file.')
        else:
            notes.append(f'{name} was given as terrain / topography, which this workflow does '
                         'not read, so the run goes on without it; ERT takes its elevations '
                         'from the electrode positions. If the file holds positions the run '
                         'needs, a temporary script can convert it.')
    if inputs.get('geophone_file') and not raw_seismic:
        name = Path(str(inputs.pop('geophone_file'))).name
        notes.append(f'{name} was given as geophone coordinates, which only raw SEG-Y data '
                     'use; travel-time files carry their own geometry, so the run goes on '
                     'without it.')
    return notes


def _carry_over_studio_inversion(config, studio, progress):
    """Let the run take over an ERT inversion finished in the studio, when it fits.

    ``studio`` is what the ERT page wrote when a step-by-step task was handed
    over (its ``agent_handover``). Only a model made from the very surveys this
    run loads is taken over: one from other files would be reported as the
    inversion of data it never saw.
    """
    import os

    from PyHydroGeophysX.agents.runtime.catalog import STUDIO_INVERSION_KEY, survey_files

    def same(paths):
        return [os.path.normcase(os.path.abspath(str(p))) for p in paths]

    run_files = survey_files(config)
    studio_files = [p for p in studio.get('source_files') or [] if p]
    if run_files and same(run_files) == same(studio_files) and studio.get('model_dir'):
        config[STUDIO_INVERSION_KEY] = studio['model_dir']
        progress('Carrying over the studio inversion', .05,
                 f'{len(studio_files)} survey(s) were inverted in the studio; the run continues from that model')
    else:
        progress('Studio inversion not carried over', .05,
                 'It was made from other survey files than this run loads, so the run inverts the data itself')


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
