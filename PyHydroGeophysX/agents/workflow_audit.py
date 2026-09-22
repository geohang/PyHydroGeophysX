"""Reproducible processing evidence and uncertainty appendix for desktop reports.

The record this module keeps has two audiences that want different things. A
reviewer reconstructing the run needs every input hash, every parameter and the
full text of every reference the retrieval step returned; that audience is
served by ``processing_audit.json``, which is written complete and unabridged.
A reader of the report needs to know which files went in, what settings were
used, and where the numbers came from - and is served badly by the same content
pasted into the document as JSON.

So the markdown carries tabulated summaries and the appendices they belong in,
and points at the JSON for the rest. The retrieval excerpts in particular are
now filtered to the sources the interpretation actually cites: a semantic search
returns its best matches whether or not they are relevant, and a report on a
time-lapse ERT survey was ending with several hundred words about the desktop
studio's gravity module and its seismic agent, under a heading that implied the
run had used them.
"""
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import sys

from ._document import bullets, control_block, digest, facts, file_size, renumber, table

PACKAGES = {'PyHydroGeophysX': 'PyHydroGeophysX', 'pygimli': 'pygimli', 'numpy': 'numpy',
            'scipy': 'scipy', 'simpeg': 'simpeg', 'flopy': 'flopy', 'resipy': 'resipy',
            'matplotlib': 'matplotlib', 'adtlert': 'adtlert', 'pandas': 'pandas'}


@contextmanager
def record_calls():
    calls = Counter()
    previous = sys.getprofile()
    def profile(frame, event, arg):
        if event == 'call':
            module = frame.f_globals.get('__name__', '')
            root = module.split('.', 1)[0]
            if root in PACKAGES:
                name = module + '.' + frame.f_code.co_name
                if name in calls or len(calls) < 2000:
                    calls[name] += 1
    sys.setprofile(profile)
    try:
        yield calls
    finally:
        sys.setprofile(previous)


def _fallback_body(config, interpretation):
    """A document head for a run whose engine produced no report of its own.

    Without this the file opened straight into interpretive prose with no
    title, no date and no statement of what it describes - which is exactly the
    document most likely to be forwarded to someone who was not there.
    """
    request = str((config or {}).get('user_request') or '').strip()
    head = control_block(
        'Geophysical Processing Report',
        [('Report date', datetime.now().strftime('%Y-%m-%d %H:%M')),
         ('Request', request or None),
         ('Prepared by', 'PyHydroGeophysX multi-agent workflow'),
         ('Status', 'Automated output - technical review required')],
        notice='This report was produced by an automated workflow. The '
               'interpretive text below is written by a language model and has '
               'not been reviewed by a geophysicist.')
    return head + '\n## Interpretation\n\n' + str(
        interpretation or 'The workflow returned no interpretation.')


def _inventory_rows(inputs):
    """One row per file, listing every role it was used in.

    The configuration names the same file under several keys - a five-survey
    run had each survey under ``time_lapse_files`` and again under
    ``timelapse_files``, plus the first one under ``data_file``,
    ``baseline_file`` and ``ert_file``. Listed once per key, seven files filled
    seventeen rows and the repetition read as seventeen inputs.
    """
    merged = {}
    for record in inputs:
        key = record.get('path')
        entry = merged.setdefault(key, {'record': record, 'roles': []})
        role = str(record.get('role', '')).replace('_', ' ')
        if role and role not in entry['roles']:
            entry['roles'].append(role)
    rows = []
    for entry in merged.values():
        record = entry['record']
        rows.append([Path(record['path']).name, ', '.join(entry['roles']),
                     _size_or_status(record), digest(record.get('sha256'))])
    return rows


def _size_or_status(record):
    """A file's size, or why it has none.

    A record without a byte count is a directory or a path that was not there;
    reporting both as "directory" would let a missing input pass as a normal
    row in the inventory.
    """
    if 'bytes' in record:
        return file_size(record['bytes'])
    if record.get('note'):
        return 'directory'
    return 'not found' if not record.get('exists') else 'directory'


def _place_limitations(body, limitations):
    """Put the limitations before any recommendations the body ends with.

    A reader meets the caveats before the advice that rests on them, which is
    the order a deliverable uses. The recommendations are written by the report
    agent and the limitations by this module, so the two are joined here rather
    than passed between them.
    """
    marker = None
    for line in body.split('\n'):
        if line.startswith('## ') and line.rstrip().endswith('Recommendations'):
            marker = line
            break
    if marker is None:
        return [body, limitations]
    head, _, tail = body.partition(marker)
    return [head.rstrip(), limitations, marker + tail]


def _warnings_block(warnings):
    """The run's own warnings, stated before the standing caveats.

    A warning is specific to this run and a limitation applies to every run;
    printing them together at the end, as raw JSON, buried the one that
    mattered. These are what the reader most needs and least expects.
    """
    if not warnings:
        return '### Warnings Raised by This Run\n\nThe workflow raised no warnings.'
    return '### Warnings Raised by This Run\n\n' + bullets(warnings)


def _quality_block(assessment):
    """Quality metrics as tables rather than a nested JSON dump."""
    if not assessment or not any(assessment.values()):
        return 'No quality assessment was recorded for this run.'
    score = assessment.get('quality_score')
    parts = [facts([('Status', str(assessment.get('status') or '').replace('_', ' ')),
                    ('Summary', assessment.get('summary')),
                    ('Score', f"{float(score):.1f} / 100"
                     if isinstance(score, (int, float)) else score),
                    ('Evaluation attempts', assessment.get('attempts'))],
                   label='Assessment', value='Value')]
    rows = []
    for group, metrics in (assessment.get('quality_metrics') or {}).items():
        if not isinstance(metrics, dict):
            continue
        for name, value in metrics.items():
            pretty = f"{value:.4g}" if isinstance(value, float) else value
            rows.append([group.replace('_', ' ').capitalize(),
                         name.replace('_', ' '), pretty])
    metric_table = table(['Group', 'Metric', 'Value'], rows,
                         align=['left', 'left', 'right'])
    if metric_table:
        parts += ['**Metrics**', metric_table]
    if assessment.get('recommendations'):
        parts += ['**Recommendations from the automated evaluation**',
                  bullets(assessment['recommendations'])]
    return '\n\n'.join(part for part in parts if part)


def _references_block(sources, report_text):
    """Only the retrieved references the report actually cites.

    Retrieval returns its closest matches whether or not any of them is
    relevant, and printing all of them with their excerpts filled the end of
    the document with material the run never used - documentation for the
    seismic agent and the gravity module under a heading that implied
    otherwise. A source counts as used here when its file name appears in the
    report text, which is how the interpretation cites one.
    """
    if not sources:
        return ('### References Consulted\n\n'
                'Local reference retrieval was disabled, or returned no match.')
    cited = [s for s in sources if Path(str(s.get('source', ''))).name in (report_text or '')]
    lines = ['### References Consulted', '']
    if cited:
        lines.append(bullets(
            f"{Path(str(s['source'])).name}, line {s['line']}" for s in cited))
        lines.append(f"{len(cited)} of {len(sources)} retrieved excerpts are cited above; "
                     f"all of them, with their text, are in `processing_audit.json`.")
    else:
        lines.append(f"{len(sources)} local reference excerpts were retrieved and none is "
                     f"cited in this report. They are listed in `processing_audit.json`.")
    return '\n'.join(lines)


def write_report(output, config, results, plan, interpretation, reports, events, calls, sources, settings):
    output = Path(output)
    inputs = []
    for role, value in config.items():
        if not (role.endswith(('_file', '_files', '_dir')) or role == 'reference_file'):
            continue
        for item in value if isinstance(value, list) else [value]:
            if not isinstance(item, str):
                continue
            path = Path(item)
            record = {'role': role, 'path': str(path.resolve()), 'exists': path.exists()}
            if path.is_file():
                hasher = hashlib.sha256()
                with path.open('rb') as stream:
                    for chunk in iter(lambda: stream.read(1024*1024), b''):
                        hasher.update(chunk)
                record.update(bytes=path.stat().st_size, sha256=hasher.hexdigest())
            elif path.is_dir():
                record['note'] = 'Directory input; individual consumed files must be checked in the engine log.'
            inputs.append(record)
    software = []
    for module, distribution in PACKAGES.items():
        if not any(name.startswith(module + '.') for name in calls):
            continue
        try:
            meta = metadata.metadata(distribution)
            software.append({'name': distribution, 'version': metadata.version(distribution),
                             'references': meta.get_all('Project-URL') or [meta.get('Home-page', '')]})
        except metadata.PackageNotFoundError:
            software.append({'name': distribution, 'version': 'unavailable', 'references': []})
    audit = dict(created_utc=datetime.now(timezone.utc).isoformat(), inputs=inputs,
                 configuration=config, execution_plan=plan, events=events, executed_python_calls=dict(calls),
                 software=software, retrieved_sources=sources, ai_settings=settings,
                 warnings=results.get('warnings', []) if isinstance(results, dict) else [])
    audit['python_trace_limit'] = 2000
    if isinstance(results, dict):
        evaluation = results.get('evaluation_results') or {}
        audit['quality_assessment'] = {key: evaluation.get(key) for key in (
            'status', 'summary', 'attempts', 'quality_score', 'quality_metrics',
            'recommendations', 'adjusted_params', 'transparent_log')}
        audit['processing'] = (results.get('inversion_results') or {}).get('processing', {})
    audit['python_trace_may_be_truncated'] = len(calls) >= 2000
    (output / 'processing_audit.json').write_text(json.dumps(audit, indent=2, default=str), encoding='utf-8')
    original = ''
    if reports.get('report_markdown'):
        try:
            original = Path(reports['report_markdown']).read_text(encoding='utf-8')
        except OSError:
            pass
    body = original or _fallback_body(config, interpretation)
    limitations = '\n\n'.join([
        '## Limitations and Uncertainty',
        _warnings_block(audit['warnings']),
        '### Standing Limitations',
        'The following qualify every result in this report.',
        bullets([
            'File classification is a model suggestion. Check roles, acquisition dates, units, coordinate systems and elevation datums; the confidence attached to it is not a calibrated probability.',
            'Measurement noise, outliers, missing observations and first-break picks all affect the data fit. A reduced chi-squared near one does not prove that the recovered model is unique or geologically correct.',
            'Mesh resolution, boundary conditions, regularization, the starting model, the stopping criteria and the solver all affect the recovered model. Inspect the convergence record and compare plausible settings before relying on a feature.',
            'Petrophysical parameters and geological assumptions can dominate the uncertainty of any derived water content. A single best-fit model is not a confidence interval.',
            'Confidence intervals, ensemble sensitivity and external validation are established only where this report explicitly says they were computed.',
            'Model-written interpretation and retrieved references may be incomplete or mistaken. Review them against field observations and domain expertise.',
        ])])
    sections = _place_limitations(body, limitations) + [
        '## Appendix A - Input File Inventory',
        table(['File', 'Used as', 'Size', 'SHA-256'], _inventory_rows(inputs),
              align=['left', 'left', 'right', 'left'])
        or 'No file inputs were recorded.',
        'Paths are abbreviated to file names and digests to their first characters; '
        '`processing_audit.json` carries both in full.',
        '## Appendix B - Workflow Configuration',
        facts([('Provider', settings.get('provider')), ('Model', settings.get('model')),
               ('Reasoning effort', settings.get('reasoning_effort')),
               ('Local reference retrieval', 'enabled' if settings.get('use_rag') else 'disabled'),
               ('External tools (MCP)', 'enabled' if settings.get('use_mcp') else 'disabled')],
              label='Setting', value='Value'),
        'Full configuration as passed to the workflow:',
        '```json\n' + json.dumps(config, indent=2, default=str) + '\n```',
        '## Appendix C - Execution Record',
        table(['Elapsed', 'Stage', 'Detail'],
              [[f"{e['elapsed_seconds']:.1f} s", e['step'], e['details']] for e in events],
              align=['right', 'left', 'left']) or 'No stages were recorded.',
        '**Steps taken**',
        # The plan is derived from the run, so it can carry each step's outcome.
        # A plan that lists only what was intended cannot show that a step
        # failed, which is the thing a reader most needs from it.
        table(['Step', 'Agent', 'Outcome', 'Purpose'],
              [[str(s.get('step', '')), str(s.get('agent', '')),
                str(s.get('status', '') or '-'), str(s.get('description', ''))]
               for s in (plan or []) if isinstance(s, dict)],
              align=['left', 'left', 'left', 'left'])
        or 'No execution plan was recorded.',
        '## Appendix D - Quality Assessment',
        'Quality scores are heuristic diagnostics. They are not calibrated probabilities or confidence levels.',
        _quality_block(audit.get('quality_assessment') or {}),
        '## Appendix E - Software and References',
        'The packages below were observed in Python calls during this run. This is runtime evidence, not a claim that every imported solver contributed to the result; native internal calls and child processes are not traced, and the trace is capped at 2000 distinct function names.',
        table(['Package', 'Version', 'Reference'],
              [[p['name'], p['version'],
                next((str(r) for r in p['references'] if r), None)] for p in software])
        or 'No package usage was recorded.',
        bullets(['PyHydroGeophysX source: https://github.com/geohang/PyHydroGeophysX',
                 'Inversion, regularization and error models: https://www.pygimli.org/user-guide/inversion/']),
        _references_block(sources, body),
        # Backticked: a bare Windows path loses its backslashes to markdown
        # escaping, so the run directory rendered with segments run together.
        '## Appendix F - Output Files',
        table(['Output', 'Path'], [[k, f'`{v}`'] for k, v in reports.items()])
        or 'No output files were recorded.']
    path = output / 'detailed_report.md'
    path.write_text(renumber('\n\n'.join(s for s in sections if s and str(s).strip())),
                    encoding='utf-8')
    return {**reports, 'engine_report_markdown': reports.get('report_markdown'),
            'report_markdown': str(path), 'processing_audit': str(output / 'processing_audit.json')}
