"""Background file classification and cited retrieval for assistant chat."""
from pathlib import Path
from PySide6.QtCore import QThread, Signal


class _FileWorker(QThread):
    ready = Signal(object)

    def __init__(self, query, scope, parent=None):
        super().__init__(parent)
        self.query, self.scope = query, scope
        if parent is not None and hasattr(parent, 'aboutToQuit'):
            parent.aboutToQuit.connect(self.requestInterruption)
            parent.aboutToQuit.connect(self.wait)


class ClipboardClassificationWorker(_FileWorker):
    def __init__(self, query, paths, roles, provider, scope, parent=None):
        super().__init__(query, scope, parent)
        self.paths, self.roles, self.provider = tuple(paths), dict(roles), provider

    def run(self):
        from PyHydroGeophysX.agents.folder_catalog import classify_catalog, TEXT_SUFFIXES
        from PyHydroGeophysX.agents.local_knowledge import _document_parts
        rows = []
        try:
            for filename in self.paths:
                if self.isInterruptionRequested():
                    return
                path = Path(filename)
                if path.is_dir():
                    names = sorted(p.name for p in path.iterdir())[:40]
                    preview = 'Selected folder; entries: ' + ', '.join(names)
                elif path.suffix.lower() in {'.pdf', '.docx'}:
                    try:
                        stat = path.stat()
                        parts = _document_parts(filename, stat.st_mtime_ns, stat.st_size)
                        preview = '\n'.join(text[:4000] for _, text in parts[:2])[:4000]
                    except Exception as exc:
                        preview = f'[Text preview unavailable: {exc}]'
                elif path.suffix.lower() in TEXT_SUFFIXES | {'.json'}:
                    with path.open('rb') as stream:
                        raw = stream.read(4000)
                    preview = '[binary]' if b'\x00' in raw else raw.decode('utf-8', errors='replace')
                else:
                    preview = '[binary file; use filename, format and task evidence]'
                rows.append({'path': filename, 'name': path.name, 'preview': preview,
                             'is_folder': path.is_dir()})
            roles = {**self.roles, 'unknown': 'Ambiguous: ask the user', 'ignore': 'Unrelated to this task'}
            catalog = classify_catalog({'files': rows}, self.query, self.provider, roles=roles)
            result = {'query': self.query, 'scope': self.scope, 'files': catalog['files']}
        except Exception as exc:
            result = {'query': self.query, 'scope': self.scope, 'error': str(exc)}
        if not self.isInterruptionRequested():
            self.ready.emit(result)


class ReferenceRetrievalWorker(_FileWorker):
    def __init__(self, query, paths, scope, parent=None, search_query=None):
        super().__init__(query, scope, parent)
        self.paths, self.search_query = tuple(paths), search_query or query

    def run(self):
        from PyHydroGeophysX.agents.local_knowledge import retrieve
        diagnostics = []
        try:
            sources = retrieve(self.search_query, self.paths, diagnostics=diagnostics,
                               cancelled=self.isInterruptionRequested)
        except Exception as exc:
            sources = []
            diagnostics.append(str(exc))
        if not self.isInterruptionRequested():
            self.ready.emit({'query': self.query, 'sources': sources,
                             'diagnostics': diagnostics, 'scope': self.scope})
