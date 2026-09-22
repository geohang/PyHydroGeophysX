"""Local lexical retrieval with explicit source paths, without a vector service."""
import re
from pathlib import Path


def retrieve(query, paths=(), limit=6):
    tokens = set(re.findall(r'[\w-]{3,}', query.lower()))
    candidates = []
    roots = [Path(p) for p in paths]
    docs = Path(__file__).resolve().parents[2] / 'docs' / 'source'
    if docs.is_dir():
        roots.append(docs)
    files = []
    for root in roots:
        if root.is_file():
            files.append(root)
        elif root.is_dir():
            from .folder_catalog import scan_folder
            files.extend(Path(row['path']) for row in scan_folder(root)['files'])
    for path in list(dict.fromkeys(files))[:300]:
        if path.suffix.lower() not in {'.md', '.rst', '.txt'} or path.is_symlink():
            continue
        try:
            with path.open('r', encoding='utf-8', errors='replace') as stream:
                text = stream.read(100000)
        except OSError:
            continue
        lines = text.splitlines()
        for start in range(0, len(lines), 24):
            chunk = '\n'.join(lines[start:start+30])[:2400]
            words = set(re.findall(r'[\w-]{3,}', chunk.lower()))
            score = len(tokens & words)
            if score:
                candidates.append({'source': str(path.resolve()), 'line': start+1,
                                   'excerpt': chunk, 'score': score})
    return sorted(candidates, key=lambda x: -x['score'])[:limit]
