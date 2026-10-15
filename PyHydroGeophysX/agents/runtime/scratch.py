"""Temporary scripts an assistant writes to get past a problem, run only on a yes.

Some failures are neither a setting nor a missing feature: a file in a layout no
reader handles, a model folder written in one module's format that another
module cannot open, a quantity none of the tools computes. A person at the
keyboard would write ten lines of Python and carry on. This lets the assistant
do the same - and only that far:

- **The user decides.** A script is shown with what it is for, and runs only
  after an explicit yes. A run with nobody to ask declines.
- **The source is checked before anyone is asked** (:func:`check_script`): one
  ``def main(inputs, out)``, imports from a short list of scientific libraries
  and this package, no ``exec``/``eval``, no dunder attributes, no ``os``,
  ``sys``, ``subprocess`` or network modules.
- **It runs in a process of its own** (:func:`run_script`), in a folder of its
  own, with a time limit and without the API keys in its environment. Inside
  that process an audit hook refuses subprocesses, sockets and native-library
  loading, and any write outside the script's folder (which holds its
  temporary directory too). The user's files are read, never written.

That is a guard rail, not a security boundary: compiled code inside numpy or
pyGIMLi does not pass through Python's audit hooks. It keeps a well-meant script
from doing something nobody approved; the approval is what stands between a
script and the user's data.

The script, its inputs, what it printed and what it wrote are kept in its
folder, so the report can say what ran and the user can run it again.
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

#: Top-level modules a script may import. Scientific libraries and this
#: package; nothing that starts processes, opens sockets or edits the system.
ALLOWED_MODULES = frozenset({
    "numpy", "scipy", "pandas", "matplotlib", "pygimli", "PyHydroGeophysX",
    "math", "cmath", "statistics", "json", "csv", "re", "datetime", "collections",
    "itertools", "functools", "pathlib", "typing", "dataclasses", "decimal",
    "fractions", "string", "textwrap", "copy", "operator", "bisect", "heapq",
})

#: Built-in names a script may not call: each one is a way round the checks.
FORBIDDEN_NAMES = frozenset({
    "exec", "eval", "compile", "__import__", "globals", "locals", "vars",
    "getattr", "setattr", "delattr", "breakpoint", "input", "help",
})

#: Longer than this and nobody reads it before saying yes.
MAX_SOURCE_CHARS = 6000

#: Seconds a script may run before it is stopped.
DEFAULT_TIMEOUT_S = 180

#: What the assistant is told a script looks like. Shared by the chat and the
#: automatic run, so both write scripts the same checks accept.
SCRIPT_CONTRACT = (
    "Write Python with one function `def main(inputs, out):`. `inputs` is a dict of "
    "name -> file or folder path (strings) to read; `out` is a pathlib.Path folder, "
    "the only place you may write. Return a small JSON-serialisable dict saying what "
    "you found or did and naming the files you wrote (relative to `out`). Imports "
    "only from: numpy, scipy, pandas, matplotlib, pygimli, PyHydroGeophysX, math, "
    "statistics, json, csv, re, datetime, collections, itertools, functools, "
    "pathlib, typing, dataclasses. No os, sys, subprocess, shutil or network; no "
    "exec, eval, getattr or dunder attributes; never modify the input files. Keep it "
    f"short - under {MAX_SOURCE_CHARS} characters - so a person can read it before "
    "approving it.")

_SECRET = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL", re.IGNORECASE)


def check_script(source: str) -> str:
    """Why ``source`` may not run, or "" when it may be offered to the user.

    Parameters
    ----------
    source : str
        The script as the model wrote it.

    Returns
    -------
    str
        A sentence the model can act on to fix the script, or "".

    Raises
    ------
    None

    Examples
    --------
    >>> check_script("import numpy as np\\ndef main(inputs, out):\\n    return {}")
    ''
    >>> check_script("import os\\ndef main(inputs, out):\\n    os.remove('x')")
    'It imports os, which a temporary script may not use.'
    >>> check_script("def main(inputs, out):\\n    return ().__class__")
    'It uses __class__; dunder attributes are not allowed.'
    >>> check_script("print('hi')")
    'It must define main(inputs, out).'
    """
    text = str(source or "")
    if not text.strip():
        return "The script is empty."
    if len(text) > MAX_SOURCE_CHARS:
        return (f"It is {len(text)} characters long; keep it under {MAX_SOURCE_CHARS} "
                "so it can be read before it runs.")
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        return f"It does not parse: {exc.msg} (line {exc.lineno})."
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in ALLOWED_MODULES:
                    return f"It imports {alias.name}, which a temporary script may not use."
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if node.level or root not in ALLOWED_MODULES:
                return (f"It imports from {node.module or '.'}, which a temporary "
                        "script may not use.")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return f"It uses {node.attr}; dunder attributes are not allowed."
        elif isinstance(node, ast.Name):
            if node.id in FORBIDDEN_NAMES:
                return f"It calls {node.id}, which a temporary script may not use."
            if node.id.startswith("__"):
                return f"It uses {node.id}; dunder names are not allowed."
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            return "It uses global or nonlocal; keep everything inside main."
    mains = [node for node in tree.body
             if isinstance(node, ast.FunctionDef) and node.name == "main"]
    if not mains or len(mains[0].args.args) != 2:
        return "It must define main(inputs, out)."
    return ""


def imported_modules(source: str) -> List[str]:
    """The modules ``source`` imports, to load before the write guard goes up."""
    names: List[str] = []
    try:
        tree = ast.parse(str(source or ""))
    except SyntaxError:
        return names
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            names.append(node.module)
    return list(dict.fromkeys(names))


def _environment(out: Path) -> Dict[str, str]:
    """The child's environment: no credentials, this package importable, no GUI."""
    env = {key: value for key, value in os.environ.items() if not _SECRET.search(key)}
    package_root = str(Path(__file__).resolve().parents[3])
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (package_root, os.environ.get("PYTHONPATH", "")) if p)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["MPLBACKEND"] = "Agg"
    env["MPLCONFIGDIR"] = str(out / ".mplconfig")
    # A temporary directory of its own, inside its folder: the system one holds
    # other programs' files, and the script may write only where it was put.
    private = out / ".tmp"
    private.mkdir(parents=True, exist_ok=True)
    for key in ("TMP", "TEMP", "TMPDIR"):
        env[key] = str(private)
    return env


def _kill_tree(process: subprocess.Popen) -> None:
    """Stop the script and anything it started."""
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)],
                           capture_output=True, timeout=20)
        else:
            os.killpg(process.pid, 9)
    except Exception:  # noqa: BLE001 - fall back to the process alone
        pass
    try:
        process.kill()
    except Exception:  # noqa: BLE001
        pass


def run_script(source: str, inputs: Mapping[str, Any], out_dir: Any,
               purpose: str = "", timeout: float = DEFAULT_TIMEOUT_S) -> Dict[str, Any]:
    """Run an approved script in a process of its own and report what it did.

    Call only after the user has said yes: this does not ask.

    Parameters
    ----------
    source : str
        The script; checked again here.
    inputs : mapping
        Name -> path given to ``main`` as ``inputs``. Every path must exist.
    out_dir : str or Path
        The script's folder, created if needed; ``main`` gets it as ``out``.
    purpose : str, optional
        What the script is for, kept with it.
    timeout : float, optional
        Seconds before the script is stopped.

    Returns
    -------
    dict
        ``status`` ("ok" or "failed"), ``result`` (what ``main`` returned),
        ``files`` (written in ``out``, relative), ``printed`` (the end of its
        output), ``error``, ``folder`` and ``seconds``.

    Raises
    ------
    None
    """
    out = Path(out_dir).resolve()
    record: Dict[str, Any] = {"status": "failed", "result": None, "files": [],
                              "printed": "", "error": "", "folder": str(out),
                              "seconds": 0.0}
    problem = check_script(source)
    if problem:
        record["error"] = problem
        return record
    paths = {str(name): str(value) for name, value in dict(inputs or {}).items()}
    missing = [p for p in paths.values() if not Path(p).exists()]
    if missing:
        record["error"] = f"Input not found: {', '.join(missing)}."
        return record
    script = out / "temporary_script.py"
    manifest = out / "manifest.json"
    try:
        out.mkdir(parents=True, exist_ok=True)
        header = f"# {purpose}\n" if purpose else ""
        script.write_text(header + str(source), encoding="utf-8")
        manifest.write_text(json.dumps({
            "script": str(script), "out_dir": str(out), "inputs": paths,
            "imports": imported_modules(source), "purpose": purpose}, indent=2),
            encoding="utf-8")
    except OSError as exc:
        record["error"] = f"The script's folder could not be written: {exc}"
        return record
    child = Path(__file__).with_name("_scratch_child.py")
    started = time.monotonic()
    kwargs: Dict[str, Any] = {}
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    else:
        kwargs["start_new_session"] = True
    try:
        process = subprocess.Popen(
            [sys.executable, "-s", str(child), str(manifest)], cwd=str(out),
            env=_environment(out), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, **kwargs)
    except OSError as exc:
        record["error"] = f"The script could not be started: {exc}"
        return record
    try:
        output, _ = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_tree(process)
        output, _ = process.communicate()
        record["error"] = f"The script was stopped after {timeout:g} s."
    record["seconds"] = round(time.monotonic() - started, 1)
    text = (output or b"").decode("utf-8", errors="replace")
    lines = text.rstrip().splitlines()
    reply = None
    if lines:
        try:
            reply = json.loads(lines[-1])
        except ValueError:
            reply = None
    if isinstance(reply, dict):
        record.update(status="ok" if reply.get("ok") else "failed",
                      result=reply.get("result"), files=list(reply.get("files") or []),
                      printed=str(reply.get("printed") or "")[-3000:])
        if not reply.get("ok") and not record["error"]:
            record["error"] = str(reply.get("error") or "The script failed.")[-1500:]
    elif not record["error"]:
        record["error"] = ("The script ended without a result: "
                           + "\n".join(lines[-15:])[-1500:])
    try:
        (out / "result.json").write_text(json.dumps(record, indent=2, default=str),
                                         encoding="utf-8")
    except OSError:  # a record beside the script; what ran is in ``record``
        pass
    return record


def script_folder(root: Any, label: str = "script") -> Path:
    """A new folder for one script under ``root``, named by time."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    folder = Path(root) / f"{stamp}_{re.sub(r'[^A-Za-z0-9_-]+', '_', label)[:40]}"
    n = 1
    while folder.exists():
        n += 1
        folder = Path(root) / f"{stamp}_{label[:40]}_{n}"
    return folder


def describe_for_user(purpose: str, source: str, inputs: Mapping[str, Any]) -> str:
    """The question put to the user: what the script is for, what it reads, the code.

    The prose comes first and the code last, so the studio can show the code in
    a box of its own (``qt_apps.modules.base._split_code``).
    """
    reads = ", ".join(Path(str(p)).name for p in dict(inputs or {}).values())
    return (f"{purpose.strip()}\n\nIt reads {reads or 'nothing'} and writes only to a "
            "folder of its own; your files are not changed. It runs in a separate "
            "process with a time limit and no network. Read it before you decide:"
            f"\n\n{str(source).strip()}")


def split_answer(reply: Any) -> Tuple[str, str]:
    """``(purpose, code)`` from a model's JSON proposal, or ``("", "")``."""
    from ..._internal.utils import parse_json_object

    parsed = parse_json_object(str(reply or ""))
    if not isinstance(parsed, dict):
        return "", ""
    return str(parsed.get("purpose") or "")[:600], str(parsed.get("code") or "")
