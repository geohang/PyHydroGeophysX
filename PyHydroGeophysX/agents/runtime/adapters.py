"""A generated reader for a text table nobody anticipated, run only if approved.

The recurring failure this exists for: a coordinate file - electrodes, geophone
positions, a topography profile - that holds the right numbers in a layout the
readers were not written for. A comma-separated file with two header lines, a
station column where none was expected, an easting/northing/elevation triple
where an x/z pair was. The data are fine; the parser is not. A person would look
at the first few lines and write four lines of Python.

So can a model, and this lets it - under conditions that are the whole point of
the module:

**The user reads the code and approves it.** Every run of generated code is put
to :meth:`RunContext.ask` with the source in the question, and runs only on an
explicit yes. Without someone to ask, the answer is no: a headless run declines
rather than taking a default, because "nobody was there" must not read as
consent to execute code.

**It never touches the filesystem, the network or the process.** The function is
handed the file's *text*, not its path, and runs in a namespace with a small
builtins whitelist and no imports beyond numpy. It returns numbers.

**Its output must look like what it claimed to produce**: a non-empty 2-D array
of finite numbers with the expected column count, or it is rejected.

**It is written down.** The source, the approval and the file it produced are
recorded as warnings, because a number that reached a report through code a
model wrote must be traceable to that fact.

This is a guard rail, not a security boundary. A restricted ``exec`` namespace
stops the obvious reaches - ``open``, ``__import__``, ``subprocess`` - and a
determined escape through Python's object graph is not something it claims to
prevent. What actually stands between a user and bad code here is that they read
it and pressed the button.
"""

import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .context import RunContext

#: Configuration keys an adapter may rewrite: plain numeric text tables, where
#: "the layout was unexpected" is the whole failure and the repaired form is
#: unambiguous. Survey data files are absent on purpose - an instrument format
#: is not a table, and a generated parser for one would be guessing at physics
#: rather than at columns.
ADAPTABLE_FILES = {
    "electrode_file": 3,
    "geophone_file": 3,
    "topography_file": 2,
}

#: Names that must not appear in generated source. Checked before it is compiled,
#: so a proposal that reaches for them is never shown to the user as runnable.
FORBIDDEN = ("import", "open(", "exec", "eval", "compile", "__", "globals",
             "locals", "getattr", "setattr", "delattr", "vars", "input",
             "breakpoint", "exit", "quit", "help", "memoryview")

#: Everything the generated function is allowed to call.
SAFE_BUILTINS = {
    "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
    "divmod": divmod, "enumerate": enumerate, "filter": filter, "float": float,
    "int": int, "isinstance": isinstance, "len": len, "list": list, "map": map,
    "max": max, "min": min, "range": range, "reversed": reversed,
    "round": round, "set": set, "sorted": sorted, "str": str, "sum": sum,
    "tuple": tuple, "zip": zip, "ValueError": ValueError, "TypeError": TypeError,
    "IndexError": IndexError, "StopIteration": StopIteration,
}

#: How long a generated function may run. It parses a few hundred lines of text;
#: anything longer is a loop that will not end.
TIMEOUT_SECONDS = 5.0

#: The file name generated code is compiled under, which is how the tracer that
#: stops it recognises its frames and leaves every other frame untraced.
_SOURCE_NAME = "<generated adapter>"

#: How long a reader that ran out of time gets to notice it has been stopped.
#: It stops at its next line, so this is only ever waited in full by one stuck
#: inside a single long call into C.
_CANCEL_GRACE_SECONDS = 1.0

#: A bare ``except:`` catches the exception that stops a reader that ran out of
#: time, and the tracer that raised it is gone once it has fired - so a reader
#: looping inside ``try: ... except: continue`` could never be stopped.
_BARE_EXCEPT = re.compile(r"\bexcept\s*:")


class _Cancelled(BaseException):
    """Raised inside a generated reader to stop it once its time is up.

    A ``BaseException``, not an ``Exception``, so an ``except Exception`` in the
    generated code cannot swallow it.
    """

ADAPTER_PROMPT = """A coordinate file could not be read. Write a reader for it.

The file: {name}
It should hold {columns} numeric columns ({meaning}).
The reader failed with: {error}

The first lines of the file:
{sample}

Reply with JSON and nothing else:
{{"describe": "one sentence on what the layout actually is",
  "code": "def adapt(text):\\n    ..."}}

The function:
- is called `adapt` and takes the file's whole text as one string
- returns a list of rows, each row {columns} numbers, in the order given above
- uses no imports, no file access, and no name starting with an underscore
- skips headers, blank lines and comments itself; raises ValueError if the text
  is not what you described

Write the reader for the layout you actually see above, not a general one."""


def sample_text(path: Any, lines: int = 12, chars: int = 1200) -> str:
    """The first few lines of a text file, for a person and a model to look at."""
    try:
        text = Path(str(path)).read_text(encoding="utf-8-sig", errors="replace")
    except OSError:
        return ""
    return "\n".join(text.splitlines()[:lines])[:chars]


def check_source(code: str) -> str:
    """Why ``code`` may not be run, or "" when nothing objects.

    Examples
    --------
    >>> check_source('def adapt(text):\\n    return [[1.0, 2.0]]')
    ''
    >>> check_source('import os\\ndef adapt(text): return []')
    "generated code may not use 'import'"
    >>> check_source('x = 1')
    'generated code must define a function called adapt'
    """
    text = str(code or "")
    if "def adapt(" not in text:
        return "generated code must define a function called adapt"
    for name in FORBIDDEN:
        if name in text:
            return f"generated code may not use '{name}'"
    if _BARE_EXCEPT.search(text):
        return "generated code may not use a bare 'except:'; name the exception"
    if len(text) > 4000:
        return "generated code is too long to review"
    return ""


def run_adapter(code: str, text: str, columns: int,
                timeout: float = TIMEOUT_SECONDS) -> np.ndarray:
    """Compile and run ``code``'s ``adapt`` on ``text``, and check what came back.

    Parameters
    ----------
    code : str
        The generated source. Must already have passed :func:`check_source`.
    text : str
        The file's contents, passed as a value - the function never learns
        where the file is.
    columns : int
        How many numeric columns the result must have.
    timeout : float
        Seconds before the attempt is abandoned.

    Returns
    -------
    numpy.ndarray
        The rows, as ``(n, columns)`` floats.

    Raises
    ------
    ValueError
        If the source is rejected, the function raises, it runs too long, or
        what it returns is not a non-empty table of finite numbers of the right
        width. Every one of these is a normal outcome, not a bug: the point of
        checking is that generated code is not trusted to be right.

    Examples
    --------
    >>> code = 'def adapt(text):\\n    return [[float(v) for v in line.split(",")]' \\
    ...        '\\n            for line in text.splitlines() if line.strip()]'
    >>> run_adapter(code, '1,2\\n3,4\\n', 2).tolist()
    [[1.0, 2.0], [3.0, 4.0]]
    >>> run_adapter('def adapt(text): return []', '', 2)
    Traceback (most recent call last):
    ValueError: the generated reader returned no rows
    """
    import sys
    import threading

    objection = check_source(code)
    if objection:
        raise ValueError(objection)
    namespace: Dict[str, Any] = {"__builtins__": dict(SAFE_BUILTINS), "np": np}
    try:
        exec(compile(code, _SOURCE_NAME, "exec"), namespace)  # noqa: S102
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"the generated reader would not compile: {exc}")
    adapt = namespace.get("adapt")
    if not callable(adapt):
        raise ValueError("generated code must define a function called adapt")

    box: Dict[str, Any] = {}
    stop = threading.Event()

    def line(frame, event, arg):
        if stop.is_set():
            raise _Cancelled()
        return line

    def enter(frame, event, arg):
        # Only the generated code's own frames are traced line by line.
        return line if frame.f_code.co_filename == _SOURCE_NAME else None

    def call():
        # Python cannot kill a thread, and abandoning this one left a runaway
        # reader holding a core and the GIL for the rest of the process - every
        # later test in the same process crawled. So it stops itself: the
        # tracer raises at its next line once the time is up.
        previous = sys.gettrace()
        sys.settrace(enter)
        try:
            box["rows"] = adapt(text)
        except _Cancelled:
            pass
        except Exception as exc:  # noqa: BLE001
            box["error"] = exc
        finally:
            sys.settrace(previous)

    worker = threading.Thread(target=call, daemon=True, name="generated-adapter")
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        stop.set()
        worker.join(_CANCEL_GRACE_SECONDS)
        raise ValueError(f"the generated reader did not finish within {timeout:g}s")
    if "error" in box:
        raise ValueError(f"the generated reader failed: {box['error']}")

    rows = np.asarray(box.get("rows"), dtype=float)
    if rows.ndim == 1 and rows.size:
        rows = rows.reshape(1, -1)
    if rows.ndim != 2 or not rows.size:
        raise ValueError("the generated reader returned no rows")
    if rows.shape[1] != columns:
        raise ValueError(f"the generated reader returned {rows.shape[1]} columns, "
                         f"not the {columns} this file should have")
    if not np.all(np.isfinite(rows)):
        raise ValueError("the generated reader returned values that are not numbers")
    return rows


def propose(ctx: RunContext, key: str, path: Any, error: str,
            ask: Optional[Callable[[str], str]]) -> Tuple[str, str]:
    """Ask the model for a reader for this file. Returns ``(code, description)``.

    Returns ``("", "")`` when there is no model, the reply is unusable, or the
    code it proposed is not allowed to run.
    """
    import json

    columns = ADAPTABLE_FILES.get(key, 0)
    if not callable(ask) or not columns:
        return "", ""
    sample = sample_text(path)
    if not sample.strip():
        return "", ""
    meanings = {"electrode_file": "electrode index, x, elevation",
                "geophone_file": "receiver id, x, elevation",
                "topography_file": "x, elevation"}
    prompt = ADAPTER_PROMPT.format(
        name=Path(str(path)).name, columns=columns,
        meaning=meanings.get(key, "numbers"), error=str(error)[:400],
        sample=sample)
    try:
        reply = ask(prompt)
    except Exception:  # noqa: BLE001 - an unreachable model proposes nothing
        return "", ""
    match = re.search(r"\{.*\}", str(reply or ""), re.S)
    if not match:
        return "", ""
    try:
        parsed = json.loads(match.group(0))
    except ValueError:
        return "", ""
    code = str(parsed.get("code") or "")
    if check_source(code):
        return "", ""
    return code, str(parsed.get("describe") or "")[:300]


def review(ctx: RunContext, key: str, path: Any, code: str,
           description: str) -> bool:
    """Show the user the code and ask whether to run it.

    The answer is no unless somebody says yes. A run with nobody to ask declines
    - :meth:`RunContext.ask` returns its default, and the default here is "do
    not run it", because silence is not consent to execute code.
    """
    question = (
        f"{Path(str(path)).name} could not be read, and AQUAH has written a "
        f"reader for it. {description}\n\nIt will be given the file's text and "
        f"must return numbers; it cannot open files or use the network. Read it "
        f"before you decide:\n\n{code}")
    answer = ctx.ask(question, [
        {"id": "decline", "label": "Do not run it",
         "detail": "Leave the step undone. The report will say the file could "
                   "not be read."},
        {"id": "run", "label": "Run this code on the file",
         "detail": "Use it to read the file for this run only. Nothing is "
                   "written back to your original file, and the code and its "
                   "result are recorded in the report."}],
        default="decline")
    return answer == "run"


def adapt_file(ctx: RunContext, key: str, error: str,
               ask: Optional[Callable[[str], str]]) -> str:
    """Offer a generated reader for ``ctx.config[key]``, and use it if approved.

    Returns
    -------
    str
        The path of the rewritten file, now a plain numeric table the ordinary
        readers accept, or "" when nothing was produced. The rewritten file is
        inside the run's output directory; the user's own file is never touched.
    """
    path = ctx.config.get(key)
    columns = ADAPTABLE_FILES.get(key, 0)
    if not path or not columns or not Path(str(path)).is_file():
        return ""
    code, description = propose(ctx, key, path, error, ask)
    if not code:
        return ""
    if not review(ctx, key, path, code, description):
        ctx.note(f"A generated reader for {Path(str(path)).name} was offered and "
                 f"declined, so that file was left unread.")
        return ""
    try:
        text = Path(str(path)).read_text(encoding="utf-8-sig", errors="replace")
        rows = run_adapter(code, text, columns)
    except (OSError, ValueError) as exc:
        ctx.note(f"The generated reader for {Path(str(path)).name} was approved "
                 f"but did not work ({exc}), so that file was left unread.")
        return ""
    folder = Path(ctx.output_dir) / "adapted"
    folder.mkdir(parents=True, exist_ok=True)
    rewritten = folder / Path(str(path)).name
    np.savetxt(rewritten, rows, fmt="%.6f")
    (folder / f"{Path(str(path)).stem}_reader.py").write_text(code, encoding="utf-8")
    ctx.note(f"{Path(str(path)).name} was read by code AQUAH wrote and you "
             f"approved: {description} It produced {len(rows)} rows, written to "
             f"{rewritten.name} and used in place of the original for this run. "
             f"The code is saved beside it as "
             f"{Path(str(path)).stem}_reader.py. Your original file is unchanged.")
    return str(rewritten)
