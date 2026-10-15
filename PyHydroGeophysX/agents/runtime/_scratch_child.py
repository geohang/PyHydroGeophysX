"""Run one approved temporary script: the child process of ``scratch.run_script``.

Started as a file, not as a module of the package, so nothing is imported
before the script's own imports. Standard library only. The last line it prints
is ``{"ok", "result", "files", "printed", "error"}`` as JSON.
"""

import contextlib
import importlib
import io
import json
import os
import sys
import traceback
from pathlib import Path

#: Audit events refused outright: starting processes, the network, native code.
_DENIED = ("subprocess.", "os.system", "os.exec", "os.spawn", "os.posix_spawn",
           "os.fork", "os.forkpty", "os.startfile", "os.kill", "socket.",
           "ctypes.", "winreg.", "_winapi.CreateProcess", "webbrowser.",
           "sys.addaudithook")

#: Audit events that change the file system, with the argument positions that
#: name the paths they change.
_WRITES = {"os.remove": (0,), "os.rmdir": (0,), "os.rename": (0, 1),
           "os.replace": (0, 1), "os.mkdir": (0,), "os.chmod": (0,),
           "os.truncate": (0,), "os.link": (1,), "os.symlink": (1,),
           "shutil.rmtree": (0,), "shutil.move": (0, 1), "shutil.copyfile": (1,),
           "shutil.copytree": (1,), "os.utime": (0,)}

_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC


def _guard(roots):
    roots = [str(Path(r).resolve()).lower() for r in roots]

    def inside(path):
        if isinstance(path, int):
            return True  # an already open descriptor
        try:
            target = str(Path(os.fsdecode(path)).resolve()).lower()
        except Exception:  # noqa: BLE001 - an unreadable path is refused
            return False
        return any(target == r or target.startswith(r + os.sep) for r in roots)

    def hook(event, args):
        if event.startswith(_DENIED):
            raise PermissionError(f"A temporary script may not use {event}.")
        if event == "open":
            path, mode, flags = (list(args) + [None, None, None])[:3]
            writing = (any(c in str(mode or "") for c in "wax+")
                       or (isinstance(flags, int) and flags & _WRITE_FLAGS))
            if writing and not inside(path):
                raise PermissionError(f"A temporary script may write only in its "
                                      f"own folder, not {path}.")
        elif event in _WRITES:
            for index in _WRITES[event]:
                if index < len(args) and not inside(args[index]):
                    raise PermissionError(f"A temporary script may change files only "
                                          f"in its own folder, not {args[index]}.")
    return hook


def _plain(value):
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        if isinstance(value, dict):
            return {str(k): _plain(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_plain(v) for v in value]
        if hasattr(value, "tolist"):
            return value.tolist()
        return str(value)


def main(manifest_path):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    out = Path(manifest["out_dir"]).resolve()
    source = Path(manifest["script"]).read_text(encoding="utf-8")
    # The script's libraries load before the guard goes up: a library writes its
    # own caches and settings when imported, outside the script's folder.
    for name in manifest.get("imports") or []:
        try:
            importlib.import_module(name)
        except Exception:  # noqa: BLE001 - the script's own import reports it
            pass
    before = {p for p in out.rglob("*") if p.is_file()}
    # The temporary directory is inside it (scratch._environment).
    sys.addaudithook(_guard([out]))
    printed = io.StringIO()
    reply = {"ok": False, "result": None, "files": [], "printed": "", "error": ""}
    try:
        namespace = {"__name__": "temporary_script"}
        with contextlib.redirect_stdout(printed):
            exec(compile(source, "temporary_script.py", "exec"), namespace)
            result = namespace["main"](dict(manifest.get("inputs") or {}), out)
        reply.update(ok=True, result=_plain(result))
    except BaseException:  # noqa: BLE001 - every failure is reported, not raised
        reply["error"] = traceback.format_exc(limit=6)[-2500:]
    skip = {"temporary_script.py", "manifest.json", "result.json"}
    reply["files"] = sorted(str(p.relative_to(out)) for p in out.rglob("*")
                            if p.is_file() and p not in before and p.name not in skip
                            and ".mplconfig" not in p.parts and ".tmp" not in p.parts)
    reply["printed"] = printed.getvalue()[-3000:]
    sys.stdout.write("\n" + json.dumps(reply, default=str) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main(sys.argv[1])
