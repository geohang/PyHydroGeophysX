"""Read Codex's model picker through its local app-server protocol.

Only initialization and model/list are sent: no inference, thread or tool run.
The CLI owns authentication and catalog refresh, just as it does for exec.
"""

from __future__ import annotations

import json
from pathlib import Path
import queue
import subprocess
import threading
import time

from .. import __version__


def list_codex_models(provider, timeout: float = 20) -> list[str]:
    """Return visible model IDs from the CLI used by this provider.

    Run off the GUI thread. A single deadline bounds the handshake and all
    catalog pages; closing the short-lived server also stops its pipe reader.
    """
    deadline = time.monotonic() + timeout
    # Stdio is the default transport, including versions before --listen.
    process = subprocess.Popen(
        provider._command() + ["app-server",
                               "-c", 'forced_login_method="chatgpt"',
                               "-c", "mcp_servers={}"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
        cwd=str(Path.home()), env=provider._environment(),
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    inbox = queue.Queue()

    def read_stdout():
        try:
            for line in process.stdout:
                try:
                    message = json.loads(line)
                except ValueError:
                    continue
                if isinstance(message, dict):
                    inbox.put(message)
        finally:
            inbox.put(None)

    reader = threading.Thread(target=read_stdout, daemon=True,
                              name="codex-model-catalog")
    reader.start()

    def send(message):
        try:
            process.stdin.write(json.dumps(message) + "\n")
            process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise RuntimeError("Codex model discovery could not start.") from exc

    def request(method, params, request_id):
        send({"method": method, "id": request_id, "params": params})
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError("Codex model discovery timed out.")
            try:
                message = inbox.get(timeout=remaining)
            except queue.Empty as exc:
                raise RuntimeError("Codex model discovery timed out.") from exc
            if message is None:
                raise RuntimeError("Codex closed before returning its model list.")
            if message.get("id") != request_id:
                continue  # Unrelated app-server notifications.
            if "error" in message:
                error = message["error"]
                detail = error.get("message", "Request failed") if isinstance(error, dict) else str(error)
                raise RuntimeError("Codex model discovery: " + detail)
            result = message.get("result")
            if not isinstance(result, dict):
                raise RuntimeError("Codex returned an invalid model catalog response.")
            return result

    try:
        request("initialize", {"clientInfo": {
            "name": "pyhydrogeophysx_studio", "version": __version__}}, 1)
        send({"method": "initialized", "params": {}})
        models, cursor, cursors = [], None, set()
        for request_id in range(2, 22):
            params = {"limit": 100, "includeHidden": False}
            if cursor:
                params["cursor"] = cursor
            result = request("model/list", params, request_id)
            entries = result.get("data")
            if not isinstance(entries, list):
                raise RuntimeError("Codex returned an invalid model list.")
            for entry in entries:
                if not isinstance(entry, dict) or entry.get("hidden"):
                    continue
                model = entry.get("model") or entry.get("id")
                if isinstance(model, str) and model.strip() and model not in models:
                    models.append(model)
            cursor = result.get("nextCursor")
            if not cursor:
                return models
            if not isinstance(cursor, str) or cursor in cursors:
                raise RuntimeError("Codex returned an invalid model list cursor.")
            cursors.add(cursor)
        raise RuntimeError("Codex returned too many model catalog pages.")
    finally:
        if process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)
        reader.join(timeout=1)
        process.stdin.close()
        process.stdout.close()
