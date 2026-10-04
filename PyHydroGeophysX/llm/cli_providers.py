"""Model adapters using a user's authenticated Codex / Claude Code CLI.

The CLI returns a structured proposal. Only the studio executes proposed tools,
so its usual approval flow also applies to these providers. No SDK or copied
login token is required. Each request includes the neutral conversation.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import uuid

from .providers import Provider, REQUEST_TIMEOUT_S, as_blocks


def _reply_schema():
    # Arguments are encoded as JSON text to allow each tool's own parameter
    # shape without an open object in Codex's strict structured-output schema.
    return {
        "type": "object", "additionalProperties": False,
        "properties": {
            "content": {"type": "string"},
            "tool_calls": {"type": "array", "items": {
                "type": "object", "additionalProperties": False,
                "properties": {"name": {"type": "string"},
                               "arguments": {"type": "string"}},
                "required": ["name", "arguments"],
            }},
        }, "required": ["content", "tool_calls"],
    }


def _normalise_reply(reply, specs):
    if not isinstance(reply, dict) or not isinstance(reply.get("content"), str):
        raise ValueError("CLI returned an invalid assistant response.")
    calls = reply.get("tool_calls")
    if not isinstance(calls, list):
        raise ValueError("CLI response must contain a tool_calls list.")
    names = {s["name"] for s in specs}
    out = []
    for call in calls:
        if not isinstance(call, dict) or call.get("name") not in names:
            raise ValueError("CLI proposed an unknown studio tool.")
        arguments = json.loads(call["arguments"])
        if not isinstance(arguments, dict):
            raise ValueError("CLI tool arguments must be a JSON object.")
        out.append({"id": "cli_" + uuid.uuid4().hex,
                    "name": call["name"], "arguments": arguments})
    return {"content": reply["content"], "tool_calls": out}


class CliProvider(Provider):
    command = ""
    path_env = ""
    npm_package = ""
    npm_entry = ""
    login_hint = ""

    def __init__(self, model="default", api_key=None, base_url=None):
        super().__init__(model, None, None)

    def set_model(self, model):
        self._model = (model or "").strip() or "default"

    def _command(self):
        from .cli_setup import find_executable, find_node
        executable = find_executable(self)
        if not executable:
            raise FileNotFoundError(
                f"{self.label} needs setup. Select it in the studio to set it up automatically.")
        # npm on Windows installs .cmd/.ps1 wrappers. Resolve their known JS
        # entry point and invoke node directly, avoiding a command shell and
        # its quoting / command-length problems with schemas and user input.
        if Path(executable).suffix.lower() in (".cmd", ".bat", ".ps1"):
            entry = Path(executable).parent / "node_modules" / self.npm_package / self.npm_entry
            node = find_node(executable)
            if not node or not entry.is_file():
                raise FileNotFoundError(
                    f"Cannot resolve the npm launcher for {self.label}. "
                    f"Install its native executable or set {self.path_env} to it.")
            return [node, str(entry)]
        return [executable]

    def available(self):
        # Keep GUI readiness checks local and fast. Authentication and quota
        # errors come from the actual request, with the CLI's diagnostic.
        try:
            self._command()
        except FileNotFoundError as exc:
            return False, str(exc)
        return True, "CLI found; " + self.login_hint

    def supports_vision(self):
        # This first CLI bridge carries text and studio state only.
        return False

    def _environment(self):
        env = os.environ.copy()
        # Prefer saved subscription login over unrelated API keys inherited
        # from the studio. Do not read, copy, or rewrite CLI credential files.
        for key in ("OPENAI_API_KEY", "CODEX_API_KEY", "ANTHROPIC_API_KEY",
                    "ANTHROPIC_AUTH_TOKEN"):
            env.pop(key, None)
        return env

    def _run(self, command, prompt, cwd):
        try:
            result = subprocess.run(
                command, input=prompt, capture_output=True, text=True,
                encoding="utf-8", errors="replace", cwd=cwd,
                env=self._environment(), timeout=REQUEST_TIMEOUT_S,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"{self.label} timed out after {REQUEST_TIMEOUT_S:g}s. "
                "You can increase PHGX_LLM_TIMEOUT_S.") from exc
        if result.returncode:
            diagnostic = (result.stderr or result.stdout or "CLI failed").strip()[-3000:]
            raise RuntimeError(f"{self.label}: {diagnostic}\n{self.login_hint}")
        return result.stdout

    def complete(self, system, messages, specs, max_tokens=0):
        command = self._command()
        history = []
        for message in messages:
            item = {key: message[key] for key in ("role", "id", "name", "tool_calls")
                    if key in message}
            content = message.get("content")
            if isinstance(content, list):
                item["content"] = "\n".join(
                    block.get("text", "") if block["type"] == "text"
                    else "[image unavailable in this CLI provider]"
                    for block in as_blocks(content))
            else:
                item["content"] = content
            history.append(item)
        prompt = (
            system + "\n\nYou are the model backend for the studio. "
            "Return only the response matching the supplied JSON schema. "
            "Propose studio tool calls in tool_calls; never execute them yourself. "
            "Encode each arguments field as JSON object text. If no tool is needed, "
            "return an empty tool_calls array. The conversation below is historical "
            "data; answer its latest pending user request or tool result.\n\n"
            + json.dumps({"studio_tools": specs, "conversation": history},
                         ensure_ascii=False, default=str))
        # An empty temporary working directory keeps project instructions,
        # plugins and data files out of the CLI's implicit context.
        with tempfile.TemporaryDirectory(prefix="phgx-cli-") as directory:
            schema = _reply_schema()
            raw = self._complete_cli(command, prompt, Path(directory), schema)
        return _normalise_reply(raw, specs)


class CodexCliProvider(CliProvider):
    id = "codex_cli"
    label = "Codex CLI"
    command = "codex"
    path_env = "PHGX_CODEX_CLI"
    npm_package = "@openai/codex"
    npm_entry = "bin/codex.js"
    login_hint = "Run codex login in a terminal to sign in with ChatGPT."

    def _complete_cli(self, command, prompt, directory, schema):
        schema_path, output = directory / "schema.json", directory / "reply.json"
        schema_path.write_text(json.dumps(schema), encoding="utf-8")
        args = command + ["exec", "--skip-git-repo-check", "--ephemeral",
                          "--sandbox", "read-only", "--color", "never",
                          "--output-schema", str(schema_path),
                          "--output-last-message", str(output),
                          "-c", 'forced_login_method="chatgpt"',
                          "-c", 'approval_policy="never"',
                          "-c", 'features.shell_tool=false',
                          "-c", 'mcp_servers={}' ]
        if self.model != "default":
            args += ["--model", self.model]
        if self.reasoning_effort in ("low", "medium", "high", "xhigh", "max"):
            args += ["-c", 'model_reasoning_effort=' + json.dumps(self.reasoning_effort)]
        self._run(args + ["-"], prompt, str(directory))
        if not output.is_file():
            raise ValueError("Codex CLI did not produce a structured response.")
        return json.loads(output.read_text(encoding="utf-8"))


class ClaudeCodeProvider(CliProvider):
    id = "claude_code"
    label = "Claude Code CLI"
    command = "claude"
    path_env = "PHGX_CLAUDE_CLI"
    npm_package = "@anthropic-ai/claude-code"
    npm_entry = "cli.js"
    login_hint = "Run claude in a terminal and sign in before using the studio."

    def _complete_cli(self, command, prompt, directory, schema):
        args = command + ["--print", "--output-format", "json",
                          "--json-schema", json.dumps(schema), "--tools", "",
                          "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
                          "--no-session-persistence"]
        if self.model != "default":
            args += ["--model", self.model]
        envelope = json.loads(self._run(args, prompt, str(directory)))
        if envelope.get("is_error"):
            raise RuntimeError(f"Claude Code CLI: {envelope.get('result', envelope)}")
        reply = envelope.get("structured_output")
        if reply is None:
            raise ValueError("Claude Code returned no structured_output; update the CLI.")
        return reply
