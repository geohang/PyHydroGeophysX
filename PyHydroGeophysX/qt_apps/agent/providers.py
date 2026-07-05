"""Provider abstraction for the AQUAH assistant: OpenAI, Claude, compatible.

The chat panel keeps a single *provider-neutral* conversation and a neutral tool
spec list; each provider adapter translates that to its own wire format, makes
one tool-enabled call, and normalises the reply back. This lets the user switch
provider and model at runtime without the panel knowing any provider details.

Neutral shapes used throughout:

- tool spec:        ``{"name", "description", "parameters"(JSON schema)}``
- message (user):   ``{"role": "user", "content": str}``
- message (asst):   ``{"role": "assistant", "content": str|None,
                       "tool_calls": [{"id", "name", "arguments"(dict)}],
                       "_anthropic_content"(optional, opaque)}``
- message (tool):   ``{"role": "tool", "id": str, "name": str, "content": any}``
- complete() reply: ``{"content": str|None, "tool_calls": [...],
                       "_anthropic_content"(optional)}``

The ``openai`` / ``anthropic`` SDKs are imported lazily, so importing this module
never requires either to be installed.
"""

from __future__ import annotations

import importlib.util
import json
import os
from typing import Any, Dict, List, Optional, Tuple


# -- neutral -> provider tool schemas -----------------------------------------
def to_openai_tools(specs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {"type": "function", "function": {
            "name": s["name"], "description": s["description"], "parameters": s["parameters"]}}
        for s in specs
    ]


def to_anthropic_tools(specs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {"name": s["name"], "description": s["description"], "input_schema": s["parameters"]}
        for s in specs
    ]


def _as_text(content: Any) -> str:
    return content if isinstance(content, str) else json.dumps(content, default=str)


# -- neutral -> provider messages ---------------------------------------------
def to_openai_messages(system: str, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = [{"role": "system", "content": system}]
    for m in messages:
        role = m.get("role")
        if role == "user":
            out.append({"role": "user", "content": m.get("content", "")})
        elif role == "assistant":
            item: Dict[str, Any] = {"role": "assistant", "content": m.get("content")}
            calls = m.get("tool_calls") or []
            if calls:
                item["tool_calls"] = [
                    {"id": c["id"], "type": "function",
                     "function": {"name": c["name"], "arguments": json.dumps(c.get("arguments") or {})}}
                    for c in calls
                ]
            out.append(item)
        elif role == "tool":
            out.append({"role": "tool", "tool_call_id": m.get("id"),
                        "content": _as_text(m.get("content"))})
    return out


def to_anthropic_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Translate neutral messages to Anthropic format (system is passed separately).

    Consecutive ``tool`` entries are coalesced into a single ``user`` message of
    ``tool_result`` blocks, as the Messages API requires.
    """
    out: List[Dict[str, Any]] = []
    pending_results: List[Dict[str, Any]] = []

    def flush_results() -> None:
        nonlocal pending_results
        if pending_results:
            out.append({"role": "user", "content": pending_results})
            pending_results = []

    for m in messages:
        role = m.get("role")
        if role == "tool":
            pending_results.append({
                "type": "tool_result",
                "tool_use_id": m.get("id"),
                "content": _as_text(m.get("content")),
            })
            continue
        flush_results()
        if role == "user":
            out.append({"role": "user", "content": m.get("content", "")})
        elif role == "assistant":
            raw = m.get("_anthropic_content")
            if raw is not None:
                out.append({"role": "assistant", "content": raw})
                continue
            blocks: List[Dict[str, Any]] = []
            if m.get("content"):
                blocks.append({"type": "text", "text": m["content"]})
            for c in (m.get("tool_calls") or []):
                blocks.append({"type": "tool_use", "id": c["id"], "name": c["name"],
                               "input": c.get("arguments") or {}})
            out.append({"role": "assistant", "content": blocks or [{"type": "text", "text": ""}]})
    flush_results()
    return out


# -- providers ----------------------------------------------------------------
class Provider:
    """Base class: owns model + key config and one tool-enabled ``complete``."""

    id = ""
    label = ""

    def __init__(self, model: str, api_key: Optional[str], base_url: Optional[str] = None) -> None:
        self._model = model
        self._api_key = api_key or None
        self._base_url = base_url or None
        self._client = None

    @property
    def model(self) -> str:
        return self._model

    def set_model(self, model: str) -> None:
        model = (model or "").strip()
        if model:
            self._model = model

    def set_api_key(self, api_key: str) -> None:
        self._api_key = (api_key or "").strip() or None
        self._client = None

    def set_base_url(self, base_url: str) -> None:
        self._base_url = (base_url or "").strip() or None
        self._client = None

    def available(self) -> Tuple[bool, str]:  # pragma: no cover - overridden
        raise NotImplementedError

    def complete(self, system: str, messages, specs, max_tokens: int = 0) -> Dict[str, Any]:  # pragma: no cover
        raise NotImplementedError


class OpenAIProvider(Provider):
    id = "openai"
    label = "OpenAI"
    env_key = "OPENAI_API_KEY"

    def available(self) -> Tuple[bool, str]:
        if importlib.util.find_spec("openai") is None:
            return False, "The 'openai' package is not installed. Run: pip install openai"
        if not self._api_key:
            return False, f"No API key. Set {self.env_key} or paste a key below."
        return True, "ready"

    def _ensure_client(self):
        import openai

        if self._client is None:
            kwargs: Dict[str, Any] = {"api_key": self._api_key}
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._client = openai.OpenAI(**kwargs)
        return self._client

    def complete(self, system, messages, specs, max_tokens: int = 1024) -> Dict[str, Any]:
        client = self._ensure_client()
        response = client.chat.completions.create(
            model=self._model,
            messages=to_openai_messages(system, messages),
            tools=to_openai_tools(specs),
            tool_choice="auto",
            temperature=0.2,
        )
        message = response.choices[0].message
        out: Dict[str, Any] = {"content": message.content, "tool_calls": []}
        for tc in (getattr(message, "tool_calls", None) or []):
            try:
                args = json.loads(tc.function.arguments or "{}")
            except Exception:
                args = {}
            out["tool_calls"].append({"id": tc.id, "name": tc.function.name, "arguments": args})
        return out


class OpenAICompatibleProvider(OpenAIProvider):
    """Any OpenAI-compatible chat-completions endpoint (DeepSeek, OpenRouter, local…)."""

    id = "openai_compatible"
    label = "OpenAI-compatible"
    env_key = "OPENAI_COMPAT_API_KEY"

    def available(self) -> Tuple[bool, str]:
        ok, reason = super().available()
        if not ok:
            return ok, reason
        if not self._base_url:
            return False, "Set a Base URL for the OpenAI-compatible endpoint."
        return True, "ready"


class AnthropicProvider(Provider):
    id = "anthropic"
    label = "Claude (Anthropic)"
    env_key = "ANTHROPIC_API_KEY"

    def available(self) -> Tuple[bool, str]:
        if importlib.util.find_spec("anthropic") is None:
            return False, "The 'anthropic' package is not installed. Run: pip install anthropic"
        if not self._api_key:
            return False, f"No API key. Set {self.env_key} or paste a key below."
        return True, "ready"

    def _ensure_client(self):
        import anthropic

        if self._client is None:
            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def complete(self, system, messages, specs, max_tokens: int = 4096) -> Dict[str, Any]:
        client = self._ensure_client()
        kwargs: Dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens,
            "messages": to_anthropic_messages(messages),
            "tools": to_anthropic_tools(specs),
        }
        if system:
            kwargs["system"] = system
        response = client.messages.create(**kwargs)
        text_parts: List[str] = []
        tool_calls: List[Dict[str, Any]] = []
        for block in response.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                text_parts.append(block.text)
            elif btype == "tool_use":
                tool_calls.append({"id": block.id, "name": block.name, "arguments": block.input})
        return {
            "content": "".join(text_parts) or None,
            "tool_calls": tool_calls,
            # Echo the raw blocks verbatim on the next turn (preserves thinking +
            # tool_use exactly, which Claude requires on same-model continuation).
            "_anthropic_content": list(response.content),
        }


# -- registry -----------------------------------------------------------------
PROVIDER_ORDER = ["openai", "anthropic", "openai_compatible"]

PROVIDER_META: Dict[str, Dict[str, Any]] = {
    "openai": {
        "label": "OpenAI",
        "env_key": "OPENAI_API_KEY",
        "model_env": "OPENAI_MODEL",
        # gpt-4.1 follows tool/agent instructions much better than 4o-mini and is
        # still inexpensive; mini variants are cheaper, gpt-4o is the older mid-tier.
        "models": ["gpt-4.1", "gpt-4.1-mini", "gpt-4o", "gpt-4o-mini"],
        "default_model": "gpt-4.1",
        "needs_base_url": False,
    },
    "anthropic": {
        "label": "Claude (Anthropic)",
        "env_key": "ANTHROPIC_API_KEY",
        "model_env": "ANTHROPIC_MODEL",
        # Sonnet is the speed/intelligence sweet spot and far cheaper than Opus;
        # claude-sonnet-5 is the current Sonnet generation (near-Opus coding quality
        # at the same price, intro pricing through 2026-08-31).
        "models": ["claude-sonnet-5", "claude-sonnet-4-6", "claude-opus-4-8", "claude-haiku-4-5", "claude-opus-4-7"],
        "default_model": "claude-sonnet-5",
        "needs_base_url": False,
    },
    "openai_compatible": {
        "label": "OpenAI-compatible",
        "env_key": "OPENAI_COMPAT_API_KEY",
        "model_env": "OPENAI_COMPAT_MODEL",
        "base_url_env": "OPENAI_COMPAT_BASE_URL",
        "models": ["deepseek-chat", "gpt-4o-mini"],
        "default_model": "deepseek-chat",
        "needs_base_url": True,
    },
}

_PROVIDER_CLASSES = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "openai_compatible": OpenAICompatibleProvider,
}


def make_provider(
    provider_id: str,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Provider:
    """Build a provider, filling model/key/base_url from the environment by default."""
    meta = PROVIDER_META.get(provider_id, PROVIDER_META["openai"])
    cls = _PROVIDER_CLASSES.get(provider_id, OpenAIProvider)
    if model is None:
        env_model = os.getenv(meta["model_env"]) if meta.get("model_env") else None
        model = env_model or meta["default_model"]
    if api_key is None:
        api_key = os.getenv(meta["env_key"])
    if base_url is None and meta.get("base_url_env"):
        base_url = os.getenv(meta["base_url_env"])
    return cls(model=model, api_key=api_key, base_url=base_url)
