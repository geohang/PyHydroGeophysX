"""Provider abstraction for the AQUAH assistant: OpenAI, Claude, compatible.

The chat panel keeps a single *provider-neutral* conversation and a neutral tool
spec list; each provider adapter translates that to its own wire format, makes
one tool-enabled call, and normalises the reply back. This lets the user switch
provider and model at runtime without the panel knowing any provider details.

Neutral shapes used throughout:

- tool spec:        ``{"name", "description", "parameters"(JSON schema)}``
- message (user):   ``{"role": "user", "content": str | [block]}``
- message (asst):   ``{"role": "assistant", "content": str|None,
                       "tool_calls": [{"id", "name", "arguments"(dict)}],
                       "_anthropic_content"(optional, opaque)}``
- message (tool):   ``{"role": "tool", "id": str, "name": str, "content": any}``
- complete() reply: ``{"content": str|None, "tool_calls": [...],
                       "_anthropic_content"(optional)}``
- block (text):     ``{"type": "text", "text": str}``
- block (image):    ``{"type": "image", "media_type": str, "data": str(base64)}``

Only ``user`` messages carry blocks, and only they may hold images. That is not a
simplification: the OpenAI chat-completions API accepts images in a ``user`` part
list but requires ``role: "tool"`` content to be a plain string, so a figure
produced by a tool has to travel as a separate follow-up user message. Anthropic
would allow an image inside ``tool_result``, but routing both providers the same
way keeps one code path and one transcript shape.

The ``openai`` / ``anthropic`` SDKs are imported lazily, so importing this module
never requires either to be installed.
"""

from __future__ import annotations

import importlib.util
import json
import os
from typing import Any, Dict, List, Optional, Tuple

#: Per-request timeout in seconds for LLM calls. The SDK defaults (about ten
#: minutes) leave a chat UI hanging when a provider stalls; override with the
#: PHGX_LLM_TIMEOUT_S environment variable when longer calls are expected.
REQUEST_TIMEOUT_S = float(os.getenv("PHGX_LLM_TIMEOUT_S", "120"))


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


# -- neutral content blocks ---------------------------------------------------
#: Substituted for a figure that has aged out of the image budget, so the model
#: knows a picture existed at that point instead of silently losing the step.
IMAGE_PLACEHOLDER = "[figure from an earlier step, dropped to save context]"


def text_block(text: str) -> Dict[str, Any]:
    """A neutral text block."""
    return {"type": "text", "text": text}


def image_block(data: str, media_type: str = "image/png") -> Dict[str, Any]:
    """A neutral image block from base64-encoded image ``data``."""
    return {"type": "image", "media_type": media_type, "data": data}


def as_blocks(content: Any) -> List[Dict[str, Any]]:
    """Normalise neutral message content to a list of well-formed blocks."""
    if isinstance(content, list):
        return [b for b in content
                if isinstance(b, dict) and b.get("type") in ("text", "image")]
    return [text_block("" if content is None else str(content))]


def has_image(content: Any) -> bool:
    """Whether neutral message content carries at least one image block."""
    return isinstance(content, list) and any(
        isinstance(b, dict) and b.get("type") == "image" for b in content
    )


def _to_openai_content(content: Any) -> Any:
    """User content -> a plain string, or a multimodal part list when blocks are used."""
    if not isinstance(content, list):
        return content if content is not None else ""
    parts: List[Dict[str, Any]] = []
    for b in as_blocks(content):
        if b["type"] == "text":
            parts.append({"type": "text", "text": b.get("text", "")})
        else:
            media = b.get("media_type", "image/png")
            parts.append({"type": "image_url",
                          "image_url": {"url": f"data:{media};base64,{b.get('data', '')}"}})
    return parts


def _to_anthropic_content(content: Any) -> Any:
    """User content -> a plain string, or an Anthropic content-block list."""
    if not isinstance(content, list):
        return content if content is not None else ""
    parts: List[Dict[str, Any]] = []
    for b in as_blocks(content):
        if b["type"] == "text":
            parts.append({"type": "text", "text": b.get("text", "")})
        else:
            parts.append({"type": "image", "source": {
                "type": "base64",
                "media_type": b.get("media_type", "image/png"),
                "data": b.get("data", ""),
            }})
    return parts


# -- neutral -> provider messages ---------------------------------------------
def to_openai_messages(system: str, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = [{"role": "system", "content": system}]
    for m in messages:
        role = m.get("role")
        if role == "user":
            out.append({"role": "user", "content": _to_openai_content(m.get("content", ""))})
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


def _provider_blocks(content: Any) -> List[Dict[str, Any]]:
    """Provider-shaped content -> a block list, preserving blocks already built.

    Unlike :func:`as_blocks` this keeps every block type, so ``tool_result``
    entries survive a merge. Empty text is dropped, as the API rejects it.
    """
    if isinstance(content, list):
        return [b for b in content
                if not (isinstance(b, dict) and b.get("type") == "text"
                        and not (b.get("text") or "").strip())]
    text = "" if content is None else str(content)
    return [{"type": "text", "text": text}] if text.strip() else []


def _push_user(out: List[Dict[str, Any]], content: Any) -> None:
    """Append a user message, merging it into a preceding user message.

    Three neutral entries can land next to each other on the user side: the tool
    results answering an assistant turn, a follow-up screenshot, and whatever the
    user types next. Anthropic pairs one assistant turn with the single user turn
    that answers it, so they are concatenated into one message.
    """
    blocks = _provider_blocks(content)
    if not blocks:
        return
    if out and out[-1].get("role") == "user":
        out[-1] = {"role": "user", "content": _provider_blocks(out[-1]["content"]) + blocks}
        return
    out.append({"role": "user", "content": blocks})


def to_anthropic_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Translate neutral messages to Anthropic format (system is passed separately).

    Consecutive ``tool`` entries are coalesced into a single ``user`` message of
    ``tool_result`` blocks, as the Messages API requires, and adjacent user
    messages are merged by :func:`_push_user`.
    """
    out: List[Dict[str, Any]] = []
    pending_results: List[Dict[str, Any]] = []

    def flush_results() -> None:
        nonlocal pending_results
        if pending_results:
            if out and out[-1].get("role") == "user":
                out[-1] = {"role": "user",
                           "content": _provider_blocks(out[-1]["content"]) + pending_results}
            else:
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
            _push_user(out, _to_anthropic_content(m.get("content", "")))
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


def cap_images(messages: List[Dict[str, Any]], max_images: int = 2) -> List[Dict[str, Any]]:
    """Keep image blocks only in the ``max_images`` most recent messages carrying one.

    A screenshot costs roughly one to two thousand tokens on every subsequent
    call, so a session that captures a figure per step would re-send the whole
    gallery each turn. Older images become :data:`IMAGE_PLACEHOLDER` text; the
    input list is never mutated.
    """
    if max_images < 0:
        return list(messages)
    out = list(messages)
    kept = 0
    for i in range(len(out) - 1, -1, -1):
        content = out[i].get("content")
        if not has_image(content):
            continue
        if kept < max_images:
            kept += 1
            continue
        stripped = [b for b in as_blocks(content) if b["type"] != "image"]
        stripped.append(text_block(IMAGE_PLACEHOLDER))
        out[i] = {**out[i], "content": stripped}
    return out


def window_messages(messages: List[Dict[str, Any]], max_items: int = 60,
                    max_images: int = 2) -> List[Dict[str, Any]]:
    """Trim a long neutral conversation to roughly ``max_items`` recent entries.

    Long chats grow the prompt (and the bill) without bound. The cut always
    lands on a ``user`` text turn so assistant ``tool_calls`` never lose their
    paired ``tool`` results, which the Anthropic Messages API requires. When no
    safe boundary exists in the tail, every entry is kept. Images are capped
    separately by :func:`cap_images`, because a handful of figures can outweigh
    a long text transcript.
    """
    if max_items <= 0 or len(messages) <= max_items:
        return cap_images(messages, max_images)
    start = len(messages) - max_items
    while start < len(messages) and messages[start].get("role") != "user":
        start += 1
    if start >= len(messages):
        return cap_images(messages, max_images)
    return cap_images(messages[start:], max_images)


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

    def supports_vision(self) -> bool:
        """Whether the configured model accepts image blocks in user messages."""
        return supports_vision(self.id, self._model)

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
            kwargs: Dict[str, Any] = {"api_key": self._api_key, "timeout": REQUEST_TIMEOUT_S}
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
            self._client = anthropic.Anthropic(api_key=self._api_key, timeout=REQUEST_TIMEOUT_S)
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
        "vision": True,
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
        "vision": True,
    },
    "openai_compatible": {
        "label": "OpenAI-compatible",
        "env_key": "OPENAI_COMPAT_API_KEY",
        "model_env": "OPENAI_COMPAT_MODEL",
        "base_url_env": "OPENAI_COMPAT_BASE_URL",
        "models": ["deepseek-chat", "gpt-4o-mini"],
        "default_model": "deepseek-chat",
        "needs_base_url": True,
        # The wire format is OpenAI's, so a vision-capable endpoint (OpenRouter,
        # a local Qwen-VL) works; the text-only models below are excluded by name.
        "vision": True,
    },
}

#: Model-name fragments that identify a text-only model on an otherwise
#: vision-capable provider. Matched case-insensitively as a substring, so
#: version suffixes ("deepseek-chat-v3") are covered without new entries.
TEXT_ONLY_MODEL_HINTS = ("deepseek-chat", "deepseek-reasoner", "deepseek-coder")


def supports_vision(provider_id: str, model: Optional[str]) -> bool:
    """Whether ``model`` on ``provider_id`` can read image blocks.

    Used to gate the workbench ``capture_view`` tool: offering a screenshot tool
    to a text-only model wastes a turn and returns an API error.
    """
    meta = PROVIDER_META.get(provider_id) or {}
    if not meta.get("vision", False):
        return False
    name = (model or "").lower()
    return not any(hint in name for hint in TEXT_ONLY_MODEL_HINTS)

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
