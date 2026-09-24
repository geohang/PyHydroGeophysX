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

Model choice is offered as a three-step cost ladder rather than a flat list -
see :data:`MODEL_TIERS`. Level 1 answers most requests, level 2 handles coding
and agent work, level 3 exists for what the levels below could not finish. Both
the desktop chat panel and the Streamlit sidebar read that one registry, so the
two surfaces always offer the same levels at the same prices.

The ``openai`` / ``anthropic`` SDKs are imported lazily, so importing this module
never requires either to be installed.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import threading
from typing import Any, Dict, List, Optional, Tuple

#: Per-request timeout in seconds for LLM calls. The SDK defaults (about ten
#: minutes) leave a chat UI hanging when a provider stalls; override with the
#: PHGX_LLM_TIMEOUT_S environment variable when longer calls are expected.
REQUEST_TIMEOUT_S = float(os.getenv("PHGX_LLM_TIMEOUT_S", "120"))

#: How long an idle connection to the provider is kept for the next request, in
#: seconds. The SDKs drop one after 5 s, which is less time than a person takes
#: to read a proposed step and approve it, so step-by-step chat opened a new TCP
#: and TLS connection before nearly every request. Override with the
#: PHGX_LLM_KEEPALIVE_S environment variable; it never goes below the SDK's own.
KEEPALIVE_S = float(os.getenv("PHGX_LLM_KEEPALIVE_S", "120"))


#: Held while a provider's client is built or discarded. The client is built on
#: first use, and first use can now come from two threads at once - the warm-up
#: started when the user begins typing, and the request itself.
_CLIENT_LOCK = threading.Lock()


def keepalive_http_client(sdk: Any) -> Optional[Any]:
    """The SDK's own HTTP client, keeping idle connections for :data:`KEEPALIVE_S`.

    Only the idle expiry differs from what the SDK would build for itself;
    connection limits, redirects, proxies and timeouts stay the SDK's defaults.
    Shared with ``BaseAgent``, whose workflow agents talk to the same APIs.

    Parameters
    ----------
    sdk : module
        The imported ``openai`` or ``anthropic`` package.

    Returns
    -------
    httpx.Client or None
        None when the SDK does not expose its default client and limits, which
        leaves the SDK to build its own exactly as before.
    """
    base = getattr(sdk, "DefaultHttpxClient", None)
    limits = getattr(sdk, "DEFAULT_CONNECTION_LIMITS", None)
    if base is None or limits is None:
        return None
    import httpx

    class _KeptAliveClient(base):
        """Closed when dropped, as the client the SDK builds for itself is.

        A provider's client is replaced whenever the key, the endpoint or the
        provider changes, and nothing else would close the old one's sockets.
        """

        def __del__(self) -> None:
            try:
                if not self.is_closed:
                    self.close()
            except Exception:  # noqa: BLE001 - may run during interpreter shutdown
                pass

    expiry = limits.keepalive_expiry
    if expiry is not None:
        expiry = max(KEEPALIVE_S, float(expiry))
    return _KeptAliveClient(limits=httpx.Limits(
        max_connections=limits.max_connections,
        max_keepalive_connections=limits.max_keepalive_connections,
        keepalive_expiry=expiry))


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


def to_responses_input(messages):
    """Keep reasoning and function calls together across stateless tool turns."""
    items = []
    for message in messages:
        role = message.get('role')
        if role == 'assistant' and message.get('_openai_output'):
            items.extend(message['_openai_output'])
        elif role == 'tool':
            items.append({'type': 'function_call_output', 'call_id': message['id'],
                          'output': _as_text(message.get('content'))})
        elif role == 'assistant':
            if message.get('content'):
                items.append({'role': 'assistant', 'content': message['content']})
            for call in message.get('tool_calls') or []:
                items.append({'type': 'function_call', 'call_id': call['id'],
                              'name': call['name'], 'arguments': json.dumps(call.get('arguments') or {})})
        elif role == 'user':
            content = []
            for block in as_blocks(message.get('content', '')):
                if block['type'] == 'text':
                    content.append({'type': 'input_text', 'text': block.get('text', '')})
                else:
                    content.append({'type': 'input_image', 'image_url':
                        f"data:{block.get('media_type', 'image/png')};base64,{block.get('data', '')}"})
            items.append({'role': 'user', 'content': content})
    return items


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
                blocks = [b for b in raw if not _is_empty_text(b)]
            else:
                blocks = []
                if m.get("content"):
                    blocks.append({"type": "text", "text": m["content"]})
                for c in (m.get("tool_calls") or []):
                    blocks.append({"type": "tool_use", "id": c["id"], "name": c["name"],
                                   "input": c.get("arguments") or {}})
            # The Messages API rejects an assistant turn with no content, or with
            # a text block that is empty, anywhere but last - and once one is in
            # the history every later call fails with a 400. Such a turn says
            # nothing and asks for no tool, so it is left out; the user turns on
            # either side then merge, as adjacent user turns always do.
            if blocks:
                out.append({"role": "assistant", "content": blocks})
    flush_results()
    return out


def _is_empty_text(block: Any) -> bool:
    """A text block holding only whitespace, as an SDK object or a plain dict."""
    if isinstance(block, dict):
        kind, text = block.get("type"), block.get("text")
    else:
        kind, text = getattr(block, "type", None), getattr(block, "text", None)
    return kind == "text" and not str(text or "").strip()


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
        self.reasoning_effort = 'medium'

    @property
    def model(self) -> str:
        return self._model

    def set_model(self, model: str) -> None:
        model = (model or "").strip()
        if model:
            self._model = model

    def set_api_key(self, api_key: str) -> None:
        # Under the lock, so a client a warm-up is building with the old key
        # cannot be stored after this has discarded it.
        with _CLIENT_LOCK:
            self._api_key = (api_key or "").strip() or None
            self._client = None

    def set_base_url(self, base_url: str) -> None:
        with _CLIENT_LOCK:
            self._base_url = (base_url or "").strip() or None
            self._client = None

    def available(self) -> Tuple[bool, str]:  # pragma: no cover - overridden
        raise NotImplementedError

    def supports_vision(self) -> bool:
        """Whether the configured model accepts image blocks in user messages."""
        return supports_vision(self.id, self._model)

    def complete(self, system: str, messages, specs, max_tokens: int = 0) -> Dict[str, Any]:  # pragma: no cover
        raise NotImplementedError

    def _warm(self) -> None:
        """Do a first request's one-off work ahead of it: imports, the client.

        Run by :func:`prewarm` on a background thread. May raise; the caller
        treats any failure as "nothing to warm".
        """


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

        with _CLIENT_LOCK:
            if self._client is None:
                kwargs: Dict[str, Any] = {"api_key": self._api_key,
                                          "timeout": REQUEST_TIMEOUT_S}
                if self._base_url:
                    kwargs["base_url"] = self._base_url
                http_client = keepalive_http_client(openai)
                if http_client is not None:
                    kwargs["http_client"] = http_client
                self._client = openai.OpenAI(**kwargs)
            return self._client

    def _warm(self) -> None:
        # The package and the chat resource: the SDK imports that on first
        # attribute access, which cost a first request about 0.3 s on its own.
        # Not openai.resources.responses: building its types holds the GIL for
        # about 100 ms at a stretch, which froze the window while the user
        # typed. The first request still imports it, as it always did.
        importlib.import_module("openai")
        importlib.import_module("openai.resources.chat")
        if self.available()[0]:
            self._ensure_client()

    def complete(self, system, messages, specs, max_tokens: int = 1024) -> Dict[str, Any]:
        from .runtime_options import openai_options
        client = self._ensure_client()
        if (self.id == 'openai' and self._model.startswith(('gpt-5', 'gpt-6', 'o1', 'o3', 'o4'))
                and (specs or any(m.get('_openai_output') for m in messages))):
            return self._complete_responses(client, system, messages, specs, max_tokens)
        response = client.chat.completions.create(
            model=self._model,
            messages=to_openai_messages(system, messages),
            **({'tools': to_openai_tools(specs), 'tool_choice': 'auto'} if specs else {}),
            **openai_options(self._model, effort=self.reasoning_effort),
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

    def _complete_responses(self, client, system, messages, specs, max_tokens):
        response = client.responses.create(
            model=self._model, instructions=system, input=to_responses_input(messages),
            tools=[{'type': 'function', **tool['function'], 'strict': False}
                   for tool in to_openai_tools(specs)],
            reasoning={'effort': self.reasoning_effort or 'medium'},
            max_output_tokens=max(16384, max_tokens), store=False,
            include=['reasoning.encrypted_content'],
        )
        if response.status != 'completed':
            raise RuntimeError(f'Model response did not complete: {response.status}. Retry or reduce reasoning effort.')
        out = {'content': response.output_text, 'tool_calls': [],
               '_openai_output': [item.model_dump(exclude_none=True) for item in response.output]}
        for item in response.output:
            if item.type == 'function_call':
                args = json.loads(item.arguments or '{}')
                if not isinstance(args, dict):
                    raise ValueError('Model returned invalid tool arguments; no action was executed.')
                out['tool_calls'].append({'id': item.call_id, 'name': item.name, 'arguments': args})
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

        with _CLIENT_LOCK:
            if self._client is None:
                kwargs: Dict[str, Any] = {"api_key": self._api_key,
                                          "timeout": REQUEST_TIMEOUT_S}
                http_client = keepalive_http_client(anthropic)
                if http_client is not None:
                    kwargs["http_client"] = http_client
                self._client = anthropic.Anthropic(**kwargs)
            return self._client

    def _warm(self) -> None:
        importlib.import_module("anthropic")
        importlib.import_module("anthropic.resources.messages")
        if self.available()[0]:
            self._ensure_client()

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
        # Ordered by the cost ladder in MODEL_TIERS: luna / terra / sol are the
        # level 1 / 2 / 3 rungs. The gpt-4.x names stay selectable for keys and
        # endpoints that have not been moved to the 5.6 family yet.
        "models": ["gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol",
                   "gpt-4.1", "gpt-4.1-mini", "gpt-4o-mini"],
        "default_model": "gpt-5.6-luna",
        "needs_base_url": False,
        "vision": True,
    },
    "anthropic": {
        "label": "Claude (Anthropic)",
        "env_key": "ANTHROPIC_API_KEY",
        "model_env": "ANTHROPIC_MODEL",
        # Ordered by the cost ladder in MODEL_TIERS: haiku / sonnet / opus are the
        # level 1 / 2 / 3 rungs. The 4.x names remain selectable for pinned work.
        "models": ["claude-haiku-4-5", "claude-sonnet-5", "claude-opus-5",
                   "claude-sonnet-4-6", "claude-opus-4-8"],
        "default_model": "claude-haiku-4-5",
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

    Used to gate the studio ``capture_view`` tool: offering a screenshot tool
    to a text-only model wastes a turn and returns an API error.
    """
    meta = PROVIDER_META.get(provider_id) or {}
    if not meta.get("vision", False):
        return False
    name = (model or "").lower()
    return not any(hint in name for hint in TEXT_ONLY_MODEL_HINTS)

# -- model tiers (the cost ladder) --------------------------------------------
#: Approximate list price in USD per million input / output tokens, by model id.
#: Rates move, and an agent turn re-sends its transcript, so read these as the
#: relative cost of one level against the next rather than as a bill.
MODEL_PRICES_USD_PER_MTOK: Dict[str, Tuple[float, float]] = {
    "gpt-5.6-luna": (0.20, 1.20),
    "gpt-5.6-terra": (2.00, 12.00),
    "gpt-5.6-sol": (5.00, 30.00),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4o": (5.00, 15.00),
    "gpt-4o-mini": (0.15, 0.60),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-opus-4-8": (5.00, 25.00),
    "deepseek-chat": (0.28, 0.42),
}

#: Tier ids in ladder order, cheapest first.
TIER_ORDER = ("level1", "level2", "level3")

#: Selected when the model does not belong to any tier - a hand-typed name, an
#: OpenAI-compatible endpoint, or a model pinned from an earlier session.
TIER_CUSTOM = "custom"

#: The three-step routing ladder offered in the chat settings and the Streamlit
#: sidebar. Sending every request to a flagship model is the expensive default;
#: starting at level 1 and escalating only when a level cannot finish the job
#: costs far less over a session, because most turns in a studio conversation
#: are short and mechanical. Each tier names one model per provider so the same
#: choice survives a provider switch.
MODEL_TIERS: Dict[str, Dict[str, Any]] = {
    "level1": {
        "name": "Level 1",
        "headline": "Default requests",
        "purpose": "Most simple tasks: reading a file, setting a parameter, a short answer.",
        "models": {"openai": "gpt-5.6-luna", "anthropic": "claude-haiku-4-5"},
        "effort": "low",
    },
    "level2": {
        "name": "Level 2",
        "headline": "Complex requests",
        "purpose": "Coding, reasoning, agent loops, and complex retrieval.",
        "models": {"openai": "gpt-5.6-terra", "anthropic": "claude-sonnet-5"},
        "effort": "medium",
    },
    "level3": {
        "name": "Level 3",
        "headline": "Genuinely hard requests",
        "purpose": "Escalate here only when a lower level could not solve it.",
        "models": {"openai": "gpt-5.6-sol", "anthropic": "claude-opus-5"},
        "effort": "high",
    },
}

#: Reasoning effort when the level does not name one, and for a custom model.
DEFAULT_EFFORT = "medium"


def tier_effort(tier_id: Optional[str]) -> str:
    """How hard the model should think, for a level of the ladder.

    Effort is the other half of the dial the ladder turns. Every level-1 model
    is a reasoning model, so leaving effort at ``medium`` there made the cheap
    level slow: chat drives the studio one tool call at a time, and each call is
    a separate request that pays a full reasoning pass before it can answer
    "navigate to the ERT module". Level 1 exists for exactly the requests that
    do not need that.

    Parameters
    ----------
    tier_id : str or None
        A key of :data:`MODEL_TIERS`, or None / :data:`TIER_CUSTOM` for a model
        that is not on the ladder.

    Returns
    -------
    str
        An effort accepted by the reasoning models: ``'low'``, ``'medium'`` or
        ``'high'``. A custom model gets :data:`DEFAULT_EFFORT`, since nothing is
        known about what it is for.

    Raises
    ------
    None

    Examples
    --------
    >>> tier_effort('level1'), tier_effort('level2'), tier_effort('level3')
    ('low', 'medium', 'high')
    >>> tier_effort(None)
    'medium'
    >>> tier_effort(TIER_CUSTOM)
    'medium'
    """
    tier = MODEL_TIERS.get(tier_id or "")
    return str((tier or {}).get("effort") or DEFAULT_EFFORT)

#: Provider names used by callers that predate this module - the Streamlit
#: sidebar and the agents engine both say "claude" where the adapters say
#: "anthropic".
PROVIDER_ALIASES = {"claude": "anthropic", "anthropic": "anthropic",
                    "openai": "openai", "gpt": "openai"}


def normalise_provider_id(provider_id: Optional[str]) -> str:
    """Map a caller's provider name onto an adapter id (``claude`` -> ``anthropic``)."""
    key = (provider_id or "").strip().lower()
    return PROVIDER_ALIASES.get(key, key)


def tier_model(tier_id: str, provider_id: str) -> Optional[str]:
    """The model a ``tier_id`` selects on ``provider_id``, or None if untiered.

    Returns None for a provider the ladder does not cover (an OpenAI-compatible
    endpoint, say), which is the caller's cue to fall back to manual entry.
    """
    tier = MODEL_TIERS.get(tier_id)
    if not tier:
        return None
    return tier["models"].get(normalise_provider_id(provider_id))


def tier_for_model(provider_id: str, model: Optional[str]) -> str:
    """Which tier ``model`` belongs to on ``provider_id``, else :data:`TIER_CUSTOM`.

    The reverse lookup keeps the level selector honest when the model is set by
    an environment variable, a saved session, or hand-typed into the model box.
    """
    name = (model or "").strip()
    for tier_id in TIER_ORDER:
        if tier_model(tier_id, provider_id) == name:
            return tier_id
    return TIER_CUSTOM


def tier_of_model(model: Optional[str]) -> str:
    """Which tier ``model`` sits on for *any* provider, else :data:`TIER_CUSTOM`.

    Lets a caller keep the level a user chose when they switch provider: the
    saved model belongs to the old provider's ladder, and the level it names is
    what should carry over.
    """
    for tier_id in TIER_ORDER:
        if (model or "").strip() in MODEL_TIERS[tier_id]["models"].values():
            return tier_id
    return TIER_CUSTOM


def price_label(model: Optional[str], compact: bool = False) -> str:
    """``"$0.20 / $1.20 per Mtok"`` for a known model, else an empty string.

    ``compact`` drops the unit, for a status line that has to stay on one row;
    the ladder itself spells the unit out, so it is stated somewhere.
    """
    rate = MODEL_PRICES_USD_PER_MTOK.get((model or "").strip())
    if rate is None:
        return ""
    unit = "" if compact else " per Mtok"
    return f"${rate[0]:.2f} / ${rate[1]:.2f}{unit}"


def tier_label(tier_id: str, provider_id: str, with_model: bool = True) -> str:
    """One line naming a tier, and on a covered provider its model and price.

    Example: ``Level 2 - Complex requests (gpt-5.6-terra, $2.00 / $12.00 per Mtok)``.
    """
    tier = MODEL_TIERS.get(tier_id)
    if not tier:
        return "Custom model"
    label = f"{tier['name']} - {tier['headline']}"
    model = tier_model(tier_id, provider_id)
    if not (with_model and model):
        return label
    price = price_label(model)
    return f"{label} ({model}, {price})" if price else f"{label} ({model})"


def provider_has_tiers(provider_id: str) -> bool:
    """Whether the ladder names a model for every tier on ``provider_id``."""
    return all(tier_model(t, provider_id) for t in TIER_ORDER)


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


def prewarm(provider: Any) -> Optional[threading.Thread]:
    """Start a first request's one-off work on a background thread.

    The first request of a session paid about 1.5 s before anything reached the
    network: importing the SDK, then the resource module behind the endpoint,
    then building the client. None of that depends on what the user is about to
    ask, so the chat starts it as soon as they begin typing and the request
    finds it done.

    Parameters
    ----------
    provider : Provider
        The provider to warm. Anything without a ``_warm`` method is ignored.

    Returns
    -------
    threading.Thread or None
        The daemon thread doing the work, or None when there is nothing to do.
        Nothing waits for it: a request that starts first simply does the same
        work itself, and the lock in ``_ensure_client`` keeps the two from each
        building a client.

    Raises
    ------
    None
        Failures are swallowed: a missing SDK, a missing key or a broken
        install is reported by the request that needs it, not by a warm-up
        nobody asked for.

    Examples
    --------
    >>> prewarm(object()) is None
    True
    """
    warm = getattr(provider, "_warm", None)
    if not callable(warm):
        return None

    def run() -> None:
        try:
            warm()
        except Exception:  # noqa: BLE001 - a warm-up must never surface an error
            pass

    thread = threading.Thread(target=run, name="phgx-llm-prewarm", daemon=True)
    thread.start()
    return thread
