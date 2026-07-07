"""Tests for the Qt-free LLM provider layer (message translation, windowing)."""

from PyHydroGeophysX.llm.providers import (
    PROVIDER_META,
    to_anthropic_messages,
    to_openai_messages,
    window_messages,
)


def test_window_keeps_short_conversations():
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello", "tool_calls": []},
    ]
    assert window_messages(msgs, max_items=10) == msgs


def test_window_cuts_on_user_boundary_and_keeps_pairs():
    msgs = []
    for i in range(30):
        msgs.append({"role": "user", "content": f"u{i}"})
        msgs.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": f"c{i}", "name": "t", "arguments": {}}],
            }
        )
        msgs.append({"role": "tool", "id": f"c{i}", "name": "t", "content": "ok"})
        msgs.append({"role": "assistant", "content": f"a{i}", "tool_calls": []})
    out = window_messages(msgs, max_items=10)
    assert 0 < len(out) <= 10
    assert out[0]["role"] == "user"
    n_tool_results = sum(1 for m in out if m["role"] == "tool")
    n_tool_calls = sum(
        len(m.get("tool_calls") or []) for m in out if m["role"] == "assistant"
    )
    assert n_tool_results == n_tool_calls


def test_anthropic_translation_pairs_tool_results():
    msgs = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "c1", "name": "t", "arguments": {"a": 1}}],
        },
        {"role": "tool", "id": "c1", "name": "t", "content": {"ok": True}},
    ]
    out = to_anthropic_messages(msgs)
    assert [m["role"] for m in out] == ["user", "assistant", "user"]
    result_block = out[2]["content"][0]
    assert result_block["type"] == "tool_result"
    assert result_block["tool_use_id"] == "c1"


def test_openai_translation_serializes_arguments():
    msgs = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "c1", "name": "t", "arguments": {"a": 1}}],
        },
        {"role": "tool", "id": "c1", "name": "t", "content": "done"},
    ]
    out = to_openai_messages("sys", msgs)
    assert out[0] == {"role": "system", "content": "sys"}
    call = out[2]["tool_calls"][0]
    assert call["function"]["arguments"] == '{"a": 1}'
    assert out[3]["tool_call_id"] == "c1"


def test_qt_shim_aliases_the_canonical_module():
    import PyHydroGeophysX.llm.providers as canonical
    import PyHydroGeophysX.qt_apps.agent.providers as shim

    assert shim is canonical


def test_default_models_are_current():
    assert PROVIDER_META["anthropic"]["default_model"] == "claude-sonnet-5"
    assert "claude-sonnet-5" in PROVIDER_META["anthropic"]["models"]
