"""Tests for the LLM cost-preview helpers."""

from PyHydroGeophysX.agents._pricing import (
    FALLBACK_RATE_USD_PER_MTOK,
    estimate_llm_cost_usd,
    estimate_tokens,
    get_rate_usd_per_mtok,
)


def test_current_generation_rates_present():
    assert get_rate_usd_per_mtok("claude", "claude-sonnet-5") == (3.00, 15.00)
    assert get_rate_usd_per_mtok("anthropic", "claude-haiku-4-5") == (1.00, 5.00)
    assert get_rate_usd_per_mtok("openai", "gpt-4.1") == (2.00, 8.00)
    assert get_rate_usd_per_mtok("gemini", "gemini-2.5-flash") == (0.30, 2.50)


def test_lookup_is_case_insensitive():
    assert get_rate_usd_per_mtok("Claude", "CLAUDE-SONNET-5") == (3.00, 15.00)


def test_unknown_model_falls_back():
    assert get_rate_usd_per_mtok("openai", "totally-new-model") == FALLBACK_RATE_USD_PER_MTOK


def test_cost_estimate_scales_with_tokens():
    cost = estimate_llm_cost_usd("claude", "claude-sonnet-5", 1_000_000, 1_000_000)
    assert cost == 18.00
    assert estimate_llm_cost_usd("claude", "claude-sonnet-5", 0, 0) == 0.0


def test_estimate_tokens():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcdefgh") == 2
