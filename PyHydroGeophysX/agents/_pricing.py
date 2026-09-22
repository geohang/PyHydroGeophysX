"""Approximate LLM token pricing helpers for agent previews.

Rates change over time. The values below are rough defaults intended for
workflow previews, not billing reconciliation. Update them when provider
pricing changes or override them in downstream applications when exact cost
tracking matters.

Current-generation rates live in
:data:`PyHydroGeophysX.llm.providers.MODEL_PRICES_USD_PER_MTOK`, which is also
what the chat UIs quote next to each request level; this module consults that
table for any model its own provider-specific entries do not cover, so a model
added to the level ladder is priced here without a second edit.
"""

from typing import Dict, Optional, Tuple


# Values are approximate USD per 1 million input/output tokens.
DEFAULT_PROVIDER_RATES_USD_PER_MTOK: Dict[Tuple[str, str], Tuple[float, float]] = {
    ("openai", "gpt-4o-mini"): (0.15, 0.60),
    ("openai", "gpt-4o"): (5.00, 15.00),
    ("openai", "gpt-4"): (30.00, 60.00),
    ("openai", "gpt-4.1"): (2.00, 8.00),
    ("openai", "gpt-4.1-mini"): (0.40, 1.60),
    ("gemini", "gemini-pro"): (0.50, 1.50),
    ("gemini", "gemini-2.5-flash"): (0.30, 2.50),
    ("gemini", "gemini-2.5-pro"): (1.25, 10.00),
    ("claude", "claude-3-haiku-20240307"): (0.25, 1.25),
    ("claude", "claude-3-5-sonnet-20241022"): (3.00, 15.00),
    ("claude", "claude-3-opus-20240229"): (15.00, 75.00),
    # Current Anthropic generation (July 2026). "claude" and "anthropic" keys both
    # work because different entry points pass different provider labels.
    ("claude", "claude-sonnet-5"): (3.00, 15.00),
    ("claude", "claude-sonnet-4-6"): (3.00, 15.00),
    ("claude", "claude-opus-4-8"): (5.00, 25.00),
    ("claude", "claude-opus-4-7"): (5.00, 25.00),
    ("claude", "claude-haiku-4-5"): (1.00, 5.00),
    ("anthropic", "claude-sonnet-5"): (3.00, 15.00),
    ("anthropic", "claude-sonnet-4-6"): (3.00, 15.00),
    ("anthropic", "claude-opus-4-8"): (5.00, 25.00),
    ("anthropic", "claude-opus-4-7"): (5.00, 25.00),
    ("anthropic", "claude-haiku-4-5"): (1.00, 5.00),
    ("claude", "claude-opus-5"): (5.00, 25.00),
    ("anthropic", "claude-opus-5"): (5.00, 25.00),
}

FALLBACK_RATE_USD_PER_MTOK: Tuple[float, float] = (1.00, 3.00)


def _shared_rate(model: str) -> Optional[Tuple[float, float]]:
    """Rate for ``model`` from the shared provider table, or None.

    Imported lazily and defensively: this module is used by the agents engine in
    environments where the Qt/web chat layer may not be importable.
    """
    try:
        from PyHydroGeophysX.llm.providers import MODEL_PRICES_USD_PER_MTOK
    except Exception:  # noqa: BLE001
        return None
    return MODEL_PRICES_USD_PER_MTOK.get(model)


def estimate_tokens(text: str) -> int:
    """Estimate token count from text without provider-specific tokenizers.

    Parameters
    ----------
    text : str
        Text to estimate.

    Returns
    -------
    int
        Approximate token count using a conservative 4 characters per token.

    Raises
    ------
    None

    Examples
    --------
    >>> estimate_tokens("abcd")
    1
    """
    if not text:
        return 0
    return max(1, int(len(text) / 4))


def get_rate_usd_per_mtok(provider: str, model: str) -> Tuple[float, float]:
    """Return approximate input/output prices for a provider and model.

    Parameters
    ----------
    provider : str
        LLM provider name.
    model : str
        Model name.

    Returns
    -------
    tuple of float
        ``(input_rate, output_rate)`` in USD per 1 million tokens.

    Raises
    ------
    None

    Examples
    --------
    >>> get_rate_usd_per_mtok("openai", "gpt-4o-mini")[0]
    0.15
    """
    provider_key = (provider or "").lower()
    model_key = (model or "").lower()
    rate = DEFAULT_PROVIDER_RATES_USD_PER_MTOK.get((provider_key, model_key))
    if rate is None:
        rate = _shared_rate(model_key)
    return rate if rate is not None else FALLBACK_RATE_USD_PER_MTOK


def estimate_llm_cost_usd(
    provider: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> float:
    """Estimate LLM cost in USD.

    Parameters
    ----------
    provider : str
        LLM provider name.
    model : str
        Model name.
    prompt_tokens : int
        Prompt/input token count.
    completion_tokens : int
        Completion/output token count.

    Returns
    -------
    float
        Approximate USD cost.

    Raises
    ------
    None

    Examples
    --------
    >>> estimate_llm_cost_usd("openai", "gpt-4o-mini", 1000, 1000) > 0
    True
    """
    input_rate, output_rate = get_rate_usd_per_mtok(provider, model)
    return (prompt_tokens / 1_000_000.0) * input_rate + (
        completion_tokens / 1_000_000.0
    ) * output_rate
