"""Token counting for OpenAI (tiktoken) and Anthropic (count_tokens API) models.

Anthropic models use the free ``messages.count_tokens`` endpoint when an API key
is available. Otherwise we fall back to tiktoken, which is an *approximation*
for Claude (it typically undercounts by ~15-20%). If tiktoken cannot load an
encoding (e.g. offline on first use) we fall back to a chars/4 heuristic.

Set ``TOKEN_EVAL_OFFLINE=1`` to never call the network.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal

Method = Literal["anthropic_api", "tiktoken", "heuristic"]


@dataclass(frozen=True)
class TokenCount:
    tokens: int
    method: Method


def is_anthropic(model: str) -> bool:
    return model.startswith("claude") or model.startswith("anthropic.")


@lru_cache(maxsize=16)
def _encoding(model: str) -> Any | None:
    try:
        import tiktoken
    except ImportError:
        return None
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        pass
    except Exception:
        return None
    try:
        return tiktoken.get_encoding("o200k_base")
    except Exception:
        return None


def _local_count(text: str, model: str) -> TokenCount:
    enc = _encoding(model)
    if enc is not None:
        return TokenCount(len(enc.encode(text, disallowed_special=())), "tiktoken")
    return TokenCount(max(1, len(text) // 4) if text else 0, "heuristic")


@lru_cache(maxsize=1)
def _anthropic_client() -> Any | None:
    if os.environ.get("TOKEN_EVAL_OFFLINE") or not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic
    except ImportError:
        return None
    return anthropic.Anthropic()


@lru_cache(maxsize=4096)
def _anthropic_count(text: str, model: str) -> int | None:
    client = _anthropic_client()
    if client is None:
        return None
    try:
        resp = client.messages.count_tokens(
            model=model, messages=[{"role": "user", "content": text}]
        )
        return int(resp.input_tokens)
    except Exception:
        # Unknown model, network error, auth error... degrade to local estimate.
        return None


def count_tokens_detailed(text: str, model: str = "gpt-4o") -> TokenCount:
    """Count tokens and report which method produced the number."""
    if not text:
        return TokenCount(0, "tiktoken")
    if is_anthropic(model):
        n = _anthropic_count(text, model)
        if n is not None:
            # The API wraps text in a user turn; strip that fixed overhead.
            return TokenCount(max(0, n - _anthropic_overhead(model)), "anthropic_api")
    return _local_count(text, model)


@lru_cache(maxsize=16)
def _anthropic_overhead(model: str) -> int:
    n = _anthropic_count(".", model)
    return max(0, (n or 1) - 1)


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Number of tokens ``text`` uses for ``model``."""
    return count_tokens_detailed(text, model).tokens


def count_tokens_local(text: str, model: str = "gpt-4o") -> int:
    """Local-only count (tiktoken/heuristic); never calls an API. Good for bulk counting."""
    return _local_count(text, model).tokens if text else 0


def truncate_to_tokens(text: str, max_tokens: int, model: str = "gpt-4o") -> str:
    """Truncate ``text`` to roughly ``max_tokens`` using the local tokenizer (no API calls)."""
    enc = _encoding(model)
    if enc is not None:
        ids = enc.encode(text, disallowed_special=())
        return text if len(ids) <= max_tokens else enc.decode(ids[:max_tokens])
    return text[: max_tokens * 4]


def reset_caches() -> None:
    """Clear cached clients/counts (useful in tests or after setting an API key)."""
    for fn in (_anthropic_client, _anthropic_count, _anthropic_overhead):
        if hasattr(fn, "cache_clear"):  # may be monkeypatched in tests
            fn.cache_clear()
