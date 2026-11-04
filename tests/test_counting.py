from __future__ import annotations

from types import SimpleNamespace

import pytest

from token_eval import counting
from token_eval.counting import count_tokens, count_tokens_detailed, truncate_to_tokens
from token_eval.pricing import PricingTable


def test_openai_uses_tiktoken():
    tc = count_tokens_detailed("hello world", "gpt-4o")
    assert tc.method in ("tiktoken", "heuristic")
    assert tc.tokens >= 2


def test_empty_text_is_zero():
    assert count_tokens("", "gpt-4o") == 0
    assert count_tokens("", "claude-opus-5-5") == 0


def test_longer_text_has_more_tokens():
    assert count_tokens("word " * 100, "gpt-4o") > count_tokens("word " * 10, "gpt-4o")


def test_anthropic_without_key_falls_back():
    tc = count_tokens_detailed("hello world", "claude-opus-5-5")
    assert tc.method in ("tiktoken", "heuristic")


class FakeMessages:
    def __init__(self) -> None:
        self.calls = 0

    def count_tokens(self, *, model: str, messages: list[dict]):
        self.calls += 1
        # 7 tokens of fixed overhead + 1 token per word.
        return SimpleNamespace(input_tokens=7 + len(messages[0]["content"].split()))


def test_anthropic_api_used_when_available(monkeypatch: pytest.MonkeyPatch):
    fake = SimpleNamespace(messages=FakeMessages())
    monkeypatch.setattr(counting, "_anthropic_client", lambda: fake)
    tc = count_tokens_detailed("one two three four", "claude-opus-5-5")
    assert tc == counting.TokenCount(4, "anthropic_api")
    # Second call with same text is cached.
    calls = fake.messages.calls
    count_tokens("one two three four", "claude-opus-5-5")
    assert fake.messages.calls == calls


def test_anthropic_api_error_falls_back(monkeypatch: pytest.MonkeyPatch):
    class Boom:
        def count_tokens(self, **_):
            raise RuntimeError("network down")

    monkeypatch.setattr(counting, "_anthropic_client", lambda: SimpleNamespace(messages=Boom()))
    assert count_tokens_detailed("hi there", "claude-opus-5-5").method != "anthropic_api"


def test_truncate():
    text = "token " * 500
    out = truncate_to_tokens(text, 50, "gpt-4o")
    assert count_tokens(out, "gpt-4o") <= 50
    assert truncate_to_tokens("short", 50) == "short"


def test_pricing_cost_and_prefix_match():
    table = PricingTable({
        "default": {"provider": "x", "tier": "medium", "input": 1, "output": 2,
                    "cache_write": 1, "cache_read": 0.1},
        "models": {
            "big": {"provider": "p", "tier": "large", "input": 10, "output": 50,
                    "cache_write": 12.5, "cache_read": 1},
            "small": {"provider": "p", "tier": "small", "input": 1, "output": 5,
                      "cache_write": 1.25, "cache_read": 0.1},
        },
    })
    assert table.cost("big", 1_000_000, 1_000_000) == pytest.approx(60.0)
    assert table.get("big-2026-01-01").input == 10  # prefix match
    assert table.get("unknown").input == 1  # default
    assert table.cheaper_alternative("big") == "small"
    assert table.cheaper_alternative("small") is None


def test_packaged_pricing_loads():
    table = PricingTable.load(counting.__file__.replace("counting.py", "pricing.json"))
    assert "claude-opus-5-5" in table.models
