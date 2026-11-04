from __future__ import annotations

from datetime import timedelta
from types import SimpleNamespace

import pytest

from token_eval.storage import CallRecord, TokenStore, parse_since, utcnow
from token_eval.tracker import TokenTracker, extract_usage


def fake_anthropic_create(**kwargs):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text="hi")],
        usage=SimpleNamespace(input_tokens=1000, output_tokens=200,
                              cache_creation_input_tokens=0, cache_read_input_tokens=500),
    )


def fake_openai_create(**kwargs):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="hello"))],
        usage=SimpleNamespace(prompt_tokens=300, completion_tokens=50,
                              prompt_tokens_details=SimpleNamespace(cached_tokens=100)),
    )


def test_extract_usage_shapes():
    a = extract_usage(fake_anthropic_create())
    assert (a.prompt_tokens, a.completion_tokens, a.cache_read_tokens) == (1000, 200, 500)
    o = extract_usage(fake_openai_create())
    assert (o.prompt_tokens, o.completion_tokens, o.cache_read_tokens) == (200, 50, 100)
    d = extract_usage({"usage": {"input_tokens": 5, "output_tokens": 6}})
    assert (d.prompt_tokens, d.completion_tokens) == (5, 6)
    assert extract_usage("plain string") is None


def test_call_records_cost_and_persists(db_path):
    store = TokenStore(db_path)
    tracker = TokenTracker(store, task_id="t1")
    res = tracker.call(fake_anthropic_create, model="claude-opus-5-5",
                       messages=[{"role": "user", "content": "hello"}], tag="search")
    rec = res.record
    # $4/MTok input, $20/MTok output, $0.20/MTok cache read
    assert rec.cost_usd == pytest.approx((1000 * 4 + 200 * 20 + 500 * 0.2) / 1e6)
    assert rec.tag == "search" and rec.task_id == "t1"
    assert "hello" in rec.prompt_preview
    rows = store.calls()
    assert len(rows) == 1 and rows[0]["model"] == "claude-opus-5-5"
    assert store.usage() == (1700, pytest.approx(rec.cost_usd))


def test_call_without_usage_counts_text(db_path):
    tracker = TokenTracker(TokenStore(db_path))
    res = tracker.call(lambda **kw: "a reply of several words", model="gpt-4o",
                       prompt="what is the answer to this question?")
    assert res.record.prompt_tokens > 0 and res.record.completion_tokens > 0


def test_model_forwarded_only_when_accepted(db_path):
    tracker = TokenTracker(TokenStore(db_path))
    seen = {}

    def create(*, model, messages):
        seen["model"] = model
        return fake_anthropic_create()

    tracker.call(create, model="claude-haiku-5-5", messages=[])
    assert seen["model"] == "claude-haiku-5-5"
    tracker.call(lambda: "no params", model="gpt-4o", prompt="x")  # must not raise


def test_decorator_and_model_required(db_path):
    tracker = TokenTracker(TokenStore(db_path))

    @tracker.track(model="gpt-4o", tag="chat")
    def ask(question: str):
        return fake_openai_create()

    assert ask("q").record.tag == "chat"
    with pytest.raises(ValueError):
        tracker.call(fake_openai_create)


def test_report_and_since(db_path):
    store = TokenStore(db_path)
    old = CallRecord("gpt-4o", 10, 10, 1.0, tag="old", ts=utcnow() - timedelta(days=10))
    store.add_call(old)
    store.add_call(CallRecord("gpt-4o", 10, 10, 2.0, tag="a"))
    store.add_call(CallRecord("gpt-4o", 10, 10, 3.0, tag="a"))
    store.add_call(CallRecord("gpt-4o", 10, 10, 0.5, tag="b"))
    rows = store.report(by="tag", since=parse_since("7d"))
    assert [r["key"] for r in rows] == ["a", "b"]
    assert rows[0]["cost_usd"] == pytest.approx(5.0) and rows[0]["calls"] == 2
    assert len(store.report(by="day")) == 2
    with pytest.raises(ValueError):
        store.report(by="nope")


def test_parse_since():
    assert parse_since(None) is None
    assert utcnow() - parse_since("2h") == pytest.approx(timedelta(hours=2), abs=timedelta(seconds=5))
    assert parse_since("2026-01-01").year == 2026
