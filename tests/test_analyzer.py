from __future__ import annotations

from pathlib import Path

import pytest

from token_eval.analyzer import PromptAnalyzer
from token_eval.prompt import Prompt

EXAMPLE = Path(__file__).parents[1] / "examples" / "prompt.json"
REPEATED = ("This paragraph is long enough to count as a real block of repeated content "
            "that wastes tokens every time it is sent.")


def make_prompt(turns: int = 3, **kw) -> dict:
    msgs = []
    for i in range(turns):
        msgs += [{"role": "user", "content": f"question {i}"},
                 {"role": "assistant", "content": f"answer {i}"}]
    return {"system": kw.get("system", "Be helpful."), "messages": msgs,
            "chunks": kw.get("chunks", [])}


def test_sections_and_total():
    r = PromptAnalyzer(model="gpt-4o").analyze(make_prompt(2))
    assert [s.name for s in r.sections][:2] == ["system", "message[0]:user"]
    assert r.total_tokens == sum(s.tokens for s in r.sections)
    assert r.tokens_by_kind["message"] > 0
    assert r.estimated_input_cost_usd > 0


def test_duplicates_detected_across_sections():
    p = make_prompt(1, system=f"Rules.\n\n{REPEATED}", chunks=[REPEATED, f"x\n\n{REPEATED}"])
    r = PromptAnalyzer(model="gpt-4o").analyze(p)
    assert len(r.duplicates) == 1
    d = r.duplicates[0]
    assert d.occurrences == 3 and d.wasted_tokens > 0
    assert set(d.locations) == {"system", "chunk[0]", "chunk[1]"}


def test_short_blocks_not_flagged_as_duplicates():
    r = PromptAnalyzer(model="gpt-4o").analyze(make_prompt(1, chunks=["ok", "ok"]))
    assert r.duplicates == []


def test_long_system_and_old_turns_and_large_chunks():
    p = make_prompt(10, system="rule " * 400, chunks=[{"id": "big", "text": "data " * 2000}])
    r = PromptAnalyzer(model="gpt-4o", system_threshold=100, keep_last_n=4,
                       chunk_threshold=500).analyze(p)
    assert r.system_too_long
    assert r.old_turns == 6 and r.old_turn_messages == 12 and r.old_turn_tokens > 0
    assert [c.name for c in r.large_chunks] == ["chunk[big]"]
    assert any("summarize" in f for f in r.findings)


def test_cache_hint():
    small = PromptAnalyzer(model="claude-opus-5-5").analyze(make_prompt(1))
    assert not small.cache.cacheable and small.cache.savings_per_cached_call_usd == 0
    big = PromptAnalyzer(model="claude-opus-5-5").analyze(make_prompt(1, system="policy " * 3000))
    assert big.cache.cacheable
    assert big.cache.prefix_tokens == big.system_tokens
    # $4 input vs $0.20 cache read on Opus 5.5
    assert big.cache.savings_per_cached_call_usd == pytest.approx(
        big.cache.prefix_tokens * 3.8 / 1e6)


def test_example_prompt():
    prompt = Prompt.load(EXAMPLE)
    r = PromptAnalyzer(chunk_threshold=800).analyze(prompt)
    assert r.model == "claude-opus-5-5"
    assert r.duplicates and r.old_turns == 3 and r.large_chunks
    assert r.to_dict()["duplicate_tokens"] == r.duplicate_tokens


def test_block_content_lists_are_joined():
    p = Prompt.from_dict({"messages": [{"role": "user", "content": [
        {"type": "text", "text": "a"}, {"type": "image", "source": {}}, {"type": "text", "text": "b"}]}]})
    assert p.messages[0].content == "a\n\nb"
