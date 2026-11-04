from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from token_eval.optimizer import (OMITTED, SUMMARY_HEADER, TRUNCATION_MARK, clean_text,
                                  extractive_summarizer, optimize)
from token_eval.prompt import Message, Prompt
from token_eval.storage import TokenStore

EXAMPLE = Path(__file__).parents[1] / "examples" / "prompt.json"
REPEATED = ("This paragraph is long enough to count as a real block of repeated content "
            "that wastes tokens every time it is sent.")


def conv(turns: int) -> list[dict]:
    msgs = []
    for i in range(turns):
        msgs += [{"role": "user", "content": f"Question number {i}. With more detail here."},
                 {"role": "assistant", "content": f"Answer number {i}. " + "Elaboration. " * 20}]
    return msgs


def test_clean_text_preserves_code():
    text = "# Title\n\n**Bold**   words   here.   \n\n\n\n---\n\n```py\nx  =  1\n\n\n\ny = 2\n```"
    out = clean_text(text)
    assert "Bold words here." in out
    assert "---" not in out and "\n\n\n" not in out.split("```")[0]
    assert "x  =  1\n\n\n\ny = 2" in out  # untouched inside fence


def test_dedupe_keeps_first_occurrence():
    p = {"system": f"Rules.\n\n{REPEATED}",
         "chunks": [REPEATED, "unique chunk text that is fairly long for the threshold ok"],
         "messages": [{"role": "user", "content": REPEATED}]}
    r = optimize(p, "gpt-4o", summarizer=None, max_chunk_tokens=0)
    assert REPEATED in r.prompt.system
    assert len(r.prompt.chunks) == 1  # identical chunk dropped
    assert r.prompt.messages[0].content == OMITTED  # alternation preserved
    assert next(s for s in r.steps if s.name == "dedupe").tokens_saved > 0


def test_summarize_history_with_mock_llm():
    fake_llm = MagicMock(return_value="User asked several questions; all were answered.")
    r = optimize({"messages": conv(10)}, "gpt-4o", keep_last_n=3, summarizer=fake_llm)
    fake_llm.assert_called_once()
    older = fake_llm.call_args.args[0]
    assert len(older) == 14 and all(isinstance(m, Message) for m in older)
    msgs = r.prompt.messages
    assert len(msgs) == 6 and msgs[0].role == "user"
    assert msgs[0].content.startswith(SUMMARY_HEADER)
    assert "Question number 7" in msgs[0].content
    assert r.saved_tokens > 0


def test_summary_skipped_if_not_shorter():
    long_summary = MagicMock(return_value="word " * 5000)
    r = optimize({"messages": conv(5)}, "gpt-4o", keep_last_n=2, summarizer=long_summary)
    assert len(r.prompt.messages) == 10
    assert "skipped" in next(s for s in r.steps if s.name == "summarize_history").detail


def test_extractive_summarizer():
    s = extractive_summarizer([Message("user", "First sentence. Second sentence."),
                               Message("assistant", "x" * 500)])
    assert "- user: First sentence." in s and "Second" not in s
    assert len(s.splitlines()[1]) < 200


def test_truncate_chunks():
    r = optimize({"chunks": [{"id": "big", "text": "data " * 3000}, "small"]}, "gpt-4o",
                 max_chunk_tokens=100)
    big = r.prompt.chunks[0]
    assert big.text.endswith(TRUNCATION_MARK)
    assert r.prompt.chunks[1].text == "small"
    assert r.tokens_after < 150


def test_savings_usd_and_store(db_path):
    store = TokenStore(db_path)
    r = optimize({"chunks": ["data " * 3000]}, "claude-opus-5-5", max_chunk_tokens=100,
                 store=store, tag="rag")
    assert r.saved_usd == pytest.approx(r.saved_tokens * 4 / 1e6)
    rows = store.savings()
    assert rows[0]["tokens_before"] == r.tokens_before and rows[0]["tag"] == "rag"


def test_cheaper_model_suggestion():
    simple = optimize({"messages": [{"role": "user", "content": "What is 2+2?"}]},
                      "claude-opus-5-5")
    assert simple.suggested_model == "claude-haiku-5-5"
    assert simple.suggested_model_saved_usd > 0
    with_tools = optimize({"tools": [{"name": "t"}],
                           "messages": [{"role": "user", "content": "hi"}]}, "claude-opus-5-5")
    assert with_tools.suggested_model is None


def test_example_prompt_end_to_end():
    r = optimize(Prompt.load(EXAMPLE), keep_last_n=4)
    assert r.tokens_after < r.tokens_before
    assert {s.name for s in r.steps} == {"cleanup", "dedupe", "summarize_history",
                                         "truncate_chunks"}
    assert all(s.tokens_saved >= 0 for s in r.steps)
    assert r.to_dict()["saved_tokens"] == r.saved_tokens
    assert r.cache is not None
