"""optimize(): apply safe, mechanical token reductions to a prompt."""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Any, Callable

from token_eval.analyzer import (DEFAULT_MODEL, CacheHint, PromptAnalyzer, block_key,
                                 normalize_block, split_blocks)
from token_eval.counting import count_tokens, truncate_to_tokens
from token_eval.pricing import PricingTable, default_pricing
from token_eval.prompt import Chunk, Message, Prompt
from token_eval.storage import TokenStore

Summarizer = Callable[[list[Message]], str]
SUMMARY_HEADER = "[Summary of earlier conversation]"
TRUNCATION_MARK = " ...[truncated]"
OMITTED = "[repeated content omitted; see above]"

_FENCE = re.compile(r"(```.*?```)", re.DOTALL)
_HR = re.compile(r"^[ \t]*([-*_])([ \t]*\1){2,}[ \t]*$", re.MULTILINE)
_BOLD = re.compile(r"(\*\*|__)(?=\S)(.+?)(?<=\S)\1")
_INNER_SPACES = re.compile(r"(?<=\S)[ \t]{2,}")
_TRAILING = re.compile(r"[ \t]+$", re.MULTILINE)
_BLANKS = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """Whitespace + markdown cleanup that leaves fenced code blocks untouched."""
    parts = _FENCE.split(text)
    for i in range(0, len(parts), 2):  # even indices are outside code fences
        p = parts[i]
        p = _HR.sub("", p)
        p = _BOLD.sub(r"\2", p)
        p = _INNER_SPACES.sub(" ", p)
        p = _TRAILING.sub("", p)
        p = _BLANKS.sub("\n\n", p)
        parts[i] = p
    return "".join(parts).strip()


def extractive_summarizer(messages: list[Message], max_chars: int = 160) -> str:
    """Free, offline summarizer: first sentence of every message."""
    lines = []
    for m in messages:
        text = normalize_block(m.content)
        first = re.split(r"(?<=[.!?])\s", text, maxsplit=1)[0]
        if len(first) > max_chars:
            first = first[: max_chars - 1] + "..."
        lines.append(f"- {m.role}: {first}")
    return "\n".join(lines)


@dataclass
class StepResult:
    name: str
    tokens_saved: int
    detail: str


@dataclass
class OptimizationResult:
    prompt: Prompt
    model: str
    tokens_before: int
    tokens_after: int
    saved_usd: float
    steps: list[StepResult] = field(default_factory=list)
    suggested_model: str | None = None
    suggested_model_saved_usd: float = 0.0
    cache: CacheHint | None = None

    @property
    def saved_tokens(self) -> int:
        return self.tokens_before - self.tokens_after

    @property
    def saved_pct(self) -> float:
        return self.saved_tokens / self.tokens_before if self.tokens_before else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "tokens_before": self.tokens_before,
            "tokens_after": self.tokens_after,
            "saved_tokens": self.saved_tokens,
            "saved_pct": round(self.saved_pct, 4),
            "saved_usd_per_call": self.saved_usd,
            "steps": [vars(s) for s in self.steps],
            "suggested_model": self.suggested_model,
            "suggested_model_saved_usd_per_call": self.suggested_model_saved_usd,
            "cache": vars(self.cache) if self.cache else None,
            "prompt": self.prompt.to_dict(),
        }


def _total(prompt: Prompt, model: str) -> int:
    return sum(count_tokens(t, model) for _, _, t in prompt.sections())


def _cleanup(p: Prompt) -> tuple[Prompt, str]:
    return replace(
        p,
        system=clean_text(p.system),
        messages=[Message(m.role, clean_text(m.content)) for m in p.messages],
        chunks=[Chunk(c.id, clean_text(c.text)) for c in p.chunks],
    ), "whitespace and markdown normalized"


def _dedupe(p: Prompt, min_chars: int) -> tuple[Prompt, str]:
    seen: set[str] = set()
    removed = 0

    def strip(text: str) -> str:
        nonlocal removed
        kept = []
        for block in split_blocks(text):
            if len(normalize_block(block)) >= min_chars:
                k = block_key(block)
                if k in seen:
                    removed += 1
                    continue
                seen.add(k)
            kept.append(block.strip("\n"))
        return "\n\n".join(kept)

    system = strip(p.system)
    chunks = [Chunk(c.id, t) for c in p.chunks if (t := strip(c.text))]
    messages = [Message(m.role, strip(m.content) or OMITTED) for m in p.messages]
    dropped = len(p.chunks) - len(chunks)
    detail = f"removed {removed} duplicate block(s)" + (f", dropped {dropped} chunk(s)" if dropped else "")
    return replace(p, system=system, chunks=chunks, messages=messages), detail


def _summarize(p: Prompt, keep_last_n: int, summarizer: Summarizer, model: str
               ) -> tuple[Prompt, str]:
    older, kept = p.split_old_turns(keep_last_n)
    if not older:
        return p, "nothing to summarize"
    summary = f"{SUMMARY_HEADER}\n{summarizer(older)}"
    if count_tokens(summary, model) >= sum(count_tokens(m.content, model) for m in older):
        return p, "summary would not be shorter; skipped"
    # split_old_turns cuts at a user message, so kept[0] is a user turn.
    first = kept[0]
    kept = [Message(first.role, f"{summary}\n\n{first.content}")] + kept[1:]
    return replace(p, messages=kept), f"summarized {len(older)} message(s) older than {keep_last_n} turns"


def _truncate_chunks(p: Prompt, max_tokens: int, model: str) -> tuple[Prompt, str]:
    n = 0
    chunks = []
    for c in p.chunks:
        if count_tokens(c.text, model) > max_tokens:
            n += 1
            chunks.append(Chunk(c.id, truncate_to_tokens(c.text, max_tokens, model).rstrip()
                                + TRUNCATION_MARK))
        else:
            chunks.append(c)
    return replace(p, chunks=chunks), f"truncated {n} chunk(s) to {max_tokens} tokens"


def is_simple(p: Prompt, tokens: int, threshold: int) -> bool:
    texts = [p.system, *(m.content for m in p.messages), *(c.text for c in p.chunks)]
    return (tokens <= threshold and not p.tools and len(p.messages) <= 4
            and not any("```" in t for t in texts))


def optimize(
    prompt: Prompt | dict[str, Any],
    model: str | None = None,
    *,
    keep_last_n: int = 6,
    max_chunk_tokens: int = 800,
    summarizer: Summarizer | None = extractive_summarizer,
    cleanup: bool = True,
    dedupe: bool = True,
    min_duplicate_chars: int = 40,
    simple_threshold: int = 1500,
    pricing: PricingTable | None = None,
    store: TokenStore | None = None,
    tag: str | None = None,
) -> OptimizationResult:
    """Apply safe reductions and report before/after tokens and estimated savings.

    ``summarizer`` receives the messages older than ``keep_last_n`` turns and
    returns a summary string. The default is a free extractive summarizer;
    pass ``None`` to skip summarization.
    """
    if isinstance(prompt, dict):
        prompt = Prompt.from_dict(prompt)
    pricing = pricing or default_pricing()
    model = model or prompt.model or DEFAULT_MODEL

    before = current = _total(prompt, model)
    steps: list[StepResult] = []
    p = prompt

    def run(name: str, fn: Callable[[Prompt], tuple[Prompt, str]]) -> None:
        nonlocal p, current
        p, detail = fn(p)
        after = _total(p, model)
        steps.append(StepResult(name, current - after, detail))
        current = after

    if cleanup:
        run("cleanup", _cleanup)
    if dedupe:
        run("dedupe", lambda q: _dedupe(q, min_duplicate_chars))
    if summarizer is not None and keep_last_n > 0:
        run("summarize_history", lambda q: _summarize(q, keep_last_n, summarizer, model))
    if max_chunk_tokens:
        run("truncate_chunks", lambda q: _truncate_chunks(q, max_chunk_tokens, model))

    price = pricing.get(model)
    saved_usd = (before - current) * price.input / 1_000_000

    suggested, suggested_saving = None, 0.0
    if is_simple(p, current, simple_threshold):
        suggested = pricing.cheaper_alternative(model)
        if suggested:
            suggested_saving = current * (price.input - pricing.get(suggested).input) / 1_000_000

    if store is not None:
        store.add_saving(model, before, current, saved_usd, tag)

    return OptimizationResult(
        prompt=p,
        model=model,
        tokens_before=before,
        tokens_after=current,
        saved_usd=saved_usd,
        steps=steps,
        suggested_model=suggested,
        suggested_model_saved_usd=suggested_saving,
        cache=PromptAnalyzer(model=model, pricing=pricing).analyze(p).cache,
    )
