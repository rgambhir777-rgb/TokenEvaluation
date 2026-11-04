"""PromptAnalyzer: where do the tokens go, and what could be saved?"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any

from token_eval.counting import count_tokens
from token_eval.pricing import PricingTable, default_pricing
from token_eval.prompt import Prompt

DEFAULT_MODEL = "claude-opus-5-5"
_PARA_SPLIT = re.compile(r"\n\s*\n")


def normalize_block(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def block_key(text: str) -> str:
    return hashlib.sha1(normalize_block(text).encode("utf-8")).hexdigest()


def split_blocks(text: str) -> list[str]:
    return [b for b in _PARA_SPLIT.split(text) if b.strip()]


@dataclass
class SectionTokens:
    name: str
    kind: str
    tokens: int


@dataclass
class Duplicate:
    preview: str
    occurrences: int
    locations: list[str]
    wasted_tokens: int


@dataclass
class CacheHint:
    prefix_tokens: int
    min_cacheable_tokens: int
    cacheable: bool
    share_of_prompt: float
    savings_per_cached_call_usd: float
    note: str


@dataclass
class AnalysisReport:
    model: str
    total_tokens: int
    estimated_input_cost_usd: float
    sections: list[SectionTokens]
    tokens_by_kind: dict[str, int]
    duplicates: list[Duplicate]
    system_tokens: int
    system_too_long: bool
    old_turns: int
    old_turn_messages: int
    old_turn_tokens: int
    large_chunks: list[SectionTokens]
    cache: CacheHint
    findings: list[str] = field(default_factory=list)

    @property
    def duplicate_tokens(self) -> int:
        return sum(d.wasted_tokens for d in self.duplicates)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["duplicate_tokens"] = self.duplicate_tokens
        return d


class PromptAnalyzer:
    def __init__(
        self,
        model: str | None = None,
        pricing: PricingTable | None = None,
        system_threshold: int = 1500,
        keep_last_n: int = 6,
        chunk_threshold: int = 800,
        min_duplicate_chars: int = 40,
    ) -> None:
        self.model = model
        self.pricing = pricing or default_pricing()
        self.system_threshold = system_threshold
        self.keep_last_n = keep_last_n
        self.chunk_threshold = chunk_threshold
        self.min_duplicate_chars = min_duplicate_chars

    def analyze(self, prompt: Prompt | dict[str, Any]) -> AnalysisReport:
        if isinstance(prompt, dict):
            prompt = Prompt.from_dict(prompt)
        model = self.model or prompt.model or DEFAULT_MODEL

        sections = [SectionTokens(n, k, count_tokens(t, model)) for n, k, t in prompt.sections()]
        total = sum(s.tokens for s in sections)
        by_kind: dict[str, int] = defaultdict(int)
        for s in sections:
            by_kind[s.kind] += s.tokens

        duplicates = self._duplicates(prompt, model)

        system_tokens = by_kind.get("system", 0)
        old, _ = prompt.split_old_turns(self.keep_last_n)
        old_tokens = sum(count_tokens(m.content, model) for m in old)
        old_turns = sum(1 for m in old if m.role == "user")

        large_chunks = [s for s in sections if s.kind == "chunk" and s.tokens > self.chunk_threshold]
        cache = self._cache_hint(by_kind.get("tools", 0) + system_tokens, total, model)

        report = AnalysisReport(
            model=model,
            total_tokens=total,
            estimated_input_cost_usd=self.pricing.cost(model, total),
            sections=sections,
            tokens_by_kind=dict(by_kind),
            duplicates=duplicates,
            system_tokens=system_tokens,
            system_too_long=system_tokens > self.system_threshold,
            old_turns=old_turns,
            old_turn_messages=len(old),
            old_turn_tokens=old_tokens,
            large_chunks=large_chunks,
            cache=cache,
        )
        report.findings = self._findings(report)
        return report

    def _duplicates(self, prompt: Prompt, model: str) -> list[Duplicate]:
        seen: dict[str, list[str]] = defaultdict(list)
        text_for: dict[str, str] = {}
        for name, kind, text in prompt.sections():
            if kind == "tools":
                continue
            for block in split_blocks(text):
                if len(normalize_block(block)) < self.min_duplicate_chars:
                    continue
                k = block_key(block)
                seen[k].append(name)
                text_for.setdefault(k, block)
        dups = []
        for k, locs in seen.items():
            if len(locs) > 1:
                block = text_for[k]
                dups.append(Duplicate(
                    preview=normalize_block(block)[:80],
                    occurrences=len(locs),
                    locations=locs,
                    wasted_tokens=count_tokens(block, model) * (len(locs) - 1),
                ))
        return sorted(dups, key=lambda d: d.wasted_tokens, reverse=True)

    def _cache_hint(self, prefix: int, total: int, model: str) -> CacheHint:
        price = self.pricing.get(model)
        cacheable = prefix >= price.min_cacheable_tokens
        savings = prefix * (price.input - price.cache_read) / 1_000_000 if cacheable else 0.0
        if cacheable:
            note = (f"Mark the end of tools+system with cache_control; each repeat call within "
                    f"the TTL saves ~${savings:.5f} (first call pays the write premium).")
        elif prefix:
            note = (f"Stable prefix ({prefix} tokens) is below the {price.min_cacheable_tokens}-token "
                    "minimum for caching on this model.")
        else:
            note = "No stable prefix (no system prompt or tools)."
        return CacheHint(prefix, price.min_cacheable_tokens, cacheable,
                         prefix / total if total else 0.0, savings, note)

    def _findings(self, r: AnalysisReport) -> list[str]:
        f = []
        if r.duplicates:
            f.append(f"{len(r.duplicates)} duplicated block(s) waste ~{r.duplicate_tokens} tokens.")
        if r.system_too_long:
            f.append(f"System prompt is {r.system_tokens} tokens (> {self.system_threshold}); "
                     "trim or move rarely-needed rules into tools/docs.")
        if r.old_turns:
            f.append(f"{r.old_turns} chat turn(s) older than the last {self.keep_last_n} use "
                     f"{r.old_turn_tokens} tokens; summarize them.")
        for c in r.large_chunks:
            f.append(f"{c.name} is {c.tokens} tokens (> {self.chunk_threshold}); truncate or re-chunk.")
        if r.cache.cacheable:
            f.append(f"{r.cache.prefix_tokens} prefix tokens ({r.cache.share_of_prompt:.0%}) "
                     "are cacheable.")
        if not f:
            f.append("No obvious waste found.")
        return f
