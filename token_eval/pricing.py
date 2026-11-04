"""Pricing table loading and cost calculation (prices are USD per 1M tokens)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

PACKAGED_PRICING = Path(__file__).with_name("pricing.json")


@dataclass(frozen=True)
class ModelPrice:
    model: str
    provider: str
    tier: str
    input: float
    output: float
    cache_write: float
    cache_read: float
    min_cacheable_tokens: int = 1024


def pricing_path() -> Path:
    """Resolve pricing.json: $TOKEN_EVAL_PRICING > ./pricing.json > packaged default."""
    env = os.environ.get("TOKEN_EVAL_PRICING")
    if env:
        return Path(env)
    local = Path.cwd() / "pricing.json"
    if local.exists():
        return local
    return PACKAGED_PRICING


class PricingTable:
    def __init__(self, data: dict) -> None:
        self._default = data.get("default", {})
        self._models: dict[str, dict] = data.get("models", {})

    @classmethod
    def load(cls, path: str | Path | None = None) -> "PricingTable":
        p = Path(path) if path else pricing_path()
        return cls(json.loads(p.read_text(encoding="utf-8")))

    @property
    def models(self) -> list[str]:
        return list(self._models)

    def get(self, model: str) -> ModelPrice:
        entry = self._models.get(model)
        if entry is None:
            # Allow dated/suffixed IDs to match their base entry (longest prefix wins).
            matches = [m for m in self._models if model.startswith(m)]
            entry = self._models[max(matches, key=len)] if matches else self._default
        fields = {k: entry.get(k, self._default.get(k)) for k in
                  ("provider", "tier", "input", "output", "cache_write", "cache_read")}
        return ModelPrice(
            model=model,
            min_cacheable_tokens=int(entry.get("min_cacheable_tokens",
                                               self._default.get("min_cacheable_tokens", 1024))),
            **fields,
        )

    def cost(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int = 0,
        cache_write_tokens: int = 0,
        cache_read_tokens: int = 0,
    ) -> float:
        """Cost in USD. prompt_tokens is the uncached input portion."""
        p = self.get(model)
        return (
            prompt_tokens * p.input
            + completion_tokens * p.output
            + cache_write_tokens * p.cache_write
            + cache_read_tokens * p.cache_read
        ) / 1_000_000

    def cheaper_alternative(self, model: str) -> str | None:
        """Cheapest known model from the same provider with a lower input price."""
        current = self.get(model)
        candidates = [
            (self.get(m).input, m) for m in self._models
            if self.get(m).provider == current.provider and self.get(m).input < current.input
        ]
        return min(candidates)[1] if candidates else None


@lru_cache(maxsize=1)
def default_pricing() -> PricingTable:
    return PricingTable.load()
