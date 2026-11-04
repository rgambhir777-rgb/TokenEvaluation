"""Budgets per task / session / day, and the CONTINUE / WARN / STOP decision."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Any

from token_eval.storage import CallRecord, TokenStore, start_of_day, utcnow

SCOPES = ("task", "session", "day")


class Level(str, Enum):
    CONTINUE = "CONTINUE"
    WARN = "WARN"
    STOP = "STOP"


@dataclass
class Limit:
    tokens: int | None = None
    usd: float | None = None


@dataclass
class BudgetLimits:
    task: Limit = field(default_factory=Limit)
    session: Limit = field(default_factory=Limit)
    day: Limit = field(default_factory=Limit)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BudgetLimits":
        return cls(**{s: Limit(**d.get(s, {})) for s in SCOPES})


@dataclass
class ScopeUsage:
    scope: str
    used_tokens: int
    used_usd: float
    limit: Limit

    @property
    def fraction(self) -> float:
        fr = [0.0]
        if self.limit.tokens:
            fr.append(self.used_tokens / self.limit.tokens)
        if self.limit.usd:
            fr.append(self.used_usd / self.limit.usd)
        return max(fr)

    @property
    def remaining_tokens(self) -> int | None:
        return None if self.limit.tokens is None else max(0, self.limit.tokens - self.used_tokens)

    @property
    def remaining_usd(self) -> float | None:
        return None if self.limit.usd is None else max(0.0, self.limit.usd - self.used_usd)


@dataclass
class BudgetStatus:
    level: Level
    used_fraction: float
    binding_scope: str | None
    remaining_tokens: int | None
    remaining_usd: float | None
    recommendation: str
    scopes: dict[str, ScopeUsage]

    def __str__(self) -> str:
        rt = "unlimited" if self.remaining_tokens is None else f"{self.remaining_tokens:,}"
        ru = "unlimited" if self.remaining_usd is None else f"${self.remaining_usd:.4f}"
        return (f"[{self.level.value}] {self.used_fraction:.0%} used "
                f"({self.binding_scope or 'no limits'}) | remaining: {rt} tokens, {ru} | "
                f"{self.recommendation}")


class Budget:
    """Tracks spend against limits.

    Day usage comes from the store (shared across processes) when one is given.
    Task / session usage comes from the store when ``task_id`` / ``session_id``
    are set, otherwise from in-memory counters for this Budget instance.
    """

    def __init__(
        self,
        limits: BudgetLimits | None = None,
        store: TokenStore | None = None,
        task_id: str | None = None,
        session_id: str | None = None,
        warn_at: float = 0.70,
        stop_at: float = 0.90,
    ) -> None:
        self.limits = limits or BudgetLimits()
        self.store = store
        self.task_id = task_id
        self.session_id = session_id
        self.warn_at = warn_at
        self.stop_at = stop_at
        self._mem: dict[str, list[float]] = {s: [0, 0.0] for s in SCOPES}
        self._mem_day: date = utcnow().date()

    # --- construction -----------------------------------------------------
    @classmethod
    def from_dict(cls, d: dict[str, Any], **kwargs: Any) -> "Budget":
        return cls(BudgetLimits.from_dict(d), warn_at=d.get("warn_at", 0.70),
                   stop_at=d.get("stop_at", 0.90), **kwargs)

    @classmethod
    def from_file(cls, path: str | Path, **kwargs: Any) -> "Budget":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")), **kwargs)

    def new_task(self, task_id: str | None = None) -> None:
        """Start counting a fresh task."""
        self.task_id = task_id
        self._mem["task"] = [0, 0.0]

    # --- accounting -------------------------------------------------------
    def register(self, rec: CallRecord) -> BudgetStatus:
        """Account for a call (the tracker calls this) and return the new status."""
        self._roll_day()
        for s in SCOPES:
            self._mem[s][0] += rec.total_tokens
            self._mem[s][1] += rec.cost_usd
        return self.status()

    def _roll_day(self) -> None:
        today = utcnow().date()
        if today != self._mem_day:
            self._mem_day, self._mem["day"] = today, [0, 0.0]

    def usage(self, scope: str) -> tuple[int, float]:
        self._roll_day()
        if self.store is not None:
            if scope == "day":
                return self.store.usage(since=start_of_day())
            if scope == "task" and self.task_id:
                return self.store.usage(task_id=self.task_id)
            if scope == "session" and self.session_id:
                return self.store.usage(session_id=self.session_id)
        t, u = self._mem[scope]
        return int(t), float(u)

    def status(self) -> BudgetStatus:
        scopes = {s: ScopeUsage(s, *self.usage(s), getattr(self.limits, s)) for s in SCOPES}
        limited = [u for u in scopes.values() if u.limit.tokens or u.limit.usd]
        binding = max(limited, key=lambda u: u.fraction) if limited else None
        frac = binding.fraction if binding else 0.0
        level = (Level.STOP if frac >= self.stop_at
                 else Level.WARN if frac >= self.warn_at else Level.CONTINUE)
        rem_t = [u.remaining_tokens for u in scopes.values() if u.remaining_tokens is not None]
        rem_u = [u.remaining_usd for u in scopes.values() if u.remaining_usd is not None]
        return BudgetStatus(
            level=level,
            used_fraction=frac,
            binding_scope=binding.scope if binding else None,
            remaining_tokens=min(rem_t) if rem_t else None,
            remaining_usd=min(rem_u) if rem_u else None,
            recommendation=self._recommend(level, binding),
            scopes=scopes,
        )

    def should_continue(self) -> bool:
        """Call before each agent step: False once any limit reaches the STOP threshold."""
        return self.status().level is not Level.STOP

    @staticmethod
    def _recommend(level: Level, binding: ScopeUsage | None) -> str:
        if binding is None:
            return "No limits configured; set limits to get guidance."
        scope = binding.scope
        if level is Level.STOP:
            return {
                "task": "Task budget exhausted: return your best partial result and move on.",
                "session": "Session budget exhausted: save state and end the session.",
                "day": "Daily budget exhausted: pause non-essential work until tomorrow.",
            }[scope]
        if level is Level.WARN:
            return (f"Approaching the {scope} limit: wrap up the current step, summarize "
                    "context, and prefer a cheaper model.")
        return f"Within budget ({binding.fraction:.0%} of the {scope} limit used)."
