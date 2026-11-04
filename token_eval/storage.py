"""SQLite persistence for call records and optimization savings."""

from __future__ import annotations

import re
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

DEFAULT_DB = "tokens.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_tokens INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    cache_write_tokens INTEGER NOT NULL DEFAULT 0,
    cache_read_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL,
    tag TEXT,
    task_id TEXT,
    session_id TEXT,
    prompt_preview TEXT
);
CREATE INDEX IF NOT EXISTS idx_calls_ts ON calls(ts);
CREATE TABLE IF NOT EXISTS savings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    model TEXT NOT NULL,
    tokens_before INTEGER NOT NULL,
    tokens_after INTEGER NOT NULL,
    usd_saved REAL NOT NULL,
    tag TEXT
);
"""


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class CallRecord:
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    tag: str | None = None
    task_id: str | None = None
    session_id: str | None = None
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    prompt_preview: str | None = None
    ts: datetime = field(default_factory=utcnow)
    id: int | None = None

    @property
    def total_tokens(self) -> int:
        return (self.prompt_tokens + self.completion_tokens
                + self.cache_write_tokens + self.cache_read_tokens)


def parse_since(value: str | None) -> datetime | None:
    """Parse '7d', '12h', '30m', '2w' or an ISO date into a UTC datetime."""
    if not value:
        return None
    m = re.fullmatch(r"(\d+)([mhdw])", value.strip())
    if m:
        n, unit = int(m.group(1)), m.group(2)
        delta = {"m": timedelta(minutes=n), "h": timedelta(hours=n),
                 "d": timedelta(days=n), "w": timedelta(weeks=n)}[unit]
        return utcnow() - delta
    dt = datetime.fromisoformat(value)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def start_of_day(now: datetime | None = None) -> datetime:
    now = now or utcnow()
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


GROUP_COLUMNS = {"tag": "tag", "model": "model", "task": "task_id",
                 "session": "session_id", "day": "substr(ts, 1, 10)"}


class TokenStore:
    def __init__(self, path: str | Path = DEFAULT_DB) -> None:
        self.path = Path(path)
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # --- writes -----------------------------------------------------------
    def add_call(self, rec: CallRecord) -> CallRecord:
        d = asdict(rec)
        d.pop("id")
        d["ts"] = rec.ts.isoformat()
        cols = ", ".join(d)
        with self._conn() as c:
            cur = c.execute(f"INSERT INTO calls ({cols}) VALUES ({', '.join('?' * len(d))})",
                            list(d.values()))
            rec.id = cur.lastrowid
        return rec

    def add_saving(self, model: str, tokens_before: int, tokens_after: int,
                   usd_saved: float, tag: str | None = None) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO savings (ts, model, tokens_before, tokens_after, usd_saved, tag) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (utcnow().isoformat(), model, tokens_before, tokens_after, usd_saved, tag),
            )

    # --- reads ------------------------------------------------------------
    def usage(self, since: datetime | None = None, task_id: str | None = None,
              session_id: str | None = None) -> tuple[int, float]:
        """(total_tokens, total_usd) matching the filters."""
        where, args = self._where(since, task_id=task_id, session_id=session_id)
        with self._conn() as c:
            row = c.execute(
                "SELECT COALESCE(SUM(prompt_tokens + completion_tokens + cache_write_tokens"
                " + cache_read_tokens), 0) AS t, COALESCE(SUM(cost_usd), 0) AS u "
                f"FROM calls {where}", args).fetchone()
        return int(row["t"]), float(row["u"])

    def report(self, by: str = "tag", since: datetime | None = None,
               session_id: str | None = None, tag: str | None = None) -> list[dict]:
        if by not in GROUP_COLUMNS:
            raise ValueError(f"by must be one of {sorted(GROUP_COLUMNS)}")
        col = GROUP_COLUMNS[by]
        where, args = self._where(since, session_id=session_id, tag=tag)
        with self._conn() as c:
            rows = c.execute(
                f"SELECT {col} AS key, COUNT(*) AS calls, SUM(prompt_tokens) AS prompt_tokens, "
                "SUM(completion_tokens) AS completion_tokens, SUM(cost_usd) AS cost_usd "
                f"FROM calls {where} GROUP BY key ORDER BY cost_usd DESC", args).fetchall()
        return [dict(r) for r in rows]

    def calls(self, since: datetime | None = None, limit: int | None = None,
              order: str = "ts DESC", session_id: str | None = None) -> list[dict]:
        assert order in ("ts DESC", "cost_usd DESC")
        where, args = self._where(since, session_id=session_id)
        sql = f"SELECT * FROM calls {where} ORDER BY {order}"
        if limit:
            sql += f" LIMIT {int(limit)}"
        with self._conn() as c:
            return [dict(r) for r in c.execute(sql, args).fetchall()]

    def savings(self, since: datetime | None = None) -> list[dict]:
        where, args = self._where(since)
        with self._conn() as c:
            return [dict(r) for r in c.execute(
                f"SELECT * FROM savings {where} ORDER BY ts", args).fetchall()]

    @staticmethod
    def _where(since: datetime | None, **eq: str | None) -> tuple[str, list]:
        clauses, args = [], []
        if since is not None:
            clauses.append("ts >= ?")
            args.append(since.isoformat())
        for col, val in eq.items():
            if val is not None:
                clauses.append(f"{col} = ?")
                args.append(val)
        return ("WHERE " + " AND ".join(clauses)) if clauses else "", args
