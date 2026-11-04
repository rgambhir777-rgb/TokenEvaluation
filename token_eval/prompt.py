"""A provider-neutral prompt structure: system + tools + messages + retrieved chunks.

JSON format::

    {
      "model": "claude-opus-5-5",            # optional
      "system": "You are ...",
      "tools": [{"name": "...", ...}],        # optional, any JSON
      "messages": [{"role": "user", "content": "..."}, ...],
      "chunks": [{"id": "doc-1", "text": "..."}, "or a bare string"]   # optional
    }

Message content may be a string or a list of Anthropic/OpenAI style blocks;
text blocks are joined (non-text blocks are dropped for analysis).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Message:
    role: str
    content: str


@dataclass
class Chunk:
    id: str
    text: str


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, str):
                parts.append(b)
            elif isinstance(b, dict) and isinstance(b.get("text"), str):
                parts.append(b["text"])
        return "\n\n".join(parts)
    return "" if content is None else str(content)


@dataclass
class Prompt:
    system: str = ""
    messages: list[Message] = field(default_factory=list)
    tools: list[Any] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)
    model: str | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Prompt":
        system = d.get("system", "")
        chunks = [
            Chunk(str(c.get("id", i)), c.get("text", "")) if isinstance(c, dict)
            else Chunk(str(i), str(c))
            for i, c in enumerate(d.get("chunks", d.get("context", [])))
        ]
        return cls(
            system=_content_text(system),
            messages=[Message(m.get("role", "user"), _content_text(m.get("content")))
                      for m in d.get("messages", [])],
            tools=list(d.get("tools", [])),
            chunks=chunks,
            model=d.get("model"),
        )

    @classmethod
    def load(cls, path: str | Path) -> "Prompt":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        if self.model:
            d["model"] = self.model
        d["system"] = self.system
        if self.tools:
            d["tools"] = self.tools
        d["messages"] = [{"role": m.role, "content": m.content} for m in self.messages]
        if self.chunks:
            d["chunks"] = [{"id": c.id, "text": c.text} for c in self.chunks]
        return d

    @property
    def tools_text(self) -> str:
        return json.dumps(self.tools, ensure_ascii=False, sort_keys=True) if self.tools else ""

    def sections(self) -> list[tuple[str, str, str]]:
        """(name, kind, text) for every section, in render order."""
        out: list[tuple[str, str, str]] = []
        if self.tools:
            out.append(("tools", "tools", self.tools_text))
        if self.system:
            out.append(("system", "system", self.system))
        for c in self.chunks:
            out.append((f"chunk[{c.id}]", "chunk", c.text))
        for i, m in enumerate(self.messages):
            out.append((f"message[{i}]:{m.role}", "message", m.content))
        return out

    def turn_starts(self) -> list[int]:
        """Indices of messages that start a turn (each user message)."""
        return [i for i, m in enumerate(self.messages) if m.role == "user"]

    def split_old_turns(self, keep_last_n: int) -> tuple[list[Message], list[Message]]:
        """(older messages, the last ``keep_last_n`` turns)."""
        starts = self.turn_starts()
        if keep_last_n <= 0 or len(starts) <= keep_last_n:
            return [], list(self.messages)
        cut = starts[-keep_last_n]
        return self.messages[:cut], self.messages[cut:]
