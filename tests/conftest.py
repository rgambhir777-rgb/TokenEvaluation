from __future__ import annotations

import pytest

from token_eval import counting


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch):
    """Never hit the network in tests unless a test opts in explicitly."""
    monkeypatch.setenv("TOKEN_EVAL_OFFLINE", "1")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    counting.reset_caches()
    yield
    counting.reset_caches()


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "tokens.db"
