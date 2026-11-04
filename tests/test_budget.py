from __future__ import annotations

from datetime import timedelta

import pytest

from token_eval.budget import Budget, BudgetLimits, Level, Limit
from token_eval.storage import CallRecord, TokenStore, utcnow
from token_eval.tracker import TokenTracker


def rec(tokens: int, usd: float = 0.0, **kw) -> CallRecord:
    return CallRecord("gpt-4o", tokens, 0, usd, **kw)


@pytest.mark.parametrize("used, level", [(0, Level.CONTINUE), (690, Level.CONTINUE),
                                         (700, Level.WARN), (899, Level.WARN),
                                         (900, Level.STOP), (1500, Level.STOP)])
def test_thresholds(used, level):
    b = Budget(BudgetLimits(task=Limit(tokens=1000)))
    status = b.register(rec(used)) if used else b.status()
    assert status.level is level
    assert b.should_continue() is (level is not Level.STOP)


def test_usd_limit_binds_when_tighter():
    b = Budget(BudgetLimits(task=Limit(tokens=1_000_000, usd=1.0)))
    s = b.register(rec(100, usd=0.95))
    assert s.level is Level.STOP and s.binding_scope == "task"
    assert s.remaining_usd == pytest.approx(0.05)
    assert s.remaining_tokens == 999_900


def test_most_used_scope_binds_and_new_task_resets():
    b = Budget(BudgetLimits(task=Limit(tokens=100), day=Limit(tokens=10_000)))
    s = b.register(rec(80))
    assert s.binding_scope == "task" and s.level is Level.WARN
    assert "task" in s.recommendation
    b.new_task("next")
    s = b.status()
    assert s.level is Level.CONTINUE and s.scopes["day"].used_tokens == 80
    assert s.remaining_tokens == 100


def test_no_limits():
    s = Budget().register(rec(10**9))
    assert s.level is Level.CONTINUE and s.remaining_tokens is None
    assert "No limits" in s.recommendation


def test_day_usage_from_store_ignores_yesterday(db_path):
    store = TokenStore(db_path)
    store.add_call(rec(5000, ts=utcnow() - timedelta(days=1, hours=1)))
    store.add_call(rec(800))
    b = Budget(BudgetLimits(day=Limit(tokens=1000)), store=store)
    s = b.status()
    assert s.scopes["day"].used_tokens == 800 and s.level is Level.WARN


def test_tracker_integration_status_after_each_call(db_path):
    store = TokenStore(db_path)
    budget = Budget(BudgetLimits(task=Limit(tokens=1000), session=Limit(tokens=5000)),
                    store=store, task_id="task-1")
    tracker = TokenTracker(budget=budget)
    assert tracker.store is store

    def llm(**_):
        return {"usage": {"input_tokens": 300, "output_tokens": 50}}

    levels = [tracker.call(llm, model="claude-haiku-5-5").status.level for _ in range(3)]
    assert levels == [Level.CONTINUE, Level.WARN, Level.STOP]  # 350, 700, 1050 of 1000
    assert store.calls()[0]["task_id"] == "task-1"
    assert not budget.should_continue()


def test_from_dict():
    b = Budget.from_dict({"task": {"tokens": 10}, "day": {"usd": 5}, "warn_at": 0.5})
    assert b.limits.task.tokens == 10 and b.limits.day.usd == 5 and b.warn_at == 0.5
