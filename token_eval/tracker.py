"""TokenTracker: wrap any LLM call, record usage + cost, persist, and check the budget."""

from __future__ import annotations

import functools
import inspect
import json
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, TypeVar

from token_eval.counting import count_tokens
from token_eval.pricing import PricingTable, default_pricing
from token_eval.storage import CallRecord, TokenStore

if TYPE_CHECKING:
    from token_eval.budget import Budget, BudgetStatus

R = TypeVar("R")
PREVIEW_CHARS = 200


@dataclass
class Usage:
    prompt_tokens: int
    completion_tokens: int
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0


@dataclass
class TrackedResult:
    response: Any
    record: CallRecord
    status: "BudgetStatus | None"


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def extract_usage(response: Any) -> Usage | None:
    """Read token usage from Anthropic, OpenAI (chat or responses API) or dict responses."""
    usage = _get(response, "usage")
    if usage is None:
        return None
    if _get(usage, "input_tokens") is not None:
        input_tokens = int(_get(usage, "input_tokens"))
        output_tokens = int(_get(usage, "output_tokens") or 0)
        details = _get(usage, "input_tokens_details")
        if details is not None:
            # OpenAI responses API: input_tokens includes cached tokens.
            cached = int(_get(details, "cached_tokens") or 0)
            return Usage(input_tokens - cached, output_tokens, 0, cached)
        # Anthropic: input_tokens is the uncached remainder only.
        return Usage(input_tokens, output_tokens,
                     int(_get(usage, "cache_creation_input_tokens") or 0),
                     int(_get(usage, "cache_read_input_tokens") or 0))
    # OpenAI chat completions: prompt_tokens includes cached tokens.
    if _get(usage, "prompt_tokens") is not None:
        cached = _get(_get(usage, "prompt_tokens_details"), "cached_tokens") or 0
        return Usage(int(_get(usage, "prompt_tokens")) - cached,
                     int(_get(usage, "completion_tokens") or 0), 0, int(cached))
    return None


def _accepts_kwarg(fn: Callable[..., Any], name: str) -> bool:
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(p.name == name or p.kind is p.VAR_KEYWORD for p in params)


def response_text(response: Any) -> str:
    if isinstance(response, str):
        return response
    content = _get(response, "content")
    if isinstance(content, list):
        return "".join(_get(b, "text", "") or "" for b in content)
    choices = _get(response, "choices")
    if choices:
        return _get(_get(choices[0], "message"), "content", "") or ""
    text = _get(response, "output_text") or _get(response, "text")
    return text if isinstance(text, str) else ""


def prompt_text(prompt: Any) -> str:
    if prompt is None:
        return ""
    if isinstance(prompt, str):
        return prompt
    return json.dumps(prompt, ensure_ascii=False, default=str)


class TokenTracker:
    """Records every LLM call to SQLite and (optionally) evaluates a Budget.

    >>> tracker = TokenTracker(TokenStore("tokens.db"), budget=budget)
    >>> result = tracker.call(client.messages.create, model="claude-opus-5-5",
    ...                       max_tokens=1024, messages=msgs, tag="search")
    >>> result.status.level  # CONTINUE / WARN / STOP
    """

    def __init__(
        self,
        store: TokenStore | None = None,
        pricing: PricingTable | None = None,
        budget: "Budget | None" = None,
        task_id: str | None = None,
        session_id: str | None = None,
    ) -> None:
        if store is None and budget is not None:
            store = budget.store
        self.store = store
        self.pricing = pricing or default_pricing()
        self.budget = budget
        self._task_id = task_id
        self.session_id = session_id or (budget.session_id if budget else None) \
            or uuid.uuid4().hex[:12]
        if budget is not None and budget.session_id is None and budget.store is store:
            budget.session_id = self.session_id

    @property
    def task_id(self) -> str | None:
        # Follow the budget's current task so Budget.new_task() is picked up automatically.
        return self._task_id or (self.budget.task_id if self.budget else None)

    @task_id.setter
    def task_id(self, value: str | None) -> None:
        self._task_id = value

    def record(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        *,
        tag: str | None = None,
        cache_write_tokens: int = 0,
        cache_read_tokens: int = 0,
        prompt_preview: str | None = None,
        task_id: str | None = None,
        session_id: str | None = None,
    ) -> tuple[CallRecord, "BudgetStatus | None"]:
        """Record usage you already know (e.g. from logs)."""
        rec = CallRecord(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cache_write_tokens=cache_write_tokens,
            cache_read_tokens=cache_read_tokens,
            cost_usd=self.pricing.cost(model, prompt_tokens, completion_tokens,
                                       cache_write_tokens, cache_read_tokens),
            tag=tag,
            task_id=task_id or self.task_id,
            session_id=session_id or self.session_id,
            prompt_preview=(prompt_preview or "")[:PREVIEW_CHARS] or None,
        )
        if self.store is not None:
            self.store.add_call(rec)
        status = self.budget.register(rec) if self.budget is not None else None
        return rec, status

    def call(
        self,
        fn: Callable[..., R],
        *args: Any,
        model: str | None = None,
        tag: str | None = None,
        prompt: Any = None,
        task_id: str | None = None,
        **kwargs: Any,
    ) -> TrackedResult:
        """Invoke ``fn(*args, **kwargs)`` and record its usage.

        ``model`` defaults to ``kwargs['model']``; it is forwarded to ``fn`` when
        ``fn`` accepts a ``model`` argument (as every SDK create() does).
        ``prompt`` is only used for the preview and as a fallback when the
        response carries no usage block; it defaults to ``kwargs['messages']``.
        """
        if model is None:
            model = kwargs.get("model")
        if model is None:
            raise ValueError("model is required (pass model= or include it in kwargs)")
        if "model" not in kwargs and _accepts_kwarg(fn, "model"):
            kwargs["model"] = model
        if prompt is None:
            prompt = kwargs.get("messages", kwargs.get("prompt", kwargs.get("input")))
        response = fn(*args, **kwargs)
        ptext = prompt_text(prompt)
        usage = extract_usage(response) or Usage(
            count_tokens(ptext, model), count_tokens(response_text(response), model))
        rec, status = self.record(
            model, usage.prompt_tokens, usage.completion_tokens, tag=tag,
            cache_write_tokens=usage.cache_write_tokens,
            cache_read_tokens=usage.cache_read_tokens,
            prompt_preview=ptext, task_id=task_id,
        )
        return TrackedResult(response, rec, status)

    def track(self, *, model: str | None = None, tag: str | None = None
              ) -> Callable[[Callable[..., R]], Callable[..., TrackedResult]]:
        """Decorator form of :meth:`call`."""
        def deco(fn: Callable[..., R]) -> Callable[..., TrackedResult]:
            @functools.wraps(fn)
            def wrapper(*args: Any, **kwargs: Any) -> TrackedResult:
                return self.call(fn, *args, model=model, tag=tag, **kwargs)
            return wrapper
        return deco
