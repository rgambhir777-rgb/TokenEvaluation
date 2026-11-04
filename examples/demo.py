"""Demo: run a few fake LLM calls through TokenTracker and print the budget after each.

No API calls are made. Run from the project root:

    python examples/demo.py
"""

from __future__ import annotations

import random
from pathlib import Path

from token_eval import Budget, BudgetLimits, Limit, TokenStore, TokenTracker, optimize
from token_eval.prompt import Prompt

HERE = Path(__file__).parent
random.seed(7)


def fake_llm(*, model: str, messages: list[dict], max_tokens: int = 1024) -> dict:
    """Stands in for client.messages.create(); returns an Anthropic-shaped dict."""
    prompt_tokens = sum(len(m["content"]) for m in messages) // 4 + random.randint(1500, 4000)
    return {
        "content": [{"type": "text", "text": "Sure, here's an answer..."}],
        "usage": {"input_tokens": prompt_tokens,
                  "output_tokens": random.randint(200, min(max_tokens, 900))},
    }


def main() -> None:
    store = TokenStore("tokens.db")
    budget = Budget(
        BudgetLimits(task=Limit(tokens=20_000, usd=0.10),
                     session=Limit(tokens=60_000),
                     day=Limit(usd=5.00)),
        store=store,
        task_id="demo-task-1",
    )
    tracker = TokenTracker(budget=budget)  # uses the budget's store

    tags = ["search", "summarize", "chat", "search", "codegen", "chat", "summarize", "search"]
    step = 0
    while budget.should_continue() and step < len(tags):
        tag = tags[step]
        result = tracker.call(
            fake_llm,
            model="claude-opus-5-5" if tag == "codegen" else "claude-sonnet-5-5",
            messages=[{"role": "user", "content": f"Step {step}: please do the {tag} task."}],
            tag=tag,
        )
        rec = result.record
        print(f"step {step} [{tag:9}] {rec.model:18} {rec.total_tokens:>6} tok "
              f"${rec.cost_usd:.4f}")
        print(f"         {result.status}")
        step += 1

    if not budget.should_continue():
        print("\nBudget says STOP -> agent loop exits and hands off.")

    # Record an optimization so the dashboard has savings to show.
    opt = optimize(Prompt.load(HERE / "prompt.json"), keep_last_n=4, store=store, tag="demo")
    print(f"\noptimize(examples/prompt.json): {opt.tokens_before} -> {opt.tokens_after} tokens "
          f"(-{opt.saved_pct:.0%}, ${opt.saved_usd:.5f} per call)")
    print("Data written to tokens.db - try `token-eval report --by tag` or `token-eval dashboard`.")


if __name__ == "__main__":
    main()
