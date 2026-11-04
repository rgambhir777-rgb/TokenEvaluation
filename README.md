# token-eval

Evaluate, optimize and budget LLM token usage, for your own code or for **any GitHub repo**.

## Measure the tokens any repo uses

```bash
# 1) Static scan, nothing executed: LLM call sites, models, prompt text, codebase size
token-eval scan anthropics/anthropic-quickstarts
token-eval scan https://github.com/owner/repo --json > scan.json

# 2) Real measurement: run the repo and meter every LLM call it makes
token-eval run owner/repo --setup "pip install -r requirements.txt" -- python main.py --question "hi"
token-eval run ./local/checkout -- npm start
token-eval run owner/repo --budget budget.json --enforce-budget -- python agent.py   # stop runaway agents
token-eval report --session <run-id> --by model
token-eval report --by tag            # one row per repo (tag defaults to the repo name)
```

**How `run` works:** the repo is shallow-cloned into `.token_eval/repos/` (a local path is used as-is). token-eval starts a local metering proxy and runs your command inside the repo with these variables set:

| variable | used by |
|---|---|
| `ANTHROPIC_BASE_URL`, `ANTHROPIC_API_URL` | Anthropic SDKs (Python/TS/...), LangChain |
| `OPENAI_BASE_URL`, `OPENAI_API_BASE` | OpenAI SDKs, LangChain, LiteLLM |

Every request is forwarded unchanged to the real API using the repo's own API keys from your environment. Keys are passed through and never stored. Usage is read from each response, both normal JSON and streaming SSE, and then recorded in `tokens.db`. The record has the run id as its session, the repo name as its tag, and the model, prompt, completion and cache tokens, cost and a prompt preview. If you already set `ANTHROPIC_BASE_URL` / `OPENAI_BASE_URL` (e.g. a gateway), that becomes the upstream. With `--enforce-budget`, calls get HTTP 429 once the budget reaches STOP.

> **Running a repo runs its code with your API keys, and its LLM calls are billed to you.** Only run repos you trust, ideally in a container or VM.

**Limitations:**
- Only Anthropic and OpenAI clients that honor the base-URL variables are metered. Code that hard-codes `base_url`, or uses Bedrock, Vertex, Gemini and so on, isn't seen. `run` warns when zero calls were observed.
- OpenAI streams without `stream_options={"include_usage": True}` have their tokens estimated from the text.
- Batch API usage isn't metered.

**`scan`** is static, so its prompt numbers are lower bounds on prompt text found in the code. User input, retrieved documents, history and how many calls an execution makes can only be measured with `run`. It also reports the size of the codebase itself in tokens, counted locally with tiktoken.

- **Count** tokens for OpenAI (tiktoken) and Anthropic (the free `count_tokens` API, with a tiktoken fallback) models.
- **Track** every LLM call (tokens, cost, model, tag) into SQLite (`tokens.db`) using an editable `pricing.json`.
- **Analyze** prompts for waste, like duplicated blocks, long system prompts, old chat turns, oversized RAG chunks, and cacheable prefixes.
- **Optimize** prompts safely and see before/after tokens and dollars.
- **Budget** per task, session and day, getting `CONTINUE` / `WARN` / `STOP` back after every call.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -e ".[dashboard,dev]"
cp .env.example .env        # optional: add ANTHROPIC_API_KEY for exact Claude counts
pytest
```

> Without `ANTHROPIC_API_KEY`, Claude counts use tiktoken, which typically **undercounts Claude by ~15-20%**.
> Use `count_tokens_detailed()` to see which method produced a number.

## Quick start

```bash
python examples/demo.py                       # fake calls + budget status after each
token-eval count README.md --model gpt-4o
token-eval analyze examples/prompt.json
token-eval optimize examples/prompt.json --out optimized.json
token-eval report --by tag --since 7d         # also: --by model|day|task|session, --json
cp examples/budget.json budget.json
token-eval budget status --task demo-task-1
token-eval dashboard
```

## Library usage

### Counting

```python
from token_eval import count_tokens, count_tokens_detailed

count_tokens("Hello world", "gpt-4o")                   # tiktoken
count_tokens_detailed("Hello world", "claude-opus-5-5") # TokenCount(tokens=..., method="anthropic_api" | "tiktoken" | "heuristic")
```

### Tracking calls + budget

```python
import anthropic
from token_eval import Budget, BudgetLimits, Limit, TokenStore, TokenTracker

store = TokenStore("tokens.db")
budget = Budget(
    BudgetLimits(task=Limit(tokens=50_000, usd=0.50), day=Limit(usd=20)),
    store=store, task_id="ticket-123",
)
tracker = TokenTracker(budget=budget)
client = anthropic.Anthropic()

while budget.should_continue():                 # check before every agent step
    result = tracker.call(
        client.messages.create,
        model="claude-opus-5-5", max_tokens=4096,
        messages=[{"role": "user", "content": "..."}],
        tag="triage",
    )
    print(result.status)   # [WARN] 78% used (task) | remaining: 11,000 tokens, $0.11 | Approaching the task limit...
    ...
```

`tracker.call()` works with anything that returns a usage block. That covers Anthropic (`usage.input_tokens`, including cache read/write), OpenAI chat (`usage.prompt_tokens`, cached tokens priced as cache reads), the OpenAI Responses API, and plain dicts. If there's no usage block, it counts the prompt and response text instead. There's also a decorator (`@tracker.track(model=..., tag=...)`) and `tracker.record(model, prompt_tokens, completion_tokens, tag=...)` for usage you already know.

Status thresholds are `CONTINUE` below 70%, `WARN` from 70% up to 90% and `STOP` at 90% or more. You can change them with `warn_at` / `stop_at`. Whichever scope (task, session or day) is the most used decides the status.
Day usage is read from the database, so it is shared across processes. Task and session usage come from the database when `task_id` / `session_id` are set, and from in-memory counters otherwise.

### Analyze and optimize

```python
from token_eval import Prompt, PromptAnalyzer, optimize

prompt = Prompt.load("examples/prompt.json")
report = PromptAnalyzer(keep_last_n=6, chunk_threshold=800).analyze(prompt)
print(report.findings, report.cache.note)

result = optimize(prompt, keep_last_n=6, max_chunk_tokens=800, store=store)
print(result.tokens_before, "->", result.tokens_after, f"${result.saved_usd:.4f}/call")
print(result.suggested_model)   # set when the prompt is short and simple
```

`optimize()` runs these steps:

1. **cleanup**: removes trailing whitespace, extra blank lines, `**bold**` markers and horizontal rules. Fenced code is left alone.
2. **dedupe**: removes repeated paragraphs (≥ 40 chars) after their first appearance. Identical chunks are dropped, and a message that ends up empty keeps a short placeholder so the user/assistant alternation stays intact.
3. **summarize_history**: replaces turns older than `keep_last_n` with a summary. It is skipped if the summary wouldn't be shorter.
4. **truncate_chunks**: truncates retrieved chunks to `max_chunk_tokens`.

The default summarizer is **free and runs locally** (it keeps the first sentence of each message). For a summary written by an LLM, pass any `Callable[[list[Message]], str]`:

```python
optimize(prompt, summarizer=my_cheap_model_summarizer)
```

### Prompt JSON format

```json
{
  "model": "claude-opus-5-5",
  "system": "You are ...",
  "tools": [{"name": "lookup_order", "input_schema": {...}}],
  "chunks": [{"id": "doc-1", "text": "..."}],
  "messages": [{"role": "user", "content": "..."}]
}
```

### Prompt caching hints

The analyzer treats `tools + system` as the stable prefix. It reports the prefix's token count, whether that meets the model's minimum cacheable length (`min_cacheable_tokens` in `pricing.json`), and how much each cached repeat call would save.

## Pricing

`token_eval/pricing.json` lists USD per 1M tokens for input, output, cache_write and cache_read, plus a `tier` and `provider` used for cheaper-model suggestions. Lookup order is `$TOKEN_EVAL_PRICING`, then `./pricing.json`, then the packaged file. Unknown models match the longest known prefix, and fall back to `default` if nothing matches.
**Check the prices against your provider's current pricing page before relying on the dollar figures.**

## Layout

```
token_eval/   counting, pricing, storage, tracker, prompt, analyzer, optimizer, budget,
              repo (clone), scanner (static scan), proxy + runner (metered execution), cli, dashboard
examples/     prompt.json, budget.json, demo.py
tests/        pytest suite (no network; LLM/count calls are mocked)
```
