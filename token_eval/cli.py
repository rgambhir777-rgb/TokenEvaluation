"""token-eval command line interface."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from token_eval.analyzer import PromptAnalyzer
from token_eval.budget import Budget, Level
from token_eval.counting import count_tokens_detailed
from token_eval.optimizer import optimize as run_optimize
from token_eval.prompt import Prompt
from token_eval.repo import RepoError, repo_name, resolve_repo
from token_eval.storage import DEFAULT_DB, TokenStore, parse_since

app = typer.Typer(help="Evaluate, optimize and budget LLM token usage.", no_args_is_help=True)
budget_app = typer.Typer(help="Budget commands.", no_args_is_help=True)
app.add_typer(budget_app, name="budget")
console = Console()

DbOpt = typer.Option(None, "--db", help="SQLite path (default: $TOKEN_EVAL_DB or tokens.db).")
JsonOpt = typer.Option(False, "--json", help="Print machine-readable JSON.")


@app.callback()
def _init() -> None:
    load_dotenv()
    # Legacy Windows consoles (cp1252) can't encode every character in user prompts.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")


def _db(path: Optional[Path]) -> TokenStore:
    return TokenStore(path or os.environ.get("TOKEN_EVAL_DB", DEFAULT_DB))


def _load_prompt(path: Path) -> Prompt:
    try:
        return Prompt.load(path)
    except (OSError, json.JSONDecodeError) as e:
        console.print(f"[red]Could not read prompt {escape(str(path))}: {escape(str(e))}[/red]")
        raise typer.Exit(1)


@app.command()
def count(
    file: Path = typer.Argument(..., exists=True, dir_okay=False, help="Text file to count."),
    model: str = typer.Option("claude-opus-5-5", "--model", "-m"),
) -> None:
    """Count tokens in a file."""
    tc = count_tokens_detailed(file.read_text(encoding="utf-8"), model)
    console.print(f"{tc.tokens:,} tokens  [dim]({model}, via {tc.method})[/dim]")


@app.command()
def analyze(
    prompt_file: Path = typer.Argument(..., exists=True, dir_okay=False),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Overrides prompt's model."),
    keep_last: int = typer.Option(6, help="Turns considered 'recent'."),
    system_threshold: int = typer.Option(1500, help="Flag system prompts above this."),
    chunk_threshold: int = typer.Option(800, help="Flag chunks above this."),
    as_json: bool = JsonOpt,
) -> None:
    """Report where a prompt's tokens go and what could be saved."""
    r = PromptAnalyzer(model, system_threshold=system_threshold, keep_last_n=keep_last,
                       chunk_threshold=chunk_threshold).analyze(_load_prompt(prompt_file))
    if as_json:
        print(json.dumps(r.to_dict(), indent=2))
        return
    console.print(f"[bold]{r.total_tokens:,} tokens[/bold] on {r.model} "
                  f"(~${r.estimated_input_cost_usd:.5f} input per call)")
    t = Table("section", "kind", "tokens", "%")
    for s in r.sections:
        t.add_row(escape(s.name), s.kind, f"{s.tokens:,}", f"{s.tokens / max(r.total_tokens, 1):.0%}")
    console.print(t)
    if r.duplicates:
        d = Table("duplicate block", "x", "wasted", "where", title="Duplicates")
        for dup in r.duplicates:
            d.add_row(escape(dup.preview), str(dup.occurrences), str(dup.wasted_tokens),
                      escape(", ".join(dup.locations)))
        console.print(d)
    console.print(f"[bold]Caching:[/bold] {r.cache.note}")
    console.print("[bold]Findings:[/bold]")
    for f in r.findings:
        console.print(f"  - {escape(f)}")


@app.command()
def optimize(
    prompt_file: Path = typer.Argument(..., exists=True, dir_okay=False),
    model: Optional[str] = typer.Option(None, "--model", "-m"),
    keep_last: int = typer.Option(6, help="Keep the last N turns verbatim; summarize older."),
    max_chunk_tokens: int = typer.Option(800, help="Truncate retrieved chunks to this size."),
    out: Optional[Path] = typer.Option(None, "--out", "-o", help="Write optimized prompt JSON."),
    record: bool = typer.Option(True, help="Record savings to the database."),
    tag: Optional[str] = typer.Option(None, help="Tag stored with the savings record."),
    db: Optional[Path] = DbOpt,
    as_json: bool = JsonOpt,
) -> None:
    """Apply safe reductions and show before/after tokens and savings."""
    r = run_optimize(_load_prompt(prompt_file), model, keep_last_n=keep_last,
                     max_chunk_tokens=max_chunk_tokens,
                     store=_db(db) if record else None, tag=tag)
    if out:
        out.write_text(json.dumps(r.prompt.to_dict(), indent=2, ensure_ascii=False),
                       encoding="utf-8")
    if as_json:
        print(json.dumps(r.to_dict(), indent=2, ensure_ascii=False))
        return
    t = Table("step", "tokens saved", "detail")
    for s in r.steps:
        t.add_row(s.name, f"{s.tokens_saved:,}", escape(s.detail))
    console.print(t)
    console.print(f"[bold]{r.tokens_before:,} -> {r.tokens_after:,} tokens[/bold] "
                  f"(-{r.saved_tokens:,}, {r.saved_pct:.0%}) ~${r.saved_usd:.5f} saved per call "
                  f"on {r.model}")
    if r.suggested_model:
        console.print(f"Prompt is short and simple: consider [bold]{r.suggested_model}[/bold] "
                      f"(saves ~${r.suggested_model_saved_usd:.5f} more per call).")
    if r.cache:
        console.print(f"Caching: {r.cache.note}")
    if out:
        console.print(f"Optimized prompt written to {out}")


@app.command()
def report(
    by: str = typer.Option("tag", help="Group by: tag, model, task, session, day."),
    since: Optional[str] = typer.Option(None, help="e.g. 7d, 24h, 2w or an ISO date."),
    session: Optional[str] = typer.Option(None, help="Only this session / run id."),
    tag: Optional[str] = typer.Option(None, help="Only this tag (e.g. a repo name)."),
    db: Optional[Path] = DbOpt,
    as_json: bool = JsonOpt,
) -> None:
    """Usage and cost from the tracking database."""
    try:
        rows = _db(db).report(by=by, since=parse_since(since), session_id=session, tag=tag)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    if as_json:
        print(json.dumps(rows, indent=2))
        return
    t = Table(by, "calls", "prompt tok", "completion tok", "cost $",
              title=f"Usage by {by}" + (f" since {since}" if since else ""))
    for r in rows:
        t.add_row(escape(str(r["key"])), str(r["calls"]), f"{r['prompt_tokens']:,}",
                  f"{r['completion_tokens']:,}", f"{r['cost_usd']:.4f}")
    if rows:
        t.add_section()
        t.add_row("TOTAL", str(sum(r["calls"] for r in rows)),
                  f"{sum(r['prompt_tokens'] for r in rows):,}",
                  f"{sum(r['completion_tokens'] for r in rows):,}",
                  f"{sum(r['cost_usd'] for r in rows):.4f}")
    console.print(t)


@app.command()
def scan(
    repo: str = typer.Argument(..., help="GitHub URL, owner/repo, or local path."),
    model: Optional[str] = typer.Option(None, "--model", "-m",
                                        help="Tokenizer/pricing model (default: most used in repo)."),
    ref: Optional[str] = typer.Option(None, help="Branch or tag to clone."),
    fresh: bool = typer.Option(False, help="Re-clone even if cached."),
    top: int = typer.Option(10, help="How many prompts / call sites to list."),
    as_json: bool = JsonOpt,
) -> None:
    """Statically scan a repo (nothing is executed): LLM call sites, models, prompt tokens."""
    from token_eval.scanner import scan_repo

    try:
        path = resolve_repo(repo, ref=ref, fresh=fresh)
    except RepoError as e:
        console.print(f"[red]{escape(str(e))}[/red]")
        raise typer.Exit(1)
    r = scan_repo(path, model=model)
    if as_json:
        print(json.dumps(r.to_dict(), indent=2, ensure_ascii=False))
        return
    console.print(f"[bold]{escape(repo)}[/bold] ({r.files_scanned} files) - codebase is "
                  f"[bold]{r.codebase_tokens:,}[/bold] tokens")
    if r.models:
        console.print("Models referenced: " + ", ".join(f"{escape(m)} x{n}" for m, n in r.models.items()))
    cs = Table("file:line", "provider", "kind", "code", title=f"LLM call sites ({len(r.call_sites)})")
    for c in r.call_sites[:top]:
        cs.add_row(escape(f"{c.file}:{c.line}"), c.provider, c.kind, escape(c.snippet))
    console.print(cs)
    pt = Table("location", "tokens", "preview",
               title=f"Static prompt text ({len(r.prompts)}), counted for {escape(r.count_model)}")
    for p in r.prompts[:top]:
        pt.add_row(escape(p.location), f"{p.tokens:,}", escape(p.preview))
    console.print(pt)
    console.print(f"Static prompt tokens: [bold]{r.static_prompt_tokens:,}[/bold] "
                  f"(~${r.static_prompt_cost_usd:.5f} if each were sent once)")
    for n in r.notes:
        console.print(f"[dim]note: {escape(n)}[/dim]")


@app.command()
def run(
    repo: str = typer.Argument(..., help="GitHub URL, owner/repo, or local path."),
    command: list[str] = typer.Argument(..., help="Command to run inside the repo (after --)."),
    setup: Optional[str] = typer.Option(None, help='Shell command run first, e.g. '
                                        '"pip install -r requirements.txt".'),
    tag: Optional[str] = typer.Option(None, help="Tag for recorded calls (default: repo name)."),
    budget_config: Optional[Path] = typer.Option(None, "--budget",
                                                 help="Budget JSON; the run is the session."),
    enforce_budget: bool = typer.Option(False, help="Block LLM calls (HTTP 429) once the "
                                        "budget reaches STOP."),
    env_file: Optional[Path] = typer.Option(None, "--env-file", exists=True, dir_okay=False,
                                            help="Dotenv file passed to the command's env "
                                                 "(e.g. the repo's .env with its API key)."),
    env: Optional[list[str]] = typer.Option(None, "--env", "-e",
                                            help="Extra KEY=VALUE for the command (repeatable)."),
    ref: Optional[str] = typer.Option(None, help="Branch or tag to clone."),
    fresh: bool = typer.Option(False, help="Re-clone even if cached."),
    db: Optional[Path] = DbOpt,
    as_json: bool = JsonOpt,
) -> None:
    """Execute a repo's command and measure every LLM token it uses.

    Example: token-eval run owner/repo --setup "pip install -r requirements.txt" -- python main.py
    """
    from token_eval.runner import load_env_file, run_repo

    try:
        path = resolve_repo(repo, ref=ref, fresh=fresh)
    except RepoError as e:
        console.print(f"[red]{escape(str(e))}[/red]")
        raise typer.Exit(1)
    budget = Budget.from_file(budget_config) if budget_config else None
    extra_env = load_env_file(env_file) if env_file else {}
    for item in env or []:
        key, sep, value = item.partition("=")
        if not sep:
            console.print(f"[red]--env expects KEY=VALUE, got {escape(item)}[/red]")
            raise typer.Exit(2)
        extra_env[key] = value
    store = _db(db)
    console.print(f"[dim]Running in {escape(str(path))} with LLM traffic metered...[/dim]")
    res = run_repo(path, command, store=store, tag=tag or repo_name(repo), setup=setup,
                   budget=budget, enforce_budget=enforce_budget, extra_env=extra_env)
    if as_json:
        print(json.dumps({k: (str(v) if isinstance(v, Path) else v)
                          for k, v in vars(res).items()}, indent=2))
        raise typer.Exit(res.exit_code)
    t = Table("model", "calls", "prompt tok", "completion tok", "cost $",
              title=f"LLM usage for run {res.run_id}")
    for r in res.by_model:
        t.add_row(escape(str(r["key"])), str(r["calls"]), f"{r['prompt_tokens']:,}",
                  f"{r['completion_tokens']:,}", f"{r['cost_usd']:.4f}")
    console.print(t)
    console.print(f"[bold]{res.calls} LLM call(s), {res.total_tokens:,} tokens "
                  f"(incl. cache), ${res.cost_usd:.4f}[/bold]  exit code {res.exit_code}")
    if res.blocked_calls:
        console.print(f"[yellow]{res.blocked_calls} call(s) blocked by the budget.[/yellow]")
    if res.failed_calls:
        statuses = ", ".join(f"HTTP {k} x{v}" for k, v in sorted(res.failed_calls.items()))
        console.print(f"[red]{sum(res.failed_calls.values())} LLM call(s) were rejected by the "
                      f"provider ({statuses}): {escape(res.last_failure or '')}[/red]")
    elif res.calls == 0:
        console.print("[yellow]No LLM calls were seen. The program may not call an LLM on this "
                      "path, or it hard-codes a base_url / uses another provider "
                      "(only base-URL-configurable Anthropic and OpenAI clients are metered).[/yellow]")
    console.print(f"[dim]Details: token-eval report --session {res.run_id} --by model[/dim]")
    raise typer.Exit(res.exit_code)


@budget_app.command("status")
def budget_status(
    config: Path = typer.Option(Path("budget.json"), "--config", "-c",
                                help="Budget limits JSON."),
    task: Optional[str] = typer.Option(None, help="Task id to evaluate."),
    session: Optional[str] = typer.Option(None, help="Session id to evaluate."),
    db: Optional[Path] = DbOpt,
    as_json: bool = JsonOpt,
) -> None:
    """Show CONTINUE / WARN / STOP against configured limits."""
    if not config.exists():
        console.print(f"[red]No budget config at {config}. See examples/budget.json.[/red]")
        raise typer.Exit(1)
    b = Budget.from_file(config, store=_db(db), task_id=task, session_id=session)
    s = b.status()
    if as_json:
        print(json.dumps({
            "level": s.level.value, "used_fraction": s.used_fraction,
            "binding_scope": s.binding_scope, "remaining_tokens": s.remaining_tokens,
            "remaining_usd": s.remaining_usd, "recommendation": s.recommendation,
            "scopes": {k: {"used_tokens": v.used_tokens, "used_usd": v.used_usd,
                           "limit_tokens": v.limit.tokens, "limit_usd": v.limit.usd}
                       for k, v in s.scopes.items()},
        }, indent=2))
        return
    color = {Level.CONTINUE: "green", Level.WARN: "yellow", Level.STOP: "red"}[s.level]
    t = Table("scope", "tokens used / limit", "USD used / limit", "used")
    for name, u in s.scopes.items():
        if name == "task" and not task or name == "session" and not session:
            continue  # without an id these would only reflect this (empty) process
        t.add_row(name,
                  f"{u.used_tokens:,} / {u.limit.tokens:,}" if u.limit.tokens else f"{u.used_tokens:,}",
                  f"{u.used_usd:.4f} / {u.limit.usd:.2f}" if u.limit.usd else f"{u.used_usd:.4f}",
                  f"{u.fraction:.0%}")
    console.print(t)
    console.print(f"[bold {color}]{s.level.value}[/bold {color}] {s.recommendation}")


@app.command()
def dashboard(db: Optional[Path] = DbOpt,
              budget_config: Path = typer.Option(Path("budget.json"), "--budget")) -> None:
    """Launch the Streamlit dashboard."""
    script = Path(__file__).with_name("dashboard.py")
    env = {**os.environ, "TOKEN_EVAL_DB": str(db or os.environ.get("TOKEN_EVAL_DB", DEFAULT_DB)),
           "TOKEN_EVAL_BUDGET": str(budget_config)}
    raise typer.Exit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(script)],
                                     env=env))


if __name__ == "__main__":
    app()
