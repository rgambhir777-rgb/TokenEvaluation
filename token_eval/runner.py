"""Run a repository's command with every LLM call metered by the proxy."""

from __future__ import annotations

import os
import shutil
import subprocess
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

from token_eval.budget import Budget
from token_eval.proxy import DEFAULT_UPSTREAMS, MeteringProxy
from token_eval.storage import TokenStore
from token_eval.tracker import TokenTracker


@dataclass
class RunResult:
    run_id: str
    repo_dir: Path
    tag: str
    exit_code: int
    calls: int
    blocked_calls: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    by_model: list[dict] = field(default_factory=list)
    failed_calls: dict[int, int] = field(default_factory=dict)
    last_failure: str | None = None
    proxy_errors: list[str] = field(default_factory=list)


def _upstreams_from_env(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Respect gateways the user already configured as the real upstream."""
    env = os.environ if env is None else env
    ups = dict(DEFAULT_UPSTREAMS)
    if env.get("ANTHROPIC_BASE_URL"):
        ups["anthropic"] = env["ANTHROPIC_BASE_URL"].rstrip("/")
    if env.get("OPENAI_BASE_URL"):
        # OPENAI_BASE_URL normally ends in /v1; the proxy route adds /v1 itself.
        ups["openai"] = env["OPENAI_BASE_URL"].rstrip("/").removesuffix("/v1")
    return ups


def load_env_file(path: str | Path) -> dict[str, str]:
    """Read a dotenv file (values are passed to the child process, never printed)."""
    from dotenv import dotenv_values

    return {k: v for k, v in dotenv_values(path).items() if v is not None}


def _shell_cmd(command: Sequence[str] | str) -> list[str] | str:
    if isinstance(command, str):
        return command
    cmd = list(command)
    exe = shutil.which(cmd[0])  # resolves npm.cmd / python.exe etc. on Windows
    return [exe, *cmd[1:]] if exe else cmd


def run_repo(
    repo_dir: Path,
    command: Sequence[str] | str,
    *,
    store: TokenStore,
    tag: str,
    setup: str | None = None,
    budget: Budget | None = None,
    enforce_budget: bool = False,
    upstreams: dict[str, str] | None = None,
    extra_env: dict[str, str] | None = None,
    run_id: str | None = None,
) -> RunResult:
    """Execute ``command`` inside ``repo_dir`` and meter all LLM traffic.

    ``command`` is a list (run without a shell) or a string (run through the shell).
    ``setup`` (always a shell string, e.g. "pip install -r requirements.txt") runs first
    and is metered too.
    """
    run_id = run_id or uuid.uuid4().hex[:12]
    if budget is not None:
        budget.store = budget.store or store
        budget.session_id = run_id
    tracker = TokenTracker(store, budget=budget, session_id=run_id, task_id=run_id)

    base_env = {**os.environ, **(extra_env or {})}
    with MeteringProxy(tracker, upstreams=upstreams or _upstreams_from_env(base_env), tag=tag,
                       enforce_budget=enforce_budget) as proxy:
        env = {**base_env, **proxy.client_env(),
               "TOKEN_EVAL_RUN_ID": run_id}
        exit_code = 0
        if setup:
            exit_code = subprocess.call(setup, cwd=repo_dir, env=env, shell=True)
        if exit_code == 0:
            cmd = _shell_cmd(command)
            exit_code = subprocess.call(cmd, cwd=repo_dir, env=env,
                                        shell=isinstance(cmd, str))
        calls, blocked, errors = proxy.calls, proxy.blocked, list(proxy.errors)
        failed, last_failure = dict(proxy.failed), proxy.last_failure

    rows = store.report(by="model", session_id=run_id)
    prompt = sum(r["prompt_tokens"] or 0 for r in rows)
    completion = sum(r["completion_tokens"] or 0 for r in rows)
    total, cost = store.usage(session_id=run_id)
    return RunResult(run_id, repo_dir, tag, exit_code, calls, blocked, prompt, completion,
                     total, cost, rows, errors, failed, last_failure)
