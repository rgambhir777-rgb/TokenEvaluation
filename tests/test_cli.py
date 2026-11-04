from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from token_eval.cli import app
from token_eval.storage import CallRecord, TokenStore

ROOT = Path(__file__).parents[1]
EXAMPLE = ROOT / "examples" / "prompt.json"
BUDGET = ROOT / "examples" / "budget.json"
runner = CliRunner()


def test_count(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("hello world " * 10)
    res = runner.invoke(app, ["count", str(f), "--model", "gpt-4o"])
    assert res.exit_code == 0 and "tokens" in res.output


def test_analyze_json():
    res = runner.invoke(app, ["analyze", str(EXAMPLE), "--json"])
    assert res.exit_code == 0, res.output
    data = json.loads(res.output)
    assert data["total_tokens"] > 0 and data["duplicates"]


def test_analyze_pretty():
    res = runner.invoke(app, ["analyze", str(EXAMPLE)])
    assert res.exit_code == 0 and "Findings" in res.output
    assert "chunk[tent-manual]" in res.output  # brackets must not be eaten as rich markup


def test_optimize_writes_output_and_records(tmp_path):
    db, out = tmp_path / "t.db", tmp_path / "opt.json"
    res = runner.invoke(app, ["optimize", str(EXAMPLE), "--out", str(out), "--db", str(db)])
    assert res.exit_code == 0, res.output
    assert json.loads(out.read_text())["messages"]
    assert len(TokenStore(db).savings()) == 1


def test_report(tmp_path):
    db = tmp_path / "t.db"
    store = TokenStore(db)
    store.add_call(CallRecord("gpt-4o", 100, 10, 0.5, tag="search"))
    res = runner.invoke(app, ["report", "--by", "tag", "--since", "7d", "--db", str(db), "--json"])
    assert res.exit_code == 0
    assert json.loads(res.output)[0]["key"] == "search"
    assert runner.invoke(app, ["report", "--by", "bad", "--db", str(db)]).exit_code == 1


def test_budget_status(tmp_path):
    db = tmp_path / "t.db"
    TokenStore(db).add_call(CallRecord("gpt-4o", 1_500_000, 0, 1.0, task_id="t1"))
    res = runner.invoke(app, ["budget", "status", "-c", str(BUDGET), "--db", str(db), "--json"])
    assert json.loads(res.output)["level"] == "WARN"  # 75% of daily tokens
    res = runner.invoke(app, ["budget", "status", "-c", str(BUDGET), "--db", str(db),
                              "--task", "t1"])
    assert res.exit_code == 0 and "STOP" in res.output
    assert runner.invoke(app, ["budget", "status", "-c", str(tmp_path / "x.json")]).exit_code == 1


def test_scan_cli(tmp_path):
    (tmp_path / "app.py").write_text(
        'import anthropic\nc = anthropic.Anthropic()\n'
        'c.messages.create(model="claude-haiku-5-5", system="You are a terse assistant that '
        'answers only with yes or no.", messages=[])\n')
    res = runner.invoke(app, ["scan", str(tmp_path), "--json"])
    assert res.exit_code == 0, res.output
    data = json.loads(res.output)
    assert data["models"] == {"claude-haiku-5-5": 1} and len(data["call_sites"]) == 1
    assert "LLM call sites" in runner.invoke(app, ["scan", str(tmp_path)]).output


def test_run_cli_meters_and_passes_exit_code(tmp_path, monkeypatch):
    import sys

    from fake_upstream import FakeUpstream
    from test_proxy import SCRIPT

    (tmp_path / "main.py").write_text(SCRIPT + "\nraise SystemExit(5)\n")
    db = tmp_path / "t.db"
    with FakeUpstream() as up:
        monkeypatch.setenv("ANTHROPIC_BASE_URL", up.url)  # treated as the real upstream
        res = runner.invoke(app, ["run", str(tmp_path), "--db", str(db), "--tag", "demo",
                                  "--", sys.executable, "main.py"])
    assert res.exit_code == 5, res.output
    assert "3 LLM call(s)" in res.output
    rows = TokenStore(db).report(by="tag")
    assert rows[0]["key"] == "demo" and rows[0]["calls"] == 3
