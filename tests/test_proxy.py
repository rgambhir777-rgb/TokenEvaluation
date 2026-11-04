from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

import anthropic
import pytest

from fake_upstream import FakeUpstream
from token_eval.budget import Budget, BudgetLimits, Limit
from token_eval.proxy import MeteringProxy, is_metered
from token_eval.repo import normalize_url, repo_name, resolve_repo
from token_eval.runner import run_repo
from token_eval.storage import TokenStore
from token_eval.tracker import TokenTracker


@pytest.fixture
def upstream():
    with FakeUpstream() as up:
        yield up


def ups(upstream) -> dict[str, str]:
    return {"anthropic": upstream.url, "openai": upstream.url}


def post(url: str, body: dict, headers: dict | None = None) -> tuple[int, bytes]:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def test_is_metered():
    assert is_metered("/v1/messages") and is_metered("/v1/chat/completions")
    assert not is_metered("/v1/messages/count_tokens") and not is_metered("/v1/models")


def test_anthropic_sdk_through_proxy(upstream, db_path):
    store = TokenStore(db_path)
    with MeteringProxy(TokenTracker(store), upstreams=ups(upstream), tag="repo-x") as proxy:
        client = anthropic.Anthropic(api_key="sk-test", base_url=proxy.base_url("anthropic"))
        msg = client.messages.create(model="claude-opus-5-5", max_tokens=100,
                                     messages=[{"role": "user", "content": "hello there"}])
        assert msg.content[0].text == "Hello!"  # response passed through intact
        with client.messages.stream(model="claude-opus-5-5", max_tokens=100,
                                    messages=[{"role": "user", "content": "stream pls"}]) as s:
            final = s.get_final_message()
        assert final.usage.output_tokens == 45
        client.messages.count_tokens(model="claude-opus-5-5",
                                     messages=[{"role": "user", "content": "x"}])
    rows = sorted(store.calls(), key=lambda r: r["id"])
    assert len(rows) == 2  # count_tokens is not metered
    assert (rows[0]["prompt_tokens"], rows[0]["completion_tokens"],
            rows[0]["cache_read_tokens"]) == (120, 30, 1000)
    assert (rows[1]["prompt_tokens"], rows[1]["completion_tokens"]) == (200, 45)
    assert rows[0]["tag"] == "repo-x" and rows[0]["prompt_preview"] == "hello there"
    # API key forwarded to upstream but never stored.
    sent = {k.lower(): v for k, v in upstream.requests[0][1].items()}
    assert "sk-test" in sent.get("x-api-key", "") + sent.get("authorization", "")
    assert "sk-test" not in json.dumps(rows)


def test_openai_json_and_stream_without_usage(upstream, db_path):
    store = TokenStore(db_path)
    with MeteringProxy(TokenTracker(store), upstreams=ups(upstream)) as proxy:
        base = proxy.client_env()["OPENAI_BASE_URL"]
        msgs = [{"role": "user", "content": "hi"}]
        assert post(f"{base}/chat/completions", {"model": "gpt-4o-mini", "messages": msgs})[0] == 200
        status, body = post(f"{base}/chat/completions",
                            {"model": "gpt-4o-mini", "messages": msgs, "stream": True})
        assert status == 200 and b"one two three" in body
    rows = sorted(store.calls(), key=lambda r: r["id"])
    assert (rows[0]["prompt_tokens"], rows[0]["completion_tokens"]) == (50, 10)
    assert rows[1]["completion_tokens"] == 3  # estimated from streamed text


def test_unknown_prefix_and_upstream_errors_not_recorded(upstream, db_path):
    store = TokenStore(db_path)
    with MeteringProxy(TokenTracker(store), upstreams=ups(upstream)) as proxy:
        assert post(f"http://127.0.0.1:{proxy.port}/nope/v1/x", {})[0] == 404
        assert post(proxy.base_url("anthropic") + "/v1/unknown", {})[0] == 404
    assert store.calls() == []


def test_budget_enforcement_blocks_calls(upstream, db_path):
    store = TokenStore(db_path)
    budget = Budget(BudgetLimits(task=Limit(tokens=1500)), store=store, task_id="t")
    tracker = TokenTracker(budget=budget)
    with MeteringProxy(tracker, upstreams=ups(upstream), enforce_budget=True) as proxy:
        url = proxy.base_url("anthropic") + "/v1/messages"
        body = {"model": "claude-opus-5-5", "max_tokens": 5,
                "messages": [{"role": "user", "content": "x"}]}
        assert post(url, body)[0] == 200   # 1150 tokens -> 77% WARN
        assert post(url, body)[0] == 200   # 2300 -> STOP
        status, payload = post(url, body)
        assert status == 429 and b"budget exhausted" in payload
        assert proxy.blocked == 1
    assert len(store.calls()) == 2


SCRIPT = """
import json, os, urllib.request
base = os.environ["ANTHROPIC_BASE_URL"]
for i in range(3):
    req = urllib.request.Request(base + "/v1/messages", method="POST",
        data=json.dumps({"model": "claude-haiku-5-5", "max_tokens": 5,
                         "messages": [{"role": "user", "content": f"q{i}"}]}).encode(),
        headers={"Content-Type": "application/json", "x-api-key": "k"})
    urllib.request.urlopen(req).read()
print("run id", os.environ["TOKEN_EVAL_RUN_ID"])
"""


def test_run_repo_meters_subprocess(upstream, db_path, tmp_path):
    repo = tmp_path / "myrepo"
    repo.mkdir()
    (repo / "main.py").write_text(SCRIPT)
    store = TokenStore(db_path)
    res = run_repo(repo, [sys.executable, "main.py"], store=store, tag="myrepo",
                   upstreams=ups(upstream))
    assert res.exit_code == 0 and res.calls == 3
    assert res.prompt_tokens == 360 and res.completion_tokens == 90
    assert res.total_tokens == 360 + 90 + 3000  # incl. cache reads
    assert res.cost_usd > 0 and res.by_model[0]["key"] == "claude-opus-5-5"
    assert {r["session_id"] for r in store.calls()} == {res.run_id}


def test_run_repo_failing_setup_skips_command(upstream, db_path, tmp_path):
    res = run_repo(tmp_path, [sys.executable, "-c", "print(1)"], store=TokenStore(db_path),
                   tag="t", setup="exit 3", upstreams=ups(upstream))
    assert res.exit_code == 3 and res.calls == 0


def test_repo_helpers(tmp_path):
    assert normalize_url("anthropics/anthropic-sdk-python") == \
        "https://github.com/anthropics/anthropic-sdk-python.git"
    assert repo_name("https://github.com/owner/proj.git") == "owner__proj"
    assert repo_name("git@github.com:owner/proj.git") == "owner__proj"
    assert resolve_repo(str(tmp_path)) == tmp_path.resolve()


def test_resolve_repo_clones_with_git(tmp_path):
    import subprocess

    src = tmp_path / "origin"
    src.mkdir()
    (src / "a.py").write_text("print(1)\n")
    git = ["git", "-c", "user.email=t@t", "-c", "user.name=t"]
    subprocess.run(["git", "init", "-q"], cwd=src, check=True)
    subprocess.run(git + ["add", "."], cwd=src, check=True)
    subprocess.run(git + ["commit", "-qm", "init"], cwd=src, check=True)
    dest = resolve_repo(src.as_uri(), cache_dir=tmp_path / "cache")
    assert (dest / "a.py").exists() and dest.parent == (tmp_path / "cache").resolve()
    assert resolve_repo(src.as_uri(), cache_dir=tmp_path / "cache") == dest  # cached


def test_env_file_values_reach_command_and_set_upstream(upstream, db_path, tmp_path):
    from token_eval.runner import load_env_file

    envf = tmp_path / ".env"
    envf.write_text(f"SECRET_X=abc   # inline comment\nANTHROPIC_BASE_URL={upstream.url}\n")
    (tmp_path / "main.py").write_text(SCRIPT + "\nassert os.environ['SECRET_X'] == 'abc'\n")
    res = run_repo(tmp_path, [sys.executable, "main.py"], store=TokenStore(db_path), tag="t",
                   extra_env=load_env_file(envf))
    assert res.exit_code == 0 and res.calls == 3  # upstream taken from the env file


def test_rejected_calls_are_counted_not_recorded(db_path):
    import json as _json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Reject(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            data = _json.dumps({"error": {"message": "Your API key has expired.",
                                          "code": "expired_secret_key"}}).encode()
            self.send_response(401)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Reject)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        store = TokenStore(db_path)
        url = f"http://127.0.0.1:{srv.server_address[1]}"
        with MeteringProxy(TokenTracker(store), upstreams={"openai": url}) as proxy:
            status, _ = post(proxy.client_env()["OPENAI_BASE_URL"] + "/chat/completions",
                             {"model": "gpt-4o", "messages": [], "stream": True})
            assert status == 401
            assert proxy.failed == {401: 1}
            assert proxy.last_failure == "Your API key has expired. (expired_secret_key)"
        assert store.calls() == []
    finally:
        srv.shutdown()
        srv.server_close()
