"""A local metering proxy for LLM APIs.

Clients are pointed at ``http://127.0.0.1:<port>/<provider>`` (via the SDKs'
standard base-URL environment variables). Each request is forwarded unchanged
to the real upstream API, the response is streamed back unchanged, and token
usage is parsed from the response (JSON or SSE stream) and recorded through a
:class:`TokenTracker`.

API keys pass through in request headers and are never stored or logged.
"""

from __future__ import annotations

import http.client
import json
import threading
from collections import Counter
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

from token_eval.counting import count_tokens
from token_eval.tracker import TokenTracker, Usage, extract_usage

DEFAULT_UPSTREAMS = {
    "anthropic": "https://api.anthropic.com",
    "openai": "https://api.openai.com",
}
HOP_BY_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te",
              "trailers", "transfer-encoding", "upgrade", "host", "accept-encoding",
              "content-length"}
# Endpoints that consume tokens. Everything else (count_tokens, models, ...) is just forwarded.
METERED_SUFFIXES = ("/messages", "/chat/completions", "/completions", "/responses", "/embeddings")


def is_metered(path: str) -> bool:
    return path.split("?")[0].rstrip("/").endswith(METERED_SUFFIXES)


@dataclass
class StreamAccumulator:
    """Collects usage from Anthropic / OpenAI server-sent-event streams."""
    model: str | None = None
    usage: dict[str, int] = field(default_factory=dict)
    text: list[str] = field(default_factory=list)
    _buf: bytes = b""

    def feed(self, data: bytes) -> None:
        self._buf += data
        while b"\n" in self._buf:
            line, self._buf = self._buf.split(b"\n", 1)
            line = line.strip()
            if line.startswith(b"data:"):
                self._event(line[5:].strip())

    def _event(self, raw: bytes) -> None:
        if not raw or raw == b"[DONE]":
            return
        try:
            ev = json.loads(raw)
        except json.JSONDecodeError:
            return
        t = ev.get("type")
        if t == "message_start":  # Anthropic
            msg = ev.get("message", {})
            self.model = msg.get("model", self.model)
            self._merge(msg.get("usage"))
        elif t == "message_delta":  # Anthropic: cumulative usage at end of stream
            self._merge(ev.get("usage"))
        elif t == "content_block_delta":
            self.text.append(ev.get("delta", {}).get("text", "") or "")
        elif t == "response.completed":  # OpenAI Responses API
            resp = ev.get("response", {})
            self.model = resp.get("model", self.model)
            self._merge_any(resp.get("usage"))
        elif "choices" in ev or "usage" in ev:  # OpenAI chat completions chunk
            self.model = ev.get("model", self.model)
            if ev.get("usage"):
                self._merge_any(ev["usage"])
            for ch in ev.get("choices") or []:
                self.text.append((ch.get("delta") or {}).get("content") or "")

    def _merge(self, usage: dict | None) -> None:
        for k, v in (usage or {}).items():
            if isinstance(v, int):
                self.usage[k] = max(v, self.usage.get(k, 0))

    def _merge_any(self, usage: dict | None) -> None:
        if usage:
            self.usage = {**self.usage, **{k: v for k, v in usage.items()}}

    def result(self) -> Usage | None:
        return extract_usage({"usage": self.usage}) if self.usage else None


def _request_text(body: dict) -> str:
    parts: list[str] = []
    system = body.get("system")
    if system:
        parts.append(system if isinstance(system, str) else json.dumps(system))
    for m in body.get("messages") or []:
        c = m.get("content")
        parts.append(c if isinstance(c, str) else json.dumps(c))
    for key in ("input", "prompt"):
        v = body.get(key)
        if v:
            parts.append(v if isinstance(v, str) else json.dumps(v))
    return "\n".join(parts)


def _error_message(body: bytes) -> str | None:
    try:
        err = json.loads(body).get("error")
    except (json.JSONDecodeError, UnicodeDecodeError, AttributeError):
        return None
    if isinstance(err, dict):
        code = err.get("code") or err.get("type")
        return f"{err.get('message', '')} ({code})" if code else err.get("message")
    return str(err) if err else None


def _preview(body: dict) -> str:
    msgs = body.get("messages") or []
    for m in reversed(msgs):
        if m.get("role") == "user":
            c = m.get("content")
            return c if isinstance(c, str) else json.dumps(c)
    return _request_text(body)


class MeteringProxy:
    """Run with ``with MeteringProxy(tracker) as proxy: ... proxy.base_url('anthropic')``."""

    def __init__(
        self,
        tracker: TokenTracker,
        upstreams: dict[str, str] | None = None,
        tag: str | None = None,
        enforce_budget: bool = False,
        host: str = "127.0.0.1",
        port: int = 0,
        timeout: float = 600,
    ) -> None:
        self.tracker = tracker
        self.upstreams = {**DEFAULT_UPSTREAMS, **(upstreams or {})}
        self.tag = tag
        self.enforce_budget = enforce_budget
        self.timeout = timeout
        self.calls = 0
        self.blocked = 0
        self.failed: Counter[int] = Counter()   # metered requests the upstream rejected
        self.last_failure: str | None = None    # provider's error message (never headers/keys)
        self.errors: list[str] = []
        self._lock = threading.Lock()
        self._server = ThreadingHTTPServer((host, port), self._handler_class())
        self._server.daemon_threads = True
        self._thread: threading.Thread | None = None

    # --- lifecycle --------------------------------------------------------
    @property
    def port(self) -> int:
        return self._server.server_address[1]

    def base_url(self, provider: str) -> str:
        return f"http://127.0.0.1:{self.port}/{provider}"

    def start(self) -> "MeteringProxy":
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    def __enter__(self) -> "MeteringProxy":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    def client_env(self) -> dict[str, str]:
        """Environment variables that route the common SDKs through this proxy."""
        a, o = self.base_url("anthropic"), self.base_url("openai") + "/v1"
        return {
            "ANTHROPIC_BASE_URL": a,
            "ANTHROPIC_API_URL": a,     # LangChain
            "OPENAI_BASE_URL": o,
            "OPENAI_API_BASE": o,       # legacy openai<1 / LangChain / LiteLLM
        }

    # --- recording --------------------------------------------------------
    def _record(self, path: str, req: dict, status: int, ctype: str, body: bytes,
                stream: StreamAccumulator | None) -> None:
        if not is_metered(path):
            return
        if status >= 400:
            with self._lock:
                self.failed[status] += 1
                self.last_failure = _error_message(body) or f"HTTP {status}"
            return
        model = req.get("model") or "unknown"
        usage: Usage | None = None
        if stream is not None:
            model = stream.model or model
            usage = stream.result()
            if usage is None:  # e.g. OpenAI stream without stream_options.include_usage
                usage = Usage(count_tokens(_request_text(req), model),
                              count_tokens("".join(stream.text), model))
        else:
            try:
                data = json.loads(body)
            except (json.JSONDecodeError, UnicodeDecodeError):
                return
            model = data.get("model") or model
            usage = extract_usage(data)
        if usage is None:
            return
        with self._lock:
            self.tracker.record(
                model, usage.prompt_tokens, usage.completion_tokens, tag=self.tag,
                cache_write_tokens=usage.cache_write_tokens,
                cache_read_tokens=usage.cache_read_tokens,
                prompt_preview=_preview(req),
            )
            self.calls += 1

    def _budget_blocks(self) -> bool:
        b = self.tracker.budget
        return bool(self.enforce_budget and b is not None and not b.should_continue())

    # --- HTTP handler -----------------------------------------------------
    def _handler_class(self) -> type[BaseHTTPRequestHandler]:
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"  # close after each response; no re-chunking

            def log_message(self, *args: Any) -> None:  # keep the user's terminal clean
                pass

            def do_GET(self) -> None:
                self._forward()

            def do_POST(self) -> None:
                self._forward()

            def do_DELETE(self) -> None:
                self._forward()

            def _send_error(self, status: int, message: str) -> None:
                payload = json.dumps({"type": "error", "error": {
                    "type": "token_eval_proxy_error", "message": message}}).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def _forward(self) -> None:
                provider, _, rest = self.path.lstrip("/").partition("/")
                upstream = proxy.upstreams.get(provider)
                if upstream is None:
                    self._send_error(404, f"unknown provider prefix '/{provider}'")
                    return
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length) if length else b""
                try:
                    req = json.loads(body) if body else {}
                except json.JSONDecodeError:
                    req = {}
                if is_metered("/" + rest) and proxy._budget_blocks():
                    proxy.blocked += 1
                    self._send_error(429, "token-eval budget exhausted (STOP); request blocked")
                    return

                u = urlsplit(upstream)
                conn_cls = (http.client.HTTPSConnection if u.scheme == "https"
                            else http.client.HTTPConnection)
                conn = conn_cls(u.netloc, timeout=proxy.timeout)
                target = (u.path.rstrip("/") + "/" + rest) or "/"
                headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP_BY_HOP}
                headers["Host"] = u.netloc
                headers["Accept-Encoding"] = "identity"  # so we can parse usage
                if body:
                    headers["Content-Length"] = str(len(body))
                try:
                    conn.request(self.command, target, body=body or None, headers=headers)
                    resp = conn.getresponse()
                except OSError as e:
                    proxy.errors.append(str(e))
                    self._send_error(502, f"upstream error: {e}")
                    return

                ctype = resp.getheader("Content-Type", "")
                streaming = "text/event-stream" in ctype
                self.send_response(resp.status, resp.reason)
                for k, v in resp.getheaders():
                    if k.lower() not in HOP_BY_HOP:
                        self.send_header(k, v)
                self.end_headers()

                acc = StreamAccumulator() if streaming else None
                chunks: list[bytes] = []
                try:
                    while True:
                        data = resp.read1(65536)
                        if not data:
                            break
                        if acc is not None:
                            acc.feed(data)
                        else:
                            chunks.append(data)
                        self.wfile.write(data)
                        self.wfile.flush()
                except OSError as e:  # client disconnected mid-stream
                    proxy.errors.append(str(e))
                finally:
                    conn.close()
                try:
                    proxy._record("/" + rest, req, resp.status, ctype, b"".join(chunks), acc)
                except Exception as e:  # never break the client because of metering
                    proxy.errors.append(f"record failed: {e}")

        return Handler
