"""A tiny fake Anthropic/OpenAI API server for tests."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ANTHROPIC_MESSAGE = {
    "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
    "content": [{"type": "text", "text": "Hello!"}], "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 120, "output_tokens": 30, "cache_creation_input_tokens": 0,
              "cache_read_input_tokens": 1000},
}


def sse(events: list[dict]) -> bytes:
    out = b""
    for ev in events:
        out += f"event: {ev.get('type', 'message')}\ndata: {json.dumps(ev)}\n\n".encode()
    return out


ANTHROPIC_STREAM = sse([
    {"type": "message_start", "message": {**ANTHROPIC_MESSAGE, "content": [],
                                          "stop_reason": None,
                                          "usage": {"input_tokens": 200, "output_tokens": 1}}},
    {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hi"}},
    {"type": "content_block_stop", "index": 0},
    {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
     "usage": {"output_tokens": 45}},
    {"type": "message_stop"},
])

OPENAI_CHAT = {
    "id": "c1", "object": "chat.completion", "model": "gpt-4o-mini",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                 "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60},
}
OPENAI_STREAM_NO_USAGE = (
    b'data: {"model": "gpt-4o-mini", "choices": [{"delta": {"content": "one two three"}}]}\n\n'
    b"data: [DONE]\n\n"
)


class FakeUpstream:
    def __init__(self) -> None:
        self.requests: list[tuple[str, dict, dict]] = []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"{}")
                outer.requests.append((self.path, dict(self.headers), body))
                if self.path.endswith("/count_tokens"):
                    return self._json({"input_tokens": 7})
                if self.path == "/v1/messages":
                    if body.get("stream"):
                        return self._raw(ANTHROPIC_STREAM, "text/event-stream")
                    return self._json(ANTHROPIC_MESSAGE)
                if self.path == "/v1/chat/completions":
                    if body.get("stream"):
                        return self._raw(OPENAI_STREAM_NO_USAGE, "text/event-stream")
                    return self._json(OPENAI_CHAT)
                self._json({"error": "not found"}, 404)

            def _json(self, obj, status=200):
                self._raw(json.dumps(obj).encode(), "application/json", status)

            def _raw(self, data: bytes, ctype: str, status=200):
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def __enter__(self):
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()
