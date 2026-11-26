"""A tiny fake Ollama HTTP server for trying ARGUS scripts without a model.

Developer tool, not needed for normal use. It answers /api/tags and /api/chat the way Ollama does,
using the keyword rules in tests/fake_rules.py, so the REAL `ollama` Python client and every ARGUS
script can be exercised end to end:

    python -m tests.fake_ollama_server          # listens on localhost:11434 (stop the real Ollama first)
"""
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

from .fake_rules import fake_reply

MODEL = "qwen2.5:7b"


class Handler(BaseHTTPRequestHandler):
    def _send(self, obj: dict, code: int = 200) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/api/tags"):
            self._send({"models": [{"name": MODEL, "model": MODEL}]})
        else:
            self._send({"error": "not found"}, 404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(length) or b"{}")
        if not self.path.startswith("/api/chat"):
            return self._send({"error": "not found"}, 404)
        prompt = req["messages"][-1]["content"]
        keys = set((req.get("format") or {}).get("properties", {}))
        try:
            content = json.dumps(fake_reply(prompt, keys))
        except ValueError as e:
            return self._send({"error": str(e)}, 400)
        self._send({
            "model": req.get("model", MODEL), "created_at": "2026-01-01T00:00:00Z",
            "message": {"role": "assistant", "content": content}, "done": True, "done_reason": "stop",
        })

    def log_message(self, *args):  # keep the console quiet
        pass


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 11434
    print(f"Fake Ollama on http://localhost:{port} (Ctrl+C to stop)")
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()
