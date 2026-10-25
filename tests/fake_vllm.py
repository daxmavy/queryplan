"""A stand-in for `vllm serve` for tests: `fake_vllm.py serve MODEL --port P [--api-key K] ...`.

Answers /health, /v1/models and /v1/chat/completions like vLLM does, and like vLLM refuses /v1 requests without the
key when started with one. With FAKE_VLLM_DELAY set it waits that many seconds before binding its port."""

import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer


def main() -> None:
    args = sys.argv[1:]
    model = args[1]
    port = int(args[args.index("--port") + 1])
    key = args[args.index("--api-key") + 1] if "--api-key" in args else None
    if "--served-model-name" in args:
        model = args[args.index("--served-model-name") + 1]

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _authorised(self):
            if key is None or not self.path.startswith("/v1"):
                return True
            return self.headers.get("Authorization") == f"Bearer {key}"

        def do_GET(self):
            if not self._authorised():
                return self._send(401, {"error": "Unauthorized"})
            if self.path == "/health":
                return self._send(200, {})
            if self.path == "/v1/models":
                return self._send(200, {"data": [{"id": model}]})
            self._send(404, {})

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(length)
            if not self._authorised():
                return self._send(401, {"error": "Unauthorized"})
            self._send(200, {"choices": [{"message": {"content": f"{model} OK"}}],
                             "usage": {"prompt_tokens": 3, "completion_tokens": 2}})

    time.sleep(float(os.environ.get("FAKE_VLLM_DELAY", "0")))
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
