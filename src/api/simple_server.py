"""Zero-dependency HTTP server (stdlib). Same routes as the FastAPI app.

    python -m src.api.simple_server --port 8000
"""
import argparse, json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from src.api import handlers


class Handler(BaseHTTPRequestHandler):
    def _send(self, status, payload):
        data = json.dumps(payload).encode()
        self.send_response(status); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)

    def _dispatch(self, method):
        fn = handlers.ROUTES.get((method, self.path.split("?")[0]))
        if fn is None:
            return self._send(404, {"error": "not found", "routes": [f"{m} {p}" for m, p in handlers.ROUTES]})
        try:
            body = {}
            if method == "POST":
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(n) or b"{}")
                if not isinstance(body, dict):
                    raise handlers.ApiError(422, "JSON body must be an object")
            self._send(200, fn(body))
        except handlers.ApiError as e:
            self._send(e.status, {"error": e.message})
        except json.JSONDecodeError:
            self._send(400, {"error": "invalid JSON"})

    def do_GET(self): self._dispatch("GET")
    def do_POST(self): self._dispatch("POST")
    def log_message(self, *a): pass


def make_server(port=8000, host="127.0.0.1"):
    return ThreadingHTTPServer((host, port), Handler)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8000); ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    print(f"Serving on http://{a.host}:{a.port}  (GET /health, POST /find-related-researchers, ...)")
    make_server(a.port, a.host).serve_forever()
