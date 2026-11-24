import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def tool_call(name: str, arguments: dict, call_id: str) -> dict:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}


class FakeOpenAI:
    """OpenAI-compatible /chat/completions server replaying scripted replies, for end-to-end CLI tests."""

    def __init__(self, replies: list[tuple[str, list[dict]]], hold_requests: set[int] = frozenset()):
        self.replies, self.requests, self.hold_requests = list(replies), [], set(hold_requests)
        self.request_arrived, self.release = threading.Condition(), threading.Event()
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                with fake.request_arrived:
                    index = len(fake.requests)
                    fake.requests.append(body)
                    fake.request_arrived.notify_all()
                if index in fake.hold_requests:
                    fake.release.wait(30)
                    return
                content, calls = fake.replies.pop(0) if fake.replies else ("done", [])
                message = {"role": "assistant", "content": content or None, **({"tool_calls": calls} if calls else {})}
                payload = {"id": f"chatcmpl-{index}", "object": "chat.completion", "created": 0, "model": body["model"],
                           "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if calls else "stop"}],
                           "usage": {"prompt_tokens": len(json.dumps(body)) // 4, "completion_tokens": 5,
                                     "total_tokens": len(json.dumps(body)) // 4 + 5,
                                     "prompt_tokens_details": {"cached_tokens": 0}}}
                data = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *_):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}/v1"

    def wait_for_requests(self, count: int, timeout: float = 30) -> None:
        with self.request_arrived:
            assert self.request_arrived.wait_for(lambda: len(self.requests) >= count, timeout), "requests did not arrive"

    def close(self) -> None:
        self.release.set()
        self.server.shutdown()
