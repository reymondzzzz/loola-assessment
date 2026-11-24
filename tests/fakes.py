import json

from harness.config import HarnessConfig
from harness.graph import run_agent
from harness.llm import ContextOverflow
from harness.metrics import MetricsCollector


def call(name: str, arguments, call_id: str) -> dict:
    raw = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": raw}}


class ScriptedLLM:
    def __init__(self, replies: list[tuple[str, list[dict]]], summary: str | Exception | None = None,
                 overflow_calls: set[int] = frozenset()):
        self.replies, self.requests, self.summary_requests = iter(replies), [], []
        self.summary, self.overflow_calls = summary, overflow_calls
        self.metrics = MetricsCollector()

    def complete(self, messages, tools, purpose="agent"):
        if purpose == "summarizer":
            self.summary_requests.append(messages)
            if isinstance(self.summary, Exception):
                raise self.summary
            usage = {"input": 100, "output": 50, "cached": 0}
            self.metrics.record_llm_call(usage, purpose)
            return self.summary or "{}", [], usage
        if len(self.requests) in self.overflow_calls:
            self.overflow_calls = self.overflow_calls - {len(self.requests)}
            raise ContextOverflow("maximum context length exceeded")
        self.requests.append({"messages": json.loads(json.dumps(messages)), "tools": tools})
        content, tool_calls = next(self.replies, ("done", []))
        usage = {"input": len(json.dumps(messages)) // 4, "output": 10, "cached": 0}
        self.metrics.record_llm_call(usage, purpose)
        return content, tool_calls, usage


def run_script(replies, config: HarnessConfig = HarnessConfig(), task: str = "task", **llm_options):
    llm = ScriptedLLM(replies, **llm_options)
    metrics = run_agent(task, config, llm)
    return metrics, llm
