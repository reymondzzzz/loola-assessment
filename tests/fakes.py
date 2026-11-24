import json

from harness.config import HarnessConfig
from harness.graph import run_agent
from harness.metrics import MetricsCollector


def call(name: str, arguments, call_id: str) -> dict:
    raw = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": raw}}


class ScriptedLLM:
    def __init__(self, replies: list[tuple[str, list[dict]]], metrics: MetricsCollector | None = None):
        self.replies, self.requests = iter(replies), []
        self.metrics = metrics or MetricsCollector()

    def complete(self, messages, tools, purpose="agent"):
        self.requests.append({"messages": json.loads(json.dumps(messages)), "tools": tools})
        content, tool_calls = next(self.replies, ("done", []))
        usage = {"input": len(json.dumps(messages)) // 4, "output": 10, "cached": 0}
        self.metrics.record_llm_call(usage, purpose)
        return content, tool_calls, usage


def run_script(replies, config: HarnessConfig = HarnessConfig(), task: str = "task"):
    llm = ScriptedLLM(replies)
    metrics = run_agent(task, config, llm)
    return metrics, llm
