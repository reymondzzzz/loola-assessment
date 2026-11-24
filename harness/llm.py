import json, os, random, time

import openai
from langchain_core.exceptions import ContextOverflowError
from langchain_openai import ChatOpenAI

from harness.metrics import MetricsCollector

RETRYABLE = (openai.APIConnectionError, openai.RateLimitError, openai.InternalServerError)


class ContextOverflow(Exception):
    pass


def to_usage(usage_metadata: dict | None) -> dict:
    usage_metadata = usage_metadata or {}
    return {"input": usage_metadata.get("input_tokens", 0), "output": usage_metadata.get("output_tokens", 0),
            "cached": usage_metadata.get("input_token_details", {}).get("cache_read", 0)}


def openai_tool_call(tool_call_id: str, name: str, arguments: str) -> dict:
    return {"id": tool_call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


class LLMClient:
    def __init__(self, metrics: MetricsCollector, retries: int, deadline_seconds: float):
        self.model_name = os.environ["LLM_MODEL"]
        self.model = ChatOpenAI(model=self.model_name, base_url=os.environ["LLM_BASE_URL"],
                                api_key=os.environ["LLM_API_KEY"], max_tokens=4096, max_retries=0, timeout=180)
        self.metrics, self.retries, self.deadline_seconds = metrics, retries, deadline_seconds

    def complete(self, messages: list[dict], tools: list[dict], purpose: str = "agent") -> tuple[str, list[dict], dict]:
        model = self.model.bind_tools(tools, tool_choice="auto") if tools else self.model
        deadline = time.monotonic() + self.deadline_seconds
        for attempt in range(self.retries + 1):
            try:
                message = model.invoke(messages)
                break
            except ContextOverflowError as error:
                raise ContextOverflow(str(error)) from error
            except RETRYABLE:
                if attempt == self.retries or time.monotonic() > deadline:
                    raise
                self.metrics.count("llm_retries")
                time.sleep(min(30.0, 2.0 ** attempt) + random.uniform(0, 1))
        usage = to_usage(message.usage_metadata)
        self.metrics.record_llm_call(usage, purpose)
        tool_calls = [openai_tool_call(call["id"], call["name"], json.dumps(call["args"])) for call in message.tool_calls]
        tool_calls += [openai_tool_call(call["id"] or f"invalid_{index}", call["name"] or "", call["args"] or "")
                       for index, call in enumerate(message.invalid_tool_calls)]
        return message.text.strip(), tool_calls, usage
