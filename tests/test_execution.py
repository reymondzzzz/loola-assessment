import json

import httpx
import openai
import pytest

from harness import executor as executor_module
from harness.config import HarnessConfig
from harness.executor import ToolExecutor, run_bash
from harness.protocol import protocol_errors
from harness.llm import LLMClient
from harness.metrics import MetricsCollector
from tests.fakes import call, run_script


def tool_messages(request):
    return [m for m in request["messages"] if m["role"] == "tool"]


def payload(message):
    return json.loads(message["content"])


def test_tool_call_protocol_integrity():
    _, llm = run_script([("two calls", [call("bash", {"command": "echo a"}, "c1"), call("bash", {"command": "echo b"}, "c2")]),
                         ("done", [])])
    messages = llm.requests[1]["messages"]
    assistants = [m for m in messages if m["role"] == "assistant"]
    assert len(assistants) == 1 and [c["id"] for c in assistants[0]["tool_calls"]] == ["c1", "c2"]
    assert [m["tool_call_id"] for m in tool_messages(llm.requests[1])] == ["c1", "c2"]
    assert protocol_errors(messages[2:]) == []
    assert protocol_errors([{"role": "tool", "tool_call_id": "x", "content": ""}]) == ["orphan tool result at 0"]


def test_error_recovery_returns_repairable_errors_to_model():
    metrics, llm = run_script([("", [call("bash", "{not json", "c1")]),
                               ("", [call("nope", {}, "c2")]),
                               ("", [call("invoke", {"name": "read_file", "arguments": {}}, "c3")]),
                               ("", [call("invoke", {"name": "read_file", "arguments": {"path": "/no/such"}}, "c4")]),
                               ("fixed", [])])
    codes = [payload(m)["code"] for m in tool_messages(llm.requests[-1])]
    assert codes == ["INVALID_JSON", "UNKNOWN_TOOL", "INVALID_ARGUMENTS", "FILE_NOT_FOUND"]
    assert all(payload(m)["model_can_repair"] for m in tool_messages(llm.requests[-1]))
    assert metrics["stop_reason"] == "finished"


def test_permission_denied_is_not_repairable():
    outcome = ToolExecutor(available={"invoke", "bash"}).execute(
        "invoke", json.dumps({"name": "send_email", "arguments": {"to": ["a@b.c"], "subject": "s", "body": "b"}}))
    assert outcome.code == "PERMISSION_DENIED" and not outcome.model_can_repair
    assert run_bash("sudo rm -rf /tmp/x").code == "PERMISSION_DENIED"
    assert run_bash("git push origin main").code == "PERMISSION_DENIED"
    assert run_bash("echo sudoku").code == "OK"


def test_loop_detection():
    executor = ToolExecutor(available={"bash"}, loop_limit=3)
    codes = [executor.execute("invoke_missing", "{}").code for _ in range(2)]
    assert codes == ["UNKNOWN_TOOL", "UNKNOWN_TOOL"]
    executor = ToolExecutor(available={"describe"}, loop_limit=3)
    failing = json.dumps({"bogus": 1})
    assert [executor.execute("describe", failing).code for _ in range(4)] == \
        ["INVALID_ARGUMENTS", "INVALID_ARGUMENTS", "INVALID_ARGUMENTS", "LOOP_DETECTED"]


def test_loop_counter_resets_after_environment_change(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    executor = ToolExecutor(available={"bash"}, loop_limit=2)
    failing = json.dumps({"command": "cat missing.txt"})
    assert [executor.execute("bash", failing).code for _ in range(2)] == ["NONZERO_EXIT", "NONZERO_EXIT"]
    assert executor.execute("bash", failing).code == "NONZERO_EXIT"
    executor.environment_version, executor.failures = 0, {}
    read = json.dumps({"name": "read_file", "arguments": {"path": "missing.txt"}})
    executor.available = {"invoke"}
    assert [executor.execute("invoke", read).code for _ in range(3)] == ["FILE_NOT_FOUND", "FILE_NOT_FOUND", "LOOP_DETECTED"]


def test_non_idempotent_retry_protection(monkeypatch):
    attempts = {"read": 0, "edit": 0}

    def flaky(kind):
        def handler(**_):
            attempts[kind] += 1
            raise ConnectionError("transient")
        return handler

    from dataclasses import replace
    from harness.catalog import CATALOG
    monkeypatch.setattr(executor_module, "TRANSIENT_RETRIES", 2)
    monkeypatch.setattr(executor_module.time, "sleep", lambda _: None)
    read = replace(CATALOG["read_file"], handler=flaky("read"))
    edit = replace(CATALOG["edit_file"], handler=flaky("edit"))
    assert executor_module.run_catalog_tool(read, {"path": "x"}).code == "TOOL_ERROR"
    assert executor_module.run_catalog_tool(edit, {"path": "x", "old": "a", "new": "b"}).code == "TOOL_ERROR"
    assert attempts == {"read": 3, "edit": 1}


def test_bash_timeout_reports_possible_side_effects(tmp_path):
    outcome = run_bash(f"touch {tmp_path}/started; sleep 5", timeout=0.5)
    body = json.loads(outcome.content)
    assert body["code"] == "TIMEOUT" and body["retry_same_call"] is False
    assert "partially run" in body["message"] and (tmp_path / "started").exists()
    assert outcome.changed_environment


def test_llm_retries_transient_errors_then_gives_up(monkeypatch):
    monkeypatch.setattr("harness.llm.time.sleep", lambda _: None)
    client = LLMClient(MetricsCollector(), retries=2, deadline_seconds=60)
    error = openai.APIConnectionError(request=httpx.Request("POST", "http://x"))
    calls = {"n": 0}

    class FlakyModel:
        def bind_tools(self, *_, **__):
            return self

        def invoke(self, _):
            calls["n"] += 1
            if calls["n"] < 3:
                raise error
            from langchain_core.messages import AIMessage
            return AIMessage(content="ok", usage_metadata={"input_tokens": 5, "output_tokens": 1, "total_tokens": 6})

    client.model = FlakyModel()
    assert client.complete([], [])[0] == "ok"
    assert client.metrics.counters["llm_retries"] == 2
    calls["n"] = -10
    with pytest.raises(openai.APIConnectionError):
        client.complete([], [])


def test_budget_exhaustion_stops_run():
    replies = [("", [call("bash", {"command": "true"}, f"c{i}")]) for i in range(10)]
    metrics, llm = run_script(replies, HarnessConfig(max_turns=3))
    assert metrics["stop_reason"] == "max_turns" and len(llm.requests) == 3
