import hashlib
import subprocess
import sys
from pathlib import Path

from tests.fake_openai import FakeOpenAI, tool_call

ROOT = Path(__file__).resolve().parent.parent
ORIGINAL = ROOT / "baseline" / "original_agent.py"
PROVIDED_SHA256 = "d1e005f625b5212bab74fb0b273839d1f9a2129fa4bdc56071385583a905eb1c"


def test_baseline_is_the_provided_file_byte_for_byte():
    assert hashlib.sha256(ORIGINAL.read_bytes()).hexdigest() == PROVIDED_SHA256
    first_commit = subprocess.run(["git", "show", "8b5e4ae:agent.py"], cwd=ROOT, capture_output=True, text=True).stdout
    changed = [(a, b) for a, b in zip(first_commit.splitlines(), ORIGINAL.read_text().splitlines()) if a != b]
    assert [b for _, b in changed] == ['import json, sys, subprocess', 'LLM_BASE_URL = "http://ip:port/v1"',
                                       'LLM_API_KEY = "..."', 'LLM_MODEL = "..."']


def test_original_mode_runs_the_provided_file_unchanged(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT / "bench"))
    import run as bench_run
    server = FakeOpenAI([("", [tool_call("bash", {"command": "./tickets next"}, "call_1")]), ("stopping here", [])])
    monkeypatch.setenv("LLM_BASE_URL", server.url)
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "fake-model")
    monkeypatch.setattr(bench_run, "RUNS_DIR", tmp_path / "runs")
    try:
        result = bench_run.run_once("route-test", 0, "original", 100_000)
    finally:
        server.close()
    first = server.requests[0]
    assert first["model"] == "fake-model" and first["temperature"] == 0.1 and first["max_tokens"] == 4096
    assert [t["function"]["name"] for t in first["tools"]] == ["bash"] and first["tool_choice"] == "auto"
    assert "TICKET 1/11" in [m for m in server.requests[1]["messages"] if m["role"] == "tool"][0]["content"]
    assert result["mode"] == "original" and result["stop_reason"] == "finished" and result["turns"] == 2
    assert result["tool_calls"] == 1 and result["input_tokens"] > 0 and result["tickets_closed"] == 0
    assert result["success"] is False and result["code_revision"]
    assert hashlib.sha256(ORIGINAL.read_bytes()).hexdigest() == PROVIDED_SHA256
