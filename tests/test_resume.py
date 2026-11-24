import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests.fake_openai import FakeOpenAI, tool_call

AGENT = str(Path(__file__).resolve().parent.parent / "agent.py")


@pytest.fixture
def cli(tmp_path):
    workdir, runs = tmp_path / "work", tmp_path / "runs"
    workdir.mkdir()
    servers = []

    def start(server: FakeOpenAI, *args: str) -> subprocess.Popen:
        servers.append(server)
        env = {**os.environ, "LLM_BASE_URL": server.url, "LLM_API_KEY": "test", "LLM_MODEL": "fake-model",
               "AGENT_RUNS_DIR": str(runs), "PYTHONUNBUFFERED": "1", "AGENT_MODE": os.environ.get("AGENT_MODE", "lazy")}
        return subprocess.Popen([sys.executable, AGENT, *args], cwd=workdir, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, start_new_session=True)

    yield start, workdir, runs
    for server in servers:
        server.close()
    subprocess.run(["pkill", "-f", "sleep 61.5"], check=False)


def run_id_of(process: subprocess.Popen) -> str:
    first_line = process.stdout.readline()
    return re.search(r"Run (\S+)", first_line).group(1)


def kill(process: subprocess.Popen) -> None:
    os.killpg(process.pid, signal.SIGKILL)
    process.wait()


def events(runs: Path, run_id: str) -> list[dict]:
    return [json.loads(line) for line in (runs / run_id / "events.jsonl").read_text().splitlines()]


def test_resume_after_crash_does_not_repeat_uncertain_side_effect(cli):
    start, workdir, runs = cli
    server = FakeOpenAI([("", [tool_call("bash", {"command": "echo once >> side.log; sleep 61.5"}, "c1")]),
                         ("finished after resume", [])])
    first = start(server, "append a line")
    run_id = run_id_of(first)
    deadline = time.monotonic() + 20
    while not (workdir / "side.log").exists() and time.monotonic() < deadline:
        time.sleep(0.1)
    kill(first)

    resumed = start(server, "--resume", run_id)
    output, _ = resumed.communicate(timeout=60)
    assert resumed.returncode == 0, output
    assert "Resuming run" in output and "✅ Agent finished" in output
    assert (workdir / "side.log").read_text() == "once\n"
    tool_results = [m for m in server.requests[1]["messages"] if m["role"] == "tool"]
    assert json.loads(tool_results[0]["content"])["code"] == "OUTCOME_UNKNOWN"
    log = events(runs, run_id)
    assert [e["kind"] for e in log].count("task") == 1 and any(e["kind"] == "resume" for e in log)
    assert {"checkpoints.sqlite", "config.json", "events.jsonl"} <= {p.name for p in (runs / run_id).iterdir()}


def test_resume_after_crash_reuses_finished_tool_result(cli):
    start, workdir, runs = cli
    server = FakeOpenAI([("", [tool_call("bash", {"command": "echo once >> side.log"}, "c1")]),
                         ("finished after resume", [])], hold_requests={1})
    first = start(server, "append a line")
    run_id = run_id_of(first)
    server.wait_for_requests(2)
    kill(first)
    server.release.set()

    resumed = start(server, "--resume", run_id)
    output, _ = resumed.communicate(timeout=60)
    assert resumed.returncode == 0, output
    assert (workdir / "side.log").read_text() == "once\n"
    final_request = server.requests[-1]["messages"]
    assert [m["tool_call_id"] for m in final_request if m["role"] == "tool"] == ["c1"]
    assert sum(e["kind"] == "tool_started" for e in events(runs, run_id)) == 1
    metrics = json.loads(output.split("📊 ", 1)[1].splitlines()[0])
    assert metrics["run_id"] == run_id and metrics["tool_calls"] == 1 and metrics["stop_reason"] == "finished"


def test_resume_unknown_run_fails_cleanly(cli):
    start, _, _ = cli
    process = start(FakeOpenAI([]), "--resume", "20990101-000000-abcdef")
    output, _ = process.communicate(timeout=30)
    assert process.returncode != 0 and "no run" in output


def test_only_read_only_idempotent_calls_are_repeated_after_a_crash():
    from harness.executor import repeat_is_safe
    assert repeat_is_safe("invoke", {"name": "read_file", "arguments": {"path": "a"}})
    assert not repeat_is_safe("invoke", {"name": "edit_file", "arguments": {}})
    assert not repeat_is_safe("invoke", {"name": "write_file", "arguments": {}})
    assert not repeat_is_safe("invoke", {"name": 5})
    assert not repeat_is_safe("bash", {"command": "ls"})
    assert repeat_is_safe("describe", {"name": "git_log"}) and repeat_is_safe("search_history", {"query": "x"})
    assert not repeat_is_safe("edit_file", {}) and repeat_is_safe("git_log", {})
