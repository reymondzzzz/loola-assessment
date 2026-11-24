import json
from pathlib import Path

from tests.fake_openai import FakeOpenAI, tool_call
from tests.fakes import call, run_script
from tests.test_resume import cli, run_id_of  # noqa: F401

TASK = "Write a greeting to out.txt."
REQUIREMENT = "Requirement change: out.txt must contain HELLO in uppercase, nothing else."
SUMMARY = {"goal": "write a greeting", "verified_facts": [], "decisions": [], "completed_work": ["wrote hello"],
           "failed_attempts": [], "pending_work": ["follow the requirement update"], "important_paths": ["out.txt"],
           "artifact_refs": [], "active_skill_refs": []}


def test_requirement_added_on_resume_survives_compaction_and_changes_the_result(cli, monkeypatch):  # noqa: F811
    monkeypatch.setenv("AGENT_CONTEXT_WINDOW", "12000")
    monkeypatch.setenv("AGENT_MODE", "hybrid")
    start, workdir, runs = cli
    noisy = [("", [tool_call("bash", {"command": f"seq 1 400 | sed 's/^/check {i} /'"}, f"n{i}")]) for i in range(14)]
    server = FakeOpenAI([("", [tool_call("bash", {"command": "echo hello > out.txt"}, "c1")]), ("done", []),
                         *noisy,
                         ("", [tool_call("bash", {"command": "echo HELLO > out.txt"}, "c2")]), ("done again", [])],
                        summary=SUMMARY)
    first = start(server, TASK)
    run_id = run_id_of(first)
    assert first.wait(60) == 0 and (workdir / "out.txt").read_text() == "hello\n"
    first_requests = len(server.requests)

    resumed = start(server, "--resume", run_id, "--message", REQUIREMENT)
    output, _ = resumed.communicate(timeout=90)
    assert resumed.returncode == 0, output

    assert (workdir / "out.txt").read_text() == "HELLO\n"
    after = server.requests[first_requests:]
    assert after and all(r["messages"][1] == {"role": "user", "content": TASK} for r in after)
    assert all(r["messages"][2]["role"] == "user" and REQUIREMENT in r["messages"][2]["content"] for r in after)
    metrics = json.loads(output.split("📊 ", 1)[1].splitlines()[0])
    assert metrics["compactions"] >= 1 and metrics["stop_reason"] == "finished"
    if server.summary_requests:
        assert REQUIREMENT in server.summary_requests[0]["messages"][1]["content"]
    log = [json.loads(line) for line in (runs / run_id / "events.jsonl").read_text().splitlines()]
    assert [e["text"] for e in log if e["kind"] == "requirement_update"] == [REQUIREMENT]


def test_repeated_tool_calls_metric(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "x.txt").write_text("1\n2\n3\n")
    script = [("", [call("invoke", '{"name": "read_file", "arguments": {"path": "x.txt", "limit": 2}}', "c1")]),
              ("", [call("invoke", '{"arguments": {"limit": 2, "path": "x.txt"}, "name": "read_file"}', "c2")]),
              ("", [call("invoke", '{"name": "read_file", "arguments": {"path": "x.txt", "limit": 3}}', "c3")]),
              ("", [call("bash", {"command": "true"}, "c4")])]
    metrics, _ = run_script(script)
    assert metrics["repeated_tool_calls"] == 1 and metrics["tool_calls"] == 4
    metrics, _ = run_script(script[2:])
    assert metrics["repeated_tool_calls"] == 0
