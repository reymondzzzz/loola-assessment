import json

from harness.config import HarnessConfig
from tests.fake_openai import FakeOpenAI, tool_call
from tests.fakes import call, run_script
from tests.test_resume import cli, kill, run_id_of  # noqa: F401

VERIFY = "test -f fixed.txt && echo verified-ok"


def test_failed_verification_returns_to_agent_and_then_passes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    script = [("done", []), ("", [call("bash", {"command": "touch fixed.txt"}, "c1")]), ("done now", [])]
    metrics, llm = run_script(script, HarnessConfig(verify_command=VERIFY))
    assert metrics["stop_reason"] == "verified" and metrics["verification_runs"] == 2
    diagnostic = llm.requests[1]["messages"][-1]
    assert diagnostic["role"] == "user" and "Automatic verification failed" in diagnostic["content"]
    log = [json.loads(line) for line in open(metrics["events_path"])]
    assert [e["exit_code"] for e in log if e["kind"] == "verification"] == [1, 0]


def test_two_failed_verifications_stop_the_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    metrics, llm = run_script([("done", []), ("really done", [])], HarnessConfig(verify_command=VERIFY))
    assert metrics["stop_reason"] == "verification_failed"
    assert metrics["verification_failures"] == 2 and len(llm.requests) == 2


def test_runs_without_verify_keep_previous_behaviour(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    metrics, _ = run_script([("done", [])], HarnessConfig())
    assert metrics["stop_reason"] == "finished" and "verification_runs" not in metrics


def test_verify_command_survives_resume(cli):  # noqa: F811
    start, workdir, runs = cli
    server = FakeOpenAI([("done", []), ("", [tool_call("bash", {"command": "touch fixed.txt"}, "c1")]), ("done now", [])],
                        hold_requests={1})
    first = start(server, "--verify", VERIFY, "create fixed.txt")
    run_id = run_id_of(first)
    server.wait_for_requests(2)
    kill(first)
    server.release.set()
    assert json.loads((runs / run_id / "config.json").read_text())["verify_command"] == VERIFY

    resumed = start(server, "--resume", run_id)
    output, _ = resumed.communicate(timeout=60)
    assert resumed.returncode == 0, output
    assert "Verified" in output and (workdir / "fixed.txt").exists()
    metrics = json.loads(output.split("📊 ", 1)[1].splitlines()[0])
    assert metrics["stop_reason"] == "verified" and metrics["verification_runs"] == 2
