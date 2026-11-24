import json

from harness.executor import ToolExecutor
from tests.fakes import call, run_script

FAILING_TEST = "def test_fails():\n    assert 1 == 2\n"


def invoke_run_tests(path: str) -> str:
    return json.dumps({"name": "run_tests", "arguments": {"path": path}})


def test_run_tests_reports_failing_exit_code(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "test_bad.py").write_text(FAILING_TEST)
    (tmp_path / "test_good.py").write_text("def test_ok():\n    assert True\n")
    executor = ToolExecutor(available={"invoke"})
    assert executor.execute("invoke", invoke_run_tests("test_bad.py"), "c1").code == "NONZERO_EXIT"
    assert executor.execute("invoke", invoke_run_tests("test_good.py"), "c2").code == "OK"


def test_repeated_bash_failure_gets_a_hint_but_is_not_blocked(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    executor = ToolExecutor(available={"bash"})
    failing = json.dumps({"command": "cat missing.txt"})
    first, second, third = (executor.execute("bash", failing, f"c{i}") for i in range(1, 4))
    assert [o.code for o in (first, second, third)] == ["NONZERO_EXIT"] * 3
    assert "[Hint:" not in first.content
    assert "failed 2 times" in second.content and "tool call c1" in second.content and "missing.txt" in second.content
    assert "failed 3 times" in third.content
    executor.execute("bash", json.dumps({"command": "echo fix > other.txt"}), "c4")
    after_change = executor.execute("bash", failing, "c5")
    assert after_change.code == "NONZERO_EXIT" and "[Hint:" not in after_change.content


def test_failing_bash_without_file_changes_does_not_count_as_environment_change(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    executor = ToolExecutor(available={"bash"})
    executor.execute("bash", json.dumps({"command": "false"}), "c1")
    assert executor.environment_version == 0
    executor.execute("bash", json.dumps({"command": "touch made.txt"}), "c2")
    assert executor.environment_version == 1


def test_repeated_run_tests_failure_hint_then_loop_block(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "test_bad.py").write_text(FAILING_TEST)
    executor = ToolExecutor(available={"invoke"})
    outcomes = [executor.execute("invoke", invoke_run_tests("test_bad.py"), f"c{i}") for i in range(1, 5)]
    assert [o.code for o in outcomes] == ["NONZERO_EXIT", "NONZERO_EXIT", "NONZERO_EXIT", "LOOP_DETECTED"]
    assert "[Hint:" in outcomes[1].content and "tool call c1" in outcomes[1].content


def test_repeat_hint_reaches_the_model_through_the_graph(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    script = [("", [call("bash", {"command": "cat missing.txt"}, f"c{i}")]) for i in range(1, 3)]
    _, llm = run_script(script)
    results = [m["content"] for m in llm.requests[2]["messages"] if m["role"] == "tool"]
    assert "[Hint:" not in results[0] and "previous attempt: tool call c1" in results[1]
