import json
import shutil
import sys
from pathlib import Path

from harness.config import HarnessConfig, OUTPUT_PIPELINE
from harness.executor import ToolExecutor, run_bash
from harness.memory import read_artifact
from harness.output import OutputProcessor, head_tail
from harness.store import ArtifactStore
from harness.tokens import count_tokens
from tests.fakes import call, run_script


def test_tool_budget_checked_before_each_call(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calls = [call("bash", {"command": f"touch f{i}"}, f"c{i}") for i in range(3)]
    metrics, llm = run_script([("", calls)], HarnessConfig(max_tool_calls=1))
    assert metrics["stop_reason"] == "max_tool_calls"
    assert sorted(p.name for p in tmp_path.iterdir() if p.name.startswith("f")) == ["f0"]
    events = [json.loads(line) for line in open(metrics["events_path"])]
    codes = [e["code"] for e in events if e["kind"] == "tool_result"]
    assert codes == ["OK", "BUDGET_EXHAUSTED", "BUDGET_EXHAUSTED"]


def test_head_tail_never_exceeds_budget():
    text = "\n".join(f"line {i} " + "word " * 30 for i in range(500))
    for budget in (10, 40, 100, 700):
        result = head_tail(text, budget, "a0123456789a", count_tokens(text))
        floor = count_tokens("[artifact a0123456789a]")
        assert count_tokens(result) <= max(budget, floor)
        assert "a0123456789a" in result


def test_bash_capture_is_bounded_and_full_output_recoverable(tmp_path):
    outcome = run_bash(f"{sys.executable} -c \"print('x' * 3_000_000); print('LAST LINE')\"")
    assert len(outcome.content) < 300_000 and outcome.spill_path and Path(outcome.spill_path).stat().st_size > 3_000_000
    artifacts = ArtifactStore(tmp_path)
    action = {"id": "c1", "name": "bash", "arguments": "{}", "parsed_arguments": {}, **vars(outcome)}
    processed, observations = OutputProcessor(HarnessConfig(**OUTPUT_PIPELINE), artifacts).process([action], {}, 1, 50_000)
    artifact_id = observations["c1"]["artifact_id"]
    assert not Path(outcome.spill_path).exists()
    assert "LAST LINE" in read_artifact(artifacts, artifact_id, offset=3, limit=5)
    assert observations["c1"]["original_tokens"] > 500_000 and count_tokens(processed[0]["visible"]) <= 2000


def test_invoke_with_wrong_argument_types_is_repairable():
    executor = ToolExecutor(available={"invoke"})
    for arguments in ({"name": 5, "arguments": {}}, {"name": "read_file", "arguments": "path=a.txt"}, {"name": "read_file"}):
        outcome = executor.execute("invoke", json.dumps(arguments))
        assert outcome.code == "INVALID_ARGUMENTS" and json.loads(outcome.content)["model_can_repair"]


def test_benchmark_detects_deleted_visible_test(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))
    import run as bench_run
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    for name in ("test_pricing.py", "test_store.py"):
        shutil.copy(bench_run.VISIBLE_TESTS_DIR / name, tests_dir / name)
    assert bench_run.tests_intact(tmp_path, current_ticket=1)
    (tests_dir / "test_store.py").unlink()
    assert not bench_run.tests_intact(tmp_path, current_ticket=1)
    shutil.copy(bench_run.VISIBLE_TESTS_DIR / "test_store.py", tests_dir / "test_store.py")
    (tests_dir / "test_store.py").write_text("def test_ok(): pass\n")
    assert not bench_run.tests_intact(tmp_path, current_ticket=1)


def test_malformed_invoke_does_not_crash_the_graph(tmp_path, monkeypatch):
    from harness.config import mode_config
    monkeypatch.chdir(tmp_path)
    script = [("", [call("invoke", {"name": "read_file", "arguments": "path=a.txt"}, "c1"),
                    call("invoke", {"name": 7, "arguments": []}, "c2")])]
    metrics, llm = run_script(script, mode_config("hybrid"))
    results = [m["content"] for m in llm.requests[1]["messages"] if m["role"] == "tool"]
    assert metrics["stop_reason"] == "finished"
    assert all(json.loads(r)["code"] == "INVALID_ARGUMENTS" for r in results)


def test_shared_output_budget_respects_twenty_percent():
    from harness.output import MIN_CALL_BUDGET, per_call_budgets
    budgets = per_call_budgets([{"tool": "bash"}] * 8, HarnessConfig(**OUTPUT_PIPELINE), available_tokens=1000)
    assert sum(budgets) <= 200 and all(b >= MIN_CALL_BUDGET for b in budgets)
    tiny = per_call_budgets([{"tool": "bash"}] * 8, HarnessConfig(**OUTPUT_PIPELINE), available_tokens=100)
    assert sum(tiny) == MIN_CALL_BUDGET * 8
    text = "\n".join("word " * 40 for _ in range(200))
    for budget in budgets + [MIN_CALL_BUDGET]:
        assert count_tokens(head_tail(text, budget, "a0123456789a", count_tokens(text))) <= budget


def test_pytest_cache_files_do_not_count_as_test_changes(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))
    import run as bench_run
    tests_dir = tmp_path / "tests"
    (tests_dir / "__pycache__").mkdir(parents=True)
    (tests_dir / "__pycache__" / "test_pricing.cpython-312-pytest.pyc").write_bytes(b"x")
    shutil.copy(bench_run.VISIBLE_TESTS_DIR / "test_pricing.py", tests_dir / "test_pricing.py")
    assert bench_run.tests_intact(tmp_path, current_ticket=0)
