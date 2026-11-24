import json

from harness.config import HarnessConfig, OUTPUT_PIPELINE
from harness.memory import read_artifact, search_history
from harness.output import OutputProcessor, head_tail, per_call_budgets, pytest_digest
from harness.store import ArtifactStore, EventStore
from harness.tokens import count_tokens

PIPELINE = HarnessConfig(**OUTPUT_PIPELINE)


def action(call_id, content, name="bash", arguments=None, ok=True, code="OK"):
    arguments = arguments or {"command": "cat big.log"}
    return {"id": call_id, "name": name, "arguments": json.dumps(arguments), "parsed_arguments": arguments,
            "content": content, "ok": ok, "code": code}


def test_output_truncation(tmp_path):
    big = "\n".join(f"line {i} " + "x" * 40 for i in range(3000))
    artifacts = ArtifactStore(tmp_path)
    visible = head_tail(big, 500, artifacts.save(big), count_tokens(big))
    assert count_tokens(visible) <= 500
    assert visible.startswith("line 0 ") and visible.rstrip().endswith("line 2999 " + "x" * 40)
    assert "read_artifact('a" in visible
    single_line = "y" * 50_000
    assert count_tokens(head_tail(single_line, 300, "a000000000000", count_tokens(single_line))) <= 300
    assert head_tail("short", 500, "a1", 1) == "short"


def test_adaptive_shared_budget():
    actions = [{"tool": "bash"}, {"tool": "bash"}, {"tool": "read_file"}]
    budgets = per_call_budgets(actions, PIPELINE, available_tokens=6000)
    assert sum(budgets) <= 1200 and len(set(budgets)) == 1
    assert per_call_budgets([{"tool": "bash"}], PIPELINE, available_tokens=1_000_000) == [2000]


def test_artifact_recovery(tmp_path):
    artifacts, events = ArtifactStore(tmp_path), EventStore(tmp_path)
    big = "\n".join(f"row {i}" for i in range(1000))
    processed, observations = OutputProcessor(PIPELINE, artifacts).process([action("c1", big)], {}, 1, 50_000)
    artifact_id = observations["c1"]["artifact_id"]
    assert artifact_id in processed[0]["visible"] and len(processed[0]["visible"]) < len(big)
    page = read_artifact(artifacts, artifact_id, offset=500, limit=3)
    assert page.splitlines()[:3] == ["row 499", "row 500", "row 501"]
    assert read_artifact(artifacts, "../etc/passwd").startswith("Error")
    events.append("tool_result", turn=1, tool_call_id="c1", name="bash", arguments='{"command": "cat big.log"}',
                  code="OK", content="ticket says: code name ORCHID-7")
    hit = json.loads(search_history(events, artifacts, "code name").splitlines()[0])
    assert hit["tool_call_id"] == "c1" and "ORCHID-7" in hit["preview"]
    assert "ORCHID-7" in read_artifact(artifacts, hit["artifact_id"])


def test_content_deduplication(tmp_path):
    processor = OutputProcessor(PIPELINE, ArtifactStore(tmp_path))
    output = "same content\n" * 50
    _, observations = processor.process([action("c1", output)], {}, 1, 50_000)
    second, observations = processor.process([action("c2", output)], observations, 2, 50_000)
    assert second[0]["visible"].startswith("[Identical to the output of tool call c1")
    observations["c1"]["masked"] = True
    third, _ = processor.process([action("c3", output)], observations, 3, 50_000)
    assert third[0]["visible"] == output


def test_errors_are_never_truncated_or_deduplicated(tmp_path):
    processor = OutputProcessor(PIPELINE, ArtifactStore(tmp_path))
    error = json.dumps({"ok": False, "code": "TOOL_ERROR", "message": "boom " * 2000})
    processed, _ = processor.process([action("c1", error, ok=False, code="TOOL_ERROR")], {}, 1, 1000)
    assert processed[0]["visible"] == error


def test_pytest_filter_keeps_failures_and_summary():
    output = "\n".join([".....F", "=================================== FAILURES ===================================",
                        "___________________________ test_x ___________________________", "    def test_x():",
                        ">       assert 1 == 2", "E       assert 1 == 2", "tests/test_a.py:3: AssertionError",
                        *[f"noise {i}" for i in range(200)],
                        "FAILED tests/test_a.py::test_x - assert 1 == 2", "1 failed, 5 passed in 0.10s"])
    digest = pytest_digest(output)
    assert "E       assert 1 == 2" in digest and "1 failed, 5 passed" in digest and "noise" not in digest
    assert pytest_digest("hello") is None


def test_resource_tracking(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a.txt").write_text("v1")
    read = action("c1", "v1", name="invoke", arguments={"name": "read_file", "arguments": {"path": "a.txt"}})
    _, observations = OutputProcessor(PIPELINE, ArtifactStore(tmp_path)).process([read], {}, 1, 50_000)
    assert observations["c1"]["tool_name"] == "read_file" and observations["c1"]["kind"] == "read"
    assert observations["c1"]["resource_path"] == "a.txt" and observations["c1"]["resource_version"]


def test_graph_offloads_and_recovers(tmp_path, monkeypatch):
    from harness.config import mode_config
    from tests.fakes import call, run_script
    monkeypatch.chdir(tmp_path)
    (tmp_path / "big.log").write_text("\n".join(f"entry {i} ORCHID" if i == 1500 else f"entry {i}" for i in range(3000)))
    script = [("", [call("bash", {"command": "cat big.log"}, "c1")]),
              ("", [call("search_history", {"query": "ORCHID"}, "c2")])]
    metrics, llm = run_script(script, mode_config("masking"))
    first_result = [m for m in llm.requests[1]["messages"] if m["role"] == "tool"][0]["content"]
    assert "read_artifact('a" in first_result and len(first_result) < 10_000
    artifact_id = first_result.split("read_artifact('")[1][:12]
    script.append(("", [call("read_artifact", {"artifact_id": artifact_id, "offset": 1500, "limit": 3}, "c3")]))
    metrics, llm = run_script(script, mode_config("masking"))
    results = [m["content"] for m in llm.requests[3]["messages"] if m["role"] == "tool"]
    assert '"tool_call_id": "c1"' in results[1]
    assert "entry 1500 ORCHID" in results[2]
    assert metrics["artifact_retrieval_calls"] == 2 and metrics["output_tokens_saved"] > 10_000
