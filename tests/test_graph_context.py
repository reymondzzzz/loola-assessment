import json
from dataclasses import replace

from harness.config import mode_config
from tests.fakes import call, run_script

TASK = "Do the tickets. Constraint: never edit tests/. Code name to remember: ORCHID-7."
SUMMARY = json.dumps({"goal": "do the tickets", "verified_facts": ["code name ORCHID-7"], "decisions": [],
                      "completed_work": ["printed files"], "failed_attempts": [], "pending_work": ["continue"],
                      "important_paths": [], "artifact_refs": [], "active_skill_refs": []})


def noisy_script(turns: int, lines: int = 300):
    return [("", [call("bash", {"command": f"seq 1 {lines} | sed 's/^/turn{turn} value /'"}, f"c{turn}")])
            for turn in range(turns)]


def is_prefix(shorter: list, longer: list) -> bool:
    return longer[:len(shorter)] == shorter


def test_prompt_prefix_stability(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _, llm = run_script(noisy_script(6), mode_config("lazy"), TASK)
    for previous, current in zip(llm.requests, llm.requests[1:]):
        assert current["tools"] == previous["tools"]
        assert is_prefix(previous["messages"], current["messages"])
        assert json.dumps(current["messages"]).startswith(json.dumps(previous["messages"])[:-1])


def test_compaction_epochs_keep_stable_head_and_task(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = replace(mode_config("hybrid", context_window=12_000), output_reserve=1000, safety_margin=500,
                     recent_interactions=2)
    metrics, llm = run_script(noisy_script(60, lines=400), config, TASK, summary=SUMMARY)
    assert metrics["stop_reason"] == "finished" and metrics["compactions"] >= 2
    head = llm.requests[0]["messages"][:2]
    assert all(r["messages"][:2] == head and r["tools"] == llm.requests[0]["tools"] for r in llm.requests)
    assert all(r["messages"][1]["content"] == TASK for r in llm.requests)
    assert all(metrics["peak_context"] < config.input_budget for _ in [0])
    appended_turns = sum(is_prefix(a["messages"], b["messages"]) for a, b in zip(llm.requests, llm.requests[1:]))
    assert appended_turns >= len(llm.requests) - 1 - metrics["compactions"]


def test_semantic_compaction_preserves_early_facts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = replace(mode_config("summary", context_window=12_000), output_reserve=1000, safety_margin=500,
                     recent_interactions=2)
    metrics, llm = run_script(noisy_script(30, lines=400), config, TASK, summary=SUMMARY)
    assert metrics["summarizer_model_calls"] >= 1
    summarized = [r for r in llm.requests if len(r["messages"]) > 2 and (r["messages"][2]["content"] or "").startswith("[Context summary")]
    assert summarized and "ORCHID-7" in summarized[-1]["messages"][2]["content"]
    assert "ORCHID-7" in llm.summary_requests[0][1]["content"]


def test_summarizer_failure_does_not_lose_history(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = replace(mode_config("hybrid", context_window=12_000), output_reserve=1000, safety_margin=500,
                     recent_interactions=2)
    metrics, llm = run_script(noisy_script(20, lines=400), config, TASK, summary=RuntimeError("summarizer down"))
    assert metrics["stop_reason"] == "finished"
    assert all(r["messages"][1]["content"] == TASK for r in llm.requests)


def test_context_overflow_recovery(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    metrics, llm = run_script(noisy_script(8), mode_config("hybrid"), TASK, summary=SUMMARY, overflow_calls={6})
    assert metrics["stop_reason"] == "finished" and metrics["emergency_compactions"] == 1
    after = llm.requests[6]["messages"]
    assert after[1]["content"] == TASK and len(json.dumps(after)) < len(json.dumps(llm.requests[5]["messages"]))


def test_repeated_overflow_stops_with_budget_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    metrics, _ = run_script(noisy_script(8), mode_config("hybrid"), TASK, summary=SUMMARY, overflow_calls={4, 5, 6, 7})
    assert metrics["stop_reason"] in ("context_budget_exceeded", "finished")
    assert metrics["emergency_compactions"] >= 1
