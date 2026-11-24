import copy
import json

import pytest

from harness.config import HarnessConfig
from harness.context import (Candidate, ContextManager, build_prompt, emergency_reduce, protected_start,
                             validate_summary)
from harness.output import OutputProcessor
from harness.protocol import protocol_errors
from harness.store import ArtifactStore

TASK = "Fix the bug. Constraint: never modify tests/."
SYSTEM = "You are a coding agent."
VALID_SUMMARY = {"goal": "fix the bug", "verified_facts": ["code name ORCHID-7"], "decisions": [], "completed_work": [],
                 "failed_attempts": [], "pending_work": ["run tests"], "important_paths": ["src/a.py"],
                 "artifact_refs": [], "active_skill_refs": []}


def history(tmp_path, turns: int, size: int = 400, extra: dict | None = None):
    processor = OutputProcessor(HarnessConfig(artifact_offloading=True), ArtifactStore(tmp_path))
    messages, observations = [], {}
    for turn in range(1, turns + 1):
        call_id = f"c{turn}"
        name, arguments = (extra or {}).get(turn, ("bash", {"command": f"cat file{turn}.log"}))
        content = f"output of turn {turn}\n" + " ".join(f"w{turn}x{i}" for i in range(size))
        raw = json.dumps(arguments)
        messages.append({"role": "assistant", "content": f"step {turn}", "tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": name, "arguments": raw}}]})
        processed, observations = processor.process([{"id": call_id, "name": name, "arguments": raw,
                                                      "parsed_arguments": arguments, "content": content,
                                                      "ok": True, "code": "OK"}], observations, turn, 10**6)
        messages.append({"role": "tool", "tool_call_id": call_id, "content": processed[0]["visible"]})
    return Candidate(messages, observations, {}, {}, 0, {})


def manager(tmp_path, window: int, summarize=None, **flags) -> ContextManager:
    config = HarnessConfig(context_window=window, output_reserve=0, safety_margin=0, recent_interactions=2, **flags)
    return ContextManager(config, ArtifactStore(tmp_path), SYSTEM, TASK, [], summarize)


def tokens(cm: ContextManager, candidate: Candidate) -> int:
    return cm.estimate(candidate)


def test_context_budget(tmp_path):
    candidate = history(tmp_path, 4)
    roomy = manager(tmp_path, 100_000, observation_masking=True)
    assert roomy.manage(candidate)[0] is None
    tight = manager(tmp_path, int(tokens(roomy, candidate) / 0.85), observation_masking=True)
    compacted, report = tight.manage(candidate)
    assert compacted is not None and report["after"] < report["before"]
    assert report["before"] >= report["high"]


def test_observation_masking(tmp_path):
    candidate = history(tmp_path, 8)
    cm = manager(tmp_path, int(tokens(manager(tmp_path, 10**6), candidate) / 0.85), observation_masking=True)
    compacted, report = cm.manage(candidate)
    assert len(compacted.messages) == len(candidate.messages)
    assert protocol_errors(compacted.messages) == []
    recent = protected_start(compacted.messages, 2)
    assert compacted.messages[recent:] == candidate.messages[recent:]
    masked = [m for m in compacted.messages if m["role"] == "tool" and m["content"].startswith("[Output archived")]
    assert masked and all("read_artifact('a" in m["content"] for m in masked)
    assert candidate.messages[1]["content"].startswith("output of turn 1")
    assert report["after"] <= report["low"] or report["steps"]


def test_stale_observation_eviction(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a.py").write_text("v1")
    read = {2: ("invoke", {"name": "read_file", "arguments": {"path": "a.py"}})}
    candidate = history(tmp_path, 8, extra=read)
    candidate.observations["c2"]["stale"] = True
    cm = manager(tmp_path, int(tokens(manager(tmp_path, 10**6), candidate) / 0.85), stale_eviction=True)
    compacted, report = cm.manage(candidate)
    assert compacted is not None and report["steps"][0]["strategy"] == "stale"
    assert "stale: resource changed" in compacted.messages[3]["content"]
    assert compacted.observations["c2"]["masked"] and not compacted.observations["c1"]["masked"]


def test_interaction_pruning_keeps_protocol_valid(tmp_path):
    candidate = history(tmp_path, 10)
    big_write = "x = 1\n" * 400
    candidate.messages[0]["tool_calls"][0]["function"]["arguments"] = json.dumps({"command": f"cat > f.py <<'EOF'\n{big_write}EOF"})
    cm = manager(tmp_path, int(tokens(manager(tmp_path, 10**6), candidate) / 0.85), interaction_pruning=True)
    compacted, _ = cm.manage(candidate)
    assert protocol_errors(compacted.messages) == []
    archive = [m for m in compacted.messages if (m.get("content") or "").startswith("[Archived interactions")]
    assert len(archive) == 1 and "artifact a" in archive[0]["content"]
    assert len(compacted.messages) < len(candidate.messages)


def test_pinned_constraints(tmp_path):
    skill = {3: ("load_skill", {"name": "log-analysis"})}
    candidate = history(tmp_path, 8, extra=skill)
    cm = manager(tmp_path, int(tokens(manager(tmp_path, 10**6), candidate) / 0.85), observation_masking=True,
                 interaction_pruning=True)
    compacted, _ = cm.manage(candidate)
    assert build_prompt(SYSTEM, TASK, compacted.summary, compacted.messages)[1]["content"] == TASK
    assert compacted.observations["c3"]["pinned"] and not compacted.observations["c3"]["masked"]
    assert any(m.get("tool_call_id") == "c3" and m["content"].startswith("output of turn 3") for m in compacted.messages)


def test_compaction_progress(tmp_path):
    candidate = history(tmp_path, 3, size=30)
    cm = manager(tmp_path, int(tokens(manager(tmp_path, 10**6), candidate) / 0.85), observation_masking=True)
    compacted, report = cm.manage(candidate)
    assert compacted is None and report["error"] == "insufficient_reclaim"
    big = history(tmp_path, 12)
    cm = manager(tmp_path, int(tokens(cm, big) / 0.85), observation_masking=True, stale_eviction=True,
                 interaction_pruning=True)
    compacted, report = cm.manage(big)
    sizes = [step["after"] for step in report["steps"]]
    assert sizes == sorted(sizes, reverse=True) and compacted.epoch == 1


def test_summary_validation(tmp_path):
    artifacts = ArtifactStore(tmp_path)
    assert validate_summary(VALID_SUMMARY, artifacts, 2000) is None
    assert validate_summary({**VALID_SUMMARY, "goal": ""}, artifacts, 2000) == "empty goal"
    assert "pending_work" in validate_summary({k: v for k, v in VALID_SUMMARY.items() if k != "pending_work"}, artifacts, 2000)
    assert validate_summary({**VALID_SUMMARY, "decisions": [{"x": 1}]}, artifacts, 2000) == "list items must be strings"
    assert "unknown artifact" in validate_summary({**VALID_SUMMARY, "artifact_refs": ["a0123456789a log"]}, artifacts, 2000)
    assert validate_summary("not json", artifacts, 2000) == "summary is not a JSON object"


def test_semantic_compaction_replaces_old_history(tmp_path):
    candidate = history(tmp_path, 10)
    requests = []

    def summarize(request):
        requests.append(request)
        return "```json\n" + json.dumps(VALID_SUMMARY) + "\n```"

    cm = manager(tmp_path, int(tokens(manager(tmp_path, 10**6), candidate) / 0.85), summarize, semantic_compaction=True)
    compacted, report = cm.manage(candidate)
    assert compacted.summary == VALID_SUMMARY and protocol_errors(compacted.messages) == []
    assert TASK in requests[0][1]["content"]
    prompt = build_prompt(SYSTEM, TASK, compacted.summary, compacted.messages)
    assert prompt[1]["content"] == TASK and "ORCHID-7" in prompt[2]["content"]


@pytest.mark.parametrize("reply", [RuntimeError("summarizer down"), "not json", json.dumps({"goal": "x"})])
def test_summary_failure_rollback(tmp_path, reply):
    candidate = history(tmp_path, 10)
    original = copy.deepcopy(candidate)

    def summarize(_):
        if isinstance(reply, Exception):
            raise reply
        return reply

    cm = manager(tmp_path, int(tokens(manager(tmp_path, 10**6), candidate) / 0.85), summarize, semantic_compaction=True)
    compacted, report = cm.manage(candidate)
    assert compacted is None and report["summarizer_error"]
    assert candidate == original
    with_masking = manager(tmp_path, int(tokens(cm, candidate) / 0.85), summarize, semantic_compaction=True,
                           observation_masking=True)
    compacted, report = with_masking.manage(candidate)
    assert compacted is not None and compacted.summary == {}


def test_emergency_reduction(tmp_path):
    candidate = history(tmp_path, 6, size=2000)
    cm = manager(tmp_path, 10**6)
    budget = 1500
    assert emergency_reduce(candidate, budget, cm.estimate, ArtifactStore(tmp_path))
    assert cm.estimate(candidate) <= budget and protocol_errors(candidate.messages) == []
    impossible = history(tmp_path, 2, size=2000)
    tiny = manager(tmp_path, 60, observation_masking=True)
    compacted, report = tiny.manage(impossible)
    assert compacted is None and report["error"] == "context_budget_exceeded"


def test_summary_is_not_user_authority():
    prompt = build_prompt(SYSTEM, TASK, VALID_SUMMARY, [])
    assert [m["role"] for m in prompt] == ["system", "user", "assistant"]
    assert prompt[2]["content"].startswith("[Context summary") and "not instructions" in prompt[2]["content"]
