import json
import re
from dataclasses import replace
from pathlib import Path

from harness.config import mode_config
from harness.prompt import initial_tools, system_prompt
from tests.fakes import call, run_script


def names(tools):
    return [tool["function"]["name"] for tool in tools]


def test_lazy_skills_is_independent_of_tool_mode():
    for base in (mode_config("lazy"), mode_config("eager")):
        lazy, eager = replace(base, lazy_skills=True), replace(base, lazy_skills=False)
        assert "# SQL reporting" not in system_prompt(lazy) and "sql-reporting:" in system_prompt(lazy)
        assert "# SQL reporting" in system_prompt(eager)
        assert {"load_skill", "read_skill_resource"} <= set(names(initial_tools(lazy)))
        assert not {"load_skill", "read_skill_resource"} & set(names(initial_tools(eager)))
        tool_part = lambda c: [n for n in names(initial_tools(c)) if n not in ("load_skill", "read_skill_resource")]  # noqa: E731
        assert tool_part(lazy) == tool_part(eager)


def big_output_script(turns: int):
    return [("", [call("bash", {"command": f"seq 1 3000 | sed 's/^/t{t} row /'"}, f"c{t}")]) for t in range(turns)]


def run_with_offloading(tmp_path, monkeypatch, enabled: bool):
    monkeypatch.chdir(tmp_path)
    config = replace(mode_config("masking", context_window=14_000), output_reserve=1000, safety_margin=500,
                     recent_interactions=2, artifact_offloading=enabled)
    metrics, llm = run_script(big_output_script(30), config)
    artifacts_dir = Path(metrics["events_path"]).parent / "artifacts"
    return metrics, llm, artifacts_dir, config


def test_artifact_offloading_on(tmp_path, monkeypatch):
    metrics, llm, artifacts_dir, config = run_with_offloading(tmp_path, monkeypatch, True)
    text = json.dumps([r["messages"] for r in llm.requests])
    referenced = set(re.findall(r"read_artifact\('(a[0-9a-f]{11})'", text))
    assert referenced and all((artifacts_dir / f"{a}.txt").exists() for a in referenced)
    assert "read_artifact" in names(llm.requests[0]["tools"]) and metrics["compactions"] >= 1


def test_artifact_offloading_off_keeps_limits_without_dangling_references(tmp_path, monkeypatch):
    metrics, llm, artifacts_dir, config = run_with_offloading(tmp_path, monkeypatch, False)
    assert metrics["stop_reason"] == "finished"
    assert not artifacts_dir.exists() or not any(artifacts_dir.iterdir())
    text = json.dumps([r["messages"] for r in llm.requests] + [r["tools"] for r in llm.requests])
    assert "read_artifact" not in text and "[artifact " not in text
    first_result = [m for m in llm.requests[1]["messages"] if m["role"] == "tool"][0]["content"]
    assert "Output not archived" in first_result and len(first_result) < 12_000
    assert metrics["peak_context"] <= config.input_budget * 1.05
