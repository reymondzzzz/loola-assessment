import json
import sqlite3

import agent
from harness import registry
from harness.prompt import initial_tools, system_prompt


def names(tools):
    return [tool["function"]["name"] for tool in tools]


def test_tool_discovery():
    assert registry.TOOLS.discover("run a sql query on the sqlite database")[0] == "sqlite_query"
    assert "git_log" in registry.TOOLS.discover("which commit introduced a regression")
    assert registry.SKILLS.discover("count errors in a log file")[0] == "log-analysis"
    assert registry.discover("zzz qqq").startswith("No matching tools")
    assert len(registry.discover("file").splitlines()) <= 6


def test_dynamic_tool_validation():
    assert "missing required ['path']" in registry.invoke("read_file", {})
    assert "unknown ['bogus']" in registry.invoke("read_file", {"path": "x", "bogus": 1})
    assert "wrong type for ['limit']" in registry.invoke("read_file", {"path": "x", "limit": "10"})
    assert "wrong type for ['limit']" in registry.invoke("read_file", {"path": "x", "limit": True})
    assert "unknown tool 'nope'" in registry.invoke("nope", {})
    assert "not available in this environment" in registry.invoke("slack_post_message", {"channel": "c", "text": "t"})


def test_catalog_tools_work(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a.py").write_text("x = 1\nx = 1\ny = 2\n")
    assert "found 2 occurrences" in registry.invoke("edit_file", {"path": "a.py", "old": "x = 1", "new": "x = 3"})
    assert registry.invoke("edit_file", {"path": "a.py", "old": "y = 2", "new": "y = 5"}) == "Edited a.py"
    assert "[lines 2-3 of 3]" in registry.invoke("read_file", {"path": "a.py", "offset": 2})
    (tmp_path / "d.json").write_text(json.dumps({"s": [{"id": "S-1"}]}))
    assert json.loads(registry.invoke("json_query", {"path": "d.json", "pointer": "s.0"})) == {"id": "S-1"}
    with sqlite3.connect(tmp_path / "x.db") as db:
        db.execute("CREATE TABLE t (v INTEGER)")
        db.execute("INSERT INTO t VALUES (7)")
    assert registry.invoke("sqlite_query", {"database": "x.db", "sql": "SELECT v FROM t"}) == "v\n7"
    delete = {"name": "sqlite_query", "arguments": {"database": "x.db", "sql": "DELETE FROM t"}}
    assert "readonly" in agent.call_tool("invoke", delete, {"invoke"})


def test_skill_progressive_loading():
    metadata_prompt = system_prompt("lazy")
    assert "sql-reporting:" in metadata_prompt and "# SQL reporting" not in metadata_prompt
    level1 = registry.load_skill("sql-reporting")
    assert "# SQL reporting" in level1 and "References: references/sqlite-datetime.md" in level1
    assert "datetime(" in registry.read_skill_resource("sql-reporting", "references/sqlite-datetime.md")
    assert registry.read_skill_resource("sql-reporting", "../log-analysis/SKILL.md").startswith("Error")
    assert registry.read_skill_resource("sql-reporting", "/etc/passwd").startswith("Error")
    assert registry.load_skill("nope").startswith("Error: unknown skill")


def test_modes_share_one_catalog():
    eager, lazy = initial_tools("eager"), initial_tools("lazy")
    assert set(names(eager)) == {"bash", *registry.TOOLS.specs}
    assert names(lazy) == ["bash", "discover", "describe", "invoke", "load_skill", "read_skill_resource"]
    assert "invoke" not in names(initial_tools("native"))
    assert not any(f"{name}" in system_prompt("lazy") for name in ("sqlite_query", "git_show", "slack_post_message"))
    assert all(f"# " in system_prompt("eager") for _ in registry.SKILLS.skills)


def test_native_binding():
    tools = initial_tools("native")
    assert "unknown tool 'git_log'" in agent.call_tool("git_log", {}, set(names(tools)))
    bound = agent.bind_described_tool(tools, "git_log")
    assert names(bound)[-1] == "git_log"
    assert agent.bind_described_tool(bound, "git_log") is bound
    assert agent.bind_described_tool(bound, "nope") is bound


def test_lazy_prefix_unchanged_by_discovery():
    before = json.dumps([system_prompt("lazy"), initial_tools("lazy")])
    registry.describe("git_log")
    registry.discover("csv")
    registry.load_skill("csv-parsing")
    assert json.dumps([system_prompt("lazy"), initial_tools("lazy")]) == before
