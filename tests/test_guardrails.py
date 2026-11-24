import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import guardrails  # noqa: E402


@pytest.fixture
def repo(tmp_path):
    for path in guardrails.FROZEN:
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / path, tmp_path / path)
    (tmp_path / "harness").mkdir()
    (tmp_path / "harness" / "ok.py").write_text("import json\nfrom harness import config\n")
    (tmp_path / ".gitignore").write_text(".env\n")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    assert guardrails.run_guardrails(tmp_path) == []
    return tmp_path


def test_real_repository_passes_all_guardrails():
    assert guardrails.run_guardrails(ROOT) == []


def test_g1_frozen_baseline(repo):
    with open(repo / "baseline" / "original_agent.py", "a") as f:
        f.write("# tweak\n")
    problems = guardrails.frozen_baseline(repo)
    assert len(problems) == 1 and problems[0].startswith("G1 baseline/original_agent.py: content changed")


def test_g2_secrets_home_paths_and_emails(repo):
    (repo / "notes.md").write_text("key sk-" + "a" * 24 + "\n")
    (repo / "log.txt").write_text("ran in " + "/Us" + "ers/someone/project\n")
    (repo / "contact.md").write_text("mail someone" + "@gm" + "ail.com, ok bot noreply@anthropic.com, a@example.com\n")
    assert sorted(guardrails.leaked_secrets(repo)) == [
        "G2 contact.md: contains a personal email address",
        "G2 log.txt: contains a home directory path",
        "G2 notes.md: looks like an API key"]


def test_g2_ignores_git_ignored_files(repo):
    (repo / ".env").write_text("LLM_API_KEY=sk-" + "b" * 30 + "\n")
    assert guardrails.leaked_secrets(repo) == []


def test_g3_env_file_tracked(repo):
    (repo / ".env").write_text("X=1\n")
    subprocess.run(["git", "add", "-f", ".env"], cwd=repo, check=True)
    assert guardrails.env_file_tracked(repo) == ["G3 .env is tracked by git; remove it from the index"]


def test_g4_layering(repo):
    (repo / "harness" / "bad.py").write_text("from bench.run import TASK\nimport tests.fakes\n")
    assert guardrails.layering(repo) == ["G4 harness/bad.py: imports bench", "G4 harness/bad.py: imports tests"]


def test_g5_trailing_whitespace_skips_run_logs(repo):
    (repo / "harness" / "ws.py").write_text("x = 1   \n")
    (repo / "bench" / "runs").mkdir(parents=True)
    (repo / "bench" / "runs" / "a.log").write_text("raw output   \n")
    assert guardrails.trailing_whitespace(repo) == ["G5 harness/ws.py:1: trailing whitespace"]
