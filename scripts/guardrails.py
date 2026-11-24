import ast, hashlib, re, subprocess, sys
from pathlib import Path

FROZEN = {"baseline/original_agent.py": "d1e005f625b5212bab74fb0b273839d1f9a2129fa4bdc56071385583a905eb1c",
          "baseline/langgraph_port.py": "e8fe1091a71ba97aaafbd2e98eb62b6e65c9566e781a25007d6f87a1d9803b35"}
SECRET = re.compile(r"sk-[A-Za-z0-9_-]{20,}")
HOME_PATH = re.compile(r"/(Users|home)/[A-Za-z0-9._-]+/")
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[a-z]{2,}")
ALLOWED_EMAILS = ("noreply@anthropic.com", "@example.com")
FORBIDDEN_IMPORTS = {"harness": {"bench", "tests", "baseline", "agent"}, "baseline": {"harness", "bench", "tests"}}
WHITESPACE_CHECKED = (".py", ".md", ".toml", ".json", ".sh")


def repository_files(root: Path) -> list[str]:
    listed = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=root,
                            capture_output=True, text=True, check=True).stdout.splitlines()
    return [path for path in listed if (root / path).is_file()]


def frozen_baseline(root: Path) -> list[str]:
    return [f"G1 {path}: content changed (sha256 {digest[:12]}…); baseline files are frozen"
            for path, expected in FROZEN.items()
            if (digest := hashlib.sha256((root / path).read_bytes()).hexdigest()) != expected]


def leaked_secrets(root: Path) -> list[str]:
    problems = []
    for path in repository_files(root):
        text = (root / path).read_text(errors="ignore")
        if SECRET.search(text):
            problems.append(f"G2 {path}: looks like an API key")
        if HOME_PATH.search(text):
            problems.append(f"G2 {path}: contains a home directory path")
        if any(not email.endswith(ALLOWED_EMAILS) for email in EMAIL.findall(text)):
            problems.append(f"G2 {path}: contains a personal email address")
    return problems


def env_file_tracked(root: Path) -> list[str]:
    tracked = subprocess.run(["git", "ls-files", ".env"], cwd=root, capture_output=True, text=True).stdout.strip()
    return ["G3 .env is tracked by git; remove it from the index"] if tracked else []


def layering(root: Path) -> list[str]:
    problems = []
    for package, forbidden in FORBIDDEN_IMPORTS.items():
        for path in sorted((root / package).glob("*.py")):
            for node in ast.walk(ast.parse(path.read_text())):
                names = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                    [node.module or ""] if isinstance(node, ast.ImportFrom) else []
                for name in names:
                    if name.split(".")[0] in forbidden:
                        problems.append(f"G4 {path.relative_to(root)}: imports {name.split('.')[0]}")
    return problems


def trailing_whitespace(root: Path) -> list[str]:
    return [f"G5 {path}:{number}: trailing whitespace"
            for path in repository_files(root) if path.endswith(WHITESPACE_CHECKED)
            for number, line in enumerate((root / path).read_text(errors="ignore").splitlines(), start=1)
            if line != line.rstrip()]


CHECKS = [frozen_baseline, leaked_secrets, env_file_tracked, layering, trailing_whitespace]


def run_guardrails(root: Path) -> list[str]:
    return [problem for check in CHECKS for problem in check(root)]


if __name__ == "__main__":
    found = run_guardrails(Path(__file__).resolve().parent.parent)
    print("\n".join(found) or "guardrails: ok")
    sys.exit(1 if found else 0)
