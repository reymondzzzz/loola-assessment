import re, subprocess, sys
from pathlib import Path

from guardrails import run_guardrails

ROOT = Path(__file__).resolve().parent.parent
MIN_TESTS = 87
README_COUNT = re.compile(r"# (\d+) deterministic tests")


def main() -> int:
    problems = run_guardrails(ROOT)
    result = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"], cwd=ROOT,
                            capture_output=True, text=True)
    summary = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else result.stderr[-500:]
    passed = int(match.group(1)) if (match := re.search(r"(\d+) passed", summary)) else 0
    if result.returncode != 0:
        problems.append(f"T1 tests failed: {summary}")
    if re.search(r"\d+ (skipped|xfailed|xpassed|deselected)", summary):
        problems.append(f"T2 skipped/xfail tests are not allowed: {summary}")
    if passed < MIN_TESTS:
        problems.append(f"T3 {passed} tests passed, fewer than the pinned minimum {MIN_TESTS}; tests went missing")
    readme = README_COUNT.search((ROOT / "README.md").read_text())
    if not readme or int(readme.group(1)) != passed:
        problems.append(f"T4 README states {readme.group(1) if readme else 'no'} tests, the suite has {passed}")
    print("\n".join(problems) or f"check: ok ({passed} tests, guardrails clean)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
