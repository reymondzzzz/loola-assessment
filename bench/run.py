import argparse, contextlib, hashlib, importlib, json, os, shutil, subprocess, sys, tempfile, time
from unittest import mock
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH_DIR.parent))
from generate import make_workdir  # noqa: E402
from tickets import TICKETS, VISIBLE_TESTS_DIR  # noqa: E402
from baseline import langgraph_port  # noqa: E402
from harness.config import EVICTION_POLICIES, MODES, mode_config  # noqa: E402
from harness.graph import run_agent  # noqa: E402

RUNS_DIR = BENCH_DIR / "runs"
RESULTS_FILE = BENCH_DIR / "results.jsonl"
BENCH_MAX_TURNS = 150
BENCH_CONTEXT_WINDOW = 100_000
PROTECTED_DIRS = ("data", "logs")

TASK = """\
You are working in a Python project in the current directory. The work is organised as tickets.
Run `./tickets next` to get the first ticket. When you have finished a ticket, run `./tickets next` again: it
verifies your work and either shows what is still wrong or gives you the next ticket. `./tickets current` shows the
current ticket again. Do not modify anything in tests/, data/ or logs/.
Keep going until `./tickets next` says all tickets are done, then reply with a short summary."""


def digest_protected(workdir: Path) -> dict[str, str]:
    return {str(path.relative_to(workdir)): hashlib.sha256(path.read_bytes()).hexdigest()
            for folder in PROTECTED_DIRS for path in sorted((workdir / folder).rglob("*")) if path.is_file()}


def install_tickets_command(workdir: Path, state_path: Path) -> None:
    script = workdir / "tickets"
    script.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{BENCH_DIR / "tickets.py"}" "{state_path}" "$@"\n')
    script.chmod(0o755)
    with open(workdir / ".git" / "info" / "exclude", "a") as exclude:
        exclude.write("tickets\n")


def tests_intact(workdir: Path, current_ticket: int) -> bool:
    issued = {ticket.visible_test for ticket in TICKETS[:min(current_ticket, len(TICKETS) - 1) + 1] if ticket.visible_test}
    present = {path.name for path in (workdir / "tests").iterdir() if path.name not in (".gitkeep", "__pycache__")}
    return present == issued and all(
        (VISIBLE_TESTS_DIR / name).read_bytes() == (workdir / "tests" / name).read_bytes() for name in issued)


def score(workdir: Path, state: dict, protected_digest: dict[str, str]) -> dict:
    final_checks = [ticket.check(workdir, state)[0] for ticket in TICKETS]
    tests_untouched = tests_intact(workdir, state["current"])
    data_untouched = digest_protected(workdir) == protected_digest
    return {"tickets_closed": state["closed"], "final_checks_passed": sum(final_checks),
            "final_checks": final_checks, "tests_untouched": tests_untouched, "data_untouched": data_untouched,
            "success": all(final_checks) and tests_untouched and data_untouched}


LITERAL_TIMEOUT_SECONDS = 600


def code_revision() -> str:
    def git(*args):
        return subprocess.run(["git", *args], cwd=BENCH_DIR, capture_output=True, text=True).stdout.strip()
    return git("rev-parse", "--short", "HEAD") + ("-dirty" if git("status", "--porcelain") else "")


def run_literal_original(task: str) -> dict:
    """Runs baseline/original_agent.py unmodified; endpoint config is injected as module attributes."""
    original = importlib.import_module("baseline.original_agent")
    original.LLM_BASE_URL, original.LLM_MODEL = os.environ["LLM_BASE_URL"], os.environ["LLM_MODEL"]
    original.LLM_HEADERS = {"Content-Type": "application/json", "Authorization": f"Bearer {os.environ['LLM_API_KEY']}"}
    original.MAX_TURNS = BENCH_MAX_TURNS
    responses, real_post = [], original.requests.post

    def recording_post(*args, **kwargs):
        response = real_post(*args, timeout=LITERAL_TIMEOUT_SECONDS, **kwargs)
        if response.ok:
            responses.append(response.json())
        return response

    stop_reason = "finished"
    with mock.patch.object(original.requests, "post", recording_post):
        try:
            original.agent_loop(task)
        except Exception as error:  # the baseline crashes on HTTP/JSON errors; record it as the run's outcome
            print(f"\n💥 {type(error).__name__}: {error}")
            stop_reason = f"crash: {type(error).__name__}"
    usage = [response.get("usage") or {} for response in responses]
    calls = [call for response in responses for call in response["choices"][0]["message"].get("tool_calls") or []]
    signatures = [f"{c['function']['name']}:{c['function']['arguments']}" for c in calls]
    if stop_reason == "finished" and calls and responses[-1]["choices"][0]["message"].get("tool_calls"):
        stop_reason = "max_turns"
    input_tokens = sum(u.get("prompt_tokens", 0) for u in usage)
    cached = sum((u.get("prompt_tokens_details") or {}).get("cached_tokens", 0) for u in usage)
    return {"stop_reason": stop_reason, "turns": len(responses), "model_calls": len(responses),
            "input_tokens": input_tokens, "output_tokens": sum(u.get("completion_tokens", 0) for u in usage),
            "cached_tokens": cached, "total_input_tokens": input_tokens, "total_cached_tokens": cached,
            "peak_context": max((u.get("prompt_tokens", 0) for u in usage), default=0), "tool_calls": len(calls),
            "repeated_tool_calls": len(signatures) - len(set(signatures))}


def run_once(label: str, seed: int, mode: str, context_window: int, eviction_policy: str = "default") -> dict:
    workdir, bad_commit = make_workdir(seed)
    state_dir = Path(tempfile.mkdtemp(prefix="bench-state-"))
    state_path = state_dir / "state.json"
    state_path.write_text(json.dumps({"workdir": str(workdir), "current": -1, "closed": 0, "bad_commit": bad_commit}))
    install_tickets_command(workdir, state_path)
    protected_digest = digest_protected(workdir)
    RUNS_DIR.mkdir(exist_ok=True)
    started = time.monotonic()
    original_cwd = os.getcwd()
    os.chdir(workdir)
    with open(RUNS_DIR / f"{label}-seed{seed}.log", "w") as log, contextlib.redirect_stdout(log):
        try:
            if mode == "original":
                metrics = run_literal_original(TASK)
            elif mode == "original-port":
                metrics = langgraph_port.agent_loop(TASK)
            else:
                metrics = run_agent(TASK, mode_config(mode, context_window, eviction_policy=eviction_policy))
        except Exception as error:  # a crashed run is a failed run, not a crashed benchmark
            print(f"\n💥 {type(error).__name__}: {error}")
            metrics = {"stop_reason": f"crash: {type(error).__name__}"}
    os.chdir(original_cwd)
    result = {"label": label, "mode": mode, "code_revision": code_revision(), "context_window": context_window, "eviction_policy": eviction_policy, "model": os.environ["LLM_MODEL"], "seed": seed, "wall_seconds": round(time.monotonic() - started, 1),
              **metrics, **score(workdir, json.loads(state_path.read_text()), protected_digest)}
    shutil.rmtree(workdir)
    shutil.rmtree(state_dir)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--mode", choices=("original", "original-port", *MODES), default="original")
    parser.add_argument("--context-window", type=int, default=BENCH_CONTEXT_WINDOW)
    parser.add_argument("--eviction-policy", choices=EVICTION_POLICIES, default="default")
    args = parser.parse_args()
    langgraph_port.MAX_TURNS = BENCH_MAX_TURNS
    for seed in range(args.first_seed, args.first_seed + args.runs):
        result = run_once(args.label, seed, args.mode, args.context_window, args.eviction_policy)
        with open(RESULTS_FILE, "a") as f:
            f.write(json.dumps(result) + "\n")
        print(json.dumps(result))


if __name__ == "__main__":
    main()
