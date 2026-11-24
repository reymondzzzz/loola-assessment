import hashlib, json, os, random, re, shutil, signal, sqlite3, subprocess, tempfile, time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from typing import Callable

from harness.catalog import ToolSpec
from harness.registry import BASH_TOOL, META_HANDLERS, META_TOOL_SCHEMAS, TOOLS, validation_error

BASH_TIMEOUT = 120
CAPTURE_HEAD_BYTES = CAPTURE_TAIL_BYTES = 128_000
FINGERPRINT_SKIP_DIRS = {".git", "__pycache__", ".venv", "node_modules", ".pytest_cache"}
FINGERPRINT_MAX_FILES = 20_000
REPEAT_HINT_AFTER = 2
TRANSIENT_ERRORS = (sqlite3.OperationalError, ConnectionError, TimeoutError, BlockingIOError)
TRANSIENT_RETRIES = 2
FORBIDDEN_COMMANDS = re.compile(r"(^|[;&|]\s*)(sudo|shutdown|reboot|mkfs|git\s+push)\b|rm\s+-rf\s+(/|~)(\s|$)|:\(\)\s*\{")


@dataclass
class ToolOutcome:
    ok: bool
    content: str
    code: str = "OK"
    model_can_repair: bool = True
    changed_environment: bool = False
    spill_path: str | None = None


def failure(code: str, message: str, *, model_can_repair: bool = True, retry_same_call: bool = False) -> ToolOutcome:
    payload = {"ok": False, "code": code, "message": message, "retry_same_call": retry_same_call,
               "model_can_repair": model_can_repair}
    return ToolOutcome(False, json.dumps(payload), code, model_can_repair)


def read_bounded(stream) -> tuple[str, bool]:
    size = stream.seek(0, os.SEEK_END)
    stream.seek(0)
    if size <= CAPTURE_HEAD_BYTES + CAPTURE_TAIL_BYTES:
        return stream.read().decode(errors="replace"), False
    head = stream.read(CAPTURE_HEAD_BYTES)
    stream.seek(size - CAPTURE_TAIL_BYTES)
    omitted = size - CAPTURE_HEAD_BYTES - CAPTURE_TAIL_BYTES
    marker = f"\n[... {omitted} bytes not loaded; the full output is saved as an artifact ...]\n".encode()
    return (head + marker + stream.read()).decode(errors="replace"), True


def spill_full_output(exit_code: int, stdout, stderr) -> str:
    with tempfile.NamedTemporaryFile("wb", delete=False, prefix="bash-output-") as full:
        full.write(f"Exit code: {exit_code}\n".encode())
        stdout.seek(0)
        shutil.copyfileobj(stdout, full)
        if stderr.seek(0, os.SEEK_END):
            stderr.seek(0)
            full.write(b"\nSTDERR:\n")
            shutil.copyfileobj(stderr, full)
    return full.name


def workspace_fingerprint() -> str | None:
    digest, files = hashlib.sha256(), 0
    for root, dirs, names in os.walk("."):
        dirs[:] = sorted(d for d in dirs if d not in FINGERPRINT_SKIP_DIRS)
        for name in sorted(names):
            files += 1
            if files > FINGERPRINT_MAX_FILES:
                return None
            try:
                status = os.stat(os.path.join(root, name))
            except OSError:
                continue
            digest.update(f"{root}/{name}:{status.st_mtime_ns}:{status.st_size}\n".encode())
    return digest.hexdigest()


def run_bash(command: str, timeout: float = BASH_TIMEOUT) -> ToolOutcome:
    if FORBIDDEN_COMMANDS.search(command):
        return failure("PERMISSION_DENIED", "This command is not permitted in this session. Do not retry it; "
                       "find another way or report that it is needed.", model_can_repair=False)
    before = workspace_fingerprint()
    with tempfile.TemporaryFile("w+b") as stdout, tempfile.TemporaryFile("w+b") as stderr:
        process = subprocess.Popen(command, shell=True, stdout=stdout, stderr=stderr, start_new_session=True)
        try:
            exit_code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            outcome = failure("TIMEOUT", f"Command exceeded {timeout:.0f}s and was killed. It may have partially run "
                              "(files written, processes started). Inspect the current state before retrying.")
            outcome.changed_environment = True
            return outcome
        (out, out_cut), (err, err_cut) = read_bounded(stdout), read_bounded(stderr)
        spill_path = spill_full_output(exit_code, stdout, stderr) if out_cut or err_cut else None
    content = f"Exit code: {exit_code}\n{out}" + (f"\nSTDERR:\n{err}" if err else "")
    changed = before is None or workspace_fingerprint() != before
    return ToolOutcome(True, content, "OK" if exit_code == 0 else "NONZERO_EXIT", changed_environment=changed,
                       spill_path=spill_path)


def permission_error(spec: ToolSpec) -> ToolOutcome | None:
    if spec.handler is None and not spec.read_only:
        return failure("PERMISSION_DENIED", f"{spec.name} has external side effects and is not permitted in this session. "
                       "Do not retry it.", model_can_repair=False)
    return None


def call_with_timeout(spec: ToolSpec, arguments: dict) -> str:
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(spec.handler, **arguments).result(timeout=spec.timeout)


def run_catalog_tool(spec: ToolSpec, arguments: dict) -> ToolOutcome:
    if (problem := validation_error(arguments, spec.parameters)) is not None:
        return failure("INVALID_ARGUMENTS", f"{problem}. Parameters: {json.dumps(spec.parameters)}")
    if (denied := permission_error(spec)) is not None:
        return denied
    if spec.handler is None:
        return failure("NOT_AVAILABLE", f"{spec.name} has no backend in this environment. Use another tool or bash.")
    attempts = TRANSIENT_RETRIES + 1 if spec.read_only and spec.idempotent else 1
    for attempt in range(attempts):
        try:
            result = call_with_timeout(spec, arguments)
            code = "NONZERO_EXIT" if getattr(result, "exit_code", 0) else "OK"
            return ToolOutcome(True, str(result), code, changed_environment=not spec.read_only)
        except FileNotFoundError as error:
            return failure("FILE_NOT_FOUND", f"{error}. Check the path with list_dir or bash `ls`.")
        except FutureTimeout:
            message = f"{spec.name} exceeded {spec.timeout:.0f}s."
            if not spec.read_only:
                outcome = failure("TIMEOUT", message + " It may have partially applied; inspect before retrying.")
                outcome.changed_environment = True
                return outcome
            if attempt == attempts - 1:
                return failure("TIMEOUT", message)
        except TRANSIENT_ERRORS as error:
            if attempt == attempts - 1 or "readonly" in str(error):
                return failure("TOOL_ERROR", f"{type(error).__name__}: {error}")
            time.sleep(0.2 * 2 ** attempt + random.uniform(0, 0.1))
        except Exception as error:
            return failure("TOOL_ERROR", f"{type(error).__name__}: {error}")
    raise AssertionError("unreachable")


def repeat_is_safe(name: str, arguments: dict) -> bool:
    if name == "invoke":
        name = arguments.get("name") if isinstance(arguments.get("name"), str) else ""
    elif name in META_HANDLERS or name in ("read_artifact", "search_history"):
        return True
    spec = TOOLS.specs.get(name)
    return spec is not None and spec.read_only and spec.idempotent


def outcome_unknown() -> ToolOutcome:
    outcome = failure("OUTCOME_UNKNOWN", "The harness process stopped while this call was running, so it is unknown "
                      "whether it completed. It was not repeated automatically. Inspect the current state (files, "
                      "processes) before deciding whether to run it again.")
    outcome.changed_environment = True
    return outcome


def failure_summary(outcome: ToolOutcome) -> str:
    if not outcome.ok:
        return json.loads(outcome.content)["message"][:160]
    lines = [line for line in outcome.content.splitlines() if line.strip()]
    return " | ".join(lines[:1] + lines[-1:])[:160]


def add_hint(outcome: ToolOutcome, hint: str) -> None:
    if outcome.ok:
        outcome.content += f"\n[Hint: {hint}]"
    else:
        outcome.content = json.dumps({**json.loads(outcome.content), "hint": hint})


@dataclass
class ToolExecutor:
    available: set[str]
    loop_limit: int = 3
    environment_version: int = 0
    failures: dict[str, dict] = field(default_factory=dict)
    extra_tools: dict[str, tuple[dict, Callable[..., str]]] = field(default_factory=dict)

    def call_key(self, name: str, arguments: dict) -> str:
        normalized = json.dumps(arguments, sort_keys=True)
        return hashlib.sha256(f"{name}|{normalized}|{self.environment_version}".encode()).hexdigest()

    def execute(self, name: str, raw_arguments: str, call_id: str = "") -> ToolOutcome:
        try:
            arguments = json.loads(raw_arguments or "{}")
        except json.JSONDecodeError as error:
            return failure("INVALID_JSON", f"Arguments are not valid JSON ({error.msg} at position {error.pos}). "
                           "Resend the call with a valid JSON object.")
        if not isinstance(arguments, dict):
            return failure("INVALID_ARGUMENTS", "Arguments must be a JSON object.")
        if name not in self.available:
            return failure("UNKNOWN_TOOL", f"Tool '{name}' is not available. Available: {', '.join(sorted(self.available))}.")
        key = self.call_key(name, arguments)
        previous = self.failures.get(key)
        repeats = previous["count"] if previous else 0
        if repeats >= self.loop_limit and name != "bash":
            return failure("LOOP_DETECTED", f"This exact call already failed {repeats} times with no change in between. "
                           "Do not repeat it; try a different approach.")
        outcome = self.dispatch(name, arguments)
        if outcome.changed_environment:
            self.environment_version += 1
        if outcome.ok and outcome.code == "OK":
            return outcome
        self.failures[key] = {"count": repeats + 1, "call_id": call_id, "error": failure_summary(outcome)}
        if previous and repeats + 1 >= REPEAT_HINT_AFTER and not outcome.changed_environment:
            add_hint(outcome, f"This exact call has now failed {repeats + 1} times with no observable file change in "
                              f"between (previous attempt: tool call {previous['call_id'] or '?'}, error: "
                              f"{previous['error']}). Change something or try a different approach before repeating it.")
        return outcome

    def dispatch(self, name: str, arguments: dict) -> ToolOutcome:
        if name == "bash":
            if (problem := validation_error(arguments, BASH_TOOL["function"]["parameters"])) is not None:
                return failure("INVALID_ARGUMENTS", problem)
            return run_bash(arguments["command"])
        if name == "invoke":
            if (problem := validation_error(arguments, META_TOOL_SCHEMAS["invoke"]["function"]["parameters"])) is not None:
                return failure("INVALID_ARGUMENTS", problem)
            target = arguments["name"]
            if target not in TOOLS.specs:
                return failure("UNKNOWN_TOOL", f"Unknown catalog tool '{target}'. Use discover(query).")
            return run_catalog_tool(TOOLS.specs[target], arguments["arguments"])
        if name in self.extra_tools:
            schema, handler = self.extra_tools[name]
            if (problem := validation_error(arguments, schema["function"]["parameters"])) is not None:
                return failure("INVALID_ARGUMENTS", problem)
            return ToolOutcome(True, handler(**arguments))
        if name in META_HANDLERS:
            if (problem := validation_error(arguments, META_TOOL_SCHEMAS[name]["function"]["parameters"])) is not None:
                return failure("INVALID_ARGUMENTS", problem)
            return ToolOutcome(True, META_HANDLERS[name](**arguments))
        return run_catalog_tool(TOOLS.specs[name], arguments)
