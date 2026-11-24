import hashlib, json, os, random, re, signal, sqlite3, subprocess, tempfile, time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass, field

from harness.catalog import ToolSpec
from harness.registry import BASH_TOOL, META_HANDLERS, META_TOOL_SCHEMAS, TOOLS, validation_error

BASH_TIMEOUT = 120
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


def failure(code: str, message: str, *, model_can_repair: bool = True, retry_same_call: bool = False) -> ToolOutcome:
    payload = {"ok": False, "code": code, "message": message, "retry_same_call": retry_same_call,
               "model_can_repair": model_can_repair}
    return ToolOutcome(False, json.dumps(payload), code, model_can_repair)


def run_bash(command: str, timeout: float = BASH_TIMEOUT) -> ToolOutcome:
    if FORBIDDEN_COMMANDS.search(command):
        return failure("PERMISSION_DENIED", "This command is not permitted in this session. Do not retry it; "
                       "find another way or report that it is needed.", model_can_repair=False)
    with tempfile.TemporaryFile("w+", errors="replace") as stdout, tempfile.TemporaryFile("w+", errors="replace") as stderr:
        process = subprocess.Popen(command, shell=True, stdout=stdout, stderr=stderr, text=True, start_new_session=True)
        try:
            exit_code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            outcome = failure("TIMEOUT", f"Command exceeded {timeout:.0f}s and was killed. It may have partially run "
                              "(files written, processes started). Inspect the current state before retrying.")
            outcome.changed_environment = True
            return outcome
        stdout.seek(0)
        stderr.seek(0)
        out, err = stdout.read(), stderr.read()
    content = f"Exit code: {exit_code}\n{out}" + (f"\nSTDERR:\n{err}" if err else "")
    return ToolOutcome(True, content, "OK" if exit_code == 0 else "NONZERO_EXIT", changed_environment=True)


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
            return ToolOutcome(True, call_with_timeout(spec, arguments), changed_environment=not spec.read_only)
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


@dataclass
class ToolExecutor:
    available: set[str]
    loop_limit: int = 3
    environment_version: int = 0
    failures: dict[str, int] = field(default_factory=dict)

    def call_key(self, name: str, arguments: dict) -> str:
        normalized = json.dumps(arguments, sort_keys=True)
        return hashlib.sha256(f"{name}|{normalized}|{self.environment_version}".encode()).hexdigest()

    def execute(self, name: str, raw_arguments: str) -> ToolOutcome:
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
        repeats = self.failures.get(key, 0)
        if repeats >= self.loop_limit:
            return failure("LOOP_DETECTED", f"This exact call already failed {repeats} times with no change in between. "
                           "Do not repeat it; try a different approach.")
        outcome = self.dispatch(name, arguments)
        if outcome.changed_environment:
            self.environment_version += 1
        if not (outcome.ok and outcome.code == "OK"):
            self.failures[key] = repeats + 1
        return outcome

    def dispatch(self, name: str, arguments: dict) -> ToolOutcome:
        if name == "bash":
            if (problem := validation_error(arguments, BASH_TOOL["function"]["parameters"])) is not None:
                return failure("INVALID_ARGUMENTS", problem)
            return run_bash(arguments["command"])
        if name == "invoke":
            target = arguments.get("name")
            if target not in TOOLS.specs:
                return failure("UNKNOWN_TOOL", f"Unknown catalog tool '{target}'. Use discover(query).")
            return run_catalog_tool(TOOLS.specs[target], arguments.get("arguments") or {})
        if name in META_HANDLERS:
            if (problem := validation_error(arguments, META_TOOL_SCHEMAS[name]["function"]["parameters"])) is not None:
                return failure("INVALID_ARGUMENTS", problem)
            return ToolOutcome(True, META_HANDLERS[name](**arguments))
        return run_catalog_tool(TOOLS.specs[name], arguments)
