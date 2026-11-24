import hashlib, json, re
from pathlib import Path

from harness.config import HarnessConfig
from harness.registry import TOOLS
from harness.store import ArtifactStore
from harness.tokens import count_tokens

READ_TOOLS = {"read_file", "csv_preview", "json_query"}
WRITE_TOOLS = {"write_file", "edit_file"}
PYTEST_SUMMARY = re.compile(r"^(=+ )?.*\b\d+ (passed|failed|error)s?\b.* in [\d.]+s", re.MULTILINE)
PYTEST_KEEP = re.compile(r"^(FAILED|ERROR|E\s|_{3,}|.*\.py:\d+|=+ (FAILURES|ERRORS|short test summary))")
DIFF_KEEP = re.compile(r"^(commit |Author:|Date:|diff --git|--- |\+\+\+ |@@|[+-]|    \S)")
HEAD_SHARE = 0.3


def effective_call(name: str, arguments: dict) -> tuple[str, dict]:
    if name == "invoke":
        return str(arguments.get("name")), arguments.get("arguments") or {}
    return name, arguments


def fingerprint(path: str) -> str | None:
    try:
        stat = Path(path).stat()
    except OSError:
        return None
    return f"{stat.st_mtime_ns}:{stat.st_size}"


def observation_kind(tool: str, arguments: dict) -> str:
    command = str(arguments.get("command", ""))
    if tool == "run_tests" or (tool == "bash" and "pytest" in command):
        return "test"
    if tool in READ_TOOLS:
        return "read"
    if tool in WRITE_TOOLS:
        return "write"
    if tool in {"discover", "describe", "load_skill", "read_skill_resource"}:
        return "meta"
    return "command" if tool == "bash" else "other"


def pytest_digest(text: str) -> str | None:
    if not PYTEST_SUMMARY.search(text):
        return None
    kept = [line for line in text.splitlines() if PYTEST_KEEP.match(line) or PYTEST_SUMMARY.match(line)]
    return "\n".join(kept)


def diff_digest(text: str) -> str | None:
    if "diff --git" not in text:
        return None
    return "\n".join(line for line in text.splitlines() if DIFF_KEEP.match(line) and not line.startswith("index "))


def filtered(kind: str, tool: str, text: str) -> str:
    if kind == "test":
        return pytest_digest(text) or text
    if tool in {"git_show", "git_diff", "bash"}:
        return diff_digest(text) or text
    return text


def head_tail(text: str, budget: int, artifact_id: str, original_tokens: int) -> str:
    if count_tokens(text) <= budget:
        return text
    minimal = f"[truncated; read_artifact('{artifact_id}')]"
    lines = text.splitlines()
    if len(lines) < 3:
        lines = [text[i:i + 200] for i in range(0, len(text), 200)]
    reference = (f"[... {{omitted}} of {len(lines)} lines omitted; original {original_tokens} tokens. "
                 f"Full output: read_artifact('{artifact_id}', offset, limit) ...]")
    content_budget = budget - count_tokens(reference.format(omitted=len(lines))) - 2
    if content_budget < 8:
        return minimal
    head, used = [], 0
    for line in lines:
        cost = count_tokens(line) + 1
        if used + cost > content_budget * HEAD_SHARE:
            break
        head.append(line)
        used += cost
    tail = []
    for line in reversed(lines[len(head):]):
        cost = count_tokens(line) + 1
        if used + cost > content_budget:
            break
        tail.insert(0, line)
        used += cost
    while True:
        result = "\n".join([*head, reference.format(omitted=len(lines) - len(head) - len(tail)), *tail])
        if count_tokens(result) <= budget or not (head or tail):
            return result if count_tokens(result) <= budget else minimal
        (tail.pop(0) if len(tail) >= len(head) else head.pop())


def per_call_budgets(actions: list[dict], config: HarnessConfig, available_tokens: int) -> list[int]:
    limits = [TOOLS.specs[a["tool"]].max_output_tokens if a["tool"] in TOOLS.specs else config.tool_output_tokens
              for a in actions]
    turn_budget = min(sum(limits), max(int(available_tokens * 0.2), 256))
    share = max(turn_budget // max(len(actions), 1), 64)
    return [min(limit, share) for limit in limits]


class OutputProcessor:
    def __init__(self, config: HarnessConfig, artifacts: ArtifactStore):
        self.config, self.artifacts = config, artifacts

    def process(self, actions: list[dict], observations: dict, turn: int, available_tokens: int) -> tuple[list[dict], dict]:
        observations = dict(observations)
        visible_digests = {obs["digest"]: obs for obs in observations.values()
                           if not obs["masked"] and obs["duplicate_of"] is None}
        prepared = []
        for action in actions:
            tool, arguments = effective_call(action["name"], action["parsed_arguments"])
            prepared.append({**action, "tool": tool, "tool_arguments": arguments})
        budgets = per_call_budgets(prepared, self.config, available_tokens)
        processed = []
        for action, budget in zip(prepared, budgets):
            content, tool, arguments = action["content"], action["tool"], action["tool_arguments"]
            kind = observation_kind(tool, arguments)
            digest = hashlib.sha256(content.encode()).hexdigest()
            if action.get("spill_path"):
                artifact_id = self.artifacts.save_file(action["spill_path"])
                original_tokens = (self.artifacts.directory / f"{artifact_id}.txt").stat().st_size // 4
            else:
                artifact_id = self.artifacts.save(content)
                original_tokens = count_tokens(content)
            visible, duplicate_of = content, None
            if action["ok"]:
                if self.config.deduplication and digest in visible_digests:
                    earlier = visible_digests[digest]
                    duplicate_of = earlier["message_id"]
                    visible = (f"[Identical to the output of tool call {earlier['message_id']} (turn "
                               f"{earlier['created_at_turn']}), which is still above. artifact {artifact_id}]")
                elif self.config.output_truncation:
                    visible = head_tail(filtered(kind, tool, content), budget, artifact_id, original_tokens)
            path = arguments.get("path") if tool in READ_TOOLS | WRITE_TOOLS else None
            observation = {"message_id": action["id"], "artifact_id": artifact_id, "tool_name": tool, "kind": kind,
                           "token_count": count_tokens(visible), "original_tokens": original_tokens,
                           "arguments_tokens": count_tokens(action["arguments"]),
                           "created_at_turn": turn, "last_access_turn": turn, "access_count": 1,
                           "resource_path": path, "resource_version": fingerprint(path) if path else None,
                           "pinned": tool == "load_skill", "stale": False, "recoverable": True,
                           "digest": digest, "masked": False, "duplicate_of": duplicate_of, "ok": action["ok"], "code": action["code"],
                           "summary": f"{tool} {json.dumps(arguments)[:120]}"}
            observations[action["id"]] = observation
            if duplicate_of is None:
                visible_digests.setdefault(digest, observation)
            processed.append({**action, "visible": visible})
        return processed, observations
