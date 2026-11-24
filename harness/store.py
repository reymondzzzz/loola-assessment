import hashlib, json, os, tempfile, time, uuid
from pathlib import Path

RUNS_ROOT = Path(os.environ.get("AGENT_RUNS_DIR", Path(tempfile.gettempdir()) / "agent-runs"))


class EventStore:

    def __init__(self, run_dir: Path):
        run_dir.mkdir(parents=True, exist_ok=True)
        self.path = run_dir / "events.jsonl"

    def append(self, kind: str, **fields) -> dict:
        event = {"id": uuid.uuid4().hex[:12], "kind": kind, "time": round(time.time(), 3), **fields}
        with open(self.path, "a") as log:
            log.write(json.dumps(event) + "\n")
        return event

    def read(self) -> list[dict]:
        return [json.loads(line) for line in self.path.read_text().splitlines()] if self.path.exists() else []


def private_root() -> Path:
    RUNS_ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
    status = RUNS_ROOT.stat()
    if status.st_uid != os.getuid() or status.st_mode & 0o077:
        raise PermissionError(f"{RUNS_ROOT} must be owned by the current user with mode 0700 (tool outputs may hold secrets)")
    return RUNS_ROOT


def new_run_dir() -> Path:
    return private_root() / f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"


class ArtifactStore:
    def __init__(self, run_dir: Path):
        self.directory = run_dir / "artifacts"
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(self, content: str) -> str:
        artifact_id = "a" + hashlib.sha256(content.encode()).hexdigest()[:11]
        path = self.directory / f"{artifact_id}.txt"
        if not path.exists():
            path.write_text(content)
        return artifact_id

    def exists(self, artifact_id: str) -> bool:
        return (self.directory / f"{artifact_id}.txt").is_file()

    def read(self, artifact_id: str, offset: int = 1, limit: int = 200) -> str | None:
        if not artifact_id.isalnum() or not self.exists(artifact_id):
            return None
        lines = (self.directory / f"{artifact_id}.txt").read_text().splitlines()
        shown = lines[offset - 1:offset - 1 + limit]
        return "\n".join(shown) + f"\n[artifact {artifact_id}: lines {offset}-{offset + len(shown) - 1} of {len(lines)}]"
