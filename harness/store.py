import hashlib, json, os, shutil, tempfile, time, uuid
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


def new_run_id() -> str:
    return f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"


def run_dir_for(run_id: str) -> Path:
    if not run_id.replace("-", "").isalnum():
        raise ValueError(f"invalid run id {run_id!r}")
    return private_root() / run_id


def new_run_dir() -> Path:
    return run_dir_for(new_run_id())


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

    def save_file(self, path: str) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as source:
            for chunk in iter(lambda: source.read(1 << 20), b""):
                digest.update(chunk)
        artifact_id = "a" + digest.hexdigest()[:11]
        target = self.directory / f"{artifact_id}.txt"
        if target.exists():
            os.remove(path)
        else:
            shutil.move(path, target)
        return artifact_id

    def exists(self, artifact_id: str) -> bool:
        return (self.directory / f"{artifact_id}.txt").is_file()

    def read(self, artifact_id: str, offset: int = 1, limit: int = 200) -> str | None:
        if not artifact_id.isalnum() or not self.exists(artifact_id):
            return None
        shown, total = [], 0
        with open(self.directory / f"{artifact_id}.txt", errors="replace") as source:
            for total, line in enumerate(source, start=1):
                if offset <= total < offset + limit:
                    shown.append(line.rstrip("\n"))
        return "\n".join(shown) + f"\n[artifact {artifact_id}: lines {offset}-{offset + len(shown) - 1} of {total}]"
