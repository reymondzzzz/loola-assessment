import csv, json, sqlite3, subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class ToolSpec:
    name: str
    summary: str
    description: str
    parameters: dict
    handler: Callable[..., str] | None
    keywords: tuple[str, ...] = ()
    read_only: bool = True
    idempotent: bool = True
    timeout: float = 60
    version: str = "1"
    max_output_tokens: int = 2000


def schema(properties: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": properties, "required": required}


def string(description: str) -> dict:
    return {"type": "string", "description": description}


def integer(description: str) -> dict:
    return {"type": "integer", "description": description}


class CommandResult(str):
    exit_code: int


def run(command: list[str]) -> CommandResult:
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    output = CommandResult(f"Exit code: {result.returncode}\n{result.stdout}"
                           + (f"\nSTDERR:\n{result.stderr}" if result.stderr else ""))
    output.exit_code = result.returncode
    return output


def read_file(path: str, offset: int = 1, limit: int = 200) -> str:
    lines = Path(path).read_text(errors="replace").splitlines()
    shown = lines[offset - 1:offset - 1 + limit]
    footer = f"\n[lines {offset}-{offset + len(shown) - 1} of {len(lines)}]"
    return "\n".join(f"{number:>6}  {line}" for number, line in enumerate(shown, start=offset)) + footer


def write_file(path: str, content: str) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(content)
    return f"Wrote {len(content)} characters to {path}"


def edit_file(path: str, old: str, new: str) -> str:
    text = Path(path).read_text()
    count = text.count(old)
    if count != 1:
        return f"Error: `old` must occur exactly once in {path}, found {count} occurrences"
    Path(path).write_text(text.replace(old, new))
    return f"Edited {path}"


def list_dir(path: str = ".") -> str:
    return "\n".join(sorted(f"{entry.name}/" if entry.is_dir() else entry.name for entry in Path(path).iterdir()))


def grep(pattern: str, path: str = ".") -> str:
    return run(["grep", "-rnE", "--exclude-dir=.git", pattern, path])


def run_tests(path: str = "tests") -> str:
    return run(["python", "-m", "pytest", "-q", path])


def sqlite_query(database: str, sql: str) -> str:
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as db:
        cursor = db.execute(sql)
        rows = cursor.fetchmany(200)
        header = "\t".join(column[0] for column in cursor.description or [])
    return "\n".join([header, *("\t".join(map(str, row)) for row in rows)])


def json_query(path: str, pointer: str = "") -> str:
    node = json.loads(Path(path).read_text())
    for key in filter(None, pointer.split(".")):
        node = node[int(key)] if isinstance(node, list) else node[key]
    return json.dumps(node, indent=2)[:8000]


def csv_preview(path: str, rows: int = 10) -> str:
    with open(path, newline="", encoding="utf-8-sig") as f:
        records = list(csv.reader(f))
    return "\n".join([f"{len(records) - 1} data rows (including blank lines)", *map(repr, records[:rows + 1])])


def git_log(path: str = "", max_count: int = 30) -> str:
    return run(["git", "log", "--oneline", f"--max-count={max_count}", *(["--", path] if path else [])])


def git_show(revision: str, path: str = "") -> str:
    return run(["git", "show", revision, *(["--", path] if path else [])])


def git_diff(revision_range: str = "", path: str = "") -> str:
    return run(["git", "diff", *([revision_range] if revision_range else []), *(["--", path] if path else [])])


WORKING_TOOLS = [
    ToolSpec("read_file", "Read a text file with line numbers.",
             "Read a text file and return its lines prefixed with line numbers. Use offset and limit to page "
             "through large files instead of reading them whole.",
             schema({"path": string("File path, relative to the working directory."),
                     "offset": integer("First line to return, 1-based. Default 1."),
                     "limit": integer("Maximum number of lines to return. Default 200.")}, ["path"]), read_file, keywords=("open", "view", "cat", "lines", "page")),
    ToolSpec("write_file", "Create or overwrite a file.",
             "Write content to a file, creating parent directories as needed. Overwrites existing files.",
             schema({"path": string("File path."), "content": string("Full new file content.")}, ["path", "content"]),
             write_file, read_only=False, keywords=("create", "save", "overwrite")),
    ToolSpec("edit_file", "Replace one exact snippet in a file.",
             "Replace an exact, unique occurrence of `old` with `new` in a file. Fails if `old` occurs zero or "
             "several times, so include enough surrounding context to make it unique.",
             schema({"path": string("File path."), "old": string("Exact text to replace; must be unique."),
                     "new": string("Replacement text.")}, ["path", "old", "new"]), edit_file, read_only=False, idempotent=False, keywords=("modify", "patch", "replace", "fix")),
    ToolSpec("list_dir", "List directory entries.", "List the entries of a directory; directories end with '/'.",
             schema({"path": string("Directory path. Default '.'.")}, []), list_dir, keywords=("ls", "files", "folder", "tree")),
    ToolSpec("grep", "Search files with a regular expression.",
             "Recursively search files for an extended regular expression and return matching lines with file "
             "names and line numbers. The .git directory is skipped.",
             schema({"pattern": string("Extended regular expression."),
                     "path": string("File or directory to search. Default '.'.")}, ["pattern"]), grep, keywords=("find", "search", "regex", "pattern")),
    ToolSpec("run_tests", "Run pytest on a path.", "Run `python -m pytest -q` on a file or directory and return the output.",
             schema({"path": string("Test file or directory. Default 'tests'.")}, []), run_tests, keywords=("pytest", "unit", "failing", "verify")),
    ToolSpec("sqlite_query", "Run a read-only SQL query on a SQLite database.",
             "Execute one SQL statement against a SQLite database opened read-only and return up to 200 rows as "
             "tab-separated values with a header row.",
             schema({"database": string("Path to the .db file."), "sql": string("SQL statement to execute.")},
                    ["database", "sql"]), sqlite_query, keywords=("sql", "database", "db", "select", "table")),
    ToolSpec("json_query", "Extract part of a JSON file.",
             "Load a JSON file and return the node at a dotted pointer such as `suppliers.0.catalog` "
             "(list indexes are numbers). An empty pointer returns the whole document (truncated to 8000 chars).",
             schema({"path": string("JSON file path."), "pointer": string("Dotted pointer; empty for the root.")},
                    ["path"]), json_query, keywords=("nested", "jq", "field", "pointer")),
    ToolSpec("csv_preview", "Show the header and first rows of a CSV file.",
             "Parse a CSV file with standard quoting (UTF-8 BOM tolerated) and show the row count, the header "
             "and the first rows as Python lists, so quoting and whitespace are visible.",
             schema({"path": string("CSV file path."), "rows": integer("Number of data rows to show. Default 10.")},
                    ["path"]), csv_preview, keywords=("spreadsheet", "rows", "columns", "header")),
    ToolSpec("git_log", "Show commit history.", "Show one-line git log, optionally limited to commits touching a path.",
             schema({"path": string("Limit to commits that touch this path."),
                     "max_count": integer("Maximum number of commits. Default 30.")}, []), git_log, keywords=("commit", "history", "regression", "changes")),
    ToolSpec("git_show", "Show a commit and its diff.", "Show a commit's message and diff, optionally limited to one path.",
             schema({"revision": string("Commit hash or revision expression."),
                     "path": string("Limit the diff to this path.")}, ["revision"]), git_show, keywords=("commit", "diff", "patch", "regression")),
    ToolSpec("git_diff", "Show differences between revisions or the working tree.",
             "Show `git diff` for a revision range (e.g. `abc123..HEAD`) or for uncommitted changes when empty.",
             schema({"revision_range": string("Revision or range; empty for working-tree changes."),
                     "path": string("Limit to this path.")}, []), git_diff, keywords=("changes", "compare", "uncommitted")),
]

UNAVAILABLE_TOOLS = [
    ToolSpec("http_get", "Fetch a URL over HTTP.", "Perform an HTTP GET request and return status, headers and body.",
             schema({"url": string("Absolute URL."), "headers": {"type": "object", "description": "Extra request headers."},
                     "timeout_seconds": integer("Request timeout.")}, ["url"]), None),
    ToolSpec("github_search_issues", "Search GitHub issues and pull requests.",
             "Search issues and pull requests in a GitHub repository using GitHub search syntax.",
             schema({"repo": string("owner/name."), "query": string("GitHub search query."),
                     "state": {"type": "string", "enum": ["open", "closed", "all"], "description": "Issue state."},
                     "limit": integer("Maximum results.")}, ["repo", "query"]), None),
    ToolSpec("github_create_pull_request", "Open a GitHub pull request.",
             "Create a pull request from a head branch into a base branch, optionally as a draft.",
             schema({"repo": string("owner/name."), "title": string("PR title."), "body": string("PR description, Markdown."),
                     "head": string("Source branch."), "base": string("Target branch."),
                     "draft": {"type": "boolean", "description": "Open as draft."}}, ["repo", "title", "head", "base"]), None, read_only=False, idempotent=False),
    ToolSpec("slack_post_message", "Post a message to Slack.", "Post a message to a Slack channel or thread.",
             schema({"channel": string("Channel ID or name."), "text": string("Message text, Slack mrkdwn."),
                     "thread_ts": string("Parent message timestamp to reply in a thread.")}, ["channel", "text"]), None, read_only=False, idempotent=False),
    ToolSpec("docker_build", "Build a Docker image.", "Build a Docker image from a build context and tag it.",
             schema({"context": string("Build context directory."), "tag": string("Image tag."),
                     "dockerfile": string("Dockerfile path."), "build_args": {"type": "object", "description": "Build arguments."}},
                    ["context", "tag"]), None, read_only=False, idempotent=False),
    ToolSpec("docker_run", "Run a command in a Docker container.", "Run a container from an image and return its output.",
             schema({"image": string("Image reference."), "command": string("Command to run."),
                     "env": {"type": "object", "description": "Environment variables."},
                     "ports": {"type": "array", "items": {"type": "string"}, "description": "Port mappings host:container."}},
                    ["image"]), None, read_only=False, idempotent=False),
    ToolSpec("k8s_get_pods", "List Kubernetes pods.", "List pods in a namespace with status, restarts and age.",
             schema({"namespace": string("Namespace."), "selector": string("Label selector, e.g. app=api.")}, ["namespace"]), None),
    ToolSpec("k8s_apply", "Apply a Kubernetes manifest.", "Apply a YAML manifest to the cluster, optionally as a dry run.",
             schema({"manifest": string("YAML manifest content."), "namespace": string("Namespace."),
                     "dry_run": {"type": "boolean", "description": "Server-side dry run only."}}, ["manifest"]), None, read_only=False, idempotent=False),
    ToolSpec("sentry_list_issues", "List Sentry issues for a project.",
             "List error issues reported to Sentry for a project, filtered by query and status.",
             schema({"project": string("Sentry project slug."), "query": string("Sentry search query."),
                     "status": {"type": "string", "enum": ["unresolved", "resolved", "ignored"], "description": "Issue status."}},
                    ["project"]), None),
    ToolSpec("jira_create_issue", "Create a Jira issue.", "Create a Jira issue with summary, description, type and priority.",
             schema({"project": string("Project key."), "summary": string("One-line summary."),
                     "description": string("Issue description."),
                     "issue_type": {"type": "string", "enum": ["Bug", "Task", "Story"], "description": "Issue type."},
                     "priority": {"type": "string", "enum": ["Low", "Medium", "High"], "description": "Priority."}},
                    ["project", "summary"]), None, read_only=False, idempotent=False),
    ToolSpec("s3_upload", "Upload a file to S3.", "Upload a local file to an S3 bucket under a key.",
             schema({"bucket": string("Bucket name."), "key": string("Object key."), "path": string("Local file path."),
                     "content_type": string("MIME type.")}, ["bucket", "key", "path"]), None, read_only=False, idempotent=False),
    ToolSpec("postgres_query", "Run a SQL query on PostgreSQL.", "Execute a parameterised SQL query against a PostgreSQL database.",
             schema({"dsn": string("Connection string."), "sql": string("SQL with $1-style placeholders."),
                     "params": {"type": "array", "items": {}, "description": "Query parameters."}}, ["dsn", "sql"]), None),
    ToolSpec("send_email", "Send an email.", "Send a plain-text email to one or more recipients.",
             schema({"to": {"type": "array", "items": {"type": "string"}, "description": "Recipient addresses."},
                     "subject": string("Subject line."), "body": string("Plain-text body."),
                     "cc": {"type": "array", "items": {"type": "string"}, "description": "CC addresses."}},
                    ["to", "subject", "body"]), None, read_only=False, idempotent=False),
]

CATALOG = {spec.name: spec for spec in WORKING_TOOLS + UNAVAILABLE_TOOLS}
