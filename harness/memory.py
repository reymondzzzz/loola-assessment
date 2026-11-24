import json

from harness.catalog import integer, schema, string
from harness.registry import BM25, openai_tool
from harness.store import ArtifactStore, EventStore

MEMORY_TOOL_SCHEMAS = {
    "read_artifact": openai_tool(
        "read_artifact", "Read part of an archived tool output by artifact id (shown in truncated or masked results).",
        schema({"artifact_id": string("Artifact id, e.g. a1b2c3d4e5f6."),
                "offset": integer("First line, 1-based. Default 1."),
                "limit": integer("Maximum lines. Default 200.")}, ["artifact_id"])),
    "search_history": openai_tool(
        "search_history", "Search earlier tool calls and outputs of this session, including ones no longer shown.",
        schema({"query": string("Keywords to look for."), "limit": integer("Maximum results. Default 5.")}, ["query"])),
}
MAX_ARTIFACT_LINES = 400


def read_artifact(artifacts: ArtifactStore, artifact_id: str, offset: int = 1, limit: int = 200) -> str:
    content = artifacts.read(artifact_id, max(offset, 1), min(max(limit, 1), MAX_ARTIFACT_LINES))
    return content if content is not None else f"Error: no artifact '{artifact_id}'."


def search_history(events: EventStore, artifacts: ArtifactStore, query: str, limit: int = 5) -> str:
    results = {event["tool_call_id"]: event for event in events.read() if event["kind"] == "tool_result"}
    if not results:
        return "No history yet."
    index = BM25({call_id: f"{e['name']} {e['arguments']} {e['content']}" for call_id, e in results.items()})
    hits = index.search(query, min(max(limit, 1), 10))
    if not hits:
        return "No matching history."
    lines = []
    for call_id in hits:
        event = results[call_id]
        preview = " ".join(event["content"].split())[:300]
        lines.append(json.dumps({"tool_call_id": call_id, "turn": event["turn"], "tool": event["name"],
                                 "arguments": event["arguments"][:200], "artifact_id": artifacts.save(event["content"]),
                                 "preview": preview}))
    return "\n".join(lines) + "\nUse read_artifact(artifact_id) for full content."
