# AGENTS.md

Test task for Loola: extend a minimal coding-agent harness (`agent.py`) with
dynamic tool/skill loading and context compaction, then benchmark before/after.

## Run
- `cp .env.example .env` and fill in an OpenAI-compatible endpoint.
- `uv run --env-file .env agent.py "<task>"`

## Rules
- Python 3.12, deps via `uv add` only. Keep the dependency list minimal (stdlib first).
- The first commit holds the unmodified baseline harness; measure "before" against it.
- Keep the harness a single readable loop; add modules only when a mechanism earns one.
- Never commit `.env` or API keys.
- Record decisions (what, why, evidence) in `docs/DEVLOG.md` as you make them.
