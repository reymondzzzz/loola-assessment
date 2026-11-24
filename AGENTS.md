# AGENTS.md

Loola test task: a cache-aware LangGraph agent harness with dynamic tool/skill loading, adaptive context
compaction, artifact memory and a before/after benchmark. Overview and results: `README.md`. Decisions and
their evidence: `docs/DEVLOG.md`.

## Layout
- `baseline/` — `original_agent.py` (provided code, byte-identical, SHA-256 pinned in a test; never edit; the
  benchmark injects endpoint config at runtime) and `langgraph_port.py` (behaviour-preserving port, benchmark
  mode `original-port`; never edit).
- `harness/` — the harness. `graph.py` only orchestrates; logic lives in `registry`/`catalog` (tools,
  skills), `executor` (tool execution, errors), `output` (output pipeline), `context` (compaction),
  `store`/`memory` (artifacts, event log, recovery tools), `llm`, `metrics`, `config` (mode presets).
- `skills/<name>/SKILL.md` + `references/` — Agent Skills, loaded progressively.
- `bench/` — ticket-queue benchmark: `generate.py` (project per seed), `tickets.py` (tickets + checks),
  `run.py` (runner + independent scoring), `report.py`, `results.jsonl`, `runs/*.log`.
- `tests/` — deterministic tests: scripted LLM (`tests/fakes.py`) and a local OpenAI-compatible fake server
  (`tests/fake_openai.py`) that drives the real CLI in subprocesses; no external network.

## Commands
- `uv sync`; `cp .env.example .env` and fill an OpenAI-compatible endpoint.
- Tests: `uv run python -m pytest -q tests` (must stay green).
- Agent: `AGENT_MODE=lazy uv run --env-file .env python agent.py "<task>"`; resume with
  `agent.py --resume RUN_ID [--message "new requirement"]`.
- Benchmark: `uv run --env-file .env python bench/run.py --label X --mode hybrid --runs 2 [--context-window N]`,
  then `uv run python bench/report.py`.

## Rules
- Python 3.12, dependencies via `uv add` only; stdlib first, no new dependency for what a few lines do.
- Keep the stable prompt prefix stable: never put changing data (timestamps, counters, loaded-tool lists)
  into the system prompt, `tools[]` or the task message; load things by appending tool results.
- Context edits are transactional and protocol-valid: never split an assistant tool-call message from its
  tool results; never rewrite the original task.
- Never auto-retry bash or non-idempotent tools.
- Benchmark claims need real runs recorded in `bench/results.jsonl`; label synthetic tools and reduced
  context windows; never use the agent's final message as proof of success.
- Never commit `.env`, keys, or personal data. Run directories hold tool outputs: keep them private (0700).
- One commit per logical part; record each decision (what, why, evidence) in `docs/DEVLOG.md` as you go.
