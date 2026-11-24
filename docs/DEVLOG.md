# Dev log — decisions

Development was done with Claude Code (agent) under my direction. This log keeps the decisions and their
reasons, not the full trajectory. Newest entries at the bottom.

## 1. Project setup
- Own git repo, Python 3.12, `uv`. Endpoint/key/model come from env (`.env`, git-ignored) instead of the
  hard-coded placeholders.
- First commit (`8b5e4ae`) is the original harness unchanged apart from env config — the reference point.

## 2. Port to LangGraph, behaviour-preserving
- Requirement: same logic and same bugs as the original loop.
- Graph: `llm → tools → llm …`, with `finish` and `max_turns` terminal nodes; `recursion_limit = 2*MAX_TURNS+2`
  so LangGraph never stops before the harness's own turn limit.
- Messages stay plain OpenAI dicts (no `add_messages`): LangChain message objects would silently change what
  the model receives — and later both compaction and prompt caching depend on exact bytes.
- Verified by replaying scripted LLM replies through both versions and diffing stdout + every request sent
  (multi tool calls, no-text finish, unknown tool, max turns, bad-JSON crash). Full 1000-turn run checked.

## 3. LLM client: `ChatOpenAI` (langchain-openai)
- Chosen over raw `requests` at the user's request; stays OpenAI-compatible (`base_url`).
- `ChatOpenAI` parses tool calls and drops the raw ones, so `call_llm` rebuilds OpenAI-format tool calls from
  `tool_calls` + `invalid_tool_calls`. This keeps the graph unchanged and keeps the bad-JSON crash.
- `max_retries=0` to keep the baseline's "any HTTP error kills the run" behaviour (to be fixed deliberately).
- Model: `gpt-6-luna`. It rejects `temperature`, so it was removed; it returns content blocks, so text is read
  via `msg.text`.

## 4. Metrics
- Per LLM call: input, output, cached tokens (`usage_metadata`). Per run: totals, peak context (max input
  tokens of one call), turns, stop reason, wall time.

## 5. Benchmark v1 rejected — too easy
- Three-part task (fix bugs, analyse log, add CLI). Baseline: 3/3 success, **4 turns, ~8k peak context**.
  The model read everything in one command and fixed everything in one heredoc.
- Nothing to measure: compaction never triggers, tool loading saves almost nothing.

## 6. Benchmark v2 — ticket queue with edge cases
- `./tickets next` reveals the next ticket only after the current one passes its checks. Reason: a single
  task lets the model batch everything; a gated queue guarantees a long, multi-step run regardless of style.
- 11 tickets: pricing, store, log analysis, tie-breaking, CSV CLI, SQLite (UTC offsets, boundaries),
  nested JSON, cross-file rename, 1500-line legacy file, git regression hunt, recall test.
- Cross-ticket dependencies (4←3, 7←2, 11←1) so lost context shows up as failures. Ticket 11 asks for a
  fact stated only once in ticket 1 (ORCHID-7), with a decoy (ORCHID-5) later — a direct compaction test.
- Edge cases baked into data and hidden tests: CRLF, stack traces containing `ERROR [payments]`,
  `auth` vs `auth-legacy`, rigged ties, BOM/quoted commas/blank lines in CSV, `+02:00` timestamps at month
  boundaries, refunds, duplicate customer names, revenue ties, case-insensitive SKUs, duplicate CSV rows,
  later commits touching the same file, half-up rounding.
- The checker was validated before spending tokens: a reference solution passes 11/11 on seeds 0–2 and every
  naive approach is rejected (grep, `most_common`, string date compare, unmerged rows, case-sensitive SKUs,
  blind `sed`, "last commit", decoy name).
- Bug found during validation: the gate let a blind `sed qty→quantity` through although it broke the CLI.
  Fix: every `./tickets next` re-checks all earlier tickets.
- Final score is computed independently after the run (all checks re-run on the final state, protected
  `tests/`, `data/`, `logs/` verified by hash) — it does not trust what `./tickets` reported.
- Known limitation: hidden-test paths are visible in failure output, so the agent could read them.

## 7. Baseline result (gpt-6-luna, seed 0)
- 11/11 tickets, 28 turns, 259,899 input tokens (91% cached), 5,571 output, peak context 21,555, 112 s.
- Consequences for the next steps:
  - 21k peak never fills the real window → compaction must be tested under an artificial context budget,
    and judged by "still 11/11, ticket 11 still correct" rather than by savings alone.
  - 91% cache hit is the bar: the new mechanisms must not destroy prefix caching.
  - Token savings from dynamic tool loading need a realistic tool catalog in the "before" variant.

## 8. Requirements update
- The user supplied a detailed spec (LangGraph harness: context manager cascade, artifact store, error
  taxonomy, benchmark matrix `original/eager/lazy/masking/summary/hybrid`). It is now the primary source;
  work proceeds in parts, one commit per part, each with a DEVLOG entry.
- Baseline frozen: `baseline/original_agent.py` (provided code, byte-identical) and
  `baseline/langgraph_port.py` (behaviour-preserving port + metrics, commit 370f1fe).
- Assumption: the provided code cannot run on gpt-6-luna (it sends `temperature`, which the model rejects
  with HTTP 400). Benchmark mode `original` therefore runs the port, which was verified equivalent to the
  provided loop by replaying scripted LLM replies. The earlier "baseline" result is relabelled `original`.

## 9. Part 1 — dynamic tools and skills
- Registry: `harness/catalog.py` (25 tools: 12 working, 13 synthetic with realistic schemas and no backend —
  they return "not available in this environment"), `ToolSpec` with keywords, read_only, idempotent,
  timeout, version, max_output_tokens. `harness/registry.py`: `ToolRegistry`, `SkillRegistry`, one BM25
  index class (stdlib) reused for tools and skills.
- Modes over the same catalog:
  - `eager`: bash + all 25 schemas in `tools[]`, all SKILL.md bodies in the system prompt.
  - `lazy` (static dispatcher): fixed `tools[]` = bash, discover, describe, invoke, load_skill,
    read_skill_resource. No registry listing in the prompt (spec), only a one-line hint of tool categories
    and skill L0 metadata (name, description, when to use).
  - `native` (optional experiment): `describe(name)` appends that schema to `tools[]` for later turns.
- Why the dispatcher is the default: `tools[]` and the system prompt are the head of the cached prefix.
  In `lazy` they never change, discovered schemas arrive as tool results (append-only), so the prefix stays
  byte-identical (`test_lazy_prefix_unchanged_by_discovery`). `native` gets real schema enforcement for
  discovered tools but every binding rewrites the head of the prompt; measured below.
- Skills follow Agent Skills layout (`skills/<name>/SKILL.md` + `references/`): L0 metadata in prompt,
  L1 `load_skill`, L2 `read_skill_resource` (path traversal blocked).
- Rejected: embedding/hybrid retrieval and hierarchical discovery (25 tools do not need them), intent
  routing (extra LLM call), skill eviction (rewrites history; handled later by the context manager).
- Validation: required/unknown/type checks before execution (bool is not accepted as integer).
- Fixed per-call prefix (measured with a real request, `input_tokens`): original 171, eager 2,887,
  lazy 576, native 535 (before any binding).
- Caveat: the skills contain advice relevant to the benchmark tickets, so `original` (no skills) vs
  `lazy` differs in available information, not only in loading strategy. `eager` vs `lazy` is the clean
  comparison: same catalog, same skills, different loading.
- Benchmark, gpt-6-luna, seed 0, 1 run per mode (n=1 — indicative only, turn counts vary run to run):

  | mode | success | turns | input tokens | cached | peak context | tool calls of note |
  |---|---|---|---|---|---|---|
  | original | 11/11 | 28 | 259,899 | 91% | 21,555 | bash ×27 |
  | eager | 11/11 | 51 | 578,172 | 97% | 19,282 | bash 16, edit_file 13, read_file 11, … |
  | lazy | 11/11 | 39 | 519,976 | 94% | 28,539 | bash 32, load_skill 6, discover 0 |
  | native | 11/11 | 37 | 290,158 | 94% | 16,858 | — |
- Findings:
  - Per-call fixed cost is the clean signal: eager pays ~2.3k tokens more than lazy on every call.
  - Total tokens are dominated by turn count, which varies with strategy: granular catalog tools in eager led
    to many small steps (51 turns). In lazy the model ignored the catalog and used bash + skills; loaded
    skill bodies raised its peak context. n=1 cannot separate these effects from noise.
  - No quality regression: every mode closed all 11 tickets including the recall ticket.

## 10. Part 2 — harness graph and execution reliability
- `harness/graph.py`: StateGraph with the spec's flow: context_manager → prompt_builder → llm → router →
  (tool_router → tool_executor → output_processor → state_update → context_manager) | final_verification.
  Business logic lives in plain modules (`executor`, `llm`, `store`, `metrics`); nodes only orchestrate.
  `InMemorySaver` checkpointer for graph state (this is persistence, unrelated to the provider KV cache).
- `AgentState` separates the active LLM context (`active_messages`, `summary`) from structured state
  (`observations`, `active_skills`, `active_tools`, `completed_actions`, `executor`), and the immutable
  history goes to an append-only JSONL `EventStore` outside the prompt.
- Protocol fix (baseline bug): one assistant message per turn holding all tool calls, followed by one tool
  result per `tool_call_id`. `final_verification` checks protocol integrity.
- Error taxonomy (`harness/executor.py`), returned as structured JSON
  `{ok, code, message, retry_same_call, model_can_repair}`:
  - model-repairable: INVALID_JSON (malformed arguments no longer crash the loop), INVALID_ARGUMENTS
    (schema), UNKNOWN_TOOL, FILE_NOT_FOUND, NOT_AVAILABLE, TOOL_ERROR, non-zero exit codes (plain output).
  - harness-recoverable: LLM connection/timeout/429/5xx → bounded retries, exponential backoff + jitter,
    overall deadline (`harness/llm.py`); transient errors of read-only idempotent tools → 2 retries.
  - non-recoverable: auth errors stop the run; PERMISSION_DENIED (synthetic side-effect tools, a small bash
    denylist: sudo, git push, rm -rf /, …) tells the model not to retry; budgets (turns, tool calls,
    tokens, wall time) stop the run with an explicit status; context overflow gets its own status.
- Never retried: bash and non-idempotent tools. A bash timeout kills the whole process group and reports
  that the command may have partially run, asking the model to inspect state first.
- Loop detection: same tool + normalized arguments failing `loop_limit` (3) times without any environment
  change in between → LOOP_DETECTED instead of executing. Any successful write/bash bumps the environment
  version, so a retry after a fix is allowed.
- Bash output is streamed to temp files instead of pipes, so huge outputs do not accumulate in memory.
- Tool calls run sequentially in model order: shell commands may depend on each other.
- Tests (mock LLM replaying scripted replies): protocol integrity, error recovery, permission, loop
  detection and reset, non-idempotent retry protection, bash timeout side effects, LLM retry/give-up,
  budget exhaustion.
