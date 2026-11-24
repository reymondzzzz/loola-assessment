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
