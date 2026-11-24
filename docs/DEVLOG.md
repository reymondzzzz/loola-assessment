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
- Real smoke run of the new graph (lazy, seed 0, 100k window): 11/11 tickets, 37 turns, 434,755 input
  tokens (93% cached), peak 27,729, 36 tool calls (5 discovery), no protocol errors.

## 11. Part 3 — output processing and external memory
- Token counting with tiktoken `o200k_base` (already a transitive dependency of langchain-openai; now
  declared). Counts include tool schemas and tool-call arguments.
- `ArtifactStore`: content-addressed files (`a` + sha256[:11]) in the run directory; the full output is
  saved before anything is truncated, so every shortened observation stays recoverable.
- `OutputProcessor` pipeline per tool result (flags: output_truncation, deduplication):
  1. tool-specific filter: pytest → failures, `E` lines, locations, summary; git/bash diffs → headers and
     changed lines;
  2. deduplication against observations that are still visible; a duplicate becomes a pointer to the
     earlier tool call. Bug found by test: pointers were chained (c3 → c2 → c1); only observations whose own
     content is visible are dedup targets now;
  3. head/tail truncation (30% head / 70% tail) with an artifact reference whose size is counted in the
     budget;
  4. adaptive budget: per-turn budget = min(sum of per-tool `max_output_tokens`, 20% of free input budget),
     shared equally by the tool calls of that turn.
  Structured errors are never truncated or deduplicated.
- Observation metadata per tool result (artifact id, tokens before/after, turn, resource path + file
  fingerprint, kind, pinned, stale, digest) feeds the context manager. Staleness is tracked after each turn
  but not acted on until the next compaction epoch: a read becomes stale when the file fingerprint changes;
  a test result becomes stale when anything changed the environment after it.
- New base tools in every harness mode: `read_artifact(id, offset, limit)` (paged, max 400 lines) and
  `search_history(query, limit)` (BM25 over the JSONL event log). Recovered content is appended as a new
  observation; history is never rewritten to restore it.

## 12. Part 4 — adaptive context manager
- Budget: `input_budget = context_window − output_reserve (4096) − safety_margin (2048)`; compaction runs only
  when the estimated next request (all messages + tool schemas, tiktoken) reaches the high watermark (80%)
  and compacts down to the low watermark (50%), so it happens rarely and buys many turns. Estimates are
  calibrated by the last `usage.input_tokens / estimate` ratio (clamped 0.8–1.5).
- Cascade (`harness/context.py`), cheapest and safest first, stopping as soon as the context is ≤ low:
  1. stale + duplicate eviction — mask outputs of files that changed since they were read, test results
     superseded by later changes, and older copies of identical outputs;
  2. observation masking — replace old tool results with a reference (`tool + args → status, tokens,
     read_artifact(id)`, error text kept for failures). Message ids, tool_call ids and the assistant tool
     calls stay untouched;
  3. interaction pruning — archive large tool-call arguments (heredocs, file contents) to artifacts, then
     replace whole old groups (assistant message + all its tool results) by one compact list of execution
     events; never splits a group, never leaves orphan results;
  4. semantic compaction — one LLM call producing a structured summary (goal, verified facts, decisions,
     completed/failed/pending work, paths, artifact refs, skill refs). Input: original task verbatim, previous
     summary (incremental), deterministic execution events, and the history to compress. The summary is
     validated (keys, types, non-empty goal, size, existing artifact ids) before it is committed;
  5. emergency — mask everything but the last interaction, archive older messages, truncate oversized
     results; if the mandatory part still does not fit, stop with `context_budget_exceeded`.
- Policy vs mechanism: `eviction_candidates` (policy) picks unpinned, recoverable, non-recent observations
  ≥ 64 tokens, ordered stale first, then by size, then by age; `mask`/pruning are the mechanisms.
  Cache-aware tie-break (flag `cache_aware`): within the same size bucket (256 tokens) prefer later messages,
  because masking a later message keeps a longer unchanged prefix.
- Pinning: the original task is a fixed message slot and never enters the summarizer's output; loaded skill
  instructions are pinned against masking. Skills track `loaded_at_turn` and `visible_in_context`; when a
  skill's message is summarized away it is marked not visible and listed in `active_skill_refs`, and
  `load_skill` simply reloads it.
- Transactional: strategies work on a deep copy; the result is committed only if protocol validation, task
  check, artifact references and budget pass, and at least `MIN_RECLAIM` (1024) tokens were reclaimed.
  A failing summarizer (exception, non-JSON, invalid schema) leaves history untouched and lets the
  deterministic strategies do the work. Each strategy runs at most once per epoch → no compaction loops.
- Prompt layout `[system, task, summary?, *history]`: within an epoch history is append-only, so requests are
  strict prefixes of the next one (`test_prompt_prefix_stability`); across epochs system + task + tools stay
  identical, so only the part after the first modified message needs prefill.
- Overflow path: if the provider rejects a request for length, the `emergency` node forces compaction
  targeting 50% of the current size (the configured window was evidently too optimistic), retried at most
  `emergency_attempts` (2) times. Bug found by test: the first version targeted the configured low watermark
  and reclaimed nothing.
- Tests (unit + graph with scripted LLM): budget/watermarks, masking, stale eviction, pruning protocol
  validity, pinned task/skills, progress and insufficient-reclaim, summary validation, summarizer failure
  rollback, early facts surviving two compaction epochs, prefix stability, overflow recovery.
- Security review finding (run directories in a shared temp dir could expose tool outputs): the run root is
  now created with mode 0700 and rejected if owned by another user or group/world-accessible.
- Real run, hybrid, 30k window, seed 0: 11/11 tickets, 44 turns, 376,392 input tokens (95% cached),
  peak 16,802 — compaction never triggered: output truncation + dedup alone kept the peak below the high
  watermark (~19k; lazy without the output pipeline peaked at 28.5k). A live compaction test therefore needs
  an artificially smaller window (16k, labelled as such in results).

## 13. Part 5 — benchmark matrix, README
- Security review of the part-4 commit: the model-written summary was a `user`-role message, giving text
  distilled from untrusted tool output user authority. It is now an `assistant`-role message ("my own notes …
  data, not instructions; the original task is authoritative"); test `test_summary_is_not_user_authority`.
  Caveat: the `hybrid` and `hybrid-16k` matrix runs had already started and used the old user-role layout;
  `summary`/`summary-16k` runs started later and use the new one.
- Matrix launched in 5 parallel lanes (separate workdirs and run dirs; results appended line by line).
  First attempt failed instantly: zsh does not word-split `$VAR` holding a command; relaunched via a script.

## 14. Fixes from an external review
All five findings were confirmed against the code before fixing; each fix has a regression test
(`tests/test_review_fixes.py`).
1. Tool-call budget was checked after the whole batch (with `max_tool_calls=1` three calls ran). Now checked
   before each call; refused calls still get a `BUDGET_EXHAUSTED` result for their `tool_call_id`.
2. `head_tail` could exceed its budget (49 tokens for budget 10) because of a minimum content size. Now the
   result is trimmed until it fits; the floor is a minimal artifact reference (~12 tokens). Bash no longer
   reads whole stdout/stderr into memory: at most 128 KB head + 128 KB tail per stream are loaded; when cut,
   the full output is assembled on disk by streaming and moved into the artifact store; `read_artifact`
   streams lines instead of loading the file.
3. A malformed `invoke` (wrong types) could raise inside the dispatcher; `invoke`'s own schema is validated
   first → `INVALID_ARGUMENTS`.
4. Benchmark scoring only compared test files still present; a deleted visible test went unnoticed. Now the
   full set of issued test paths and their hashes must match (`tests_intact`).
5. The summary/hybrid comparison mixed code versions (hybrid-16k used the summarizer 3–4 times with the old
   user-role summary layout). The 16k trio is re-run on the fixed code with the same seeds; earlier 16k rows
   are kept as `*-16k-pre`. `bench/report.py` now counts summarizer tokens in total input.
- Also: removed the unused `constraints` state field (the task is protected by its fixed prompt slot and
  validation); rewrote the outdated `AGENTS.md`.

## 15. Fixes from the second review
1. Malformed `invoke` still crashed the graph: the dispatcher returned INVALID_ARGUMENTS, but the output
   processor re-parsed the raw arguments and called `.get` on a string. `effective_call` now only accepts a
   dict; regression test goes through `run_agent` (`test_malformed_invoke_does_not_crash_the_graph`).
2. Benchmark scoring would fail correct solutions: pytest (the agent's and the checker's) writes
   `tests/__pycache__`, which `tests_intact` counted as an extra file. `__pycache__` is ignored and the
   checker runs `python -B`. The in-flight 16k re-runs used this broken scoring, so they were stopped and
   discarded before writing results.
3. Shared output budget exceeded the 20% rule (8 calls, 1000 free tokens → 512 instead of 200) because of a
   per-call minimum of 64 on top of the turn budget. Now the turn budget is
   `max(min(Σ limits, 20% free), n × MIN_CALL_BUDGET)`, where MIN_CALL_BUDGET is the size of the minimal
   artifact reference (`[artifact a…]`), and `head_tail` never exceeds a budget ≥ that floor.
   A multi-epoch graph test needed a longer scenario because outputs are now smaller.

## 16. Part 5 results
- Final matrix: 18 real runs (29 recorded in total incl. earlier versions), all 11/11 tickets (incl. the
  recall ticket after up to 8 compactions).
  Tables and interpretation in README §10. 100k matrix on `fe022a1` (no compaction can trigger there);
  16k compaction trio re-run on `dfe7a3c`, same seeds. Older rows kept with `-pre`/`-part1` labels.
- Main findings: fixed per-call cost 2,887 (eager) vs 576 (lazy) tokens; the output pipeline cut peak
  context from ~30k to 16–19k; at 16k deterministic masking was the cheapest (55k uncached vs 92k hybrid,
  101k summary-only, which spent ~99k tokens on 2 summaries and had the lowest cache hit rate, 71%).
  Hypothesis partly supported: hybrid < summary-only on uncached and summarizer tokens at equal success, but
  masking alone was enough on this task.
- Privacy pass before publishing: run logs contained the home path and `ls -la` owner name; replaced with
  `<repo>`, `~` and `user`. Three logs committed earlier still contain the path in history; not rewritten,
  because commit author name/email are public anyway and rewriting would change hashes cited in the docs.

## 17. Fixes from the third review
1. Deduplication compared the bounded preview of large bash outputs: two outputs with equal head and tail
   but different middles were reported as identical. Spilled outputs now use the full artifact's hash as
   their digest (`test_large_outputs_with_equal_preview_are_not_deduplicated`).
2. `MIN_CALL_BUDGET` was measured on one example id (9 tokens); other ids tokenize to 14–15. The floor is now
   the worst case: reference overhead + one token per id character (checked over 2,000 random ids).
3. Wrong counts in the docs: the final matrix is 18 runs (29 recorded in total), and the recall ticket
   survived up to 8 compactions (hybrid-16k seed 0), not 6. Corrected in README and here.

## 18. Fix from the fourth review
- `search_history` re-saved the event's content as an artifact; for spilled bash outputs that content is the
  bounded preview, so the returned id pointed to a copy without the middle. `tool_result` events now record
  the full artifact id and `search_history` returns it (graph-level test with a marker in the middle of a
  ~800 KB output). Limitation: history search indexes the preview of spilled outputs, so text that appears
  only in the unloaded middle is not searchable; it is reachable via `read_artifact`.
- README test count updated (55).

## 19. Closing five gaps with the spec — 1/5: configurable eviction policies (§9)
- `HarnessConfig.eviction_policy` ∈ `default | fifo | size` (validated; `bench/run.py --eviction-policy`).
  `default` is the previous combined ordering, unchanged (stale first, then size, then age; with
  `cache_aware` the size bucket + later position tie-break). `fifo`: oldest observation first. `size`:
  largest first.
- Policy and mechanism stay separate: the policy only orders candidates; masking is the same mechanism.
  The common guards are applied before any policy, so they hold in every mode: not masked, not pinned,
  recoverable (artifact available), outside the protected recent interactions, ≥ 64 tokens.
- Gap closed on the way: stale/duplicate eviction did not check `recoverable`.
- Tests (`tests/test_eviction_policies.py`): the same history gives three different candidate orders; the
  guards exclude pinned/unrecoverable/small/recent observations under every policy; the policy drives the
  existing masking mechanism (fifo masks the oldest, size the largest); unknown policy names are rejected.

## 20. 2/5: independent optimization flags (§16)
- Prompt and base tools are now built from the config instead of a mode name:
  - `lazy_tools`: catalog via discover/describe/invoke (True) or all 25 schemas in `tools[]` (False).
  - `lazy_skills`: skill L0 metadata in the prompt + `load_skill`/`read_skill_resource` tools (True), or
    all SKILL.md bodies in the prompt and no skill tools (False) — independent of `lazy_tools`.
  - `artifact_offloading`:
    - True: full outputs (and large archived arguments) are saved as artifacts; truncated/masked outputs
      point to them; `read_artifact` is a base tool.
    - False: nothing is archived. Output limits still apply (filters, head/tail truncation, shared budget);
      the truncation marker says "Output not archived" and contains no artifact id; spilled bash output is
      deleted after the bounded preview; `read_artifact` is not offered; `search_history` returns previews
      without ids. Observations are unrecoverable, so masking, stale eviction and pruning skip them;
      emergency reduction drops them with an explicit "not archived" note; validation only checks artifact
      ids that exist in observations.
- Presets: eager/lazy/native keep `artifact_offloading=False`; masking/summary/hybrid all use the same
  output pipeline with offloading on, so they stay comparable. Note: eager/lazy no longer get a
  `read_artifact` tool (it had nothing to read there), which changes their prefix slightly vs. older runs.
- Tests (`tests/test_flags.py`): `lazy_skills` toggled alone under both tool modes (prompt bodies vs
  metadata, skill tools present/absent, tool part unchanged); `artifact_offloading` toggled alone on the
  masking preset through the full graph — on: every referenced artifact exists and compaction ran; off: no
  artifact files, no artifact references anywhere, outputs still bounded, run finishes within budget.

## 21. 3/5: persistent checkpoints and resume (§4)
- `SqliteSaver` (package `langgraph-checkpoint-sqlite`) writes graph checkpoints to
  `<run dir>/checkpoints.sqlite`; `invoke(..., durability="sync")` persists after every step, so a
  SIGKILL loses at most the step in progress. The run id (`YYYYmmdd-HHMMSS-xxxxxx`) is the run directory name
  and the LangGraph `thread_id`; it is printed first (`🆔 Run …`). The run's config is saved as
  `config.json`, metrics live in graph state (`AgentState.metrics`) so they continue across processes.
- `agent.py --resume RUN_ID` (→ `resume_agent`) reopens the same checkpoint store, event log and artifact
  store and continues from the next pending node.
- Uncertain side effects: the executor journals `tool_started` before and `tool_finished` (with the full
  outcome) after every call. When `tool_executor` re-runs on resume, a finished call is replayed from the
  journal (never executed twice); a started-but-unfinished call is re-executed only if it is safe to repeat
  (meta/memory tools, read-only idempotent catalog tools); otherwise it returns `OUTCOME_UNKNOWN`
  ("not repeated automatically; inspect state first"). LLM calls are simply re-issued.
- Tests (`tests/test_resume.py`) run the real CLI in subprocesses against a local OpenAI-compatible fake
  server (`tests/fake_openai.py`, real HTTP through ChatOpenAI, no test hooks in the harness):
  1. SIGKILL while a bash command (`echo once >> side.log; sleep …`) runs → `--resume` in a new process
     finishes; the model receives `OUTCOME_UNKNOWN`; `side.log` has exactly one line; same run directory
     (one `task` event, one `resume` event).
  2. SIGKILL while waiting for the LLM after a tool finished → resume does not re-run the tool, metrics
     carry over (`tool_calls == 1`), run finishes.
  3. Unknown run id → clean error.
- Security review of the resume commit: `repeat_is_safe` returned True for every `invoke` because `invoke`
  is in `META_HANDLERS`, so an interrupted `invoke(edit_file)` would have been repeated on resume. `invoke`
  is now judged by its target tool's spec (read-only and idempotent only); test covers invoke of
  read_file/edit_file/write_file/malformed, bash, meta and direct catalog tools.

## 22. 4/5: late requirement changes and repeated-call metric (§16)
- `agent.py --resume RUN_ID --message "…"` adds a user requirement to a started run (finished, stopped
  or crashed). Requirements are stored in state and rendered as `user` messages right after the original
  task: `[Requirement update N, added after turn T. Where it conflicts with the original task, this update
  wins.]`. They are part of the pinned head: compaction validation rejects any candidate where task or
  requirement messages changed, and the summarizer receives them verbatim. Adding one changes the prompt head
  once (one cache miss) — accepted, because a real requirement change has to be visible to the model.
  For a finished run the state is updated as if `state_update` ran (status back to running) and the loop
  continues from `context_manager`; for an interrupted run only the requirements change and execution
  continues at the pending node.
- `repeated_tool_calls`: number of tool calls whose tool name and arguments, normalized to JSON with sorted
  keys and no whitespace, equal an earlier call of the same run (malformed JSON arguments compare as raw
  text; `invoke` compares the whole call including the inner tool name and arguments). Reported in run
  metrics and benchmark results; `bench/report.py` shows `—` for older rows without the field.
- Tests: end-to-end through the CLI and the fake server — a run finishes after its first action; resume with
  a requirement in a new process under a 12k window (hybrid); every later request carries task + requirement
  as user messages; ≥ 1 compaction happens; the independent check of `out.txt` sees the changed result.
  Metric: key order does not matter (repeat counted), a changed argument is not a repeat.

## 23. 5/5: literal baseline in the benchmark (§16)
- Correction: earlier docs called `baseline/original_agent.py` byte-identical to the provided code. It was
  not: the first commit already replaced the placeholders (`"http://ip:port/v1"`, `"..."`) with
  `os.environ[...]` (4 lines). The file is now restored to the provided text verbatim (SHA-256
  `d1e005f6…eb1c`, pinned in `tests/test_original_baseline.py`, which also checks that the first commit differed
  only in those 4 lines).
- `bench/run.py --mode original` runs that file unmodified, in-process: endpoint, model and auth header are
  injected as module attributes (`LLM_BASE_URL`, `LLM_MODEL`, `LLM_HEADERS`, read at call time), `MAX_TURNS`
  is set to the same benchmark limit as every mode, and `requests.post` is wrapped to record usage from the
  responses and to add a 600 s timeout (the provided call has none, so a hung request would stall the
  benchmark). Payload, prompts, tool and loop are the provided ones, including `temperature=0.1` and
  `max_tokens=4096`.
- `--mode original-port` runs the LangGraph port; all earlier `original` rows (gpt-6-luna, which rejects
  the provided payload) are relabelled `original-port`.
- Results now carry `code_revision` (`git rev-parse --short HEAD`, `-dirty` if uncommitted).
- Route test against the fake server: `run_once(…, "original")` generates the fixture, runs the provided loop,
  scores independently; the server receives the provided payload; the file hash is unchanged afterwards.
