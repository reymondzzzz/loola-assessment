# Cache-aware agent harness (Loola test task)

A LangGraph coding-agent harness grown from the provided single-file loop. It adds **dynamic tool and skill
loading**, **adaptive context compaction**, external artifact memory, a structured error taxonomy, and a
reproducible benchmark that measures tokens, cache hits and task success against the original loop.

- Original code, unchanged: [`baseline/original_agent.py`](baseline/original_agent.py)
- Decisions, commands, results and trade-offs: [`docs/DEVLOG.md`](docs/DEVLOG.md) (the session log)
- Agent instructions for this repo: [`AGENTS.md`](AGENTS.md)

## 1. Problem

The original loop keeps every tool description and every byte of tool output in the prompt forever. With a
large tool catalog the fixed prefix grows; on long tasks the context overflows; and any edit to earlier
messages would destroy the provider's prefix (KV) cache. It also crashes on malformed tool-call JSON or any
HTTP error, and duplicates the assistant message when one reply holds several tool calls.

## 2. Architecture

```text
START → context_manager → prompt_builder → llm ─┬─ no tool calls → final_verification → END
              ↑                                  ├─ provider context overflow → emergency → prompt_builder
              │                                  └─ tool calls → tool_router → tool_executor
              │                                                        → output_processor → state_update ─┐
              └────────────────────────────────────────────────────────────────────────────────────────────┘
```

| Component | File | Responsibility |
|---|---|---|
| AgentState, graph wiring | `harness/graph.py` | orchestration only; InMemorySaver checkpoints |
| HarnessConfig, mode presets | `harness/config.py` | flags, budgets, limits |
| ToolRegistry, SkillRegistry | `harness/registry.py`, `harness/catalog.py` | catalog, BM25 discovery, validation |
| PromptBuilder | `harness/prompt.py`, `harness/context.py:build_prompt` | stable system prompt, base tools, prompt layout |
| ToolExecutor | `harness/executor.py` | validation, permissions, retries, timeouts, loop detection |
| OutputProcessor | `harness/output.py` | filters, dedup, truncation, adaptive budget, observation metadata |
| ArtifactStore, EventStore | `harness/store.py` | content-addressed outputs, append-only JSONL history |
| ContextManager | `harness/context.py` | watermarks, strategy cascade, validation, emergency reduction |
| Memory tools | `harness/memory.py` | `read_artifact`, `search_history` |
| LLM client | `harness/llm.py` | ChatOpenAI, bounded retries, overflow detection, usage |
| MetricsCollector | `harness/metrics.py` | tokens, cache, calls, compactions, reclaimed tokens |
| BenchmarkRunner | `bench/` | ticket-queue task, independent checker, report |

State is split three ways: the immutable history (JSONL event log + artifacts on disk, outside the prompt),
the active LLM context (`active_messages`, `summary`), and structured execution state (`observations`,
`active_skills`, `active_tools`, `completed_actions`, executor counters).

## 3. Dynamic tools and skills

The catalog has 25 tools: 12 working (files, grep, tests, SQLite, JSON, CSV, git) and 13 **synthetic**
external-service tools with realistic schemas and no backend (labelled; calling them returns
`NOT_AVAILABLE` or `PERMISSION_DENIED`). Each `ToolSpec` carries keywords, read_only, idempotent, timeout,
version and max_output_tokens.

| Mode | `tools[]` | System prompt | Fixed cost per call (measured) |
|---|---|---|---|
| original | bash | original | 171 tokens |
| eager | bash + 25 schemas (+ memory tools) | + all SKILL.md bodies | 2,887 tokens |
| lazy — static dispatcher (default) | bash, discover, describe, invoke, load_skill, read_skill_resource, read_artifact, search_history | one-line catalog hint + skill L0 metadata | 576 tokens |
| native (experiment) | lazy minus invoke; `describe(x)` appends x's schema | same as lazy | 535 tokens before binding |

- **Static dispatcher**: `discover(query, kind="tool"|"skill")` (BM25 over names, descriptions, keywords) →
  `describe(name)` → `invoke(name, arguments)`. The executor validates arguments against the schema and
  checks permissions; discovery grants nothing. `tools[]` never changes during a session.
- **Skills** use the Agent Skills layout (`skills/<name>/SKILL.md`, `references/`) with progressive
  disclosure: L0 metadata in the prompt, L1 `load_skill`, L2 `read_skill_resource` (path-traversal safe).
  Loaded skills record `skill_id`, `version`, `loaded_at_turn`, `visible_in_context`; a skill compacted
  out of the context is marked invisible and can be reloaded.
- **Trade-off, native vs dispatcher**: native binding gives the model real function definitions (provider
  side schema enforcement), but each binding changes `tools[]`, which is rendered at the very start of the
  prompt, so every later token must be prefilled again. The dispatcher keeps the prefix byte-identical and
  pays with harness-side validation of nested arguments. Sticky tool sets (change only between phases) are a
  middle ground; not implemented.

## 4. Context management

Budget: `input_budget = context_window − output_reserve − safety_margin`. Nothing happens below the high
watermark (80%). Above it, a cascade runs on a copy of the state until the estimate is at the low watermark
(50%), so compaction is rare and buys many turns:

| Level | Strategy | LLM | What changes |
|---|---|---|---|
| 0 | no-op | — | append-only history |
| 1 | stale + duplicate eviction | no | outputs of files changed since read, superseded test results, older identical outputs → references |
| 2 | observation masking | no | old tool results → `tool + args → status, tokens, read_artifact(id)` (errors keep their text); ids and assistant tool calls untouched |
| 3 | interaction pruning / action masking | no | large tool arguments → artifacts; whole old groups (assistant + all its results) → one list of execution events |
| 4 | structured semantic summary | 1 call | old history → validated JSON summary (goal, verified facts, decisions, completed/failed/pending work, paths, artifact refs, skill refs), incremental over the previous summary, fed with deterministic execution events |
| 5 | emergency | no | everything but the last interaction masked/archived; if mandatory content still does not fit → `context_budget_exceeded` |

Transactional: the candidate is committed only if tool-call protocol, the verbatim task, artifact
references and the budget validate and ≥ 1024 tokens were reclaimed; otherwise the original history stays.
A failed or invalid summary never replaces history. Each strategy runs at most once per epoch (no loops).
On a provider overflow error the emergency node forces compaction to 50% of the current size, at most twice.

Output processing keeps data out of the context in the first place: full output saved as an artifact,
pytest/diff filters, dedup against still-visible observations, head/tail truncation, and a per-turn output
budget of min(per-tool limits, 20% of free input budget) shared by all calls of the turn.

## 5. Eviction policies

Policy (what) is separate from mechanism (how). Default policy: candidates are unpinned, recoverable
(archived), outside the last `recent_interactions` (5) interactions and ≥ 64 tokens; ordered stale first,
then largest, then oldest. With `cache_aware`, ties within a 256-token size bucket go to the **later**
message, because modifying a later message preserves a longer cached prefix. Pinned: the original task (fixed
slot, never summarized) and loaded skill instructions. Version-aware staleness uses file fingerprints
(mtime + size); unrestricted bash conservatively invalidates earlier test results.

## 6. External memory

Full tool outputs and large archived arguments are content-addressed files; the JSONL event log holds every
assistant message, tool call and result, compaction report and summary. The model recovers anything with
`read_artifact(id, offset, limit)` (paged, ≤ 400 lines) or `search_history(query)` (BM25 over the log).
Recovered content is appended as a new observation, never restored in place, so the prefix survives until
the next compaction. LangGraph checkpoints persist graph state; they are unrelated to the KV cache.

## 7. Prompt cache — answer to question 1

> How to do compaction and skill loading without invalidating the prompt (KV) cache on every step?

- **Stable head.** System prompt, base `tools[]` (fixed order), the verbatim task and, within an epoch, the
  summary never change. No timestamps, counters or "loaded tools" lists at the top.
- **Load by appending.** Tool schemas (`describe`) and skills (`load_skill`, `read_skill_resource`) arrive as
  tool results at the end of the history instead of being added to `tools[]` or the system prompt.
  `test_lazy_prefix_unchanged_by_discovery` and `test_prompt_prefix_stability` assert byte-level prefix
  equality between consecutive requests.
- **Compact rarely, in batches.** Stale flags are queued, not applied immediately; compaction starts only at
  the high watermark, reclaims down to the low watermark and is skipped if it would reclaim < 1024 tokens.
  Between epochs the history is strictly append-only, so every request extends the previous one.
- **Prefer edits that keep a long prefix**: cheap deterministic masking first, cache-aware tie-breaking
  toward later messages, and the summary slot placed right after the task.
- **What caching can and cannot do.** Standard prefix caching reuses KV blocks only up to the first changed
  token; unchanged blocks after an edit cannot be reused without special backend support. A compaction
  therefore keeps only the unchanged head and the rest is prefilled once; the next turns hit the cache again.
  Caching reduces repeated computation and cost, not the logical context length. Real behaviour depends on
  the provider; the numbers below are the provider's reported `cached_tokens`.

## 8. Error handling — answer to question 2

> Which tool errors should go back to the model for self-correction, and which should the harness catch
> deterministically?

Return to the model (structured `{"ok": false, "code", "message", "retry_same_call", "model_can_repair"}`)
everything the model can fix by changing its next call: `INVALID_JSON` (malformed arguments; the baseline
crashed here), `INVALID_ARGUMENTS` (schema), `UNKNOWN_TOOL` (with what is available), `FILE_NOT_FOUND`,
`NOT_AVAILABLE`, tool exceptions, non-zero exit codes and test failures (plain output), and `TIMEOUT` of a
command with a warning that it may have partially run.

Handle in the harness, deterministically:
- **recoverable**: LLM connection errors, timeouts, 429 and 5xx → bounded retries with exponential backoff,
  jitter and a deadline; transient failures of read-only idempotent tools → 2 retries; context overflow →
  compaction / emergency path; oversized output → truncation + artifact;
- **never auto-retried**: bash and non-idempotent tools (a timeout does not prove nothing happened; the
  process group is killed and the model is told to inspect state first);
- **non-recoverable**: auth errors stop the run; `PERMISSION_DENIED` (side-effecting external tools, a small
  bash denylist) is reported with `model_can_repair: false`; exhausted budgets (turns, tool calls, tokens,
  wall time) stop with an explicit status; `LOOP_DETECTED` replaces the third identical failing call made
  without any environment change in between.

## 9. Benchmark methodology

`bench/` generates a fresh project per seed and gates the work behind `./tickets next`: the next ticket is
revealed only after the current one passes its checks, so the task is long by construction. 11 tickets:
pricing and store bug fixes, log analysis, tie-breaking, a CSV CLI, a SQLite report, nested JSON, a
cross-file rename, a 1,500-line legacy file, a git regression hunt, and a **recall ticket** asking for a code
name stated only once in ticket 1 (with a decoy later). Data contains edge cases (CRLF, stack traces with
`ERROR [payments]`, `auth` vs `auth-legacy`, rigged ties, BOM and quoted commas, `+02:00` timestamps at month
boundaries, refunds, duplicate names, case-insensitive SKUs, later commits touching the same file).

Success is decided independently after the run: every ticket's checks re-run on the final state (hidden
edge-case tests, reference implementations), and `tests/`, `data/`, `logs/` are verified untouched by hash.
The checker was validated before use: a reference solution scores 11/11 on seeds 0–2 and naive solutions are
rejected. The agent's final message is never used as evidence.

Modes (`harness/config.py`): eager and lazy share the catalog; masking, summary and hybrid share the output
processor, artifact store, model and environment and differ only in context strategies.

| Mode | Lazy loading | Output pipeline | Masking/stale/pruning | Summary |
|---|---|---|---|---|
| original | — | — | — | — |
| eager | no | no | no | no |
| lazy | yes | no | no | no |
| masking | yes | yes | yes | no |
| summary | yes | yes | no | yes |
| hybrid | yes | yes | yes | yes |

Model: `gpt-6-luna` via the OpenAI API. Main matrix at a 100k context window. Compaction does not trigger
there (runs peak below 30k), so the compaction modes were also run with an **artificially reduced 16k
window** (16k), labelled `-16k`. `original` runs the behaviour-preserving LangGraph port of the provided loop,
because the provided file sends `temperature`, which this model rejects (verified equivalent by replaying
scripted replies through both).

## 10. Results

All numbers are real `gpt-6-luna` runs recorded in `bench/results.jsonl` (per-run logs in `bench/runs/`).
Means over seeds 0 and 1 (n = 2 per mode). "Input" includes summarizer calls; "uncached" = input − provider
reported `cached_tokens` (cached tokens are a subset of input, not added twice). Produce the tables with
`uv run python bench/report.py <labels…>`.

**Main matrix, 100k context window** (code at `fe022a1`; `summary` lane started after the summary-role
change, `hybrid` before it — at 100k neither ever summarized, so it does not matter here):

| mode | success | turns | input tokens | cached | uncached | peak context | compactions | wall s |
|---|---|---|---|---|---|---|---|---|
| original | 2/2 | 28 | 432,538 | 93% | 30,948 | 29,002 | 0 | 113 |
| eager | 2/2 | 42 | 511,664 | 96% | 21,437 | 22,890 | 0 | 140 |
| lazy | 2/2 | 31 | 442,501 | 93% | 32,300 | 30,380 | 0 | 143 |
| masking | 2/2 | 40 | 342,127 | 95% | 18,112 | 16,170 | 0 | 155 |
| summary | 2/2 | 40 | 376,228 | 95% | 20,308 | 18,362 | 0 | 162 |
| hybrid | 2/2 | 40 | 400,825 | 95% | 19,932 | 18,606 | 0 | 164 |

**Compaction stress test, artificially reduced 16k window** (input budget 9,856, compaction at 7,884;
all three modes on the same code revision `dfe7a3c`, same seeds):

| mode | success | turns | input tokens | cached | uncached | summarizer tokens | peak context | compactions | wall s |
|---|---|---|---|---|---|---|---|---|---|
| masking-16k | 2/2 | 60 | 345,065 | 84% | 55,036 | 0 | 7,833 | 6 | 194 |
| summary-16k | 2/2 | 51 | 345,268 | 71% | 100,764 | 99,448 | 8,973 | 2 | 365 |
| hybrid-16k | 2/2 | 69 | 417,870 | 78% | 92,114 | 49,514 | 7,874 | 6 | 346 |

What the numbers show (and do not show):

- **Quality**: every run in every mode closed all 11 tickets, including the recall ticket (a code name stated
  once in ticket 1 and needed in ticket 11, after up to 8 compactions at 16k). No quality regression was
  observed; with n = 2 that is evidence, not proof.
- **Dynamic loading**: the clean signal is the fixed per-call cost — eager sends 2,887 tokens of schemas and
  skills on every call vs 576 for lazy. In totals, eager also took more turns (42 vs 31): with 25 granular
  tools the model made many small calls. Its *uncached* tokens are lower because the bigger fixed prefix is
  cached — caching hides much of the eager overhead on this provider, but not the context it occupies.
  In lazy the model barely used the catalog (bash + `load_skill`), which matches the task.
- **Output pipeline** (masking/summary/hybrid at 100k, no compaction triggered): peak context fell from
  ~30k (lazy) to 16–19k and input tokens by 9–23%, with turns higher (40 vs 31). Turn counts vary by tens of
  percent between seeds, so the token difference is suggestive, not significant.
- **Context strategies (16k)**: deterministic masking alone was the cheapest: about half the uncached tokens
  of the other two and no summarizer cost. Summary-only spent ~99k summarizer tokens for 2 compactions and
  had the lowest cache hit rate (71%): every summary rewrites everything after the task, so the next request
  is mostly prefill. Hybrid ran the cheap strategies first and summarized only when they were not enough:
  it used half the summarizer tokens of summary-only and ~9% fewer uncached tokens, but more turns.
- **Hypothesis** ("deterministic pruning + occasional summarization is cheaper than summarization alone while
  preserving success"): partly supported — hybrid beat summary-only on uncached and summarizer tokens at
  equal success, but not on total input (more turns). The stronger result is that on this task
  deterministic masking alone was enough; the summarizer added cost without a measurable quality benefit.
  A task whose early observations are needed verbatim late would be needed to show summarization's value.
- Earlier rows kept for transparency (`*-16k-pre`: same comparison on older code with the user-role summary
  and old scoring; `*-part1`, `part2-lazy`, `part4-hybrid-30k`: intermediate versions, n = 1). They show the
  same ordering: masking cheapest, summary-only most expensive in uncached tokens.
- Not measured: TTFT (no streaming), monetary cost (model pricing unknown to me).

## 11. Trade-offs and limitations

- Sample sizes are small (see the `runs` column); turn counts vary by tens of percent between runs of the
  same mode, so differences in totals smaller than that are not significant.
- The skills contain advice relevant to the tickets, so `original` (no skills) vs the harness modes differs in
  information, not only in mechanics; `eager` vs `lazy` is the clean loading comparison.
- The agent can read hidden-test paths in failure output; final scoring is independent of it.
- Summaries are validated structurally, not semantically: a summary that drops a fact still passes. Recovery
  then relies on `search_history`/`read_artifact`.
- Staleness of files read through bash (`cat`) is not tracked; only catalog reads have fingerprints.
- No late user messages are supported (single-shot task), so "late requirement changes" is untested.
- Tool calls run sequentially; parallel read-only execution was not needed for this task.
- Cost in dollars is not reported: pricing for the model was not available to me.

## 12. Future improvements

Sticky tool sets between phases; semantic checks of summaries (every verified fact traceable to an event);
SQLite FTS5 for history; AST/symbol retrieval and a repository map instead of full-file reads; parallel
independent read-only tools; a cost model for compaction timing (reprefill cost vs. tokens saved).

## 13. Install and run

```bash
uv sync
cp .env.example .env            # LLM_BASE_URL, LLM_API_KEY, LLM_MODEL (any OpenAI-compatible endpoint)
uv run --env-file .env python agent.py "list files in the current directory"     # AGENT_MODE=lazy by default
AGENT_MODE=hybrid uv run --env-file .env python agent.py "<task>"

uv run python -m pytest -q tests                                                # 52 deterministic tests
uv run --env-file .env python bench/run.py --label hybrid --mode hybrid --runs 2 [--context-window 16000]
uv run python bench/report.py                                                   # results table
```

Run directories (events, artifacts) go to `$AGENT_RUNS_DIR` or a private (0700) folder in the temp dir.
