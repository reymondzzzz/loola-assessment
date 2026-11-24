# AGENTS.md

Rules only. Architecture, results and the answers to the task questions live in `README.md`; every decision,
with its evidence, lives in `docs/DEVLOG.md` (append-only). Read the README section for the area you touch.

## Commands

```
# THE ONE COMMAND. Nothing is committed without it passing.
uv run python scripts/check.py        # tests + guardrails + pinned test count + README count; ~35 s
```

The pieces it orchestrates; run them directly only when debugging one of them:

```
uv run python scripts/guardrails.py              # G1–G5 structural guardrails
uv run python -m pytest -q tests                 # deterministic suite, no network
uv run python -m pytest -q tests/test_resume.py  # narrowest rung for persistence/CLI changes
```

Live commands (need `.env`; never part of the gate):

```
uv run --env-file .env python agent.py [--verify "<cmd>"] "<task>"     # prints 🆔 Run <id>
uv run --env-file .env python agent.py --resume <id> [--message "<new requirement>"]
uv run --env-file .env python bench/run.py --label X --mode hybrid --runs 2 [--context-window 16000]
uv run python bench/report.py [labels…]
```

Environment traps:
- Always `uv run`; the system `python3` lacks the dependencies.
- `gpt-6-luna` rejects `temperature`/`max_tokens`: `--mode original` (the provided file) only works on models
  that accept them (e.g. `gpt-4.1-mini`); use `--mode original-port` otherwise.
- Do not `pgrep -f "bench/run.py"` from a loop whose own command line contains that string: it matches
  itself and never exits. Match `python bench/run.py --runs` or use the PID.

## Rules

Zero-tolerance guardrails, all in `scripts/guardrails.py`, each with a regression test in
`tests/test_guardrails.py` that fails when the rule is disabled:

- **G1 frozen baseline** — `baseline/original_agent.py` (the provided file, byte-identical) and
  `baseline/langgraph_port.py` never change; their SHA-256 values are pinned. Need different behaviour? It is
  not the baseline: change `harness/`. Never edit the pinned hashes to make G1 pass.
- **G2 no secrets or personal data** in tracked or untracked (non-ignored) files: no API keys, home-directory
  paths, personal emails. Run logs record absolute paths: scrub before committing
  (`sed -i '' -e 's#<abs repo path>#<repo>#g' -e 's#/Users/<you>#~#g' bench/runs/*.log`). Test fixtures that
  need such strings assemble them from parts (see `tests/test_guardrails.py`).
- **G3 `.env` is never tracked.**
- **G4 layering** — `harness/` never imports `bench`, `tests`, `baseline` or `agent`; `baseline/` never imports
  `harness`. Dependencies point inward: `agent.py` and `bench/` use `harness/`.
- **G5 no trailing whitespace** in `.py/.md/.toml/.json/.sh`. `bench/runs/*.log` are raw recorded output and
  exempt (`.gitattributes` marks them `-whitespace`); never "clean" them.

Gate mechanics, enforced by `scripts/check.py`:

- **T1–T3** the suite passes, nothing is skipped/xfailed, and at least `MIN_TESTS` tests ran. `MIN_TESTS` is a
  ratchet: raise it when you add tests, never lower it. Fix or delete a broken test; never skip-and-forget.
- **T4** the README line `# N deterministic tests` equals the real count.
- If you are about to change `check.py`/`guardrails.py` to make a failure go away, fix the cause instead.

Harness invariants (each pinned by the named tests; a red test means the invariant broke):

1. **Stable prompt head.** System prompt, base `tools[]` and their order, the task and requirement messages
   never contain changing data; load tools and skills by appending tool results. →
   `test_prompt_prefix_stability`, `test_lazy_prefix_unchanged_by_discovery`,
   `test_compaction_epochs_keep_stable_head_and_task`.
2. **Context edits are transactional and protocol-valid.** Never split an assistant tool-call message from its
   results; never rewrite or summarize away the task or a requirement update. →
   `test_tool_call_protocol_integrity`, `test_pinned_constraints`,
   `test_requirement_added_on_resume_survives_compaction_and_changes_the_result`.
3. **Never auto-retry bash or non-idempotent tools**, including on resume (`OUTCOME_UNKNOWN`). A new
   `ToolSpec` must set `read_only`/`idempotent` truthfully: they decide retries and resume safety. →
   `test_non_idempotent_retry_protection`, `test_only_read_only_idempotent_calls_are_repeated_after_a_crash`,
   `test_resume_after_crash_does_not_repeat_uncertain_side_effect`.
4. **Tool failures never escape a node.** Model-repairable problems become a structured
   `{"ok": false, "code", …}` result. → `test_error_recovery_returns_repairable_errors_to_model`,
   `test_malformed_invoke_does_not_crash_the_graph`.
5. **No artifact reference without an artifact.** With `artifact_offloading=False` nothing may point to
   `read_artifact`. → `test_artifact_offloading_off_keeps_limits_without_dangling_references`.

Reviewer-enforced (no gate can see these):

6. **Benchmark claims come only from `bench/results.jsonl` rows**, with sample size and `code_revision`;
   synthetic tools and reduced context windows are labelled; the agent's final message is never evidence.
   Never edit result numbers by hand; regenerate tables with `bench/report.py`.
7. **Do not change code in this tree while a benchmark is running**: each mode starts a new process that imports
   the code on disk, so later modes would measure different code. Work in a `git worktree` and merge after.
8. **Do not push** without the owner's explicit OK.

## Layout

- New tool → a `ToolSpec` in `harness/catalog.py` (discovery, validation and dispatch pick it up from there).
- New skill → `skills/<name>/SKILL.md` with front matter `name`, `version`, `description`, `when_to_use`, plus
  `references/`.
- New context strategy → a class in `harness/context.py` with `apply(candidate, target, estimate, config)`,
  added to the cascade in `ContextManager.__init__`, cheapest first.
- New flag or mode → `HarnessConfig` / `MODES` in `harness/config.py`; prompt and base tools derive from flags
  in `harness/prompt.py`, never from a mode name.
- New benchmark ticket → `bench/tickets.py` (+ hidden test in `bench/hidden_tests/`); validate it with a
  reference solution and a naive wrong one before any live run.
- `harness/graph.py` only orchestrates; logic goes into the module that owns it.

## Testing

Deterministic only: `tests/fakes.py` (`ScriptedLLM`, `run_script`) for graph behaviour; `tests/fake_openai.py`
(local OpenAI-compatible server) when the real CLI or `bench/run.py` must run in a subprocess. No external
network in tests. A behavioural bug fix ships with the test that reproduces it.

## Documentation

| You changed | You update, in the same commit |
|---|---|
| harness behaviour, flags, CLI | the README section for that area + a `docs/DEVLOG.md` entry |
| benchmark data | README §10 from `bench/report.py` output, with code revision and n |
| number of tests | `MIN_TESTS` in `scripts/check.py` (up only) and the README count |
| a rule in this file | its gate or named test, in the same commit |

`docs/DEVLOG.md` is append-only: add entries, never rewrite history; correct a wrong claim with a new entry.
A stale doc is a bug: fix or delete it on sight.
