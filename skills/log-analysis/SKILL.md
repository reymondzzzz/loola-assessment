---
name: log-analysis
version: 1
description: Count or aggregate structured log entries correctly.
when_to_use: A task asks for counts, rates or top-N from a log file.
---
# Log analysis

1. Look at the format first: `head -20` and `grep -vnE '^<timestamp-regex>'` to find lines that are NOT entries
   (stack traces, wrapped messages). Never count those as entries.
2. Parse fields with an anchored regex on the entry structure, e.g.
   `^\d{4}-\d\d-\d\dT\S+ (\w+) \[([^\]]+)\] `. Never `grep LEVEL`: the level word also appears inside messages
   and stack traces.
3. Compare fields exactly (`level == "ERROR"`, service names as whole tokens: `auth` is not `auth-legacy`).
4. Strip `\r` (CRLF lines) before matching, or use `splitlines()`.
5. Cross-check: total of your per-key counts == number of entry lines with that level.
6. Ties: apply the tie-break rule from the task explicitly; `Counter.most_common` keeps insertion order on ties.

More regex patterns: read_skill_resource("log-analysis", "references/regex-cookbook.md").
