---
name: sql-reporting
version: 1
description: Produce correct aggregate reports from a SQL database.
when_to_use: A task asks for top-N, totals or revenue from SQLite/SQL data, especially with dates.
---
# SQL reporting

1. Inspect the schema first: `SELECT sql FROM sqlite_master`, then sample rows of each table.
2. Timestamps stored as text may carry different UTC offsets. Do not filter by string comparison;
   normalise to UTC first (in Python: `datetime.fromisoformat(s).astimezone(timezone.utc)`).
3. Month boundaries: `start <= t < next_month_start`, in UTC.
4. Filter by status exactly as specified (e.g. only 'paid'; refunded/pending excluded).
5. Money: sum with `Decimal(str(value))`, round at the end, and break ties on the rounded value with the
   specified key (e.g. smaller id first). Float sums can make equal totals look different.
6. Join names by id, not by name: different customers can share a name.

SQLite date functions and pitfalls: read_skill_resource("sql-reporting", "references/sqlite-datetime.md").
