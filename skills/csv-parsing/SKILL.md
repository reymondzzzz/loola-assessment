---
name: csv-parsing
version: 1
description: Parse real-world CSV files robustly.
when_to_use: A task reads or aggregates a CSV file.
---
# CSV parsing

1. Always use the `csv` module; never `line.split(",")` (quoted fields contain commas and `""` escapes).
2. Open with `encoding="utf-8-sig"` (strips a BOM, otherwise the first header becomes '﻿name')
   and `newline=""`.
3. `csv.DictReader(f, skipinitialspace=True)`, and still `.strip()` values: trailing spaces survive.
4. Blank lines are skipped by DictReader; header-only files yield zero rows (handle totals of 0).
5. Money: `Decimal(value)` and format with `f"{total:.2f}"`.
6. Duplicate keys (same name on several rows) may need merging; follow the task's rule.
7. CLI errors: catch `FileNotFoundError` at the entry point, print to stderr, `sys.exit(1)`.

Dialect notes: read_skill_resource("csv-parsing", "references/dialects.md").
