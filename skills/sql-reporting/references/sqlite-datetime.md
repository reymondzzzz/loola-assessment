# SQLite datetime pitfalls

- `datetime('2026-04-01T01:30:00+02:00')` returns `2026-03-31 23:30:00` (UTC); string comparison of the raw
  text does not.
- `strftime('%Y-%m', created_at)` also converts offsets to UTC.
- `ROUND(x, 2)` in SQLite uses binary floats; for money, aggregate in Python with Decimal.
- `GROUP BY name` merges different customers with the same name; group by id.
