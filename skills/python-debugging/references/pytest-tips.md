# Pytest tips

- `python -m pytest -q path::test_name` runs one test; `-x` stops at the first failure; `-vv` full diffs.
- `python -m pytest --lf` reruns only last failures.
- Quick property check: `python -c "from pkg.mod import f; print(f(0), f(-1), f(10))"`.
