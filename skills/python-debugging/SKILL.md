---
name: python-debugging
version: 1
description: Fix failing Python tests by matching documented behaviour, including edge cases.
when_to_use: Tests fail or code disagrees with its docstrings.
---
# Python debugging

1. Run the failing tests first and read the assertion diff.
2. Treat docstrings as the specification. Implement every sentence: bounds, inclusive/exclusive thresholds
   ("10 or more" means `>=`), errors to raise (type matters: KeyError vs ValueError), side effects.
3. Check edge cases the visible tests do not cover: zero, negatives, exact thresholds, empty inputs,
   removing exactly all stock, unknown keys.
4. Aliasing: storing a caller's mutable object and later mutating it changes the caller's object.
   Copy when the docstring says the argument must not be modified.
5. Rerun the whole test suite after each fix, not just one test.

Pytest tricks: read_skill_resource("python-debugging", "references/pytest-tips.md").
