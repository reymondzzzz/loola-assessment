---
name: git-regression-hunt
version: 1
description: Find the commit that introduced a regression and fix it.
when_to_use: Something "used to work" and the repository has history.
---
# Git regression hunt

1. Get a failing check you can run (a test or a one-line python -c assertion).
2. List candidate commits: `git log --oneline -- <file>`. The latest commit touching the file is often NOT
   the culprit; inspect each with `git show <hash> -- <file>`.
3. If several commits are plausible, bisect: `git bisect start HEAD <good>` then
   `git bisect run <check>`; finish with `git bisect reset`.
4. Do not lose uncommitted work: never `git checkout` old revisions with a dirty tree. Use
   `git show <rev>:<path>` to view old file versions instead.
5. Fix only the regression; keep unrelated changes from later commits.
6. Report the full 40-character hash (`git rev-parse <short>`).

Bisect details: read_skill_resource("git-regression-hunt", "references/bisect-run.md").
