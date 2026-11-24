# git bisect run

```sh
git stash            # only if the tree is dirty and you must check out revisions
git bisect start HEAD <known-good>
git bisect run python -c "from inventory.tax import compute_tax; assert compute_tax(100, 'GB') == 20.0"
git bisect reset
git stash pop
```
Exit code 0 = good, 1-124 = bad, 125 = skip. The first bad commit is printed at the end.
