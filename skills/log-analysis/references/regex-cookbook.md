# Regex cookbook for logs

| Need | Pattern |
|---|---|
| ISO timestamp at line start | `^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})` |
| Level token after timestamp | `^\S+ (DEBUG|INFO|WARN|ERROR) ` |
| Bracketed service | `\[([A-Za-z0-9_-]+)\]` |
| Continuation line | `^(\s|Traceback|[A-Za-z]+Error:)` |

Python skeleton:
```python
import re, collections
entry = re.compile(r"^\d{4}-\d\d-\d\dT\S+ (\w+) \[([^\]]+)\] ")
counts = collections.Counter()
for line in open(path, encoding="utf-8").read().splitlines():
    m = entry.match(line)
    if m and m.group(1) == "ERROR":
        counts[m.group(2)] += 1
```
