# CSV dialect notes

- `csv.Sniffer().sniff(sample)` guesses delimiter/quoting; prefer explicit settings when the format is known.
- Excel exports often use `;` in European locales and CRLF line endings (`newline=""` handles CRLF).
- `quotechar='"'` with `doublequote=True` (default) turns `""` into `"` inside a quoted field.
