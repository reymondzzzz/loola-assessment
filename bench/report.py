import json, statistics, sys
from collections import defaultdict
from pathlib import Path

RESULTS_FILE = Path(__file__).resolve().parent / "results.jsonl"
COLUMNS = [("runs", None), ("success", "success"), ("tickets", "tickets_closed"), ("turns", "turns"),
           ("input tok", "input_tokens"), ("cached %", None), ("uncached tok", None), ("output tok", "output_tokens"),
           ("summarizer tok", None), ("peak ctx", "peak_context"), ("compactions", "compactions"),
           ("wall s", "wall_seconds")]


def mean(values: list[float]) -> str:
    return f"{statistics.mean(values):,.0f}" if values else "—"


def row(label: str, runs: list[dict]) -> str:
    cells = [label, str(len(runs)), f"{sum(r['success'] for r in runs)}/{len(runs)}",
             mean([r.get("tickets_closed", 0) for r in runs]), mean([r.get("turns", 0) for r in runs])]
    inputs = [r.get("input_tokens", 0) for r in runs]
    cached = [r.get("cached_tokens", 0) for r in runs]
    cells += [mean(inputs), f"{100 * sum(cached) / max(sum(inputs), 1):.0f}%",
              mean([i - c for i, c in zip(inputs, cached)]), mean([r.get("output_tokens", 0) for r in runs]),
              mean([r.get("summarizer_input_tokens", 0) + r.get("summarizer_output_tokens", 0) for r in runs]),
              mean([r.get("peak_context", 0) for r in runs]), mean([r.get("compactions", 0) for r in runs]),
              mean([r.get("wall_seconds", 0) for r in runs])]
    return "| " + " | ".join(cells) + " |"


def main(labels: list[str]) -> None:
    groups: dict[str, list[dict]] = defaultdict(list)
    for line in RESULTS_FILE.read_text().splitlines():
        result = json.loads(line)
        groups[result["label"]].append(result)
    print("| mode | " + " | ".join(name for name, _ in COLUMNS) + " |")
    print("|" + "---|" * (len(COLUMNS) + 1))
    for label in labels or groups:
        if label in groups:
            print(row(label, groups[label]))


if __name__ == "__main__":
    main(sys.argv[1:])
