from collections import Counter


class MetricsCollector:
    def __init__(self, exported: dict | None = None):
        exported = exported or {}
        self.counters: Counter = Counter(exported.get("counters", {}))
        self.peak_context = exported.get("peak_context", 0)

    def export(self) -> dict:
        return {"counters": dict(self.counters), "peak_context": self.peak_context}

    def record_llm_call(self, usage: dict, purpose: str = "agent") -> None:
        prefix = "" if purpose == "agent" else f"{purpose}_"
        self.counters[f"{prefix}model_calls"] += 1
        self.counters[f"{prefix}input_tokens"] += usage["input"]
        self.counters[f"{prefix}output_tokens"] += usage["output"]
        self.counters[f"{prefix}cached_tokens"] += usage["cached"]
        if purpose == "agent":
            self.peak_context = max(self.peak_context, usage["input"])

    def count(self, name: str, amount: int = 1) -> None:
        self.counters[name] += amount

    def snapshot(self) -> dict:
        counters = dict(self.counters)
        total_input = counters.get("input_tokens", 0) + counters.get("summarizer_input_tokens", 0)
        total_cached = counters.get("cached_tokens", 0) + counters.get("summarizer_cached_tokens", 0)
        return {**counters, "peak_context": self.peak_context, "total_input_tokens": total_input,
                "total_cached_tokens": total_cached, "total_uncached_input_tokens": total_input - total_cached}
