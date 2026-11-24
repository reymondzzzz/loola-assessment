from dataclasses import dataclass, replace

EVICTION_POLICIES = ("default", "fifo", "size")


@dataclass(frozen=True)
class HarnessConfig:
    lazy_tools: bool = True
    lazy_skills: bool = True
    native_binding: bool = False

    output_truncation: bool = False
    artifact_offloading: bool = False
    deduplication: bool = False

    observation_masking: bool = False
    stale_eviction: bool = False
    interaction_pruning: bool = False
    semantic_compaction: bool = False
    cache_aware: bool = False
    eviction_policy: str = "default"

    context_window: int = 400_000
    output_reserve: int = 4096
    safety_margin: int = 2048
    high_watermark: float = 0.80
    low_watermark: float = 0.50
    recent_interactions: int = 5
    tool_output_tokens: int = 2000

    max_turns: int = 150
    max_tool_calls: int = 400
    max_total_tokens: int = 3_000_000
    max_seconds: float = 1800
    llm_retries: int = 4
    llm_deadline_seconds: float = 300
    emergency_attempts: int = 2
    loop_limit: int = 3
    verify_command: str | None = None
    verify_timeout: float = 300
    verify_attempts: int = 2

    def __post_init__(self):
        if self.eviction_policy not in EVICTION_POLICIES:
            raise ValueError(f"eviction_policy must be one of {EVICTION_POLICIES}")

    @property
    def input_budget(self) -> int:
        return self.context_window - self.output_reserve - self.safety_margin


OUTPUT_PIPELINE = dict(output_truncation=True, artifact_offloading=True, deduplication=True)
MODES = {
    "eager": HarnessConfig(lazy_tools=False, lazy_skills=False),
    "lazy": HarnessConfig(),
    "native": HarnessConfig(native_binding=True),
    "masking": HarnessConfig(**OUTPUT_PIPELINE, observation_masking=True, stale_eviction=True, interaction_pruning=True,
                             cache_aware=True),
    "summary": HarnessConfig(**OUTPUT_PIPELINE, semantic_compaction=True),
    "hybrid": HarnessConfig(**OUTPUT_PIPELINE, observation_masking=True, stale_eviction=True, interaction_pruning=True,
                            semantic_compaction=True, cache_aware=True),
}


def mode_config(mode: str, context_window: int | None = None, **overrides) -> HarnessConfig:
    if context_window:
        overrides["context_window"] = context_window
    return replace(MODES[mode], **overrides)
