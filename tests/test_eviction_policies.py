import pytest

from harness.config import HarnessConfig, mode_config
from harness.context import ContextManager, eviction_candidates
from harness.store import ArtifactStore
from tests.test_context import SYSTEM, TASK, history


def varied_history(tmp_path):
    sizes = {1: 150, 2: 900, 3: 300, 4: 600, 5: 120, 6: 80, 7: 80}
    candidate = history(tmp_path, 7, size=100)
    for turn, size in sizes.items():
        candidate.observations[f"c{turn}"]["token_count"] = size
    candidate.observations["c3"]["stale"] = True
    return candidate


def order(candidate, policy):
    config = HarnessConfig(eviction_policy=policy, recent_interactions=2)
    return [call_id for call_id, _ in eviction_candidates(candidate, config)]


def test_policies_order_the_same_history_differently(tmp_path):
    candidate = varied_history(tmp_path)
    fifo, size, default = order(candidate, "fifo"), order(candidate, "size"), order(candidate, "default")
    assert fifo == ["c1", "c2", "c3", "c4", "c5"]
    assert size == ["c2", "c4", "c3", "c1", "c5"]
    assert default == ["c3", "c2", "c4", "c1", "c5"]
    assert len({tuple(fifo), tuple(size), tuple(default)}) == 3


def test_common_guards_apply_to_every_policy(tmp_path):
    candidate = varied_history(tmp_path)
    candidate.observations["c2"]["pinned"] = True
    candidate.observations["c4"]["recoverable"] = False
    candidate.observations["c1"]["token_count"] = 10
    for policy in ("default", "fifo", "size"):
        chosen = order(candidate, policy)
        assert "c2" not in chosen and "c4" not in chosen and "c1" not in chosen
        assert "c6" not in chosen and "c7" not in chosen


def test_policy_drives_the_existing_masking_mechanism(tmp_path):
    masked = {}
    for policy in ("fifo", "size"):
        candidate = varied_history(tmp_path)
        config = HarnessConfig(observation_masking=True, eviction_policy=policy, recent_interactions=2,
                               context_window=10**6, output_reserve=0, safety_margin=0)
        manager = ContextManager(config, ArtifactStore(tmp_path), SYSTEM, TASK, [])
        from harness.context import ObservationMasking
        ObservationMasking().apply(candidate, manager.estimate(candidate) - 150, manager.estimate, config)
        masked[policy] = sorted(c for c, o in candidate.observations.items() if o["masked"])
    assert masked["fifo"] == ["c1"] and masked["size"] == ["c2"]


def test_unknown_policy_is_rejected():
    with pytest.raises(ValueError):
        HarnessConfig(eviction_policy="lru")
    assert mode_config("hybrid").eviction_policy == "default"
