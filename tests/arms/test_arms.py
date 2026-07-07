"""Tests for the A0/A1/A2/ORACLE arm presets (SPEC §10)."""

from typing import Any

import pytest

from ragsynth.arms.base import ARMS, run_arm
from ragsynth.pipeline.base import Resources
from ragsynth.pipeline.registry import RegistryError

# Gate without retrieval checks: candidate embeddings in the unit-test world
# come from MockEmbedder (random directions), so round_trip would reject
# everything. The toy-world e2e exercises the full check chain.
LIGHT_PARAMS: dict[str, Any] = {
    "n_seeds": 6,
    "gate": {"checks": ["dedup", "zero_context", "answerability"]},
    "quota": {"n_min": 2},
}


class _SpyChat:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, system: str, user: str, **kwargs: Any) -> str:
        self.calls += 1
        return f"spy generated query {self.calls}?"


def test_all_arms_registered() -> None:
    registered = ARMS.keys()
    assert registered == ["a0", "a1", "a2", "oracle"]
    with pytest.raises(RegistryError, match="unknown arm"):
        ARMS.get("a3")


def test_a0_uses_uniform_seeds_and_no_exemplars(steps_world: Resources) -> None:
    preset = ARMS.get("a0")()
    _resources, steps = preset.build_steps(steps_world, LIGHT_PARAMS)
    names = [s.name for s in steps]
    assert names == [
        "seed_sampler.uniform",
        "context_assembler",
        "generator",
        "gate",
        "qrel_builder",
        "curator",
    ]
    assembler = steps[1]
    assert assembler.to_config()["k_style"] == 0  # no steering (SPEC §10)


def test_a1_uses_quota_and_exemplars(steps_world: Resources) -> None:
    preset = ARMS.get("a1")()
    _resources, steps = preset.build_steps(steps_world, LIGHT_PARAMS)
    assert steps[0].name == "seed_sampler.quota"
    assert steps[0].to_config()["lam"] == 0.7
    assert steps[1].to_config()["k_style"] == 3


def test_a2_uses_spec_sampler(steps_world: Resources) -> None:
    preset = ARMS.get("a2")()
    _resources, steps = preset.build_steps(steps_world, LIGHT_PARAMS)
    assert steps[0].name == "seed_sampler.spec"
    assert steps[1].to_config()["k_style"] == 3


def test_generative_arms_produce_records(steps_world: Resources) -> None:
    for arm in ("a0", "a1", "a2"):
        records = run_arm(arm, steps_world, LIGHT_PARAMS)
        assert records, f"arm {arm} produced no records"
        for record in records:
            assert record.qrels
            assert record.query.gen_meta["gate"]  # went through the gate


def test_llm_override_swaps_generator(steps_world: Resources) -> None:
    spy = _SpyChat()
    preset = ARMS.get("a0")()
    resources, _steps = preset.build_steps(
        steps_world, {**LIGHT_PARAMS, "llm_override": {"type": "mock", "params": {"seed": 9}}}
    )
    # The override built a NEW MockChatModel(seed=9), not the world's seed=0 one.
    assert resources.generator_llm is not steps_world.generator_llm
    assert resources.generator_llm.to_config() == {"seed": 9}
    del spy  # spy pattern exercised in test_oracle below


def test_oracle_needs_no_llm_and_uses_oracle_split(steps_world: Resources) -> None:
    spy = _SpyChat()
    resources = steps_world.with_overrides(
        generator_llm=spy,
        oracle_qrels={
            q.query_id: {steps_world.chunks[0].chunk_id: 1} for q in steps_world.queries_oracle
        },
    )
    records = run_arm("oracle", resources, {"n_records": 3})
    assert spy.calls == 0
    assert len(records) == 3
    oracle_ids = {q.query_id for q in resources.queries_oracle}
    for record in records:
        assert record.query.query_id in oracle_ids
        assert record.qrels == {steps_world.chunks[0].chunk_id: 1}
        assert record.query.gen_meta["arm"] == "oracle"
        assert record.query.embedding_ref == record.query.query_id


def test_oracle_caps_at_split_size(steps_world: Resources) -> None:
    resources = steps_world.with_overrides(
        oracle_qrels={
            q.query_id: {steps_world.chunks[0].chunk_id: 1} for q in steps_world.queries_oracle
        }
    )
    records = run_arm("oracle", resources, {"n_records": 100})
    assert len(records) == len(resources.queries_oracle)
