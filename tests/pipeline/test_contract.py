"""LSP contract tests parametrized over the registries (SPEC §15.2, §3.2).

Every concrete step serializes/deserializes/runs; every GateCheck produces
a Rejection with a reason on its failing fixture; every adapter Protocol is
satisfied by its offline implementation. Keys with the ``test.`` prefix are
test doubles and skipped.
"""

from typing import Any

import numpy as np
import pytest

import ragsynth.arms
import ragsynth.datasets
import ragsynth.steps  # noqa: F401 - populate STEPS registry
from ragsynth.adapters.embedder.base import EMBEDDERS
from ragsynth.adapters.judge.base import JUDGES
from ragsynth.adapters.judge.mock import MockJudge
from ragsynth.adapters.llm.base import CHAT_MODELS
from ragsynth.adapters.retriever.base import RETRIEVERS
from ragsynth.domain import Seed, Stratum, SyntheticQuery
from ragsynth.gate.checks.base import CHECKS
from ragsynth.pipeline.base import STEPS, PipelineState, Resources
from ragsynth.pipeline.serialization import build_resources, make_initial_state
from ragsynth.steps.gate import VerificationGate
from tests.conftest import toy_config

EXPECTED_STEPS = {
    "seed_sampler.uniform",
    "seed_sampler.quota",
    "seed_sampler.spec",
    "context_assembler",
    "generator",
    "gate",
    "qrel_builder",
    "curator",
    "validator",
}
EXPECTED_CHECKS = {"dedup", "zero_context", "answerability", "round_trip", "uniqueness"}

# Constructor params per step for the contract run (small but functional).
STEP_PARAMS: dict[str, dict[str, Any]] = {
    "seed_sampler.uniform": {"n_seeds": 6, "p_group": 0.5},
    "seed_sampler.quota": {"n_seeds": 12, "n_min": 3, "p_group": 0.2},
    "seed_sampler.spec": {"n_seeds": 6, "n_chunks_per_seed": 2},
    "context_assembler": {"k_style": 2},
    "generator": {"n_candidates": 1},
    "gate": {"checks": ["dedup", "zero_context", "answerability"]},
    "qrel_builder": {"strategy": "binary"},
    "curator": {"memorization_cos": 0.9},
    "validator": {
        "arms": ["oracle"],
        "n_boot": 10,
        "n_per_arm": 8,
        "arm_params": {"oracle": {"n_records": 8}},
    },
}


def _registered(registry: Any) -> list[str]:
    keys = registry.keys()  # Registry method, not dict.keys
    return [key for key in keys if not key.startswith("test.")]


@pytest.fixture(scope="module")
def contract_resources(tmp_path_factory: pytest.TempPathFactory) -> Resources:
    """A real (small) toy world so every step can actually run."""
    return build_resources(toy_config(tmp_path_factory.mktemp("contract")))


def test_registries_hold_the_specced_inventory() -> None:
    assert set(_registered(STEPS)) == EXPECTED_STEPS
    assert set(_registered(CHECKS)) == EXPECTED_CHECKS


@pytest.mark.parametrize("key", sorted(EXPECTED_STEPS))
def test_step_config_fixed_point(key: str, contract_resources: Resources) -> None:
    step_class = STEPS.get(key)
    step = step_class.from_config(STEP_PARAMS[key], contract_resources)
    config = step.to_config()
    rebuilt = step_class.from_config(config, contract_resources)
    assert rebuilt.to_config() == config, f"{key} config not a fixed point"


def test_all_steps_run_in_sequence(contract_resources: Resources) -> None:
    """The full chain runs; fit is idempotent (SPEC §3.1, §15.2)."""
    order = [
        "seed_sampler.quota",
        "context_assembler",
        "generator",
        "gate",
        "qrel_builder",
        "curator",
        "validator",
    ]
    state = make_initial_state(toy_config(contract_resources.artifacts.root.parent.parent))
    state.provenance["config"] = {}  # keep the report light for the contract run
    for key in order:
        step = STEPS.get(key).from_config(STEP_PARAMS[key], contract_resources)
        step.fit(contract_resources)
        step.fit(contract_resources)  # idempotent by contract
        result = step.run(state)
        assert isinstance(result, PipelineState), key
        state = result
    assert state.seeds
    assert state.contexts
    assert state.candidates
    assert state.accepted
    assert "eval_report" in state.metrics


@pytest.mark.parametrize("key", sorted(EXPECTED_STEPS - {"validator", "gate"}))
def test_steps_never_mutate_resources(key: str, contract_resources: Resources) -> None:
    chunks_before = contract_resources.chunks
    p_hat_before = contract_resources.demand.p_hat.copy()
    step = STEPS.get(key).from_config(STEP_PARAMS[key], contract_resources)
    step.fit(contract_resources)
    assert contract_resources.chunks is chunks_before
    np.testing.assert_array_equal(contract_resources.demand.p_hat, p_hat_before)


def _failing_setup(
    key: str, resources: Resources
) -> tuple[Resources, SyntheticQuery, PipelineState]:
    """Per-check scenario that MUST produce a rejection."""
    chunk = resources.chunks[0]
    state = PipelineState()
    seed = Seed(
        seed_id=f"contract-{key}",
        chunk_ids=(chunk.chunk_id,),
        cluster_id=0,
        stratum=Stratum(dims={"query_type": "factoid"}),
    )
    emb_ref = f"contract-{key}-emb"
    if emb_ref not in resources.embeddings:
        resources.embeddings.add([emb_ref], resources.embeddings.get([chunk.chunk_id]))
    candidate = SyntheticQuery(
        query_id=f"q-{key}",
        text=f"contract candidate {key}?",
        seed=seed,
        embedding_ref=emb_ref,
        gen_meta={},
    )
    if key == "dedup":
        state.gate_accepted.append(candidate)  # exact duplicate of itself
        return resources, candidate, state
    if key == "zero_context":
        return (
            resources.with_overrides(judge=MockJudge(answerable_without_evidence=True)),
            candidate,
            state,
        )
    if key == "answerability":
        return (
            resources.with_overrides(judge=MockJudge(answerable_with_evidence=False)),
            candidate,
            state,
        )
    if key == "round_trip":
        # Gold = the chunk most dissimilar to the candidate embedding; k=1.
        embs = resources.chunk_embs()
        farthest = int(np.argmin(embs @ resources.embeddings.get([emb_ref])[0]))
        far_seed = seed.model_copy(update={"chunk_ids": (resources.chunks[farthest].chunk_id,)})
        return resources, candidate.model_copy(update={"seed": far_seed}), state
    if key == "uniqueness":
        # A judge that answers from ANY evidence: every non-gold hit is a leak.
        return (
            resources.with_overrides(judge=MockJudge(answerable_with_evidence=True)),
            candidate,
            state,
        )
    raise AssertionError(f"no failing fixture for {key}")


@pytest.mark.parametrize("key", sorted(EXPECTED_CHECKS))
def test_every_check_emits_rejection_with_reason(key: str, contract_resources: Resources) -> None:
    check_params: dict[str, Any] = {}
    if key == "round_trip":
        check_params = {"k": 1}
    if key == "uniqueness":
        check_params = {"mode": "reject", "top_m": 3}
    resources, candidate, state = _failing_setup(key, contract_resources)
    gate = VerificationGate(resources, checks=[key], **{key: check_params})
    state.candidates = [candidate]
    state = gate.run(state)
    assert len(state.rejected) == 1, f"{key} did not reject"
    rejection = state.rejected[0]
    assert rejection.check == key
    assert rejection.reason  # non-empty human-readable reason (SPEC §15.2)
    assert state.metrics["gate_reject_reasons"] == {key: 1}


def test_adapter_registries_resolve_and_mocks_satisfy_protocols(
    contract_resources: Resources,
) -> None:
    for registry, expected in (
        (CHAT_MODELS, {"mock", "openai_compatible", "toy_chat"}),
        (EMBEDDERS, {"mock", "hashed_ngram", "sentence_transformer", "passthrough"}),
        (RETRIEVERS, {"dense_inmemory", "bm25s"}),
        (JUDGES, {"mock", "llm", "toy_judge"}),
    ):
        assert expected <= set(_registered(registry))
    # Behavioral smoke through the Protocol surfaces on the live toy world.
    text = contract_resources.generator_llm.complete("system", "user [chunk] toychunk:0001")
    assert isinstance(text, str)
    embs = contract_resources.embedder.encode([contract_resources.chunks[0].text])
    assert embs.shape[0] == 1
    assert np.isclose(np.linalg.norm(embs[0]), 1.0)
    verdict = contract_resources.judge.judge(text, [contract_resources.chunks[0].text])
    assert isinstance(verdict.answerable, bool)
    hits = contract_resources.retriever.search(embs[0], k=3)
    assert len(hits) == 3
    assert all(isinstance(cid, str) and isinstance(score, float) for cid, score in hits)
