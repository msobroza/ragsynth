"""Tests for the binary qrel builder (SPEC §6.5)."""

import pytest

from ragsynth.domain import Seed, Stratum, SyntheticQuery
from ragsynth.pipeline.base import PipelineState, Resources
from ragsynth.pipeline.registry import RegistryError
from ragsynth.steps.qrel_builder import QrelBuilder


def _accepted_candidate(
    resources: Resources, text: str, gold: tuple[int, ...], promoted: tuple[int, ...] = ()
) -> SyntheticQuery:
    seed = Seed(
        seed_id=f"s-{text}",
        chunk_ids=tuple(resources.chunks[i].chunk_id for i in gold),
        cluster_id=0,
        stratum=Stratum(dims={"query_type": "howto"}),
    )
    return SyntheticQuery(
        query_id=f"q-{text}",
        text=text,
        seed=seed,
        embedding_ref=None,
        gen_meta={
            "gate": {"round_trip": {"passed": True, "score": 1.0, "reason": "gold at rank 1"}},
            "promoted": [resources.chunks[i].chunk_id for i in promoted],
        },
    )


def test_binary_qrels_include_seed_chunks_and_promotions(min_resources: Resources) -> None:
    state = PipelineState(
        gate_accepted=[_accepted_candidate(min_resources, "q1", gold=(0, 1), promoted=(2,))],
        provenance={"benchmark_version": "v1-test@abc123"},
    )
    state = QrelBuilder(min_resources).run(state)
    assert len(state.accepted) == 1
    record = state.accepted[0]
    expected_gold = {min_resources.chunks[i].chunk_id for i in (0, 1, 2)}
    assert set(record.qrels) == expected_gold
    assert all(grade == 1 for grade in record.qrels.values())
    # crucial = all gold in v1 (SPEC §4).
    assert set(record.crucial) == expected_gold
    # content_hashes snapshot every annotated chunk (v2 lifecycle key).
    assert set(record.content_hashes) == expected_gold
    for cid in expected_gold:
        assert record.content_hashes[cid] == min_resources.chunk_index[cid].content_hash
    assert record.stratum.dims == {"query_type": "howto"}
    assert record.gate_meta["round_trip"]["passed"] is True
    assert record.benchmark_version == "v1-test@abc123"
    assert record.dialogue_context is None


def test_record_ids_deterministic_and_unique(min_resources: Resources) -> None:
    state = PipelineState(
        gate_accepted=[
            _accepted_candidate(min_resources, "q1", gold=(0,)),
            _accepted_candidate(min_resources, "q2", gold=(1,)),
        ]
    )
    ids_a = [r.record_id for r in QrelBuilder(min_resources).run(state).accepted]
    state_b = PipelineState(
        gate_accepted=[
            _accepted_candidate(min_resources, "q1", gold=(0,)),
            _accepted_candidate(min_resources, "q2", gold=(1,)),
        ]
    )
    ids_b = [r.record_id for r in QrelBuilder(min_resources).run(state_b).accepted]
    assert ids_a == ids_b
    assert len(set(ids_a)) == 2


def test_unknown_strategy_raises_with_known_keys(min_resources: Resources) -> None:
    with pytest.raises(RegistryError, match="unknown qrel strategy 'graded'"):
        QrelBuilder(min_resources, strategy="graded")


def test_config_round_trip(min_resources: Resources) -> None:
    step = QrelBuilder(min_resources)
    assert step.to_config() == {"strategy": "binary"}
    assert QrelBuilder.from_config(step.to_config(), min_resources).to_config() == {
        "strategy": "binary"
    }


def test_relabel_nearest_strategy(min_resources: Resources) -> None:
    """Gold is the single nearest chunk of the emitted query (SPEC §10)."""
    # Candidate embedding == chunk 1's vector, but seed gold is chunk 0.
    cand = _accepted_candidate(min_resources, "relabel me", gold=(0,))
    ref = f"emb-{cand.query_id}"
    min_resources.embeddings.add(
        [ref], min_resources.embeddings.get([min_resources.chunks[1].chunk_id])
    )
    relabeled = cand.model_copy(update={"embedding_ref": ref})
    state = PipelineState(gate_accepted=[relabeled])
    state = QrelBuilder(min_resources, strategy="relabel_nearest").run(state)
    record = state.accepted[0]
    assert record.qrels == {min_resources.chunks[1].chunk_id: 1}
    assert record.crucial == (min_resources.chunks[1].chunk_id,)
