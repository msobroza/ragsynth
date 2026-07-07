"""Tests for the frozen pydantic v2 domain model (SPEC §4)."""

from datetime import UTC, datetime

import pydantic
import pytest

from ragsynth.domain import (
    AnnotationRecord,
    Chunk,
    GenerationContext,
    ProductionQuery,
    Rejection,
    Seed,
    Stratum,
    SyntheticQuery,
    Turn,
)


@pytest.fixture
def stratum() -> Stratum:
    return Stratum(dims={"query_type": "factoid"})


@pytest.fixture
def seed(stratum: Stratum) -> Seed:
    return Seed(seed_id="s1", chunk_ids=("c1", "c2"), cluster_id=3, stratum=stratum)


@pytest.fixture
def query(seed: Seed) -> SyntheticQuery:
    return SyntheticQuery(
        query_id="q1",
        text="how do I rebalance the index?",
        seed=seed,
        embedding_ref="q1",
        gen_meta={"model": "mock", "prompt_version": "answer_first_v1"},
    )


def test_all_domain_objects_are_frozen(seed: Seed, query: SyntheticQuery) -> None:
    chunk = Chunk.create(text="hello world", doc_id="d1")
    for obj, field, value in [
        (chunk, "text", "x"),
        (seed, "cluster_id", 0),
        (query, "text", "x"),
        (Stratum(dims={}), "dims", {}),
    ]:
        with pytest.raises(pydantic.ValidationError):
            setattr(obj, field, value)


def test_stratum_key_is_sorted_canonical() -> None:
    s = Stratum(dims={"persona": "broker", "query_type": "factoid"})
    t = Stratum(dims={"query_type": "factoid", "persona": "broker"})
    assert s.key() == "persona=broker|query_type=factoid"
    assert s.key() == t.key()


def test_chunk_create_is_deterministic_and_content_addressed() -> None:
    a = Chunk.create(text="Interest  accrues \n daily.", doc_id="doc-9", page=4)
    b = Chunk.create(text="Interest accrues daily.", doc_id="doc-9", page=4)
    assert a.chunk_id == b.chunk_id  # whitespace-normalized
    assert a.content_hash == b.content_hash
    assert len(a.chunk_id) == 16
    assert int(a.chunk_id, 16) >= 0  # hex
    c = Chunk.create(text="Interest accrues daily.", doc_id="doc-9", page=5)
    assert c.chunk_id != a.chunk_id  # page participates in the id
    assert c.content_hash == a.content_hash  # ...but not in the content hash


def test_chunk_metadata_defaults_are_independent() -> None:
    a = Chunk.create(text="t", doc_id="d")
    b = Chunk.create(text="t2", doc_id="d")
    assert a.metadata == {}
    assert a.metadata is not b.metadata


def test_production_query_optional_fields() -> None:
    q = ProductionQuery(query_id="p1", text="fee schedule?")
    assert q.timestamp is None
    assert q.stratum is None
    assert q.embedding_ref is None


def test_annotation_record_json_roundtrip(query: SyntheticQuery, stratum: Stratum) -> None:
    record = AnnotationRecord(
        record_id="r1",
        query=query,
        qrels={"c1": 1, "c2": 1, "c9": 1},
        crucial=("c1",),
        stratum=stratum,
        gate_meta={"round_trip": {"passed": True, "score": 1.0}},
        content_hashes={"c1": "ab" * 8, "c2": "cd" * 8, "c9": "ef" * 8},
        benchmark_version="v1",
        created_at=datetime(2026, 7, 7, tzinfo=UTC),
    )
    restored = AnnotationRecord.model_validate_json(record.model_dump_json())
    assert restored == record
    assert restored.dialogue_context is None
    assert restored.supplemental == ()


def test_turn_role_is_validated() -> None:
    Turn(role="user", text="hi")
    with pytest.raises(pydantic.ValidationError):
        Turn(role="system", text="nope")


def test_generation_context_and_rejection(seed: Seed, query: SyntheticQuery) -> None:
    ctx = GenerationContext(
        seed=seed,
        chunk_texts=("evidence",),
        style_exemplars=("real query 1", "real query 2"),
        instruction="Write a factoid question.",
    )
    assert ctx.seed.seed_id == "s1"
    rej = Rejection(candidate=query, check="round_trip", reason="gold not in top-10", score=0.0)
    assert rej.check == "round_trip"
