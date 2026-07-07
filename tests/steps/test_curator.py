"""Tests for the curator (SPEC §6.6)."""

from datetime import UTC, datetime

from ragsynth.domain import AnnotationRecord, Seed, Stratum, SyntheticQuery
from ragsynth.pipeline.base import PipelineState, Resources
from ragsynth.steps.curator import Curator


def _record(
    resources: Resources, text: str, query_type: str = "factoid", emb_ref: str | None = None
) -> AnnotationRecord:
    stratum = Stratum(dims={"query_type": query_type})
    seed = Seed(
        seed_id=f"s-{text}",
        chunk_ids=(resources.chunks[0].chunk_id,),
        cluster_id=0,
        stratum=stratum,
    )
    query = SyntheticQuery(
        query_id=f"q-{text}", text=text, seed=seed, embedding_ref=emb_ref, gen_meta={}
    )
    return AnnotationRecord(
        record_id=f"r-{text}",
        query=query,
        qrels={resources.chunks[0].chunk_id: 1},
        stratum=stratum,
        benchmark_version="v1",
        created_at=datetime(2026, 7, 7, tzinfo=UTC),
    )


def test_final_dedup_drops_exact_text_repeats(min_resources: Resources) -> None:
    state = PipelineState(
        accepted=[
            _record(min_resources, "same question?"),
            _record(min_resources, "same question?"),
            _record(min_resources, "other question?"),
        ]
    )
    state = Curator(min_resources).run(state)
    assert [r.query.text for r in state.accepted] == ["same question?", "other question?"]


def test_memorization_flagging_keeps_but_flags(min_resources: Resources) -> None:
    # Give the record the EXACT embedding of a production train query.
    train_id = min_resources.queries_train[0].query_id
    min_resources.embeddings.add(["memorized-emb"], min_resources.embeddings.get([train_id]))
    memorized = _record(min_resources, "verbatim production query?", emb_ref="memorized-emb")
    fresh = _record(min_resources, "novel question?")
    state = Curator(min_resources, memorization_cos=0.9).run(
        PipelineState(accepted=[memorized, fresh])
    )
    assert len(state.accepted) == 2
    flagged = state.accepted[0]
    assert flagged.gate_meta["memorization_flag"] is True
    assert flagged.gate_meta["memorization_cos"] >= 0.99
    assert "memorization_flag" not in state.accepted[1].gate_meta
    assert state.metrics["curator_memorization_flags"] == 1


def test_target_mix_stratified_subsample(min_resources: Resources) -> None:
    records = [_record(min_resources, f"factoid {i}?", "factoid") for i in range(8)] + [
        _record(min_resources, f"howto {i}?", "howto") for i in range(2)
    ]
    state = Curator(
        min_resources, target_mix={"query_type=factoid": 0.5, "query_type=howto": 0.5}
    ).run(PipelineState(accepted=records))
    # howto is the binding stratum (2 records at 50% => total 4).
    counts: dict[str, int] = {}
    for r in state.accepted:
        counts[r.stratum.key()] = counts.get(r.stratum.key(), 0) + 1
    assert counts == {"query_type=factoid": 2, "query_type=howto": 2}


def test_max_records_cap_preserves_order(min_resources: Resources) -> None:
    records = [_record(min_resources, f"q {i}?") for i in range(10)]
    state = Curator(min_resources, max_records=4).run(PipelineState(accepted=records))
    assert len(state.accepted) == 4
    texts = [r.query.text for r in state.accepted]
    assert texts == sorted(texts, key=lambda t: int(t.split()[1].rstrip("?")))


def test_deterministic_under_seed(min_resources: Resources) -> None:
    records = [_record(min_resources, f"q {i}?") for i in range(10)]
    a = Curator(min_resources, max_records=4).run(PipelineState(accepted=list(records)))
    b = Curator(min_resources, max_records=4).run(PipelineState(accepted=list(records)))
    assert [r.record_id for r in a.accepted] == [r.record_id for r in b.accepted]


def test_config_round_trip(min_resources: Resources) -> None:
    step = Curator(min_resources, memorization_cos=0.85, max_records=100)
    config = step.to_config()
    assert config == {"memorization_cos": 0.85, "target_mix": None, "max_records": 100}
    assert Curator.from_config(config, min_resources).to_config() == config


def test_records_without_embedding_skip_memorization(min_resources: Resources) -> None:
    state = Curator(min_resources).run(
        PipelineState(accepted=[_record(min_resources, "no embedding?")])
    )
    assert len(state.accepted) == 1
    assert "memorization_flag" not in state.accepted[0].gate_meta
