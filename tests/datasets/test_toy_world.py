"""Tests for the geometric toy world: bundle invariants + coupled toy adapters."""

import dataclasses
import hashlib
import logging

import numpy as np
import pytest

from ragsynth.adapters.embedder.base import EMBEDDERS
from ragsynth.adapters.judge.base import JUDGES
from ragsynth.adapters.llm.base import CHAT_MODELS
from ragsynth.datasets.base import DATASETS, DatasetBundle, EmbeddingBank
from ragsynth.datasets.toy_world import (
    PassthroughEmbedder,
    ToyChatModel,
    ToyJudge,
    ToyWorldDataset,
)
from ragsynth.pipeline.base import stable_hash64
from ragsynth.sampling.vmf import l2_normalize, sphere_uniform

SMALL = {"d": 16, "k_true": 4, "n_chunks": 80, "n_prod": 400}


@pytest.fixture(scope="module")
def bundle() -> DatasetBundle:
    return ToyWorldDataset.build(dict(SMALL), seed=0)


def _chunk_matrix(bundle: DatasetBundle) -> np.ndarray:
    return bundle.embeddings.get([c.chunk_id for c in bundle.chunks]).astype(np.float64)


# ---------------------------------------------------------------------------
# Bundle invariants
# ---------------------------------------------------------------------------


def test_registrations() -> None:
    assert DATASETS.get("toy_world") is ToyWorldDataset
    assert CHAT_MODELS.get("toy_chat") is ToyChatModel
    assert JUDGES.get("toy_judge") is ToyJudge
    assert EMBEDDERS.get("passthrough") is PassthroughEmbedder


def test_split_sizes(bundle) -> None:
    assert len(bundle.queries_train) == 240
    assert len(bundle.queries_anchor) == 100
    assert len(bundle.queries_oracle) == 60


def test_chunk_text_and_doc_conventions(bundle) -> None:
    assert len(bundle.chunks) == 80
    for i, chunk in enumerate(bundle.chunks):
        assert chunk.text == f"toychunk:{i:04d}"
        assert chunk.doc_id == f"toydoc:{i // 5}"
        assert chunk.content_hash  # built via Chunk.create


def test_query_text_conventions(bundle) -> None:
    for split, queries in (
        ("train", bundle.queries_train),
        ("anchor", bundle.queries_anchor),
        ("oracle", bundle.queries_oracle),
    ):
        for i, q in enumerate(queries):
            assert q.text == f"toyquery:{split}:{i:05d}"


def test_chunks_per_component(bundle) -> None:
    # The build's first rng consumption is the component means: replay it.
    rng = np.random.default_rng([0, stable_hash64("toy_world")])
    mus = sphere_uniform(SMALL["k_true"], SMALL["d"], rng)
    assign = np.argmax(_chunk_matrix(bundle) @ mus.T, axis=1)
    assert np.bincount(assign, minlength=4).tolist() == [20, 20, 20, 20]


def test_all_ids_in_store_and_texts_in_bank(bundle) -> None:
    for chunk in bundle.chunks:
        assert chunk.chunk_id in bundle.embeddings
        assert chunk.text in bundle.bank
    for queries in (bundle.queries_train, bundle.queries_anchor, bundle.queries_oracle):
        for q in queries:
            assert q.query_id in bundle.embeddings
            assert q.text in bundle.bank


def test_all_vectors_unit_norm(bundle) -> None:
    ids = [c.chunk_id for c in bundle.chunks] + [
        q.query_id
        for queries in (bundle.queries_train, bundle.queries_anchor, bundle.queries_oracle)
        for q in queries
    ]
    norms = np.linalg.norm(bundle.embeddings.get(ids), axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_qrels_are_nearest_chunk_gold(bundle) -> None:
    chunk_ids = [c.chunk_id for c in bundle.chunks]
    matrix = _chunk_matrix(bundle)
    # verify one anchor and one oracle qrel by hand against the store matrices
    for queries, qrels in (
        (bundle.queries_anchor, bundle.anchor_qrels),
        (bundle.queries_oracle, bundle.oracle_qrels),
    ):
        q = queries[0]
        qv = bundle.embeddings.get([q.query_id])[0].astype(np.float64)
        nearest = chunk_ids[int(np.argmax(matrix @ qv))]
        assert qrels[q.query_id] == {nearest: 1}


def test_qrels_cover_anchor_and_oracle_splits(bundle) -> None:
    assert set(bundle.anchor_qrels) == {q.query_id for q in bundle.queries_anchor}
    assert set(bundle.oracle_qrels) == {q.query_id for q in bundle.queries_oracle}
    for grades in bundle.anchor_qrels.values():
        assert len(grades) == 1
        assert set(grades.values()) == {1}


def test_build_determinism_byte_identical(bundle) -> None:
    again = ToyWorldDataset.build(dict(SMALL), seed=0)
    assert [c.chunk_id for c in again.chunks] == [c.chunk_id for c in bundle.chunks]
    assert _chunk_matrix(again).tobytes() == _chunk_matrix(bundle).tobytes()
    q_ids = [q.query_id for q in bundle.queries_train]
    assert bundle.embeddings.get(q_ids).tobytes() == again.embeddings.get(q_ids).tobytes()


# ---------------------------------------------------------------------------
# ToyChatModel
# ---------------------------------------------------------------------------


def test_toy_chat_deterministic_per_prompt(bundle) -> None:
    user = f"write a question from {bundle.chunks[0].text}"
    t1 = ToyChatModel(bundle.bank, seed=0, d=16).complete("sys", user)
    v1 = bundle.bank.get(t1).copy()
    t2 = ToyChatModel(bundle.bank, seed=0, d=16).complete("sys", user)
    assert t1 == t2
    np.testing.assert_array_equal(v1, bundle.bank.get(t2))
    assert np.isclose(np.linalg.norm(v1), 1.0)


def test_toy_chat_text_contract_and_banked(bundle) -> None:
    chat = ToyChatModel(bundle.bank, seed=7, d=16)
    system, user = "sys prompt", f"evidence: {bundle.chunks[3].text}"
    text = chat.complete(system, user)
    digest = hashlib.sha256(f"{system}\x00{user}\x00{7}".encode()).hexdigest()[:12]
    assert text == f"toysynth:{digest}"
    assert text in bundle.bank


def test_toy_chat_emission_matches_prototype_emit(bundle) -> None:
    seed, d, style, noise = 0, 16, 0.15, 0.68
    chat = ToyChatModel(bundle.bank, style=style, noise=noise, seed=seed, d=d)
    system, user = "gen", f"ground: {bundle.chunks[5].text}"
    text = chat.complete(system, user)
    base = l2_normalize(bundle.bank.get(bundle.chunks[5].text))
    style_rng = np.random.default_rng([seed, stable_hash64("toy_style_dir")])
    style_dir = l2_normalize(style_rng.standard_normal(d))
    rng = np.random.default_rng([seed, stable_hash64(f"{system}\x00{user}")])
    g = l2_normalize(rng.standard_normal(d))
    expected = l2_normalize(base + style * style_dir + noise * g)
    np.testing.assert_allclose(bundle.bank.get(text), expected, atol=1e-12)


def test_toy_chat_parses_chunks_and_stays_near_base(bundle) -> None:
    chat = ToyChatModel(bundle.bank, style=0.05, noise=0.15, seed=0, d=16)
    c0, c1 = bundle.chunks[0], bundle.chunks[1]
    text = chat.complete("gen", f"Write a question grounded in {c0.text} and {c1.text}.")
    emission = bundle.bank.get(text)
    base = l2_normalize(bundle.bank.get(c0.text) + bundle.bank.get(c1.text))
    assert float(emission @ base) > 0.9


def test_toy_chat_base_exemplars_prefers_exemplar_tokens() -> None:
    bank = EmbeddingBank()
    chunk_vec = np.zeros(16)
    chunk_vec[0] = 1.0
    exemplar_vec = np.zeros(16)
    exemplar_vec[1] = 1.0
    bank.put("toychunk:0000", chunk_vec)
    bank.put("toyquery:train:00000", exemplar_vec)
    chat = ToyChatModel(bank, style=0.05, noise=0.15, base="exemplars", seed=1, d=16)
    text = chat.complete("gen", "style like toyquery:train:00000 given toychunk:0000")
    emission = bank.get(text)
    assert float(emission @ exemplar_vec) > float(emission @ chunk_vec)
    assert float(emission @ exemplar_vec) > 0.9


def test_toy_chat_base_exemplars_falls_back_to_chunks() -> None:
    bank = EmbeddingBank()
    chunk_vec = np.zeros(16)
    chunk_vec[0] = 1.0
    bank.put("toychunk:0000", chunk_vec)
    chat = ToyChatModel(bank, style=0.05, noise=0.15, base="exemplars", seed=2, d=16)
    text = chat.complete("gen", "no exemplars, only toychunk:0000")
    assert float(bank.get(text) @ chunk_vec) > 0.9


def test_toy_chat_revision_tightens_toward_base(bundle) -> None:
    chat = ToyChatModel(bundle.bank, style=0.15, noise=0.68, seed=0, d=16)
    c0 = bundle.chunks[0]
    base = bundle.bank.get(c0.text)
    first, revised = [], []
    for i in range(20):
        e1 = bundle.bank.get(chat.complete("gen", f"prompt {i}: {c0.text}"))
        e2 = bundle.bank.get(chat.complete("gen", f"prompt {i}: {c0.text} REVISE_REQUEST"))
        first.append(float(e1 @ base))
        revised.append(float(e2 @ base))
    assert float(np.mean(revised)) > float(np.mean(first))


def test_toy_chat_ignores_unbanked_tokens(bundle) -> None:
    chat = ToyChatModel(bundle.bank, style=0.05, noise=0.15, seed=0, d=16)
    c0 = bundle.chunks[0]
    # toychunk:9999 matches the regex but was never banked -> ignored
    text = chat.complete("gen", f"{c0.text} plus phantom toychunk:9999")
    assert float(bundle.bank.get(text) @ bundle.bank.get(c0.text)) > 0.9


def test_toy_chat_unparseable_prompt_raises(bundle) -> None:
    chat = ToyChatModel(bundle.bank, seed=0, d=16)
    with pytest.raises(ValueError, match="token"):
        chat.complete("sys", "a prompt with no toy tokens at all")
    with pytest.raises(ValueError, match="token"):
        chat.complete("sys", "only a phantom toychunk:9999 token")


def test_toy_chat_to_config_and_from_config(bundle) -> None:
    chat = ToyChatModel(bundle.bank, style=0.05, noise=0.15, base="exemplars", seed=3, d=16)
    assert chat.to_config() == {
        "style": 0.05,
        "noise": 0.15,
        "base": "exemplars",
        "seed": 3,
        "d": 16,
    }
    built = ToyChatModel.from_config({"style": 0.05}, bundle, np.random.default_rng(0))
    assert built.bank is bundle.bank
    assert built.style == 0.05
    assert built.d == 16  # inferred from the bundle's embedding store


def test_toy_chat_from_config_requires_bank(bundle) -> None:
    no_bank = dataclasses.replace(bundle, bank=None)
    with pytest.raises(ValueError, match="bank"):
        ToyChatModel.from_config({}, no_bank, np.random.default_rng(0))


# ---------------------------------------------------------------------------
# ToyJudge
# ---------------------------------------------------------------------------


def test_toy_judge_no_evidence_mostly_unanswerable(bundle) -> None:
    judge = ToyJudge(bundle.bank)
    verdicts = [judge.judge(q.text, []) for q in bundle.queries_anchor]
    rate = float(np.mean([v.answerable for v in verdicts]))
    assert rate < 0.10
    for v in verdicts:
        assert v.answer == ("toy answer" if v.answerable else "")


def test_toy_judge_answerable_with_own_nearest_chunk(bundle) -> None:
    judge = ToyJudge(bundle.bank)
    matrix = _chunk_matrix(bundle)
    for q in bundle.queries_anchor[:25]:
        qv = bundle.embeddings.get([q.query_id])[0].astype(np.float64)
        nearest = bundle.chunks[int(np.argmax(matrix @ qv))]
        verdict = judge.judge(q.text, [nearest.text])
        assert verdict.answerable
        assert verdict.answer == "toy answer"
        assert 0.5 <= verdict.confidence <= 1.0


def test_toy_judge_unknown_query_warns(bundle, caplog) -> None:
    judge = ToyJudge(bundle.bank)
    with caplog.at_level(logging.WARNING, logger="ragsynth.datasets.toy_world"):
        verdict = judge.judge("never banked query", ["whatever evidence"])
    assert verdict.answerable is False
    assert verdict.confidence == 0.0
    assert verdict.answer == ""
    assert any(r.levelno == logging.WARNING for r in caplog.records)


def test_toy_judge_to_config_and_from_config(bundle) -> None:
    judge = ToyJudge(bundle.bank, tau_ans=0.4, common_knowledge_pct=5)
    assert judge.to_config() == {"tau_ans": 0.4, "common_knowledge_pct": 5}
    built = ToyJudge.from_config({"tau_ans": 0.4}, bundle, np.random.default_rng(0))
    assert built.bank is bundle.bank
    assert built.tau_ans == 0.4
    no_bank = dataclasses.replace(bundle, bank=None)
    with pytest.raises(ValueError, match="bank"):
        ToyJudge.from_config({}, no_bank, np.random.default_rng(0))


# ---------------------------------------------------------------------------
# PassthroughEmbedder
# ---------------------------------------------------------------------------


def test_passthrough_embedder_lookup(bundle) -> None:
    emb = PassthroughEmbedder(bundle.bank)
    texts = [bundle.chunks[0].text, bundle.queries_train[0].text]
    out = emb.encode(texts)
    assert out.shape == (2, 16)
    np.testing.assert_array_equal(out[0], bundle.bank.get(texts[0]))
    np.testing.assert_array_equal(out[1], bundle.bank.get(texts[1]))


def test_passthrough_embedder_unknown_text_raises(bundle) -> None:
    emb = PassthroughEmbedder(bundle.bank)
    with pytest.raises(KeyError):
        emb.encode([bundle.chunks[0].text, "never banked text"])


def test_passthrough_embedder_from_config(bundle) -> None:
    built = PassthroughEmbedder.from_config({}, bundle, np.random.default_rng(0))
    assert built.bank is bundle.bank
    assert built.to_config() == {}
    no_bank = dataclasses.replace(bundle, bank=None)
    with pytest.raises(ValueError, match="bank"):
        PassthroughEmbedder.from_config({}, no_bank, np.random.default_rng(0))
