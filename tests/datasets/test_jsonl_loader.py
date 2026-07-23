"""Tests for the JSONL corpus loader and the jsonl dataset builder."""

import hashlib
import json
from datetime import datetime
from pathlib import Path

import pytest

from ragsynth.datasets.base import DATASETS
from ragsynth.datasets.jsonl_loader import (
    JsonlDataset,
    load_anchor_qrels,
    load_chunks,
    load_queries,
    load_query_metadata,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_jsonl(path, rows) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


@pytest.fixture
def corpus_paths(tmp_path):
    chunks = [
        {"text": "Invoices are generated nightly.", "doc_id": "doc-billing"},
        {
            "text": "Webhook deliveries retry with backoff.",
            "doc_id": "doc-webhooks",
            "page": 3,
            "metadata": {"topic": "webhooks"},
        },
    ] + [{"text": f"Filler chunk number {i}.", "doc_id": f"doc-{i}"} for i in range(8)]
    queries = [
        {"query_id": "custom-id", "text": "How are invoices generated?"},
        {"text": "webhook retry policy"},
        {
            "query_id": "q-ts",
            "text": "When do exports run?",
            "timestamp": "2026-01-02T03:04:05",
        },
    ] + [{"query_id": f"qq{i}", "text": f"question number {i}?"} for i in range(17)]
    qrels = [
        {"query_id": q["query_id"], "qrels": {"chunk-a": 1, "chunk-b": 1}}
        for q in queries
        if "query_id" in q
    ]
    chunks_path = tmp_path / "chunks.jsonl"
    queries_path = tmp_path / "queries.jsonl"
    qrels_path = tmp_path / "anchor_qrels.jsonl"
    _write_jsonl(chunks_path, chunks)
    _write_jsonl(queries_path, queries)
    _write_jsonl(qrels_path, qrels)
    return chunks_path, queries_path, qrels_path


def test_load_chunks_round_trip(corpus_paths) -> None:
    chunks_path, _, _ = corpus_paths
    chunks = load_chunks(chunks_path)
    assert len(chunks) == 10
    assert chunks[0].text == "Invoices are generated nightly."
    assert chunks[0].doc_id == "doc-billing"
    assert chunks[0].page is None
    assert chunks[0].metadata == {}
    assert chunks[0].chunk_id
    assert chunks[0].content_hash
    assert chunks[1].page == 3
    assert chunks[1].metadata == {"topic": "webhooks"}


def test_load_queries_round_trip(corpus_paths) -> None:
    _, queries_path, _ = corpus_paths
    queries = load_queries(queries_path)
    assert len(queries) == 20
    assert queries[0].query_id == "custom-id"
    assert queries[1].query_id == "q00001"  # default id from line index
    assert queries[1].timestamp is None
    assert queries[2].timestamp == datetime.fromisoformat("2026-01-02T03:04:05")
    assert queries[2].text == "When do exports run?"


def test_load_anchor_qrels_round_trip(corpus_paths) -> None:
    _, _, qrels_path = corpus_paths
    qrels = load_anchor_qrels(qrels_path)
    assert qrels["custom-id"] == {"chunk-a": 1, "chunk-b": 1}
    assert all(isinstance(g, int) for grades in qrels.values() for g in grades.values())


def test_missing_files_raise_actionable_error(tmp_path) -> None:
    missing = tmp_path / "nope.jsonl"
    for loader in (load_chunks, load_queries, load_anchor_qrels):
        with pytest.raises(FileNotFoundError, match=r"nope\.jsonl"):
            loader(missing)


def test_jsonl_dataset_registered() -> None:
    assert DATASETS.get("jsonl") is JsonlDataset


def test_jsonl_dataset_build_split_and_qrels(corpus_paths) -> None:
    chunks_path, queries_path, qrels_path = corpus_paths
    params = {
        "chunks_path": str(chunks_path),
        "queries_path": str(queries_path),
        "anchor_qrels_path": str(qrels_path),
    }
    bundle = JsonlDataset.build(params, seed=0)
    # 20 queries at 0.60/0.25/0.15 -> 12/5/3
    assert len(bundle.queries_train) == 12
    assert len(bundle.queries_anchor) == 5
    assert len(bundle.queries_oracle) == 3
    all_ids = {
        q.query_id
        for split in ("train", "anchor", "oracle")
        for q in getattr(bundle, f"queries_{split}")
    }
    assert len(all_ids) == 20  # disjoint partition covering every query
    # anchor_qrels filtered to queries that landed in the anchor split
    anchor_ids = {q.query_id for q in bundle.queries_anchor}
    assert set(bundle.anchor_qrels) <= anchor_ids
    assert bundle.oracle_qrels == {}
    # text corpora embed later via a real featurizer
    assert bundle.embeddings is None
    assert bundle.bank is None
    assert len(bundle.chunks) == 10


def test_jsonl_dataset_build_deterministic(corpus_paths) -> None:
    chunks_path, queries_path, _ = corpus_paths
    params = {"chunks_path": str(chunks_path), "queries_path": str(queries_path)}
    a = JsonlDataset.build(params, seed=0)
    b = JsonlDataset.build(params, seed=0)
    for split in ("train", "anchor", "oracle"):
        ids_a = [q.query_id for q in getattr(a, f"queries_{split}")]
        ids_b = [q.query_id for q in getattr(b, f"queries_{split}")]
        assert ids_a == ids_b
    assert a.anchor_qrels == {}


def test_jsonl_dataset_custom_split(corpus_paths) -> None:
    chunks_path, queries_path, _ = corpus_paths
    params = {
        "chunks_path": str(chunks_path),
        "queries_path": str(queries_path),
        "split": (0.5, 0.3, 0.2),
    }
    bundle = JsonlDataset.build(params, seed=1)
    assert len(bundle.queries_train) == 10
    assert len(bundle.queries_anchor) == 6
    assert len(bundle.queries_oracle) == 4


def test_jsonl_dataset_bad_split_raises(corpus_paths) -> None:
    chunks_path, queries_path, _ = corpus_paths
    params = {
        "chunks_path": str(chunks_path),
        "queries_path": str(queries_path),
        "split": (0.9, 0.9, 0.9),
    }
    with pytest.raises(ValueError, match="split"):
        JsonlDataset.build(params, seed=0)


# --- query metadata + split_stratify_by (schema-2 feature, spec01 §6, D36) ----------


def _split_hash(split) -> str:
    return hashlib.sha256("|".join(q.query_id for q in split).encode()).hexdigest()[:16]


def test_load_query_metadata_parses_and_defaults(tmp_path) -> None:
    queries = [
        {"query_id": "a", "text": "one", "metadata": {"subcorpus": "cuad", "n": 3}},
        {"query_id": "b", "text": "two"},
    ]
    path = tmp_path / "queries.jsonl"
    _write_jsonl(path, queries)
    meta = load_query_metadata(path)
    # values coerced to str (like load_chunks); absent metadata -> {}
    assert meta == [{"subcorpus": "cuad", "n": "3"}, {}]
    # load_queries stays metadata-free (byte-behavior unchanged)
    assert [q.query_id for q in load_queries(path)] == ["a", "b"]


@pytest.fixture
def stratified_paths(tmp_path):
    # 12 group-"alpha" queries + 8 group-"beta" queries, interleaved in file order.
    rows = []
    for i in range(20):
        group = "alpha" if i % 5 != 0 else "beta"  # 16 alpha / 4 beta -> adjust below
        rows.append({"query_id": f"q{i:02d}", "text": f"q {i}", "metadata": {"subcorpus": group}})
    # Force exactly 12 alpha / 8 beta for clean 60/25/15 arithmetic.
    for i in range(20):
        rows[i]["metadata"]["subcorpus"] = "alpha" if i < 12 else "beta"
    chunks_path = tmp_path / "chunks.jsonl"
    queries_path = tmp_path / "queries.jsonl"
    _write_jsonl(chunks_path, [{"text": "c", "doc_id": "d"}])
    _write_jsonl(queries_path, rows)
    return chunks_path, queries_path


def test_split_stratify_by_splits_per_group(stratified_paths) -> None:
    chunks_path, queries_path = stratified_paths
    params = {
        "chunks_path": str(chunks_path),
        "queries_path": str(queries_path),
        "split_stratify_by": "subcorpus",
    }
    bundle = JsonlDataset.build(params, seed=0)
    meta = {r["query_id"]: r["subcorpus"] for r in _read_meta(queries_path)}
    # alpha 12 -> 7/3/2, beta 8 -> 4/2/2  =>  train 11, anchor 5, oracle 4
    assert len(bundle.queries_train) == 11
    assert len(bundle.queries_anchor) == 5
    assert len(bundle.queries_oracle) == 4
    # disjoint partition covering all 20
    all_ids = [
        q.query_id for s in ("train", "anchor", "oracle") for q in getattr(bundle, f"queries_{s}")
    ]
    assert sorted(all_ids) == sorted(f"q{i:02d}" for i in range(20))
    assert len(set(all_ids)) == 20
    # concatenation is in group-sorted order: all alpha then all beta within train
    train_groups = [meta[q.query_id] for q in bundle.queries_train]
    assert train_groups == ["alpha"] * 7 + ["beta"] * 4


def test_split_stratify_by_deterministic(stratified_paths) -> None:
    chunks_path, queries_path = stratified_paths
    params = {
        "chunks_path": str(chunks_path),
        "queries_path": str(queries_path),
        "split_stratify_by": "subcorpus",
    }
    a = JsonlDataset.build(params, seed=0)
    b = JsonlDataset.build(params, seed=0)
    for split in ("train", "anchor", "oracle"):
        assert [q.query_id for q in getattr(a, f"queries_{split}")] == [
            q.query_id for q in getattr(b, f"queries_{split}")
        ]


def test_split_stratify_by_missing_key_raises(tmp_path) -> None:
    rows = [{"query_id": "a", "text": "x", "metadata": {"other": "z"}}]
    chunks_path = tmp_path / "chunks.jsonl"
    queries_path = tmp_path / "queries.jsonl"
    _write_jsonl(chunks_path, [{"text": "c", "doc_id": "d"}])
    _write_jsonl(queries_path, rows)
    params = {
        "chunks_path": str(chunks_path),
        "queries_path": str(queries_path),
        "split_stratify_by": "subcorpus",
    }
    with pytest.raises(ValueError, match="subcorpus"):
        JsonlDataset.build(params, seed=0)


def test_sample_corpus_split_byte_identical_regression() -> None:
    """Absent split_stratify_by ⇒ exact v1 split for the bundled sample corpus.

    These hashes are the pre-schema-2 golden; the stratification refactor must
    not perturb the default (schema-1) split by a single query.
    """
    bundle = JsonlDataset.build(
        {
            "chunks_path": str(REPO_ROOT / "data" / "sample" / "chunks.jsonl"),
            "queries_path": str(REPO_ROOT / "data" / "sample" / "queries.jsonl"),
        },
        seed=0,
    )
    assert (len(bundle.queries_train), len(bundle.queries_anchor), len(bundle.queries_oracle)) == (
        72,
        30,
        18,
    )
    assert _split_hash(bundle.queries_train) == "02da5774298c0b89"
    assert _split_hash(bundle.queries_anchor) == "4de8a8e93960555d"
    assert _split_hash(bundle.queries_oracle) == "734fb4e1b61c0b16"


def _read_meta(queries_path):
    rows = (json.loads(line) for line in queries_path.read_text().splitlines() if line.strip())
    return [{"query_id": r["query_id"], "subcorpus": r["metadata"]["subcorpus"]} for r in rows]
