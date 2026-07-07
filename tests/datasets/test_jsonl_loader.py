"""Tests for the JSONL corpus loader and the jsonl dataset builder."""

import json
from datetime import datetime

import pytest

from ragsynth.datasets.base import DATASETS
from ragsynth.datasets.jsonl_loader import (
    JsonlDataset,
    load_anchor_qrels,
    load_chunks,
    load_queries,
)


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
