"""Tests for the bundled deterministic sample corpus (PLAN D18)."""

import re
from pathlib import Path

from ragsynth.datasets.jsonl_loader import load_chunks, load_queries
from ragsynth.datasets.sample_corpus import generate_sample_corpus

REPO_ROOT = Path(__file__).resolve().parents[2]
COMMITTED = REPO_ROOT / "data" / "sample"


def _lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def test_generates_exact_counts(tmp_path) -> None:
    generate_sample_corpus(tmp_path)
    assert len(_lines(tmp_path / "chunks.jsonl")) == 200
    assert len(_lines(tmp_path / "queries.jsonl")) == 120


def test_two_runs_byte_identical(tmp_path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    generate_sample_corpus(a)
    generate_sample_corpus(b)
    for name in ("chunks.jsonl", "queries.jsonl"):
        assert (a / name).read_bytes() == (b / name).read_bytes()


def test_regeneration_matches_committed_files(tmp_path) -> None:
    # PLAN D18: the committed data/sample corpus is regenerable byte-for-byte.
    generate_sample_corpus(tmp_path)
    for name in ("chunks.jsonl", "queries.jsonl"):
        assert (tmp_path / name).read_bytes() == (COMMITTED / name).read_bytes(), (
            f"{name} drifted from `python -m ragsynth.datasets.sample_corpus` output"
        )


def test_loads_via_jsonl_loader(tmp_path) -> None:
    generate_sample_corpus(tmp_path)
    chunks = load_chunks(tmp_path / "chunks.jsonl")
    queries = load_queries(tmp_path / "queries.jsonl")
    assert len(chunks) == 200
    assert len(queries) == 120
    assert all(c.text and c.doc_id for c in chunks)
    assert all(q.query_id and q.text for q in queries)
    assert queries[0].query_id == "q00000"


def test_chunk_texts_nontrivial(tmp_path) -> None:
    generate_sample_corpus(tmp_path)
    chunks = load_chunks(tmp_path / "chunks.jsonl")
    avg_len = sum(len(c.text) for c in chunks) / len(chunks)
    assert avg_len > 100
    words_by_topic: dict[str, set[str]] = {}
    for c in chunks:
        topic = c.metadata["topic"]
        words_by_topic.setdefault(topic, set()).update(re.findall(r"[a-z']+", c.text.lower()))
    assert len(words_by_topic) == 8  # eight topical clusters
    for topic, words in words_by_topic.items():
        assert len(words) > 30, f"topic {topic} vocabulary too small: {len(words)}"


def test_queries_mixed_registers_and_skew(tmp_path) -> None:
    generate_sample_corpus(tmp_path)
    queries = load_queries(tmp_path / "queries.jsonl")
    texts = [q.text for q in queries]
    questions = [t for t in texts if t.endswith("?")]
    non_questions = [t for t in texts if not t.endswith("?")]
    assert questions  # mixed registers: some natural-language questions...
    assert non_questions  # ...and some keyword-ish / how-to fragments
    assert len(set(texts)) > 60  # not degenerate
