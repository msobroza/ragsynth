"""FiQA converter tests against the hand-made mini-fixture (spec01 §13.4).

Fixture layout: ``tests/fixtures/benchmarks/fiqa/raw/`` -- 5 docs, 4 queries,
qrels split across ``train.tsv``/``test.tsv`` with a deliberate cross-split
score conflict on (q2, d3) that only a true max-merge resolves correctly
(see ``test_base.py``'s dedup test for the isolated version of this check).
"""

import hashlib
import json
from pathlib import Path

from ragsynth.datasets.converters import fiqa
from ragsynth.datasets.converters.base import LICENSE_NOTES
from ragsynth.datasets.jsonl_loader import load_anchor_qrels, load_chunks, load_queries
from ragsynth.domain import Chunk

FIXTURE_RAW = Path(__file__).resolve().parents[2] / "fixtures" / "benchmarks" / "fiqa" / "raw"

_EXPECTED_CHUNKS_JSONL = (
    '{"doc_id": "d3", "metadata": {"upstream_id": "d3"}, '
    '"text": "Compound interest grows savings faster than simple interest."}\n'
    '{"doc_id": "d1", "metadata": {"upstream_id": "d1"}, '
    '"text": "Diversification Basics\\n\\nSpreading investments across assets '
    'reduces portfolio risk."}\n'
    '{"doc_id": "d8", "metadata": {"upstream_id": "d8"}, '
    '"text": "Index funds track a market benchmark at low cost."}\n'
    '{"doc_id": "d2", "metadata": {"upstream_id": "d2"}, '
    '"text": "Emergency Fund\\n\\nKeep three to six months of expenses in a liquid account."}\n'
    '{"doc_id": "d5", "metadata": {"upstream_id": "d5"}, '
    '"text": "Rebalancing restores a portfolio\'s target asset allocation."}\n'
)

_EXPECTED_QUERIES_JSONL = (
    '{"query_id": "q2", "text": "How does compounding interest work for savings accounts?"}\n'
    '{"query_id": "q1", "text": "What is portfolio diversification?"}\n'
    '{"query_id": "q4", "text": "Why keep an emergency fund?"}\n'
    '{"query_id": "q3", "text": "What are index funds?"}\n'
)

_EXPECTED_ANCHOR_QRELS_JSONL = (
    '{"qrels": {"b4d2f992fda8fb28": 1}, "query_id": "q2"}\n'
    '{"qrels": {"4372aafc6c7883b5": 1}, "query_id": "q1"}\n'
    '{"qrels": {"8ee7005152703650": 1}, "query_id": "q4"}\n'
    '{"qrels": {"a05b0168879d8bf2": 1}, "query_id": "q3"}\n'
)


def test_name_attribute() -> None:
    assert fiqa.name == "fiqa"


def test_convert_emits_exact_chunks_jsonl_bytes(tmp_path: Path) -> None:
    fiqa.convert(FIXTURE_RAW, tmp_path)
    assert (tmp_path / "chunks.jsonl").read_text(encoding="utf-8") == _EXPECTED_CHUNKS_JSONL


def test_convert_emits_exact_queries_jsonl_bytes(tmp_path: Path) -> None:
    fiqa.convert(FIXTURE_RAW, tmp_path)
    assert (tmp_path / "queries.jsonl").read_text(encoding="utf-8") == _EXPECTED_QUERIES_JSONL


def test_convert_emits_exact_anchor_qrels_jsonl_bytes(tmp_path: Path) -> None:
    fiqa.convert(FIXTURE_RAW, tmp_path)
    assert (tmp_path / "anchor_qrels.jsonl").read_text(
        encoding="utf-8"
    ) == _EXPECTED_ANCHOR_QRELS_JSONL


def test_convert_manifest_counts_and_provenance(tmp_path: Path) -> None:
    manifest = fiqa.convert(FIXTURE_RAW, tmp_path)

    assert manifest.dataset == "fiqa"
    assert manifest.n_chunks == 5
    assert manifest.n_queries == 4
    assert manifest.n_qrel_entries == 4
    assert manifest.license_note == LICENSE_NOTES["fiqa"]
    assert set(manifest.output_sha256) == {"chunks.jsonl", "queries.jsonl", "anchor_qrels.jsonl"}
    for filename, digest in manifest.output_sha256.items():
        assert digest == hashlib.sha256((tmp_path / filename).read_bytes()).hexdigest()

    manifest_on_disk = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest_on_disk["n_chunks"] == 5


def test_convert_twice_is_byte_identical(tmp_path: Path) -> None:
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    fiqa.convert(FIXTURE_RAW, out_a)
    fiqa.convert(FIXTURE_RAW, out_b)

    for filename in ("chunks.jsonl", "queries.jsonl", "anchor_qrels.jsonl", "manifest.json"):
        assert (out_a / filename).read_bytes() == (out_b / filename).read_bytes()


def test_qrels_are_rekeyed_to_the_content_hash_chunk_id_the_loader_will_produce(
    tmp_path: Path,
) -> None:
    """spec01 §5: converters build the SAME Chunk the loader builds to re-key qrels.

    Independently construct the fixture's "d1" chunk exactly as the
    converter/loader do and assert its content-hash ``chunk_id`` is the key
    used for query ``q1`` in the emitted anchor_qrels.jsonl.
    """
    fiqa.convert(FIXTURE_RAW, tmp_path)

    expected_text = (
        "Diversification Basics\n\nSpreading investments across assets reduces portfolio risk."
    )
    expected_chunk = Chunk.create(text=expected_text, doc_id="d1", metadata={"upstream_id": "d1"})

    qrels = load_anchor_qrels(tmp_path / "anchor_qrels.jsonl")
    assert expected_chunk.chunk_id in qrels["q1"]
    assert qrels["q1"][expected_chunk.chunk_id] == 1


def test_convert_raw_dir_without_qrels_fails_loudly(tmp_path: Path) -> None:
    """A raw dir missing qrels/ must error, never emit an empty anchor_qrels.jsonl."""
    import shutil

    import pytest

    broken_raw = tmp_path / "raw"
    broken_raw.mkdir()
    shutil.copy(FIXTURE_RAW / "corpus.jsonl", broken_raw / "corpus.jsonl")
    shutil.copy(FIXTURE_RAW / "queries.jsonl", broken_raw / "queries.jsonl")

    with pytest.raises(FileNotFoundError, match="qrels"):
        fiqa.convert(broken_raw, tmp_path / "out")


def test_loader_round_trip_every_qrel_chunk_id_resolves_in_loaded_chunks(tmp_path: Path) -> None:
    """End-to-end proof: run the REAL jsonl_loader over the converter's output."""
    fiqa.convert(FIXTURE_RAW, tmp_path)

    chunks = load_chunks(tmp_path / "chunks.jsonl")
    queries = load_queries(tmp_path / "queries.jsonl")
    qrels = load_anchor_qrels(tmp_path / "anchor_qrels.jsonl")

    chunk_ids = {c.chunk_id for c in chunks}
    query_ids = {q.query_id for q in queries}

    assert len(chunks) == 5
    assert len(queries) == 4
    for query_id, gold in qrels.items():
        assert query_id in query_ids
        for chunk_id in gold:
            assert chunk_id in chunk_ids
