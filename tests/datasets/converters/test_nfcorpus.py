"""NFCorpus converter tests against the hand-made mini-fixture (spec01 §13.4).

Fixture layout: ``tests/fixtures/benchmarks/nfcorpus/raw/`` -- 5 docs, 4
queries, graded qrels (0/1/2) split across ``train.tsv``/``test.tsv``:
- (PLAIN-1, MED-1) and (PLAIN-2, MED-3): present in both splits at
  different grades -- dedup-max, then binarize (spec01 §5, D33/§15.2).
- (PLAIN-4, MED-8): train.tsv says grade 0, test.tsv says grade 2 -- proves
  the merge takes the true max, not "last file wins" (files are visited
  ``test.tsv`` before ``train.tsv`` in sorted-filename order).
- (PLAIN-3, MED-2): grade 0 only -- must be binarized away entirely, so
  PLAIN-3 has no ``anchor_qrels.jsonl`` row even though it is still a valid
  query in ``queries.jsonl``.
"""

import hashlib
import json
from pathlib import Path

from ragsynth.datasets.converters import nfcorpus
from ragsynth.datasets.converters.base import LICENSE_NOTES
from ragsynth.datasets.jsonl_loader import load_anchor_qrels, load_chunks, load_queries

FIXTURE_RAW = Path(__file__).resolve().parents[2] / "fixtures" / "benchmarks" / "nfcorpus" / "raw"

_EXPECTED_CHUNKS_JSONL = (
    '{"doc_id": "MED-3", "metadata": {"upstream_id": "MED-3"}, '
    '"text": "Antioxidants and Cancer Risk\\n\\nDietary antioxidants may reduce '
    'oxidative stress linked to cancer risk."}\n'
    '{"doc_id": "MED-1", "metadata": {"upstream_id": "MED-1"}, '
    '"text": "Vitamin D deficiency is associated with bone density loss."}\n'
    '{"doc_id": "MED-8", "metadata": {"upstream_id": "MED-8"}, '
    '"text": "Omega-3 Fatty Acids\\n\\nOmega-3 fatty acids from fish oil support '
    'cardiovascular health."}\n'
    '{"doc_id": "MED-2", "metadata": {"upstream_id": "MED-2"}, '
    '"text": "Fiber intake improves gut microbiome diversity."}\n'
    '{"doc_id": "MED-5", "metadata": {"upstream_id": "MED-5"}, '
    '"text": "Sodium Intake\\n\\nReducing sodium intake is linked to lower blood pressure."}\n'
)

_EXPECTED_QUERIES_JSONL = (
    '{"query_id": "PLAIN-2", "text": "Do antioxidants help prevent cancer?"}\n'
    '{"query_id": "PLAIN-1", "text": "What does vitamin D do for bones?"}\n'
    '{"query_id": "PLAIN-4", "text": "Are omega-3 supplements good for the heart?"}\n'
    '{"query_id": "PLAIN-3", "text": "How does fiber affect gut health?"}\n'
)

_EXPECTED_ANCHOR_QRELS_JSONL = (
    '{"qrels": {"794f07166a954030": 1}, "query_id": "PLAIN-2"}\n'
    '{"qrels": {"4ab93f57d0199e9b": 1}, "query_id": "PLAIN-1"}\n'
    '{"qrels": {"2c2fe5d3926d7744": 1}, "query_id": "PLAIN-4"}\n'
)


def test_name_attribute() -> None:
    assert nfcorpus.name == "nfcorpus"


def test_convert_emits_exact_chunks_jsonl_bytes(tmp_path: Path) -> None:
    nfcorpus.convert(FIXTURE_RAW, tmp_path)
    assert (tmp_path / "chunks.jsonl").read_text(encoding="utf-8") == _EXPECTED_CHUNKS_JSONL


def test_convert_emits_exact_queries_jsonl_bytes(tmp_path: Path) -> None:
    nfcorpus.convert(FIXTURE_RAW, tmp_path)
    assert (tmp_path / "queries.jsonl").read_text(encoding="utf-8") == _EXPECTED_QUERIES_JSONL


def test_convert_emits_exact_anchor_qrels_jsonl_bytes_grade_0_query_dropped(
    tmp_path: Path,
) -> None:
    """PLAIN-3's only judgment is grade 0, so it gets no anchor_qrels row at all."""
    nfcorpus.convert(FIXTURE_RAW, tmp_path)
    text = (tmp_path / "anchor_qrels.jsonl").read_text(encoding="utf-8")
    assert text == _EXPECTED_ANCHOR_QRELS_JSONL
    assert "PLAIN-3" not in text


def test_convert_manifest_counts_reflect_binarization_drop(tmp_path: Path) -> None:
    manifest = nfcorpus.convert(FIXTURE_RAW, tmp_path)

    assert manifest.dataset == "nfcorpus"
    assert manifest.n_chunks == 5
    assert manifest.n_queries == 4
    # 4 candidate (query, doc) pairs after cross-split dedup; PLAIN-3's sole
    # judgment binarizes away, leaving 3 emitted qrel entries.
    assert manifest.n_qrel_entries == 3
    assert manifest.license_note == LICENSE_NOTES["nfcorpus"]
    for filename, digest in manifest.output_sha256.items():
        assert digest == hashlib.sha256((tmp_path / filename).read_bytes()).hexdigest()


def test_convert_twice_is_byte_identical(tmp_path: Path) -> None:
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    nfcorpus.convert(FIXTURE_RAW, out_a)
    nfcorpus.convert(FIXTURE_RAW, out_b)

    for filename in ("chunks.jsonl", "queries.jsonl", "anchor_qrels.jsonl", "manifest.json"):
        assert (out_a / filename).read_bytes() == (out_b / filename).read_bytes()


def test_max_dedup_across_splits_keeps_the_higher_grade_not_the_last_file_read(
    tmp_path: Path,
) -> None:
    """(PLAIN-4, MED-8): train.tsv=0, test.tsv=2 -- must survive at grade 1."""
    nfcorpus.convert(FIXTURE_RAW, tmp_path)
    qrels = load_anchor_qrels(tmp_path / "anchor_qrels.jsonl")
    lines = (tmp_path / "anchor_qrels.jsonl").read_text(encoding="utf-8").splitlines()
    third_row = json.loads(lines[2])
    assert third_row["query_id"] == "PLAIN-4"
    assert list(qrels["PLAIN-4"].values()) == [1]


def test_loader_round_trip_every_qrel_chunk_id_resolves_in_loaded_chunks(tmp_path: Path) -> None:
    """End-to-end proof: run the REAL jsonl_loader over the converter's output."""
    nfcorpus.convert(FIXTURE_RAW, tmp_path)

    chunks = load_chunks(tmp_path / "chunks.jsonl")
    queries = load_queries(tmp_path / "queries.jsonl")
    qrels = load_anchor_qrels(tmp_path / "anchor_qrels.jsonl")

    chunk_ids = {c.chunk_id for c in chunks}
    query_ids = {q.query_id for q in queries}

    assert len(chunks) == 5
    assert len(queries) == 4
    assert "PLAIN-3" not in qrels
    for query_id, gold in qrels.items():
        assert query_id in query_ids
        for chunk_id in gold:
            assert chunk_id in chunk_ids
