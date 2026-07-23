"""LegalBench-RAG converter tests: D34 splitter, D35 spans_to_qrels, D36 merge.

The §13.4 known-value fixtures are written FIRST and every number is
hand-derived in a comment (spec01 §13.4, SPEC §15.4 style). The mini-fixture
under ``tests/fixtures/benchmarks/legalbench_rag/raw/`` is 2 sub-corpora
(``contractnli`` < ``cuad``) x 2 small docs; ``contractnli/doc_a.txt`` is a
1,300-char document with predictable whitespace (a single space at every
global index ``== 5 (mod 10)``) so the split offsets are hand-checkable, and
its query span crosses a chunk boundary to exercise the 200-char absolute
branch of D35.
"""

import hashlib
import json
import logging
import shutil
from pathlib import Path

import pytest

from ragsynth.datasets.converters import legalbench_rag
from ragsynth.datasets.converters.base import LICENSE_NOTES
from ragsynth.datasets.converters.legalbench_rag import spans_to_qrels, split_document
from ragsynth.datasets.jsonl_loader import load_anchor_qrels, load_chunks, load_queries
from ragsynth.domain import Chunk

FIXTURE_RAW = (
    Path(__file__).resolve().parents[2] / "fixtures" / "benchmarks" / "legalbench_rag" / "raw"
)

# The exact bytes of the four raw corpus docs (must match the on-disk fixture).
DOC_A = "".join(f"{k:05d} ABCD" for k in range(130))  # 1,300 chars, space at idx == 5 (mod 10)
DOC_B = "Confidential Information means all data disclosed by a party."
DOC_C = "This Agreement is entered into by Acme Corp and Beta LLC."
DOC_D = "This Agreement is governed by the laws of the State of Delaware."


# ---------------------------------------------------------------------------
# D34 splitter — §13.4 crafted 2,300-char doc, exact offsets + snap + overlap.
# ---------------------------------------------------------------------------
def test_fixture_docs_match_on_disk() -> None:
    """Guard: the hand-coded doc constants equal the committed fixture bytes."""
    assert (FIXTURE_RAW / "corpus" / "contractnli" / "doc_a.txt").read_text("utf-8") == DOC_A
    assert (FIXTURE_RAW / "corpus" / "contractnli" / "doc_b.txt").read_text("utf-8") == DOC_B
    assert (FIXTURE_RAW / "corpus" / "cuad" / "doc_c.txt").read_text("utf-8") == DOC_C
    assert (FIXTURE_RAW / "corpus" / "cuad" / "doc_d.txt").read_text("utf-8") == DOC_D


def test_split_document_2300_char_doc_exact_offsets_snap_and_overlap() -> None:
    """§13.4: 2,300-char doc, block ``"ABCDE FGHI"`` (space at local idx 5).

    Whitespace sits at every global index ``== 5 (mod 10)`` (5, 15, ..., 2295).
    Windows are 1,000 wide, 200 overlap, end snapped back to the last space:

    * W0: start=0, raw_end=1000 (<2300, snap). Last space <= 999 with idx==5
      (mod 10) is 995 -> end=995. Next start = 995 - 200 = 795.
    * W1: start=795, raw_end=1795 (<2300, snap). Last space <= 1794 with
      idx==5 (mod 10) is 1785 -> end=1785. Next start = 1785 - 200 = 1585.
    * W2: start=1585, raw_end=2585 (>=2300) -> last window, end=2300 (no snap).
    """
    doc = "ABCDE FGHI" * 230
    assert len(doc) == 2300

    windows = split_document(doc)

    assert windows == [(0, 995), (795, 1785), (1585, 2300)]
    # snap really moved the boundary off the raw window edge, onto a space:
    assert doc[995] == " "
    assert doc[994] != " "
    assert doc[1785] == " "
    assert doc[1784] != " "
    # overlap region text is byte-for-byte duplicated between consecutive chunks:
    assert doc[0:995][795:] == doc[795:1785][:200]  # W0 tail == W1 head (200 chars)
    assert doc[795:1785][1585 - 795 :] == doc[1585:2300][:200]  # W1 tail == W2 head


def test_split_document_hard_cut_when_window_has_no_whitespace() -> None:
    """A 1,500-char run with no whitespace is cut hard at 1,000 (D34)."""
    doc = "x" * 1500

    windows = split_document(doc)

    # W0 raw_end=1000 (<1500), no space -> hard cut at 1000; next start=800.
    # W1 raw_end=1800 (>=1500) -> end=1500.
    assert windows == [(0, 1000), (800, 1500)]


def test_split_document_short_doc_is_single_window() -> None:
    assert split_document("short contract text") == [(0, len("short contract text"))]


def test_split_document_empty_doc_is_no_windows() -> None:
    assert split_document("") == []


# ---------------------------------------------------------------------------
# D35 spans_to_qrels — §13.4 hand-derived thresholds.
# ---------------------------------------------------------------------------
def test_spans_to_qrels_long_span_50pct_branch() -> None:
    """§13.4: span (100, 400) => |span|=300, threshold=min(0.5*300, 200)=150.

    * chunk (0, 250):   overlap = 250-100 = 150 >= 150 -> gold
    * chunk (200, 450): overlap = 400-200 = 200 >= 150 -> gold
    * chunk (380, 600): overlap = 400-380 =  20 <  150 -> not gold
    """
    spans = {"qA": [("doc", 100, 400)]}
    chunk_offsets = {
        "c0": ("doc", 0, 250),
        "c1": ("doc", 200, 450),
        "c2": ("doc", 380, 600),
    }

    assert spans_to_qrels(spans, chunk_offsets) == {"qA": {"c0": 1, "c1": 1}}


def test_spans_to_qrels_short_span_fully_inside_one_chunk() -> None:
    """§13.4: span (100, 130) => |span|=30, threshold=min(15, 200)=15.

    Fully inside c0 (overlap 30 >= 15 -> gold); c1 does not overlap it at all.
    """
    spans = {"qB": [("doc", 100, 130)]}
    chunk_offsets = {"c0": ("doc", 0, 250), "c1": ("doc", 200, 450)}

    assert spans_to_qrels(spans, chunk_offsets) == {"qB": {"c0": 1}}


def test_spans_to_qrels_200_char_absolute_branch() -> None:
    """The 200-char absolute branch credits a half of a long split span.

    span (0, 1000) => |span|=1000, threshold=min(0.5*1000, 200)=200.

    * chunk (800, 1000): overlap = 1000-800 = 200 >= 200 -> gold
      (and 200 < 0.5*|span|=500, so ONLY the absolute branch makes it gold)
    * chunk (900, 1150): overlap = 1000-900 = 100 <  200 -> not gold
    """
    spans = {"q": [("doc", 0, 1000)]}
    chunk_offsets = {"lo": ("doc", 800, 1000), "hi": ("doc", 900, 1150)}

    assert spans_to_qrels(spans, chunk_offsets) == {"q": {"lo": 1}}


def test_spans_to_qrels_ignores_chunks_from_other_documents() -> None:
    """A numerically overlapping chunk in a different doc is never gold."""
    spans = {"q": [("docX", 100, 400)]}
    chunk_offsets = {"same": ("docX", 0, 250), "other": ("docY", 0, 250)}

    assert spans_to_qrels(spans, chunk_offsets) == {"q": {"same": 1}}


def test_spans_to_qrels_query_with_no_gold_is_omitted() -> None:
    spans = {"q": [("doc", 100, 130)]}  # threshold 15
    chunk_offsets = {"far": ("doc", 500, 800)}  # no overlap

    assert spans_to_qrels(spans, chunk_offsets) == {}


# ---------------------------------------------------------------------------
# D36 merge — exact jsonl bytes, manifest, subcorpus on chunks AND queries.
# ---------------------------------------------------------------------------
_META_A = {"subcorpus": "contractnli", "upstream_id": "contractnli/doc_a.txt"}
_META_B = {"subcorpus": "contractnli", "upstream_id": "contractnli/doc_b.txt"}
_META_C = {"subcorpus": "cuad", "upstream_id": "cuad/doc_c.txt"}
_META_D = {"subcorpus": "cuad", "upstream_id": "cuad/doc_d.txt"}

# Emission order: sub-corpora sorted (contractnli, cuad); docs sorted by path;
# doc_a splits into (0, 995) + (795, 1300) (see the 1,300-char derivation below).
_EXPECTED_CHUNK_ROWS = [
    {"text": DOC_A[0:995], "doc_id": "contractnli/doc_a.txt", "metadata": _META_A},
    {"text": DOC_A[795:1300], "doc_id": "contractnli/doc_a.txt", "metadata": _META_A},
    {"text": DOC_B, "doc_id": "contractnli/doc_b.txt", "metadata": _META_B},
    {"text": DOC_C, "doc_id": "cuad/doc_c.txt", "metadata": _META_C},
    {"text": DOC_D, "doc_id": "cuad/doc_d.txt", "metadata": _META_D},
]


def _dumps(rows: list[dict[str, object]]) -> str:
    return "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows)


def _chunk_id(text: str, doc_id: str, metadata: dict[str, str]) -> str:
    return Chunk.create(text=text, doc_id=doc_id, metadata=metadata).chunk_id


def test_convert_emits_exact_chunks_jsonl_bytes(tmp_path: Path) -> None:
    """doc_a 1,300 chars => (0,995) + (795,1300); other docs are single chunks.

    W0: raw_end=1000, last space (idx==5 mod 10) <= 999 is 995 -> end=995.
    W1: start=795, raw_end=1795 (>=1300) -> end=1300 (last window).
    """
    legalbench_rag.convert(FIXTURE_RAW, tmp_path)
    assert (tmp_path / "chunks.jsonl").read_text("utf-8") == _dumps(_EXPECTED_CHUNK_ROWS)


def test_convert_emits_exact_queries_jsonl_bytes_with_subcorpus(tmp_path: Path) -> None:
    legalbench_rag.convert(FIXTURE_RAW, tmp_path)
    expected = _dumps(
        [
            {
                "query_id": "contractnli_0000",
                "text": "What triggers confidentiality across the whole agreement?",
                "metadata": {"subcorpus": "contractnli"},
            },
            {
                "query_id": "contractnli_0001",
                "text": "What counts as confidential information?",
                "metadata": {"subcorpus": "contractnli"},
            },
            {
                "query_id": "cuad_0000",
                "text": "Who are the contracting parties?",
                "metadata": {"subcorpus": "cuad"},
            },
            {
                "query_id": "cuad_0001",
                "text": "What law governs the agreement?",
                "metadata": {"subcorpus": "cuad"},
            },
        ]
    )
    assert (tmp_path / "queries.jsonl").read_text("utf-8") == expected


def test_convert_emits_exact_anchor_qrels_jsonl_bytes(tmp_path: Path) -> None:
    """doc_a query span (750, 1300) crosses the chunk boundary -> BOTH chunks gold.

    chunk0 = (contractnli/doc_a.txt, 0, 995), chunk1 = (..., 795, 1300).
    |span|=550, threshold=min(0.5*550, 200)=min(275, 200)=200.
    * chunk0 overlap = 995-750 = 245 >= 200 -> gold
      (245 < 275, so the 200-char ABSOLUTE branch is what credits chunk0)
    * chunk1 overlap = 1300-795 = 505 >= 200 -> gold
    Short single-chunk docs (b/c/d) each yield exactly their one chunk.
    """
    legalbench_rag.convert(FIXTURE_RAW, tmp_path)

    a0 = _chunk_id(DOC_A[0:995], "contractnli/doc_a.txt", _META_A)
    a1 = _chunk_id(DOC_A[795:1300], "contractnli/doc_a.txt", _META_A)
    b = _chunk_id(DOC_B, "contractnli/doc_b.txt", _META_B)
    c = _chunk_id(DOC_C, "cuad/doc_c.txt", _META_C)
    d = _chunk_id(DOC_D, "cuad/doc_d.txt", _META_D)

    expected = _dumps(
        [
            {"query_id": "contractnli_0000", "qrels": {a0: 1, a1: 1}},
            {"query_id": "contractnli_0001", "qrels": {b: 1}},
            {"query_id": "cuad_0000", "qrels": {c: 1}},
            {"query_id": "cuad_0001", "qrels": {d: 1}},
        ]
    )
    assert (tmp_path / "anchor_qrels.jsonl").read_text("utf-8") == expected


def test_convert_manifest_counts_and_provenance(tmp_path: Path) -> None:
    manifest = legalbench_rag.convert(FIXTURE_RAW, tmp_path)

    assert manifest.dataset == "legalbench_rag"
    assert manifest.n_chunks == 5  # doc_a x2 + doc_b + doc_c + doc_d
    assert manifest.n_queries == 4
    assert manifest.n_qrel_entries == 5  # 2 (doc_a) + 1 + 1 + 1
    assert manifest.license_note == LICENSE_NOTES["legalbench_rag"]
    assert set(manifest.output_sha256) == {"chunks.jsonl", "queries.jsonl", "anchor_qrels.jsonl"}
    for filename, digest in manifest.output_sha256.items():
        assert digest == hashlib.sha256((tmp_path / filename).read_bytes()).hexdigest()

    on_disk = json.loads((tmp_path / "manifest.json").read_text("utf-8"))
    assert on_disk["n_chunks"] == 5


def test_convert_twice_is_byte_identical(tmp_path: Path) -> None:
    out_a, out_b = tmp_path / "a", tmp_path / "b"
    legalbench_rag.convert(FIXTURE_RAW, out_a)
    legalbench_rag.convert(FIXTURE_RAW, out_b)
    for filename in ("chunks.jsonl", "queries.jsonl", "anchor_qrels.jsonl", "manifest.json"):
        assert (out_a / filename).read_bytes() == (out_b / filename).read_bytes()


def test_convert_loader_round_trip_every_qrel_chunk_id_resolves(tmp_path: Path) -> None:
    """End-to-end: the REAL jsonl_loader re-keys span-gold to loaded chunk ids."""
    legalbench_rag.convert(FIXTURE_RAW, tmp_path)

    chunks = load_chunks(tmp_path / "chunks.jsonl")
    queries = load_queries(tmp_path / "queries.jsonl")
    qrels = load_anchor_qrels(tmp_path / "anchor_qrels.jsonl")

    chunk_ids = {c.chunk_id for c in chunks}
    query_ids = {q.query_id for q in queries}
    assert len(chunks) == 5
    assert len(queries) == 4
    # the boundary-crossing query is gold for both doc_a chunks:
    assert len(qrels["contractnli_0000"]) == 2
    for query_id, gold in qrels.items():
        assert query_id in query_ids
        for chunk_id in gold:
            assert chunk_id in chunk_ids


def test_convert_chunk_and_query_carry_subcorpus_metadata(tmp_path: Path) -> None:
    """D36: metadata.subcorpus on every chunk AND every query row."""
    legalbench_rag.convert(FIXTURE_RAW, tmp_path)

    chunk_rows = [
        json.loads(ln)
        for ln in (tmp_path / "chunks.jsonl").read_text("utf-8").splitlines()
        if ln.strip()
    ]
    query_rows = [
        json.loads(ln)
        for ln in (tmp_path / "queries.jsonl").read_text("utf-8").splitlines()
        if ln.strip()
    ]
    assert {r["metadata"]["subcorpus"] for r in chunk_rows} == {"contractnli", "cuad"}
    assert {r["metadata"]["subcorpus"] for r in query_rows} == {"contractnli", "cuad"}


def test_unmatched_snippet_path_warns_and_leaves_output_unchanged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A snippet whose file_path matches no converted doc warns; outputs stay byte-identical."""
    baseline_out = tmp_path / "baseline"
    legalbench_rag.convert(FIXTURE_RAW, baseline_out)

    mutated_raw = tmp_path / "raw"
    shutil.copytree(FIXTURE_RAW, mutated_raw)
    bench = mutated_raw / "benchmarks" / "cuad.json"
    entries = json.loads(bench.read_text("utf-8"))
    entries[0]["snippets"].append({"file_path": "cuad/ghost.txt", "span": [0, 50]})
    bench.write_text(json.dumps(entries), "utf-8")

    mutated_out = tmp_path / "mutated"
    with caplog.at_level(logging.WARNING, logger="ragsynth.datasets.converters.legalbench_rag"):
        legalbench_rag.convert(mutated_raw, mutated_out)

    messages = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("cuad" in m and "cuad/ghost.txt" in m and "1" in m for m in messages), (
        f"expected an unmatched-path warning naming cuad/ghost.txt, got {messages!r}"
    )

    for fname in ("chunks.jsonl", "queries.jsonl", "anchor_qrels.jsonl"):
        assert (mutated_out / fname).read_bytes() == (baseline_out / fname).read_bytes()


def test_clean_fixture_conversion_emits_no_unmatched_path_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="ragsynth.datasets.converters.legalbench_rag"):
        legalbench_rag.convert(FIXTURE_RAW, tmp_path / "out")
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]
