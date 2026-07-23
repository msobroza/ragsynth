"""Unit tests for converters/base.py shared helpers (spec01 §5)."""

import hashlib
import json

import pytest

from ragsynth.datasets.converters.base import (
    BENCHMARK_URLS,
    LICENSE_NOTES,
    ConversionManifest,
    binarize_qrels,
    build_chunks,
    read_beir_corpus,
    read_beir_qrels_dir,
    read_beir_queries,
    render_benchmarks_readme,
    resolve_source_version,
    sha256_of_file,
    write_benchmarks_readme,
    write_jsonl,
    write_manifest,
)


def test_benchmark_urls_known_datasets() -> None:
    assert BENCHMARK_URLS["fiqa"] == (
        "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip"
    )
    assert BENCHMARK_URLS["nfcorpus"] == (
        "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip"
    )


def test_license_notes_present_for_both_datasets() -> None:
    assert set(LICENSE_NOTES) == {"fiqa", "nfcorpus"}
    assert all(isinstance(note, str) and note for note in LICENSE_NOTES.values())


def test_sha256_of_file(tmp_path) -> None:
    path = tmp_path / "f.bin"
    path.write_bytes(b"hello world")
    assert sha256_of_file(path) == hashlib.sha256(b"hello world").hexdigest()


def test_write_jsonl_deterministic_bytes_sorted_keys(tmp_path) -> None:
    path = tmp_path / "out" / "rows.jsonl"
    write_jsonl(path, [{"b": 1, "a": 2}, {"z": "x"}])
    assert path.read_text(encoding="utf-8") == '{"a": 2, "b": 1}\n{"z": "x"}\n'


def test_write_jsonl_twice_is_byte_identical(tmp_path) -> None:
    path = tmp_path / "rows.jsonl"
    rows = [{"text": "café", "doc_id": "d1"}]
    write_jsonl(path, rows)
    first = path.read_bytes()
    write_jsonl(path, rows)
    second = path.read_bytes()
    assert first == second
    # ensure_ascii=False: non-ascii stays literal, not \uXXXX escaped.
    assert "café".encode() in first


def test_write_manifest_canonical_json_round_trips(tmp_path) -> None:
    manifest = ConversionManifest(
        dataset="fiqa",
        n_chunks=5,
        n_queries=4,
        n_qrel_entries=4,
        source_version="dircontent:abc",
        license_note="note",
        output_sha256={"chunks.jsonl": "aaa"},
    )
    path = write_manifest(tmp_path, manifest)
    assert path == tmp_path / "manifest.json"
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded == {
        "dataset": "fiqa",
        "n_chunks": 5,
        "n_queries": 4,
        "n_qrel_entries": 4,
        "source_version": "dircontent:abc",
        "license_note": "note",
        "output_sha256": {"chunks.jsonl": "aaa"},
    }


def test_resolve_source_version_reads_marker_file(tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "SOURCE_SHA256").write_text("deadbeef\n", encoding="utf-8")
    assert resolve_source_version(raw_dir) == "sha256:deadbeef"


def test_resolve_source_version_falls_back_to_deterministic_dircontent_marker(tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "corpus.jsonl").write_text('{"_id": "1", "text": "x"}\n', encoding="utf-8")

    first = resolve_source_version(raw_dir)
    second = resolve_source_version(raw_dir)
    assert first == second
    assert first.startswith("dircontent:")

    (raw_dir / "corpus.jsonl").write_text('{"_id": "1", "text": "y"}\n', encoding="utf-8")
    assert resolve_source_version(raw_dir) != first


def test_read_beir_corpus_preserves_upstream_order_and_defaults_missing_title(tmp_path) -> None:
    path = tmp_path / "corpus.jsonl"
    path.write_text(
        '{"_id": "d3", "title": "T", "text": "b"}\n{"_id": "d1", "text": "a"}\n',
        encoding="utf-8",
    )
    rows = read_beir_corpus(path)
    assert rows == [
        {"_id": "d3", "title": "T", "text": "b"},
        {"_id": "d1", "title": "", "text": "a"},
    ]


def test_read_beir_queries_preserves_upstream_order(tmp_path) -> None:
    path = tmp_path / "queries.jsonl"
    path.write_text('{"_id": "q2", "text": "b"}\n{"_id": "q1", "text": "a"}\n', encoding="utf-8")
    rows = read_beir_queries(path)
    assert rows == [{"_id": "q2", "text": "b"}, {"_id": "q1", "text": "a"}]


def test_read_beir_qrels_dir_dedup_keeps_true_max_regardless_of_file_processing_order(
    tmp_path,
) -> None:
    """Regression fixture: a naive 'last file wins' merge would get this wrong.

    Files are visited in sorted-filename order (``test.tsv`` before
    ``train.tsv``); ``train.tsv``'s lower score for (q4, d8) must NOT
    overwrite ``test.tsv``'s higher score merely because it is read second.
    """
    qrels_dir = tmp_path / "qrels"
    qrels_dir.mkdir()
    (qrels_dir / "test.tsv").write_text("query-id\tcorpus-id\tscore\nq4\td8\t2\n", encoding="utf-8")
    (qrels_dir / "train.tsv").write_text(
        "query-id\tcorpus-id\tscore\nq4\td8\t0\n", encoding="utf-8"
    )
    merged = read_beir_qrels_dir(qrels_dir)
    assert merged == {("q4", "d8"): 2}


def test_build_chunks_concatenates_title_and_text_only_when_title_nonempty() -> None:
    chunks = build_chunks(
        [
            {"_id": "d1", "title": "Title", "text": "Body one."},
            {"_id": "d2", "title": "", "text": "Body two."},
        ]
    )
    assert chunks[0].text == "Title\n\nBody one."
    assert chunks[0].doc_id == "d1"
    assert chunks[0].metadata == {"upstream_id": "d1"}
    assert chunks[1].text == "Body two."


def test_binarize_qrels_drops_subthreshold_and_unknown_corpus_ids() -> None:
    raw = {("q1", "d1"): 2, ("q1", "d2"): 0, ("q2", "d9"): 1}
    upstream_to_chunk_id = {"d1": "chunk-1", "d2": "chunk-2"}  # "d9" intentionally missing

    out = binarize_qrels(raw, upstream_to_chunk_id, threshold=1)

    assert out == {"q1": {"chunk-1": 1}}


def test_render_benchmarks_readme_sorted_and_deterministic() -> None:
    entries = {
        "nfcorpus": {"url": "u2", "sha256": "s2", "license_note": "n2"},
        "fiqa": {"url": "u1", "sha256": "s1", "license_note": "n1"},
    }
    text = render_benchmarks_readme(entries)
    assert text.index("## fiqa") < text.index("## nfcorpus")
    assert "u1" in text
    assert "s2" in text


def test_write_benchmarks_readme_merges_across_separate_calls(tmp_path) -> None:
    root = tmp_path / "benchmarks"
    write_benchmarks_readme(root, "fiqa", "url-fiqa", "sha-fiqa", "note-fiqa")
    write_benchmarks_readme(root, "nfcorpus", "url-nf", "sha-nf", "note-nf")

    text = (root / "README.md").read_text(encoding="utf-8")
    assert "## fiqa" in text
    assert "## nfcorpus" in text
    assert "url-fiqa" in text
    assert "url-nf" in text


def test_write_benchmarks_readme_idempotent_for_same_dataset(tmp_path) -> None:
    root = tmp_path / "benchmarks"
    write_benchmarks_readme(root, "fiqa", "url", "sha", "note")
    first = (root / "README.md").read_bytes()
    write_benchmarks_readme(root, "fiqa", "url", "sha", "note")
    second = (root / "README.md").read_bytes()
    assert first == second


@pytest.mark.parametrize("threshold", [1])
def test_binarize_qrels_keeps_exactly_threshold(threshold: int) -> None:
    raw = {("q1", "d1"): 1}
    out = binarize_qrels(raw, {"d1": "c1"}, threshold=threshold)
    assert out == {"q1": {"c1": 1}}
