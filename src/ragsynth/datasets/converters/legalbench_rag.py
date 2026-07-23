"""LegalBench-RAG converter: D34 splitter, D35 span->chunk gold, D36 merge (spec01 §5).

Unlike the BEIR 1:1 converters (``fiqa``/``nfcorpus``), LegalBench-RAG ships
long contracts with *span-level* ground truth, so this module adds three
legalbench-specific pieces on top of the shared ``base`` helpers:

* :func:`split_document` -- the D34 deterministic character-window splitter
  (1,000-char windows, 200-char overlap, window end snapped back to the last
  whitespace; a window with no whitespace is cut hard at 1,000). Every chunk
  keeps its ``(start, end)`` char offsets so D35 can measure span overlap.
* :func:`spans_to_qrels` -- the D35 gold rule (spec01 §5 signature verbatim):
  a chunk is gold for a query iff ``|span ∩ chunk| >= min(0.5*|span|, 200)``
  characters for at least one gold span.
* :func:`convert` -- the D36 merge: the CUAD / ContractNLI / MAUD / PrivacyQA
  sub-corpora form ONE corpus with ``metadata.subcorpus`` on every chunk AND
  query. Span-derived gold is re-keyed to content-hash chunk ids through the
  same :meth:`Chunk.create` path ``base`` uses, so the loader recomputes the
  identical ids (spec01 §5).

Raw layout (the mini-fixture under ``tests/fixtures/benchmarks/legalbench_rag/``
defines it; the real 2024 release fetch/adjustment is a later task -- see the
report's known-assumptions note)::

    raw/corpus/<subcorpus>/<doc>.txt          # UTF-8 contract text
    raw/benchmarks/<subcorpus>.json           # queries + span-level gold

Each ``<subcorpus>.json`` is a JSON list of entries (an object with a
``"tests"`` list is also accepted, matching the upstream release), each entry
``{"query": str, "snippets": [{"file_path": str, "span": [start, end]}, ...]}``
where ``file_path`` is relative to ``raw/corpus/`` (so it carries the
sub-corpus prefix) and ``span`` is a half-open ``[start, end)`` char range.

Emission order is deterministic: sub-corpora in sorted name order; within a
sub-corpus, documents in sorted file-path order and queries in benchmark-file
(upstream) order; window chunks in split order. Two conversions of the same
raw tree are byte-identical (the manifest sha256s prove it).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from ragsynth.datasets.converters.base import (
    LICENSE_NOTES,
    ConversionManifest,
    resolve_source_version,
    sha256_of_file,
    write_jsonl,
    write_manifest,
)
from ragsynth.domain import Chunk

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Any

name = "legalbench_rag"

#: The four sub-corpora that D36 merges into one corpus (documentation only;
#: the converter processes whichever sub-corpora are present in ``raw/``).
SUBCORPORA: tuple[str, ...] = ("contractnli", "cuad", "maud", "privacy_qa")

_WINDOW_CHARS = 1000
_OVERLAP_CHARS = 200
_MIN_FRAC = 0.5
_MIN_CHARS = 200


# ---------------------------------------------------------------------------
# D34 — deterministic character-window splitter.
# ---------------------------------------------------------------------------
def split_document(
    text: str, *, window_chars: int = _WINDOW_CHARS, overlap_chars: int = _OVERLAP_CHARS
) -> list[tuple[int, int]]:
    """Split ``text`` into overlapping char windows, returning ``(start, end)`` offsets.

    Windows are ``window_chars`` wide with ``overlap_chars`` of overlap. Every
    non-final window's end is snapped back to the last whitespace character
    inside the window so words are not split mid-token; a window containing no
    snappable whitespace is cut hard at ``window_chars`` (D34). The next window
    starts ``overlap_chars`` before the (snapped) end, so consecutive chunks
    share exactly ``overlap_chars`` characters of duplicated text. The final
    window always ends at ``len(text)`` (no snap).

    Args:
        text: The full document text.
        window_chars: Target window width in characters (D34: 1,000).
        overlap_chars: Overlap between consecutive windows (D34: 200).

    Returns:
        ``[(start, end), ...]`` half-open char offsets, in document order;
        empty for empty ``text``.
    """
    n = len(text)
    if n == 0:
        return []
    windows: list[tuple[int, int]] = []
    start = 0
    while True:
        raw_end = start + window_chars
        if raw_end >= n:
            windows.append((start, n))
            break
        end = _snap_back(text, start, raw_end, overlap_chars)
        windows.append((start, end))
        start = end - overlap_chars
    return windows


def _snap_back(text: str, start: int, raw_end: int, overlap_chars: int) -> int:
    """Return the last whitespace index in ``text[start:raw_end]``, else ``raw_end``.

    The snap is only taken when it still lets the next window advance past
    ``start`` (i.e. the whitespace sits beyond ``start + overlap_chars``);
    otherwise the window is cut hard at ``raw_end``. This both honours the D34
    "no whitespace -> hard cut" rule and guarantees the split loop terminates
    on pathological runs with whitespace only in the overlap prefix.
    """
    for i in range(raw_end - 1, start, -1):
        if text[i].isspace():
            if i - overlap_chars > start:
                return i
            break  # last whitespace is too early to make progress -> hard cut
    return raw_end


# ---------------------------------------------------------------------------
# D35 — span -> chunk gold rule (spec01 §5 signature verbatim).
# ---------------------------------------------------------------------------
def spans_to_qrels(
    spans: Mapping[str, Sequence[tuple[str, int, int]]],  # query_id -> [(doc_id, start, end)]
    chunk_offsets: Mapping[str, tuple[str, int, int]],  # chunk_id -> (doc_id, start, end)
    *,
    min_frac: float = _MIN_FRAC,
    min_chars: int = _MIN_CHARS,  # D35
) -> dict[str, dict[str, int]]:
    """Map query gold spans to gold chunk ids by character overlap (D35).

    A chunk is gold for a query iff, for at least one of the query's spans in
    the *same document*, ``|span ∩ chunk| >= min(min_frac*|span|, min_chars)``
    characters. The two branches: a long span split across chunks credits both
    halves (the 200-char absolute branch), while a chunk merely grazing a short
    span is excluded (the 50%-of-span branch). Grade is binary (1).

    Args:
        spans: ``query_id -> [(doc_id, start, end), ...]`` gold spans
            (half-open char ranges).
        chunk_offsets: ``chunk_id -> (doc_id, start, end)`` for every chunk.
        min_frac: Fraction-of-span threshold (D35: 0.5).
        min_chars: Absolute char threshold (D35: 200).

    Returns:
        ``query_id -> {chunk_id: 1}``, omitting queries with no gold chunk.
    """
    out: dict[str, dict[str, int]] = {}
    for query_id, query_spans in spans.items():
        gold: dict[str, int] = {}
        for chunk_id, chunk in chunk_offsets.items():
            if _chunk_is_gold(query_spans, chunk, min_frac=min_frac, min_chars=min_chars):
                gold[chunk_id] = 1
        if gold:
            out[query_id] = gold
    return out


def _chunk_is_gold(
    query_spans: Sequence[tuple[str, int, int]],
    chunk: tuple[str, int, int],
    *,
    min_frac: float,
    min_chars: int,
) -> bool:
    """Return whether ``chunk`` meets the D35 threshold for any of ``query_spans``."""
    c_doc, c_start, c_end = chunk
    for s_doc, s_start, s_end in query_spans:
        if s_doc != c_doc:
            continue
        overlap = min(s_end, c_end) - max(s_start, c_start)
        if overlap <= 0:
            continue
        threshold = min(min_frac * (s_end - s_start), min_chars)
        if overlap >= threshold:
            return True
    return False


# ---------------------------------------------------------------------------
# D36 — merge sub-corpora into one corpus; convert() entry point.
# ---------------------------------------------------------------------------
def convert(raw_dir: Path, out_dir: Path) -> ConversionManifest:
    """Convert a LegalBench-RAG raw tree into ``chunks``/``queries``/``anchor_qrels`` jsonl.

    Args:
        raw_dir: Directory holding ``corpus/<subcorpus>/<doc>.txt`` and
            ``benchmarks/<subcorpus>.json`` (see the module docstring).
        out_dir: Destination directory for the three emitted jsonl files and
            ``manifest.json``.

    Returns:
        The :class:`~ragsynth.datasets.converters.base.ConversionManifest`
        written alongside the output.
    """
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    corpus_root = raw_dir / "corpus"
    benchmarks_root = raw_dir / "benchmarks"
    subcorpora = _discover_subcorpora(benchmarks_root)

    chunk_rows: list[dict[str, object]] = []
    chunk_offsets: dict[str, tuple[str, int, int]] = {}
    for subcorpus in subcorpora:
        for doc_path in sorted((corpus_root / subcorpus).glob("*.txt")):
            doc_id = doc_path.relative_to(corpus_root).as_posix()
            text = doc_path.read_text(encoding="utf-8")
            metadata = {"subcorpus": subcorpus, "upstream_id": doc_id}
            for start, end in split_document(text):
                chunk = Chunk.create(text=text[start:end], doc_id=doc_id, metadata=dict(metadata))
                chunk_offsets[chunk.chunk_id] = (doc_id, start, end)
                chunk_rows.append(
                    {"text": chunk.text, "doc_id": chunk.doc_id, "metadata": dict(chunk.metadata)}
                )

    query_rows: list[dict[str, object]] = []
    query_order: list[str] = []
    spans: dict[str, list[tuple[str, int, int]]] = {}
    for subcorpus in subcorpora:
        for i, entry in enumerate(_read_benchmark(benchmarks_root / f"{subcorpus}.json")):
            query_id = f"{subcorpus}_{i:04d}"
            query_order.append(query_id)
            query_rows.append(
                {
                    "query_id": query_id,
                    "text": str(entry["query"]),
                    "metadata": {"subcorpus": subcorpus},
                }
            )
            spans[query_id] = [
                (str(s["file_path"]), int(s["span"][0]), int(s["span"][1]))
                for s in entry["snippets"]
            ]

    qrels_by_query = spans_to_qrels(spans, chunk_offsets)
    qrel_rows = [
        {"query_id": qid, "qrels": qrels_by_query[qid]}
        for qid in query_order
        if qid in qrels_by_query
    ]

    write_jsonl(out_dir / "chunks.jsonl", chunk_rows)
    write_jsonl(out_dir / "queries.jsonl", query_rows)
    write_jsonl(out_dir / "anchor_qrels.jsonl", qrel_rows)

    output_sha256 = {
        fname: sha256_of_file(out_dir / fname)
        for fname in ("chunks.jsonl", "queries.jsonl", "anchor_qrels.jsonl")
    }
    manifest = ConversionManifest(
        dataset=name,
        n_chunks=len(chunk_rows),
        n_queries=len(query_rows),
        n_qrel_entries=sum(len(row["qrels"]) for row in qrel_rows),
        source_version=resolve_source_version(raw_dir),
        license_note=LICENSE_NOTES[name],
        output_sha256=output_sha256,
    )
    write_manifest(out_dir, manifest)
    return manifest


def _discover_subcorpora(benchmarks_root: Path) -> list[str]:
    """Return sub-corpus names (sorted ``benchmarks/*.json`` stems), failing loudly if none.

    Raises:
        FileNotFoundError: If ``benchmarks_root`` is missing or holds no
            ``*.json`` files -- refusing to emit an empty conversion.
    """
    if not benchmarks_root.is_dir():
        raise FileNotFoundError(
            f"benchmarks directory not found: {benchmarks_root} -- expected the "
            "LegalBench-RAG layout <raw_dir>/benchmarks/<subcorpus>.json (see "
            "ragsynth.datasets.converters.legalbench_rag for the schema)"
        )
    stems = sorted(p.stem for p in benchmarks_root.glob("*.json"))
    if not stems:
        raise FileNotFoundError(
            f"no <subcorpus>.json files under {benchmarks_root} -- refusing to emit "
            "an empty conversion"
        )
    return stems


def _read_benchmark(path: Path) -> list[dict[str, Any]]:
    """Read one ``<subcorpus>.json`` as a list of ``{query, snippets}`` entries.

    Accepts either a bare JSON list of entries or an object with a ``"tests"``
    list (the upstream 2024 release wraps entries under ``"tests"``).

    Raises:
        TypeError: If the file is neither a list nor a ``{"tests": [...]}``
            object.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("tests"), list):
        entries = data["tests"]
    elif isinstance(data, list):
        entries = data
    else:
        raise TypeError(
            f"unexpected benchmark shape in {path}: expected a JSON list of entries "
            'or an object with a "tests" list'
        )
    return [dict(entry) for entry in entries]
