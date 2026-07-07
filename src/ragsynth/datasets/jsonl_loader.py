"""JSONL corpus loader: chunks, production queries, anchor qrels (SPEC §10).

File shapes (one JSON object per line):
    - ``chunks.jsonl``: ``{"text": ..., "doc_id": ..., "page"?: int,
      "metadata"?: {str: str}}``
    - ``queries.jsonl``: ``{"query_id"?: str, "text": ...,
      "timestamp"?: ISO-8601 str}``
    - ``anchor_qrels.jsonl``: ``{"query_id": str, "qrels": {chunk_id: grade}}``

The ``jsonl`` dataset splits queries train/anchor/oracle via a seeded
permutation (PLAN D10) and leaves ``embeddings``/``bank`` as ``None`` --
text corpora embed through a real featurizer at composition time.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from ragsynth.datasets.base import DATASETS, DatasetBundle
from ragsynth.domain import Chunk, ProductionQuery
from ragsynth.pipeline.base import stable_hash64

logger = logging.getLogger(__name__)

_DEFAULT_SPLIT = (0.60, 0.25, 0.15)
_SPLIT_PARTS = 3
_SPLIT_SUM_TOL = 1e-6


def _read_jsonl(path: Path, kind: str) -> list[dict[str, Any]]:
    """Read a JSONL file, one object per non-blank line.

    Raises:
        FileNotFoundError: With the expected shape spelled out.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"{kind} file not found: {path} -- expected a JSONL file "
            "(one JSON object per line); see ragsynth.datasets.jsonl_loader for the schema"
        )
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_chunks(path: Path | str) -> list[Chunk]:
    """Load knowledge-base chunks from JSONL via :meth:`Chunk.create`.

    Args:
        path: Path to a ``chunks.jsonl`` with required ``text``/``doc_id``
            and optional ``page``/``metadata`` fields per line.

    Returns:
        Chunks in file order, with content-addressed ids filled in.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    rows = _read_jsonl(Path(path), "chunks")
    return [
        Chunk.create(
            text=str(row["text"]),
            doc_id=str(row["doc_id"]),
            page=int(row["page"]) if row.get("page") is not None else None,
            metadata={str(k): str(v) for k, v in row.get("metadata", {}).items()},
        )
        for row in rows
    ]


def load_queries(path: Path | str) -> list[ProductionQuery]:
    """Load production queries from JSONL.

    Args:
        path: Path to a ``queries.jsonl`` with required ``text`` and
            optional ``query_id`` (default ``q{line:05d}``) / ISO-8601
            ``timestamp`` fields per line.

    Returns:
        Queries in file order.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    rows = _read_jsonl(Path(path), "queries")
    queries = []
    for i, row in enumerate(rows):
        timestamp = row.get("timestamp")
        queries.append(
            ProductionQuery(
                query_id=str(row.get("query_id") or f"q{i:05d}"),
                text=str(row["text"]),
                timestamp=datetime.fromisoformat(timestamp) if timestamp else None,
            )
        )
    return queries


def load_anchor_qrels(path: Path | str) -> dict[str, dict[str, int]]:
    """Load anchor relevance judgments from JSONL.

    Args:
        path: Path to an ``anchor_qrels.jsonl`` with ``query_id`` and a
            ``qrels`` mapping (chunk_id -> integer grade) per line.

    Returns:
        Mapping ``query_id -> {chunk_id: grade}``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    rows = _read_jsonl(Path(path), "anchor qrels")
    return {
        str(row["query_id"]): {str(cid): int(grade) for cid, grade in row["qrels"].items()}
        for row in rows
    }


@DATASETS.register("jsonl")
class JsonlDataset:
    """Local JSONL corpus: the first-real-run dataset (SPEC §10)."""

    @classmethod
    def build(cls, params: dict[str, Any], seed: int) -> DatasetBundle:
        """Load the corpus and split queries with a seeded permutation.

        Args:
            params: ``chunks_path``, ``queries_path``, optional
                ``anchor_qrels_path``, optional ``split`` fractions
                (default 0.60/0.25/0.15, PLAN D10).
            seed: Config seed; the split uses the substream
                ``[seed, stable_hash64("jsonl_split")]``.

        Returns:
            A :class:`DatasetBundle` with ``embeddings``/``bank`` left
            ``None`` and ``anchor_qrels`` filtered to the anchor split.

        Raises:
            FileNotFoundError: If a referenced file does not exist.
            ValueError: If ``split`` is not three fractions summing to 1.
        """
        chunks = tuple(load_chunks(str(params["chunks_path"])))
        queries = load_queries(str(params["queries_path"]))
        split = tuple(float(f) for f in params.get("split", _DEFAULT_SPLIT))
        if len(split) != _SPLIT_PARTS or abs(sum(split) - 1.0) > _SPLIT_SUM_TOL:
            raise ValueError(
                f"split must be three train/anchor/oracle fractions summing to 1.0, got {split}"
            )

        rng = np.random.default_rng([seed, stable_hash64("jsonl_split")])
        shuffled = [queries[int(i)] for i in rng.permutation(len(queries))]
        n_train = int(split[0] * len(queries))
        n_anchor = int(split[1] * len(queries))
        train = tuple(shuffled[:n_train])
        anchor = tuple(shuffled[n_train : n_train + n_anchor])
        oracle = tuple(shuffled[n_train + n_anchor :])

        anchor_qrels: dict[str, dict[str, int]] = {}
        qrels_path = params.get("anchor_qrels_path")
        if qrels_path:
            all_qrels = load_anchor_qrels(str(qrels_path))
            anchor_ids = {q.query_id for q in anchor}
            anchor_qrels = {qid: qr for qid, qr in all_qrels.items() if qid in anchor_ids}
            dropped = len(all_qrels) - len(anchor_qrels)
            if dropped:
                logger.info(
                    f"jsonl dataset: {dropped} qrel entries fell outside the anchor split "
                    "and were dropped"
                )
        logger.info(
            f"jsonl dataset built: {len(chunks)} chunks, "
            f"{len(train)}/{len(anchor)}/{len(oracle)} train/anchor/oracle queries"
        )
        return DatasetBundle(
            chunks=chunks,
            queries_train=train,
            queries_anchor=anchor,
            queries_oracle=oracle,
            anchor_qrels=anchor_qrels,
            oracle_qrels={},
            embeddings=None,
            bank=None,
        )
