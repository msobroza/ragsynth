"""Deterministic retrieval-system zoo and per-query nDCG@k scoring.

The validity meter needs a fixed family of retrieval systems of graded
quality: an exact system, increasingly distorted embedding mixes, and
low-rank projections (prototype ``reference/synth_query_eval.py``
L779-821). Each system scores a query set against a shared chunk corpus
with binary-qrel nDCG@k (PLAN D16), which reduces exactly to the
prototype's single-gold ``1/log2(1+rank)`` when a query has one gold.

The metrics layer intentionally does not import ``ragsynth.sampling``
(dependency direction: sampling may use metrics-free utilities, metrics
stay leaf-importable), so a module-private ``_l2_normalize`` is defined
here instead of reusing the sampling copy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from numpy.typing import NDArray

__all__ = [
    "MatrixSystem",
    "RetrievalSystem",
    "evaluate_zoo",
    "make_system_zoo",
]

_EPS = 1e-12
_DEFAULT_K = 10
_DISTORT_SIGMAS = (0.10, 0.28, 0.30, 0.32, 0.34, 0.50, 0.75)
_RANK_FRACTIONS = (0.75, 0.5, 0.4375, 0.375)  # 48/32/28/24 at d=64 (prototype L787)
_MIN_RANK = 2


def _l2_normalize(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Project rows onto the unit sphere (module-private, see module docstring)."""
    return np.asarray(
        x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), _EPS), dtype=np.float64
    )


class RetrievalSystem(Protocol):
    """A system that scores queries against its own indexed corpus."""

    def per_query_scores(
        self,
        query_embs: NDArray[np.float64],
        qrels: Sequence[Mapping[str, int]],
        k: int = _DEFAULT_K,
        drop_mask: NDArray[np.bool_] | None = None,
    ) -> NDArray[np.float64]:
        """Score each query with nDCG@k against binary qrels.

        Args:
            query_embs: Shape ``(Q, d)`` query embeddings.
            qrels: One ``{chunk_id: grade}`` mapping per query; grades > 0
                mark relevant chunks.
            k: Retrieval cutoff depth.
            drop_mask: Optional shape ``(N,)`` boolean mask of chunks
                removed from the index (positive-control degradation).

        Returns:
            Shape ``(Q,)`` per-query nDCG@k scores.
        """
        ...


@dataclass(frozen=True)
class MatrixSystem:
    """Retrieval system defined by a ``(d, d)`` embedding transform.

    Queries and chunks are both mapped through ``matrix``, L2-normalized,
    and ranked by cosine similarity -- the prototype's ``score_system``
    (L793-814) generalized to binary multi-gold qrels via nDCG@k (PLAN D16).

    Attributes:
        name: Zoo key, e.g. ``"exact"`` or ``"distort-0.3"``.
        matrix: Shape ``(d, d)`` transform applied to queries and chunks.
        chunk_ids: Corpus chunk ids, aligned with ``chunk_embs`` rows.
        chunk_embs: Shape ``(N, d)`` corpus embeddings.
    """

    name: str
    matrix: NDArray[np.float64]
    chunk_ids: tuple[str, ...]
    chunk_embs: NDArray[np.float64]

    def per_query_scores(
        self,
        query_embs: NDArray[np.float64],
        qrels: Sequence[Mapping[str, int]],
        k: int = _DEFAULT_K,
        drop_mask: NDArray[np.bool_] | None = None,
    ) -> NDArray[np.float64]:
        """Score each query with binary-qrel nDCG@k under this transform.

        Rank of a relevant chunk is ``1 + #{chunks with strictly higher
        cosine}`` (optimistic ties, prototype semantics); DCG sums
        ``1/log2(1+rank)`` over relevant chunks retrieved in the top k;
        IDCG places the ``min(R, k)`` relevant chunks at ranks ``1..``.
        With a single gold this reduces exactly to the prototype's
        ``1/log2(1+rank)``-if-in-top-k-else-0.

        Reference:
            Jarvelin & Kekalainen, "Cumulated Gain-Based Evaluation of IR
            Techniques", ACM TOIS 2002 (nDCG).

        Args:
            query_embs: Shape ``(Q, d)`` query embeddings.
            qrels: One ``{chunk_id: grade}`` mapping per query; grades > 0
                mark relevant chunks.
            k: Retrieval cutoff depth.
            drop_mask: Optional shape ``(N,)`` boolean mask of dropped
                chunks; dropped chunks are unretrievable and contribute
                nothing to DCG (their qrel entry still counts toward IDCG).

        Returns:
            Shape ``(Q,)`` per-query nDCG@k; ``0.0`` for a query with no
            relevant chunk retrievable.

        Raises:
            ValueError: If a qrel references a chunk_id not in the corpus.
        """
        qe = _l2_normalize(np.asarray(query_embs, dtype=np.float64) @ self.matrix)
        ce = _l2_normalize(np.asarray(self.chunk_embs, dtype=np.float64) @ self.matrix)
        sims = qe @ ce.T  # (Q, N)
        dropped = (
            np.zeros(len(self.chunk_ids), dtype=np.bool_)
            if drop_mask is None
            else np.asarray(drop_mask, dtype=np.bool_)
        )
        sims[:, dropped] = -np.inf
        index = {chunk_id: i for i, chunk_id in enumerate(self.chunk_ids)}

        scores = np.zeros(sims.shape[0], dtype=np.float64)
        for qi, qrel in enumerate(qrels):
            relevant: list[int] = []
            for chunk_id, grade in qrel.items():
                if chunk_id not in index:
                    raise ValueError(
                        f"qrel for query {qi} references unknown chunk_id {chunk_id!r}"
                    )
                if grade > 0:
                    relevant.append(index[chunk_id])
            if not relevant:
                continue
            row = sims[qi]
            dcg = 0.0
            for j in relevant:
                if dropped[j]:
                    continue
                rank = int((row > row[j]).sum()) + 1
                if rank <= k:
                    dcg += 1.0 / math.log2(rank + 1.0)
            idcg = sum(1.0 / math.log2(r + 1.0) for r in range(1, min(len(relevant), k) + 1))
            scores[qi] = dcg / idcg
        return scores


def make_system_zoo(
    chunk_ids: Sequence[str],
    chunk_embs: NDArray[np.float64],
    seed: int = 0,
) -> dict[str, MatrixSystem]:
    """Build the deterministic 12-system zoo over a shared corpus.

    Embedding-model variants of graded quality (prototype L779-790): the
    identity ("exact"), seven increasingly distorted mixes
    ``I + sigma * G / sqrt(d)``, and four low-rank projections ``Q Q^T``
    with rank ``r = max(2, round(d * f))`` for ``f`` in ``(0.75, 0.5,
    0.4375, 0.375)`` -- 48/32/28/24 at d=64, matching the prototype. For
    very small d (< 8) two fractions may round to the same rank and
    collide on the zoo key; use d >= 8.

    Args:
        chunk_ids: Corpus chunk ids, aligned with ``chunk_embs`` rows.
        chunk_embs: Shape ``(N, d)`` corpus embeddings.
        seed: Seed for the zoo's matrix RNG.

    Returns:
        Insertion-ordered mapping ``{name: MatrixSystem}`` -- "exact"
        first, then ``distort-*``, then ``rank-*``.
    """
    embs = np.asarray(chunk_embs, dtype=np.float64)
    d = embs.shape[1]
    rng = np.random.default_rng(seed)
    matrices: dict[str, NDArray[np.float64]] = {"exact": np.eye(d)}
    for sigma in _DISTORT_SIGMAS:
        matrices[f"distort-{sigma}"] = np.eye(d) + sigma * rng.standard_normal((d, d)) / np.sqrt(d)
    for frac in _RANK_FRACTIONS:
        r = max(_MIN_RANK, round(d * frac))
        q, _ = np.linalg.qr(rng.standard_normal((d, r)))
        matrices[f"rank-{r}"] = q @ q.T
    ids = tuple(chunk_ids)
    return {
        name: MatrixSystem(name=name, matrix=matrix, chunk_ids=ids, chunk_embs=embs)
        for name, matrix in matrices.items()
    }


def evaluate_zoo(
    zoo: Mapping[str, RetrievalSystem],
    query_embs: NDArray[np.float64],
    qrels: Sequence[Mapping[str, int]],
    k: int = _DEFAULT_K,
) -> NDArray[np.float64]:
    """Score every zoo system on a query set (prototype L817-821).

    Args:
        zoo: Insertion-ordered mapping of retrieval systems.
        query_embs: Shape ``(Q, d)`` query embeddings.
        qrels: One ``{chunk_id: grade}`` mapping per query.
        k: Retrieval cutoff depth.

    Returns:
        Shape ``(S, Q)`` per-query score matrix, rows in the zoo's
        insertion order.
    """
    return np.vstack([system.per_query_scores(query_embs, qrels, k=k) for system in zoo.values()])
