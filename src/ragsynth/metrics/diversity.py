"""Light diversity metrics for synthetic query sets (SPEC §8-9, v1).

Two v1 meters: surface diversity via distinct-n (Li et al., "A
Diversity-Promoting Objective Function for Neural Conversation Models",
NAACL 2016) and semantic redundancy via a greedy cosine-threshold dedup
pass in the spirit of MMR (Carbonell & Goldstein, SIGIR 1998; PLAN D13).
The Vendi Score (Friedman & Dieng, "The Vendi Score: A Diversity Evaluation
Metric for Machine Learning", TMLR 2023) is the planned v2 extension,
shipped behind an optional extra.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

_NORM_EPS = 1e-12
"""Floor on row norms so zero vectors do not divide by zero."""


def distinct_n(texts: Sequence[str], n: int = 1) -> float:
    """Fraction of word n-grams that are unique across ``texts``.

    Distinct-n (Li et al., NAACL 2016): unique n-grams divided by total
    n-grams, with whitespace tokenization. Texts shorter than ``n`` tokens
    contribute nothing. 1.0 = no repetition; lower = more repetitive.

    Args:
        texts: Corpus of texts (one synthetic query each).
        n: N-gram order (1 = distinct-1, 2 = distinct-2, ...).

    Returns:
        Unique-to-total n-gram ratio, or 0.0 when no n-grams exist (empty
        input or all texts shorter than ``n``).

    Raises:
        ValueError: If ``n`` is not a positive integer.
    """
    if n < 1:
        raise ValueError(f"n-gram order must be >= 1, got {n}")
    total = 0
    unique: set[tuple[str, ...]] = set()
    for text in texts:
        tokens = text.split()
        for i in range(len(tokens) - n + 1):
            unique.add(tuple(tokens[i : i + n]))
            total += 1
    if total == 0:
        return 0.0
    return len(unique) / total


def semantic_dedup_rate(embs: NDArray[np.float64], threshold: float = 0.95) -> float:
    """Fraction of embeddings dropped by a greedy cosine-threshold dedup pass.

    Greedy in-order scan (the documented MMR simplification, Carbonell &
    Goldstein, SIGIR 1998; PLAN D13): a vector is dropped when its cosine
    similarity to any already-kept vector is >= ``threshold``, otherwise it
    joins the kept set. 0.0 = no near-duplicates; higher = more semantic
    redundancy. Rows are re-normalized defensively before comparison.

    Args:
        embs: Embedding matrix, shape ``(n, d)``.
        threshold: Cosine similarity at or above which a vector counts as a
            duplicate of a kept one.

    Returns:
        Dropped fraction in [0, 1]; 0.0 for empty input.
    """
    x = np.asarray(embs, dtype=np.float64)
    if len(x) == 0:
        return 0.0
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / np.maximum(norms, _NORM_EPS)
    kept: list[int] = []
    dropped = 0
    for i in range(len(x)):
        if kept and float(np.max(x[kept] @ x[i])) >= threshold:
            dropped += 1
        else:
            kept.append(i)
    return dropped / len(x)
