"""Positive-control degradations and the paired bootstrap significance test.

A benchmark is only useful if it *detects* known regressions: this module
provides the degradation factory (index deletion, embedding-space noise)
and the one-sided paired bootstrap that decides whether a degraded system
scores significantly below its baseline (SPEC §8-9 "positive-control
battery"; Sakai's discriminative-power tradition, SIGIR 2006). Ported from
the frozen prototype (``reference/synth_query_eval.py`` L543-554, L866-868).

Top-k truncation -- the third degradation named in the SPEC -- needs no
factory: pass a smaller ``k`` to ``RetrievalSystem.per_query_scores``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = [
    "drop_index_mask",
    "noise_transform",
    "paired_bootstrap_pvalue",
]


def paired_bootstrap_pvalue(
    scores_base: NDArray[np.float64],
    scores_degraded: NDArray[np.float64],
    n_boot: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    """Test whether a degradation lowered the per-query metric.

    One-sided paired bootstrap on the per-query score deltas: resamples
    queries with replacement and reports the fraction of bootstrap mean
    deltas that are non-positive. The benchmark "detects" the regression
    when ``delta > 0`` and ``p < 0.05``.

    Reference:
        Sakai, "Evaluating Evaluation Metrics based on the Bootstrap",
        SIGIR 2006.

    Args:
        scores_base: Shape ``(Q,)`` per-query scores of the intact system.
        scores_degraded: Shape ``(Q,)`` paired scores of the degraded system.
        n_boot: Number of bootstrap resamples.
        seed: Seed for the bootstrap RNG.

    Returns:
        Tuple ``(mean_delta, p_value)`` where ``mean_delta`` is the mean of
        ``scores_base - scores_degraded``.
    """
    base = np.asarray(scores_base, dtype=np.float64)
    degraded = np.asarray(scores_degraded, dtype=np.float64)
    delta = base - degraded
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(delta), size=(n_boot, len(delta)))
    boot = delta[idx].mean(axis=1)
    p = float((boot <= 0).mean())
    return float(delta.mean()), p


def drop_index_mask(n_chunks: int, frac: float, rng: np.random.Generator) -> NDArray[np.bool_]:
    """Build a boolean mask marking a random fraction of chunks as dropped.

    Simulates index loss (the "10% of the index silently deleted"
    positive control): exactly ``int(frac * n_chunks)`` entries are True,
    chosen uniformly without replacement.

    Reference:
        Positive-control degradation for discriminative-power evaluation
        (Sakai, SIGIR 2006 tradition).

    Args:
        n_chunks: Total number of indexed chunks.
        frac: Fraction of chunks to drop, in ``[0, 1]``.
        rng: Source of randomness (pass ``Resources.rng(name)``).

    Returns:
        Shape ``(n_chunks,)`` boolean mask; True marks a dropped chunk.
    """
    n_drop = int(frac * n_chunks)  # truncation, matching prototype L865
    mask = np.zeros(n_chunks, dtype=np.bool_)
    if n_drop > 0:
        mask[rng.choice(n_chunks, size=n_drop, replace=False)] = True
    return mask


def noise_transform(d: int, sigma: float, rng: np.random.Generator) -> NDArray[np.float64]:
    """Build a noisy embedding transform ``I + sigma * G / sqrt(d)``.

    ``G`` has i.i.d. standard-normal entries; the ``1/sqrt(d)`` scaling
    keeps the perturbation's spectral norm roughly ``sigma`` independent of
    dimension, so a fixed sigma degrades retrieval comparably across d.
    Apply the matrix to both query and chunk embeddings (prototype
    L866-868 pattern).

    Reference:
        Positive-control degradation for discriminative-power evaluation
        (Sakai, SIGIR 2006 tradition).

    Args:
        d: Embedding dimensionality.
        sigma: Noise magnitude; ``0.0`` returns the exact identity.
        rng: Source of randomness (pass ``Resources.rng(name)``).

    Returns:
        Shape ``(d, d)`` transform matrix.
    """
    noise: NDArray[np.float64] = rng.standard_normal((d, d))
    return np.asarray(np.eye(d, dtype=np.float64) + sigma * noise / np.sqrt(d), dtype=np.float64)
