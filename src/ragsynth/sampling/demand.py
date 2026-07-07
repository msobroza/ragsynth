"""Demand map, coverage tilting, and the on-manifold radius.

Port of the vendored prototype ``reference/synth_query_eval.py`` (L252-L286,
SPEC §7.3/§7.5): per-component demand from movMF responsibilities with
optional exponential time decay, the coverage-guaranteeing weight tilt
``pi' ~ lambda * p_hat + (1 - lambda)/C``, and the nearest-neighbour cosine
threshold used by the spec sampler's rejection guard.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.neighbors import NearestNeighbors

if TYPE_CHECKING:
    from numpy.typing import NDArray


def demand_from_responsibilities(
    resp: NDArray[np.float64],
    timestamps: NDArray[np.float64] | None = None,
    half_life: float | None = None,
) -> NDArray[np.float64]:
    """Estimate per-component demand ``p_hat``, optionally time-decayed.

    ``p_hat_c ~ sum_i exp(-(t_now - t_i) * ln2 / half_life) * gamma_c(q_i)``
    where ``t_now`` is the newest timestamp observed (prototype L252).

    Args:
        resp: Responsibility matrix ``(n_queries, K)`` from
            :meth:`MovMF.responsibilities`.
        timestamps: Optional per-query timestamps ``(n_queries,)``. Any
            monotone time unit works (epoch seconds, days, ...).
        half_life: Optional decay half-life, expressed in the **same time
            unit as** ``timestamps`` (e.g. if timestamps are epoch seconds,
            a 7-day half-life is ``7 * 86400``). A query older than the
            newest one by one half-life counts half as much. Decay applies
            only when both ``timestamps`` and ``half_life`` are given.

    Returns:
        Demand vector ``(K,)`` summing to 1.
    """
    resp64 = np.asarray(resp, dtype=np.float64)
    if timestamps is None or half_life is None:
        w = np.ones(resp64.shape[0])
    else:
        ts = np.asarray(timestamps, dtype=np.float64)
        age = ts.max() - ts
        w = np.exp(-age * np.log(2.0) / half_life)
    p_hat = (w[:, None] * resp64).sum(axis=0)
    return np.asarray(p_hat / p_hat.sum(), dtype=np.float64)


def tilt_weights(p_hat: NDArray[np.float64], lam: float) -> NDArray[np.float64]:
    """Tilt demand toward coverage: ``pi'_c ~ lam * p_hat_c + (1 - lam)/C``.

    The uniform mixture component guarantees every cluster keeps sampling
    mass even when observed demand is zero (prototype L270, SPEC §7.3).

    Args:
        p_hat: Demand vector ``(C,)``.
        lam: Demand weight in ``[0, 1]``; ``lam=1`` is pure demand,
            ``lam=0`` is uniform.

    Returns:
        Tilted weight vector ``(C,)`` summing to 1.
    """
    p64 = np.asarray(p_hat, dtype=np.float64)
    c = len(p64)
    w = lam * p64 + (1.0 - lam) / c
    return np.asarray(w / w.sum(), dtype=np.float64)


def nn_cos_threshold(prod_emb: NDArray[np.float64], pct: float = 5.0) -> float:
    """Estimate the on-manifold radius ``tau_r`` from production traffic.

    ``tau_r`` is the ``pct``-th percentile of each production query's
    nearest-neighbour cosine (self excluded). A sampled ``z`` is considered
    on-manifold if its nearest production query is at least this similar —
    i.e. no farther than the sparsest ``pct``% of real traffic is from its
    own neighbourhood (prototype L277, SPEC §7.5).

    Args:
        prod_emb: L2-normalized production query embeddings ``(n, d)``.
        pct: Percentile of the NN-cosine distribution to return.

    Returns:
        The cosine threshold ``tau_r``.
    """
    prod64 = np.asarray(prod_emb, dtype=np.float64)
    nn = NearestNeighbors(n_neighbors=2, metric="cosine").fit(prod64)
    dist, _ = nn.kneighbors(prod64)
    return float(np.percentile(1.0 - dist[:, 1], pct))
