"""Efficiency layer: importance weights, ESS, post-stratification, coverage.

Port of the prototype's efficiency block (reference/synth_query_eval.py
L329-367) plus the coverage trio from the PLAN §3 contract. The reporting
rule (SPEC §8-9): always show BOTH the demand-weighted headline + ESS and
the unweighted per-cluster table -- never a single blended average.

References:
    Kong (1992) for the ESS identity; Chatterjee & Diaconis (2018) for the
    importance-sampling cost bound; BCG (arXiv 2510.00001) and "Coverage,
    Not Averages" (arXiv 2604.20763) for the coverage floors.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

_EPS = 1e-12


def cluster_importance_weights(
    labels_synth: NDArray[np.int_], p_hat: NDArray[np.float64]
) -> tuple[NDArray[np.float64], float]:
    """Per-sample weights ``w_i = p_hat[c(i)] / q[c(i)]`` and the coverage gap.

    ``q`` is the empirical cluster distribution of the synthetic set.
    Clusters with zero synthetic samples contribute to ``coverage_gap`` (the
    share of production demand the set cannot speak for); ``p_hat`` is
    renormalized over covered clusters for the weights.

    Args:
        labels_synth: Reference-partition labels of the synthetic queries.
        p_hat: Demand distribution over the partition's clusters.

    Returns:
        Tuple of (per-sample weights, coverage gap in [0, 1]).

    Reference:
        Kong (1992); prototype L329-347.
    """
    labels = np.asarray(labels_synth, dtype=np.int_)
    demand = np.asarray(p_hat, dtype=np.float64)
    n_clusters = len(demand)
    counts = np.bincount(labels, minlength=n_clusters).astype(np.float64)
    q = counts / counts.sum()
    covered = q > 0
    coverage_gap = float(demand[~covered].sum())
    p_cov = np.where(covered, demand, 0.0)
    p_cov = p_cov / max(p_cov.sum(), _EPS)
    weights = np.where(q[labels] > 0, p_cov[labels] / np.maximum(q[labels], _EPS), 0.0)
    return np.asarray(weights, dtype=np.float64), coverage_gap


def effective_sample_size(weights: NDArray[np.float64]) -> float:
    """ESS = (sum w)^2 / sum w^2; equals N iff all weights are equal.

    The identity ``variance inflation = n / ESS`` (Kong, 1992) is why we
    match the demand at the source instead of reweighting after the fact
    (SPEC §16 theory anchors).

    Args:
        weights: Per-sample importance weights.

    Returns:
        The effective sample size in [1, len(weights)] (0-safe).

    Reference:
        Kong (1992); prototype L350-353.
    """
    w = np.asarray(weights, dtype=np.float64)
    s1 = float(w.sum())
    s2 = float((w**2).sum())
    return s1 * s1 / max(s2, _EPS)


def post_stratified_estimate(
    per_query_metric: NDArray[np.float64],
    labels: NDArray[np.int_],
    p_hat: NDArray[np.float64],
) -> float:
    """Demand-weighted headline, renormalized over covered clusters.

    ``M_hat = sum_c p_hat_c * mean(M | cluster c) / sum_covered p_hat_c``.

    Args:
        per_query_metric: One score per synthetic query.
        labels: Reference-partition labels aligned with the metric.
        p_hat: Demand distribution over the partition's clusters.

    Returns:
        The post-stratified estimate.

    Reference:
        Classical survey post-stratification (Holt & Smith, JRSS A 1979);
        residual analysis in SPEC §16; prototype L356-367.
    """
    metric = np.asarray(per_query_metric, dtype=np.float64)
    labels_arr = np.asarray(labels, dtype=np.int_)
    demand = np.asarray(p_hat, dtype=np.float64)
    total, mass = 0.0, 0.0
    for cluster in range(len(demand)):
        mask = labels_arr == cluster
        if mask.any():
            total += float(demand[cluster]) * float(metric[mask].mean())
            mass += float(demand[cluster])
    return total / max(mass, _EPS)


def demand_weighted_coverage(labels_synth: NDArray[np.int_], p_hat: NDArray[np.float64]) -> float:
    """Share of production demand in clusters with at least one synthetic query.

    ``sum_c p_hat_c * 1[cluster c covered]`` (BCG arXiv 2510.00001).

    Args:
        labels_synth: Reference-partition labels of the synthetic queries.
        p_hat: Demand distribution over the partition's clusters.

    Returns:
        Coverage in [0, 1]; ``1 - coverage_gap``.
    """
    return minimum_semantic_coverage(labels_synth, p_hat, floor=1)


def zero_query_clusters(labels_synth: NDArray[np.int_], n_clusters: int) -> list[int]:
    """Cluster ids with zero synthetic queries, sorted ascending.

    The worst-k companion of the dual-view reporting rule ("Coverage, Not
    Averages", arXiv 2604.20763).

    Args:
        labels_synth: Reference-partition labels of the synthetic queries.
        n_clusters: Total number of clusters in the partition.

    Returns:
        Sorted list of uncovered cluster ids.
    """
    counts = np.bincount(np.asarray(labels_synth, dtype=np.int_), minlength=n_clusters)
    return [int(c) for c in np.flatnonzero(counts == 0)]


def minimum_semantic_coverage(
    labels_synth: NDArray[np.int_], p_hat: NDArray[np.float64], floor: int
) -> float:
    """Demand mass in clusters holding at least ``floor`` synthetic queries.

    A per-cluster sample floor turns coverage into a minimum-support
    guarantee (BCG arXiv 2510.00001; "Coverage, Not Averages",
    arXiv 2604.20763); ``floor=1`` reduces to
    :func:`demand_weighted_coverage`.

    Args:
        labels_synth: Reference-partition labels of the synthetic queries.
        p_hat: Demand distribution over the partition's clusters.
        floor: Minimum synthetic-sample count for a cluster to count.

    Returns:
        Covered demand mass in [0, 1].
    """
    demand = np.asarray(p_hat, dtype=np.float64)
    counts = np.bincount(np.asarray(labels_synth, dtype=np.int_), minlength=len(demand))
    return float(demand[counts >= floor].sum())
