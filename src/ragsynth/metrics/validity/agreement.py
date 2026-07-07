"""Ranking-agreement metrics between a synthetic arm and the anchor set.

Given per-query scores of a fixed zoo of retrieval systems on two query
sets (the real *anchor* queries and one synthetic *arm*), these functions
measure whether the synthetic benchmark ranks the systems the same way the
real one does -- the criterion validity meter of SPEC §8-9. Ported from the
frozen prototype (``reference/synth_query_eval.py`` L470-540) with math
kept semantically identical.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import kendalltau

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

__all__ = [
    "RankingAgreement",
    "ranking_agreement",
    "rbo_ext",
    "system_ranking",
    "tau_ap",
]

_MIN_RANKING_LEN = 2
_CI_PERCENTILES = (2.5, 97.5)
_DEFAULT_RBO_P = 0.9


def system_ranking(mean_scores: NDArray[np.float64]) -> list[int]:
    """Order system indices best-first by mean score.

    Args:
        mean_scores: Shape ``(S,)`` mean per-system scores.

    Returns:
        System indices sorted by descending score.
    """
    scores = np.asarray(mean_scores, dtype=np.float64)
    return [int(i) for i in np.argsort(-scores)]


def tau_ap(reference: Sequence[int], candidate: Sequence[int]) -> float:
    """Compute tau_AP, the top-weighted rank correlation of two rankings.

    For each candidate item below the top, counts the fraction of items
    ranked above it that the reference also ranks above it, then averages
    and rescales to ``[-1, 1]``. Errors near the top are penalized more
    than errors near the bottom.

    Reference:
        Yilmaz, Aslam & Robertson, "A New Rank Correlation Coefficient for
        Information Retrieval", SIGIR 2008.

    Args:
        reference: Reference ranking, best-first (permutation of items).
        candidate: Candidate ranking, best-first (same items).

    Returns:
        tau_AP in ``[-1, 1]``; ``1.0`` for rankings shorter than two items.
    """
    ref_pos = {item: i for i, item in enumerate(reference)}
    n = len(candidate)
    if n < _MIN_RANKING_LEN:
        return 1.0
    total = 0.0
    for i in range(1, n):
        item = candidate[i]
        concordant = sum(1 for j in range(i) if ref_pos[candidate[j]] < ref_pos[item])
        total += concordant / i
    return 2.0 * total / (n - 1) - 1.0


def rbo_ext(s: Sequence[int], t: Sequence[int], p: float = _DEFAULT_RBO_P) -> float:
    """Compute extrapolated Rank-Biased Overlap of two rankings.

    Top-weighted set-overlap of the two rankings' prefixes, geometrically
    discounted by depth with persistence ``p``, extrapolated to infinite
    depth from the deepest common prefix.

    Reference:
        Webber, Moffat & Zobel, "A Similarity Measure for Indefinite
        Rankings", ACM TOIS 2010.

    Args:
        s: First ranking, best-first.
        t: Second ranking, best-first.
        p: Persistence parameter in ``(0, 1)``; higher weights deeper ranks.

    Returns:
        RBO_ext in ``[0, 1]``; ``1.0`` for identical rankings.
    """
    k = min(len(s), len(t))
    seen_s: set[int] = set()
    seen_t: set[int] = set()
    inter = 0
    acc = 0.0
    for depth in range(1, k + 1):
        a, b = s[depth - 1], t[depth - 1]
        if a == b:
            inter += 1
        else:
            if a in seen_t:
                inter += 1
            if b in seen_s:
                inter += 1
        seen_s.add(a)
        seen_t.add(b)
        acc += (p**depth) * (inter / depth)
    return (1.0 - p) / p * acc + (inter / k) * (p**k)


@dataclass(frozen=True)
class RankingAgreement:
    """Agreement of an arm's system ranking with the anchor ranking.

    Attributes:
        tau: Kendall tau between anchor and arm mean scores (Kendall 1938).
        tau_ap_: Top-weighted tau_AP of the two best-first rankings.
        rbo: Extrapolated Rank-Biased Overlap of the two rankings.
        tau_ci_low: 2.5th percentile of the bootstrap tau distribution.
        tau_ci_high: 97.5th percentile of the bootstrap tau distribution.
    """

    tau: float
    tau_ap_: float
    rbo: float
    tau_ci_low: float
    tau_ci_high: float


def ranking_agreement(
    anchor_scores: NDArray[np.float64],
    arm_scores: NDArray[np.float64],
    n_boot: int = 1000,
    seed: int = 0,
) -> RankingAgreement:
    """Measure how well the arm's system ranking matches the anchor's.

    Computes Kendall tau, tau_AP and RBO between the system rankings induced
    by mean per-query scores, plus a percentile bootstrap CI on tau obtained
    by resampling the arm's queries (columns) with replacement.

    References:
        Kendall, "A New Measure of Rank Correlation", Biometrika 1938;
        Yilmaz, Aslam & Robertson, SIGIR 2008 (tau_AP);
        Webber, Moffat & Zobel, ACM TOIS 2010 (RBO).

    Args:
        anchor_scores: Shape ``(S, Q_anchor)`` per-query scores of the S
            systems on the real anchor query set.
        arm_scores: Shape ``(S, Q_arm)`` per-query scores of the same
            systems on the synthetic arm's query set.
        n_boot: Number of bootstrap resamples of the arm's queries.
        seed: Seed for the bootstrap RNG.

    Returns:
        A :class:`RankingAgreement` with point estimates and the tau CI.
    """
    anchor = np.asarray(anchor_scores, dtype=np.float64)
    arm = np.asarray(arm_scores, dtype=np.float64)
    anchor_means = anchor.mean(axis=1)
    arm_means = arm.mean(axis=1)
    ref, cand = system_ranking(anchor_means), system_ranking(arm_means)

    tau = float(kendalltau(anchor_means, arm_means).statistic)
    rng = np.random.default_rng(seed)
    q = arm.shape[1]
    idx = rng.integers(0, q, size=(n_boot, q))
    boot_means = arm[:, idx].mean(axis=2)  # (S, n_boot)
    taus = np.array(
        [float(kendalltau(anchor_means, boot_means[:, b]).statistic) for b in range(n_boot)]
    )
    lo, hi = np.percentile(taus, _CI_PERCENTILES)
    return RankingAgreement(
        tau=tau,
        tau_ap_=tau_ap(ref, cand),
        rbo=rbo_ext(ref, cand),
        tau_ci_low=float(lo),
        tau_ci_high=float(hi),
    )
