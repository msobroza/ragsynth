"""Fidelity metrics: does the synthetic query cloud match real traffic?

Ports of the vendored prototype (``reference/synth_query_eval.py`` L375-462),
math unchanged. The real reference is the anchor embedding set (equal-n
subsample, SPEC §8):

- :func:`kl_similarity_distributions` — the Chroma representativeness monitor
  (Chroma Generative Benchmarking, 2025).
- :func:`c2st_auc` / :func:`c2st_auc_with_coefs` — classifier two-sample test
  (Lopez-Paz & Oquab, ICLR 2017), with the "what does the discriminator use"
  coefficient diagnostic.
- :func:`mmd_rbf` — unbiased squared MMD (Gretton et al., JMLR 2012).
- :func:`within_cluster_c2st` — the A2 mechanism meter: C2ST inside each
  reference-partition cluster.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

if TYPE_CHECKING:
    from numpy.typing import NDArray

_EDGE_PAD = 1e-9
"""Padding on the top histogram edge so the max sample falls inside the last bin."""

_LOGREG_MAX_ITER = 1000
"""Iteration budget for the C2ST logistic discriminator (prototype L405)."""

_MEDIAN_HEURISTIC_MAX = 500
"""Subsample size for the median-heuristic bandwidth estimate (prototype L425)."""

_MIN_BANDWIDTH_SQ_DIST = 1e-12
"""Floor on the median squared distance to keep gamma finite (prototype L428)."""

_WITHIN_CLUSTER_SPLITS = 3
"""CV folds for the per-cluster C2ST, where per-side n is small (prototype L460)."""


def kl_similarity_distributions(
    real_q: NDArray[np.float64],
    synth_q: NDArray[np.float64],
    chunk_emb: NDArray[np.float64],
    bins: int = 50,
    eps: float = 1e-6,
) -> float:
    """KL(real || synth) between query -> top-1-chunk cosine distributions.

    The Chroma 2025 representativeness monitor (Chroma Generative
    Benchmarking, 2025): histogram each side's top-1 query-to-chunk cosine
    over shared bin edges, epsilon-smooth, and take the KL divergence.
    Steered target on real data is roughly <= 0.16 — a reference band, not a
    hard gate (SPEC §8). Port of prototype L375-392.

    Args:
        real_q: Real query embeddings, shape ``(n_real, d)``, L2-normalized.
        synth_q: Synthetic query embeddings, shape ``(n_synth, d)``,
            L2-normalized.
        chunk_emb: Chunk embeddings, shape ``(n_chunks, d)``, L2-normalized.
        bins: Number of shared histogram bins.
        eps: Additive smoothing applied to every bin count before
            normalization.

    Returns:
        KL divergence in nats; ~0 when the two profiles match.
    """
    real = np.asarray(real_q, dtype=np.float64)
    synth = np.asarray(synth_q, dtype=np.float64)
    chunks = np.asarray(chunk_emb, dtype=np.float64)
    s_real = (real @ chunks.T).max(axis=1)
    s_synth = (synth @ chunks.T).max(axis=1)
    lo = float(min(s_real.min(), s_synth.min()))
    hi = float(max(s_real.max(), s_synth.max())) + _EDGE_PAD
    edges = np.linspace(lo, hi, bins + 1)
    p = np.histogram(s_real, bins=edges)[0].astype(np.float64) + eps
    q = np.histogram(s_synth, bins=edges)[0].astype(np.float64) + eps
    p /= p.sum()
    q /= q.sum()
    return float(np.sum(p * np.log(p / q)))


def c2st_auc(
    x_real: NDArray[np.float64],
    x_synth: NDArray[np.float64],
    n_splits: int = 5,
    seed: int = 0,
) -> float:
    """Classifier two-sample test (Lopez-Paz & Oquab, ICLR 2017).

    Cross-validated ROC-AUC of a logistic regression separating real from
    synthetic embeddings. 0.5 = indistinguishable; higher = detectable gap.
    Port of prototype L395-407.

    Args:
        x_real: Real embeddings, shape ``(n_real, d)``.
        x_synth: Synthetic embeddings, shape ``(n_synth, d)``.
        n_splits: Stratified CV folds.
        seed: Shuffle seed for the CV splitter.

    Returns:
        Mean ROC-AUC over the CV folds.
    """
    xr = np.asarray(x_real, dtype=np.float64)
    xs = np.asarray(x_synth, dtype=np.float64)
    x = np.vstack([xr, xs])
    y = np.concatenate([np.zeros(len(xr)), np.ones(len(xs))])
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=_LOGREG_MAX_ITER))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = cross_val_score(clf, x, y, cv=cv, scoring="roc_auc")
    return float(np.asarray(scores, dtype=np.float64).mean())


def c2st_auc_with_coefs(
    x_real: NDArray[np.float64],
    x_synth: NDArray[np.float64],
    seed: int = 0,
) -> tuple[float, NDArray[np.float64]]:
    """C2ST AUC plus the discriminator's standardized coefficients.

    The "what does the discriminator use" diagnostic (SPEC §8): the AUC comes
    from :func:`c2st_auc` (cross-validated, Lopez-Paz & Oquab, ICLR 2017);
    the coefficients come from a StandardScaler + LogisticRegression refit on
    ALL data, so ``|coef[j]|`` ranks embedding dimensions by how much the
    discriminator leans on them.

    Args:
        x_real: Real embeddings, shape ``(n_real, d)``.
        x_synth: Synthetic embeddings, shape ``(n_synth, d)``.
        seed: Shuffle seed forwarded to :func:`c2st_auc`.

    Returns:
        Tuple of (cross-validated AUC, coefficient vector of shape ``(d,)``
        in standardized-feature space).
    """
    auc = c2st_auc(x_real, x_synth, seed=seed)
    xr = np.asarray(x_real, dtype=np.float64)
    xs = np.asarray(x_synth, dtype=np.float64)
    x = np.vstack([xr, xs])
    y = np.concatenate([np.zeros(len(xr)), np.ones(len(xs))])
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=_LOGREG_MAX_ITER))
    model.fit(x, y)
    coefs = np.asarray(model[-1].coef_, dtype=np.float64).ravel()
    return auc, coefs


def mmd_rbf(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    gamma: float | None = None,
    max_n: int = 2000,
    seed: int = 0,
) -> float:
    """Unbiased squared MMD with an RBF kernel (Gretton et al., JMLR 2012).

    Median-heuristic bandwidth over a pooled subsample; each side is
    subsampled to ``max_n`` first; the unbiased estimate is clamped at 0.
    Port of prototype L410-436.

    Args:
        x: First sample, shape ``(n, d)``.
        y: Second sample, shape ``(m, d)``.
        gamma: RBF kernel coefficient; ``None`` selects the median heuristic
            ``1 / median(squared distances)``.
        max_n: Per-side subsample cap.
        seed: Seed for the subsampling RNG.

    Returns:
        Unbiased squared MMD estimate, clamped to be >= 0. Exactly 0.0 when
        ``x`` and ``y`` are the same sample.
    """
    rng = np.random.default_rng(seed)
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if len(xa) > max_n:
        xa = xa[rng.choice(len(xa), max_n, replace=False)]
    if len(ya) > max_n:
        ya = ya[rng.choice(len(ya), max_n, replace=False)]
    if gamma is None:
        pool = np.vstack([xa, ya])
        sub = pool[rng.choice(len(pool), min(len(pool), _MEDIAN_HEURISTIC_MAX), replace=False)]
        d2 = pairwise_distances(sub, metric="sqeuclidean")
        med = float(np.median(d2[d2 > 0]))
        gamma = 1.0 / max(med, _MIN_BANDWIDTH_SQ_DIST)
    kxx = rbf_kernel(xa, xa, gamma)
    kyy = rbf_kernel(ya, ya, gamma)
    kxy = rbf_kernel(xa, ya, gamma)
    n, m = len(xa), len(ya)
    mmd2 = (
        (kxx.sum() - np.trace(kxx)) / (n * (n - 1))
        + (kyy.sum() - np.trace(kyy)) / (m * (m - 1))
        - 2.0 * kxy.mean()
    )
    return float(max(mmd2, 0.0))


def within_cluster_c2st(
    x_real: NDArray[np.float64],
    x_synth: NDArray[np.float64],
    labels_real: NDArray[np.int_],
    labels_synth: NDArray[np.int_],
    min_per_side: int = 30,
    seed: int = 0,
) -> tuple[float, dict[int, float]]:
    """Mean and per-cluster C2ST AUC inside each reference cluster.

    The diagnostic that isolates within-cluster shape mismatch — exactly the
    mechanism the A2 arm claims to improve; matched cluster *marginals* (as
    in A1) contribute nothing here (SPEC §8). Clusters with fewer than
    ``min_per_side`` members on either side are skipped; both sides are
    subsampled to equal n before the test. C2ST per Lopez-Paz & Oquab
    (ICLR 2017). Port of prototype L439-462.

    Args:
        x_real: Real embeddings, shape ``(n_real, d)``.
        x_synth: Synthetic embeddings, shape ``(n_synth, d)``.
        labels_real: Reference-partition labels for ``x_real``, shape
            ``(n_real,)``.
        labels_synth: Reference-partition labels for ``x_synth``, shape
            ``(n_synth,)``.
        min_per_side: Minimum members per side for a cluster to be scored.
        seed: Base seed; each cluster subsamples with ``seed + cluster_id``.

    Returns:
        Tuple of (mean AUC over scored clusters — NaN when none qualify,
        mapping of cluster id -> AUC for scored clusters only).
    """
    xr_all = np.asarray(x_real, dtype=np.float64)
    xs_all = np.asarray(x_synth, dtype=np.float64)
    lr = np.asarray(labels_real)
    ls = np.asarray(labels_synth)
    per: dict[int, float] = {}
    for c in np.unique(lr):
        xr = xr_all[lr == c]
        xs = xs_all[ls == c]
        if len(xr) >= min_per_side and len(xs) >= min_per_side:
            n = min(len(xr), len(xs))
            rng = np.random.default_rng(seed + int(c))
            xr = xr[rng.choice(len(xr), n, replace=False)]
            xs = xs[rng.choice(len(xs), n, replace=False)]
            per[int(c)] = c2st_auc(xr, xs, n_splits=_WITHIN_CLUSTER_SPLITS, seed=seed)
    mean = float(np.mean(list(per.values()))) if per else float("nan")
    return mean, per
