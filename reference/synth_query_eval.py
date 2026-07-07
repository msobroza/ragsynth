"""Empirical test harness for synthetic query generation strategies.

Implements the A0 / A1 / A2 / A_oracle comparison discussed for validating a
spec-first ("z-sampling") synthetic query generator against quota-based and
naive baselines, on any corpus where you have:

  * chunk embeddings            (the knowledge base, L2-normalised)
  * production query embeddings (real traffic, L2-normalised)
  * a held-out anchor set       (real queries + gold chunk ids)
  * a system zoo                (>= 8 retriever variants to rank)

Arms
----
  A0      naive chunk-first generation (uniform seeds, no steering)
  A1      cluster-quota chunk-first (lambda-mixture allocation + exemplars)
  A2      spec-first: sample z from a demand-tilted movMF fitted on
          production queries, condition on kNN chunks/queries around z
  ORACLE  random subsample of real held-out queries (the ceiling)

Metric stack
------------
  Fidelity   : KL on query->chunk cosine-similarity distributions (Chroma 2025),
               classifier two-sample test AUC (Lopez-Paz & Oquab, ICLR 2017),
               MMD with RBF kernel (Gretton et al., JMLR 2012),
               within-cluster C2ST (the mechanism-specific diagnostic).
  Efficiency : per-cluster importance weights, ESS, coverage gap,
               post-stratified headline estimate.
  Validity   : Kendall tau (scipy), tau_AP (Yilmaz et al., SIGIR 2008),
               RBO (Webber et al., TOIS 2010), bootstrap CIs,
               positive controls via paired bootstrap (Sakai, SIGIR 2006).

Statistical core
----------------
  movMF mixture (Banerjee et al., JMLR 2005) with EM on the unit sphere,
  Wood (1994) rejection sampler for von Mises-Fisher draws,
  demand tilting  pi' ~ lambda * p_hat + (1 - lambda)/C,
  on-manifold guard for sampled specs.

Dependencies: numpy, scipy, scikit-learn (all BSD -- air-gap friendly).
Run ``python synth_query_eval.py`` for a fully synthetic end-to-end demo that
exercises every component and prints the comparison table. Plug real data and
a real LLM through the Protocols in the ``LLM integration`` section.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence

import numpy as np
from numpy.random import Generator
from scipy.special import gammaln, ive, logsumexp
from scipy.stats import kendalltau
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------
# Sphere utilities
# --------------------------------------------------------------------------


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    """Project vectors onto the unit sphere."""
    return x / np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), eps)


def sphere_uniform(n: int, d: int, rng: Generator) -> np.ndarray:
    """n uniform samples on S^{d-1}."""
    return l2_normalize(rng.standard_normal((n, d)))


def log_sphere_area(d: int) -> float:
    """log surface area of S^{d-1}: 2 pi^{d/2} / Gamma(d/2)."""
    return np.log(2.0) + (d / 2.0) * np.log(np.pi) - gammaln(d / 2.0)


# --------------------------------------------------------------------------
# von Mises-Fisher sampling (Wood, 1994) and log-density
# --------------------------------------------------------------------------


def sample_vmf(mu: np.ndarray, kappa: float, n: int, rng: Generator) -> np.ndarray:
    """Draw n samples from vMF(mu, kappa) on S^{d-1}.

    Rejection sampler of Wood (1994); Householder reflection maps the
    canonical pole e1 onto mu.
    """
    d = mu.shape[0]
    if kappa < 1e-8:
        return sphere_uniform(n, d, rng)

    b = (-2.0 * kappa + np.sqrt(4.0 * kappa**2 + (d - 1.0) ** 2)) / (d - 1.0)
    x0 = (1.0 - b) / (1.0 + b)
    c = kappa * x0 + (d - 1.0) * np.log(1.0 - x0**2)

    w = np.empty(n)
    for i in range(n):
        while True:
            z = rng.beta((d - 1.0) / 2.0, (d - 1.0) / 2.0)
            wi = (1.0 - (1.0 + b) * z) / (1.0 - (1.0 - b) * z)
            u = rng.uniform()
            if kappa * wi + (d - 1.0) * np.log(1.0 - x0 * wi) - c >= np.log(u):
                w[i] = wi
                break

    v = l2_normalize(rng.standard_normal((n, d - 1)))
    x = np.concatenate([w[:, None], np.sqrt(np.maximum(1.0 - w**2, 0.0))[:, None] * v], axis=1)

    e1 = np.zeros(d)
    e1[0] = 1.0
    u_vec = e1 - mu
    nu = np.linalg.norm(u_vec)
    if nu < 1e-12:  # mu is already the pole
        return x
    u_vec = u_vec / nu
    return x - 2.0 * np.outer(x @ u_vec, u_vec)  # Householder: e1 -> mu


def vmf_log_norm_const(d: int, kappa: np.ndarray) -> np.ndarray:
    """log C_d(kappa) with the exponentially scaled Bessel for stability.

    C_d(k) = k^{d/2-1} / ((2 pi)^{d/2} I_{d/2-1}(k));
    log I_v(k) = log(ive(v, k)) + k.
    """
    kappa = np.asarray(kappa, dtype=float)
    v = d / 2.0 - 1.0
    small = kappa < 1e-8
    log_iv = np.log(np.maximum(ive(v, np.maximum(kappa, 1e-8)), 1e-300)) + kappa
    out = v * np.log(np.maximum(kappa, 1e-8)) - (d / 2.0) * np.log(2.0 * np.pi) - log_iv
    if np.any(small):  # kappa -> 0: uniform density on the sphere
        out = np.where(small, -log_sphere_area(d), out)
    return out


# --------------------------------------------------------------------------
# movMF mixture with EM (Banerjee et al., JMLR 2005)
# --------------------------------------------------------------------------


@dataclass
class MovMF:
    """Mixture of von Mises-Fisher distributions on the unit sphere.

    Attributes after fit: weights_ (K,), means_ (K, d), kappas_ (K,).
    """

    n_components: int
    max_iter: int = 200
    tol: float = 1e-6
    kappa_min: float = 1e-2
    kappa_max: float = 1e5
    seed: int = 0

    weights_: np.ndarray = field(default=None, repr=False)
    means_: np.ndarray = field(default=None, repr=False)
    kappas_: np.ndarray = field(default=None, repr=False)
    log_likelihood_: float = field(default=np.nan, repr=False)

    # -- internals ----------------------------------------------------------

    def _component_log_pdf(self, x: np.ndarray) -> np.ndarray:
        """(n, K) matrix of log f(x | mu_k, kappa_k)."""
        d = x.shape[1]
        log_c = vmf_log_norm_const(d, self.kappas_)  # (K,)
        return log_c[None, :] + (x @ self.means_.T) * self.kappas_[None, :]

    @staticmethod
    def _kappa_banerjee(rbar: np.ndarray, d: int) -> np.ndarray:
        """Banerjee et al. approximation kappa = rbar (d - rbar^2)/(1 - rbar^2)."""
        rbar = np.clip(rbar, 1e-6, 1.0 - 1e-6)
        return rbar * (d - rbar**2) / (1.0 - rbar**2)

    # -- API ----------------------------------------------------------------

    def fit(self, x: np.ndarray) -> "MovMF":
        x = l2_normalize(np.asarray(x, dtype=float))
        n, d = x.shape
        k = self.n_components

        km = KMeans(n_clusters=k, n_init=10, random_state=self.seed).fit(x)
        self.means_ = l2_normalize(km.cluster_centers_)
        self.weights_ = np.bincount(km.labels_, minlength=k).astype(float) / n
        rbar0 = np.array(
            [
                np.linalg.norm(x[km.labels_ == c].sum(axis=0))
                / max((km.labels_ == c).sum(), 1)
                for c in range(k)
            ]
        )
        self.kappas_ = np.clip(self._kappa_banerjee(rbar0, d), self.kappa_min, self.kappa_max)

        prev_ll = -np.inf
        for _ in range(self.max_iter):
            log_joint = np.log(np.maximum(self.weights_, 1e-300))[None, :] + self._component_log_pdf(x)
            log_norm = logsumexp(log_joint, axis=1)  # (n,)
            ll = float(log_norm.mean())
            resp = np.exp(log_joint - log_norm[:, None])  # (n, K)

            nk = resp.sum(axis=0) + 1e-12
            self.weights_ = nk / n
            r = resp.T @ x  # (K, d)
            norms = np.linalg.norm(r, axis=1)
            self.means_ = l2_normalize(np.where(norms[:, None] > 1e-12, r, self.means_))
            rbar = norms / nk
            self.kappas_ = np.clip(self._kappa_banerjee(rbar, d), self.kappa_min, self.kappa_max)

            if abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll

        self.log_likelihood_ = prev_ll
        return self

    def responsibilities(self, x: np.ndarray) -> np.ndarray:
        """(n, K) soft assignments gamma_c(x)."""
        x = l2_normalize(np.asarray(x, dtype=float))
        log_joint = np.log(np.maximum(self.weights_, 1e-300))[None, :] + self._component_log_pdf(x)
        return np.exp(log_joint - logsumexp(log_joint, axis=1)[:, None])

    def log_prob(self, x: np.ndarray) -> np.ndarray:
        x = l2_normalize(np.asarray(x, dtype=float))
        log_joint = np.log(np.maximum(self.weights_, 1e-300))[None, :] + self._component_log_pdf(x)
        return logsumexp(log_joint, axis=1)

    def sample(self, n: int, rng: Generator, weights: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Ancestral sampling: returns (samples (n, d), component ids (n,)).

        ``weights`` overrides the fitted mixing weights -- pass the tilted
        pi' here to sample from the demand-tilted mixture.
        """
        w = self.weights_ if weights is None else np.asarray(weights, dtype=float)
        w = w / w.sum()
        comps = rng.choice(len(w), size=n, p=w)
        d = self.means_.shape[1]
        out = np.empty((n, d))
        for c in np.unique(comps):
            idx = np.where(comps == c)[0]
            out[idx] = sample_vmf(self.means_[c], float(self.kappas_[c]), len(idx), rng)
        return out, comps


# --------------------------------------------------------------------------
# Demand map, tilting, and the spec sampler (the A2 core)
# --------------------------------------------------------------------------


def demand_from_responsibilities(
    resp: np.ndarray,
    timestamps: np.ndarray | None = None,
    half_life: float | None = None,
) -> np.ndarray:
    """Per-component demand p_hat, optionally with exponential time decay.

    p_hat_c ~ sum_i  exp(-(t_now - t_i) * ln2 / half_life) * gamma_c(q_i)
    """
    if timestamps is None or half_life is None:
        w = np.ones(resp.shape[0])
    else:
        age = timestamps.max() - np.asarray(timestamps, dtype=float)
        w = np.exp(-age * np.log(2.0) / half_life)
    p_hat = (w[:, None] * resp).sum(axis=0)
    return p_hat / p_hat.sum()


def tilt_weights(p_hat: np.ndarray, lam: float) -> np.ndarray:
    """pi'_c ~ lambda * p_hat_c + (1 - lambda) / C  (coverage-guaranteeing mix)."""
    c = len(p_hat)
    w = lam * p_hat + (1.0 - lam) / c
    return w / w.sum()


def nn_cos_threshold(prod_emb: np.ndarray, pct: float = 5.0) -> float:
    """On-manifold radius: pct-th percentile of each prod query's NN cosine.

    A sampled z is considered on-manifold if its nearest production query is
    at least this similar -- i.e. no farther than the sparsest 5% of real
    traffic is from its own neighbourhood.
    """
    nn = NearestNeighbors(n_neighbors=2, metric="cosine").fit(prod_emb)
    dist, _ = nn.kneighbors(prod_emb)
    return float(np.percentile(1.0 - dist[:, 1], pct))


@dataclass
class SpecSampler:
    """Samples target embeddings z from the demand-tilted movMF with an
    on-manifold rejection guard (the step-1/2 loop of the A2 mechanism)."""

    model: MovMF
    tilted_weights: np.ndarray
    prod_emb: np.ndarray
    tau_r: float
    exploration: np.ndarray | None = None  # bool mask over components
    max_tries: int = 50

    def __post_init__(self) -> None:
        self._nn = NearestNeighbors(n_neighbors=1, metric="cosine").fit(self.prod_emb)
        if self.exploration is None:
            self.exploration = np.zeros(len(self.tilted_weights), dtype=bool)

    def sample(self, n: int, rng: Generator) -> tuple[np.ndarray, np.ndarray]:
        """Returns (z (n, d), component ids (n,)). Guarded ancestral sampling."""
        zs, cs = [], []
        while len(zs) < n:
            batch = max(n - len(zs), 8)
            z, c = self.model.sample(batch, rng, weights=self.tilted_weights)
            dist, _ = self._nn.kneighbors(z)
            on_manifold = (1.0 - dist[:, 0]) >= self.tau_r
            keep = on_manifold | self.exploration[c]
            zs.append(z[keep])
            cs.append(c[keep])
            if sum(len(a) for a in zs) == 0 and len(zs) > self.max_tries:
                raise RuntimeError("SpecSampler: guard rejects everything; lower tau_r.")
        z = np.vstack(zs)[:n]
        c = np.concatenate(cs)[:n]
        return z, c


# --------------------------------------------------------------------------
# Importance weights, ESS, post-stratified headline (efficiency layer)
# --------------------------------------------------------------------------


def cluster_importance_weights(
    labels_synth: np.ndarray, p_hat: np.ndarray
) -> tuple[np.ndarray, float]:
    """Per-sample weights w_i = p_hat[c(i)] / q[c(i)] and the coverage gap.

    q is the empirical cluster distribution of the synthetic set. Clusters
    with zero synthetic samples contribute to ``coverage_gap`` (the share of
    production demand your set cannot speak for); p_hat is renormalised over
    covered clusters for the weights.
    """
    n_clusters = len(p_hat)
    counts = np.bincount(labels_synth, minlength=n_clusters).astype(float)
    q = counts / counts.sum()
    covered = q > 0
    coverage_gap = float(p_hat[~covered].sum())
    p_cov = np.where(covered, p_hat, 0.0)
    p_cov = p_cov / max(p_cov.sum(), 1e-12)
    w = np.where(q[labels_synth] > 0, p_cov[labels_synth] / np.maximum(q[labels_synth], 1e-12), 0.0)
    return w, coverage_gap


def effective_sample_size(weights: np.ndarray) -> float:
    """ESS = (sum w)^2 / sum w^2.  ESS == N iff all weights are equal."""
    s1, s2 = weights.sum(), (weights**2).sum()
    return float(s1 * s1 / max(s2, 1e-12))


def post_stratified_estimate(
    per_query_metric: np.ndarray, labels: np.ndarray, p_hat: np.ndarray
) -> float:
    """Demand-weighted headline: M_hat = sum_c p_hat_c * mean(M | cluster c),
    renormalised over covered clusters."""
    total, mass = 0.0, 0.0
    for c in range(len(p_hat)):
        mask = labels == c
        if mask.any():
            total += p_hat[c] * float(per_query_metric[mask].mean())
            mass += p_hat[c]
    return total / max(mass, 1e-12)


# --------------------------------------------------------------------------
# Fidelity metrics
# --------------------------------------------------------------------------


def kl_similarity_distributions(
    real_q: np.ndarray,
    synth_q: np.ndarray,
    chunk_emb: np.ndarray,
    bins: int = 50,
    eps: float = 1e-6,
) -> float:
    """KL(real || synth) between query->top-1-chunk cosine distributions
    (the Chroma 2025 representativeness monitor)."""
    s_real = (real_q @ chunk_emb.T).max(axis=1)
    s_synth = (synth_q @ chunk_emb.T).max(axis=1)
    lo = min(s_real.min(), s_synth.min())
    hi = max(s_real.max(), s_synth.max()) + 1e-9
    edges = np.linspace(lo, hi, bins + 1)
    p = np.histogram(s_real, bins=edges)[0].astype(float) + eps
    q = np.histogram(s_synth, bins=edges)[0].astype(float) + eps
    p, q = p / p.sum(), q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def c2st_auc(
    x_real: np.ndarray, x_synth: np.ndarray, n_splits: int = 5, seed: int = 0
) -> float:
    """Classifier two-sample test (Lopez-Paz & Oquab, ICLR 2017).

    Cross-validated ROC-AUC of a logistic regression separating real from
    synthetic embeddings. 0.5 = indistinguishable; higher = detectable gap.
    """
    x = np.vstack([x_real, x_synth])
    y = np.concatenate([np.zeros(len(x_real)), np.ones(len(x_synth))])
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return float(cross_val_score(clf, x, y, cv=cv, scoring="roc_auc").mean())


def mmd_rbf(x: np.ndarray, y: np.ndarray, gamma: float | None = None, max_n: int = 2000,
            seed: int = 0) -> float:
    """Unbiased squared MMD with an RBF kernel (Gretton et al., JMLR 2012).

    Median-heuristic bandwidth; subsamples to ``max_n`` per side.
    """
    rng = np.random.default_rng(seed)
    if len(x) > max_n:
        x = x[rng.choice(len(x), max_n, replace=False)]
    if len(y) > max_n:
        y = y[rng.choice(len(y), max_n, replace=False)]
    if gamma is None:
        from sklearn.metrics import pairwise_distances

        pool = np.vstack([x, y])
        sub = pool[rng.choice(len(pool), min(len(pool), 500), replace=False)]
        d2 = pairwise_distances(sub, metric="sqeuclidean")
        med = np.median(d2[d2 > 0])
        gamma = 1.0 / max(med, 1e-12)
    kxx, kyy, kxy = rbf_kernel(x, x, gamma), rbf_kernel(y, y, gamma), rbf_kernel(x, y, gamma)
    n, m = len(x), len(y)
    mmd2 = (
        (kxx.sum() - np.trace(kxx)) / (n * (n - 1))
        + (kyy.sum() - np.trace(kyy)) / (m * (m - 1))
        - 2.0 * kxy.mean()
    )
    return float(max(mmd2, 0.0))


def within_cluster_c2st(
    x_real: np.ndarray,
    x_synth: np.ndarray,
    labels_real: np.ndarray,
    labels_synth: np.ndarray,
    min_per_side: int = 30,
    seed: int = 0,
) -> tuple[float, dict[int, float]]:
    """Mean (and per-cluster) C2ST AUC computed *inside* each reference
    cluster -- the diagnostic that isolates within-cluster shape mismatch,
    which is exactly the mechanism A2 claims to improve. Cluster marginals
    being matched (as in A1) contributes nothing here."""
    per: dict[int, float] = {}
    for c in np.unique(labels_real):
        xr = x_real[labels_real == c]
        xs = x_synth[labels_synth == c]
        if len(xr) >= min_per_side and len(xs) >= min_per_side:
            n = min(len(xr), len(xs))
            rng = np.random.default_rng(seed + int(c))
            xr = xr[rng.choice(len(xr), n, replace=False)]
            xs = xs[rng.choice(len(xs), n, replace=False)]
            per[int(c)] = c2st_auc(xr, xs, n_splits=3, seed=seed)
    mean = float(np.mean(list(per.values()))) if per else float("nan")
    return mean, per


# --------------------------------------------------------------------------
# Validity metrics: ranking agreement + positive controls
# --------------------------------------------------------------------------


def system_ranking(mean_scores: np.ndarray) -> list[int]:
    """System indices ordered best-first."""
    return list(np.argsort(-mean_scores))


def tau_ap(reference: Sequence[int], candidate: Sequence[int]) -> float:
    """tau_AP (Yilmaz, Aslam & Robertson, SIGIR 2008): top-weighted rank
    correlation of ``candidate`` against ``reference`` (both best-first)."""
    ref_pos = {item: i for i, item in enumerate(reference)}
    n = len(candidate)
    if n < 2:
        return 1.0
    total = 0.0
    for i in range(1, n):
        item = candidate[i]
        concordant = sum(1 for j in range(i) if ref_pos[candidate[j]] < ref_pos[item])
        total += concordant / i
    return 2.0 * total / (n - 1) - 1.0


def rbo_ext(s: Sequence[int], t: Sequence[int], p: float = 0.9) -> float:
    """Extrapolated Rank-Biased Overlap (Webber, Moffat & Zobel, TOIS 2010)."""
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
    tau: float
    tau_ap_: float
    rbo: float
    tau_ci_low: float
    tau_ci_high: float


def ranking_agreement(
    anchor_scores: np.ndarray,  # (S, Q_anchor) per-query scores on the anchor set
    arm_scores: np.ndarray,  # (S, Q_arm) per-query scores on the synthetic set
    n_boot: int = 1000,
    seed: int = 0,
) -> RankingAgreement:
    """tau / tau_AP / RBO of the arm's system ranking vs the anchor ranking,
    with a bootstrap CI on tau obtained by resampling the arm's queries."""
    anchor_means = anchor_scores.mean(axis=1)
    arm_means = arm_scores.mean(axis=1)
    ref, cand = system_ranking(anchor_means), system_ranking(arm_means)

    tau = float(kendalltau(anchor_means, arm_means).statistic)
    rng = np.random.default_rng(seed)
    q = arm_scores.shape[1]
    idx = rng.integers(0, q, size=(n_boot, q))
    boot_means = arm_scores[:, idx].mean(axis=2)  # (S, n_boot)
    taus = np.array([kendalltau(anchor_means, boot_means[:, b]).statistic for b in range(n_boot)])
    lo, hi = np.percentile(taus, [2.5, 97.5])
    return RankingAgreement(tau, tau_ap(ref, cand), rbo_ext(ref, cand), float(lo), float(hi))


def paired_bootstrap_pvalue(
    scores_base: np.ndarray, scores_degraded: np.ndarray, n_boot: int = 2000, seed: int = 0
) -> tuple[float, float]:
    """One-sided paired bootstrap (Sakai, SIGIR 2006 tradition) that a
    degradation lowered the per-query metric. Returns (mean delta, p-value);
    the benchmark 'detects' the regression when delta > 0 and p < 0.05."""
    delta = scores_base - scores_degraded
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(delta), size=(n_boot, len(delta)))
    boot = delta[idx].mean(axis=1)
    p = float((boot <= 0).mean())
    return float(delta.mean()), p


# --------------------------------------------------------------------------
# LLM integration (plug your vLLM / LiteLLM stack here)
# --------------------------------------------------------------------------

EmbedFn = Callable[[Sequence[str]], np.ndarray]
"""Batch text -> L2-normalised embeddings in the *planning* space."""


@dataclass(frozen=True)
class GenerationContext:
    """Everything the generator sees for one sample (Protocol-based DI so the
    harness never imports an LLM client)."""

    component: int
    stratum: str
    style_exemplars: tuple[str, ...]  # nearest production queries around z
    chunk_ids: tuple[int, ...]  # nearest chunks around z (content seeds)
    chunk_texts: tuple[str, ...]
    z: np.ndarray | None = None  # the target embedding (A2 only)


class QueryGenerator(Protocol):
    def __call__(self, ctx: GenerationContext) -> str: ...


class QueryReviser(Protocol):
    def __call__(self, query: str, ctx: GenerationContext, cos_to_target: float) -> str: ...


def spec_first_generate(
    sampler: SpecSampler,
    embed_fn: EmbedFn,
    generator: QueryGenerator,
    prod_texts: Sequence[str],
    chunk_texts: Sequence[str],
    chunk_emb: np.ndarray,
    n: int,
    rng: Generator,
    tau_t: float = 0.6,
    m_style: int = 3,
    n_chunks: int = 5,
    reviser: QueryReviser | None = None,
    stratum: str = "standalone",
) -> list[dict]:
    """The full A2 loop against a *real* LLM: sample z -> kNN style exemplars
    and content seeds -> generate -> target-check (one optional revision).

    Output records feed your existing verification gate unchanged; ``z`` and
    ``cos_to_target`` land in gen_meta as the optimizer-loop signal.
    """
    q_nn = NearestNeighbors(n_neighbors=m_style, metric="cosine").fit(sampler.prod_emb)
    c_nn = NearestNeighbors(n_neighbors=n_chunks, metric="cosine").fit(chunk_emb)
    z_all, comp_all = sampler.sample(n, rng)
    records: list[dict] = []
    for z, comp in zip(z_all, comp_all):
        _, qi = q_nn.kneighbors(z[None, :])
        _, ci = c_nn.kneighbors(z[None, :])
        ctx = GenerationContext(
            component=int(comp),
            stratum=stratum,
            style_exemplars=tuple(prod_texts[j] for j in qi[0]),
            chunk_ids=tuple(int(j) for j in ci[0]),
            chunk_texts=tuple(chunk_texts[j] for j in ci[0]),
            z=z,
        )
        query = generator(ctx)
        emb = l2_normalize(embed_fn([query]))[0]
        cos = float(emb @ z)
        if cos < tau_t and reviser is not None:
            query = reviser(query, ctx, cos)
            emb = l2_normalize(embed_fn([query]))[0]
            cos = float(emb @ z)
        records.append(
            {
                "query": query,
                "embedding": emb,
                "gold_candidates": list(ctx.chunk_ids),
                "gen_meta": {"z": z, "component": int(comp), "cos_to_target": cos},
            }
        )
    return records


# --------------------------------------------------------------------------
# Simulated world for the self-verifying demo
# --------------------------------------------------------------------------
#
# Ground truth: K_true topical components on the sphere; *within* each
# component, production queries come from two asymmetric sub-modes (the
# within-cluster structure quota sampling cannot see). Chunks sit near
# component means, uniformly across components (KB coverage != demand).
# An LLM "style monoculture" is a fixed direction added to generated
# embeddings; steering shrinks it (A1), the target check shrinks it more (A2).


@dataclass
class World:
    chunk_emb: np.ndarray
    prod_train: np.ndarray
    prod_anchor: np.ndarray
    prod_oracle: np.ndarray
    gold_train: np.ndarray
    gold_anchor: np.ndarray
    gold_oracle: np.ndarray
    style_dir: np.ndarray


def _nearest_chunk(q: np.ndarray, chunk_emb: np.ndarray) -> np.ndarray:
    return np.argmax(q @ chunk_emb.T, axis=1)


def make_world(
    d: int = 64,
    k_true: int = 8,
    n_chunks: int = 640,
    n_prod: int = 5000,
    kappa_chunk: float = 150.0,
    kappa_query: float = 400.0,
    seed: int = 0,
) -> World:
    rng = np.random.default_rng(seed)
    mus = sphere_uniform(k_true, d, rng)
    weights = 1.0 / np.arange(1, k_true + 1) ** 1.1  # skewed demand
    weights = weights / weights.sum()

    # two asymmetric sub-modes per component (the A2-visible structure)
    sub_mus = np.empty((k_true, 2, d))
    for c in range(k_true):
        for s in range(2):
            t = rng.standard_normal(d)
            t -= (t @ mus[c]) * mus[c]  # tangent perturbation
            sub_mus[c, s] = l2_normalize(mus[c] + 0.35 * l2_normalize(t))

    per = n_chunks // k_true
    chunk_emb = np.vstack([sample_vmf(mus[c], kappa_chunk, per, rng) for c in range(k_true)])

    comps = rng.choice(k_true, size=n_prod, p=weights)
    subs = rng.choice(2, size=n_prod, p=[0.65, 0.35])
    queries = np.vstack(
        [sample_vmf(sub_mus[c, s], kappa_query, 1, rng) for c, s in zip(comps, subs)]
    )
    perm = rng.permutation(n_prod)
    n_tr, n_an = int(0.60 * n_prod), int(0.25 * n_prod)
    tr, an, orc = np.split(queries[perm], [n_tr, n_tr + n_an])

    style_dir = sphere_uniform(1, d, rng)[0]
    return World(
        chunk_emb=chunk_emb,
        prod_train=tr,
        prod_anchor=an,
        prod_oracle=orc,
        gold_train=_nearest_chunk(tr, chunk_emb),
        gold_anchor=_nearest_chunk(an, chunk_emb),
        gold_oracle=_nearest_chunk(orc, chunk_emb),
        style_dir=style_dir,
    )


def _emit(base: np.ndarray, style_dir: np.ndarray, style: float, noise: float,
          rng: Generator) -> np.ndarray:
    """Simulated 'LLM output embedding': seed + style monoculture + jitter.

    ``style`` and ``noise`` are geometric magnitudes (the noise direction is
    unit-normalised), so 0.35 means a perturbation of norm 0.35 on unit seeds.
    """
    g = l2_normalize(rng.standard_normal(base.shape))
    return l2_normalize(base + style * style_dir[None, :] + noise * g)


@dataclass
class ArmResult:
    name: str
    emb: np.ndarray
    gold: np.ndarray


def arm_a0(world: World, n: int, rng: Generator) -> ArmResult:
    """Naive chunk-first: uniform seeds, strong style bias, no steering.\n\n    Gold is relabeled to the nearest chunk of the *emitted* query, mirroring\n    what the round-trip + uniqueness gate produces (Gecko-style relabeling).\n    """
    seeds = rng.integers(0, len(world.chunk_emb), size=n)
    emb = _emit(world.chunk_emb[seeds], world.style_dir, 0.45, 0.55, rng)
    return ArmResult("A0 naive", emb, _nearest_chunk(emb, world.chunk_emb))


def arm_a1(
    world: World, kmeans: KMeans, p_hat: np.ndarray, lam: float, n: int, rng: Generator
) -> ArmResult:
    """Cluster-quota chunk-first: lambda-mixture allocation over the frozen
    reference partition + (simulated) cluster-level exemplar steering.
    Gold is gate-relabeled to the nearest chunk of the emitted query."""
    quotas = tilt_weights(p_hat, lam)
    chunk_cluster = kmeans.predict(world.chunk_emb)
    seeds = np.empty(n, dtype=int)
    draws = rng.choice(len(quotas), size=n, p=quotas)
    for c in np.unique(draws):
        pool = np.where(chunk_cluster == c)[0]
        if len(pool) == 0:
            pool = np.arange(len(world.chunk_emb))
        seeds[draws == c] = rng.choice(pool, size=(draws == c).sum())
    emb = _emit(world.chunk_emb[seeds], world.style_dir, 0.15, 0.68, rng)
    return ArmResult("A1 quota", emb, _nearest_chunk(emb, world.chunk_emb))


def arm_a2(
    world: World, sampler: SpecSampler, n: int, rng: Generator
) -> ArmResult:
    """Spec-first: z from the demand-tilted movMF (captures sub-modes),
    tight target check keeps the emission near z; gold = nearest chunk to z."""
    z, _ = sampler.sample(n, rng)
    emb = _emit(z, world.style_dir, 0.05, 0.15, rng)
    return ArmResult("A2 spec", emb, _nearest_chunk(emb, world.chunk_emb))


def arm_oracle(world: World, n: int, rng: Generator) -> ArmResult:
    idx = rng.choice(len(world.prod_oracle), size=min(n, len(world.prod_oracle)), replace=False)
    return ArmResult("ORACLE", world.prod_oracle[idx], world.gold_oracle[idx])


# --------------------------------------------------------------------------
# System zoo and retrieval scoring
# --------------------------------------------------------------------------


def make_system_zoo(d: int, seed: int = 0) -> dict[str, np.ndarray]:
    """Deterministic embedding-model variants: identity, increasingly
    distorted mixes, and low-rank projections. Each is a (d, d) matrix
    applied to both queries and chunks."""
    rng = np.random.default_rng(seed)
    zoo: dict[str, np.ndarray] = {"exact": np.eye(d)}
    for sigma in (0.10, 0.28, 0.30, 0.32, 0.34, 0.50, 0.75):
        zoo[f"distort-{sigma}"] = np.eye(d) + sigma * rng.standard_normal((d, d)) / np.sqrt(d)
    for r in (48, 32, 28, 24):
        q, _ = np.linalg.qr(rng.standard_normal((d, r)))
        zoo[f"rank-{r}"] = q @ q.T
    return zoo


def score_system(
    m: np.ndarray,
    q_emb: np.ndarray,
    gold: np.ndarray,
    chunk_emb: np.ndarray,
    k: int = 10,
    drop_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Per-query nDCG@k with a single gold chunk: 1/log2(1+rank) if the gold
    is retrieved in the top k, else 0. ``drop_mask`` simulates index loss."""
    qe = l2_normalize(q_emb @ m)
    ce = l2_normalize(chunk_emb @ m)
    sims = qe @ ce.T
    if drop_mask is not None:
        sims[:, drop_mask] = -np.inf
        gold_dropped = drop_mask[gold]
    else:
        gold_dropped = np.zeros(len(gold), dtype=bool)
    gold_sims = sims[np.arange(len(gold)), gold]
    ranks = (sims > gold_sims[:, None]).sum(axis=1) + 1
    scores = np.where(ranks <= k, 1.0 / np.log2(ranks + 1.0), 0.0)
    return np.where(gold_dropped, 0.0, scores)


def evaluate_zoo(
    zoo: dict[str, np.ndarray], q_emb: np.ndarray, gold: np.ndarray, chunk_emb: np.ndarray
) -> np.ndarray:
    """(S, Q) per-query score matrix in the zoo's insertion order."""
    return np.vstack([score_system(m, q_emb, gold, chunk_emb) for m in zoo.values()])


# --------------------------------------------------------------------------
# Demo orchestration
# --------------------------------------------------------------------------


def run_demo(n_arm: int = 500, lam: float = 0.7, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    print("Building world (8 topical components x 2 hidden sub-modes, skewed demand) ...")
    world = make_world(seed=seed)

    # frozen reference partition + hard demand map (shared by quotas/ESS/reporting)
    ref_km = KMeans(n_clusters=8, n_init=10, random_state=seed).fit(world.prod_train)
    p_hat = np.bincount(ref_km.labels_, minlength=8).astype(float)
    p_hat /= p_hat.sum()

    # A2 planning density: finer movMF on production queries + tilt + guard
    print("Fitting movMF demand map (K=16) on production train split ...")
    movmf = MovMF(n_components=16, seed=seed).fit(world.prod_train)
    demand = demand_from_responsibilities(movmf.responsibilities(world.prod_train))
    sampler = SpecSampler(
        model=movmf,
        tilted_weights=tilt_weights(demand, lam),
        prod_emb=world.prod_train,
        tau_r=nn_cos_threshold(world.prod_train, pct=5.0),
    )

    arms = [
        arm_a0(world, n_arm, rng),
        arm_a1(world, ref_km, p_hat, lam, n_arm, rng),
        arm_a2(world, sampler, n_arm, rng),
        arm_oracle(world, n_arm, rng),
    ]

    zoo = make_system_zoo(world.chunk_emb.shape[1], seed=seed)
    anchor_scores = evaluate_zoo(zoo, world.prod_anchor, world.gold_anchor, world.chunk_emb)

    real_ref = world.prod_anchor[rng.choice(len(world.prod_anchor), n_arm, replace=False)]
    labels_real = ref_km.predict(real_ref)

    # positive controls: can each arm detect a known regression of 'exact'?
    drop_mask = np.zeros(len(world.chunk_emb), dtype=bool)
    drop_mask[rng.choice(len(world.chunk_emb), int(0.10 * len(world.chunk_emb)), replace=False)] = True
    noisy_m = np.eye(world.chunk_emb.shape[1]) + 0.5 * rng.standard_normal(
        (world.chunk_emb.shape[1],) * 2
    ) / np.sqrt(world.chunk_emb.shape[1])

    header = (
        f"{'arm':<10}{'KL':>7}{'C2ST':>7}{'wC2ST':>7}{'MMD*1e3':>9}"
        f"{'ESS/N':>7}{'gap':>6}{'tau':>7}{'tau95CI':>15}{'tauAP':>7}{'RBO':>6}"
        f"{'PCdrop':>8}{'PCnoise':>8}"
    )
    print("\n" + header)
    print("-" * len(header))

    for arm in arms:
        labels_synth = ref_km.predict(arm.emb)
        kl = kl_similarity_distributions(real_ref, arm.emb, world.chunk_emb)
        auc = c2st_auc(real_ref, arm.emb, seed=seed)
        wauc, _ = within_cluster_c2st(real_ref, arm.emb, labels_real, labels_synth, seed=seed)
        mmd = mmd_rbf(real_ref, arm.emb, seed=seed)
        w, gap = cluster_importance_weights(labels_synth, p_hat)
        ess = effective_sample_size(w) / len(w)

        arm_scores = evaluate_zoo(zoo, arm.emb, arm.gold, world.chunk_emb)
        agree = ranking_agreement(anchor_scores, arm_scores, seed=seed)

        base = score_system(zoo["exact"], arm.emb, arm.gold, world.chunk_emb)
        deg_drop = score_system(zoo["exact"], arm.emb, arm.gold, world.chunk_emb, drop_mask=drop_mask)
        deg_noise = score_system(noisy_m, arm.emb, arm.gold, world.chunk_emb)
        _, p_drop = paired_bootstrap_pvalue(base, deg_drop, seed=seed)
        _, p_noise = paired_bootstrap_pvalue(base, deg_noise, seed=seed)

        print(
            f"{arm.name:<10}{kl:>7.3f}{auc:>7.3f}{wauc:>7.3f}{mmd * 1e3:>9.2f}"
            f"{ess:>7.2f}{gap:>6.2f}{agree.tau:>7.3f}"
            f"{'[' + format(agree.tau_ci_low, '.2f') + ',' + format(agree.tau_ci_high, '.2f') + ']':>15}"
            f"{agree.tau_ap_:>7.3f}{agree.rbo:>6.2f}{p_drop:>8.3f}{p_noise:>8.3f}"
        )

    print(
        "\nRead: KL/C2ST/wC2ST/MMD lower is better (0.5 = indistinguishable for AUCs);"
        "\nESS/N higher is better; tau/tauAP/RBO vs anchor higher is better;"
        "\nPC* are positive-control p-values (< 0.05 = the arm detects the"
        "\ninjected regression of the 'exact' system)."
        "\n\nWhat the demo shows: quota matching (A1) fixes ESS but contributes"
        "\nnothing within-cluster (wC2ST ~= 1.0); spec-first sampling (A2) is the"
        "\nonly arm that matches within-cluster shape, and it passes the tau gate"
        "\nwhile A0/A1 fail it. Note A1 also MISSES the noise regression"
        "\n(PCnoise > 0.05): a distributionally-off benchmark can be insensitive"
        "\nto real regressions -- run positive controls. ORACLE is a subsample of"
        "\nreal queries, so its tau is itself noisy at this N (the ceiling has a CI)."
    )


if __name__ == "__main__":
    run_demo()
