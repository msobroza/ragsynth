"""Mixture of von Mises-Fisher distributions with EM on the unit sphere.

Port of the vendored prototype ``reference/synth_query_eval.py`` (L145-L244,
SPEC §7.2): log-space responsibilities, the Banerjee et al. closed-form
concentration update, KMeans initialization, plus npz artifact IO — the
fitted demand map is a versioned frozen artifact per benchmark epoch.

References:
    Banerjee, A., Dhillon, I. S., Ghosh, J., & Sra, S. (2005). Clustering on
    the unit hypersphere using von Mises-Fisher distributions. Journal of
    Machine Learning Research, 6, 1345-1382.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Self

import numpy as np
from scipy.special import logsumexp
from sklearn.cluster import KMeans

from ragsynth.io.artifacts import sha256_hex
from ragsynth.sampling.vmf import l2_normalize, sample_vmf, vmf_log_norm_const

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.random import Generator
    from numpy.typing import NDArray

_LOG_FLOOR = 1e-300
"""Floor applied to mixture weights before taking their log."""

_RESP_EPS = 1e-12
"""Additive floor on per-component responsibility mass (empty-component guard)."""

_RESULTANT_EPS = 1e-12
"""Resultant-vector norm under which a component's mean is left unchanged."""

_RBAR_CLIP = 1e-6
"""Clip margin keeping the mean resultant length inside the open unit interval."""


class MovMF:
    """Mixture of von Mises-Fisher distributions on the unit sphere.

    EM with log-space responsibilities and the Banerjee et al. (2005)
    concentration approximation ``kappa = rbar (d - rbar^2) / (1 - rbar^2)``,
    ``kappa`` clipped to ``[kappa_min, kappa_max]``, KMeans initialization.

    Attributes after :meth:`fit`: ``weights_`` (K,), ``means_`` (K, d),
    ``kappas_`` (K,), ``log_likelihood_``, and ``fitted_on_hash`` — the
    sha256 of the (normalized) training matrix bytes, recorded so the fitted
    planning density can be pinned as a frozen artifact (SPEC §7.2).

    References:
        Banerjee, A., Dhillon, I. S., Ghosh, J., & Sra, S. (2005).
        Clustering on the unit hypersphere using von Mises-Fisher
        distributions. JMLR, 6, 1345-1382.
    """

    def __init__(
        self,
        n_components: int,
        max_iter: int = 200,
        tol: float = 1e-6,
        kappa_min: float = 1e-2,
        kappa_max: float = 1e5,
        seed: int = 0,
    ) -> None:
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.kappa_min = kappa_min
        self.kappa_max = kappa_max
        self.seed = seed
        self.weights_: NDArray[np.float64] = np.empty(0, dtype=np.float64)
        self.means_: NDArray[np.float64] = np.empty((0, 0), dtype=np.float64)
        self.kappas_: NDArray[np.float64] = np.empty(0, dtype=np.float64)
        self.log_likelihood_: float = float("nan")
        self.fitted_on_hash: str = ""

    # -- internals ----------------------------------------------------------

    def _require_fitted(self) -> None:
        """Raise unless the mixture parameters have been set.

        Raises:
            RuntimeError: If the model is not fitted.
        """
        if self.means_.size == 0:
            raise RuntimeError("MovMF is not fitted; call fit() or from_artifact() first.")

    def _component_log_pdf(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute the ``(n, K)`` matrix of ``log f(x | mu_k, kappa_k)``."""
        d = x.shape[1]
        log_c = vmf_log_norm_const(d, self.kappas_)  # (K,)
        return np.asarray(
            log_c[None, :] + (x @ self.means_.T) * self.kappas_[None, :], dtype=np.float64
        )

    @staticmethod
    def _kappa_banerjee(rbar: NDArray[np.float64], d: int) -> NDArray[np.float64]:
        """Apply the Banerjee et al. approximation ``kappa = rbar (d - rbar^2)/(1 - rbar^2)``."""
        rbar64 = np.clip(rbar, _RBAR_CLIP, 1.0 - _RBAR_CLIP)
        return np.asarray(rbar64 * (d - rbar64**2) / (1.0 - rbar64**2), dtype=np.float64)

    # -- API ----------------------------------------------------------------

    def fit(self, x: NDArray[np.float64]) -> Self:
        """Fit the mixture by EM (prototype L180-L217).

        Args:
            x: Sample matrix of shape ``(n, d)``; rows are L2-normalized
                before fitting.

        Returns:
            The fitted model (self), with ``fitted_on_hash`` set to the
            sha256 of the normalized training matrix bytes.
        """
        x64 = l2_normalize(np.asarray(x, dtype=np.float64))
        self.fitted_on_hash = sha256_hex(np.ascontiguousarray(x64).tobytes())
        n, d = x64.shape
        k = self.n_components

        km = KMeans(n_clusters=k, n_init=10, random_state=self.seed).fit(x64)
        self.means_ = l2_normalize(np.asarray(km.cluster_centers_, dtype=np.float64))
        self.weights_ = np.bincount(km.labels_, minlength=k).astype(np.float64) / n
        rbar0 = np.array(
            [
                np.linalg.norm(x64[km.labels_ == c].sum(axis=0)) / max((km.labels_ == c).sum(), 1)
                for c in range(k)
            ]
        )
        self.kappas_ = np.clip(self._kappa_banerjee(rbar0, d), self.kappa_min, self.kappa_max)

        prev_ll = -np.inf
        for _ in range(self.max_iter):
            log_joint = np.log(np.maximum(self.weights_, _LOG_FLOOR))[
                None, :
            ] + self._component_log_pdf(x64)
            log_norm = logsumexp(log_joint, axis=1)  # (n,)
            ll = float(np.mean(log_norm))
            resp = np.exp(log_joint - log_norm[:, None])  # (n, K)

            nk = resp.sum(axis=0) + _RESP_EPS
            self.weights_ = nk / n
            r = resp.T @ x64  # (K, d)
            norms = np.linalg.norm(r, axis=1)
            self.means_ = l2_normalize(np.where(norms[:, None] > _RESULTANT_EPS, r, self.means_))
            rbar = norms / nk
            self.kappas_ = np.clip(self._kappa_banerjee(rbar, d), self.kappa_min, self.kappa_max)

            if abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll

        self.log_likelihood_ = float(prev_ll)
        return self

    def responsibilities(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute the ``(n, K)`` soft assignments ``gamma_c(x)``.

        Raises:
            RuntimeError: If the model is not fitted.
        """
        self._require_fitted()
        x64 = l2_normalize(np.asarray(x, dtype=np.float64))
        log_joint = np.log(np.maximum(self.weights_, _LOG_FLOOR))[
            None, :
        ] + self._component_log_pdf(x64)
        return np.asarray(
            np.exp(log_joint - logsumexp(log_joint, axis=1)[:, None]), dtype=np.float64
        )

    def log_prob(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute the mixture log-density at each row of ``x``.

        Raises:
            RuntimeError: If the model is not fitted.
        """
        self._require_fitted()
        x64 = l2_normalize(np.asarray(x, dtype=np.float64))
        log_joint = np.log(np.maximum(self.weights_, _LOG_FLOOR))[
            None, :
        ] + self._component_log_pdf(x64)
        return np.asarray(logsumexp(log_joint, axis=1), dtype=np.float64)

    def sample(
        self, n: int, rng: Generator, weights: NDArray[np.float64] | None = None
    ) -> tuple[NDArray[np.float64], NDArray[np.int_]]:
        """Sample ancestrally: component ids then vMF draws (prototype L230).

        Args:
            n: Number of samples.
            rng: Source of randomness.
            weights: Optional override of the fitted mixing weights — pass
                the tilted ``pi'`` here to sample from the demand-tilted
                mixture. Renormalized internally.

        Returns:
            Tuple of samples ``(n, d)`` and component ids ``(n,)``.

        Raises:
            RuntimeError: If the model is not fitted.
        """
        self._require_fitted()
        w = self.weights_ if weights is None else np.asarray(weights, dtype=np.float64)
        w = w / w.sum()
        comps = np.asarray(rng.choice(len(w), size=n, p=w), dtype=np.int_)
        d = self.means_.shape[1]
        out = np.empty((n, d))
        for c in np.unique(comps):
            idx = np.where(comps == c)[0]
            out[idx] = sample_vmf(self.means_[c], float(self.kappas_[c]), len(idx), rng)
        return out, comps

    # -- artifact IO ----------------------------------------------------------

    def to_artifact(self, path: Path) -> None:
        """Persist the fitted mixture as an npz artifact at ``path``.

        Stores ``weights_``, ``means_``, ``kappas_``, every ``__init__``
        parameter, ``log_likelihood_``, and ``fitted_on_hash`` (SPEC §7.2:
        the demand map is a versioned frozen artifact per benchmark epoch).

        Args:
            path: Destination file path (written exactly, no suffix added).

        Raises:
            RuntimeError: If the model is not fitted.
        """
        self._require_fitted()
        with path.open("wb") as fh:
            np.savez(
                fh,
                weights_=self.weights_,
                means_=self.means_,
                kappas_=self.kappas_,
                n_components=np.array(self.n_components),
                max_iter=np.array(self.max_iter),
                tol=np.array(self.tol),
                kappa_min=np.array(self.kappa_min),
                kappa_max=np.array(self.kappa_max),
                seed=np.array(self.seed),
                log_likelihood_=np.array(self.log_likelihood_),
                fitted_on_hash=np.array(self.fitted_on_hash),
            )

    @classmethod
    def from_artifact(cls, path: Path) -> Self:
        """Load a mixture previously written by :meth:`to_artifact`.

        Args:
            path: Path to the npz artifact.

        Returns:
            A fitted :class:`MovMF` with identical parameters.
        """
        with np.load(path, allow_pickle=False) as data:
            model = cls(
                n_components=int(data["n_components"]),
                max_iter=int(data["max_iter"]),
                tol=float(data["tol"]),
                kappa_min=float(data["kappa_min"]),
                kappa_max=float(data["kappa_max"]),
                seed=int(data["seed"]),
            )
            model.weights_ = np.asarray(data["weights_"], dtype=np.float64)
            model.means_ = np.asarray(data["means_"], dtype=np.float64)
            model.kappas_ = np.asarray(data["kappas_"], dtype=np.float64)
            model.log_likelihood_ = float(data["log_likelihood_"])
            model.fitted_on_hash = str(data["fitted_on_hash"])
        return model
