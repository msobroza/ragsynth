"""Sphere utilities and von Mises-Fisher sampling.

Port of the vendored prototype ``reference/synth_query_eval.py`` (L67-L137,
SPEC §7.1). The rejection sampler follows Wood (1994); the log normalization
constant uses the exponentially scaled Bessel function ``ive`` for numerical
stability across the full concentration range.

References:
    Wood, A. T. A. (1994). Simulation of the von Mises Fisher distribution.
    Communications in Statistics - Simulation and Computation, 23(1), 157-164.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.special import gammaln, ive

if TYPE_CHECKING:
    from numpy.random import Generator
    from numpy.typing import NDArray

_KAPPA_UNIFORM = 1e-8
"""Below this concentration the vMF is numerically uniform on the sphere."""

_POLE_EPS = 1e-12
"""Reflection-vector norm under which mu already equals the canonical pole."""

_BESSEL_FLOOR = 1e-300
"""Floor applied to the scaled Bessel value before taking its log."""


def l2_normalize(x: NDArray[np.float64], axis: int = -1, eps: float = 1e-12) -> NDArray[np.float64]:
    """Project vectors onto the unit sphere (prototype L67).

    Args:
        x: Array of vectors; any shape.
        axis: Axis holding the vector components.
        eps: Norm floor guarding against division by zero.

    Returns:
        ``x`` scaled to unit L2 norm along ``axis``, as float64.
    """
    x64 = np.asarray(x, dtype=np.float64)
    normed = x64 / np.maximum(np.linalg.norm(x64, axis=axis, keepdims=True), eps)
    return np.asarray(normed, dtype=np.float64)


def sphere_uniform(n: int, d: int, rng: Generator) -> NDArray[np.float64]:
    """Draw ``n`` uniform samples on S^{d-1} (prototype L72).

    Args:
        n: Number of samples.
        d: Ambient dimension.
        rng: Source of randomness.

    Returns:
        Array of shape ``(n, d)`` with unit rows.
    """
    return l2_normalize(rng.standard_normal((n, d)))


def log_sphere_area(d: int) -> float:
    """Compute the log surface area of S^{d-1} (prototype L77).

    Args:
        d: Ambient dimension.

    Returns:
        ``log(2 pi^{d/2} / Gamma(d/2))``.
    """
    return float(np.log(2.0) + (d / 2.0) * np.log(np.pi) - gammaln(d / 2.0))


def sample_vmf(
    mu: NDArray[np.float64], kappa: float, n: int, rng: Generator
) -> NDArray[np.float64]:
    """Draw ``n`` samples from vMF(mu, kappa) on S^{d-1} (prototype L87).

    Rejection sampler of Wood (1994); a Householder reflection maps the
    canonical pole ``e1`` onto ``mu``. For ``kappa`` below 1e-8 the
    distribution is numerically uniform and sampled directly.

    Args:
        mu: Unit mean direction of shape ``(d,)``.
        kappa: Concentration parameter, ``kappa >= 0``.
        n: Number of samples.
        rng: Source of randomness.

    Returns:
        Array of shape ``(n, d)`` with unit rows.

    References:
        Wood, A. T. A. (1994). Simulation of the von Mises Fisher
        distribution. Communications in Statistics - Simulation and
        Computation, 23(1), 157-164.
    """
    mu64 = np.asarray(mu, dtype=np.float64)
    d = mu64.shape[0]
    if kappa < _KAPPA_UNIFORM:
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
    u_vec = e1 - mu64
    nu = float(np.linalg.norm(u_vec))
    if nu < _POLE_EPS:  # mu is already the pole
        return np.asarray(x, dtype=np.float64)
    u_vec = u_vec / nu
    reflected = x - 2.0 * np.outer(x @ u_vec, u_vec)  # Householder: e1 -> mu
    return np.asarray(reflected, dtype=np.float64)


def vmf_log_norm_const(d: int, kappa: NDArray[np.float64] | float) -> NDArray[np.float64]:
    """Compute ``log C_d(kappa)`` with the scaled Bessel function (prototype L124).

    ``C_d(k) = k^{d/2-1} / ((2 pi)^{d/2} I_{d/2-1}(k))`` and
    ``log I_v(k) = log(ive(v, k)) + k`` keeps the evaluation finite for large
    ``kappa``. As ``kappa -> 0`` the density is uniform on the sphere, so the
    log constant tends to ``-log_sphere_area(d)``.

    Args:
        d: Ambient dimension.
        kappa: Concentration parameter(s), scalar or array.

    Returns:
        ``log C_d(kappa)`` with the same shape as ``kappa``.
    """
    kappa64 = np.asarray(kappa, dtype=np.float64)
    v = d / 2.0 - 1.0
    small = kappa64 < _KAPPA_UNIFORM
    log_iv = (
        np.log(np.maximum(ive(v, np.maximum(kappa64, _KAPPA_UNIFORM)), _BESSEL_FLOOR)) + kappa64
    )
    out = v * np.log(np.maximum(kappa64, _KAPPA_UNIFORM)) - (d / 2.0) * np.log(2.0 * np.pi) - log_iv
    if np.any(small):  # kappa -> 0: uniform density on the sphere
        out = np.where(small, -log_sphere_area(d), out)
    return np.asarray(out, dtype=np.float64)
