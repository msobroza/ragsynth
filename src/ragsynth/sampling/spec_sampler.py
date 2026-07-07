"""Guarded ancestral sampling of target embeddings (the A2 core).

Port of the vendored prototype ``reference/synth_query_eval.py`` (L289-L321,
SPEC §7.5): ``c ~ Cat(pi')``, ``z ~ vMF(mu_c, kappa_c)``, rejected unless
``max_j z.q_j >= tau_r`` (the on-manifold guard) or component ``c`` is an
exploration component.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.neighbors import NearestNeighbors

if TYPE_CHECKING:
    from numpy.random import Generator
    from numpy.typing import NDArray

    from ragsynth.sampling.movmf import MovMF

_MIN_BATCH = 8
"""Smallest draw batch, so the guard sees enough candidates per round."""


class SpecSampler:
    """Sample target embeddings ``z`` from the demand-tilted movMF.

    Each draw passes the on-manifold rejection guard — its nearest
    production query must have cosine at least ``tau_r`` — unless its
    component is flagged in ``exploration`` (the step-1/2 loop of the A2
    mechanism, SPEC §7.5).
    """

    def __init__(
        self,
        model: MovMF,
        tilted_weights: NDArray[np.float64],
        prod_emb: NDArray[np.float64],
        tau_r: float,
        exploration: NDArray[np.bool_] | None = None,
        max_tries: int = 50,
    ) -> None:
        self.model = model
        self.tilted_weights = np.asarray(tilted_weights, dtype=np.float64)
        self.prod_emb = np.asarray(prod_emb, dtype=np.float64)
        self.tau_r = float(tau_r)
        self.max_tries = int(max_tries)
        mask: NDArray[np.bool_]
        if exploration is None:
            mask = np.zeros(len(self.tilted_weights), dtype=bool)
        else:
            mask = np.asarray(exploration, dtype=bool)
        self.exploration = mask
        self._nn = NearestNeighbors(n_neighbors=1, metric="cosine").fit(self.prod_emb)

    def sample(self, n: int, rng: Generator) -> tuple[NDArray[np.float64], NDArray[np.int_]]:
        """Draw ``n`` guarded samples from the tilted mixture.

        Guarded ancestral sampling (prototype L306-L321; bookkeeping here
        counts accepted *samples* where the prototype counted batches — the
        accepted stream and its distribution are identical).

        Args:
            n: Number of samples to return.
            rng: Source of randomness.

        Returns:
            Tuple of ``z`` ``(n, d)`` and component ids ``(n,)``.

        Raises:
            RuntimeError: If the guard rejects every draw for more than
                ``max_tries`` consecutive batches — lower ``tau_r``.
        """
        zs: list[NDArray[np.float64]] = []
        cs: list[NDArray[np.int_]] = []
        total = 0
        tries = 0
        while total < n:
            batch = max(n - total, _MIN_BATCH)
            z, c = self.model.sample(batch, rng, weights=self.tilted_weights)
            dist, _ = self._nn.kneighbors(z)
            on_manifold = (1.0 - dist[:, 0]) >= self.tau_r
            keep = on_manifold | self.exploration[c]
            zs.append(z[keep])
            cs.append(c[keep])
            total += int(keep.sum())
            tries += 1
            if total == 0 and tries > self.max_tries:
                raise RuntimeError("SpecSampler: guard rejects everything; lower tau_r.")
        z_out = np.asarray(np.vstack(zs)[:n], dtype=np.float64)
        c_out = np.asarray(np.concatenate(cs)[:n], dtype=np.int_)
        return z_out, c_out
