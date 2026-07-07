"""Frozen KMeans reference partition over the query embedding space.

The reference partition (SPEC §7.4) defines the cluster ids used by ALL
reporting: demand estimates, quota allocation, importance weights/ESS, and
per-cluster metric tables. It is fit once on training production queries and
then frozen as a versioned artifact.

**Changing the partition is a benchmark-migration event** (SPEC §7.4): every
quantity above is expressed in its cluster ids, so numbers computed under
different partitions are not comparable. Refit only when deliberately opening
a new benchmark epoch, and re-baseline every tracked metric when you do.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Self

import numpy as np
from sklearn.cluster import KMeans

from ragsynth.io.artifacts import sha256_hex

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray


class ReferencePartition:
    """Frozen KMeans partition: stores only the cluster centers.

    Inputs are assumed L2-normalized (the pipeline's embedding contract).
    :meth:`assign` reproduces ``KMeans.predict`` semantics — nearest center
    by euclidean distance, ties resolved to the lowest cluster id.

    Attributes:
        centers: Cluster centers ``(n_clusters, d)``.
        n_clusters: Number of clusters (derived from ``centers``).
        seed: Seed the partition was fit with.
        fitted_on_hash: sha256 of the training matrix bytes, pinning the
            partition to the exact data it was fit on.
    """

    def __init__(
        self,
        centers: NDArray[np.float64],
        seed: int = 0,
        fitted_on_hash: str = "",
    ) -> None:
        self.centers: NDArray[np.float64] = np.asarray(centers, dtype=np.float64)
        self.n_clusters: int = int(self.centers.shape[0])
        self.seed = int(seed)
        self.fitted_on_hash = fitted_on_hash

    @classmethod
    def fit(cls, query_embs: NDArray[np.float64], n_clusters: int = 8, seed: int = 0) -> Self:
        """Fit the partition with KMeans and keep only the centers.

        Args:
            query_embs: L2-normalized training query embeddings ``(n, d)``.
            n_clusters: Number of reference clusters (``C``); default 8
                per PLAN D2.
            seed: ``random_state`` for KMeans (``n_init=10``).

        Returns:
            The fitted partition.
        """
        x = np.asarray(query_embs, dtype=np.float64)
        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit(x)
        return cls(
            centers=np.asarray(km.cluster_centers_, dtype=np.float64),
            seed=seed,
            fitted_on_hash=sha256_hex(np.ascontiguousarray(x).tobytes()),
        )

    def assign(self, embs: NDArray[np.float64]) -> NDArray[np.int_]:
        """Assign each row to its nearest center (euclidean argmin).

        Matches ``KMeans.predict``: the squared distance
        ``||e||^2 - 2 e.c + ||c||^2`` is minimized over centers; the
        row-constant ``||e||^2`` term is dropped.

        Args:
            embs: Embedding matrix ``(n, d)``.

        Returns:
            Cluster labels ``(n,)`` in ``[0, n_clusters)``.
        """
        embs64 = np.asarray(embs, dtype=np.float64)
        center_sq = np.einsum("kd,kd->k", self.centers, self.centers)
        sq_dist = -2.0 * embs64 @ self.centers.T + center_sq[None, :]
        return np.asarray(np.argmin(sq_dist, axis=1), dtype=np.int_)

    def proportions(self, embs: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute the normalized cluster histogram of ``embs``.

        ``p_hat = proportions(train_query_embs)`` is the hard-label demand
        estimate over the reference partition (PLAN §3).

        Args:
            embs: Embedding matrix ``(n, d)`` with ``n >= 1``.

        Returns:
            Vector ``(n_clusters,)`` summing to 1.

        Raises:
            ValueError: If ``embs`` is empty.
        """
        embs64 = np.asarray(embs, dtype=np.float64)
        if embs64.shape[0] == 0:
            raise ValueError("proportions of an empty embedding set are undefined")
        counts = np.bincount(self.assign(embs64), minlength=self.n_clusters).astype(np.float64)
        return np.asarray(counts / counts.sum(), dtype=np.float64)

    def to_artifact(self, path: Path) -> None:
        """Persist the partition as an npz artifact at ``path``.

        Args:
            path: Destination file path (written exactly, no suffix added).
        """
        with path.open("wb") as fh:
            np.savez(
                fh,
                centers=self.centers,
                seed=np.array(self.seed),
                fitted_on_hash=np.array(self.fitted_on_hash),
            )

    @classmethod
    def from_artifact(cls, path: Path) -> Self:
        """Load a partition previously written by :meth:`to_artifact`.

        Args:
            path: Path to the npz artifact.

        Returns:
            The reconstructed partition.
        """
        with np.load(path, allow_pickle=False) as data:
            return cls(
                centers=np.asarray(data["centers"], dtype=np.float64),
                seed=int(data["seed"]),
                fitted_on_hash=str(data["fitted_on_hash"]),
            )
