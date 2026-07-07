"""Externalized embedding storage.

Domain objects never hold vectors inline (SPEC §4 rule); they hold an
``embedding_ref`` key into an :class:`EmbeddingStore`, which owns the
float32 matrix and persists as a single ``.npz`` (ids + matrix).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


_MATRIX_NDIM = 2


class EmbeddingStore:
    """Append-only id -> vector map with npz persistence.

    Vectors are stored as float32 rows; ids are arbitrary strings
    (chunk_id / query_id). Overwriting an existing id is an error --
    embeddings are content-addressed by their owning object's id and
    must not silently change within a run.
    """

    def __init__(self) -> None:
        self._index: dict[str, int] = {}
        self._rows: list[np.ndarray] = []
        self._dim: int | None = None

    @property
    def dim(self) -> int | None:
        """Vector dimensionality, or ``None`` while the store is empty."""
        return self._dim

    def __contains__(self, key: str) -> bool:
        return key in self._index

    def __len__(self) -> int:
        return len(self._index)

    def add(self, ids: Sequence[str], vectors: np.ndarray) -> None:
        """Register vectors under new ids.

        Args:
            ids: One id per row of ``vectors``.
            vectors: Array of shape ``(len(ids), d)``.

        Raises:
            ValueError: If lengths mismatch, an id already exists, or the
                dimensionality differs from previously added vectors.
        """
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != _MATRIX_NDIM or matrix.shape[0] != len(ids):
            raise ValueError(
                f"ids/vectors mismatch: {len(ids)} ids vs vectors of shape {matrix.shape}"
            )
        if self._dim is not None and matrix.shape[1] != self._dim:
            raise ValueError(f"dimension mismatch: store has d={self._dim}, got {matrix.shape[1]}")
        dupes = [i for i in ids if i in self._index]
        if dupes:
            raise ValueError(f"duplicate embedding ids: {dupes[:5]}")
        if self._dim is None:
            self._dim = int(matrix.shape[1])
        base = len(self._rows)
        for offset, key in enumerate(ids):
            self._index[key] = base + offset
        self._rows.extend(matrix)

    def get(self, ids: Sequence[str]) -> np.ndarray:
        """Return vectors for ``ids`` as an ``(len(ids), d)`` float32 array.

        Raises:
            KeyError: If any id is absent.
        """
        missing = [i for i in ids if i not in self._index]
        if missing:
            raise KeyError(f"unknown embedding ids: {missing[:5]}")
        return np.stack([self._rows[self._index[i]] for i in ids])

    def save(self, path: Path) -> None:
        """Persist as npz with ``ids`` (unicode array) and ``matrix``."""
        ids = sorted(self._index, key=self._index.__getitem__)
        matrix = (
            np.stack(self._rows) if self._rows else np.empty((0, self._dim or 0), dtype=np.float32)
        )
        np.savez(path, ids=np.array(ids), matrix=matrix)

    @classmethod
    def load(cls, path: Path) -> EmbeddingStore:
        """Load a store previously written by :meth:`save`."""
        data = np.load(path, allow_pickle=False)
        store = cls()
        ids = [str(i) for i in data["ids"]]
        if ids:
            store.add(ids, data["matrix"])
        return store
