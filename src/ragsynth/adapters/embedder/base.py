"""Embedder Protocol and its registry."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from ragsynth.pipeline.registry import Registry

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    from numpy.typing import NDArray


class Embedder(Protocol):
    """Batch text -> L2-normalized embedding matrix (SPEC §12)."""

    def encode(self, texts: Sequence[str]) -> NDArray[np.float64]:
        """Return an ``(len(texts), d)`` matrix of unit-norm rows."""
        ...


EMBEDDERS: Registry[Any] = Registry("embedder")
