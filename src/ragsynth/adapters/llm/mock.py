"""Deterministic offline ChatModel for tests/CI (SPEC §6.3, §12)."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

from ragsynth.adapters.llm.base import CHAT_MODELS

if TYPE_CHECKING:
    import numpy as np

    from ragsynth.datasets.base import DatasetBundle


@CHAT_MODELS.register("mock")
class MockChatModel:
    """Hash-seeded deterministic chat model.

    Same ``(seed, system, user)`` always yields the same templated text, so
    the full pipeline runs offline and reproducibly. The output embeds a
    short digest plus the tail of the user prompt, giving distinct texts for
    distinct prompts (dedup/diversity metrics stay meaningful).
    """

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def complete(self, system: str, user: str, **kwargs: Any) -> str:
        """Return deterministic templated text for the exchange.

        Echoes words from ``[chunk]`` evidence lines when present so that
        surface-similarity embedders (hashed n-grams) place mock queries
        near their gold chunks -- keeping the round-trip gate meaningful
        in fully-offline runs.
        """
        digest = hashlib.sha256(f"{self.seed}|{system}|{user}".encode()).hexdigest()[:8]
        chunk_words = [
            word
            for line in user.splitlines()
            if line.startswith("[chunk]")
            for word in line.split()[1:]
        ]
        salient = " ".join(chunk_words[:12]) if chunk_words else " ".join(user.split()[-6:])
        return f"what about {salient} ({digest})?"

    def to_config(self) -> dict[str, Any]:
        """JSON-safe constructor params."""
        return {"seed": self.seed}

    @classmethod
    def from_config(
        cls, params: dict[str, Any], bundle: DatasetBundle, rng: np.random.Generator
    ) -> MockChatModel:
        """Build from a config params block (composition-root factory contract)."""
        return cls(seed=int(params.get("seed", 0)))
