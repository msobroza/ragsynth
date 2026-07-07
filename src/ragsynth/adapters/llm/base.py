"""ChatModel Protocol and its registry."""

from __future__ import annotations

from typing import Any, Protocol

from ragsynth.pipeline.registry import Registry


class ChatModel(Protocol):
    """Minimal chat-completion contract (SPEC §12).

    Implementations target OpenAI-compatible endpoints (vLLM/LiteLLM/internal
    gateways) or are fully offline mocks.
    """

    def complete(self, system: str, user: str, **kwargs: Any) -> str:
        """Return the assistant text for a (system, user) exchange."""
        ...


CHAT_MODELS: Registry[Any] = Registry("chat model")
