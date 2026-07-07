"""LLM chat adapters: Protocol, OpenAI-compatible client, offline mock."""

from ragsynth.adapters.llm.base import CHAT_MODELS, ChatModel
from ragsynth.adapters.llm.mock import MockChatModel

__all__ = ["CHAT_MODELS", "ChatModel", "MockChatModel"]
