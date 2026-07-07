"""Turn: one dialogue turn (v1: model only, unused -- conversational-ready)."""

from typing import Literal

from pydantic import BaseModel, ConfigDict


class Turn(BaseModel):
    """A single dialogue turn for future conversational strata (SPEC §4, R8)."""

    model_config = ConfigDict(frozen=True)

    role: Literal["user", "assistant"]
    text: str
