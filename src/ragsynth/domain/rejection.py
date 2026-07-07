"""Rejection: a gate-rejected candidate with the reason."""

from pydantic import BaseModel, ConfigDict

from ragsynth.domain.candidate import SyntheticQuery


class Rejection(BaseModel):
    """A candidate rejected by a verification gate check (SPEC §4, §6.4).

    The per-check tallies of these are the v2 prompt-optimizer's routing
    signal (SPEC §11, R2).
    """

    model_config = ConfigDict(frozen=True)

    candidate: SyntheticQuery
    check: str
    reason: str
    score: float | None
