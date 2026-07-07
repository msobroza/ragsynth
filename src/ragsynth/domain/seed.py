"""Seed: what one synthetic query is generated from."""

from pydantic import BaseModel, ConfigDict

from ragsynth.domain.stratum import Stratum


class Seed(BaseModel):
    """Generation seed (SPEC §4, §6.1).

    ``chunk_ids`` of length 1 is a single-chunk seed; length > 1 is a
    chunk-group (multi-evidence). ``z`` is the A2 target embedding in the
    planning space (spec-first sampling); ``None`` for chunk-first arms.
    """

    model_config = ConfigDict(frozen=True)

    seed_id: str
    chunk_ids: tuple[str, ...]
    cluster_id: int
    stratum: Stratum
    z: list[float] | None = None
