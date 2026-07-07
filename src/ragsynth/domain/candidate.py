"""SyntheticQuery: one generated candidate query."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ragsynth.domain.seed import Seed


class SyntheticQuery(BaseModel):
    """A generated query candidate (pre- or post-gate) (SPEC §4).

    ``gen_meta`` records generation provenance: model, prompt_version,
    cos_to_target (A2), n_candidates, revision count -- the v2 optimizer's
    routing signal rides here.
    """

    model_config = ConfigDict(frozen=True)

    query_id: str
    text: str
    seed: Seed
    embedding_ref: str | None
    gen_meta: dict[str, Any] = Field(default_factory=dict)
