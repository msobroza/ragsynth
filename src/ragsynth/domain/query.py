"""ProductionQuery: one observed real-traffic query."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict

from ragsynth.domain.stratum import Stratum


class ProductionQuery(BaseModel):
    """A production query used for demand estimation, exemplars, validation.

    ``timestamp`` feeds the exponential time-decay demand weighting
    (SPEC §7.3); ``stratum`` is optional v1 metadata (real traffic is
    usually unlabeled).
    """

    model_config = ConfigDict(frozen=True)

    query_id: str
    text: str
    timestamp: datetime | None = None
    embedding_ref: str | None = None
    stratum: Stratum | None = None
