"""AnnotationRecord: THE canonical output of the pipeline."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ragsynth.domain.candidate import SyntheticQuery
from ragsynth.domain.conversation import Turn
from ragsynth.domain.stratum import Stratum


class AnnotationRecord(BaseModel):
    """One gate-verified synthetic query with its relevance annotations (SPEC §4).

    ``qrels`` maps chunk_id -> grade ({0, 1} in v1); gate-promoted golds
    from the uniqueness check are included. ``crucial`` is filled by the
    v3 masking-ablation check (v1: all gold). ``content_hashes`` snapshot
    the annotated chunks' text hashes so the v2 churn lifecycle needs no
    migration. ``dialogue_context`` is ``None`` in v1 (conversational-ready).
    """

    model_config = ConfigDict(frozen=True)

    record_id: str
    query: SyntheticQuery
    qrels: dict[str, int]
    crucial: tuple[str, ...] = ()
    supplemental: tuple[str, ...] = ()
    stratum: Stratum
    dialogue_context: tuple[Turn, ...] | None = None
    gate_meta: dict[str, Any] = Field(default_factory=dict)
    content_hashes: dict[str, str] = Field(default_factory=dict)
    benchmark_version: str
    created_at: datetime
