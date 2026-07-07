"""Frozen pydantic v2 domain model (SPEC §4) -- conversation- and visual-ready."""

from ragsynth.domain.annotation import AnnotationRecord
from ragsynth.domain.candidate import SyntheticQuery
from ragsynth.domain.chunk import Chunk
from ragsynth.domain.context import GenerationContext
from ragsynth.domain.conversation import Turn
from ragsynth.domain.query import ProductionQuery
from ragsynth.domain.rejection import Rejection
from ragsynth.domain.report import EvalReport
from ragsynth.domain.seed import Seed
from ragsynth.domain.stratum import Stratum

__all__ = [
    "AnnotationRecord",
    "Chunk",
    "EvalReport",
    "GenerationContext",
    "ProductionQuery",
    "Rejection",
    "Seed",
    "Stratum",
    "SyntheticQuery",
    "Turn",
]
