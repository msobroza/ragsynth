"""Pipeline steps, self-registered on import (SPEC §6)."""

from ragsynth.steps.context_assembler import ContextAssembler
from ragsynth.steps.curator import Curator
from ragsynth.steps.gate import VerificationGate
from ragsynth.steps.generator import QueryGenerator
from ragsynth.steps.qrel_builder import QrelBuilder
from ragsynth.steps.seed_sampler import (
    QuotaSeedSampler,
    SpecSeedSampler,
    UniformSeedSampler,
)
from ragsynth.steps.validator import Validator

__all__ = [
    "ContextAssembler",
    "Curator",
    "QrelBuilder",
    "QueryGenerator",
    "QuotaSeedSampler",
    "SpecSeedSampler",
    "UniformSeedSampler",
    "Validator",
    "VerificationGate",
]
