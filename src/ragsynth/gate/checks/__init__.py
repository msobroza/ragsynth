"""The five v1 gate checks, self-registered on import (SPEC §6.4)."""

from ragsynth.gate.checks.answerability import AnswerabilityCheck
from ragsynth.gate.checks.base import CHECKS, CheckResult, GateCheck
from ragsynth.gate.checks.dedup import DedupCheck
from ragsynth.gate.checks.round_trip import RoundTripCheck
from ragsynth.gate.checks.uniqueness import UniquenessCheck
from ragsynth.gate.checks.zero_context import ZeroContextCheck

__all__ = [
    "CHECKS",
    "AnswerabilityCheck",
    "CheckResult",
    "DedupCheck",
    "GateCheck",
    "RoundTripCheck",
    "UniquenessCheck",
    "ZeroContextCheck",
]
