"""GateCheck ABC, CheckResult, and the check registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Self

from ragsynth.pipeline.registry import Registry

if TYPE_CHECKING:
    from ragsynth.domain import SyntheticQuery
    from ragsynth.pipeline.base import PipelineState, Resources


@dataclass(frozen=True)
class CheckResult:
    """Outcome of one gate check on one candidate (SPEC §6.4).

    Attributes:
        passed: Whether the candidate survives this check.
        score: Check-specific diagnostic score (may be ``None``).
        reason: Human-readable pass/reject explanation (rejections feed the
            ``gate_reject_reasons`` tally -- the v2 optimizer routing signal).
        promoted: Chunk ids the uniqueness check promotes into the qrels.
    """

    passed: bool
    score: float | None
    reason: str
    promoted: tuple[str, ...] = ()


class GateCheck(ABC):
    """One verification check; the gate orchestrator runs them cheap-to-expensive.

    Checks are separate classes, not branches (SPEC §3.2 S-rule), registered
    in ``CHECKS`` so configs list them by key and the contract test can
    enumerate them.
    """

    name: ClassVar[str]

    @abstractmethod
    def check(
        self, candidate: SyntheticQuery, state: PipelineState, resources: Resources
    ) -> CheckResult:
        """Evaluate one candidate against this check."""

    def to_config(self) -> dict[str, Any]:
        """JSON-safe constructor params (default: parameterless)."""
        return {}

    @classmethod
    def from_config(cls, config: dict[str, Any], resources: Resources) -> Self:
        """Build from a config params block (default: ``cls(**config)``)."""
        return cls(**config)


CHECKS: Registry[GateCheck] = Registry("gate check")
