"""Verification gate orchestrator: ordered checks, short-circuit, tallies (SPEC §6.4)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self

from ragsynth.domain import Rejection
from ragsynth.gate.checks.base import CHECKS
from ragsynth.pipeline.base import STEPS, PipelineStep

if TYPE_CHECKING:
    from ragsynth.gate.checks.base import GateCheck
    from ragsynth.pipeline.base import PipelineState, Resources


@STEPS.register("gate")
class VerificationGate(PipelineStep):
    """Run the configured checks cheap-to-expensive over every candidate.

    First failing check rejects the candidate (short-circuit); reject
    reasons are tallied into ``state.metrics['gate_reject_reasons']`` --
    the v2 prompt-optimizer's routing signal (SPEC §6.4, R2). Uniqueness
    promotions accumulate in the accepted candidate's
    ``gen_meta['promoted']`` for the qrel builder.
    """

    name = "gate"

    def __init__(
        self,
        resources: Resources,
        checks: list[str],
        **check_params: dict[str, Any],
    ) -> None:
        self._resources = resources
        self.check_names = list(checks)
        self.check_params = {name: dict(check_params.get(name, {})) for name in checks}
        self._checks: list[GateCheck] = [
            CHECKS.get(name).from_config(self.check_params[name], resources)
            for name in self.check_names
        ]

    def run(self, state: PipelineState) -> PipelineState:
        """Filter ``state.candidates`` into ``gate_accepted``/``rejected``."""
        reject_reasons: dict[str, int] = state.metrics.setdefault("gate_reject_reasons", {})
        for candidate in state.candidates:
            promoted: list[str] = []
            gate_log: dict[str, Any] = {}
            rejected = False
            for check in self._checks:
                result = check.check(candidate, state, self._resources)
                gate_log[check.name] = {
                    "passed": result.passed,
                    "score": result.score,
                    "reason": result.reason,
                }
                if not result.passed:
                    reject_reasons[check.name] = reject_reasons.get(check.name, 0) + 1
                    state.rejected.append(
                        Rejection(
                            candidate=candidate,
                            check=check.name,
                            reason=result.reason,
                            score=result.score,
                        )
                    )
                    rejected = True
                    break
                promoted.extend(result.promoted)
            if not rejected:
                accepted = candidate.model_copy(
                    update={
                        "gen_meta": {
                            **candidate.gen_meta,
                            "gate": gate_log,
                            "promoted": promoted,
                        }
                    }
                )
                state.gate_accepted.append(accepted)
        total = len(state.candidates)
        state.metrics["gate_pass_rate"] = len(state.gate_accepted) / total if total else 0.0
        return state

    def to_config(self) -> dict[str, Any]:
        """The §13 config shape: check list + per-check param blocks."""
        config: dict[str, Any] = {"checks": list(self.check_names)}
        for name, check in zip(self.check_names, self._checks, strict=True):
            params = check.to_config()
            if params:
                config[name] = params
        return config

    @classmethod
    def from_config(cls, config: dict[str, Any], resources: Resources) -> Self:
        """Build from a config params block (per-check dicts nested by name)."""
        params = dict(config)
        checks = params.pop("checks")
        return cls(resources, checks=checks, **params)
