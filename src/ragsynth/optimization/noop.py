"""NoOpOptimizer: the identity optimizer that proves the §11 plumbing."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from ragsynth.optimization.base import (
    OPTIMIZERS,
    BasePromptOptimizer,
    OptimizationResult,
    TrialRecord,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ragsynth.optimization.base import OptimizationMetric


@OPTIMIZERS.register("noop")
class NoOpOptimizer(BasePromptOptimizer):
    """Returns the input prompt unchanged (SPEC §11).

    Exercises the full optimization contract -- result shapes, registry
    dispatch, config serialization -- without performing any search. The
    v1 stand-in that keeps call sites honest until v2 backends land.
    """

    def optimize(
        self,
        current_prompt: str,
        samples: Sequence[Any] | None,
        metric: OptimizationMetric | None,
    ) -> OptimizationResult:
        """Evaluate ``current_prompt`` once and return it as the optimum.

        Args:
            current_prompt: The prompt to (not) optimize.
            samples: Ignored; accepted for contract compatibility.
            metric: Scored once on ``current_prompt`` when given;
                otherwise the score is 0.0.

        Returns:
            OptimizationResult where baseline == optimized and the
            improvement is exactly 0.0, with a single trial recorded.
        """
        score = metric(current_prompt) if metric is not None else 0.0
        trial = TrialRecord(
            trial_id=1,
            prompt=current_prompt,
            score=score,
            timestamp=datetime.now(tz=UTC).isoformat(),
        )
        return OptimizationResult(
            optimized_prompt=current_prompt,
            baseline_score=score,
            optimized_score=score,
            improvement=0.0,
            num_trials=1,
            trial_history=[trial],
            optimizer_name="noop",
            config=self.config.model_dump(),
        )
