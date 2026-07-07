"""MIPROv2 adapter stub behind the ``optimization`` extra (SPEC §11, §2.2).

Optimizer *execution* is a v2 deliverable (roadmap R2; Opsahl-Ong et al.,
EMNLP 2024): v1 registers the stub so configs can already name
``mipro_v2`` and fail actionably instead of silently.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ragsynth.optimization.base import OPTIMIZERS, BasePromptOptimizer
from ragsynth.optional_deps import require_optional

try:
    import dspy
except ImportError:  # pragma: no cover - exercised via require_optional tests
    dspy = None

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ragsynth.optimization.base import (
        BaseOptimizerConfig,
        OptimizationMetric,
        OptimizationResult,
    )


@OPTIMIZERS.register("mipro_v2")
class MIPROv2Optimizer(BasePromptOptimizer):
    """DSPy MIPROv2 adapter stub (Opsahl-Ong et al., EMNLP 2024).

    Requires the ``optimization`` extra (``uv sync --extra optimization``);
    construction fails actionably without it. ``optimize`` itself is a v2
    deliverable (SPEC §2.2, R2).
    """

    def __init__(self, config: BaseOptimizerConfig) -> None:
        """Check the optional ``dspy`` backend, then store the config.

        Args:
            config: Optimizer configuration.

        Raises:
            ImportError: If ``dspy`` is not installed, with the exact
                install command.
        """
        require_optional(dspy, "MIPROv2Optimizer", "optimization")
        super().__init__(config)

    def optimize(
        self,
        current_prompt: str,
        samples: Sequence[Any] | None,
        metric: OptimizationMetric | None,
    ) -> OptimizationResult:
        """Not implemented in v1 (contract-only release).

        Args:
            current_prompt: The starting prompt text.
            samples: Evaluation dataset (unused in v1).
            metric: Prompt fitness callable (unused in v1).

        Raises:
            NotImplementedError: Always in v1.
        """
        raise NotImplementedError("MIPROv2 execution is a v2 deliverable (SPEC §2.2, R2)")
