"""Prompt-optimization abstraction (SPEC §11; v1 = contract only, no execution).

Importing this package registers the built-in optimizers (``noop``,
``mipro_v2``) in :data:`~ragsynth.optimization.base.OPTIMIZERS`, so
:func:`~ragsynth.optimization.base.create_optimizer` works out of the box.
"""

from ragsynth.optimization import mipro_adapter
from ragsynth.optimization.base import (
    OPTIMIZERS,
    BaseOptimizerConfig,
    BasePromptOptimizer,
    OptimizationMetric,
    OptimizationResult,
    TrialRecord,
    create_optimizer,
)
from ragsynth.optimization.mipro_adapter import MIPROv2Optimizer
from ragsynth.optimization.noop import NoOpOptimizer
from ragsynth.optimization.objectives import HARD_PENALTY, FidelityObjective

__all__ = [
    "HARD_PENALTY",
    "OPTIMIZERS",
    "BaseOptimizerConfig",
    "BasePromptOptimizer",
    "FidelityObjective",
    "MIPROv2Optimizer",
    "NoOpOptimizer",
    "OptimizationMetric",
    "OptimizationResult",
    "TrialRecord",
    "create_optimizer",
    "mipro_adapter",
]
