"""Prompt-optimization contract: metric Protocol, result types, ABC, registry (SPEC §11).

v1 ships the CONTRACT ONLY -- no optimizer executes here. Mirrors
``healthbench_agent/prompt_optimization/optimizer.py`` in spirit: frozen
result dataclasses for immutability and easy serialization, a structural
:class:`OptimizationMetric` Protocol, a :class:`BasePromptOptimizer` ABC
that backend adapters subclass, and a registry factory. Concrete v2
backends (DSPy MIPROv2, TextGrad) plug in behind the ``optimization``
extra (SPEC §2.2, roadmap R2).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import BaseModel, ConfigDict

from ragsynth.pipeline.registry import Registry

if TYPE_CHECKING:
    from collections.abc import Sequence


class OptimizationMetric(Protocol):
    """Callable contract shared by every prompt fitness function.

    Objectives (e.g. :class:`~ragsynth.optimization.objectives.FidelityObjective`)
    satisfy this Protocol structurally -- no inheritance required.
    """

    def __call__(self, prompt: str) -> float:
        """Score a candidate prompt.

        Args:
            prompt: The candidate prompt text to evaluate.

        Returns:
            A scalar fitness score where higher is better.
        """
        ...


@dataclass(frozen=True)
class TrialRecord:
    """Single optimization trial.

    Attributes:
        trial_id: Sequential trial number (1-based).
        prompt: The candidate prompt evaluated in this trial.
        score: Evaluation score, or ``None`` when no metric was supplied.
        timestamp: ISO 8601 timestamp of when the trial completed.
    """

    trial_id: int
    prompt: str
    score: float | None
    timestamp: str


@dataclass(frozen=True)
class OptimizationResult:
    """Result of a prompt-optimization run.

    Attributes:
        optimized_prompt: The best prompt found during optimization.
        baseline_score: Score of the original prompt before optimization.
        optimized_score: Score of the best prompt found.
        improvement: Score delta (``optimized_score - baseline_score``).
        num_trials: Total number of candidate prompts evaluated.
        trial_history: Per-trial details for reproducibility.
        optimizer_name: Registry key of the optimizer used.
        config: Serialized optimizer configuration for reproducibility.
    """

    optimized_prompt: str
    baseline_score: float
    optimized_score: float
    improvement: float
    num_trials: int
    trial_history: list[TrialRecord]
    optimizer_name: str
    config: dict[str, Any]


class BaseOptimizerConfig(BaseModel):
    """Common optimizer knobs; each backend adapter subclasses with its own params.

    Attributes:
        max_trials: Maximum number of candidate prompts to evaluate.
        seed: Base seed for any stochastic search the backend performs.
    """

    model_config = ConfigDict(frozen=True)

    max_trials: int = 10
    seed: int = 0


class BasePromptOptimizer(ABC):
    """Abstract base every prompt-optimization backend implements (SPEC §11).

    Subclasses implement the search for a specific framework (DSPy,
    TextGrad, critique-refine) and return an :class:`OptimizationResult`.
    """

    def __init__(self, config: BaseOptimizerConfig) -> None:
        """Store the optimizer configuration.

        Args:
            config: Optimizer configuration; concrete subclass varies
                per backend adapter.
        """
        self.config = config

    @abstractmethod
    def optimize(
        self,
        current_prompt: str,
        samples: Sequence[Any] | None,
        metric: OptimizationMetric | None,
    ) -> OptimizationResult:
        """Optimize a prompt against a scoring metric.

        Args:
            current_prompt: The starting prompt text.
            samples: Evaluation dataset; backends that score end-to-end
                require it, mutation-only backends may accept ``None``.
            metric: Callable scoring a candidate prompt (higher is
                better); same optionality as ``samples``.

        Returns:
            OptimizationResult with the best prompt and trial history.
        """


OPTIMIZERS: Registry[Any] = Registry("prompt optimizer")
"""Prompt-optimizer registry; adapters self-register at import time (SPEC §3.3)."""


def create_optimizer(name: str, config: BaseOptimizerConfig) -> BasePromptOptimizer:
    """Instantiate the optimizer registered under ``name``.

    Args:
        name: Registry key (e.g. ``"noop"``, ``"mipro_v2"``).
        config: Configuration forwarded to the optimizer's constructor.

    Returns:
        A ready-to-use optimizer instance.

    Raises:
        RegistryError: If ``name`` is unknown; the message lists known keys.
        ImportError: If the optimizer's optional backend is not installed.
    """
    optimizer_class = OPTIMIZERS.get(name)
    optimizer: BasePromptOptimizer = optimizer_class(config)
    return optimizer
