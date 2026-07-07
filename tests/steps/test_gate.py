"""Tests for the gate orchestrator step (SPEC §6.4)."""

import contextlib
from typing import Any, ClassVar

from ragsynth.domain import Seed, Stratum, SyntheticQuery
from ragsynth.gate.checks.base import CHECKS, CheckResult, GateCheck
from ragsynth.pipeline.base import PipelineState, Resources
from ragsynth.pipeline.registry import RegistryError
from ragsynth.steps.gate import VerificationGate


class _RecordingCheck(GateCheck):
    """Configurable pass/fail check that records the candidates it sees."""

    name = "test.recording"
    seen: ClassVar[list[str]] = []
    fail_texts: ClassVar[set[str]] = set()
    promote: tuple[str, ...] = ()

    def check(
        self, candidate: SyntheticQuery, state: PipelineState, resources: Resources
    ) -> CheckResult:
        type(self).seen.append(candidate.text)
        if candidate.text in self.fail_texts:
            return CheckResult(passed=False, score=0.0, reason="configured failure")
        return CheckResult(passed=True, score=1.0, reason="ok", promoted=self.promote)


class _AfterCheck(_RecordingCheck):
    name = "test.after"
    seen: ClassVar[list[str]] = []
    fail_texts: ClassVar[set[str]] = set()


for _class in (_RecordingCheck, _AfterCheck):
    with contextlib.suppress(RegistryError):
        CHECKS.register(_class.name)(_class)


def _candidate(text: str, chunk_id: str = "c0") -> SyntheticQuery:
    seed = Seed(
        seed_id=f"s-{text}",
        chunk_ids=(chunk_id,),
        cluster_id=0,
        stratum=Stratum(dims={"query_type": "factoid"}),
    )
    return SyntheticQuery(
        query_id=f"q-{text}", text=text, seed=seed, embedding_ref=None, gen_meta={}
    )


def _reset() -> None:
    _RecordingCheck.seen = []
    _RecordingCheck.fail_texts = set()
    _RecordingCheck.promote = ()
    _AfterCheck.seen = []
    _AfterCheck.fail_texts = set()


def test_accepts_and_moves_to_gate_accepted(min_resources: Resources) -> None:
    _reset()
    gate = VerificationGate(min_resources, checks=["test.recording", "test.after"])
    state = PipelineState(candidates=[_candidate("good one"), _candidate("good two")])
    state = gate.run(state)
    assert [c.text for c in state.gate_accepted] == ["good one", "good two"]
    assert state.rejected == []
    assert state.metrics["gate_pass_rate"] == 1.0


def test_short_circuits_on_first_failure(min_resources: Resources) -> None:
    _reset()
    _RecordingCheck.fail_texts = {"bad"}
    gate = VerificationGate(min_resources, checks=["test.recording", "test.after"])
    state = PipelineState(candidates=[_candidate("bad"), _candidate("fine")])
    state = gate.run(state)
    assert _AfterCheck.seen == ["fine"]  # 'bad' never reached the second check
    assert len(state.rejected) == 1
    rejection = state.rejected[0]
    assert rejection.check == "test.recording"
    assert rejection.reason == "configured failure"
    assert state.metrics["gate_reject_reasons"] == {"test.recording": 1}
    assert state.metrics["gate_pass_rate"] == 0.5


def test_promotions_accumulate_into_gen_meta(min_resources: Resources) -> None:
    _reset()
    _RecordingCheck.promote = ("promoted-chunk",)
    gate = VerificationGate(min_resources, checks=["test.recording"])
    state = gate.run(PipelineState(candidates=[_candidate("promoter")]))
    accepted = state.gate_accepted[0]
    assert accepted.gen_meta["promoted"] == ["promoted-chunk"]
    assert accepted.gen_meta["gate"]["test.recording"]["passed"] is True


def test_config_round_trip_with_nested_check_params(min_resources: Resources) -> None:
    config: dict[str, Any] = {
        "checks": ["dedup", "round_trip"],
        "dedup": {"cos_threshold": 0.9},
        "round_trip": {"k": 5},
    }
    gate = VerificationGate.from_config(config, min_resources)
    assert gate.to_config() == config
    rebuilt = VerificationGate.from_config(gate.to_config(), min_resources)
    assert rebuilt.to_config() == config


def test_unknown_check_name_raises(min_resources: Resources) -> None:
    import pytest

    with pytest.raises(RegistryError, match="unknown gate check"):
        VerificationGate(min_resources, checks=["no_such_check"])
