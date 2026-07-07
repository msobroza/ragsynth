"""EvalReport: the validator's frozen output (per-arm metric blocks)."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ragsynth.io.artifacts import canonical_json


class EvalReport(BaseModel):
    """Per-arm fidelity/efficiency/validity/diversity blocks + provenance (SPEC §4, §6.7).

    ``arms`` maps arm name -> metric block (nested plain dicts so the report
    serializes to ``metrics.json`` without loss). ``provenance`` carries
    wall-clock and environment details and is EXCLUDED from ``metrics.json``
    so the determinism criterion (SPEC §15.1) is byte-exact (PLAN D14).
    """

    model_config = ConfigDict(frozen=True)

    name: str
    schema_version: int = 1
    config: dict[str, Any]
    config_hash: str
    seed: int
    arms: dict[str, dict[str, Any]]
    gates: dict[str, float]
    gates_passed: dict[str, bool]
    provenance: dict[str, Any] = Field(default_factory=dict)

    def to_json(self) -> str:
        """Serialize the full report (including provenance) to canonical JSON."""
        return canonical_json(self.model_dump(mode="json"))

    def metrics_payload(self) -> dict[str, Any]:
        """Deterministic subset for ``metrics.json`` (no provenance/wall-clock)."""
        payload = self.model_dump(mode="json")
        payload.pop("provenance")
        return payload
