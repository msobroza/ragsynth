"""Curator: stratified subsample, final dedup, memorization audit (SPEC §6.6)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self

import numpy as np

from ragsynth.pipeline.base import STEPS, PipelineStep

if TYPE_CHECKING:
    from ragsynth.domain import AnnotationRecord
    from ragsynth.pipeline.base import PipelineState, Resources


@STEPS.register("curator")
class Curator(PipelineStep):
    """Final curation of the accepted record set.

    Memorization: records whose query embedding has cosine >=
    ``memorization_cos`` to ANY production train query are FLAGGED (kept,
    not dropped) -- Chroma's verbatim-reproduction audit semantics.
    ``target_mix`` maps stratum keys (``Stratum.key()``) to target
    fractions; the binding stratum determines the total.
    """

    name = "curator"

    def __init__(
        self,
        resources: Resources,
        memorization_cos: float = 0.9,
        target_mix: dict[str, float] | None = None,
        max_records: int | None = None,
    ) -> None:
        self._resources = resources
        self.memorization_cos = memorization_cos
        self.target_mix = target_mix
        self.max_records = max_records

    def _dedup(self, records: list[AnnotationRecord]) -> list[AnnotationRecord]:
        seen: set[str] = set()
        out: list[AnnotationRecord] = []
        for record in records:
            if record.query.text in seen:
                continue
            seen.add(record.query.text)
            out.append(record)
        return out

    def _flag_memorization(self, records: list[AnnotationRecord]) -> list[AnnotationRecord]:
        resources = self._resources
        train_ids = [q.query_id for q in resources.queries_train]
        if not train_ids:
            return records
        train_embs = resources.embeddings.get(train_ids).astype(np.float64)
        out: list[AnnotationRecord] = []
        for record in records:
            ref = record.query.embedding_ref
            if ref is None or ref not in resources.embeddings:
                out.append(record)
                continue
            emb = resources.embeddings.get([ref])[0].astype(np.float64)
            max_cos = float(np.max(train_embs @ emb))
            if max_cos >= self.memorization_cos:
                out.append(
                    record.model_copy(
                        update={
                            "gate_meta": {
                                **record.gate_meta,
                                "memorization_flag": True,
                                "memorization_cos": max_cos,
                            }
                        }
                    )
                )
            else:
                out.append(record)
        return out

    def _apply_mix(
        self, records: list[AnnotationRecord], rng: np.random.Generator
    ) -> list[AnnotationRecord]:
        if self.target_mix is None:
            return records
        by_stratum: dict[str, list[AnnotationRecord]] = {}
        for record in records:
            by_stratum.setdefault(record.stratum.key(), []).append(record)
        # The binding stratum caps the total: total = min_s count_s / mix_s.
        total = min(
            len(by_stratum.get(key, [])) / frac for key, frac in self.target_mix.items() if frac > 0
        )
        keep: set[str] = set()
        for key, frac in self.target_mix.items():
            pool = by_stratum.get(key, [])
            n_keep = round(total * frac)
            picks = rng.choice(len(pool), size=min(n_keep, len(pool)), replace=False)
            keep.update(pool[int(i)].record_id for i in picks)
        return [r for r in records if r.record_id in keep]

    def run(self, state: PipelineState) -> PipelineState:
        """Dedup, audit, and subsample ``state.accepted`` in place."""
        rng = self._resources.rng(self.name)
        records = self._dedup(state.accepted)
        records = self._flag_memorization(records)
        records = self._apply_mix(records, rng)
        if self.max_records is not None and len(records) > self.max_records:
            picks = sorted(rng.choice(len(records), size=self.max_records, replace=False).tolist())
            records = [records[i] for i in picks]
        state.metrics["curator_memorization_flags"] = sum(
            1 for r in records if r.gate_meta.get("memorization_flag")
        )
        state.accepted = records
        return state

    def to_config(self) -> dict[str, Any]:
        """JSON-safe constructor params."""
        return {
            "memorization_cos": self.memorization_cos,
            "target_mix": self.target_mix,
            "max_records": self.max_records,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any], resources: Resources) -> Self:
        """Build from a config params block."""
        return cls(resources, **config)
