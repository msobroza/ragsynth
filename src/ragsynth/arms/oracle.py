"""ORACLE: held-out real queries -- the tau ceiling (SPEC §10)."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from ragsynth.arms.base import ARMS, ArmPreset
from ragsynth.domain import AnnotationRecord, Seed, Stratum, SyntheticQuery

if TYPE_CHECKING:
    from ragsynth.pipeline.base import Resources


@ARMS.register("oracle")
class OraclePreset(ArmPreset):
    """Random subsample of the oracle split with its qrels; no generation.

    The oracle's tau against the anchor is itself noisy at benchmark-sized
    N -- the ceiling has a CI (prototype epilogue).
    """

    name = "oracle"

    def run(self, resources: Resources, params: dict[str, Any]) -> list[AnnotationRecord]:
        """Sample real held-out queries verbatim as records."""
        n = int(params.get("n_records", params.get("n_seeds", 200)))
        rng = resources.rng("arm.oracle")
        pool = resources.queries_oracle
        picks = rng.choice(len(pool), size=min(n, len(pool)), replace=False)
        created_at = datetime.now(tz=UTC)
        records: list[AnnotationRecord] = []
        for i in sorted(int(p) for p in picks):
            query = pool[i]
            qrels = dict(resources.oracle_qrels.get(query.query_id, {}))
            if not qrels:
                continue
            stratum = query.stratum or Stratum(dims={"query_type": "real"})
            seed = Seed(
                seed_id=f"oracle-{query.query_id}",
                chunk_ids=tuple(sorted(qrels)),
                cluster_id=int(
                    resources.partition.assign(
                        resources.embeddings.get([query.query_id]).astype(float)
                    )[0]
                ),
                stratum=stratum,
            )
            synthetic = SyntheticQuery(
                query_id=query.query_id,
                text=query.text,
                seed=seed,
                embedding_ref=query.query_id,
                gen_meta={"arm": "oracle"},
            )
            record_id = hashlib.sha256(query.query_id.encode()).hexdigest()[:16]
            records.append(
                AnnotationRecord(
                    record_id=f"rec-oracle-{record_id}",
                    query=synthetic,
                    qrels=qrels,
                    crucial=tuple(sorted(qrels)),
                    stratum=stratum,
                    gate_meta={},
                    content_hashes={
                        cid: resources.chunk_index[cid].content_hash
                        for cid in qrels
                        if cid in resources.chunk_index
                    },
                    benchmark_version="arm-oracle",
                    created_at=created_at,
                )
            )
        return records
