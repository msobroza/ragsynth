"""Composition root: config loading, validation, and Resources/Pipeline assembly.

This module is the ONLY place adapters and fitted artifacts are constructed
(SPEC §3.1 DIP rule) and ``load_config``/``dump_config`` define the byte-stable
YAML round-trip (SPEC §13). The reference partition and movMF demand map are
fit here per run (deterministic under the config seed) and persisted through
the ArtifactStore as versioned frozen artifacts (SPEC §7, PLAN D19).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml

from ragsynth.datasets.base import DATASETS, DatasetBundle
from ragsynth.io.artifacts import ArtifactStore, canonical_json, sha256_hex
from ragsynth.io.embeddings import EmbeddingStore
from ragsynth.pipeline.base import STEPS, DemandArtifact, Resources, stable_hash64
from ragsynth.pipeline.pipeline import Pipeline

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ragsynth.domain import Chunk, ProductionQuery

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

_DEFAULT_PARTITION: dict[str, Any] = {"n_clusters": 8}
_DEFAULT_DEMAND: dict[str, Any] = {
    "n_components": 16,
    "lam": 0.7,
    "tau_r_pct": 5.0,
    "half_life": None,
}


def load_config(path: Path) -> dict[str, Any]:
    """Load and validate a run config (SPEC §13 schema v1).

    Raises:
        ValueError: On schema-version mismatch or missing required blocks.
        RegistryError: If any ``type`` key is unknown (message lists known keys).
    """
    config: dict[str, Any] = yaml.safe_load(Path(path).read_text())
    validate_config(config)
    return config


def dump_config(config: dict[str, Any]) -> str:
    """Serialize a config byte-stably (sorted keys, SPEC §13 round-trip rule)."""
    return yaml.safe_dump(config, sort_keys=True, default_flow_style=False)


def config_hash(config: dict[str, Any]) -> str:
    """sha256 of the canonical-JSON form of the config."""
    return sha256_hex(canonical_json(config).encode("utf-8"))


def _llm_family(block: dict[str, Any]) -> str:
    """Best-effort model-family identity of an adapter config block."""
    params = block.get("params") or {}
    return str(params.get("model", block.get("type", "")))


def validate_config(config: dict[str, Any]) -> list[str]:
    """Validate schema, registry keys, and the cross-family judge rule (SPEC §6.4).

    Returns:
        Human-readable warnings (also logged).

    Raises:
        ValueError: On schema-version mismatch or missing required blocks.
        RegistryError: If any ``type`` is not registered.
    """
    meta = config.get("ragsynth") or {}
    if meta.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported schema_version {meta.get('schema_version')!r}; expected {SCHEMA_VERSION}"
        )
    for key in ("resources", "pipeline", "artifacts_dir"):
        if key not in config:
            raise ValueError(f"config is missing the required '{key}' block")

    resources = config["resources"]
    from ragsynth.adapters.embedder.base import EMBEDDERS
    from ragsynth.adapters.judge.base import JUDGES
    from ragsynth.adapters.llm.base import CHAT_MODELS
    from ragsynth.adapters.retriever.base import RETRIEVERS

    DATASETS.get(resources["dataset"]["type"])
    EMBEDDERS.get(resources["embedder"]["type"])
    CHAT_MODELS.get(resources["generator_llm"]["type"])
    JUDGES.get(resources["judge_llm"]["type"])
    RETRIEVERS.get(resources["retriever"]["type"])
    for step in config["pipeline"]:
        STEPS.get(step["type"])

    warnings: list[str] = []
    judge_block = resources["judge_llm"]
    judge_chat = (judge_block.get("params") or {}).get("chat", judge_block)
    generator_family = _llm_family(resources["generator_llm"])
    judge_family = _llm_family(judge_chat)
    if generator_family and generator_family == judge_family:
        message = (
            f"judge and generator share the model family '{generator_family}': LLM assessors "
            "overestimate same-family systems by 9-17 rank positions (Froebe et al., SIGIR 2025); "
            "prefer a cross-family judge"
        )
        warnings.append(message)
        logger.warning(message)
    return warnings


def _ensure_embedded(
    store: EmbeddingStore,
    embedder: Any,
    ids: list[str],
    texts: list[str],
) -> None:
    """Encode and register any ids missing from the store."""
    missing = [(i, t) for i, t in zip(ids, texts, strict=True) if i not in store]
    if missing:
        store.add([i for i, _ in missing], embedder.encode([t for _, t in missing]))


def _nearest_chunk_qrels(
    query_ids: list[str],
    query_embs: NDArray[np.float64],
    chunks: tuple[Chunk, ...],
    chunk_embs: NDArray[np.float64],
) -> dict[str, dict[str, int]]:
    """Gate-style nearest-chunk gold relabeling fallback (SPEC §10, PLAN D17)."""
    nearest = np.argmax(query_embs @ chunk_embs.T, axis=1)
    return {qid: {chunks[int(c)].chunk_id: 1} for qid, c in zip(query_ids, nearest, strict=True)}


def build_resources(config: dict[str, Any]) -> Resources:
    """Assemble the frozen Resources bundle from a validated config.

    Order matters: dataset -> embedder -> fill embedding store -> partition +
    demand artifacts (train split only, PLAN D10) -> zoo -> adapters.
    """
    from ragsynth.adapters.embedder.base import EMBEDDERS
    from ragsynth.adapters.judge.base import JUDGES
    from ragsynth.adapters.llm.base import CHAT_MODELS
    from ragsynth.adapters.retriever.base import RETRIEVERS
    from ragsynth.metrics.validity.systems import make_system_zoo
    from ragsynth.sampling.demand import (
        demand_from_responsibilities,
        nn_cos_threshold,
        tilt_weights,
    )
    from ragsynth.sampling.movmf import MovMF
    from ragsynth.sampling.partition import ReferencePartition

    resources_cfg = config["resources"]
    seed = int(config["ragsynth"]["seed"])
    rng = np.random.default_rng([seed, stable_hash64("composition")])

    dataset_cfg = resources_cfg["dataset"]
    bundle: DatasetBundle = DATASETS.get(dataset_cfg["type"]).build(
        dataset_cfg.get("params") or {}, seed
    )

    store = bundle.embeddings if bundle.embeddings is not None else EmbeddingStore()
    bundle = DatasetBundle(
        chunks=bundle.chunks,
        queries_train=bundle.queries_train,
        queries_anchor=bundle.queries_anchor,
        queries_oracle=bundle.queries_oracle,
        anchor_qrels=bundle.anchor_qrels,
        oracle_qrels=bundle.oracle_qrels,
        embeddings=store,
        bank=bundle.bank,
    )

    embedder_cfg = resources_cfg["embedder"]
    embedder = EMBEDDERS.get(embedder_cfg["type"]).from_config(
        embedder_cfg.get("params") or {}, bundle, rng
    )

    _ensure_embedded(
        store, embedder, [c.chunk_id for c in bundle.chunks], [c.text for c in bundle.chunks]
    )
    for split in (bundle.queries_train, bundle.queries_anchor, bundle.queries_oracle):
        _ensure_embedded(store, embedder, [q.query_id for q in split], [q.text for q in split])

    artifacts = ArtifactStore(Path(config["artifacts_dir"]))
    chunk_ids = [c.chunk_id for c in bundle.chunks]
    chunk_embs = store.get(chunk_ids).astype(np.float64)
    train_embs = store.get([q.query_id for q in bundle.queries_train]).astype(np.float64)

    partition_cfg = {**_DEFAULT_PARTITION, **(resources_cfg.get("partition") or {})}
    partition = ReferencePartition.fit(
        train_embs, n_clusters=int(partition_cfg["n_clusters"]), seed=seed
    )
    partition.to_artifact(artifacts.path_for(f"partition-c{partition.n_clusters}.npz"))
    artifacts.record_file(f"partition-c{partition.n_clusters}.npz")

    demand_cfg = {**_DEFAULT_DEMAND, **(resources_cfg.get("demand") or {})}
    movmf = MovMF(n_components=int(demand_cfg["n_components"]), seed=seed).fit(train_embs)
    movmf.to_artifact(artifacts.path_for("demand-movmf.npz"))
    artifacts.record_file("demand-movmf.npz")
    timestamps = _train_timestamps(bundle.queries_train, demand_cfg["half_life"])
    movmf_demand = demand_from_responsibilities(
        movmf.responsibilities(train_embs),
        timestamps=timestamps,
        half_life=demand_cfg["half_life"],
    )
    lam = float(demand_cfg["lam"])
    demand = DemandArtifact(
        p_hat=partition.proportions(train_embs),
        movmf=movmf,
        movmf_demand=movmf_demand,
        tilted=tilt_weights(movmf_demand, lam),
        tau_r=nn_cos_threshold(train_embs, pct=float(demand_cfg["tau_r_pct"])),
        lam=lam,
    )

    anchor_qrels = bundle.anchor_qrels or _nearest_chunk_qrels(
        [q.query_id for q in bundle.queries_anchor],
        store.get([q.query_id for q in bundle.queries_anchor]).astype(np.float64),
        bundle.chunks,
        chunk_embs,
    )
    oracle_qrels = bundle.oracle_qrels or _nearest_chunk_qrels(
        [q.query_id for q in bundle.queries_oracle],
        store.get([q.query_id for q in bundle.queries_oracle]).astype(np.float64),
        bundle.chunks,
        chunk_embs,
    )

    generator_cfg = resources_cfg["generator_llm"]
    judge_cfg = resources_cfg["judge_llm"]
    retriever_cfg = resources_cfg["retriever"]
    return Resources(
        chunks=bundle.chunks,
        queries_train=bundle.queries_train,
        queries_anchor=bundle.queries_anchor,
        queries_oracle=bundle.queries_oracle,
        anchor_qrels=anchor_qrels,
        oracle_qrels=oracle_qrels,
        embedder=embedder,
        generator_llm=CHAT_MODELS.get(generator_cfg["type"]).from_config(
            generator_cfg.get("params") or {}, bundle, rng
        ),
        judge=JUDGES.get(judge_cfg["type"]).from_config(judge_cfg.get("params") or {}, bundle, rng),
        retriever=RETRIEVERS.get(retriever_cfg["type"]).from_config(
            retriever_cfg.get("params") or {}, bundle, rng
        ),
        embeddings=store,
        partition=partition,
        demand=demand,
        zoo=make_system_zoo(chunk_ids, chunk_embs, seed=seed),
        artifacts=artifacts,
        seed=seed,
        bundle=bundle,
    )


def _train_timestamps(
    queries: tuple[ProductionQuery, ...], half_life: float | None
) -> NDArray[np.float64] | None:
    """Epoch-seconds timestamps when decay is configured and data has them."""
    if half_life is None:
        return None
    if any(q.timestamp is None for q in queries):
        logger.warning("demand half_life configured but some queries lack timestamps; no decay")
        return None
    return np.array([q.timestamp.timestamp() for q in queries if q.timestamp is not None])


def build_pipeline(config: dict[str, Any], resources: Resources) -> Pipeline:
    """Instantiate the step chain from the ``pipeline:`` config section."""
    steps = [
        STEPS.get(entry["type"]).from_config(entry.get("params") or {}, resources)
        for entry in config["pipeline"]
    ]
    return Pipeline(steps, config=config)
