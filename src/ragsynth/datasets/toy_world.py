"""Geometric toy world: dataset builder plus coupled toy adapters (SPEC §10).

Port of the prototype's simulated world (``reference/synth_query_eval.py``
L652-723): ``k_true`` topical components on the sphere, two asymmetric
tangent sub-modes per component, skewed demand, and an LLM "style
monoculture" direction. Chunk and query *texts* are opaque tokens
(``toychunk:0007``, ``toyquery:train:00042``); their geometry lives in the
bundle's :class:`~ragsynth.io.embeddings.EmbeddingStore` (keyed by id) and
the shared :class:`~ragsynth.datasets.base.EmbeddingBank` (keyed by text),
so the full text pipeline runs end to end with exact, deterministic
embeddings. The coupled adapters -- :class:`ToyChatModel`,
:class:`ToyJudge`, :class:`PassthroughEmbedder` -- close the loop through
the bank (PLAN D11/D12).
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import TYPE_CHECKING, Any

import numpy as np

from ragsynth.adapters.embedder.base import EMBEDDERS
from ragsynth.adapters.judge.base import JUDGES, JudgeVerdict
from ragsynth.adapters.llm.base import CHAT_MODELS
from ragsynth.datasets.base import DATASETS, DatasetBundle, EmbeddingBank
from ragsynth.domain import Chunk, ProductionQuery
from ragsynth.io.embeddings import EmbeddingStore
from ragsynth.pipeline.base import stable_hash64
from ragsynth.sampling.vmf import l2_normalize, sample_vmf, sphere_uniform

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

_CHUNKS_PER_DOC = 5
_N_SUB_MODES = 2
_TRAIN_FRAC = 0.60
_ANCHOR_FRAC = 0.25

_DEFAULT_D = 64
_DEFAULT_K_TRUE = 8
_DEFAULT_N_CHUNKS = 640
_DEFAULT_N_PROD = 5000
_DEFAULT_KAPPA_CHUNK = 150.0
_DEFAULT_KAPPA_QUERY = 400.0
_DEFAULT_SUB_OFFSET = 0.35
_DEFAULT_SUB_PROBS = (0.65, 0.35)
_DEFAULT_DEMAND_SKEW = 1.1

_DEFAULT_STYLE = 0.15
_DEFAULT_NOISE = 0.68
_REVISION_STYLE_DIV = 2.0
_REVISION_NOISE_DIV = 3.0
_REVISE_TOKEN = "REVISE_REQUEST"  # noqa: S105 - prompt marker token, not a secret
_TEXT_DIGEST_LEN = 12
_CHUNK_TOKEN_RE = re.compile(r"toychunk:\d{4}")
_EXEMPLAR_TOKEN_RE = re.compile(r"toyquery:[a-z]+:\d{5}")
_BASE_MODES = ("chunks", "exemplars")

_DEFAULT_TAU_ANS = 0.5
_DEFAULT_COMMON_KNOWLEDGE_PCT = 2
_PERCENT = 100


@DATASETS.register("toy_world")
class ToyWorldDataset:
    """The prototype's 8-component / 2-hidden-sub-mode world (proto L668-712).

    Chunks sit near component means uniformly across components (KB
    coverage != demand); production queries follow a skewed demand over
    components and two asymmetric within-component sub-modes -- the
    structure quota sampling cannot see but the spec sampler can.
    """

    @classmethod
    def build(cls, params: dict[str, Any], seed: int) -> DatasetBundle:
        """Build the toy world bundle.

        Args:
            params: Geometry knobs -- ``d``, ``k_true``, ``n_chunks``,
                ``n_prod``, ``kappa_chunk``, ``kappa_query``, ``sub_offset``,
                ``sub_probs``, ``demand_skew`` (prototype defaults).
            seed: Config seed; the world uses the deterministic substream
                ``[seed, stable_hash64("toy_world")]``.

        Returns:
            A :class:`DatasetBundle` with pre-split queries (PLAN D10),
            nearest-chunk anchor/oracle qrels (PLAN D17), a filled
            :class:`EmbeddingStore`, and the shared :class:`EmbeddingBank`.
        """
        d = int(params.get("d", _DEFAULT_D))
        k_true = int(params.get("k_true", _DEFAULT_K_TRUE))
        n_chunks = int(params.get("n_chunks", _DEFAULT_N_CHUNKS))
        n_prod = int(params.get("n_prod", _DEFAULT_N_PROD))
        kappa_chunk = float(params.get("kappa_chunk", _DEFAULT_KAPPA_CHUNK))
        kappa_query = float(params.get("kappa_query", _DEFAULT_KAPPA_QUERY))
        sub_offset = float(params.get("sub_offset", _DEFAULT_SUB_OFFSET))
        sub_probs = [float(p) for p in params.get("sub_probs", _DEFAULT_SUB_PROBS)]
        demand_skew = float(params.get("demand_skew", _DEFAULT_DEMAND_SKEW))

        rng = np.random.default_rng([seed, stable_hash64("toy_world")])
        chunk_emb, split_embs = _make_geometry(
            rng,
            d=d,
            k_true=k_true,
            n_chunks=n_chunks,
            n_prod=n_prod,
            kappa_chunk=kappa_chunk,
            kappa_query=kappa_query,
            sub_offset=sub_offset,
            sub_probs=sub_probs,
            demand_skew=demand_skew,
        )

        chunks = tuple(
            Chunk.create(text=f"toychunk:{i:04d}", doc_id=f"toydoc:{i // _CHUNKS_PER_DOC}")
            for i in range(len(chunk_emb))
        )
        store = EmbeddingStore()
        bank = EmbeddingBank()
        store.add([c.chunk_id for c in chunks], chunk_emb)
        for chunk, vec in zip(chunks, chunk_emb, strict=True):
            bank.put(chunk.text, vec)

        splits: dict[str, tuple[ProductionQuery, ...]] = {}
        qrels: dict[str, dict[str, dict[str, int]]] = {"anchor": {}, "oracle": {}}
        for split_name, emb in split_embs.items():
            queries = tuple(
                ProductionQuery(
                    query_id=f"toyquery:{split_name}:{i:05d}",
                    text=f"toyquery:{split_name}:{i:05d}",
                )
                for i in range(len(emb))
            )
            store.add([q.query_id for q in queries], emb)
            for query, vec in zip(queries, emb, strict=True):
                bank.put(query.text, vec)
            splits[split_name] = queries
            if split_name in qrels:  # gate-style nearest-chunk gold (PLAN D17)
                nearest = np.argmax(emb @ chunk_emb.T, axis=1)
                qrels[split_name] = {
                    q.query_id: {chunks[int(j)].chunk_id: 1}
                    for q, j in zip(queries, nearest, strict=True)
                }
        logger.info(
            f"toy_world built: {len(chunks)} chunks, "
            f"{len(splits['train'])}/{len(splits['anchor'])}/{len(splits['oracle'])} "
            f"train/anchor/oracle queries (d={d}, k_true={k_true})"
        )
        return DatasetBundle(
            chunks=chunks,
            queries_train=splits["train"],
            queries_anchor=splits["anchor"],
            queries_oracle=splits["oracle"],
            anchor_qrels=qrels["anchor"],
            oracle_qrels=qrels["oracle"],
            embeddings=store,
            bank=bank,
        )


def _make_geometry(
    rng: np.random.Generator,
    *,
    d: int,
    k_true: int,
    n_chunks: int,
    n_prod: int,
    kappa_chunk: float,
    kappa_query: float,
    sub_offset: float,
    sub_probs: list[float],
    demand_skew: float,
) -> tuple[NDArray[np.float64], dict[str, NDArray[np.float64]]]:
    """Port of the prototype's ``make_world`` geometry (proto L668-712).

    Returns:
        ``(chunk_emb, split_embs)`` where ``split_embs`` maps
        train/anchor/oracle to unit-row query matrices, split 60/25/15 via
        a seeded permutation (PLAN D10).
    """
    mus = sphere_uniform(k_true, d, rng)
    weights = 1.0 / np.arange(1, k_true + 1) ** demand_skew  # skewed demand
    weights = weights / weights.sum()

    # two asymmetric tangent sub-modes per component (the A2-visible structure)
    sub_mus = np.empty((k_true, _N_SUB_MODES, d))
    for c in range(k_true):
        for s in range(_N_SUB_MODES):
            t = rng.standard_normal(d)
            t -= (t @ mus[c]) * mus[c]  # tangent perturbation
            sub_mus[c, s] = l2_normalize(mus[c] + sub_offset * l2_normalize(t))

    per = n_chunks // k_true
    chunk_emb = np.vstack([sample_vmf(mus[c], kappa_chunk, per, rng) for c in range(k_true)])

    comps = rng.choice(k_true, size=n_prod, p=weights)
    subs = rng.choice(_N_SUB_MODES, size=n_prod, p=sub_probs)
    query_emb = np.vstack(
        [
            sample_vmf(sub_mus[int(c), int(s)], kappa_query, 1, rng)
            for c, s in zip(comps, subs, strict=True)
        ]
    )
    perm = rng.permutation(n_prod)
    n_tr, n_an = int(_TRAIN_FRAC * n_prod), int(_ANCHOR_FRAC * n_prod)
    train, anchor, oracle = np.split(query_emb[perm], [n_tr, n_tr + n_an])
    return chunk_emb, {"train": train, "anchor": anchor, "oracle": oracle}


@CHAT_MODELS.register("toy_chat")
class ToyChatModel:
    """Simulated generator LLM: emits geometric embeddings, not language.

    ``complete`` parses ``toychunk:``/``toyquery:`` tokens out of the user
    prompt, averages their banked vectors into a base, and applies the
    prototype's emission model ``_emit`` (proto L715-723): unit-normalized
    ``base + style * style_dir + noise * g``. Prompts containing the
    literal ``REVISE_REQUEST`` token are revision calls and use
    ``style / 2`` and ``noise / 3`` -- the target-check tightening
    (PLAN D12). The returned text is banked so the
    :class:`PassthroughEmbedder` can look it up later.
    """

    def __init__(
        self,
        bank: EmbeddingBank,
        style: float = _DEFAULT_STYLE,
        noise: float = _DEFAULT_NOISE,
        base: str = "chunks",
        seed: int = 0,
        d: int = _DEFAULT_D,
    ) -> None:
        if base not in _BASE_MODES:
            raise ValueError(f"unknown base mode '{base}'; known: {list(_BASE_MODES)}")
        self.bank = bank
        self.style = style
        self.noise = noise
        self.base = base
        self.seed = seed
        self.d = d
        style_rng = np.random.default_rng([seed, stable_hash64("toy_style_dir")])
        self._style_dir: NDArray[np.float64] = l2_normalize(style_rng.standard_normal(d))

    def _banked_vectors(self, tokens: list[str]) -> list[NDArray[np.float64]]:
        """Vectors for the banked subset of ``tokens`` (unbanked are ignored)."""
        return [self.bank.get(t) for t in tokens if t in self.bank]

    def complete(self, system: str, user: str, **kwargs: Any) -> str:
        """Emit one synthetic query: bank its embedding, return its text token.

        Raises:
            ValueError: If the user prompt contains no banked
                ``toychunk:``/``toyquery:`` token to seed the emission.
        """
        revision = _REVISE_TOKEN in user
        style = self.style / _REVISION_STYLE_DIV if revision else self.style
        noise = self.noise / _REVISION_NOISE_DIV if revision else self.noise

        chunk_vecs = self._banked_vectors(_CHUNK_TOKEN_RE.findall(user))
        exemplar_vecs = self._banked_vectors(_EXEMPLAR_TOKEN_RE.findall(user))
        primary, fallback = (
            (chunk_vecs, exemplar_vecs) if self.base == "chunks" else (exemplar_vecs, chunk_vecs)
        )
        vectors = primary or fallback
        if not vectors:
            raise ValueError(
                "toy_chat could not parse any banked toychunk:/toyquery: token from the "
                "user prompt; toy-world prompts must embed evidence or exemplar texts"
            )
        base_vec = l2_normalize(np.stack(vectors).mean(axis=0))

        # prototype _emit (L715-723), deterministic per (seed, system, user)
        rng = np.random.default_rng([self.seed, stable_hash64(f"{system}\x00{user}")])
        g = l2_normalize(rng.standard_normal(base_vec.shape))
        emission = l2_normalize(base_vec + style * self._style_dir + noise * g)

        digest = hashlib.sha256(f"{system}\x00{user}\x00{self.seed}".encode()).hexdigest()
        text = f"toysynth:{digest[:_TEXT_DIGEST_LEN]}"
        self.bank.put(text, emission)
        return text

    def to_config(self) -> dict[str, Any]:
        """JSON-safe constructor params (the bank is injected, not serialized)."""
        return {
            "style": self.style,
            "noise": self.noise,
            "base": self.base,
            "seed": self.seed,
            "d": self.d,
        }

    @classmethod
    def from_config(
        cls, params: dict[str, Any], bundle: DatasetBundle, rng: np.random.Generator
    ) -> ToyChatModel:
        """Build from a config params block, wiring in the bundle's bank.

        ``d`` defaults to the bundle's embedding dimensionality when the
        store is filled (toy world), else to the toy default.

        Raises:
            ValueError: If the bundle carries no :class:`EmbeddingBank`.
        """
        if bundle.bank is None:
            raise ValueError(
                "toy_chat requires a dataset with an embedding bank (toy_world); "
                "text corpora should use a real chat model instead"
            )
        default_d = _DEFAULT_D
        if bundle.embeddings is not None and bundle.embeddings.dim is not None:
            default_d = bundle.embeddings.dim
        return cls(
            bundle.bank,
            style=float(params.get("style", _DEFAULT_STYLE)),
            noise=float(params.get("noise", _DEFAULT_NOISE)),
            base=str(params.get("base", "chunks")),
            seed=int(params.get("seed", 0)),
            d=int(params.get("d", default_d)),
        )


@JUDGES.register("toy_judge")
class ToyJudge:
    """Geometric relevance judge over the shared bank (PLAN D11).

    With evidence: answerable iff the max query-evidence cosine reaches
    ``tau_ans``. Without evidence: a deterministic ``common_knowledge_pct``
    percent of queries are "common knowledge" (hash rule), simulating the
    zero-context self-test's rare false positives.
    """

    def __init__(
        self,
        bank: EmbeddingBank,
        tau_ans: float = _DEFAULT_TAU_ANS,
        common_knowledge_pct: int = _DEFAULT_COMMON_KNOWLEDGE_PCT,
    ) -> None:
        self.bank = bank
        self.tau_ans = tau_ans
        self.common_knowledge_pct = common_knowledge_pct

    def judge(self, query: str, evidence_texts: Sequence[str]) -> JudgeVerdict:
        """Return the geometric verdict for one (query, evidence) pair."""
        if query not in self.bank:
            logger.warning(f"toy_judge: query text not banked, ruling unanswerable: {query[:60]!r}")
            return JudgeVerdict(answerable=False, answer="", confidence=0.0)
        if not evidence_texts:
            answerable = stable_hash64(query) % _PERCENT < self.common_knowledge_pct
            return JudgeVerdict(
                answerable=answerable,
                answer="toy answer" if answerable else "",
                confidence=1.0 if answerable else 0.0,
            )
        evidence_vecs = [self.bank.get(t) for t in evidence_texts if t in self.bank]
        if not evidence_vecs:
            logger.warning(
                f"toy_judge: none of the {len(evidence_texts)} evidence texts are banked, "
                "ruling unanswerable"
            )
            return JudgeVerdict(answerable=False, answer="", confidence=0.0)
        query_vec = l2_normalize(self.bank.get(query))
        max_cos = float(np.max(l2_normalize(np.stack(evidence_vecs)) @ query_vec))
        answerable = max_cos >= self.tau_ans
        return JudgeVerdict(
            answerable=answerable,
            answer="toy answer" if answerable else "",
            confidence=float(np.clip(max_cos, 0.0, 1.0)),
        )

    def to_config(self) -> dict[str, Any]:
        """JSON-safe constructor params (the bank is injected, not serialized)."""
        return {"tau_ans": self.tau_ans, "common_knowledge_pct": self.common_knowledge_pct}

    @classmethod
    def from_config(
        cls, params: dict[str, Any], bundle: DatasetBundle, rng: np.random.Generator
    ) -> ToyJudge:
        """Build from a config params block, wiring in the bundle's bank.

        Raises:
            ValueError: If the bundle carries no :class:`EmbeddingBank`.
        """
        if bundle.bank is None:
            raise ValueError(
                "toy_judge requires a dataset with an embedding bank (toy_world); "
                "text corpora should use a real judge instead"
            )
        return cls(
            bundle.bank,
            tau_ans=float(params.get("tau_ans", _DEFAULT_TAU_ANS)),
            common_knowledge_pct=int(
                params.get("common_knowledge_pct", _DEFAULT_COMMON_KNOWLEDGE_PCT)
            ),
        )


@EMBEDDERS.register("passthrough")
class PassthroughEmbedder:
    """Bank-lookup embedder: the toy world bypasses text entirely (SPEC §12).

    Every text a pipeline embeds must have been banked (by the dataset
    builder or by :class:`ToyChatModel`); an unknown text raises ``KeyError``
    -- that propagation is the toy invariant, not an error to swallow.
    """

    def __init__(self, bank: EmbeddingBank) -> None:
        self.bank = bank

    def encode(self, texts: Sequence[str]) -> NDArray[np.float64]:
        """Look up the banked vector for each text.

        Raises:
            KeyError: If any text was never banked.
        """
        return np.stack([self.bank.get(t) for t in texts])

    def to_config(self) -> dict[str, Any]:
        """JSON-safe constructor params (the bank is injected, not serialized)."""
        return {}

    @classmethod
    def from_config(
        cls, params: dict[str, Any], bundle: DatasetBundle, rng: np.random.Generator
    ) -> PassthroughEmbedder:
        """Build from a config params block, wiring in the bundle's bank.

        Raises:
            ValueError: If the bundle carries no :class:`EmbeddingBank`.
        """
        if bundle.bank is None:
            raise ValueError(
                "passthrough embedder requires a dataset with an embedding bank "
                "(toy_world); text corpora should use hashed_ngram or st instead"
            )
        return cls(bundle.bank)
