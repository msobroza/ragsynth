# SPEC — `ragsynth` v2.A: single-node parallel execution + ChromaDB chunk-store backend

> **Audience:** Claude Fable 5 coding agent (with superpowers skills).
> **Author of record:** Max (ML Tech Lead). Lifts the v1 non-goal in SPEC §2.2 ("Distributed execution, DB backends, web UI. Everything is in-memory/npz/jsonl") — execution and storage only; **no web UI**.
> **Status:** v2 scaffold spec. SPEC.md remains law for everything not amended here; PLAN.md D1–D31 stand. Extend PLAN.md with D69+ before coding.

---

## 0. How to execute this spec (read first)

1. **Plan before coding.** Append a v2.A section to `PLAN.md` (task breakdown, file map) before implementation; record the pre-made decisions in §11 as D69–D76 plus any new calls as D77 (allocated range per `specs/v2/README.md`).
2. **The one law:** a parallel run produces **byte-identical `metrics.json`** to the **serial v2-execution run** at the same seed (§4), and `schema_version: 1` configs remain byte-identical to the v1 baseline (§4.3). Every design choice defers to this. If a proposed optimization cannot honor it, don't build it.
3. **v1 stays green.** All §15 v1 acceptance criteria (SPEC.md) keep passing; `schema_version: 1` configs load and run unchanged, byte-identical to the v1 baseline forever (§4.3).
4. **TDD.** Known-value fixtures from §9 are written FIRST (SPEC §15.4 style).
5. **Do not modify** `reference/synth_query_eval.py` or any frozen v1 artifact semantics.

---

## 1. Context & motivation

v1 runs everything serially in one process over in-memory stores. Two pressures:

- **Wall-clock.** With a real OpenAI-compatible endpoint, generation + gate dominate runtime: per accepted record ≥ 5 LLM round-trips (3 candidates + zero-context + answerability + uniqueness judges). This is **I/O latency, not CPU** — threads suffice; frameworks don't help.
- **Corpus scale.** `DenseInMemoryRetriever` holds the whole matrix; real KBs live in vector DBs and churn (RAGOps, arXiv 2506.03401). SPEC §16 R6 (lifecycle under churn) needs a store keyed by `content_hash` to hook recycle/re-gate/regenerate onto — v1 already stamps `content_hashes` on every record so this needs no migration.

**Key decisions already made (do not relitigate):**

- **Single-node only. joblib only.** joblib is already a core dep (BSD-3). Thread pools (`prefer="threads"`) for I/O-bound adapter calls; `loky` processes for per-arm validator fan-out. **NO ray/dask/celery/queue framework** — the bottleneck is LLM endpoint latency and provider rate limits, which a queue framework cannot fix and a thread pool saturates; multi-node adds failure modes (partial results, retries, clock skew) that directly threaten the byte-identity law. Revisit only if a single node cannot saturate the endpoint (measure first).
- **Parallel == serial bytes** at the same seed (§4). Non-negotiable acceptance gate.
- **ChromaDB is the one v2 DB backend** (Apache-2.0 → permissive rule holds), behind a `chromadb` extra with `require_optional` lazy import. Local `PersistentClient` only — no client/server mode, no chroma cloud.
- **ragsynth's `Embedder` is the single source of embeddings.** Chroma stores precomputed vectors; its embedding functions **never run** (they can download models — violates air-gap and determinism).
- **Cosine space** (`hnsw:space: cosine`), matching the L2-normalized store invariant.
- **Checkpointing is opt-in, off by default.** A resumed run must be byte-identical to an unresumed one.

---

## 2. Goals & non-goals

### 2.1 v2.A goals

1. `execution` config block + `deterministic_map` helper; parallel candidate generation, parallel gate judge calls (two-phase, §3.4), process-parallel validator arms.
2. `ChunkStore` Protocol (get/add/upsert keyed by `content_hash` — the R6 hook) with `InMemoryChunkStore` (default, wraps v1 behavior) and `ChromaChunkStore`.
3. `ChromaRetriever` implementing the existing `Retriever` Protocol (SPEC §12) — drop-in via `resources.retriever: {type: chromadb}`.
4. `ragsynth ingest` CLI command: corpus jsonl → embed → chroma collection + D19-style sha256 content manifest (ArtifactStore pattern).
5. Per-step checkpoint/resume for large corpora.
6. Config `schema_version: 2` for configs using any of the above (canonical trigger list: `specs/v2/README.md`); v1 configs untouched.

### 2.2 v2.A non-goals (stub or omit)

- Multi-node/distributed anything; async adapters; GPU batching. Ray/dask: never (see §1).
- Other DB backends (qdrant, pgvector, …) — the `ChunkStore`/`Retriever` Protocols are the extension seam; no second concrete.
- Chroma server mode, auth, multi-tenant collections, chroma-side filtering/metadata queries beyond ids.
- Lifecycle *execution* (recycle/re-gate on churn events) — v2.B/R6; this spec only ships the `upsert` diff report it will consume.
- Web UI, experiment tracking, retries/backoff policy tuning (a single `max_retries` knob on `OpenAICompatibleChat` is allowed, default 0).

---

## 3. Execution architecture (`src/ragsynth/execution/`)

### 3.1 Policy & helper

```python
@dataclass(frozen=True)
class ExecutionPolicy:                      # execution/policy.py; threaded through Resources
    parallel: bool = False                  # False ⇒ serial v2-execution path; schema-1 configs always take the v1 path (§4.3)
    n_workers: int = 8                      # threads for adapter calls
    arm_processes: int = 1                  # loky processes for validator arms (1 ⇒ in-process)
    checkpoint: bool = False                # §7
    checkpoint_dir: str | None = None       # default: <artifacts_dir>/checkpoints

def deterministic_map(
    fn: Callable[[int, T], R], items: Sequence[T],
    *, policy: ExecutionPolicy, backend: Literal["threads", "loky"] = "threads",
) -> list[R]:
    """Map fn over (index, item); results ALWAYS in input-index order.

    joblib.Parallel preserves submission order; this helper additionally
    forbids order-sensitive fn (fn must not read/mutate shared state) and
    runs serially when policy.parallel is False or len(items) < 2.
    """
```

`Resources` gains `policy: ExecutionPolicy` (frozen field, default serial — `with_overrides` compatible). Steps receive parallelism ONLY through `deterministic_map`; raw `joblib.Parallel`/threading in step code is a review reject.

### 3.2 Where parallelism applies (and where it must not)

| Site | Backend | Item key |
|---|---|---|
| `generator`: candidate LLM calls | threads | `f"{seed.seed_id}/{cand_idx}"` |
| `gate`: zero_context / answerability / round_trip / uniqueness verdicts | threads | `candidate.query_id` |
| `validator`: per-arm sub-runs | loky processes | arm name |

**Never parallelized:** dedup (order-dependent by definition), curator, qrel_builder, all `metrics/` and `sampling/` code (CPU-cheap; prototype-faithful rng streams per D31), artifact writes (single writer), figure rendering (parent process only, Agg).

### 3.3 Merge rule

Workers return `(index, result)`; the ONLY merge is ascending-index reassembly inside `deterministic_map`. Completion order must be unobservable: no shared accumulators, no logging-derived state, no `dict` built in completion order. `state` mutation happens exclusively in the parent after reassembly.

### 3.4 Two-phase gate

The v1 gate is a serial short-circuit fold (cheap→expensive) whose dedup check depends on the accepted-so-far set — inherently sequential. v2 splits it:

- **Phase 1 (parallel):** for every candidate, precompute the *item-local* verdicts (zero_context, answerability, round_trip, uniqueness) via `deterministic_map`. These depend only on `(candidate, resources)` — never on `state`.
- **Phase 2 (serial fold):** replay the v1 orchestrator in candidate-index order, consuming precomputed verdicts instead of calling adapters; dedup runs live against the accepted set. Short-circuit semantics, `Rejection` objects, reject-reason tallies, and promotions are computed by the fold ⇒ **identical to serial output**.

**Documented cost:** phase 1 spends LLM calls on candidates the serial gate would have short-circuited (e.g. dedup-rejected). That buys latency with call volume; `parallel: false` restores minimal-call behavior. The report gains `metrics["execution"] = {n_llm_calls, n_saved_by_shortcircuit}` (informational; lives in report.md provenance, NOT metrics.json — D14 byte rule).

---

## 4. Determinism & the seeding scheme (the hard requirement)

### 4.1 Per-item RNG streams

On the **v2 execution path only** (§4.3), all randomness in a parallelized site derives from a per-item substream, never a loop-carried generator:

```python
# reuses v1 Resources.rng(name) = default_rng([seed, stable_hash64(name)])
item_rng = resources.rng(f"{step_name}/{item_key}")     # e.g. "generator/s0042/1"
```

`item_key` is a **stable identity** (seed_id, query_id, arm name — content-derived, never enumeration order of a set/dict). The stream a worker uses is a pure function of `(seed, step_name, item_key)`; worker assignment and completion order cannot reach it. (Counter-based per-item streams: Salmon et al., SC 2011; numpy `SeedSequence` mechanics.)

### 4.2 Acceptance identity

Two hard identities, both CI tests, not doc claims:

1. **Parallel == serial v2-execution:** for a schema-2 config declaring the `execution` block, `run(config, seed, parallel=True, n_workers=k)` ⇒ `metrics.json` byte-identical to the **serial v2-execution run** (`parallel: false`, same config, same seed), for k ∈ {1, 4, 8}, on both bundled schema-2 configs. Same for `arm_processes ∈ {1, 4}`.
2. **Schema-1 == v1 baseline:** every `schema_version: 1` config produces `metrics.json` byte-identical to the frozen v1 baseline at the same seed.

### 4.3 Two frozen RNG paths (no stream migration)

Per-item RNG streams (§4.1) apply **only to the v2 execution path** — schema-2 configs that declare the `execution` block. `schema_version: 1` configs keep v1's loop-carried generator/gate substreams and remain **bit-exact to the v1 baseline forever**: the v1 byte-stability law (SPEC §15) is never broken, and acceptance criteria pinned to v1 bytes in `01-real-benchmarks.md` and `02-graded-pooled-qrels-umbrela.md` stay satisfiable. There is **no benchmark-migration event** and `experiments/` metrics are not re-locked. Serial v2-execution output (per-item streams) legitimately differs from v1-path output at the same seed — a property of the new path, not a migration of the old one; the determinism law for the v2 path is identity 1 of §4.2. Metric-internal rngs (D31) are untouched on both paths.

### 4.4 What is NOT guaranteed (state this in README)

- With a **real LLM endpoint** (`OpenAICompatibleChat`), outputs are provider-nondeterministic regardless of parallelism. Guaranteed instead: the *request set* is identical (per-item independence) and the merge is index-ordered — parallelism adds **zero additional** nondeterminism. Full byte-identity holds for all deterministic adapters (mocks, toy, hashed embedder).
- **Pre-existing chroma collections** (§6): only manifest-verified collections participate in any determinism claim.

---

## 5. ChunkStore Protocol + ChromaDB backend

### 5.1 Protocol (`adapters/chunk_store/base.py`)

```python
@dataclass(frozen=True)
class UpsertReport:                       # the R6 churn-lifecycle hook
    added: tuple[str, ...]                # chunk_ids new to the store
    replaced: tuple[str, ...]             # same chunk_id, content_hash changed
    unchanged: tuple[str, ...]            # content_hash matched ⇒ skipped

class ChunkStore(Protocol):
    def get(self, chunk_ids: Sequence[str]) -> list[Chunk]: ...          # KeyError on miss
    def get_embeddings(self, chunk_ids: Sequence[str]) -> NDArray[np.float32]: ...
    def add(self, chunks: Sequence[Chunk], embeddings: NDArray[np.float32]) -> None: ...
        # duplicate chunk_id ⇒ ValueError (write-once, mirrors EmbeddingStore)
    def upsert(self, chunks: Sequence[Chunk], embeddings: NDArray[np.float32]) -> UpsertReport: ...
        # keyed by content_hash: unchanged content never rewritten
    def all_chunk_ids(self) -> list[str]: ...                            # ingestion order
    def __len__(self) -> int: ...
```

`CHUNK_STORES` registry; `InMemoryChunkStore` (dict + `EmbeddingStore`) is the default and the contract-test reference implementation. Embeddings passed in are **always** produced by the configured ragsynth `Embedder` at the composition root — stores never embed.

### 5.2 `ChromaChunkStore` + `ChromaRetriever` (`adapters/{chunk_store,retriever}/chroma.py`, extra `chromadb`)

- `require_optional(chromadb, "ChromaChunkStore", "chromadb")`; `PersistentClient(path, settings=Settings(anonymized_telemetry=False))` — telemetry off is mandatory (air-gap).
- Collection created with `metadata={"hnsw:space": "cosine"}` and **no embedding function**; every add/query passes explicit vectors. A guard asserts at open time that the collection has no configured EF and dim matches the manifest; mismatch ⇒ actionable `ValueError`.
- Chunk fields ride chroma metadata (`doc_id`, `content_hash`, `page`, `ingest_index`, extra `metadata` flattened); `text` in documents; `chunk_id` as chroma id. `ingest_index` = position in corpus order at ingest.
- `ChromaRetriever.search(query_emb, k)`: query with `n_results = min(k + tie_buffer, len(collection))` (default `tie_buffer=8`), then re-sort by `(-score, ingest_index)` and truncate to k. This pins tie order to `DenseInMemoryRetriever`'s stable row order — required for §6 rank parity.
- `search_ef` config (default 256): parity holds in the brute-force regime (`search_ef ≥ collection size`); above that HNSW is approximate — document, don't pretend otherwise.
- Writers: single-process only. Validator arm processes open chroma **read-only**; ingest is the sole writer. Enforced by convention + a lock-file check, not chroma internals.

### 5.3 Config

```yaml
ragsynth: {schema_version: 2, name: v2-chroma, seed: 0}
execution: {parallel: true, n_workers: 8, arm_processes: 1, checkpoint: false}
resources:
  dataset: {type: jsonl, params: {chunks_path: data/sample/chunks.jsonl, queries_path: data/sample/queries.jsonl}}
  embedder: {type: hashed_ngram, params: {dim: 256}}
  retriever: {type: chromadb, params: {path: .chroma, collection: ragsynth_sample, search_ef: 256}}
  chunk_store: {type: chromadb, params: {path: .chroma, collection: ragsynth_sample}}  # optional; default inmemory
```

`ragsynth ingest --config <cfg>` builds the collection (embed via configured embedder → `upsert` → write manifest) and prints the `UpsertReport`. Licenses: chromadb is Apache-2.0; its transitive deps are documented in the README extras table (they are extra-only, never core runtime deps).

---

## 6. Determinism boundary for persistent DBs

A persistent collection is external mutable state; the v1 "same seed ⇒ same bytes" claim cannot blanket it. The boundary:

- **Guaranteed:** a **fresh** collection ingested from a given corpus with a given embedder config is reproducible. `ingest` writes a D19-style sha256 content manifest (ArtifactStore pattern, D19) next to the collection: `{corpus_hash: sha256 over sorted (chunk_id, content_hash) pairs, embedder_config_hash, dim, count, ragsynth_version, schema: 1}`. At run time the composition root recomputes both hashes and **refuses** a mismatched or missing manifest unless `trust_existing: true` (then it WARNs and the run's report provenance records `unverified_store: true`).
- **Not guaranteed:** anything about pre-existing/hand-modified collections, cross-chromadb-version byte layout, or on-disk file bytes (D30 already exempts artifact bytes; determinism claims attach to `metrics.json` only).
- **Parity acceptance test (the teeth):** on the bundled 200-chunk sample corpus, a `chromadb`-backed run reproduces the `dense_inmemory` run **exactly** — per-query retrieval ranks equal, and therefore `metrics.json` byte-identical. Runs in the `chromadb` CI job (§8).

---

## 7. Step-level checkpointing (opt-in, off by default)

- `execution.checkpoint: true` ⇒ after each step, the parent serializes the state delta to `<checkpoint_dir>/<config_hash>/<NN>_<step.name>.jsonl` (one pydantic-JSON object per record; `metrics`/`provenance` in a trailing header line). Key = `(config_hash, seed, step index, step.name, step.version)` — config_hash already covers params, so any config edit invalidates every checkpoint.
- **Resume:** on run, the pipeline skips leading steps whose checkpoint key matches, loads the newest matching prefix, and continues. Partial-step (intra-item) resume is v2.B; a step interrupted mid-run leaves no checkpoint.
- **Byte-stability interaction:** checkpoints are outputs, not config — the SPEC §13 YAML round-trip rule is untouched. The binding rule instead: a resumed run's `metrics.json` is byte-identical to an uninterrupted run (§8 test). Guaranteed by per-item streams (§4.1): resumption re-derives identical rngs because nothing is loop-carried.
- Checkpoint dirs live under `artifacts/` ⇒ gitignored per D30. `ragsynth run --no-resume` ignores existing checkpoints; stale-key checkpoints are ignored silently (logged at INFO).

---

## 8. Schema, typing, CI policy

- **schema_version:** of this spec's features, the `execution` block, the `chunk_store` resource, and `retriever.type: chromadb` require `schema_version: 2` (validation error names the offending key). The canonical trigger list and the single-owner `from_yaml` loader change are defined once in `specs/v2/README.md` ("schema_version 2 — canonical trigger list"); this spec does not re-define them. v1 configs (`schema_version: 1`) parse to serial `ExecutionPolicy` + `InMemoryChunkStore` — byte-identical to the v1 baseline (§4.3).
- **mypy strict:** chromadb ships incomplete typing — add `chromadb.*` to the existing `ignore_missing_imports` override list (same policy as `sklearn.*`/`joblib.*`, D31 spirit); the ragsynth-side adapter modules stay fully typed; no `type: ignore` outside the import seam.
- **ruff select=ALL** stays green; new modules follow the no-`utils.py`, Google-docstring, paper-citation rules.
- **CI jobs:** (a) core — no extras, air-gapped, full suite; chroma tests auto-skip via `pytest.importorskip("chromadb")`; (b) `--extra chromadb` — still air-gapped (PersistentClient is local, telemetry off; the parity + manifest tests live here). Every LLM/embedding call in every new test goes through the existing deterministic mocks.

---

## 9. Known-value fixtures (write FIRST — SPEC §15.4 style)

1. `deterministic_map` with a fn that sleeps `hash(item) % 5` ms and records completion order ⇒ results in input order; completion order provably scrambled in the fixture yet output identical to serial.
2. Per-item rng: `resources.rng("generator/s0/0")` equal across two `Resources` instances and across thread submission orders; ≠ `"generator/s0/1"`.
3. Two-phase gate fold on a hand-crafted 4-candidate fixture (candidate 1 dedup-dup of 0, candidate 2 fails answerability, candidate 3 promotes): accepted set, `Rejection` reasons, reject tally, promotions all `==` the v1 serial orchestrator's output on the same fixture.
4. `ChunkStore` contract test parametrized over `CHUNK_STORES` (in-memory always; chroma when importable): add/get round-trip, duplicate-add ValueError, `upsert` diff fixture — 1 new, 1 changed-hash, 1 unchanged ⇒ exact `UpsertReport`.
5. Tie-break fixture: two chunks with identical embeddings ⇒ `ChromaRetriever` and `DenseInMemoryRetriever` return identical order (ingest_index).
6. Manifest: ingest twice from same corpus ⇒ identical `corpus_hash`; mutate one chunk's text ⇒ hash changes and the run refuses without `trust_existing`.
7. Checkpoint: run tiny toy config, kill after gate (simulated), resume ⇒ `metrics.json` byte-identical to uninterrupted run.
8. Parallel-vs-serial: tiny schema-2 toy config, `n_workers ∈ {1, 4}` ⇒ byte-identical `metrics.json` (§4.2 identity 1, miniaturized for CI speed).
9. Schema-1 baseline: `configs/v1_toy.yaml` (schema 1, untouched file) ⇒ `metrics.json` byte-identical to the frozen v1 baseline fixture (§4.2 identity 2).

---

## 10. Acceptance criteria (Definition of Done)

- [ ] `uv run ragsynth run --config configs/v1_toy.yaml` (schema 1, untouched file) passes all v1 §15 criteria with `metrics.json` byte-identical to the frozen v1 baseline (no stream migration — §4.3).
- [ ] **Parallel == serial v2-execution bytes:** §4.2 identity 1 holds on both bundled schema-2 configs, k ∈ {1, 4, 8}, `arm_processes ∈ {1, 4}`; §4.2 identity 2 (schema-1 == v1 baseline) enforced alongside — CI-enforced (miniature) + full run documented in `experiments/v2a/`.
- [ ] **Chroma parity:** §6 test — sample-corpus chromadb run reproduces dense_inmemory retrieval ranks exactly and `metrics.json` byte-identically.
- [ ] Manifest guard: unmanifested/mismatched collection refused; `trust_existing: true` warns + records provenance flag.
- [ ] Checkpoint resume byte-identity (§7); checkpoints off by default; `--no-resume` works.
- [ ] Fresh env without extras: full core suite green, air-gapped; chroma tests skip cleanly. `--extra chromadb` job green, still air-gapped, telemetry off verified (no network syscalls in test — assert via chroma settings, not tcpdump).
- [ ] `uv run ruff check . && uv run ruff format --check .` · `uv run mypy src` (chromadb override per §8) · `uv run pytest -q --cov=ragsynth --cov-fail-under=70` — green at every commit.
- [ ] Every new step/strategy/adapter implements `to_config()/from_config()`; contract tests parametrize the new `CHUNK_STORES` registry; YAML round-trip byte-stable for schema-2 configs; schema-1 configs load unchanged.
- [ ] README: `chromadb` extra row (license + transitive-dep note), determinism-boundary paragraph (§4.4, §6), ingest quickstart. ARCHITECTURE.md: execution + chunk-store paragraph.
- [ ] PLAN.md extended with D69+ (§11) and v2.A task list; `experiments/v2a/` holds the frozen config, metrics, and a latency-vs-call-volume note for the two-phase gate.

---

## 11. Decisions this spec pre-makes (append to PLAN.md)

Decision numbers D69–D77 allocated per `specs/v2/README.md`.

| # | Decision |
|---|---|
| D69 | Single-node joblib-only parallelism; no ray/dask/celery ever in v2 (bottleneck = LLM latency; byte-identity law). |
| D70 | Per-item RNG streams `Resources.rng(f"{step}/{item_key}")` on the v2 execution path only (schema-2 configs declaring `execution`), item_key content-derived; loop-carried rngs banned at parallel sites; schema-1 configs keep v1 loop-carried streams, bit-exact to the v1 baseline forever — no stream migration (§4.3). |
| D71 | Two-phase gate: parallel item-local verdict precompute + serial index-order fold; identical outputs to v1 orchestrator; extra LLM-call cost documented, reported in report.md provenance only (D14 holds). |
| D72 | ChromaDB behind `chromadb` extra; local PersistentClient only; telemetry off; cosine space; chroma embedding functions never configured; ties pinned by `ingest_index` metadata re-sort. |
| D73 | Collection manifest = {corpus_hash, embedder_config_hash, dim, count, version} — sha256 content-manifest per D19's ArtifactStore pattern; unverified collections refused unless `trust_existing: true` (warn + provenance flag). |
| D74 | Checkpoints keyed `(config_hash, seed, step index, name, version)`, jsonl under artifacts (gitignored per D30), off by default; resumed == uninterrupted bytes. |
| D75 | schema_version 2 required for this spec's `execution`/`chunk_store`/chromadb features per the canonical trigger table in `specs/v2/README.md` (single-owner loader change; other specs' features also trigger schema 2); v1 configs behave serially with in-memory stores. |
| D76 | mypy policy for chromadb: `ignore_missing_imports` override at the import seam only, adapter modules fully typed (extends D31 typing posture). |

---

## 12. Reference table (v2.A additions)

| Feature | References |
|---|---|
| Counter-based / splittable parallel RNG | Salmon et al., SC 2011; numpy SeedSequence docs |
| HNSW (why parity needs the brute-force regime) | Malkov & Yashunin, TPAMI 2020 |
| ChromaDB (Apache-2.0) | Chroma docs, 2025 |
| Churn lifecycle framing (upsert hook) | RAGOps, arXiv 2506.03401; SPEC §16 R6 |
| joblib threading/loky backends | joblib docs (BSD-3) |

---

## 13. Open questions — defaults for the agent (decide, document, proceed)

1. `n_workers` default: **8** threads (LLM endpoints tolerate ~8 concurrent easily; config-exposed).
2. `tie_buffer` for ChromaRetriever: **8**; `search_ef` default **256** — both config-exposed.
3. Checkpoint granularity: **per-step only** in v2.A; intra-step item-level resume deferred to v2.B (note in PLAN).
4. `OpenAICompatibleChat` gains `max_retries` (default **0**) + thread-safety audit (urllib per-call, no shared session state) — no backoff policy framework.
5. Chroma collection naming on ingest: config `collection` verbatim; no auto-versioned names (manifest catches drift).
6. Single `chromadb` extra pin: `chromadb>=1.0` (older 0.x churned the client API); document the floor's rationale in pyproject comment.
