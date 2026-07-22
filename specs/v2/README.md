# ragsynth v2 specs — index

One spec per workstream, written in the SPEC.md house style (context → contracts →
config → acceptance → open-questions-with-defaults). SPEC.md remains the v1 law;
these documents extend it and pre-make their own decisions, to be recorded in
PLAN.md as D32+ when implementation starts. All five inherit the v1 global
constraints: permissive-license runtime deps only, offline-first adapters with
deterministic mocks (air-gapped CI), mypy strict + ruff `select=ALL`, TDD with
known-value fixtures first, `to_config()/from_config()` round-trip, and
schema_version 2 for configs using v2 features (schema_version 1 configs keep
loading unchanged).

| # | spec | roadmap anchor | depends on |
|---|---|---|---|
| 01 | [Real-benchmark validation](01-real-benchmarks.md) — FiQA-2018, NFCorpus, LegalBench-RAG | SPEC §16 "benchmark validation experiments" | none (first real-data step) |
| 02 | [Graded/pooled qrels (UMBRELA) + judge calibration](02-graded-pooled-qrels-umbrela.md) | R5 | none; consumes 01's runs as its trigger evidence |
| 03 | [Prompt-optimizer execution (MIPROv2, GEPA)](03-prompt-optimizer-execution.md) | R2 | none; triggered when 01 shows gate pass-rate / τ plateau |
| 04 | [Vendi, MAUVE, soft movMF demand map](04-soft-demand-diversity-metrics.md) | §2.2 optional/v2 | none (report-only additions) |
| 05 | [Distributed execution + ChromaDB backend](05-distributed-execution-db-backends.md) | §2.2 non-goal lifted; R6 hook | none; 01's larger corpora benefit from it |

Suggested execution order: **01 first** — it is the trigger-evaluation run that
decides whether 02 (τ_AP stalls while τ passes) and 03 (pass-rate/τ plateau)
fire at all. 04 and 05 are independent and can proceed in parallel with anything.

## PLAN.md decision-number allocation (binding)

Each spec pre-makes decisions destined for PLAN.md. Ranges are disjoint and
allocated in execution order; a spec MUST use only its range for its own
decisions (references to existing v1 decisions D1–D31 are unaffected):

| spec | range |
|---|---|
| 01 real benchmarks | **D32–D41** |
| 02 graded qrels | **D42–D51** |
| 03 prompt optimizer | **D52–D61** |
| 04 soft demand / diversity | **D62–D68** |
| 05 distributed + chromadb | **D69–D77** |

## schema_version 2 — canonical trigger list (single owner: this file)

The loader change is made ONCE (whichever spec merges first implements it,
citing this section): `Pipeline.from_yaml` accepts `schema_version ∈ {1, 2}`;
schema 1 semantics, outputs, and bytes are untouched. A config MUST declare
`schema_version: 2` iff it uses any of:

- `resources.execution` block, `chunk_store`, or `retriever.type: chromadb` (05)
- `qrel_builder.strategy: graded_umbrela` or `resources.qrel_judge` (02)
- `optimization` block or `resources.reflection_llm` (03)
- `resources.demand.estimator: soft_movmf` (04)
- `generator_llm.type: cached` (transcript replay), `partition.ladder`,
  `split_stratify_by`, or `validator.audit_export` (01)

Individual specs reference this table instead of re-defining the trigger set.
