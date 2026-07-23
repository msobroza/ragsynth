# ragsynth

Synthetic query generation & validation for RAG retrieval evaluation over
**dynamic knowledge bases**. ragsynth turns corpus chunks into gate-verified
synthetic queries with relevance annotations (qrels), and — the part that
makes the output trustworthy — validates every generated set against your
production query distribution: system-ranking agreement (Kendall τ / τ_AP)
against a real-query anchor set, positive controls (does the benchmark
detect injected regressions?), demand-weighted coverage, and
distribution-fidelity metrics (KL, C2ST, MMD, within-cluster C2ST).

`SPEC.md` is the design source of truth; `PLAN.md` records every
implementation decision; `ARCHITECTURE.md` shows the pipeline state flow
and the extension recipe.

## Quickstart

```bash
uv sync
uv run ragsynth run --config configs/v1_toy.yaml           # 4-arm toy world, ~30 s
uv run ragsynth run --config configs/v1_local_corpus.yaml  # bundled 200-chunk corpus
uv run ragsynth validate --config configs/v1_toy.yaml      # config check only
uv run ragsynth report --config configs/v1_toy.yaml        # re-render report + figures
```

Outputs land under the experiment directory (`experiments/v1/` for the toy
config): `metrics.json` (deterministic — same seed, same bytes), `report.md`
(the 4-arm table + dual-view per-cluster breakdown), `records.jsonl` (the
benchmark), `figures/`, and `artifacts/` (sha256-manifested partition and
demand-map artifacts).

## v2 benchmarks quickstart

Real-benchmark validation (FiQA-2018, NFCorpus, LegalBench-RAG) against
production-scale text, embeddings, and cross-family LLMs — `specs/v2/01-real-benchmarks.md`
is the design source of truth; PLAN.md D32-D41 records the decisions.

```bash
cp .env.example .env                              # fill in GEMINI_API_KEY
uv sync --extra gemini --extra chromadb            # gemini embedder + ChromaDB cache
uv run python scripts/fetch_benchmarks.py fiqa     # or nfcorpus
uv run python scripts/convert_benchmark.py fiqa    # -> data/benchmarks/fiqa/*.jsonl
uv run ragsynth run --config configs/v2_fiqa.yaml  # record run: embeds once (cached_chroma),
                                                    # calls the live LLM endpoints once (cached
                                                    # transcripts); replay run (offline, byte-
                                                    # identical metrics.json) lands in the second
                                                    # half of this work.
```

Embeddings go through `cached_chroma` (gemini-embedding-2, 768-dim, keyed by
`sha256(embedder_id)`) so a dataset is only ever embedded once.
`convert_benchmark.py` supports all three datasets, but LegalBench-RAG's
fetch entry is deferred to the second half — place its raw archive under
`data/benchmarks/legalbench_rag/raw/` manually until then. Outputs land
under `experiments/v2_<name>/`.

## The four arms

Every run can validate itself against three baselines plus the ceiling
(SPEC §10): **A0** naive uniform chunk seeds, **A1** demand-quota seeds +
style exemplars, **A2** spec-first sampling from a demand-tilted movMF over
production embeddings, and **ORACLE** — a held-out slice of real queries
(the τ ceiling; it has its own CI).

## Offline-first

Runtime dependencies are Apache/MIT/BSD only; every LLM/embedding call goes
through an adapter Protocol with deterministic offline mocks, so the entire
test suite and both bundled configs run air-gapped. Real deployments point
`openai_compatible` at any vLLM/LiteLLM-style endpoint.

## Extras

| extra | install | provides |
|---|---|---|
| `st` | `uv sync --extra st` | `sentence_transformer` embedder (real semantic quality; default is the pure-numpy `hashed_ngram`) |
| `gemini` | `uv sync --extra gemini` | `gemini` embedder (google-genai; the v2 benchmark default, D39-A) |
| `chromadb` | `uv sync --extra chromadb` | `cached_chroma` embedder cache (embed each dataset once, replay from ChromaDB) |
| `bm25` | `uv sync --extra bm25` | `bm25s` retriever (text-query path) |
| `optimization` | `uv sync --extra optimization` | dspy/textgrad backends for the v2 prompt-optimization loop (v1 ships the contract + `NoOpOptimizer`) |
| `notebooks` | `uv sync --extra notebooks` | jupyterlab + seaborn for `experiments/*/analysis.ipynb` |

## Development

```bash
uv run ruff check . && uv run ruff format --check .
uv run mypy src
uv run pytest -q --cov=ragsynth --cov-fail-under=70
```

All three plus the toy-config run are green at every commit (SPEC §15.3).
Roadmap (R1-R10: scaled A2, closed-loop prompt optimization, graded qrels,
churn lifecycle, conversational and visual grounding) lives in SPEC §16;
experiment tracking (mlflow) is a v2 option.
