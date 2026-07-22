# SPEC v2.06 — Content-coverage exploration: combining chunk clusters with the demand map

**Status:** proposed (v2.1 — after spec 01's first real-corpus run).
Decision numbers **D78–D84** allocated per [README](README.md).

---

## 0. Context & hypothesis

The demand map stratifies by **user intent** (query clusters, p̂); BCG-style coverage
stratifies by **corpus content** (chunk clusters). Each is blind where the other sees:
demand-matched sampling (A2) cannot test corpus regions users have not queried yet — by
construction, an injected regression under a zero-demand chunk cluster is invisible to every
demand-matched strategy. Content-only sampling (the BCG-implied quota cousin of A1, see spec 01's
lineage notes) is demand-blind and wrecks ESS.

**Hypothesis (R-item form).** H: adding a small, firewalled *exploration slice* routed by
content-coverage floors gives early-warning validity on cold corpus regions at ≤ 0.02 ESS cost
on the demand-weighted headline. Method: mixture-of-stratifications sampling (below). Meter:
cold-cluster positive control + unchanged headline metrics. Trigger: spec 01's real-corpus run
shows uncovered-but-growing chunk clusters, or the first churn epoch lands (R6).

## 1. Method

1. **Mixture of stratifications.** `π′ = (1−ε)·demand-tilted movMF + ε·exploration components`.
   The `SpecSampler` already supports this: it takes an `exploration` boolean mask per component
   and bypasses the τ_r on-manifold guard for flagged components
   (`sampling/spec_sampler.py:89`). No new sampler surface.
2. **Content clusters.** Fit KMeans on chunk embeddings with `C_d = 2 × C_q` (frozen artifact
   per epoch, D30 gitignore semantics). Content-coverage = MSC over chunk clusters
   (`minimum_semantic_coverage`, floor k=3) — the BCG adaptation already in
   `metrics/efficiency.py`.
3. **Routing.** Exploration components are placed on chunk clusters failing the floor, ranked:
   zero demand-projection first, then by chunk-count growth since the previous epoch.
4. **Exploration targets.** No queries exist there, so: generate one HyDE-style hypothetical
   query per routed cluster (generator LLM, one call), embed it, place a vMF component at that
   direction with κ = median κ of the fitted demand map. τ_r is bypassed (existing mask).
5. **Firewall (non-negotiable).** Exploration records carry `stratum=exploration`, are excluded
   from the demand-weighted headline and from fidelity/ESS metrics, and are reported in their own
   content-coverage block. This is the SPEC §2.1 two-products rule applied inside one run.
6. **Promotion.** An exploration cluster that receives ≥ `promote_min` real queries within an
   epoch is folded into the demand map at the next epoch refresh — a benchmark-migration event;
   the reporting partition never changes mid-epoch.

## 2. Config (schema_version 2 — trigger registered in README)

```yaml
resources:
  demand:
    estimator: hard_kmeans          # unchanged
    exploration: {eps: 0.10, content_cluster_factor: 2, floor_k: 3, promote_min: 25}
pipeline:
  - {type: seed_sampler.spec, params: {lam: 0.7, exploration_eps: 0.10}}
```

`eps` validated in [0, 0.2]; `eps: 0` reproduces v2 A2 byte-identically (regression fixture).

## 3. Meters & experiment — how we test it

Add a fifth strategy **A2X** (A2 + exploration slice) to the harness and run
`[a0, a1, a2, a2x, oracle]` on the toy world and one spec-01 corpus:

1. **Headline unchanged:** A2X's demand-weighted τ, ESS/N, wC2ST equal A2's within the
   bootstrap CI; ESS/N penalty ≤ 0.02 at eps = 0.10.
2. **Content coverage gained:** MSC over chunk clusters (floor 3) ≥ 0.95 for A2X where A2
   scores its unmodified (lower) value; per-cluster table reported (dual-view rule).
3. **The killer control — cold-cluster regression:** inject `drop_index` restricted to chunks of
   a zero-demand cluster. Expected: a2 p ≈ 1.0 (blind by design — this is the honest baseline
   statement, mirroring A1-on-noise), a2x p < 0.05 via its exploration records. This control
   ships in the battery as `drop_cold_cluster`.
4. **Firewall regression:** with the exploration slice present, headline metrics.json for the
   demand stratum is byte-identical to an `eps: 0` run at the same seed.
5. **Promotion:** unit fixture — synthetic epoch where a cold cluster gains `promote_min`
   queries → next-epoch demand map contains a component within cos ≥ 0.95 of its centroid.

## 4. Acceptance (Definition of Done)

- [ ] `eps: 0` byte-identical to A2 (fixture).
- [ ] Toy-world 5-strategy table: criteria 1–3 above green at seed 0.
- [ ] `drop_cold_cluster` control added to `metrics/validity/controls.py` battery with
      parity across strategies documented.
- [ ] Exploration records flagged, excluded from headline, reported in a content-coverage block.
- [ ] Promotion fixture green; ruff/mypy strict; round-trip byte-stable; schema-1 untouched.

## 5. Pre-made decisions

| # | Decision | Rationale |
|---|---|---|
| D78 | Exploration slice is a flagged stratum excluded from the demand headline. | Two-products firewall (SPEC §2.1). |
| D79 | `eps` default 0.10, validated [0, 0.2]; 0 ⇒ byte-identical A2. | ESS cost bound; safe rollout. |
| D80 | Content clusters: KMeans on chunk embeddings, `C_d = 2×C_q`, frozen per epoch. | Finer content structure; D30 artifact semantics. |
| D81 | Routing: under-floor (k=3) clusters, zero-demand first, then growth rate. | Early warning targets churn (R6). |
| D82 | Exploration targets via HyDE pseudo-query; κ = median demand-map κ; τ_r bypassed via the existing mask. | No queries to fit; reuse shipped sampler surface. |
| D83 | Promotion at ≥ `promote_min` real queries/epoch, applied only at epoch refresh. | Partition stability is v1 law. |
| D84 | Joint pooled query+chunk clustering **rejected** (won't-do): the modality gap makes clusters split by type, not topic. | Recorded so it is not relitigated. |

## 6. Open questions — defaults for the agent

- HyDE pseudo-query prompt: reuse `answer_first_v1` with a "write the question this passage
  answers" preamble; revisit only if exploration gate pass-rate < 50%.
- `promote_min`: default 25 (≈ the wC2ST 30/side floor minus margin); config-exposed.
- Whether exploration records may seed the *stress suite* directly: default yes (they are
  discovery by nature), behind the same stratum flag.
