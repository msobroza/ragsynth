"""Validator: the 4-arm metric harness producing the EvalReport (SPEC §6.7).

Runs the configured comparison arms (via the ``arms/`` presets, PLAN D9),
computes the fidelity/efficiency/validity/diversity blocks against the
anchor split + system zoo, and writes ``metrics.json`` (deterministic, no
wall-clock -- PLAN D14), ``report.md``, ``records.jsonl``, and figures
under the experiment directory (the parent of the artifacts dir).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Self

import numpy as np

from ragsynth.domain import EvalReport
from ragsynth.io.artifacts import canonical_json
from ragsynth.metrics.diversity import distinct_n, semantic_dedup_rate
from ragsynth.metrics.efficiency import (
    cluster_importance_weights,
    demand_weighted_coverage,
    effective_sample_size,
    post_stratified_estimate,
    zero_query_clusters,
)
from ragsynth.metrics.fidelity import c2st_auc, kl_similarity_distributions, mmd_rbf
from ragsynth.metrics.fidelity import within_cluster_c2st as wc2st
from ragsynth.metrics.validity.agreement import ranking_agreement
from ragsynth.metrics.validity.controls import (
    drop_index_mask,
    noise_transform,
    paired_bootstrap_pvalue,
)
from ragsynth.metrics.validity.systems import MatrixSystem, evaluate_zoo
from ragsynth.pipeline.base import STEPS, PipelineStep

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from ragsynth.domain import AnnotationRecord
    from ragsynth.pipeline.base import PipelineState, Resources

logger = logging.getLogger(__name__)

_MIN_STRATUM_N = 30
_DEFAULT_CONTROLS: dict[str, float] = {"drop_frac": 0.10, "noise_sigma": 0.5, "truncate_k": 3}
_WORST_K = 3
_MIN_USABLE = 2


@STEPS.register("validator")
class Validator(PipelineStep):
    """Compute the per-arm metric suite and emit the EvalReport.

    ``reuse_pipeline_for`` names the arm whose records are the OUTER
    pipeline's accepted set (no double generation, PLAN D9); every other
    requested arm is generated fresh through its preset with
    ``arm_params[arm]`` merged over the preset defaults.
    """

    name = "validator"

    def __init__(
        self,
        resources: Resources,
        arms: list[str],
        n_boot: int = 1000,
        gates: dict[str, float] | None = None,
        k: int = 10,
        n_per_arm: int = 500,
        reuse_pipeline_for: str | None = None,
        arm_params: dict[str, dict[str, Any]] | None = None,
        controls: dict[str, float] | None = None,
        wc2st_min_per_side: int = 30,
    ) -> None:
        self._resources = resources
        self.arms = list(arms)
        self.n_boot = n_boot
        self.gates = dict(gates) if gates is not None else {"tau": 0.9, "tau_ap": 0.8}
        self.k = k
        self.n_per_arm = n_per_arm
        self.reuse_pipeline_for = reuse_pipeline_for
        self.arm_params = {key: dict(val) for key, val in (arm_params or {}).items()}
        self.controls = {**_DEFAULT_CONTROLS, **(controls or {})}
        self.wc2st_min_per_side = wc2st_min_per_side

    # ---- record collection -------------------------------------------------

    def _arm_records(self, arm: str, state: PipelineState) -> tuple[list[AnnotationRecord], bool]:
        if arm == self.reuse_pipeline_for:
            return self._cap(arm, list(state.accepted)), True
        from ragsynth.arms.base import run_arm

        params = dict(self.arm_params.get(arm, {}))
        params.setdefault("n_seeds", self.n_per_arm)
        params.setdefault("n_records", self.n_per_arm)
        return self._cap(arm, run_arm(arm, self._resources, params)), False

    def _cap(self, arm: str, records: list[AnnotationRecord]) -> list[AnnotationRecord]:
        """Deterministically subsample to ``n_per_arm`` for arm comparability.

        The prototype evaluates exactly ``n_arm`` records per arm; without the
        cap, arms that overgenerate would enjoy tighter bootstrap CIs than the
        oracle and the positive-control p-values would not be comparable.
        """
        if len(records) <= self.n_per_arm:
            return records
        rng = self._resources.rng(f"validator.subsample.{arm}")
        picks = sorted(rng.choice(len(records), size=self.n_per_arm, replace=False).tolist())
        return [records[i] for i in picks]

    def _record_embs(
        self, records: list[AnnotationRecord]
    ) -> tuple[NDArray[np.float64], list[AnnotationRecord]]:
        usable = [
            r
            for r in records
            if r.query.embedding_ref is not None
            and r.query.embedding_ref in self._resources.embeddings
        ]
        if len(usable) < len(records):
            logger.warning("%d records lack embeddings; excluded", len(records) - len(usable))
        refs = [r.query.embedding_ref for r in usable if r.query.embedding_ref is not None]
        return self._resources.embeddings.get(refs).astype(np.float64), usable

    # ---- metric blocks -----------------------------------------------------

    def _fidelity_block(
        self,
        arm_embs: NDArray[np.float64],
        real_ref: NDArray[np.float64],
        labels_real: NDArray[np.int_],
        labels_synth: NDArray[np.int_],
        chunk_embs: NDArray[np.float64],
    ) -> dict[str, Any]:
        seed = self._resources.seed
        wc2st_mean, per_cluster = wc2st(
            real_ref,
            arm_embs,
            labels_real,
            labels_synth,
            min_per_side=self.wc2st_min_per_side,
            seed=seed,
        )
        return {
            "kl": kl_similarity_distributions(real_ref, arm_embs, chunk_embs),
            "c2st_auc": c2st_auc(real_ref, arm_embs, seed=seed),
            "wc2st_mean": None if np.isnan(wc2st_mean) else wc2st_mean,
            "wc2st_per_cluster": {str(c): v for c, v in sorted(per_cluster.items())},
            "mmd": mmd_rbf(real_ref, arm_embs, seed=seed),
        }

    def _efficiency_block(
        self,
        labels_synth: NDArray[np.int_],
        base_scores: NDArray[np.float64],
    ) -> dict[str, Any]:
        p_hat = self._resources.demand.p_hat
        weights, coverage_gap = cluster_importance_weights(labels_synth, p_hat)
        per_cluster: list[dict[str, Any]] = []
        for cluster in range(len(p_hat)):
            mask = labels_synth == cluster
            per_cluster.append(
                {
                    "cluster": cluster,
                    "p_hat": float(p_hat[cluster]),
                    "n_synth": int(mask.sum()),
                    "mean_ndcg": float(base_scores[mask].mean()) if mask.any() else None,
                }
            )
        covered = [row for row in per_cluster if row["mean_ndcg"] is not None]
        worst = sorted(covered, key=lambda row: row["mean_ndcg"])[:_WORST_K]
        return {
            "ess_ratio": effective_sample_size(weights) / max(len(weights), 1),
            "coverage_gap": coverage_gap,
            "demand_weighted_coverage": demand_weighted_coverage(labels_synth, p_hat),
            "zero_query_clusters": zero_query_clusters(labels_synth, len(p_hat)),
            "post_stratified_ndcg": post_stratified_estimate(base_scores, labels_synth, p_hat),
            "unweighted_mean_ndcg": float(base_scores.mean()) if len(base_scores) else None,
            "per_cluster": per_cluster,
            "worst_clusters": [row["cluster"] for row in worst],
        }

    def _validity_block(
        self,
        arm_embs: NDArray[np.float64],
        arm_qrels: list[dict[str, int]],
        anchor_scores: NDArray[np.float64],
        records: list[AnnotationRecord],
    ) -> tuple[dict[str, Any], NDArray[np.float64]]:
        resources = self._resources
        seed = resources.seed
        zoo = dict(resources.zoo)
        arm_scores = evaluate_zoo(zoo, arm_embs, arm_qrels, k=self.k)
        agreement = ranking_agreement(anchor_scores, arm_scores, n_boot=self.n_boot, seed=seed)

        exact = zoo["exact"]
        if not isinstance(exact, MatrixSystem):
            raise TypeError("controls need the 'exact' zoo system to be a MatrixSystem")
        base = exact.per_query_scores(arm_embs, arm_qrels, k=self.k)
        controls_rng = np.random.default_rng([seed, 0xC0]).spawn(2)
        n_chunks = len(resources.chunks)
        drop = exact.per_query_scores(
            arm_embs,
            arm_qrels,
            k=self.k,
            drop_mask=drop_index_mask(n_chunks, self.controls["drop_frac"], controls_rng[0]),
        )
        noisy_system = MatrixSystem(
            name="noise-control",
            matrix=noise_transform(
                arm_embs.shape[1], self.controls["noise_sigma"], controls_rng[1]
            ),
            chunk_ids=exact.chunk_ids,
            chunk_embs=exact.chunk_embs,
        )
        noisy = noisy_system.per_query_scores(arm_embs, arm_qrels, k=self.k)
        truncated = exact.per_query_scores(arm_embs, arm_qrels, k=int(self.controls["truncate_k"]))
        controls = {}
        for control_name, degraded in (
            ("drop_index", drop),
            ("noise", noisy),
            ("truncate_topk", truncated),
        ):
            delta, p_value = paired_bootstrap_pvalue(base, degraded, seed=seed)
            controls[control_name] = {"delta": delta, "p_value": p_value}

        per_stratum: dict[str, Any] = {}
        by_stratum: dict[str, list[int]] = {}
        for i, record in enumerate(records):
            by_stratum.setdefault(record.stratum.key(), []).append(i)
        for key, indices in sorted(by_stratum.items()):
            if len(indices) >= _MIN_STRATUM_N:
                sub = ranking_agreement(
                    anchor_scores, arm_scores[:, indices], n_boot=self.n_boot, seed=seed
                )
                per_stratum[key] = {
                    "tau": sub.tau,
                    "tau_ap": sub.tau_ap_,
                    "n": len(indices),
                    "passed": bool(
                        sub.tau >= self.gates.get("tau", 0.9)
                        and sub.tau_ap_ >= self.gates.get("tau_ap", 0.8)
                    ),
                }
            else:
                per_stratum[key] = {
                    "tau": None,
                    "tau_ap": None,
                    "n": len(indices),
                    "passed": None,
                }

        block = {
            "tau": agreement.tau,
            "tau_ci": [agreement.tau_ci_low, agreement.tau_ci_high],
            "tau_ap": agreement.tau_ap_,
            "rbo": agreement.rbo,
            "controls": controls,
            "per_stratum": per_stratum,
        }
        return block, base

    # ---- run ---------------------------------------------------------------

    def run(self, state: PipelineState) -> PipelineState:
        """Build every arm's metric block, then the report + outputs."""
        resources = self._resources
        chunk_embs = resources.chunk_embs()
        anchor_embs = resources.query_embs("anchor")
        anchor_qrels = [
            dict(resources.anchor_qrels.get(q.query_id, {})) for q in resources.queries_anchor
        ]
        anchor_scores = evaluate_zoo(dict(resources.zoo), anchor_embs, anchor_qrels, k=self.k)

        arm_blocks: dict[str, dict[str, Any]] = {}
        gates_passed: dict[str, bool] = {}
        # ONE shared anchor subsample base for every arm's fidelity block
        # (prototype L860 protocol): per-arm equal-n views are nested prefixes.
        ref_rng = resources.rng("validator.fidelity")
        n_ref_base = min(len(anchor_embs), self.n_per_arm)
        real_ref_base = anchor_embs[
            ref_rng.choice(len(anchor_embs), size=n_ref_base, replace=False)
        ]
        for arm in self.arms:
            records, reused = self._arm_records(arm, state)
            arm_embs, usable = self._record_embs(records)
            if len(usable) < _MIN_USABLE:
                logger.warning("arm '%s' produced %d usable records; skipping", arm, len(usable))
                arm_blocks[arm] = {"n_records": len(usable), "skipped": True}
                gates_passed[arm] = False
                continue
            # Equal-n on BOTH sides (SPEC §8): trim the shared reference to the
            # arm size, and subsample the arm when it exceeds the reference.
            n_fid = min(n_ref_base, len(arm_embs))
            real_ref = real_ref_base[:n_fid]
            if len(arm_embs) > n_fid:
                arm_rng = resources.rng(f"validator.fidelity.arm.{arm}")
                fid_embs = arm_embs[arm_rng.choice(len(arm_embs), size=n_fid, replace=False)]
            else:
                fid_embs = arm_embs
            labels_real = resources.partition.assign(real_ref)
            labels_synth = resources.partition.assign(arm_embs)
            arm_qrels = [dict(r.qrels) for r in usable]

            validity, base_scores = self._validity_block(arm_embs, arm_qrels, anchor_scores, usable)
            block: dict[str, Any] = {
                "n_records": len(usable),
                "reused_pipeline_records": reused,
                "fidelity": self._fidelity_block(
                    fid_embs,
                    real_ref,
                    labels_real,
                    resources.partition.assign(fid_embs),
                    chunk_embs,
                ),
                "efficiency": self._efficiency_block(labels_synth, base_scores),
                "validity": validity,
                "diversity": {
                    "distinct_1": distinct_n([r.query.text for r in usable], 1),
                    "distinct_2": distinct_n([r.query.text for r in usable], 2),
                    "semantic_dedup_rate": semantic_dedup_rate(arm_embs),
                },
                "gate_reject_reasons": dict(state.metrics.get("gate_reject_reasons", {}))
                if reused
                else {},
            }
            passed = validity["tau"] >= self.gates.get("tau", 0.9) and validity[
                "tau_ap"
            ] >= self.gates.get("tau_ap", 0.8)
            block["gates_passed"] = passed
            gates_passed[arm] = passed
            arm_blocks[arm] = block

        provenance = dict(state.provenance)
        report = EvalReport(
            name=str(provenance.get("name", "ragsynth")),
            config=dict(provenance.get("config", {})),
            config_hash=str(provenance.get("config_hash", "")),
            seed=resources.seed,
            arms=arm_blocks,
            gates=self.gates,
            gates_passed=gates_passed,
            provenance=provenance,
        )
        state.metrics["eval_report"] = report.metrics_payload()
        self._write_outputs(report, state)
        return state

    # ---- outputs -----------------------------------------------------------

    def _output_dir(self) -> Path:
        return self._resources.artifacts.root.parent

    def _write_outputs(self, report: EvalReport, state: PipelineState) -> None:
        out = self._output_dir()
        out.mkdir(parents=True, exist_ok=True)
        (out / "metrics.json").write_text(canonical_json(report.metrics_payload()))
        (out / "report.md").write_text(self._render_markdown(report))
        with (out / "records.jsonl").open("w") as fh:
            for record in state.accepted:
                fh.write(record.model_dump_json() + "\n")
        self._render_figures(report, out / "figures")
        logger.info("validator outputs written under %s", out)

    def _render_markdown(self, report: EvalReport) -> str:
        return render_markdown(report, k=self.k)

    def _render_figures(self, report: EvalReport, fig_dir: Path) -> None:
        render_figures(report, fig_dir, tau_gate=self.gates.get("tau", 0.9))

    def to_config(self) -> dict[str, Any]:
        """JSON-safe constructor params."""
        return {
            "arms": self.arms,
            "n_boot": self.n_boot,
            "gates": self.gates,
            "k": self.k,
            "n_per_arm": self.n_per_arm,
            "reuse_pipeline_for": self.reuse_pipeline_for,
            "arm_params": self.arm_params,
            "controls": self.controls,
            "wc2st_min_per_side": self.wc2st_min_per_side,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any], resources: Resources) -> Self:
        """Build from a config params block."""
        return cls(resources, **config)


def render_markdown(report: EvalReport, k: int) -> str:
    """Render the report.md content from an EvalReport (used by ``ragsynth report``)."""

    def fmt(value: Any, spec: str = ".3f") -> str:
        if value is None:
            return "-"
        return format(value, spec)

    lines = [
        f"# EvalReport - {report.name}",
        "",
        f"Config hash: `{report.config_hash[:12]}` - seed {report.seed}",
        "",
        "Reading guide: KL/C2ST/wC2ST/MMD lower is better (0.5 = indistinguishable",
        "for AUCs); ESS/N higher is better; tau/tau_AP/RBO vs the anchor ranking",
        "higher is better; control p-values < 0.05 mean the arm detects the",
        "injected regression.",
        "",
        "| arm | n | KL | C2ST | wC2ST | MMD | ESS/N | gap | tau | tau 95% CI | tau_AP |"
        " RBO | PC drop | PC noise | gates |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for arm, block in report.arms.items():
        if block.get("skipped"):
            lines.append(f"| {arm} | {block['n_records']} | skipped |" + " - |" * 12)
            continue
        fid, eff, val = block["fidelity"], block["efficiency"], block["validity"]
        ci = val["tau_ci"]
        lines.append(
            f"| {arm} | {block['n_records']} | {fmt(fid['kl'])} | {fmt(fid['c2st_auc'])} "
            f"| {fmt(fid['wc2st_mean'])} | {fmt(fid['mmd'], '.4f')} "
            f"| {fmt(eff['ess_ratio'], '.2f')} | {fmt(eff['coverage_gap'], '.2f')} "
            f"| {fmt(val['tau'])} | [{fmt(ci[0], '.2f')}, {fmt(ci[1], '.2f')}] "
            f"| {fmt(val['tau_ap'])} | {fmt(val['rbo'], '.2f')} "
            f"| {fmt(val['controls']['drop_index']['p_value'])} "
            f"| {fmt(val['controls']['noise']['p_value'])} "
            f"| {'PASS' if block['gates_passed'] else 'fail'} |"
        )
    lines.append("")
    for arm, block in report.arms.items():
        if block.get("skipped"):
            continue
        eff = block["efficiency"]
        lines += [
            f"## {arm}",
            "",
            "Dual view (SPEC §8-9): demand-weighted headline "
            f"nDCG@{k} = {fmt(eff['post_stratified_ndcg'])} "
            f"(unweighted {fmt(eff['unweighted_mean_ndcg'])}, "
            f"ESS/N {fmt(eff['ess_ratio'], '.2f')}); per-cluster table below; "
            f"worst clusters: {eff['worst_clusters']}; "
            f"zero-query clusters: {eff['zero_query_clusters']}.",
            "",
            "| cluster | p_hat | n_synth | mean nDCG |",
            "|---|---|---|---|",
        ]
        lines += [
            f"| {row['cluster']} | {fmt(row['p_hat'])} | {row['n_synth']} "
            f"| {fmt(row['mean_ndcg'])} |"
            for row in eff["per_cluster"]
        ]
        if block.get("gate_reject_reasons"):
            ordered = dict(sorted(block["gate_reject_reasons"].items()))
            lines += ["", f"Gate reject reasons: {ordered}"]
        lines.append("")
    return "\n".join(lines)


def render_figures(report: EvalReport, fig_dir: Path, tau_gate: float) -> None:
    """Render the four report figures (used by ``ragsynth report``)."""
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    arms = [a for a, b in report.arms.items() if not b.get("skipped")]

    def bars(filename: str, series: dict[str, list[float | None]], title: str, ylabel: str) -> None:
        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(len(arms))
        width = 0.8 / max(len(series), 1)
        for i, (label, values) in enumerate(series.items()):
            heights = [v if v is not None else 0.0 for v in values]
            ax.bar(x + i * width, heights, width, label=label)
        ax.set_xticks(x + 0.4 - width / 2)
        ax.set_xticklabels(arms)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / filename)
        plt.close(fig)

    bars(
        "fidelity_bars.png",
        {
            "KL": [report.arms[a]["fidelity"]["kl"] for a in arms],
            "C2ST": [report.arms[a]["fidelity"]["c2st_auc"] for a in arms],
            "wC2ST": [report.arms[a]["fidelity"]["wc2st_mean"] for a in arms],
        },
        "Fidelity per arm (lower is better)",
        "value",
    )
    bars(
        "ess_coverage.png",
        {
            "ESS/N": [report.arms[a]["efficiency"]["ess_ratio"] for a in arms],
            "coverage": [report.arms[a]["efficiency"]["demand_weighted_coverage"] for a in arms],
        },
        "Efficiency per arm (higher is better)",
        "ratio",
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    taus = [report.arms[a]["validity"]["tau"] for a in arms]
    los = [report.arms[a]["validity"]["tau_ci"][0] for a in arms]
    his = [report.arms[a]["validity"]["tau_ci"][1] for a in arms]
    x = np.arange(len(arms))
    # tau is computed on the full data and can land outside the bootstrap
    # percentile band at small n_boot; errorbar arms must be non-negative.
    lower = np.maximum(np.subtract(taus, los), 0.0)
    upper = np.maximum(np.subtract(his, taus), 0.0)
    ax.errorbar(x, taus, yerr=[lower, upper], fmt="o", capsize=4)
    ax.axhline(tau_gate, linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(arms)
    ax.set_title("System-ranking agreement (tau vs anchor, bootstrap 95% CI)")
    ax.set_ylabel("Kendall tau")
    fig.tight_layout()
    fig.savefig(fig_dir / "tau_ci.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    reasons: dict[str, int] = {}
    for block in report.arms.values():
        for reason, count in sorted(block.get("gate_reject_reasons", {}).items()):
            reasons[reason] = reasons.get(reason, 0) + count
    reasons = dict(sorted(reasons.items()))
    if reasons:
        ax.bar(list(reasons.keys()), list(reasons.values()))
    ax.set_title("Gate reject reasons (pipeline arm)")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(fig_dir / "gate_rejects.png")
    plt.close(fig)
