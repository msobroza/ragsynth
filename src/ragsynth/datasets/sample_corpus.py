"""Deterministic bundled sample corpus generator (PLAN D18).

Writes ``chunks.jsonl`` and ``queries.jsonl`` describing "Ledgerline", a
fictional B2B fintech product suite, across eight topics that act as the
corpus's latent clusters. Chunk texts are 2-4 template-driven sentences
with topic-specific vocabulary; queries are user questions with a skewed
topic distribution (``1/r^1.1``) and mixed registers (questions,
keyword-ish fragments, how-to phrasing).

The only randomness source is ``default_rng([seed,
stable_hash64("sample_corpus")])``, so regeneration with default arguments
is byte-identical to the committed ``data/sample/`` files -- guarded by
``tests/datasets/test_sample_corpus.py``.

Usage:
    python -m ragsynth.datasets.sample_corpus [out_dir]   # default data/sample
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

from ragsynth.pipeline.base import stable_hash64

logger = logging.getLogger(__name__)

_DEFAULT_N_CHUNKS = 200
_DEFAULT_N_QUERIES = 120
_CHUNKS_PER_DOC = 5
_TOPIC_SKEW = 1.1
_MIN_SENTENCES = 2
_MAX_SENTENCES = 4
_DEFAULT_OUT_DIR = "data/sample"

TOPICS = (
    "billing",
    "api-auth",
    "webhooks",
    "rate-limits",
    "compliance",
    "sdk",
    "dashboards",
    "exports",
)

_VOCAB: dict[str, dict[str, tuple[str, ...]]] = {
    "billing": {
        "subject": (
            "the invoicing engine",
            "Ledgerline Billing",
            "the subscription ledger",
            "the proration service",
        ),
        "action": (
            "issue consolidated invoices",
            "apply prorated credits",
            "reconcile settlement batches",
            "schedule dunning reminders",
        ),
        "object": (
            "invoice line items",
            "billing cycles",
            "credit notes",
            "payment settlements",
        ),
        "detail": (
            "with per-currency rounding applied automatically",
            "before the statement period closes at midnight UTC",
            "so finance teams keep a clean audit trail",
            "without disturbing downstream tax calculations",
        ),
    },
    "api-auth": {
        "subject": (
            "the token service",
            "Ledgerline API authentication",
            "the key-rotation scheduler",
            "the OAuth gateway",
        ),
        "action": (
            "rotate signing keys",
            "mint scoped access tokens",
            "revoke compromised credentials",
            "enforce mutual TLS",
        ),
        "object": (
            "API keys",
            "refresh tokens",
            "service-account scopes",
            "session signatures",
        ),
        "detail": (
            "with a fifteen-minute expiry by default",
            "while older credentials keep working during the grace window",
            "so integrations never share long-lived secrets",
            "and every issuance is written to the security log",
        ),
    },
    "webhooks": {
        "subject": (
            "the webhook dispatcher",
            "Ledgerline Events",
            "the delivery retry queue",
            "the endpoint verifier",
        ),
        "action": (
            "replay failed deliveries",
            "sign event payloads",
            "deduplicate concurrent notifications",
            "verify subscriber endpoints",
        ),
        "object": (
            "event subscriptions",
            "delivery receipts",
            "payload signatures",
            "callback URLs",
        ),
        "detail": (
            "with exponential backoff capped at one hour",
            "so consumers can confirm authenticity offline",
            "even when the receiving service is briefly unreachable",
            "and dead-lettered events stay queryable for seven days",
        ),
    },
    "rate-limits": {
        "subject": (
            "the throttling layer",
            "Ledgerline rate limiting",
            "the quota manager",
            "the burst allowance tracker",
        ),
        "action": (
            "smooth traffic spikes",
            "allocate per-tenant quotas",
            "return retry-after headers",
            "raise soft limits temporarily",
        ),
        "object": (
            "request budgets",
            "concurrency ceilings",
            "burst windows",
            "throttle responses",
        ),
        "detail": (
            "measured over a sliding sixty-second window",
            "so a single tenant cannot starve the cluster",
            "with headroom reserved for interactive traffic",
            "before hard rejections ever kick in",
        ),
    },
    "compliance": {
        "subject": (
            "the audit vault",
            "Ledgerline Compliance",
            "the data-retention planner",
            "the residency controller",
        ),
        "action": (
            "export immutable audit logs",
            "enforce retention schedules",
            "pin records to a region",
            "produce SOC 2 evidence",
        ),
        "object": (
            "audit trails",
            "retention policies",
            "residency zones",
            "access reviews",
        ),
        "detail": (
            "with tamper-evident hashing on every entry",
            "as required by PCI DSS assessors",
            "so regulators can verify chain of custody",
            "without moving customer data across borders",
        ),
    },
    "sdk": {
        "subject": (
            "the Python SDK",
            "Ledgerline client libraries",
            "the typed request builder",
            "the pagination helper",
        ),
        "action": (
            "retry idempotent calls",
            "stream large result sets",
            "validate payloads locally",
            "surface typed errors",
        ),
        "object": (
            "client sessions",
            "response iterators",
            "request models",
            "error hierarchies",
        ),
        "detail": (
            "with sensible defaults baked into the constructor",
            "so developers catch mistakes before any network call",
            "across Python, TypeScript, and Go",
            "while keeping memory usage flat on huge collections",
        ),
    },
    "dashboards": {
        "subject": (
            "the analytics dashboard",
            "Ledgerline Insights",
            "the widget composer",
            "the alerting panel",
        ),
        "action": (
            "chart settlement volumes",
            "pin saved views",
            "drill into tenant activity",
            "trigger threshold alerts",
        ),
        "object": (
            "revenue widgets",
            "usage heatmaps",
            "saved filters",
            "alert rules",
        ),
        "detail": (
            "refreshed every five minutes from the reporting replica",
            "so on-call staff notice anomalies quickly",
            "with per-team visibility controls",
            "and each view can be shared by URL",
        ),
    },
    "exports": {
        "subject": (
            "the export scheduler",
            "Ledgerline data exports",
            "the CSV streamer",
            "the warehouse sync job",
        ),
        "action": (
            "ship nightly snapshots",
            "stream incremental changes",
            "compress archival dumps",
            "backfill historical periods",
        ),
        "object": (
            "ledger snapshots",
            "change feeds",
            "parquet files",
            "destination buckets",
        ),
        "detail": (
            "delivered to S3-compatible storage of your choice",
            "with checksums published alongside every file",
            "so downstream warehouses stay consistent",
            "and partial failures resume from the last good offset",
        ),
    },
}

_SENTENCE_TEMPLATES = (
    "{cap_subject} lets teams {action} across {object} {detail}.",
    "To {action}, open {subject} and review the affected {object} {detail}.",
    "{cap_subject} tracks {object} continuously and can {action} {detail}.",
    "Operators use {subject} to {action} whenever {object} change {detail}.",
    "When enabled, {subject} will {action} against the current {object} {detail}.",
    "{cap_subject} keeps {object} consistent and can {action} on demand {detail}.",
)

_QUERY_REGISTERS = ("question", "howto", "keyword")
_QUERY_REGISTER_PROBS = (0.5, 0.3, 0.2)

_QUERY_TEMPLATES: dict[str, tuple[str, ...]] = {
    "question": (
        "How do I {action} in Ledgerline?",
        "Why does Ledgerline show stale {object} after I {action}?",
        "Can Ledgerline {action} automatically for all {object}?",
        "What happens to {object} when I {action} mid-cycle?",
    ),
    "howto": (
        "how to {action} in ledgerline",
        "steps to {action} without breaking {object}",
        "guide for setting up {object}",
    ),
    "keyword": (
        "ledgerline {topic} {object}",
        "{topic} {object} not working",
        "{topic} {object} configuration",
    ),
}


def _capitalize(phrase: str) -> str:
    """Uppercase only the first character (proper nouns stay intact)."""
    return phrase[0].upper() + phrase[1:]


def _pick(rng: np.random.Generator, pool: tuple[str, ...]) -> str:
    """Draw one element from ``pool`` uniformly."""
    return pool[int(rng.integers(len(pool)))]


def _chunk_rows(n_chunks: int, rng: np.random.Generator) -> list[dict[str, Any]]:
    """Template-driven chunk texts, topic-major, five chunks per doc."""
    rows: list[dict[str, Any]] = []
    doc_counters: dict[str, int] = {}
    for i in range(n_chunks):
        topic = TOPICS[(i * len(TOPICS)) // n_chunks]
        within_topic = doc_counters.get(topic, 0)
        doc_counters[topic] = within_topic + 1
        vocab = _VOCAB[topic]
        n_sentences = int(rng.integers(_MIN_SENTENCES, _MAX_SENTENCES + 1))
        sentences = []
        for _ in range(n_sentences):
            template = _pick(rng, _SENTENCE_TEMPLATES)
            subject = _pick(rng, vocab["subject"])
            sentences.append(
                template.format(
                    subject=subject,
                    cap_subject=_capitalize(subject),
                    action=_pick(rng, vocab["action"]),
                    object=_pick(rng, vocab["object"]),
                    detail=_pick(rng, vocab["detail"]),
                )
            )
        rows.append(
            {
                "text": " ".join(sentences),
                "doc_id": f"{topic}-doc-{within_topic // _CHUNKS_PER_DOC:02d}",
                "metadata": {"topic": topic},
            }
        )
    return rows


def _query_rows(n_queries: int, rng: np.random.Generator) -> list[dict[str, Any]]:
    """Realistic user queries with skewed topic demand and mixed registers."""
    weights = 1.0 / np.arange(1, len(TOPICS) + 1) ** _TOPIC_SKEW
    weights = weights / weights.sum()
    rows: list[dict[str, Any]] = []
    for i in range(n_queries):
        topic = TOPICS[int(rng.choice(len(TOPICS), p=weights))]
        vocab = _VOCAB[topic]
        register = _QUERY_REGISTERS[
            int(rng.choice(len(_QUERY_REGISTERS), p=list(_QUERY_REGISTER_PROBS)))
        ]
        template = _pick(rng, _QUERY_TEMPLATES[register])
        text = template.format(
            topic=topic,
            action=_pick(rng, vocab["action"]),
            object=_pick(rng, vocab["object"]),
        )
        rows.append({"query_id": f"q{i:05d}", "text": text})
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write one compact JSON object per line (byte-stable)."""
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def generate_sample_corpus(
    out_dir: Path,
    n_chunks: int = _DEFAULT_N_CHUNKS,
    n_queries: int = _DEFAULT_N_QUERIES,
    seed: int = 0,
) -> None:
    """Write ``chunks.jsonl`` and ``queries.jsonl`` into ``out_dir``.

    Args:
        out_dir: Target directory (created if missing).
        n_chunks: Number of chunk lines (spread evenly over the topics).
        n_queries: Number of query lines (skewed over the topics).
        seed: Config seed; the sole rng is
            ``default_rng([seed, stable_hash64("sample_corpus")])``.
    """
    rng = np.random.default_rng([seed, stable_hash64("sample_corpus")])
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    _write_jsonl(target / "chunks.jsonl", _chunk_rows(n_chunks, rng))
    _write_jsonl(target / "queries.jsonl", _query_rows(n_queries, rng))
    logger.info(
        f"sample corpus written to {target}: {n_chunks} chunks, {n_queries} queries (seed={seed})"
    )


def main(argv: list[str] | None = None) -> None:
    """CLI entry: ``python -m ragsynth.datasets.sample_corpus [out_dir]``."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = sys.argv[1:] if argv is None else argv
    out_dir = Path(args[0]) if args else Path(_DEFAULT_OUT_DIR)
    generate_sample_corpus(out_dir)


if __name__ == "__main__":
    main()
