"""ragsynth: synthetic query generation & validation for RAG retrieval evaluation.

Generates gate-verified synthetic queries with relevance annotations (qrels)
from corpus chunks and validates them against production query distributions
(fidelity / efficiency / validity metric suites, 4-arm experiment harness).
See ``SPEC.md`` for the design rationale and ``ARCHITECTURE.md`` for the
pipeline state flow.
"""

__version__ = "0.1.0"
