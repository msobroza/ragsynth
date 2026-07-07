"""Helper for optional-extra dependencies (lazy import pattern, SPEC §3.3).

Adapters and optimizers lazy-import their backend (``dspy``, ``bm25s``,
``sentence_transformers``, ...) so the core package works without any
extras installed. This module centralises the error message so every
call site points users to the same install command.
"""

from __future__ import annotations


def require_optional(module: object | None, feature: str, extra: str) -> None:
    """Raise an actionable ImportError if an optional dependency is missing.

    Args:
        module: The optionally-imported module, or ``None`` if the import
            failed (call sites use ``try: import x except ImportError: x = None``).
        feature: User-facing name of the feature requesting the dependency
            (e.g. ``"BM25sRetriever"``).
        extra: Name of the extra that provides the dependency
            (e.g. ``"bm25"``).

    Raises:
        ImportError: If ``module`` is ``None``, with the exact install command.
    """
    if module is None:
        raise ImportError(
            f"{feature} requires the '{extra}' extra. Install with: uv sync --extra {extra}"
        )
