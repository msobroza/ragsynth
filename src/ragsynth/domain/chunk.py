"""Chunk: one retrievable unit of the knowledge base."""

from __future__ import annotations

import hashlib

from pydantic import BaseModel, ConfigDict, Field

_ID_LEN = 16


def _norm(text: str) -> str:
    """Whitespace-normalize text for content addressing."""
    return " ".join(text.split())


def _sha16(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:_ID_LEN]


class Chunk(BaseModel):
    """A knowledge-base chunk (SPEC §4).

    ``canonical_bbox``/``page_image_ref`` are carried for the v3 visual
    grounding setting (ColPali/ColQwen) but unused in v1 generation, which
    conditions on text only. Embeddings live in the ``EmbeddingStore``;
    ``embedding_ref`` is the key (never inline vectors).
    """

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    doc_id: str
    text: str
    content_hash: str
    embedding_ref: str | None = None
    page: int | None = None
    canonical_bbox: tuple[float, float, float, float] | None = None
    page_image_ref: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)

    @classmethod
    def create(
        cls,
        text: str,
        doc_id: str,
        page: int | None = None,
        canonical_bbox: tuple[float, float, float, float] | None = None,
        page_image_ref: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> Chunk:
        """Build a chunk with content-addressed ids.

        ``chunk_id = sha256(norm_text|doc_id|page|bbox)[:16]`` identifies the
        chunk *placement*; ``content_hash = sha256(norm_text)[:16]`` identifies
        the text alone (the v2 lifecycle key -- SPEC §2.2).

        Args:
            text: Raw chunk text (whitespace is normalized for hashing only).
            doc_id: Owning document id.
            page: Optional 1-based page number.
            canonical_bbox: Optional (x0, y0, x1, y1) region on the page (v3).
            page_image_ref: Optional rendered-page artifact key (v3).
            metadata: Optional free-form string metadata.

        Returns:
            A frozen :class:`Chunk` with ids filled in.
        """
        norm = _norm(text)
        return cls(
            chunk_id=_sha16(f"{norm}|{doc_id}|{page}|{canonical_bbox}"),
            doc_id=doc_id,
            text=text,
            content_hash=_sha16(norm),
            page=page,
            canonical_bbox=canonical_bbox,
            page_image_ref=page_image_ref,
            metadata=metadata or {},
        )
