"""Data structures shared across the ingestion and retrieval pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

BlockKind = Literal["heading", "paragraph", "table", "figure"]
ExtractSource = Literal["text", "vision", "native"]


@dataclass
class PageMarkdown:
    """One page as structured Markdown. Every extractor produces these, so
    nothing downstream can tell which path (text or vision) produced them."""

    page_no: int
    markdown: str
    source: ExtractSource


@dataclass
class Block:
    """A structural unit of a document: a heading, paragraph, table, or figure.

    `page_start`/`page_end` differ when a block was stitched across a page
    break, which is why citations can render as "pp. 4-5".
    """

    kind: BlockKind
    text: str
    page_start: int
    page_end: int
    level: int = 0  # heading depth; 0 for non-headings

    @property
    def is_open_ended(self) -> bool:
        """True when the text stops mid-sentence -- the signal that this block
        probably continues onto the next page."""
        stripped = self.text.rstrip()
        return bool(stripped) and stripped[-1] not in ".!?:;\"')]}"


@dataclass
class Chunk:
    """A retrieval unit: what gets embedded and stored."""

    text: str
    heading_path: str
    page_start: int
    page_end: int
    chunk_index: int
    parent_id: str
    char_start: int
    char_end: int

    def embedding_text(self) -> str:
        """Text actually sent to the embedding model.

        The heading path is prefixed so the vector carries the section context
        a bare excerpt loses -- a cheap and substantial retrieval gain.
        """
        return f"{self.heading_path}\n\n{self.text}" if self.heading_path else self.text


@dataclass
class DocumentMeta:
    doc_id: str
    doc_name: str
    mime_type: str
    size_bytes: int
    page_count: int
    chunk_count: int
    vision_pages: list[int] = field(default_factory=list)
    uploaded_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class RetrievedChunk:
    """A search hit.

    After parent expansion, `parent_text` holds the whole enclosing section --
    what the LLM actually reads -- while `char_start`/`char_end` still point at
    the smaller chunk that matched, so citation highlighting stays precise.
    """

    text: str
    doc_id: str
    doc_name: str
    heading_path: str
    page_start: int
    page_end: int
    chunk_index: int
    parent_id: str
    char_start: int
    char_end: int
    # Weaviate's fused hybrid score. RELATIVE to the other results in the same
    # query -- the top hit is ~1.0 however poor the match -- so it ranks but
    # must never be used to decide groundedness.
    score: float
    # Absolute cosine similarity between the query and chunk vectors. Independent
    # of the result set, which is what makes it a valid grounding threshold.
    similarity: float = 0.0
    parent_text: str | None = None
    parent_char_start: int | None = None

    @property
    def context_text(self) -> str:
        """Text sent to the model: the parent section when expanded."""
        return self.parent_text or self.text


@dataclass
class Citation:
    n: int
    doc_id: str
    doc_name: str
    heading_path: str
    page_start: int
    page_end: int
    chunk_text: str
    char_start: int
    char_end: int
    # Offsets of the matched span *within* chunk_text, for UI highlighting.
    highlight_start: int
    highlight_end: int

    @property
    def page_label(self) -> str:
        if self.page_start == self.page_end:
            return f"p.{self.page_start}"
        return f"pp.{self.page_start}-{self.page_end}"
