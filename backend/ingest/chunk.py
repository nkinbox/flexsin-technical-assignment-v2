"""Stage 3 -- structural chunking.

Chunking defines the unit of retrieval, which makes it the highest-leverage
decision in a RAG system. Too small and a chunk loses the context needed to
answer; too large and its embedding becomes a blurry average that matches
nothing precisely.

This chunker works on *structural units* rather than character counts:

  * a section under a heading is one chunk
  * a table is atomic -- never split, because a table without its header row
    yields confidently wrong numbers
  * the heading path is carried on every chunk and prefixed at embedding time,
    restoring the context a bare excerpt loses
  * sections that exceed the budget split recursively: sub-heading, then
    paragraph, then sentence -- structure first, arbitrary cuts last

`parent_id` groups every chunk cut from the same section, which is what makes
parent-document retrieval possible at query time: match a small chunk, answer
from the whole section.
"""

from __future__ import annotations

import re
import uuid

from backend.config import get_settings
from backend.models import Block, Chunk
from backend.providers.base import ModelProvider

_BLOCK_SEPARATOR = "\n\n"
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# Namespace so parent ids are stable across re-ingests of the same document.
_PARENT_NAMESPACE = uuid.UUID("6f1f5f5e-9a2e-4f3a-8a1f-2c9a7c3b4d5e")


def chunk_blocks(
    blocks: list[Block], provider: ModelProvider, doc_id: str
) -> tuple[list[Chunk], str]:
    """Chunk an assembled document.

    Returns the chunks and the assembled document text they index into, so the
    UI can highlight a citation span against the same string.
    """
    settings = get_settings()
    doc_text, spans = _assemble(blocks)

    chunks: list[Chunk] = []
    for section_index, section in enumerate(_iter_sections(blocks, spans)):
        parent_id = str(
            uuid.uuid5(_PARENT_NAMESPACE, f"{doc_id}:{section_index}")
        )
        chunks.extend(
            _chunk_section(
                section=section,
                parent_id=parent_id,
                provider=provider,
                target_tokens=settings.chunk_target_tokens,
                overlap_tokens=settings.chunk_overlap_tokens,
                start_index=len(chunks),
            )
        )

    return chunks, doc_text


# ---------------------------------------------------------------------------
# Assembly -- one document-level string, with a span per block
# ---------------------------------------------------------------------------

def _assemble(blocks: list[Block]) -> tuple[str, list[tuple[int, int]]]:
    parts: list[str] = []
    spans: list[tuple[int, int]] = []
    offset = 0

    for block in blocks:
        spans.append((offset, offset + len(block.text)))
        parts.append(block.text)
        offset += len(block.text) + len(_BLOCK_SEPARATOR)

    return _BLOCK_SEPARATOR.join(parts), spans


# ---------------------------------------------------------------------------
# Sectioning -- group blocks under their heading path
# ---------------------------------------------------------------------------

class _Section:
    __slots__ = ("heading_path", "items")

    def __init__(self, heading_path: str) -> None:
        self.heading_path = heading_path
        self.items: list[tuple[Block, tuple[int, int]]] = []


def _iter_sections(blocks: list[Block], spans: list[tuple[int, int]]):
    """Yield sections, each carrying its full heading path (`A > B > C`)."""
    stack: list[tuple[int, str]] = []
    current = _Section("")

    for block, span in zip(blocks, spans):
        if block.kind == "heading":
            if current.items:
                yield current
            while stack and stack[-1][0] >= block.level:
                stack.pop()
            stack.append((block.level, block.text))
            current = _Section(" > ".join(text for _, text in stack))
            continue

        current.items.append((block, span))

    if current.items:
        yield current


# ---------------------------------------------------------------------------
# Packing
# ---------------------------------------------------------------------------

def _chunk_section(
    section: _Section,
    parent_id: str,
    provider: ModelProvider,
    target_tokens: int,
    overlap_tokens: int,
    start_index: int,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    buffer: list[tuple[str, int, int]] = []  # (text, char_start, char_end)
    buffer_tokens = 0

    def flush(carry_overlap: bool) -> None:
        nonlocal buffer, buffer_tokens
        if not buffer:
            return

        chunks.append(
            Chunk(
                text=_BLOCK_SEPARATOR.join(text for text, _, _ in buffer).strip(),
                heading_path=section.heading_path,
                page_start=0,  # filled in by the caller loop below
                page_end=0,
                chunk_index=start_index + len(chunks),
                parent_id=parent_id,
                char_start=buffer[0][1],
                char_end=buffer[-1][2],
            )
        )

        if carry_overlap and overlap_tokens > 0:
            carried: list[tuple[str, int, int]] = []
            carried_tokens = 0
            for item in reversed(buffer):
                item_tokens = provider.count_tokens(item[0])
                if carried_tokens + item_tokens > overlap_tokens:
                    break
                carried.insert(0, item)
                carried_tokens += item_tokens
            buffer = carried
            buffer_tokens = carried_tokens
        else:
            buffer = []
            buffer_tokens = 0

    # Track page ranges alongside the buffer so chunks inherit them.
    page_ranges: list[tuple[int, int]] = []

    for block, (span_start, span_end) in section.items:
        block_tokens = provider.count_tokens(block.text)

        # Tables are atomic. They never merge into an oversized buffer and are
        # never split, even when larger than the target on their own.
        if block.kind == "table":
            if buffer and buffer_tokens + block_tokens > target_tokens:
                flush(carry_overlap=False)
                page_ranges = []
            buffer.append((block.text, span_start, span_end))
            page_ranges.append((block.page_start, block.page_end))
            buffer_tokens += block_tokens
            continue

        if block_tokens > target_tokens:
            # Oversized prose with no internal structure left to exploit.
            if buffer:
                flush(carry_overlap=False)
                page_ranges = []
            for piece_text, piece_start, piece_end in _split_oversized(
                block.text, span_start, provider, target_tokens, overlap_tokens
            ):
                chunks.append(
                    Chunk(
                        text=piece_text,
                        heading_path=section.heading_path,
                        page_start=block.page_start,
                        page_end=block.page_end,
                        chunk_index=start_index + len(chunks),
                        parent_id=parent_id,
                        char_start=piece_start,
                        char_end=piece_end,
                    )
                )
            continue

        if buffer and buffer_tokens + block_tokens > target_tokens:
            flushed_before = len(chunks)
            flush(carry_overlap=True)
            if len(chunks) > flushed_before and page_ranges:
                chunks[-1].page_start = min(p[0] for p in page_ranges)
                chunks[-1].page_end = max(p[1] for p in page_ranges)
            page_ranges = []

        buffer.append((block.text, span_start, span_end))
        page_ranges.append((block.page_start, block.page_end))
        buffer_tokens += block_tokens

    if buffer:
        flushed_before = len(chunks)
        flush(carry_overlap=False)
        if len(chunks) > flushed_before and page_ranges:
            chunks[-1].page_start = min(p[0] for p in page_ranges)
            chunks[-1].page_end = max(p[1] for p in page_ranges)

    # Any chunk that never received a page range (pure overlap carry) inherits
    # from its neighbour so citations always have a page to point at.
    for i, chunk in enumerate(chunks):
        if chunk.page_start == 0:
            neighbour = chunks[i - 1] if i else (chunks[i + 1] if i + 1 < len(chunks) else None)
            if neighbour:
                chunk.page_start = neighbour.page_start
                chunk.page_end = neighbour.page_end
            else:
                chunk.page_start = chunk.page_end = 1

    return chunks


def _split_oversized(
    text: str,
    base_offset: int,
    provider: ModelProvider,
    target_tokens: int,
    overlap_tokens: int,
) -> list[tuple[str, int, int]]:
    """Last-resort split of a single oversized block, at sentence boundaries.

    Sentences are packed greedily to the token target with overlap carried
    forward, and a word is never split. Embedding-similarity ("semantic")
    splitting would choose the cut point by topic shift instead; it is
    documented as a future refinement and confined to exactly this path, so it
    would never cost a per-sentence embedding across the whole corpus.
    """
    sentences = _SENTENCE_SPLIT_RE.split(text)
    pieces: list[tuple[str, int, int]] = []

    buffer: list[str] = []
    buffer_tokens = 0
    cursor = 0          # offset within `text` of the buffer's first character
    search_from = 0

    def offset_of(sentence: str, start_at: int) -> int:
        found = text.find(sentence, start_at)
        return found if found >= 0 else start_at

    for sentence in sentences:
        if not sentence.strip():
            continue

        sentence_start = offset_of(sentence, search_from)
        search_from = sentence_start + len(sentence)
        sentence_tokens = provider.count_tokens(sentence)

        if buffer and buffer_tokens + sentence_tokens > target_tokens:
            joined = " ".join(buffer)
            pieces.append(
                (joined, base_offset + cursor, base_offset + cursor + len(joined))
            )

            carried: list[str] = []
            carried_tokens = 0
            for previous in reversed(buffer):
                previous_tokens = provider.count_tokens(previous)
                if carried_tokens + previous_tokens > overlap_tokens:
                    break
                carried.insert(0, previous)
                carried_tokens += previous_tokens

            buffer = carried
            buffer_tokens = carried_tokens
            cursor = offset_of(carried[0], 0) if carried else sentence_start

        if not buffer:
            cursor = sentence_start

        buffer.append(sentence)
        buffer_tokens += sentence_tokens

    if buffer:
        joined = " ".join(buffer)
        pieces.append((joined, base_offset + cursor, base_offset + cursor + len(joined)))

    return pieces
