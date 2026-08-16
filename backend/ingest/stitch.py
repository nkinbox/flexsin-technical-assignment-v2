"""Stage 2b -- repair the damage page boundaries do to meaning.

A page break is an artifact of paper size. Chunking page-by-page produces
fragments that begin mid-sentence and have the running header, footer, and page
number welded into the middle of a clause -- neither half retrieves well, and a
half-table without its header row produces confidently wrong numbers.

Order matters: strip furniture *first*, then stitch. Reversed, the footer gets
glued into the sentence it interrupted.
"""

from __future__ import annotations

import math
import re

from backend.models import Block, PageMarkdown
from backend.ingest.normalize import (
    has_divider,
    strip_header,
    table_header,
    table_width,
)

# A line that is only a page number, in the shapes documents actually use.
_PAGE_NUMBER_RE = re.compile(
    r"^\s*(?:page\s+)?\d+\s*(?:/|of|\|)?\s*\d*\s*$", re.IGNORECASE
)
# How many lines at each edge of a page count as header/footer territory.
_EDGE_LINES = 3
# A line must recur on at least this many pages to be considered furniture.
_MIN_REPEATS = 3

_CONTINUES_STRUCTURE = re.compile(r"^\s*(#{1,6}\s|[-*+]\s|\d+[.)]\s|\|)")


def strip_repeating_furniture(pages: list[PageMarkdown]) -> list[PageMarkdown]:
    """Remove running headers, footers, and page numbers.

    Furniture is identified by *recurrence at the page edges*, not by content,
    so it works for any language or template without a hardcoded list.
    """
    if len(pages) < 2:
        return [
            PageMarkdown(p.page_no, _drop_page_numbers(p.markdown), p.source)
            for p in pages
        ]

    counts: dict[str, int] = {}
    for page in pages:
        for line in _edge_lines(page.markdown):
            key = _normalize_line(line)
            if key:
                counts[key] = counts.get(key, 0) + 1

    # Scale with document length, but never demand more repeats than there are
    # pages -- a 2-page document has a running header appearing exactly twice.
    threshold = min(len(pages), max(2, math.ceil(0.5 * len(pages))))
    furniture = {key for key, count in counts.items() if count >= threshold}

    cleaned: list[PageMarkdown] = []
    for page in pages:
        lines = page.markdown.splitlines()
        edge_indices = _edge_indices(len(lines))
        kept = [
            line
            for i, line in enumerate(lines)
            if not (i in edge_indices and _normalize_line(line) in furniture)
        ]
        cleaned.append(
            PageMarkdown(page.page_no, _drop_page_numbers("\n".join(kept)), page.source)
        )
    return cleaned


def _edge_indices(total: int) -> set[int]:
    head = set(range(min(_EDGE_LINES, total)))
    tail = set(range(max(0, total - _EDGE_LINES), total))
    return head | tail


def _edge_lines(markdown: str) -> list[str]:
    lines = markdown.splitlines()
    return [lines[i] for i in sorted(_edge_indices(len(lines)))]


def _normalize_line(line: str) -> str:
    """Collapse whitespace and digits so 'Page 3 of 20' and 'Page 4 of 20'
    compare equal -- the page number varies, the furniture does not."""
    collapsed = re.sub(r"\s+", " ", line.strip())
    return re.sub(r"\d+", "#", collapsed).lower()


def _drop_page_numbers(markdown: str) -> str:
    return "\n".join(
        line for line in markdown.splitlines() if not _PAGE_NUMBER_RE.match(line)
    )


def stitch_blocks(blocks: list[Block]) -> list[Block]:
    """Merge blocks that a page break split in two.

    Returns a new list; merged blocks carry a page *range*, which is what lets
    a citation render as "report.pdf, pp. 4-5".
    """
    if not blocks:
        return []

    merged: list[Block] = [blocks[0]]

    for block in blocks[1:]:
        previous = merged[-1]
        crosses_page = block.page_start > previous.page_end

        if crosses_page and _should_merge_paragraphs(previous, block):
            merged[-1] = _merge_paragraphs(previous, block)
            continue

        if crosses_page and _should_merge_tables(previous, block):
            merged[-1] = _merge_tables(previous, block)
            continue

        merged.append(block)

    return merged


def _should_merge_paragraphs(previous: Block, nxt: Block) -> bool:
    if previous.kind != "paragraph" or nxt.kind != "paragraph":
        return False
    if not previous.is_open_ended:
        return False
    # The continuation must not itself start a new structure.
    if _CONTINUES_STRUCTURE.match(nxt.text):
        return False
    first_char = nxt.text.lstrip()[:1]
    # Lowercase start, or a hyphenated word break, means mid-sentence.
    return bool(first_char) and (first_char.islower() or previous.text.rstrip().endswith("-"))


def _merge_paragraphs(previous: Block, nxt: Block) -> Block:
    left = previous.text.rstrip()
    right = nxt.text.lstrip()
    if left.endswith("-"):
        text = left[:-1] + right  # de-hyphenate across the break
    else:
        text = f"{left} {right}"
    return Block(
        kind="paragraph",
        text=text,
        page_start=previous.page_start,
        page_end=nxt.page_end,
        level=previous.level,
    )


def _should_merge_tables(previous: Block, nxt: Block) -> bool:
    """A table continued on the next page repeats its header (or omits it), and
    keeps the same column count. Merging restores the header row that the
    continuation fragment would otherwise lack."""
    if previous.kind != "table" or nxt.kind != "table":
        return False
    if table_width(previous.text) != table_width(nxt.text):
        return False

    previous_header = table_header(previous.text)
    next_header = table_header(nxt.text)
    return (not has_divider(nxt.text)) or previous_header == next_header


def _merge_tables(previous: Block, nxt: Block) -> Block:
    continuation = strip_header(nxt.text) if has_divider(nxt.text) else nxt.text
    return Block(
        kind="table",
        text=f"{previous.text}\n{continuation}".strip(),
        page_start=previous.page_start,
        page_end=nxt.page_end,
    )
