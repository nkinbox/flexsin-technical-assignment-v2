"""Stage 2a -- parse the unified Markdown contract into structural blocks.

Every extractor emits Markdown, so this parser is the single place where text
becomes structure. Downstream code works only with `Block` objects and never
needs to know whether a page came from pypdf, python-docx, or Gemini vision.
"""

from __future__ import annotations

import re

from backend.models import Block, PageMarkdown

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_DIVIDER_RE = re.compile(r"^\s*\|[\s:|-]+\|\s*$")
_FIGURE_RE = re.compile(r"^\s*(Figure:|!\[|>\s*_\[)", re.IGNORECASE)


def normalize_pages(pages: list[PageMarkdown]) -> list[Block]:
    """Flatten pages into one ordered block list, tagged with page numbers."""
    blocks: list[Block] = []
    for page in pages:
        blocks.extend(_parse_page(page))
    return blocks


def _parse_page(page: PageMarkdown) -> list[Block]:
    blocks: list[Block] = []
    lines = page.markdown.splitlines()
    buffer: list[str] = []
    index = 0

    def flush_paragraph() -> None:
        nonlocal buffer
        text = " ".join(part.strip() for part in buffer if part.strip()).strip()
        if text:
            blocks.append(
                Block(
                    kind="figure" if _FIGURE_RE.match(text) else "paragraph",
                    text=text,
                    page_start=page.page_no,
                    page_end=page.page_no,
                )
            )
        buffer = []

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()

        if not stripped:
            flush_paragraph()
            index += 1
            continue

        heading = _HEADING_RE.match(stripped)
        if heading:
            flush_paragraph()
            blocks.append(
                Block(
                    kind="heading",
                    text=heading.group(2).strip(),
                    page_start=page.page_no,
                    page_end=page.page_no,
                    level=len(heading.group(1)),
                )
            )
            index += 1
            continue

        if _TABLE_ROW_RE.match(line):
            flush_paragraph()
            table_lines: list[str] = []
            while index < len(lines) and _TABLE_ROW_RE.match(lines[index]):
                table_lines.append(lines[index].strip())
                index += 1
            blocks.append(
                Block(
                    kind="table",
                    text="\n".join(table_lines),
                    page_start=page.page_no,
                    page_end=page.page_no,
                )
            )
            continue

        buffer.append(line)
        index += 1

    flush_paragraph()
    return blocks


# --- Table helpers, used by the stitcher ----------------------------------

def table_rows(table_text: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in table_text.splitlines():
        if _TABLE_DIVIDER_RE.match(line):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        rows.append(cells)
    return rows


def table_width(table_text: str) -> int:
    rows = table_rows(table_text)
    return max((len(r) for r in rows), default=0)


def table_header(table_text: str) -> list[str]:
    rows = table_rows(table_text)
    return rows[0] if rows else []


def has_divider(table_text: str) -> bool:
    return any(_TABLE_DIVIDER_RE.match(line) for line in table_text.splitlines())


def strip_header(table_text: str) -> str:
    """Drop a repeated header row and its divider when merging a continued table."""
    lines = table_text.splitlines()
    out: list[str] = []
    dropped_header = False
    for line in lines:
        if not dropped_header:
            if _TABLE_DIVIDER_RE.match(line):
                dropped_header = True
                continue
            if not out:
                out.append("__HEADER__")
                continue
        out.append(line)
    return "\n".join(l for l in out if l != "__HEADER__")
