"""Stage 1 -- extraction with per-page routing.

Design principle: normalise the *output*, not the *input*. Every path below
emits the same structured Markdown, so nothing downstream knows or cares which
one produced a given page.

Vision is applied per page, driven by evidence, never as a blanket rule:

    DOCX / TXT              -> native parser  (structure is explicit)
    PDF page, clean prose   -> pypdf text     (exact characters, free)
    PDF page, <50 chars     -> vision         (scanned; no text exists)
    PDF page, table/columns -> vision         (layout carries meaning)
    Image file              -> vision         (only option)

Rendering *every* page to an image was considered and rejected: it discards the
exact characters a digital PDF already contains and re-derives them
probabilistically, multiplies cost and latency, makes extraction
non-deterministic, and would require LibreOffice to render DOCX.
"""

from __future__ import annotations

import io
import logging
import re
from pathlib import Path

from backend.config import get_settings
from backend.models import PageMarkdown
from backend.providers.base import ModelProvider

logger = logging.getLogger(__name__)

TEXT_EXTENSIONS = {".txt", ".md", ".markdown"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff"}
DOCX_EXTENSIONS = {".docx"}
PDF_EXTENSIONS = {".pdf"}
SUPPORTED_EXTENSIONS = (
    TEXT_EXTENSIONS | IMAGE_EXTENSIONS | DOCX_EXTENSIONS | PDF_EXTENSIONS
)

_MIME_BY_EXTENSION = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".tiff": "image/tiff",
}


class UnsupportedFileType(ValueError):
    pass


def extract(path: Path, provider: ModelProvider) -> list[PageMarkdown]:
    """Route a file to the appropriate extractor(s)."""
    suffix = path.suffix.lower()

    if suffix in TEXT_EXTENSIONS:
        return _extract_text_file(path)
    if suffix in DOCX_EXTENSIONS:
        return _extract_docx(path)
    if suffix in PDF_EXTENSIONS:
        return _extract_pdf(path, provider)
    if suffix in IMAGE_EXTENSIONS:
        return _extract_image(path, provider)

    raise UnsupportedFileType(
        f"Unsupported file type '{suffix}'. "
        f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
    )


# ---------------------------------------------------------------------------
# Plain text
# ---------------------------------------------------------------------------

def _extract_text_file(path: Path) -> list[PageMarkdown]:
    raw = _read_text_with_fallback(path)
    return [PageMarkdown(page_no=1, markdown=_promote_text_headings(raw), source="native")]


def _read_text_with_fallback(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    # latin-1 cannot fail, but be explicit rather than relying on that.
    return path.read_bytes().decode("utf-8", errors="replace")


_HEADING_HINT = re.compile(r"^[A-Z0-9][^.!?]{0,79}$")


def _promote_text_headings(raw: str) -> str:
    """Give plain text a light structural spine.

    A short line with no terminal punctuation, followed by a blank line, is
    almost always a heading. Marking it as one lets the structural chunker keep
    sections intact instead of splitting on arbitrary character counts.
    """
    lines = raw.splitlines()
    out: list[str] = []

    for i, line in enumerate(lines):
        stripped = line.strip()
        next_blank = i + 1 < len(lines) and not lines[i + 1].strip()
        prev_blank = i == 0 or not lines[i - 1].strip()

        looks_like_heading = (
            stripped
            and prev_blank
            and next_blank
            and len(stripped) <= 80
            and _HEADING_HINT.match(stripped)
            and not stripped.startswith(("#", "-", "*", ">", "|"))
        )
        out.append(f"## {stripped}" if looks_like_heading else line)

    return "\n".join(out)


# ---------------------------------------------------------------------------
# DOCX -- structure is explicit, so no rendering and no LibreOffice dependency
# ---------------------------------------------------------------------------

def _extract_docx(path: Path) -> list[PageMarkdown]:
    from docx import Document
    from docx.oxml.ns import qn
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    document = Document(str(path))
    pages: list[PageMarkdown] = []
    current: list[str] = []
    page_no = 1

    def flush() -> None:
        nonlocal current
        if current:
            pages.append(
                PageMarkdown(
                    page_no=page_no,
                    markdown="\n\n".join(current).strip(),
                    source="native",
                )
            )
        current = []

    body = document.element.body
    for child in body.iterchildren():
        if child.tag == qn("w:p"):
            paragraph = Paragraph(child, document)

            # An explicit page break starts a new logical page, which keeps
            # citation page numbers meaningful for paginated Word documents.
            if _docx_has_page_break(child, qn):
                flush()
                page_no += 1

            rendered = _docx_paragraph_to_markdown(paragraph)
            if rendered:
                current.append(rendered)

        elif child.tag == qn("w:tbl"):
            table = Table(child, document)
            rendered = _docx_table_to_markdown(table)
            if rendered:
                current.append(rendered)

    flush()
    return pages or [PageMarkdown(page_no=1, markdown="", source="native")]


def _docx_has_page_break(paragraph_element, qn) -> bool:
    for br in paragraph_element.iter(qn("w:br")):
        if br.get(qn("w:type")) == "page":
            return True
    return False


def _docx_paragraph_to_markdown(paragraph) -> str:
    text = paragraph.text.strip()
    if not text:
        return ""

    style = (paragraph.style.name or "").lower()

    if style.startswith("heading"):
        digits = "".join(ch for ch in style if ch.isdigit())
        level = min(int(digits), 6) if digits else 1
        return f"{'#' * level} {text}"
    if style in {"title", "subtitle"}:
        return f"# {text}"
    if style.startswith("list"):
        return f"- {text}"
    return text


def _docx_table_to_markdown(table) -> str:
    """Tables are where answers hide and where naive extractors fail -- a
    flattened table produces confidently wrong numbers."""
    rows: list[list[str]] = []
    for row in table.rows:
        cells = [c.text.strip().replace("\n", " ").replace("|", "\\|") for c in row.cells]
        if any(cells):
            rows.append(cells)

    if not rows:
        return ""

    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]

    header, *body = rows
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# PDF -- per-page routing between exact text and vision
# ---------------------------------------------------------------------------

def _extract_pdf(path: Path, provider: ModelProvider) -> list[PageMarkdown]:
    from pypdf import PdfReader

    settings = get_settings()
    reader = PdfReader(str(path))
    pages: list[PageMarkdown] = []

    for index, page in enumerate(reader.pages):
        page_no = index + 1
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            logger.warning("pypdf failed on page %d: %s", page_no, exc)
            text = ""

        needs_vision, reason = _page_needs_vision(page, text, settings)

        if needs_vision:
            logger.info("page %d -> vision (%s)", page_no, reason)
            markdown = _vision_for_pdf_page(path, index, provider)
            pages.append(PageMarkdown(page_no, markdown, "vision"))
        else:
            pages.append(PageMarkdown(page_no, _plain_text_to_markdown(text), "text"))

    return pages


def _page_needs_vision(page, text: str, settings) -> tuple[bool, str]:
    """Decide the route for one PDF page.

    Two independent triggers: too little text to be anything but a scan, or a
    layout whose meaning lives in its geometry (tables, multiple columns) and
    would be destroyed by flattening to a text stream.
    """
    if len(text.strip()) < settings.scanned_page_char_threshold:
        return True, "scanned or near-empty"

    try:
        fragments = _collect_text_positions(page)
    except Exception as exc:
        logger.debug("layout analysis unavailable: %s", exc)
        return False, ""

    if _looks_tabular(fragments):
        return True, "table-like layout"
    if _looks_multi_column(fragments):
        return True, "multi-column layout"
    return False, ""


def _collect_text_positions(page) -> list[tuple[float, float]]:
    """Collect (x, y) of every text-drawing operation on the page."""
    positions: list[tuple[float, float]] = []

    def visitor(text, cm, tm, font_dict, font_size):  # noqa: ANN001 - pypdf callback
        if text and text.strip():
            positions.append((float(tm[4]), float(tm[5])))

    page.extract_text(visitor_text=visitor)
    return positions


def _group_rows(fragments: list[tuple[float, float]], tolerance: float = 3.0):
    """Bucket fragments into visual rows by y-coordinate."""
    rows: dict[int, list[float]] = {}
    for x, y in fragments:
        key = int(y / tolerance)
        rows.setdefault(key, []).append(x)
    return rows


def _looks_tabular(fragments: list[tuple[float, float]], min_rows: int = 4) -> bool:
    """A table shows up as many rows sharing 3+ well-separated x positions."""
    if len(fragments) < 20:
        return False

    rows = _group_rows(fragments)
    multi_column_rows = 0
    for xs in rows.values():
        clusters = _cluster_positions(sorted(xs), gap=25.0)
        if len(clusters) >= 3:
            multi_column_rows += 1

    return multi_column_rows >= min_rows and multi_column_rows >= 0.3 * len(rows)


def _looks_multi_column(fragments: list[tuple[float, float]]) -> bool:
    """Two dense vertical bands separated by a wide gutter = two columns."""
    if len(fragments) < 40:
        return False

    xs = sorted(x for x, _ in fragments)
    clusters = _cluster_positions(xs, gap=60.0)
    dense = [c for c in clusters if len(c) >= 0.2 * len(xs)]
    return len(dense) >= 2


def _cluster_positions(sorted_values: list[float], gap: float) -> list[list[float]]:
    if not sorted_values:
        return []
    clusters = [[sorted_values[0]]]
    for value in sorted_values[1:]:
        if value - clusters[-1][-1] > gap:
            clusters.append([value])
        else:
            clusters[-1].append(value)
    return clusters


def _vision_for_pdf_page(path: Path, page_index: int, provider: ModelProvider) -> str:
    """Render one PDF page to PNG and transcribe it."""
    try:
        import pypdfium2 as pdfium
    except ImportError:
        logger.error("pypdfium2 not installed; cannot render page for vision")
        return "> _[page could not be rendered: pypdfium2 is not installed]_"

    try:
        pdf = pdfium.PdfDocument(str(path))
        try:
            # scale=2 is roughly 144 DPI -- enough detail for small print
            # without inflating the image token count unnecessarily.
            bitmap = pdf[page_index].render(scale=2)
            image = bitmap.to_pil()
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            payload = buffer.getvalue()
        finally:
            pdf.close()
    except Exception as exc:
        logger.error("failed to render page %d: %s", page_index + 1, exc)
        return f"> _[page {page_index + 1} could not be rendered: {exc}]_"

    return provider.extract_from_image(payload, "image/png")


def _plain_text_to_markdown(text: str) -> str:
    """Normalise PDF text while *preserving line structure*.

    Line structure has to survive this stage: header/footer detection works by
    finding lines that recur at the top and bottom of many pages, so collapsing
    a page into one paragraph here would destroy the only signal it has.

    Wrapped lines are rejoined into paragraphs later, by `normalize._parse_page`,
    which runs after furniture has been stripped.
    """
    lines = [ln.rstrip() for ln in text.splitlines()]
    joined = "\n".join(lines)
    # De-hyphenate words broken across a line wrap: "sub-\nsidiary".
    return re.sub(r"(\w)-\n(\w)", r"\1\2", joined)


# ---------------------------------------------------------------------------
# Standalone images
# ---------------------------------------------------------------------------

def _extract_image(path: Path, provider: ModelProvider) -> list[PageMarkdown]:
    mime = _MIME_BY_EXTENSION.get(path.suffix.lower(), "image/png")
    markdown = provider.extract_from_image(path.read_bytes(), mime)
    return [PageMarkdown(page_no=1, markdown=markdown, source="vision")]
