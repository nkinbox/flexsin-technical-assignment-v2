"""Answer generation with enforced grounding.

Preventing hallucination is the primary evaluation criterion, so it is enforced
at four independent layers rather than trusting the prompt alone:

  1. Relevance floor  -- nothing retrieved above threshold means we refuse
                         *before* calling the model. A model that is never
                         asked cannot hallucinate, and the refusal is free.
  2. Constrained prompt -- answer only from numbered context, or say you don't
                         know.
  3. Citation validation -- every [n] the model emits must map to a passage we
                         actually supplied; invented markers are stripped.
  4. User verification -- citations are clickable and highlight the source span.

Layers 1 and 3 are code, not prompting, which is what makes them reliable.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from backend.models import Citation, RetrievedChunk
from backend.providers.base import ModelProvider

logger = logging.getLogger(__name__)

REFUSAL_TEXT = (
    "I don't have enough information in the uploaded documents to answer that."
)

_ANSWER_SYSTEM = f"""\
# task: answer
You are a document analysis assistant. You answer strictly from the numbered
context passages supplied with each question.

Rules:
- Use ONLY the numbered context. Never use outside or prior knowledge.
- End every factual claim with the citation marker of the passage that supports
  it, like this [1]. A sentence drawing on two passages ends with [1][2].
- Only cite numbers that appear in the context. Never invent a citation.
- If the context does not contain the answer, reply with exactly:
  {REFUSAL_TEXT}
- Do not apologise, hedge, or explain your reasoning. Answer directly.
- Prefer the wording of the source text over paraphrase for figures, names,
  dates, and quoted terms.
"""

_CITATION_MARKER_RE = re.compile(r"\[(\d+)\]")


@dataclass
class AnswerResult:
    answer: str
    grounded: bool
    confidence: float
    citations: list[Citation] = field(default_factory=list)
    stripped_citations: list[int] = field(default_factory=list)


def build_context(chunks: list[RetrievedChunk]) -> str:
    """Render retrieved passages as a numbered block the model can cite."""
    lines: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        page = (
            f"p.{chunk.page_start}"
            if chunk.page_start == chunk.page_end
            else f"pp.{chunk.page_start}-{chunk.page_end}"
        )
        location = f"{chunk.doc_name}, {page}"
        if chunk.heading_path:
            location += f", {chunk.heading_path}"
        # context_text is the expanded parent section when available.
        lines.append(f"[{index}] ({location}) {chunk.context_text}")
    return "\n\n".join(lines)


def generate_answer(
    question: str, chunks: list[RetrievedChunk], provider: ModelProvider
) -> AnswerResult:
    # --- Layer 1: refuse before spending an LLM call ----------------------
    if not chunks:
        return AnswerResult(
            answer=REFUSAL_TEXT, grounded=False, confidence=0.0, citations=[]
        )

    # --- Layer 2: constrained prompt --------------------------------------
    context = build_context(chunks)
    user = (
        f"<context>\n{context}\n</context>\n\n"
        f"<question>{question}</question>"
    )

    try:
        raw = provider.generate(_ANSWER_SYSTEM, user)
    except Exception as exc:
        logger.error("generation failed: %s", exc)
        raise

    # --- Layer 3: citation validation -------------------------------------
    answer, used, stripped = _validate_citations(raw, valid_count=len(chunks))

    is_refusal = _is_refusal(answer)
    grounded = bool(used) and not is_refusal

    citations = [
        _to_citation(number, chunks[number - 1]) for number in sorted(used)
    ]

    return AnswerResult(
        answer=answer,
        grounded=grounded,
        confidence=_confidence(chunks, used, is_refusal),
        citations=citations,
        stripped_citations=stripped,
    )


def _validate_citations(raw: str, valid_count: int) -> tuple[str, set[int], list[int]]:
    """Strip any citation marker that does not map to a supplied passage.

    A model that invents [7] when six passages were supplied is fabricating a
    source. Removing the marker keeps the prose readable while making the
    fabrication visible in `stripped_citations`.
    """
    used: set[int] = set()
    stripped: list[int] = []

    def replace(match: re.Match[str]) -> str:
        number = int(match.group(1))
        if 1 <= number <= valid_count:
            used.add(number)
            return match.group(0)
        stripped.append(number)
        return ""

    cleaned = _CITATION_MARKER_RE.sub(replace, raw or "")
    # Tidy the whitespace left behind by removed markers.
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([.,;:])", r"\1", cleaned)

    if stripped:
        logger.warning("stripped invented citation markers: %s", stripped)

    return cleaned.strip(), used, stripped


def _is_refusal(answer: str) -> bool:
    normalized = answer.lower().strip().rstrip(".")
    return normalized.startswith(REFUSAL_TEXT.lower().rstrip("."))


def _confidence(
    chunks: list[RetrievedChunk], used: set[int], is_refusal: bool
) -> float:
    """A transparency signal, not a probability.

    Combines how strongly the best passage matched with how much of the
    supplied context the answer actually leaned on.
    """
    if is_refusal or not used:
        return 0.0

    # Absolute similarity, not the relative fusion score -- the latter is ~1.0
    # for the top hit of every query and would make confidence meaningless.
    top_similarity = max((c.similarity for c in chunks), default=0.0)
    coverage = len(used) / max(len(chunks), 1)
    score = 0.75 * min(max(top_similarity, 0.0), 1.0) + 0.25 * coverage
    return round(min(max(score, 0.0), 1.0), 2)


def _to_citation(number: int, chunk: RetrievedChunk) -> Citation:
    """Show the reader the full section, with the matched span highlighted."""
    display_text = chunk.context_text
    base = chunk.parent_char_start if chunk.parent_text else chunk.char_start

    highlight_start = max(0, chunk.char_start - base)
    highlight_end = min(len(display_text), chunk.char_end - base)
    if highlight_end <= highlight_start:
        highlight_start, highlight_end = 0, min(len(display_text), 240)

    return Citation(
        n=number,
        doc_id=chunk.doc_id,
        doc_name=chunk.doc_name,
        heading_path=chunk.heading_path,
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        chunk_text=display_text,
        char_start=chunk.char_start,
        char_end=chunk.char_end,
        highlight_start=highlight_start,
        highlight_end=highlight_end,
    )
