"""Grounding is the primary evaluation criterion.

The guarantees under test are the ones enforced in *code* rather than by
prompting -- refusing before an LLM call when nothing was retrieved, and
stripping citation markers the model invented.
"""

from __future__ import annotations

from backend.models import RetrievedChunk
from backend.rag.generate import REFUSAL_TEXT, build_context, generate_answer


class ScriptedProvider:
    """Returns a fixed answer, so the test exercises the guardrail, not a model."""

    name = "scripted"

    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.calls = 0

    def generate(self, system: str, user: str) -> str:
        self.calls += 1
        return self.answer

    def embed_documents(self, texts):  # pragma: no cover - unused
        return [[0.0] for _ in texts]

    def embed_query(self, text):  # pragma: no cover - unused
        return [0.0]

    def extract_from_image(self, image_bytes, mime_type):  # pragma: no cover
        return ""

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)


def _chunk(text: str = "Q3 revenue was 4.2 million dollars.", similarity: float = 0.8):
    return RetrievedChunk(
        text=text,
        doc_id="d1",
        doc_name="report.pdf",
        heading_path="Report > Q3",
        page_start=4,
        page_end=4,
        chunk_index=0,
        parent_id="p1",
        char_start=100,
        char_end=100 + len(text),
        score=1.0,
        similarity=similarity,
    )


def test_no_retrieval_refuses_without_calling_the_model():
    """Layer 1: a model that is never asked cannot hallucinate, and the
    refusal costs nothing."""
    provider = ScriptedProvider("this should never be produced")

    result = generate_answer("Who won the 2019 World Cup?", [], provider)

    assert provider.calls == 0
    assert result.answer == REFUSAL_TEXT
    assert result.grounded is False
    assert result.confidence == 0.0
    assert result.citations == []


def test_invented_citation_markers_are_stripped_and_reported():
    """Layer 3: citing [7] when four passages were supplied is fabrication."""
    provider = ScriptedProvider("Revenue rose [1] and margins improved [7].")

    result = generate_answer("What happened?", [_chunk()], provider)

    assert "[7]" not in result.answer
    assert "[1]" in result.answer
    assert result.stripped_citations == [7]
    # A valid citation remains, so the answer is still grounded.
    assert result.grounded is True


def test_answer_with_no_citation_at_all_is_not_grounded():
    provider = ScriptedProvider("Revenue rose sharply during the quarter.")

    result = generate_answer("What happened?", [_chunk()], provider)

    assert result.grounded is False
    assert result.citations == []


def test_model_refusal_is_reported_as_ungrounded():
    provider = ScriptedProvider(REFUSAL_TEXT)

    result = generate_answer("Unrelated question?", [_chunk()], provider)

    assert result.grounded is False
    assert result.confidence == 0.0


def test_citations_resolve_to_the_supplied_passages():
    provider = ScriptedProvider("Revenue was 4.2 million [1].")

    result = generate_answer("What was revenue?", [_chunk()], provider)

    assert len(result.citations) == 1
    citation = result.citations[0]
    assert citation.n == 1
    assert citation.doc_name == "report.pdf"
    assert citation.page_label == "p.4"


def test_citation_highlight_span_lands_inside_the_displayed_text():
    """The drawer highlights char offsets against the text it shows; an
    out-of-range span would highlight nothing or crash the UI."""
    chunk = _chunk()
    chunk.parent_text = "Preamble sentence. " + chunk.text + " Trailing sentence."
    chunk.parent_char_start = chunk.char_start - len("Preamble sentence. ")

    provider = ScriptedProvider("Revenue was 4.2 million [1].")
    result = generate_answer("What was revenue?", [chunk], provider)

    citation = result.citations[0]
    assert 0 <= citation.highlight_start < citation.highlight_end
    assert citation.highlight_end <= len(citation.chunk_text)
    highlighted = citation.chunk_text[citation.highlight_start : citation.highlight_end]
    assert "4.2 million" in highlighted


def test_confidence_uses_absolute_similarity_not_the_fusion_score():
    """Weaviate's hybrid score is ~1.0 for the top hit of every query, so
    confidence derived from it would be meaningless."""
    weak = _chunk(similarity=0.2)
    strong = _chunk(similarity=0.9)

    provider = ScriptedProvider("Answer [1].")
    low = generate_answer("q", [weak], provider).confidence
    high = generate_answer("q", [strong], provider).confidence

    assert low < high


def test_context_block_is_numbered_and_labelled_for_citation():
    context = build_context([_chunk()])

    assert context.startswith("[1] (report.pdf, p.4, Report > Q3)")


def test_multi_page_chunk_renders_a_page_range():
    chunk = _chunk()
    chunk.page_start, chunk.page_end = 4, 5

    assert "pp.4-5" in build_context([chunk])
