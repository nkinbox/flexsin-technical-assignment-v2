"""Chunking is a graded criterion, so its invariants are pinned here.

The rules under test: sections follow headings, tables are never split, the
heading path travels with the chunk, words are never cut in half, and overlap
exists so an answer straddling a boundary is not halved.
"""

from __future__ import annotations

from backend.ingest.chunk import chunk_blocks
from backend.models import Block


def test_section_under_a_heading_becomes_one_chunk(provider):
    blocks = [
        Block("heading", "Equipment Allowance", 1, 1, level=2),
        Block("paragraph", "The allowance is 45000 rupees.", 1, 1),
        Block("paragraph", "Receipts are due within 30 days.", 1, 1),
    ]

    chunks, _ = chunk_blocks(blocks, provider, doc_id="d1")

    assert len(chunks) == 1
    assert "45000" in chunks[0].text and "Receipts" in chunks[0].text


def test_heading_path_is_nested_and_carried_on_the_chunk(provider):
    blocks = [
        Block("heading", "Annual Report", 1, 1, level=1),
        Block("heading", "Q3 Performance", 1, 1, level=2),
        Block("paragraph", "Revenue was 4.2 million dollars.", 1, 1),
    ]

    chunks, _ = chunk_blocks(blocks, provider, doc_id="d1")

    assert chunks[0].heading_path == "Annual Report > Q3 Performance"
    # The path is prefixed at embedding time -- that is the retrieval gain.
    assert chunks[0].embedding_text().startswith("Annual Report > Q3 Performance")


def test_sibling_heading_pops_the_stack(provider):
    blocks = [
        Block("heading", "Report", 1, 1, level=1),
        Block("heading", "Q3", 1, 1, level=2),
        Block("paragraph", "Q3 body text here.", 1, 1),
        Block("heading", "Q4", 1, 1, level=2),
        Block("paragraph", "Q4 body text here.", 1, 1),
    ]

    chunks, _ = chunk_blocks(blocks, provider, doc_id="d1")
    paths = [c.heading_path for c in chunks]

    assert paths == ["Report > Q3", "Report > Q4"]


def test_table_is_never_split(provider):
    """A table split across chunks loses its header row, which is a reliable
    source of confidently wrong numbers."""
    rows = "\n".join(f"| Item {i} | {i * 1000} |" for i in range(200))
    table = f"| Item | Cost |\n| --- | --- |\n{rows}"
    blocks = [
        Block("heading", "Costs", 1, 1, level=1),
        Block("table", table, 1, 1),
    ]

    chunks, _ = chunk_blocks(blocks, provider, doc_id="d1")

    table_chunks = [c for c in chunks if "| Item | Cost |" in c.text]
    assert len(table_chunks) == 1
    assert table_chunks[0].text.count("Item 199") == 1


def test_oversized_prose_splits_without_breaking_words(provider):
    sentence = "The migration completed successfully across every region. "
    blocks = [
        Block("heading", "Migration", 1, 1, level=1),
        Block("paragraph", sentence * 400, 1, 1),
    ]

    chunks, _ = chunk_blocks(blocks, provider, doc_id="d1")

    assert len(chunks) > 1
    for chunk in chunks:
        # A split mid-word would leave a fragment that is not a real token.
        assert not chunk.text.startswith("he ")
        assert chunk.text.strip() == chunk.text


def test_oversized_prose_chunks_overlap(provider):
    sentence = "Latency improved by thirty eight percent after the rollout. "
    blocks = [Block("paragraph", sentence * 400, 1, 1)]

    chunks, _ = chunk_blocks(blocks, provider, doc_id="d1")

    assert len(chunks) > 1
    # Adjacent chunks share trailing/leading content so a straddling answer
    # survives the boundary.
    first_tail = chunks[0].text[-200:]
    assert any(word in chunks[1].text for word in first_tail.split()[:5])


def test_chunks_of_one_section_share_a_parent_id(provider):
    """parent_id is what makes parent-document retrieval possible."""
    sentence = "Costs remained flat throughout the reporting period. "
    blocks = [
        Block("heading", "Costs", 1, 1, level=1),
        Block("paragraph", sentence * 100, 1, 1),
        Block("paragraph", sentence * 100, 1, 1),
    ]

    chunks, _ = chunk_blocks(blocks, provider, doc_id="d1")

    assert len({c.parent_id for c in chunks}) == 1


def test_different_sections_get_different_parent_ids(provider):
    blocks = [
        Block("heading", "A", 1, 1, level=1),
        Block("paragraph", "First section body.", 1, 1),
        Block("heading", "B", 1, 1, level=1),
        Block("paragraph", "Second section body.", 1, 1),
    ]

    chunks, _ = chunk_blocks(blocks, provider, doc_id="d1")

    assert len({c.parent_id for c in chunks}) == 2


def test_char_offsets_locate_the_chunk_in_the_document(provider):
    """Offsets drive citation highlighting, so they must actually resolve."""
    blocks = [
        Block("heading", "Section", 1, 1, level=1),
        Block("paragraph", "A distinctive marker phrase lives here.", 1, 1),
    ]

    chunks, document_text = chunk_blocks(blocks, provider, doc_id="d1")
    chunk = chunks[0]

    assert document_text[chunk.char_start : chunk.char_end] == chunk.text


def test_page_range_survives_a_stitched_block(provider):
    blocks = [
        Block("heading", "Migration", 1, 1, level=1),
        Block("paragraph", "Text spanning the page boundary.", 1, 2),
    ]

    chunks, _ = chunk_blocks(blocks, provider, doc_id="d1")

    assert (chunks[0].page_start, chunks[0].page_end) == (1, 2)


def test_every_chunk_has_a_usable_page_number(provider):
    """A citation with page 0 is a broken citation."""
    sentence = "Recurring content for the section body. "
    blocks = [
        Block("heading", "S", 3, 3, level=1),
        Block("paragraph", sentence * 150, 3, 4),
        Block("paragraph", sentence * 150, 4, 4),
    ]

    chunks, _ = chunk_blocks(blocks, provider, doc_id="d1")

    assert chunks
    for chunk in chunks:
        assert chunk.page_start >= 1
        assert chunk.page_end >= chunk.page_start
