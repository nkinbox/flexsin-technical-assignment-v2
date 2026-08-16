"""Page boundaries are an artifact of layout, not meaning.

These tests cover the failure mode that motivated the whole stitch stage: a
paragraph split across a page break, with a running header and page number
wedged into the middle of the sentence.
"""

from __future__ import annotations

from backend.ingest.normalize import normalize_pages
from backend.ingest.stitch import stitch_blocks, strip_repeating_furniture
from backend.models import Block, PageMarkdown


def _pages(*bodies: str) -> list[PageMarkdown]:
    return [
        PageMarkdown(page_no=i, markdown=body, source="text")
        for i, body in enumerate(bodies, start=1)
    ]


def test_running_header_and_footer_are_stripped():
    pages = _pages(
        "Annual Report 2024\nACME Holdings\n\nRevenue grew steadily.\n\nPage 1 of 3",
        "Annual Report 2024\nACME Holdings\n\nCosts remained flat.\n\nPage 2 of 3",
        "Annual Report 2024\nACME Holdings\n\nOutlook is positive.\n\nPage 3 of 3",
    )

    cleaned = strip_repeating_furniture(pages)
    combined = "\n".join(p.markdown for p in cleaned)

    assert "ACME Holdings" not in combined
    assert "Annual Report 2024" not in combined
    assert "Page 1 of 3" not in combined
    # Real content must survive.
    assert "Revenue grew steadily." in combined
    assert "Outlook is positive." in combined


def test_two_page_document_still_strips_its_header():
    """A running header appears exactly twice in a 2-page document; requiring
    three repeats would never strip it."""
    pages = _pages(
        "Confidential Draft\n\nFirst body line.",
        "Confidential Draft\n\nSecond body line.",
    )

    combined = "\n".join(p.markdown for p in strip_repeating_furniture(pages))

    assert "Confidential Draft" not in combined
    assert "First body line." in combined


def test_paragraph_split_across_pages_is_reassembled():
    blocks = [
        Block("paragraph", "The rollout required a maintenance window because the", 1, 1),
        Block("paragraph", "connection pool had to be rebuilt for the new topology.", 2, 2),
    ]

    merged = stitch_blocks(blocks)

    assert len(merged) == 1
    assert "because the connection pool had to be rebuilt" in merged[0].text
    # Provenance survives the merge, so the citation reads "pp.1-2".
    assert (merged[0].page_start, merged[0].page_end) == (1, 2)


def test_hyphenated_word_split_across_pages_is_repaired():
    blocks = [
        Block("paragraph", "The change affected three sub-", 1, 1),
        Block("paragraph", "sidiaries in the group.", 2, 2),
    ]

    merged = stitch_blocks(blocks)

    assert len(merged) == 1
    assert "subsidiaries" in merged[0].text


def test_complete_sentence_is_not_merged_into_the_next_page():
    blocks = [
        Block("paragraph", "The quarter closed on schedule.", 1, 1),
        Block("paragraph", "A new section begins here.", 2, 2),
    ]

    assert len(stitch_blocks(blocks)) == 2


def test_new_structure_on_the_next_page_is_not_merged():
    """An open-ended line followed by a bullet is a list, not a continuation."""
    blocks = [
        Block("paragraph", "The following items were reviewed", 1, 1),
        Block("paragraph", "- the migration plan", 2, 2),
    ]

    assert len(stitch_blocks(blocks)) == 2


def test_table_continued_on_next_page_is_merged_and_keeps_one_header():
    first = "| Item | Cost |\n| --- | --- |\n| Chair | 8500 |"
    second = "| Item | Cost |\n| --- | --- |\n| Desk | 12000 |"
    blocks = [Block("table", first, 1, 1), Block("table", second, 2, 2)]

    merged = stitch_blocks(blocks)

    assert len(merged) == 1
    assert merged[0].text.count("| Item | Cost |") == 1
    assert "Chair" in merged[0].text and "Desk" in merged[0].text
    assert (merged[0].page_start, merged[0].page_end) == (1, 2)


def test_unrelated_tables_are_not_merged():
    first = "| Item | Cost |\n| --- | --- |\n| Chair | 8500 |"
    second = "| Region | Revenue | Margin |\n| --- | --- | --- |\n| APAC | 2.6M | 41% |"

    assert len(stitch_blocks([Block("table", first, 1, 1), Block("table", second, 2, 2)])) == 2


def test_full_page_pipeline_strips_then_stitches():
    """Order matters: stripping must happen first, or the footer is welded into
    the middle of the sentence it interrupted."""
    pages = _pages(
        "Infrastructure Review\n\nLatency improved because the connection\n\nPage 1 of 2",
        "Infrastructure Review\n\npooling configuration was rebuilt.\n\nPage 2 of 2",
    )

    blocks = stitch_blocks(normalize_pages(strip_repeating_furniture(pages)))
    text = " ".join(b.text for b in blocks)

    assert "because the connection pooling configuration was rebuilt" in text
    assert "Infrastructure Review" not in text
    assert "Page 1 of 2" not in text
