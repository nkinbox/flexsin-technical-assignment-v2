"""Retrieval: hybrid search, parent expansion, and bounded query handling.

Three ideas, in order of how much they matter:

1. **Hybrid search.** Pure vector search misses exact tokens -- invoice
   numbers, surnames, SKUs. Pure keyword search misses paraphrase. Weaviate
   runs both and fuses the rankings.

2. **Parent expansion.** Match on small chunks so ranking is precise, then hand
   the model the enclosing section so it has enough context to answer fully.

3. **Bounded query handling.** Decompose multi-part questions; retry once with
   a broadened query before refusing. Two rounds maximum -- never an
   open-ended agent loop, so latency stays predictable.
"""

from __future__ import annotations

import logging

from backend.config import get_settings
from backend.models import RetrievedChunk
from backend.providers.base import ModelProvider
from backend.rag.rewrite import decompose_query
from backend.store.weaviate_store import WeaviateStore

logger = logging.getLogger(__name__)

_BROADEN_SYSTEM = """\
# task: rewrite
Rewrite the search query to retrieve more broadly. Expand acronyms, drop
jargon, and use plainer synonyms for rare terms. Keep the same meaning.
Output only the rewritten query.
"""


def retrieve(
    question: str,
    provider: ModelProvider,
    store: WeaviateStore,
    doc_ids: list[str] | None = None,
) -> list[RetrievedChunk]:
    """Run the full retrieval path for one question."""
    settings = get_settings()
    # Gate on absolute cosine similarity, never on the hybrid score: fusion
    # scores are relative to the result set, so the best hit is always ~1.0
    # even when nothing in the corpus is remotely relevant.
    floor = settings.effective_relevance_floor

    # --- Round 1: decompose, then search each part ------------------------
    sub_queries = decompose_query(question, provider)
    if len(sub_queries) > 1:
        logger.info("decomposed into %d sub-queries: %s", len(sub_queries), sub_queries)

    hits = _search_all(sub_queries, provider, store, doc_ids, settings)
    grounded = [h for h in hits if h.similarity >= floor]

    # --- Round 2 (only if round 1 found nothing usable) -------------------
    if not grounded:
        broadened = _broaden(question, provider)
        if broadened and broadened.lower() != question.lower():
            logger.info("nothing above floor; retrying with: %s", broadened)
            hits = _search_all([broadened], provider, store, doc_ids, settings)
            grounded = [h for h in hits if h.similarity >= floor]

    if not grounded:
        best = max((h.similarity for h in hits), default=0.0)
        logger.info(
            "no chunk cleared the floor (best similarity %.3f < %.3f) -- will refuse",
            best,
            floor,
        )
        return []

    grounded.sort(key=lambda h: h.score, reverse=True)
    selected = grounded[: settings.top_k]

    return _expand_to_parents(selected, store, settings)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def _search_all(
    queries: list[str],
    provider: ModelProvider,
    store: WeaviateStore,
    doc_ids: list[str] | None,
    settings,
) -> list[RetrievedChunk]:
    """Search every sub-query and merge, keeping the best score per chunk."""
    best: dict[tuple[str, int], RetrievedChunk] = {}

    for query in queries:
        vector = provider.embed_query(query)
        results = store.hybrid_search(
            query=query,
            vector=vector,
            limit=settings.overfetch,
            alpha=settings.hybrid_alpha,
            doc_ids=doc_ids,
        )
        for hit in results:
            key = (hit.doc_id, hit.chunk_index)
            existing = best.get(key)
            if existing is None or hit.score > existing.score:
                best[key] = hit

    return list(best.values())


def _broaden(question: str, provider: ModelProvider) -> str | None:
    try:
        return provider.generate(_BROADEN_SYSTEM, f"<question>{question}</question>").strip()
    except Exception as exc:
        logger.warning("query broadening failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Parent expansion
# ---------------------------------------------------------------------------

def _expand_to_parents(
    selected: list[RetrievedChunk], store: WeaviateStore, settings
) -> list[RetrievedChunk]:
    """Replace each hit's context with its enclosing section.

    Capped at `max_parents` distinct sections and truncated to a fixed
    character budget, so a handful of large sections cannot overflow the
    model's context window.
    """
    expanded: list[RetrievedChunk] = []
    seen_parents: set[str] = set()
    # ~4 characters per token is close enough for a safety budget.
    budget = settings.parent_context_tokens * 4
    used = 0

    for hit in selected:
        if hit.parent_id in seen_parents:
            continue
        if len(seen_parents) >= settings.max_parents:
            break

        siblings = store.fetch_parent(hit.doc_id, hit.parent_id)
        if siblings:
            section_text = "\n\n".join(s.text for s in siblings)
            section_start = min(s.char_start for s in siblings)
            page_start = min(s.page_start for s in siblings)
            page_end = max(s.page_end for s in siblings)
        else:
            section_text = hit.text
            section_start = hit.char_start
            page_start, page_end = hit.page_start, hit.page_end

        remaining = budget - used
        if remaining <= 0:
            break
        if len(section_text) > remaining:
            section_text = section_text[:remaining].rstrip() + " ..."

        used += len(section_text)
        seen_parents.add(hit.parent_id)

        hit.parent_text = section_text
        hit.parent_char_start = section_start
        hit.page_start = page_start
        hit.page_end = page_end
        expanded.append(hit)

    return expanded
