"""Chat endpoint: the online half of the RAG pipeline.

    rewrite -> decompose -> retrieve -> expand -> ground -> generate
"""

from __future__ import annotations

import logging
from dataclasses import asdict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.api import deps
from backend.rag.generate import generate_answer
from backend.rag.retrieve import retrieve
from backend.rag.rewrite import rewrite_query

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    session_id: str | None = None
    # Multi-document search across everything is the default; this scopes it.
    doc_ids: list[str] | None = None


@router.post("")
async def chat(request: ChatRequest) -> dict:
    provider = deps.provider()
    store = deps.store()
    sessions = deps.sessions()

    session_id = request.session_id or sessions.new_session_id()
    history = sessions.history(session_id)

    # 1. Resolve pronouns and ellipsis against recent turns. Without this,
    #    "what about the second one?" embeds to nothing useful.
    search_query = rewrite_query(request.question, history, provider)
    if search_query != request.question:
        logger.info("rewrote %r -> %r", request.question, search_query)

    # 2. Decompose, search, retry once, expand to parent sections.
    try:
        chunks = retrieve(search_query, provider, store, request.doc_ids)
    except Exception as exc:
        logger.exception("retrieval failed")
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {exc}") from exc

    # 3. Generate under the grounding guardrail.
    try:
        result = generate_answer(search_query, chunks, provider)
    except Exception as exc:
        logger.exception("generation failed")
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}") from exc

    sessions.append(session_id, "user", request.question)
    sessions.append(session_id, "assistant", result.answer)

    return {
        "session_id": session_id,
        "question": request.question,
        "search_query": search_query,
        "answer": result.answer,
        "grounded": result.grounded,
        "confidence": result.confidence,
        "citations": [asdict(c) for c in result.citations],
        "stripped_citations": result.stripped_citations,
        "retrieved_count": len(chunks),
    }


@router.get("/{session_id}")
async def get_history(session_id: str) -> dict:
    return {"session_id": session_id, "turns": deps.sessions().history(session_id)}


@router.delete("/{session_id}")
async def clear_history(session_id: str) -> dict:
    deps.sessions().clear(session_id)
    return {"session_id": session_id, "cleared": True}
