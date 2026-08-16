"""FastAPI application entry point."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.api import chat, deps, documents
from backend.config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info(
        "starting with provider=%s collection=%s",
        settings.provider,
        settings.collection_name,
    )
    if settings.provider == "dev":
        logger.warning(
            "PROVIDER=dev -- stub embeddings and template answers. "
            "The pipeline is real; answer quality is not. Set PROVIDER=vertex "
            "once Google Cloud is configured."
        )

    try:
        deps.store().connect()
    except Exception as exc:
        # Fail loudly rather than serving an app whose every query 500s.
        logger.error(
            "could not connect to Weaviate at %s:%s -- is `docker compose up -d` running? (%s)",
            settings.weaviate_host,
            settings.weaviate_http_port,
            exc,
        )
        raise

    yield

    deps.store().close()
    logger.info("shutdown complete")


app = FastAPI(
    title="Document-Aware AI Chatbot (RAG)",
    description=(
        "Answers questions using only uploaded documents, with citations back "
        "to the source passage."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # POC only; scope to known origins in production.
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router)
app.include_router(chat.router)


@app.get("/health", tags=["meta"])
async def health() -> dict:
    settings = get_settings()
    return {
        "status": "ok",
        "provider": settings.provider,
        "collection": settings.collection_name,
        "embed_dim": settings.embed_dim,
        "weaviate_ready": deps.store().is_ready(),
    }


if FRONTEND_DIR.exists():
    app.mount(
        "/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static"
    )

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(str(FRONTEND_DIR / "index.html"))
