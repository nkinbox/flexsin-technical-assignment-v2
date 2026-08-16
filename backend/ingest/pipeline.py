"""Ingestion orchestration: file in, chunks in the vector store out.

    extract -> strip furniture -> normalize -> stitch -> chunk -> embed -> upsert

Each stage is a separate module so the pipeline reads as the architecture
diagram in the README, and so any stage can be tested in isolation.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

from backend.config import get_settings
from backend.ingest.chunk import chunk_blocks
from backend.ingest.extract import SUPPORTED_EXTENSIONS, UnsupportedFileType, extract
from backend.ingest.normalize import normalize_pages
from backend.ingest.stitch import stitch_blocks, strip_repeating_furniture
from backend.models import DocumentMeta
from backend.providers.base import ModelProvider
from backend.store.weaviate_store import WeaviateStore

logger = logging.getLogger(__name__)


def ingest_file(
    path: Path,
    original_name: str,
    provider: ModelProvider,
    store: WeaviateStore,
) -> DocumentMeta:
    """Run a saved file through the full ingestion pipeline."""
    settings = get_settings()

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileType(
            f"Unsupported file type '{path.suffix}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    doc_id = str(uuid.uuid4())

    # 1. Extract -- per-page routing between exact text and vision.
    pages = extract(path, provider)
    vision_pages = [p.page_no for p in pages if p.source == "vision"]
    logger.info(
        "%s: %d page(s), %d via vision", original_name, len(pages), len(vision_pages)
    )

    # 2. Strip running headers/footers *before* stitching, or the footer gets
    #    welded into the sentence it interrupted.
    pages = strip_repeating_furniture(pages)

    # 3. Parse the unified Markdown contract into structural blocks.
    blocks = normalize_pages(pages)

    # 4. Reassemble paragraphs and tables split by page breaks.
    blocks = stitch_blocks(blocks)

    # 5. Chunk on structural units.
    chunks, _document_text = chunk_blocks(blocks, provider, doc_id)

    if not chunks:
        logger.warning("%s produced no chunks -- nothing to index", original_name)
        return DocumentMeta(
            doc_id=doc_id,
            doc_name=original_name,
            mime_type=path.suffix.lstrip("."),
            size_bytes=path.stat().st_size,
            page_count=len(pages),
            chunk_count=0,
            vision_pages=vision_pages,
        )

    # 6. Embed. The heading path is prefixed inside embedding_text() so the
    #    vector carries section context the bare excerpt lacks.
    vectors = provider.embed_documents([c.embedding_text() for c in chunks])

    # 7. Store.
    store.upsert_chunks(
        doc_id=doc_id, doc_name=original_name, chunks=chunks, vectors=vectors
    )

    meta = DocumentMeta(
        doc_id=doc_id,
        doc_name=original_name,
        mime_type=path.suffix.lstrip("."),
        size_bytes=path.stat().st_size,
        page_count=len(pages),
        chunk_count=len(chunks),
        vision_pages=vision_pages,
    )
    logger.info("%s ingested: %d chunk(s)", original_name, len(chunks))
    return meta


def save_upload(content: bytes, filename: str) -> Path:
    """Persist an upload under a collision-proof name, preserving the extension."""
    settings = get_settings()
    safe_suffix = Path(filename).suffix.lower()
    target = settings.upload_dir / f"{uuid.uuid4()}{safe_suffix}"
    target.write_bytes(content)
    return target
