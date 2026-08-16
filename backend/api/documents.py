"""Document upload, listing, and deletion."""

from __future__ import annotations

import logging
from dataclasses import asdict

from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.api import deps
from backend.ingest.extract import SUPPORTED_EXTENSIONS, UnsupportedFileType
from backend.ingest.pipeline import ingest_file, save_upload

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])

MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB


@router.post("")
async def upload_document(file: UploadFile = File(...)) -> dict:
    """Ingest one document through the full pipeline."""
    content = await file.read()

    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.",
        )

    filename = file.filename or "upload"
    saved_path = save_upload(content, filename)

    try:
        meta = ingest_file(
            path=saved_path,
            original_name=filename,
            provider=deps.provider(),
            store=deps.store(),
        )
    except UnsupportedFileType as exc:
        saved_path.unlink(missing_ok=True)
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    except Exception as exc:
        saved_path.unlink(missing_ok=True)
        logger.exception("ingestion failed for %s", filename)
        raise HTTPException(
            status_code=500, detail=f"Ingestion failed: {exc}"
        ) from exc

    if meta.chunk_count == 0:
        raise HTTPException(
            status_code=422,
            detail=(
                "No text could be extracted from this document. If it is a "
                "scanned file or an image, set PROVIDER=vertex to enable "
                "vision-based extraction."
            ),
        )

    deps.registry().add(meta)
    return asdict(meta)


@router.get("")
async def list_documents() -> dict:
    return {
        "documents": [asdict(meta) for meta in deps.registry().list()],
        "supported_types": sorted(SUPPORTED_EXTENSIONS),
    }


@router.delete("/{doc_id}")
async def delete_document(doc_id: str) -> dict:
    meta = deps.registry().remove(doc_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="Document not found.")

    deleted = deps.store().delete_document(doc_id)
    return {"doc_id": doc_id, "deleted_chunks": deleted, "doc_name": meta.doc_name}
