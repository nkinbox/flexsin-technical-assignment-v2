"""Document registry -- metadata Weaviate doesn't need for retrieval.

Weaviate stays the single source of truth for *chunks*. This holds the
document-level facts the UI needs (page count, which pages used vision, upload
time, size) which would otherwise be duplicated onto every chunk row.

A JSON file is deliberate for a POC: transparent, inspectable, and zero
infrastructure. Production would put this in Postgres alongside user ownership.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict
from pathlib import Path

from backend.config import get_settings
from backend.models import DocumentMeta

logger = logging.getLogger(__name__)


class DocumentRegistry:
    def __init__(self, path: Path | None = None) -> None:
        settings = get_settings()
        self._path = path or (settings.upload_dir.parent / "documents.json")
        self._lock = threading.Lock()
        self._documents: dict[str, DocumentMeta] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            self._documents = {
                doc_id: DocumentMeta(**payload) for doc_id, payload in raw.items()
            }
        except Exception as exc:
            logger.warning("could not read document registry (%s); starting empty", exc)
            self._documents = {}

    def _flush(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {doc_id: asdict(meta) for doc_id, meta in self._documents.items()}
        self._path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def add(self, meta: DocumentMeta) -> None:
        with self._lock:
            self._documents[meta.doc_id] = meta
            self._flush()

    def remove(self, doc_id: str) -> DocumentMeta | None:
        with self._lock:
            meta = self._documents.pop(doc_id, None)
            if meta:
                self._flush()
            return meta

    def get(self, doc_id: str) -> DocumentMeta | None:
        return self._documents.get(doc_id)

    def list(self) -> list[DocumentMeta]:
        return sorted(
            self._documents.values(), key=lambda m: m.uploaded_at, reverse=True
        )
