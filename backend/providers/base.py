"""The single interface every model backend implements.

Nothing outside `providers/` imports Vertex or Google libraries. Swapping
`PROVIDER=dev` for `PROVIDER=vertex` changes the implementation behind this
Protocol and nothing else -- which is what lets the whole pipeline run and be
tested before any cloud credentials exist.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ModelProvider(Protocol):
    """Generation, vision, and embeddings behind one seam."""

    name: str

    # --- Embeddings -------------------------------------------------------
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed chunks for storage.

        Uses the asymmetric 'document' task type where the backend supports it:
        document and query embeddings live in the same space but are optimised
        for their respective roles, which measurably improves retrieval.
        """
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query (asymmetric counterpart of embed_documents)."""
        ...

    # --- Generation -------------------------------------------------------
    def generate(self, system: str, user: str) -> str:
        """Single-turn completion. History is folded into `user` by the caller."""
        ...

    # --- Vision -----------------------------------------------------------
    def extract_from_image(self, image_bytes: bytes, mime_type: str) -> str:
        """Transcribe a page image or standalone image to structured Markdown."""
        ...

    # --- Tokenisation -----------------------------------------------------
    def count_tokens(self, text: str) -> int:
        """Token count used by the chunker to size sections."""
        ...


def get_provider() -> ModelProvider:
    """Construct the provider named in settings. Imports are deferred so that
    dev mode never requires google-genai to be installed or importable."""
    from backend.config import get_settings

    settings = get_settings()

    if settings.provider == "vertex":
        from backend.providers.vertex import VertexProvider

        return VertexProvider()

    from backend.providers.dev import DevProvider

    return DevProvider()
