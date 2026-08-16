"""Typed application settings, loaded from .env."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- Provider selection ---
    provider: Literal["dev", "vertex"] = "dev"

    # --- Vertex AI ---
    gcp_project_id: str | None = None
    gcp_region: str = "us-central1"
    gen_model: str = "gemini-2.5-pro"
    embed_model: str = "gemini-embedding-001"

    # --- Weaviate ---
    weaviate_host: str = "localhost"
    weaviate_http_port: int = 8080
    weaviate_grpc_port: int = 50051

    # --- Chunking ---
    chunk_target_tokens: int = 600
    chunk_overlap_ratio: float = 0.15
    # A page yielding fewer characters than this is treated as scanned and
    # routed to vision instead of the text extractor.
    scanned_page_char_threshold: int = 50

    # --- Retrieval ---
    top_k: int = 5
    overfetch: int = 15
    # Absolute cosine-similarity floor for grounding. Set to None to use the
    # per-provider default below, which differs because embedding models place
    # unrelated text at very different baseline similarities.
    relevance_floor: float | None = None
    max_parents: int = 3
    parent_context_tokens: int = 6000
    hybrid_alpha: float = 0.5

    # --- Storage ---
    upload_dir: Path = Path("./data/uploads")

    @property
    def embed_dim(self) -> int:
        """Vector width. The Weaviate collection is dimension-bound, so changing
        provider requires recreating it -- guarded at startup in weaviate_store."""
        return 3072 if self.provider == "vertex" else 768

    @property
    def collection_name(self) -> str:
        """Namespaced by provider so dev and vertex vectors can never mix.

        Vectors from different embedding models are not comparable, and a
        collection is bound to one dimension. Separate collections make the
        switch safe instead of silently returning nonsense.
        """
        return f"DocumentChunk_{self.provider}"

    @property
    def chunk_overlap_tokens(self) -> int:
        return int(self.chunk_target_tokens * self.chunk_overlap_ratio)

    @property
    def effective_relevance_floor(self) -> float:
        """Grounding threshold on absolute cosine similarity.

        Provider-specific because the baseline differs: sparse feature-hashed
        vectors put unrelated text near 0, while dense semantic embeddings
        routinely score unrelated text around 0.4-0.5, so a floor tuned for one
        would either refuse everything or refuse nothing on the other.
        """
        if self.relevance_floor is not None:
            return self.relevance_floor
        # dev: measured on the sample corpus -- relevant questions score
        # 0.38-0.57, unrelated ones 0.00, so 0.15 separates cleanly.
        # vertex: a starting point only. Dense embedding models place unrelated
        # text far above zero, so this must be re-tuned against real documents
        # once credentials exist (see README "Tuning the grounding threshold").
        return 0.55 if self.provider == "vertex" else 0.15


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    return settings
