"""Shared singletons.

The provider, store, registry, and session store are created once at startup
and injected everywhere, so the Weaviate gRPC connection and the Vertex client
are not rebuilt per request.
"""

from __future__ import annotations

from functools import lru_cache

from backend.providers.base import ModelProvider, get_provider
from backend.rag.session import SessionStore
from backend.store.registry import DocumentRegistry
from backend.store.weaviate_store import WeaviateStore


@lru_cache
def provider() -> ModelProvider:
    return get_provider()


@lru_cache
def store() -> WeaviateStore:
    return WeaviateStore()


@lru_cache
def registry() -> DocumentRegistry:
    return DocumentRegistry()


@lru_cache
def sessions() -> SessionStore:
    return SessionStore()
