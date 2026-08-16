"""Weaviate vector store.

Configured with self-provided vectors: embeddings come from Vertex AI, so
Weaviate must never try to vectorize anything itself. BM25 stays enabled
because hybrid search needs the lexical half.

The collection name is namespaced by provider (`DocumentChunk_dev` vs
`DocumentChunk_vertex`). Vectors from different embedding models are not
comparable and a collection is bound to one dimension, so separate collections
make switching providers safe instead of silently returning nonsense.
"""

from __future__ import annotations

import logging
from typing import Any

import weaviate
from weaviate.classes.config import Configure, DataType, Property, Tokenization
from weaviate.classes.query import Filter, MetadataQuery

from backend.config import get_settings
from backend.models import Chunk, RetrievedChunk

logger = logging.getLogger(__name__)


def _extract_vector(raw: Any) -> list[float]:
    """Normalise the shape Weaviate returns for `include_vector=True`.

    Self-provided vectors come back under the 'default' named-vector key; older
    shapes return a bare list.
    """
    if isinstance(raw, dict):
        for key in ("default", "vector"):
            if key in raw:
                return list(raw[key])
        return list(next(iter(raw.values()), []))
    return list(raw or [])


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot / (norm_a * norm_b)))


class WeaviateStore:
    """Thin wrapper owning client lifecycle and the collection schema."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._client: weaviate.WeaviateClient | None = None

    # --- Lifecycle --------------------------------------------------------
    def connect(self) -> None:
        if self._client is not None:
            return
        self._client = weaviate.connect_to_local(
            host=self._settings.weaviate_host,
            port=self._settings.weaviate_http_port,
            grpc_port=self._settings.weaviate_grpc_port,
        )
        self.ensure_schema()
        logger.info(
            "connected to Weaviate; collection=%s", self._settings.collection_name
        )

    def close(self) -> None:
        """The v4 client holds a gRPC channel that must be closed explicitly,
        otherwise the process hangs on shutdown."""
        if self._client is not None:
            self._client.close()
            self._client = None

    def is_ready(self) -> bool:
        try:
            return bool(self._client and self._client.is_ready())
        except Exception:
            return False

    @property
    def _collection(self):
        if self._client is None:
            raise RuntimeError("WeaviateStore.connect() was not called")
        return self._client.collections.get(self._settings.collection_name)

    # --- Schema -----------------------------------------------------------
    def ensure_schema(self) -> None:
        assert self._client is not None
        name = self._settings.collection_name

        if self._client.collections.exists(name):
            return

        self._client.collections.create(
            name=name,
            description="Document chunks with externally supplied embeddings.",
            # We supply vectors from Vertex AI; Weaviate must not vectorize.
            vector_config=Configure.Vectors.self_provided(),
            properties=[
                Property(name="text", data_type=DataType.TEXT),
                Property(
                    name="doc_id",
                    data_type=DataType.TEXT,
                    tokenization=Tokenization.FIELD,
                ),
                Property(name="doc_name", data_type=DataType.TEXT),
                Property(
                    name="parent_id",
                    data_type=DataType.TEXT,
                    tokenization=Tokenization.FIELD,
                ),
                Property(name="heading_path", data_type=DataType.TEXT),
                Property(name="page_start", data_type=DataType.INT),
                Property(name="page_end", data_type=DataType.INT),
                Property(name="chunk_index", data_type=DataType.INT),
                Property(name="char_start", data_type=DataType.INT),
                Property(name="char_end", data_type=DataType.INT),
            ],
        )
        logger.info("created collection %s", name)

    def reset(self) -> None:
        """Drop and recreate the collection. Used by tests."""
        assert self._client is not None
        name = self._settings.collection_name
        if self._client.collections.exists(name):
            self._client.collections.delete(name)
        self.ensure_schema()

    # --- Writes -----------------------------------------------------------
    def upsert_chunks(
        self,
        doc_id: str,
        doc_name: str,
        chunks: list[Chunk],
        vectors: list[list[float]],
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(
                f"chunk/vector count mismatch: {len(chunks)} vs {len(vectors)}"
            )
        if not chunks:
            return

        collection = self._collection
        with collection.batch.dynamic() as batch:
            for chunk, vector in zip(chunks, vectors):
                batch.add_object(
                    properties={
                        "text": chunk.text,
                        "doc_id": doc_id,
                        "doc_name": doc_name,
                        "parent_id": chunk.parent_id,
                        "heading_path": chunk.heading_path,
                        "page_start": chunk.page_start,
                        "page_end": chunk.page_end,
                        "chunk_index": chunk.chunk_index,
                        "char_start": chunk.char_start,
                        "char_end": chunk.char_end,
                    },
                    vector=vector,
                )

        failures = collection.batch.failed_objects
        if failures:
            first = failures[0].message
            raise RuntimeError(
                f"{len(failures)} of {len(chunks)} chunks failed to index: {first}"
            )

    def delete_document(self, doc_id: str) -> int:
        result = self._collection.data.delete_many(
            where=Filter.by_property("doc_id").equal(doc_id)
        )
        return int(getattr(result, "successful", 0))

    # --- Reads ------------------------------------------------------------
    def hybrid_search(
        self,
        query: str,
        vector: list[float],
        limit: int,
        alpha: float,
        doc_ids: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        """Vector similarity fused with BM25.

        Pure vector search misses exact identifiers and figures; pure keyword
        search misses paraphrase. `alpha` balances the two (0 = keyword only,
        1 = vector only).
        """
        response = self._collection.query.hybrid(
            query=query,
            vector=vector,
            alpha=alpha,
            limit=limit,
            filters=self._doc_filter(doc_ids),
            return_metadata=MetadataQuery(score=True),
            # Vectors come back so we can compute an *absolute* similarity.
            # Weaviate's fused score is relative to the result set (the best
            # hit is ~1.0 regardless of quality), so it cannot gate grounding.
            include_vector=True,
        )

        results: list[RetrievedChunk] = []
        for obj in response.objects:
            chunk = self._to_retrieved(obj.properties, float(obj.metadata.score or 0.0))
            chunk.similarity = _cosine(vector, _extract_vector(obj.vector))
            results.append(chunk)
        return results

    def fetch_parent(self, doc_id: str, parent_id: str) -> list[RetrievedChunk]:
        """All chunks of one section, in document order -- the unit handed to
        the model after parent expansion."""
        response = self._collection.query.fetch_objects(
            filters=Filter.by_property("doc_id").equal(doc_id)
            & Filter.by_property("parent_id").equal(parent_id),
            limit=200,
        )
        chunks = [self._to_retrieved(o.properties, 0.0) for o in response.objects]
        chunks.sort(key=lambda c: c.chunk_index)
        return chunks

    def document_summaries(self) -> dict[str, dict[str, Any]]:
        """Chunk counts per document, read back from the index itself."""
        summaries: dict[str, dict[str, Any]] = {}
        for obj in self._collection.iterator(
            return_properties=["doc_id", "doc_name"]
        ):
            doc_id = str(obj.properties.get("doc_id", ""))
            if not doc_id:
                continue
            entry = summaries.setdefault(
                doc_id,
                {"doc_id": doc_id, "doc_name": obj.properties.get("doc_name", ""), "chunk_count": 0},
            )
            entry["chunk_count"] += 1
        return summaries

    # --- Helpers ----------------------------------------------------------
    @staticmethod
    def _doc_filter(doc_ids: list[str] | None):
        """Scope a search to selected documents. Multi-document search across
        the whole collection is the default when this is None."""
        if not doc_ids:
            return None
        if len(doc_ids) == 1:
            return Filter.by_property("doc_id").equal(doc_ids[0])
        return Filter.any_of(
            [Filter.by_property("doc_id").equal(d) for d in doc_ids]
        )

    @staticmethod
    def _to_retrieved(properties: dict[str, Any], score: float) -> RetrievedChunk:
        return RetrievedChunk(
            text=str(properties.get("text", "")),
            doc_id=str(properties.get("doc_id", "")),
            doc_name=str(properties.get("doc_name", "")),
            heading_path=str(properties.get("heading_path", "")),
            page_start=int(properties.get("page_start", 1) or 1),
            page_end=int(properties.get("page_end", 1) or 1),
            chunk_index=int(properties.get("chunk_index", 0) or 0),
            parent_id=str(properties.get("parent_id", "")),
            char_start=int(properties.get("char_start", 0) or 0),
            char_end=int(properties.get("char_end", 0) or 0),
            score=score,
        )
