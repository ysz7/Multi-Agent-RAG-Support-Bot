"""Vector store backends, selected by `VECTOR_STORE`.

pgvector is the default and the only backend bundled in Docker Compose or
covered by tests. Qdrant has a real implementation but ships unverified.
"""

from __future__ import annotations

from app.core.config import Settings, get_settings
from app.rag.retrievers.base import (
    RetrievedChunk,
    Retriever,
    RetrieverError,
    RetrieverUnavailable,
)

__all__ = [
    "RetrievedChunk",
    "Retriever",
    "RetrieverError",
    "RetrieverUnavailable",
    "get_retriever",
]


def get_retriever(settings: Settings | None = None) -> Retriever:
    """Build the retriever named by `VECTOR_STORE`.

    Imports are deferred so a default install never needs `qdrant_client`.
    """
    settings = settings or get_settings()

    if settings.vector_store == "pgvector":
        from app.rag.retrievers.pgvector import PgVectorRetriever

        return PgVectorRetriever(settings)

    if settings.vector_store == "qdrant":
        from app.rag.retrievers.qdrant import QdrantRetriever

        return QdrantRetriever(settings)

    # Settings already constrains the value; this guards a future addition.
    raise RetrieverError(f"unknown VECTOR_STORE: {settings.vector_store}")
