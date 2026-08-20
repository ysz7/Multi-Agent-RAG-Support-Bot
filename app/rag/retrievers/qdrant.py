"""Qdrant adapter — WRITTEN BUT UNVERIFIED.

No Qdrant service is bundled (see `docker-compose.qdrant.yml`) and nothing here
is covered by an integration test. pgvector is the default and the only backend
exercised in CI. This module exists so `VECTOR_STORE=qdrant` has a real
implementation to switch to, and so the multi-tenant roadmap item has a
starting point — treat it as a sketch to validate, not as working code.

To try it:

    pip install -e ".[qdrant]"
    docker compose -f docker-compose.yml -f docker-compose.qdrant.yml up -d
    # set VECTOR_STORE=qdrant in .env

`qdrant_client` is imported lazily so a default install never needs it.
"""

from __future__ import annotations

from typing import Any

from app.core.config import Settings, get_settings
from app.rag.retrievers.base import RetrievedChunk, RetrieverError, RetrieverUnavailable

_INSTALL_HINT = (
    'VECTOR_STORE=qdrant requires the optional extra: pip install -e ".[qdrant]". '
    "Note the Qdrant adapter ships unverified — pgvector is the supported default."
)


def _import_client() -> Any:
    try:
        from qdrant_client import AsyncQdrantClient
    except ImportError as exc:  # pragma: no cover - depends on install shape
        raise RetrieverUnavailable(_INSTALL_HINT) from exc
    return AsyncQdrantClient


class QdrantRetriever:
    """Read path against a Qdrant collection.

    Payload is expected to mirror the pgvector rows: `tenant_id`, `content`,
    `chunk_index`, `document_id`, `title`, `source_path`, `metadata`.
    """

    name = "qdrant"

    def __init__(self, settings: Settings | None = None, client: Any | None = None) -> None:
        self._settings = settings or get_settings()
        self._collection = self._settings.qdrant_collection
        if client is not None:
            self._client = client
        else:
            client_cls = _import_client()
            self._client = client_cls(url=self._settings.qdrant_url)

    async def similarity_search(
        self,
        query_embedding: list[float],
        *,
        tenant_id: str,
        top_k: int | None = None,
        filters: dict | None = None,
    ) -> list[RetrievedChunk]:
        if len(query_embedding) != self._settings.embedding_dim:
            raise RetrieverError(
                f"query embedding has {len(query_embedding)} dims, "
                f"expected {self._settings.embedding_dim}"
            )

        try:
            from qdrant_client import models
        except ImportError as exc:  # pragma: no cover
            raise RetrieverUnavailable(_INSTALL_HINT) from exc

        # Tenant is a hard server-side condition, exactly as in pgvector.
        conditions = [
            models.FieldCondition(key="tenant_id", match=models.MatchValue(value=tenant_id))
        ]
        for key, value in (filters or {}).items():
            conditions.append(
                models.FieldCondition(key=f"metadata.{key}", match=models.MatchValue(value=value))
            )

        try:
            response = await self._client.query_points(
                collection_name=self._collection,
                query=query_embedding,
                query_filter=models.Filter(must=conditions),
                limit=top_k or self._settings.retrieval_top_k,
                with_payload=True,
            )
        except Exception as exc:
            raise RetrieverError(f"qdrant search failed: {exc}") from exc

        results = []
        for point in getattr(response, "points", response):
            payload = point.payload or {}
            results.append(
                RetrievedChunk(
                    content=payload.get("content", ""),
                    score=float(point.score),
                    chunk_index=int(payload.get("chunk_index", 0)),
                    document_id=str(payload.get("document_id", "")),
                    title=payload.get("title", ""),
                    source_path=payload.get("source_path", ""),
                    metadata=payload.get("metadata") or {},
                )
            )
        return results

    async def aclose(self) -> None:
        close = getattr(self._client, "close", None)
        if close is not None:
            result = close()
            if hasattr(result, "__await__"):
                await result
