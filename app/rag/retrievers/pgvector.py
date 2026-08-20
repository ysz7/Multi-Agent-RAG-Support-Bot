"""pgvector-backed store: the default (and only bundled) vector backend.

This module owns both halves of the pgvector collection: the write path
(`upsert_document`, used by the indexer) and the read path
(`PgVectorRetriever.similarity_search`, used by the RAG chain).

Idempotency: a document's identity is `(tenant_id, content_hash)`. Re-indexing
unchanged bytes is a no-op. An edited file keeps its `source_path` but changes
hash, so the old row (and its chunks, via ON DELETE CASCADE) is replaced.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from app.core.config import Settings, get_settings
from app.rag.chunking import Chunk, LoadedDocument
from app.rag.retrievers.base import RetrievedChunk, RetrieverError


class IndexOutcome(StrEnum):
    INSERTED = "inserted"
    REPLACED = "replaced"
    UNCHANGED = "unchanged"


@dataclass(frozen=True, slots=True)
class IndexResult:
    outcome: IndexOutcome
    document_id: str | None
    chunk_count: int


class PgVectorStore:
    """Thin async wrapper over psycopg. Callers own the connection lifecycle."""

    name = "pgvector"

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    async def connect(self) -> AsyncConnection:
        from pgvector.psycopg import register_vector_async

        connection = await AsyncConnection.connect(
            self._settings.database_url, row_factory=dict_row, autocommit=False
        )
        # Teaches psycopg to adapt Python lists to the `vector` type.
        await register_vector_async(connection)
        return connection

    async def upsert_document(
        self,
        connection: AsyncConnection,
        *,
        tenant_id: str,
        document: LoadedDocument,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> IndexResult:
        if len(chunks) != len(embeddings):
            raise ValueError(f"{len(chunks)} chunks but {len(embeddings)} embeddings")

        async with connection.cursor() as cursor:
            # Same bytes, same tenant -> nothing to do.
            await cursor.execute(
                "SELECT id FROM documents WHERE tenant_id = %s AND content_hash = %s",
                (tenant_id, document.content_hash),
            )
            if existing := await cursor.fetchone():
                return IndexResult(IndexOutcome.UNCHANGED, str(existing["id"]), 0)

            # Same path but different bytes -> the file was edited; replace it.
            await cursor.execute(
                "DELETE FROM documents WHERE tenant_id = %s AND source_path = %s RETURNING id",
                (tenant_id, document.source_path),
            )
            replaced = await cursor.fetchone() is not None

            await cursor.execute(
                """
                INSERT INTO documents (tenant_id, source_path, title, content_hash, metadata)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    tenant_id,
                    document.source_path,
                    document.title,
                    document.content_hash,
                    Jsonb(document.metadata),
                ),
            )
            row = await cursor.fetchone()
            document_id = row["id"]

            if chunks:
                await cursor.executemany(
                    """
                    INSERT INTO chunks
                        (document_id, tenant_id, chunk_index, content, metadata, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    [
                        (
                            document_id,
                            tenant_id,
                            chunk.chunk_index,
                            chunk.content,
                            Jsonb(chunk.metadata),
                            embedding,
                        )
                        for chunk, embedding in zip(chunks, embeddings, strict=True)
                    ],
                )

        outcome = IndexOutcome.REPLACED if replaced else IndexOutcome.INSERTED
        return IndexResult(outcome, str(document_id), len(chunks))

    async def delete_missing(
        self, connection: AsyncConnection, *, tenant_id: str, keep_paths: set[str]
    ) -> int:
        """Drop indexed documents whose source file no longer exists."""
        async with connection.cursor() as cursor:
            await cursor.execute(
                "SELECT id, source_path FROM documents WHERE tenant_id = %s", (tenant_id,)
            )
            stale = [r["id"] for r in await cursor.fetchall() if r["source_path"] not in keep_paths]
            if stale:
                await cursor.execute("DELETE FROM documents WHERE id = ANY(%s)", (stale,))
        return len(stale)

    async def stats(self, connection: AsyncConnection, *, tenant_id: str) -> dict[str, int]:
        async with connection.cursor() as cursor:
            await cursor.execute(
                """
                SELECT
                    (SELECT count(*) FROM documents WHERE tenant_id = %(t)s) AS documents,
                    (SELECT count(*) FROM chunks    WHERE tenant_id = %(t)s) AS chunks
                """,
                {"t": tenant_id},
            )
            return dict(await cursor.fetchone())


class PgVectorRetriever:
    """Read path over the same tables the indexer writes.

    Holds a connection pool because it is used per-request by the API, unlike
    `PgVectorStore` whose caller (a one-shot script) owns its connection.
    """

    name = "pgvector"

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._pool = None

    async def _get_pool(self):
        if self._pool is None:
            from pgvector.psycopg import register_vector_async
            from psycopg_pool import AsyncConnectionPool

            async def configure(connection: AsyncConnection) -> None:
                await register_vector_async(connection)

            self._pool = AsyncConnectionPool(
                self._settings.database_url,
                min_size=1,
                max_size=self._settings.db_pool_max_size,
                kwargs={"row_factory": dict_row},
                configure=configure,
                open=False,
            )
            await self._pool.open(wait=True)
        return self._pool

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

        limit = top_k or self._settings.retrieval_top_k
        # tenant_id is always a bound parameter and always applied, regardless
        # of what the caller passes in `filters`.
        params: dict = {"tenant": tenant_id, "embedding": query_embedding, "limit": limit}
        filter_sql = ""
        if filters:
            filter_sql = " AND c.metadata @> %(filters)s"
            params["filters"] = Jsonb(filters)

        sql = f"""
            SELECT
                c.id, c.content, c.chunk_index, c.metadata,
                c.document_id, d.title, d.source_path,
                1 - (c.embedding <=> %(embedding)s::vector) AS score
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE c.tenant_id = %(tenant)s
              AND c.embedding IS NOT NULL
              {filter_sql}
            ORDER BY c.embedding <=> %(embedding)s::vector
            LIMIT %(limit)s
        """

        pool = await self._get_pool()
        try:
            async with pool.connection() as connection, connection.cursor() as cursor:
                await cursor.execute(sql, params)
                rows = await cursor.fetchall()
        except Exception as exc:
            raise RetrieverError(f"pgvector search failed: {exc}") from exc

        return [
            RetrievedChunk(
                content=row["content"],
                score=float(row["score"]),
                chunk_index=row["chunk_index"],
                document_id=str(row["document_id"]),
                title=row["title"] or "",
                source_path=row["source_path"],
                metadata=row["metadata"] or {},
            )
            for row in rows
        ]

    async def aclose(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
