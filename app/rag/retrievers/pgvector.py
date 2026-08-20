"""pgvector-backed store: the default (and only bundled) vector backend.

This module owns both halves of the pgvector collection. The write path
(`upsert_document`) lands in Phase 4 so indexing can run end to end; the read
path (`similarity_search`) arrives in Phase 5.

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
