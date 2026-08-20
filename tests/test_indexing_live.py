"""Phase 4: indexing against a real Postgres.

Skipped when the database is unreachable, so the default suite stays offline.
"""

from pathlib import Path

import pytest

from app.core.config import Settings
from app.rag.chunking import chunk_document, load_document
from app.rag.retrievers.pgvector import IndexOutcome, PgVectorStore

pytestmark = pytest.mark.live

ROOT = Path(__file__).resolve().parent.parent
TENANT = "test-tenant"


@pytest.fixture
async def store_and_conn(env):
    env(ANTHROPIC_API_KEY="sk-test")
    settings = Settings(_env_file=ROOT / ".env")
    store = PgVectorStore(settings)
    try:
        connection = await store.connect()
    except Exception as exc:
        pytest.skip(f"postgres unreachable: {exc}")

    async with connection.cursor() as cursor:
        await cursor.execute("DELETE FROM documents WHERE tenant_id = %s", (TENANT,))
    await connection.commit()

    yield store, connection, settings

    async with connection.cursor() as cursor:
        await cursor.execute("DELETE FROM documents WHERE tenant_id = %s", (TENANT,))
    await connection.commit()
    await connection.close()


def _doc(tmp_path, text: str, name: str = "doc.md"):
    path = tmp_path / name
    path.write_text(text)
    document = load_document(path)
    chunks = chunk_document(document, size=400, overlap=50)
    return document, chunks


def _fake_embeddings(chunks, dim: int):
    return [[0.01 * (i + 1)] * dim for i, _ in enumerate(chunks)]


async def test_insert_then_reindex_is_unchanged(store_and_conn, tmp_path):
    store, conn, settings = store_and_conn
    document, chunks = _doc(tmp_path, "# Title\n\nSome body text.")
    vectors = _fake_embeddings(chunks, settings.embedding_dim)

    first = await store.upsert_document(
        conn, tenant_id=TENANT, document=document, chunks=chunks, embeddings=vectors
    )
    await conn.commit()
    assert first.outcome is IndexOutcome.INSERTED
    assert first.chunk_count == len(chunks)

    second = await store.upsert_document(
        conn, tenant_id=TENANT, document=document, chunks=chunks, embeddings=vectors
    )
    await conn.commit()
    assert second.outcome is IndexOutcome.UNCHANGED
    assert second.chunk_count == 0


async def test_edited_file_replaces_and_leaves_no_orphans(store_and_conn, tmp_path):
    store, conn, settings = store_and_conn
    document, chunks = _doc(tmp_path, "# Title\n\nOriginal text.")
    await store.upsert_document(
        conn,
        tenant_id=TENANT,
        document=document,
        chunks=chunks,
        embeddings=_fake_embeddings(chunks, settings.embedding_dim),
    )
    await conn.commit()

    edited, edited_chunks = _doc(tmp_path, "# Title\n\nOriginal text.\n\n## New\n\nMore text.")
    result = await store.upsert_document(
        conn,
        tenant_id=TENANT,
        document=edited,
        chunks=edited_chunks,
        embeddings=_fake_embeddings(edited_chunks, settings.embedding_dim),
    )
    await conn.commit()
    assert result.outcome is IndexOutcome.REPLACED

    stats = await store.stats(conn, tenant_id=TENANT)
    assert stats["documents"] == 1
    assert stats["chunks"] == len(edited_chunks)

    async with conn.cursor() as cursor:
        await cursor.execute(
            """
            SELECT count(*) AS orphans FROM chunks c
            LEFT JOIN documents d ON d.id = c.document_id
            WHERE d.id IS NULL
            """
        )
        assert (await cursor.fetchone())["orphans"] == 0


async def test_prune_removes_documents_with_no_source_file(store_and_conn, tmp_path):
    store, conn, settings = store_and_conn
    document, chunks = _doc(tmp_path, "# Gone\n\nBody.", name="gone.md")
    await store.upsert_document(
        conn,
        tenant_id=TENANT,
        document=document,
        chunks=chunks,
        embeddings=_fake_embeddings(chunks, settings.embedding_dim),
    )
    await conn.commit()

    removed = await store.delete_missing(conn, tenant_id=TENANT, keep_paths=set())
    await conn.commit()

    assert removed == 1
    assert (await store.stats(conn, tenant_id=TENANT))["documents"] == 0


async def test_mismatched_embedding_count_is_rejected(store_and_conn, tmp_path):
    store, conn, _ = store_and_conn
    document, chunks = _doc(tmp_path, "# T\n\nBody.")
    with pytest.raises(ValueError, match="embeddings"):
        await store.upsert_document(
            conn, tenant_id=TENANT, document=document, chunks=chunks, embeddings=[]
        )
