"""Phase 5: pgvector search against a real Postgres.

Uses hand-built orthogonal vectors rather than a live embedding model, so
ranking assertions are exact and Ollama is not required.
"""

from pathlib import Path

import pytest

from app.core.config import Settings
from app.rag.chunking import Chunk, LoadedDocument
from app.rag.retrievers.base import RetrieverError
from app.rag.retrievers.pgvector import PgVectorRetriever, PgVectorStore

pytestmark = pytest.mark.live

ROOT = Path(__file__).resolve().parent.parent
TENANT_A = "tenant-a"
TENANT_B = "tenant-b"


def _unit_vector(dim: int, axis: int) -> list[float]:
    vector = [0.0] * dim
    vector[axis] = 1.0
    return vector


def _document(name: str, digest: str) -> LoadedDocument:
    return LoadedDocument(
        source_path=f"/corpus/{name}",
        title=name.rsplit(".", 1)[0],
        text="",
        content_hash=digest,
        metadata={"type": "markdown"},
    )


@pytest.fixture
async def seeded(env):
    env(ANTHROPIC_API_KEY="sk-test")
    settings = Settings(_env_file=ROOT / ".env")
    store = PgVectorStore(settings)
    try:
        connection = await store.connect()
    except Exception as exc:
        pytest.skip(f"postgres unreachable: {exc}")

    async def purge() -> None:
        async with connection.cursor() as cursor:
            await cursor.execute(
                "DELETE FROM documents WHERE tenant_id = ANY(%s)", ([TENANT_A, TENANT_B],)
            )
        await connection.commit()

    await purge()
    dim = settings.embedding_dim

    # Tenant A: three chunks on distinct axes, so nearest-neighbour is exact.
    await store.upsert_document(
        connection,
        tenant_id=TENANT_A,
        document=_document("refunds.md", "hash-a"),
        chunks=[
            Chunk(0, "Refunds within 30 days.", {"section": "Refunds", "source_path": "a"}),
            Chunk(1, "Billing runs monthly.", {"section": "Billing", "source_path": "a"}),
            Chunk(2, "Passwords reset by email.", {"section": "Access", "source_path": "a"}),
        ],
        embeddings=[_unit_vector(dim, 0), _unit_vector(dim, 1), _unit_vector(dim, 2)],
    )
    # Tenant B: sits on the SAME axis as tenant A's first chunk. If tenant
    # filtering is broken, it will surface in tenant A's results.
    await store.upsert_document(
        connection,
        tenant_id=TENANT_B,
        document=_document("other.md", "hash-b"),
        chunks=[Chunk(0, "TENANT B SECRET.", {"section": "Secret", "source_path": "b"})],
        embeddings=[_unit_vector(dim, 0)],
    )
    await connection.commit()

    retriever = PgVectorRetriever(settings)
    yield retriever, settings

    await retriever.aclose()
    await purge()
    await connection.close()


async def test_returns_nearest_chunk_first(seeded):
    retriever, settings = seeded
    results = await retriever.similarity_search(
        _unit_vector(settings.embedding_dim, 1), tenant_id=TENANT_A
    )
    assert results
    assert results[0].content == "Billing runs monthly."
    assert results[0].score == pytest.approx(1.0, abs=1e-6)


async def test_results_are_ordered_by_descending_score(seeded):
    retriever, settings = seeded
    results = await retriever.similarity_search(
        _unit_vector(settings.embedding_dim, 0), tenant_id=TENANT_A
    )
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


async def test_tenant_filter_excludes_other_tenants(seeded):
    """The security claim: tenant A must never see tenant B's content."""
    retriever, settings = seeded
    results = await retriever.similarity_search(
        _unit_vector(settings.embedding_dim, 0), tenant_id=TENANT_A, top_k=10
    )
    assert results
    assert all("TENANT B" not in r.content for r in results)


async def test_unknown_tenant_returns_nothing(seeded):
    retriever, settings = seeded
    results = await retriever.similarity_search(
        _unit_vector(settings.embedding_dim, 0), tenant_id="no-such-tenant"
    )
    assert results == []


async def test_top_k_limits_results(seeded):
    retriever, settings = seeded
    results = await retriever.similarity_search(
        _unit_vector(settings.embedding_dim, 0), tenant_id=TENANT_A, top_k=2
    )
    assert len(results) == 2


async def test_metadata_filter_narrows_results(seeded):
    retriever, settings = seeded
    results = await retriever.similarity_search(
        _unit_vector(settings.embedding_dim, 0),
        tenant_id=TENANT_A,
        filters={"section": "Access"},
    )
    assert len(results) == 1
    assert results[0].content == "Passwords reset by email."


async def test_results_carry_citation_data(seeded):
    retriever, settings = seeded
    results = await retriever.similarity_search(
        _unit_vector(settings.embedding_dim, 0), tenant_id=TENANT_A
    )
    top = results[0]
    assert top.source_path.endswith("refunds.md")
    assert top.title == "refunds"
    assert top.document_id
    assert top.citation() == "refunds.md (Refunds)"


async def test_wrong_embedding_width_is_rejected(seeded):
    retriever, _ = seeded
    with pytest.raises(RetrieverError, match="dims"):
        await retriever.similarity_search([0.1, 0.2], tenant_id=TENANT_A)
