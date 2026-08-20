"""Phase 5: retriever factory and the shared contract. Offline."""

import sys

import pytest

from app.core.config import Settings
from app.rag.retrievers import RetrieverUnavailable, get_retriever
from app.rag.retrievers.base import RetrievedChunk, Retriever
from app.rag.retrievers.pgvector import PgVectorRetriever
from app.rag.retrievers.qdrant import QdrantRetriever


def _settings(env, **overrides) -> Settings:
    env(ANTHROPIC_API_KEY="sk-test", **overrides)
    return Settings()


def test_default_backend_is_pgvector(env):
    retriever = get_retriever(_settings(env))
    assert isinstance(retriever, PgVectorRetriever)
    assert retriever.name == "pgvector"


def test_factory_selects_qdrant(env, monkeypatch):
    """qdrant_client is not installed by default, so stub the constructor."""
    monkeypatch.setattr(
        "app.rag.retrievers.qdrant._import_client", lambda: lambda **kwargs: object()
    )
    assert isinstance(get_retriever(_settings(env, VECTOR_STORE="qdrant")), QdrantRetriever)


def test_both_backends_satisfy_the_protocol(env):
    """The Qdrant adapter is unverified but must still honour the interface."""
    assert isinstance(get_retriever(_settings(env)), Retriever)
    assert isinstance(QdrantRetriever(_settings(env), client=object()), Retriever)


def test_unknown_backend_is_rejected_by_settings(env):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        _settings(env, VECTOR_STORE="faiss")


def test_missing_qdrant_extra_gives_an_actionable_error(env, monkeypatch):
    """Selecting qdrant without the extra must say how to install it."""
    monkeypatch.setitem(sys.modules, "qdrant_client", None)
    with pytest.raises(RetrieverUnavailable, match=r"\[qdrant\]"):
        get_retriever(_settings(env, VECTOR_STORE="qdrant"))


# --- citation formatting ---------------------------------------------------


def _chunk(**overrides) -> RetrievedChunk:
    base = dict(
        content="body",
        score=0.9,
        chunk_index=0,
        document_id="doc-1",
        title="Handbook",
        source_path="/data/documents/handbook.md",
        metadata={},
    )
    return RetrievedChunk(**{**base, **overrides})


def test_citation_prefers_section():
    chunk = _chunk(metadata={"section": "Setup > Install"})
    assert chunk.citation() == "handbook.md (Setup > Install)"


def test_citation_falls_back_to_page():
    chunk = _chunk(source_path="/data/documents/policy.pdf", metadata={"page": 3})
    assert chunk.citation() == "policy.pdf (p. 3)"


def test_citation_without_location_is_just_the_filename():
    assert _chunk().citation() == "handbook.md"
