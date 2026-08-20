"""Shared retriever contract.

Both backends return the same `RetrievedChunk` shape so the RAG chain and the
graph nodes never learn which store answered.

`tenant_id` is a **required keyword argument** on every search. It is not
optional and never comes from a request body — Phase 11 resolves it from the
caller's `Principal`. Making it required means a caller cannot accidentally
query across tenants by omitting a filter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


class RetrieverError(RuntimeError):
    """Any retriever-level failure, normalised across backends."""


class RetrieverUnavailable(RetrieverError):
    """Backend is selected but its optional dependency is not installed."""


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """One retrieved chunk plus everything needed to cite it."""

    content: str
    score: float
    chunk_index: int
    document_id: str
    title: str
    source_path: str
    metadata: dict = field(default_factory=dict)

    def citation(self) -> str:
        """Human-readable location, e.g. 'handbook.md (Setup > Install)'."""
        name = self.source_path.rsplit("/", 1)[-1] or self.title
        if section := self.metadata.get("section"):
            return f"{name} ({section})"
        if page := self.metadata.get("page"):
            return f"{name} (p. {page})"
        return name


@runtime_checkable
class Retriever(Protocol):
    name: str

    async def similarity_search(
        self,
        query_embedding: list[float],
        *,
        tenant_id: str,
        top_k: int | None = None,
        filters: dict | None = None,
    ) -> list[RetrievedChunk]: ...

    async def aclose(self) -> None: ...
