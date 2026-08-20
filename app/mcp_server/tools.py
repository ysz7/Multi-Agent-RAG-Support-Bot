"""Tool implementations and their sensitivity metadata.

Two rules shape everything here:

* **Least privilege.** Read-only tools are the default. The one write/send tool
  (`send_email`) is tagged sensitive and, in this reference implementation,
  writes to a local outbox instead of sending real mail — the pipeline is
  demonstrable without granting the process an SMTP credential.
* **Tenant never comes from tool arguments.** `search_documents` takes no
  `tenant_id`; the server resolves it from configuration (and, in Phase 11,
  from the caller's `Principal`). A model that could pass its own tenant id
  could read another tenant's documents.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from app.core.config import Settings, get_settings
from app.core.llm_provider import get_embedding_provider
from app.rag.retrievers import get_retriever

OUTBOX_FILENAME = "outbox.jsonl"


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """What the graph needs to know about a tool before calling it."""

    name: str
    description: str
    sensitive: bool
    read_only: bool


TOOL_SPECS: dict[str, ToolSpec] = {
    "search_documents": ToolSpec(
        name="search_documents",
        description="Semantic search over the indexed document corpus.",
        sensitive=False,
        read_only=True,
    ),
    "list_documents": ToolSpec(
        name="list_documents",
        description="List indexed documents with chunk counts.",
        sensitive=False,
        read_only=True,
    ),
    "send_email": ToolSpec(
        name="send_email",
        description="Send an email to a customer. Requires human approval.",
        sensitive=True,
        read_only=False,
    ),
}


def is_sensitive(name: str) -> bool:
    """Unknown tools are treated as sensitive — fail closed, not open."""
    spec = TOOL_SPECS.get(name)
    return True if spec is None else spec.sensitive


class Toolbox:
    """Holds the backends the tools need, so handlers stay simple."""

    def __init__(self, settings: Settings | None = None, *, tenant_id: str | None = None) -> None:
        self._settings = settings or get_settings()
        self.tenant_id = tenant_id or self._settings.local_tenant_id
        self._retriever = None
        self._embedder = None

    def _ensure(self):
        if self._retriever is None:
            self._retriever = get_retriever(self._settings)
            self._embedder = get_embedding_provider(self._settings)
        return self._retriever, self._embedder

    # --- read-only ---------------------------------------------------------

    async def search_documents(self, query: str, top_k: int = 5, section: str | None = None):
        retriever, embedder = self._ensure()
        [vector] = await embedder.embed([query])
        chunks = await retriever.similarity_search(
            vector,
            tenant_id=self.tenant_id,  # server-side, never from arguments
            top_k=top_k,
            filters={"section": section} if section else None,
        )
        return [
            {
                "citation": chunk.citation(),
                "content": chunk.content,
                "score": round(chunk.score, 4),
                "source_path": chunk.source_path,
                "document_id": chunk.document_id,
                "chunk_index": chunk.chunk_index,
            }
            for chunk in chunks
        ]

    async def list_documents(self) -> list[dict]:
        from psycopg import AsyncConnection
        from psycopg.rows import dict_row

        connection = await AsyncConnection.connect(
            self._settings.database_url, row_factory=dict_row
        )
        try:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    """
                    SELECT d.title, d.source_path, count(c.id) AS chunks
                    FROM documents d
                    LEFT JOIN chunks c ON c.document_id = d.id
                    WHERE d.tenant_id = %s
                    GROUP BY d.id, d.title, d.source_path
                    ORDER BY d.source_path
                    """,
                    (self.tenant_id,),
                )
                return [dict(row) for row in await cursor.fetchall()]
        finally:
            await connection.close()

    # --- sensitive ---------------------------------------------------------

    async def send_email(self, to: str, subject: str, body: str) -> dict:
        """Append to a local outbox. Deliberately not a real mail send.

        The human-in-the-loop gate (Phase 10) runs *before* this is reached;
        nothing here re-checks approval, because by design the graph never
        dispatches a sensitive call that has not been approved.
        """
        outbox = Path(self._settings.documents_dir).parent / OUTBOX_FILENAME
        outbox.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "to": to,
            "subject": subject,
            "body": body,
            "tenant_id": self.tenant_id,
            "queued_at": datetime.now(UTC).isoformat(),
        }
        with outbox.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        return {"status": "queued", "outbox": str(outbox), "to": to, "subject": subject}

    async def aclose(self) -> None:
        if self._retriever is not None:
            await self._retriever.aclose()
        if self._embedder is not None:
            await self._embedder.aclose()
        self._retriever = self._embedder = None
