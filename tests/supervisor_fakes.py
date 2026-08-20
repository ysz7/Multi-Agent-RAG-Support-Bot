"""Shared fakes for the supervisor branch tests."""

from __future__ import annotations

from app.mcp_server.tools import ToolSpec


class FakeToolResult:
    def __init__(self, content):
        self.content = content
        self.is_error = False


class FakeTools:
    """Stands in for `MCPToolClient` without touching Postgres or Ollama."""

    def __init__(self, hits=None, specs=None):
        self.hits = (
            hits
            if hits is not None
            else [
                {
                    "citation": "handbook.md (Refunds)",
                    "content": "Refunds are issued within 30 days of purchase.",
                    "score": 0.9,
                    "source_path": "/corpus/handbook.md",
                    "document_id": "doc-1",
                    "chunk_index": 0,
                }
            ]
        )
        self.specs = specs or [
            ToolSpec("search_documents", "search", sensitive=False, read_only=True),
            ToolSpec("send_email", "send", sensitive=True, read_only=False),
        ]
        self.calls: list[tuple[str, dict]] = []

    async def list_tools(self):
        return self.specs

    async def call(self, name, arguments=None, *, approved=False):
        self.calls.append((name, arguments or {}))
        if name == "send_email" and not approved:
            raise AssertionError("action_taker must never execute a sensitive tool")
        return FakeToolResult(self.hits if name == "search_documents" else {"status": "queued"})

    async def aclose(self):
        return None
