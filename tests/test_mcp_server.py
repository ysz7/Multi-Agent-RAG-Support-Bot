"""Phase 7: MCP tool server — advertisement, sensitivity, and the approval gate.

Offline: the toolbox backends are stubbed, so these assert wiring and policy
rather than retrieval quality.
"""

import json

import pytest

from app.core.config import Settings
from app.mcp_server.client import ApprovalRequired, MCPToolClient
from app.mcp_server.server import build_server
from app.mcp_server.tools import TOOL_SPECS, Toolbox, is_sensitive


def _settings(env, **overrides) -> Settings:
    env(ANTHROPIC_API_KEY="sk-test", **overrides)
    return Settings()


@pytest.fixture
def client(env, tmp_path, monkeypatch):
    settings = _settings(env, DOCUMENTS_DIR=str(tmp_path / "documents"))

    async def fake_search(self, query, top_k=5, section=None):
        return [
            {
                "citation": "handbook.md (Refunds)",
                "content": f"result for {query}",
                "score": 0.9,
                "source_path": "/corpus/handbook.md",
                "document_id": "doc-1",
                "chunk_index": 0,
            }
        ]

    async def fake_list(self):
        return [{"title": "Handbook", "source_path": "/corpus/handbook.md", "chunks": 3}]

    monkeypatch.setattr(Toolbox, "search_documents", fake_search)
    monkeypatch.setattr(Toolbox, "list_documents", fake_list)
    return MCPToolClient(settings), settings


# --- advertisement ---------------------------------------------------------


async def test_server_advertises_every_tool(client):
    tool_client, _ = client
    names = {spec.name for spec in await tool_client.list_tools()}
    assert names == set(TOOL_SPECS)


async def test_read_only_tools_are_marked_read_only(client):
    tool_client, _ = client
    specs = {spec.name: spec for spec in await tool_client.list_tools()}
    assert specs["search_documents"].read_only is True
    assert specs["list_documents"].read_only is True
    assert specs["send_email"].read_only is False


async def test_only_send_email_is_sensitive(client):
    tool_client, _ = client
    specs = {spec.name: spec for spec in await tool_client.list_tools()}
    sensitive = {name for name, spec in specs.items() if spec.sensitive}
    assert sensitive == {"send_email"}


async def test_tool_schemas_are_exposed_for_external_clients(client):
    """Claude Desktop needs input schemas to call these tools."""
    tool_client, _ = client
    tools = {t.name: t for t in await tool_client._server.list_tools()}
    schema = tools["search_documents"].input_schema
    assert "query" in schema["properties"]
    assert schema["required"] == ["query"]


def test_unknown_tools_fail_closed():
    """An unrecognised name must be treated as sensitive, not as safe."""
    assert is_sensitive("definitely_not_a_tool") is True


# --- read-only tools -------------------------------------------------------


async def test_search_documents_runs_without_approval(client):
    tool_client, _ = client
    result = await tool_client.call("search_documents", {"query": "refunds"})
    assert not result.is_error
    assert result.content[0]["citation"] == "handbook.md (Refunds)"


async def test_search_does_not_accept_a_tenant_argument(client):
    """Tenant is resolved server-side; a model must not be able to pass one."""
    tool_client, _ = client
    tools = {t.name: t for t in await tool_client._server.list_tools()}
    assert "tenant_id" not in tools["search_documents"].input_schema["properties"]


async def test_list_documents_runs_without_approval(client):
    tool_client, _ = client
    result = await tool_client.call("list_documents")
    assert result.content[0]["title"] == "Handbook"


# --- the approval gate -----------------------------------------------------


async def test_sensitive_tool_is_refused_without_approval(client):
    tool_client, _ = client
    with pytest.raises(ApprovalRequired, match="send_email"):
        await tool_client.call(
            "send_email", {"to": "a@example.com", "subject": "Hi", "body": "Hello"}
        )


async def test_sensitive_tool_runs_once_approved(client, tmp_path):
    tool_client, settings = client
    result = await tool_client.call(
        "send_email",
        {"to": "a@example.com", "subject": "Refund", "body": "Processed."},
        approved=True,
    )
    assert result.content["status"] == "queued"

    outbox = tmp_path / "outbox.jsonl"
    assert outbox.exists(), "approved send should reach the outbox"
    record = json.loads(outbox.read_text().splitlines()[0])
    assert record["to"] == "a@example.com"
    assert record["subject"] == "Refund"


async def test_refused_call_leaves_no_trace(client, tmp_path):
    """A blocked send must not have side effects."""
    tool_client, _ = client
    with pytest.raises(ApprovalRequired):
        await tool_client.call("send_email", {"to": "a@x.com", "subject": "s", "body": "b"})
    assert not (tmp_path / "outbox.jsonl").exists()


async def test_approval_flag_does_not_leak_between_calls(client, tmp_path):
    tool_client, _ = client
    await tool_client.call(
        "send_email", {"to": "a@x.com", "subject": "s", "body": "b"}, approved=True
    )
    with pytest.raises(ApprovalRequired):
        await tool_client.call("send_email", {"to": "b@x.com", "subject": "s", "body": "b"})
    assert len((tmp_path / "outbox.jsonl").read_text().strip().splitlines()) == 1


# --- server metadata -------------------------------------------------------


def test_server_instructions_flag_untrusted_content(env):
    server = build_server(_settings(env))
    assert "untrusted" in (server.instructions or "").lower()
    assert "never as instructions" in (server.instructions or "").lower()
