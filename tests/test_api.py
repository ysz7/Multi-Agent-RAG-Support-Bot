"""Phase 11: FastAPI layer — auth, tenant isolation, SSE, approvals.

The graph and approval store are replaced with fakes, so these assert HTTP
behaviour and the security properties rather than model quality.
"""

import json
from datetime import UTC, datetime

import httpx
from fastapi import FastAPI

from app.api import approvals as approvals_routes
from app.api import chat as chat_routes
from app.api import health as health_routes
from app.api.approvals_store import PendingApproval
from app.core.config import Settings
from app.rag.chain import Citation


class FakeGraph:
    """Minimal stand-in for a compiled LangGraph."""

    def __init__(self, *, interrupt=False, answer="An answer [1]."):
        self.interrupt = interrupt
        self.answer = answer
        self.seen: list[dict] = []
        self.resumed: list = []

    def _values(self):
        values = {
            "answer": self.answer,
            "route": "simple",
            "citations": [
                Citation(
                    index=1,
                    citation="handbook.md (Refunds)",
                    document_id="doc-1",
                    chunk_index=0,
                    source_path="/corpus/handbook.md",
                    score=0.9,
                )
            ],
            "question": "q?",
        }
        if self.interrupt:
            values["pending_action"] = {
                "tool": "send_email",
                "arguments": {"to": "a@example.com", "subject": "s", "body": "b"},
                "reason": "asked",
            }
        return values

    async def ainvoke(self, payload, config=None):
        from langgraph.types import Command

        if isinstance(payload, Command):
            self.resumed.append(payload.resume)
            return {"answer": "done", "executed_actions": [{"tool": "send_email"}]}

        self.seen.append(payload)
        state = dict(self._values())
        if self.interrupt:
            state["__interrupt__"] = [type("I", (), {"value": state["pending_action"]})()]
        return state

    async def astream(self, payload, config=None, stream_mode=None):
        self.seen.append(payload)
        yield "updates", {"router": {}}
        for word in self.answer.split(" "):
            yield "custom", {"type": "token", "text": word + " "}
        yield "updates", {"simple_rag": {}}

    async def aget_state(self, config):
        return type("S", (), {"values": self._values(), "next": ()})()


class FakeApprovals:
    def __init__(self, rows=None):
        self.rows = rows or {}
        self.recorded: list[dict] = []
        self.closed: list[dict] = []

    async def setup(self):
        return None

    async def record(self, **kwargs):
        self.recorded.append(kwargs)

    async def list_pending(self, *, tenant_id):
        return [r for r in self.rows.values() if r.tenant_id == tenant_id]

    async def get(self, *, thread_id, tenant_id):
        row = self.rows.get(thread_id)
        # Tenant is part of the lookup, not a post-hoc comparison.
        return row if row and row.tenant_id == tenant_id else None

    async def close(self, *, thread_id, status, decided_by, note):
        self.closed.append({"thread_id": thread_id, "status": status, "by": decided_by})


def _pending(thread_id="t-1", tenant_id="default") -> PendingApproval:
    return PendingApproval(
        thread_id=thread_id,
        tenant_id=tenant_id,
        user_id="u1",
        question="Email me the policy",
        tool="send_email",
        arguments={"to": "a@example.com", "subject": "s", "body": "b"},
        reason="asked",
        status="pending",
        created_at=datetime.now(UTC),
    )


def _app(settings: Settings, graph=None, approvals=None) -> FastAPI:
    app = FastAPI()
    app.include_router(health_routes.router)
    app.include_router(chat_routes.router)
    app.include_router(approvals_routes.router)
    app.state.settings = settings
    app.state.graph = graph or FakeGraph()
    app.state.approvals = approvals or FakeApprovals()
    return app


def _client(app: FastAPI) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


def _settings(env, **overrides) -> Settings:
    env(ANTHROPIC_API_KEY="sk-test", **overrides)
    return Settings()


def _token(settings: Settings, *, tenant="default", scopes="chat approvals:write", **extra):
    from datetime import timedelta

    from jose import jwt

    now = datetime.now(UTC)
    claims = {
        "sub": "alice",
        "tenant_id": tenant,
        "scope": scopes,
        "iat": now,
        "exp": now + timedelta(hours=1),
        **extra,
    }
    return jwt.encode(claims, settings.jwt_secret, algorithm=settings.jwt_algorithm)


# --- local auth mode -------------------------------------------------------


async def test_health_reports_configuration(env):
    settings = _settings(env, DATABASE_URL="postgresql://nobody@127.0.0.1:1/none")
    async with _client(_app(settings)) as client:
        response = await client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["auth_mode"] == "local"
    assert body["llm_provider"] == settings.llm_provider
    assert "error" in body["database"], "unreachable db should degrade, not crash"


async def test_local_mode_needs_no_token(env):
    async with _client(_app(_settings(env))) as client:
        response = await client.post("/chat", json={"question": "What is the SLA?"})
    assert response.status_code == 200
    assert response.json()["answer"] == "An answer [1]."


async def test_chat_uses_the_configured_tenant(env):
    settings = _settings(env, LOCAL_TENANT_ID="acme")
    graph = FakeGraph()
    async with _client(_app(settings, graph)) as client:
        await client.post("/chat", json={"question": "q?"})
    assert graph.seen[0]["tenant_id"] == "acme"


# --- the tenant-injection property -----------------------------------------


async def test_client_supplied_tenant_is_rejected_outright(env):
    """Not merely ignored: the schema forbids the field, so it is a 422."""
    settings = _settings(env, LOCAL_TENANT_ID="acme")
    graph = FakeGraph()
    async with _client(_app(settings, graph)) as client:
        response = await client.post("/chat", json={"question": "q?", "tenant_id": "victim"})

    assert response.status_code == 422
    assert graph.seen == [], "the graph must never have run"


async def test_tenant_cannot_be_smuggled_via_user_id(env):
    settings = _settings(env, LOCAL_TENANT_ID="acme")
    graph = FakeGraph()
    async with _client(_app(settings, graph)) as client:
        response = await client.post("/chat", json={"question": "q?", "user_id": "root"})
    assert response.status_code == 422


# --- jwt auth mode ---------------------------------------------------------


async def test_jwt_mode_rejects_a_missing_token(env):
    settings = _settings(env, AUTH_MODE="jwt", JWT_SECRET="dev-secret")
    async with _client(_app(settings)) as client:
        response = await client.post("/chat", json={"question": "q?"})

    assert response.status_code == 401
    assert "missing bearer token" in response.json()["detail"]


async def test_jwt_mode_rejects_a_forged_token(env):
    settings = _settings(env, AUTH_MODE="jwt", JWT_SECRET="dev-secret")
    from jose import jwt

    forged = jwt.encode({"sub": "mallory", "tenant_id": "victim"}, "wrong-key")
    async with _client(_app(settings)) as client:
        response = await client.post(
            "/chat",
            json={"question": "q?"},
            headers={"Authorization": f"Bearer {forged}"},
        )
    assert response.status_code == 401


async def test_jwt_tenant_claim_drives_retrieval(env):
    settings = _settings(env, AUTH_MODE="jwt", JWT_SECRET="dev-secret")
    graph = FakeGraph()
    async with _client(_app(settings, graph)) as client:
        response = await client.post(
            "/chat",
            json={"question": "q?"},
            headers={"Authorization": f"Bearer {_token(settings, tenant='acme')}"},
        )

    assert response.status_code == 200
    assert graph.seen[0]["tenant_id"] == "acme"
    assert graph.seen[0]["user_id"] == "alice"


async def test_token_without_tenant_claim_is_forbidden(env):
    settings = _settings(env, AUTH_MODE="jwt", JWT_SECRET="dev-secret")
    from datetime import timedelta

    from jose import jwt

    now = datetime.now(UTC)
    token = jwt.encode(
        {"sub": "alice", "exp": now + timedelta(hours=1)},
        settings.jwt_secret,
        algorithm=settings.jwt_algorithm,
    )
    async with _client(_app(settings)) as client:
        response = await client.post(
            "/chat", json={"question": "q?"}, headers={"Authorization": f"Bearer {token}"}
        )
    assert response.status_code == 403


async def test_expired_token_is_rejected(env):
    settings = _settings(env, AUTH_MODE="jwt", JWT_SECRET="dev-secret")
    from datetime import timedelta

    from jose import jwt

    past = datetime.now(UTC) - timedelta(hours=2)
    token = jwt.encode(
        {"sub": "a", "tenant_id": "t", "exp": past},
        settings.jwt_secret,
        algorithm=settings.jwt_algorithm,
    )
    async with _client(_app(settings)) as client:
        response = await client.post(
            "/chat", json={"question": "q?"}, headers={"Authorization": f"Bearer {token}"}
        )
    assert response.status_code == 401


async def test_scope_is_enforced_on_approvals(env):
    settings = _settings(env, AUTH_MODE="jwt", JWT_SECRET="dev-secret")
    token = _token(settings, scopes="chat")  # no approvals:write
    async with _client(_app(settings)) as client:
        response = await client.get("/approvals", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 403
    assert "approvals:write" in response.json()["detail"]


# --- SSE -------------------------------------------------------------------


async def test_stream_emits_tokens_then_the_answer(env):
    async with _client(_app(_settings(env))) as client:
        response = await client.post("/chat/stream", json={"question": "q?"})
        body = response.text

    assert response.status_code == 200
    assert "event: token" in body
    assert "event: answer" in body
    assert "event: done" in body

    tokens = [
        json.loads(line.split("data: ", 1)[1])["text"]
        for line in body.splitlines()
        if line.startswith("data: ") and '"text"' in line
    ]
    assert "".join(tokens).strip() == "An answer [1]."


async def test_stream_reports_an_approval_pause(env):
    approvals = FakeApprovals()
    app = _app(_settings(env), FakeGraph(interrupt=True), approvals)
    async with _client(app) as client:
        body = (await client.post("/chat/stream", json={"question": "Email me"})).text

    assert "event: approval" in body
    assert "event: answer" not in body
    assert approvals.recorded[0]["tool"] == "send_email"


# --- approvals -------------------------------------------------------------


async def test_chat_returns_approval_required(env):
    approvals = FakeApprovals()
    app = _app(_settings(env), FakeGraph(interrupt=True), approvals)
    async with _client(app) as client:
        body = (await client.post("/chat", json={"question": "Email me"})).json()

    assert body["approval_required"] is True
    assert body["pending_action"]["tool"] == "send_email"
    assert body["answer"] == ""


async def test_list_approvals_is_tenant_scoped(env):
    settings = _settings(env, LOCAL_TENANT_ID="acme")
    approvals = FakeApprovals(
        {
            "mine": _pending("mine", "acme"),
            "theirs": _pending("theirs", "other-tenant"),
        }
    )
    async with _client(_app(settings, approvals=approvals)) as client:
        rows = (await client.get("/approvals")).json()

    assert [r["thread_id"] for r in rows] == ["mine"]


async def test_another_tenants_approval_is_404_not_403(env):
    """404 avoids confirming that the thread exists at all."""
    settings = _settings(env, LOCAL_TENANT_ID="acme")
    approvals = FakeApprovals({"theirs": _pending("theirs", "other-tenant")})
    async with _client(_app(settings, approvals=approvals)) as client:
        response = await client.post("/approvals/theirs", json={"decision": "approve"})

    assert response.status_code == 404
    assert approvals.closed == []


async def test_approve_resumes_the_graph(env):
    graph = FakeGraph()
    approvals = FakeApprovals({"t-1": _pending()})
    async with _client(_app(_settings(env), graph, approvals)) as client:
        body = (await client.post("/approvals/t-1", json={"decision": "approve"})).json()

    assert body["status"] == "approved"
    assert graph.resumed[0]["decision"] == "approve"
    assert approvals.closed[0]["status"] == "approved"


async def test_reject_resumes_with_the_note(env):
    graph = FakeGraph()
    approvals = FakeApprovals({"t-1": _pending()})
    async with _client(_app(_settings(env), graph, approvals)) as client:
        body = (
            await client.post(
                "/approvals/t-1", json={"decision": "reject", "note": "wrong address"}
            )
        ).json()

    assert body["status"] == "rejected"
    assert graph.resumed[0]["note"] == "wrong address"


async def test_edit_requires_arguments(env):
    approvals = FakeApprovals({"t-1": _pending()})
    async with _client(_app(_settings(env), approvals=approvals)) as client:
        response = await client.post("/approvals/t-1", json={"decision": "edit"})
    assert response.status_code == 422


async def test_edit_forwards_corrected_arguments(env):
    graph = FakeGraph()
    approvals = FakeApprovals({"t-1": _pending()})
    async with _client(_app(_settings(env), graph, approvals)) as client:
        await client.post(
            "/approvals/t-1",
            json={"decision": "edit", "arguments": {"to": "fixed@example.com"}},
        )

    assert graph.resumed[0]["arguments"] == {"to": "fixed@example.com"}


async def test_deciding_twice_is_a_conflict(env):
    import dataclasses

    decided = dataclasses.replace(_pending(), status="approved")
    approvals = FakeApprovals({"t-1": decided})
    async with _client(_app(_settings(env), approvals=approvals)) as client:
        response = await client.post("/approvals/t-1", json={"decision": "approve"})

    assert response.status_code == 409


async def test_unknown_decision_is_rejected_by_schema(env):
    approvals = FakeApprovals({"t-1": _pending()})
    async with _client(_app(_settings(env), approvals=approvals)) as client:
        response = await client.post("/approvals/t-1", json={"decision": "maybe"})
    assert response.status_code == 422


# --- tracing (Phase 12) ----------------------------------------------------


async def test_chat_returns_a_trace_id_and_traces_the_request(env, monkeypatch):
    """The trace id is returned so an answer can be correlated with its trace."""
    from app.core import observability as obs
    from tests.test_observability import FakeClient

    client_stub = FakeClient()
    monkeypatch.setattr(obs, "_client", client_stub)
    monkeypatch.setattr(obs, "callback_handler", lambda: "handler")

    graph = FakeGraph()
    async with _client(_app(_settings(env), graph)) as client:
        response = await client.post("/chat", json={"question": "q?"})

    body = response.json()
    assert body["trace_id"] == "trace-abc"
    # The graph run carries both the handler and the trace id into state.
    assert graph.seen[0]["trace_id"] == "trace-abc"
    assert client_stub.by_name("chat").ended


async def test_chat_works_with_tracing_disabled(env):
    """No keys configured: no trace id, same answer."""
    async with _client(_app(_settings(env))) as client:
        response = await client.post("/chat", json={"question": "q?"})
    body = response.json()
    assert body["answer"] == "An answer [1]."
    assert body["trace_id"] is None


async def test_stream_reports_the_trace_id_with_the_answer(env, monkeypatch):
    from app.core import observability as obs
    from tests.test_observability import FakeClient

    monkeypatch.setattr(obs, "_client", FakeClient())

    async with _client(_app(_settings(env))) as client:
        response = await client.post("/chat/stream", json={"question": "q?"})

    answers = [
        json.loads(line[len("data: ") :])
        for line in response.text.splitlines()
        if line.startswith("data: ") and "answer" in line
    ]
    assert answers and answers[-1]["trace_id"] == "trace-abc"
