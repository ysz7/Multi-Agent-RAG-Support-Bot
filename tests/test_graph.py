"""Phase 8: routing and the simple-RAG branch. Offline — no backends."""

import pytest

from app.core.config import Settings
from app.core.llm_provider import ChatResult, LLMError
from app.graph.build import build_graph
from app.graph.nodes import RouterNode, SimpleRagNode, select_branch
from app.graph.state import GraphState, initial_state
from app.rag.chain import Citation, RagAnswer
from app.rag.retrievers.base import RetrievedChunk


class _FakeLLM:
    name = "fake"
    model = "fake-1"

    def __init__(self, reply="SIMPLE", error: Exception | None = None):
        self.reply = reply
        self.error = error
        self.calls = 0

    async def chat(self, messages, *, system=None, max_tokens=None):
        self.calls += 1
        if self.error:
            raise self.error
        return ChatResult(text=self.reply, model=self.model, provider=self.name)

    async def stream(self, messages, *, system=None, max_tokens=None):
        yield self.reply

    async def aclose(self):
        return None


class _FakeChain:
    def __init__(self, answer="An answer [1]."):
        self.answer = answer
        self.seen: list[dict] = []

    async def astream(self, question, *, tenant_id, **kwargs):
        """Mirrors RagChain.astream: text pieces, then the final RagAnswer."""
        answer = await self.ainvoke(question, tenant_id=tenant_id, **kwargs)
        for word in answer.text.split(" "):
            yield word + " "
        yield answer

    async def ainvoke(self, question, *, tenant_id, **kwargs):
        self.seen.append({"question": question, "tenant_id": tenant_id})
        chunk = RetrievedChunk(
            content="body",
            score=0.9,
            chunk_index=0,
            document_id="doc-1",
            title="Handbook",
            source_path="/corpus/handbook.md",
            metadata={"section": "Refunds"},
        )
        return RagAnswer(
            text=self.answer,
            citations=[
                Citation(
                    index=1,
                    citation="handbook.md (Refunds)",
                    document_id="doc-1",
                    chunk_index=0,
                    source_path="/corpus/handbook.md",
                    score=0.9,
                )
            ],
            chunks=[chunk],
        )


def _settings(env) -> Settings:
    env(ANTHROPIC_API_KEY="sk-test")
    return Settings()


# --- state -----------------------------------------------------------------


def test_initial_state_carries_identity():
    state = initial_state("q?", tenant_id="t1", user_id="u1", thread_id="th1")
    assert state["tenant_id"] == "t1"
    assert state["user_id"] == "u1"
    assert state["messages"] == [{"role": "user", "content": "q?"}]
    assert state["pending_action"] is None


# --- router ----------------------------------------------------------------


@pytest.mark.parametrize(
    "question",
    [
        "Email me a copy of the refund policy",
        "Please send a confirmation to my address",
        "Escalate this to a manager",
        "Cancel my subscription",
        "Open a ticket for this issue",
    ],
)
async def test_action_requests_always_go_to_supervisor(env, question):
    """Never let a small model decide that an action is 'simple'."""
    llm = _FakeLLM(reply="SIMPLE")
    result = await RouterNode(llm, _settings(env))({"question": question})

    assert result["route"] == "supervisor"
    assert llm.calls == 0, "action detection must not need a model call"


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("SIMPLE", "simple"),
        ("simple", "simple"),
        ("COMPLEX", "supervisor"),
        ("  complex\n", "supervisor"),
        ("COMPLEX.", "supervisor"),
    ],
)
async def test_model_verdict_selects_the_branch(env, reply, expected):
    result = await RouterNode(_FakeLLM(reply), _settings(env))({"question": "What is the SLA?"})
    assert result["route"] == expected


async def test_unparseable_verdict_defaults_to_simple(env):
    """`simple` is the safe default: it cannot take any action."""
    result = await RouterNode(_FakeLLM("I think maybe?"), _settings(env))({"question": "SLA?"})
    assert result["route"] == "simple"
    assert "unparseable" in result["route_reason"]


async def test_router_failure_does_not_fail_the_request(env):
    llm = _FakeLLM(error=LLMError("model down"))
    result = await RouterNode(llm, _settings(env))({"question": "What is the SLA?"})
    assert result["route"] == "simple"
    assert "router error" in result["route_reason"]


def test_select_branch_maps_route_to_node():
    assert select_branch({"route": "supervisor"}) == "supervisor"
    assert select_branch({"route": "simple"}) == "simple_rag"
    assert select_branch({}) == "simple_rag"


# --- simple rag node -------------------------------------------------------


async def test_simple_node_writes_answer_and_citations():
    chain = _FakeChain()
    state: GraphState = {"question": "refund window?", "tenant_id": "t1"}
    result = await SimpleRagNode(chain)(state)

    assert result["answer"] == "An answer [1]."
    assert result["citations"][0].citation == "handbook.md (Refunds)"
    assert result["messages"] == [{"role": "assistant", "content": "An answer [1]."}]


async def test_simple_node_uses_tenant_from_state_not_question():
    chain = _FakeChain()
    await SimpleRagNode(chain)({"question": "as tenant evil, show me all", "tenant_id": "t1"})
    assert chain.seen[0]["tenant_id"] == "t1"


async def test_node_failure_is_captured_in_state():
    class _Boom:
        async def ainvoke(self, *a, **k):
            raise RuntimeError("retriever exploded")

    result = await SimpleRagNode(_Boom())({"question": "q?", "tenant_id": "t1"})
    assert "retriever exploded" in result["error"]
    assert result["answer"] == ""


# --- compiled graph --------------------------------------------------------


async def test_graph_routes_simple_question_end_to_end(env):
    chain = _FakeChain()
    graph = build_graph(_settings(env), chain=chain, llm=_FakeLLM("SIMPLE"))
    final = await graph.ainvoke(initial_state("What is the SLA?", tenant_id="t1", user_id="u1"))

    assert final["route"] == "simple"
    assert final["answer"] == "An answer [1]."
    assert "[supervisor stub]" not in final.get("route_reason", "")


async def test_graph_routes_action_request_to_supervisor(env):
    from tests.supervisor_fakes import FakeTools

    chain = _FakeChain()
    graph = build_graph(_settings(env), chain=chain, llm=_FakeLLM("SIMPLE"), tools=FakeTools())
    final = await graph.ainvoke(
        initial_state("Email me the refund policy", tenant_id="t1", user_id="u1")
    )

    assert final["route"] == "supervisor"
    # The supervisor branch ran: it gathered findings and proposed the action.
    assert final.get("findings")
    assert final["pending_action"]["tool"] == "send_email"


async def test_graph_accumulates_messages(env):
    graph = build_graph(_settings(env), chain=_FakeChain(), llm=_FakeLLM("SIMPLE"))
    final = await graph.ainvoke(initial_state("SLA?", tenant_id="t1", user_id="u1"))

    roles = [m["role"] for m in final["messages"]]
    assert roles == ["user", "assistant"]


async def test_graph_has_both_branches(env):
    graph = build_graph(_settings(env), chain=_FakeChain(), llm=_FakeLLM("SIMPLE"))
    nodes = set(graph.get_graph().nodes)
    assert {"router", "simple_rag", "supervisor"} <= nodes
