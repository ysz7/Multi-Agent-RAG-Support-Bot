"""Phase 10: the human-in-the-loop gate.

Interrupts need a checkpointer, so these use an in-memory saver. Durability
across processes is covered by the live tests against Postgres.
"""

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app.core.config import Settings
from app.core.llm_provider import ChatResult
from app.graph.approval import ApprovalGateNode, DispatchActionNode, RejectionNode, _normalise
from app.graph.build import GRAPH_RECURSION_LIMIT, build_graph
from app.graph.state import initial_state
from tests.supervisor_fakes import FakeTools


class _ScriptedLLM:
    name = "fake"
    model = "fake-1"

    def __init__(self, *replies):
        self.replies = list(replies) or [""]
        self.calls = 0

    async def chat(self, messages, *, system=None, max_tokens=None):
        self.calls += 1
        return ChatResult(
            text=self.replies[min(self.calls - 1, len(self.replies) - 1)],
            model=self.model,
            provider=self.name,
        )

    async def stream(self, messages, *, system=None, max_tokens=None):
        yield self.replies[0]

    async def aclose(self):
        return None


def _settings(env) -> Settings:
    env(ANTHROPIC_API_KEY="sk-test")
    return Settings()


def _action_graph(env, tools=None):
    """Graph primed to route an action request through the gate."""
    llm = _ScriptedLLM("COMPLEX", '["refund"]', "Draft body [1].", '{"verdict": "approved"}')
    return build_graph(
        _settings(env),
        llm=llm,
        tools=tools or FakeTools(),
        chain=None,
        checkpointer=InMemorySaver(),
    )


def _config(thread: str) -> dict:
    return {
        "configurable": {"thread_id": thread},
        "recursion_limit": GRAPH_RECURSION_LIMIT,
    }


PROPOSAL = {
    "tool": "send_email",
    "arguments": {"to": "a@example.com", "subject": "s", "body": "b"},
    "reason": "asked for it",
    "proposed_by": "action_taker",
}


# --- decision parsing (fails closed) ---------------------------------------


@pytest.mark.parametrize(
    ("decision", "expected"),
    [
        (True, "approved"),
        ("approve", "approved"),
        ("yes", "approved"),
        ({"decision": "approve"}, "approved"),
        ({"decision": "edit"}, "approved"),
        (False, "rejected"),
        (None, "rejected"),
        ("no", "rejected"),
        ("nonsense", "rejected"),
        ({"decision": "maybe"}, "rejected"),
        (12345, "rejected"),
        ([], "rejected"),
    ],
)
def test_resume_values_fail_closed(decision, expected):
    verdict, _, _ = _normalise(decision, PROPOSAL)
    assert verdict == expected


def test_edit_replaces_argument_values():
    _, arguments, _ = _normalise({"decision": "edit", "arguments": {"body": "corrected"}}, PROPOSAL)
    assert arguments["body"] == "corrected"
    assert arguments["to"] == "a@example.com", "unedited arguments must survive"


def test_edit_cannot_switch_the_tool():
    """A human may correct arguments, not escalate to a different tool."""
    decision = {"decision": "edit", "tool": "rm_rf", "arguments": {"body": "x"}}
    verdict, arguments, _ = _normalise(decision, PROPOSAL)
    assert verdict == "approved"
    assert "tool" not in arguments


# --- gate node -------------------------------------------------------------


async def test_gate_passes_through_when_nothing_is_pending():
    assert await ApprovalGateNode()({"pending_action": None}) == {"approval": None}


async def test_gate_does_not_stop_a_read_only_tool():
    proposal = {"tool": "search_documents", "arguments": {}}
    result = await ApprovalGateNode()({"pending_action": proposal})
    assert result["approval"] == "approved"


async def test_dispatch_refuses_without_approval(env):
    tools = FakeTools()
    result = await DispatchActionNode(tools, _settings(env))(
        {"pending_action": PROPOSAL, "approval": "pending"}
    )
    assert "without approval" in result["error"]
    assert tools.calls == []


async def test_dispatch_clears_pending_action(env):
    """Clearing it is what makes a replay a no-op."""
    result = await DispatchActionNode(FakeTools(), _settings(env))(
        {"pending_action": PROPOSAL, "approval": "approved"}
    )
    assert result["pending_action"] is None
    assert result["executed_actions"][0]["tool"] == "send_email"


async def test_rejection_records_and_explains():
    result = await RejectionNode()({"pending_action": PROPOSAL, "approval_note": "wrong recipient"})
    assert result["pending_action"] is None
    assert result["rejected_actions"][0]["tool"] == "send_email"
    assert "not approved" in result["draft"]
    assert "wrong recipient" in result["draft"]


# --- full interrupt / resume cycle -----------------------------------------


async def test_graph_pauses_before_executing(env):
    tools = FakeTools()
    graph = _action_graph(env, tools)
    config = _config("pause")

    result = await graph.ainvoke(
        initial_state("Email me the refund policy", tenant_id="t1", user_id="u1"), config
    )

    assert "__interrupt__" in result, "graph must pause, not finish"
    payload = result["__interrupt__"][0].value
    assert payload["type"] == "approval_request"
    assert payload["tool"] == "send_email"
    assert payload["arguments"]["to"]
    assert all(name != "send_email" for name, _ in tools.calls), "nothing may run yet"


async def test_pending_run_is_resumable_by_thread_id(env):
    graph = _action_graph(env)
    config = _config("resumable")
    await graph.ainvoke(initial_state("Email me the policy", tenant_id="t1", user_id="u1"), config)

    snapshot = await graph.aget_state(config)
    assert snapshot.next, "a paused run must have a next node"
    assert snapshot.values["pending_action"]["tool"] == "send_email"


async def test_approval_dispatches_the_action(env):
    tools = FakeTools()
    graph = _action_graph(env, tools)
    config = _config("approve")
    await graph.ainvoke(initial_state("Email me the policy", tenant_id="t1", user_id="u1"), config)

    final = await graph.ainvoke(Command(resume={"decision": "approve"}), config)

    assert ("send_email", tools.calls[-1][1]) in tools.calls
    assert final["executed_actions"][0]["tool"] == "send_email"
    assert final["pending_action"] is None


async def test_rejection_never_dispatches(env):
    tools = FakeTools()
    graph = _action_graph(env, tools)
    config = _config("reject")
    await graph.ainvoke(initial_state("Email me the policy", tenant_id="t1", user_id="u1"), config)

    final = await graph.ainvoke(
        Command(resume={"decision": "reject", "note": "not appropriate"}), config
    )

    assert all(name != "send_email" for name, _ in tools.calls)
    assert final["rejected_actions"][0]["tool"] == "send_email"
    assert "not approved" in final["answer"]
    assert "not appropriate" in final["answer"], "the reason must reach the user"


async def test_edited_arguments_are_what_get_sent(env):
    tools = FakeTools()
    graph = _action_graph(env, tools)
    config = _config("edit")
    await graph.ainvoke(initial_state("Email me the policy", tenant_id="t1", user_id="u1"), config)

    await graph.ainvoke(
        Command(resume={"decision": "edit", "arguments": {"to": "corrected@example.com"}}),
        config,
    )

    sent = [args for name, args in tools.calls if name == "send_email"]
    assert sent and sent[0]["to"] == "corrected@example.com"


async def test_action_executes_exactly_once(env):
    """Re-invoking a finished thread must not re-send."""
    tools = FakeTools()
    graph = _action_graph(env, tools)
    config = _config("once")
    await graph.ainvoke(initial_state("Email me the policy", tenant_id="t1", user_id="u1"), config)
    await graph.ainvoke(Command(resume={"decision": "approve"}), config)

    sends_after_first = len([1 for name, _ in tools.calls if name == "send_email"])
    assert sends_after_first == 1

    # A second resume on the same thread must be a no-op.
    await graph.ainvoke(Command(resume={"decision": "approve"}), config)
    assert len([1 for name, _ in tools.calls if name == "send_email"]) == 1


async def test_simple_questions_never_reach_the_gate(env):
    tools = FakeTools()
    llm = _ScriptedLLM("SIMPLE")

    class _Chain:
        async def ainvoke(self, question, *, tenant_id, **kwargs):
            from app.rag.chain import RagAnswer

            return RagAnswer(text="Plain answer.", citations=[], chunks=[])

    graph = build_graph(
        _settings(env), llm=llm, tools=tools, chain=_Chain(), checkpointer=InMemorySaver()
    )
    result = await graph.ainvoke(
        initial_state("What is the SLA?", tenant_id="t1", user_id="u1"), _config("simple")
    )

    assert "__interrupt__" not in result
    assert result["answer"] == "Plain answer."
