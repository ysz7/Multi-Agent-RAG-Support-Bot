"""Graph assembly.

    START → router ─┬→ simple_rag ──────────────────────────────────→ END
                    └→ supervisor ─┬→ researcher ───────────→ supervisor
                                   ├→ action_taker → approval_gate
                                   │                   ├→ dispatch_action → supervisor
                                   │                   └→ rejected_action → supervisor
                                   ├→ reviewer ─────────────→ supervisor
                                   └→ END

The first conditional edge reads `state["route"]`; the second reads
`state["next_step"]`. `approval_gate` interrupts the run, so **no sensitive tool
can be reached without passing a human decision first** — it is a structural
property of the graph, not a check a node has to remember to make.

Checkpointing is Postgres-backed so a graph interrupted for human approval can
be resumed later — in a different process, after a restart. That is what makes
the Phase 10 gate durable rather than an in-memory pause.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from langgraph.graph import END, START, StateGraph

from app.core.config import Settings, get_settings
from app.core.llm_provider import get_llm_provider
from app.core.observability import instrument_llm, instrument_tools
from app.graph.approval import (
    ApprovalGateNode,
    DispatchActionNode,
    RejectionNode,
    approval_branch,
)
from app.graph.nodes import RouterNode, SimpleRagNode, select_branch
from app.graph.state import MAX_SUPERVISOR_ITERATIONS, GraphState
from app.graph.supervisor import (
    ActionTakerNode,
    ResearcherNode,
    ReviewerNode,
    SupervisorNode,
    supervisor_branch,
)
from app.rag.chain import Citation, build_rag_chain
from app.rag.retrievers.base import RetrievedChunk


def build_graph_builder(
    settings: Settings | None = None,
    *,
    chain=None,
    llm=None,
    tools=None,
) -> StateGraph:
    """Wire nodes and edges. Dependencies are injectable so tests need no backends."""
    settings = settings or get_settings()
    chain = chain if chain is not None else build_rag_chain(settings)
    # Only what we construct is instrumented: injected test doubles are left
    # exactly as the test wrote them.
    llm = llm if llm is not None else instrument_llm(get_llm_provider(settings), settings)
    if tools is None:
        from app.mcp_server.client import MCPToolClient

        tools = instrument_tools(MCPToolClient(settings))

    builder = StateGraph(GraphState)
    builder.add_node("router", RouterNode(llm, settings))
    builder.add_node("simple_rag", SimpleRagNode(chain))
    builder.add_node("supervisor", SupervisorNode(llm, settings))
    builder.add_node("researcher", ResearcherNode(llm, tools, settings))
    builder.add_node("action_taker", ActionTakerNode(llm, tools, settings))
    builder.add_node("reviewer", ReviewerNode(llm, settings))
    builder.add_node("approval_gate", ApprovalGateNode())
    builder.add_node("dispatch_action", DispatchActionNode(tools, settings))
    builder.add_node("rejected_action", RejectionNode())

    builder.add_edge(START, "router")
    builder.add_conditional_edges(
        "router",
        select_branch,
        {"simple_rag": "simple_rag", "supervisor": "supervisor"},
    )
    builder.add_conditional_edges(
        "supervisor",
        supervisor_branch,
        {
            "researcher": "researcher",
            "action_taker": "action_taker",
            "reviewer": "reviewer",
            "__end__": END,
        },
    )
    # A proposed action goes to the gate, never straight back to the supervisor.
    builder.add_edge("action_taker", "approval_gate")
    builder.add_conditional_edges(
        "approval_gate",
        approval_branch,
        {"dispatch_action": "dispatch_action", "rejected_action": "rejected_action"},
    )

    # Every specialist reports back; only the supervisor may end the run.
    builder.add_edge("researcher", "supervisor")
    builder.add_edge("dispatch_action", "supervisor")
    builder.add_edge("rejected_action", "supervisor")
    builder.add_edge("reviewer", "supervisor")
    builder.add_edge("simple_rag", END)
    return builder


# Each supervisor turn costs up to 2 graph steps, plus entry and slack.
GRAPH_RECURSION_LIMIT = MAX_SUPERVISOR_ITERATIONS * 2 + 6


def build_graph(settings: Settings | None = None, *, checkpointer=None, **kwargs):
    """Compile the graph. Pass a checkpointer to make runs resumable."""
    return build_graph_builder(settings, **kwargs).compile(checkpointer=checkpointer)


def make_serde():
    """Serializer that knows our state dataclasses.

    `RetrievedChunk` and `Citation` live in graph state, so the checkpointer has
    to deserialize them. Unregistered types currently warn and will be **blocked**
    in a future LangGraph release, so they are allow-listed explicitly rather
    than left to the default.
    """
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    return JsonPlusSerializer(allowed_msgpack_modules=[RetrievedChunk, Citation])


@asynccontextmanager
async def postgres_checkpointer(settings: Settings | None = None):
    """Postgres-backed checkpointer, tables created on first use.

    `setup()` is idempotent, so calling it on every startup is safe.
    """
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    settings = settings or get_settings()
    async with AsyncPostgresSaver.from_conn_string(
        settings.database_url, serde=make_serde()
    ) as saver:
        await saver.setup()
        yield saver


@asynccontextmanager
async def compiled_graph(settings: Settings | None = None, **kwargs):
    """The graph with durable checkpointing, ready to invoke."""
    settings = settings or get_settings()
    async with postgres_checkpointer(settings) as checkpointer:
        yield build_graph(settings, checkpointer=checkpointer, **kwargs)
