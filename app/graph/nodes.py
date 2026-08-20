"""Graph nodes.

Phase 8 ships the entry router and the simple-RAG branch. The supervisor branch
is a clearly-marked stub that Phase 9 replaces with real specialist nodes.

Routing is two-stage on purpose:

1. A cheap deterministic pre-check catches questions that plainly request an
   *action* ("email me a refund confirmation"). Those always need the supervisor
   and its approval gate, and we do not want that decision left to a small
   model's judgement.
2. Everything else goes to the model for a one-word classification, with an
   unparseable reply defaulting to `simple` — the cheaper, safer branch, since
   `simple` cannot take any action.
"""

from __future__ import annotations

import logging
import re

from app.core.config import Settings, get_settings
from app.core.llm_provider import ChatMessage, LLMError, LLMProvider
from app.graph.state import GraphState
from app.rag.chain import RagChain

logger = logging.getLogger("graph")

# Verb-led phrasing that asks the agent to *do* something to the outside world.
_ACTION_RE = re.compile(
    r"\b(send|email|e-mail|mail|notify|escalate|open a ticket|file a ticket|"
    r"raise a ticket|cancel|refund me|issue a refund|process my refund|"
    r"delete my|close my account|update my)\b",
    re.IGNORECASE,
)

ROUTER_SYSTEM = """You classify support questions for a document-grounded assistant.

Reply with exactly one word:

SIMPLE - a single factual question answerable by looking up one thing in the docs.
COMPLEX - needs several lookups, compares or reconciles sources, is ambiguous and
needs clarification, or asks the assistant to take an action.

Reply with the single word only. No punctuation, no explanation."""


class RouterNode:
    """Entry node: decides `simple` vs `supervisor`."""

    def __init__(self, llm: LLMProvider, settings: Settings | None = None) -> None:
        self._llm = llm
        self._settings = settings or get_settings()

    async def __call__(self, state: GraphState) -> GraphState:
        question = state.get("question", "")

        if _ACTION_RE.search(question):
            return {
                "route": "supervisor",
                "route_reason": "requests an action, which needs the approval gate",
            }

        messages: list[ChatMessage] = [{"role": "user", "content": question}]
        try:
            result = await self._llm.chat(messages, system=ROUTER_SYSTEM)
        except LLMError as exc:
            # A router failure must not fail the request: answer it simply.
            logger.warning("router classification failed, defaulting to simple: %s", exc)
            return {"route": "simple", "route_reason": f"router error: {exc}"}

        verdict = (result.text or "").strip().upper()
        if "COMPLEX" in verdict:
            return {"route": "supervisor", "route_reason": "model classified as complex"}
        if "SIMPLE" in verdict:
            return {"route": "simple", "route_reason": "model classified as simple"}

        logger.warning("unparseable router reply %r, defaulting to simple", result.text)
        return {"route": "simple", "route_reason": "unparseable classification"}


class SimpleRagNode:
    """Single-shot RAG answer, wrapping the Phase 6 LCEL chain."""

    def __init__(self, chain: RagChain) -> None:
        self._chain = chain

    async def __call__(self, state: GraphState) -> GraphState:
        try:
            answer = await self._chain.ainvoke(
                state["question"],
                tenant_id=state["tenant_id"],  # from Principal, never from the question
            )
        except Exception as exc:
            logger.exception("simple rag node failed")
            return {"error": f"{type(exc).__name__}: {exc}", "answer": "", "messages": []}

        return {
            "answer": answer.text,
            "citations": answer.citations,
            "chunks": answer.chunks,
            "messages": [{"role": "assistant", "content": answer.text}],
        }


class SupervisorStubNode:
    """Placeholder for the Phase 9 multi-agent branch.

    Answers via the simple chain so the graph is runnable end to end, and marks
    the state so it is obvious this is not the real supervisor yet.
    """

    def __init__(self, chain: RagChain) -> None:
        self._delegate = SimpleRagNode(chain)

    async def __call__(self, state: GraphState) -> GraphState:
        logger.info("supervisor stub handling question (Phase 9 replaces this)")
        result = await self._delegate(state)
        return {**result, "route_reason": f"{state.get('route_reason', '')} [supervisor stub]"}


def select_branch(state: GraphState) -> str:
    """Conditional edge function: map the routing decision to a node name."""
    return "supervisor" if state.get("route") == "supervisor" else "simple_rag"
