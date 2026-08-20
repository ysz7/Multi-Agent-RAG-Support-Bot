"""Graph nodes.

The entry router and the simple-RAG branch. The supervisor branch and its
specialist nodes live in `app/graph/supervisor.py`.

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


def _stream_writer():
    """LangGraph's per-run token sink, or None outside a streaming run."""
    try:
        from langgraph.config import get_stream_writer

        return get_stream_writer()
    except Exception:  # not inside a graph run, or streaming not requested
        return None


class SimpleRagNode:
    """Single-shot RAG answer, wrapping the Phase 6 LCEL chain.

    Streams tokens through LangGraph's custom stream when one is active, so the
    API can forward them over SSE; falls back to a plain invoke otherwise.
    """

    def __init__(self, chain: RagChain) -> None:
        self._chain = chain

    async def __call__(self, state: GraphState) -> GraphState:
        try:
            writer = _stream_writer()
            if writer is None:
                answer = await self._chain.ainvoke(
                    state["question"],
                    tenant_id=state["tenant_id"],  # from Principal, not the question
                )
            else:
                answer = None
                async for piece in self._chain.astream(
                    state["question"], tenant_id=state["tenant_id"]
                ):
                    if isinstance(piece, str):
                        writer({"type": "token", "text": piece})
                    else:
                        answer = piece
                if answer is None:
                    raise RuntimeError("stream ended without a final answer")
        except Exception as exc:
            logger.exception("simple rag node failed")
            return {"error": f"{type(exc).__name__}: {exc}", "answer": "", "messages": []}

        return {
            "answer": answer.text,
            "citations": answer.citations,
            "chunks": answer.chunks,
            "messages": [{"role": "assistant", "content": answer.text}],
        }


def select_branch(state: GraphState) -> str:
    """Conditional edge function: map the routing decision to a node name."""
    return "supervisor" if state.get("route") == "supervisor" else "simple_rag"
