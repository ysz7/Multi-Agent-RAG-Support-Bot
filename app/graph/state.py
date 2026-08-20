"""Shared state passed between graph nodes.

Two conventions worth knowing:

* **`tenant_id` and `user_id` are set once, at entry, from the caller's
  `Principal`.** No node may derive them from the question text or from tool
  output. Every retrieval reads them straight from state.
* **State is additive.** Nodes return only the keys they changed; LangGraph
  merges. `messages` is the one accumulating channel.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from app.rag.chain import Citation
from app.rag.retrievers.base import RetrievedChunk

Route = Literal["simple", "supervisor"]
ApprovalStatus = Literal["pending", "approved", "rejected"]
SupervisorStep = Literal["research", "act", "draft", "review", "finalize"]
ReviewVerdict = Literal["approved", "revise"]

# Hard stops. The graph must terminate even if the model keeps asking for more.
MAX_SUPERVISOR_ITERATIONS = 8
MAX_REVISIONS = 1
MAX_RESEARCH_QUERIES = 3


class PendingAction(TypedDict, total=False):
    """A sensitive tool call awaiting human approval (Phases 9–10)."""

    tool: str
    arguments: dict[str, Any]
    reason: str
    proposed_by: str


class Finding(TypedDict, total=False):
    """One retrieved fact, kept with its provenance so the reviewer can check it."""

    query: str
    citation: str
    content: str
    score: float
    source_path: str
    document_id: str
    chunk_index: int


class GraphState(TypedDict, total=False):
    # --- identity, set at entry from the caller's Principal -----------------
    tenant_id: str
    user_id: str
    thread_id: str

    # --- request ------------------------------------------------------------
    question: str
    messages: Annotated[list[dict[str, str]], operator.add]

    # --- routing ------------------------------------------------------------
    route: Route
    route_reason: str

    # --- retrieval and answer ----------------------------------------------
    chunks: list[RetrievedChunk]
    answer: str
    citations: list[Citation]

    # --- supervisor branch (Phase 9) ----------------------------------------
    next_step: SupervisorStep
    findings: Annotated[list[Finding], operator.add]
    draft: str
    review_verdict: ReviewVerdict | None
    review_note: str | None
    iterations: int
    revisions: int
    needs_action: bool

    # --- human-in-the-loop (Phases 9–10) ------------------------------------
    pending_action: PendingAction | None
    approval: ApprovalStatus | None
    approval_note: str | None

    # --- observability / failure -------------------------------------------
    trace_id: str | None
    error: str | None


def initial_state(
    question: str, *, tenant_id: str, user_id: str, thread_id: str | None = None, **extra: Any
) -> GraphState:
    """Build entry state. Identity is supplied by the caller, never inferred."""
    state: GraphState = {
        "question": question,
        "tenant_id": tenant_id,
        "user_id": user_id,
        "messages": [{"role": "user", "content": question}],
        "pending_action": None,
        "approval": None,
        "error": None,
    }
    if thread_id:
        state["thread_id"] = thread_id
    state.update(extra)  # type: ignore[typeddict-item]
    return state
