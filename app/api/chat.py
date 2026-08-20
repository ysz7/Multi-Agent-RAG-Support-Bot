"""Chat endpoints.

`POST /chat` streams Server-Sent Events:

    event: token     {"text": "..."}      incremental answer text
    event: node      {"node": "..."}      which graph node just ran
    event: approval  {...}                run paused; a human must decide
    event: answer    {"answer", "citations", "thread_id"}
    event: error     {"detail": "..."}
    event: done      {}

`tenant_id` never appears in the request body — it comes from the `Principal`
and is written into the graph state at entry.
"""

from __future__ import annotations

import json
import logging
import uuid

from fastapi import APIRouter, Depends, Request
from sse_starlette.sse import EventSourceResponse

from app.api.approvals_store import ApprovalStore
from app.api.deps import get_approvals, get_settings_dep
from app.api.schemas import ChatRequest, ChatResponse, CitationOut
from app.core.auth import Principal, RequireChat
from app.core.config import Settings
from app.graph.build import GRAPH_RECURSION_LIMIT
from app.graph.state import initial_state

logger = logging.getLogger("api.chat")

router = APIRouter(tags=["chat"])


def _citations(state: dict) -> list[CitationOut]:
    return [
        CitationOut(index=c.index, citation=c.citation, source_path=c.source_path, score=c.score)
        for c in (state.get("citations") or [])
    ]


async def _record_pending(
    approvals: ApprovalStore, *, thread_id: str, principal: Principal, state: dict
) -> dict | None:
    """Index a paused run so it shows up in GET /approvals."""
    action = state.get("pending_action")
    if not action:
        return None
    await approvals.record(
        thread_id=thread_id,
        tenant_id=principal.tenant_id,
        user_id=principal.user_id,
        question=state.get("question", ""),
        tool=action.get("tool", ""),
        arguments=action.get("arguments") or {},
        reason=action.get("reason"),
    )
    return action


@router.post("/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    request: Request,
    principal: Principal = RequireChat,
    settings: Settings = Depends(get_settings_dep),
    approvals: ApprovalStore = Depends(get_approvals),
) -> ChatResponse:
    """Non-streaming answer. Returns `approval_required` if the run paused."""
    graph = request.app.state.graph
    thread_id = payload.thread_id or uuid.uuid4().hex
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": GRAPH_RECURSION_LIMIT,
    }

    state = await graph.ainvoke(
        initial_state(
            payload.question,
            tenant_id=principal.tenant_id,  # server-side, always
            user_id=principal.user_id,
            thread_id=thread_id,
        ),
        config,
    )

    if "__interrupt__" in state:
        snapshot = await graph.aget_state(config)
        action = await _record_pending(
            approvals, thread_id=thread_id, principal=principal, state=snapshot.values
        )
        return ChatResponse(
            thread_id=thread_id,
            answer="",
            approval_required=True,
            pending_action=action,
            route=snapshot.values.get("route"),
        )

    return ChatResponse(
        thread_id=thread_id,
        answer=state.get("answer", ""),
        citations=_citations(state),
        route=state.get("route"),
    )


@router.post("/chat/stream")
async def chat_stream(
    payload: ChatRequest,
    request: Request,
    principal: Principal = RequireChat,
    settings: Settings = Depends(get_settings_dep),
    approvals: ApprovalStore = Depends(get_approvals),
) -> EventSourceResponse:
    """Streaming answer over SSE."""
    graph = request.app.state.graph
    thread_id = payload.thread_id or uuid.uuid4().hex
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": GRAPH_RECURSION_LIMIT,
    }
    entry = initial_state(
        payload.question,
        tenant_id=principal.tenant_id,
        user_id=principal.user_id,
        thread_id=thread_id,
    )

    async def events():
        interrupted = False
        try:
            async for mode, chunk in graph.astream(
                entry, config, stream_mode=["custom", "updates"]
            ):
                if mode == "custom" and chunk.get("type") == "token":
                    yield {"event": "token", "data": json.dumps({"text": chunk["text"]})}
                elif mode == "updates":
                    for node in chunk:
                        if node == "__interrupt__":
                            interrupted = True
                            continue
                        yield {"event": "node", "data": json.dumps({"node": node})}

            snapshot = await graph.aget_state(config)
            values = snapshot.values

            if interrupted or values.get("pending_action"):
                action = await _record_pending(
                    approvals, thread_id=thread_id, principal=principal, state=values
                )
                yield {
                    "event": "approval",
                    "data": json.dumps({"thread_id": thread_id, "pending_action": action}),
                }
            else:
                yield {
                    "event": "answer",
                    "data": json.dumps(
                        {
                            "thread_id": thread_id,
                            "answer": values.get("answer", ""),
                            "citations": [c.model_dump() for c in _citations(values)],
                        }
                    ),
                }
        except Exception as exc:
            logger.exception("chat stream failed")
            yield {"event": "error", "data": json.dumps({"detail": str(exc)})}
        finally:
            yield {"event": "done", "data": "{}"}

    return EventSourceResponse(events())
