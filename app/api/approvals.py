"""Approval endpoints — the HTTP face of the Phase 10 gate.

    GET  /approvals              list runs awaiting a decision (this tenant only)
    GET  /approvals/{thread_id}  inspect one proposal before deciding
    POST /approvals/{thread_id}  approve / edit / reject, resuming the graph

Tenant isolation is enforced in the store query, not by comparing values in
Python: a thread belonging to another tenant simply does not match, so it
returns 404 rather than leaking its existence.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request, status
from langgraph.types import Command

from app.api.approvals_store import ApprovalStore, PendingApproval
from app.api.schemas import ApprovalDecision, ApprovalOut, ApprovalResult
from app.core.auth import Principal, RequireApprove
from app.graph.build import GRAPH_RECURSION_LIMIT

logger = logging.getLogger("api.approvals")

router = APIRouter(prefix="/approvals", tags=["approvals"])


def _to_out(row: PendingApproval) -> ApprovalOut:
    return ApprovalOut(
        thread_id=row.thread_id,
        question=row.question,
        tool=row.tool,
        arguments=row.arguments,
        reason=row.reason,
        status=row.status,
        created_at=row.created_at,
    )


@router.get("", response_model=list[ApprovalOut])
async def list_approvals(
    request: Request,
    principal: Principal = RequireApprove,
) -> list[ApprovalOut]:
    approvals: ApprovalStore = request.app.state.approvals
    rows = await approvals.list_pending(tenant_id=principal.tenant_id)
    return [_to_out(row) for row in rows]


@router.get("/{thread_id}", response_model=ApprovalOut)
async def get_approval(
    thread_id: str,
    request: Request,
    principal: Principal = RequireApprove,
) -> ApprovalOut:
    approvals: ApprovalStore = request.app.state.approvals
    row = await approvals.get(thread_id=thread_id, tenant_id=principal.tenant_id)
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="no such approval")
    return _to_out(row)


@router.post("/{thread_id}", response_model=ApprovalResult)
async def decide(
    thread_id: str,
    decision: ApprovalDecision,
    request: Request,
    principal: Principal = RequireApprove,
) -> ApprovalResult:
    approvals: ApprovalStore = request.app.state.approvals
    graph = request.app.state.graph

    row = await approvals.get(thread_id=thread_id, tenant_id=principal.tenant_id)
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="no such approval")
    if row.status != "pending":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"already {row.status}",
        )

    resume: dict = {"decision": decision.decision, "note": decision.note}
    if decision.decision == "edit":
        if not decision.arguments:
            raise HTTPException(
                status_code=422,
                detail="'edit' requires arguments",
            )
        resume["arguments"] = decision.arguments

    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": GRAPH_RECURSION_LIMIT,
    }
    final = await graph.ainvoke(Command(resume=resume), config)

    outcome = "approved" if decision.decision in {"approve", "edit"} else "rejected"
    await approvals.close(
        thread_id=thread_id,
        status=outcome,
        decided_by=principal.user_id,
        note=decision.note,
    )
    logger.info("approval %s for thread %s by %s", outcome, thread_id, principal.user_id)

    return ApprovalResult(
        thread_id=thread_id,
        status=outcome,
        answer=final.get("answer"),
        executed=final.get("executed_actions") or [],
    )
